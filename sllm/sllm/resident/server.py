"""Local HTTP control and token-ID inference service for resident GPU pools.

Start once with `python -m sllm.resident.server --torchrun ... --gpu-count N`.
One pool process stays alive per GPU. HTTP callers can operate concurrently;
the existing controller serializes model operations and checks generations.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import logging
import math
import os
from pathlib import Path

from sllm.resident.client import ResidentPoolClient
from sllm.resident.controller import ResidentController


class IdleReaper:
    def __init__(self, controller, interval=0.5):
        self.controller = controller
        self.interval = interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name='resident-idle-reaper')

    def run(self):
        while not self.stop.wait(self.interval):
            try:
                self.controller.reap_idle()
            except Exception:
                logging.exception('Resident idle reclamation failed')

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


class ResidentAPI:
    def __init__(self, controller, relay=None, recovery=None):
        self.controller = controller
        self.relay = relay
        self.recovery = recovery

    def dispatch(self, method, path, payload=None):
        if method == 'POST' and path == '/v1/relay':
            if self.relay is None:
                return 404, dict(error='Relay models are not configured')
            if (not isinstance(payload, dict) or
                    not {'name', 'prompt_token_ids'} <= payload.keys() or
                    payload.keys() - {'name', 'prompt_token_ids', 'max_tokens'} or
                    not isinstance(payload['name'], str)):
                return 400, dict(error='Invalid relay request fields')
            try:
                return 200, self.relay.generate(payload['name'], payload['prompt_token_ids'],
                                                payload.get('max_tokens', 8))
            except ValueError as exc:
                return 409, dict(error=str(exc))
            except RuntimeError as exc:
                return 503, dict(error=str(exc))
        if method == 'GET' and path == '/v1/status':
            status = self.controller.status()
            status['instances'] = [asdict(instance) for instance in status['instances']]
            if self.recovery is not None:
                status['recovery'] = self.recovery.status()
            return 200, status
        operations = {'/v1/create', '/v1/hydra_stage', '/v1/generate', '/v1/switch',
                      '/v1/lazy_init', '/v1/release'}
        if method != 'POST' or path not in operations:
            return 404, dict(error='Unknown resident endpoint')
        if self.relay is not None and self.relay._recovering.is_set():
            return 503, dict(error='Resident service is recovering; retry a new request later')
        if not isinstance(payload, dict):
            return 400, dict(error='Request must be a JSON object')
        fields = {
            '/v1/create': ({'name', 'model', 'tp'}, {'load_options', 'pp'}),
            '/v1/hydra_stage': ({'name', 'native_name', 'native_generation', 'model', 'tp', 'pp'}, {'load_options'}),
            '/v1/generate': ({'name', 'generation', 'prompt_token_ids'}, {'max_tokens', 'request_id', 'kv_transfer_params'}),
            '/v1/switch': ({'name', 'generation', 'model'}, {'load_options'}),
            '/v1/lazy_init': ({'name', 'generation', 'layers', 'request_id'}, set()),
            '/v1/release': ({'name', 'generation'}, set()),
        }
        required, optional = fields[path]
        if not required <= payload.keys() or payload.keys() - required - optional:
            return 400, dict(error='Missing or unknown request fields')
        if not isinstance(payload['name'], str) or not payload['name']:
            return 400, dict(error='Instance name must be a nonempty string')
        if self.relay is not None and payload['name'].startswith(self.relay.prefix):
            return 409, dict(error='Relay service owns this instance route')
        options = payload.get('load_options', {})
        allowed_load = {'load_format', 'load_method', 'shm_info', 'max_model_len', 'gpu_memory_utilization', 'startup_mode'}
        if not isinstance(options, dict) or options.keys() - allowed_load:
            return 400, dict(error='Invalid load options')
        if path == '/v1/generate':
            prompt = payload['prompt_token_ids']
            maximum = payload.get('max_tokens', 8)
            if (not isinstance(prompt, list) or not prompt or
                    any(type(token) is not int or token < 0 for token in prompt) or
                    type(maximum) is not int or maximum < 1):
                return 400, dict(error='Invalid prompt token IDs or generation length')
        if path == '/v1/lazy_init':
            layers = payload['layers']
            if (not isinstance(layers, list) or not layers or
                    any(not (type(layer) is int and layer >= 0) and not
                        (isinstance(layer, list) and layer and
                         all(type(item) is int and item >= 0 for item in layer))
                        for layer in layers) or
                    (any(isinstance(layer, list) for layer in layers) and
                     not all(isinstance(layer, list) for layer in layers)) or
                    not isinstance(payload['request_id'], str) or not payload['request_id']):
                return 400, dict(error='Invalid layers or request ID')
        try:
            if path == '/v1/create':
                instance = self.controller.create(payload['name'], payload['model'], payload['tp'],
                    **({'pp': payload['pp']} if 'pp' in payload else {}), **options)
                return 201, asdict(instance)
            if path == '/v1/hydra_stage':
                native = self.controller.lookup(payload['native_name'], payload['native_generation'])
                options.setdefault('startup_mode', 'hydraserve')
                instance = self.controller.create(payload['name'], payload['model'], payload['tp'],
                    pp=payload['pp'], staging_for=native, **options)
                return 201, asdict(instance)
            instance = self.controller.lookup(payload['name'], payload['generation'])
            if path == '/v1/generate':
                result = self.controller.generate(instance, payload['prompt_token_ids'],
                    **{key: payload[key] for key in ('max_tokens', 'request_id', 'kv_transfer_params') if key in payload})
                return 200, result
            if path == '/v1/switch':
                return 200, asdict(self.controller.switch(instance, payload['model'], **options))
            if path == '/v1/lazy_init':
                return 200, asdict(self.controller.lazy_init(instance, payload['layers'], payload['request_id']))
            self.controller.release(instance)
            return 200, dict(released=True)
        except ValueError as exc:
            return 409, dict(error=str(exc))
        except RuntimeError as exc:
            return 503, dict(error=str(exc))


def make_server(address, api, *, max_callers=32, max_body_bytes=2 * 1024 * 1024):
    if max_callers < 1 or max_body_bytes < 1:
        raise ValueError('HTTP limits must be positive')

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(10)
            super().setup()

        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(*api.dispatch('GET', self.path))

        def do_POST(self):
            if self.headers.get('Transfer-Encoding'):
                self.reply(400, dict(error='Content-Length is required'))
                return
            try:
                size = int(self.headers.get('Content-Length', '-1'))
            except ValueError:
                size = -1
            if size < 0:
                self.reply(400, dict(error='Invalid Content-Length'))
                return
            if size > max_body_bytes:
                self.reply(413, dict(error='Request body exceeds configured limit'))
                return
            try:
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError('Incomplete body')
                payload = json.loads(body)
            except (ValueError, UnicodeError):
                self.reply(400, dict(error='Invalid JSON request body'))
                return
            if self.path == '/v1/relay/stream':
                if api.relay is None:
                    self.reply(404, dict(error='Relay models are not configured'))
                    return
                if (not isinstance(payload, dict) or
                        not {'name', 'prompt_token_ids'} <= payload.keys() or
                        payload.keys() - {'name', 'prompt_token_ids', 'max_tokens'} or
                        not isinstance(payload['name'], str)):
                    self.reply(400, dict(error='Invalid relay request fields'))
                    return
                try:
                    api.relay.validate_request(payload['name'], payload['prompt_token_ids'],
                                               payload.get('max_tokens', 8))
                except ValueError as exc:
                    self.reply(409, dict(error=str(exc)))
                    return
                except RuntimeError as exc:
                    self.reply(503, dict(error=str(exc)))
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')
                self.end_headers()
                events = api.relay.stream(payload['name'], payload['prompt_token_ids'],
                                          payload.get('max_tokens', 8))
                try:
                    for event in events:
                        data = json.dumps(event, separators=(',', ':'))
                        self.wfile.write(f"event: {event['event']}\ndata: {data}\n\n".encode())
                        self.wfile.flush()
                except OSError:
                    pass
                finally:
                    # The iterator owns warm cancellation and cold handoff
                    # draining; closing it joins that owner before teardown.
                    events.close()
                self.close_connection = True
                return
            self.reply(*api.dispatch('POST', self.path, payload))

        def log_message(self, *_):
            pass

    class Server(ThreadingHTTPServer):
        # Non-daemon request threads are drained by server_close before the
        # GPU pool is shut down. Bound queueing avoids unbounded handler threads.
        daemon_threads = False

        def __init__(self, address):
            self.callers = threading.BoundedSemaphore(max_callers)
            super().__init__(address, Handler)

        def process_request(self, request, client_address):
            if not self.callers.acquire(blocking=False):
                try:
                    request.settimeout(1)
                    request.sendall(b'HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n')
                finally:
                    self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                self.callers.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self.callers.release()

    return Server(address)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--torchrun')
    sources.add_argument('--node-config', type=Path,
                         help='JSON mapping node names to url and key_file')
    sources.add_argument('--cluster-config', type=Path,
                         help='Global WORLD TCP endpoints, physical nodes and key_file')
    sources.add_argument('--global-world-config', type=Path,
                         help='Owned single-host logical-node WORLD config')
    parser.add_argument('--gpu-count', type=int)
    parser.add_argument('--gpu-cpu-ids',
                        help='Comma-separated local GPU worker CPU allocation; each rank selects its NUMA subset')
    parser.add_argument('--concurrent-topologies', action='store_true',
                        help='Use independent rank sockets for disjoint GPU instances')
    parser.add_argument('--copy-throttle', action=argparse.BooleanOptionalAction,
                        default=True, help='Throttle weight DMA submissions during GPU initialization')
    parser.add_argument('--copy-throttle-tasks', type=int, default=32,
                        help='Copy tasks per host fence; larger values weaken warmup throttling')
    parser.add_argument('--copy-throttle-first-only', action='store_true',
                        help='Throttle the first SHM model in each GPU process, then disable throttling')
    parser.add_argument('--relay-config', type=Path,
                        help='Local tokenwise model registrations as JSON')
    parser.add_argument('--cpu-python')
    parser.add_argument('--cpu-root', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--auto-recovery', action=argparse.BooleanOptionalAction, default=True,
                        help='Automatically rebuild failed locally owned pools and CPU producers')
    parser.add_argument('--recovery-interval', type=float, default=.1)
    parser.add_argument('--recovery-backoff', type=float, default=1.)
    parser.add_argument('--recovery-max-backoff', type=float, default=30.)
    parser.add_argument('--idle-timeout', type=float, default=10,
                        help='Idle seconds before model unload; 0 disables reclamation')
    parser.add_argument('--max-callers', type=int, default=32)
    parser.add_argument('--prewarm-tp', default='1,2,4',
                        help='Comma-separated TP sizes prepared before accepting HTTP requests')
    parser.add_argument('--prewarm-pp', default='1',
                        help='Comma-separated PP sizes prepared before accepting HTTP requests')
    args = parser.parse_args()
    if (any(not math.isfinite(value) or value <= 0 for value in
            (args.recovery_interval, args.recovery_backoff, args.recovery_max_backoff))
            or args.recovery_backoff > args.recovery_max_backoff):
        parser.error('Recovery intervals must be positive and initial backoff <= maximum backoff')
    if not math.isfinite(args.idle_timeout) or args.idle_timeout < 0:
        parser.error('idle-timeout must be finite and nonnegative')
    try:
        sizes = [int(value) for value in args.prewarm_tp.split(',')]
        pp_sizes = [int(value) for value in args.prewarm_pp.split(',')]
        if not sizes or not pp_sizes or any(value < 1 for value in sizes + pp_sizes):
            raise ValueError()
    except ValueError:
        parser.error('prewarm-tp and prewarm-pp must contain positive integers')
    if args.torchrun and (args.gpu_count is None or args.gpu_count < 1):
        parser.error('Local pool requires positive gpu-count')
    gpu_cpu_ids = None
    if args.gpu_cpu_ids is not None:
        if not args.torchrun:
            parser.error('gpu-cpu-ids applies to a local torchrun pool')
        try:
            gpu_cpu_ids = [int(value) for value in args.gpu_cpu_ids.split(',')]
            if (not gpu_cpu_ids or len(set(gpu_cpu_ids)) != len(gpu_cpu_ids)
                    or any(cpu < 0 for cpu in gpu_cpu_ids)
                    or not set(gpu_cpu_ids) <= os.sched_getaffinity(0)):
                raise ValueError()
        except ValueError:
            parser.error('gpu-cpu-ids must contain distinct allowed CPU IDs')
    if args.relay_config and ((not args.torchrun and not args.cluster_config
                               and not args.global_world_config)
                              or not args.cpu_python or not args.cpu_root):
        parser.error('relay-config requires a local torchrun or cluster WORLD, cpu-python and cpu-root')
    relay_gpu_capacity = args.gpu_count
    if relay_gpu_capacity is None and args.cluster_config:
        try:
            cluster_manifest = json.loads(args.cluster_config.read_text())
            relay_gpu_capacity = len(cluster_manifest['ranks'])
        except (OSError, ValueError, KeyError, TypeError):
            parser.error('cluster-config must provide a valid rank list before relay validation')
    if relay_gpu_capacity is None and args.global_world_config:
        try:
            world_config = json.loads(args.global_world_config.read_text())
            relay_gpu_capacity = sum(len(value.split(','))
                                     for value in world_config['devices_by_node'])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            parser.error('global-world-config must provide devices_by_node')
    with ExitStack() as stack:
        # Separate owner stacks also own replacements. Supervision stops before
        # these stacks unwind; all GPU consumers close before any CPU buffer.
        producer_owners = stack.enter_context(ExitStack())
        pool_owners = stack.enter_context(ExitStack())
        # Producers are registered first so GPU consumers close before the
        # CPU-owned SHM sources at shutdown or failed startup.
        registrations = {}
        if args.relay_config:
            from sllm.resident.cpu_client import CpuProducerClient
            config = json.loads(args.relay_config.read_text())
            if not isinstance(config, dict) or not config:
                parser.error('relay-config requires a nonempty model mapping')
            for name, value in config.items():
                if (not isinstance(name, str) or not name or not isinstance(value, dict) or
                        not {'checkpoint', 'gpu_tp', 'prefix_tokens'} <= value.keys() or
                        value.keys() - {'checkpoint', 'gpu_tp', 'gpu_pp', 'layer_idxes_by_pp', 'prefix_tokens', 'cpu_tp', 'max_model_len', 'cpu_layers', 'planner_profile', 'startup_mode', 'hydra_profile', 'ttft_slo_ms', 'tpot_slo_ms', 'hydra_min_free_memory_ratio'}):
                    parser.error('Invalid relay model registration fields')
                if (not isinstance(value['checkpoint'], str) or not value['checkpoint'] or
                        any(type(value.get(key, default)) is not int or value.get(key, default) < 1
                            for key, default in [('gpu_tp', 1), ('gpu_pp', 1),
                                                 ('cpu_tp', 2), ('max_model_len', 40960)]) or
                        value['gpu_tp'] * value.get('gpu_pp', 1) > relay_gpu_capacity):
                    parser.error('Invalid relay checkpoint or parallelism/length settings')
                if type(value['prefix_tokens']) is not int or value['prefix_tokens'] < 0:
                    parser.error('prefix_tokens must be a nonnegative integer')
                if type(value.get('cpu_layers', 0)) is not int or value.get('cpu_layers', 0) < 0:
                    parser.error('cpu_layers must be a nonnegative integer')
                if value.get('startup_mode', 'relay') not in ('relay', 'serverlessllm', 'hydraserve'):
                    parser.error('startup_mode must be relay, serverlessllm or hydraserve')
                try:
                    ResidentController._validate_hydra_memory_ratio(
                        value.get('hydra_min_free_memory_ratio', 0.0))
                except ValueError as exc:
                    parser.error(str(exc))
                if value.get('startup_mode') == 'hydraserve':
                    if (not {'hydra_profile', 'ttft_slo_ms', 'tpot_slo_ms'} <= value.keys()
                            or 'planner_profile' in value or value['prefix_tokens']
                            or value.get('cpu_layers', 0) or value.get('gpu_pp', 1) != 1
                            or value.get('layer_idxes_by_pp') is not None):
                        parser.error('Hydra requires profile/SLOs, native PP1 and no CPU relay')
                elif {'hydra_profile', 'ttft_slo_ms', 'tpot_slo_ms', 'hydra_min_free_memory_ratio'} & value.keys():
                    parser.error('Hydra fields require hydraserve startup mode')
                if value.get('startup_mode') == 'serverlessllm' and (
                        'planner_profile' in value or value.get('cpu_layers', 0)):
                    parser.error('ServerlessLLM baseline cannot use CPU relay planning')
            for name, value in config.items():
                producer = producer_owners.enter_context(CpuProducerClient(args.cpu_python, args.cpu_root,
                    value['checkpoint'], tp=value.get('cpu_tp', 2), timeout=args.timeout,
                    max_model_len=value.get('max_model_len', 40960)))
                registrations[name] = (producer, value)
        physical_nodes = None
        if args.global_world_config:
            from sllm.resident.global_world import GlobalWorldOwner
            config = json.loads(args.global_world_config.read_text())
            allowed = {'torchrun', 'devices_by_node', 'timeout', 'rank_host',
                       'master_host', 'worker_module', 'log_dir'}
            if (not isinstance(config, dict) or set(config) - allowed
                    or not isinstance(config.get('torchrun'), str)
                    or not isinstance(config.get('devices_by_node'), list)
                    or not config['devices_by_node']):
                parser.error('Invalid global-world-config')
            owner = pool_owners.enter_context(GlobalWorldOwner(
                config['torchrun'], config['devices_by_node'],
                timeout=config.get('timeout', args.timeout),
                rank_host=config.get('rank_host', '127.0.0.1'),
                master_host=config.get('master_host', '127.0.0.1'),
                worker_module=config.get('worker_module', 'sllm.resident.worker'),
                log_dir=config.get('log_dir')))
            pools = {'world': owner}
            physical_nodes = {'world': [f'node{node}'
                for node, devices in enumerate(config['devices_by_node'])
                for _ in devices.split(',')]}
        elif args.cluster_config:
            from sllm.resident.cluster import parse_cluster_config
            from sllm.resident.rank_transport import RankTransport
            endpoints, nodes, key = parse_cluster_config(json.loads(args.cluster_config.read_text()))
            pool = RankTransport(endpoints, timeout=args.timeout, key=key)
            # ExitStack closes HTTP handlers and reaper first, then unloads
            # models and explicitly shuts down cluster ranks before channels.
            stack.callback(pool.close)
            stack.callback(pool.command, 'shutdown')
            pools = {'world': pool}
            physical_nodes = {'world': nodes}
        elif args.node_config:
            from sllm.resident.remote import RemotePoolClient
            config = json.loads(args.node_config.read_text())
            if (not isinstance(config, dict) or not config or
                    any(not isinstance(name, str) or not name or not isinstance(value, dict)
                        or value.keys() != {'url', 'key_file'} for name, value in config.items())):
                parser.error('node-config must map node names to url and key_file')
            pools = {name: pool_owners.enter_context(RemotePoolClient(value['url'],
                Path(value['key_file']).read_text().strip(), timeout=args.timeout))
                for name, value in config.items()}
        else:
            pools = {'local': pool_owners.enter_context(ResidentPoolClient(
                args.torchrun, args.gpu_count, timeout=args.timeout,
                concurrent_topologies=args.concurrent_topologies,
                copy_throttle=args.copy_throttle,
                copy_throttle_tasks=args.copy_throttle_tasks,
                copy_throttle_first_only=args.copy_throttle_first_only,
                gpu_cpu_ids=gpu_cpu_ids))}
        controller = ResidentController(pools, idle_timeout=args.idle_timeout,
                                        physical_nodes_by_rank=physical_nodes)
        preparation = controller.prewarm(sizes, pp_sizes=pp_sizes)
        relay = None
        weight_preparation = {}
        if registrations:
            from sllm.resident.relay import RelayService
            relay = RelayService(controller)
            stack.callback(relay.close)
            for name, (producer, value) in registrations.items():
                weight_pool = pools['local'] if 'local' in pools else pools['world']
                weight_preparation[name] = weight_pool.command('prepare_weights',
                    model=str(producer.model_path), gpu_tp=value['gpu_tp'],
                    shm_info=producer.info['shm_info'],
                    chunk_tokens=producer.max_num_batched_tokens).ranks
                planner = None
                if 'hydra_profile' in value:
                    from sllm.resident.hydra import HydraPlanner
                    planner = HydraPlanner(json.loads(Path(value['hydra_profile']).read_text()))
                if 'planner_profile' in value:
                    from sllm.resident.planner import ResidentPlanner
                    planner = ResidentPlanner(json.loads(Path(value['planner_profile']).read_text()),
                        producer.checkpoint, gpu_tp=value['gpu_tp'], cpu_tp=producer.tp,
                        max_model_len=value.get('max_model_len', 40960),
                        copy_throttle=args.copy_throttle,
                        copy_throttle_tasks=args.copy_throttle_tasks,
                        copy_throttle_first_only=args.copy_throttle_first_only,
                        config=json.loads((producer.model_path / 'config.json').read_text()))
                relay.register(name, producer, tp=value['gpu_tp'], prefix_tokens=value['prefix_tokens'],
                               cpu_layers=value.get('cpu_layers', 0), planner=planner,
                               pp=value.get('gpu_pp', 1), layer_idxes_by_pp=value.get('layer_idxes_by_pp'),
                               startup_mode=value.get('startup_mode', 'relay'),
                               ttft_slo_ms=value.get('ttft_slo_ms'),
                               tpot_slo_ms=value.get('tpot_slo_ms'),
                               hydra_min_free_memory_ratio=value.get('hydra_min_free_memory_ratio', 0.0))
        recovery = None
        if args.auto_recovery and (args.torchrun or args.node_config or args.global_world_config):
            from sllm.resident.recovery import RecoverySupervisor
            def pool_factory(node):
                return lambda: pool_owners.enter_context(controller.pools[node].replacement())
            def rebuild_producer(previous):
                return producer_owners.enter_context(previous.replacement())
            recovery = RecoverySupervisor(controller, {node: pool_factory(node) for node in pools}, relay=relay,
                producer_factory=rebuild_producer, interval=args.recovery_interval,
                retry_initial=args.recovery_backoff, retry_max=args.recovery_max_backoff)
        server = make_server((args.host, args.port), ResidentAPI(controller, relay, recovery),
                             max_callers=args.max_callers)
        stack.enter_context(IdleReaper(controller))
        if recovery is not None:
            stack.enter_context(recovery)
        print(json.dumps(dict(listening=server.server_address, numa_by_rank=controller.numa,
                              weight_preparation=weight_preparation,
                              communication_preparation=preparation)), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
