"""Ray-free resident rank loop. Launch one rank per GPU using torchrun.

Rank zero reads JSON commands from stdin; replies carry the same sequence.
The controller sends each command once and waits for its reply. Other ranks
block on the CPU control group while idle. Model commands use prepared groups;
probe is a communication diagnostic.
"""
import argparse
import json
import os
import sys
import time


def topology_from_json(value):
    from sllm.resident.groups import Topology
    return Topology(tuple(value['ranks']), value['tp'], value.get('pp', 1))


class RankLoop:
    def __init__(self, registry, commands=None, gpu_inventory=None):
        self.registry = registry
        self.commands = commands or sys.stdin
        self.next_sequence = 0
        self.model_runtime = None
        from sllm.resident.runtime_state import ModelRuntimeState
        self.runtime_state = ModelRuntimeState()
        self.model_slots = {}
        self._pending_slots = {}
        self.gpu_inventory = gpu_inventory

    def _slot(self, command):
        name = command.get('slot', 'default')
        if not isinstance(name, str) or not name or len(name) > 128:
            raise ValueError('Model slot must be a nonempty string of at most 128 characters')
        if name == 'default':
            return self.model_runtime, self.runtime_state
        return self.model_slots.get(name, self._pending_slots.get(name, (None, None)))

    def _validate_model(self, command, topology):
        runtime, binding = self._slot(command)
        if self.registry.rank not in topology.ranks:
            return
        if (command.get('slot', 'default') != 'default' and command['op'] == 'load'
                and command['slot'] in self.model_slots
                and getattr(self.model_slots[command['slot']][0], 'engine', None) is not None):
            raise ValueError('Model slot is already loaded')
        if runtime is None:
            if command['op'] != 'load':
                raise ValueError('Target model slot is not loaded')
            from sllm.resident.model import ResidentModel
            from sllm.resident.runtime_state import ModelRuntimeState
            runtime = ResidentModel(self.registry)
            binding = ModelRuntimeState()
            # A rejected load must not publish a model slot.
            runtime.validate(command, topology)
            if command.get('slot', 'default') == 'default':
                self.model_runtime, self.runtime_state = runtime, binding
            else:
                self._pending_slots[command['slot']] = runtime, binding
        else:
            runtime.validate(command, topology)


    def execute(self, command):
        op = command['op']
        if op == 'recover_pp':
            from sllm.resident.migration import recover_requests
            topology = topology_from_json(command['topology'])
            runtime, binding = self._slot(command)
            target, _ = self._slot(dict(slot=command['target_slot']))
            if self.registry.rank not in command['target_topology']['ranks']:
                target = None
            try:
                with binding.activate(), self.registry.activate(topology, retain_active=True):
                    return recover_requests(runtime, target)
            except BaseException:
                self.registry._poisoned = True
                raise
        if op in ('status', 'shutdown'):
            slots = dict(self.model_slots)
            if self.model_runtime is not None:
                slots['default'] = self.model_runtime, self.runtime_state
            if op == 'shutdown':
                # Validate every slot before releasing any model. A retained
                # request in one slot must not partially tear down its peers.
                if any(runtime.sessions.retained for runtime, _ in slots.values()):
                    raise ValueError('Drain active requests before shutting down models')
                for runtime, binding in slots.values():
                    with binding.activate():
                        runtime.unload()
            memory = {}
            if op == 'status' and command.get('include_memory'):
                import torch
                free, total = torch.cuda.mem_get_info(self.registry.device)
                memory['gpu_memory'] = dict(free_bytes=free, total_bytes=total)
            return dict(**memory, pid=os.getpid(), rank=self.registry.rank,
                        gpu_inventory=self.gpu_inventory,
                        cached_groups=len(self.registry._groups),
                        topologies=len(self.registry._topologies),
                        model_slot=(self.model_runtime.status()
                                    if self.model_runtime else None),
                        model_slots={name: runtime.status()
                                     for name, (runtime, _) in slots.items()})
        if op in ('load', 'lazy_init', 'unload', 'generate', 'profile_memory',
                  'submit', 'step', 'collect', 'abort', 'weights_ready'):
            try:
                topology = topology_from_json(command['topology'])
                if self.registry.rank not in topology.ranks:
                    return dict(idle=True)
                runtime, binding = self._slot(command)
                with binding.activate():
                    result = runtime.execute(command, topology)
                if command['op'] == 'load' and command.get('slot', 'default') != 'default':
                    pending = self._pending_slots.pop(command['slot'], None)
                    if pending is not None:
                        self.model_slots[command['slot']] = pending
                return result
            except BaseException:
                # A rank failing during model/device execution can leave its
                # peers in unmatched collectives. Never reuse the pool then.
                # Readiness only queries native events: no model execution or
                # rank collective has started. Return its loader error so the
                # controller quarantines the node, retaining the command loop
                # for status/teardown rather than terminating every rank.
                if op != 'weights_ready':
                    self.registry._poisoned = True
                raise
        if op == 'prepare':
            # Library imports belong to process prewarming, not the first
            # model's configuration-validation interval.
            from sllm.resident.model import prewarm_model_runtime
            prewarm_model_runtime()
            self.registry.prepare([topology_from_json(value)
                                   for value in command['topologies']])
            return self.execute(dict(op='status'))
        if op == 'prepare_weights':
            from sllm.resident.weights import prepare_weights, source_ranks
            return prepare_weights(command['model'], source_ranks(self.registry, command['model'], command['gpu_tp']),
                shm_info=command.get('shm_info'), chunk_tokens=command.get('chunk_tokens', 4096))
        if op == 'probe':
            import torch
            topology = topology_from_json(command['topology'])
            if self.registry.rank not in topology.ranks:
                return dict(rank=self.registry.rank, idle=True)
            with self.registry.activate(topology) as groups:
                probe = torch.tensor([[self.registry.rank]],
                                     device=self.registry.device,
                                     dtype=torch.float32)
                values = groups['tp'].all_gather(probe).cpu().tolist()
                return dict(rank=self.registry.rank, pid=os.getpid(),
                            tp_ranks=groups['tp'].ranks,
                            pp_ranks=groups['pp'].ranks, values=values)
        raise ValueError(f'Unknown resident operation: {op}')

    def cancel(self, command):
        """Discard transient state built during validation of a rejected load."""
        if (isinstance(command, dict) and command.get('op') == 'load'
                and command.get('slot', 'default') != 'default'):
            slot = command['slot']
            runtime = self._pending_slots.get(slot, (None, None))[0]
            if runtime is not None and getattr(runtime, 'engine', None) is None:
                self._pending_slots.pop(slot, None)

    def validate(self, command):
        """Validate everywhere before dispatching any device collective."""
        if not isinstance(command, dict):
            raise ValueError('Command must be an object')
        if type(command.get('sequence')) is not int:
            raise ValueError('Command sequence must be an integer')
        if command['sequence'] != self.next_sequence:
            raise ValueError(f'Expected command sequence {self.next_sequence}')
        op = command.get('op')
        if op == 'shutdown':
            slots = list(self.model_slots.values())
            if self.model_runtime is not None:
                slots.append((self.model_runtime, self.runtime_state))
            if any(runtime.sessions.retained for runtime, _ in slots):
                raise ValueError('Drain active requests before shutting down models')
        if op == 'prepare':
            if not isinstance(command.get('topologies'), list):
                raise ValueError('prepare requires a topology list')
            for value in command['topologies']:
                topology_from_json(value).partition('tp', self.registry.world_size)
        elif op == 'prepare_weights':
            if not isinstance(command.get('model'), str) or not command['model']:
                raise ValueError('prepare_weights requires model path')
            from sllm.resident.weights import preparation_sources, source_ranks
            ranks = source_ranks(self.registry, command['model'], command.get('gpu_tp'))
            preparation_sources(command['model'], ranks,
                shm_info=command.get('shm_info'), chunk_tokens=command.get('chunk_tokens', 4096))
        elif op == 'recover_pp':
            from sllm.resident.migration import validate_recovery
            topology = topology_from_json(command['topology'])
            target_topology = topology_from_json(command['target_topology'])
            runtime, source_binding = self._slot(command)
            target, target_binding = self._slot(dict(slot=command['target_slot']))
            if runtime is None or runtime.generation != command['generation']:
                raise ValueError('Stale recovery source')
            if self.registry.rank in target_topology.ranks:
                if target is None or target.generation != command['target_generation']:
                    raise ValueError('Stale recovery destination')
            else:
                target = None
            with source_binding.activate():
                source_ready = runtime.weights_ready()
            target_ready = None
            if target is not None:
                with target_binding.activate():
                    target_ready = target.weights_ready()
            validate_recovery(runtime, target, topology, target_topology,
                source_weights_ready=source_ready, target_weights_ready=target_ready)
        elif op in ('probe', 'load', 'lazy_init', 'unload', 'generate', 'profile_memory',
                    'submit', 'step', 'collect', 'abort', 'weights_ready'):
            topology = topology_from_json(command['topology'])
            topology.partition('tp', self.registry.world_size)
            if topology not in self.registry._topologies:
                raise ValueError('Topology was not prepared')
            if op != 'probe':
                self._validate_model(command, topology)
        elif op not in ('status', 'shutdown'):
            raise ValueError(f'Unknown resident operation: {op}')

    def run(self):
        import torch.distributed as dist
        rank = self.registry.rank
        group = self.registry.control_group
        while True:
            envelope = [None]
            if rank == 0:
                line = self.commands.readline()
                if not line:
                    envelope[0] = dict(sequence=self.next_sequence, op='shutdown')
                else:
                    try:
                        envelope[0] = json.loads(line)
                    except (ValueError, TypeError):
                        envelope[0] = None
            dist.broadcast_object_list(envelope, src=0, group=group)
            command = envelope[0]
            error = None
            validation_start = time.perf_counter()
            try:
                self.validate(command)
            except Exception as exc:
                # A source manifest/SHM segment may disappear or become
                # unreadable before registration. Engine configuration can
                # also raise RuntimeError/NotImplementedError. Join every
                # validation exception before any model/device operation,
                # instead of stranding idle peer ranks on the control group.
                error = str(exc)
            validation_seconds = time.perf_counter() - validation_start
            errors = [None] * self.registry.world_size
            dist.all_gather_object(errors, error, group=group)
            if any(errors):
                self.cancel(command)
                if rank == 0:
                    self.reply(command, False, errors, advance=False)
                continue
            result = None
            error = None
            execution_start = time.perf_counter()
            try:
                result = self.execute(command)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            execution_seconds = time.perf_counter() - execution_start
            outcomes = [None] * self.registry.world_size
            dist.all_gather_object(outcomes, dict(
                result=result, error=error, fatal=self.registry._poisoned,
                validation_seconds=validation_seconds,
                execution_seconds=execution_seconds),
                                   group=group)
            failed = any(item['error'] for item in outcomes)
            # Valid dispatched commands always consume their sequence,
            # including a failed operation. Rejected commands do not.
            self.next_sequence += 1
            if rank == 0:
                self.reply(command, not failed, outcomes, advance=True)
            if command['op'] == 'shutdown' or any(
                    item['fatal'] for item in outcomes):
                return

    def reply(self, command, ok, outcomes, advance):
        sequence = command.get('sequence') if isinstance(command, dict) else None
        print('RESIDENT_REPLY ' + json.dumps(dict(
            sequence=sequence, next_sequence=self.next_sequence,
            accepted=advance, ok=ok, ranks=outcomes)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-seconds', type=int, default=120)
    parser.add_argument('--command-file',
                        help='Rank-zero JSON-lines input, including a named pipe')
    parser.add_argument('--rank-socket-dir',
                        help='Independent local command socket for each resident rank')
    parser.add_argument('--rank-tcp-host', help='Bind authenticated TCP rank control')
    parser.add_argument('--rank-tcp-port-base', type=int, default=19000)
    parser.add_argument('--rank-key-file')
    parser.add_argument('--weight-cache-bytes', type=int, default=80 * 1024**3,
                        help='Per-process idle pinned weight retention budget')
    args = parser.parse_args()
    if args.rank_tcp_host and (args.rank_socket_dir or not args.rank_key_file):
        parser.error('TCP rank control requires key-file and excludes Unix sockets')
    if not 1 <= args.rank_tcp_port_base <= 65535:
        parser.error('Invalid TCP port base')
    if args.weight_cache_bytes < 0:
        parser.error('weight-cache-bytes must be nonnegative')
    os.environ['VLLM_SHM_WEIGHT_CACHE_BYTES'] = str(args.weight_cache_bytes)
    if args.timeout_seconds < 1:
        parser.error('timeout must be positive')
    import datetime
    import torch
    import torch.distributed as dist
    from vllm.distributed import parallel_state as state
    from sllm.resident.groups import CommunicationRegistry
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    from sllm.resident.inventory import current_gpu_inventory, bind_worker_to_gpu_numa
    inventory = bind_worker_to_gpu_numa(current_gpu_inventory(device))
    # Keep the default group on CPU: idle control commands need no CUDA work.
    dist.init_process_group('gloo', timeout=datetime.timedelta(
        seconds=args.timeout_seconds))
    state.set_custom_all_reduce(True)
    state.init_distributed_environment(local_rank=device.index, backend='nccl')
    registry = CommunicationRegistry(device)
    commands = (open(args.command_file) if args.command_file
                and registry.rank == 0 else sys.stdin)
    try:
        loop = RankLoop(registry, commands, inventory)
        if args.rank_tcp_host:
            from pathlib import Path
            from sllm.resident.rank_transport import serve_rank
            key = Path(args.rank_key_file).read_text().strip()
            serve_rank((args.rank_tcp_host, args.rank_tcp_port_base + device.index), loop, key=key)
        elif args.rank_socket_dir:
            from pathlib import Path
            from sllm.resident.rank_transport import serve_rank
            serve_rank(Path(args.rank_socket_dir) / f'rank-{registry.rank}.sock', loop)
        else:
            loop.run()
    finally:
        if commands is not sys.stdin:
            commands.close()
        from shm_manager._C import set_pinned_weight_cache_budget
        set_pinned_weight_cache_budget(0)
        registry.close()
        state.destroy_distributed_environment()


if __name__ == '__main__':
    main()
