"""Single-node TP/PP relay admission over resident GPU/CPU owners."""
from concurrent.futures import ThreadPoolExecutor, CancelledError, TimeoutError
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
import math
import queue
import threading
import time


def _timed_response(result, started, *, cpu_first_token_at=None, model_create_started_at=None, **details):
    """Measure token production from service entry, including queued work.

    All clocks belong to the local host. GPU generation timing excludes
    admission/initialization; a complete CPU prefix produces the first sample
    even if the GPU command later produces additional tokens.
    """
    first_at = cpu_first_token_at
    if first_at is None and result.get('first_output_seconds') is not None:
        first_at = result['generation_started_at'] + result['first_output_seconds']
    return dict(result, **details, request_started_at=started,
                first_token_seconds=first_at - started if first_at is not None else None,
                model_create_started_at=model_create_started_at,
                initialization_seconds=(details['gpu_ready'] - model_create_started_at
                    if model_create_started_at is not None
                    and details.get('gpu_ready') is not None else None),
                model_create_to_first_token_seconds=(first_at - model_create_started_at
                    if first_at is not None and model_create_started_at is not None else None),
                first_token_source='cpu' if cpu_first_token_at is not None else
                                   'gpu' if first_at is not None else None,
                relay_seconds=time.perf_counter() - started)


@dataclass
class RelayModel:
    producer: object
    tp: int
    prefix_tokens: int
    layers: int
    vocab_size: int
    max_model_len: int
    cpu_layers: int = 0
    planner: object = None
    instance: object = None
    failed: bool = False
    lock: object = field(default_factory=threading.Lock)
    pp: int = 1
    layer_idxes_by_pp: object = None
    pump: object = None
    pump_route_instance: object = None
    startup_mode: str = 'relay'
    hydra_slos: object = None
    recovery: object = None


class RelayService:
    """Own routes whose GPU instance may span physical nodes.

    A route uses one controller WORLD pool. Its PP stages may be placed on
    different physical nodes through ``physical_nodes_by_rank``. The CPU
    producer is the first PP stage's AMX owner. The stage is derived from the
    ordered topology, so CPU and planner layer partitions cannot diverge.
    """
    prefix = '__relay__:'

    def __init__(self, controller, *, admission_timeout=30.0):
        if (isinstance(admission_timeout, bool) or
                not math.isfinite(admission_timeout) or admission_timeout < 0):
            raise ValueError('Admission timeout must be finite and nonnegative')
        self.admission_timeout = admission_timeout
        if len(controller.pools) != 1:
            raise ValueError('Cross-node relay requires one global WORLD pool')
        self.controller = controller
        self.models = {}
        self._closing = threading.Event()
        self._recovering = threading.Event()

    def _ensure_available(self):
        if self._closing.is_set():
            raise RuntimeError('Relay service is closing')
        with self.controller._lock:
            quarantined = bool(self.controller._quarantined)
        if self._recovering.is_set() or quarantined:
            raise RuntimeError('Relay service is recovering through worker restart; retry a new request later')

    def close(self):
        """Stop route admission and join iteration owners before pool shutdown."""
        self._closing.set()
        errors = []
        for model in self.models.values():
            with model.lock:
                if model.pump is not None:
                    try:
                        model.pump.close()
                    except Exception as exc:
                        errors.append(exc)
                    finally:
                        model.pump = None
        if errors:
            raise RuntimeError('Relay iteration drain failed') from errors[0]

    def recover(self, pool_factory, producer_factory):
        """Recover this local service after device failure without replaying requests.

        Factories return started owners. Old GPU processes exit before any CPU
        SHM owner closes. All routes remain locked through replacement; failed
        requests keep their error and subsequent requests cold-start afresh.
        """
        node = next(iter(self.controller.pools))
        with ExitStack() as locks:
            for name in sorted(self.models):
                locks.enter_context(self.models[name].lock)
            if self._closing.is_set():
                raise RuntimeError('Relay service is closing')
            if node not in self.controller._quarantined:
                raise ValueError('Recovery requires a quarantined GPU node')
            for model in self.models.values():
                if model.pump is not None:
                    try:
                        model.pump.close()
                    except RuntimeError:
                        # The failed pump has joined its owners before raising.
                        pass
                    model.pump = None
            recovered = self.controller.restart_node(node, pool_factory)
            try:
                for model in self.models.values():
                    old = model.producer
                    if model.failed or getattr(old, '_failed', False) or getattr(old, '_closed', False):
                        old.close()
                        producer = producer_factory(old)
                        if (producer is old or producer.info is None
                                or producer.max_model_len != model.max_model_len):
                            if producer is not old:
                                producer.close()
                            raise ValueError('Recovery requires a fresh compatible CPU producer')
                        model.producer = producer
                    model.instance = model.pump_route_instance = None
                    model.recovery = None
                    model.failed = False
            except BaseException as exc:
                # Keep successfully replaced producers for the next attempt.
                # Admission remains fenced until every route is prepared.
                self.controller._quarantine(node, exc)
                raise
            return recovered

    def register(self, name, producer, *, tp, prefix_tokens, cpu_layers=0,
                 planner=None, pp=1, layer_idxes_by_pp=None, startup_mode='relay',
                 ttft_slo_ms=None, tpot_slo_ms=None, hydra_min_free_memory_ratio=0.0):
        self.controller._validate_hydra_memory_ratio(hydra_min_free_memory_ratio)
        if hydra_min_free_memory_ratio and startup_mode != 'hydraserve':
            raise ValueError('Hydra memory threshold requires hydraserve startup mode')
        if startup_mode not in ('relay', 'serverlessllm', 'hydraserve'):
            raise ValueError('Unsupported resident startup_mode')
        if startup_mode == 'hydraserve':
            from sllm.resident.hydra import HydraPlanner
            if (not isinstance(planner, HydraPlanner) or pp != 1 or cpu_layers
                    or prefix_tokens or layer_idxes_by_pp is not None):
                raise ValueError('Hydra requires a measured planner and native PP1 without CPU relay')
            for value in (ttft_slo_ms, tpot_slo_ms):
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Hydra SLOs must be finite positive milliseconds')
            if (planner.profile['model'] not in (str(producer.model_path),
                                                str(getattr(producer, 'checkpoint', None)))
                    or planner.profile['tp'] != tp
                    or planner.profile['max_model_len'] != producer.max_model_len):
                raise ValueError('Hydra profile must match producer model/TP/context')
            # A producer's SHM directory is ephemeral; bind a verified
            # checkpoint profile to this producer without mutating the profile.
            if planner.profile['model'] != str(producer.model_path):
                planner = HydraPlanner(dict(planner.profile, model=str(producer.model_path)))
        elif ttft_slo_ms is not None or tpot_slo_ms is not None:
            raise ValueError('Hydra SLOs require hydraserve startup mode')
        if startup_mode == 'serverlessllm' and (planner is not None or cpu_layers):
            raise ValueError('ServerlessLLM baseline cannot use CPU relay planning')
        if (not isinstance(name, str) or not name or name in self.models
                or type(tp) is not int or tp < 1
                or type(pp) is not int or pp < 1
                or type(prefix_tokens) is not int or prefix_tokens < 0):
            raise ValueError('Invalid relay registration')
        if any(model.producer is producer for model in self.models.values()):
            raise ValueError('Each producer requires one exclusive model route')
        if producer.info is None or producer.model_path is None:
            raise ValueError('CPU producer must be started before registration')
        config = json.loads((producer.model_path / 'config.json').read_text())
        if type(cpu_layers) is not int or not 0 <= cpu_layers < config['num_hidden_layers']:
            raise ValueError('Invalid CPU layer partition')
        if pp > config['num_hidden_layers'] - cpu_layers:
            raise ValueError('Every PP stage requires at least one GPU layer')
        from sllm.resident.planner import ResidentPlanner
        if isinstance(planner, ResidentPlanner):
            pool = next(iter(self.controller.pools.values()))
            policy = getattr(pool, 'copy_throttle', None)
            if (getattr(pool, 'copy_throttle_first_only', False) !=
                    getattr(planner, 'copy_throttle_first_only', False)):
                raise ValueError('Planner first-only copy policy does not match GPU pool')
            if policy is not None and policy != planner.copy_throttle:
                raise ValueError('Planner copy throttle policy does not match GPU pool')
            if (policy and getattr(pool, 'copy_throttle_tasks', 32) !=
                    getattr(planner, 'copy_throttle_tasks', 32)):
                raise ValueError('Planner copy throttle tasks do not match GPU pool')
        if pp > 1 and isinstance(planner, ResidentPlanner):
            raise ValueError('ResidentPlanner currently supports PP=1; provide a PP planner')
        model = RelayModel(producer, tp, prefix_tokens,
            config['num_hidden_layers'], config['vocab_size'], producer.max_model_len,
            cpu_layers, planner, pp=pp, layer_idxes_by_pp=layer_idxes_by_pp,
            startup_mode=startup_mode)
        if startup_mode == 'hydraserve':
            model.hydra_slos = dict(ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
                                   hydra_min_free_memory_ratio=hydra_min_free_memory_ratio)
        self._partitions(model, cpu_layers, None)
        self.models[name] = model

    @staticmethod
    def _partitions(model, cpu_layers, plan):
        if type(cpu_layers) is not int or not 0 <= cpu_layers < model.layers:
            raise ValueError('Invalid planned CPU layer partition')
        partitions = (plan.get('layer_idxes_by_pp') if plan else None)
        if partitions is None:
            partitions = model.layer_idxes_by_pp
        if partitions is None:
            count, remainder = divmod(model.layers - cpu_layers, model.pp)
            partitions, start = [], cpu_layers
            for stage in range(model.pp):
                end = start + count + (stage < remainder)
                partitions.append(list(range(start, end)))
                start = end
        if (not isinstance(partitions, (list, tuple)) or len(partitions) != model.pp
                or any(not isinstance(stage, (list, tuple)) or not stage
                       or any(type(layer) is not int for layer in stage)
                       for stage in partitions)
                or [layer for stage in partitions for layer in stage]
                   != list(range(cpu_layers, model.layers))):
            raise ValueError('PP partitions must cover the contiguous GPU suffix exactly')
        return [list(stage) for stage in partitions]

    def validate_request(self, name, prompt_token_ids, max_tokens=8):
        self._ensure_available()
        if name not in self.models:
            raise ValueError('Unknown relay model')
        model = self.models[name]
        if (not isinstance(prompt_token_ids, list) or not prompt_token_ids or
                any(type(token) is not int or not 0 <= token < model.vocab_size for token in prompt_token_ids)
                or type(max_tokens) is not int or max_tokens < 1
                or len(prompt_token_ids) + max_tokens > model.max_model_len):
            raise ValueError('Invalid relay prompt or generation length')
        return model

    def stream(self, name, prompt_token_ids, max_tokens=8):
        """Emit CPU prefill and incremental GPU decode without replaying tokens.

        Closing the iterator cancels warm distributed work. Cold handoffs
        drain before the owner thread exits to preserve CPU/GPU KV ownership.
        """
        self.validate_request(name, prompt_token_ids, max_tokens)
        events = queue.Queue()
        cancelled = threading.Event()
        def run():
            emitted = []
            try:
                def emit(tokens, source):
                    emitted.extend(tokens)
                    for token in tokens:
                        if not cancelled.is_set():
                            events.put(dict(event='tokens', token_ids=[token], source=source,
                                            emitted_at=time.perf_counter()))
                result = self.generate(name, prompt_token_ids, max_tokens,
                                       on_cpu_tokens=lambda tokens: emit(tokens, 'cpu'),
                                       on_gpu_tokens=lambda tokens: emit(tokens, 'gpu'),
                                       cancel_event=cancelled)
                if result['token_ids'] != emitted:
                    raise RuntimeError('Streamed output differs from completed relay')
                events.put(dict(event='done', result=result))
            except Exception as exc:
                events.put(dict(event='error', error=f'{type(exc).__name__}: {exc}'))
            finally:
                events.put(None)
        thread = threading.Thread(target=run, name='relay-stream')
        thread.start()
        try:
            while True:
                try:
                    event = events.get(timeout=1)
                except queue.Empty:
                    yield dict(event='heartbeat')
                    continue
                if event is None:
                    break
                yield event
        finally:
            cancelled.set()
            thread.join()

    def generate(self, name, prompt_token_ids, max_tokens=8, *, on_cpu_tokens=None,
                 on_gpu_tokens=None, cancel_event=None):
        model = self.validate_request(name, prompt_token_ids, max_tokens)
        started = time.perf_counter()
        pending = None
        with self.controller.request_admission(self.prefix + name):
            with model.lock:
                self._ensure_available()
                if model.failed:
                    raise RuntimeError('Relay model requires service restart after failed handoff')
                if model.instance is not None:
                    from sllm.resident.pump import InstancePump
                    try:
                        if (model.pump is not None and model.pump_route_instance is not None
                                and model.instance != model.pump_route_instance):
                            # An explicit route replacement is distinct from
                            # recovery performed internally by the pump.
                            model.pump.close()
                            model.pump = None
                        if model.pump is not None:
                            # submit pins the current generation under the same
                            # mutex as PP recovery. The route's cached handle
                            # can still refer to the released temporary source.
                            pending = model.pump.submit(prompt_token_ids,
                                request_id=f'relay-warm-{time.monotonic_ns()}',
                                max_tokens=max_tokens, **(
                                    {'on_tokens': on_gpu_tokens} if on_gpu_tokens is not None else {}))
                            model.instance = model.pump.instance
                            model.pump_route_instance = model.instance
                        else:
                            with self.controller.instance_admission(model.instance):
                                model.pump = InstancePump(self.controller, model.instance)
                                model.pump_route_instance = model.instance
                                pending = model.pump.submit(prompt_token_ids,
                                    request_id=f'relay-warm-{time.monotonic_ns()}',
                                    max_tokens=max_tokens, **(
                                        {'on_tokens': on_gpu_tokens} if on_gpu_tokens is not None else {}))
                    except ValueError:
                        # Admission failed before queueing: the old generation
                        # has expired. Execution errors propagate from result().
                        if model.pump is not None:
                            model.pump.close()
                            model.pump = None
                        model.instance = None
            if pending is not None:
                if cancel_event is not None:
                    while True:
                        if cancel_event.is_set():
                            pending.cancel()
                            raise CancelledError('Relay client disconnected')
                        try:
                            return _timed_response(pending.result(timeout=.05), started, cold=False)
                        except TimeoutError:
                            if pending.done():
                                return _timed_response(pending.result(), started, cold=False)
                return _timed_response(pending.result(), started, cold=False)
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError('Relay client disconnected before cold start')
            if model.startup_mode == 'hydraserve':
                return self._generate_hydra(name, prompt_token_ids, max_tokens, started,
                                            on_gpu_tokens=on_gpu_tokens, cancel_event=cancel_event)
            return self._generate_serial(name, prompt_token_ids, max_tokens,
                                         on_cpu_tokens=on_cpu_tokens, on_gpu_tokens=on_gpu_tokens)

    def _generate_hydra(self, name, prompt, max_tokens, started, *, on_gpu_tokens=None,
                        cancel_event=None):
        from sllm.resident.pump import InstancePump
        model = self.models[name]
        cold = False
        created = ready = None
        decision = None
        with model.lock:
            self._ensure_available()
            if model.instance is None:
                cold = True
                native = stage = None
                options = dict(load_format='shm', load_method='tokenwise',
                    shm_info=model.producer.info['shm_info'],
                    max_model_len=model.max_model_len,
                    max_num_batched_tokens=getattr(model.producer, 'max_num_batched_tokens', 4096))
                try:
                    native = self.controller.create(self.prefix + name,
                        str(model.producer.model_path), model.tp, reserve_only=True,
                        admission_timeout=self.admission_timeout)
                    # Reserve native capacity through temporary initialization.
                    with ExitStack() as native_admissions:
                        native_admissions.enter_context(self.controller.instance_admission(native))
                        created = time.perf_counter()
                        with self.controller.hydra_lenders(native) as lenders:
                            stage, decision = self.controller.promote_hydra(
                                self.prefix + name + ':hydra', native, model.planner,
                                len(prompt), borrow_from=lenders, **model.hydra_slos, **options)
                        if stage is native:
                            stage = self.controller.materialize(native,
                                startup_mode='hydraserve', **options)
                        with self.controller.instance_admission(stage):
                            if not stage.ready:
                                count, remainder = divmod(model.layers, stage.pp_size)
                                partitions, offset = [], 0
                                for index in range(stage.pp_size):
                                    end = offset + count + (index < remainder)
                                    partitions.append(list(range(offset, end)))
                                    offset = end
                                stage = self.controller.lazy_init(stage,
                                    partitions[0] if stage.pp_size == 1 else partitions, None)
                            ready = time.perf_counter()
                            model.pump = InstancePump(self.controller, stage)
                            model.instance = model.pump_route_instance = stage
                            pending = model.pump.submit(prompt,
                                request_id=f'hydra-{time.monotonic_ns()}', max_tokens=max_tokens,
                                **({'on_tokens': on_gpu_tokens} if on_gpu_tokens is not None else {}))
                        if stage.pp_size > 1:
                            # Match pump admission lock ordering and transfer
                            # the native pin atomically against idle reaping.
                            with model.pump._condition, self.controller._lock:
                                native_admissions.close()
                                model.recovery = model.pump.materialize_and_recover(
                                    native, list(range(model.layers)),
                                    after_first_output=True,
                                    startup_mode='hydraserve', **options)
                except BaseException as startup_error:
                    if model.pump is not None:
                        try:
                            model.pump.close()
                        except BaseException as cleanup_error:
                            startup_error.add_note(f'Hydra pump cleanup failed: {cleanup_error}')
                            model.failed = True
                        model.pump = None
                    model.instance = None
                    for instance in (stage, native):
                        if instance is not None:
                            try:
                                self.controller.release(instance)
                            except ValueError:
                                pass
                            except BaseException as cleanup_error:
                                startup_error.add_note(f'Hydra instance cleanup failed: {cleanup_error}')
                                model.failed = True
                    raise
            else:
                if model.pump is None:
                    model.pump = InstancePump(self.controller, model.instance)
                pending = model.pump.submit(prompt,
                    request_id=f'hydra-{time.monotonic_ns()}', max_tokens=max_tokens,
                    **({'on_tokens': on_gpu_tokens} if on_gpu_tokens is not None else {}))
                model.instance = model.pump_route_instance = model.pump.instance
        while True:
            if cancel_event is not None and cancel_event.is_set():
                pending.cancel()
                raise CancelledError('Hydra client disconnected')
            try:
                result = pending.result(timeout=.05)
                break
            except TimeoutError:
                if pending.done():
                    result = pending.result()
                    break
        return _timed_response(result, started, cold=cold, startup_mode='hydraserve',
            amx_used=False, plan=decision, model_create_started_at=created, gpu_ready=ready,
            vllm_initialization_seconds=(
                getattr(stage, 'vllm_initialization_seconds', None) if cold else None))

    def _generate_serial(self, name, prompt_token_ids, max_tokens=8, *, on_cpu_tokens=None,
                         on_gpu_tokens=None):
        model = self.validate_request(name, prompt_token_ids, max_tokens)
        prompt = list(prompt_token_ids)
        started = time.perf_counter()
        with self.controller.request_admission(self.prefix + name), model.lock:
            self._ensure_available()
            if model.failed:
                raise RuntimeError('Relay model requires service restart after failed handoff')
            if model.instance is not None:
                entered = False
                try:
                    # Resolve and reserve atomically against idle reclamation.
                    with self.controller._instance_operation(
                            name=model.instance.name, generation=model.instance.generation) as instance:
                        entered = True
                        result = self.controller.generate(instance, prompt, max_tokens=max_tokens,
                            **({'on_tokens': on_gpu_tokens} if on_gpu_tokens is not None else {}))
                except ValueError:
                    if entered:
                        raise
                    model.instance = None
                else:
                    return _timed_response(result, started, cold=False)
            from sllm.resident.planner import ResidentPlanner
            phase_planner = isinstance(model.planner, ResidentPlanner)
            plan = model.planner(len(prompt)) if model.planner is not None and not phase_planner else None
            if model.startup_mode == 'serverlessllm':
                plan = dict(cpu_layers=0, cpu_prefix_tokens=0, method='gpu',
                            baseline='serverlessllm')
            cpu_layers = plan['cpu_layers'] if plan else model.cpu_layers
            prefix = plan['cpu_prefix_tokens'] if plan else min(model.prefix_tokens, len(prompt))
            partitions = self._partitions(model, cpu_layers, plan)
            producer = model.producer
            executor = ThreadPoolExecutor(1, thread_name_prefix='relay-cpu')
            future = None
            model_create_started_at = None
            cpu_node = None
            amx_used = False
            def cold_response(result, **details):
                return _timed_response(result, started,
                    model_create_started_at=model_create_started_at,
                    vllm_initialization_seconds=getattr(
                        model.instance, 'vllm_initialization_seconds', None),
                    amx_used=amx_used, startup_mode=model.startup_mode, **details)
            try:
                try:
                    layerwise = cpu_layers > 0
                    # The service lock owns this producer session through GPU
                    # acknowledgment. Both engines use the same request ID.
                    request_id = f'relay-{time.monotonic_ns()}'
                    def cpu_prefill():
                        handoff = producer.command('generate', prompt_token_ids=prompt[:prefix], max_tokens=1)
                        if (handoff['kv_transfer_params']['num_computed_tokens'] != prefix
                                or len(handoff['token_ids']) != 1):
                            raise RuntimeError('CPU producer returned invalid prefill result')
                        if on_cpu_tokens is not None and prefix == len(prompt):
                            on_cpu_tokens(handoff['token_ids'])
                        return handoff
                    def start_cpu(instance):
                        nonlocal future, plan, cpu_layers, prefix, layerwise, model_create_started_at, partitions, amx_used, cpu_node
                        cpu_node = instance.node
                        if phase_planner:
                            plan = model.planner(len(prompt), initialization_phase=instance.initialization_phase,
                                max_num_batched_tokens=getattr(producer, 'max_num_batched_tokens', 4096))
                            cpu_layers = plan['cpu_layers']
                            prefix = plan['cpu_prefix_tokens']
                            layerwise = cpu_layers > 0
                            partitions = self._partitions(model, cpu_layers, plan)
                        # CPU is the first PP stage. The controller's ordered
                        # placement keeps its first TP ranks on one node.
                        first_stage_node = self.controller.physical_nodes[instance.node][instance.ranks[0]]
                        lease = self.controller.amx_lease(
                            instance, physical_node=first_stage_node)
                        amx_used = (layerwise or prefix > 0) and lease.acquire(blocking=False)
                        if not amx_used:
                            # Full GPU execution must cover the CPU prefix too.
                            # A registered uneven suffix is a relay plan, not a
                            # valid full-model partition when CPU is bypassed.
                            # Keep the solver candidate for diagnostics, but
                            # explicitly publish the effective execution plan;
                            # callers must never mistake a bypassed candidate
                            # for CPU work that actually ran.
                            if plan is not None:
                                plan = dict(plan)
                                plan.update(amx_bypassed=True,
                                            effective_method='gpu',
                                            effective_cpu_layers=0,
                                            effective_cpu_prefix_tokens=0,
                                            effective_execution_layers=list(range(model.layers)))
                            cpu_layers, prefix, layerwise = 0, 0, False
                            count, remainder = divmod(model.layers, model.pp)
                            partitions, start = [], 0
                            for stage in range(model.pp):
                                end = start + count + (stage < remainder)
                                partitions.append(list(range(start, end)))
                                start = end
                        # Admission and planner selection have finished. Model
                        # creation and CPU work begin on this shared timeline.
                        model_create_started_at = time.perf_counter()
                        try:
                            future = (executor.submit(producer.command, 'layerwise',
                                prompt_token_ids=prompt, max_tokens=max_tokens,
                                cpu_layers=cpu_layers, request_id=request_id) if layerwise else
                                executor.submit(cpu_prefill) if prefix else None)
                        except BaseException:
                            if amx_used:
                                lease.release()
                            raise
                        if amx_used:
                            # The lease covers computation, including a CPU
                            # command still awaiting feedback after GPU failure.
                            future.add_done_callback(lambda completed: lease.release())
                        return {'load_method': 'layerwise' if layerwise else 'tokenwise'}
                    create_started = time.perf_counter()
                    instance = self.controller.create(self.prefix + name, str(producer.model_path), model.tp,
                        **({'pp': model.pp} if model.pp != 1 else {}),
                        admission_timeout=self.admission_timeout,
                        load_format='shm', load_method='layerwise' if layerwise else 'tokenwise',
                        startup_mode=model.startup_mode,
                        shm_info=producer.info['shm_info'],
                        max_model_len=model.max_model_len, on_reserved=start_cpu,
                        max_num_batched_tokens=getattr(producer, 'max_num_batched_tokens', 4096))
                    create_seconds = time.perf_counter() - create_started
                    model.instance = instance
                    with self.controller._instance_operation(instance):
                        cpu_only = not layerwise and prefix == len(prompt) and max_tokens == 1
                        lazy_started = time.perf_counter()
                        instance = self.controller.lazy_init(instance,
                                                             partitions[0] if model.pp == 1 else partitions,
                                                             None if cpu_only or future is None else request_id)
                        model.instance = instance
                        gpu_ready = time.perf_counter()
                        lazy_seconds = gpu_ready - lazy_started
                        # A failed CPU prefix cannot publish the state that a
                        # layerwise GPU request waits for. Surface an already
                        # completed failure before entering that handshake.
                        if future is not None and future.done():
                            future.result()
                        if future is None:
                            result = self.controller.generate(instance, prompt, max_tokens=max_tokens,
                                **({'on_tokens': on_gpu_tokens} if on_gpu_tokens is not None else {}))
                            return cold_response(result, cold=True, plan=plan, cpu_prefix_tokens=0,
                                cpu_layers=0, create_seconds=create_seconds, lazy_seconds=lazy_seconds,
                                gpu_ready=gpu_ready)
                        if layerwise:
                            generating = threading.Event()
                            generating.set()
                            def cpu_failed_during_generation(completed):
                                if (generating.is_set() and not completed.cancelled()
                                        and completed.exception() is not None):
                                    # No CPU state will arrive. A GPU command
                                    # already waiting on it cannot be recycled.
                                    self.controller._quarantine(instance.node, completed.exception())
                                    abort = getattr(self.controller.pools[instance.node],
                                                    'abort_pending', None)
                                    if abort is not None:
                                        abort()
                            future.add_done_callback(cpu_failed_during_generation)
                            try:
                                result = self.controller.generate(instance, prompt, max_tokens=max_tokens,
                                    request_id=request_id,
                                    **({'on_tokens': on_gpu_tokens} if on_gpu_tokens is not None else {}))
                            except BaseException as gpu_error:
                                if future.done() and not future.cancelled() and future.exception() is not None:
                                    cpu_error = future.exception()
                                    cpu_error.add_note(f'GPU relay interrupted: {gpu_error}')
                                    raise cpu_error from gpu_error
                                raise
                            finally:
                                generating.clear()
                            handoff = future.result()
                            return cold_response(result, cold=True, plan=plan, cpu_layers=cpu_layers,
                                cpu_session_seconds=handoff['seconds'],
                                cpu_timing_scope='session_including_gpu_feedback',
                                cpu_seconds=handoff['seconds'], cpu_start=handoff['started_at'],
                                cpu_end=handoff['finished_at'], gpu_ready=gpu_ready,
                                create_seconds=create_seconds, lazy_seconds=lazy_seconds)
                        handoff = future.result()
                        metadata = handoff['kv_transfer_params']
                        if metadata['num_computed_tokens'] != prefix:
                            raise RuntimeError('CPU producer returned incorrect prefix length')
                        # A full-prefix CPU result contains the first output
                        # token. A partial-prefix sample must be discarded.
                        cpu_tokens = handoff['token_ids'] if prefix == len(prompt) else []
                        remaining = max_tokens - len(cpu_tokens)
                        if remaining:
                            result = self.controller.generate(instance, prompt + cpu_tokens,
                                max_tokens=remaining, request_id=request_id,
                                kv_transfer_params=metadata,
                                **({'on_tokens': on_gpu_tokens} if on_gpu_tokens is not None else {}))
                            if result['num_cached_tokens'] != prefix:
                                raise RuntimeError('GPU did not import the planned CPU prefix')
                            tokens = cpu_tokens + result['token_ids']
                        else:
                            # The CPU has already satisfied this request.  Do
                            # not issue a dummy GPU decode merely to establish
                            # a warm gate: initialization deliberately omitted
                            # the designated cold request ID, so ordinary warm
                            # requests are admitted directly.
                            result = dict(model=instance.model, generation=instance.generation,
                                          ready=True, request_id=handoff['request_id'],
                                          num_cached_tokens=0)
                            tokens = cpu_tokens
                        producer.command('release', request_id=handoff['request_id'])
                        return cold_response(result,
                            cpu_first_token_at=handoff['finished_at'] if cpu_tokens else None,
                            token_ids=tokens, cold=True, plan=plan,
                            cpu_prefix_tokens=prefix, cpu_seconds=handoff['seconds'],
                            cpu_session_seconds=handoff['seconds'],
                            cpu_timing_scope='prefill_request',
                            cpu_start=handoff['started_at'], cpu_end=handoff['finished_at'],
                            gpu_ready=gpu_ready, create_seconds=create_seconds,
                            lazy_seconds=lazy_seconds)
                finally:
                    # A layerwise CPU command may be waiting for feedback
                    # from a failed GPU. Do not join it while reporting that
                    # failure; its client deadline bounds the pending command.
                    # Keep its SHM owner alive until GPU pool shutdown.
                    executor.shutdown(wait=False, cancel_futures=True)
            except BaseException as startup_error:
                if (cpu_node in self.controller._quarantined
                        and future is not None and not future.done()):
                    # A lost GPU cannot send the next layerwise feedback.
                    # Stop actual CPU computation before its Future releases
                    # the node's AMX lease; retain the producer's SHM owner
                    # until the failed GPU pool has been shut down.
                    abort = getattr(producer, 'abort_pending', None)
                    if abort is not None:
                        try:
                            abort()
                        except BaseException as cleanup_error:
                            startup_error.add_note(f'CPU handoff abort failed: {cleanup_error}')
                # Never reuse a producer whose retained KV may still be read
                # by a failed/uncertain GPU command. Pool shutdown precedes
                # producer teardown in the service owner's ExitStack.
                if future is not None or model.instance is not None:
                    model.failed = True
                if (future is not None and future.done() and not future.cancelled()
                        and future.exception() is not None and model.instance is not None
                        and model.instance.node not in self.controller._quarantined):
                    # The operation pin has unwound. Unload joins background
                    # weight loading before releasing this GPU reservation.
                    # The failed producer itself remains unavailable.
                    try:
                        self.controller.release(model.instance)
                        model.instance = None
                    except BaseException as cleanup_error:
                        startup_error.add_note(f'CPU failure GPU cleanup failed: {cleanup_error}')
                raise
