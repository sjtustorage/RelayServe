"""Synchronous resident model commands, serialized by RankLoop.

One slot per GPU rank. Every operation executes inside its prepared topology;
generations prevent queued requests from silently reaching a replacement model.
The command loop serializes requests and switches, so admission/draining is
implicit here. Concurrent serving will use ResidentModelSlot leases separately.
"""
import os
import math
import time


def prewarm_model_runtime():
    """Import model-independent engine dependencies before request admission.

    This builds no model config or engine and attaches no producer storage.
    Configuration validation remains mandatory for every load command.
    """
    from vllm.engine.arg_utils import EngineArgs
    from sllm.resident.executor import ResidentTPExecutor, bind_executor
    from sllm.resident.engine import ResidentEngineCore
    from vllm.v1.sample.logits_processor import _load_logitsprocs_plugins
    # Discovery is process-wide and cached; model-specific processor instances
    # are still constructed by the runner with each model's configuration.
    _load_logitsprocs_plugins()


class ResidentModel:
    def __init__(self, registry):
        self.registry = registry
        self.engine = None
        self.topology = None
        self.model = None
        self.generation = 0
        self.pending_config = None
        self.ready = False
        from sllm.resident.outputs import RequestSessions
        self.sessions = RequestSessions()

    def status(self):
        result = dict(model=self.model, generation=self.generation, ready=self.ready)
        config = getattr(getattr(self.engine, 'vllm_config', None), 'load_config', None)
        if getattr(config, 'load_format', None) == 'shm':
            asynchronous = getattr(config, 'load_async', None)
            if type(asynchronous) is bool:
                result['weight_loading_policy'] = 'async' if asynchronous else 'serial'
        return result

    def weights_ready(self):
        """Query copy AND unpack events without fencing the loading stream.

        Called with this model's runtime binding active. Engine readiness and
        the engine's execution-layer set do not establish weight readiness.
        Inspect the actual allocated tensors, including embedding/head storage.
        """
        if self.engine is None:
            return False
        if self.engine.vllm_config.load_config.load_format == 'safetensors':
            return True
        from shm_manager.torch import model_info
        allocated = getattr(model_info, 'per_layer_gpu_ptrs', None)
        if model_info.manager is None or not allocated:
            return False
        layers = sorted(allocated)
        return bool(model_info.manager.check_load_state(layers))

    def validate(self, command, topology):
        if self.registry.rank not in topology.ranks:
            return
        op = command['op']
        if op == 'generate' and self.sessions.pending:
            raise ValueError('Use incremental stepping while requests are active')
        if op in ('load', 'unload') and self.sessions.retained:
            raise ValueError('Drain active requests before changing the model')
        if op == 'load':
            validation_started = time.perf_counter()
            ratio = command.get('hydra_min_free_memory_ratio', 0.0)
            if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
                raise ValueError('hydra_min_free_memory_ratio must be a finite number in [0, 1]')
            if ratio and self.registry.rank in command.get('hydra_borrowed_ranks', []):
                import torch
                free, total = torch.cuda.mem_get_info(self.registry.device)
                if total > 0 and free / total < ratio:
                    # Unloaded resident models retain allocator-owned free
                    # segments for fast reuse. Reclaim only under pressure;
                    # live hot models/KV/graphs remain allocated.
                    torch.cuda.empty_cache()
                    free, total = torch.cuda.mem_get_info(self.registry.device)
                if total <= 0 or free / total < ratio:
                    raise ValueError(
                        'Insufficient free GPU memory for Hydra borrowing threshold: '
                        f'free={free} total={total} ratio={ratio}')
            required_free = command.get('required_free_bytes')
            rank_budgets = command.get('required_free_bytes_by_rank')
            if rank_budgets is not None:
                if (not isinstance(rank_budgets, dict)
                        or set(rank_budgets) != {str(rank) for rank in topology.ranks}
                        or any(type(value) is not int or value <= 0
                               for value in rank_budgets.values())):
                    raise ValueError('Invalid per-rank Hydra memory budgets')
                required_free = rank_budgets[str(self.registry.rank)]
            if required_free is not None:
                if type(required_free) is not int or required_free <= 0:
                    raise ValueError('required_free_bytes must be a positive integer')
                import torch
                free, _ = torch.cuda.mem_get_info(self.registry.device)
                if free < required_free:
                    # Compare the measured incremental budget with driver-free
                    # memory after releasing reusable allocator segments. Never
                    # double-credit cached bytes or unload another model.
                    torch.cuda.empty_cache()
                    free, _ = torch.cuda.mem_get_info(self.registry.device)
                if free < required_free:
                    raise ValueError(
                        'Insufficient free GPU memory for Hydra staging budget: '
                        f'free={free} required={required_free}')
            generation = command.get('generation', self.generation + 1)
            if type(generation) is not int or generation <= self.generation:
                raise ValueError('load generation must exceed the rank generation')
            if self.topology is not None and self.topology != topology:
                raise ValueError('Unload the old topology before changing membership')
            if not isinstance(command.get('model'), str) or not command['model']:
                raise ValueError('load requires a non-empty model path')
            load_format = command.get('load_format', 'safetensors')
            if load_format not in ('shm', 'safetensors'):
                raise ValueError('Unsupported resident load format')
            startup_mode = command.get('startup_mode', 'relay')
            if startup_mode not in ('relay', 'serverlessllm', 'hydraserve'):
                raise ValueError('Unsupported resident startup_mode')
            if startup_mode in ('serverlessllm', 'hydraserve') and load_format != 'shm':
                raise ValueError('Serial baseline requires cached SHM weights')
            max_model_len = command.get('max_model_len', 8192)
            if type(max_model_len) is not int or max_model_len < 1:
                raise ValueError('max_model_len must be a positive integer')
            chunk_tokens = command.get('max_num_batched_tokens', 4096)
            if type(chunk_tokens) is not int or chunk_tokens < 1:
                raise ValueError('max_num_batched_tokens must be a positive integer')
            cache_blocks = command.get('num_gpu_blocks_override')
            if cache_blocks is not None and (type(cache_blocks) is not int or cache_blocks <= 0):
                raise ValueError('num_gpu_blocks_override must be a positive integer')
            relay = {}
            if load_format == 'shm':
                method = command.get('load_method')
                if method not in ('layerwise', 'tokenwise'):
                    raise ValueError('SHM load requires tokenwise or layerwise')
                info = command.get('shm_info')
                if (not isinstance(info, list) or len(info) != 4 or
                        any(type(value) is not int or value <= 0 for value in info)):
                    raise ValueError('shm_info requires positive size, blocks, block length and CPU TP')
                size, blocks, block_len, cpu_tp = info
                if startup_mode in ('serverlessllm', 'hydraserve') and method != 'tokenwise':
                    raise ValueError('Serial baseline requires full GPU execution')
                relay = dict(load_method=method, lazy_load=True,
                    load_async=startup_mode == 'relay',
                    shm_kv_cache_size=size, shm_num_blocks=blocks,
                    shm_block_len=block_len, shm_tp_size=cpu_tp, block_size=32)
                if method == 'tokenwise':
                    relay['kv_transfer_config'] = dict(kv_connector='ShmConnector',
                                                       kv_role='kv_consumer')
            from vllm.engine.arg_utils import EngineArgs
            from sllm.resident.executor import ResidentTPExecutor, ResidentPPExecutor
            # Metadata/config validation precedes every device collective and
            # old-model teardown, and errors are gathered by the rank loop.
            self.pending_config = EngineArgs(
                model=command['model'], skip_tokenizer_init=True,
                dtype='bfloat16', tensor_parallel_size=topology.tp,
                pipeline_parallel_size=topology.pp,
                distributed_executor_backend=(ResidentPPExecutor if topology.pp > 1
                                              else ResidentTPExecutor),
                load_format=load_format, enforce_eager=True,
                max_model_len=max_model_len, max_num_batched_tokens=chunk_tokens,
                num_gpu_blocks_override=cache_blocks,
                enable_chunked_prefill=True,
                enable_prefix_caching=False,
                gpu_memory_utilization=command.get('gpu_memory_utilization',
                    0.5 if load_format == 'shm' else 0.3), seed=42,
                **relay).create_engine_config()
            if load_format == 'shm':
                self.pending_config.cache_config._relay_group_inventory = (
                    self.registry.memory_profile_inventory())
                from vllm.utils.relay_memory_profile import (
                    find_memory_profile, memory_profile_identity, read_non_kv_memory)
                profile_path = find_memory_profile(command['model'])
                if profile_path is not None:
                    # Every rank joins validation errors before GPU collectives.
                    # Freeze the validated value against file removal/replacement
                    # between preflight and determine_available_memory.
                    budget = read_non_kv_memory(profile_path, memory_profile_identity(
                        self.pending_config, self.registry.device))
                    self.pending_config.cache_config._relay_profile_non_kv_bytes = budget
            if os.environ.get('VLLM_PROFILE_RELAY'):
                print(f'[resident create] rank={self.registry.rank} '
                      f'config_validation={time.perf_counter() - validation_started:.6f}s',
                      flush=True)
        else:
            if self.engine is None or self.topology != topology:
                raise ValueError('Target model topology is not loaded')
            if (type(command.get('generation')) is not int or
                    command['generation'] != self.generation):
                raise ValueError('Stale or missing model generation')
            if op == 'profile_memory':
                if not self.ready:
                    raise ValueError('Memory calibration requires a ready model')
                config = self.engine.vllm_config
                if (config.load_config.load_format != 'shm'
                        or not config.model_config.enforce_eager):
                    raise ValueError('Memory calibration requires eager SHM execution')
            if op in ('step', 'collect') and not self.ready:
                raise ValueError('Model requires lazy_init before scheduling')
            if op == 'collect':
                self.sessions.validate_collection(command.get('request_id'))
            if op == 'abort':
                self.sessions.validate_cancellation(command.get('request_id'))
            if op == 'lazy_init':
                if self.ready or self.engine.vllm_config.load_config.load_format != 'shm':
                    raise ValueError('lazy_init requires an uninitialized SHM model')
                partitions = command.get('layer_idxes_by_pp')
                total = self.engine.total_layers
                from vllm.v1.worker.cold_start_transition import compute_pp_layer_ranges
                compute_pp_layer_ranges(total, topology.pp, partitions)
                layers = [layer for part in partitions for layer in part]
                if not layers:
                    raise ValueError('lazy_init requires a nonempty GPU suffix')
                if self.engine.vllm_config.load_config.load_method == 'tokenwise' and layers != list(range(total)):
                    raise ValueError('Tokenwise initialization requires all layers')
                request_id = command.get('request_id')
                if request_id is not None and (not isinstance(request_id, str) or not request_id):
                    raise ValueError('Invalid lazy_init request ID')
            if op in ('generate', 'submit'):
                if not self.ready:
                    raise ValueError('Model requires lazy_init before generation')
                if command.get('model') != self.model:
                    raise ValueError('Request model does not match resident model')
                prompt = command.get('prompt_token_ids')
                maximum = command.get('max_tokens', 8)
                if (not isinstance(prompt, list) or not prompt or
                        any(type(token) is not int or token < 0 or token >=
                            self.engine.model_config.get_vocab_size() for token in prompt)):
                    raise ValueError('Invalid prompt token IDs')
                if (type(maximum) is not int or maximum < 1 or
                        len(prompt) + maximum > self.engine.model_config.max_model_len):
                    raise ValueError('Invalid generation length')
                request_id = command.get('request_id')
                if request_id is not None and (not isinstance(request_id, str) or not request_id):
                    raise ValueError('Invalid request ID')
                if op == 'submit':
                    self.sessions.validate_submission(request_id or
                        f"resident-{self.generation}-{command['sequence']}")
                if (not self.engine.cold_start_request_come and
                        self.engine.cold_start_request_id is not None and
                        request_id != self.engine.cold_start_request_id):
                    raise ValueError('First generation must use the designated relay request ID')
                if 'kv_transfer_params' in command:
                    config = self.engine.vllm_config
                    params = command['kv_transfer_params']
                    if (config.kv_transfer_config is None or not isinstance(params, dict)
                            or params.get('do_remote_prefill') is not True):
                        raise ValueError('Invalid tokenwise transfer metadata')
                    count = params.get('num_computed_tokens')
                    blocks = params.get('remote_block_ids')
                    block_size = config.cache_config.block_size
                    if (type(count) is not int or not 0 < count <= len(prompt) or
                            not isinstance(blocks, list) or
                            len(blocks) < (count + block_size - 1) // block_size or
                            any(type(block) is not int or block < 0 or
                                block >= config.load_config.shm_num_blocks for block in blocks)):
                        raise ValueError('Invalid tokenwise cache range')

    def _step(self):
        # Optional calling-thread diagnostics, without CUDA profiler
        # setup or additional stream fences. Profile only the first
        # iteration of each instance; benchmark metadata must mark
        # these runs as diagnostic rather than latency baselines.
        profile_dir = os.environ.get('SLLM_PROFILE_FIRST_STEP')
        if profile_dir and getattr(self, '_profiled_generation', None) != self.generation:
            import cProfile
            from pathlib import Path
            directory = Path(profile_dir)
            directory.mkdir(parents=True, exist_ok=True)
            profiler = cProfile.Profile()
            try:
                with profiler:
                    result = self.engine.step()
            finally:
                profiler.dump_stats(str(directory /
                    f'first-step-{os.getpid()}-{self.generation}.prof'))
            self._profiled_generation = self.generation
        else:
            result = self.engine.step()
        return result

    def unload(self):
        if self.sessions.retained:
            raise ValueError('Drain active requests before changing the model')
        if self.engine is None:
            return
        with self.registry.activate(self.topology, retain_active=True):
            self.engine.shutdown()
        self.engine = None
        self.topology = None
        self.model = None
        self.ready = False
        from sllm.resident.outputs import RequestSessions
        self.sessions = RequestSessions()

    def execute(self, command, topology):
        if self.registry.rank not in topology.ranks:
            return dict(idle=True)
        from sllm.resident.executor import bind_executor, ResidentTPExecutor
        if command['op'] == 'unload':
            self.unload()
            return self.status()
        if command['op'] == 'weights_ready':
            return dict(**self.status(), weights_ready=self.weights_ready())
        if command['op'] == 'load':
            from sllm.resident.engine import ResidentEngineCore
            self.unload()
            start = time.perf_counter()
            with self.registry.activate(topology, retain_active=True), bind_executor(self.registry, topology):
                activated_at = time.perf_counter()
                executor = self.pending_config.parallel_config.distributed_executor_backend
                profile_dir = os.environ.get('SLLM_PROFILE_ENGINE_CREATE')
                if profile_dir:
                    import cProfile
                    from pathlib import Path
                    directory = Path(profile_dir)
                    directory.mkdir(parents=True, exist_ok=True)
                    profiler = cProfile.Profile()
                    try:
                        with profiler:
                            self.engine = ResidentEngineCore(self.pending_config, executor, False)
                    finally:
                        profiler.dump_stats(str(directory /
                            f'engine-create-{os.getpid()}-{command.get("generation", self.generation + 1)}.prof'))
                else:
                    self.engine = ResidentEngineCore(self.pending_config, executor, False)
                constructed_at = time.perf_counter()
            if os.environ.get('VLLM_PROFILE_RELAY'):
                print(f'[resident create] rank={self.registry.rank} '
                      f'activation={activated_at - start:.6f}s '
                      f'engine={constructed_at - activated_at:.6f}s '
                      f'context_exit={time.perf_counter() - constructed_at:.6f}s',
                      flush=True)
            self.pending_config = None
            self.topology = topology
            self.model = command['model']
            self.generation = command.get('generation', self.generation + 1)
            self.ready = self.engine.scheduler is not None
            return dict(**self.status(), load_seconds=time.perf_counter() - start)
        if command['op'] == 'lazy_init':
            start = time.perf_counter()
            with self.registry.activate(topology, retain_active=True):
                self.engine.lazy_load_model(command['layer_idxes_by_pp'],
                                            request_id=command['request_id'])
            self.ready = True
            ready_at = time.perf_counter()
            origin = getattr(self.engine.vllm_config, '_relay_initialization_started_at', None)
            return dict(**self.status(), lazy_init_seconds=ready_at - start,
                        vllm_initialization_seconds=(ready_at - origin
                            if origin is not None else None))
        if command['op'] == 'profile_memory':
            with self.registry.activate(topology, retain_active=True):
                profiles = self.engine.model_executor.collective_rpc(
                    'profile_shm_non_kv_memory')
            return dict(**self.status(), memory_profiles=profiles,
                        load_async=self.engine.vllm_config.load_config.load_async,
                        num_gpu_blocks=self.engine.vllm_config.cache_config.num_gpu_blocks)
        if command['op'] == 'step':
            with self.registry.activate(topology, retain_active=True):
                outputs, _ = self._step()
                token_updates = self.sessions.consume(outputs, time.perf_counter())
            from sllm.resident.migration import layerwise_handoff_complete
            config = self.engine.vllm_config.load_config
            recovery_ready = (config.load_format != 'shm' or config.load_method != 'layerwise'
                              or layerwise_handoff_complete(self.engine))
            return dict(**self.status(), pending=self.sessions.pending,
                        token_updates=token_updates,
                        recovery_ready=recovery_ready,
                        pp_transition_complete=self.engine._transition_complete,
                        first_output_request_ids=self.sessions.first_output_ids,
                        finished_request_ids=self.sessions.finished_ids)
        if command['op'] == 'collect':
            state = self.sessions.take_finished(command['request_id'])
            return dict(**self.status(), request_id=state.request_id,
                        token_ids=state.token_ids,
                        num_cached_tokens=state.num_cached_tokens,
                        first_output_seconds=state.first_output_seconds,
                        generation_started_at=state.submitted_at,
                        generation_finished_at=state.finished_at,
                        generation_seconds=state.finished_at - state.submitted_at,
                        pp_transition_complete=self.engine._transition_complete)
        if command['op'] == 'abort':
            request_id = command['request_id']
            with self.registry.activate(topology, retain_active=True):
                self.engine.abort_requests([request_id])
            state = self.sessions.cancel(request_id, expect_completion=(
                self.engine.scheduler.finished_req_ids_dict is not None))
            return dict(**self.status(), request_id=request_id, cancelled=True,
                        token_ids=state.token_ids)
        from vllm import SamplingParams
        from vllm.v1.request import Request
        request_id = command.get('request_id',
                                 f"resident-{self.generation}-{command['sequence']}")
        request = Request(request_id, command['prompt_token_ids'], None, None, None,
            SamplingParams(temperature=0, max_tokens=command.get('max_tokens', 8),
                           ignore_eos=True, extra_args={'kv_transfer_params':
                               command['kv_transfer_params']} if 'kv_transfer_params' in command else None),
            None, None)
        if command['op'] == 'submit':
            # Admission and stepping run on the same rank loop, so no engine
            # mutations race model teardown or distributed collectives.
            self.sessions.submit(request_id, time.perf_counter())
            try:
                with self.registry.activate(topology, retain_active=True):
                    self.engine.add_request(request)
            except BaseException:
                self.sessions.discard(request_id)
                raise
            return dict(**self.status(), request_id=request_id, submitted=True)
        from sllm.resident.outputs import RequestOutput
        collected = RequestOutput(request_id)
        start = time.perf_counter()
        with self.registry.activate(topology, retain_active=True):
            self.engine.add_request(request)
            while self.engine.scheduler.has_requests():
                outputs, _ = self._step()
                collected.consume(outputs, time.perf_counter() - start)
        end = time.perf_counter()
        return dict(**self.status(), request_id=request_id, token_ids=collected.token_ids,
                    num_cached_tokens=collected.num_cached_tokens,
                    first_output_seconds=collected.first_output_seconds,
                    generation_seconds=end - start,
                    generation_started_at=start, generation_finished_at=end,
                    pp_transition_complete=self.engine._transition_complete)
