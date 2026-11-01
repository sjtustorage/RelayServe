"""Live greedy decoder recovery from a temporary PP slot to its native TP.

Called by the serialized rank loop with the source topology activated. Source
KV remains owned until all destination ranks acknowledge the copy and install.
No model forward or prompt recomputation occurs in this operation.
"""
import copy
import math


def layerwise_handoff_complete(engine):
    return (getattr(engine, '_transition_complete', False)
            and getattr(engine, '_next_cpu_boundary', None) == 0
            and not getattr(engine, 'need_send_shm_model_output', True))


def _validate_weight_source(runtime, weights_ready, *, source=False):
    config = runtime.engine.vllm_config.load_config
    if config.load_format == 'safetensors':
        return
    method = getattr(config, 'load_method', None)
    if (config.load_format != 'shm' or method not in ('tokenwise', 'layerwise')
            or weights_ready is not True):
        raise ValueError('Recovery requires fully loaded SHM or safetensors weights')
    engine = runtime.engine
    if method == 'tokenwise' and engine._waiting_for_tokenwise_request():
        raise ValueError('Recovery cannot bypass a pending CPU tokenwise handoff')
    if method == 'layerwise':
        if source and not layerwise_handoff_complete(engine):
            raise ValueError('Recovery requires completed layerwise CPU handoff and PP transition')
        if not source:
            designated_pending = (getattr(engine, 'cold_start_request_id', None) is not None
                                  and not getattr(engine, 'cold_start_request_come', False))
            relay_pending = (getattr(engine, '_cpu_relay_step_active', False)
                             and (getattr(engine, '_next_cpu_boundary', None) != 0
                                  or getattr(engine, 'need_send_shm_model_output', True)))
            if designated_pending or relay_pending:
                raise ValueError('Recovery cannot bypass a pending CPU layerwise handoff')


def validate_recovery(source, target, source_topology, target_topology, *,
                      source_weights_ready=None, target_weights_ready=None):
    from vllm.v1.request import RequestStatus
    if (source_topology.pp <= 1 or target_topology.pp != 1
            or source_topology.tp != target_topology.tp
            or tuple(source_topology.ranks[:source_topology.tp]) != target_topology.ranks):
        raise ValueError('Recovery requires matching native TP at temporary PP stage zero')
    if not source.ready or source.engine is None:
        raise ValueError('Recovery source must be ready')
    if source.topology != source_topology:
        raise ValueError('Recovery source topology mismatch')
    config = source.engine.vllm_config
    _validate_weight_source(source, source_weights_ready, source=True)
    if config.speculative_config is not None:
        raise ValueError('Speculative requests cannot use greedy recovery')
    for request in source.engine.scheduler.requests.values():
        if (request.status not in (RequestStatus.RUNNING, RequestStatus.WAITING)
                or (request.status == RequestStatus.RUNNING and request.num_computed_tokens < 1)
                or (request.status == RequestStatus.WAITING and request.num_computed_tokens != 0)
                or request.has_encoder_inputs or request.lora_request is not None
                or request.use_structured_output or request.sampling_params is None
                or request.sampling_params.temperature != 0
                or request.spec_token_ids or request.num_output_placeholders):
            raise ValueError('Recovery supports synchronous greedy text requests')
    if target is not None:
        if target.engine is None:
            raise ValueError('Native destination must be initialized')
        _validate_weight_source(target, target_weights_ready)
        if target.engine.vllm_config.speculative_config is not None:
            raise ValueError('Speculative destination cannot use greedy recovery')
        if (not target.ready or target.model != source.model
                or target.topology != target_topology or target.sessions.retained
                or target.engine.scheduler.has_unfinished_requests()):
            raise ValueError('Native destination must be ready, matching and drained')
        if (target.engine.vllm_config.cache_config.block_size != config.cache_config.block_size
                or target.engine.model_config.max_model_len < source.engine.model_config.max_model_len):
            raise ValueError('Recovery cache block size/context mismatch')


def recover_requests(source, target):
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_pp_group, get_tp_group
    from vllm.v1.request import RequestStatus
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    group = get_pp_group()
    scheduler = source.engine.scheduler
    computed_tokens = {key: request.num_computed_tokens
                       for key, request in scheduler.requests.items()}
    runner = source.engine.model_executor.worker.model_runner
    source_cache = runner._build_kv_caches_dict()
    target_runner = target.engine.model_executor.worker.model_runner if target else None
    target_cache = target_runner._build_kv_caches_dict() if target else {}
    copied_elements = 0
    layout = lambda caches: {name: (tuple(tensor.shape[:1]) + tuple(tensor.shape[2:]),
                                    str(tensor.dtype)) for name, tensor in caches.items()}
    layouts = [None] * group.world_size
    dist.all_gather_object(layouts, (layout(source_cache), layout(target_cache)),
                           group=group.cpu_group)
    native_rank = dist.get_process_group_ranks(group.cpu_group).index(group.ranks[0])
    native_layout = layouts[native_rank][1]
    invalid_layout = any(name not in native_layout or native_layout[name] != value
                         for stage, _ in layouts for name, value in stage.items())
    covered = [name for stage, _ in layouts for name in stage]
    if invalid_layout or len(covered) != len(set(covered)) or set(covered) != set(native_layout):
        raise ValueError('PP source layers do not exactly match native KV layouts')
    allocated = []
    error = None
    destination = {}
    clones = {}
    if target is not None:
        try:
            manager = target.engine.scheduler.kv_cache_manager
            if manager.num_kv_cache_groups != 1:
                raise ValueError('Recovery requires one full-attention KV cache group')
            for request_id, request in scheduler.requests.items():
                clone = copy.deepcopy(request)
                clones[request_id] = clone
                if clone.status == RequestStatus.WAITING:
                    destination[request_id] = ([],)
                    continue
                # Allocate space for all committed tokens, including the last
                # sampled token whose KV will be computed by the next decode.
                computed = clone.num_computed_tokens
                clone.num_computed_tokens = 0
                blocks = manager.allocate_slots(clone, clone.num_tokens)
                clone.num_computed_tokens = computed
                if blocks is None:
                    raise ValueError('Native KV cache lacks recovery capacity')
                allocated.append(clone)
                destination[request_id] = manager.get_block_ids(request_id)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    tp = get_tp_group()
    lane_errors = [None] * tp.world_size
    dist.all_gather_object(lane_errors, error, group=tp.cpu_group)
    errors = [None] * group.world_size
    dist.all_gather_object(errors, lane_errors, group=group.cpu_group)
    if any(item for stage in errors for item in stage):
        if target is not None:
            for clone in allocated:
                target.engine.scheduler.kv_cache_manager.free(clone)
        raise ValueError(f'Recovery allocation rejected: {errors}')
    message = [destination if group.is_first_rank else None]
    dist.broadcast_object_list(message, src=group.first_rank, group=group.cpu_group)
    destination = message[0]
    # Every PP rank owns distinct layers. Physical source/destination block
    # numbers may differ; only logical request-token order is preserved.
    for stage in range(group.world_size):
        for request_id, request in scheduler.requests.items():
            count = math.ceil(request.num_computed_tokens /
                              source.engine.vllm_config.cache_config.block_size)
            if not count:
                continue
            source_blocks = scheduler.kv_cache_manager.get_block_ids(request_id)[0][:count]
            target_blocks = destination[request_id][0][:count]
            if group.rank_in_group == stage:
                tensors = {name: tensor[:, source_blocks].contiguous()
                           for name, tensor in source_cache.items()}
                if stage:
                    group.send_tensor_dict(tensors, dst=0)
            elif group.is_first_rank:
                tensors = group.recv_tensor_dict(src=stage)
            if group.is_first_rank:
                for name, tensor in tensors.items():
                    target_cache[name][:, target_blocks] = tensor
                    copied_elements += tensor.numel()
    torch.cuda.synchronize()
    group.barrier()
    if target is not None:
        dest_scheduler = target.engine.scheduler
        for request_id, clone in clones.items():
            if clone.status == RequestStatus.WAITING:
                dest_scheduler.add_request(clone)
                continue
            # Install as cached RUNNING requests so V1 never classifies an
            # already computed prefix as a new prefill. Runner state carries
            # the complete output history for token positions and penalties.
            clone.status = RequestStatus.RUNNING
            dest_scheduler.finished_req_ids.discard(request_id)
            target_runner.input_batch.remove_request(request_id)
            dest_scheduler.requests[request_id] = clone
            dest_scheduler.running.append(clone)
            target_runner.requests[request_id] = CachedRequestState(
                req_id=request_id, prompt_token_ids=clone.prompt_token_ids,
                mm_kwargs=[], mm_positions=[], mm_hashes=[],
                sampling_params=clone.sampling_params, pooling_params=None,
                generator=None, block_ids=destination[request_id],
                num_computed_tokens=clone.num_computed_tokens,
                output_token_ids=list(clone.output_token_ids), lora_request=None)
        target.sessions = copy.deepcopy(source.sessions)
        if (target.engine.vllm_config.load_config.load_format == 'shm'
                and target.engine.vllm_config.load_config.load_method == 'layerwise'):
            # The copied KV is now entirely GPU-owned. A fresh native
            # layerwise slot must never wait for another CPU scheduler batch.
            engine = target.engine
            target_runner.update_shm_scheduler_output(None)
            engine._next_cpu_boundary = 0
            engine.need_send_shm_model_output = False
            engine._cpu_relay_step_active = False
            engine.loaded_layers = set(engine.target_layers)
            engine.cold_start = True
    group.barrier()
    # Only after the native copy and state install are acknowledged may the
    # temporary scheduler release its blocks. No request is sampled twice.
    recovered_ids = sorted(source.sessions._requests)
    source.engine.abort_requests(list(scheduler.requests))
    from sllm.resident.outputs import RequestSessions
    source.sessions = RequestSessions()
    return dict(recovered_request_ids=recovered_ids,
                recovered_computed_tokens=computed_tokens,
                copied_kv_elements=copied_elements,
                prompt_recomputed=False)
