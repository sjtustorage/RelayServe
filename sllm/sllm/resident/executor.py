"""Synchronous V1 executors using activated resident TP/PP topologies.

Each TP rank hosts a deterministic EngineCore scheduler, like vLLM's external
launcher. No worker subprocess or communication group is created here. The
controller must execute identical commands on every participating rank and
hold registry.activate for construction, steps and teardown. The PP variant
broadcasts stage-zero schedules and final-stage output to keep replicas aligned.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import gc

import torch
import torch.distributed as dist

from vllm.config import set_current_vllm_config
from vllm.distributed import get_pp_group, get_tp_group
from vllm.v1.executor.abstract import Executor
from vllm.v1.worker.gpu_worker import Worker


_RESIDENT = ContextVar('resident_executor_binding', default=None)


@contextmanager
def bind_executor(registry, topology):
    token = _RESIDENT.set((registry, topology))
    try:
        yield
    finally:
        _RESIDENT.reset(token)


class ResidentTPExecutor(Executor):
    uses_ray = False
    supports_pp = False

    def _init_executor(self):
        binding = _RESIDENT.get()
        if binding is None:
            raise RuntimeError('Resident executor requires controller binding')
        registry, topology = binding
        if not self.supports_pp and (topology.pp != 1 or get_pp_group().world_size != 1):
            raise ValueError('ResidentTPExecutor currently requires PP=1')
        stage = topology.ranks.index(registry.rank) // topology.tp
        if (get_pp_group().world_size != topology.pp or
                get_tp_group().ranks != list(topology.ranks[stage * topology.tp:(stage + 1) * topology.tp])):
            raise ValueError('Activate the bound resident topology first')
        self.worker = None
        self._closed = False
        try:
            self.worker = Worker(self.vllm_config, registry.device.index,
                topology.ranks.index(registry.rank), 'env://', True)
            with set_current_vllm_config(self.vllm_config):
                self.worker.init_device_with_prepared_groups()
                if self.load_config.load_format == 'shm':
                    self.worker.init_model_info()
                    if self.load_config.load_method != 'gpu':
                        self.worker.init_shm_kv_cache_info()
                if not self.load_config.lazy_load:
                    self.worker.load_model()
        except BaseException:
            self.shutdown()
            raise

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        if self._closed or self.worker is None:
            raise RuntimeError('Resident executor is closed')
        if not isinstance(method, str):
            raise ValueError('Resident RPC requires a named worker method')
        with set_current_vllm_config(self.vllm_config):
            result = getattr(self.worker, method)(*args, **(kwargs or {}))
        if method in ('get_ready_layerwise_boundary', 'check_uniform_weights_ready'):
            # Physical completion can differ across ranks. All schedulers
            # must choose the same next CPU boundary and collective sequence.
            results = [None] * get_tp_group().world_size
            dist.all_gather_object(results, result, group=get_tp_group().cpu_group)
            return results
        return [result]

    def initialize_from_config(self, kv_cache_configs, no_warmup=False):
        self.collective_rpc('initialize_from_config', args=(kv_cache_configs[0],))
        if not no_warmup:
            self.collective_rpc('compile_or_warm_up_model')

    def determine_available_memory(self):
        value = super().determine_available_memory()[0]
        memory = torch.tensor(value, dtype=torch.int64, device='cpu')
        dist.all_reduce(memory, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
        return [memory.item()]

    def check_health(self):
        if self._closed:
            raise RuntimeError('Resident executor is closed')

    def shutdown(self):
        if getattr(self, '_closed', True):
            return
        self._closed = True
        if self.load_config.load_format == 'shm':
            from shm_manager.torch import reset_relay_info
            reset_relay_info()
            from vllm.distributed.kv_transfer.kv_transfer_state import reset_shm_kv_transfer
            reset_shm_kv_transfer()
        torch.cuda.synchronize()
        self.worker = None
        # Attention modules are retained by the config's forward registry.
        self.vllm_config.compilation_config.static_forward_context.clear()
        # get_rope caches actual modules (including device tensor buffers),
        # not just configuration metadata. A new model must construct its own
        # modules even when it has the same rotary configuration as the old one.
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
        _ROPE_DICT.clear()
        gc.collect()
        # Keep only allocator-owned free blocks for the next model. Live model
        # tensors and forward registrations have been released above. Flushing
        # the cache here forces subsequent allocations through cudaMalloc while
        # asynchronous weight transfers are running.


class ResidentPPExecutor(ResidentTPExecutor):
    """Serialized PP execution with stage-zero scheduling and output feedback.

    EngineCore replicas consume the same scheduler output and final-stage
    result. PP lanes communicate only inside their prepared model membership;
    idle pool ranks and other model instances do not join these collectives.
    """
    supports_pp = True

    def _stage_values(self, value):
        group = get_pp_group()
        values = [None] * group.world_size
        dist.all_gather_object(values, value, group=group.cpu_group)
        backend_ranks = dist.get_process_group_ranks(group.cpu_group)
        return [values[backend_ranks.index(rank)] for rank in group.ranks]

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        results = super().collective_rpc(method, timeout, args, kwargs)
        if method in ('get_ready_layerwise_boundary', 'check_uniform_weights_ready'):
            # A transition runs NCCL collectives across PP stages. A TP-local
            # readiness decision can send only some stages into migration.
            return [result for stage in self._stage_values(results)
                    for result in stage]
        return results

    def get_kv_cache_specs(self):
        return self._stage_values(super().get_kv_cache_specs()[0])

    def determine_available_memory(self):
        return self._stage_values(super().determine_available_memory()[0])

    def initialize_from_config(self, kv_cache_configs, no_warmup=False):
        self.collective_rpc('initialize_from_config',
                            args=(kv_cache_configs[get_pp_group().rank_in_group],))
        if not no_warmup:
            self.collective_rpc('compile_or_warm_up_model')

    def execute_model(self, scheduler_output):
        group = get_pp_group()
        schedule = [scheduler_output if group.is_first_rank else None]
        dist.broadcast_object_list(schedule, src=group.first_rank, group=group.cpu_group)
        result = super().execute_model(schedule[0])
        output = [result if group.is_last_rank else None]
        dist.broadcast_object_list(output, src=group.ranks[-1], group=group.cpu_group)
        return output[0]
