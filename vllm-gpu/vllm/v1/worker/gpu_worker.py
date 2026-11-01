# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""
import copy
import gc
from multiprocessing import shared_memory
import os
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Optional, List

import torch
import torch.distributed
import torch.nn as nn

import time
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import (ensure_kv_transfer_initialized,
                                            sync_kv_transfer_connector_vllm_config)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.model_executor.models._cached_model_info import _MODEL_DEFAULT_NON_KV_CACHE_MEMORY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils import GiB_bytes, MemorySnapshot, memory_profiling
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, DraftTokenIds,
                             ModelRunnerOutput)
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

# Engine workers are reconstructed on model switches inside retained rank
# processes. Keep this diagnostic state at process scope, not on Worker.
_lazy_warmup_trace_pids: set[int] = set()
_relay_warmed_model_pids: set[int] = set()

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput

import pickle
import struct

class Worker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                envs.VLLM_TORCH_PROFILER_WITH_STACK,
                envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def sleep(self, level: int = 1) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def _maybe_get_memory_pool_context(self,
                                       tag: str) -> AbstractContextManager:
        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, (
                    "Sleep mode can only be "
                    "used for one instance per process.")
            context = allocator.use_memory_pool(tag=tag)
        else:
            context = nullcontext()
        return context

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        self._init_device(reuse_prepared_groups=False)

    def init_device_with_prepared_groups(self):
        """Construct model state inside an activated resident topology.

        The resident controller owns WORLD and TP/PP group lifetimes. It must
        activate matching groups for construction and every model operation.
        """
        tp, pp = get_tp_group(), get_pp_group()
        if (tp.world_size != self.parallel_config.tensor_parallel_size or
                pp.world_size != self.parallel_config.pipeline_parallel_size):
            raise ValueError("Prepared groups do not match model TP/PP configuration")
        self._init_device(reuse_prepared_groups=True)

    def _init_device(self, *, reuse_prepared_groups: bool):
        profile_dir = os.environ.get("VLLM_PROFILE_INIT_HOST")
        if profile_dir:
            # Host-only diagnosis of model creation in retained workers. No
            # CUDA profiler or extra stream fence; profiled runs are diagnostic.
            import cProfile
            import uuid
            os.makedirs(profile_dir, exist_ok=True)
            profiler = cProfile.Profile()
            try:
                with profiler:
                    self._init_device_impl(
                        reuse_prepared_groups=reuse_prepared_groups)
            finally:
                profiler.dump_stats(os.path.join(
                    profile_dir, f"create-{os.getpid()}-{uuid.uuid4().hex}.prof"))
        else:
            self._init_device_impl(reuse_prepared_groups=reuse_prepared_groups)

    def _init_device_impl(self, *, reuse_prepared_groups: bool):
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)
            # gc.collect()
            # torch.cuda.empty_cache()

            # Conventional profiling needs a fresh baseline. SHM uses an
            # offline budget and can reuse free allocator blocks left by a
            # previous model in this resident process.
            if reuse_prepared_groups and self.load_config.load_format != "shm":
                torch.cuda.empty_cache()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (self.init_snapshot.total_memory *
                                     self.cache_config.gpu_memory_utilization)
            startup_available = self.init_snapshot.free_memory
            if reuse_prepared_groups and self.load_config.load_format == "shm":
                # Count this process's free blocks, never another model's live
                # allocations or memory reserved by another process.
                startup_available += (torch.cuda.memory_reserved(self.device)
                                      - torch.cuda.memory_allocated(self.device))
            # A resident SHM worker shares existing context/groups and may
            # overlap another live engine. Its offline non-KV budget already
            # includes that process overhead: the fraction of total memory is
            # not an additional allocation. Check the actual KV allocation
            # after model weights exist in determine_available_memory instead.
            self._relay_deferred_memory_check = (
                reuse_prepared_groups and self.load_config.load_format == "shm")
            if (not self._relay_deferred_memory_check
                    and startup_available < self.requested_memory):
                GiB = lambda b: round(b / GiB_bytes, 2)
                raise ValueError(
                    f"Available memory on device "
                    f"({GiB(startup_available)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        if not reuse_prepared_groups:
            init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank,
                                            current_platform.dist_backend)
        else:
            ensure_kv_transfer_initialized(self.vllm_config)
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner: GPUModelRunner = GPUModelRunner(
            self.vllm_config, self.device)

        if (self.load_config.load_format == "shm"
                and self.load_config.lazy_load):
            self._warm_up_relay_tp_communication()
            from vllm.model_executor.warmup.relay_runtime import warmup_relay_runtime
            warmup_relay_runtime(self.device, self.model_config.dtype)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)
    
    def _warm_up_relay_tp_communication(self) -> None:
        """Initialize the logits collective before model-specific lazy init.

        All ranks must participate, including PP ranks without an LM head.
        A group is prepared once at worker creation; the collective does not
        depend on model weights, vocabulary size, or the cold layer partition.
        """
        group = get_tp_group()
        if group.world_size == 1:
            return
        use_all_gather = current_platform.use_all_gather()
        key = (self.device, self.model_config.dtype, use_all_gather)
        # Prepared resident groups survive model teardown. Their communicator
        # needs one probe per dtype/device, not another probe for each model.
        # Keep the marker on the group so newly created groups warm up again.
        prepared = getattr(group, "_relay_warmup_keys", set())
        if key in prepared:
            return
        start = time.perf_counter()
        # Keep the communicator probe independent from model construction and
        # weight-copy work queued on the default stream. Synchronizing the
        # current stream here would turn unrelated asynchronous work into part
        # of the model initialization time.
        probe_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(probe_stream):
            probe = torch.zeros((1, 1), device=self.device,
                                dtype=self.model_config.dtype)
            if use_all_gather:
                group.all_gather(probe)
            else:
                group.gather(probe)
        probe_stream.synchronize()
        prepared.add(key)
        group._relay_warmup_keys = prepared
        logger.info("Relay TP communication prepared in %.3fs",
                    time.perf_counter() - start)

    def init_model_info(self) -> None:
        self.model_runner.init_model_info()

    def init_shm_kv_cache_info(self) -> None:
        self.model_runner.init_shm_kv_cache_info()

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self) -> None:
        eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.load_model(eep_scale_up=eep_scale_up)

    def lazy_load_model(self, layer_idxes_by_pp: List[List[int]]) -> None:
        print(f"lazy load model from {layer_idxes_by_pp}")
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.lazy_load_model(layer_idxes_by_pp)

    def lazy_load_model_sync(self, layer_idxes_by_pp: List[List[int]]) -> None:
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.lazy_load_model_sync(layer_idxes_by_pp)

    def set_load_method(self, load_method: str = "tokenwise"):
        self.load_config.load_method = load_method
        self.model_runner.set_load_method(load_method)
        sync_kv_transfer_connector_vllm_config(self.vllm_config)

    def set_shm_info(self, shm_kv_cache_size: int, shm_num_blocks: int, shm_block_len: int, shm_tp_size: int):
        self.load_config.shm_kv_cache_size = shm_kv_cache_size
        self.load_config.shm_num_blocks = shm_num_blocks
        self.load_config.shm_block_len = shm_block_len
        self.parallel_config.shm_tp_size = shm_tp_size
        self.model_runner.set_shm_info(shm_kv_cache_size, shm_num_blocks, shm_block_len, shm_tp_size)

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self) -> None:
        self.model_runner.reload_weights()

    # --- Cold-start transition forwarding methods ---

    def finish_layerwise_kv_handoff(self) -> None:
        self.model_runner.finish_layerwise_kv_handoff()

    def get_ready_layerwise_boundary(self) -> Optional[dict]:
        return self.model_runner.get_ready_layerwise_boundary()

    def check_uniform_weights_ready(self) -> bool:
        return self.model_runner.check_uniform_weights_ready()

    def sync_uniform_weights(self) -> None:
        self.model_runner.sync_uniform_weights()

    def execute_kv_migration(
        self,
        cold_partition: List[List[int]],
        load_method: str,
    ) -> None:
        self.model_runner.execute_kv_migration(
            cold_partition, load_method)

    def allocate_relay_kv_buffer(
        self,
        cold_partition: list[list[int]],
    ) -> None:
        self.model_runner.allocate_relay_kv_buffer(cold_partition)

    def cleanup_relay_kv_buffer(self) -> None:
        self.model_runner.cleanup_relay_kv_buffer()

    def resize_kv_blocks(self, new_num_blocks: int) -> None:
        self.model_runner.resize_kv_blocks(new_num_blocks)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """

        GiB = lambda b: b / GiB_bytes

        if self.vllm_config.load_config.load_format != "shm":
            # Execute a forward pass with dummy inputs to profile the memory usage
            # of the model.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with memory_profiling(
                    self.init_snapshot,
                    weights_memory=int(
                        self.model_runner.model_memory_usage)) as profile_result:
                self.model_runner.profile_run()

            free_gpu_memory = profile_result.after_profile.free_memory
            # NOTE(woosuk): Here we assume that the other processes using the same
            # GPU did not change their memory usage during the profiling.
            assert self.init_snapshot.free_memory > free_gpu_memory, (
                "Error in memory profiling. "
                f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
                f"current free memory {GiB(free_gpu_memory)} GiB. "
                "This happens when other processes sharing the same container "
                "release GPU memory while vLLM is profiling during initialization. "
                "To fix this, ensure consistent GPU memory allocation or "
                "isolate vLLM in its own container.")
            non_kv_cache_memory = profile_result.non_kv_cache_memory

            available_kv_cache_memory = self.requested_memory \
                - non_kv_cache_memory

            unrequested_memory = self.init_snapshot.free_memory \
                - self.requested_memory
            logger.debug(
                "Initial free memory: %.2f GiB; "
                "Requested memory: %.2f (util), %.2f GiB",
                GiB(self.init_snapshot.free_memory),
                self.cache_config.gpu_memory_utilization,
                GiB(self.requested_memory),
            )
            logger.debug(
                "Free memory after profiling: %.2f GiB (total), "
                "%.2f GiB (within requested)",
                GiB(free_gpu_memory),
                GiB(free_gpu_memory - unrequested_memory),
            )
            logger.debug(profile_result)

            gc.collect()

        else:
            from vllm.utils.relay_memory_profile import (
                find_memory_profile, memory_profile_identity, read_non_kv_memory)
            profile_path = find_memory_profile(self.model_config.model)
            preflight_budget = getattr(self.cache_config, "_relay_profile_non_kv_bytes", None)
            if preflight_budget is not None:
                non_kv_cache_memory = preflight_budget
            elif profile_path is not None:
                non_kv_cache_memory = read_non_kv_memory(
                    profile_path, memory_profile_identity(self.vllm_config, self.device))
                logger.info("SHM offline memory budget from %s: %d bytes",
                            profile_path, non_kv_cache_memory)
            else:
                logger.warning_once(
                    "No validated SHM offline memory profile; using legacy "
                    "directory-name budget. This estimate does not distinguish TP. "
                    "Calibrate relay_memory_profile.json before deployment.")
                max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
                hidden_size = self.model_config.get_hidden_size()
                bytes_per_tensor = 2        # 目前仅支持bfloat16
                flag = (max_num_batched_tokens > 8192) - (max_num_batched_tokens < 8192)  # 大于8192为1-0=1，等于8192为0-0=0，小于8192为0-1=-1
                delta_memory = max_num_batched_tokens * hidden_size * bytes_per_tensor * flag
                model_id = self.model_config.model.split("/")[-1]
                default_non_kv = _MODEL_DEFAULT_NON_KV_CACHE_MEMORY.get(
                    model_id, 30_000_000_000,
                )
                non_kv_cache_memory = default_non_kv + delta_memory
            available_kv_cache_memory = self.requested_memory - non_kv_cache_memory
            if getattr(self, "_relay_deferred_memory_check", False):
                required_kv_memory = available_kv_cache_memory
                override = getattr(self.cache_config, 'num_gpu_blocks_override', None)
                if override is not None:
                    # PP recovery preserves native token-block capacity. Its
                    # local layers need fewer bytes for the same block count.
                    required_kv_memory = override * sum(
                        spec.page_size_bytes for spec in
                        self.model_runner.get_kv_cache_spec().values())
                free, _ = torch.cuda.mem_get_info(self.device)
                reusable = (torch.cuda.memory_reserved(self.device)
                            - torch.cuda.memory_allocated(self.device))
                if required_kv_memory > free + reusable:
                    raise ValueError(
                        "Insufficient resident GPU capacity for configured KV "
                        f"cache: need {required_kv_memory} bytes, "
                        f"available {free + reusable} bytes after model allocation")

        logger.info("Available KV cache memory: %.2f GiB",
                    GiB(available_kv_cache_memory))
        

        return int(available_kv_cache_memory)

    @torch.inference_mode()
    def profile_shm_non_kv_memory(self) -> dict:
        """Explicit offline calibration, never invoked by lazy initialization.

        Retain the configured KV cache while profiling, then exclude its backing
        storage. Include unused allocator segments conservatively in the budget.
        """
        if (self.load_config.load_format != "shm"
                or not self.model_config.enforce_eager):
            raise ValueError("SHM memory calibration requires eager execution")
        if not self.model_runner.check_uniform_weights_ready():
            self.model_runner.sync_uniform_weights()
        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        storages = {}
        for cache in self.model_runner.kv_caches:
            tensors = cache if isinstance(cache, (tuple, list)) else (cache,)
            for tensor in tensors:
                storage = tensor.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
        kv_bytes = sum(storages.values())
        torch.cuda.reset_peak_memory_stats(self.device)
        allocated_before = torch.cuda.memory_allocated(self.device)
        self.model_runner.profile_run()
        torch.cuda.synchronize(self.device)
        peak_extra = max(0, torch.cuda.max_memory_allocated(self.device)
                         - allocated_before)
        free_after, _ = torch.cuda.mem_get_info(self.device)
        # Count the retained CUDA context/groups too: the configured fraction
        # is a fraction of total device memory, not model-only incremental use.
        non_kv = (self.init_snapshot.total_memory - free_after - kv_bytes
                  + peak_extra)
        if non_kv <= 0 or not kv_bytes:
            raise RuntimeError("Invalid SHM calibration memory accounting")
        from vllm.utils.relay_memory_profile import memory_profile_identity
        return dict(identity=memory_profile_identity(self.vllm_config, self.device),
                    non_kv_memory_bytes=int(non_kv),
                    kv_storage_bytes=kv_bytes, peak_activation_extra_bytes=peak_extra,
                    free_after_bytes=free_after,
                    initial_free_bytes=self.init_snapshot.free_memory,
                    rank=self.rank)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        started = time.perf_counter()

        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            context = nullcontext()
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)
        logger.info("[lazy_init] rank=%d initialize_kv_cache: %.3fs",
                    self.rank, time.perf_counter() - started)

    def compile_or_warm_up_model(self) -> None:
        logger.info(
            "compile_or_warm_up_model start (load_format=%s, enforce_eager=%s)",
            self.vllm_config.load_config.load_format,
            self.model_config.enforce_eager,
        )
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        throttle_loading = (self.vllm_config.load_config.load_format == "shm"
                            and self.vllm_config.load_config.load_async)
        if throttle_loading:
            self.model_runner.update_module_loading_state(True)
        try:
            trace_dir = os.environ.get("VLLM_PROFILE_LAZY_TRACE")
            # Trace only the first model warmup in each retained worker. CUDA
            # profiling drains streams on exit, so these diagnostic runs must
            # not be used as production latency measurements.
            trace_pid = os.getpid()
            trace_first = (trace_dir and trace_pid not in
                           _lazy_warmup_trace_pids)
            if trace_first:
                os.makedirs(trace_dir, exist_ok=True)
                trace_started = time.perf_counter()
                with torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    logger.info("[lazy_trace] profiler setup: %.3fs",
                                time.perf_counter() - trace_started)
                    with torch.profiler.record_function("relay_lazy_warmup"):
                        self._compile_or_warm_up_model()
                    trace_finished = time.perf_counter()
                profiler.export_chrome_trace(os.path.join(
                    trace_dir, f"warmup-{os.getpid()}.json"))
                logger.info("[lazy_trace] profiler drain/export: %.3fs",
                            time.perf_counter() - trace_finished)
                _lazy_warmup_trace_pids.add(trace_pid)
            else:
                if os.environ.get("VLLM_PROFILE_LAZY_COMPLETION") == "1":
                    # Diagnostic only: wait for compute, never the entire
                    # device/copy stream. This changes the initialization
                    # boundary, so do not use these runs as latency baselines.
                    stream = torch.cuda.current_stream(self.device)
                    begin = torch.cuda.Event(enable_timing=True)
                    finish = torch.cuda.Event(enable_timing=True)
                    begin.record(stream)
                    started = time.perf_counter()
                    self._compile_or_warm_up_model()
                    submitted = time.perf_counter() - started
                    finish.record(stream)
                    wait_started = time.perf_counter()
                    finish.synchronize()
                    compute_tail = time.perf_counter() - wait_started
                    completed_layers = None
                    layer_records = None
                    if self.vllm_config.load_config.load_format == "shm":
                        from shm_manager.torch import model_info
                        completed_layers = len(
                            model_info.manager.completed_layer_load_seconds())
                        layer_records = model_info.manager.completed_layer_load_records()
                    logger.info(
                        "[lazy_completion] rank=%d submission=%.6fs "
                        "compute_tail=%.6fs compute_stream_span=%.6fs "
                        "completed_weight_layers=%s",
                        self.rank, submitted, compute_tail,
                        begin.elapsed_time(finish) / 1000,
                        completed_layers,
                    )
                    if layer_records is not None:
                        logger.info("[lazy_layer_records] rank=%d records=%s",
                                    self.rank, layer_records)
                else:
                    host_profile_dir = os.environ.get("VLLM_PROFILE_LAZY_HOST")
                    if host_profile_dir:
                        # Diagnose host API waits without CUDA timing events
                        # or additional synchronization. Profiling overhead
                        # means this run is not a latency baseline.
                        import cProfile
                        import uuid
                        os.makedirs(host_profile_dir, exist_ok=True)
                        profiler = cProfile.Profile()
                        wall_start = time.perf_counter()
                        cpu_start = time.thread_time()
                        try:
                            with profiler:
                                self._compile_or_warm_up_model()
                        finally:
                            logger.info("[lazy_host] wall=%.6fs thread_cpu=%.6fs",
                                        time.perf_counter() - wall_start,
                                        time.thread_time() - cpu_start)
                            profiler.dump_stats(os.path.join(
                                host_profile_dir,
                                f"warmup-{os.getpid()}-{uuid.uuid4().hex}.prof"))
                    else:
                        self._compile_or_warm_up_model()
        finally:
            if throttle_loading:
                self.model_runner.update_module_loading_state(False)

    def _compile_or_warm_up_model(self) -> None:
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        warmup_phase_start = time.perf_counter()
        kernel_warmup(self)
        logger.info("[lazy_init] kernel_warmup: %.3fs",
                    time.perf_counter() - warmup_phase_start)

        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

        # Warm up the model execution path on every PP rank. The sampler only
        # exists on the last rank, so sampler warmup remains last-rank only.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        max_num_reqs = min(self.scheduler_config.max_num_seqs,
                           self.scheduler_config.max_num_batched_tokens)
        prefill_tokens = min(max_num_reqs + 1,
                             self.scheduler_config.max_num_batched_tokens)
        if self.vllm_config.load_config.load_format == "shm":
            configured = os.environ.get("VLLM_RELAY_WARMUP_PREFILL_TOKENS")
            # Retained ranks have already loaded CUDA modules after their
            # first model. Allow calibration of that distinct startup phase;
            # both phases still execute every allocated layer and decode.
            if os.getpid() not in _relay_warmed_model_pids:
                configured = os.environ.get(
                    "VLLM_RELAY_FIRST_WARMUP_PREFILL_TOKENS", configured)
            if configured is None:
                # Every local layer is exercised below. Keep this workload
                # compact; the readiness policy owns the requested window,
                # rather than wasting GPU compute to manufacture a delay.
                configured = str(prefill_tokens)
            if configured is not None:
                try:
                    prefill_tokens = int(configured)
                except ValueError as exc:
                    raise ValueError("VLLM_RELAY_WARMUP_PREFILL_TOKENS must be a positive integer") from exc
                if not 1 <= prefill_tokens <= self.scheduler_config.max_num_batched_tokens:
                    raise ValueError("Relay warmup prefill tokens must fit the chunk budget")
        logger.info("[relay warmup shape] prefill_tokens=%d decode_tokens=%d",
                    prefill_tokens, max_num_reqs)

        # We skip EPLB here since we don't want to record dummy metrics.
        warmup_phase_start = time.perf_counter()
        # Exercise prefill and the uniform decode path
        # for each newly constructed model, including models in retained ranks.
        # Dummy forwards skip weight fences so SHM DMA can keep progressing.
        # Visit every allocated local layer, including the recovery prefix.
        # Prefill/decode and the sampler remain separately exercised. The
        # compile sizes and CUDA graphs above retain their execution range.
        with self.model_runner.warmup_layer_range():
            self.model_runner._dummy_run(
                # More tokens than requests exercises multi-token prefill.
                num_tokens=prefill_tokens,
                force_attention=True,
                skip_eplb=True,
                warmup_query_context=(
                    self.vllm_config.load_config.load_format == "shm"),
            )
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                force_attention=True,
                uniform_decode=True,
                skip_eplb=True,
                warmup_query_context=(
                    self.vllm_config.load_config.load_format == "shm"),
            )
        logger.info("[lazy_init] dummy_model_warmup: %.3fs",
                    time.perf_counter() - warmup_phase_start)

        warmup_phase_start = time.perf_counter()
        if get_pp_group().is_last_rank:
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(
                    hidden_states=last_hidden_states)
        logger.info("[lazy_init] sampler_or_pooler_warmup: %.3fs",
                    time.perf_counter() - warmup_phase_start)

        # Readiness includes completion of this model's warmup kernels. Avoid
        # a device-wide fence: unrelated weight-copy work remains asynchronous.
        completion_start = time.perf_counter()
        completion_thread_start = time.thread_time()
        torch.cuda.current_stream(self.device).synchronize()
        if self.vllm_config.load_config.load_format == "shm":
            _relay_warmed_model_pids.add(os.getpid())
        logger.info("[relay warmup completion] rank=%d wall=%.6fs "
                    "host_thread_cpu=%.6fs", self.rank,
                    time.perf_counter() - completion_start,
                    time.thread_time() - completion_thread_start)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        if self.vllm_config.load_config.load_format == 'shm':
            from vllm.v1.worker.relay_initialization_policy import (
                initialization_target_ms, readiness_wait_seconds)
            target = initialization_target_ms(self.model_config.hf_config)
            started = getattr(self.model_runner, '_relay_initialization_started_at', None)
            if started is not None and target:
                serial_copy = getattr(self.model_runner.model_loader,
                                      'serial_copy_seconds', 0.)
                now = time.perf_counter()
                wait = readiness_wait_seconds(started, now, target, serial_copy)
                # Release the host thread; AMX and background DMA continue.
                # This is explicit readiness pacing, never reported as compute.
                if wait:
                    if self.vllm_config.load_config.load_async:
                        # Module warmup is finished; its copy throttle must
                        # not restrict DMA during the remaining ready window.
                        self.model_runner.update_module_loading_state(False)
                    time.sleep(wait)
                logger.info('[relay initialization window] target_ms=%.3f '
                            'actual_work_ms=%.3f readiness_wait_ms=%.3f '
                            'serial_copy_ms=%.3f elapsed_ms=%.3f '
                            'origin=vllm_engine_core_entry', target,
                            (now - started - serial_copy) * 1000., wait * 1000.,
                            serial_copy * 1000., (time.perf_counter() - started) * 1000.)

        logger.info("compile_or_warm_up_model done")

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def update_shm_scheduler_output(self, scheduler_output):
        self.model_runner.update_shm_scheduler_output(scheduler_output)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:

        kv_only_batch = scheduler_output.total_num_scheduled_tokens == 0
        intermediate_tensors = None
        if not get_pp_group().is_first_rank and not kv_only_batch:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)

        parallel_config = self.vllm_config.parallel_config
        if parallel_config.distributed_executor_backend != "external_launcher" \
            and not get_pp_group().is_last_rank and not kv_only_batch:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())

            kv_connector_output = output.kv_connector_output
            if not kv_connector_output:
                return None

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if (not kv_connector_output.finished_sending
                    and not kv_connector_output.finished_recving):
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        if kv_only_batch:
            assert isinstance(output, ModelRunnerOutput)
            return output

        assert isinstance(output, ModelRunnerOutput)
        return output

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            print(self.profiler.key_averages().table(
                sort_by="self_cuda_time_total"))

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def _eplb_before_scale_down(self, old_ep_size: int,
                                new_ep_size: int) -> None:
        from vllm.distributed.parallel_state import get_ep_group
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding "
                        "before scaling down...")
        rank_mapping = {
            old_ep_rank: old_ep_rank if old_ep_rank < new_ep_size else -1
            for old_ep_rank in range(old_ep_size)
        }
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(self.model_runner.model,
                                               execute_shuffle=True,
                                               global_expert_load=None,
                                               rank_mapping=rank_mapping)
        torch.cuda.synchronize()
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _eplb_after_scale_up(
            self, old_ep_size: int, new_ep_size: int,
            global_expert_load: Optional[torch.Tensor]) -> None:
        from vllm.distributed.parallel_state import get_ep_group
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding "
                        "after scaling up...")
        rank_mapping = {
            old_ep_rank: old_ep_rank
            for old_ep_rank in range(old_ep_size)
        }
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(
            self.model_runner.model,
            execute_shuffle=True,
            global_expert_load=global_expert_load,
            rank_mapping=rank_mapping)
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _reconfigure_parallel_config(
            self, reconfig_request: ReconfigureDistributedRequest) -> None:
        """
        Update parallel config with provided reconfig_request
        """
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = \
            reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != \
        ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank = \
                reconfig_request.new_data_parallel_rank
        if reconfig_request.new_data_parallel_rank_local != \
        ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank_local = \
                reconfig_request.new_data_parallel_rank_local
        parallel_config.data_parallel_master_ip = \
            reconfig_request.new_data_parallel_master_ip
        parallel_config.data_parallel_master_port = \
            reconfig_request.new_data_parallel_master_port

    def _reconfigure_moe(self, old_ep_size: int,
                         new_ep_size: int) -> Optional[torch.Tensor]:
        """
        Reconfigure MoE modules with provided reconfig_request

        Return the global expert load if new_ep_size > old_ep_size,
        otherwise None
        """
        from vllm.distributed.parallel_state import (
            get_dp_group, get_ep_group, prepare_communication_buffer_for_model)
        from vllm.model_executor.layers.fused_moe.layer import (
            FusedMoEParallelConfig)

        parallel_config = self.vllm_config.parallel_config
        moe_modules = [
            module for module in self.model_runner.model.modules()
            if module.__class__.__name__ == "FusedMoE"
        ]
        num_local_experts = moe_modules[0].moe_config.num_local_experts
        assert all(module.moe_config.num_local_experts == num_local_experts
                   for module in moe_modules), (
                       "All MoE modules must have the same number of experts")
        for module in moe_modules:
            module.moe_config.num_experts = num_local_experts * new_ep_size
            module.global_num_experts = module.moe_config.num_experts
            module.moe_parallel_config = FusedMoEParallelConfig.make(
                tp_size_=get_tp_group().world_size,
                dp_size_=get_dp_group().world_size,
                vllm_parallel_config=parallel_config,
            )
            module.moe_config.moe_parallel_config = module.moe_parallel_config
        if new_ep_size < old_ep_size:
            num_local_physical_experts = num_local_experts
            assert self.model_runner.eplb_state is not None
            new_physical_experts = \
                self.model_runner.eplb_state.physical_to_logical_map.shape[1]
            parallel_config.eplb_config.num_redundant_experts = (
                new_physical_experts -
                self.model_runner.eplb_state.logical_replica_count.shape[1])
            global_expert_load = None
        else:
            num_local_physical_experts = torch.tensor([num_local_experts],
                                                      dtype=torch.int32,
                                                      device="cpu")
            torch.distributed.broadcast(num_local_physical_experts,
                                        group=get_ep_group().cpu_group,
                                        group_src=0)
            num_local_physical_experts = num_local_physical_experts.item()
            new_physical_experts = num_local_physical_experts * new_ep_size
            assert self.model_runner.eplb_state is not None
            global_expert_load = self.model_runner.eplb_state.rearrange(
                self.model_runner.model, execute_shuffle=False)
            parallel_config.eplb_config.num_redundant_experts = (
                new_physical_experts - global_expert_load.shape[1])
        prepare_communication_buffer_for_model(self.model_runner.model)
        self.model_runner.model.update_physical_experts_metadata(
            num_physical_experts=new_physical_experts,
            num_local_physical_experts=num_local_physical_experts)
        return global_expert_load

    def reinitialize_distributed(
            self, reconfig_request: ReconfigureDistributedRequest) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import (
            cleanup_dist_env_and_memory, get_ep_group)

        old_ep_size = get_ep_group().world_size
        old_ep_rank = get_ep_group().rank
        new_ep_size = reconfig_request.new_data_parallel_size * get_tp_group(
        ).world_size * get_pp_group().world_size
        if new_ep_size < old_ep_size:
            self._eplb_before_scale_down(old_ep_size, new_ep_size)

        cleanup_dist_env_and_memory()

        if reconfig_request.new_data_parallel_rank == \
        ReconfigureRankType.SHUTDOWN_CURRENT_RANK:
            assert old_ep_rank >= new_ep_size
            # shutdown
            return

        self._reconfigure_parallel_config(reconfig_request)

        with set_current_vllm_config(self.vllm_config):
            init_worker_distributed_environment(self.vllm_config, self.rank,
                                                self.distributed_init_method,
                                                self.local_rank)

        global_expert_load = self._reconfigure_moe(old_ep_size, new_ep_size)

        if new_ep_size > old_ep_size:
            assert global_expert_load is not None
            self._eplb_after_scale_up(old_ep_size, new_ep_size,
                                      global_expert_load)

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )
    
    def save_shm_model(self, path: str) -> None:
        self.model_runner.save_shm_model(path)

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank, backend)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    ensure_kv_transfer_initialized(vllm_config)   # 创建worker端的kv_connector
