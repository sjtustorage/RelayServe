# ---------------------------------------------------------------------------- #
#  vLLM on GPU: ShmConnector kv_consumer, lazy load, scheduler / router hooks    #
# ---------------------------------------------------------------------------- #
import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import ray
from vllm import AsyncLLMEngine, RequestOutput, SamplingParams

from sllm.backends.backend_utils import (
    BackendStatus,
    LLMEngineStatusDict,
    SllmBackend,
    async_engine_args_from_dict,
    build_core_filtered_engine_config,
    parse_vllm_generate_request,
    process_output,
    read_numa_cpu_affinity,
    vllm_engine_generate,
)
from sllm.sllm.loading_perf_profile import (
    _normalize_layer_idxes_for_vllm_pp,
    expand_first_stage_load_list_compute_layers_first,
)
from sllm.sllm.logger import init_logger

logger = init_logger(__name__)

class GPUBackend(SllmBackend):
    """vLLM ``AsyncLLMEngine`` on GPU: KV consumer, lazy weights, NUMA affinity."""

    _vllm_device_log_label = "GPU"

    def __init__(
        self,
        instance_id: str,
        model: str,
        device: str,
        backend_config: Optional[Dict[str, Any]] = None,
        runtime_env: Optional[Dict[str, Any]] = None,
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")
        if device != "gpu":
            raise ValueError(f"GPUBackend expects device 'gpu', got {device!r}")

        self.status = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.instance_id = instance_id
        self.model_name = model
        self.backend_config = backend_config
        self.runtime_env = runtime_env
        self.request_trace = LLMEngineStatusDict()
        self.trace_debug = backend_config.get("trace_debug", False)
        self.load_method = backend_config.get("load_method", "tokenwise")
        self.lazy_load = backend_config.get("lazy_load", True)
        self.pp_size = backend_config.get("pipeline_parallel_size", 1)
        self.profile = False
        self.node_id = backend_config.get("node_id", -1)

        filtered = build_core_filtered_engine_config(model, backend_config)
        filtered["enforce_eager"] = True
        filtered["enable_prefix_caching"] = False
        filtered["dtype"] = "bfloat16"
        filtered["load_method"] = "tokenwise"
        filtered["block_size"] = 128
        filtered["max_num_batched_tokens"] = 4096
        filtered["max_model_len"] = 40960
        # hydraserve / serverlessllm: custom async load is managed externally.
        # vllm / default: let the engine manage its own async behaviour.
        if backend_config.get("hydraserve") or backend_config.get("serverlessllm"):
            filtered["load_async"] = False
        else:
            filtered["load_async"] = True

        if filtered["load_format"] == "shm":
            filtered["shm_tp_size"] = backend_config.get("shm_tp_size", 2)
            filtered["shm_kv_cache_size"] = backend_config.get("shm_kv_cache_size", 4293918720)
            filtered["shm_num_blocks"] = backend_config.get("shm_num_blocks", 455)
            filtered["shm_block_len"] = backend_config.get("shm_block_len", 131072)
            filtered["kv_transfer_config"] = {
                "kv_connector": "ShmConnector",
                "kv_role": "kv_consumer",
                "kv_rank": 1,
            }
            self.lazy_load = bool(filtered.get("lazy_load", False))

        self.engine_args = async_engine_args_from_dict(filtered)
        self.engine: Optional[AsyncLLMEngine] = None
        self.weights_loaded = False

        self.gpu_router = ray.get_actor(self.model_name, namespace="gpu_models")
        self.scheduler = ray.get_actor("model_loading_scheduler")

    async def lazy_init(
        self, layer_idxes: list[list[int]], request_id: str = None
    ):
        start_time = time.time()
        # Same shape rules as ``solve_lazy_load_method`` (idempotent if already normalized).
        layer_idxes = _normalize_layer_idxes_for_vllm_pp(
            str(self.load_method or "tokenwise"),
            layer_idxes,
            self.backend_config,
        )
        if self.pp_size > 1: 
            layer_idxes = expand_first_stage_load_list_compute_layers_first(
                layer_idxes, self.backend_config
            )
        await self.engine.lazy_init(
            layer_idxes_by_pp=layer_idxes, no_warmup=False, request_id=request_id
        )
        logger.info(
            "lazy_init done instance_id=%s elapsed_s=%.3f",
            self.instance_id,
            time.time() - start_time,
        )
        self.weights_loaded = True
        return True

    async def init_backend(self) -> None:
        # Non-PP only: ``_sllm_cuda_visible_devices`` from stripped ``runtime_env`` (see
        # ``router_utils``). PP: do not set ``CUDA_VISIBLE_DEVICES`` here.
        _pp = int(
            (self.backend_config or {}).get("pipeline_parallel_size", 1)
            or (self.backend_config or {}).get("gpu_pp_size", 1)
            or 1
        ) or 1
        cvd = (self.backend_config or {}).get("_sllm_cuda_visible_devices")
        if _pp <= 1 and cvd is not None and str(cvd).strip() != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd).strip()

        if self.runtime_env is not None:
            ev = self.runtime_env.get("env_vars")
            if isinstance(ev, dict) and ev:
                os.environ.update(
                    {
                        str(k): str(v)
                        for k, v in ev.items()
                        if str(k) != "CUDA_VISIBLE_DEVICES"
                    }
                )
        # Bind the backend process to the CPUs local to its assigned GPU/NUMA
        # domain.  The old implementation used ``{0, 96}`` unconditionally,
        # which silently placed every worker on the same two CPUs and defeated
        # NUMA-aware weight loading and AMX execution.
        local_gpu_numa = self.backend_config.get("local_gpu_numa_node")
        if local_gpu_numa is None:
            local_gpu_numa = os.environ.get("SLLM_LOCAL_NUMA_NODE")
        if local_gpu_numa is not None:
            try:
                numa_cpus = read_numa_cpu_affinity(int(local_gpu_numa))
            except (TypeError, ValueError):
                numa_cpus = []
            if numa_cpus:
                # Preserve CPU isolation imposed by the launcher/container.
                local_cpus = set(numa_cpus) & os.sched_getaffinity(0)
                if local_cpus:
                    os.sched_setaffinity(0, local_cpus)
                else:
                    logger.warning(
                        "NUMA node %s has no CPUs in the inherited affinity; "
                        "preserving launcher affinity", local_gpu_numa,
                    )

        os.environ["VLLM_SLEEP_WHEN_IDLE"] = "1"
        os.environ["VLLM_TORCH_PROFILER_DIR"] = "/workspace/sllm/traces"

        # PP multi-node: vLLM v1 verifies the Ray driver/worker sees the same placement
        # group used for GPU bundles. Attach PG in this worker if we have a handle.
        pg = self.backend_config.get("_sllm_ray_placement_group")
        if pg is not None:
            try:
                if ray.util.get_current_placement_group() is None:
                    if hasattr(ray.util, "set_current_placement_group"):
                        ray.util.set_current_placement_group(pg)  # type: ignore[attr-defined]
                    else:
                        gw = getattr(
                            getattr(ray, "worker", None), "global_worker", None
                        )
                        cw = getattr(gw, "core_worker", None) if gw else None
                        if cw is not None and hasattr(cw, "set_placement_group"):
                            cw.set_placement_group(pg)
            except Exception as exc:
                logger.warning(
                    "Could not attach Ray placement group before vLLM init: %s",
                    exc,
                )

        try:
            _ctx = ray.get_runtime_context()
            logger.info(
                "GPUBackend actor Ray node_id=%s has_pg_in_worker=%s before "
                "AsyncLLMEngine.from_engine_args (spawned vLLM driver may differ)",
                _ctx.get_node_id(),
                ray.util.get_current_placement_group() is not None,
            )
        except Exception:
            pass

        logger.info("Initializing vLLM GPU backend...")
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                logger.warning("vLLM GPU backend already initialized")
                return
            start_time = time.time()
            if self.lazy_load:
                logger.info("init engine except load model weights")
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            load_time = time.time() - start_time
            logger.info("vLLM GPU backend initialized in %.2f ms", load_time * 1000)
            self.status = BackendStatus.RUNNING

    async def set_shm_kv_cache_info(self, shm_kv_cache_info: tuple[int, int, int, int]):
        if self.engine is None:
            return
        await self.engine.set_shm_kv_cache_info(
            shm_kv_cache_size=shm_kv_cache_info[0],
            shm_num_blocks=shm_kv_cache_info[1],
            shm_block_len=shm_kv_cache_info[2],
            shm_tp_size=shm_kv_cache_info[3],
        )

    async def generate(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        if request_data is None:
            return {"error": "Request data is missing"}

        try:
            model_name, inputs, request_id, sampling_kw, extra_args = (
                parse_vllm_generate_request(request_data, self.model_name)
            )
            sampling_params = SamplingParams(**sampling_kw)
        except Exception as e:
            return {"error": f"Invalid request or sampling parameters: {e}"}

        try:
            engine = await self._begin_request(request_id)
        except (RuntimeError, ValueError) as exc:
            return {"error": str(exc)}

        latency_metrics: Dict[str, Any] = {}
        start_time = time.perf_counter()
        final_output: Optional[RequestOutput] = None
        first_chunk_time: Optional[float] = None
        ttft: float = 0.0
        itl_token_count = 0
        observed_tokens = 0
        first_token_count = 0
        itl_list: List[float] = []
        most_recent_timestamp = start_time
        notify_flag = True
        completed = False
        try:
            results_generator = vllm_engine_generate(
                engine, inputs, sampling_params, request_id, extra_args,
                sampling_kw_without_extra=sampling_kw,
            )
            async for response_output in results_generator:
                final_output = response_output
                if getattr(response_output, "load_weights_finished", False) and notify_flag:
                    notify_flag = False
                    await self.gpu_router.notify_weights_loaded.remote(instance_id=self.instance_id)
                await self.request_trace.update_status(request_id, response_output)
                token_count = len(response_output.outputs[0].token_ids) if response_output.outputs else 0
                if token_count > observed_tokens:
                    current_time = time.perf_counter()
                    if first_chunk_time is None:
                        first_chunk_time = current_time
                        ttft = first_chunk_time - start_time
                        first_token_count = token_count
                        try:
                            self.scheduler.notify_first_token_by_instance.remote(
                                instance_id=self.instance_id, model_name=self.model_name
                            )
                        except Exception:
                            pass
                    else:
                        itl_token_count += token_count - observed_tokens
                        itl_list.append(current_time - most_recent_timestamp)
                    most_recent_timestamp = current_time
                    observed_tokens = token_count
            completed = final_output is not None
        except Exception as exc:
            logger.exception(
                "Streaming generation failed for request %s model=%s: %s",
                request_id, self.model_name, exc,
            )
            return {"error": f"Generation stream error: {exc}"}
        finally:
            await self._finish_request(engine, request_id, completed)
        end_time = time.perf_counter()
        if final_output is None or not final_output.outputs:
            logger.error(
                "No output produced for request %s model=%s", request_id, self.model_name
            )
            return {"error": "No output produced"}
        latency_metrics["e2e"] = end_time - start_time
        latency_metrics["ttft"] = ttft
        latency_metrics["first_token_time"] = first_chunk_time
        latency_metrics["first_token_count"] = first_token_count
        latency_metrics["last_token_offset_s"] = most_recent_timestamp - start_time if observed_tokens else None
        latency_metrics["tpot"] = (
            (most_recent_timestamp - first_chunk_time) / itl_token_count
            if itl_token_count > 0 and first_chunk_time is not None
            else 0.0
        )
        latency_metrics["itls"] = itl_list if itl_token_count > 0 else []
        latency_metrics["output_length"] = len(final_output.outputs[0].token_ids)

        return process_output(final_output, latency_metrics, model_name)
