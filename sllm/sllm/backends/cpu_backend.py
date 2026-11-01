# ---------------------------------------------------------------------------- #
#  vLLM on CPU: ShmConnector kv_producer, SHM layout probe, layerwise prefill   #
# ---------------------------------------------------------------------------- #
import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from vllm import AsyncLLMEngine, SamplingParams

from sllm.backends.backend_utils import (
    BackendStatus,
    LLMEngineStatusDict,
    SllmBackend,
    async_engine_args_from_dict,
    build_core_filtered_engine_config,
    parse_vllm_generate_request,
    process_output,
    snapshot_kv_transfer_params,
    vllm_engine_generate,
)
from sllm.sllm.logger import init_logger

logger = init_logger(__name__)


class CPUBackend(SllmBackend):
    """vLLM ``AsyncLLMEngine`` on CPU: KV producer for shared-memory handoff to GPU."""

    _vllm_device_log_label = "CPU"

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
        if device != "cpu":
            raise ValueError(f"CPUBackend expects device 'cpu', got {device!r}")

        self.status = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.instance_id = instance_id
        self.model_name = model
        self.backend_config = backend_config
        self.runtime_env = runtime_env
        self.request_trace = LLMEngineStatusDict()
        self.trace_debug = backend_config.get("trace_debug", False)
        self.load_method = backend_config.get("load_method", "tokenwise")

        filtered = build_core_filtered_engine_config(model, backend_config)
        filtered["tensor_parallel_size"] = 2
        filtered["pipeline_parallel_size"] = 1

        filtered["max_num_batched_tokens"] = 4096
        filtered["enforce_eager"] = True
        filtered["enable_prefix_caching"] = False
        filtered["task"] = "auto"
        filtered["dtype"] = "bfloat16"
        filtered["load_method"] = "tokenwise"
        filtered["max_model_len"] = 40960

        if filtered["load_format"] == "shm":
            filtered["kv_transfer_config"] = {
                "kv_connector": "ShmConnector",
                "kv_role": "kv_producer",
                "kv_rank": 0,
            }

        self.engine_args = async_engine_args_from_dict(filtered)
        self.engine: Optional[AsyncLLMEngine] = None
        self.profile = False

    async def init_backend(self) -> None:
        if self.runtime_env is not None:
            ev = self.runtime_env.get("env_vars")
            if isinstance(ev, dict) and ev:
                os.environ.update({str(k): str(v) for k, v in ev.items()})

        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "98-142|146-190"
        os.environ["VLLM_SLEEP_WHEN_IDLE"] = "1"
        os.environ["VLLM_CPU_KVCACHE_SPACE"] = "8"

        logger.info("Initializing vLLM CPU backend...")
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                logger.warning("vLLM CPU backend already initialized")
                return
            start_time = time.time()
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            load_time = time.time() - start_time
            logger.info("vLLM CPU backend initialized in %.2f ms", load_time * 1000)
            self.status = BackendStatus.RUNNING

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
        final_output = None
        first_chunk_time = None
        itl_token_count = 0
        observed_tokens = 0
        first_token_count = 0
        ttft = 0.0
        itl_list: List[float] = []
        most_recent_timestamp = start_time
        last_kv_transfer_params: Any = None
        completed = False
        try:
            results_generator = vllm_engine_generate(
                engine, inputs, sampling_params, request_id, extra_args,
                sampling_kw_without_extra=sampling_kw,
            )
            async for response_output in results_generator:
                final_output = response_output
                kvp = getattr(response_output, "kv_transfer_params", None)
                if kvp is not None:
                    last_kv_transfer_params = kvp
                await self.request_trace.update_status(request_id, response_output)
                token_count = len(response_output.outputs[0].token_ids) if response_output.outputs else 0
                if token_count > observed_tokens:
                    current_time = time.perf_counter()
                    if first_chunk_time is None:
                        first_chunk_time = current_time
                        ttft = first_chunk_time - start_time
                        first_token_count = token_count
                    else:
                        itl_token_count += token_count - observed_tokens
                        itl_list.append(current_time - most_recent_timestamp)
                    most_recent_timestamp = current_time
                    observed_tokens = token_count
            end_time = time.perf_counter()
            if final_output is None or not final_output.outputs:
                return {"error": "No output produced"}
            latency_metrics["e2e"] = end_time - start_time
            latency_metrics["ttft"] = ttft
            latency_metrics["first_token_time"] = first_chunk_time
            latency_metrics["first_token_count"] = first_token_count
            latency_metrics["last_token_offset_s"] = most_recent_timestamp - start_time if observed_tokens else None
            latency_metrics["tpot"] = (
                (most_recent_timestamp - first_chunk_time) / itl_token_count
                if itl_token_count > 0 else 0.0
            )
            latency_metrics["itls"] = itl_list if itl_token_count > 0 else []
            latency_metrics["output_length"] = len(final_output.outputs[0].token_ids)
            latency_metrics["sampling_max_tokens"] = getattr(sampling_params, "max_tokens", None)
            completed = True
        finally:
            await self._finish_request(engine, request_id, completed)
        api_resp, metrics = process_output(final_output, latency_metrics, model_name)
        api_resp = dict(api_resp)
        api_resp["prompt_token_ids"] = list(final_output.prompt_token_ids)
        api_resp["output_token_ids"] = list(final_output.outputs[0].token_ids)
        snap = snapshot_kv_transfer_params(last_kv_transfer_params)
        if snap:
            api_resp = dict(api_resp)
            api_resp["kv_transfer_params"] = snap
        return api_resp, metrics

    async def update_computing_layers(self, computing_layers: int):
        if self.load_method != "layerwise" or self.engine is None:
            return
        await self.engine.update_computing_layers(computing_layers)

    async def update_pp_cleanup(self, enabled: bool):
        if self.engine is None:
            return 
        await self.engine.set_pp_cleanup(enabled)

    async def get_shm_kv_cache_info(self) -> Optional[Tuple[int, int, int, int]]:
        if self.engine is None:
            return None
        return await self.engine.get_shm_kv_cache_info()
