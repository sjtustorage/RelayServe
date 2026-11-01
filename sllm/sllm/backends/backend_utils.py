# ---------------------------------------------------------------------------- #
#  SLLM backend ABC + vLLM helpers and package exports                          #
#  (merged former backend_utils + vllm_shared + this module)                  #
# ---------------------------------------------------------------------------- #
import asyncio
import inspect
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import fields
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm.inputs import TokensPrompt

from sllm.logger import init_logger
from sllm.utils import request_text_prompt

logger = init_logger(__name__)

# ----------------------------------------------------------------------------- #
#  Backend ABC (all SLLM model backends)                                       #
# ----------------------------------------------------------------------------- #


class BackendStatus(Enum):
    UNINITIALIZED = auto()
    RUNNING = auto()
    STOPPING = auto()
    DELETING = auto()


class LLMEngineStatusDict:
    def __init__(self):
        self.status_dict: Dict[str, Union[RequestOutput, str]] = {}
        self.completed_status_dict: Dict[str, Union[RequestOutput, str]] = {}
        self.lock = asyncio.Lock()

    async def update_status(
        self, request_id: str, request_output: Union[RequestOutput, str]
    ):
        async with self.lock:
            self.status_dict[request_id] = request_output

    async def register_request(self, request_id: str) -> bool:
        async with self.lock:
            if request_id in self.status_dict:
                return False
            self.status_dict[request_id] = "pending"
            return True

    async def delete_request(self, request_id: str, keep_history: bool = False):
        async with self.lock:
            result = self.status_dict.pop(request_id, None)
            if keep_history and result is not None:
                self.completed_status_dict[request_id] = result

    async def return_all_results(self) -> List[Union[RequestOutput, str]]:
        async with self.lock:
            return list(self.status_dict.values())

    async def return_all_request_ids(self) -> List[str]:
        async with self.lock:
            return list(self.status_dict.keys())

    async def request_count(self) -> int:
        async with self.lock:
            return len(self.status_dict)


class SllmBackend(ABC):
    """Model backends; vLLM CPU/GPU share profiling, shutdown, and token trace helpers."""

    engine: Optional[AsyncLLMEngine]
    instance_id: str
    model_name: str
    status: BackendStatus
    status_lock: asyncio.Lock
    request_trace: LLMEngineStatusDict
    trace_debug: bool

    _vllm_device_log_label: str = "device"

    @abstractmethod
    def __init__(
        self,
        instance_id: str,
        model_name: str,
        device: str,
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.profile = False
        pass

    async def start_profile(self):
        await self.engine.start_profile()
        self.profile = True

    async def stop_profile(self):
        await self.engine.stop_profile()
        self.profile = False

    @abstractmethod
    async def init_backend(self) -> None:
        pass

    @abstractmethod
    async def generate(self, request_data: Dict[str, Any]):
        pass

    async def set_load_method(self, method: str) -> None:
        """Update the load method on this backend and propagate to the engine."""
        self.load_method = method
        if self.engine is not None:
            await self.engine.set_load_method(load_method=method)

    async def _begin_request(self, request_id: str):
        # Register before the first engine output, atomically with stop admission.
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING or self.engine is None:
                raise RuntimeError("Engine is not running")
            if not await self.request_trace.register_request(request_id):
                raise ValueError(f"Request {request_id} is already active")
            return self.engine

    async def _finish_request(self, engine, request_id: str, completed: bool):
        try:
            if not completed:
                try:
                    await asyncio.wait_for(engine.abort(request_id), timeout=self._abort_timeout())
                except Exception:
                    logger.exception("Abort failed for request %s", request_id)
        finally:
            await self.request_trace.delete_request(
                request_id, keep_history=self.trace_debug
            )

    def _abort_timeout(self) -> float:
        return float((getattr(self, "backend_config", None) or {}).get("request_abort_timeout_s", 30))

    async def shutdown(self):
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        engine = self.engine
        if engine is not None:
            requests = await self.request_trace.return_all_request_ids()
            results = await asyncio.gather(
                *(asyncio.wait_for(engine.abort(request_id), timeout=self._abort_timeout()) for request_id in requests),
                return_exceptions=True,
            )
            for request_id, result in zip(requests, results):
                if isinstance(result, BaseException):
                    logger.warning("Abort failed for request %s: %s", request_id, result)
            logger.info(
                "Shutting down vLLM %s engine instance_id=%s",
                self._vllm_device_log_label,
                self.instance_id,
            )
            try:
                engine.shutdown()
            finally:
                self.engine = None
                for request_id in requests:
                    await self.request_trace.delete_request(request_id)

        logger.info(
            "vLLM %s backend shut down instance_id=%s",
            self._vllm_device_log_label,
            self.instance_id,
        )

    async def stop(self):
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        deadline = time.monotonic() + float(
            (getattr(self, "backend_config", None) or {}).get("request_drain_timeout_s", 600)
        )
        while await self.request_trace.request_count() > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("Request drain timed out; aborting remaining requests")
                break
            logger.info("Waiting for all requests to finish")
            await asyncio.sleep(min(1, remaining))
        logger.info(
            "All requests finished. Shutting down the %s backend.",
            self._vllm_device_log_label,
        )
        await self.shutdown()

# ----------------------------------------------------------------------------- #
#  vLLM shared helpers (CPU / GPU backends import from here)                    #
# ----------------------------------------------------------------------------- #


def process_output(
    output: RequestOutput, latency_metrics: Dict[str, Any], model_name: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    choices: List[Dict[str, Any]] = [
        {
            "index": idx,
            "text": result.text,
            "logprobs": result.logprobs,
            "finish_reason": result.finish_reason,
        }
        for idx, result in enumerate(output.outputs)
    ]

    api_response = {
        "id": output.request_id,
        "object": "chat.completion",
        "created": (
            int(time.time())
            if output.metrics is None
            else output.metrics.arrival_time
        ),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": sum(
                len(result.token_ids) for result in output.outputs
            ),
            "total_tokens": len(output.prompt_token_ids)
            + sum(len(result.token_ids) for result in output.outputs),
        },
    }

    return api_response, latency_metrics


def snapshot_kv_transfer_params(value: Any) -> Optional[Dict[str, Any]]:
    """Copy vLLM ``RequestOutput.kv_transfer_params`` into a plain ``dict`` for Ray/handoff."""
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        out = model_dump()
        return dict(out) if isinstance(out, dict) else None
    as_dict = getattr(value, "__dict__", None)
    if isinstance(as_dict, dict):
        return {k: v for k, v in as_dict.items() if not k.startswith("_")}
    return None


def build_core_filtered_engine_config(
    model: str, backend_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Fields for ``AsyncEngineArgs`` before ShmConnector / lazy-load overrides."""
    async_engine_fields = {f.name for f in fields(AsyncEngineArgs)}
    filtered: Dict[str, Any] = {
        k: v for k, v in backend_config.items() if k in async_engine_fields
    }

    torch_dtype = backend_config.get("torch_dtype")
    if torch_dtype is not None:
        filtered["dtype"] = torch_dtype

    # Deploy-time weight path (``pretrained_model_name_or_path``) is set by routers
    # from ``model_config.model_path``; if absent, treat *model* as the path/id.
    pretrained_path = backend_config.get("pretrained_model_name_or_path")
    filtered["model"] = pretrained_path if pretrained_path else model

    # vllm mode: use standard vLLM weight loading.
    # serverlessllm / hydraserve / default: use shared-memory loading.
    if backend_config.get("vllm"):
        filtered["load_format"] = "auto"
    else:
        filtered["load_format"] = "shm"

    filtered["enforce_eager"] = backend_config.get("enforce_eager", False)
    filtered["enable_prefix_caching"] = backend_config.get(
        "enable_prefix_caching", True
    )
    filtered["task"] = backend_config.get("task", "auto")

    load_method = backend_config.get("load_method", "tokenwise")
    if load_method is not None:
        filtered["load_method"] = load_method

    return filtered


def parse_vllm_generate_request(
    request_data: Dict[str, Any], default_model_name: str
) -> Tuple[
    str,
    Union[str, TokensPrompt],
    str,
    Dict[str, Any],
    Optional[Dict[str, Any]],
]:
    """
    Returns ``(model_name, inputs, request_id, sampling_kwargs, extra_args)`` for
    ``SamplingParams(**sampling_kwargs)``. ``extra_args`` (e.g. ShmConnector
    ``kv_transfer_params``) is passed to ``AsyncLLMEngine.generate`` when supported.
    """
    rd = dict(request_data)
    model_name: str = rd.pop("model", default_model_name)
    text_prompt = request_text_prompt(rd)
    rd.pop("messages", None)
    input_tokens = rd.pop("input_tokens", None)
    rd.pop("prompt", None)
    inputs: Union[str, TokensPrompt] = text_prompt
    if input_tokens is not None:
        inputs = TokensPrompt(prompt_token_ids=input_tokens)
    request_id: str = rd.pop("request_id", f"chatcmpl-{uuid.uuid4()}")
    # Set by the controller for scheduling; not a vLLM SamplingParams field.
    rd.pop("input_length", None)
    extra_args = rd.pop("extra_args", None)
    return model_name, inputs, request_id, rd, extra_args


def vllm_engine_generate(
    engine: Any,
    inputs: Any,
    sampling_params: Any,
    request_id: str,
    extra_args: Optional[Dict[str, Any]] = None,
    *,
    sampling_kw_without_extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call ``engine.generate``, passing ``extra_args`` when the engine supports it."""
    if extra_args is None:
        return engine.generate(inputs, sampling_params, request_id)
    try:
        signature = inspect.signature(engine.generate)
    except (TypeError, ValueError):
        # Uninspectable callables get one submission, never an error-driven retry.
        signature = None
    if signature is not None:
        try:
            signature.bind(inputs, sampling_params, request_id, extra_args=extra_args)
        except TypeError:
            if sampling_kw_without_extra is None:
                raise TypeError("Engine cannot accept extra_args; KV handoff parameters must not be dropped")
            merged = dict(sampling_kw_without_extra)
            merged["extra_args"] = extra_args
            sp = SamplingParams(**merged)
            return engine.generate(inputs, sp, request_id)
    return engine.generate(inputs, sampling_params, request_id, extra_args=extra_args)


def _parse_cpu_list(cpu_list: str) -> List[int]:
    cpus: List[int] = []
    for item in cpu_list.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(item))
    return cpus


def read_numa_cpu_affinity(numa_node_id: int) -> List[int]:
    cpulist_path = f"/sys/devices/system/node/node{numa_node_id}/cpulist"
    if not os.path.exists(cpulist_path):
        return []
    try:
        with open(cpulist_path, "r", encoding="utf-8") as f:
            return _parse_cpu_list(f.read().strip())
    except Exception:
        return []


def async_engine_args_from_dict(filtered: Dict[str, Any]) -> AsyncEngineArgs:
    logger.info("Creating new VLLM engine with config: %s", filtered)
    return AsyncEngineArgs(**filtered)
