# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
import os
import uuid
import time
from typing import Any, Dict, List, Mapping, Optional

import ray

from sllm.sllm.logger import init_logger
from sllm.sllm.loading_perf_profile import (
    layer_idxes_by_pp_stages,
    load_loading_perf_profile,
    solve_lazy_load_method,
    solve_static,
)
from sllm.routers import CpuModelRouter, RoundRobinRouter
from sllm.routers.router_utils import await_backend_call, validate_auto_scaling_config
from sllm.schedulers import FcfsScheduler
from sllm.sllm.utils import TokenizerWrapper, request_text_prompt

logger = init_logger(__name__)


def _parse_boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _positive_parallel_size(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a positive integer") from None
    if parsed <= 0 or (not isinstance(value, str) and parsed != value):
        raise ValueError(f"{name} must be a positive integer")
    return parsed


class _ColdStartError(Exception):
    def __init__(self, response):
        self.response = response
        super().__init__(response["error"])


def merge_tokenwise_metrics(cpu, gpu, start, cpu_start, gpu_start, end, cpu_prefill_complete):
    """Merge local duration offsets, without comparing clocks across workers."""
    gpu_first = gpu_start + gpu.get("ttft", 0)
    gpu_last = gpu_start + (gpu.get("last_token_offset_s") or gpu.get("e2e", 0))
    cpu_tokens = cpu.get("output_length", 0) if cpu_prefill_complete else 0
    gpu_tokens = gpu.get("output_length", 0)
    intervals = list(gpu.get("itls") or [])
    if cpu_tokens:
        first = cpu_start + cpu.get("ttft", 0)
        cpu_last = cpu_start + (cpu.get("last_token_offset_s") or cpu.get("e2e", cpu.get("ttft", 0)))
        intervals = list(cpu.get("itls") or []) + ([max(0, gpu_first - cpu_last)] if gpu_tokens else []) + intervals
        if not gpu_tokens:
            gpu_last = cpu_last
        gpu["first_token_time"] = cpu.get("first_token_time")
        first_count = cpu.get("first_token_count", 1)
    else:
        first = gpu_first
        first_count = gpu.get("first_token_count", 1)
    count = cpu_tokens + gpu_tokens
    denominator = count - first_count
    gpu.update(
        ttft=max(0, first - start), e2e=end - start, output_length=count,
        itls=intervals, first_token_count=first_count,
        last_token_offset_s=max(0, gpu_last - start),
        tpot=max(0, gpu_last - first) / denominator if denominator > 0 else 0.0,
    )


def merge_tokenwise_response(cpu, gpu):
    """Return CPU-generated prefix and GPU continuation as one completion."""
    merged = dict(gpu)
    prefixes = {choice.get("index", index): choice for index, choice in enumerate(cpu.get("choices") or [])}
    merged["choices"] = [
        dict(choice, text=prefixes.get(choice.get("index", index), {}).get("text", "") + choice.get("text", ""))
        for index, choice in enumerate(gpu.get("choices") or [])
    ]
    cpu_usage = cpu.get("usage")
    gpu_usage = gpu.get("usage")
    if cpu_usage is not None and gpu_usage is not None:
        prompt = cpu_usage.get("prompt_tokens", 0)
        completion = cpu_usage.get("completion_tokens", 0) + gpu_usage.get("completion_tokens", 0)
        merged["usage"] = dict(gpu_usage, prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)
    return merged


def _cold_start_mode(config, model_config):
    backend = model_config.get("backend_config") or {}
    def enabled(name):
        return any(_parse_boolish(source.get(name, False)) for source in (config, model_config, backend))
    if (enabled("hydraserve") and int(backend.get("hydraserve_pp_size", 0) or 0) >= 2
            and int(backend.get("num_layers", 0) or 0) > 0):
        return "hydraserve"
    return next((name for name in ("serverlessllm", "vllm") if enabled(name)), "default")


class _ColdStartOperations:
    """Own local waiters and remote calls for one CPU/GPU handoff."""

    def __init__(self):
        self.calls = {}

    def start(self, awaitable):
        task = asyncio.ensure_future(awaitable)
        self.calls[task] = awaitable
        return task

    async def wait(self, primary, companion):
        pending = {primary, companion}
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result = task.result()
                if isinstance(result, dict) and result.get("error"):
                    raise _ColdStartError(result)
            if primary.done():
                return primary.result()

    async def close(self):
        for task, call in self.calls.items():
            if not task.done() or task.cancelled():
                # Cancelling a local ObjectRef waiter alone does not stop its actor call.
                if isinstance(call, ray.ObjectRef):
                    try:
                        ray.cancel(call, recursive=True)
                    except Exception:
                        logger.exception("Failed to cancel cold-start actor call")
                task.cancel()
        await asyncio.gather(*self.calls, return_exceptions=True)


class SllmController:
    def __init__(self, config: Optional[Mapping] = None):
        self.config = dict(config or {})
        original_amx = _parse_boolish(self.config.get("original_amx", False))
        self.loading_perf_profile = load_loading_perf_profile(original_amx=original_amx)

        self.running_lock = asyncio.Lock()
        self.running = False
        self._shutting_down = False

        self.metadata_lock = asyncio.Lock()
        self._model_lifecycle_lock = asyncio.Lock()
        self._cpu_plan_locks: Dict[str, asyncio.Lock] = {}
        self._active_requests = set()
        self.scheduler = None
        self.gpu_request_routers = {}
        self.cpu_request_routers = {} # 每个model在CPU的router中只有一个instance
        # Register model info
        self.registered_models = {}
        self.tokenizers: Dict[str, TokenizerWrapper] = {}
        # Tracks how many models reference each CPU router (for safe shared deletion).
        # Key: authoritative owner model name; Value: set of model names sharing it.
        self._cpu_router_refs: Dict[str, set] = {}

        # ── Observability: solver timing & resource monitoring ────────────────
        # Each entry: {wall_clock, model, input_length, method, duration_s}
        self._solver_timings: List[Dict] = []
        # Each entry: {timestamp, active_instances, active_gpus, per_model}
        self._resource_stats: List[Dict] = []
        self._monitor_task: Optional[asyncio.Task] = None
        # perf_counter epoch when the monitor loop started (set in start()).
        self._monitor_t0: float = 0.0
        # Output directory for stats files; configurable via config["stats_dir"].
        self._stats_dir: str = str(self.config.get("stats_dir", "."))


    def _gpu_worker_resources(self) -> Dict[str, float]:
        """Ray ``resources=`` for scheduler / routers on the head (``gpu_worker_0`` slot)."""
        frac = float(self.config.get("head_plane_resource_fraction", 0.001))
        return {"gpu_worker_0": frac}

    def _head_plane_resources(self) -> Dict[str, float]:
        """Alias for :meth:`_gpu_worker_resources` (tests and call sites)."""
        return self._gpu_worker_resources()

    def _generate_lazy_load_method(
        self, model_name: str, input_length: int, resolved_model_config=None
    ) -> List[Any]:
        model_config = resolved_model_config if resolved_model_config is not None else self.registered_models.get(model_name, {})
        backend_config = dict(model_config.get("backend_config", {}))

        t_solve_start = time.perf_counter()
        if _parse_boolish(self.config.get("static", False)):
            logger.info(
                "static ablation: bypassing solver for model=%s input_length=%d",
                model_name,
                input_length,
            )
            solved = solve_static(
                self.loading_perf_profile,
                model_name,
                input_length,
                backend_config,
                model_config,
            )
        else:
            solved = solve_lazy_load_method(
                self.loading_perf_profile,
                model_name,
                input_length,
                backend_config,
                model_config,
            )
        t_solve_end = time.perf_counter()

        self._solver_timings.append({
            "wall_clock": t_solve_start,
            "model": model_name,
            "input_length": input_length,
            "method": solved[0] if solved else None,
            "duration_s": t_solve_end - t_solve_start,
        })
        return solved

    def _resolve_num_gpus_per_instance(self, mname: str, models_cfg: Dict) -> int:
        """Return the number of physical GPUs consumed by one loaded instance of *mname*.

        Priority:
        1. Top-level ``num_gpus`` in the model config (if > 0).
        2. ``tensor_parallel_size * pipeline_parallel_size`` from backend_config.
        3. Falls back to 1.
        """
        mcfg = models_cfg.get(mname) or {}
        top_level = int(mcfg.get("num_gpus") or 0)
        if top_level > 0:
            return top_level
        bc = mcfg.get("backend_config") or {}
        tp = int(bc.get("tensor_parallel_size", 1) or 1)
        pp = int(bc.get("pipeline_parallel_size", 1) or 1)
        return max(1, tp * pp)

    async def _snapshot_resources(self, t0: float) -> None:
        """Capture one resource snapshot and append to ``_resource_stats``."""
        ts = time.perf_counter() - t0
        per_model: Dict[str, Dict] = {}
        total_instances = 0
        total_gpus = 0
        async with self.metadata_lock:
            routers = dict(self.gpu_request_routers)
            models_cfg = dict(self.registered_models)
        for mname, router in routers.items():
            try:
                status = await router.get_instance_pool_status.remote()
                loaded = int(status.get("loaded_ready", 0))
                available = int(status.get("loaded_available", 0))
                num_gpus_per_inst = self._resolve_num_gpus_per_instance(mname, models_cfg)
                per_model[mname] = {
                    "loaded_instances": loaded,
                    "loaded_available": available,
                    "gpus": loaded * num_gpus_per_inst,
                    "num_gpus_per_instance": num_gpus_per_inst,
                    "empty_ready": int(status.get("empty_ready", 0)),
                    "empty_starting": int(status.get("empty_starting", 0)),
                }
                total_instances += loaded
                total_gpus += loaded * num_gpus_per_inst
            except Exception as exc:
                logger.warning(
                    "resource_monitor: failed to query router for %s: %s",
                    mname, exc,
                )
        self._resource_stats.append({
            "timestamp_s": ts,
            "active_instances": total_instances,
            "active_gpus": total_gpus,
            "per_model": per_model,
        })

    async def _resource_monitor_loop(self, interval_s: float = 2.0) -> None:
        """Background task: periodically snapshot active GPU and instance counts.

        The first snapshot is taken immediately (t=0) so that even very short
        benchmark runs produce at least one data point.  Subsequent snapshots
        follow every *interval_s* seconds.
        """
        self._monitor_t0 = time.perf_counter()
        # Immediate snapshot at t≈0 so short runs still produce data.
        await self._snapshot_resources(self._monitor_t0)
        while True:
            await asyncio.sleep(interval_s)
            await self._snapshot_resources(self._monitor_t0)

    async def save_stats(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Write solver timing and resource stats to JSON files in *output_dir*.

        A final resource snapshot is taken immediately before writing so the
        file always ends with the most up-to-date state.

        Returns a dict mapping stat type → saved file path.
        """
        import json
        import os
        # Flush one last snapshot so the file always ends with the state at the
        # exact moment save_stats was called (not the last periodic poll).
        if self._monitor_task is not None:
            try:
                await self._snapshot_resources(self._monitor_t0)
            except Exception as exc:
                logger.warning("save_stats: final snapshot failed: %s", exc)
        out_dir = output_dir or self._stats_dir
        os.makedirs(out_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        solver_path = os.path.join(out_dir, "solver_timings.json")
        with open(solver_path, "w") as f:
            json.dump(self._solver_timings, f, indent=2)
        paths["solver_timings"] = solver_path
        logger.info("Saved %d solver timing records to %s", len(self._solver_timings), solver_path)

        resource_path = os.path.join(out_dir, "resource_stats.json")
        with open(resource_path, "w") as f:
            json.dump(self._resource_stats, f, indent=2)
        paths["resource_stats"] = resource_path
        logger.info("Saved %d resource snapshots to %s", len(self._resource_stats), resource_path)

        return paths

    async def start(self):
        async with self._model_lifecycle_lock:
            return await self._start_controller()

    async def _start_controller(self):
        async with self.running_lock:
            if self.running:
                logger.error("Controller already started")
                raise RuntimeError("Controller already started")
            try:
                await self._start()
                self.running = True
                self._shutting_down = False
            except BaseException:
                if self.scheduler is not None:
                    ray.kill(self.scheduler, no_restart=True)
                    self.scheduler = None
                raise

    async def _start(self):
        logger.info("Starting scheduler")
        ray_scheduler_cls = ray.remote(FcfsScheduler)

        self.router_cls = ray.remote(RoundRobinRouter)
        self.cpu_router_cls = ray.remote(CpuModelRouter)

        self.scheduler = ray_scheduler_cls.options(
            name="model_loading_scheduler",
            resources=self._gpu_worker_resources(),
            max_concurrency=100,
        ).remote(scheduler_config=self.config.get("scheduler_config", {}))
        await await_backend_call(self.scheduler.start.remote())

        monitor_interval = float(self.config.get("resource_monitor_interval_s", 2.0))
        self._monitor_task = asyncio.create_task(
            self._resource_monitor_loop(interval_s=monitor_interval)
        )
        logger.info(
            "Resource monitor started (interval=%.1fs, stats_dir=%s)",
            monitor_interval, self._stats_dir,
        )

    async def register(self, model_config):
        async with self._model_lifecycle_lock:
            owned_routers = []
            try:
                return await self._register(model_config, owned_routers)
            except BaseException:
                await asyncio.gather(
                    *(self._stop_router(router) for router in owned_routers),
                    return_exceptions=True,
                )
                model_name = model_config.get("model")
                if model_name not in self.registered_models:
                    self.tokenizers.pop(model_name, None)
                raise

    async def _stop_router(self, router):
        # Only kill the actor after a successful graceful shutdown.  If shutdown
        # fails, keep the handle alive so the caller can retry instead of losing
        # the only reference to a potentially still-running router.
        await router.shutdown.remote()
        ray.kill(router, no_restart=True)

    async def _register(self, model_config, owned_routers):
        if not self.running:
            raise RuntimeError("Controller not running")
        model_name = model_config.get("model")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Model name is required")
        backend = model_config.get("backend", None)
        if backend is None:
            raise ValueError(f"Backend not specified for model {model_name}")
        backend_config = model_config.get("backend_config", {})
        auto_scaling_config = model_config.get("auto_scaling_config", None)
        validate_auto_scaling_config(auto_scaling_config)

        # model_path is the path/repo used to load weights; falls back to model_name.
        model_path = (
            model_config.get("model_path")
            or backend_config.get("pretrained_model_name_or_path")
            or model_name
        )
        # Ensure backend_config always carries the resolved path.
        backend_config = dict(backend_config)
        backend_config.setdefault("pretrained_model_name_or_path", model_path)
        # Keep top-level deployment aliases and backend configuration consistent;
        # GPU placement and the planner must see one layout; CPU always uses PP=1.
        for key in ("tensor_parallel_size", "pipeline_parallel_size", "num_layers"):
            if key in model_config:
                if key != "num_layers" and key in backend_config and _positive_parallel_size(model_config[key], key) != _positive_parallel_size(backend_config[key], key):
                    raise ValueError(f"Conflicting {key} in model and backend configuration")
                backend_config[key] = model_config[key]
        pp = _positive_parallel_size(backend_config.get("pipeline_parallel_size", 1), "pipeline_parallel_size")
        total_gpus = _positive_parallel_size(model_config["num_gpus"], "num_gpus") if "num_gpus" in model_config else None
        if "tensor_parallel_size" not in backend_config and total_gpus is not None:
            if total_gpus % pp:
                raise ValueError("num_gpus must be divisible by pipeline_parallel_size")
            backend_config["tensor_parallel_size"] = total_gpus // pp
        for key in ("tensor_parallel_size", "pipeline_parallel_size"):
            backend_config[key] = _positive_parallel_size(backend_config.get(key, 1), key)
        expected_gpus = backend_config["tensor_parallel_size"] * backend_config["pipeline_parallel_size"]
        if total_gpus is not None and total_gpus != expected_gpus:
            raise ValueError("num_gpus must equal tensor_parallel_size * pipeline_parallel_size")
        model_config = dict(model_config)
        model_config["num_gpus"] = expected_gpus
        hpp = model_config.get("hydraserve_pp_size")
        if hpp is not None:
            backend_config["hydraserve_pp_size"] = int(hpp)
        nlay = model_config.get("num_layers")
        if nlay is not None:
            backend_config["num_layers"] = int(nlay)
        for mode_name in ("hydraserve", "serverlessllm", "vllm"):
            backend_config[mode_name] = any(
                _parse_boolish(source.get(mode_name, False))
                for source in (self.config, model_config, backend_config)
            )
        if sum(backend_config[name] for name in ("hydraserve", "serverlessllm", "vllm")) > 1:
            raise ValueError("Only one cold-start baseline mode may be enabled")

        async with self.metadata_lock:
            if model_name in self.registered_models:
                logger.info(f"Model {model_name} already registered")
                return

        logger.info(f"Registering new model {model_name} (path={model_path})")

        self.tokenizers[model_name] = TokenizerWrapper(model_path, trust_remote_code=True)

        use_vllm = backend_config.get("vllm", False)
        cpu_request_router = None
        _cpu_owner: Optional[str] = None
        cpu_worker_id: Optional[str] = None

        backend_config["lazy_load"] = True

        if not use_vllm:
            # ── CPU router: create a new one or reuse a shared one ───────────────
            # ``shared_cpu_model`` lets multiple GPU-router replicas share one CPU
            # backend on the same node, avoiding duplicate CPU engine allocations in
            # multi-model benchmarks.
            shared_cpu_model: Optional[str] = model_config.get("shared_cpu_model")

            if shared_cpu_model:
                shared_cfg = self.registered_models.get(shared_cpu_model)
                if shared_cfg is None or shared_cpu_model not in self.cpu_request_routers:
                    raise ValueError(f"Shared CPU model {shared_cpu_model!r} is not available")
                if shared_cfg.get("model_path") != model_path:
                    raise ValueError("Shared CPU models must use the same model_path")
                placement = backend_config.get("cpu_placement_node_id")
                if placement is not None:
                    shared_node = await self.cpu_request_routers[shared_cpu_model].get_cpu_placement_node_id.remote()
                    if str(placement) != str(shared_node):
                        raise ValueError("Shared CPU models must use the same CPU placement node")

            if shared_cpu_model and shared_cpu_model in self.cpu_request_routers:
                # Reuse the existing CPU router; do NOT start a new one.
                cpu_request_router = self.cpu_request_routers[shared_cpu_model]
                _cpu_owner = shared_cfg.get("_cpu_owner", shared_cpu_model)
                logger.info(
                    "Model %s: reusing CPU router from %s (shared_cpu_model)",
                    model_name,
                    shared_cpu_model,
                )
                async with self.metadata_lock:
                    if model_name in self.cpu_request_routers:
                        logger.error(f"Model {model_name} already registered")
                        return
            else:
                # TODO: put resource requirements in model_config
                cpu_resource_requirements = {
                    "num_cpus": 1,
                    "num_gpus": 0,
                }
                # Copy so CPU router sees ``cpu_placement_node_id`` / ``cpu_placement_resources``
                # (``gpu_worker_*`` only); GPU path still mutates ``backend_config`` (e.g. ``lazy_load``).
                cpu_backend_config = {
                    "cpu_placement_node_id": backend_config.get("cpu_placement_node_id", None),
                }
                if "backend_shutdown_timeout_s" in backend_config:
                    cpu_backend_config["backend_shutdown_timeout_s"] = backend_config["backend_shutdown_timeout_s"]
                if "cpu_backend_runtime_env" in backend_config:
                    cpu_backend_config["cpu_backend_runtime_env"] = backend_config.get(
                        "cpu_backend_runtime_env"
                    )
                if _parse_boolish(self.config.get("original_amx", False)):
                    renv = dict(cpu_backend_config.get("cpu_backend_runtime_env") or {})
                    ev = dict(renv.get("env_vars") or {})
                    ev["VLLM_CPU_USE_AMX_GEMM_BF16"] = "0"
                    renv["env_vars"] = ev
                    cpu_backend_config["cpu_backend_runtime_env"] = renv
                    logger.info(
                        "original_amx: injecting VLLM_CPU_USE_AMX_GEMM_BF16=0 into CPU backend env for model=%s",
                        model_name,
                    )
                cpu_request_router = self.cpu_router_cls.options(
                    name=model_name,
                    namespace="cpu_models",
                    num_cpus=1,
                    resources=self._gpu_worker_resources(),
                ).remote(
                    model_name=model_name,
                    resource_requirements=cpu_resource_requirements,
                    backend=backend,
                    backend_config=cpu_backend_config,
                    model_path=model_path,
                )
                owned_routers.append(cpu_request_router)

                async with self.metadata_lock:
                    if model_name in self.cpu_request_routers:
                        logger.error(f"Model {model_name} already registered")
                        return

                await await_backend_call(cpu_request_router.start.remote(auto_scaling_config=None))
                _cpu_owner = model_name

            cpu_worker_id = await cpu_request_router.get_cpu_placement_node_id.remote()
            if cpu_worker_id:
                logger.info(
                    "CPU instance for %s placed on node %s; GPU prewarm will prefer same node",
                    model_name,
                    cpu_worker_id,
                )
            else:
                logger.info(
                    "CPU instance for %s has no logical node_id; GPU prewarm will use default placement",
                    model_name,
                )

            # GPUBackend builds AsyncLLMEngine args at construction time; ShmConnector params
            # must match the primary CPUBackend. CpuModelRouter already probed
            # get_shm_kv_cache_info() — merge that tuple into config before creating the GPU router.
            cached_shm = await cpu_request_router.get_cached_shm_kv_cache_info.remote()
            if cached_shm is not None and len(cached_shm) == 4:
                (
                    backend_config["shm_kv_cache_size"],
                    backend_config["shm_num_blocks"],
                    backend_config["shm_block_len"],
                    backend_config["shm_tp_size"],
                ) = (int(cached_shm[0]), int(cached_shm[1]), int(cached_shm[2]), int(cached_shm[3]))
                logger.info(
                    "GPU backend_config SHM fields from primary CPU: "
                    "shm_kv_cache_size=%s shm_num_blocks=%s shm_block_len=%s shm_tp_size=%s",
                    backend_config["shm_kv_cache_size"],
                    backend_config["shm_num_blocks"],
                    backend_config["shm_block_len"],
                    backend_config["shm_tp_size"],
                )
            else:
                logger.info(
                    "No cached primary CPU SHM layout for %s; GPUBackend will use config defaults",
                    model_name,
                )

        gpu_resource_requirements = {
            "num_cpus": 1,
            "num_gpus": model_config.get("num_gpus", 0),
        }
        gpu_request_router = self.router_cls.options(   #register时直接创建cpu_router和一个未加载权重的gpu_router
            name=model_name,
            namespace="gpu_models",
            num_cpus=1,
            resources=self._gpu_worker_resources(),
        ).remote(
            model_name=model_name,
            resource_requirements=gpu_resource_requirements,
            backend=backend,
            backend_config=backend_config,
            model_path=model_path,
        )
        owned_routers.append(gpu_request_router)

        await await_backend_call(gpu_request_router.start.remote(
            auto_scaling_config=auto_scaling_config,
            cpu_worker_id=cpu_worker_id,
        ))

        logger.info(f"Model {model_name} registered")

        # Mark model as registered only after model registered successfully
        async with self.metadata_lock:
            if not self.running:
                raise RuntimeError("Controller stopped during model registration")
            stored_cfg = dict(model_config)
            stored_cfg["model_path"] = model_path
            stored_cfg["backend_config"] = dict(backend_config)
            self.registered_models[model_name] = stored_cfg
            self.gpu_request_routers[model_name] = gpu_request_router
            if not use_vllm:
                # Record which CPU owner this model refers to so delete() can do
                # safe reference-counted teardown.
                shared_cpu_model_stored = model_config.get("shared_cpu_model")
                if shared_cpu_model_stored and _cpu_owner:
                    stored_cfg["_cpu_owner"] = _cpu_owner
                self.cpu_request_routers[model_name] = cpu_request_router
                # Update ref-count map: owner key → set of model names sharing it.
                if _cpu_owner is not None:
                    self._cpu_router_refs.setdefault(_cpu_owner, set()).add(model_name)
                else:
                    self._cpu_router_refs.setdefault(model_name, set()).add(model_name)

    async def generate_stream(self, model_name: str, request_data: Dict[str, Any]):
        if getattr(self, "_shutting_down", False):
            return {"error": "Controller is shutting down"}
        request_data = dict(request_data)
        request_id = request_data.setdefault("request_id", str(uuid.uuid4()))
        if not isinstance(request_id, str) or not request_id:
            return {"error": "request_id must be a nonempty string"}
        active_requests = self._active_requests
        model_cfg = self.registered_models.get(model_name, {})
        mode = _cold_start_mode(self.config, model_cfg)
        # Shared CPU routers use one vLLM engine and therefore one request-id
        # namespace even when the GPU models differ.
        namespace = (
            ("cpu", model_cfg.get("_cpu_owner", model_name))
            if mode == "default" else ("gpu", model_name)
        )
        key = (namespace, request_id)
        # No await between checking and claiming: concurrent actor calls cannot
        # both enter the routing path for the same backend namespace/request pair.
        if key in active_requests:
            return {"error": f"Request {request_id} is already active"}
        active_requests.add(key)
        try:
            return await self._generate_stream(model_name, request_data)
        finally:
            active_requests.discard(key)

    async def _generate_stream(self, model_name: str, request_data: Dict[str, Any]):
        logger.info(f"Received request {request_data['request_id']}")

        cpu_router = None
        async with self.metadata_lock:
            if model_name not in self.gpu_request_routers:
                raise ValueError(f"Model {model_name} not found")
            model_cfg = self.registered_models.get(model_name, {})
            mode = _cold_start_mode(self.config, model_cfg)
            use_vllm = mode == "vllm"
            if not use_vllm and model_name not in self.cpu_request_routers:
                raise ValueError(f"Model {model_name} has no CPU router (non-vllm mode requires one)")
            if not use_vllm:
                cpu_router = self.cpu_request_routers.get(model_name)
            gpu_router = self.gpu_request_routers[model_name]

        logger.info(f"Got request router for {model_name}")

        try:
            request_data["prompt"] = request_text_prompt(request_data)
        except ValueError as exc:
            return {"error": str(exc)}

        input_tokens = request_data.get("input_tokens")
        if input_tokens is not None:
            if (not isinstance(input_tokens, list) or not input_tokens or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in input_tokens
            )):
                return {"error": "input_tokens must be a nonempty list of nonnegative integers"}
            input_length = len(input_tokens)
        else:
            input_length = self.tokenizers[model_name].get_prompt_len(request_data["prompt"])

        # ---- Hot path: route to loaded GPU instance when capacity is available ---- #
        pool_status = await gpu_router.get_instance_pool_status.remote()
        loaded_ready = int(pool_status.get("loaded_ready", 0))
        loaded_available = int(pool_status.get("loaded_available", 0))
        empty_ready = int(pool_status.get("empty_ready", 0))
        if mode == "default" and loaded_available == 0:
            transition = await self._try_layerwise_transition(gpu_router, cpu_router, request_data)
            if transition is not None:
                return transition
        if loaded_ready > 0:
            if loaded_available > 0:
                result_ref = gpu_router.inference.remote(
                    request_data=request_data, action="warm"
                )
                return await await_backend_call(result_ref)
            if empty_ready == 0:
                if mode == "hydraserve":
                    await gpu_router.hydraserve_ensure_prewarm_spike.remote()
                else:
                    await gpu_router.ensure_one_empty_instance.remote()
            # Hydra / serverlessllm: no CPU fallback; fall through to cold start

        reservation_id = str(uuid.uuid4())
        node_id = None
        claimed = False
        deadline = time.monotonic() + float(self.config.get("cold_start_wait_timeout_s", 600))
        try:
            while time.monotonic() < deadline:
                if mode == "default":
                    transition = await self._try_layerwise_transition(gpu_router, cpu_router, request_data)
                    if transition is not None:
                        return transition
                try:
                    reservation = await asyncio.wait_for(
                        await_backend_call(gpu_router.reserve_cold_start.remote(
                            request_id=request_data["request_id"], reservation_id=reservation_id, mode=mode,
                        )), timeout=max(0, deadline - time.monotonic()),
                    )
                except TimeoutError:
                    return {"error": "Timed out waiting for cold-start capacity"}
                state = reservation["state"]
                if state in ("unavailable", "stopped"):
                    return {"error": "No GPU instance available for cold start"}
                if state == "reserved":
                    node_id = reservation["node_id"]
                    try:
                        claimed, _ = await asyncio.wait_for(
                            await_backend_call(self.scheduler.try_claim_cold_start.remote(
                                node_id=node_id, model_name=model_name, claim_id=reservation_id,
                            )), timeout=max(0, deadline - time.monotonic()),
                        )
                    except TimeoutError:
                        return {"error": "Timed out waiting for cold-start capacity"}
                    if claimed:
                        break
                    await gpu_router.release_cold_start.remote(reservation_id)
                try:
                    refreshed = await asyncio.wait_for(
                        await_backend_call(gpu_router.get_instance_pool_status.remote()),
                        timeout=max(0, deadline - time.monotonic()),
                    )
                except TimeoutError:
                    return {"error": "Timed out waiting for cold-start capacity"}
                if int(refreshed.get("loaded_available", 0)) > 0:
                    return await await_backend_call(gpu_router.inference.remote(
                        request_data=request_data, action="warm"
                    ))
                await asyncio.sleep(0.01)
            else:
                return {"error": "Timed out waiting for cold-start capacity"}
            layout = reservation.get("parallel_layout") or {}
            if layout and mode != "hydraserve":
                model_cfg = dict(model_cfg)
                model_cfg["backend_config"] = dict(model_cfg.get("backend_config") or {})
                for key in ("tensor_parallel_size", "pipeline_parallel_size"):
                    if key in layout:
                        model_cfg[key] = model_cfg["backend_config"][key] = layout[key]
            return await self._run_cold_start(
                model_name, request_data, node_id, gpu_router, cpu_router,
                input_length, model_cfg, reservation_id,
            )
        finally:
            try:
                if node_id is not None:
                    await self.scheduler.finish_cold_start.remote(
                        node_id=node_id, model_name=model_name, claim_id=reservation_id,
                    )
            finally:
                await gpu_router.release_cold_start.remote(reservation_id)

    async def _try_layerwise_transition(self, gpu_router, cpu_router, request_data):
        request_id = request_data["request_id"]
        admission_id = str(uuid.uuid4())
        operations = _ColdStartOperations()
        try:
            instance_id = await await_backend_call(gpu_router.admit_layerwise_transition.remote(request_id, admission_id))
            if instance_id is None:
                return None
            cpu_instance = await cpu_router.get_instance.remote()
            if cpu_instance is None or cpu_instance.backend_instance is None:
                return {"error": "CPU instance not available for layerwise transition"}
            cpu_task = operations.start(cpu_router.inference.remote(request_data=request_data, action="generate"))
            gpu_task = operations.start(gpu_router.inference.remote(request_data=request_data, action="generate"))
            return await operations.wait(gpu_task, cpu_task)
        except _ColdStartError as exc:
            return exc.response
        finally:
            try:
                await operations.close()
            finally:
                await gpu_router.release_transition_request.remote(admission_id)

    async def _run_cold_start(
        self, model_name: str, request_data: Dict[str, Any], node_id: str,
        gpu_router: Any, cpu_router: Any, input_length: int, model_cfg: Mapping,
        reservation_id: str,
    ):
        """Execute a claimed cold start; the caller owns claim release."""
        operations = _ColdStartOperations()
        cpu_lock = None
        lock_acquired = False
        waiting_for_cpu = True
        try:
            if _cold_start_mode(self.config, model_cfg) == "default":
                owner = model_cfg.get("_cpu_owner", model_name)
                cpu_lock = self._cpu_plan_locks.setdefault(owner, asyncio.Lock())
                async with asyncio.timeout(float(self.config.get("cold_start_wait_timeout_s", 600))):
                    await cpu_lock.acquire()
                    lock_acquired = True
                    while True:
                        async with self.metadata_lock:
                            sharers = [
                                router for name, router in self.gpu_request_routers.items()
                                if name != model_name
                                and self.registered_models.get(name, {}).get("_cpu_owner", name) == owner
                            ]
                        busy = False
                        for router in sharers:
                            if await await_backend_call(router.has_cpu_handoff.remote()):
                                busy = True
                                break
                        if not busy:
                            break
                        await asyncio.sleep(0.01)
            waiting_for_cpu = False
            return await self._execute_cold_start(
                model_name, request_data, node_id, gpu_router, cpu_router,
                input_length, model_cfg, operations, reservation_id,
            )
        except _ColdStartError as exc:
            return exc.response
        except TimeoutError:
            if not waiting_for_cpu:
                raise
            return {"error": "Timed out waiting for shared CPU cold-start capacity"}
        finally:
            try:
                await operations.close()
            finally:
                if lock_acquired:
                    cpu_lock.release()

    async def _execute_cold_start(
        self, model_name: str, request_data: Dict[str, Any], node_id: str,
        gpu_router: Any, cpu_router: Any, input_length: int, model_cfg: Mapping,
        operations: _ColdStartOperations,
        reservation_id: str,
    ):
        bc0 = model_cfg.get("backend_config") or {}
        pp_size = int(model_cfg.get("pipeline_parallel_size", bc0.get("pipeline_parallel_size", 1)) or 1)
        is_pp_model = pp_size > 1
        mode = _cold_start_mode(self.config, model_cfg)

        # Cold start logic
        start_time = time.perf_counter()

        if mode == "hydraserve":
            # HydraServe: no CPU path, no loading_perf solver. GPU-only pair+primary full load, then generate on pair.
            logger.info(
                "Cold start (HydraServe) model=%s input_length=%s",
                model_name,
                input_length,
            )
            await self.scheduler.start_cold_start.remote(
                node_id=node_id, model_name=model_name, method="hydraserve"
            )
            t0 = time.perf_counter()
            await await_backend_call(gpu_router.hydraserve_parallel_cold_load.remote(
                request_id=request_data.get("request_id"), reservation_id=reservation_id,
            ))
            await self.scheduler.signal_cold_start_ready.remote(node_id=node_id)
            gpu_result = await await_backend_call(gpu_router.inference.remote(
                request_data=request_data, action="generate"
            ))
            end_t = time.perf_counter()
            if isinstance(gpu_result, (list, tuple)) and len(gpu_result) > 1:
                gpu_result[1]["e2e"] = end_t - t0
            if isinstance(gpu_result, (list, tuple)) and len(gpu_result) > 1:
                _, met = gpu_result
                logger.info(
                    f"HydraServe cold start finished. E2E: {met.get('e2e', 'N/A')}, "
                    f"TTFT: {met.get('ttft', 'N/A')}"
                )
            try:
                await gpu_router.hydraserve_retire_spike.remote()
            except Exception:
                logger.exception(
                    "hydraserve_retire_spike failed model=%s", model_name
                )
            try:
                await gpu_router.hydraserve_ensure_prewarm_spike.remote()
            except Exception:
                logger.exception(
                    "hydraserve_ensure_prewarm_spike failed model=%s",
                    model_name,
                )
            return gpu_result

        if mode in ("serverlessllm", "vllm"):
            # serverlessllm: single prewarmed instance, full lazy_init, no CPU, no solver
            method = mode
            logger.info(
                "Cold start (%s baseline) model=%s input_length=%s",
                method,
                model_name,
                input_length,
            )
            await self.scheduler.start_cold_start.remote(
                node_id=node_id, model_name=model_name, method=method
            )
            t0 = time.perf_counter()
            n = int(bc0.get("num_layers", 0) or 0)
            full_layers = layer_idxes_by_pp_stages(n, pp_size)
            await await_backend_call(gpu_router.lazy_init.remote(
                layer_idxes=full_layers,
                load_method="tokenwise",
                request_id=request_data.get("request_id"),
                mark_weights_loaded=True,
                mode=method,
                reservation_id=reservation_id,
            ))
            t1 = time.perf_counter()
            await self.scheduler.signal_cold_start_ready.remote(node_id=node_id)
            gpu_result = await await_backend_call(gpu_router.inference.remote(
                request_data=request_data, action="generate"
            ))
            end_t = time.perf_counter()
            if isinstance(gpu_result, (list, tuple)) and len(gpu_result) > 1:
                gpu_result[1]["e2e"] = end_t - t0
                gpu_result[1]["ttft"] += t1 - t0
            if isinstance(gpu_result, (list, tuple)) and len(gpu_result) > 1:
                _, met = gpu_result
                logger.info(
                    "%s cold start finished. E2E: %s, TTFT: %s",
                    method,
                    met.get("e2e", "N/A"),
                    met.get("ttft", "N/A"),
                )
            try:
                await gpu_router.serverlessllm_ensure_prewarm.remote()
            except Exception:
                logger.exception(
                    "ensure_prewarm failed model=%s", model_name
                )
            return gpu_result

        # Standard cold start: loading_perf solver + CPU-assisted lazy load
        registered_backend = self.registered_models.get(model_name, {}).get("backend_config") or {}
        layout_changed = any(
            int(bc0.get(key, 1)) != int(registered_backend.get(key, 1))
            for key in ("tensor_parallel_size", "pipeline_parallel_size")
        )
        methods = (
            self._generate_lazy_load_method(model_name, input_length, model_cfg)
            if layout_changed else self._generate_lazy_load_method(model_name, input_length)
        )
        logger.info("input_length: %s, methods: %s", input_length, methods)
        if not methods or len(methods) < 3:
            return {"error": "No loading plan available", "model": model_name}

        cpu_instance = await cpu_router.get_instance.remote()
        if cpu_instance is None or cpu_instance.backend_instance is None:
            return {"error": "CPU instance not available"}
        cpu_backend = cpu_instance.backend_instance
        await await_backend_call(cpu_backend.update_pp_cleanup.remote(enabled=is_pp_model))
        load_method = methods[0]
        await await_backend_call(cpu_backend.set_load_method.remote(method=load_method))

        # Push the resolved load method into the CPU backend immediately.
        # GPU backend is set inside the router's lazy_load_weights once the target
        # instance has been selected.
        if methods[0] == "layerwise":
            # Mark cold start in progress (layerwise) at scheduler (node) level
            await self.scheduler.start_cold_start.remote(
                node_id=node_id, model_name=model_name, method="layerwise"
            )
            logger.info(f"Starting cold start for model {model_name} (layerwise)")

            # 1. Update CPU computing layers
            await await_backend_call(cpu_backend.update_computing_layers.remote(
                computing_layers=methods[1]
            ))

            start_lazy_init_time = time.perf_counter()
            # 2. Start CPU generation (background)
            cpu_task = operations.start(cpu_backend.generate.remote(request_data=request_data))

            # 3. Wait for GPU weights
            layer_idxes = methods[2]
            gpu_load_task = operations.start(gpu_router.lazy_init.remote(
                reservation_id=reservation_id,
                layer_idxes=layer_idxes,
                load_method=methods[0],
                request_id=request_data.get("request_id"),
            ))
            await operations.wait(gpu_load_task, cpu_task)
            lazy_init_time = time.perf_counter() - start_lazy_init_time

            # 4. Start GPU generation
            # NOTE: For layerwise, the first-token callback in vllm_backend
            # calls router.notify_first_token → scheduler.notify_first_token,
            # which sets the node's ready event so the next cold start on
            # the same node may begin.
            gpu_task = operations.start(gpu_router.inference.remote(
                request_data=request_data, action="generate"
            ))
            result = await operations.wait(gpu_task, cpu_task)

            if isinstance(result, dict) and result.get("error"):
                logger.error(
                    "Layerwise cold start inference error model=%s: %s",
                    model_name, result["error"],
                )
                return result

            if not isinstance(result, (list, tuple)) or len(result) < 2:
                logger.error(
                    "Layerwise cold start unexpected result shape model=%s: %r",
                    model_name, result,
                )
                return {"error": "Unexpected inference result", "detail": repr(result)}

            _, metrics = result

            logger.info(f"original ttft: {metrics.get('ttft', 'N/A')}")

            metrics['e2e'] += lazy_init_time
            metrics['ttft'] += lazy_init_time
            logger.info(
                f"Cold start finished. E2E: {metrics['e2e']:.4f}s, "
                f"TTFT: {metrics.get('ttft', 'N/A')}, "
                f"TPOT: {metrics.get('tpot', 'N/A')}"
            )
            try:
                await gpu_router.ensure_one_empty_instance.remote()
            except Exception:
                logger.exception(
                    "Failed to replenish empty prewarm instance after cold start (layerwise) model=%s",
                    model_name,
                )
            return result

        else:
            # Tokenwise cold start
            await self.scheduler.start_cold_start.remote(
                node_id=node_id, model_name=model_name, method="tokenwise"
            )
            logger.info(f"Starting cold start for model {model_name} (tokenwise)")

            start_time = time.perf_counter()
            cpu_compute_tokens = methods[1]
            cpu_request_data = dict(request_data)
            cpu_request_data["extra_args"] = dict(request_data.get("extra_args") or {})
            cpu_request_data["extra_args"]["kv_transfer_params"] = {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "max_num_prefill_compute_tokens": cpu_compute_tokens,
            }
            cpu_request_start_time = time.perf_counter()
            cpu_task = operations.start(cpu_backend.generate.remote(request_data=cpu_request_data))

            layer_idxes = methods[2]
            gpu_load_task = operations.start(gpu_router.lazy_init.remote(
                reservation_id=reservation_id,
                layer_idxes=layer_idxes,
                load_method=methods[0],
                request_id=request_data.get("request_id"),
            ))

            cpu_result = await operations.wait(cpu_task, gpu_load_task)
            if (
                not isinstance(cpu_result, (list, tuple))
                or len(cpu_result) < 2
            ):
                logger.error(
                    "Unexpected CPU generate result model=%s: %r",
                    model_name,
                    cpu_result,
                )
                return {
                    "error": "Invalid CPU generation result",
                    "detail": repr(cpu_result),
                }

            cpu_api, cpu_metrics = cpu_result[0], cpu_result[1]
            choices = (cpu_api.get("choices") or []) if isinstance(
                cpu_api, dict
            ) else []
            if not choices:
                return {
                    "error": "CPU generation returned no choices",
                    "detail": cpu_api,
                }
            cpu_text = choices[0].get("text")
            if cpu_text is None:
                return {
                    "error": "CPU generation choice has no text",
                    "detail": choices[0],
                }

            # CPU computation done → tokenwise readiness criteria met.
            # Signal at the node level so that the next cold start may begin.
            await self.scheduler.signal_cold_start_ready.remote(node_id=node_id)
            logger.info("signal_cold_start_ready")

            await operations.wait(gpu_load_task, cpu_task)

            cpu_prefill_complete = cpu_compute_tokens >= input_length
            cpu_output_length = cpu_metrics.get("output_length", 0) if cpu_prefill_complete else 0
            max_tokens = request_data.get("max_tokens", cpu_metrics.get("sampling_max_tokens"))
            cpu_stopped = cpu_prefill_complete and all(choice.get("finish_reason") == "stop" for choice in choices)
            if cpu_stopped or (isinstance(max_tokens, int) and cpu_output_length >= max_tokens):
                response = dict(cpu_api)
                response.pop("prompt_token_ids", None)
                response.pop("output_token_ids", None)
                response.pop("kv_transfer_params", None)
                metrics = dict(cpu_metrics)
                metrics["e2e"] = time.perf_counter() - start_time
                metrics["ttft"] = cpu_request_start_time - start_time + cpu_metrics.get("ttft", 0)
                try:
                    await gpu_router.ensure_one_empty_instance.remote()
                except Exception:
                    logger.exception("Failed to replenish prewarm after CPU completed request model=%s", model_name)
                return response, metrics

            gpu_request_data = dict(request_data)
            gpu_request_data["extra_args"] = dict(request_data.get("extra_args") or {})
            gpu_request_data["request_id"] = request_data.get("request_id")
            if cpu_prefill_complete:
                gpu_request_data["prompt"] += cpu_text
            if isinstance(max_tokens, int) and cpu_output_length:
                gpu_request_data["max_tokens"] = max_tokens - cpu_output_length
            min_tokens = request_data.get("min_tokens")
            if isinstance(min_tokens, int) and cpu_output_length:
                gpu_request_data["min_tokens"] = max(0, min_tokens - cpu_output_length)
            prompt_tokens = request_data.get("input_tokens")
            if prompt_tokens is None:
                prompt_tokens = cpu_api.get("prompt_token_ids")
            if prompt_tokens is not None:
                output_tokens = cpu_api.get("output_token_ids") if cpu_prefill_complete else []
                if output_tokens is None and cpu_text:
                    return {"error": "CPU tokenwise handoff did not return output token IDs"}
                gpu_request_data["input_tokens"] = list(prompt_tokens) + list(output_tokens or [])
            cpu_kv = cpu_api.get("kv_transfer_params") or {}
            gpu_kv = dict(cpu_kv)
            gpu_kv["do_remote_decode"] = False
            gpu_kv["do_remote_prefill"] = True
            gpu_kv["num_computed_tokens"] = cpu_kv.get("num_computed_tokens", cpu_compute_tokens)
            gpu_kv["remote_block_ids"] = cpu_kv.get("remote_block_ids", [])
            gpu_request_data["extra_args"]["kv_transfer_params"] = gpu_kv

            gpu_request_start_time = time.perf_counter()
            gpu_task = operations.start(gpu_router.inference.remote(
                request_data=gpu_request_data, action="generate"
            ))
            gpu_result = await gpu_task

            if isinstance(gpu_result, dict) and gpu_result.get("error"):
                logger.error(
                    "Tokenwise cold start GPU inference error model=%s: %s",
                    model_name, gpu_result["error"],
                )
                return gpu_result

            if not isinstance(gpu_result, (list, tuple)) or len(gpu_result) < 2:
                logger.error(
                    "Tokenwise cold start unexpected GPU result shape model=%s: %r",
                    model_name, gpu_result,
                )
                return {"error": "Unexpected GPU inference result", "detail": repr(gpu_result)}

            end_time = time.perf_counter()
            merge_tokenwise_metrics(
                cpu_metrics, gpu_result[1], start_time, cpu_request_start_time,
                gpu_request_start_time, end_time,
                cpu_prefill_complete,
            )

            if cpu_prefill_complete and cpu_output_length:
                gpu_result = (merge_tokenwise_response(cpu_api, gpu_result[0]), gpu_result[1])

            _, metrics = gpu_result
            logger.info(
                f"Cold start finished. E2E: {metrics['e2e']:.4f}s, "
                f"TTFT: {metrics.get('ttft', 'N/A')}, "
                f"TPOT: {metrics.get('tpot', 'N/A')}"
            )
            try:
                await gpu_router.ensure_one_empty_instance.remote()
            except Exception:
                logger.exception(
                    "Failed to replenish empty prewarm instance after cold start (tokenwise) model=%s",
                    model_name,
                )
            return gpu_result

    async def update(self, model_name: str, model_config: Mapping):
        async with self._model_lifecycle_lock:
            return await self._update(model_name, model_config)

    async def _update(self, model_name: str, model_config: Mapping):
        async with self.metadata_lock:
            if (
                model_name not in self.registered_models
                or model_name not in self.gpu_request_routers
            ):
                logger.error(f"Model {model_name} not found")
                raise ValueError(
                    f"Model {model_name} not found, please register first"
                )

            request_router = self.gpu_request_routers[model_name]
        auto_scaling_config = model_config.get("auto_scaling_config", None)
        validate_auto_scaling_config(auto_scaling_config)
        logger.info(f"Try to update the model {model_name} config")
        if auto_scaling_config is not None:
            await await_backend_call(request_router.update.remote(auto_scaling_config))
            async with self.metadata_lock:
                self.registered_models[model_name]["auto_scaling_config"] = dict(auto_scaling_config)
        # TODO: update other config (if possible)

    async def exists(self, model_name: str):
        async with self.metadata_lock:
            return model_name in self.registered_models

    async def delete(self, model_name: str):
        async with self._model_lifecycle_lock:
            return await self._delete(model_name)

    async def _delete(self, model_name: str):
        routers_to_stop = []
        cpu_router = None
        gpu_router = None
        stored_cfg = None
        tokenizer = None
        previous_refs = None
        async with self.metadata_lock:
            if model_name not in self.gpu_request_routers:
                logger.error(f"Model {model_name} not found")
                return

            # Determine who owns the CPU router and update the ref-count.
            stored_cfg = dict(self.registered_models.get(model_name, {}))
            cpu_owner = stored_cfg.get("_cpu_owner", model_name)
            previous_refs = set(self._cpu_router_refs.get(cpu_owner, set()))
            refs = set(previous_refs)
            refs.discard(model_name)

            if model_name in self.cpu_request_routers:
                if not refs:
                    # Last reference: actually shut down the CPU router.
                    cpu_router = self.cpu_request_routers.pop(model_name)
                    routers_to_stop.append(cpu_router)
                    self._cpu_router_refs.pop(cpu_owner, None)
                else:
                    # Other models still share this CPU router; only remove the
                    # mapping for this model without shutting down the actor.
                    self.cpu_request_routers.pop(model_name)
                    self._cpu_router_refs[cpu_owner] = refs
                    logger.info(
                        "CPU router for %s (owner=%s) kept alive; "
                        "remaining sharers: %s",
                        model_name,
                        cpu_owner,
                        refs,
                    )

            if model_name in self.gpu_request_routers:
                gpu_router = self.gpu_request_routers.pop(model_name)
                routers_to_stop.append(gpu_router)
            self.registered_models.pop(model_name, None)
            tokenizer = self.tokenizers.pop(model_name, None)
        # Stop the GPU router first.  If it fails, the CPU actor remains alive
        # and the metadata rollback below can restore a usable shared owner.
        routers_to_stop = (
            ([gpu_router] if gpu_router is not None else [])
            + ([cpu_router] if cpu_router is not None else [])
        )
        for router in routers_to_stop:
            try:
                await self._stop_router(router)
            except BaseException as result:
                async with self.metadata_lock:
                    self.registered_models[model_name] = stored_cfg
                    self.gpu_request_routers[model_name] = gpu_router
                    if tokenizer is not None:
                        self.tokenizers[model_name] = tokenizer
                    if cpu_router is not None:
                        self.cpu_request_routers[model_name] = cpu_router
                    if previous_refs:
                        self._cpu_router_refs[cpu_owner] = previous_refs
                    else:
                        self._cpu_router_refs.pop(cpu_owner, None)
                raise result
        logger.info(f"Model {model_name} deleted")

    async def get_models(self):
        async with self.metadata_lock:
            return self.registered_models

    async def status(self):
        """
        Returns the status of all registered models in OpenAI-compliant format.
        """
        async with self.metadata_lock:
            models = []
            for model_name, config in self.registered_models.items():
                created_time = config.get("created", None)
                allow_create_engine = config.get("allow_create_engine", None)
                allow_sampling = config.get("allow_sampling", None)
                allow_logprobs = config.get("allow_logprobs", None)
                allow_search_indices = config.get("allow_search_indices", None)
                allow_view = config.get("allow_view", None)
                organization = config.get("organization", "*")
                group = config.get("group", None)
                is_blocking = config.get("is_blocking", None)

                max_model_len = config.get("max_position_embeddings", None)

                model_permission_id = f"modelperm-{model_name}"
                permission = [
                    {
                        "id": model_permission_id,
                        "object": "model_permission",
                        "created": created_time
                        if created_time is not None
                        else None,
                        "allow_create_engine": allow_create_engine
                        if allow_create_engine is not None
                        else None,
                        "allow_sampling": allow_sampling
                        if allow_sampling is not None
                        else None,
                        "allow_logprobs": allow_logprobs
                        if allow_logprobs is not None
                        else None,
                        "allow_search_indices": allow_search_indices
                        if allow_search_indices is not None
                        else None,
                        "allow_view": allow_view
                        if allow_view is not None
                        else None,
                        "organization": organization
                        if organization is not None
                        else None,
                        "group": group if group is not None else None,
                        "is_blocking": is_blocking
                        if is_blocking is not None
                        else None,
                    }
                ]

                # Build the model metadata entry
                model_metadata = {
                    "id": model_name,
                    "object": "model",
                    "created": created_time
                    if created_time is not None
                    else None,
                    "owned_by": "sllm",
                    "root": model_name,
                    "parent": None,
                    "max_model_len": max_model_len
                    if max_model_len is not None
                    else None,
                    "permission": permission,
                }
                models.append(model_metadata)

            return {"object": "list", "models": models}

    async def shutdown(self):
        async with self._model_lifecycle_lock:
            return await self._shutdown_controller()

    async def _shutdown_controller(self):
        # stop the control loop
        async with self.running_lock:
            if not self.running:
                logger.error("Controller not running")
                raise RuntimeError("Controller not running")
            self.running = False
            self._shutting_down = True

        # Stop resource monitor and flush stats to disk.
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        try:
            saved = await self.save_stats()
            logger.info("Stats saved on shutdown: %s", saved)
        except Exception:
            logger.exception("Failed to save stats on shutdown")

        try:
            async with self.metadata_lock:
                model_names = list(self.gpu_request_routers.keys())
            results = await asyncio.gather(
                *(self._delete(m) for m in model_names), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.error("Model teardown failed during shutdown: %s", result)
        finally:
            sched = getattr(self, "scheduler", None)
            if sched is not None:
                try:
                    ray.kill(sched, no_restart=True)
                    logger.info("Actor model_loading_scheduler removed")
                except Exception as e:
                    logger.warning("Failed to kill model_loading_scheduler: %s", e)
                self.scheduler = None
