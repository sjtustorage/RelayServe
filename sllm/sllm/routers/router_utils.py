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
import math
from collections.abc import Mapping
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import ray

from sllm.sllm.logger import init_logger

logger = init_logger(__name__)


def validate_auto_scaling_config(config):
    """Validate before publishing a policy to the background autoscaler."""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("auto_scaling_config must be a mapping")
    policy = dict(config or {})
    for key, default, minimum in (
        ("min_instances", 0, 0), ("max_instances", 10, 0),
        ("target", 10, 1), ("max_queue_length", 256, 1),
    ):
        value = policy.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if policy.get("min_instances", 0) > policy.get("max_instances", 10):
        raise ValueError("min_instances must not exceed max_instances")
    keep_alive = policy.get("keep_alive", 0)
    if (isinstance(keep_alive, bool) or not isinstance(keep_alive, (int, float))
            or not math.isfinite(keep_alive) or keep_alive < 0):
        raise ValueError("keep_alive must be finite and nonnegative")
    return policy


def instance_request_capacity(policy):
    return policy.get("target", 256) if policy.get("metric") == "concurrency" else policy.get("max_queue_length", 256)


async def await_backend_call(call):
    """Propagate router request cancellation to its backend actor call."""
    try:
        return await call
    except asyncio.CancelledError:
        if isinstance(call, ray.ObjectRef):
            try:
                ray.cancel(call, recursive=True)
            except Exception:
                logger.exception("Failed to cancel backend actor call")
        raise


async def close_backend_actor(backend, instance_id, config, *, drain=False):
    """Bound graceful backend exit and always dispose of the Ray actor."""
    method = "stop" if drain else "shutdown"
    default = (
        float(config.get("request_drain_timeout_s", 600))
        + float(config.get("request_abort_timeout_s", 30)) + 30
        if drain else 30
    )
    timeout = float(config.get("backend_stop_timeout_s" if drain else "backend_shutdown_timeout_s", default))
    try:
        await asyncio.wait_for(await_backend_call(getattr(backend, method).remote()), timeout=timeout)
    except Exception:
        logger.exception("Backend %s failed for instance %s", method, instance_id)
    finally:
        try:
            ray.kill(backend)
        except Exception:
            logger.exception("Backend kill failed for instance %s", instance_id)


def merge_runtime_env(
    base: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Merge two Ray :class:`RuntimeEnv` dicts (JSON-serializable, per Ray API).

    * *extra* overrides *base* for same top-level key except ``env_vars``, which is
      merged with *extra* winning on duplicate keys. Either argument may be ``None``.
    """
    if not base and not extra:
        return None
    if not extra:
        return dict(base) if base else None
    if not base:
        return dict(extra) if extra else None
    out = dict(base)
    for k, v in extra.items():
        if k == "env_vars" and isinstance(v, dict):
            ev_base = out.get("env_vars")
            if isinstance(ev_base, dict):
                out["env_vars"] = {**ev_base, **v}
            else:
                out["env_vars"] = dict(v)
        else:
            out[k] = v
    return out


def _strip_cuda_visible_from_runtime_env_for_ray(
    re: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Return a copy of *re* without ``CUDA_VISIBLE_DEVICES`` and the removed value.

    Ray applies ``runtime_env`` before the actor runs; if ``CUDA_VISIBLE_DEVICES`` is set
    there, ``set_visible_accelerator_ids`` can raise ``IndexError`` when the restricted
    device list does not line up with Ray's assigned GPU resource indices. We strip it
    from what is passed to ``.options(runtime_env=...)`` and set visibility in
    :class:`~sllm.backends.gpu_backend.GPUBackend` via ``backend_config`` instead.
    """
    if re is None:
        return None, None
    out: Dict[str, Any] = dict(re)
    cvd: Optional[str] = None
    ev = out.get("env_vars")
    if isinstance(ev, dict) and "CUDA_VISIBLE_DEVICES" in ev:
        cvd = str(ev["CUDA_VISIBLE_DEVICES"])
        ev2 = {k: v for k, v in ev.items() if k != "CUDA_VISIBLE_DEVICES"}
        if ev2:
            out["env_vars"] = ev2
        else:
            out.pop("env_vars", None)
    if "CUDA_VISIBLE_DEVICES" in out:
        if cvd is None:
            cvd = str(out["CUDA_VISIBLE_DEVICES"])
        del out["CUDA_VISIBLE_DEVICES"]
    if not out:
        return None, cvd
    return out, cvd


def start_instance_spawner_options(
    resources: Optional[Dict[str, float]],
    startup_config: Dict[str, Any],
    *,
    for_gpu: bool = False,
) -> Dict[str, Any]:
    """Build ``kwargs`` for ``start_instance.options(**kwargs)``.

    The ``@ray.remote`` *spawner* task runs ``import sllm.backends...`` and must see
    vLLM in the same interpreter as the backend actor, so it needs the same
    ``runtime_env`` as in ``startup_config`` (e.g. ``py_executable`` / ``conda``), not
    only the child actor.
    """
    out: Dict[str, Any] = {}
    if resources:
        out["resources"] = resources
    re = startup_config.get("runtime_env")
    re_san, _ = _strip_cuda_visible_from_runtime_env_for_ray(
        re if isinstance(re, dict) else None
    )
    if re_san is not None:
        out["runtime_env"] = re_san
    return out


class SllmRouter(ABC):
    """Abstract router for a model (CPU cold-start or GPU pool)."""

    @abstractmethod
    async def start(self, auto_scaling_config: Optional[Dict[str, int]] = None):
        pass

    @abstractmethod
    async def shutdown(self):
        pass

    @abstractmethod
    async def inference(self, request_data: dict, action: str):
        pass

@ray.remote
def start_instance(
    instance_id, backend, model_name, backend_config, startup_config, device
):
    logger.info(f"Starting instance {instance_id} with backend {backend}")
    if backend == "vllm":
        if device == "cpu":
            from sllm.backends.cpu_backend import CPUBackend

            model_backend_cls = CPUBackend
        elif device == "gpu":
            from sllm.backends.gpu_backend import GPUBackend

            model_backend_cls = GPUBackend
        else:
            raise ValueError(f"Unknown device for vllm: {device}")
    else:
        logger.error(f"Unknown backend: {backend}")
        raise ValueError(f"Unknown backend: {backend}")

    model_actor_cls = ray.remote(model_backend_cls)

    raw_re = startup_config.get("runtime_env")
    raw_dict = raw_re if isinstance(raw_re, dict) else None
    runtime_env, cuda_visible = _strip_cuda_visible_from_runtime_env_for_ray(raw_dict)
    bc: Dict[str, Any] = (
        dict(backend_config) if isinstance(backend_config, dict) else {}
    )
    pp = int(
        (startup_config or {}).get("_sllm_pipeline_parallel_size")
        or bc.get("pipeline_parallel_size")
        or bc.get("gpu_pp_size")
        or 1
    ) or 1
    if (
        device == "gpu"
        and pp <= 1
        and cuda_visible is not None
        and str(cuda_visible).strip() != ""
        and not str(bc.get("_sllm_cuda_visible_devices") or "").strip()
    ):
        bc["_sllm_cuda_visible_devices"] = str(cuda_visible).strip()
    # Pin resources for the vLLM backend actor. When the caller also passes the same
    # ``resources`` on ``start_instance.options(...)``, the spawner task and the
    # backend actor both consume the same logical ``gpu_worker_*`` slot.
    # ``runtime_env`` must be on ``.options(..., runtime_env=...)`` so Ray applies conda/pip
    # / env in the worker process; it is not a valid keyword to ``.remote()`` for the actor.
    # Do not pass CUDA_VISIBLE_DEVICES in runtime_env (see _strip_...); use bc instead
    # for non-PP only. PP: do not set CVD here (``pp``>1).
    actor_options: Dict[str, Any] = {
        "name": instance_id,
        "max_concurrency": 10,
        "lifetime": "detached",
    }
    for k, v in startup_config.items():
        if k in ("runtime_env",) or k.startswith("_sllm_"):
            continue
        if v is not None:
            actor_options[k] = v
    if runtime_env is not None:
        actor_options["runtime_env"] = runtime_env
    return model_actor_cls.options(**actor_options).remote(
        instance_id=instance_id,
        model=model_name,
        device=device,
        backend_config=bc,
        runtime_env=runtime_env,
    )
