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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import ray

# Ray custom resource keys for worker placement (must match ``ray start --resources``).
_CPU_WORKER_HEAD = "cpu_worker_"
_GPU_WORKER_HEAD = "gpu_worker_"


def request_text_prompt(request: Mapping[str, Any]) -> str:
    """Resolve text input consistently before planning and backend submission."""
    prompt = request.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        return prompt
    if request.get("input_tokens") is not None:
        return ""
    messages = request.get("messages")
    if messages is None:
        return ""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    lines = []
    for message in messages:
        if (not isinstance(message, Mapping) or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)):
            raise ValueError("Each message must have string role and content fields")
        lines.append(f"{message['role']}: {message['content']}")
    return "\n".join(lines)


def cpu_placement_resource_prefix(backend_config: Optional[Mapping[str, Any]] = None) -> str:
    """Prefix for ``cpu_placement_node_id`` → Ray key (default ``cpu_worker_``)."""
    p = str((dict(backend_config or {}).get("cpu_placement_resource_prefix")) or _CPU_WORKER_HEAD)
    if not p.endswith("_"):
        p = f"{p}_"
    return p


def ray_cpu_worker_placement_resources(
    node_id: str,
    backend_config: Dict[str, Any],
    worker_resource_fraction: Optional[float] = None,
) -> Dict[str, float]:
    """Ray ``resources=`` using ``<prefix><node_id>`` (default ``cpu_worker_<id>``)."""
    frac = float(
        worker_resource_fraction
        if worker_resource_fraction is not None
        else backend_config.get("ray_worker_resource_fraction", 0.001)
    )
    prefix = cpu_placement_resource_prefix(backend_config)
    return {f"{prefix}{node_id}": frac}


def _logical_worker_id_from_ray_resource(resource_key: str) -> Optional[str]:
    """Map ``gpu_worker_<id>`` to the scheduler *node_id* suffix."""
    if resource_key.startswith(_GPU_WORKER_HEAD):
        return resource_key[len(_GPU_WORKER_HEAD) :]
    return None


def get_worker_nodes():
    ray_nodes = ray.nodes()
    worker_node_info = {}
    for node in ray_nodes:
        ray_node_id = node.get("NodeID", None)
        if ray_node_id is None:
            continue
        resources = node.get("Resources", {})
        if not resources:
            continue
        node_address = node.get("NodeManagerAddress", None)
        if node_address is None or node_address == "":
            continue
        total_gpu = float(resources.get("GPU", 0) or 0)
        for key in resources.keys():
            logical_id = _logical_worker_id_from_ray_resource(key)
            if logical_id is None:
                continue
            if total_gpu <= 0:
                continue
            worker_node_info[logical_id] = {
                "ray_node_id": ray_node_id,
                "address": node_address,
                "remaining_instance_slots": int(resources.get("GPU", 0)),
                "total_gpu": int(resources.get("GPU", 0)),
            }

    return worker_node_info


def ray_node_id_for_logical_slot(
    logical_id: str,
    backend_config: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Return the Ray NodeID for the machine that advertises this logical slot.

    Tries :func:`get_worker_nodes` first (GPU logical slots), then scans ``ray.nodes()``
    for ``cpu_worker_*`` or ``gpu_worker_*`` (including ``pp_pg_gpu_resource_prefix`` when
    it differs from the default). ``cpu_worker_x`` and ``gpu_worker_x`` on the same box
    share one NodeID.
    """
    if not ray.is_initialized():
        return None
    lid = str(logical_id).strip()
    if not lid:
        return None
    w = get_worker_nodes()
    info = w.get(lid)
    if info and info.get("ray_node_id"):
        return str(info["ray_node_id"])

    bc = dict(backend_config or {})
    cpu_key = f"{cpu_placement_resource_prefix(bc)}{lid}"
    gpu_pg_prefix = str(bc.get("pp_pg_gpu_resource_prefix", _GPU_WORKER_HEAD))
    if not gpu_pg_prefix.endswith("_"):
        gpu_pg_prefix = f"{gpu_pg_prefix}_"
    gpu_pg_slot_key = f"{gpu_pg_prefix}{lid}"
    gpu_worker_key = f"{_GPU_WORKER_HEAD}{lid}"
    slot_keys = {cpu_key, gpu_pg_slot_key, gpu_worker_key}

    for node in ray.nodes():
        if not node.get("Alive", True):
            continue
        resources = node.get("Resources") or {}
        for key, raw in resources.items():
            if not isinstance(key, str) or key not in slot_keys:
                continue
            try:
                amount = float(raw)
            except (TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            nid = node.get("NodeID")
            if nid is not None:
                return str(nid)
    return None


def ray_gpu_actor_placement_resources(
    node_id: Optional[str],
    backend_config: Dict[str, Any],
    worker_resource_fraction: Optional[float] = None,
) -> Dict[str, float]:
    """Ray ``resources=`` for placing a GPU model actor on a logical worker *node_id*."""
    frac = float(
        worker_resource_fraction
        if worker_resource_fraction is not None
        else backend_config.get("ray_worker_resource_fraction", 0.001)
    )
    out: Dict[str, float] = {}
    if node_id is None:
        return out
    out[f"{_GPU_WORKER_HEAD}{node_id}"] = frac
    return out


def ray_gpu_pg_logical_pin_resources(
    node_id: Optional[str],
    backend_config: Dict[str, Any],
    worker_resource_fraction: Optional[float] = None,
) -> Dict[str, float]:
    """Ray ``resources=`` for PP placement groups using ``gpu_worker_<id>`` (or custom prefix)."""
    frac = float(
        worker_resource_fraction
        if worker_resource_fraction is not None
        else backend_config.get("ray_worker_resource_fraction", 0.001)
    )
    out: Dict[str, float] = {}
    if node_id is None:
        return out
    prefix = str(backend_config.get("pp_pg_gpu_resource_prefix", _GPU_WORKER_HEAD))
    if not prefix.endswith("_"):
        prefix = f"{prefix}_"
    out[f"{prefix}{node_id}"] = frac
    return out


def ray_node_resource_key_for_node_id(ray_node_id: Optional[str]) -> Optional[str]:
    """Return the Ray ``node:*`` resource key for this cluster node (stable per NodeID).

    Placement group bundles cannot use raw ``NodeID`` hex strings as keys; each Ray node
    registers a ``node:<address>`` (or similar) entry in ``Resources``. We resolve it by
    matching ``NodeID`` from :func:`ray.nodes` so pinning survives duplicate
    ``NodeManagerAddress`` / IP ambiguity across logical slots.

    Ray may expose multiple ``node:*`` keys on one machine (e.g. ``node:__internal_head__``
    and ``node:1.2.3.4``). Returning the *first* key only is iteration-order dependent and
    can diverge between building a placement group and scheduling an actor onto bundle 0.
    We therefore collect candidates and prefer a non-placeholder key (real address) over
    internal Ray placeholders.
    """
    if not ray_node_id or not ray.is_initialized():
        return None
    target = str(ray_node_id).strip()
    if not target:
        return None
    candidates: List[str] = []
    for node in ray.nodes():
        if not node.get("Alive", True):
            continue
        if str(node.get("NodeID", "")) != target:
            continue
        resources = node.get("Resources") or {}
        for key, raw in resources.items():
            if not isinstance(key, str) or not key.startswith("node:"):
                continue
            try:
                amount = float(raw)
            except (TypeError, ValueError):
                continue
            if amount > 0:
                candidates.append(key)
    if not candidates:
        return None

    def _is_placeholder(k: str) -> bool:
        kl = k.lower()
        return "__internal" in kl or "internal_head" in kl

    prefer = [k for k in candidates if not _is_placeholder(k)]
    pick = prefer if prefer else candidates
    return sorted(pick)[0]


def pp_placement_pin_resources(
    stage: Mapping[str, Any],
    backend_config: Dict[str, Any],
    *,
    worker_resource_fraction: Optional[float] = None,
    pin_mode: Optional[str] = None,
) -> Dict[str, float]:
    """Extra Ray ``resources`` for one PP placement bundle (and matching actor options).

    *pin_mode* (or ``backend_config["pp_pg_pin_mode"]``):

    - ``auto`` (default): resolve ``node:*`` from ``stage["ray_node_id"]``; if missing or
      not found, fall back to ``gpu_worker_<scheduler node_id>`` (prefix: ``pp_pg_gpu_resource_prefix``).
    - ``logical``: only ``gpu_worker_*`` logical pin (requires matching ``ray start --resources``).
    - ``ray_node``: only ``node:*`` from ``ray_node_id`` (no logical fallback).
    """
    frac = float(
        worker_resource_fraction
        if worker_resource_fraction is not None
        else backend_config.get("ray_worker_resource_fraction", 0.001)
    )
    mode = (pin_mode or backend_config.get("pp_pg_pin_mode") or "auto").strip().lower()
    if mode not in ("auto", "logical", "ray_node"):
        mode = "auto"

    rid = stage.get("ray_node_id")
    if rid is not None:
        rid = str(rid).strip() or None

    if mode in ("auto", "ray_node"):
        nk = ray_node_resource_key_for_node_id(rid)
        if nk is not None:
            return {nk: frac}
        if mode == "ray_node":
            return {}

    logical_id = stage.get("node_id")
    if logical_id is None:
        return {}
    s = str(logical_id).strip()
    if not s:
        return {}
    return ray_gpu_pg_logical_pin_resources(s, backend_config, frac)


def discover_cpu_worker_placement_keys(
    backend_config: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Return sorted Ray resource keys ``<cpu_placement_prefix><id>`` on non-control nodes.

    Prefix is :func:`cpu_placement_resource_prefix` (default ``cpu_worker_``). Used when no
    explicit CPU placement is set: the controller can round-robin models across these slots.
    """
    if not ray.is_initialized():
        return []
    head = cpu_placement_resource_prefix(backend_config)
    found: set = set()
    for node in ray.nodes():
        resources = node.get("Resources") or {}
        for key, raw in resources.items():
            if not isinstance(key, str) or not key.startswith(head):
                continue
            suffix = key[len(head) :]
            if not suffix:
                continue
            try:
                amount = float(raw)
            except (TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            found.add(key)
    return sorted(found)


def discover_cpu_only_worker_slots(
    backend_config: Optional[Mapping[str, Any]] = None,
) -> List[Tuple[str, str]]:
    """Return ``(ray_node_id, logical_id)`` for nodes with **no** GPU and ``cpu_worker_*`` > 0."""
    if not ray.is_initialized():
        return []
    head = cpu_placement_resource_prefix(backend_config)
    pairs: List[Tuple[str, str]] = []
    for node in ray.nodes():
        if not node.get("Alive", True):
            continue
        resources = node.get("Resources") or {}
        if float(resources.get("GPU", 0) or 0) > 0:
            continue
        ray_node_id = node.get("NodeID")
        if ray_node_id is None:
            continue
        for key, raw in resources.items():
            if not isinstance(key, str) or not key.startswith(head):
                continue
            suffix = key[len(head) :]
            if not suffix:
                continue
            try:
                amount = float(raw)
            except (TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            pairs.append((str(ray_node_id), suffix))
    pairs.sort(key=lambda x: (x[0], x[1]))
    return pairs


def discover_cpu_only_worker_logical_ids(
    backend_config: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Return logical ids for ``<cpu_placement_prefix><id>`` on nodes with **no** GPU.

    :func:`get_worker_nodes` ignores these nodes; PP SHM fan-out still needs to load weights
    onto CPU-only machines when those slots use the CPU-worker resource prefix.
    """
    return sorted({lid for _, lid in discover_cpu_only_worker_slots(backend_config)})


def ray_cpu_actor_placement_resources(
    backend_config: Dict[str, Any],
) -> Tuple[Dict[str, float], Optional[str]]:
    """Ray ``resources=`` for the CPU vLLM backend actor (SHM producer) on a logical slot.

    Uses the same ``gpu_worker_<id>`` convention as GPU backends so one physical
    ``ray start --resources`` line (e.g. ``gpu_worker_0``) suffices; no separate
    ``cpu_worker_*`` key is required.

    If ``cpu_placement_node_id`` is set, resources are
    ``{"gpu_worker_<id>": ray_worker_resource_fraction}`` (default *frac* ``0.001``).

    If it is absent, a non-empty ``cpu_placement_resources`` dict is used as a manual
    override (typically ``gpu_worker_*`` keys); otherwise returns ``({}, None)``.
    """
    frac = float(backend_config.get("ray_worker_resource_fraction", 0.001))
    nid = backend_config.get("cpu_placement_node_id")
    if nid is not None and str(nid).strip() != "":
        logical = str(nid).strip()
        return ray_gpu_actor_placement_resources(logical, backend_config, frac), logical

    custom = backend_config.get("cpu_placement_resources")
    if isinstance(custom, dict) and custom:
        out = {str(k): float(v) for k, v in custom.items()}
        return out, None

    return {}, None


@dataclass
class InstanceStatus:
    instance_id: str
    node_id: Optional[str]
    concurrency: int

    model_name: Optional[str] = None
    num_current_tokens: Optional[int] = None
    resuming_latency: Optional[float] = None


@dataclass
class InstanceHandle:
    instance_id: str
    max_queue_length: int

    node_id: Optional[str] = None
    backend_instance: Optional[ray.actor.ActorHandle] = None
    ready: bool = False
    concurrency: int = 0
    num_gpu: int = 0
    empty_instance: bool = False
    load_method: Optional[str] = None
    # NUMA domains touched by this instance's GPUs (from scheduler allocation PP0 stage).
    gpu_numa_nodes: Optional[Tuple[int, ...]] = None
    preferred_gpu_ids: Optional[List[int]] = None
    preferred_pp0_node_id: Optional[str] = None
    allocation_info: Optional[Dict[str, Any]] = None
    status: bool = True
    # Ray placement group used to pin PP stage workers to designated nodes.
    # Created by the router for pp_size > 1; must be removed on teardown.
    placement_group: Optional[Any] = None
    # HydraServe: "primary" (TP, pp=1) and "pair" (PP extension); pair shares primary GPUs.
    hydraserve_role: Optional[str] = None
    # Pair instance reuses the same node/gpu info as the primary; set at adopt time.
    hydraserve_primary_id: Optional[str] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add_requests(self, num_requests: int = 1):
        async with self.lock:
            if num_requests > 0 and (not self.ready or not self.status):
                return False
            if (
                (num_requests > 0 and self.concurrency + num_requests > self.max_queue_length)
                or self.concurrency + num_requests < 0
            ):
                return False
            self.concurrency += num_requests
            return True

    async def check_request_queue(self):
        async with self.lock:
            return self.ready and self.status and self.concurrency + 1 <= self.max_queue_length

    async def get_status(self):
        async with self.lock:
            return InstanceStatus(
                self.instance_id,
                self.node_id,
                self.concurrency,
            )


class TokenizerWrapper:
    def __init__(
        self,
        tokenizer_name: str,
        trust_remote_code: bool = False,
    ) -> None:
        # Lazy import: ``transformers`` is optional for many workers (e.g. Ray tasks
        # that only import :mod:`sllm.utils` for placement helpers).
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
        )

    def get_prompt_len(self, text: str) -> int:
        if not isinstance(text, str) or text == "":
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))
