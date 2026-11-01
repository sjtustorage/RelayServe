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
import copy
import json
import os
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import ray
from ray.runtime_env import RuntimeEnv
from sllm.logger import init_logger
from sllm.utils import get_worker_nodes, ray_gpu_actor_placement_resources

from .scheduler_utils import (
    SllmScheduler,
    load_gpu_numa_affinity,
    probe_gputil_memory_used_mib,
    resolve_gpu_numa_affinity,
)

logger = init_logger(__name__)

# Keep in sync with ``sllm.utils`` ``gpu_worker_`` custom resource keys.
_GPU_WORKER_RES_PREFIX = "gpu_worker_"


@ray.remote(runtime_env=RuntimeEnv(py_executable="/workspace/sllm/.venv/bin/python"))
def _remote_probe_gpu_vram_mib() -> Optional[List[int]]:
    return probe_gputil_memory_used_mib()


class FcfsScheduler(SllmScheduler):
    """First-Come-First-Served scheduler for GPU resource allocation.

    Design principles (simplified):
    - Every instance (including prewarm/empty) immediately occupies its GPUs
      at ``allocate_resource`` time.  There is no "soft" prewarm quota that
      doesn't count against capacity.
    - One GPU slot per instance: no ``max_instances_per_gpu`` oversubscription.
    - TP GPUs are spread evenly across NUMA domains within the node.
    - PP stages are placed on distinct Ray nodes (enforced by default).
    - Same TP/PP configuration models are preferentially co-located on the same
      nodes for better resource organisation.
    - When ``preferred_node_id`` is set (CPU–GPU colocation hint from the router),
      allocation is **strict**: GPUs are only reserved on that logical node. If
      capacity is insufficient, the request stays queued and the control loop
      retries until space appears (no fallback to other nodes).
    - **VRAM** is probed on each GPU after **backend init completes** and after
      **deallocate** (so CUDA has typically allocated, readings are useful).  Used memory
      at or above ``gpu_occupied_vram_threshold_mib`` (default 2 GiB) marks a GPU as
      unschedulable, together with sllm instance slot accounting (no separate GPU
      init locks: allocation is **serial** in the control loop).  Disable VRAM with
      ``SLLM_DISABLE_GPU_VRAM_PROBE`` or ``enable_gpu_vram_occupancy: false``.
    """

    def __init__(self, scheduler_config: Optional[Mapping] = None):
        super().__init__()
        self.scheduler_config = scheduler_config

        # Last GPUtil snapshot per *logical* node_id: ``[used_mib per GPU index]``.
        self._node_gpu_vram_mib: Dict[str, List[int]] = {}
        _cfg = dict(self.scheduler_config) if self.scheduler_config else {}
        # if os.environ.get("SLLM_DISABLE_GPU_VRAM_PROBE", "").strip().lower() in (
        #     "1",
        #     "true",
        #     "yes",
        # ):
        self._gpu_vram_occupancy_enabled = False
        self._gpu_occupied_vram_threshold_mib = int(
            _cfg.get("gpu_occupied_vram_threshold_mib", 8 * 1024)
        )

        self.queue_lock = asyncio.Lock()
        self.model_loading_queues: Dict[str, List] = {}

        self.metadata_lock = asyncio.Lock()
        self.worker_nodes: Dict[str, Dict] = {}
        # Only log at INFO when *scheduling*/*VRAM* state changes (control loop is noisy).
        self._log_fp_control_loop_worker_nodes: Optional[str] = None
        self._log_fp_persisted_worker_nodes: Optional[str] = None
        self._log_fp_stall_no_node: Optional[Tuple[Any, ...]] = None
        self.model_instance: Dict[str, Dict[str, str]] = {}

        # Round-robin counters for GPU cycling per node (used when NUMA info is absent).
        self.node_gpu_counters: Dict[str, int] = {}

        # Per-node cold-start serialisation (shared CPU memory bandwidth).
        self.cold_start_state_lock = asyncio.Lock()
        self.cold_start_status: Dict[Tuple[str, str], Dict] = {}
        self.node_cold_start_ready_events: Dict[str, asyncio.Event] = {}

        self.loop = asyncio.get_running_loop()
        self.loop_task: Optional[asyncio.Task] = None

        self.running_lock = asyncio.Lock()
        self.running = False

        self.load_window_seconds = int(
            self.scheduler_config.get("load_window_seconds", 120)
        ) if self.scheduler_config else 120
        self.forecast_horizon_seconds = int(
            self.scheduler_config.get("forecast_horizon_seconds", 20)
        ) if self.scheduler_config else 20
        # Control loop polling interval; reduce in tests for faster feedback.
        self.control_loop_interval_s = float(
            self.scheduler_config.get("control_loop_interval_s", 1.0)
        ) if self.scheduler_config else 1.0
        self.model_load_history: Dict[str, Deque[Tuple[float, float]]] = defaultdict(deque)

        # instance_id → allocation metadata
        self.instance_allocations: Dict[str, Dict] = {}

        self.gpu_numa_affinity = load_gpu_numa_affinity(
            config=dict(self.scheduler_config) if self.scheduler_config else None
        )
        self.auto_pipeline_split = bool(
            (self.scheduler_config or {}).get("auto_pipeline_split", True)
        )
        logger.info("FCFS scheduler auto_pipeline_split=%s", self.auto_pipeline_split)
        logger.info(
            "FCFS GPU VRAM occupancy: enabled=%s threshold_mib=%s",
            self._gpu_vram_occupancy_enabled,
            self._gpu_occupied_vram_threshold_mib,
        )

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        async with self.running_lock:
            if self.running:
                logger.error("FCFS scheduler already started")
                return
            self.running = True
            logger.info("Starting FCFS scheduler")
            self.loop_task = self.loop.create_task(self._control_loop())

    async def _release_waiters(self, failure: Optional[Exception] = None) -> None:
        async with self.queue_lock:
            pending = [
                allocation_result
                for loading_queue in self.model_loading_queues.values()
                for _request_time, _request_info, allocation_result in loading_queue
            ]
            self.model_loading_queues = {}
            for allocation_result in pending:
                if not allocation_result.done():
                    if failure is None:
                        allocation_result.set_result(None)
                    else:
                        error = RuntimeError("Scheduler control loop failed")
                        error.__cause__ = failure
                        allocation_result.set_exception(error)
        async with self.cold_start_state_lock:
            self.cold_start_status.clear()
            for event in self.node_cold_start_ready_events.values():
                event.set()

    async def shutdown(self) -> None:
        async with self.running_lock:
            if not self.running:
                logger.error("FCFS scheduler not running")
            self.running = False
            if self.loop_task is not None:
                self.loop_task.cancel()
                await asyncio.gather(self.loop_task, return_exceptions=True)
            await self._release_waiters()

    # ------------------------------------------------------------------ #
    #  Load tracking / forecasting                                        #
    # ------------------------------------------------------------------ #

    async def report_model_load(self, model_name: str, current_load: float) -> None:
        now = time.time()
        history = self.model_load_history[model_name]
        history.append((now, float(current_load)))
        cutoff = now - self.load_window_seconds
        while history and history[0][0] < cutoff:
            history.popleft()

    def _predict_model_load(self, model_name: str) -> Dict[str, float]:
        history = self.model_load_history.get(model_name, deque())
        if not history:
            return {"current_load": 0.0, "predicted_load": 0.0, "trend": 0.0}
        now = time.time()
        recent = [(t, v) for t, v in history if t >= now - self.load_window_seconds]
        if not recent:
            return {"current_load": 0.0, "predicted_load": 0.0, "trend": 0.0}
        current_load = recent[-1][1]
        if len(recent) == 1:
            return {"current_load": current_load, "predicted_load": current_load, "trend": 0.0}
        start_t, start_v = recent[0]
        end_t, end_v = recent[-1]
        dt = max(end_t - start_t, 1e-6)
        trend = (end_v - start_v) / dt
        predicted = max(0.0, end_v + trend * self.forecast_horizon_seconds)
        return {"current_load": current_load, "predicted_load": predicted, "trend": trend}

    async def get_model_load_stats(self, model_name: str) -> Dict[str, float]:
        return self._predict_model_load(model_name)

    # ------------------------------------------------------------------ #
    #  Cluster capacity                                                   #
    # ------------------------------------------------------------------ #

    async def get_cluster_gpu_capacity(self) -> Dict[str, int]:
        worker_nodes = await self._get_worker_nodes()
        total_gpus = sum(int(n.get("total_gpu", 0)) for n in worker_nodes.values())
        free_gpus = 0
        for node_id, n in worker_nodes.items():
            t = int(n.get("total_gpu", 0) or 0)
            if t > 0:
                free_gpus += self._compute_node_free_gpus(str(node_id), t)
        return {
            "total_gpus": total_gpus,
            "total_nodes": len(worker_nodes),
            "capacity_free_gpus": free_gpus,
        }

    async def instance_eligible_for_request(self, instance_id: str) -> bool:
        async with self.metadata_lock:
            return instance_id in self.instance_allocations

    # ------------------------------------------------------------------ #
    #  Request normalisation                                              #
    # ------------------------------------------------------------------ #

    def _normalize_parallel_request(self, resources: Mapping) -> Dict:
        requested_total_gpus = float(resources.get("num_gpus", 0) or 0)
        pipeline_parallel_size = int(
            resources.get("pipeline_parallel_size", resources.get("pp_size", 1)) or 1
        )
        if pipeline_parallel_size < 1:
            pipeline_parallel_size = 1

        has_tp_key = "tensor_parallel_size" in resources or "tp_size" in resources
        if has_tp_key:
            raw_tp = resources.get("tensor_parallel_size", resources.get("tp_size"))
            tensor_parallel_size = int(raw_tp) if raw_tp is not None else None
        else:
            tensor_parallel_size = None

        if tensor_parallel_size is None or tensor_parallel_size <= 0:
            tensor_parallel_size = (
                max(1, int(round(requested_total_gpus / pipeline_parallel_size)))
                if requested_total_gpus >= 1
                else 1
            )

        logical_total_gpus = max(
            requested_total_gpus,
            float(tensor_parallel_size * pipeline_parallel_size),
        )

        pref = resources.get("preferred_node_id")
        if pref is None or (isinstance(pref, str) and not str(pref).strip()):
            pref = resources.get("preferred_pp0_node_id")

        preferred_gpu_ids = resources.get("preferred_gpu_ids")
        if preferred_gpu_ids is not None:
            preferred_gpu_ids = [int(g) for g in preferred_gpu_ids]
            if (pipeline_parallel_size != 1 or len(preferred_gpu_ids) != tensor_parallel_size
                    or len(set(preferred_gpu_ids)) != len(preferred_gpu_ids)
                    or any(g < 0 for g in preferred_gpu_ids)):
                raise ValueError("preferred_gpu_ids must contain distinct nonnegative IDs matching a single TP stage")

        return {
            "num_gpus": logical_total_gpus,
            "tp_size": tensor_parallel_size,
            "pp_size": pipeline_parallel_size,
            "empty_instance": bool(resources.get("empty_instance", False)),
            "preferred_node_id": pref,
            "preferred_gpu_ids": preferred_gpu_ids,
        }

    # ------------------------------------------------------------------ #
    #  GPU accounting                                                     #
    # ------------------------------------------------------------------ #

    async def get_per_node_schedulable_capacity(self) -> List[Dict[str, Any]]:
        """Return schedulable free GPU counts per logical worker node.

        Each row matches :meth:`_compute_node_free_gpus` (Ray slot accounting plus
        optional VRAM-over-threshold occupancy).  Used by GPU routers for VRAM-poll
        autoscale.
        """
        worker_nodes = await self._get_worker_nodes()
        out: List[Dict[str, Any]] = []
        for node_id, n in sorted(worker_nodes.items()):
            tg = int(n.get("total_gpu", 0) or 0)
            if tg <= 0:
                continue
            nid = str(node_id)
            free = self._compute_node_free_gpus(nid, tg)
            out.append(
                {
                    "node_id": nid,
                    "total_gpu": tg,
                    "free_gpus": free,
                }
            )
        return out

    def _scheduling_state_fingerprint(
        self, worker_nodes: Mapping[str, Any]
    ) -> str:
        """Compact digest of capacity (and optional VRAM cache) for log de-duplication."""
        cap: List[Tuple[str, int, int]] = []
        for nid in sorted(worker_nodes.keys()):
            ni = worker_nodes.get(nid) or {}
            tg = int(ni.get("total_gpu", 0) or 0)
            free = (
                self._compute_node_free_gpus(str(nid), tg) if tg > 0 else 0
            )
            cap.append((str(nid), tg, free))
        vram: Dict[str, List[int]] = self._node_gpu_vram_mib
        vram_brief: Dict[str, Any] = {}
        for nid in sorted(vram.keys()):
            row = vram[nid]
            vram_brief[nid] = (len(row), row[0] if row else 0, row[-1] if row else 0)
        return json.dumps({"cap": cap, "vram": vram_brief}, sort_keys=True)

    def _compact_worker_nodes_log(self, worker_nodes: Mapping[str, Any]) -> str:
        """Human-readable, small summary (no per-GPU list)."""
        lines: List[str] = []
        for nid in sorted(worker_nodes.keys()):
            ni = worker_nodes.get(nid) or {}
            tg = int(ni.get("total_gpu", 0) or 0)
            free = self._compute_node_free_gpus(str(nid), tg) if tg > 0 else 0
            lines.append(f"{nid}: total={tg} free_gpus={free}")
        return "; ".join(lines) if lines else "(no worker nodes)"

    def _vram_mib_value(self, node_id: str, gpu_id: int) -> Optional[int]:
        row = self._node_gpu_vram_mib.get(node_id)
        if not row or gpu_id < 0 or gpu_id >= len(row):
            return None
        return int(row[gpu_id])

    def _vram_over_threshold_occupies(self, node_id: str, gpu_id: int) -> bool:
        """True if the probed used MiB is at/above ``gpu_occupied_vram_threshold_mib``."""
        if not self._gpu_vram_occupancy_enabled:
            return False
        m = self._vram_mib_value(node_id, gpu_id)
        if m is None:
            return False
        return m >= self._gpu_occupied_vram_threshold_mib

    def _unmapped_pp_slots_on_node(self, node_id: str) -> int:
        """PP stages that only reserve a TP *count* (no per-GPU ``gpu_ids``) on *node_id*."""
        n = 0
        for info in self.instance_allocations.values():
            for allocation in info.get("allocations", []):
                if allocation.get("node_id") != node_id:
                    continue
                gids = allocation.get("gpu_ids") or []
                if gids:
                    continue
                n += int(
                    allocation.get("tp_slots")
                    or info.get("tp_size")
                    or info.get("tensor_parallel_size", 0)
                    or 0
                )
        return n

    def _count_gpu_instances_on_node(self, node_id: str) -> Dict[int, int]:
        """Per-GPU occupied-instance count on *node_id*.

        Every allocated instance (including prewarm/empty) is counted immediately;
        there is no separate capacity-committed distinction.
        """
        out: Dict[int, int] = defaultdict(int)
        for info in self.instance_allocations.values():
            for allocation in info.get("allocations", []):
                if allocation.get("node_id") != node_id:
                    continue
                for gid in allocation.get("gpu_ids", []):
                    out[int(gid)] += 1
        return out

    def _effective_gpu_counts(
        self,
        node_id: str,
        total_gpus: int,
        reserved_gpu_ids: Optional[Sequence[int]],
    ) -> Dict[int, int]:
        """Instance counts per GPU, plus *reserved_gpu_ids* from the current allocation loop."""
        counts = self._count_gpu_instances_on_node(node_id)
        effective: Dict[int, int] = {g: counts.get(g, 0) for g in range(total_gpus)}
        for gid in reserved_gpu_ids or ():
            ig = int(gid)
            if 0 <= ig < total_gpus:
                effective[ig] = effective.get(ig, 0) + 1
        return effective

    def _node_schedulable_free_gpus(
        self,
        node_id: str,
        total_gpus: int,
        reserved_gpu_ids: Optional[Sequence[int]],
        extra_batch_slot_reservations: int = 0,
    ) -> int:
        """Count of schedulable GPU slots: physical free, minus unmapped PP slots, minus batch slot holds."""
        if total_gpus <= 0:
            return 0
        eff = self._effective_gpu_counts(node_id, total_gpus, reserved_gpu_ids)
        unmapped = self._unmapped_pp_slots_on_node(node_id)
        n_phys = sum(
            1
            for g in range(total_gpus)
            if not eff.get(g, 0)
            and not self._vram_over_threshold_occupies(node_id, g)
        )
        n_phys -= unmapped + int(extra_batch_slot_reservations)
        return max(0, n_phys)

    def _compute_node_free_gpus(self, node_id: str, total_gpus: int) -> int:
        """Unoccupied GPU slots on *node_id* (PP abstract slots + VRAM in :meth:`_node_schedulable_free_gpus`)."""
        return self._node_schedulable_free_gpus(node_id, total_gpus, None, 0)

    def _build_per_gpu_snapshot(self, node_id: str, total_gpus: int) -> List[Dict[str, Any]]:
        eff = self._effective_gpu_counts(node_id, total_gpus, None)
        out: List[Dict[str, Any]] = []
        for g in range(total_gpus):
            row: Dict[str, Any] = {
                "gpu_id": g,
                "occupied": bool(eff.get(g, 0)),
            }
            m = self._vram_mib_value(node_id, g)
            if m is not None:
                row["vram_mib"] = m
                row["vram_over_threshold"] = self._vram_over_threshold_occupies(
                    node_id, g
                )
            row["available"] = (
                not eff.get(g, 0)
                and not self._vram_over_threshold_occupies(node_id, g)
            )
            out.append(row)
        return out

    def _can_place_tp_on_node(
        self,
        node_id: str,
        total_gpus: int,
        tp_size: int,
        reserved_gpu_ids: Optional[Sequence[int]],
        extra_batch_slot_reservations: int = 0,
    ) -> bool:
        if tp_size <= 0:
            return True
        return (
            self._node_schedulable_free_gpus(
                node_id, total_gpus, reserved_gpu_ids, extra_batch_slot_reservations
            )
            >= tp_size
        )

    def _free_gpu_ids_on_node(
        self,
        node_id: str,
        total_gpus: int,
        reserved_gpu_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """GPU indices on *node_id* that are unoccupied and under VRAM threshold."""
        eff = self._effective_gpu_counts(node_id, total_gpus, reserved_gpu_ids)
        return [
            g
            for g in range(total_gpus)
            if not eff.get(g, 0)
            and not self._vram_over_threshold_occupies(node_id, g)
        ]


    def _build_stage_allocation(
        self,
        node_id: str,
        gpu_ids: List[int],
        stage_idx: int,
        tp_size: Optional[int] = None,
    ) -> Dict:
        numa_map = self.gpu_numa_affinity.get(node_id, {})
        gids = [int(gpu_id) for gpu_id in gpu_ids]
        out: Dict[str, Any] = {
            "stage_idx": stage_idx,
            "node_id": node_id,
            "gpu_ids": gids,
            "numa_nodes": sorted(
                {int(numa_map[gpu_id]) for gpu_id in gids if gpu_id in numa_map}
            ),
        }
        if not gids and tp_size is not None and int(tp_size) > 0:
            out["tp_slots"] = int(tp_size)
        return out

    def _pick_balanced_numa_gpu_ids(
        self,
        node_id: str,
        candidate_gpu_ids: List[int],
        gpu_group_size: int,
        numa_loads: Optional[Mapping[int, int]] = None,
    ) -> Optional[List[int]]:
        """Pick *gpu_group_size* GPUs spreading evenly across NUMA domains.

        Greedy selection: pick the least occupied NUMA domain, including GPUs
        already selected for this allocation (ties broken by free capacity then ID).
        Falls back to round-robin when no NUMA affinity data is available.
        """
        if len(candidate_gpu_ids) < gpu_group_size:
            return None

        numa_map = self.gpu_numa_affinity.get(node_id, {})
        if not numa_map:
            # Round-robin to cycle through GPUs across successive allocations.
            if node_id not in self.node_gpu_counters:
                self.node_gpu_counters[node_id] = 0
            ordered = sorted(candidate_gpu_ids)
            n = len(ordered)
            start = self.node_gpu_counters[node_id] % n
            assigned = [ordered[(start + i) % n] for i in range(gpu_group_size)]
            self.node_gpu_counters[node_id] = (start + gpu_group_size) % n
            return assigned

        numa_to_gpu: Dict[int, List[int]] = defaultdict(list)
        for gpu_id in candidate_gpu_ids:
            numa_to_gpu[numa_map.get(gpu_id, -1)].append(gpu_id)
        for gpu_list in numa_to_gpu.values():
            gpu_list.sort()

        selected: List[int] = []
        selected_numa_counts: Dict[int, int] = defaultdict(int)
        while len(selected) < gpu_group_size:
            available_numas = [numa for numa, gpus in numa_to_gpu.items() if gpus]
            if not available_numas:
                break
            available_numas.sort(
                key=lambda numa: (
                    (numa_loads or {}).get(numa, 0) + selected_numa_counts[numa],
                    -len(numa_to_gpu[numa]), numa,
                )
            )
            chosen_numa = available_numas[0]
            selected.append(numa_to_gpu[chosen_numa].pop(0))
            selected_numa_counts[chosen_numa] += 1

        return selected if len(selected) == gpu_group_size else None

    def _select_gpu_ids(
        self,
        node_id: str,
        node_info: Mapping,
        gpu_group_size: int,
        preferred_gpu_ids: Optional[Sequence[int]] = None,
        reserved_gpu_ids: Optional[Sequence[int]] = None,
    ) -> Optional[List[int]]:
        """Select *gpu_group_size* free GPUs on *node_id* with NUMA load balancing."""
        total_gpus = int(node_info.get("total_gpu", 0))
        if gpu_group_size <= 0:
            return []
        free_gpu_ids = self._free_gpu_ids_on_node(node_id, total_gpus, reserved_gpu_ids)
        if preferred_gpu_ids is not None:
            preferred = [int(g) for g in preferred_gpu_ids]
            if len(set(preferred)) != len(preferred):
                return None
            free_gpu_ids = [g for g in preferred if g in free_gpu_ids]
            if len(free_gpu_ids) < gpu_group_size:
                return None
        numa_map = self.gpu_numa_affinity.get(node_id, {})
        numa_loads: Dict[int, int] = defaultdict(int)
        for gpu_id, count in self._effective_gpu_counts(node_id, total_gpus, reserved_gpu_ids).items():
            numa_loads[numa_map.get(gpu_id, -1)] += count
        return self._pick_balanced_numa_gpu_ids(
            node_id, free_gpu_ids, int(gpu_group_size), numa_loads
        )

    # ------------------------------------------------------------------ #
    #  Post-init VRAM (router calls this after ``init_backend``)          #
    # ------------------------------------------------------------------ #

    async def refresh_cluster_gpu_vram(self) -> None:
        """Refresh per-GPU used MiB on all workers (call after instance backend init)."""
        if not self._gpu_vram_occupancy_enabled:
            return
        try:
            await self._refresh_all_worker_nodes_vram()
        except Exception as exc:
            logger.warning("Cluster VRAM refresh after init failed: %s", exc)

    # ------------------------------------------------------------------ #
    #  Resource allocation / deallocation                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_pp_gpu_ids_for_public_api(info: Mapping[str, Any]) -> Dict[str, Any]:
        """Omit per-GPU indices in the dict returned to the router (PP path)."""
        out: Dict[str, Any] = copy.deepcopy(dict(info))
        out["gpu_ids"] = []
        for st in out.get("allocations") or []:
            if isinstance(st, dict):
                st["gpu_ids"] = []
                st["numa_nodes"] = []
        return out

    async def allocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ) -> Any:
        logger.info("Model %s allocation requested (instance %s)", model_name, instance_id)
        normalized = self._normalize_parallel_request(resources)
        normalized["instance_id"] = instance_id
        async with self.queue_lock:
            if not self.running:
                raise RuntimeError("Scheduler is not running")
            if instance_id in self.instance_allocations or any(
                info["instance_id"] == instance_id
                for queue in self.model_loading_queues.values()
                for _, info, future in queue
                if not future.done()
            ):
                raise ValueError(f"Instance {instance_id!r} already has an allocation or pending request")
            if model_name not in self.model_loading_queues:
                self.model_loading_queues[model_name] = []
            allocation_result = self.loop.create_future()
            self.model_loading_queues[model_name].append(
                (time.time(), normalized, allocation_result)
            )
        logger.info("Model %s added to the loading queue", model_name)
        try:
            allocation_info = await allocation_result
        except asyncio.CancelledError:
            async with self.queue_lock:
                queue = self.model_loading_queues.get(model_name, [])
                self.model_loading_queues[model_name] = [
                    entry for entry in queue if entry[2] is not allocation_result
                ]
            allocation_result.cancel()
            await self.deallocate_resource(model_name, instance_id, resources)
            raise
        if allocation_info is None:
            raise RuntimeError(
                f"Allocation for instance {instance_id!r} was cancelled during scheduler shutdown"
            )

        if (
            isinstance(allocation_info, dict)
            and int(allocation_info.get("pipeline_parallel_size", 1) or 1) > 1
        ):
            return self._strip_pp_gpu_ids_for_public_api(allocation_info)
        return allocation_info

    def _record_allocation(self, model_name: str, normalized: Mapping, allocation_info: Mapping) -> None:
        """Publish capacity before waking the allocation caller (metadata lock held)."""
        instance_id = normalized["instance_id"]
        node_id = (
            allocation_info.get("node_id")
            if isinstance(allocation_info, dict)
            else allocation_info
        )

        gpu_ids = list(allocation_info.get("gpu_ids", []))
        allocations = copy.deepcopy(list(allocation_info.get("allocations", [])))
        if not allocations and node_id is not None:
            allocations = [self._build_stage_allocation(node_id, gpu_ids, stage_idx=0)]
        self.model_instance.setdefault(model_name, {})[instance_id] = node_id
        self.instance_allocations[instance_id] = {
            "model_name": model_name,
            "node_id": node_id,
            "gpu_ids": gpu_ids,
            "allocations": allocations,
            "empty_instance": normalized["empty_instance"],
            "tp_size": int(allocation_info.get("tensor_parallel_size", normalized["tp_size"])),
            "pp_size": int(allocation_info.get("pipeline_parallel_size", normalized["pp_size"])),
            "weights_loaded": not normalized["empty_instance"],
        }

    async def adopt_hydraserve_pair_allocation(
        self,
        model_name: str,
        pair_instance_id: str,
        primary_instance_id: str,
        *,
        pair_pp_size: int,
        pair_tp_size: int,
    ) -> Optional[Dict[str, Any]]:
        """Register HydraServe *pair* PP placement without the FCFS queue.

        * **Stage 0** uses the *primary* instance’s ``node_id`` and ``gpu_ids`` (same
          ``gpu_worker`` pin as primary in the placement group’s first stage bundles).
        * **Stages 1..pp-1** follow the same **one node per stage**, **tp = pair_tp_size**
          policy as :meth:`_try_allocate_pipeline_stages` (distinct nodes; abstract
          per-stage ``tp_slots`` on each node, matching normal PP).
        """
        async with self.metadata_lock:
            src = self.instance_allocations.get(primary_instance_id)
            if not src or src.get("model_name") != model_name:
                logger.info(
                    "adopt_hydraserve_pair: primary %s missing for model %s",
                    primary_instance_id,
                    model_name,
                )
                return None
            src = copy.deepcopy(src)
            p_node = src.get("node_id")
            p_gids = list(src.get("gpu_ids") or [])
            if not p_node or not p_gids:
                logger.info("adopt_hydraserve_pair: primary %s has no node_id or gpu_ids", primary_instance_id)
                return None

        worker_nodes = await self._get_worker_nodes()
        p_node_s = str(p_node)
        if p_node_s not in worker_nodes:
            logger.info(
                "adopt_hydraserve_pair: primary node %s not in worker_nodes",
                p_node_s,
            )
            return None

        stage_allocations = self._build_hydraserve_pair_stage_allocations(
            worker_nodes,
            p_node_s,
            p_gids,
            int(pair_pp_size),
            int(pair_tp_size),
        )
        if not stage_allocations:
            logger.info("adopt_hydraserve_pair: no stage allocations found for pair %s", pair_instance_id)
            return None

        first = stage_allocations[0]
        row = {
            "model_name": model_name,
            "node_id": first.get("node_id"),
            "gpu_ids": list(first.get("gpu_ids") or []),
            "allocations": copy.deepcopy(stage_allocations),
            "empty_instance": True,
            "tp_size": int(pair_tp_size),
            "pp_size": int(pair_pp_size),
            "weights_loaded": False,
        }
        async with self.queue_lock:
            async with self.metadata_lock:
                if self.instance_allocations.get(primary_instance_id) != src:
                    return None
                if pair_instance_id in self.instance_allocations or any(
                    info["instance_id"] == pair_instance_id
                    for queue in self.model_loading_queues.values()
                    for _, info, future in queue if not future.done()
                ):
                    raise ValueError(f"Instance {pair_instance_id!r} already has an allocation or pending request")
                self.instance_allocations[pair_instance_id] = row
                self.model_instance.setdefault(model_name, {})[pair_instance_id] = str(first.get("node_id"))

        public = {
            "node_id": first.get("node_id"),
            "gpu_ids": list(first.get("gpu_ids") or []),
            "allocations": copy.deepcopy(stage_allocations),
            "tensor_parallel_size": int(pair_tp_size),
            "pipeline_parallel_size": int(pair_pp_size),
        }
        if int(pair_pp_size) > 1:
            logger.info("adopt_hydraserve_pair: returning public api for pair %s", pair_instance_id)
            return self._strip_pp_gpu_ids_for_public_api(public)
        return public

    async def get_instance_scheduler_allocation(
        self, instance_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the full internal record for *instance_id* (includes PP ``gpu_ids``)."""
        async with self.metadata_lock:
            row = self.instance_allocations.get(instance_id)
            if row is None:
                return None
            return copy.deepcopy(row)

    async def deallocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ) -> None:
        logger.info("Deallocating model %s instance %s", model_name, instance_id)
        async with self.metadata_lock:
            if model_name not in self.model_instance:
                logger.error("Model %s not found", model_name)
                return
            if instance_id not in self.model_instance[model_name]:
                logger.error("Instance %s not found for model %s", instance_id, model_name)
                return
            allocation = self.instance_allocations.get(instance_id)
            if allocation is not None and allocation.get("model_name") != model_name:
                raise ValueError(f"Instance {instance_id!r} allocation belongs to another model")
            node_id = self.model_instance[model_name].pop(instance_id)
            self.instance_allocations.pop(instance_id, None)
            logger.info("Removed instance %s from allocations (was on node %s)", instance_id, node_id)

        if self._gpu_vram_occupancy_enabled:
            try:
                await self._refresh_all_worker_nodes_vram()
            except Exception as exc:
                logger.warning("VRAM refresh after instance deallocate failed: %s", exc)

        logger.info("Deallocated model %s instance %s", model_name, instance_id)

    async def _patch_instance_allocation(
        self, instance_id: str, updates: Mapping[str, Any], log_line: str
    ) -> None:
        async with self.metadata_lock:
            info = self.instance_allocations.get(instance_id)
            if info is None:
                logger.warning("%s: unknown instance_id=%s", log_line, instance_id)
                return
            for key, value in updates.items():
                info[key] = value
        logger.info("Scheduler: instance %s marked %s", instance_id, log_line)

    async def mark_cold_start_committed(self, instance_id: str) -> None:
        """No-op: kept for API compatibility.

        With the simplified GPU accounting model every instance occupies its GPUs
        from the moment ``allocate_resource`` is called, so there is no separate
        "commit" step.
        """
        pass

    async def mark_weights_loaded(self, instance_id: str) -> None:
        await self._patch_instance_allocation(
            instance_id, {"weights_loaded": True}, "weights_loaded"
        )

    # ------------------------------------------------------------------ #
    #  Control loop                                                       #
    # ------------------------------------------------------------------ #

    async def _control_loop(self) -> None:
        try:
            await self._run_control_loop()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Scheduler control loop failed")
            async with self.running_lock:
                self.running = False
                await self._release_waiters(exc)

    async def _run_control_loop(self) -> None:
        logger.info("Starting control loop")
        while self.running:
            loading_requests: List[Tuple] = []
            reserved_gpu_ids_by_node: Dict[str, Set[int]] = defaultdict(set)
            reserved_slot_count_by_node: Dict[str, int] = defaultdict(int)

            async with self.queue_lock:
                for model_name, loading_queue in self.model_loading_queues.items():
                    for request_time, request_info, allocation_result in loading_queue:
                        loading_requests.append(
                            (model_name, request_time, request_info, allocation_result)
                        )

            if loading_requests:
                worker_nodes = await self._get_worker_nodes()
                _fp = self._scheduling_state_fingerprint(worker_nodes)
                if _fp != self._log_fp_control_loop_worker_nodes:
                    self._log_fp_control_loop_worker_nodes = _fp
                    logger.info(
                        "Worker nodes (summary): %s", self._compact_worker_nodes_log(worker_nodes)
                    )
                loading_requests.sort(key=lambda x: x[1])  # FCFS ordering

                for model_name, request_time, request_info, allocation_result in loading_requests:
                    tp_size = int(request_info["tp_size"])
                    pp_size = int(request_info["pp_size"])

                    if (self.auto_pipeline_split and pp_size == 1 and tp_size > 1
                            and request_info.get("preferred_gpu_ids") is None):
                        available = [
                            self._node_schedulable_free_gpus(
                                str(nid), int(info.get("total_gpu", 0) or 0),
                                list(reserved_gpu_ids_by_node[nid]),
                                reserved_slot_count_by_node[nid],
                            )
                            for nid, info in worker_nodes.items()
                        ]
                        if available and max(available) < tp_size:
                            total_requested = int(
                                request_info.get("num_gpus", tp_size) or tp_size
                            )
                            limit = max(available)
                            for candidate_tp in range(min(limit, total_requested), 0, -1):
                                if total_requested % candidate_tp == 0 and total_requested // candidate_tp > 1:
                                    tp_size = candidate_tp
                                    pp_size = total_requested // candidate_tp
                                    break

                    allocation_info = None

                    preferred_node_id = request_info.get("preferred_node_id")
                    same_shape_nodes: List[str] = []
                    if preferred_node_id is None:
                        # Keep replicas and identically shaped TP/PP models together.
                        # This reduces fragmentation and makes a second prewarm replica
                        # land beside the first one when capacity permits.
                        for existing in self.instance_allocations.values():
                            if (
                                existing.get("model_name") == model_name
                                or (
                                    int(existing.get("tp_size", existing.get("tensor_parallel_size", 1)) or 1) == tp_size
                                    and int(existing.get("pp_size", existing.get("pipeline_parallel_size", 1)) or 1) == pp_size
                                )
                            ):
                                for stage in existing.get("allocations", []):
                                    nid = stage.get("node_id")
                                    if nid and nid not in same_shape_nodes:
                                        same_shape_nodes.append(nid)

                    if pp_size > 1:
                        allocation_info = self._allocate_pipeline_stages(
                            worker_nodes=worker_nodes,
                            tp_size=tp_size,
                            pp_size=pp_size,
                            reserved_gpu_ids_by_node=reserved_gpu_ids_by_node,
                            preferred_pp0_node_id=preferred_node_id,
                            reserved_slot_count_by_node=reserved_slot_count_by_node,
                        )
                    elif preferred_node_id is not None:
                        # Strict CPU–GPU colocation: only place on the hinted node; wait
                        # (keep request queued) until TP fits there — no other-node fallback.
                        node_info = worker_nodes.get(preferred_node_id)
                        if node_info is not None and self._can_place_tp_on_node(
                            preferred_node_id,
                            int(node_info.get("total_gpu", 0)),
                            tp_size,
                            list(reserved_gpu_ids_by_node[preferred_node_id]),
                        ):
                            candidate_gpu_ids = self._select_gpu_ids(
                                node_id=preferred_node_id,
                                node_info=node_info,
                                gpu_group_size=tp_size,
                                preferred_gpu_ids=request_info.get("preferred_gpu_ids"),
                                reserved_gpu_ids=list(
                                    reserved_gpu_ids_by_node[preferred_node_id]
                                ),
                            )
                            if candidate_gpu_ids:
                                reserved_gpu_ids_by_node[preferred_node_id].update(
                                    candidate_gpu_ids
                                )
                                allocation_info = {
                                    "node_id": preferred_node_id,
                                    "gpu_ids": candidate_gpu_ids,
                                    "allocations": [
                                        self._build_stage_allocation(
                                            preferred_node_id, candidate_gpu_ids, 0
                                        )
                                    ],
                                    "tensor_parallel_size": tp_size,
                                    "pipeline_parallel_size": 1,
                                }
                    else:
                        # TP-only, no colocation hint: prefer most free GPUs.
                        def _tp_node_rank(item: Tuple[str, Mapping]) -> Tuple:
                            _nid, ninfo = item
                            _t = int(ninfo.get("total_gpu", 0) or 0)
                            _free = (
                                self._compute_node_free_gpus(_nid, _t) if _t else 0
                            )
                            return (_nid not in same_shape_nodes, -_free, _nid)

                        for node_id, node_info in sorted(
                            worker_nodes.items(), key=_tp_node_rank
                        ):
                            if not self._can_place_tp_on_node(
                                node_id,
                                int(node_info.get("total_gpu", 0)),
                                tp_size,
                                list(reserved_gpu_ids_by_node[node_id]),
                                reserved_slot_count_by_node[node_id],
                            ):
                                continue
                            candidate_gpu_ids = self._select_gpu_ids(
                                node_id=node_id,
                                node_info=node_info,
                                gpu_group_size=tp_size,
                                preferred_gpu_ids=request_info.get("preferred_gpu_ids"),
                                reserved_gpu_ids=list(reserved_gpu_ids_by_node[node_id]),
                            )
                            if not candidate_gpu_ids:
                                continue
                            reserved_gpu_ids_by_node[node_id].update(candidate_gpu_ids)
                            allocation_info = {
                                "node_id": node_id,
                                "gpu_ids": candidate_gpu_ids,
                                "allocations": [
                                    self._build_stage_allocation(node_id, candidate_gpu_ids, 0)
                                ],
                                "tensor_parallel_size": tp_size,
                                "pipeline_parallel_size": 1,
                            }
                            break

                    if allocation_info:
                        for stage in allocation_info.get("allocations", []):
                            stage_nid = stage.get("node_id")
                            if not stage_nid:
                                continue
                            s_gids = stage.get("gpu_ids") or []
                            if s_gids:
                                reserved_gpu_ids_by_node[stage_nid].update(s_gids)
                            else:
                                reserved_slot_count_by_node[stage_nid] += int(
                                    stage.get("tp_slots")
                                    or allocation_info.get("tensor_parallel_size", 0)
                                    or 0
                                )
                        async with self.queue_lock:
                            if allocation_result.done():
                                continue
                            try:
                                self.model_loading_queues[model_name].remove(
                                    (request_time, request_info, allocation_result)
                                )
                                async with self.metadata_lock:
                                    if allocation_result.done():
                                        continue
                                    self._record_allocation(model_name, request_info, allocation_info)
                                    allocation_result.set_result(allocation_info)
                                # The committed record now accounts for these slots.
                                for stage in allocation_info.get("allocations", []):
                                    nid = stage["node_id"]
                                    gids = stage.get("gpu_ids") or []
                                    if gids:
                                        reserved_gpu_ids_by_node[nid].difference_update(gids)
                                    else:
                                        reserved_slot_count_by_node[nid] -= int(
                                            stage.get("tp_slots") or allocation_info["tensor_parallel_size"]
                                        )
                            except ValueError:
                                break
                        self._log_fp_stall_no_node = None
                        logger.info(
                            "Allocated node=%s for model=%s TP=%s PP=%s",
                            allocation_info.get("node_id"), model_name, tp_size, pp_size,
                        )
                    else:
                        if preferred_node_id is not None:
                            logger.debug(
                                "Colocation wait: model=%s needs node=%s (TP=%s PP=%s)",
                                model_name, preferred_node_id, tp_size, pp_size,
                            )
                        else:
                            stall_key = (
                                model_name,
                                tp_size,
                                pp_size,
                                self._scheduling_state_fingerprint(worker_nodes),
                            )
                            if stall_key != self._log_fp_stall_no_node:
                                self._log_fp_stall_no_node = stall_key
                                logger.info(
                                    "No available node for model=%s (TP=%s PP=%s); %s",
                                    model_name, tp_size, pp_size,
                                    self._compact_worker_nodes_log(worker_nodes),
                                )
                            else:
                                logger.debug(
                                    "No available node (unchanged capacity) for model=%s",
                                    model_name,
                                )

                await self._update_worker_nodes(worker_nodes)

            await asyncio.sleep(self.control_loop_interval_s)

    # ------------------------------------------------------------------ #
    #  Worker node queries / updates                                      #
    # ------------------------------------------------------------------ #

    async def _refresh_all_worker_nodes_vram(self) -> None:
        """Run GPUtil on every GPU *logical* worker (Ray placement) and cache MiB/ GPU."""
        if not self._gpu_vram_occupancy_enabled:
            return
        if not ray.is_initialized():
            logger.debug("Ray not initialized; skip cluster VRAM refresh")
            return
        wn = get_worker_nodes()
        if not wn:
            return
        bc = dict(self.scheduler_config) if self.scheduler_config else {}
        refs: List = []
        node_ids: List[str] = []
        for node_id in sorted(wn.keys()):
            res = ray_gpu_actor_placement_resources(node_id, bc)
            if not res:
                continue
            refs.append(_remote_probe_gpu_vram_mib.options(resources=res).remote())
            node_ids.append(node_id)
        if not refs:
            return

        def _fetch() -> List[Optional[List[int]]]:
            return list(ray.get(refs, timeout=90.0))

        try:
            results = await asyncio.to_thread(_fetch)
        except Exception as exc:
            logger.warning("Cluster VRAM probe (ray.get) failed: %s", exc)
            return
        if len(results) != len(node_ids):
            logger.warning(
                "VRAM probe result count mismatch: nodes=%s results=%s",
                len(node_ids),
                len(results),
            )
            return
        new_map = dict(self._node_gpu_vram_mib)
        n_ok = 0
        n_none = 0
        for nid, per in zip(node_ids, results):
            if per is not None:
                new_map[nid] = [int(x) for x in per]
                n_ok += 1
            else:
                n_none += 1
        self._node_gpu_vram_mib = new_map
        logger.info(
            "Refreshed GPU VRAM: %d node(s) with data, %d node(s) returned no probe ",
            n_ok, n_none,
        )

    async def _get_worker_nodes(self) -> Dict:
        worker_nodes = get_worker_nodes()
        resolved_affinity = resolve_gpu_numa_affinity(worker_nodes, self.gpu_numa_affinity)

        for node_id, node_info in worker_nodes.items():
            total_gpus = int(node_info.get("total_gpu", 0))
            node_info["per_gpu"] = self._build_per_gpu_snapshot(node_id, total_gpus)

        async with self.metadata_lock:
            updated = copy.deepcopy(self.worker_nodes)
            for node_id, node_info in worker_nodes.items():
                if node_id not in updated:
                    updated[node_id] = copy.deepcopy(node_info)
                else:
                    updated[node_id]["total_gpu"] = node_info["total_gpu"]
                    updated[node_id]["per_gpu"] = node_info.get("per_gpu", [])
            self.worker_nodes = updated
            self.gpu_numa_affinity = resolved_affinity

        return updated

    async def _update_worker_nodes(self, worker_nodes: Dict) -> None:
        async with self.metadata_lock:
            updated = copy.deepcopy(self.worker_nodes)
        for node_id, node_info in worker_nodes.items():
            if node_id not in updated:
                logger.error("Node %s not found in worker_nodes during update", node_id)
                continue
            updated[node_id] = copy.deepcopy(node_info)
        async with self.metadata_lock:
            self.worker_nodes = updated
        _pfp = self._scheduling_state_fingerprint(updated)
        if _pfp != self._log_fp_persisted_worker_nodes:
            self._log_fp_persisted_worker_nodes = _pfp
            logger.info(
                "Worker nodes persisted: %s", self._compact_worker_nodes_log(updated)
            )

    # ------------------------------------------------------------------ #
    #  Pipeline stage allocation                                          #
    # ------------------------------------------------------------------ #

    def _max_hardware_gpus_per_node(self, worker_nodes: Mapping[str, Mapping]) -> int:
        return max((int(n.get("total_gpu", 0)) for n in worker_nodes.values()), default=0)

    def _allocate_pipeline_stages(
        self,
        worker_nodes: Mapping[str, Mapping],
        tp_size: int,
        pp_size: int,
        reserved_gpu_ids_by_node: Mapping[str, Set[int]],
        preferred_pp0_node_id: Optional[str] = None,
        reserved_slot_count_by_node: Optional[Mapping[str, int]] = None,
        **kwargs: Any,
    ) -> Optional[Dict]:
        """Place *pp_size* pipeline stages on distinct nodes, *tp_size* GPUs each.

        Availability is by **free-GPU count** per node (instances, optional VRAM threshold,
        unmapped PP ``tp_slots``, and in-flight slot reservations in this batch).
        Stages do not pin concrete ``gpu_ids``; each stage records ``tp_slots`` for
        capacity accounting. Public return from :meth:`allocate_resource` may still
        omit per-GPU fields; the router may resolve devices on the worker.
        """
        if kwargs:
            logger.debug("Ignoring legacy pipeline kwargs: %s", sorted(kwargs.keys()))
        if tp_size <= 0 or pp_size <= 1:
            return None

        return self._try_allocate_pipeline_stages(
            worker_nodes=worker_nodes,
            tp_size=tp_size,
            pp_size=pp_size,
            reserved_gpu_ids_by_node=reserved_gpu_ids_by_node,
            preferred_pp0_node_id=preferred_pp0_node_id,
            reserved_slot_count_by_node=reserved_slot_count_by_node,
        )

    def _try_allocate_pipeline_stages(
        self,
        worker_nodes: Mapping[str, Mapping],
        tp_size: int,
        pp_size: int,
        reserved_gpu_ids_by_node: Mapping[str, Set[int]],
        preferred_pp0_node_id: Optional[str] = None,
        reserved_slot_count_by_node: Optional[Mapping[str, int]] = None,
    ) -> Optional[Dict]:
        batch_slots: Mapping[str, int] = reserved_slot_count_by_node or {}
        stage_allocations: List[Dict] = []
        used_nodes: Set[str] = set()
        local_slot_reserved: Dict[str, int] = defaultdict(int)

        for stage_idx in range(pp_size):
            # Stage 0 with preferred_pp0_node_id: only that node (strict colocation).
            _preferred = preferred_pp0_node_id if stage_idx == 0 else None

            def _stage_node_rank(
                item: Tuple[str, Mapping],
                _p=_preferred,
                _bs: Mapping[str, int] = batch_slots,
            ) -> Tuple:
                nid, ninfo = item
                tn = int(ninfo.get("total_gpu", 0) or 0)
                headroom = (
                    self._node_schedulable_free_gpus(
                        nid, tn, None, int(_bs.get(nid, 0) or 0) + int(local_slot_reserved[nid] or 0)
                    )
                    if tn > 0
                    else 0
                )
                return (
                    0 if _p is not None and nid == _p else 1,
                    -headroom,
                    nid,
                )

            chosen_stage = None

            if stage_idx == 0 and _preferred is not None:
                if _preferred not in worker_nodes:
                    return None
                node_iter: List[Tuple[str, Mapping]] = [
                    (_preferred, worker_nodes[_preferred])
                ]
            else:
                node_iter = sorted(worker_nodes.items(), key=_stage_node_rank)

            for node_id, node_info in node_iter:
                if node_id in used_nodes:
                    continue
                total_n = int(node_info.get("total_gpu", 0))
                merged_gpu_reserved = list(reserved_gpu_ids_by_node.get(node_id, set()))
                extra_slots = int(batch_slots.get(node_id, 0) or 0) + int(
                    local_slot_reserved[node_id] or 0
                )
                if not self._can_place_tp_on_node(
                    node_id, total_n, tp_size, merged_gpu_reserved, extra_slots
                ):
                    continue
                chosen_stage = self._build_stage_allocation(
                    node_id, [], stage_idx, tp_size=tp_size
                )
                local_slot_reserved[node_id] += tp_size
                used_nodes.add(node_id)
                break

            if chosen_stage is None:
                return None
            stage_allocations.append(chosen_stage)

        for stage in stage_allocations:
            nid = stage["node_id"]
            # Expose the node's IP address (diagnostics) and Ray node id (unique per Ray node).
            wn = worker_nodes.get(nid, {})
            stage["address"] = str(wn.get("address") or "")
            stage["ray_node_id"] = wn.get("ray_node_id")

        first_stage = stage_allocations[0]
        return {
            "node_id": first_stage["node_id"],
            "gpu_ids": first_stage["gpu_ids"],
            "allocations": stage_allocations,
            "tensor_parallel_size": tp_size,
            "pipeline_parallel_size": pp_size,
        }

    def _build_hydraserve_pair_stage_allocations(
        self,
        worker_nodes: Mapping[str, Mapping],
        primary_node_id: str,
        primary_gpu_ids: List[int],
        pair_pp_size: int,
        pair_tp_size: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Build PP *allocations* for HydraServe pair: stage0 = primary (same ``gpu_worker``/GPUs as primary).

        Stages ``1..pp-1`` prefer **distinct** Ray nodes (same as
        :meth:`_try_allocate_pipeline_stages`), *pair_tp_size* GPUs each (unmapped
        ``tp_slots`` until workers bind devices). If no other node qualifies
        (typical single-machine cluster), later stages may fall back to the
        primary node when enough **additional** GPUs remain there.
        """
        pp = int(pair_pp_size)
        tp = int(pair_tp_size)
        if pp < 1 or tp < 1:
            return None
        p_node = str(primary_node_id)
        p_gids = [int(x) for x in primary_gpu_ids]
        if len(p_gids) != tp:
            logger.warning(
                "Hydra pair stage0: primary has %d gpu_ids but pair_tp_size=%d; "
                "PG may not match primary TP group",
                len(p_gids),
                tp,
            )
        st0 = self._build_stage_allocation(p_node, p_gids, 0)
        if pp == 1:
            out = [st0]
        else:
            used_nodes: Set[str] = {p_node}
            local_slot_reserved: Dict[str, int] = defaultdict(int)
            batch_slots: Dict[str, int] = {}
            out: List[Dict[str, Any]] = [st0]
            for stage_idx in range(1, pp):

                def _stage_node_rank(
                    item: Tuple[str, Mapping],
                    _bs: Mapping[str, int] = batch_slots,
                ) -> Tuple:
                    nid, ninfo = item
                    tnn = int(ninfo.get("total_gpu", 0) or 0)
                    headroom = (
                        self._node_schedulable_free_gpus(
                            nid,
                            tnn,
                            None,
                            int(_bs.get(nid, 0) or 0)
                            + int(local_slot_reserved.get(nid, 0) or 0),
                        )
                        if tnn > 0
                        else 0
                    )
                    return (-headroom, nid)

                chosen_stage: Optional[Dict] = None
                for n_id, n_info in sorted(worker_nodes.items(), key=_stage_node_rank):
                    if n_id in used_nodes:
                        continue
                    total_n = int(n_info.get("total_gpu", 0))
                    merged_gpu_reserved: List[int] = []
                    extra_slots = int(batch_slots.get(n_id, 0) or 0) + int(
                        local_slot_reserved.get(n_id, 0) or 0
                    )
                    if not self._can_place_tp_on_node(
                        n_id, total_n, tp, merged_gpu_reserved, extra_slots
                    ):
                        continue
                    chosen_stage = self._build_stage_allocation(
                        n_id, [], stage_idx, tp_size=tp
                    )
                    local_slot_reserved[n_id] += tp
                    used_nodes.add(n_id)
                    break

                if chosen_stage is None and p_node in worker_nodes:
                    # Primary is excluded from the loop above; allow same-node PP when
                    # there is no second worker or every other node lacks capacity.
                    n_id = p_node
                    n_info = worker_nodes.get(n_id) or {}
                    total_n = int(n_info.get("total_gpu", 0))
                    merged_gpu_reserved: List[int] = []
                    extra_slots = int(batch_slots.get(n_id, 0) or 0) + int(
                        local_slot_reserved.get(n_id, 0) or 0
                    )
                    if self._can_place_tp_on_node(
                        n_id, total_n, tp, merged_gpu_reserved, extra_slots
                    ):
                        chosen_stage = self._build_stage_allocation(
                            n_id, [], stage_idx, tp_size=tp
                        )
                        local_slot_reserved[n_id] += tp
                        logger.info(
                            "Hydra pair: PP stage %d placed on primary node %s "
                            "(no distinct worker; tp=%d)",
                            stage_idx,
                            n_id,
                            tp,
                        )

                if chosen_stage is None:
                    logger.warning(
                        "Hydra pair: no node for PP stage %d (need one node, tp=%d)",
                        stage_idx,
                        tp,
                    )
                    return None
                out.append(chosen_stage)

        for stage in out:
            nid = str(stage.get("node_id", "") or "")
            if not nid:
                continue
            wn = worker_nodes.get(nid, {}) or {}
            stage["address"] = str(wn.get("address") or "")
            stage["ray_node_id"] = wn.get("ray_node_id")

        return out

    # ------------------------------------------------------------------ #
    #  Per-node cold-start coordination                                   #
    # ------------------------------------------------------------------ #

    def _get_node_cold_start_event(self, node_id: str) -> asyncio.Event:
        if node_id not in self.node_cold_start_ready_events:
            event = asyncio.Event()
            event.set()
            self.node_cold_start_ready_events[node_id] = event
        return self.node_cold_start_ready_events[node_id]

    async def get_node_for_model(self, model_name: str) -> Optional[str]:
        async with self.metadata_lock:
            for _instance_id, node_id in self.model_instance.get(model_name, {}).items():
                return node_id
        return None
    
    async def try_claim_cold_start(
        self, node_id:str, model_name:str, claim_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        async with self.cold_start_state_lock:
            status = self.cold_start_status.get((node_id, model_name))
            if status and status.get("in_progress"):
                return False, status.get("method")
            self.cold_start_status[(node_id, model_name)] = {
                "in_progress": True,
                "method": "pending",
                "claim_id": claim_id,
            }
            self._get_node_cold_start_event(node_id).clear()
            logger.info("cold start slot claimed")
            return True, None

    async def start_cold_start(self, node_id: str, model_name: str, method: str) -> None:
        async with self.cold_start_state_lock:
            status = self.cold_start_status.setdefault((node_id, model_name), {})
            status.update(in_progress=True, method=method)
        logger.info("Cold start started: node=%s model=%s method=%s", node_id, model_name, method)

    async def finish_cold_start(self, node_id: str, model_name: str, claim_id: Optional[str] = None) -> None:
        async with self.cold_start_state_lock:
            status = self.cold_start_status.get((node_id, model_name))
            if claim_id is not None and (status is None or status.get("claim_id") != claim_id):
                return
            self.cold_start_status.pop((node_id, model_name), None)
            any_active = any(
                k[0] == node_id and v.get("in_progress")
                for k, v in self.cold_start_status.items()
            )
            if not any_active:
                self._get_node_cold_start_event(node_id).set()
        logger.info("Cold start finished: node=%s model=%s", node_id, model_name)

    async def cancel_cold_start(self, node_id: str, model_name: str) -> None:
        """Release a claim when admission fails before the loading method starts."""
        await self.finish_cold_start(node_id, model_name)

    async def signal_cold_start_ready(self, node_id: str) -> None:
        self._get_node_cold_start_event(node_id).set()
        logger.info("Cold start ready signal: node=%s", node_id)

    async def get_cold_start_status(
        self, node_id: str, model_name: str
    ) -> Tuple[bool, Optional[str]]:
        async with self.cold_start_state_lock:
            status = self.cold_start_status.get((node_id, model_name))
            if status and status.get("in_progress"):
                return True, status.get("method")
            return False, None

    async def wait_cold_start_ready(self, node_id: str) -> None:
        await self._get_node_cold_start_event(node_id).wait()

    async def notify_first_token(self, node_id: str, model_name: str) -> None:
        async with self.cold_start_state_lock:
            status = self.cold_start_status.get((node_id, model_name))
            if status and status.get("in_progress") and status.get("method") == "layerwise":
                self._get_node_cold_start_event(node_id).set()
                logger.info(
                    "Layerwise cold start: first token → node ready (node=%s model=%s)",
                    node_id, model_name,
                )

    async def notify_first_token_by_instance(
        self, instance_id: str, model_name: str
    ) -> None:
        node_id = None
        async with self.metadata_lock:
            alloc = self.instance_allocations.get(instance_id)
            if alloc:
                node_id = alloc.get("node_id")
        if node_id:
            await self.notify_first_token(node_id, model_name)
        else:
            logger.warning(
                "notify_first_token_by_instance: no allocation for instance %s", instance_id
            )
