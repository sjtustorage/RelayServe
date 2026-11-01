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
import logging
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import ray

from sllm.sllm.logger import init_logger
from sllm.sllm.loading_perf_profile import layer_idxes_by_pp_stages

from ..utils import (
    InstanceHandle,
    get_worker_nodes,
    pp_placement_pin_resources,
    ray_gpu_actor_placement_resources,
)
from .router_utils import (
    SllmRouter,
    await_backend_call,
    close_backend_actor,
    merge_runtime_env,
    start_instance,
    start_instance_spawner_options,
    validate_auto_scaling_config,
    instance_request_capacity,
)

logger = init_logger(__name__)


def _pp_stage0_coordinator_bundle(
    stage0: Dict[str, Any],
    backend_config: Dict[str, Any],
    frac: float,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """PP bundle 0 (CPU coordinator) and matching ``resources=`` for the driver actor.

    Ray requires the actor's resource request to fit **bundle index 0** exactly. This must
    use the same pin logic as :func:`_build_pp_placement_group` (single source of truth).

    We pin with **logical** ``gpu_worker_<node_id>`` only (``pin_mode="logical"``), never
    ``node:<IP>`` / ``node:*`` from ``ray_node_id``, so PG bundles and the coordinator
    actor stay aligned without IP-based Ray resource keys.

    ``resources`` omits ``CPU`` because ``startup_config`` sets ``num_cpus`` separately.
    """
    bc = dict(backend_config or {})
    pin0 = pp_placement_pin_resources(
        stage0, bc, worker_resource_fraction=frac, pin_mode="logical"
    )
    b0: Dict[str, Any] = {"CPU": 1.0, **pin0}
    actor_pins = {k: float(v) for k, v in b0.items() if k != "CPU"}
    return b0, actor_pins


async def _build_pp_placement_group(
    allocations: List[Dict[str, Any]],
    tp_size: int,
    *,
    backend_config: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Create a Ray PlacementGroup that pins each PP stage using logical ``gpu_worker_*`` only.

    This router always uses ``pin_mode="logical"`` (``gpu_worker_<scheduler node_id>`` via
    :func:`~sllm.utils.pp_placement_pin_resources`), not ``node:*`` keys from ``ray_node_id``.
    That avoids mismatches between ``placement_group`` bundles and the coordinator actor when
    Ray exposes different ``node:`` resource strings (IP vs internal names) across calls.

    Bundle layout (strategy=PACK), default ``pp_pg_one_bundle_per_stage`` = False:
      - Bundle 0 (CPU coordinator): ``{"CPU": 1, **pin}`` with the stage-0 logical pin.
      - Per stage: ``tp_size`` bundles ``{"GPU": 1.0, **pin}`` (e.g. two bundles with
        ``gpu_worker_0``, then two with ``gpu_worker_1`` for PP=2 TP=2).
    If ``pp_pg_one_bundle_per_stage`` is True, use one bundle per PP stage with
    ``{"GPU": tensor_parallel_size, **pin}`` instead.

    Use **PACK** (not STRICT_PACK) so PP stages can span multiple physical nodes.
    """
    try:
        from ray.util.placement_group import placement_group as ray_placement_group
    except ImportError:
        logger.error("ray.util.placement_group unavailable; cannot build PP placement group")
        return None

    if not allocations:
        return None

    bc = dict(backend_config or {})
    frac = float(bc.get("ray_worker_resource_fraction", 0.001))
    one_bundle_per_stage = bool(bc.get("pp_pg_one_bundle_per_stage", False))

    stage0 = allocations[0]
    b0, _ = _pp_stage0_coordinator_bundle(stage0, bc, frac)
    if b0 == {"CPU": 1.0}:
        logger.warning(
            "PP placement group: stage-0 has no logical node_id and no ray_node pin; "
            "coordinator bundle is not node-pinned"
        )
    bundles: List[Dict[str, Any]] = [b0]

    for stage in allocations:
        pin = pp_placement_pin_resources(
            stage, bc, worker_resource_fraction=frac, pin_mode="logical"
        )
        if one_bundle_per_stage:
            g: Dict[str, Any] = {"GPU": float(tp_size)}
            if pin:
                g.update(pin)
            else:
                logger.warning(
                    "PP placement group: stage %d could not be pinned (node_id / ray_node_id); "
                    "GPU bundle is not node-pinned",
                    stage.get("stage_idx", -1),
                )
            bundles.append(g)
        else:
            for _ in range(tp_size):
                if pin:
                    bundles.append({"GPU": 1.0, **pin})
                else:
                    logger.warning(
                        "PP placement group: stage %d could not be pinned (node_id / ray_node_id); "
                        "GPU bundle is not node-pinned",
                        stage.get("stage_idx", -1),
                    )
                    bundles.append({"GPU": 1.0})

    logger.info(
        "Creating PP placement group with %d bundles (1 CPU coordinator + %d GPU bundle(s)): %s",
        len(bundles), len(bundles) - 1, bundles,
    )
    pg = ray_placement_group(bundles, strategy="PACK")
    await pg.ready()
    logger.info("PP placement group is ready")
    return pg


async def auto_scaler(
    auto_scaling_metrics: Dict[str, int], auto_scaling_config: Dict[str, int]
) -> int:
    """
    Returns desired number of instances for a model based on the auto-scaling policy
    """

    request_count = auto_scaling_metrics.get("request_count", 0)

    min_instances = auto_scaling_config.get("min_instances", 0)
    max_instances = auto_scaling_config.get("max_instances", 10)
    target_ongoing_requests = auto_scaling_config.get("target", 10)

    desired_instances = (
        request_count + target_ongoing_requests - 1
    ) // target_ongoing_requests
    desired_instances = min(
        max_instances, max(min_instances, desired_instances)
    )

    return desired_instances


def _remove_placement_group(instance: "InstanceHandle", instance_id: str) -> None:
    """Best-effort removal of a Ray PlacementGroup stored on *instance*."""
    pg = instance.placement_group
    if pg is None:
        return
    try:
        from ray.util.placement_group import remove_placement_group
        remove_placement_group(pg)
        instance.placement_group = None
        logger.info("Removed placement group for instance %s", instance_id)
    except Exception as exc:
        logger.warning(
            "Failed to remove placement group for instance %s: %s", instance_id, exc
        )


class RoundRobinRouter(SllmRouter):
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
        model_path: str,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.resource_requirements = resource_requirements
        self.backend = backend
        self.backend_config = backend_config

        self.loop_interval = 1
        self.loop = asyncio.get_running_loop()
        self.request_queue = asyncio.Queue()  # type:ignore
        self._pending_allocations: Set[asyncio.Future] = set()
        # Inference instance pools
        self.starting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.instance_management_lock = asyncio.Lock()
        # Serializes the check-and-create operation used when a cold request
        # replenishes the empty prewarm slot.  Without this lock, concurrent
        # requests can all observe zero empty instances and launch duplicates.
        self._empty_instance_lock = asyncio.Lock()
        self._pool_changed_event = asyncio.Event()
        self._startup_tasks: Set[asyncio.Task] = set()
        self._retirement_tasks: Set[asyncio.Task] = set()

        self.req_to_instance_id: Dict[str, str] = {}
        self._active_requests = set()
        self.instance_to_load_status: Dict[str, bool] = {}
        # Instance IDs currently in router-driven lazy weight load (see ``lazy_load_weights``).
        # Prevents concurrent duplicate ``lazy_load_weights`` on the same empty instance; pairing
        # with controller ``start_cold_start`` covers cross-request dedup at a higher layer.
        self._lazy_load_in_flight: Set[str] = set()
        self._cold_start_reservation: Optional[Dict[str, Any]] = None

        self._begin_lazy_init: Dict[str, bool] = {}
        self._lazy_init_done: Dict[str, bool] = {}
        self._instance_load_methods: Dict[str, str] = {}
        self._transition_requests: Dict[str, Tuple[str, str]] = {}
        self._lazy_init_events: Dict[str, asyncio.Event] = {}

        self.auto_scaling_config = {}
        self.auto_scaling_lock = asyncio.Lock()

        self.request_count = 0
        self.request_count_lock = asyncio.Lock()

        self.running = False
        self._closing = False
        self._router_start_task: Optional[asyncio.Task] = None
        self.running_lock = asyncio.Lock()

        self.idle_time = 0
        self.idle_time_lock = asyncio.Lock()

        self.auto_scaler = None
        self.load_balancer = None
        self.round_robin_index = 0
        self.cpu_instance_id: Optional[str] = None
        self.request_arrivals = deque()
        # Preferred node for prewarm GPU placement (co-located with the CPU instance).
        self.cpu_worker_id: Optional[str] = None

        self.load_window_seconds = int(
            self.backend_config.get("load_window_seconds", 120)
        )
        self.forecast_horizon_seconds = int(
            self.backend_config.get("forecast_horizon_seconds", 20)
        )
        self.predictive_prewarm_threshold = float(
            self.backend_config.get("predictive_prewarm_threshold", 0.85)
        )
        # No predictive scale/prewarm from load forecast until this many requests
        # arrived in the sliding window (avoids 1 cold-start from spawning extra empties).
        self.predictive_prewarm_min_arrivals = int(
            self.backend_config.get("predictive_prewarm_min_arrivals", 2)
        )
        self._hydraserve = bool(self.backend_config.get("hydraserve"))
        self._hydraserve_pp_size = int(self.backend_config.get("hydraserve_pp_size", 0) or 0)
        self.num_layers = int(self.backend_config.get("num_layers", 0) or 0)
        self._instance_backend_overrides: Dict[str, Dict[str, Any]] = {}
        self._hydraserve_pair_by_primary: Dict[str, str] = {}
        self._last_hydra_primary_id: Optional[str] = None
        self._serverlessllm = bool(self.backend_config.get("serverlessllm"))
        self._vllm = bool(self.backend_config.get("vllm"))

        self.scheduler = ray.get_actor("model_loading_scheduler")

        logger.info(f"Created new handler for model {self.model_name}")

    async def start(
        self,
        auto_scaling_config: Dict[str, int],
        cpu_worker_id: Optional[str] = None,
    ):
        if self.running:
            return
        if self._router_start_task is not None:
            raise RuntimeError(f"Router {self.model_name} is already starting")
        self._router_start_task = asyncio.current_task()
        self._closing = False
        try:
            await self._start(auto_scaling_config, cpu_worker_id)
        except BaseException:
            await self.shutdown()
            raise
        finally:
            self._router_start_task = None

    async def _start(self, auto_scaling_config, cpu_worker_id):
        policy = validate_auto_scaling_config(auto_scaling_config)
        self.model_loading_scheduler = ray.get_actor("model_loading_scheduler")
        self.cpu_worker_id = cpu_worker_id
        async with self.auto_scaling_lock:
            self.auto_scaling_config = policy
        prewarm = self.backend_config.get(
            "prewarm_gpu_instances", 1
        )
        hydra_prewarm = (
            self._hydraserve
            and self._hydraserve_pp_size >= 2
            and self.num_layers > 0
        )
        for _ in range(prewarm):
            if hydra_prewarm:
                await self._hydraserve_prewarm_one_pair(
                    preferred_node_id=cpu_worker_id
                )
            else:
                await self._create_instance(
                    empty_instance=True, preferred_node_id=cpu_worker_id
                )
        self.auto_scaler = asyncio.create_task(self._auto_scaler_loop())
        self.load_balancer = asyncio.create_task(self._load_balancer_loop())
        async with self.running_lock:
            self.running = True
        logger.info(f"Started handler for model {self.model_name}")

    async def update(
        self,
        auto_scaling_config: Optional[Dict[str, int]] = None,
    ):
        if auto_scaling_config is not None:
            policy = validate_auto_scaling_config(auto_scaling_config)
            async with self.auto_scaling_lock:
                self.auto_scaling_config = policy
            async with self.instance_management_lock:
                instances = list(self.ready_inference_instances.values()) + list(self.starting_inference_instances.values())
                for instance in instances:
                    instance.max_queue_length = instance_request_capacity(policy)
            logger.info(
                "Model %s auto scaling config updated to %s",
                self.model_name,
                auto_scaling_config,
            )

    def _new_instance_id(self):
        pattern = "{model_name}_{id}"
        return pattern.format(model_name=self.model_name, id=uuid.uuid4())

    def _track_task(self, task: asyncio.Task, tasks: Set[asyncio.Task]) -> None:
        tasks.add(task)

        def _done(completed: asyncio.Task) -> None:
            tasks.discard(completed)
            if not completed.cancelled():
                error = completed.exception()
                if error is not None:
                    logger.error("Instance lifecycle task failed for %s: %s", self.model_name, error)

        task.add_done_callback(_done)

    async def _create_instance(
        self,
        empty_instance: bool = False,
        preferred_node_id: Optional[str] = None,
        wait_for_ready: bool = False,
    ):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}"
        )
        # get max_queue_length from auto_scaling_config.
        # When metric == "concurrency", use the configured target; otherwise
        # allow a large default so the vLLM backend's continuous-batching can
        # be fully utilised (the real cap is vLLM's max_num_seqs).
        max_request_length = instance_request_capacity(self.auto_scaling_config)
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}, max queue length is {max_request_length}"
        )
        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=max_request_length,
            num_gpu=self.resource_requirements["num_gpus"],
            empty_instance=empty_instance,
        )
        async with self.instance_management_lock:
            if self._closing:
                raise RuntimeError(f"Router {self.model_name} is shutting down")
            instance.max_queue_length = instance_request_capacity(self.auto_scaling_config)
            self.starting_inference_instances[instance_id] = instance
        event = getattr(self, "_pool_changed_event", None)
        if event is not None:
            event.set()
        task = self.loop.create_task(
            self._start_instance(instance_id, preferred_node_id=preferred_node_id)
        )
        self._track_task(task, self._startup_tasks)
        if wait_for_ready:
            await asyncio.wait_for(task, timeout=float(self.backend_config.get("instance_start_timeout_s", 600)))

        return instance_id

    async def _start_instance(self, instance_id, preferred_node_id: Optional[str] = None):
        """Start one instance and always clean up a failed startup.

        Startup is launched as a background task for normal autoscaling.  An
        uncaught exception used to leave the handle in ``starting`` forever,
        which made both autoscaling and request admission wait indefinitely.
        """
        try:
            return await self._start_instance_impl(
                instance_id, preferred_node_id=preferred_node_id
            )
        except asyncio.CancelledError:
            await self._abort_starting_instance(instance_id)
            raise
        except Exception:
            logger.exception("Failed to start instance %s", instance_id)
            await self._abort_starting_instance(instance_id)
            raise

    async def _abort_starting_instance(self, instance_id: str) -> None:
        """Remove a failed startup and release any scheduler/backend resources."""
        async with self.instance_management_lock:
            instance = self.starting_inference_instances.pop(instance_id, None)
            ready = self.ready_inference_instances.pop(instance_id, None)
            deleting = self.deleting_inference_instances.pop(instance_id, None)
            instance = instance or ready or deleting
            self.instance_to_load_status.pop(instance_id, None)
        if instance is None:
            self._pool_changed_event.set()
            return
        try:
            await self._teardown_instance_for_shutdown(instance_id, instance)
        finally:
            self._pool_changed_event.set()

    async def _start_instance_impl(self, instance_id, preferred_node_id: Optional[str] = None):
        async with self.instance_management_lock:
            if instance_id not in self.starting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.starting_inference_instances[instance_id]
        # Now ask model loading scheduler to load the model
        logger.info(
            f"Allocating resources for model {self.model_name} on instance {instance_id}"
        )
        backend_config = dict(self.backend_config)
        backend_config.update(self._instance_backend_overrides.get(instance_id, {}))
        backend_config["pretrained_model_name_or_path"] = self.model_path
        if instance.empty_instance:
            backend_config["lazy_load"] = True
        else:
            backend_config["lazy_load"] = False

        scheduler_resources = dict(self.resource_requirements)
        scheduler_resources["empty_instance"] = instance.empty_instance

        # Derive pp_size / tp_size from (possibly per-instance) backend_config.
        total_gpus = int(self.resource_requirements.get("num_gpus", 1))
        pp_size = int(
            backend_config.get(
                "pipeline_parallel_size",
                self.resource_requirements.get("pipeline_parallel_size", 1),
            )
            or 1
        )
        tp_size_cfg = int(backend_config.get("tensor_parallel_size", 0) or 0)
        if tp_size_cfg > 0:
            scheduler_resources["tp_size"] = tp_size_cfg
        elif total_gpus > 1:
            # Infer per-stage TP from total GPU count divided by PP stages.
            scheduler_resources["tp_size"] = max(1, total_gpus // max(pp_size, 1))
        if pp_size > 1:
            scheduler_resources["pipeline_parallel_size"] = pp_size
        if instance.empty_instance:
            scheduler_resources["num_gpus"] = 0.001

        # Propagate the CPU-colocation hint so the scheduler can prefer the
        # same node when placing this (prewarm) GPU instance.
        if preferred_node_id is not None:
            scheduler_resources["preferred_node_id"] = preferred_node_id
            logger.info(
                "Instance %s: hinting scheduler to prefer node %s (CPU colocation)",
                instance_id, preferred_node_id,
            )

        if (
            getattr(instance, "hydraserve_role", None) == "pair"
            and instance.hydraserve_primary_id
        ):
            allocation_info = await self.model_loading_scheduler.adopt_hydraserve_pair_allocation.remote(
                model_name=self.model_name,
                pair_instance_id=instance_id,
                primary_instance_id=instance.hydraserve_primary_id,
                pair_pp_size=int(
                    backend_config.get("pipeline_parallel_size", 1) or 1
                ),
                pair_tp_size=int(
                    backend_config.get("tensor_parallel_size", 1) or 1
                ),
            )
            if not allocation_info:
                raise RuntimeError(
                    f"HydraServe pair allocation adopt failed for {instance_id}"
                )
        else:
            allocation_info = await self.model_loading_scheduler.allocate_resource.remote(
                model_name=self.model_name, instance_id=instance_id, resources=scheduler_resources
            )
        if allocation_info is None:
            raise RuntimeError(f"Scheduler returned no allocation for {instance_id}")

        # PP responses omit per-GPU ids for the public API; reload full allocation.
        if isinstance(allocation_info, dict) and int(
            allocation_info.get("pipeline_parallel_size", 1) or 1
        ) > 1:
            full_alloc = await self.model_loading_scheduler.get_instance_scheduler_allocation.remote(
                instance_id
            )
            if isinstance(full_alloc, dict) and full_alloc:
                allocation_info = {
                    **allocation_info,
                    "gpu_ids": list(full_alloc.get("gpu_ids", [])),
                    "allocations": copy.deepcopy(
                        list(full_alloc.get("allocations", []))
                    ),
                }

        # Decode what the scheduler actually allocated.
        if isinstance(allocation_info, dict):
            startup_node = allocation_info.get("node_id")
            gpu_ids = allocation_info.get("gpu_ids", [])
            actual_pp = int(allocation_info.get("pipeline_parallel_size", 1))
            actual_tp = int(allocation_info.get("tensor_parallel_size", 1))
            stage_allocations = allocation_info.get("allocations", [])
        else:
            startup_node = allocation_info
            gpu_ids = []
            actual_pp = 1
            actual_tp = 1
            stage_allocations = []

        backend_config["tensor_parallel_size"] = actual_tp
        backend_config["pipeline_parallel_size"] = actual_pp
        if actual_tp <= 0 or actual_pp <= 0:
            raise RuntimeError("Scheduler returned invalid TP/PP sizes")
        if actual_pp > 1 and (
            len(stage_allocations) != actual_pp
            or any(stage.get("stage_idx") != index or stage.get("node_id") is None
                   for index, stage in enumerate(stage_allocations))
        ):
            raise RuntimeError("Scheduler returned incomplete PP stage allocations")
        instance.allocation_info = {
            "tensor_parallel_size": actual_tp,
            "pipeline_parallel_size": actual_pp,
        }

        # Inject the resolved node_id into backend_config so that the GPU
        # backend actor knows which node it is on (used for cold-start tracking).
        if startup_node is not None:
            backend_config["node_id"] = startup_node

        # ------------------------------------------------------------------ #
        #  Pipeline-parallel path: build a Placement Group that pins each    #
        #  stage to its designated Ray node.                                  #
        # ------------------------------------------------------------------ #
        if actual_pp > 1 and stage_allocations:
            logger.info(
                "Instance %s is PP=%d TP=%d; building placement group for stages: %s",
                instance_id, actual_pp, actual_tp,
                [(s.get("stage_idx"), s.get("node_id"), s.get("address")) for s in stage_allocations],
            )
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            pg = await _build_pp_placement_group(
                stage_allocations,
                actual_tp,
                backend_config=backend_config,
            )
            if pg is None:
                raise RuntimeError(
                    f"Failed to build PP placement group for {instance_id}"
                )

            async with instance.lock:
                instance.placement_group = pg
                instance.node_id = startup_node

            # GPUBackend driver on bundle 0 (CPU bundle pinned to stage-0 ``gpu_worker_*`` via PG).
            # capture_child_tasks places vLLM workers in subsequent GPU bundles.
            pg_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )
            _pp_frac = float(self.backend_config.get("ray_worker_resource_fraction", 0.001))
            _, coordinator_resources = _pp_stage0_coordinator_bundle(
                stage_allocations[0], backend_config, _pp_frac
            )
            startup_config: Dict[str, Any] = {
                "num_cpus": 1,
                "num_gpus": 0,
                "scheduling_strategy": pg_strategy,
            }
            if coordinator_resources:
                startup_config["resources"] = coordinator_resources

            re_gpu = backend_config.get("gpu_backend_runtime_env")
            stage0_gpu_ids = []
            if stage_allocations:
                stage0_gpu_ids = [
                    int(x) for x in (stage_allocations[0].get("gpu_ids") or [])
                ]
            logger.info(f"Stage0 GPU IDs: {stage0_gpu_ids}")
            if isinstance(re_gpu, dict) and re_gpu:
                startup_config["runtime_env"] = re_gpu
            # PP: do not set ``CUDA_VISIBLE_DEVICES`` / ``_sllm_cuda_visible_devices`` here;
            # placement + vLLM handle visibility per stage.

            pp_backend_config = dict(backend_config)
            pp_backend_config["_sllm_ray_placement_group"] = pg
            startup_config["_sllm_pipeline_parallel_size"] = int(actual_pp)

            logger.info("Startup config (PP/PG): %s", startup_config)
            # Do not put the ``start_instance`` task on bundle 0 (PG): only the
            # GPUBackend actor may use that bundle, or Ray falls back the actor to
            # another node. Pin the *spawner* with the same logical resources as
            # ``startup_config["resources"]`` (``gpu_worker_*`` for stage 0) so the
            # task is charged against the GPU slot, like ``CpuModelRouter`` does with
            # ``options(resources=startup_resources)``.
            spawner_resources: Dict[str, float] = (
                dict(coordinator_resources) if coordinator_resources else {}
            )
            if not spawner_resources and stage_allocations:
                st0 = stage_allocations[0]
                spawner_resources = (
                    pp_placement_pin_resources(
                        st0,
                        dict(backend_config),
                        worker_resource_fraction=_pp_frac,
                        pin_mode="logical",
                    )
                    or {}
                )
            _si_kws: Dict[str, Any] = {
                "instance_id": instance_id,
                "backend": self.backend,
                "model_name": self.model_name,
                "backend_config": pp_backend_config,
                "startup_config": startup_config,
                "device": "gpu",
            }
            sp_opts = start_instance_spawner_options(
                spawner_resources if spawner_resources else None,
                startup_config,
                for_gpu=True,
            )
            if sp_opts:
                await await_backend_call(start_instance.options(**sp_opts).remote(**_si_kws))
            else:
                await await_backend_call(start_instance.remote(**_si_kws))

            logger.info("Started PP instance %s for model %s", instance_id, self.model_name)
            instance.backend_instance = ray.get_actor(instance_id)
            async with instance.lock:
                instance.ready = True
            try:
                await instance.backend_instance.init_backend.remote()
            finally:
                await self.model_loading_scheduler.refresh_cluster_gpu_vram.remote()

            async with self.instance_management_lock:
                self.ready_inference_instances[instance_id] = instance
                self.starting_inference_instances.pop(instance_id)
                lazy_load = backend_config.get("lazy_load", False)
                self.instance_to_load_status[instance_id] = not lazy_load
                ev = asyncio.Event()
                if not lazy_load:
                    ev.set()
                self._lazy_init_events[instance_id] = ev
                # Eager-loaded instances are immediately eligible for routing.
                self._lazy_init_done[instance_id] = not lazy_load
                self._begin_lazy_init[instance_id] = False
            self._pool_changed_event.set()

            return instance_id

        # ------------------------------------------------------------------ #
        #  TP-only (single-node) path: pin the GPU backend to ``gpu_worker_<node_id>``. #
        #  (CPU-side placement stays in :class:`~sllm.routers.cpu_router.CpuModelRouter`.)
        # ------------------------------------------------------------------ #
        _tp_frac = float(self.backend_config.get("ray_worker_resource_fraction", 0.001))
        startup_resources = ray_gpu_actor_placement_resources(
            None if startup_node is None else str(startup_node).strip() or None,
            backend_config,
            worker_resource_fraction=_tp_frac,
        )
        num_gpus = 0.001

        startup_config = {
            "resources": startup_resources,
            "num_gpus": num_gpus,
        }
        logger.info(f"Startup config: {startup_config}, {backend_config}")

        re_gpu = backend_config.get("gpu_backend_runtime_env")
        if gpu_ids or (isinstance(re_gpu, dict) and re_gpu):
            re_base: Optional[Dict[str, Any]] = None
            if gpu_ids:
                re_base = {
                    "env_vars": {
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
                    },
                }
            re_m = merge_runtime_env(
                re_base, re_gpu if isinstance(re_gpu, dict) else None
            )
            if re_m:
                startup_config["runtime_env"] = re_m
        if gpu_ids:
            logger.info(f"Assigned GPU group {gpu_ids} to instance {instance_id}")

        spawner_opts = start_instance_spawner_options(
            startup_resources, startup_config, for_gpu=True
        )
        await await_backend_call(start_instance.options(**spawner_opts).remote(
            instance_id=instance_id,
            backend=self.backend,
            model_name=self.model_name,
            backend_config=backend_config,
            startup_config=startup_config,
            device="gpu",
        ))
        logger.info(
            f"Started instance {instance_id} for model {self.model_name}"
        )
        instance.backend_instance = ray.get_actor(instance_id)
        async with instance.lock:
            instance.ready = True
            instance.node_id = startup_node
        try:
            await instance.backend_instance.init_backend.remote()
        finally:
            await self.model_loading_scheduler.refresh_cluster_gpu_vram.remote()

        async with self.instance_management_lock:
            self.ready_inference_instances[instance_id] = instance
            self.starting_inference_instances.pop(instance_id)
            lazy_load = backend_config.get("lazy_load", False)
            self.instance_to_load_status[instance_id] = (
                False if lazy_load else True
            )
            ev = asyncio.Event()
            if not lazy_load:
                ev.set()
            self._lazy_init_events[instance_id] = ev
            # Eager-loaded instances are immediately eligible for routing.
            self._lazy_init_done[instance_id] = not lazy_load
            self._begin_lazy_init[instance_id] = False
        event = getattr(self, "_pool_changed_event", None)
        if event is not None:
            event.set()
        return instance_id

    async def _await_instance_in_ready_pool(
        self, instance_id: str, timeout_s: Optional[float] = None
    ) -> None:
        """Block until *instance_id* is moved into ``ready_inference_instances`` (post-``init_backend``)."""
        if timeout_s is None:
            timeout_s = float(
                self.backend_config.get("register_ready_timeout_s", 600.0)
            )
        deadline = time.monotonic() + timeout_s
        while instance_id not in self.ready_inference_instances:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for instance {instance_id!r} to become "
                    f"ready (model={self.model_name!r})"
                )
            await asyncio.sleep(0.05)

    def _cleanup_lazy_init_state(self, instance_id: str):
        events = getattr(self, "_lazy_init_events", {})
        ev = events.get(instance_id, None)
        if ev is not None:
            ev.set()
        getattr(self, "_begin_lazy_init", {}).pop(instance_id, None)
        getattr(self, "_lazy_init_done", {}).pop(instance_id, None)
        self._instance_load_methods.pop(instance_id, None)
        events.pop(instance_id, None)
        self.instance_to_load_status.pop(instance_id, None)
        self._lazy_load_in_flight.discard(instance_id)
        for primary_id, pair_id in list(self._hydraserve_pair_by_primary.items()):
            if instance_id in (primary_id, pair_id):
                self._hydraserve_pair_by_primary.pop(primary_id, None)
        for request_id, mapped_id in list(self.req_to_instance_id.items()):
            if mapped_id == instance_id:
                self.req_to_instance_id.pop(request_id, None)

    def _build_deallocate_resources(self, instance: InstanceHandle) -> Dict:
        resources = dict(getattr(self, "resource_requirements", {}))
        if instance.empty_instance:
            resources["num_gpus"] = 0.001
            resources["tp_size"] = getattr(self, "resource_requirements", {}).get("num_gpus", 1)
        return resources

    async def _release_scheduler_allocation(
        self, instance_id: str, instance: InstanceHandle
    ) -> bool:
        """Release scheduler ownership with bounded retries.

        Backend teardown is intentionally independent from scheduler metadata.  A
        transient Ray failure must not leave a stale GPU allocation forever, so retry
        the idempotent scheduler operation before reporting a durable leak.
        """
        scheduler = getattr(self, "model_loading_scheduler", None)
        if scheduler is None:
            return True
        attempts = max(1, int(self.backend_config.get("deallocate_retry_attempts", 3)))
        timeout = max(0.1, float(self.backend_config.get("deallocate_timeout_s", 10.0)))
        resources = self._build_deallocate_resources(instance)
        async def release() -> bool:
            for attempt in range(attempts):
                try:
                    await asyncio.wait_for(
                        await_backend_call(scheduler.deallocate_resource.remote(
                            model_name=self.model_name,
                            instance_id=instance_id,
                            resources=resources,
                        )),
                        timeout=timeout,
                    )
                    return True
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if attempt + 1 < attempts:
                        await asyncio.sleep(min(0.25 * (2 ** attempt), 1.0))
                    else:
                        logger.error(
                            "Failed to release scheduler allocation for %s after %d attempts: %s",
                            instance_id, attempts, exc,
                        )
            return False

        task = asyncio.create_task(release())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Finish metadata release before allowing caller cancellation to unwind.
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def _predictive_load_stats(self) -> Dict[str, float]:
        now = time.time()
        while self.request_arrivals and self.request_arrivals[0] < now - self.load_window_seconds:
            self.request_arrivals.popleft()
        short_window = min(30, self.load_window_seconds)
        short_count = len([ts for ts in self.request_arrivals if ts >= now - short_window])
        long_count = len(self.request_arrivals)
        short_rate = short_count / max(float(short_window), 1.0)
        long_rate = long_count / max(float(self.load_window_seconds), 1.0)
        trend = short_rate - long_rate
        predicted_rate = max(0.0, short_rate + trend)
        predicted_requests = predicted_rate * self.forecast_horizon_seconds
        if long_count < self.predictive_prewarm_min_arrivals:
            predicted_requests = 0.0
        return {
            "short_rate": short_rate,
            "long_rate": long_rate,
            "trend": trend,
            "predicted_requests": predicted_requests,
        }


    async def _stop_instance(self, instance_id: Optional[str] = None):
        # A scale-down must never wait forever while all replicas are still
        # starting. The next autoscaler tick will retry once a replica is ready.
        async with self.instance_management_lock:
            if not self.ready_inference_instances and instance_id is None:
                return

        async with self.auto_scaling_lock:
            min_instances = int(self.auto_scaling_config.get("min_instances", 0))

        async with self.instance_management_lock:
            if instance_id is None:
                num_total = len(self.ready_inference_instances) + len(
                    self.starting_inference_instances
                )
                if num_total <= min_instances:
                    logger.info(
                        "Skip shrink for model %s: num_total=%s <= min_instances=%s",
                        self.model_name,
                        num_total,
                        min_instances,
                    )
                    return

                # Prefer reclaiming VRAM-heavy loaded replicas; unload cheap empty prewarm
                # slots only when still above min_instances after shrink attempts on loaded.
                loaded_ready = [
                    (i_id, inst)
                    for i_id, inst in self.ready_inference_instances.items()
                    if self.instance_to_load_status.get(i_id, False)
                    and self._instance_can_shrink(i_id, inst)
                ]
                if loaded_ready:
                    instance_id, instance = loaded_ready[-1]
                else:
                    unloaded_empty_ready = [
                        (i_id, inst)
                        for i_id, inst in self.ready_inference_instances.items()
                        if not self.instance_to_load_status.get(i_id, False)
                        and inst.empty_instance
                        and self._instance_can_shrink(i_id, inst)
                    ]
                    if not unloaded_empty_ready:
                        logger.info(
                            "Skip shrink for model %s: no loaded replica and no "
                            "unloaded empty/prewarm instance to reclaim",
                            self.model_name,
                        )
                        return
                    instance_id, instance = unloaded_empty_ready[-1]

            elif instance_id in self.ready_inference_instances:
                instance = self.ready_inference_instances[instance_id]
            else:
                logger.error(f"Instance {instance_id} not found")
                return
            async with instance.lock:
                if not self._instance_can_shrink(instance_id, instance):
                    return
                instance.status = False
            self.ready_inference_instances.pop(instance_id, None)
            self.deleting_inference_instances[instance_id] = instance
            self._pool_changed_event.set()
        logger.info(
            f"Stopping instance {instance_id} for model {self.model_name}"
        )
        self._track_task(
            self.loop.create_task(self._finish_instance(instance_id)), self._retirement_tasks
        )

    def _instance_can_shrink(self, instance_id: str, instance: InstanceHandle) -> bool:
        """Protect admitted requests and incomplete cold-start transitions."""
        return (
            instance.status
            and instance.concurrency == 0
            and instance_id not in self._lazy_load_in_flight
            and instance_id not in self.req_to_instance_id.values()
            and not (
                self._begin_lazy_init.get(instance_id, False)
                and not self.instance_to_load_status.get(instance_id, False)
            )
        )

    async def _finish_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.deleting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.deleting_inference_instances.pop(instance_id)
        async with instance.lock:
            instance.status = False

        try:
            if instance.backend_instance is not None:
                await close_backend_actor(instance.backend_instance, instance_id, self.backend_config, drain=True)
        finally:
            await self._release_scheduler_allocation(instance_id, instance)
            _remove_placement_group(instance, instance_id)
            getattr(self, "_instance_backend_overrides", {}).pop(instance_id, None)
            self._cleanup_lazy_init_state(instance_id)
            event = getattr(self, "_pool_changed_event", None)
            if event is not None:
                event.set()

    async def _shutdown_instance(self, instance_id: str):
        logger.info(
            f"Force deleting an instance (even if it is busy) for model {self.model_name}"
        )
        async with self.instance_management_lock:
            pool = self.ready_inference_instances
            if instance_id not in pool:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = pool.pop(instance_id)
            async with instance.lock:
                instance.status = False
        
        await self._teardown_instance_for_shutdown(instance_id, instance)
        self._pool_changed_event.set()

    async def targeted_lazy_init(
        self,
        instance_id: str,
        layer_idxes: List[List[int]],
        load_method: str = "tokenwise",
        request_id: Optional[str] = None,
        mark_weights_loaded: bool = True,
    ) -> None:
        """Run :meth:`lazy_init` on a specific instance (HydraServe primary / pair)."""
        async with self.idle_time_lock:
            self.idle_time = 0
        async with self.instance_management_lock:
            if instance_id not in self.ready_inference_instances:
                raise RuntimeError(f"targeted_lazy_init: instance {instance_id} not found")
            if self._begin_lazy_init.get(instance_id, False):
                raise RuntimeError(f"Instance {instance_id} has already begun lazy initialization")
            instance = self.ready_inference_instances[instance_id]
            self._begin_lazy_init[instance_id] = True
            self._instance_load_methods[instance_id] = load_method
        if request_id is not None:
            self.req_to_instance_id[request_id] = instance_id
        try:
            if instance.backend_instance is None:
                raise RuntimeError(f"Instance {instance_id} has no backend")
            await await_backend_call(instance.backend_instance.set_load_method.remote(method=load_method))
            await await_backend_call(instance.backend_instance.lazy_init.remote(
                layer_idxes=layer_idxes, request_id=request_id
            ))
            async with self.instance_management_lock:
                self._lazy_init_done[instance_id] = True
                event = self._lazy_init_events.get(instance_id)
                if event is not None:
                    event.set()
            if load_method == "tokenwise" and mark_weights_loaded:
                await self.notify_weights_loaded(instance_id)
        except BaseException:
            await self._abort_starting_instance(instance_id)
            raise
        finally:
            async with self.instance_management_lock:
                self._lazy_load_in_flight.discard(instance_id)

    def _instance_eligible_for_lazy_init(
        self, i_id: str, inst: InstanceHandle, mode: str
    ) -> bool:
        """Shared eligibility with :meth:`lazy_init` (status, in_flight, node, Hydra roles)."""
        if self.instance_to_load_status.get(i_id, False):
            return False
        if i_id in self._lazy_load_in_flight:
            return False
        if self._begin_lazy_init.get(i_id, False):
            return False
        if not inst.node_id:
            return False
        if getattr(inst, "hydraserve_role", None) == "pair":
            return False
        if self._hydraserve and getattr(inst, "hydraserve_role", None) == "primary":
            return False
        if mode == "serverlessllm":
            if not self._serverlessllm or self.num_layers <= 0:
                return False
            if not inst.empty_instance:
                return False
            if getattr(inst, "hydraserve_role", None) is not None:
                return False
        if mode == "vllm":
            if not self._vllm or self.num_layers <= 0:
                return False
            if not inst.empty_instance:
                return False
            if getattr(inst, "hydraserve_role", None) is not None:
                return False
        return True

    async def _reserve_hydraserve_pair_for_lazy_init(self) -> Tuple[str, str]:
        """Block until a (primary, pair) staging set is reserved under ``_lazy_load_in_flight``."""
        if not (self._hydraserve and self._hydraserve_pp_size >= 2 and self.num_layers > 0):
            raise RuntimeError("HydraServe not configured or num_layers not set")
        deadline = time.monotonic() + float(
            self.backend_config.get("lazy_load_wait_timeout_s", 600.0)
        )
        while True:
            got: Optional[Tuple[str, str]] = None
            async with self.instance_management_lock:
                for p, s in list(self._hydraserve_pair_by_primary.items()):
                    if p not in self.ready_inference_instances or s not in self.ready_inference_instances:
                        continue
                    if self.instance_to_load_status.get(p, False) or self.instance_to_load_status.get(
                        s, False
                    ):
                        continue
                    if p in self._lazy_load_in_flight or s in self._lazy_load_in_flight:
                        continue
                    p_inst = self.ready_inference_instances[p]
                    s_inst = self.ready_inference_instances[s]
                    if not p_inst.node_id or not s_inst.node_id:
                        continue
                    self._lazy_load_in_flight.add(p)
                    self._lazy_load_in_flight.add(s)
                    got = (p, s)
                    break
            if got is not None:
                return got[0], got[1]
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for HydraServe pair for model {self.model_name}"
                )
            logger.info(
                "No available HydraServe primary+pair for lazy_init, waiting..."
            )
            await asyncio.sleep(0.1)

    async def reserve_cold_start(self, request_id: str, reservation_id: str, mode: str = "default"):
        """Atomically choose the replica and node before controller admission."""
        target_instance = None
        async with self.instance_management_lock:
            if not self.running:
                return {"state": "stopped"}
            if (
                request_id in self._active_requests
                or request_id in self.req_to_instance_id
                or self._cold_start_reservation is not None
                or self._transition_requests
            ):
                return {"state": "busy"}
            if any(
                self._instance_load_methods.get(i) == "layerwise"
                and self._begin_lazy_init.get(i, False)
                and not self.instance_to_load_status.get(i, False)
                for i in self.ready_inference_instances
            ):
                # A new plan would mutate the shared CPU backend's active split.
                return {"state": "busy"}
            targets = []
            if mode == "hydraserve":
                for primary, pair in self._hydraserve_pair_by_primary.items():
                    ids = [primary, pair]
                    if all(
                        i in self.ready_inference_instances
                        and not self.instance_to_load_status.get(i, False)
                        and not self._begin_lazy_init.get(i, False)
                        and i not in self._lazy_load_in_flight
                        for i in ids
                    ):
                        targets = ids
                        break
            else:
                for i, inst in self.ready_inference_instances.items():
                    if self._instance_eligible_for_lazy_init(i, inst, mode):
                        targets = [i]
                        break
            if not targets:
                if not self.ready_inference_instances and not self.starting_inference_instances:
                    return {"state": "unavailable"}
                return {"state": "busy"}
            node_id = self.ready_inference_instances[targets[0]].node_id
            if node_id is None:
                return {"state": "busy"}
            target_instance = self.ready_inference_instances[targets[-1]]
            reservation = {
                "state": "reserved", "reservation_id": reservation_id,
                "request_id": request_id, "instance_id": targets[-1],
                "node_id": node_id, "instance_ids": targets,
                "parallel_layout": dict(self.ready_inference_instances[targets[-1]].allocation_info or {}),
                "slot_reserved": False,
            }
            self._cold_start_reservation = reservation
            self._lazy_load_in_flight.update(targets)
            self.req_to_instance_id[request_id] = targets[-1]
        # Never await instance.lock while holding instance_management_lock: startup
        # and teardown paths may acquire these locks in the opposite order.
        if target_instance is None or not await target_instance.add_requests(1):
            async with self.instance_management_lock:
                if self._cold_start_reservation is not None and self._cold_start_reservation.get("reservation_id") == reservation_id:
                    self._cold_start_reservation = None
                    self._lazy_load_in_flight.difference_update(targets)
                    if self.req_to_instance_id.get(request_id) == targets[-1]:
                        self.req_to_instance_id.pop(request_id, None)
                    self._pool_changed_event.set()
            return {"state": "busy"}
        rollback_slot = False
        async with self.instance_management_lock:
            if self._cold_start_reservation is None or self._cold_start_reservation.get("reservation_id") != reservation_id:
                rollback_slot = True
            else:
                self._cold_start_reservation["slot_reserved"] = True
                return dict(self._cold_start_reservation)
        if rollback_slot:
            await target_instance.add_requests(-1)
            return {"state": "busy"}

    async def release_cold_start(self, reservation_id: str):
        instance = None
        release_slot = False
        async with self.instance_management_lock:
            reservation = self._cold_start_reservation
            if reservation is None or reservation["reservation_id"] != reservation_id:
                return
            self._cold_start_reservation = None
            instance = self.ready_inference_instances.get(reservation["instance_id"])
            release_slot = instance is not None and reservation.get("slot_reserved", True)
            for i in reservation["instance_ids"]:
                self._lazy_load_in_flight.discard(i)
            request_id = reservation["request_id"]
            if self.req_to_instance_id.get(request_id) == reservation["instance_id"]:
                self.req_to_instance_id.pop(request_id, None)
            self._pool_changed_event.set()
        if release_slot:
            await instance.add_requests(-1)

    def _reserved_cold_start(self, reservation_id: str):
        reservation = self._cold_start_reservation
        if reservation is None or reservation["reservation_id"] != reservation_id:
            raise RuntimeError("Cold-start reservation no longer exists")
        return reservation

    async def lazy_init(
        self,
        layer_idxes: List[List[int]],
        load_method: str = "tokenwise",
        request_id: Optional[str] = None,
        mark_weights_loaded: bool = True,
        mode: str = "default",
        reservation_id: Optional[str] = None,
    ):
        async with self.idle_time_lock:
            self.idle_time = 0
        if mode in ["serverlessllm", "vllm"] and self.num_layers <= 0:
            raise RuntimeError(
                "lazy_init mode=%s requires num_layers > 0 in backend_config" % mode
            )

        # Pick an empty instance, reserve in ``_lazy_load_in_flight`` (same as standard SLLM).
        instance_id: Optional[str] = None
        if reservation_id is not None:
            async with self.instance_management_lock:
                reservation = self._reserved_cold_start(reservation_id)
                if reservation["request_id"] != request_id:
                    raise RuntimeError("Cold-start reservation request mismatch")
                instance_id = reservation["instance_id"]
        deadline = time.monotonic() + float(
            self.backend_config.get("lazy_load_wait_timeout_s", 600.0)
        )
        while instance_id is None:
            async with self.instance_management_lock:
                for i_id, inst in self.ready_inference_instances.items():
                    if not self._instance_eligible_for_lazy_init(i_id, inst, mode):
                        continue
                    self._lazy_load_in_flight.add(i_id)
                    instance_id = i_id
                    break
            if instance_id is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for an empty instance for model {self.model_name}"
                    )
                logger.info("No available instance to lazy-load weights, waiting...")
                await asyncio.sleep(0.1)

        async with self.instance_management_lock:
            if instance_id not in self.ready_inference_instances:
                raise RuntimeError(f"Reserved instance {instance_id} disappeared")
            if self._begin_lazy_init.get(instance_id, False):
                raise RuntimeError(f"Instance {instance_id} already began lazy initialization")
            instance = self.ready_inference_instances[instance_id]
            self._begin_lazy_init[instance_id] = True
            self._instance_load_methods[instance_id] = load_method
            if request_id is not None:
                self.req_to_instance_id[request_id] = instance_id
            logger.info(
                f"Lazy load weights request for model {self.model_name} (mode={mode})"
            )

        try:
            await await_backend_call(instance.backend_instance.set_load_method.remote(method=load_method))
            await await_backend_call(instance.backend_instance.lazy_init.remote(
                layer_idxes=layer_idxes, request_id=request_id
            ))
            async with self.instance_management_lock:
                self._lazy_init_events[instance_id].set()
                self._lazy_init_done[instance_id] = True
            if load_method == "tokenwise" and mark_weights_loaded:
                await self.notify_weights_loaded(instance_id)
        except BaseException:
            if request_id is not None:
                self.req_to_instance_id.pop(request_id, None)
            await self._abort_starting_instance(instance_id)
            raise
        finally:
            async with self.instance_management_lock:
                self._lazy_load_in_flight.discard(instance_id)

    async def hydraserve_parallel_cold_load(self, request_id: str, reservation_id: Optional[str] = None) -> None:
        """Primary and pair each load *all* layers in parallel (same in_flight discipline as :meth:`lazy_init`)."""
        if not self._hydraserve:
            raise RuntimeError("HydraServe not enabled")
        n = int(self.num_layers)
        if n <= 0:
            raise RuntimeError("HydraServe requires backend_config num_layers > 0")
        hpp = int(self._hydraserve_pp_size)
        layers_pair = layer_idxes_by_pp_stages(n, hpp)
        layers_primary = layer_idxes_by_pp_stages(n, 1)
        async with self.idle_time_lock:
            self.idle_time = 0
        if reservation_id is None:
            pid, sid = await self._reserve_hydraserve_pair_for_lazy_init()
        else:
            async with self.instance_management_lock:
                reservation = self._reserved_cold_start(reservation_id)
                if reservation["request_id"] != request_id:
                    raise RuntimeError("Cold-start reservation request mismatch")
                pid, sid = reservation["instance_ids"]
        tasks = []
        try:
            self.req_to_instance_id[request_id] = sid
            tasks = [
                asyncio.create_task(self.targeted_lazy_init(
                    sid,
                    layers_pair,
                    load_method="tokenwise",
                    request_id=request_id,
                    mark_weights_loaded=True,
                )),
                asyncio.create_task(self.targeted_lazy_init(
                    pid,
                    layers_primary,
                    load_method="tokenwise",
                    request_id=None,
                    mark_weights_loaded=True,
                )),
            ]
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.req_to_instance_id.pop(request_id, None)
            raise
        finally:
            async with self.instance_management_lock:
                self._lazy_load_in_flight.discard(pid)
                self._lazy_load_in_flight.discard(sid)

    async def hydraserve_retire_spike(self) -> None:
        """Destroy the pair instance after a HydraServe cold-start request."""
        for pid, sid in list(self._hydraserve_pair_by_primary.items()):
            if sid in self.ready_inference_instances:
                await self._shutdown_instance(sid)
            self._hydraserve_pair_by_primary.pop(pid, None)
            break

    async def _hydraserve_prewarm_one_pair(
        self, preferred_node_id: Optional[str] = None
    ) -> None:
        """Create one HydraServe (primary + pair) staging set.

        Primary: TP-only (``pipeline_parallel_size=1``), lazy/empty; becomes the
        steady-state inference instance after a cold start completes.

        Pair: PP with ``_hydraserve_pp_size`` stages, stage-0 co-located with the
        primary (same node + GPU IDs via :meth:`adopt_hydraserve_pair_allocation`);
        used only during cold-start parallel weight loading, then destroyed.

        Primary is fully started (blocking) before the pair is launched so that the
        scheduler's ``instance_allocations[primary_id]`` exists when
        ``adopt_hydraserve_pair_allocation`` queries it for the pair's stage-0.
        """
        if not (
            self._hydraserve
            and self._hydraserve_pp_size >= 2
            and self.num_layers > 0
        ):
            logger.warning(
                "_hydraserve_prewarm_one_pair: HydraServe not fully configured "
                "(hydraserve=%s, pp_size=%s, num_layers=%s); skipping",
                self._hydraserve,
                self._hydraserve_pp_size,
                self.num_layers,
            )
            return

        max_request_length = instance_request_capacity(self.auto_scaling_config)

        # ── Step 1: primary (TP-only, pp=1, lazy empty) ──────────────────────
        primary_id = self._new_instance_id()
        primary_inst = InstanceHandle(
            instance_id=primary_id,
            max_queue_length=max_request_length,
            num_gpu=self.resource_requirements["num_gpus"],
            empty_instance=True,
            hydraserve_role="primary",
        )
        # Ensure _start_instance treats this as a single-stage (TP-only) instance.
        self._instance_backend_overrides[primary_id] = {"pipeline_parallel_size": 1}
        async with self.instance_management_lock:
            primary_inst.max_queue_length = instance_request_capacity(self.auto_scaling_config)
            self.starting_inference_instances[primary_id] = primary_inst
            self._last_hydra_primary_id = primary_id
        logger.info(
            "HydraServe prewarm: starting primary instance %s for model %s "
            "(preferred_node=%s)",
            primary_id,
            self.model_name,
            preferred_node_id,
        )
        # Block until primary's init_backend completes so its scheduler allocation
        # entry is present before the pair calls adopt_hydraserve_pair_allocation.
        await self._start_instance(primary_id, preferred_node_id=preferred_node_id)

        # ── Step 2: pair (PP, pp=_hydraserve_pp_size, stage-0 = primary) ─────
        pair_id = self._new_instance_id()
        pair_inst = InstanceHandle(
            instance_id=pair_id,
            max_queue_length=max_request_length,
            num_gpu=self.resource_requirements["num_gpus"],
            empty_instance=True,
            hydraserve_role="pair",
            hydraserve_primary_id=primary_id,
        )
        self._instance_backend_overrides[pair_id] = {
            "pipeline_parallel_size": self._hydraserve_pp_size,
        }
        async with self.instance_management_lock:
            pair_inst.max_queue_length = instance_request_capacity(self.auto_scaling_config)
            self.starting_inference_instances[pair_id] = pair_inst
            # Register the mapping immediately so _reserve_hydraserve_pair_for_lazy_init
            # can track both while the pair is still starting up.
            self._hydraserve_pair_by_primary[primary_id] = pair_id
        logger.info(
            "HydraServe prewarm: starting pair instance %s "
            "(primary=%s, pp_size=%s) for model %s",
            pair_id,
            primary_id,
            self._hydraserve_pp_size,
            self.model_name,
        )
        task = self.loop.create_task(self._start_instance(pair_id, preferred_node_id=None))
        self._track_task(task, self._startup_tasks)

    async def hydraserve_ensure_prewarm_spike(self) -> bool:
        """Prewarm a full (primary + pair) staging set for the next cold start if missing."""
        if not (
            self._hydraserve
            and self._hydraserve_pp_size >= 2
            and self.num_layers > 0
        ):
            return False
        async with self._empty_instance_lock:
            async with self.instance_management_lock:
                for ep_id, sp_id in self._hydraserve_pair_by_primary.items():
                    if (
                        (ep_id in self.ready_inference_instances or ep_id in self.starting_inference_instances)
                        and (sp_id in self.ready_inference_instances or sp_id in self.starting_inference_instances)
                        and not self.instance_to_load_status.get(ep_id, False)
                        and not self.instance_to_load_status.get(sp_id, False)
                    ):
                        return False
            await self._hydraserve_prewarm_one_pair(self.cpu_worker_id)
            return True

    async def serverlessllm_ensure_prewarm(self) -> bool:
        """After a cold start, create one new empty lazy instance for the next cold."""
        if self.num_layers <= 0:
            return False
        return await self.ensure_one_empty_instance()

    async def notify_weights_loaded(self, instance_id: str):
        logger.info(f"Received weight loading notification for instance {instance_id}")
        async with self.instance_management_lock:
            if instance_id not in self.ready_inference_instances:
                return
            self.instance_to_load_status[instance_id] = True
            # Also mark lazy-init as done and unblock any waiters, so the
            # instance becomes eligible for load-balancing immediately.
            self._lazy_init_done[instance_id] = True
            ev = self._lazy_init_events.get(instance_id)
            if ev is not None:
                ev.set()
            logger.info(f"Lazy load weights request for model {self.model_name} completed")

    async def wait_for_lazy_init(self, instance_id: str):
        # Acquire lock only to read state and grab the event reference,
        # then release before awaiting — holding an asyncio.Lock across an
        # await would deadlock any coroutine that needs the same lock to set
        # the event (e.g. lazy_init / notify_weights_loaded).
        async with self.instance_management_lock:
            if instance_id not in self.ready_inference_instances:
                return False
            if self._lazy_init_done.get(instance_id, False):
                return True
            ev = self._lazy_init_events.get(instance_id, None)
        # Lock released — safe to await.
        if ev is None:
            return False
        try:
            await asyncio.wait_for(ev.wait(), timeout=float(self.backend_config.get("lazy_load_wait_timeout_s", 600)))
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for lazy init of instance {instance_id}")
            return False
        async with self.instance_management_lock:
            return instance_id in self.ready_inference_instances and self._lazy_init_done.get(instance_id, False)

    async def admit_layerwise_transition(self, request_id: str, admission_id: str):
        """Bind a new request to an initialized layerwise replica, without reloading."""
        async with self.instance_management_lock:
            if (
                not self.running
                or request_id in self._active_requests
                or request_id in self.req_to_instance_id
                or admission_id in self._transition_requests
                or self._cold_start_reservation is not None
                or any(entry[0] == request_id for entry in self._transition_requests.values())
            ):
                return None
            for instance_id, instance in self.ready_inference_instances.items():
                if (
                    self._instance_load_methods.get(instance_id) == "layerwise"
                    and self._lazy_init_done.get(instance_id, False)
                    and not self.instance_to_load_status.get(instance_id, False)
                    and getattr(instance, "hydraserve_role", None) is None
                ):
                    # Reserve before publishing ownership. InstanceHandle only
                    # takes its own lock here, never the router management lock.
                    # Cancellation while waiting therefore leaves no mapping,
                    # and teardown cannot remove the replica during admission.
                    if not await instance.add_requests(1):
                        continue
                    self.req_to_instance_id[request_id] = instance_id
                    self._transition_requests[admission_id] = (request_id, instance_id)
                    return instance_id
        return None

    async def has_cpu_handoff(self) -> bool:
        """Whether this router still depends on its shared CPU computing split."""
        async with self.instance_management_lock:
            return bool(self._transition_requests) or any(
                self._instance_load_methods.get(instance_id) == "layerwise"
                and self._begin_lazy_init.get(instance_id, False)
                and not self.instance_to_load_status.get(instance_id, False)
                for instance_id in self.ready_inference_instances
            )

    async def release_transition_request(self, admission_id: str):
        instance = None
        async with self.instance_management_lock:
            entry = self._transition_requests.pop(admission_id, None)
            if entry is None:
                return
            request_id, instance_id = entry
            instance = self.ready_inference_instances.get(instance_id)
            if self.req_to_instance_id.get(request_id) == instance_id:
                self.req_to_instance_id.pop(request_id, None)
        if instance is not None:
            await instance.add_requests(-1)

    async def get_load_status(self, request_id: str):
        instance_id = self.req_to_instance_id.get(request_id, None)
        if instance_id is None:
            return False
        async with self.instance_management_lock:
            return self.instance_to_load_status.get(instance_id, False)

    async def has_loaded_instance(self) -> bool:
        async with self.instance_management_lock:
            for instance_id in self.ready_inference_instances.keys():
                if self.instance_to_load_status.get(instance_id, False):
                    return True
        return False

    async def get_instance_pool_status(self) -> Dict[str, int]:
        """Return routing state for controller-side load decisions (GPU router).

        HydraServe note: a (primary, pair) set counts as **one** serving instance.
        Pair instances are excluded from ``loaded_ready`` so the controller does not
        double-count them; only the primary's GPU allocation is accounted for.
        """
        async with self.instance_management_lock:
            ready_instances = list(self.ready_inference_instances.values())
            starting_instances = list(self.starting_inference_instances.values())
            loaded_instances = [
                inst
                for inst in ready_instances
                if self.instance_to_load_status.get(inst.instance_id, False)
                # HydraServe: pair shares the primary's GPUs and is not an independent
                # serving unit — exclude it so primary+pair is counted as one instance.
                and not (
                    self._hydraserve
                    and getattr(inst, "hydraserve_role", None) == "pair"
                )
            ]
            empty_ready = [
                inst
                for inst in ready_instances
                if inst.empty_instance
                and not self.instance_to_load_status.get(inst.instance_id, False)
            ]
            empty_starting = [inst for inst in starting_instances if inst.empty_instance]

        loaded_available = 0
        for inst in loaded_instances:
            try:
                if await inst.check_request_queue():
                    loaded_available += 1
            except Exception:
                continue

        return {
            "loaded_ready": len(loaded_instances),
            "loaded_available": loaded_available,
            "empty_ready": len(empty_ready),
            "empty_starting": len(empty_starting),
        }

    async def ensure_one_empty_instance(self) -> bool:
        """Ensure at least one **unloaded** empty instance exists or is starting."""
        async with self._empty_instance_lock:
            status = await self.get_instance_pool_status()
            if status["empty_ready"] > 0 or status["empty_starting"] > 0:
                return False
            await self._create_instance(
                empty_instance=True, preferred_node_id=self.cpu_worker_id
            )
            return True

    async def get_instance(self):
        async with self.instance_management_lock:
            for instance_id in self.ready_inference_instances.keys():
                return self.ready_inference_instances[instance_id]
        return None

    async def inference(self, request_data: dict, action: str):
        request_data = dict(request_data)
        request_id = request_data.setdefault("request_id", str(uuid.uuid4()))
        if not isinstance(request_id, str) or not request_id:
            return {"error": "request_id must be a nonempty string"}
        if request_id in self._active_requests:
            return {"error": f"Request {request_id} is already active"}
        self._active_requests.add(request_id)
        try:
            return await self._inference(request_data, action)
        finally:
            self._active_requests.discard(request_id)

    async def _inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        self.request_arrivals.append(time.time())
        async with self.idle_time_lock:
            self.idle_time = 0
        async with self.request_count_lock:
            self.request_count += 1
        request_id = request_data.get("request_id", str(uuid.uuid4()))
        instance_id = self.req_to_instance_id.pop(request_id, None)
        mapped_instance = instance_id is not None
        # Controller admissions own their slots until both CPU/GPU calls are closed.
        cold_reservation = self._cold_start_reservation
        admission_reserved = (
            (request_id, instance_id) in self._transition_requests.values()
            or (
                cold_reservation is not None
                and cold_reservation["request_id"] == request_id
                and cold_reservation["instance_id"] == instance_id
            )
        )
        reserved = False
        instance = None
        try:
            if instance_id is None:
                instance_allocation = self.loop.create_future()
                self._pending_allocations.add(instance_allocation)
                await self.request_queue.put(instance_allocation)
                logger.info(f"Enqueued {action} request for model {self.model_name}")
                try:
                    instance_id = await asyncio.wait_for(
                        asyncio.shield(instance_allocation),
                        timeout=float(self.backend_config.get("request_queue_timeout_s", 600)),
                    )
                except (asyncio.CancelledError, TimeoutError) as exc:
                    # Cancellation can land after the balancer reserved a slot
                    # and resolved the future, but before this task resumed.
                    if not instance_allocation.cancelled() and instance_allocation.done():
                        allocated_id = instance_allocation.result()
                        # Read the event-loop-local map without suspending; slot
                        # cleanup must not wait for the allocation publisher's lock.
                        allocated_instance = self.ready_inference_instances.get(allocated_id)
                        if allocated_instance is not None:
                            await allocated_instance.add_requests(-1)
                    instance_allocation.cancel()
                    if isinstance(exc, TimeoutError):
                        return {"error": "Timed out waiting for GPU request capacity"}
                    raise
                finally:
                    self._pending_allocations.discard(instance_allocation)
                reserved = instance_id is not None

            if instance_id is None:
                return {"error": "Instance cancelled"}

            async with self.instance_management_lock:
                instance = self.ready_inference_instances.get(instance_id)
            if instance is None:
                logger.error(f"Allocated instance {instance_id} disappeared before inference")
                return {"error": "Instance not found"}

            if mapped_instance and not admission_reserved:
                reserved = await instance.add_requests(1)
                if not reserved:
                    return {"error": "Mapped instance has no request capacity"}

            result = await await_backend_call(instance.backend_instance.generate.remote(
                request_data=request_data
            ))
        except Exception as exc:
            logger.exception(
                "inference failed for request %s on instance %s: %s",
                request_id, instance_id, exc,
            )
            result = {"error": f"Inference error: {exc}"}
        finally:
            logger.info(f"Finished processing request")
            if reserved and instance is None:
                instance = self.ready_inference_instances.get(instance_id)
            if reserved and instance is not None:
                await instance.add_requests(-1)
            self.req_to_instance_id.pop(request_id, None)
            async with self.request_count_lock:
                self.request_count -= 1
        return result

    async def _teardown_instance_for_shutdown(
        self, instance_id: str, instance: InstanceHandle
    ) -> None:
        """Best-effort teardown for any instance handle (ready / starting / deleting)."""
        async with instance.lock:
            instance.status = False

        try:
            be = instance.backend_instance
            if be is not None:
                await close_backend_actor(be, instance_id, self.backend_config)
            elif ray.is_initialized():
                # Do not let teardown implicitly connect to or start a Ray
                # cluster.  A cancelled local/resident startup may never have
                # created an actor; calling ray.get_actor in that case can
                # block on an unavailable GCS and stall admission cleanup.
                try:
                    ray.kill(ray.get_actor(instance_id))
                except Exception:
                    pass
        finally:
            await self._release_scheduler_allocation(instance_id, instance)
            _remove_placement_group(instance, instance_id)
            self._instance_backend_overrides.pop(instance_id, None)
            self._cleanup_lazy_init_state(instance_id)
            self._pool_changed_event.set()

    async def shutdown(self):
        async with self.running_lock:
            self.running = False
            self._closing = True

        startup = self._router_start_task
        if startup is not None and startup is not asyncio.current_task():
            startup.cancel()
            await asyncio.gather(startup, return_exceptions=True)

        # Include the waiter already dequeued by the load balancer, not only
        # futures that are still visible in request_queue.
        for future in self._pending_allocations:
            if not future.done():
                future.set_result(None)

        for task in (self.auto_scaler, self.load_balancer):
            if task is not None and not task.done():
                task.cancel()
        background_tasks = [task for task in (self.auto_scaler, self.load_balancer) if task is not None]
        if background_tasks:
            results = await asyncio.gather(*background_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Router background task failed during shutdown: %s", result)

        startup_tasks = list(self._startup_tasks)
        for task in startup_tasks:
            if not task.done():
                task.cancel()
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)

        retirement_tasks = list(self._retirement_tasks)
        for task in retirement_tasks:
            task.cancel()
        if retirement_tasks:
            await asyncio.gather(*retirement_tasks, return_exceptions=True)

        while not self.request_queue.empty():
            try:
                fut = self.request_queue.get_nowait()
                fut.set_result(None)
            except Exception:
                break

        async with self.instance_management_lock:
            load_snapshot = dict(self.instance_to_load_status)
            all_items: List[Tuple[str, InstanceHandle]] = []
            for pool in (
                self.ready_inference_instances,
                self.starting_inference_instances,
                self.deleting_inference_instances,
            ):
                all_items.extend(list(pool.items()))
                pool.clear()
            self.instance_to_load_status.clear()
            self._lazy_load_in_flight.clear()
            self._cold_start_reservation = None
            self._transition_requests.clear()
            self.req_to_instance_id.clear()
            self._pool_changed_event.set()

        # Tear down instances that hold weights first (VRAM-heavy), then empties/prewarm.
        all_items.sort(
            key=lambda item: (
                0 if load_snapshot.get(item[0], False) else 1,
                item[0],
            )
        )

        await asyncio.gather(
            *[
                self._teardown_instance_for_shutdown(iid, inst)
                for iid, inst in all_items
            ],
            return_exceptions=True,
        )
        return [iid for iid, _ in all_items]

    def _instance_eligible_for_load_balancing(
        self, i_id: str, inst: InstanceHandle
    ) -> bool:
        """HydraServe: pair is cold-path only. Unloaded primary is staging only. Loaded primaries take steady traffic."""
        if not self._lazy_init_done.get(i_id, True):
            return False
        if inst.empty_instance and not self.instance_to_load_status.get(i_id, False):
            return False
        if not (
            self._hydraserve
            and self._hydraserve_pp_size >= 2
        ):
            return True
        role = getattr(inst, "hydraserve_role", None)
        if role == "pair":
            return False
        if role == "primary" and not self.instance_to_load_status.get(i_id, False):
            return False
        return True

    async def _load_balancer_loop(self):
        try:
            await self._run_load_balancer_loop()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Load balancer failed for model %s", self.model_name)
            async with self.running_lock:
                self.running = False
            for future in list(self._pending_allocations):
                if not future.done():
                    future.set_exception(RuntimeError("GPU load balancer failed"))
            self._pool_changed_event.set()

    async def _run_load_balancer_loop(self):
        while True:
            instance_allocation = await self.request_queue.get()
            allocated = False
            logger.info(f"A request is waiting for model {self.model_name}")
            while not allocated:
                if instance_allocation.done():
                    break
                # 1. get ready instances
                instance_options = None
                while not instance_options:
                    if instance_allocation.done():
                        allocated = True
                        break
                    async with self.running_lock:
                        running = self.running
                    if not running:
                        if not instance_allocation.done():
                            instance_allocation.set_result(None)
                        allocated = True
                        break
                    async with self.instance_management_lock:
                        instance_options = [
                            i
                            for i, h in self.ready_inference_instances.items()
                            if self._instance_eligible_for_load_balancing(i, h)
                        ]
                    if instance_options:
                        break
                    # Wake promptly on startup success/failure, while retaining
                    # a bounded poll for legacy callers that mutate the pools.
                    self._pool_changed_event.clear()
                    try:
                        await asyncio.wait_for(
                            self._pool_changed_event.wait(), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        pass
                    logger.info(f"{instance_options}")

                if allocated:
                    continue
                
                logger.info(f"Got ready instances {instance_options}")
                # Scan all eligible instances in round-robin order so a single
                # busy instance does not block allocation when others are free.
                #
                # IMPORTANT: never ``await`` while holding ``instance_management_lock``.
                # asyncio.Lock is not re-entrant; awaiting inside the lock lets other
                # tasks run that need the same lock (e.g. ``lazy_init`` finishing
                # and setting ``_lazy_init_done`` / the lazy-init event). That
                # deadlocks the first request: GPU cold start completes but the
                # router never signals readiness.
                n = len(instance_options)
                for offset in range(n):
                    candidate_id = instance_options[
                        (self.round_robin_index + offset) % n
                    ]
                    async with self.instance_management_lock:
                        if candidate_id not in self.ready_inference_instances:
                            continue
                        candidate = self.ready_inference_instances[candidate_id]
                    if await candidate.check_request_queue():
                        ok = await candidate.add_requests(1)
                        if ok:
                            # Do not suspend between taking the slot and handing it
                            # to the waiter; cancellation would orphan the slot.
                            self.round_robin_index = (
                                self.round_robin_index + offset + 1
                            ) % n
                            if instance_allocation.done():
                                await candidate.add_requests(-1)
                                allocated = True
                                break
                            try:
                                instance_allocation.set_result(candidate_id)
                            except asyncio.InvalidStateError:
                                await candidate.add_requests(-1)
                                allocated = True
                                break
                            allocated = True
                            break
                    else:
                        logger.info(
                            f"Instance {candidate_id} cannot add another request"
                        )
                if not allocated:
                    await asyncio.sleep(self.loop_interval)

    async def _auto_scaler_loop(self):
        while True:
            # logger.info(f"Auto-scaling for model {self.model_name}")
            async with self.auto_scaling_lock:
                auto_scaling_config = self.auto_scaling_config.copy()
            async with self.request_count_lock:
                request_count = self.request_count
            await self.model_loading_scheduler.report_model_load.remote(
                model_name=self.model_name,
                current_load=float(request_count),
            )
            auto_scaling_metrics = {"request_count": request_count}
            desired_instances = await auto_scaler(
                auto_scaling_metrics, auto_scaling_config
            )
            predictive_stats = await self._predictive_load_stats()
            target_per_instance = max(float(auto_scaling_config.get("target", 1)), 1.0)
            predicted_instances = int(
                (predictive_stats["predicted_requests"] + target_per_instance - 1)
                // target_per_instance
            )
            if predictive_stats["predicted_requests"] > 0:
                desired_instances = max(desired_instances, predicted_instances)
            max_instances = int(auto_scaling_config.get("max_instances", 10))
            desired_instances = min(desired_instances, max_instances)
            async with self.instance_management_lock:
                num_running_instances = len(
                    self.starting_inference_instances
                ) + len(self.ready_inference_instances)
            if desired_instances != num_running_instances:
                logger.info(
                    "%s: %s instances, need %s instances",
                    self.model_name,
                    num_running_instances,
                    desired_instances,
                )
            if desired_instances > num_running_instances:
                logger.info("Creating new instance")
                if (
                    self._hydraserve
                    and self._hydraserve_pp_size >= 2
                ):
                    await self._hydraserve_prewarm_one_pair(self.cpu_worker_id)
                else:
                    await self._create_instance()
            elif desired_instances < num_running_instances:
                keep_alive = auto_scaling_config.get("keep_alive", 0)
                if self.idle_time >= keep_alive:
                    logger.info(
                        f"Stopping instance, idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    await self._stop_instance()
                    async with self.idle_time_lock:
                        self.idle_time = 0
                else:
                    logger.info(
                        f"idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    async with self.idle_time_lock:
                        self.idle_time += self.loop_interval
            else:
                predicted_ratio = predictive_stats["predicted_requests"] / target_per_instance
                if (
                    predicted_ratio > num_running_instances * self.predictive_prewarm_threshold
                    and num_running_instances < max_instances
                ):
                    if (
                        self._hydraserve
                        and self._hydraserve_pp_size >= 2
                    ):
                        await self._hydraserve_prewarm_one_pair(
                            self.cpu_worker_id
                        )
                    else:
                        await self._create_instance(
                            empty_instance=True,
                            preferred_node_id=self.cpu_worker_id,
                        )
            await asyncio.sleep(self.loop_interval)
