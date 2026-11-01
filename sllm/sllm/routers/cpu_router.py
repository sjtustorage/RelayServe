# ---------------------------------------------------------------------------- #
#  CPU-side model router (vLLM CPU: SHM KV probe for GPU alignment)            #
# ---------------------------------------------------------------------------- #
import asyncio
import uuid
from typing import Dict, Optional, Tuple

import ray

from sllm.logger import init_logger
from sllm.utils import InstanceHandle, ray_cpu_actor_placement_resources

from .router_utils import SllmRouter, await_backend_call, close_backend_actor, start_instance, start_instance_spawner_options

logger = init_logger(__name__)


class CpuModelRouter(SllmRouter):
    """One PP=1 CPU backend per model; expose its SHM KV layout to the GPU router."""

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

        self._cached_shm_kv_cache_info: Optional[Tuple[int, int, int, int]] = None
        self.loop = asyncio.get_running_loop()
        self.starting_inference_instances: Dict[str, InstanceHandle] = {}
        self.ready_inference_instances: Dict[str, InstanceHandle] = {}
        self.instance_management_lock = asyncio.Lock()

        self.auto_scaling_config: Dict[str, int] = {}
        self.auto_scaling_lock = asyncio.Lock()

        self.running = False
        self.running_lock = asyncio.Lock()
        self._startup_task: Optional[asyncio.Task] = None

        self.cpu_instance_id: Optional[str] = None
        self.model_loading_scheduler = None  # type: ignore

        logger.info("Created CPU handler for model %s", self.model_name)

    async def start(
        self,
        auto_scaling_config: Optional[Dict[str, int]] = None,
    ):
        if self.running:
            return
        if self._startup_task is not None:
            raise RuntimeError(f"CPU router {self.model_name} is already starting")
        self._startup_task = asyncio.current_task()
        try:
            await self._start()
        except BaseException:
            await self._shutdown_resources()
            raise
        finally:
            self._startup_task = None

    async def _start(self):
        async with self.auto_scaling_lock:
            self.auto_scaling_config = {}

        self.cpu_instance_id = await self._create_instance()

        # The backend and its SHM probe finish before _create_instance returns.
        if self._cached_shm_kv_cache_info is None:
            raise RuntimeError(f"CPU SHM KV cache info unavailable for {self.model_name}")
        async with self.running_lock:
            self.running = True
        logger.info("Started CPU handler for model %s", self.model_name)

    def _new_instance_id(self):
        pattern = "{model_name}_{id}"
        return pattern.format(model_name=self.model_name, id=uuid.uuid4())

    async def _create_instance(self) -> str:
        instance_id = self._new_instance_id()
        logger.info(
            "Creating new CPU instance %s for model %s placement_node_id=%s",
            instance_id,
            self.model_name,
            self.backend_config.get("cpu_placement_node_id"),
        )
        max_request_length = 5
        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=max_request_length,
            num_gpu=0,
            empty_instance=False,
        )
        async with self.instance_management_lock:
            self.starting_inference_instances[instance_id] = instance
        # Finish CPU initialization before the controller starts the GPU router.
        try:
            await self._start_instance(instance_id)
        except BaseException:
            await self._shutdown_instance(instance_id)
            raise
        return instance_id

    async def _start_instance(
        self,
        instance_id: str,
    ):
        async with self.instance_management_lock:
            if instance_id not in self.starting_inference_instances:
                logger.error("Instance %s not found", instance_id)
                return
            instance = self.starting_inference_instances[instance_id]

        backend_config = dict(self.backend_config)
        backend_config["pretrained_model_name_or_path"] = self.model_path
        backend_config.pop("gpu_pp_size", None)
        backend_config["pipeline_parallel_size"] = 1

        logger.info(
            "Allocating resources for model %s instance %s",
            self.model_name,
            instance_id,
        )
        startup_resources, placement_node_id = ray_cpu_actor_placement_resources(
            backend_config
        )
        num_gpus = 0
        runtime_env = backend_config.get("cpu_backend_runtime_env")
        startup_config = {
            "resources": startup_resources,
            "num_gpus": num_gpus,
            "runtime_env": runtime_env,
        }
        logger.info(
            "CPU startup config: %s placement_node_id=%s",
            startup_config,
            placement_node_id,
        )

        spawner_opts = start_instance_spawner_options(startup_resources, startup_config)
        await await_backend_call(start_instance.options(**spawner_opts).remote(
            instance_id=instance_id,
            backend=self.backend,
            model_name=self.model_name,
            backend_config=backend_config,
            startup_config=startup_config,
            device="cpu",
        ))
        logger.info(
            "Started CPU instance %s for model %s", instance_id, self.model_name
        )
        instance.backend_instance = ray.get_actor(instance_id)
        async with instance.lock:
            instance.ready = True
            instance.node_id = placement_node_id
        await instance.backend_instance.init_backend.remote()

        async with self.instance_management_lock:
            self.ready_inference_instances[instance_id] = instance
            self.starting_inference_instances.pop(instance_id)

        self.cpu_instance_id = instance_id
        shm_info = await instance.backend_instance.get_shm_kv_cache_info.remote()
        if isinstance(shm_info, (list, tuple)) and len(shm_info) == 4:
            self._cached_shm_kv_cache_info = tuple(int(value) for value in shm_info)
            logger.info("Cached CPU SHM KV cache info: %s", self._cached_shm_kv_cache_info)
        else:
            raise RuntimeError(f"CPU SHM KV cache info unavailable for {self.model_name}: {shm_info!r}")

    async def _shutdown_instance(self, instance_id: str):
        logger.info(
            "Force deleting CPU instance for model %s id=%s",
            self.model_name,
            instance_id,
        )
        async with self.instance_management_lock:
            instance = self.ready_inference_instances.pop(instance_id, None)
            starting = self.starting_inference_instances.pop(instance_id, None)
            instance = instance or starting
            if instance is None:
                logger.error("Instance %s not found", instance_id)
                return
        async with instance.lock:
            instance.status = False

        backend = instance.backend_instance
        if backend is None:
            try:
                backend = ray.get_actor(instance_id)
            except Exception:
                return
        await close_backend_actor(backend, instance_id, self.backend_config)

    async def get_cached_shm_kv_cache_info(
        self,
    ) -> Optional[Tuple[int, int, int, int]]:
        return self._cached_shm_kv_cache_info

    async def get_cpu_placement_node_id(self) -> Optional[str]:
        """Logical placement id of the CPU backend for GPU colocation."""
        instance = await self.get_instance()
        return str(instance.node_id) if instance is not None and instance.node_id is not None else None

    async def get_instance(self) -> Optional[InstanceHandle]:
        async with self.instance_management_lock:
            return self.ready_inference_instances.get(self.cpu_instance_id)

    async def inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        instance = await self.get_instance()
        if instance is None:
            return {"error": "CPU instance not ready"}

        payload_keys = sorted(
            k for k in request_data.keys() if k not in ("prompt", "messages")
        )
        logger.info(
            "CPU inference action=%s model=%s request_id=%s payload_keys=%s (node_id=%s)",
            action,
            self.model_name,
            request_data.get("request_id"),
            payload_keys,
            instance.node_id,
        )

        slot_taken = False
        try:
            if not await instance.add_requests(1):
                return {"error": "Instance busy"}
            slot_taken = True

            if action == "generate":
                result = await await_backend_call(instance.backend_instance.generate.remote(
                    request_data=request_data
                ))
            else:
                result = {"error": "Invalid action"}
            logger.info("Finished processing CPU request")
            return result
        finally:
            if slot_taken:
                await instance.add_requests(-1)

    async def shutdown(self):
        async with self.running_lock:
            self.running = False

        startup = self._startup_task
        if startup is not None and startup is not asyncio.current_task():
            startup.cancel()
            await asyncio.gather(startup, return_exceptions=True)
        return await self._shutdown_resources()

    async def _shutdown_resources(self):
        async with self.instance_management_lock:
            deleted_instance_id = list(
                set(self.ready_inference_instances) | set(self.starting_inference_instances)
            )
        delete_tasks = [
            self._shutdown_instance(instance_id) for instance_id in deleted_instance_id
        ]
        await asyncio.gather(*delete_tasks)
        self.cpu_instance_id = None
        self._cached_shm_kv_cache_info = None

        return deleted_instance_id
