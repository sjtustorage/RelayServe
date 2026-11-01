# modified from ServerlessLLM

import torch
from torch import nn
import collections
import os
import gc
import time
import uuid
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List
from vllm.config import LoadConfig, VllmConfig, ModelConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model, set_default_torch_dtype)
from shm_manager.torch import (
    init_model_info,
    load_model_to_gpu,
    init_cuda_memory_dict,
    load_model_to_gpu_sync,
    save_dict,
    update_module_loading_state,
    update_model_arch_init_state,
    load_model_to_gpu_blocking,
    check_model_load_state,
)
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

from vllm.utils.relay_profile import (
    thread_runqueue_wait_seconds as _thread_runqueue_wait_seconds)


class SharedMemoryLoader(BaseModelLoader):
    # DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self._weight_dtype = torch.float32
        self._layer_idxes = None
        self._submitted_layers: set[int] = set()
        self.relay_weight_wait_events = None
        self.relay_weight_wait_seconds = 0.0
        self.profile_weight_wait = bool(os.environ.get("VLLM_PROFILE_WEIGHT_WAIT"))
        extra_config = ({} if load_config.model_loader_extra_config is None
                        else load_config.model_loader_extra_config.copy())
        # self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(f"Unexpected extra config keys for load format "
                             f"{load_config.load_format}: "
                             f"{load_config.model_loader_extra_config.keys()}")

    @staticmethod
    def _filter_subtensors(
            tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        # Group by UntypedStorage identity. Views into the same allocation share
        # one storage object, so id(storage) matches (device, storage.data_ptr())
        # without calling data_ptr() / touching device on every tensor.
        same_storage_groups = collections.defaultdict(list)
        for key, tensor in tensors.items():
            if tensor.numel():
                same_storage_groups[id(tensor.untyped_storage())].append(
                    (key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result = {}
        for group in same_storage_groups.values():
            meta = [
                (k, t, t.data_ptr(), get_end_ptr(t), t.is_contiguous())
                for k, t in group
            ]
            for k, t, a, b, t_contig in meta:
                for k2, t2, a2, b2, t2_contig in meta:
                    if not t2_contig:
                        continue
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t_contig:
                        break  # t2 covers strictly more memory than t.
                    if k2 > k:
                        # Same tensors, keep the one with the longer key.
                        break
                else:
                    result[k] = t
        return result

    def init_model_info(self, vllm_config: VllmConfig) -> None:
        model_config = vllm_config.model_config
        model_path = model_config.model
        rank = get_tensor_model_parallel_rank()
        gpu_tp_size = get_tensor_model_parallel_world_size()
        total_num_q_heads = model_config.get_total_num_q_heads()
        total_num_kv_heads = model_config.get_total_num_kv_heads()
        head_dim = model_config.get_head_size()


        init_model_info(
            model_path, rank, 
            gpu_tp_size, cpu_tp_size=vllm_config.parallel_config.shm_tp_size, 
            total_num_q_heads=total_num_q_heads, 
            total_num_kv_heads=total_num_kv_heads,
            head_dim=head_dim)

    def load_model(self, *, vllm_config: VllmConfig, model_config: ModelConfig, **kwargs) -> nn.Module:    
        assert os.path.isdir(model_config.model)

        layer_idxes = kwargs.get("layer_idxes", None)
        if layer_idxes is None:
            start, end = model_config.get_layers_start_end_indices(
                vllm_config.parallel_config)
            layer_idxes = list(range(start, end))
        # Storage and execution may cover different ranges during cold PP.
        # Validate priority before allocation or background submission.
        execution_layers = kwargs.get("execution_layer_idxes", layer_idxes)
        from vllm.v1.relay_step import layerwise_load_order
        priority_layers = (layerwise_load_order(layer_idxes, execution_layers)
                           if self.load_config.load_method == "layerwise"
                           else list(layer_idxes))
        self._layer_idxes = list(layer_idxes)
        self._weight_dtype = model_config.dtype

        local_model_path = model_config.model

        def remove_prefix(path, prefix):
            # Normalize the paths to ensure consistency across different platforms
            path = os.path.normpath(path)
            prefix = os.path.normpath(prefix)
            
            # Check if the path starts with the prefix
            if path.startswith(prefix):
                # Return the path without the prefix
                return path[len(prefix):].lstrip(os.sep)
            
            # Return the original path if the prefix doesn't exist
            return path
        
        # vLLM needs a local model path to read model config but
        # ServerlessLLM Store requires a global model path as the model ID
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = remove_prefix(local_model_path, storage_path)

        model_config.model_path = model_path
        load_async = self.load_config.load_async
        self.serial_copy_seconds = 0.0
        
        with set_default_torch_dtype(model_config.dtype):  
            print(f"load model {layer_idxes}, load_async: {load_async}")
            device_id = torch.cuda.current_device()
            if load_async:
                update_model_arch_init_state(
                    True, bound_submissions=self.load_config.load_method == "layerwise")

            def prepare_weights():
                # Native allocation/planning releases the GIL. Structure
                # construction does not consume these weights until binding.
                torch.cuda.set_device(device_id)
                started = time.perf_counter()
                tensors = init_cuda_memory_dict(device_id, layer_idxes)
                print(f"init model memory takes {time.perf_counter() - started} seconds")
                submission_started = time.perf_counter()
                if load_async:
                    load_model_to_gpu(priority_layers)
                    if os.environ.get('VLLM_PROFILE_RELAY'):
                        print(f'[relay weight submission] rank={get_tensor_model_parallel_rank()} '
                              f'at={submission_started:.9f}', flush=True)
                else:
                    load_model_to_gpu_blocking(priority_layers)
                    self.serial_copy_seconds = time.perf_counter() - submission_started
                    print(f"load model to gpu blocking takes "
                          f"{time.perf_counter() - submission_started} seconds")
                return tensors

            # Optional Python-only diagnostics: no CUDA events or stream fences,
            # so the weight transfer continues concurrently with construction.
            profile_dir = os.environ.get("VLLM_PROFILE_SHM_ARCH")
            profiler = None
            if profile_dir:
                import cProfile
                from pathlib import Path
                Path(profile_dir).mkdir(parents=True, exist_ok=True)
                profiler = cProfile.Profile()
            storage_executor = ThreadPoolExecutor(max_workers=1,
                                                   thread_name_prefix="shm-weights")
            start_time_init_architecture = time.perf_counter()
            architecture_thread_start = time.thread_time()
            architecture_runqueue_start = _thread_runqueue_wait_seconds()
            try:
                storage_future = storage_executor.submit(prepare_weights)
                if not load_async:
                    # Serial cold-start baseline: finish host-to-device copy
                    # before constructing or warming the model. A blocking
                    # copy in a background thread alone still overlaps both.
                    storage_future.result()
                if profiler is not None:
                    profiler.enable()
                # CUDA's current device does not change the default allocation
                # device. Avoid a DeviceContext when allocations already use
                # CPU: its TorchFunctionMode intercepts every construction op.
                construction_device = (nullcontext()
                                       if torch.get_default_device().type == "cpu"
                                       else torch.device("cpu"))
                from vllm.model_executor.layers.rotary_embedding.base import rotary_cache_device
                # Parameters are CPU metadata bound to preallocated GPU
                # storage. RoPE must use the same arithmetic as native GPU
                # construction, rather than CPU trig rounded to BF16.
                with construction_device, rotary_cache_device(f'cuda:{device_id}'):
                    model = initialize_model(vllm_config=vllm_config, model_config=model_config)
                    model = model.eval()
            finally:
                architecture_finished = time.perf_counter()
                architecture_thread_seconds = time.thread_time() - architecture_thread_start
                architecture_runqueue_end = _thread_runqueue_wait_seconds()
                if profiler is not None:
                    profiler.disable()
                storage_executor.shutdown(wait=True)
                if profiler is not None:
                    profiler.dump_stats(str(Path(profile_dir) /
                        f"architecture-pid-{os.getpid()}-{uuid.uuid4().hex}.prof"))
            end_time_init_architecture = architecture_finished
            sllm_state_dict = storage_future.result()
            self._submitted_layers = set(priority_layers)
            if os.environ.get('VLLM_PROFILE_RELAY'):
                print(f'[relay storage join] rank={get_tensor_model_parallel_rank()} '
                      f'wall={time.perf_counter() - architecture_finished:.6f}s', flush=True)
            print(f"init architecture takes {end_time_init_architecture - start_time_init_architecture} seconds")
            if os.environ.get('VLLM_PROFILE_RELAY'):
                print(f'[relay architecture] rank={get_tensor_model_parallel_rank()} '
                      f'wall={end_time_init_architecture - start_time_init_architecture:.6f}s '
                      f'host_thread_cpu={architecture_thread_seconds:.6f}s',
                      flush=True)
                if (architecture_runqueue_start is not None
                        and architecture_runqueue_end is not None):
                    print(f'[relay architecture scheduling] '
                          f'rank={get_tensor_model_parallel_rank()} '
                          f'runqueue_wait={architecture_runqueue_end - architecture_runqueue_start:.6f}s '
                          f'cpu_affinity={sorted(os.sched_getaffinity(0))}',
                          flush=True)

            t_sd = time.perf_counter()
            state_dict = model.state_dict()
            t_filter = time.perf_counter()
            # state_dict = self._filter_subtensors(state_dict)
            t_after = time.perf_counter()
            print(
                f"state_dict() takes {t_filter - t_sd} seconds, "
                f"filter subtensors takes {t_after - t_filter} seconds"
            )

            start_time_bind_tensors = time.perf_counter()
            for key, param in model.named_parameters(recurse=True):
                # Membership in the remaining state is constant time. Scanning
                # every key for every parameter made binding quadratic in the
                # number of layers and allocated an unnecessary second index.
                if key in state_dict:
                    tensor = sllm_state_dict[key]
                    param.data = tensor
                    state_dict.pop(key)
            if state_dict:
                raise ValueError(
                    f"Missing keys {tuple(state_dict)} in loaded state!")
            end_time_bind_tensors = time.perf_counter()
            print(f"bind tensors takes {end_time_bind_tensors - start_time_bind_tensors} seconds")
            
            start_time_bind_buffers = time.perf_counter()
            for _, buffer in model.named_buffers(recurse=True):
                if buffer.device.type != 'cuda':
                    buffer.data = buffer.data.to(f'cuda:{device_id}')
            end_time_bind_buffers = time.perf_counter()
            print(f"bind buffers takes {end_time_bind_buffers - start_time_bind_buffers} seconds")
            
            if load_async:
                update_model_arch_init_state(False)

            #load_model_to_gpu_sync(end_layer)
            # end_load_model_time = time.perf_counter()
            # print(f"load model takes {end_load_model_time - start_load_model_time} seconds")

        return model
    
    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def update_module_loading_state(self, is_module_loading: bool):
        update_module_loading_state(is_module_loading)

    def launch_load_weights(self, **kwargs):
        layer_idxes = kwargs.get("layer_idxes", None)
        if layer_idxes is None:
            if self._layer_idxes is None:
                raise RuntimeError("load_model must run before launching default weights")
            layer_idxes = list(self._layer_idxes)
        # The cold allocation already submits the full recovery range.
        # Recopying pending/completed storage races inference and wastes PCIe.
        submitted = self._submitted_layers
        layer_idxes = [layer for layer in dict.fromkeys(layer_idxes)
                       if layer not in submitted]
        if not layer_idxes:
            return
        with set_default_torch_dtype(self._weight_dtype):
            load_model_to_gpu(layer_idxes)
        submitted.update(layer_idxes)

    def wait_for_weights_loaded(self, layer_idxes: List[int]):
        events = self.relay_weight_wait_events
        measure_wait = events is not None or self.profile_weight_wait
        if measure_wait:
            started = time.perf_counter()
        # The readiness query already observes copy and unpack completion.
        # Do not enter the native fence again for a completed layer.
        if events is not None and check_model_load_state(layer_idxes):
            if measure_wait:
                self.relay_weight_wait_seconds += time.perf_counter() - started
            return
        record_gap = events is not None
        if record_gap:
            before = torch.cuda.Event(enable_timing=True)
            after = torch.cuda.Event(enable_timing=True)
            before.record()
        try:
            # Waiting creates no tensors. Changing the process-wide default
            # dtype here adds overhead per layer and races background loading.
            load_model_to_gpu_sync(layer_idxes)
        finally:
            if record_gap:
                # The preceding layer can run while this host waits. CUDA
                # timestamps capture only the resulting compute-stream gap,
                # rather than subtracting overlapping host wait time.
                after.record()
                events.append((before, after))
            if measure_wait:
                self.relay_weight_wait_seconds += time.perf_counter() - started
        
    def load_weights(self, model, **kwargs):
        pass

    @staticmethod
    def save_model(model: nn.Module, path: str) -> None:
        rank = get_tensor_model_parallel_rank()
        state_dict = SharedMemoryLoader._filter_subtensors(model.state_dict())
        
        # move all tensors to CPU
        for key, tensor in state_dict.items():
            state_dict[key] = tensor.cpu().contiguous()

        save_path = os.path.join(path, f"rank_{rank}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        save_dict(state_dict, save_path)
