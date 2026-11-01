import torch
from torch import nn
import collections
import os
import time
import json
import tempfile
import weakref
from dataclasses import replace
from typing import Dict
from vllm.config import LoadConfig, VllmConfig, ModelConfig
from vllm import envs
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model, set_default_torch_dtype, process_weights_after_loading)
from shm_manager.torch import load_model_to_cpu, save_dict, share_cpu_state_dict, _get_tensor_sharding_dim
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_tp_group
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

class SharedMemoryLoader(BaseModelLoader):
    # DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
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
        same_storage_groups = collections.defaultdict(list)
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 > k:
                        # Same tensors, keep the one with the longer key.
                        break
                else:
                    result[k] = t
        return result
        
    def load_model(self, *, vllm_config: VllmConfig, **kwargs) -> nn.Module:
        self._direct_shm_path = None
        try:
            return self._load_model(vllm_config=vllm_config, **kwargs)
        except BaseException:
            if self._direct_shm_path is not None and os.path.exists(self._direct_shm_path):
                os.unlink(self._direct_shm_path)
            raise

    def _load_model(self, *, vllm_config: VllmConfig, **kwargs) -> nn.Module:
        assert os.path.isdir(vllm_config.model_config.model)
        
        rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()

        model_config = vllm_config.model_config
        model_path = os.path.join(model_config.model, f"rank_{rank}")
        tensor_index_path = model_config.model

        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = device_config.device if load_config.device is None else \
                      load_config.device
        target_device = torch.device(load_device)
        direct = not os.path.isfile(os.path.join(model_path, "tensor.data_0"))
        if direct:
            if not envs.VLLM_CPU_USE_AMX_GEMM_BF16 or model_config.dtype != torch.bfloat16:
                raise ValueError("Direct shared GPU weights currently require BF16 AMX packing")
            if model_config.quantization is not None:
                raise ValueError("Direct shared weights require an unquantized checkpoint")
            if vllm_config.parallel_config.pipeline_parallel_size != 1:
                raise ValueError("Direct shared CPU weights require CPU PP=1")
            if model_config.get_total_num_kv_heads() % tp_world_size:
                raise ValueError("Direct shared weights do not yet support replicated CPU KV heads")
        
        with set_default_torch_dtype(model_config.dtype):
            # with torch.device(device_config.device):   
            with torch.device("cpu"):
                model = initialize_model(vllm_config=vllm_config, model_config=model_config)
                model = model.eval()
                
            # Note: storage path is already included in the local model path
            start_time = time.perf_counter()
            if direct:
                phase_start = time.perf_counter()
                native_config = replace(load_config, load_format="safetensors",
                                        model_loader_extra_config=None)
                source = self._filter_subtensors(model.state_dict())
                if any(tensor.dtype != torch.bfloat16 for tensor in source.values()):
                    raise ValueError("Shared GPU layout requires all weights to be BF16")
                sllm_state_dict, metadata, shm_path = share_cpu_state_dict(
                    source, model_path, alignment=1, copy=False)
                # Keep explicit ownership for shutdown and failure cleanup.
                model._shared_weight_path = shm_path
                self._direct_shm_path = shm_path
                del source
                print(f"shared weight arena allocated in {time.perf_counter() - phase_start:.3f}s")
            else:
                sllm_state_dict = load_model_to_cpu(model_path, tensor_index_path, tp_world_size, "256MB", 4)
            state_dict = self._filter_subtensors(model.state_dict())
            key_list = list(state_dict.keys())

            for key, param in model.named_parameters(recurse=True):
                if key in key_list:
                    tensor = sllm_state_dict[key]
                    param.data = tensor
                    state_dict.pop(key)
            if state_dict:
                raise ValueError(
                    f"Missing keys {tuple(state_dict)} in loaded state!")
            if direct:
                # Model-specific loaders now write fused TP shards straight
                # into their final shared storage, avoiding a full weight copy.
                read_start = time.perf_counter()
                hf_config = getattr(model_config, "hf_config", None)
                if (getattr(hf_config, "model_type", None) == "qwen3"
                        and not getattr(hf_config, "tie_word_embeddings", False)):
                    from shm_manager.safetensors_reader import load_qwen_regions
                    load_qwen_regions(model_config.model, sllm_state_dict,
                                      hf_config, rank, tp_world_size,
                                      cpu_bind=getattr(vllm_config,
                                                       "_cpu_weight_loader_bind", ""))
                else:
                    DefaultModelLoader(native_config).load_weights(model, model_config)
                print(f"weight file read takes {time.perf_counter() - read_start:.3f}s")
            end_time = time.perf_counter()
            print(f"load to cpu takes {end_time - start_time} seconds")
            postprocess_start = time.perf_counter()
            process_weights_after_loading(model, model_config, target_device)
            if direct:
                print(f"weight postprocessing takes {time.perf_counter() - postprocess_start:.3f}s")
            if direct:
                # Publishing readiness follows packing on every CPU TP rank.
                publish_start = time.perf_counter()
                get_tp_group().barrier()
                if rank == 0:
                    global_index = {}
                    for name, (offset, size, shape, stride, dtype) in metadata.items():
                        shape = list(shape)
                        dim = _get_tensor_sharding_dim(name)
                        if dim >= 0:
                            shape[dim] *= tp_world_size
                            size *= tp_world_size
                        stride = []
                        step = 1
                        for dimension in reversed(shape):
                            stride.insert(0, step)
                            step *= dimension
                        global_index[name] = (offset, size, shape, stride, dtype)
                    segments = {}
                    for cpu_rank in range(tp_world_size):
                        segment_name = os.path.basename(os.path.normpath(model_config.model)) + f"-rank_{cpu_rank}"
                        segment_stat = os.stat(os.path.join("/dev/shm", segment_name))
                        segments[str(cpu_rank)] = {"inode": segment_stat.st_ino,
                                                   "size": segment_stat.st_size}
                    manifest = {"version": 1, "cpu_tp_size": tp_world_size,
                                "segments": segments, "tensor_index": global_index}
                    fd, temporary = tempfile.mkstemp(dir=tensor_index_path, prefix=".shm-weights-")
                    try:
                        with os.fdopen(fd, "w") as stream:
                            json.dump(manifest, stream)
                        os.replace(temporary, os.path.join(tensor_index_path, "shm_weights.json"))
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
                weakref.finalize(model, self._unlink_shared_weights, shm_path,
                                 os.stat(shm_path).st_ino)
                print(f"shared weight publication takes {time.perf_counter() - publish_start:.3f}s")
                print(f"shared weight preparation total takes {time.perf_counter() - start_time:.3f}s")
        
        return model

    @staticmethod
    def _unlink_shared_weights(path, inode):
        try:
            if os.stat(path).st_ino == inode:
                os.unlink(path)
        except FileNotFoundError:
            pass
    
    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model, model_config):
        pass

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
    ) -> None:        
        rank = get_tensor_model_parallel_rank()
        state_dict = SharedMemoryLoader._filter_subtensors(model.state_dict())
        
        # move all tensors to CPU
        for key, tensor in state_dict.items():
            state_dict[key] = tensor.cpu().contiguous()

        save_path = os.path.join(path, f"rank_{rank}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        save_dict(state_dict, save_path)
