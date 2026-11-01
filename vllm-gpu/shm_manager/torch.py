import json
import os
import time
import re
from typing import Dict, Optional, Union, List
import torch
from shm_manager._C import (
    save_tensors,
    get_pinned_shared_memory_ptrs,
    release_pinned_shared_memory,
    ModelManager,
    KVCacheManager,
    HiddenStatesManager,
)

class ModelInfo:
    def __init__(self):
        self.total_layers = -1
        self.state_dict = None
        self.manager = None
        self.cpu_memory_ptrs = None
        self.cpu_memory_sizes = {}
        self.per_layer_gpu_sizes = {}
        self.tensor_layer_map = {}

class KVCacheInfo:
    def __init__(self):
        self.total_layers = -1
        self.num_streams_per_device = 12
        self.manager = None
        self.shm_memory_sizes = {}
        self.gpu_caches_ptrs = None
        self.gpu_num_blocks = -1
        self.gpu_block_len = -1
        self.shm_caches_ptrs = None
        self.shm_num_blocks = -1
        self.shm_block_len = -1
        self.init_shm_kv_cache_info = False
        self.init_gpu_ptr = False


class HiddenStatesInfo:
    def __init__(self):
        self.manager = None
        self.gpu_hidden_states_ptr = None
        self.shm_hidden_states_ptr = None
        self.shm_memory_size = 0

model_info = ModelInfo()
kv_cache_info = KVCacheInfo()
hidden_states_info = HiddenStatesInfo()
total_layers = -1

def reset_model_info():
    """Detach weight sources after draining native loads.

    The resident controller must close admission and drain inference before
    calling this. It preserves producer SHM and any surviving weight tensors.
    KV/hidden-state managers have separate lifetimes and are not reset here.
    """
    global total_layers
    if model_info.manager is not None:
        model_info.manager.close()
    for rank, ptr in list((model_info.cpu_memory_ptrs or {}).items()):
        release_pinned_shared_memory(ptr, model_info.cpu_memory_sizes[rank])
        del model_info.cpu_memory_ptrs[rank]
    model_info.__init__()
    total_layers = -1

def reset_relay_info():
    """Release all relay sources after the controller drains inference.

    Dedicated native streams are fenced before any consumer mapping is
    detached. Communication groups and CPU producer segments are preserved.
    """
    if kv_cache_info.manager is not None:
        kv_cache_info.manager.close()
    if hidden_states_info.manager is not None:
        hidden_states_info.manager.close()
    for rank, ptr in list((kv_cache_info.shm_caches_ptrs or {}).items()):
        release_pinned_shared_memory(ptr, kv_cache_info.shm_memory_sizes[rank])
        del kv_cache_info.shm_caches_ptrs[rank]
    if hidden_states_info.shm_hidden_states_ptr is not None:
        release_pinned_shared_memory(hidden_states_info.shm_hidden_states_ptr,
                                     hidden_states_info.shm_memory_size)
        hidden_states_info.shm_hidden_states_ptr = None
    kv_cache_info.__init__()
    hidden_states_info.__init__()
    reset_model_info()

def _get_total_layers(tensor_index: dict) -> int:
    """从tensor_index中解析出模型的总层数。"""
    max_layer_idx = -1
    for name in tensor_index.keys():
        match = re.match(r"model\.layers\.(\d+)\.", name)
        if match:
            max_layer_idx = max(max_layer_idx, int(match.group(1)))
    return max_layer_idx    # lm_head与embedding视为一层

def _get_tensor_layer_index(
    name: str,
):
    total_layers = model_info.total_layers
    if name == "model.embed_tokens.weight":
        return 0
    if name == "model.norm.weight" or \
       name == "lm_head.weight":
        return total_layers  # 将最开始的embed_tokens视为独立的一层，在第一次调用load时就加载
    match = re.match(r"model\.layers\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Tensor {name} not found in tensor_index.")

def _get_tensor_layer_end( #用于判断是否是当前层的最后一个张量
    name: str,
    total_layers: int,
):
    match = re.match(r"model\.layers\.(\d+)\.", name)
    if "post_attention_layernorm" in name:
        return int(match.group(1))
    if "lm_head.weight" in name:
        return total_layers
    return -1

def _get_tensor_sharding_dim(
    name: str,
):
    if "norm" in name:
        return -1
    if "self_attn.o_proj" in name or "mlp.down_proj" in name:
        return 1
    return 0

def _get_tensor_is_packed(
    name: str,
):
    if "embed" in name or "norm" in name or "bias" in name:
        return False
    return True

def init_hidden_states_info(
    model_name: str,
    max_num_tokens: int,
    hidden_size: int,
    gpu_tp_rank: int,
    dtype: torch.dtype,
):
    global hidden_states_info
    hidden_states_info.manager = HiddenStatesManager(
        hidden_size,
        dtype.itemsize,
    )

    shm_name_map = {0: f"{model_name}_hidden_states"}
    shm_size_map = {0: max_num_tokens * hidden_size * dtype.itemsize * 2}
    # Retain the registration, not hidden-state contents or the GPU manager.
    # The native cache matches inode, mapping size and CUDA device, and uses
    # the same bounded idle budget as weights. A replaced producer cannot hit.
    shm_hidden_states_ptrs = get_pinned_shared_memory_ptrs(
        shm_name_map, shm_size_map, cacheable=True)
    hidden_states_info.shm_hidden_states_ptr = shm_hidden_states_ptrs[0]
    hidden_states_info.shm_memory_size = shm_size_map[0]
    hidden_states_info.manager.update_shm_ptr_info(
        shm_hidden_states_ptrs[0],
        max_num_tokens * hidden_size * dtype.itemsize,
    )

def load_hidden_states_to_gpu(
    num_input_tokens: int,
    gpu_hidden_states_ptr: int,
    gpu_residual_ptr: int,
    spans=None,
):
    global hidden_states_info
    hidden_states_info.manager.update_gpu_ptr_info(
        gpu_hidden_states_ptr,
        gpu_residual_ptr,
    )
    if spans is None:
        spans = [(0, 0, num_input_tokens)]
    hidden_states_info.manager.load_hidden_state_spans(spans)


def init_gpu_kv_cache_info(
    gpu_caches_ptrs: Dict[int, List[int]], #(layer_idx, [key_cache_ptr, value_cache_ptr])
    gpu_num_blocks: int,
    gpu_block_len: int,
    total_num_kv_heads: int,
):
    global kv_cache_info

    kv_cache_info.manager.update_gpu_ptr_info(
        gpu_num_blocks,
        gpu_block_len,
        total_num_kv_heads,
        gpu_caches_ptrs,
    )
    print(f"gpu_block_len: {gpu_block_len}, gpu_num_blocks: {gpu_num_blocks}, total_num_kv_heads: {total_num_kv_heads}")

def init_shm_kv_cache_info(
    model_name: str,
    gpu_tp_rank: int,
    gpu_tp_size: int,
    cpu_tp_size: int,
    shm_num_blocks: int,
    shm_block_len: int,
    shm_kv_cache_size: int,
    block_size: int,
    head_dim: int,
):
    global kv_cache_info
    global total_layers

    cpu_tp_ranks = []
    if gpu_tp_size < cpu_tp_size:
        ratio = cpu_tp_size // gpu_tp_size
        cpu_tp_ranks = list(range(gpu_tp_rank * ratio, (gpu_tp_rank + 1) * ratio))
    else:
        ratio = gpu_tp_size // cpu_tp_size
        cpu_tp_ranks = [gpu_tp_rank // ratio]

    shm_name_map = {rank: f"{model_name}_kv_cache_rank_{rank}" for rank in cpu_tp_ranks}
    shm_size_map = {rank: shm_kv_cache_size for rank in cpu_tp_ranks}
    shm_caches_ptrs = get_pinned_shared_memory_ptrs(
        shm_name_map, shm_size_map, cacheable=True)
    kv_cache_info.shm_caches_ptrs = shm_caches_ptrs
    kv_cache_info.shm_memory_sizes = shm_size_map

    device = torch.cuda.current_device()

    kv_cache_info.manager = KVCacheManager(
        # Weight loading uses an inclusive final layer index (the output
        # projection shares that bucket). KVCacheManager requires a count.
        total_layers + 1,
        gpu_tp_rank,
        gpu_tp_size,
        cpu_tp_size,
        shm_num_blocks,
        shm_block_len,
        block_size,
        head_dim,
        device,
    )

    kv_cache_info.manager.update_shm_ptr_info(
        cpu_tp_ranks,
        shm_caches_ptrs,
    )

def load_kv_cache_to_gpu(
    gpu_block_ids: List[int],
    cpu_block_ids: List[int],
    layer_ids: List[int],
):
    global kv_cache_info

    kv_cache_info.manager.load_kv_caches_async( 
        gpu_block_ids,
        cpu_block_ids,
        layer_ids,
    )

def load_kv_cache_to_gpu_sync(
):
    kv_cache_info.manager.synchronize()

def check_kv_cache_load_state(
):
    global kv_cache_info
    return kv_cache_info.manager.check_load_state()

def init_model_info(
    model_path: Union[str, os.PathLike],
    gpu_tp_rank: int,
    gpu_tp_size: int,
    cpu_tp_size: int,
    total_num_q_heads: int,
    total_num_kv_heads: int,
    head_dim: int,
    storage_path: Optional[str] = None,
):
    global model_info
    global total_layers

    print("gpu_tp_rank: ", gpu_tp_rank)
    print("gpu_tp_size: ", gpu_tp_size)
    print("cpu_tp_size: ", cpu_tp_size)

    cpu_tp_ranks = []
    if gpu_tp_size < cpu_tp_size:
        ratio = cpu_tp_size // gpu_tp_size
        cpu_tp_ranks = list(range(gpu_tp_rank * ratio, (gpu_tp_rank + 1) * ratio))
    else:
        ratio = gpu_tp_size // cpu_tp_size
        cpu_tp_ranks = [gpu_tp_rank // ratio]

    print(f"cpu_tp_ranks: {cpu_tp_ranks}")

    device = torch.cuda.current_device()
    print(f"device: {device}")

    model_info.manager = ModelManager(
        torch.tensor([], dtype=torch.bfloat16).element_size(),
        cpu_tp_size,
        gpu_tp_rank,
        gpu_tp_size,
        device,
    )

    if not storage_path:
        storage_path = os.getenv("STORAGE_PATH", "./models")
    manifest_path = os.path.join(storage_path, model_path, "shm_weights.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as stream:
            manifest = json.load(stream)
        if manifest.get("version") != 1 or manifest.get("cpu_tp_size") != cpu_tp_size:
            raise ValueError("Shared weight manifest version or CPU TP size mismatch")
        # A manifest can outlive its CPU producer. Reject missing/replaced SHM
        # before the native pinning path can attach to stale or incomplete data.
        if "segments" in manifest:
            for cpu_rank in cpu_tp_ranks:
                segment_name = os.path.basename(os.path.normpath(model_path)) + f"-rank_{cpu_rank}"
                try:
                    segment_stat = os.stat(os.path.join("/dev/shm", segment_name))
                except FileNotFoundError as exc:
                    raise ValueError("Shared weight producer is not ready: missing segment") from exc
                expected = manifest["segments"].get(str(cpu_rank), {})
                if (segment_stat.st_ino != expected.get("inode") or
                        segment_stat.st_size != expected.get("size")):
                    raise ValueError("Shared weight manifest refers to a replaced segment")
        tensor_index = manifest["tensor_index"]
    else:
        with open(os.path.join(storage_path, model_path, "tensor_index.json"), "r") as stream:
            tensor_index = json.load(stream)
    model_info.total_layers = _get_total_layers(tensor_index)
    model_info.manager.update_total_layers(model_info.total_layers)
    total_layers = model_info.total_layers

    tensor_meta_index = {}
    tensor_layer_map = {}
    cpu_memory = 0
    per_layer_gpu_memory = {}

    for name, (_, size, shape, stride, dtype) in tensor_index.items():
        sharding_dim = _get_tensor_sharding_dim(name)
        layer_index = _get_tensor_layer_index(name)
        layer_end = _get_tensor_layer_end(name, model_info.total_layers)
        is_packed = _get_tensor_is_packed(name)

        gpu_memory = per_layer_gpu_memory.get(layer_index, 0)
        gpu_shape = shape.copy()

        if "gate_up_proj" in name:
            cpu_size = size // cpu_tp_size
            gpu_size = size // gpu_tp_size
            gpu_shape[sharding_dim] = gpu_shape[sharding_dim] // gpu_tp_size
            tensor_meta_index[name] = (gpu_memory, gpu_shape, stride, dtype)
            tensor_layer_map[name] = layer_index

            shape[0] = shape[0] // 2
            gpu_half_shape = gpu_shape.copy()
            gpu_half_shape[0] = gpu_half_shape[0] // 2
            model_info.manager.generate_tensor_metadata(
                cpu_memory,
                gpu_memory,
                shape,
                gpu_half_shape,
                cpu_tp_ranks,
                sharding_dim,
                layer_index,
                -1,
                head_dim=-1,
                is_packed=is_packed,
                tensor_name=name
            )
            model_info.manager.generate_tensor_metadata(
                cpu_memory + cpu_size // 2,
                gpu_memory + gpu_size // 2,
                shape,
                gpu_half_shape,
                cpu_tp_ranks,
                sharding_dim,
                layer_index,
                layer_end,
                head_dim=-1,
                is_packed=is_packed,
                tensor_name=name
            )
            cpu_memory += cpu_size
            per_layer_gpu_memory[layer_index] = gpu_memory + gpu_size

        elif "self_attn.qkv_proj" in name:
            total_num_qkv_heads = total_num_q_heads + 2 * total_num_kv_heads
            total_q_heads_ratio = total_num_q_heads / total_num_qkv_heads
            total_kv_heads_ratio = total_num_kv_heads / total_num_qkv_heads

            per_gpu_num_q_heads = total_num_q_heads // gpu_tp_size
            per_gpu_num_kv_heads = max(1, total_num_kv_heads // gpu_tp_size)
            per_cpu_num_q_heads = total_num_q_heads // cpu_tp_size
            per_cpu_num_kv_heads = max(1, total_num_kv_heads // cpu_tp_size)

            gpu_q_heads_ratio = per_gpu_num_q_heads / total_num_qkv_heads
            gpu_kv_heads_ratio = per_gpu_num_kv_heads / total_num_qkv_heads
            gpu_q_shape = shape.copy()
            gpu_kv_shape = shape.copy()
            gpu_q_shape[0] = int(gpu_q_shape[0] * gpu_q_heads_ratio)
            gpu_kv_shape[0] = int(gpu_kv_shape[0] * gpu_kv_heads_ratio)
            gpu_q_size = int(size * gpu_q_heads_ratio)
            gpu_kv_size = int(size * gpu_kv_heads_ratio)

            cpu_q_heads_ratio = per_cpu_num_q_heads / total_num_qkv_heads
            cpu_kv_heads_ratio = per_cpu_num_kv_heads / total_num_qkv_heads
            cpu_q_size = int(size * cpu_q_heads_ratio)
            cpu_kv_size = int(size * cpu_kv_heads_ratio)

            q_shape = shape.copy()
            q_shape[0] = int(q_shape[0] * total_q_heads_ratio)
            kv_shape = shape.copy()
            kv_shape[0] = int(kv_shape[0] * total_kv_heads_ratio)
        
            gpu_shape[0] = int(gpu_q_shape[0] + 2 * gpu_kv_shape[0])
            tensor_meta_index[name] = (gpu_memory, gpu_shape, stride, dtype)
            tensor_layer_map[name] = layer_index

            is_1d = (len(shape) == 1)
            if is_1d:
                q_shape = [q_shape[0], 1]
                gpu_q_shape = [gpu_q_shape[0], 1]
                kv_shape = [kv_shape[0], 1]
                gpu_kv_shape = [gpu_kv_shape[0], 1]

            model_info.manager.generate_tensor_metadata(
                cpu_memory,
                gpu_memory,
                q_shape,
                gpu_q_shape,
                cpu_tp_ranks,
                sharding_dim,
                layer_index,
                -1,
                head_dim=-1,
                is_packed=is_packed,
                tensor_name=name
            )
            model_info.manager.generate_tensor_metadata(
                cpu_memory + cpu_q_size,
                gpu_memory + gpu_q_size,
                kv_shape,
                gpu_kv_shape,
                cpu_tp_ranks,
                sharding_dim,
                layer_index,
                -1,
                head_dim=head_dim if not is_1d else -1,
                is_packed=is_packed,
                tensor_name=name
            )
            model_info.manager.generate_tensor_metadata(
                cpu_memory + cpu_q_size + cpu_kv_size,
                gpu_memory + gpu_q_size + gpu_kv_size,
                kv_shape,
                gpu_kv_shape,
                cpu_tp_ranks,
                sharding_dim,
                layer_index,
                layer_end,
                head_dim=head_dim if not is_1d else -1,
                is_packed=is_packed,
                tensor_name=name
            )
            cpu_memory += cpu_q_size + 2 * cpu_kv_size
            per_layer_gpu_memory[layer_index] = gpu_memory + gpu_q_size + 2 * gpu_kv_size
            
        else:
            cpu_size = size
            gpu_size = size
            if sharding_dim != -1:
                cpu_size = cpu_size // cpu_tp_size
                gpu_size = gpu_size // gpu_tp_size
                gpu_shape[sharding_dim] = gpu_shape[sharding_dim] // gpu_tp_size
                if sharding_dim == 1:
                    stride[0] = stride[0] // gpu_tp_size
            tensor_meta_index[name] = (gpu_memory, gpu_shape, stride, dtype)
            tensor_layer_map[name] = layer_index

            model_info.manager.generate_tensor_metadata(
                cpu_memory,
                gpu_memory,
                shape,
                gpu_shape,
                cpu_tp_ranks,
                sharding_dim,
                layer_index,
                layer_end,
                head_dim=-1,
                is_packed=is_packed,
                tensor_name=name
            )
            cpu_memory += cpu_size
            per_layer_gpu_memory[layer_index] = gpu_memory + gpu_size

    model_info.tensor_meta_index = tensor_meta_index
    model_info.tensor_layer_map = tensor_layer_map
    model_info.per_layer_gpu_sizes = per_layer_gpu_memory

    cpu_all_memory = {i: cpu_memory for i in cpu_tp_ranks}
    # Zero preserves release-on-unload outside the resident pool.
    cache_budget = int(os.environ.get("VLLM_SHM_WEIGHT_CACHE_BYTES", "0"))
    if cache_budget < 0:
        raise ValueError("VLLM_SHM_WEIGHT_CACHE_BYTES must be nonnegative")
    from shm_manager._C import set_pinned_weight_cache_budget
    set_pinned_weight_cache_budget(cache_budget)
    cpu_all_name = {i: os.path.join(storage_path, model_path, f"rank_{i}") for i in cpu_tp_ranks}
    cpu_memory_ptrs = get_pinned_shared_memory_ptrs(cpu_all_name, cpu_all_memory, True)
    model_info.cpu_memory_ptrs = cpu_memory_ptrs
    model_info.cpu_memory_sizes = cpu_all_memory


def init_cuda_memory_dict(
    device_id: int,
    layer_idxes: list[int]
):
    global model_info
    selected = set(layer_idxes)
    layer_sizes = {
        layer: size for layer, size in model_info.per_layer_gpu_sizes.items()
        if layer in selected
    }
    profile = bool(os.environ.get('VLLM_PROFILE_RELAY'))
    started = time.perf_counter() if profile else 0.0
    bases = model_info.manager.allocate_cuda_arena(
        device_id, list(layer_sizes.values()))
    allocated = time.perf_counter() if profile else 0.0
    per_layer_gpu_ptrs = dict(zip(layer_sizes, bases))
    print(f"cuda total size: {sum(layer_sizes.values())}")

    model_info.manager.update_manager_info(
        model_info.cpu_memory_ptrs, per_layer_gpu_ptrs)
    manager_ready = time.perf_counter() if profile else 0.0
    planning_thread_start = time.thread_time() if profile else 0.0
    model_info.manager.generate_copy_tasks()
    planned = time.perf_counter() if profile else 0.0
    planning_thread_seconds = (time.thread_time() - planning_thread_start
                               if profile else 0.0)

    # Build the layer index once rather than rescan every tensor for each layer.
    metadata_by_layer = {layer: {} for layer in per_layer_gpu_ptrs}
    for name, meta in model_info.tensor_meta_index.items():
        layer = model_info.tensor_layer_map[name]
        if layer in metadata_by_layer:
            metadata_by_layer[layer][name] = meta
    state_dict = {}
    for layer_idx, base_ptr in per_layer_gpu_ptrs.items():
        layer_dict = model_info.manager.restore_tensors(
            metadata_by_layer[layer_idx], device_id, base_ptr)
        state_dict.update(layer_dict)
    
    model_info.per_layer_gpu_ptrs = per_layer_gpu_ptrs
    if profile:
        print(f'[relay weight storage] device={device_id} '
              f'arena={allocated - started:.6f}s '
              f'copy_metadata={planned - allocated:.6f}s '
              f'manager_setup={manager_ready - allocated:.6f}s '
              f'copy_plan={planned - manager_ready:.6f}s '
              f'copy_plan_thread_cpu={planning_thread_seconds:.6f}s '
              f'tensor_restore={time.perf_counter() - planned:.6f}s',
              flush=True)
    return state_dict


def load_model_to_gpu(
    layer_idxes: List[int],
):
    global model_info
    model_info.manager.load_layers_async(layer_idxes)


def load_model_to_gpu_blocking(layer_idxes: List[int]) -> None:
    """Load the given layers on the current GPU; return only after all copies and unpack complete."""
    global model_info
    model_info.manager.load_layers_sync(layer_idxes)


def load_model_to_gpu_sync(layer_idxes: List[int]):
    # global model_info
    # total_layers = model_info.total_layers

    # if end_layer < 0:
    #     end_layer = max(0, total_layers + end_layer)
    # elif end_layer >= total_layers - 1:
    #     end_layer = total_layers - 1

    model_info.manager.synchronize(layer_idxes)

def check_model_load_state(layer_idxes: List[int]):
    global model_info
    return model_info.manager.check_load_state(layer_idxes)

def update_module_loading_state(
    is_module_loading: bool,
):
    global model_info
    model_info.manager.update_module_loading_state(is_module_loading)

def update_model_arch_init_state(
    is_model_arch_init: bool,
    bound_submissions: bool = False,
):
    global model_info
    model_info.manager.update_model_arch_init_state(
        is_model_arch_init, bound_submissions)


def save_dict(
    state_dict: Dict[str, torch.Tensor], model_path: Union[str, os.PathLike]
):
    tensor_names = list(state_dict.keys())
    tensor_data_index = {}
    for name, param in state_dict.items():
        param_storage = param.untyped_storage()
        data_ptr = param_storage.data_ptr()
        size = param_storage.size()
        tensor_data_index[name] = (data_ptr, size)

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    # save tensors
    tensor_offsets = save_tensors(tensor_names, tensor_data_index, model_path)

    # create tensor index
    tensor_index = {}
    for name, param in state_dict.items():
        # name: offset, size
        tensor_index[name] = (
            tensor_offsets[name],
            tensor_data_index[name][1],
            tuple(param.shape),
            tuple(param.stride()),
            str(param.dtype),
        )

    # save tensor index
    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f)
