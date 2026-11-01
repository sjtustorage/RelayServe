# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
import json
import os
from typing import Dict, Union, List, Tuple
import re
import mmap
import torch

from shm_manager._C import (
    save_tensors,
    ModelManager,
    KVCacheManager,
    HiddenStatesManager
)


kvcache_manager = None
model_manager = None


def share_cpu_state_dict(state_dict, model_path, *, alignment=64, copy=True):
    """Publish already loaded CPU tensors without serializing weight files.

    The POSIX name matches the native GPU loader. Each returned tensor retains
    the mmap through torch.frombuffer, so deleting one tensor cannot unmap the
    storage of the others. The caller owns unlinking the returned name after
    consumers have detached. No metadata file is published by this operation.
    """
    tensors = {}
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    metadata = {}
    offset = 0
    for name, tensor in state_dict.items():
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError(f"{name}: expected a strided CPU tensor")
        if tensor.numel() == 0:
            raise ValueError(f"{name}: empty tensors cannot be shared")
        # All dtype views need an aligned byte offset, even with mixed dtypes.
        tensor_alignment = max(alignment, tensor.element_size())
        offset = (offset + tensor_alignment - 1) // tensor_alignment * tensor_alignment
        size = tensor.numel() * tensor.element_size()
        stride = []
        running_stride = 1
        for dimension in reversed(tensor.shape):
            stride.insert(0, running_stride)
            running_stride *= dimension
        metadata[name] = (offset, size, tuple(tensor.shape),
                          tuple(stride), str(tensor.dtype))
        offset += size
    if not metadata:
        raise ValueError("Cannot publish an empty state dict")
    path = os.path.normpath(os.fspath(model_path))
    shm_name = os.path.basename(os.path.dirname(path)) + "-" + os.path.basename(path)
    if shm_name in ("", "-") or "/" in shm_name:
        raise ValueError("model_path must identify a model and rank")
    shm_path = os.path.join("/dev/shm", shm_name)
    fd = os.open(shm_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, offset)
        arena = mmap.mmap(fd, offset, access=mmap.ACCESS_WRITE)
        for name, tensor in state_dict.items():
            byte_offset = metadata[name][0]
            shared = torch.frombuffer(arena, dtype=tensor.dtype,
                                      count=tensor.numel(), offset=byte_offset)
            shared = shared.reshape(tensor.shape)
            if copy:
                shared.copy_(tensor)
            tensors[name] = shared
    except BaseException:
        os.unlink(shm_path)
        raise
    finally:
        os.close(fd)
    return tensors, metadata, shm_path

def to_num_bytes(value: str) -> int:
    # Define the regular expression pattern for validation
    pattern = r"^(\d+)(B|KB|MB|GB|TB|PB|EB|ZB|YB)$"
    match = re.fullmatch(pattern, value)

    if not match:
        error_message = (
            "Invalid format. The input must be a positive integer "
            "followed immediately by a unit "
            "(B, KB, MB, GB, TB, PB, EB, ZB, YB), case sensitive, "
            "with no spaces or other characters."
        )
        raise ValueError(error_message)

    number_str, unit = match.groups()
    number = int(number_str)

    # Define the multiplier for each unit
    unit_multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
        "EB": 1024**6,
        "ZB": 1024**7,
        "YB": 1024**8,
    }

    bytes_value = number * unit_multipliers[unit]
    return bytes_value

def _get_tensor_sharding_dim(
    name: str,
):
    if "norm" in name:
        return -1
    if "self_attn.o_proj" in name or "mlp.down_proj" in name:
        return 1
    return 0

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

def load_model_to_cpu(
    model_path: Union[str, os.PathLike],
    tensor_index_path: Union[str, os.PathLike],
    cpu_world_size: int,
    chunk_size: str,
    num_threads: int,
):
    global model_manager
    with open(
        os.path.join(tensor_index_path, "tensor_index.json"), "r"
    ) as f:
        tensor_index = json.load(f)

    cpu_memory = 0
    tensor_meta_index = {}
    for name, (_, size, shape, stride, dtype) in tensor_index.items():
        sharding_dim = _get_tensor_sharding_dim(name)
        if sharding_dim != -1:
            shape[sharding_dim] = shape[sharding_dim] // cpu_world_size
            size = size // cpu_world_size
            if sharding_dim == 1:
                stride[0] = stride[0] // cpu_world_size

        tensor_meta_index[name] = (cpu_memory, shape, stride, dtype)
        cpu_memory += size

    model_manager = ModelManager(model_path, cpu_memory, to_num_bytes(chunk_size), num_threads)
    model_manager.load_model_to_host()
    return model_manager.restore_tensors(tensor_meta_index)

def destroy_shm_model():
    global model_manager
    if model_manager is None:
        return
    print("destroy shm model")
    model_manager.destroy_model()
    model_manager = None

def create_shm_kvcache(
    shm_name: str,
    total_size: int,
    tensor_meta_index: Dict[str, Tuple[int, List[int], List[int]]],
):
    global kvcache_manager
    kvcache_manager = KVCacheManager(shm_name, total_size)
    return kvcache_manager.restore_kvcache(tensor_meta_index)

def destroy_shm_kvcache():
    global kvcache_manager
    if kvcache_manager is None:
        return
    print("destroy shm kvcache")
    kvcache_manager.destroy_kvcache()
    kvcache_manager = None

def create_shm_hidden_states(
    shm_name: str,
    total_size: int,
    tensor_meta_index: Dict[str, Tuple[int, List[int], List[int]]],
):
    global hidden_states_manager
    hidden_states_manager = HiddenStatesManager(shm_name, total_size)
    return hidden_states_manager.restore_hidden_states(tensor_meta_index)

def destroy_shm_hidden_states():
    global hidden_states_manager
    if hidden_states_manager is None:
        return
    print("destroy shm hidden states")
    hidden_states_manager.destroy_hidden_states()
    hidden_states_manager = None


