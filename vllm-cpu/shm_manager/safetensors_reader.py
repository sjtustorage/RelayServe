"""Read dense Qwen safetensors TP shards directly into final shared tensors."""
import json
import math
import os
from pathlib import Path
import struct
import torch

from shm_manager._C import read_file_regions


def load_qwen_regions(model_path, tensors, config, rank, tp_size, num_threads=4,
                      cpu_bind=""):
    """Validate every mapping before issuing any writes to model storage."""
    if tp_size < 1 or rank < 0 or rank >= tp_size:
        raise ValueError("Invalid CPU tensor parallel rank or size")
    if config.num_key_value_heads % tp_size:
        raise ValueError("Replicated CPU KV heads are not supported")
    for name, tensor in tensors.items():
        if (tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous() or tensor.numel() == 0):
            raise ValueError(f"{name}: expected nonempty contiguous CPU BF16 storage")
    index = {}
    root = Path(model_path)
    shard_index = root / "model.safetensors.index.json"
    files = (sorted(set(json.loads(shard_index.read_text())["weight_map"].values()))
             if shard_index.exists() else [p.name for p in sorted(root.glob("*.safetensors"))])
    for filename in files:
        path = root / filename
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Invalid safetensors header: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size > path.stat().st_size - 8:
                raise ValueError(f"Invalid safetensors header length: {path}")
            header = json.loads(stream.read(header_size))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            if (meta["dtype"] != "BF16" or start < 0 or end < start
                    or end - start != math.prod(meta["shape"]) * 2
                    or 8 + header_size + end > path.stat().st_size):
                raise ValueError(f"Invalid BF16 tensor metadata: {name}")
            if name in index:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            index[name] = (str(path), 8 + header_size + start, meta["shape"])
    regions = []
    padding = []
    for name, target in tensors.items():
        sources = [(name, target)]
        if "qkv_proj" in name:
            dims = [config.num_attention_heads * config.head_dim // tp_size,
                    config.num_key_value_heads * config.head_dim // tp_size]
            parts = target.split([dims[0], dims[1], dims[1]], dim=0)
            sources = [(name.replace("qkv_proj", suffix), part)
                       for suffix, part in zip(("q_proj", "k_proj", "v_proj"), parts)]
        elif "gate_up_proj" in name:
            sources = [(name.replace("gate_up_proj", suffix), part)
                       for suffix, part in zip(("gate_proj", "up_proj"), target.chunk(2, dim=0))]
        for source_name, destination in sources:
            if source_name not in index:
                raise ValueError(f"Missing checkpoint tensor: {source_name}")
            path, offset, shape = index[source_name]
            pointer = destination.data_ptr()
            if "norm" in name:
                if list(destination.shape) != shape:
                    raise ValueError(f"Replicated tensor shape mismatch: {name}")
                regions.append((path, offset, pointer, destination.numel() * 2))
            elif "o_proj" in name or "down_proj" in name:
                if len(shape) != 2 or shape != [destination.shape[0], destination.shape[1] * tp_size]:
                    raise ValueError(f"Row parallel tensor shape mismatch: {name}")
                row_bytes = destination.shape[1] * 2
                for row in range(shape[0]):
                    regions.append((path, offset + row * shape[1] * 2 + rank * row_bytes,
                                    pointer + row * row_bytes, row_bytes))
            else:
                rows = shape[0]
                # Vocab parameters can include padding allocated by vLLM.
                local_rows = destination.shape[0]
                if name.startswith(("model.embed_tokens.", "lm_head.")):
                    copied_rows = max(0, min(local_rows, rows - rank * local_rows))
                    if copied_rows < local_rows:
                        padding.append(destination[copied_rows:])
                else:
                    if rows != local_rows * tp_size:
                        raise ValueError(f"Column parallel tensor shape mismatch: {name}")
                    copied_rows = local_rows
                if list(destination.shape[1:]) != shape[1:]:
                    raise ValueError(f"Tensor trailing dimensions mismatch: {name}")
                row_bytes = math.prod(shape[1:]) * 2
                if copied_rows:
                    regions.append((path, offset + rank * local_rows * row_bytes,
                                    pointer, copied_rows * row_bytes))
    for destination in padding:
        destination.zero_()
    read_file_regions(regions, num_threads, cpu_bind)
