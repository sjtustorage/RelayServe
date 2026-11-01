# SPDX-License-Identifier: Apache-2.0
"""
Cold-start transition utilities for pipeline-parallel KV cache migration
and extension when transitioning from a non-uniform cold-start layer
distribution to a uniform steady-state distribution.

Supports two CPU computation modes during cold start:
  1. Layer-pipeline (layerwise): CPU computes first N layers, sends
     activations + KV cache to GPU0 via SHM.  Only PP0 pulls from SHM.
  2. Token-pipeline (tokenwise): CPU computes all layers for some tokens,
     sends all KV cache to GPU0.  GPU0 temporarily holds all layers' KV
     and distributes to other PP ranks via NCCL.

After cold start, a unified transition is triggered:
  - Each GPU loads missing weights for its uniform layer assignment
  - Once weights are ready, all GPUs synchronize
  - KV cache is migrated between PP ranks via NCCL send_tensor_dict
  - Each rank extends its KV cache (shrink extra layers, grow blocks)
"""

import re
import time
from typing import TYPE_CHECKING, List, Optional

import torch

from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tp_group,
)
from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.models.utils import extract_layer_index
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheTensor,
        KVCacheGroupSpec,
    )

logger = init_logger(__name__)


def compute_pp_layer_ranges(
    num_hidden_layers: int,
    pp_size: int,
    partition: Optional[List[List[int]]] = None,
) -> list[tuple[int, int]]:
    """Compute layer ranges for each PP rank.

    If *partition* is given as layer_idxes_by_pp
    (e.g. [[0,1,..,6], [7,..,14], [15,..,23], [24,..,35]]),
    derive (start, end) from each sub-list.
    Otherwise fall back to uniform distribution via get_pp_indices.
    """
    if partition is not None:
        if not isinstance(partition, (list, tuple)) or len(partition) != pp_size:
            raise ValueError('Cold partition must contain one layer list per PP stage')
        if any(not isinstance(layers, (list, tuple)) for layers in partition):
            raise ValueError('Cold partition stages must be layer lists')
        flattened = [layer for layers in partition for layer in layers]
        if (not flattened or any(type(layer) is not int or layer < 0
                                 or layer >= num_hidden_layers for layer in flattened)):
            raise ValueError('Cold partition contains no layers or invalid layer indices')
        if flattened != list(range(flattened[0], num_hidden_layers)):
            raise ValueError('Cold partition must cover a contiguous ordered model suffix exactly once')
        ranges = []
        cursor = flattened[0]
        for layers in partition:
            end = cursor + len(layers)
            ranges.append((cursor, end))
            cursor = end
        return ranges
    return [get_pp_indices(num_hidden_layers, r, pp_size) for r in range(pp_size)]


def compute_migration_plan(
    cold_ranges: list[tuple[int, int]],
    uniform_ranges: list[tuple[int, int]],
) -> dict[int, dict[str, dict[int, list[int]]]]:
    """Compute which layers each rank needs to send/receive.

    Returns:
        {rank: {"send": {dst_rank: [layer_ids]},
                "recv": {src_rank: [layer_ids]}}}
    """
    pp_size = len(cold_ranges)
    cold_map: dict[int, int] = {}
    for rank, (s, e) in enumerate(cold_ranges):
        for l in range(s, e):
            cold_map[l] = rank

    plan: dict[int, dict[str, dict[int, list[int]]]] = {
        r: {"send": {}, "recv": {}} for r in range(pp_size)
    }

    for rank, (s, e) in enumerate(uniform_ranges):
        for l in range(s, e):
            src = cold_map.get(l)
            if src is not None and src != rank:
                plan[src]["send"].setdefault(rank, []).append(l)
                plan[rank]["recv"].setdefault(src, []).append(l)

    return plan


def _build_layer_idx_to_name(kv_caches: dict[str, torch.Tensor],
                             ) -> dict[int, str]:
    """Build a mapping from global layer index to KV cache key name.

    Works regardless of the model's naming convention (e.g.
    ``model.layers.16.self_attn.attn`` or ``model.layers.16.self_attn``).
    """
    mapping: dict[int, str] = {}
    for name in kv_caches:
        idx = extract_layer_index(name)
        mapping[idx] = name
    return mapping


def migrate_kv_between_pp_ranks(
    kv_caches: dict[str, torch.Tensor],
    used_block_ids: list[int],
    migration_plan: dict[int, dict[str, dict[int, list[int]]]],
    pp_rank: int,
) -> None:
    """Execute KV cache migration using PP group NCCL send/recv.

    All ranks traverse the same ordered list of migration edges, including
    non-adjacent and ascending edges. Each pair does a synchronous transfer;
    one final barrier completes migration before topology changes.

    Args:
        kv_caches: layer_name -> KV tensor (reshaped, not raw)
        used_block_ids: block IDs that contain valid KV data
        migration_plan: complete plan returned by compute_migration_plan
        pp_rank: this rank's PP index
    """
    pp_group = get_pp_group()
    idx2name = _build_layer_idx_to_name(kv_caches)
    edges = [(src, dst, layers)
             for src in sorted(migration_plan, reverse=True)
             for dst, layers in sorted(migration_plan[src]["send"].items())
             if layers]
    # A bad endpoint must fail on all ranks before anyone blocks in recv.
    # PP uses one scheduler, hence active physical block IDs must agree.
    metadata = [None] * pp_group.world_size
    layouts = {idx: (tuple(kv_caches[name].shape), str(kv_caches[name].dtype))
               for idx, name in idx2name.items()}
    torch.distributed.all_gather_object(
        metadata, (migration_plan, used_block_ids, layouts),
        group=pp_group.cpu_group)
    for plan, blocks, _ in metadata:
        if plan != migration_plan or blocks != used_block_ids:
            raise ValueError("PP migration plans or active block IDs differ")
    for src, dst, layers in edges:
        if (not set(layers) <= metadata[src][2].keys()
                or not set(layers) <= metadata[dst][2].keys()):
            raise ValueError(f"Missing migration KV at endpoint {src} -> {dst}")
        for layer in layers:
            source_shape, source_dtype = metadata[src][2][layer]
            target_shape, target_dtype = metadata[dst][2][layer]
            if (len(source_shape) < 2 or len(target_shape) < 2
                    or source_shape[:1] + source_shape[2:]
                    != target_shape[:1] + target_shape[2:]
                    or source_dtype != target_dtype):
                raise ValueError(f"Incompatible migration KV at endpoint {src} -> {dst}")
            if any(type(block) is not int or block < 0
                   or block >= min(source_shape[1], target_shape[1])
                   for block in used_block_ids):
                raise ValueError(f"Invalid migration block at endpoint {src} -> {dst}")
    for sender_rank, receiver_rank, layers in edges:
        if pp_rank == sender_rank:
            send_dict = {}
            for layer_idx in layers:
                name = idx2name[layer_idx]
                send_dict[f"kv_{layer_idx}"] = \
                    kv_caches[name][:, used_block_ids].contiguous()
            pp_group.send_tensor_dict(send_dict, dst=receiver_rank)

        elif pp_rank == receiver_rank:
            recv_dict = pp_group.recv_tensor_dict(src=sender_rank)
            for layer_idx in layers:
                name = idx2name.get(layer_idx)
                key = f"kv_{layer_idx}"
                if key not in recv_dict or name is None:
                    raise ValueError(f"Missing migration KV for layer {layer_idx}")
                kv_caches[name][:, used_block_ids] = recv_dict[key]

    pp_group.barrier()



# NOTE: tokenwise KV distribution is handled by ShmConnectorWorker.start_load_kv(),
# which streams each non-owned layer from SHM through a single-layer relay
# buffer on GPU0, forwarding to target PP ranks one layer at a time via NCCL.


def resize_kv_cache_for_uniform(
    kv_caches: dict[str, torch.Tensor],
    kv_cache_raw: dict[str, torch.Tensor],
    kv_cache_config: "KVCacheConfig",
    uniform_layers: list[int],
    old_num_blocks: int,
    new_num_blocks: int,
    page_size_bytes: int,
    reshape_fn,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Remove extra layers and resize all kept layers to *new_num_blocks*.

    *new_num_blocks* is computed centrally by the engine core so that
    all PP ranks end up with the same block count.

    When block count changes, allocate one slab for all kept layers and
    copy logical blocks. Even shrinking requires logical copies: truncating
    raw bytes would move the V-cache base and corrupt K/V layouts.

    The caller must have cleared external references
    (``static_forward_context``, ``kv_caches`` list) **before** calling
    this function so that ``del`` actually frees CUDA memory.

    Returns:
        (updated kv_caches, updated kv_cache_raw)
    """
    idx2name = _build_layer_idx_to_name(kv_caches)
    missing = set(uniform_layers) - idx2name.keys()
    if missing:
        raise ValueError(f"Missing uniform KV layers: {sorted(missing)}")
    if new_num_blocks <= 0:
        raise ValueError("KV block count must be positive")
    uniform_names = {idx2name[l] for l in uniform_layers}

    extra_names = [n for n in list(kv_caches.keys()) if n not in uniform_names]
    keep_names = [n for n in list(kv_caches.keys()) if n in uniform_names]

    # Phase 1: free extra layers first to reclaim CUDA memory.
    if extra_names:
        for name in extra_names:
            kv_cache_raw.pop(name, None)
            del kv_caches[name]

    # Phase 2: resize kept layers (if block count changed).
    if new_num_blocks != old_num_blocks:
        new_per_layer_size = new_num_blocks * page_size_bytes
        # One cudaMalloc for all layers avoids N slow large allocations; slices
        # are views into the same storage (reshape_fn still sees correct numel).
        slab = (torch.empty(len(keep_names) * new_per_layer_size,
                            dtype=torch.int8,
                            device=kv_caches[keep_names[0]].device)
                if keep_names else None)
        for index, name in enumerate(keep_names):
            old_reshaped = kv_caches[name]
            new_raw = slab[index * new_per_layer_size:
                           (index + 1) * new_per_layer_size]
            new_reshaped = reshape_fn(name, new_raw)
            copy_blocks = min(old_num_blocks, new_num_blocks)
            new_reshaped[:, :copy_blocks] = old_reshaped[:, :copy_blocks]
            kv_cache_raw[name] = new_raw
            kv_caches[name] = new_reshaped
            del old_reshaped

    # Update config.
    new_per_layer_size = page_size_bytes * new_num_blocks
    from vllm.v1.kv_cache_interface import KVCacheTensor
    kv_cache_config.num_blocks = new_num_blocks
    kv_cache_config.kv_cache_tensors = [
        KVCacheTensor(size=new_per_layer_size, shared_by=[n])
        for n in keep_names
    ]
    for group in kv_cache_config.kv_cache_groups:
        group.layer_names = [n for n in group.layer_names
                             if n in uniform_names]

    return kv_caches, kv_cache_raw
