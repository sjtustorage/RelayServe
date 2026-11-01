"""Prepare persistent CUDA registrations for existing CPU-owned weight SHM."""
import json
import os
from pathlib import Path
import time


def weight_sources(model, ranks=None):
    folder = Path(model)
    manifest = json.loads((folder / 'shm_weights.json').read_text())
    count = manifest.get('cpu_tp_size')
    if manifest.get('version') != 1 or type(count) is not int or count < 1:
        raise ValueError('Invalid weight source manifest')
    sizes = {}
    names = {}
    for rank in range(count) if ranks is None else ranks:
        expected = manifest.get('segments', {}).get(str(rank), {})
        source = Path('/dev/shm') / f'{folder.name}-rank_{rank}'
        info = source.stat()
        if (info.st_ino != expected.get('inode') or info.st_size != expected.get('size')
                or info.st_size <= 0):
            raise ValueError('Weight source was replaced or is incomplete')
        sizes[rank] = info.st_size
        names[rank] = str(folder / f'rank_{rank}')
    budget = int(os.environ.get('VLLM_SHM_WEIGHT_CACHE_BYTES', '0'))
    if budget < sum((size + 4095) // 4096 * 4096 for size in sizes.values()):
        raise ValueError('Weight registration preparation exceeds idle cache budget')
    return names, sizes, budget


def relay_sources(model, ranks, shm_info, chunk_tokens):
    """Validate CPU-owned KV/hidden sources without constructing GPU state."""
    if (not isinstance(shm_info, list) or len(shm_info) != 4 or
            any(type(value) is not int or value <= 0 for value in shm_info)):
        raise ValueError('Invalid relay SHM metadata')
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError('Invalid relay chunk size')
    folder = Path(model)
    manifest = json.loads((folder / 'shm_weights.json').read_text())
    if shm_info[3] != manifest['cpu_tp_size']:
        raise ValueError('Relay source CPU TP mismatch')
    hidden_size = json.loads((folder / 'config.json').read_text())['hidden_size']
    names = {rank: f'{folder.name}_kv_cache_rank_{rank}' for rank in ranks}
    sizes = {rank: shm_info[0] for rank in ranks}
    if ranks:
        key = shm_info[3]
        names[key] = f'{folder.name}_hidden_states'
        sizes[key] = chunk_tokens * hidden_size * 4  # Two BF16 buffers.
    for key, name in names.items():
        if (Path('/dev/shm') / name).stat().st_size < sizes[key]:
            raise ValueError('Relay source is incomplete')
    return names, sizes


def preparation_sources(model, ranks=None, *, shm_info=None, chunk_tokens=4096):
    """One source/budget plan shared by rank preflight and CUDA preparation."""
    names, sizes, budget = weight_sources(model, ranks)
    weight_bytes = sum(sizes.values())
    if shm_info is not None:
        relay_names, relay_sizes = relay_sources(model, list(names), shm_info, chunk_tokens)
        offset = max(names, default=-1) + 1
        for index, key in enumerate(relay_names):
            names[offset + index] = relay_names[key]
            sizes[offset + index] = relay_sizes[key]
        if budget < sum((size + 4095) // 4096 * 4096 for size in sizes.values()):
            raise ValueError('Relay registration preparation exceeds idle cache budget')
    return names, sizes, budget, weight_bytes


def prepare_weights(model, ranks=None, *, shm_info=None, chunk_tokens=4096):
    from shm_manager._C import (get_pinned_shared_memory_ptrs,
        release_pinned_shared_memory, set_pinned_weight_cache_budget,
        pinned_weight_cache_stats)
    names, sizes, budget, weight_bytes = preparation_sources(
        model, ranks, shm_info=shm_info, chunk_tokens=chunk_tokens)
    started = time.perf_counter()
    set_pinned_weight_cache_budget(budget)
    pointers = get_pinned_shared_memory_ptrs(names, sizes, True)
    # No transfers were submitted. Release consumer references immediately;
    # registrations remain in the existing bounded idle cache.
    for rank, address in pointers.items():
        release_pinned_shared_memory(address, sizes[rank])
    return dict(model=model, prepare_seconds=time.perf_counter() - started,
                weight_bytes=weight_bytes, relay_bytes=sum(sizes.values()) - weight_bytes,
                cache=pinned_weight_cache_stats())


def source_ranks(registry, model, gpu_tp):
    """Union of CPU shards this GPU may consume in prepared TP placements."""
    cpu_tp = json.loads((Path(model) / 'shm_weights.json').read_text())['cpu_tp_size']
    if type(cpu_tp) is not int or cpu_tp < 1:
        raise ValueError('Invalid CPU TP size in weight source manifest')
    if type(gpu_tp) is not int or gpu_tp < 1 or max(cpu_tp, gpu_tp) % min(cpu_tp, gpu_tp):
        raise ValueError('Invalid TP ratio for weight preparation')
    topologies = [item for item in registry._topologies if item.tp == gpu_tp]
    if not topologies:
        raise ValueError('Prepare communication topology before its weight sources')
    ranks = set()
    for item in topologies:
        if registry.rank not in item.ranks:
            continue
        # Every PP stage repeats the same TP shard positions. Source selection
        # depends on the stage-local TP rank, not the global pipeline position.
        gpu_rank = item.ranks.index(registry.rank) % gpu_tp
        if gpu_tp >= cpu_tp:
            ranks.add(gpu_rank // (gpu_tp // cpu_tp))
        else:
            ratio = cpu_tp // gpu_tp
            ranks.update(range(gpu_rank * ratio, (gpu_rank + 1) * ratio))
    return sorted(ranks)
