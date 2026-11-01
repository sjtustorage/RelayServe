"""Resident initialization validates groups before touching CUDA state."""
from types import SimpleNamespace

import pytest

from vllm.v1.worker import gpu_worker as module


@pytest.mark.parametrize('actual_tp,actual_pp', [(1, 1), (4, 1), (2, 2)])
def test_prepared_group_mismatch(monkeypatch, actual_tp, actual_pp):
    monkeypatch.setattr(module, 'get_tp_group', lambda: SimpleNamespace(world_size=actual_tp))
    monkeypatch.setattr(module, 'get_pp_group', lambda: SimpleNamespace(world_size=actual_pp))
    calls = []
    worker = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1),
        _init_device=lambda **kw: calls.append(kw))
    with pytest.raises(ValueError, match='Prepared groups'):
        module.Worker.init_device_with_prepared_groups(worker)
    assert not calls


def test_prepared_groups_use_resident_path(monkeypatch):
    monkeypatch.setattr(module, 'get_tp_group', lambda: SimpleNamespace(world_size=2))
    monkeypatch.setattr(module, 'get_pp_group', lambda: SimpleNamespace(world_size=1))
    calls = []
    worker = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1),
        _init_device=lambda **kw: calls.append(kw))
    module.Worker.init_device_with_prepared_groups(worker)
    assert calls == [dict(reuse_prepared_groups=True)]


@pytest.mark.parametrize('resident,load_format,allocated,accepted,flushed', [
    (True, 'shm', 10, True, False),
    (True, 'shm', 45, True, False),
    (False, 'shm', 10, False, False),
    (True, 'safetensors', 10, True, True),
])
def test_resident_startup_counts_only_reusable_allocator_memory(
        monkeypatch, resident, load_format, allocated, accepted, flushed):
    # 30 device-free + (50 reserved - 10 live) permits a 60-byte budget.
    # Live allocations must never be credited as available. Conventional
    # profiling must release the cache before taking its initial snapshot.
    memory = {'free': 30, 'flushes': 0}

    def empty_cache():
        memory['flushes'] += 1
        memory['free'] = 80

    class PassedMemoryCheck(Exception):
        pass

    def reached_collectives(_config):
        raise PassedMemoryCheck

    monkeypatch.setattr(module, 'current_platform', SimpleNamespace(
        set_device=lambda device: None,
        check_if_supports_dtype=lambda dtype: None))
    monkeypatch.setattr(module, 'MemorySnapshot', lambda: SimpleNamespace(
        total_memory=100, free_memory=memory['free']))
    monkeypatch.setattr(module.torch.cuda, 'empty_cache', empty_cache)
    monkeypatch.setattr(module.torch.cuda, 'memory_reserved', lambda device: 50)
    monkeypatch.setattr(module.torch.cuda, 'memory_allocated', lambda device: allocated)
    monkeypatch.setattr(module, 'ensure_kv_transfer_initialized', reached_collectives)
    worker = SimpleNamespace(
        device_config=SimpleNamespace(device=SimpleNamespace(type='cuda')),
        local_rank=0, model_config=SimpleNamespace(dtype=module.torch.bfloat16),
        load_config=SimpleNamespace(load_format=load_format),
        cache_config=SimpleNamespace(gpu_memory_utilization=0.6),
        vllm_config=object())
    expected = PassedMemoryCheck if accepted else ValueError
    with pytest.raises(expected):
        module.Worker._init_device_impl(worker, reuse_prepared_groups=resident)
    assert memory['flushes'] == int(flushed)
    assert worker.init_snapshot.free_memory == (80 if flushed else 30)
    assert worker._relay_deferred_memory_check == (resident and load_format == 'shm')


@pytest.mark.parametrize('free,reserved,allocated,accepted', [
    (40, 20, 20, True), (20, 30, 10, True), (20, 30, 25, False),
])
def test_deferred_shm_check_preserves_configured_kv_and_excludes_live_allocations(
        monkeypatch, free, reserved, allocated, accepted):
    from vllm.utils import relay_memory_profile
    monkeypatch.setattr(relay_memory_profile, 'find_memory_profile', lambda path: None)
    monkeypatch.setattr(module.torch.cuda, 'mem_get_info', lambda device: (free, 100))
    monkeypatch.setattr(module.torch.cuda, 'memory_reserved', lambda device: reserved)
    monkeypatch.setattr(module.torch.cuda, 'memory_allocated', lambda device: allocated)
    worker = SimpleNamespace(load_config=SimpleNamespace(load_format='shm'),
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(load_format='shm')),
        model_config=SimpleNamespace(model='model'), device='cuda:0',
        cache_config=SimpleNamespace(_relay_profile_non_kv_bytes=30),
        requested_memory=60, _relay_deferred_memory_check=True)
    if accepted:
        assert module.Worker.determine_available_memory(worker) == 30
    else:
        with pytest.raises(ValueError, match='configured KV cache'):
            module.Worker.determine_available_memory(worker)


@pytest.mark.parametrize('blocks,accepted', [(2, True), (4, False)])
def test_deferred_capacity_check_uses_actual_overridden_block_bytes(monkeypatch, blocks, accepted):
    from vllm.utils import relay_memory_profile
    monkeypatch.setattr(relay_memory_profile, 'find_memory_profile', lambda path: None)
    monkeypatch.setattr(module.torch.cuda, 'mem_get_info', lambda device: (20, 100))
    monkeypatch.setattr(module.torch.cuda, 'memory_reserved', lambda device: 0)
    monkeypatch.setattr(module.torch.cuda, 'memory_allocated', lambda device: 0)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(load_format='shm')),
        model_config=SimpleNamespace(model='model'), device='cuda:0',
        cache_config=SimpleNamespace(_relay_profile_non_kv_bytes=30,
                                     num_gpu_blocks_override=blocks),
        requested_memory=60, _relay_deferred_memory_check=True,
        model_runner=SimpleNamespace(get_kv_cache_spec=lambda: {
            'layer0': SimpleNamespace(page_size_bytes=4),
            'layer1': SimpleNamespace(page_size_bytes=4)}))
    if accepted:
        assert module.Worker.determine_available_memory(worker) == 30
    else:
        with pytest.raises(ValueError, match='need 32 bytes'):
            module.Worker.determine_available_memory(worker)
