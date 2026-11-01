import json
from types import SimpleNamespace

import pytest
import torch

from vllm.utils.relay_memory_profile import (
    find_memory_profile, memory_profile_identity, read_non_kv_memory)


def identity_config(path, tp=2):
    return SimpleNamespace(
        model_config=SimpleNamespace(model=str(path), dtype=torch.bfloat16,
                                     max_model_len=4200, enforce_eager=True),
        cache_config=SimpleNamespace(cache_dtype='auto', block_size=32, enable_prefix_caching=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096, max_num_seqs=2))


def test_identity_survives_unique_producer_directory_and_separates_tp(tmp_path, monkeypatch):
    source = tmp_path / 'checkpoint'
    producer = tmp_path / 'unique-producer'
    source.mkdir(); producer.mkdir()
    (source / 'config.json').write_text(json.dumps(dict(hidden_size=5120, num_hidden_layers=64)))
    (producer / 'config.json').symlink_to(source / 'config.json')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device:
        SimpleNamespace(name='NVIDIA H20', major=9, minor=0, total_memory=102085492736))
    original = memory_profile_identity(identity_config(source), None)
    assert memory_profile_identity(identity_config(producer), None) == original
    assert memory_profile_identity(identity_config(producer, tp=4), None) != original
    changed = identity_config(producer)
    changed.scheduler_config.max_num_batched_tokens = 2048
    assert memory_profile_identity(changed, None) != original


@pytest.mark.parametrize('budget', [None, True, 0, -1, 1.5, '123'])
def test_invalid_budget_rejected(tmp_path, budget):
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(dict(version=1, profiles=[
        dict(identity={'tp': 2}, non_kv_memory_bytes=budget)])))
    with pytest.raises(ValueError, match='Invalid non-KV'):
        read_non_kv_memory(path, {'tp': 2})


def test_wrong_topology_or_duplicate_profile_rejected(tmp_path):
    path = tmp_path / 'profile.json'
    identity = dict(tp=2, pp=1, max_num_batched_tokens=4096)
    entry = dict(identity=identity, non_kv_memory_bytes=35_000_000_000)
    path.write_text(json.dumps(dict(version=1, profiles=[entry])))
    assert read_non_kv_memory(path, identity) == 35_000_000_000
    with pytest.raises(ValueError, match='found 0'):
        read_non_kv_memory(path, dict(identity, tp=4))
    path.write_text(json.dumps(dict(version=1, profiles=[entry, entry])))
    with pytest.raises(ValueError, match='found 2'):
        read_non_kv_memory(path, identity)


def test_explicit_missing_profile_does_not_silently_fallback(tmp_path, monkeypatch):
    path = tmp_path / 'missing.json'
    monkeypatch.setenv('VLLM_SHM_MEMORY_PROFILE', str(path))
    assert find_memory_profile(tmp_path) == path
    with pytest.raises(FileNotFoundError):
        read_non_kv_memory(path, {})


def test_resident_group_inventory_changes_profile_identity(tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text(json.dumps(dict(hidden_size=5120)))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device:
        SimpleNamespace(name='NVIDIA H20', major=9, minor=0, total_memory=102085492736))
    config = identity_config(tmp_path)
    unscoped = memory_profile_identity(config, None)
    config.cache_config._relay_group_inventory = dict(world_size=2, groups=[
        dict(role='tp', partitions=[[0, 1]])])
    calibrated = memory_profile_identity(config, None)
    assert calibrated != unscoped
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(dict(version=1, profiles=[
        dict(identity=calibrated, non_kv_memory_bytes=35_000_000_000)])))
    assert read_non_kv_memory(path, calibrated) == 35_000_000_000
    config.cache_config._relay_group_inventory = dict(world_size=2, groups=[
        dict(role='tp', partitions=[[0, 1]]), dict(role='tp', partitions=[[0], [1]])])
    with pytest.raises(ValueError, match='found 0'):
        read_non_kv_memory(path, memory_profile_identity(config, None))


def test_pipeline_partition_cannot_reuse_another_layout_budget(tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text(json.dumps(dict(num_hidden_layers=32)))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device:
        SimpleNamespace(name='NVIDIA H20', major=9, minor=0, total_memory=102085492736))
    config = identity_config(tmp_path)
    config.parallel_config.pipeline_parallel_size = 2
    monkeypatch.delenv('VLLM_PP_LAYER_PARTITION', raising=False)
    uniform = memory_profile_identity(config, None)
    monkeypatch.setenv('VLLM_PP_LAYER_PARTITION', '8,24')
    uneven = memory_profile_identity(config, None)
    assert uniform != uneven
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(dict(version=1, profiles=[
        dict(identity=uniform, non_kv_memory_bytes=1234)])))
    with pytest.raises(ValueError, match='found 0'):
        read_non_kv_memory(path, uneven)
