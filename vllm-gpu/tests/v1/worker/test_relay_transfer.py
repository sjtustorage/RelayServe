import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.relay_transfer import RelayKVPlan


def test_plan_batches_requests_by_layer_interval():
    plan = RelayKVPlan()
    plan.add_request((2, 3), [10, 11], [20, 21])
    plan.add_request((2, 3), [12], [22])
    assert plan.transfers[(2, 3)].gpu_block_ids == [10, 11, 12]
    assert plan.transfers[(2, 3)].cpu_block_ids == [20, 21, 22]


def test_plan_rejects_mismatched_blocks():
    with pytest.raises(ValueError, match="block counts"):
        RelayKVPlan().add_request((0,), [1], [2, 3])


def test_plan_ignores_empty_layer_interval():
    plan = RelayKVPlan()
    plan.add_request((), [1], [2])
    assert not plan.transfers


def test_tokenwise_connector_batches_native_transfer(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1 import shm_connector
    copy, sync = Mock(), Mock()
    monkeypatch.setattr(shm_connector, "load_kv_cache_to_gpu", copy)
    monkeypatch.setattr(shm_connector, "load_kv_cache_to_gpu_sync", sync)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format="shm", load_method="tokenwise")),
        reqs_to_load=[], _cold_start_partition=None,
        _registered_layer_ids=[0, 1])
    metadata = SimpleNamespace(reqs_to_load={
        "a": SimpleNamespace(local_block_ids=[10, 11], remote_block_ids=[20, 21]),
        "b": SimpleNamespace(local_block_ids=[12], remote_block_ids=[22])})
    shm_connector.ShmConnectorWorker.start_load_kv(worker, metadata)
    copy.assert_called_once_with(gpu_block_ids=[10, 11, 12],
                                 cpu_block_ids=[20, 21, 22], layer_ids=[0, 1])
    sync.assert_called_once_with()
    assert worker.reqs_to_load == ["a", "b"]
