"""SHM model metadata must not survive switches in retained GPU processes."""
import pytest

from vllm.distributed.kv_transfer import kv_transfer_state as state
from vllm.distributed.kv_transfer.kv_connector.v1.shm_connector import ShmConnector


def test_reset_releases_model_connector_and_is_idempotent(monkeypatch):
    connector = object.__new__(ShmConnector)
    connector._connector_metadata = object()
    monkeypatch.setattr(state, '_KV_CONNECTOR_AGENT', connector)
    state.reset_shm_kv_transfer()
    assert not state.has_kv_transfer_group()
    assert connector._connector_metadata is None
    state.reset_shm_kv_transfer()


def test_reset_preserves_unknown_connector(monkeypatch):
    connector = object()
    monkeypatch.setattr(state, '_KV_CONNECTOR_AGENT', connector)
    with pytest.raises(RuntimeError, match='non-SHM'):
        state.reset_shm_kv_transfer()
    assert state.get_kv_transfer_group() is connector
