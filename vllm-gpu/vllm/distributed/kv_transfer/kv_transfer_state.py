# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Optional

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_KV_CONNECTOR_AGENT: Optional[KVConnectorBaseType] = None


def get_kv_transfer_group() -> KVConnectorBaseType:
    assert _KV_CONNECTOR_AGENT is not None, (
        "disaggregated KV cache transfer parallel group is not initialized")
    return _KV_CONNECTOR_AGENT


def has_kv_transfer_group() -> bool:
    return _KV_CONNECTOR_AGENT is not None


def reset_shm_kv_transfer() -> None:
    """Release a drained SHM model connector without destroying TP/PP groups.

    The caller must stop inference and drain native KV transfers first.
    SHM connectors own model metadata, not communication groups. Retaining
    their process-global reference across switches reuses stale model state.
    Other connector implementations require their own teardown protocol.
    """
    global _KV_CONNECTOR_AGENT
    if _KV_CONNECTOR_AGENT is None:
        return
    from vllm.distributed.kv_transfer.kv_connector.v1.shm_connector import ShmConnector
    if not isinstance(_KV_CONNECTOR_AGENT, ShmConnector):
        raise RuntimeError('Cannot reset a non-SHM connector with SHM teardown')
    _KV_CONNECTOR_AGENT.clear_connector_metadata()
    _KV_CONNECTOR_AGENT = None


def sync_kv_transfer_connector_vllm_config(
        vllm_config: "VllmConfig") -> None:
    """Rebind the process-global KV connector(s) to *vllm_config*.

    Connectors keep a reference to :class:`VllmConfig` at init time. If
    that reference ever diverged from the worker's config (e.g. shallow
    copies in multi-connector) or you want to be explicit after in-place
    updates to :class:`LoadConfig` (e.g. ``set_load_method``), call this
    with the same :class:`VllmConfig` the worker uses so
    ``load_config.load_method`` and related fields are read correctly.
    """
    if _KV_CONNECTOR_AGENT is None:
        return

    def _bind(conn: object, cfg: "VllmConfig") -> None:
        if conn is None:
            return
        if hasattr(conn, "vllm_config"):
            conn.vllm_config = cfg
        w = getattr(conn, "connector_worker", None)
        if w is not None and hasattr(w, "vllm_config"):
            w.vllm_config = cfg
        for ch in getattr(conn, "_connectors", None) or ():
            _bind(ch, cfg)

    _bind(_KV_CONNECTOR_AGENT, vllm_config)


def is_v1_kv_transfer_group(
        connector: Optional[KVConnectorBaseType] = None) -> bool:
    """Check if the KV connector is the v1 connector.
    If the argument is None, it will check the global KV connector

    Args:
        connector: The KV connector to check. If None, it will check the
            global KV connector.

    Note:
        This function will no-longer be needed after the v1 KV connector
        becomes the default.
    """
    if connector is None:
        connector = _KV_CONNECTOR_AGENT

    if connector is None:
        return False

    return isinstance(connector, KVConnectorBase_V1)


def ensure_kv_transfer_initialized(vllm_config: "VllmConfig") -> None:
    """
    Initialize KV cache transfer parallel group.
    """

    global _KV_CONNECTOR_AGENT

    if vllm_config.kv_transfer_config is None:
        return

    if (vllm_config.kv_transfer_config.is_kv_transfer_instance
            and _KV_CONNECTOR_AGENT is None):
        if envs.VLLM_USE_V1:
            _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
                config=vllm_config, role=KVConnectorRole.WORKER)
        else:
            raise ValueError("V0 is no longer supported")
