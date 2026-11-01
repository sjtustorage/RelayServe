# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared Memory KV Connector for remote-to-GPU KV Cache Transfer

This connector enables KV cache transfer between two vLLM processes:
- remote process: Performs prefill and writes KV cache to shared memory
- GPU process: Reads KV cache from shared memory and completes remaining prefill

Features:
- Handles heterogeneous TP (remote TP may differ from GPU TP)
- Supports partial prefill on remote side
- Uses shared memory for efficient data transfer
"""

import logging
import re
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Any, Optional
import time
from vllm import envs
import torch
import numpy as np
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp, KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_tp_group)
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

ReqId = str
EngineId = str
logger = init_logger(__name__)


class ShmConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        pass


class ShmConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        self.vllm_config = vllm_config
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id
        
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: Optional[ShmConnectorScheduler] = \
                ShmConnectorScheduler(vllm_config, self.engine_id)
            self.connector_worker: Optional[ShmConnectorWorker] = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = ShmConnectorWorker(vllm_config)

    ############################################################
    # Class Methods
    ############################################################
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        """SHM connector works with HND layout"""
        return "HND"

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Not used for SHM connector"""
        pass

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished loading requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, ShmConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for layer to be loaded from shared memory."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Save a layer of KV cache (not implemented for SHM connector).
        
        SHM connector only loads from remote to GPU, it doesn't save back to remote.
        This method exists to match the interface.
        """
        # No-op for SHM connector
        pass

    def wait_for_save(self):
        """
        Wait for all save operations to complete.
        For SHM connector, we don't save, so this is a no-op.
        """
        pass

    def is_shared_memory_connector(self) -> bool:
        return True


class ShmConnectorScheduler:
    """Scheduler-side implementation"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        self.engine_id: EngineId = engine_id

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        """
        Get number of tokens that can be loaded from remote's prefill KV cache.
        
        Returns:
            tuple: (number of tokens, True if async loading)
        """
        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
            self, request: "Request",
            blocks: "KVCacheBlocks",
            num_external_tokens: int):
        """
        Update connector state after blocks are allocated for external tokens.
        """
        params = request.kv_transfer_params
        logger.debug(
            "ShmConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)

        if not params or not params.get("do_remote_prefill"):
            return

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build connector metadata for worker-side."""
        return ShmConnectorMetadata()

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        params = request.kv_transfer_params
        logger.debug(
            "ShmConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s", request.status, params)
        if not params:
            return False, None

        if (not params.get("do_remote_decode")
                or (request.status != RequestStatus.FINISHED_LENGTH_CAPPED and 
                request.status != RequestStatus.FINISHED_PREFILL_LENGTH_CAPPED)):
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_host=None,
            remote_port=None,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            num_computed_tokens=request.num_computed_tokens)

class ShmConnectorWorker:
    """Worker-side implementation"""

    def __init__(self, vllm_config: VllmConfig):
        pass

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        pass

    def get_finished(self):
        return set(), set()

    def start_load_kv(self, metadata: ShmConnectorMetadata):
        pass

    def wait_for_load_kv_cache(self) -> None:
        pass

