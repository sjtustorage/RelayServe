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
from vllm.v1.relay_transfer import RelayKVTransfer

from shm_manager.torch import (
    init_gpu_kv_cache_info, load_kv_cache_to_gpu, load_kv_cache_to_gpu_sync, check_kv_cache_load_state)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

ReqId = str
EngineId = str
logger = init_logger(__name__)


@dataclass
class ShmReqMeta:
    """Metadata for a request's KV cache transfer"""
    local_block_ids: list[int]  # Blocks to write on GPU
    remote_block_ids: list[int]  # Blocks to read from remote


class ShmConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        # Requests that need to load KV cache from remote
        self.reqs_to_load: dict[ReqId, ShmReqMeta] = {}

    def add_new_req(
        self, 
        request_id: ReqId, 
        local_block_ids: list[int], 
        kv_transfer_params: dict[str, Any],
    ):
        _req = ShmReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"]
        )
        self.reqs_to_load[request_id] = _req


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
            self.connector_worker = ShmConnectorWorker(vllm_config, self.engine_id)

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

    def set_relay_config(
        self,
        cold_partition: list[list[int]],
        kv_caches_dict: dict[str, torch.Tensor],
        pp_rank: int,
        pp_size: int,
        relay_kv_reshaped: Optional[torch.Tensor] = None,
    ) -> None:
        """Configure PP relay for tokenwise cold-start KV distribution."""
        assert self.connector_worker is not None
        self.connector_worker.set_relay_config(
            cold_partition, kv_caches_dict, pp_rank, pp_size,
            relay_kv_reshaped)

    def clear_relay_config(self) -> None:
        """Clear PP relay state and re-register own-layer GPU pointers."""
        assert self.connector_worker is not None
        self.connector_worker.clear_relay_config()

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


class ShmConnectorScheduler:
    """Scheduler-side implementation"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        self.engine_id: EngineId = engine_id

        # Track requests that need to load KV cache from remote
        self._reqs_need_load: dict[ReqId, tuple["Request", list[int]]] = {}
        # Track requests to send (not needed for SHM)
        self._reqs_need_send: dict[ReqId, float] = {}

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        """
        Get number of tokens that can be loaded from remote's prefill KV cache.
        
        Returns:
            tuple: (number of tokens, True if async loading)
        """
        params = request.kv_transfer_params
        logger.debug(
            "ShmConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params)

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            num_external_computed_tokens = \
                params.get("num_computed_tokens", request.num_tokens)
            count = num_external_computed_tokens - num_computed_tokens
            if count > 0:
                return count, True

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

        if params.get("remote_block_ids"):
            # If remote_blocks and num_external_tokens = 0, we have
            # a full prefix cache hit on the D worker. We need to call
            # send_notif in _read_blocks to free the memory on the P.
            # prefill已经计算出，但是decode实例上已经有prefix的KV cache，因此不需要从外部加载
            # 因此local_block_ids为空，不需要新的blocks
            local_block_ids = (blocks.get_unhashed_block_ids()
                                if num_external_tokens > 0 else [])
            # Get unhashed blocks to pull from remote.
            self._reqs_need_load[request.request_id] = (
                request, local_block_ids)
        else:
            assert num_external_tokens == 0
        # Only trigger 1 KV transfer per request.
        # 改为False就不会进入上面的分支，不需要加入_reqs_need_recv了
        params["do_remote_prefill"] = False 

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build connector metadata for worker-side."""
        meta = ShmConnectorMetadata()

        # Convert requests to load into metadata
        for req_id, (req, local_block_ids) in \
                self._reqs_need_load.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=local_block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        # Clear the list
        self._reqs_need_load.clear()

        return meta

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
            "NIXLConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s", request.status, params)
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_load[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if (not params.get("do_remote_decode")
                or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
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

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.engine_id: EngineId = engine_id
        self.vllm_config = vllm_config
        self.device_type = vllm_config.device_config.device_type
        self.block_size = vllm_config.cache_config.block_size
        self.model_name = vllm_config.model_config.model.split("/")[-1]
        
        # KV caches
        self.kv_caches: dict[str, torch.Tensor] = {}
        
        # TP info
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.tp_group = get_tp_group()
        self.num_blocks = 0
        
        # KV Caches
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # Note: host xfer buffer ops when use_host_buffer is True
        self.copy_blocks: Optional[CopyBlocksOp] = None

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        # rank will still only pull from a single remote TP worker.
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}

        self.num_regions = 0
        self.num_layers = 0

        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # TODO(mgoin): remove this once we have hybrid memory allocator
        # Optimization for models with local attention (Llama 4)
        # List of block window sizes for each layer for local attention
        self.block_window_per_layer: list[Optional[int]] = []
        self.use_mla = self.model_config.use_mla

        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self.reqs_to_load: list[str] = []

        # PP relay state (tokenwise cold-start)
        self._relay_kv_reshaped: Optional[torch.Tensor] = None
        self._cold_start_partition: Optional[list[list[int]]] = None
        self._kv_caches_dict: Optional[dict[str, torch.Tensor]] = None


    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register GPU KV caches."""
        seen_base_addresses = []
        tensor_size_bytes = None
        layer_to_caches_ptrs: dict[int, list[int]] = {}

        for layer_name, caches in kv_caches.items():
            layer_idx = int(re.match(r"model\.layers\.(\d+)\.", layer_name).group(1))
            layer_to_caches_ptrs[layer_idx] = []
            cache_list = caches if not self.use_mla else [caches]
            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue
                seen_base_addresses.append(base_addr)
                layer_to_caches_ptrs[layer_idx].append(base_addr)
                curr_tensor_size_bytes = cache.numel() * cache.element_size()
                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0] 
                assert tensor_size_bytes == curr_tensor_size_bytes, \
                "All kv cache tensors must have the same size"
        self.num_layers = len(layer_to_caches_ptrs)
        self._registered_layer_ids = sorted(layer_to_caches_ptrs.keys())

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        assert tensor_size_bytes % self.num_blocks == 0

        gpu_block_len = tensor_size_bytes // self.num_blocks
        total_num_kv_heads = self.model_config.get_total_num_kv_heads()
        self._gpu_block_len = gpu_block_len
        self._total_num_kv_heads = total_num_kv_heads

        if self.device_type == "cuda":
            init_gpu_kv_cache_info(
                gpu_caches_ptrs = layer_to_caches_ptrs,
                gpu_num_blocks = self.num_blocks,
                gpu_block_len = gpu_block_len,
                total_num_kv_heads = total_num_kv_heads,
            )

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or loading.
        
        Returns:
            tuple[set[str], set[str]]: (finished_sending, finished_loading)
        """
        finished_loading = set()
        if check_kv_cache_load_state is not None and check_kv_cache_load_state():
            for req_id in list(self.reqs_to_load):
                finished_loading.add(req_id)
                self.reqs_to_load.remove(req_id)
        
        # No sending in SHM connector
        return set(), finished_loading

    # ------------------------------------------------------------------
    # PP relay config (tokenwise cold-start)
    # ------------------------------------------------------------------

    def set_relay_config(
        self,
        cold_partition: list[list[int]],
        kv_caches_dict: dict[str, torch.Tensor],
        pp_rank: int,
        pp_size: int,
        relay_kv_reshaped: Optional[torch.Tensor] = None,
    ) -> None:
        """Store state needed for one-shot PP relay during the first
        ``start_load_kv`` call.

        Args:
            cold_partition: layer indices per PP rank, e.g.
                ``[[0..6], [7..14], [15..23], [24..35]]``.
            kv_caches_dict: this rank's KV caches (name → tensor).
            pp_rank: this rank's PP index.
            pp_size: total PP world size.
            relay_kv_reshaped: single-layer relay buffer (PP0 only).
        """
        self._cold_start_partition = cold_partition
        self._kv_caches_dict = kv_caches_dict
        self._pp_rank = pp_rank
        self._pp_size = pp_size
        self._relay_kv_reshaped = relay_kv_reshaped

    def clear_relay_config(self) -> None:
        """Free relay buffer and clear cold-start state.
        Re-registers own-layer GPU pointers so the C++ KVCacheManager
        is in a clean state after relay may have overridden them."""
        self._cold_start_partition = None
        self._relay_kv_reshaped = None
        self._kv_caches_dict = None
        # Re-register own-layer GPU pointers (relay may have overridden them)
        if hasattr(self, '_kv_caches_for_reregister') and \
                self._kv_caches_for_reregister is not None:
            self.register_kv_caches(self._kv_caches_for_reregister)
            self._kv_caches_for_reregister = None

    # ------------------------------------------------------------------
    # start_load_kv  (with PP relay support)
    # ------------------------------------------------------------------

    def start_load_kv(self, metadata: ShmConnectorMetadata):
        t1 = time.perf_counter()
        transfer = RelayKVTransfer()
        load_cfg = self.vllm_config.load_config
        layerwise = (load_cfg.load_format == "shm"
                     and load_cfg.load_method == "layerwise")
        for req_id, req_meta in metadata.reqs_to_load.items():
            logger.debug(
                f"start_load_kv for request {req_id}. "
                f"Num local_block_ids: {len(req_meta.local_block_ids)}. "
                f"Num remote_block_ids: {len(req_meta.remote_block_ids)}.")

            self.reqs_to_load.append(req_id)
            if layerwise:
                continue
            if self._cold_start_partition is not None and self._pp_rank > 0:
                # PP receivers get remote data from PP0, not from host SHM.
                transfer.gpu_block_ids.extend(req_meta.local_block_ids)
            else:
                transfer.add_request(req_meta.local_block_ids,
                                     req_meta.remote_block_ids)

        all_local_block_ids = transfer.gpu_block_ids
        all_remote_block_ids = transfer.cpu_block_ids

        # ``load_method`` here is cold-start *weight* loading from SHM (see
        # ``LoadConfig``).  For ``layerwise``, remote KV is pulled in
        # ``GPUModelRunner.recv_shm_kv_cache()`` during forward, not here;
        # running the connector load path would duplicate or fight that.
        if layerwise:
            return

        if self._cold_start_partition is not None:
            # Tokenwise cold-start PP relay (one-shot, first iteration only)
            if self._pp_rank == 0:
                # PP0 needs both local/remote block ids to pull from SHM.
                if (not all_local_block_ids) or (not all_remote_block_ids):
                    logger.warning(
                        "PP0 relay skipped due to missing block ids: "
                        "local=%d remote=%d",
                        len(all_local_block_ids), len(all_remote_block_ids))
                    self._cold_start_partition = None
                    return
                self._load_own_and_relay(
                    all_local_block_ids, all_remote_block_ids)
            else:
                # Receiver ranks only need local block ids for placement.
                if not all_local_block_ids:
                    logger.warning(
                        "PP rank %d relay recv skipped due to empty local "
                        "block ids", self._pp_rank)
                    self._cold_start_partition = None
                    return
                self._recv_relay(all_local_block_ids)
            self._cold_start_partition = None
            return

        # Normal path: load from SHM (PP0 only has SHM pointers)
        no_data = (not all_local_block_ids or not all_remote_block_ids)
        if no_data or load_kv_cache_to_gpu is None:
            return
        load_kv_cache_to_gpu(
            gpu_block_ids=all_local_block_ids,
            cpu_block_ids=all_remote_block_ids,
            layer_ids=self._registered_layer_ids,
        )
        load_kv_cache_to_gpu_sync()
        t2 = time.perf_counter()
        logger.info(f"start_load_kv time: {t2 - t1} seconds")

    # ------------------------------------------------------------------
    # PP relay internals
    # ------------------------------------------------------------------

    def _load_own_and_relay(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
    ) -> None:
        """PP0: load own layers from SHM, then relay non-owned layers
        to target PP ranks one layer at a time through the relay buffer."""
        from vllm.distributed.parallel_state import get_pp_group
        pp_group = get_pp_group()

        # Step 1: load PP0's own cold-start layers from SHM
        load_kv_cache_to_gpu(
            gpu_block_ids=local_block_ids,
            cpu_block_ids=remote_block_ids,
            layer_ids=self._registered_layer_ids,
        )
        load_kv_cache_to_gpu_sync()

        # Step 2: relay each non-owned layer via the single-layer buffer
        relay = self._relay_kv_reshaped
        assert relay is not None, "relay buffer must be set on PP0"
        relay_ptrs = [cache.data_ptr() for cache in relay]

        for dst_rank in range(1, self._pp_size):
            dst_layers = self._cold_start_partition[dst_rank]
            for layer_idx in dst_layers:
                init_gpu_kv_cache_info(
                    gpu_caches_ptrs={layer_idx: relay_ptrs},
                    gpu_num_blocks=self.num_blocks,
                    gpu_block_len=self._gpu_block_len,
                    total_num_kv_heads=self._total_num_kv_heads,
                )
                load_kv_cache_to_gpu(
                    gpu_block_ids=local_block_ids,
                    cpu_block_ids=remote_block_ids,
                    layer_ids=[layer_idx],
                )
                load_kv_cache_to_gpu_sync()

                pp_group.send_tensor_dict(
                    {f"kv_{layer_idx}":
                     relay[:, local_block_ids].contiguous()},
                    dst=dst_rank)

        # Step 3: restore own-layer GPU pointers
        self._kv_caches_for_reregister = self._kv_caches_dict
        self.register_kv_caches(self._kv_caches_dict)

    def _recv_relay(self, local_block_ids: list[int]) -> None:
        """Non-PP0: receive cold-start KV layers from PP0 via NCCL."""
        from vllm.distributed.parallel_state import get_pp_group
        from vllm.v1.worker.cold_start_transition import _build_layer_idx_to_name
        pp_group = get_pp_group()

        my_layers = self._cold_start_partition[self._pp_rank]
        kv_dict = self._kv_caches_dict
        layer_names = _build_layer_idx_to_name(kv_dict)
        for layer_idx in my_layers:
            data = pp_group.recv_tensor_dict(src=0)
            name = layer_names.get(layer_idx)
            key = f"kv_{layer_idx}"
            if key not in data or name is None:
                raise ValueError(f"Missing relayed KV for layer {layer_idx}")
            kv_dict[name][:, local_block_ids] = data[key]

    def wait_for_load_kv_cache(self) -> None:
        if load_kv_cache_to_gpu_sync is not None:
            load_kv_cache_to_gpu_sync()

