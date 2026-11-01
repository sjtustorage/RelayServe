# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import DeviceConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.shm_connector import (
    ShmConnectorMetadata, ShmConnectorWorker)
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

from tests.v1.kv_connector.unit.utils import create_request, create_scheduler, create_vllm_config


def test_basic_interface():
    """Unit test for basic ShmConnector interface functionality."""

    vllm_config = create_vllm_config()
    # Update kv_transfer_config to use ShmConnector
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="ShmConnector",
        kv_role="kv_both",
        shm_sizes=[1024 * 1024 * 1024],  # 1GB
        shm_num_blocks=1000,
        shm_block_len=4096,
        shm_tp_world_size=1,
    )
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(request_id=1,
                             block_size=BLOCK_SIZE,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True)
    request_id = request.request_id

    scheduler.add_request(request)

    # Remote Prefill, triggers ShmConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, ShmConnectorMetadata)

    assert len(kv_connector_metadata.reqs_to_load) == 1
    assert request_id in kv_connector_metadata.reqs_to_load
    req_meta = kv_connector_metadata.reqs_to_load[request_id]

    for block_id, block in zip(
            req_meta.local_block_ids, scheduler.kv_cache_manager.coordinator.
            single_type_managers[0].req_to_blocks[request_id]):
        assert block_id == block.block_id
    print(f"test_basic_interface passed")


# if __name__ == "__main__":
#     test_basic_interface()

def test_prompt_less_than_block_size():
    """
    Test that we can handle case where prompt is < block.

    In this case, the P worker will still send remote_block_ids of the
    partial block. The D worker should schedule an async read
    in this case.
    """
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="ShmConnector",
        kv_role="kv_both",
        shm_sizes=[1024 * 1024 * 1024],
        shm_num_blocks=1000,
        shm_block_len=4096,
        shm_tp_world_size=1,
    )
    scheduler = create_scheduler(vllm_config)

    # Half of a block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = int(BLOCK_SIZE * 0.5)

    # Request will have 1 partial remote block.
    request = create_request(request_id=1,
                             block_size=BLOCK_SIZE,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True,
                             num_remote_blocks=1)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()

    # This request will read async.
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, ShmConnectorMetadata)
    assert len(kv_connector_metadata.reqs_to_load) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0

    print(f"test_prompt_less_than_block_size passed")


NATIVE = "vllm.distributed.kv_transfer.kv_connector.v1.shm_connector."


def make_worker(dist_init):
    config = create_vllm_config()
    config.device_config = DeviceConfig("cuda")
    worker = ShmConnectorWorker(config, "test-engine")
    shape = FlashAttentionBackend.get_kv_cache_shape(
        num_blocks=4, block_size=16, num_kv_heads=4, head_size=64)
    caches = {f"model.layers.{layer}.self_attn": torch.zeros(
        shape, dtype=torch.float16) for layer in (2, 5)}
    with patch(NATIVE + "init_gpu_kv_cache_info") as register:
        worker.register_kv_caches(caches)
    return worker, caches, register


def metadata_for(requests):
    metadata = ShmConnectorMetadata()
    for req_id, local, remote in requests:
        metadata.add_new_req(req_id, local, {"remote_block_ids": remote})
    return metadata


def test_register_kv_caches(dist_init):
    worker, caches, register = make_worker(dist_init)
    register.assert_called_once_with(
        gpu_caches_ptrs={layer: [tensor[0].data_ptr(), tensor[1].data_ptr()]
                         for layer, tensor in zip((2, 5), caches.values())},
        gpu_num_blocks=4,
        gpu_block_len=caches["model.layers.2.self_attn"][0].numel() * 2 // 4,
        total_num_kv_heads=worker.model_config.get_total_num_kv_heads())
    assert worker._registered_layer_ids == [2, 5]


def test_load_completion_uses_native_batch_fence(dist_init):
    worker, _, _ = make_worker(dist_init)
    metadata = metadata_for([("id", [1, 2, 3], [4, 5, 6])])
    with patch(NATIVE + "load_kv_cache_to_gpu") as load, \
         patch(NATIVE + "load_kv_cache_to_gpu_sync") as sync, \
         patch(NATIVE + "check_kv_cache_load_state", side_effect=[False, True, True]) as ready:
        worker.start_load_kv(metadata)
        load.assert_called_once_with(gpu_block_ids=[1, 2, 3],
            cpu_block_ids=[4, 5, 6], layer_ids=[2, 5])
        sync.assert_called_once_with()
        assert worker.get_finished() == (set(), set())
        assert worker.get_finished() == (set(), {"id"})
        assert worker.get_finished() == (set(), set())
        assert all(call.args == () for call in ready.call_args_list)


def test_concurrent_requests_coalesce_one_transfer(dist_init):
    worker, _, _ = make_worker(dist_init)
    requests = [(f"id_{i}", [i + 1], [i + 10]) for i in range(5)]
    with patch(NATIVE + "load_kv_cache_to_gpu") as load, \
         patch(NATIVE + "load_kv_cache_to_gpu_sync") as sync, \
         patch(NATIVE + "check_kv_cache_load_state", return_value=True):
        worker.start_load_kv(metadata_for(requests))
        load.assert_called_once_with(gpu_block_ids=[1, 2, 3, 4, 5],
            cpu_block_ids=[10, 11, 12, 13, 14], layer_ids=[2, 5])
        sync.assert_called_once_with()
        assert worker.get_finished() == (set(), {req[0] for req in requests})


def test_repeated_request_loads_notify_once_after_fence(dist_init):
    worker, _, _ = make_worker(dist_init)
    with patch(NATIVE + "load_kv_cache_to_gpu") as load, \
         patch(NATIVE + "load_kv_cache_to_gpu_sync") as sync, \
         patch(NATIVE + "check_kv_cache_load_state", side_effect=[False, True, True]):
        for i in range(4):
            worker.start_load_kv(metadata_for([("id", [i + 1], [i + 10])]))
        assert load.call_count == sync.call_count == 4
        assert worker.get_finished() == (set(), set())
        assert worker.get_finished() == (set(), {"id"})
        assert worker.get_finished() == (set(), set())


def test_layerwise_connector_leaves_copy_to_model_runner(dist_init):
    worker, _, _ = make_worker(dist_init)
    worker.vllm_config.load_config.load_format = "shm"
    worker.vllm_config.load_config.load_method = "layerwise"
    with patch(NATIVE + "load_kv_cache_to_gpu") as load, \
         patch(NATIVE + "load_kv_cache_to_gpu_sync") as sync:
        worker.start_load_kv(metadata_for([("id", [1], [2])]))
        load.assert_not_called()
        sync.assert_not_called()
        assert worker.reqs_to_load == ["id"]


def test_empty_metadata_submits_no_native_copy(dist_init):
    worker, _, _ = make_worker(dist_init)
    with patch(NATIVE + "load_kv_cache_to_gpu") as load, \
         patch(NATIVE + "load_kv_cache_to_gpu_sync") as sync:
        worker.start_load_kv(ShmConnectorMetadata())
        load.assert_not_called()
        sync.assert_not_called()
        assert worker.reqs_to_load == []


@pytest.mark.parametrize('suffix', ['self_attn', 'self_attn.attn'])
def test_pp_relay_writes_registered_attention_name(suffix):
    worker = object.__new__(ShmConnectorWorker)
    worker._pp_rank = 1
    worker._cold_start_partition = [[0, 1], [2]]
    cache = torch.zeros(2, 4, 3)
    worker._kv_caches_dict = {f'model.layers.2.{suffix}': cache}
    payload = torch.arange(12).reshape(2, 2, 3).float()
    group = MagicMock()
    group.recv_tensor_dict.return_value = {'kv_2': payload}
    with patch('vllm.distributed.parallel_state.get_pp_group', return_value=group):
        worker._recv_relay([1, 3])
    torch.testing.assert_close(cache[:, [1, 3]], payload)
    assert torch.count_nonzero(cache[:, [0, 2]]) == 0


@pytest.mark.parametrize('missing', ['payload', 'cache'])
def test_pp_relay_rejects_missing_layer(missing):
    worker = object.__new__(ShmConnectorWorker)
    worker._pp_rank = 1
    worker._cold_start_partition = [[0, 1], [2]]
    worker._kv_caches_dict = ({} if missing == 'cache' else
        {'model.layers.2.self_attn.attn': torch.zeros(2, 4, 3)})
    group = MagicMock()
    group.recv_tensor_dict.return_value = ({} if missing == 'payload' else
        {'kv_2': torch.ones(2, 1, 3)})
    with patch('vllm.distributed.parallel_state.get_pp_group', return_value=group):
        with pytest.raises(ValueError, match='Missing relayed KV for layer 2'):
            worker._recv_relay([1])
