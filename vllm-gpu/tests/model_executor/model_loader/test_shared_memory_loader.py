from types import SimpleNamespace
from unittest.mock import Mock, call, patch
import json

import pytest
import torch

from vllm.model_executor.model_loader import shared_memory_loader as shm


def make_loader():
    return shm.SharedMemoryLoader(SimpleNamespace(
        model_loader_extra_config=None, load_async=False, load_method="tokenwise"))


def test_default_load_uses_global_pp_layer_indices(tmp_path):
    loader = make_loader()
    model = torch.nn.Linear(2, 2, bias=False)
    config = SimpleNamespace(model=str(tmp_path), dtype=torch.bfloat16,
                             get_layers_start_end_indices=Mock(return_value=(7, 12)))
    parallel = object()
    vllm_config = SimpleNamespace(parallel_config=parallel)
    weights = {"weight": torch.ones(2, 2, dtype=torch.bfloat16)}
    with patch.object(shm.torch.cuda, "current_device", return_value=0), patch.object(
        shm, "init_cuda_memory_dict", return_value=weights,
    ) as allocate, patch.object(shm, "load_model_to_gpu_blocking") as load, patch.object(
        shm, "initialize_model", return_value=model,
    ):
        assert loader.load_model(vllm_config=vllm_config, model_config=config) is model
    config.get_layers_start_end_indices.assert_called_once_with(parallel)
    allocate.assert_called_once_with(0, [7, 8, 9, 10, 11])
    load.assert_called_once_with([7, 8, 9, 10, 11])
    assert model.weight.data_ptr() == weights["weight"].data_ptr()
    with patch.object(shm, "load_model_to_gpu") as launch:
        loader.launch_load_weights()
    launch.assert_not_called()
    # Recovery must submit only newly requested layers, never recopy live storage.
    with patch.object(shm, "load_model_to_gpu") as launch:
        loader.launch_load_weights(layer_idxes=[11, 12, 12, 13])
    launch.assert_called_once_with([12, 13])


def test_serial_load_joins_storage_before_constructing_model(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    weights = {"weight": torch.ones(2, 2, dtype=torch.bfloat16)}
    state = {}
    future = Mock()
    def result():
        if 'weights' not in state:
            state['weights'] = state['prepare']()
        return state['weights']
    future.result.side_effect = result
    executor = Mock()
    def submit(fn):
        state['prepare'] = fn
        return future
    executor.submit.side_effect = submit
    monkeypatch.setattr(shm, 'ThreadPoolExecutor', lambda **kwargs: executor)
    def construct(**kwargs):
        assert state.get('weights') is weights, 'serial construction overlapped storage'
        return model
    config = SimpleNamespace(model=str(tmp_path), dtype=torch.bfloat16,
                             get_layers_start_end_indices=Mock(return_value=(0, 1)))
    with patch.object(shm.torch.cuda, 'current_device', return_value=0), patch.object(
        shm, 'init_cuda_memory_dict', return_value=weights), patch.object(
        shm, 'load_model_to_gpu_blocking'), patch.object(
        shm, 'initialize_model', side_effect=construct):
        assert make_loader().load_model(vllm_config=SimpleNamespace(
            parallel_config=object()), model_config=config) is model


def test_default_launch_before_model_load_fails_without_submitting_copy():
    with patch.object(shm, "load_model_to_gpu") as launch:
        with pytest.raises(RuntimeError, match="load_model must run"):
            make_loader().launch_load_weights()
    launch.assert_not_called()


def test_explicit_layer_order_and_dtype_are_local_to_loader():
    first, second = make_loader(), make_loader()
    first._weight_dtype = torch.bfloat16
    second._weight_dtype = torch.float16
    observed = []
    with patch.object(shm, "load_model_to_gpu", side_effect=lambda layers:
                      observed.append((list(layers), torch.get_default_dtype()))), patch.object(
        shm, "load_model_to_gpu_sync", side_effect=lambda layers:
        observed.append((list(layers), torch.get_default_dtype())),
    ):
        first.launch_load_weights(layer_idxes=[9, 8, 7])
        second.launch_load_weights(layer_idxes=[1, 2])
        first.wait_for_weights_loaded([9])
    assert observed == [([9, 8, 7], torch.bfloat16), ([1, 2], torch.float16),
                        ([9], torch.float32)]
    assert torch.get_default_dtype() == torch.float32


def test_relay_timer_keeps_individual_layer_fences():
    loader = make_loader()
    loader.relay_weight_wait_events = []
    events = [Mock() for _ in range(4)]
    order = []
    for index, event in enumerate(events):
        event.record.side_effect = lambda index=index: order.append(('event', index))
    with patch.object(shm, 'check_model_load_state', return_value=False), patch.object(
        shm.torch.cuda, 'Event', side_effect=events), patch.object(
        shm, 'load_model_to_gpu_sync', side_effect=lambda layers: order.append(('wait', layers))):
        loader.wait_for_weights_loaded([5])
        loader.wait_for_weights_loaded([6])
    assert order == [('event', 0), ('wait', [5]), ('event', 1),
                     ('event', 2), ('wait', [6]), ('event', 3)]
    assert loader.relay_weight_wait_events == [(events[0], events[1]), (events[2], events[3])]
    assert loader.relay_weight_wait_seconds >= 0


def test_ordinary_weight_fence_does_not_allocate_timing_events():
    with patch.object(shm.torch.cuda, 'Event') as event, patch.object(
        shm, 'load_model_to_gpu_sync') as sync:
        make_loader().wait_for_weights_loaded([1])
    sync.assert_called_once_with([1])
    event.assert_not_called()


def test_host_weight_timer_keeps_fences_without_cuda_timing():
    loader = make_loader()
    loader.profile_weight_wait = True
    with patch.object(shm.torch.cuda, 'Event') as event, patch.object(
        shm, 'check_model_load_state') as readiness, patch.object(
        shm, 'load_model_to_gpu_sync') as sync, patch.object(
        shm.time, 'perf_counter', side_effect=[1.0, 1.25, 2.0, 2.5]):
        loader.wait_for_weights_loaded([0])
        loader.wait_for_weights_loaded([1])
    assert sync.call_args_list == [call([0]), call([1])]
    assert loader.relay_weight_wait_seconds == 0.75
    event.assert_not_called()
    readiness.assert_not_called()


def test_completed_layer_does_not_record_an_idle_gap():
    loader = make_loader()
    loader.relay_weight_wait_events = []
    with patch.object(shm, 'check_model_load_state', return_value=True), patch.object(
        shm.torch.cuda, 'Event') as event, patch.object(shm, 'load_model_to_gpu_sync') as sync:
        loader.wait_for_weights_loaded([5])
    sync.assert_not_called()
    event.assert_not_called()
    assert loader.relay_weight_wait_events == []


def test_weight_wait_does_not_change_global_dtype():
    loader = make_loader()
    with patch.object(shm, 'set_default_torch_dtype') as dtype, patch.object(
        shm, 'load_model_to_gpu_sync') as sync:
        loader.wait_for_weights_loaded([5])
    dtype.assert_not_called()
    sync.assert_called_once_with([5])


def test_runtime_manifest_initializes_gpu_layout_without_sllm_index(tmp_path):
    from shm_manager import torch as shared
    index = {"model.layers.0.self_attn.o_proj.weight":
             [0, 2048, [32, 32], [32, 1], "torch.bfloat16"]}
    (tmp_path / "shm_weights.json").write_text(json.dumps(
        {"version": 1, "cpu_tp_size": 1, "tensor_index": index}))
    manager = Mock()
    with patch.object(shared.torch.cuda, "current_device", return_value=0), patch.object(
        shared, "ModelManager", return_value=manager,
    ), patch.object(shared, "model_info", shared.ModelInfo()), patch.object(
        shared, "get_pinned_shared_memory_ptrs", return_value={0: 1234},
    ) as pin:
        shared.init_model_info(str(tmp_path), 0, 1, 1, 4, 2, 8)
        assert shared.model_info.tensor_meta_index[
            "model.layers.0.self_attn.o_proj.weight"][1] == [32, 32]
        manager.generate_tensor_metadata.assert_called_once()
        pin.assert_called_once_with({0: str(tmp_path / "rank_0")}, {0: 2048}, True)
    assert not (tmp_path / "tensor_index.json").exists()


def test_runtime_manifest_rejects_cpu_tp_mismatch(tmp_path):
    from shm_manager import torch as shared
    (tmp_path / "shm_weights.json").write_text(json.dumps(
        {"version": 1, "cpu_tp_size": 2, "tensor_index": {}}))
    with patch.object(shared.torch.cuda, "current_device", return_value=0), patch.object(
        shared, "ModelManager", return_value=Mock(),
    ), patch.object(shared, "model_info", shared.ModelInfo()):
        with pytest.raises(ValueError, match="CPU TP size mismatch"):
            shared.init_model_info(str(tmp_path), 0, 1, 1, 4, 2, 8)


def test_runtime_manifest_rejects_missing_producer_before_pinning(tmp_path):
    from shm_manager import torch as shared
    (tmp_path / "shm_weights.json").write_text(json.dumps(
        {"version": 1, "cpu_tp_size": 1, "tensor_index": {},
         "segments": {"0": {"inode": -1, "size": 2048}}}))
    with patch.object(shared.torch.cuda, "current_device", return_value=0), patch.object(
        shared, "ModelManager", return_value=Mock(),
    ), patch.object(shared, "model_info", shared.ModelInfo()), patch.object(
        shared, "get_pinned_shared_memory_ptrs",
    ) as pin:
        with pytest.raises(ValueError, match="missing segment"):
            shared.init_model_info(str(tmp_path), 0, 1, 1, 4, 2, 8)
        pin.assert_not_called()
