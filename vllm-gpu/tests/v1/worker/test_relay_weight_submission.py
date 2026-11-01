from contextlib import nullcontext
"""Control-plane submission must preserve in-flight model storage."""
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import pytest

from vllm.model_executor.model_loader import shared_memory_loader as module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker import gpu_worker as worker_module


def test_recovery_does_not_resubmit_pending_or_completed_layers(monkeypatch):
    loader = object.__new__(module.SharedMemoryLoader)
    loader._submitted_layers = {0, 1, 2}
    loader._layer_idxes = [0, 1, 2, 3]
    loader._weight_dtype = torch.bfloat16
    submit = Mock()
    monkeypatch.setattr(module, 'load_model_to_gpu', submit)
    loader.launch_load_weights(layer_idxes=[2, 1, 3, 3])
    submit.assert_called_once_with([3])
    loader.launch_load_weights()
    submit.assert_called_once_with([3])


def test_uniform_readiness_covers_already_submitted_assignment():
    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format='shm'),
        model_config=SimpleNamespace(
            get_layers_start_end_indices=lambda _: (4, 8)),
        parallel_config=None, loaded_layers={4, 5, 6, 7},
        model_loader=SimpleNamespace(launch_load_weights=Mock()))
    GPUModelRunner.start_loading_uniform_weights(runner)
    runner.model_loader.launch_load_weights.assert_not_called()
    # Absence of new submissions is not proof of completed CUDA work.
    assert runner._pending_uniform_layers == [4, 5, 6, 7]


@pytest.mark.parametrize('fails', [False, True])
@pytest.mark.parametrize('throttle', [False, True])
def test_warmup_throttle_ends_before_first_inference(fails, throttle):
    events = []

    def warmup():
        events.append('warmup')
        if fails:
            raise RuntimeError('warmup failed')

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format='shm', load_async=throttle)),
        model_config=SimpleNamespace(enforce_eager=True),
        model_runner=SimpleNamespace(
            update_module_loading_state=lambda value: events.append(value)),
        _compile_or_warm_up_model=warmup)
    if fails:
        with pytest.raises(RuntimeError, match='warmup failed'):
            Worker.compile_or_warm_up_model(worker)
    else:
        Worker.compile_or_warm_up_model(worker)
    assert events == ([True, 'warmup', False] if throttle else ['warmup'])


def test_lazy_trace_profiles_only_first_warmup(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    monkeypatch.setenv('VLLM_PROFILE_LAZY_TRACE', str(tmp_path))
    monkeypatch.setattr(worker_module, '_lazy_warmup_trace_pids', set())
    profiler = MagicMock()
    profiler.__enter__.return_value = profiler
    start = Mock(return_value=profiler)
    monkeypatch.setattr(worker_module.torch.profiler, 'profile', start)
    events = []
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format='shm', load_async=True)),
        model_config=SimpleNamespace(enforce_eager=True),
        model_runner=SimpleNamespace(
            update_module_loading_state=lambda value: events.append(value)),
        _compile_or_warm_up_model=lambda: events.append('warmup'))
    Worker.compile_or_warm_up_model(worker)
    # A model switch replaces Worker while retaining the rank process.
    replacement = SimpleNamespace(**vars(worker))
    Worker.compile_or_warm_up_model(replacement)
    start.assert_called_once()
    profiler.export_chrome_trace.assert_called_once()
    assert events == [True, 'warmup', False, True, 'warmup', False]


@pytest.mark.parametrize('fails', [False, True])
def test_host_warmup_profile_preserves_throttle_and_exception(
        monkeypatch, tmp_path, fails):
    import pstats

    monkeypatch.setenv('VLLM_PROFILE_LAZY_HOST', str(tmp_path))
    monkeypatch.delenv('VLLM_PROFILE_LAZY_TRACE', raising=False)
    monkeypatch.delenv('VLLM_PROFILE_LAZY_COMPLETION', raising=False)
    events = []

    def warmup():
        events.append('warmup')
        if fails:
            raise RuntimeError('warmup failed')

    synchronize = Mock(side_effect=AssertionError('extra CUDA fence'))
    monkeypatch.setattr(worker_module.torch.cuda, 'synchronize', synchronize)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format='shm', load_async=True)),
        model_config=SimpleNamespace(enforce_eager=True),
        model_runner=SimpleNamespace(
            update_module_loading_state=lambda value: events.append(value)),
        _compile_or_warm_up_model=warmup)
    if fails:
        with pytest.raises(RuntimeError, match='warmup failed'):
            Worker.compile_or_warm_up_model(worker)
    else:
        Worker.compile_or_warm_up_model(worker)
    assert events == [True, 'warmup', False]
    profiles = list(tmp_path.glob('warmup-*.prof'))
    assert len(profiles) == 1
    assert any(key[2] == 'warmup' for key in pstats.Stats(str(profiles[0])).stats)
    synchronize.assert_not_called()


@pytest.mark.parametrize('world_size', [1, 2, 4])
@pytest.mark.parametrize('all_gather', [False, True])
def test_relay_communication_preparation_uses_logits_collective(
        monkeypatch, world_size, all_gather):
    group = SimpleNamespace(world_size=world_size, all_gather=Mock(),
                            gather=Mock())
    monkeypatch.setattr(worker_module, 'get_tp_group', lambda: group)
    monkeypatch.setattr(worker_module.current_platform, 'use_all_gather',
                        lambda: all_gather)
    allocate = Mock(return_value=object())
    synchronize = Mock()
    monkeypatch.setattr(worker_module.torch, 'zeros', allocate)
    monkeypatch.setattr(worker_module.torch.cuda, 'synchronize', synchronize)
    probe_stream = SimpleNamespace(synchronize=Mock())
    make_stream = Mock(return_value=probe_stream)
    stream_context = Mock(side_effect=lambda _: nullcontext())
    monkeypatch.setattr(worker_module.torch.cuda, 'Stream', make_stream)
    monkeypatch.setattr(worker_module.torch.cuda, 'stream', stream_context)
    worker = SimpleNamespace(device=torch.device('cuda:0'),
                             model_config=SimpleNamespace(dtype=torch.bfloat16))
    Worker._warm_up_relay_tp_communication(worker)
    if world_size == 1:
        allocate.assert_not_called()
        synchronize.assert_not_called()
        make_stream.assert_not_called()
    else:
        allocate.assert_called_once_with((1, 1), device=worker.device,
                                         dtype=torch.bfloat16)
        collective = group.all_gather if all_gather else group.gather
        collective.assert_called_once_with(allocate.return_value)
        make_stream.assert_called_once_with(device=worker.device)
        stream_context.assert_called_once_with(probe_stream)
        probe_stream.synchronize.assert_called_once_with()
        synchronize.assert_not_called()
    unused = group.gather if all_gather else group.all_gather
    unused.assert_not_called()
