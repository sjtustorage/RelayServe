from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_relay_step_uses_explicit_presence_and_real_layer_boundary():
    runner = SimpleNamespace(pp_size=1, pp_rank=0, step_cnt=0,
        shm_scheduler_output=None, start_computing_layer=0,
        _update_model_layer_range=Mock())
    for boundary in (8, 5, 2, 0):
        output = {'shm_computed_layers': boundary}
        GPUModelRunner.update_shm_scheduler_output(runner, output)
        assert runner.shm_scheduler_output is output
        assert runner.start_computing_layer == boundary
    with pytest.raises(ValueError, match='must not increase'):
        GPUModelRunner.update_shm_scheduler_output(runner,
            {'shm_computed_layers': 1})
    GPUModelRunner.update_shm_scheduler_output(runner, None)
    assert runner.shm_scheduler_output is None
    assert runner.start_computing_layer == 0
    runner._update_model_layer_range.assert_called_with(start=0)


def test_relay_boundary_is_independent_of_execution_range():
    runner = SimpleNamespace(pp_size=1, pp_rank=0, step_cnt=1,
        shm_scheduler_output={'shm_computed_layers': 8},
        start_computing_layer=0, _update_model_layer_range=Mock())
    GPUModelRunner.update_shm_scheduler_output(
        runner, {'shm_computed_layers': 5})
    assert runner.start_computing_layer == 5


@pytest.mark.parametrize('completed', [True, False])
def test_compute_budget_excludes_only_cuda_idle_weight_gaps(monkeypatch, completed):
    from shm_manager import torch as shared
    monkeypatch.setattr(shared, 'check_model_load_state', lambda layers: True)
    monkeypatch.setattr(shared, 'model_info', SimpleNamespace(manager=SimpleNamespace(
        completed_layer_load_records=lambda: [(0, 100, .01), (1, 100, .02),
                                              (4, 1000, 1.)])))
    end = SimpleNamespace(query=lambda: completed)
    runner = SimpleNamespace(pp_size=1, total_layers=4,
        _relay_compute_start=SimpleNamespace(elapsed_time=lambda _: 150.),
        _relay_compute_end=end,
        _relay_weight_idle_events=[
            (SimpleNamespace(elapsed_time=lambda _: 20.), object()),
            (SimpleNamespace(elapsed_time=lambda _: 30.), object())])
    result = GPUModelRunner.get_ready_layerwise_boundary(runner)
    assert result['ready_boundary'] == 0
    assert result['layer_transfer_seconds'] == .015
    assert result['gpu_compute_seconds'] == (pytest.approx(.1) if completed else None)


def test_pp_prefix_readiness_checks_only_stage_zero_execution_range(monkeypatch):
    from shm_manager import torch as shared
    queried = []
    def ready(layers):
        queried.append(layers)
        return min(layers) >= 2
    monkeypatch.setattr(shared, 'check_model_load_state', ready)
    monkeypatch.setattr(shared, 'model_info', SimpleNamespace(manager=SimpleNamespace(
        completed_layer_load_records=lambda: [(0, 100, .01), (6, 100, 1.),
                                              (8, 1000, 2.)])))
    runner = SimpleNamespace(pp_size=2, pp_rank=0, total_layers=8,
                             _cold_execution_ranges=[(4, 6), (6, 8)])
    result = GPUModelRunner.get_ready_layerwise_boundary(runner)
    assert result['ready_boundary'] == 2
    assert result['layer_transfer_seconds'] == .01
    assert all(max(layers) < 6 for layers in queried)
    runner.pp_rank = 1
    assert GPUModelRunner.get_ready_layerwise_boundary(runner) is None
