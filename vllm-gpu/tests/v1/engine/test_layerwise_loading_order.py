"""Layerwise steps must preserve readiness until CPU metadata is received."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.engine.core import EngineCore


class ScrambledSet(set):
    def __iter__(self):
        return iter(sorted(set.copy(self), reverse=True))


@pytest.mark.parametrize('initial,cold', [
    ([4, 5], False),
    ([4, 5, 6, 7], True),
    ([1, 2, 3, 4, 5, 6, 7], True),
])
def test_step_does_not_submit_fixed_layer_batches_or_claim_completion(initial, cold):
    executor = Mock()
    core = SimpleNamespace(
        scheduler=Mock(), model_executor=executor,
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format='shm', lazy_load=True, load_method='layerwise')),
        loaded_layers=ScrambledSet(initial), target_layers=set(range(8)),
        cold_start=cold, step_cnt=1, _waiting_for_tokenwise_request=lambda: False)
    # Layerwise loading starts with the execution suffix and reverse prefix
    # during initialization. A step consumes CPU metadata before calculating
    # its next boundary from completed events and measured compute/transfer.
    # Stop at real SHM reception, which
    # needs a live CPU peer and is verified by the integration harness.
    class ReachedReceive(Exception):
        pass

    def receive():
        raise ReachedReceive

    core.recv_shm_scheduler_output = receive
    with pytest.raises(ReachedReceive):
        EngineCore.step(core)
    executor.lazy_load_model.assert_not_called()
    assert core.loaded_layers == set(initial)


@pytest.mark.parametrize('loaded,boundary', [(list(range(8)), None), ([4, 5, 6, 7], 0)])
def test_complete_gpu_clears_previous_cpu_step(loaded, boundary):
    core = SimpleNamespace(
        scheduler=Mock(), model_executor=Mock(),
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format='shm', lazy_load=True, load_method='layerwise')),
        loaded_layers=set(loaded), target_layers=set(range(8)),
        _next_cpu_boundary=boundary,
        cold_start=True, step_cnt=4, _waiting_for_tokenwise_request=lambda: False,
        _cpu_relay_step_active=True,
        update_shm_scheduler_output=Mock())
    class ReachedSchedule(Exception):
        pass
    core.scheduler.schedule.side_effect = ReachedSchedule
    with pytest.raises(ReachedSchedule):
        EngineCore.step(core)
    core.update_shm_scheduler_output.assert_called_once_with(None)
    core.model_executor.lazy_load_model.assert_not_called()
    assert not core._cpu_relay_step_active
