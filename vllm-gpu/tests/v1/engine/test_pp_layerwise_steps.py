"""Serialized PP relay advances CPU ownership only after the current forward."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.engine.core import EngineCore


@pytest.mark.parametrize('uniform_ready,expected', [(False, 2), (True, 0)])
def test_pp_feedback_follows_completed_forward_and_recovery(uniform_ready, expected):
    events = []
    metadata = {'shm_computed_layers': 4, 'req_to_block_ids': {'r': [1]}}
    executor = Mock()
    def rpc(method, **kwargs):
        events.append(method)
        if method == 'check_uniform_weights_ready':
            return [uniform_ready, uniform_ready]
        if method == 'get_ready_layerwise_boundary':
            return [dict(ready_boundary=3, gpu_compute_seconds=.01,
                         layer_transfer_seconds=.01), None]
        return [None]
    executor.collective_rpc.side_effect = rpc
    def execute(schedule):
        events.append('forward')
        return object()
    executor.execute_model.side_effect = execute
    core = SimpleNamespace(
        scheduler=Mock(), model_executor=executor,
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format='shm', lazy_load=True, load_method='layerwise'),
            parallel_config=SimpleNamespace(pipeline_parallel_size=2)),
        total_layers=8, loaded_layers=set(range(4, 8)), target_layers=set(range(8)),
        cold_start=False, step_cnt=1, _waiting_for_tokenwise_request=lambda: False,
        _cold_start_partition=[[4, 5], [6, 7]], _transition_complete=False,
        recv_shm_scheduler_output=Mock(return_value=metadata),
        update_shm_scheduler_output=Mock(), _annotate_load_weights_finished=Mock(),
        execute_model_with_error_logging=lambda fn, arg: fn(arg))
    def recover():
        events.append('recover')
        core._transition_complete = True
        core.loaded_layers = core.target_layers.copy()
    core._execute_cold_start_transition = Mock(side_effect=recover)
    core.send_shm_model_output = Mock(side_effect=lambda _: events.append('feedback'))
    core.scheduler.schedule_for_layerwise_cold_start.return_value = SimpleNamespace(
        total_num_scheduled_tokens=128)
    EngineCore.step(core)
    assert core._next_cpu_boundary == expected
    assert events.index('forward') < events.index('check_uniform_weights_ready')
    assert events[-1] == 'feedback'
    assert core._execute_cold_start_transition.call_count == int(uniform_ready)
    if uniform_ready:
        assert events.index('recover') < events.index('feedback')
    else:
        assert core.loaded_layers == set(range(4, 8))


def test_serialized_recovery_without_batch_queue():
    executor = Mock()
    cache = SimpleNamespace(num_gpu_blocks=10)
    manager = Mock()
    core = SimpleNamespace(
        batch_queue=None, model_executor=executor,
        scheduler=SimpleNamespace(kv_cache_manager=manager),
        vllm_config=SimpleNamespace(cache_config=cache,
            load_config=SimpleNamespace(load_method='layerwise')),
        _cold_start_partition=[[4, 5], [6, 7]],
        target_layers=set(range(8)), loaded_layers=set(range(4, 8)),
        _transition_complete=False)
    EngineCore._execute_cold_start_transition(core)
    assert [call.args[0] for call in executor.collective_rpc.call_args_list] == [
        'sync_uniform_weights', 'execute_kv_migration']
    assert executor.collective_rpc.call_args_list[1].kwargs['args'] == (
        core._cold_start_partition, 'layerwise')
    manager.extend_blocks.assert_called_once_with(10)
    assert core._transition_complete
    assert core.loaded_layers == core.target_layers
