"""PP must not reuse CPU first-chunk hidden states for later chunks."""
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.engine.core import EngineCore


def test_drain_first_chunk_then_recover_before_scheduling_next_chunk():
    first_chunk = SimpleNamespace(total_num_scheduled_tokens=4096)
    first_output = object()
    core = SimpleNamespace(
        batch_queue=Queue(maxsize=2), weight_ready=False,
        vllm_config=SimpleNamespace(
            load_config=SimpleNamespace(load_format='shm', lazy_load=True,
                                        load_method='layerwise'),
            parallel_config=SimpleNamespace(pipeline_parallel_size=2)),
        _saw_nonempty_batch=True, _transition_complete=False,
        _transition_weights_loading=True, _cold_start_partition=[[8], [9]],
        step_cnt=1, scheduler=Mock(), model_executor=Mock(),
        _execute_cold_start_transition=Mock(),
        _annotate_load_weights_finished=Mock(),
        _unwrap_execute_model_handle=Mock(return_value=first_output),
        execute_model_with_error_logging=lambda fn, arg: fn(arg))
    core.batch_queue.put((object(), first_chunk))
    core.scheduler.update_from_output.return_value = {'first': first_output}
    core.model_executor.collective_rpc.return_value = [False, False]

    # A free queue slot must not admit the second 4096-token chunk.
    outputs, dispatched = EngineCore.step_with_batch_queue(core)
    assert outputs == {'first': first_output}
    assert not dispatched
    core.scheduler.schedule.assert_not_called()
    core._execute_cold_start_transition.assert_not_called()
    assert core.batch_queue.empty()

    events = []
    def recover():
        events.append('recover')
        core._transition_complete = True
    def schedule():
        events.append('schedule')
        return SimpleNamespace(total_num_scheduled_tokens=4096,
                               kv_connector_metadata=None)
    core._execute_cold_start_transition.side_effect = recover
    core.scheduler.schedule.side_effect = schedule
    EngineCore.step_with_batch_queue(core)
    assert events == ['recover', 'schedule']
    core._execute_cold_start_transition.assert_called_once()
