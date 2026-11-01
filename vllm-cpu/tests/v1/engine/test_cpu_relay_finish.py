import pytest
from types import SimpleNamespace
from unittest.mock import Mock
from concurrent.futures import Future
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.core import EngineCore


@pytest.mark.parametrize("load_format,lazy,method,serial", [
    ("shm", True, "layerwise", True),
    ("shm", True, "tokenwise", False),
    ("shm", False, "layerwise", False),
    ("auto", True, "layerwise", False),
])
def test_layerwise_cpu_uses_handshake_step_with_concurrent_executor(
        load_format, lazy, method, serial):
    core = SimpleNamespace(
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format=load_format, lazy_load=lazy, load_method=method)),
        model_executor=SimpleNamespace(max_concurrent_batches=4))
    EngineCore._initialize_batch_queue(core)
    assert (core.batch_queue is None) == serial
    assert core.batch_queue_size == (1 if serial else 4)


def test_cpu_waits_for_executor_future_before_publishing():
    events = []

    class RecordingFuture(Future):
        def result(self, timeout=None):
            events.append("cpu_complete")
            return super().result(timeout)

    future = RecordingFuture()
    future.set_result(None)
    scheduler = SimpleNamespace(
        has_requests=lambda: True,
        schedule=lambda: SimpleNamespace(total_num_scheduled_tokens=4096),
        update_from_output=lambda *_: {})
    core = SimpleNamespace(
        scheduler=scheduler, step_cnt=0, total_layers=64,
        shm_gpu_loaded_layers=0, pp_cleanup_enabled=False,
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format="shm", lazy_load=True, load_method="layerwise")),
        model_executor=SimpleNamespace(
            execute_model=lambda _: future,
            publish_pending_relay_step=lambda: events.append("publish")),
        recv_shm_model_output=lambda: (
            events.append("gpu_feedback") or SimpleNamespace(
                shm_next_cpu_boundary=8, shm_gpu_loaded_layers=56)))
    core.execute_model_with_error_logging = lambda fn, out: (
        EngineCore.execute_model_with_error_logging(core, fn, out))
    EngineCore.step(core)
    assert events == ["cpu_complete", "publish", "gpu_feedback"]
    assert core.shm_gpu_loaded_layers == 56
    assert core.step_cnt == 1

def test_takeover_terminates_running_and_waiting_frontends():
    requests = {'r': SimpleNamespace(request_id='r',client_index=0),
                'w': SimpleNamespace(request_id='w',client_index=1)}
    scheduler = SimpleNamespace(requests=requests,abort_all_requests=Mock())
    core = SimpleNamespace(scheduler=scheduler)
    token = EngineCoreOutput(request_id='r',new_token_ids=[9707])
    outputs = {0: EngineCoreOutputs(outputs=[token])}
    EngineCore._finish_cpu_relay(core,outputs)
    assert token.new_token_ids == [9707]
    assert token.finish_reason == FinishReason.ABORT
    assert outputs[1].outputs[0].request_id == 'w'
    assert outputs[1].outputs[0].finished
    assert outputs[1].outputs[0].new_token_ids == []
    scheduler.abort_all_requests.assert_called_once_with(keep_kv_cache=True)


def test_pp_prefix_without_sampled_output_still_finishes_frontend():
    scheduler = SimpleNamespace(
        has_requests=lambda: True,
        schedule=lambda: SimpleNamespace(total_num_scheduled_tokens=65),
        requests={'r': SimpleNamespace(request_id='r', client_index=0)},
        abort_all_requests=Mock())
    core = SimpleNamespace(
        scheduler=scheduler, step_cnt=0,
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format='shm', load_method='layerwise', lazy_load=True)),
        shm_gpu_loaded_layers=0, total_layers=64,
        pp_cleanup_enabled=True, pp_first_output_received=False,
        model_executor=SimpleNamespace(execute_model=Mock(),
            publish_pending_relay_step=Mock()),
        execute_model_with_error_logging=lambda *args: None,
        update_load_method=Mock())
    core._finish_cpu_relay = lambda outputs, **kwargs: EngineCore._finish_cpu_relay(
        core, outputs, **kwargs)
    outputs, executed = EngineCore.step(core)
    assert executed
    assert outputs[0].outputs[0].request_id == 'r'
    assert outputs[0].outputs[0].finished
    scheduler.abort_all_requests.assert_called_once_with(keep_kv_cache=True)
    core.update_load_method.assert_called_once_with('cpu')


@pytest.mark.parametrize("completed", [63, 64])
def test_single_stage_acknowledged_takeover_releases_cpu_kv(completed):
    request = SimpleNamespace(request_id='r', client_index=0)
    scheduler = SimpleNamespace(
        has_requests=lambda: True,
        schedule=lambda: SimpleNamespace(total_num_scheduled_tokens=4096),
        requests={'r': request}, abort_all_requests=Mock(),
        update_from_output=lambda *_: {})
    response = SimpleNamespace(shm_next_cpu_boundary=0, shm_gpu_loaded_layers=completed)
    core = SimpleNamespace(
        scheduler=scheduler, step_cnt=0,
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format='shm', load_method='layerwise', lazy_load=True)),
        shm_gpu_loaded_layers=0, total_layers=64,
        pp_cleanup_enabled=False, pp_first_output_received=False,
        model_executor=SimpleNamespace(execute_model=Mock(),
            publish_pending_relay_step=Mock()),
        execute_model_with_error_logging=lambda *_: None,
        recv_shm_model_output=lambda: response, update_load_method=Mock())
    core._finish_cpu_relay = lambda outputs, **kwargs: EngineCore._finish_cpu_relay(
        core, outputs, **kwargs)
    outputs, executed = EngineCore.step(core)
    assert executed and outputs[0].outputs[0].finished
    scheduler.abort_all_requests.assert_called_once_with(keep_kv_cache=False)
    core.update_load_method.assert_called_once_with('cpu')
