"""A delayed relay request must not block the engine's input processing."""
from queue import Queue
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

from vllm.v1.engine.core import EngineCore, EngineCoreProc


def waiting_core():
    core = SimpleNamespace(
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format='shm', lazy_load=True, load_method='tokenwise')),
        scheduler=Mock(), cold_start_request_id='relay',
        cold_start_request_come=False, engines_running=False)
    core.scheduler.has_requests.return_value = True
    core._waiting_for_tokenwise_request = MethodType(
        EngineCore._waiting_for_tokenwise_request, core)
    return core


def test_step_yields_to_input_loop_while_relay_is_missing():
    core = waiting_core()
    # There is an ordinary request, but the designated relay arrives later.
    # The old implementation slept indefinitely inside step at this point.
    assert EngineCore.step(core) == ({}, False)
    core.scheduler.schedule.assert_not_called()


def test_waiting_input_loop_handles_utility_then_relay():
    core = waiting_core()
    core.input_queue = Queue()
    core.input_queue.put(('utility',))
    core.input_queue.put(('relay',))
    handled = []

    def handle(kind):
        handled.append(kind)
        if kind == 'relay':
            core.cold_start_request_come = True

    core._handle_client_request = handle
    EngineCoreProc._process_input_queue(core)
    assert handled == ['utility', 'relay']
    assert not core._waiting_for_tokenwise_request()


def test_abort_missing_relay_releases_ordinary_requests():
    core = waiting_core()
    EngineCore.abort_requests(core, ['relay'])
    assert not core._waiting_for_tokenwise_request()
    core.scheduler.finish_requests.assert_called_once()
