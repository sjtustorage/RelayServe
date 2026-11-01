from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from vllm.v1.core.sched import scheduler as module


def test_nonrelay_waiting_head_is_removed_before_next_iteration(monkeypatch):
    class NextIteration(Exception):
        pass
    request = SimpleNamespace(request_id='ordinary')
    class Waiting:
        popped = False
        def __bool__(self):
            if self.popped:
                raise NextIteration
            return True
        def peek_request(self):
            return request
        def pop_request(self):
            self.popped = True
            return request
    waiting = Waiting()
    skipped = Mock()
    monkeypatch.setattr(module, 'create_request_queue', lambda policy: skipped)
    core = SimpleNamespace(waiting=waiting, running=[],
        max_num_scheduled_tokens=8, max_num_encoder_input_tokens=0,
        max_num_running_reqs=2, lora_config=None, policy=None)
    with pytest.raises(NextIteration):
        module.Scheduler.schedule_for_layerwise_cold_start(core,['relay'])
    skipped.prepend_request.assert_called_once_with(request)
