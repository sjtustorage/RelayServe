from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_hidden_written_before_metadata_signal_and_no_duplicate_publication():
    events = []
    hidden, schedule = object(), object()
    runner = SimpleNamespace(_pending_hidden_states=hidden,
                             _pending_scheduler_output=(schedule, 3))
    def write_hidden(value):
        assert value is hidden
        events.append('hidden')
    def signal_metadata(value, layers):
        assert value is schedule and layers == 3
        assert runner._pending_hidden_states is None
        assert events == ['hidden']
        events.append('metadata')
    runner.send_shm_hidden_states = write_hidden
    runner.send_shm_scheduler_output = signal_metadata
    GPUModelRunner.publish_pending_relay_step(runner)
    assert runner._pending_scheduler_output is None
    GPUModelRunner.publish_pending_relay_step(runner)
    assert events == ['hidden', 'metadata']


def test_hidden_write_failure_does_not_notify_gpu_or_discard_pending_state():
    hidden = object()
    runner = SimpleNamespace(_pending_hidden_states=hidden,
                             _pending_scheduler_output=(object(), 0))
    def fail(_):
        raise RuntimeError('hidden write failed')
    runner.send_shm_hidden_states = fail
    runner.send_shm_scheduler_output = lambda *_: pytest.fail('metadata published prematurely')
    with pytest.raises(RuntimeError, match='hidden write'):
        GPUModelRunner.publish_pending_relay_step(runner)
    assert runner._pending_hidden_states is hidden
    assert runner._pending_scheduler_output is not None
