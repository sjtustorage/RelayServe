import pytest
from vllm.v1.relay_step import build_relay_step, hidden_state_spans


def test_actual_batch_order_overrides_scheduler_order():
    counts = {"prefill": 65, "decode": 1}
    step = build_relay_step(["decode", "prefill"], counts,
                           {"decode": [1], "prefill": [2, 3, 4]}, 5, 2)
    assert list(step["req_to_num_scheduled_tokens"]) == ["decode", "prefill"]
    assert step["total_num_scheduled_tokens"] == 66
    assert hidden_state_spans(step, ["prefill", "decode"], counts) == [
        (1, 0, 65), (0, 65, 1)]
    assert hidden_state_spans(step, ["decode", "prefill"], counts) == [(0, 0, 66)]


def test_changed_token_counts_rejected_before_copy():
    step = build_relay_step(["a"], {"a": 3}, {"a": [1]}, 2, 0)
    with pytest.raises(ValueError, match="scheduled tokens"):
        hidden_state_spans(step, ["a"], {"a": 2})


def test_missing_scheduled_request_rejected():
    with pytest.raises(ValueError, match="scheduled requests"):
        build_relay_step(["a"], {"a": 1, "b": 2}, {"a": [1]}, 2, 0)


def test_invalid_row_extent_rejected():
    with pytest.raises(ValueError, match="token count"):
        build_relay_step(["a"], {"a": -1}, {"a": [1]}, 2, 0)
    step = build_relay_step(["a"], {"a": 3}, {"a": [1]}, 2, 0)
    step["total_num_scheduled_tokens"] = 2
    with pytest.raises(ValueError, match="total token count"):
        hidden_state_spans(step, ["a"], {"a": 3})
