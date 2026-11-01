import pytest
from vllm.v1.relay_step import (build_relay_step, hidden_state_spans,
                                ready_suffix_boundary, layerwise_load_budget,
                                layerwise_load_order, next_layerwise_boundary)


def test_tp_budget_uses_common_suffix_and_slowest_timings():
    progress = [dict(ready_boundary=7, gpu_compute_seconds=.1,
                     layer_transfer_seconds=.01),
                dict(ready_boundary=9, gpu_compute_seconds=.12,
                     layer_transfer_seconds=.03)]
    # The suffix is usable only when every TP rank is ready.  The slower
    # transfer rank also limits the number of layers that can be loaded.
    assert next_layerwise_boundary(progress, 10, 16) == (7, 3, 4)
    # Prior predicted boundary must not increase if copies run late.
    assert next_layerwise_boundary(progress, 3, 16) == (7, 3, 3)


def test_missing_measurement_does_not_invent_loading_budget():
    progress = [dict(ready_boundary=7, gpu_compute_seconds=None,
                     layer_transfer_seconds=.01)]
    assert next_layerwise_boundary(progress, 10, 16) == (7, 0, 7)


def test_complete_weights_end_cpu_assistance_without_predicted_steps():
    progress = [dict(ready_boundary=0, gpu_compute_seconds=None,
                     layer_transfer_seconds=None)]
    assert next_layerwise_boundary(progress, 8, 16) == (0, 0, 0)


def test_first_suffix_precedes_reverse_prefix_recovery():
    assert layerwise_load_order(range(8), range(3, 8)) == [3, 4, 5, 6, 7, 2, 1, 0]


def test_asymmetric_pp_storage_union_keeps_execution_first():
    # Cold assignment 8..11 and final uniform assignment 4..9.
    assert layerwise_load_order(range(4, 12), range(8, 12)) == [8, 9, 10, 11, 7, 6, 5, 4]


def test_empty_cold_stage_loads_final_range_in_reverse():
    assert layerwise_load_order(range(4, 8), []) == [7, 6, 5, 4]


def test_invalid_priority_rejected_before_native_submission():
    with pytest.raises(ValueError, match='allocated'):
        layerwise_load_order(range(4), [4])


@pytest.mark.parametrize('compute, transfer, remaining, expected', [
    (0.1, 0.03, 8, 3), (0.01, 0.03, 8, 0),
    (0.3, 0.03, 4, 4), (0, 0.03, 8, 0),
])
def test_loading_budget_uses_measured_window(compute, transfer, remaining, expected):
    assert layerwise_load_budget(compute, transfer, remaining) == expected


@pytest.mark.parametrize('ready, expected', [
    (set(), 8), (set(range(8)), 0), (set(range(5, 8)), 5),
    ({0, 1, 3, 5, 6, 7}, 5), ({0, 1, 2, 3, 4, 5, 7}, 7),
])
def test_boundary_requires_complete_contiguous_suffix(ready, expected):
    queries = []
    def complete(layers):
        queries.append(layers)
        return set(layers) <= ready
    assert ready_suffix_boundary(8, complete) == expected
    assert len(queries) <= 4


def test_empty_model_needs_no_native_query():
    assert ready_suffix_boundary(0, lambda _: pytest.fail('queried')) == 0


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
