# SPDX-License-Identifier: Apache-2.0
"""CPU/GPU relay metadata. Keep this module identical in both engines."""


def layerwise_load_order(allocated_layers, execution_layers):
    """Load the current GPU assignment first, then recover towards layer 0.

    Allocation covers the cold and final PP ranges; that union does not
    determine which layers are on the first batch's critical path.
    """
    allocated = set(allocated_layers)
    execution = set(execution_layers)
    if not execution <= allocated:
        raise ValueError("Execution layers lack allocated weight storage")
    return sorted(execution) + sorted(allocated - execution, reverse=True)


def ready_suffix_boundary(total_layers, check_ready):
    """Find the smallest suffix whose copies and unpacking have completed.

    Readiness may contain gaps, so checking one layer is insufficient.
    Suffix predicates are monotone even when layers finish out of order.
    Completion must be monotone for the duration of this query: callers
    must not resubmit writes into already published model weights.
    """
    if total_layers < 0:
        raise ValueError("Invalid layer count")
    low, high = 0, total_layers
    while low < high:
        middle = (low + high) // 2
        if check_ready(list(range(middle, total_layers))):
            high = middle
        else:
            low = middle + 1
    return low


def layerwise_load_budget(gpu_compute_seconds, layer_transfer_seconds,
                          remaining_layers):
    """Layers that fit in a measured GPU compute window, rounded down."""
    import math
    if remaining_layers < 0:
        raise ValueError("Invalid remaining layer count")
    if (not math.isfinite(gpu_compute_seconds) or gpu_compute_seconds < 0
            or not math.isfinite(layer_transfer_seconds)
            or layer_transfer_seconds <= 0):
        raise ValueError("Invalid layerwise timing sample")
    return min(remaining_layers,
               math.floor(gpu_compute_seconds / layer_transfer_seconds))


def next_layerwise_boundary(progress, current_boundary, total_layers):
    """Return common completed boundary, loading budget and planned prefix."""
    if not progress or not 0 <= current_boundary <= total_layers:
        raise ValueError("Invalid layerwise progress or current boundary")
    boundaries = [p["ready_boundary"] for p in progress]
    if any(not 0 <= boundary <= total_layers for boundary in boundaries):
        raise ValueError("Invalid tensor-parallel readiness boundary")
    # A layer is usable only after every tensor-parallel rank has finished
    # copying and unpacking it.  Taking the maximum admits a suffix that is
    # ready on one rank but still stale on another, which is especially
    # visible when a request spans multiple chunked-prefill iterations.
    ready = min(boundaries)
    samples = all(p["gpu_compute_seconds"] is not None
                  and p["layer_transfer_seconds"] is not None for p in progress)
    budget = 0
    if samples and ready:
        budget = layerwise_load_budget(
            min(p["gpu_compute_seconds"] for p in progress),
            max(p["layer_transfer_seconds"] for p in progress), ready)
    # A prior prediction is an ownership promise. Late copies may force a
    # fence but must never make the CPU execute GPU-owned history again.
    return ready, budget, min(current_boundary, ready - budget)


def build_relay_step(req_ids, num_scheduled_tokens, block_ids, boundary,
                     gpu_loading_layers):
    """Publish requests in the order used to produce hidden-state rows."""
    counts = {rid: num_scheduled_tokens[rid] for rid in req_ids}
    if set(counts) != set(num_scheduled_tokens):
        raise ValueError("CPU relay batch does not match scheduled requests")
    if boundary < 0 or any(count <= 0 for count in counts.values()):
        raise ValueError("Invalid CPU relay boundary or token count")
    return {
        "req_to_block_ids": {rid: block_ids[rid] for rid in counts},
        "req_to_num_scheduled_tokens": counts,
        "total_num_scheduled_tokens": sum(counts.values()),
        "shm_computed_layers": boundary,
        "gpu_loading_layers": gpu_loading_layers,
    }


def hidden_state_spans(step, gpu_req_ids, gpu_counts):
    """Return coalesced (host row, device row, row count) direct-copy spans."""
    counts = step["req_to_num_scheduled_tokens"]
    if (any(count <= 0 for count in counts.values())
            or sum(counts.values()) != step["total_num_scheduled_tokens"]):
        raise ValueError("Invalid CPU relay total token count")
    if counts != gpu_counts or set(gpu_req_ids) != set(counts):
        raise ValueError("CPU/GPU relay scheduled tokens differ")
    offsets = {}
    source = 0
    for rid, count in counts.items():
        offsets[rid] = source
        source += count
    spans = []
    target = 0
    for rid in gpu_req_ids:
        count = counts[rid]
        source = offsets[rid]
        if spans and spans[-1][0] + spans[-1][2] == source:
            host, device, length = spans[-1]
            spans[-1] = (host, device, length + count)
        elif count:
            spans.append((source, target, count))
        target += count
    return spans
