"""
Summarize trace.pkl: per-model request counts and arrival-time stats.

Usage:
  python stats_trace_pkl.py [trace.pkl]

If path is omitted, uses trace.pkl in the current directory.
"""

import pickle
import sys
from collections import defaultdict

import numpy as np


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "trace.pkl"
    with open(path, "rb") as f:
        reqs = pickle.load(f)

    by_model = defaultdict(list)
    for row in reqs:
        arrival, model_id, _prompt = row
        by_model[model_id].append(float(arrival))

    total = len(reqs)
    print(f"File: {path}")
    print(f"Total requests: {total}")
    print(f"Distinct model ids: {len(by_model)}\n")

    # Sort by count descending
    rows = []
    for model_id, times in by_model.items():
        arr = np.array(times, dtype=np.float64)
        arr.sort()
        n = len(arr)
        rows.append(
            (
                model_id,
                n,
                float(arr.min()),
                float(arr.max()),
                float(arr.mean()),
                float(np.median(arr)),
                float(arr[-1] - arr[0]) if n > 1 else 0.0,
                n / (arr[-1] - arr[0]) if n > 1 and (arr[-1] > arr[0]) else 0.0,
            )
        )

    rows.sort(key=lambda r: -r[1])

    hdr = (
        f"{'model_id':<48} {'count':>10} {'min_s':>14} {'max_s':>14} "
        f"{'span_s':>14} {'qps*':>10} {'pct':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for model_id, n, tmin, tmax, mean_t, median_t, span, approx_qps in rows:
        pct = 100.0 * n / total
        mid = model_id if len(model_id) <= 47 else model_id[:44] + "..."
        print(
            f"{mid:<48} {n:>10} {tmin:>14.4g} {tmax:>14.4g} "
            f"{span:>14.4g} {approx_qps:>10.4g} {pct:>6.2f}%"
        )

    print("\n* qps ≈ count / (max_arrival - min_arrival) for that model (not global interleaving).")
    print("  For workload-wide rate, sort all requests by time first.")


if __name__ == "__main__":
    main()
