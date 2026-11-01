"""
Inspect trace.pkl as time moves forward: which model_id is invoked when.

Usage:
  python timeline_trace_pkl.py trace.pkl              # first 200 events + hourly summary
  python timeline_trace_pkl.py trace.pkl --events 500
  python timeline_trace_pkl.py trace.pkl --bucket 300 --buckets 48   # 5-min buckets

Or import:
  from timeline_trace_pkl import load_sorted_requests, format_timeline_lines
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter
from typing import Any, Iterator, List, Tuple

RequestRow = Tuple[float, str, Any]


def load_sorted_requests(path: str) -> List[RequestRow]:
    with open(path, "rb") as f:
        reqs: List[RequestRow] = pickle.load(f)
    return sorted(reqs, key=lambda r: (r[0], r[1]))


def iter_timeline(path: str) -> Iterator[Tuple[int, float, str]]:
    """Yield (index_in_sorted_order, arrival_s, model_id)."""
    for i, row in enumerate(load_sorted_requests(path)):
        arrival, model_id, _ = row
        yield i, float(arrival), model_id


def format_timeline_lines(
    path: str,
    *,
    max_events: int = 200,
    start_at: int = 0,
) -> List[str]:
    """Human-readable lines: global time order -> model."""
    lines: List[str] = []
    for idx, (i, t, model_id) in enumerate(iter_timeline(path)):
        if idx < start_at:
            continue
        if len(lines) >= max_events:
            break
        lines.append(f"t={t:.6f}s  event#{i:>8}  model={model_id}")
    return lines


def summarize_buckets(
    path: str,
    *,
    bucket_seconds: float,
    max_buckets: int | None = None,
) -> List[str]:
    """For each time bucket, show request counts per model (sorted by count desc)."""
    reqs = load_sorted_requests(path)
    if not reqs:
        return ["(empty trace)"]

    t0 = reqs[0][0]
    lines: List[str] = []
    cur_bucket = -1
    counts: Counter[str] = Counter()

    def flush(b: int):
        if not counts:
            return
        top = counts.most_common(12)
        rest = sum(c for m, c in counts.items() if m not in dict(top[:12]))
        detail = ", ".join(f"{m}×{c}" for m, c in top)
        if rest:
            detail += f", …(+{rest} other models)"
        lines.append(
            f"[{b * bucket_seconds:.0f}s — {(b + 1) * bucket_seconds:.0f}s)  "
            f"total={sum(counts.values())}  {detail}"
        )

    for arrival, model_id, _ in reqs:
        b = int((float(arrival) - t0) // bucket_seconds)
        if b != cur_bucket:
            if cur_bucket >= 0:
                flush(cur_bucket)
            if max_buckets is not None and b >= max_buckets:
                lines.append(f"... truncated after {max_buckets} buckets")
                return lines
            cur_bucket = b
            counts = Counter()
        counts[model_id] += 1

    if counts:
        flush(cur_bucket)
    return lines


def print_timeline_report(
    path: str,
    *,
    max_events: int = 200,
    bucket_seconds: float = 3600.0,
    max_buckets: int = 24,
) -> None:
    """Print sorted event tail + per-bucket model mix (default: hourly for 24 buckets)."""
    reqs = load_sorted_requests(path)
    print(f"file={path}  total_requests={len(reqs)}")
    if not reqs:
        return
    t_min, t_max = reqs[0][0], reqs[-1][0]
    print(f"arrival_range: [{t_min:.6f}, {t_max:.6f}] span={t_max - t_min:.6f}s\n")

    print("--- first events in global time order ---")
    for line in format_timeline_lines(path, max_events=max_events):
        print(line)

    print(f"\n--- per {bucket_seconds:.0f}s window (from first arrival) ---")
    for line in summarize_buckets(
        path, bucket_seconds=bucket_seconds, max_buckets=max_buckets
    ):
        print(line)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: python timeline_trace_pkl.py <trace.pkl> [--events N] [--bucket SEC] [--buckets M]")
        sys.exit(1)

    path = args[0]
    max_events = 200
    bucket_s = 3600.0
    max_buckets = 24
    i = 1
    while i < len(args):
        if args[i] == "--events" and i + 1 < len(args):
            max_events = int(args[i + 1])
            i += 2
        elif args[i] == "--bucket" and i + 1 < len(args):
            bucket_s = float(args[i + 1])
            i += 2
        elif args[i] == "--buckets" and i + 1 < len(args):
            max_buckets = int(args[i + 1])
            i += 2
        else:
            i += 1

    print_timeline_report(
        path,
        max_events=max_events,
        bucket_seconds=bucket_s,
        max_buckets=max_buckets,
    )


if __name__ == "__main__":
    main()
