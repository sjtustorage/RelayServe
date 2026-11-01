"""
Export trace.pkl to a human-readable text file (JSON Lines).

Each line is one request: arrival time (seconds), model id, prompt (chat messages as JSON).

Usage:
  python export_trace_txt.py                    # trace.pkl -> trace_preview.jsonl (first 500 rows)
  python export_trace_txt.py trace.pkl out.txt 10000
"""

import json
import pickle
import sys


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "trace.pkl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "trace_preview.jsonl"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    with open(src, "rb") as f:
        reqs = pickle.load(f)

    n = len(reqs)
    with open(dst, "w", encoding="utf-8") as w:
        for i in range(n):
            arrival, model, prompt = reqs[i]
            line = {
                "idx": i,
                "arrival_s": float(arrival),
                "model": model,
                "prompt": prompt,
            }
            w.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Wrote {n} of {len(reqs)} requests to {dst}")
    print(f"Format: one JSON object per line (JSONL); open with any text editor or `head -n 5 {dst}`")


if __name__ == "__main__":
    main()
