#!/usr/bin/env python3
"""CDF of input lengths (chat-template token count) for HumanEval, ShareGPT, LongBench.

Uses the same ``get_prompts(...)`` entry points as ``generate_trace.py``, i.e. the full
HumanEval test split, full LongBench (gov_report / qmsum / vcsum), and full ShareGPT train
unless you pass ``--sharegpt-max-conversations`` for a quicker dev run.

Uses the same truncation cap as ``prompts.py`` (default 32k; set ``TRACE_PROMPT_MAX_LEN``,
or ``TRACE_NO_TRUNCATE=1`` for no cap).

For the figure: by default lengths are clipped and x-axis is capped at **32k** tokens so
the plot matches a typical context window, even if ``TRACE_PROMPT_MAX_LEN`` is larger
(e.g. 50k+). Use ``--plot-max-tokens`` or ``--no-plot-cap`` to change.

Usage:
  python plot_dataset_input_length_cdf.py
  # writes dataset_input_len_cdf.png next to this script; x-axis default max 32768

  python plot_dataset_input_length_cdf.py --plot-max-tokens 65536
  python plot_dataset_input_length_cdf.py --no-plot-cap
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import redirect_stdout

# get_prompts_from_ds reads TRACE_PROMPT_MAX_LEN / TRACE_NO_TRUNCATE at runtime.

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTPUT = os.path.join(_SCRIPT_DIR, "dataset_input_len_cdf.png")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import matplotlib.pyplot as plt
import numpy as np
from modelscope import MsDataset
from modelscope.utils.constant import DownloadMode

from prompts import _max_context_tokens, _truncate_enabled, get_prompts, get_prompts_from_ds, tokenizer

# Same dataset ids as generate_trace.py
_DS_HUMANEVAL = "modelscope/humaneval"
_DS_LONGBENCH = "ZhipuAI/LongBench"
_DS_SHAREGPT = "gliang1001/ShareGPT_V3_unfiltered_cleaned_split"

# Default figure cap (independent of TRACE_PROMPT_MAX_LEN; aligns with typical 32k context).
_DEFAULT_PLOT_X_MAX_TOKENS = 32768


def _token_ids_len_from_apply_chat_template_result(enc) -> int:
    """``apply_chat_template(tokenize=True)`` may return list[int], BatchEncoding, dict, or (ids, mask)."""
    if enc is None:
        return 0
    if isinstance(enc, list):
        return len(enc)
    if isinstance(enc, tuple) and enc:
        # Some builds return (input_ids, attention_mask)
        ids = enc[0]
        if isinstance(ids, list):
            return len(ids)
        return int(ids.shape[0]) if hasattr(ids, "shape") else len(ids)
    if isinstance(enc, dict) and "input_ids" in enc:
        ids = enc["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return len(ids[0])
        return len(ids)
    ids = getattr(enc, "input_ids", None)
    if ids is not None:
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return len(ids[0])
        return len(ids)
    raise TypeError(f"Unexpected apply_chat_template return type {type(enc)!r}")  # pyright: ignore[reportUnreachable]


def _messages_token_len(messages: list[dict], plot_cap: int | None) -> int:
    if not messages:
        return 0
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    L = _token_ids_len_from_apply_chat_template_result(enc)
    if plot_cap is not None:
        return int(min(L, plot_cap))
    if _truncate_enabled():
        return int(min(L, _max_context_tokens()))
    return L


def _lengths_via_get_prompts(dataset_id: str, plot_cap: int | None) -> list[int]:
    """Same prompt list as ``generate_trace.py`` via ``get_prompts``."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        prompts = get_prompts(dataset_id)
    return [_messages_token_len(p, plot_cap) for p in prompts]


def _sharegpt_lengths_capped(max_conversations: int, plot_cap: int | None) -> list[int]:
    ds = MsDataset.load(
        _DS_SHAREGPT,
        subset_name="default",
        split="train",
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        trust_remote_code=True,
    )

    def _iter():
        for i, row in enumerate(ds):
            if i >= max_conversations:
                break
            yield row

    buf = io.StringIO()
    with redirect_stdout(buf):
        prompts = get_prompts_from_ds(_iter())
    return [_messages_token_len(p, plot_cap) for p in prompts]


def _empirical_cdf(x: np.ndarray):
    xs = np.sort(x.astype(np.float64))
    ys = np.arange(1, len(xs) + 1, dtype=np.float64) / len(xs)
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot CDF of chat input token lengths for trace datasets.")
    ap.add_argument(
        "-o",
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Output image path (.png or .pdf). Default: {_DEFAULT_OUTPUT}",
    )
    ap.add_argument(
        "--sharegpt-max-conversations",
        type=int,
        default=None,
        metavar="N",
        help="Optional: only first N ShareGPT train conversations (faster debug). "
        "Omit for full train split (same as generate_trace.py).",
    )
    ap.add_argument(
        "--plot-max-tokens",
        type=int,
        default=_DEFAULT_PLOT_X_MAX_TOKENS,
        metavar="N",
        help=f"Clip each displayed length and set x-axis max to N (default: {_DEFAULT_PLOT_X_MAX_TOKENS}). "
        "Ignores TRACE_PROMPT_MAX_LEN for the figure unless you set this higher.",
    )
    ap.add_argument(
        "--no-plot-cap",
        action="store_true",
        help="Do not clip for the figure: lengths follow prompts env (trunc / NO_TRUNCATE); x autoscales.",
    )
    args = ap.parse_args()

    if args.no_plot_cap:
        hard_cap = None
        plot_msg_cap = None
        print("Plot: --no-plot-cap (no figure clip; x-axis autoscale).")
    else:
        hard_cap = args.plot_max_tokens
        plot_msg_cap = hard_cap
        print(f"Plot: clip display lengths and x-axis to {hard_cap:,} tokens.")

    print(f"Loading HumanEval ({_DS_HUMANEVAL}) …")
    len_he = np.array(_lengths_via_get_prompts(_DS_HUMANEVAL, plot_msg_cap), dtype=np.int64)
    print(f"  samples={len(len_he)} min={len_he.min()} p50={np.median(len_he)} max={len_he.max()}")

    print(f"Loading LongBench ({_DS_LONGBENCH}) …")
    len_lb = np.array(_lengths_via_get_prompts(_DS_LONGBENCH, plot_msg_cap), dtype=np.int64)
    print(f"  samples={len(len_lb)} min={len_lb.min()} p50={np.median(len_lb)} max={len_lb.max()}")

    if args.sharegpt_max_conversations is None:
        print(f"Loading ShareGPT full train ({_DS_SHAREGPT}) …")
        len_sg = np.array(_lengths_via_get_prompts(_DS_SHAREGPT, plot_msg_cap), dtype=np.int64)
    else:
        print(f"Loading ShareGPT first {args.sharegpt_max_conversations} conversations …")
        len_sg = np.array(
            _sharegpt_lengths_capped(args.sharegpt_max_conversations, plot_msg_cap), dtype=np.int64
        )
    print(f"  samples={len(len_sg)} min={len_sg.min()} p50={np.median(len_sg)} max={len_sg.max()}")

    plt.figure(figsize=(8, 5))
    for name, arr, color in (
        ("HumanEval", len_he, "C0"),
        ("LongBench", len_lb, "C1"),
        ("ShareGPT", len_sg, "C2"),
    ):
        if arr.size == 0:
            continue
        xs, ys = _empirical_cdf(arr)
        plt.step(xs, ys, where="post", label=f"{name} (n={len(arr):,})", color=color)

    plt.xlabel("Input length (tokens, Qwen chat template)")
    plt.ylabel("CDF")
    if hard_cap is not None:
        plt.title(f"Prompt pools: input length CDF (≤ {hard_cap:,} tokens)")
        plt.xlim(0, hard_cap)
    else:
        plt.title("Prompt pools: input length CDF")
        plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
