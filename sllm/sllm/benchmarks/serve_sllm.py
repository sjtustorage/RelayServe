# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions adapted from the vLLM project
r"""Benchmark SLLM HTTP serving (vLLM ``bench serve``-style driver).

Start the SLLM API (e.g. ``sllm`` CLI / Ray controller + uvicorn), register a model,
then run from the *relayserve* repo root::

    python -m sllm.benchmarks.serve_sllm \\
        --model <registered_model_name> \\
        --tokenizer <hf_model_id_for_token_counting> \\
        --host 127.0.0.1 --port 8343 \\
        --endpoint /v1/chat/completions \\
        --num-prompts 100 --request-rate 10

Multi-model + per-model SLO (JSON values in milliseconds)::

    python -m sllm.benchmarks.serve_sllm \\
        --models llama-7b qwen-7b --model-mix round_robin \\
        --tokenizer-config tokmap.json \\
        --model-slo-config slos.json \\
        --num-prompts 200 --request-rate 20

This driver uses **JSON** ``/v1/chat/completions`` (same shape as ``app_lib``:
``model``, ``prompt``, ``max_tokens``, …) and parses the response body
``[completion_dict, latency_metrics]`` returned by the SLLM stack.

Only ``--dataset-name random`` is built in; extend ``dataset_random.py`` for
other datasets or point the upstream vLLM bench at an HTTP proxy.

**Single request (``--num-prompts 1``)**

Use ``--single-request`` to force one sampled prompt; combine with
``--random-input-len`` / ``--random-output-len`` (aliases ``--input-len`` / ``--output-len``)
and ``--random-range-ratio 0`` for fixed token sizes.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Literal, Optional

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sllm.benchmarks.dataset_random import SampleRequest, get_samples, load_trace_requests
from sllm.benchmarks.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
)
from sllm.benchmarks.traffic import TaskType, calculate_metrics, get_request, get_request_from_trace
from sllm.benchmarks.utils import convert_to_pytorch_benchmark_format, write_to_json


def check_goodput_args(args: argparse.Namespace) -> dict[str, float]:
    goodput_config_dict: dict[str, float] = {}
    valid_names = ["ttft", "tpot", "e2el"]
    if args.goodput:
        if getattr(args, "model_slo_config", None):
            print(
                "Note: --goodput is ignored when --model-slo-config is set "
                "(SLO is per model only).",
                file=sys.stderr,
            )
        else:
            goodput_config_dict = parse_goodput(args.goodput)
            for slo_name, slo_val in goodput_config_dict.items():
                if slo_name not in valid_names:
                    raise ValueError(
                        f"Invalid metric name {slo_name}. Expected one of {valid_names}."
                    )
                if slo_val < 0:
                    raise ValueError(f"Invalid SLO value {slo_name}: {slo_val}")
    return goodput_config_dict


def parse_model_weights(s: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --model-weights fragment: {part!r} (use name:weight)")
        name, w = part.rsplit(":", 1)
        out[name.strip()] = float(w)
    return out


def load_model_slo_config(path: str) -> dict[str, dict[str, float]]:
    valid = frozenset({"ttft", "tpot", "e2el"})
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("model SLO JSON must be an object: {model_name: {ttft: ms, ...}, ...}")
    out: dict[str, dict[str, float]] = {}
    for model_name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"SLO for {model_name!r} must be an object")
        inner: dict[str, float] = {}
        for k, v in spec.items():
            if k not in valid:
                raise ValueError(f"Unknown SLO key {k!r} for model {model_name}")
            inner[k] = float(v)
        out[str(model_name)] = inner
    return out


def load_tokenizer_config(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("tokenizer config must be {registered_model: hf_tokenizer_id, ...}")
    return {str(k): str(v) for k, v in raw.items()}


def parse_goodput(slo_pairs: list[str]) -> dict[str, float]:
    goodput_config_dict: dict[str, float] = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            'Use "KEY:VALUE" pairs in milliseconds (e.g. ttft:200 e2el:5000).'
        ) from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any], file_name: str
) -> None:
    metrics = [
        "median_ttft_ms",
        "mean_ttft_ms",
        "std_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p99_tpot_ms",
        "median_itl_ms",
        "mean_itl_ms",
        "std_itl_ms",
        "p99_itl_ms",
    ]
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]] for k in metrics if k in results},
        extra_info={
            k: results[k]
            for k in results
            if k not in metrics and k not in ignored_metrics
        },
    )
    if pt_records:
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def add_dataset_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random"],
        help="Only `random` is bundled; extend dataset_random.py for more.",
    )
    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        "--input-len",
        dest="random_input_len",
        type=int,
        default=1024,
        help="Target prompt length in tokens (alias: --input-len).",
    )
    random_group.add_argument(
        "--random-output-len",
        "--output-len",
        dest="random_output_len",
        type=int,
        default=128,
        help="Target max new tokens per request (alias: --output-len).",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Must be in [0, 1).",
    )
    random_group.add_argument("--random-prefix-len", type=int, default=0)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_parser(parser)
    parser.add_argument(
        "--endpoint-type",
        type=str,
        default="sllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Prefix for saved result filenames.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Full base URL (overrides host/port), e.g. http://127.0.0.1:8343",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8343)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="Must match SLLM route (default: OpenAI-style chat path).",
    )
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single registered model (alternative to --models).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Multiple registered SLLM models; combine with --model-mix / --model-weights.",
    )
    parser.add_argument(
        "--model-mix",
        type=str,
        default="round_robin",
        choices=["round_robin", "random"],
        help="How to assign requests to models when using --models.",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default=None,
        help='With --model-mix random: comma weights, e.g. ``llama:0.7,qwen:0.3``.',
    )
    parser.add_argument(
        "--model-slo-config",
        type=str,
        default=None,
        help="JSON file: per-model SLO in ms, "
        'e.g. ``{"m1":{"ttft":500,"e2el":8000},"m2":{"ttft":200}}``. '
        "Drives overall & per-model goodput when set.",
    )
    parser.add_argument(
        "--model-slo",
        type=str,
        default=None,
        help="Inline JSON string for per-model SLO in ms (same format as --model-slo-config "
        "but passed directly on the command line without a file). "
        'E.g. \'{"8B":{"ttft":500},"72B":{"ttft":2000,"e2el":30000}}\'. '
        "Merged with --model-slo-config if both are given (inline takes precedence).",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=str,
        default=None,
        help="JSON file mapping registered model name → HF tokenizer id/path.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="If set, overrides the JSON ``model`` field sent to the server.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF model id or path for token counting (single-model or default for multi).",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--logprobs", type=int, default=None)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Target requests/s (inf = fire all at t=0).",
    )
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Not supported for SLLM (ignored).",
    )
    parser.add_argument("--save-result", action="store_true")
    parser.add_argument("--save-detailed", action="store_true")
    parser.add_argument("--append-result", action="store_true")
    parser.add_argument("--metadata", metavar="KEY=VALUE", nargs="*")
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--result-filename", type=str, default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
    )
    parser.add_argument("--metric-percentiles", type=str, default="99")
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="Global SLO (ms) as ttft:200 e2el:5000 — ignored if --model-slo-config is set.",
    )
    parser.add_argument(
        "--request-id-prefix",
        type=str,
        default="benchmark-serving",
    )

    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument("--top-p", type=float, default=None)
    sampling_group.add_argument("--top-k", type=int, default=None)
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument("--temperature", type=float, default=None)

    parser.add_argument(
        "--ramp-up-strategy",
        type=str,
        default=None,
        choices=["linear", "exponential"],
    )
    parser.add_argument("--ramp-up-start-rps", type=int, default=None)
    parser.add_argument("--ramp-up-end-rps", type=int, default=None)
    parser.add_argument(
        "--single-request",
        action="store_true",
        help="Sample exactly one random prompt (--num-prompts 1). Combine with "
        "--random-input-len/--random-output-len (or --input-len/--output-len).",
    )

    trace_group = parser.add_argument_group("trace replay (from generate_trace.py)")
    trace_group.add_argument(
        "--trace-file",
        type=str,
        default=None,
        help="Path to trace.pkl produced by generate_trace.py. "
        "When set, --request-rate / --burstiness are ignored and real trace "
        "timestamps drive request dispatch.",
    )
    trace_group.add_argument(
        "--trace-model-map",
        type=str,
        default=None,
        help="JSON file mapping trace model names to deployed SLLM model names. "
        'E.g. {"Qwen/Qwen3-8B/0": "8B-1", "Qwen/Qwen3-8B/1": "8B-2"}. '
        "When absent, trace model names are used as-is.",
    )
    trace_group.add_argument(
        "--trace-time-scale",
        type=float,
        default=1.0,
        help="Divide trace inter-arrival times by this factor to compress the "
        "timeline (e.g. 14 maps a 14-day trace to ~1 day of real time).",
    )
    trace_group.add_argument(
        "--trace-output-len",
        type=int,
        default=None,
        help="Cap max output tokens per trace request.  Omit (default) to let "
        "each request generate until EOS with no token limit.",
    )
    trace_group.add_argument(
        "--trace-max-requests",
        type=int,
        default=None,
        help="Truncate the trace to this many requests (useful for quick tests).",
    )
    trace_group.add_argument(
        "--trace-models",
        nargs="+",
        type=str,
        default=None,
        metavar="MODEL",
        help="Only replay requests whose model name is in this list. "
             "Other models in the trace are dropped. "
             'E.g. --trace-models Qwen3-8B Qwen3-14B',
    )

    stats_group = parser.add_argument_group("controller stats collection")
    stats_group.add_argument(
        "--save-stats-dir",
        type=str,
        default=None,
        help="After all requests finish, POST /save_stats to the controller with this "
             "output directory so solver timings and resource snapshots are flushed to disk. "
             "When omitted the controller uses its own default stats_dir.",
    )


async def benchmark(
    endpoint_type: str,
    api_url: str,
    base_url: str,
    model_name: Optional[str],
    tokenizer: PreTrainedTokenizerBase,
    tokenizers_by_model: dict[str, PreTrainedTokenizerBase],
    input_requests: list[SampleRequest],
    logprobs: Optional[int],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    model_slo_config: Optional[dict[str, dict[str, float]]],
    max_concurrency: Optional[int],
    extra_body: Optional[dict],
    ramp_up_strategy: Optional[Literal["linear", "exponential"]] = None,
    ramp_up_start_rps: Optional[int] = None,
    ramp_up_end_rps: Optional[int] = None,
    trace_time_scale: float = 1.0,
) -> dict[str, Any]:
    del base_url
    task_type = (
        TaskType.EMBEDDING
        if api_url.endswith("/v1/embeddings")
        else TaskType.GENERATION
    )
    if task_type == TaskType.EMBEDDING:
        raise ValueError("SLLM benchmark driver does not support embeddings yet.")

    request_func = ASYNC_REQUEST_FUNCS[endpoint_type]

    if profile:
        print("Note: --profile is not supported for SLLM; ignoring.", file=sys.stderr)

    connector = aiohttp.TCPConnector(
        limit=max_concurrency or 0,
        limit_per_host=max_concurrency or 0,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
        ssl=("https://" in api_url),
    )
    session = aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    )

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    benchmark_start_time = time.perf_counter()
    # completion_times_map[task_index] = elapsed seconds from benchmark start when the
    # task finished.  Used to build the throughput timeline after all tasks complete.
    completion_times_map: dict[int, float] = {}

    async def limited_request_func(
        request_func_input: RequestFuncInput,
        sess: aiohttp.ClientSession,
        bar: Any,
    ) -> RequestFuncOutput:
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, session=sess, pbar=bar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, session=sess, pbar=bar)

    async def timed_limited_request_func(
        task_idx: int,
        request_func_input: RequestFuncInput,
        sess: aiohttp.ClientSession,
        bar: Any,
    ) -> RequestFuncOutput:
        out = await limited_request_func(request_func_input, sess, bar)
        completion_times_map[task_idx] = time.perf_counter() - benchmark_start_time
        return out

    tasks: list[asyncio.Task[RequestFuncOutput]] = []

    # Choose dispatch strategy: trace timestamps vs. synthetic Gamma distribution.
    use_trace = bool(input_requests and input_requests[0].arrival_time is not None)
    if use_trace:
        request_stream = get_request_from_trace(input_requests, trace_time_scale)
    else:
        request_stream = get_request(
            input_requests,
            request_rate,
            burstiness,
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
        )

    task_idx = 0
    async for request, _current_request_rate in request_stream:
        prompt, prompt_len, output_len, mm_content, request_id, target_model = (
            request.prompt,
            request.prompt_len,
            request.expected_output_len,
            request.multi_modal_data,
            request.request_id,
            request.target_model,
        )
        if mm_content is not None:
            raise ValueError("Multimodal requests are not supported.")
        if not target_model:
            raise ValueError("Internal error: SampleRequest missing target_model")
        request_func_input = RequestFuncInput(
            model=target_model,
            model_name=model_name,
            prompt=str(prompt),
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            logprobs=logprobs,
            multi_modal_content=mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
            request_id=request_id,
        )
        tasks.append(
            asyncio.create_task(
                timed_limited_request_func(task_idx, request_func_input, session, pbar)
            )
        )
        task_idx += 1
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens, slo_report = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
        tokenizer_by_model=tokenizers_by_model,
        model_slo_config=model_slo_config,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result (SLLM) ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    if max_concurrency is not None:
        print("{:<40} {:<10}".format("Maximum request concurrency:", max_concurrency))
    if request_rate != float("inf"):
        print("{:<40} {:<10.2f}".format("Request rate configured (RPS):", request_rate))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    if goodput_config_dict or model_slo_config:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput))
        print(
            "{:<40} {:<10.4f}".format(
                "Overall SLO satisfied rate:",
                slo_report["overall_slo_satisfied_rate"],
            )
        )
        print("Per-model SLO (completed / satisfied / rate):")
        for mk, mv in slo_report["goodput_by_model"].items():
            print(
                f"  {mk!r}: completed={mv['completed']}, "
                f"slo_satisfied={mv['slo_satisfied']}, "
                f"rate={mv['slo_satisfied_rate']:.4f}"
            )
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))

    ttft_filtered_count = slo_report.get("ttft_filtered_count", 0)
    ttft_filtered_indices: set[int] = set(slo_report.get("ttft_filtered_indices", []))

    stats_by_dataset = slo_report.get("stats_by_dataset", {})
    if stats_by_dataset:
        print("\n{s:{c}^{n}}".format(s=" Per-Dataset Latency Breakdown ", n=50, c="-"))
        for ds_label, ds_stats in sorted(stats_by_dataset.items()):
            print(f"  [{ds_label}]  completed={ds_stats['completed']}")
            print("    {:<36} {:<10.2f}".format("Mean TTFT (ms):",   ds_stats["mean_ttft_ms"]))
            print("    {:<36} {:<10.2f}".format("Median TTFT (ms):", ds_stats["median_ttft_ms"]))
            print("    {:<36} {:<10.2f}".format("P99 TTFT (ms):",    ds_stats["p99_ttft_ms"]))
            print("    {:<36} {:<10.2f}".format("Mean TPOT (ms):",   ds_stats["mean_tpot_ms"]))
            print("    {:<36} {:<10.2f}".format("Mean E2EL (ms):",   ds_stats["mean_e2el_ms"]))

    if ttft_filtered_count > 0:
        print(
            f"\n[WARNING] {ttft_filtered_count} request(s) had TTFT > benchmark "
            f"duration ({benchmark_duration:.1f}s) and were excluded from TTFT "
            "statistics (likely cross-machine clock-skew artefacts).  "
            "They are saved in request_details with ttft_filtered=true."
        )

    # ── Per-request details (always saved — used for CDF plots) ─────────────────
    # Requests with clearly invalid TTFT (> benchmark duration) are dropped entirely.
    request_details: list[dict[str, Any]] = []
    for i, (req, out) in enumerate(zip(input_requests, outputs)):
        if i in ttft_filtered_indices:
            continue
        tpot_ms: Optional[float] = None
        if out.success:
            srv_tpot = float(getattr(out, "tpot", 0.0) or 0.0)
            if srv_tpot > 0.0:
                tpot_ms = srv_tpot * 1000.0
            elif len(out.itl) > 0:
                tpot_ms = (out.latency - out.ttft) / len(out.itl) * 1000.0
            elif actual_output_lens[i] > 1:
                tpot_ms = (out.latency - out.ttft) / (actual_output_lens[i] - 1) * 1000.0
        request_details.append({
            "model": getattr(req, "target_model", None) or "",
            "dataset": getattr(req, "dataset", None) or "",
            "success": out.success,
            "ttft_ms": out.ttft * 1000.0 if out.success else None,
            "tpot_ms": tpot_ms,
            "e2el_ms": out.latency * 1000.0 if out.success else None,
            "output_tokens": actual_output_lens[i],
        })

    # ── Throughput timeline: token & request throughput per 1-second window ────
    throughput_timeline: list[dict[str, Any]] = []
    completion_events: list[tuple[float, int]] = []  # (completion_time_s, output_tokens)
    for i, out in enumerate(outputs):
        if out.success:
            t = completion_times_map.get(i, benchmark_duration)
            completion_events.append((t, actual_output_lens[i]))
    completion_events.sort(key=lambda x: x[0])
    if completion_events:
        window_s = 1.0
        max_t = completion_events[-1][0]
        t_end = window_s
        while t_end <= max_t + window_s:
            t_start = t_end - window_s
            reqs_in = sum(1 for et, _ in completion_events if t_start <= et < t_end)
            toks_in = sum(tok for et, tok in completion_events if t_start <= et < t_end)
            throughput_timeline.append({
                "window_end_s": round(t_end, 2),
                "req_throughput_rps": round(reqs_in / window_s, 4),
                "tok_throughput_tps": round(toks_in / window_s, 4),
            })
            t_end += window_s

    result: dict[str, Any] = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "request_target_models": [getattr(r, "target_model", None) for r in input_requests],
        "slo_report": slo_report,
        # Always included for CDF plotting.
        "request_details": request_details,
        "throughput_timeline": throughput_timeline,
    }

    def process_one_metric(metric_attribute_name: str, metric_name: str, metric_header: str) -> None:
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")
    print("=" * 50)

    await session.close()
    return result


async def _trigger_save_stats(base_url: str, output_dir: Optional[str]) -> None:
    """POST /save_stats to the controller and log the result."""
    url = f"{base_url.rstrip('/')}/save_stats"
    body = {}
    if output_dir:
        body["output_dir"] = output_dir
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"\nController stats saved: {data.get('files', data)}", file=sys.stderr)
                else:
                    text = await resp.text()
                    print(f"\nWarning: /save_stats returned {resp.status}: {text}", file=sys.stderr)
    except Exception as exc:
        print(f"\nWarning: failed to POST /save_stats to {url}: {exc}", file=sys.stderr)


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.single_request:
        if args.num_prompts != 1:
            print(
                "Note: --single-request forces --num-prompts 1 "
                f"(was {args.num_prompts}).",
                file=sys.stderr,
            )
        args.num_prompts = 1

    if args.ramp_up_strategy is not None:
        if args.request_rate != float("inf"):
            raise ValueError("With ramp-up, omit --request-rate (use ramp-up RPS only).")
        if args.ramp_up_start_rps is None or args.ramp_up_end_rps is None:
            raise ValueError("Ramp-up requires --ramp-up-start-rps and --ramp-up-end-rps.")
        if args.ramp_up_start_rps < 0 or args.ramp_up_end_rps < 0:
            raise ValueError("Ramp-up RPS must be non-negative.")
        if args.ramp_up_start_rps > args.ramp_up_end_rps:
            raise ValueError("Ramp-up start RPS must be <= end RPS.")
        if args.ramp_up_strategy == "exponential" and args.ramp_up_start_rps == 0:
            raise ValueError("Exponential ramp-up cannot start at 0 RPS.")

    if args.base_url is not None:
        api_url = f"{args.base_url.rstrip('/')}{args.endpoint}"
        base_url = args.base_url.rstrip("/")
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    if args.models:
        models_list = list(args.models)
    elif args.model:
        models_list = [args.model]
    else:
        raise ValueError("Specify --model for a single workload or --models for multiple.")

    model_slo_config: Optional[dict[str, dict[str, float]]] = None
    if args.model_slo_config:
        model_slo_config = load_model_slo_config(args.model_slo_config)
    if getattr(args, "model_slo", None):
        # Parse inline JSON directly (same validation as load_model_slo_config).
        try:
            raw_inline = json.loads(args.model_slo)
        except json.JSONDecodeError as err:
            raise ValueError(f"--model-slo is not valid JSON: {err}") from err
        valid = frozenset({"ttft", "tpot", "e2el"})
        inline_parsed: dict[str, dict[str, float]] = {}
        for mname, spec in raw_inline.items():
            if not isinstance(spec, dict):
                raise ValueError(f"--model-slo: SLO for {mname!r} must be an object")
            inner: dict[str, float] = {}
            for k, v in spec.items():
                if k not in valid:
                    raise ValueError(f"--model-slo: unknown SLO key {k!r} for model {mname}")
                inner[k] = float(v)
            inline_parsed[str(mname)] = inner
        if model_slo_config is None:
            model_slo_config = inline_parsed
        else:
            # Inline overrides file for same model keys.
            model_slo_config = {**model_slo_config, **inline_parsed}
    if model_slo_config:
        for m in models_list:
            if m not in model_slo_config:
                print(
                    f"Warning: model {m!r} missing from SLO config; "
                    "that model has no latency SLO (always counts as satisfied).",
                    file=sys.stderr,
                )

    tok_cfg = load_tokenizer_config(args.tokenizer_config) if args.tokenizer_config else {}
    default_tid = args.tokenizer if args.tokenizer is not None else models_list[0]
    tokenizers_by_model: dict[str, PreTrainedTokenizerBase] = {}
    for m in models_list:
        tid = tok_cfg.get(m, default_tid)
        tokenizers_by_model[m] = AutoTokenizer.from_pretrained(
            tid, trust_remote_code=args.trust_remote_code
        )

    args.models_list = models_list
    args.tokenizers_by_model = tokenizers_by_model
    args.model_weights_dict = (
        parse_model_weights(args.model_weights) if args.model_weights else None
    )
    if args.model_weights_dict:
        for k in args.model_weights_dict:
            if k not in models_list:
                raise ValueError(
                    f"--model-weights key {k!r} must appear in --models {models_list!r}"
                )

    tokenizer = tokenizers_by_model[models_list[0]]
    tokenizer_id = default_tid

    # ── Request source: trace file or synthetic random dataset ───────────────
    trace_file = getattr(args, "trace_file", None)
    if trace_file:
        trace_model_map: Optional[dict[str, str]] = None
        if getattr(args, "trace_model_map", None):
            with open(args.trace_model_map, encoding="utf-8") as _f:
                trace_model_map = json.load(_f)
        _trace_output_len = getattr(args, "trace_output_len", None)
        input_requests = load_trace_requests(
            trace_file,
            tokenizers_by_model,
            model_map=trace_model_map,
            default_output_len=_trace_output_len,
            max_requests=getattr(args, "trace_max_requests", None),
            request_id_prefix=getattr(args, "request_id_prefix", "trace"),
        )
        print(
            f"Loaded {len(input_requests)} requests from trace file {trace_file!r} "
            f"(time_scale={getattr(args, 'trace_time_scale', 1.0)}, "
            f"output_len={'EOS' if not _trace_output_len else _trace_output_len})",
            file=sys.stderr,
        )
        # Filter to requested models only (--trace-models).
        trace_model_filter = getattr(args, "trace_models", None)
        if trace_model_filter:
            allowed = set(trace_model_filter)
            all_trace_models = sorted({r.target_model for r in input_requests if r.target_model})
            before = len(input_requests)
            input_requests = [r for r in input_requests if r.target_model in allowed]
            print(
                f"--trace-models filter: kept {len(input_requests)}/{before} requests "
                f"(models: {sorted(allowed)})",
                file=sys.stderr,
            )
            if not input_requests:
                print(
                    f"WARNING: no requests left after --trace-models filter. "
                    f"Trace contains: {all_trace_models}",
                    file=sys.stderr,
                )

        # Update models_list from the trace so SLO / tokenizer lookups work.
        trace_models = sorted({r.target_model for r in input_requests if r.target_model})
        for tm in trace_models:
            if tm not in models_list:
                models_list.append(tm)
                if tm not in tokenizers_by_model:
                    tokenizers_by_model[tm] = tokenizer
    else:
        input_requests = get_samples(args, tokenizer)

    if args.single_request and len(input_requests) > 1:
        print(
            "Note: --single-request: truncating to the first request only "
            f"(had {len(input_requests)}).",
            file=sys.stderr,
        )
        input_requests = input_requests[:1]

    goodput_config_dict = check_goodput_args(args)

    sampling_params = {
        k: v
        for k, v in {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "temperature": args.temperature,
        }.items()
        if v is not None
    }
    if sampling_params and args.endpoint_type not in OPENAI_COMPATIBLE_BACKENDS:
        raise ValueError("Sampling params require an OpenAI-compatible backend key.")
    if "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.0

    gc.collect()
    gc.freeze()

    benchmark_result = await benchmark(
        endpoint_type=args.endpoint_type,
        api_url=api_url,
        base_url=base_url,
        model_name=args.served_model_name,
        tokenizer=tokenizer,
        tokenizers_by_model=tokenizers_by_model,
        input_requests=input_requests,
        logprobs=args.logprobs,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        disable_tqdm=args.disable_tqdm,
        profile=args.profile,
        selected_percentile_metrics=args.percentile_metrics.split(","),
        selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
        ignore_eos=args.ignore_eos,
        goodput_config_dict=goodput_config_dict,
        model_slo_config=model_slo_config,
        max_concurrency=args.max_concurrency,
        extra_body=sampling_params or None,
        ramp_up_strategy=args.ramp_up_strategy,
        ramp_up_start_rps=args.ramp_up_start_rps,
        ramp_up_end_rps=args.ramp_up_end_rps,
        trace_time_scale=getattr(args, "trace_time_scale", 1.0),
    )

    # Flush controller stats (solver timings + resource snapshots) to disk.
    save_stats_dir = getattr(args, "save_stats_dir", None)
    if save_stats_dir is not None:
        await _trigger_save_stats(base_url, save_stats_dir)

    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json: dict[str, Any] = {
        "date": current_dt,
        "endpoint_type": args.endpoint_type,
        "label": args.label,
        "model_ids": models_list,
        "tokenizer_id": tokenizer_id,
        "num_prompts": args.num_prompts,
        "request_rate": args.request_rate if args.request_rate < float("inf") else "inf",
        "burstiness": args.burstiness,
        "max_concurrency": args.max_concurrency,
    }
    if args.metadata:
        for item in args.metadata:
            if "=" in item:
                k, v = item.split("=", 1)
                result_json[k.strip()] = v.strip()
            else:
                raise ValueError("Invalid metadata; use KEY=VALUE.")
    if args.ramp_up_strategy is not None:
        result_json["ramp_up_strategy"] = args.ramp_up_strategy
        result_json["ramp_up_start_rps"] = args.ramp_up_start_rps
        result_json["ramp_up_end_rps"] = args.ramp_up_end_rps

    result_json = {**result_json, **benchmark_result}

    if not args.save_detailed:
        for field in [
            "input_lens",
            "output_lens",
            "ttfts",
            "itls",
            "generated_texts",
            "errors",
            "request_target_models",
        ]:
            result_json.pop(field, None)
    # request_details and throughput_timeline are always kept (used for CDF plots).

    if args.save_result or args.append_result:
        base_model_id = "-".join(m.split("/")[-1] for m in models_list)
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}" if args.max_concurrency is not None else ""
        )
        label = args.label or args.endpoint_type
        if args.ramp_up_strategy is not None:
            file_name = (
                f"{label}-ramp-up-{args.ramp_up_strategy}-{args.ramp_up_start_rps}qps-"
                f"{args.ramp_up_end_rps}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"
            )
        else:
            file_name = f"{label}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)
        mode = "a+" if args.append_result else "w"
        with open(file_name, mode=mode, encoding="utf-8") as outfile:
            if args.append_result and outfile.tell() != 0:
                outfile.write("\n")
            json.dump(result_json, outfile, indent=2, ensure_ascii=False)
        save_to_pytorch_benchmark_format(args, result_json, file_name)

    return result_json


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="SLLM serving benchmark (vLLM bench-style).")
    add_cli_args(parser)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
