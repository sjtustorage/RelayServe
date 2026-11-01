# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions adapted from the vLLM project
"""Traffic generation and benchmark metrics (vLLM bench-compatible)."""

from __future__ import annotations

import asyncio
import time
import warnings
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from typing import Any, Dict, Literal, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sllm.benchmarks.dataset_random import SampleRequest
from sllm.benchmarks.endpoint_request_func import RequestFuncOutput

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


class TaskType(Enum):
    GENERATION = "generation"
    EMBEDDING = "embedding"


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


@dataclass
class EmbedBenchmarkMetrics:
    completed: int
    total_input: int
    request_throughput: float
    total_token_throughput: float
    mean_e2el_ms: float
    std_e2el_ms: float
    median_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


def _get_current_request_rate(
    ramp_up_strategy: Optional[Literal["linear", "exponential"]],
    ramp_up_start_rps: Optional[int],
    ramp_up_end_rps: Optional[int],
    request_index: int,
    total_requests: int,
    request_rate: float,
) -> float:
    if ramp_up_strategy and ramp_up_start_rps is not None and ramp_up_end_rps is not None:
        progress = request_index / max(total_requests - 1, 1)
        if ramp_up_strategy == "linear":
            increase = (ramp_up_end_rps - ramp_up_start_rps) * progress
            return ramp_up_start_rps + increase
        if ramp_up_strategy == "exponential":
            ratio = ramp_up_end_rps / ramp_up_start_rps
            return ramp_up_start_rps * (ratio**progress)
        raise ValueError(f"Unknown ramp-up strategy: {ramp_up_strategy}")
    return request_rate


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
    ramp_up_strategy: Optional[Literal["linear", "exponential"]] = None,
    ramp_up_start_rps: Optional[int] = None,
    ramp_up_end_rps: Optional[int] = None,
) -> AsyncGenerator[tuple[SampleRequest, float], None]:
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    if isinstance(input_requests, Iterable) and not isinstance(input_requests, list):
        input_requests = list(input_requests)

    total_requests = len(input_requests)
    assert total_requests > 0, "No requests provided."

    request_rates: list[float] = []
    delay_ts: list[float] = []
    for request_index, _request in enumerate(input_requests):
        current_request_rate = _get_current_request_rate(
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
            request_index,
            total_requests,
            request_rate,
        )
        request_rates.append(current_request_rate)
        if current_request_rate == float("inf"):
            delay_ts.append(0)
        else:
            theta = 1.0 / (current_request_rate * burstiness)
            delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))

    for i in range(1, len(delay_ts)):
        delay_ts[i] += delay_ts[i - 1]
    if ramp_up_strategy is None and delay_ts[-1] != 0:
        target_total_delay_s = total_requests / request_rate
        normalize_factor = target_total_delay_s / delay_ts[-1]
        delay_ts = [delay * normalize_factor for delay in delay_ts]

    start_ts = time.time()
    for request_index, request in enumerate(input_requests):
        if delay_ts[request_index] > 0:
            current_ts = time.time()
            sleep_interval_s = start_ts + delay_ts[request_index] - current_ts
            if sleep_interval_s > 0:
                await asyncio.sleep(sleep_interval_s)
        yield request, request_rates[request_index]


async def get_request_from_trace(
    trace_requests: list[SampleRequest],
    time_scale: float = 1.0,
) -> AsyncGenerator[tuple[SampleRequest, float], None]:
    """Replay a trace by honouring the ``arrival_time`` timestamps embedded in
    each :class:`~sllm.benchmarks.dataset_random.SampleRequest`.

    Args:
        trace_requests: Requests sorted by ``arrival_time`` (seconds, relative
            to the first request, i.e. ``arrival_time[0] == 0``).
        time_scale: Compress (> 1) or stretch (< 1) the timeline.  For example
            ``time_scale=14`` maps a 14-day trace into ~1 day of real time.

    Yields:
        ``(request, 0.0)`` pairs where the second element is a placeholder rate
        (trace replay has no single RPS value).
    """
    if not trace_requests:
        return

    start_real = time.time()
    for request in trace_requests:
        t_target = (request.arrival_time or 0.0) / time_scale
        elapsed = time.time() - start_real
        sleep_for = t_target - elapsed
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        yield request, 0.0


def calculate_metrics_for_embeddings(
    outputs: list[RequestFuncOutput],
    dur_s: float,
    selected_percentiles: list[float],
) -> EmbedBenchmarkMetrics:
    total_input = 0
    completed = 0
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            e2els.append(outputs[i].latency)
            completed += 1
            total_input += outputs[i].prompt_len

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    return EmbedBenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        request_throughput=completed / dur_s,
        total_token_throughput=total_input / dur_s,
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
    )


def _request_meets_global_slo(
    ttft: float,
    tpot: float,
    e2el: float,
    slo_ms: dict[str, float],
) -> bool:
    """Same semantics as vLLM bench: satisfied iff each observed latency <= SLO (ms)."""
    if "ttft" in slo_ms:
        if ttft > slo_ms["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION:
            return False
    if "tpot" in slo_ms:
        if tpot > slo_ms["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION:
            return False
    if "e2el" in slo_ms:
        if e2el > slo_ms["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION:
            return False
    return True


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
    *,
    tokenizer_by_model: Optional[Dict[str, PreTrainedTokenizerBase]] = None,
    model_slo_config: Optional[Dict[str, Dict[str, float]]] = None,
) -> tuple[BenchmarkMetrics, list[int], Dict[str, Any]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    per_model_completed: Dict[str, int] = {}
    successful_models: list[str] = []

    # Per-dataset raw latency accumulation.
    _ds_ttfts: Dict[str, list] = defaultdict(list)
    _ds_tpots: Dict[str, list] = defaultdict(list)
    _ds_e2els: Dict[str, list] = defaultdict(list)
    _ds_input_tokens: Dict[str, int] = defaultdict(int)
    _ds_output_tokens: Dict[str, int] = defaultdict(int)
    _ds_completed: Dict[str, int] = defaultdict(int)

    for i in range(len(outputs)):
        if outputs[i].success:
            raw_ttft = outputs[i].ttft
            if raw_ttft > dur_s:
                actual_output_lens.append(0)
                continue
            output_len = outputs[i].output_tokens

            if not output_len:
                tm = input_requests[i].target_model
                tok = (
                    tokenizer_by_model.get(tm, tokenizer)
                    if tokenizer_by_model and tm
                    else tokenizer
                )
                output_len = len(tok.encode(outputs[i].generated_text, add_special_tokens=False))
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0.0
            latency_minus_ttft = outputs[i].latency - outputs[i].ttft
            # Prefer ``tpot`` from the HTTP payload (gpu/cpu backend ``process_output``, or
            # ``controller.generate_stream`` tokenwise merges). Previously we recomputed using
            # ``completion_tokens - 1``, which can disagree with how ``itls`` is recorded.
            srv_tpot = float(getattr(outputs[i], "tpot", 0.0) or 0.0)
            n_itl = len(outputs[i].itl)
            if srv_tpot > 0.0:
                tpot = srv_tpot
                tpots.append(tpot)
            elif n_itl > 0:
                tpot = latency_minus_ttft / n_itl
                tpots.append(tpot)
            elif output_len > 1:
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1

            mname = input_requests[i].target_model or ""
            successful_models.append(mname)
            per_model_completed[mname] = per_model_completed.get(mname, 0) + 1

            ds = input_requests[i].dataset or ""
            _ds_ttfts[ds].append(outputs[i].ttft)
            _ds_tpots[ds].append(tpot)
            _ds_e2els[ds].append(outputs[i].latency)
            _ds_input_tokens[ds] += input_requests[i].prompt_len
            _ds_output_tokens[ds] += output_len
            _ds_completed[ds] += 1
        else:
            actual_output_lens.append(0)

    per_model_satisfied: Dict[str, int] = defaultdict(int)

    if model_slo_config:
        for j in range(len(ttfts)):
            m = successful_models[j]
            slo_m = model_slo_config.get(m, {})
            if _request_meets_global_slo(ttfts[j], all_tpots[j], e2els[j], slo_m):
                good_completed += 1
                per_model_satisfied[m] += 1
    elif goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )

        if valid_metrics:
            for j, req_metric in enumerate(zip(*valid_metrics)):
                is_good_req = all(s >= r for s, r in zip(slo_values, req_metric))
                if is_good_req:
                    good_completed += 1
                    per_model_satisfied[successful_models[j]] += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles],
    )

    slo_report: Dict[str, Any] = {
        "overall_slo_satisfied_requests": good_completed,
        "overall_slo_satisfied_rate": good_completed / max(completed, 1),
        "goodput_by_model": {},
        "slo_mode": "per_model" if model_slo_config else ("global" if goodput_config_dict else "none"),
    }
    for m, c in per_model_completed.items():
        s = int(per_model_satisfied.get(m, 0))
        key = m if m else "(unset)"
        slo_report["goodput_by_model"][key] = {
            "completed": c,
            "slo_satisfied": s,
            "slo_satisfied_rate": s / max(c, 1),
        }

    # Per-dataset latency breakdown.
    stats_by_dataset: Dict[str, Any] = {}
    for ds, ds_ttfts in _ds_ttfts.items():
        ds_tpots = _ds_tpots[ds]
        ds_e2els = _ds_e2els[ds]
        n = _ds_completed[ds]
        label = ds if ds else "(unset)"
        stats_by_dataset[label] = {
            "completed": n,
            "total_input_tokens": _ds_input_tokens[ds],
            "total_output_tokens": _ds_output_tokens[ds],
            "mean_ttft_ms":   float(np.mean(ds_ttfts) * 1000)   if ds_ttfts else 0.0,
            "median_ttft_ms": float(np.median(ds_ttfts) * 1000) if ds_ttfts else 0.0,
            "p99_ttft_ms":    float(np.percentile(ds_ttfts, 99) * 1000) if ds_ttfts else 0.0,
            "mean_tpot_ms":   float(np.mean(ds_tpots) * 1000)   if ds_tpots else 0.0,
            "median_tpot_ms": float(np.median(ds_tpots) * 1000) if ds_tpots else 0.0,
            "p99_tpot_ms":    float(np.percentile(ds_tpots, 99) * 1000) if ds_tpots else 0.0,
            "mean_e2el_ms":   float(np.mean(ds_e2els) * 1000)   if ds_e2els else 0.0,
            "median_e2el_ms": float(np.median(ds_e2els) * 1000) if ds_e2els else 0.0,
            "p99_e2el_ms":    float(np.percentile(ds_e2els, 99) * 1000) if ds_e2els else 0.0,
        }
    slo_report["stats_by_dataset"] = stats_by_dataset

    return metrics, actual_output_lens, slo_report
