# ---------------------------------------------------------------------------- #
#  Load external loading perf profiles and solve tokenwise vs layerwise cold    #
#  start parameters from bandwidth + analytic CPU time models.                 #
#  Solver: loading_perf_profile_solver.cpp. Times in ms, GB.                   #
# ---------------------------------------------------------------------------- #
from __future__ import annotations

import os
import re
import math
from typing import Any, Dict, List, Mapping, Optional

# Deploy / router may register logical ids like "Qwen3-8B-1"; bundled solver data uses "Qwen3-8B".
_TRAILING_NUMERIC_INSTANCE_SUFFIX = re.compile(r"-\d+$")


from sllm.logger import init_logger

logger = init_logger(__name__)

try:
    import sllm._loading_perf_profile_solver as solver
except ImportError as exc:
    # The extension is architecture/Python-version specific.  Keep imports
    # usable for controller/router unit tests and fail with a focused message
    # only when the analytic solver is actually invoked.
    solver = None
    _solver_import_error = exc


def layer_idxes_by_pp_stages(
    num_layers: int, pipeline_parallel_size: int
) -> List[List[int]]:
    """Split global layer indices ``0 .. num_layers-1`` into *pp* stage lists, sizes as even
    as possible (the first *r* stages get one extra layer when *num_layers* is not divisible
    by *pp*; matches contiguous PP sharding in vLLM).

    * ``pipeline_parallel_size == 1`` → ``[[0, 1, ..., n-1]]`` (single vLLM stage list).
    """
    n = int(num_layers)
    if n <= 0:
        return []
    pp = max(1, int(pipeline_parallel_size or 1))
    if pp == 1:
        return [list(range(n))]
    base, rem = divmod(n, pp)
    out: List[List[int]] = []
    start = 0
    for stage in range(pp):
        sz = base + (1 if stage < rem else 0)
        if sz > 0:
            out.append(list(range(start, start + sz)))
        else:
            out.append([])
        start += sz
    return out


def _normalize_layer_idxes_for_vllm_pp(
    method: str,
    layer_idxes: Any,
    backend_config: Optional[Mapping[str, Any]],
) -> Any:
    """Ensure ``layer_idxes_by_pp`` matches vLLM: outer length == ``pipeline_parallel_size``.

    Legacy single-GPU C++ returned ``[tokenwise_layers, layerwise_layers]`` (length 2).
    vLLM PP=1 expects ``[[layer_ids...]]``. Flat ``[0,1,...,N]`` must be wrapped.
    """
    bc = dict(backend_config or {})
    pp = int(bc.get("pipeline_parallel_size", 1) or 1)
    if pp != 1 or not isinstance(layer_idxes, list):
        return layer_idxes
    if not layer_idxes:
        return [[]]
    if all(isinstance(x, int) for x in layer_idxes):
        return [list(layer_idxes)]
    if (
        len(layer_idxes) == 2
        and all(isinstance(x, list) for x in layer_idxes)
        and method in ("tokenwise", "layerwise")
    ):
        pick = 0 if method == "tokenwise" else 1
        return [list(layer_idxes[pick])]
    return layer_idxes


def expand_first_stage_load_list_compute_layers_first(
    layer_idxes: List[List[int]],
    backend_config: Optional[Mapping[str, Any]] = None,
) -> List[List[int]]:
    """Put **compute** layer indices first, then append ``0..x-1`` where *x* = min(compute).

    The solver lists the layers this stage must run; if their minimum index is *x* > 0, the
    first GPU can still load weights for ``0,1,...,x-1`` opportunistically. Those earlier
    layers are **appended after** the compute list so loading prioritizes the layers needed
    for the forward path first, with fill-in weights following in order.

    Controlled by ``backend_config["lazy_load_include_prefix_layers"]`` (default: True).
    """
    bc = dict(backend_config or {})
    if not bc.get("lazy_load_include_prefix_layers", True):
        return layer_idxes
    if not layer_idxes or not layer_idxes[0]:
        return layer_idxes
    first = layer_idxes[0]
    if not all(isinstance(i, int) for i in first):
        return layer_idxes
    x = min(first)
    if x <= 0:
        return layer_idxes
    fill = list(range(0, x))
    new_first = list(first) + fill
    if len(layer_idxes) == 1:
        return [new_first]
    return [new_first] + [list(s) for s in layer_idxes[1:]]


def load_loading_perf_profile(original_amx: bool = False) -> Dict[str, Any]:
    if original_amx:
        from sllm.data.loading_perf_profile_original_amx import LOADING_PERF_PROFILE
    else:
        from sllm.data.loading_perf_profile_default import LOADING_PERF_PROFILE
    return dict(LOADING_PERF_PROFILE)


def _require_solver():
    if solver is None:
        raise RuntimeError(
            "The loading-performance solver extension is unavailable for this "
            "Python interpreter; rebuild sllm._loading_perf_profile_solver."
        ) from _solver_import_error
    return solver


def _int_positive_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _iter_parallel_config_sources(
    backend_config: Optional[Mapping[str, Any]],
    model_config: Optional[Mapping[str, Any]],
):
    """Order: flat backend_config, then model's nested backend_config, then model top-level."""
    if backend_config:
        yield backend_config
    if model_config:
        nested = model_config.get("backend_config")
        if nested and isinstance(nested, Mapping):
            yield nested
        yield model_config


def _resolve_tp_pp_for_solver(
    model_entry: Mapping[str, Any],
    backend_config: Optional[Mapping[str, Any]],
    model_config: Optional[Mapping[str, Any]],
) -> tuple[int, int]:
    """Prefer deployment config (e.g. config.json) over *model_entry* (bundled profile)."""
    prof_tp = int(model_entry.get("tp_size", 1) or 1)
    prof_pp = int(model_entry.get("pp_size", 1) or 1)
    sources = list(
        _iter_parallel_config_sources(backend_config, model_config)
    )

    def _pick_first(keys: tuple[str, ...], profile_val: int) -> int:
        for src in sources:
            for k in keys:
                if k in src:
                    v = _int_positive_or_none(src.get(k))
                    if v is not None:
                        return v
        return profile_val

    tp = _pick_first(("tensor_parallel_size", "tp_size"), prof_tp)
    pp = _pick_first(("pipeline_parallel_size", "gpu_pp_size"), prof_pp)
    return tp, pp


def _resolve_profile_model_key(
    models: Mapping[str, Any], model_name: str
) -> Optional[str]:
    """Match *model_name* to a profile ``models`` entry (path or HF id → basename).

    Strips a trailing ``-<digits>`` instance suffix (possibly repeated) so ids like
    ``Qwen3-8B-1`` resolve to a bundled key ``Qwen3-8B`` when present.
    """
    if model_name in models:
        return model_name
    tail = os.path.basename(str(model_name).rstrip("/"))
    if tail and tail in models:
        return tail
    cur = tail
    while cur and _TRAILING_NUMERIC_INSTANCE_SUFFIX.search(cur):
        cur = _TRAILING_NUMERIC_INSTANCE_SUFFIX.sub("", cur)
        if cur and cur in models:
            return cur
    return None


def _prepare_solver_inputs(
    profile: Mapping[str, Any],
    model_name: str,
    input_length: int,
    backend_config: Optional[Mapping[str, Any]] = None,
    model_config: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    models = profile.get("models", {})
    key = _resolve_profile_model_key(models, model_name)
    if key is None:
        logger.warning(
            "No loading perf entry for model %r (tried basename); solver skipped",
            model_name,
        )
        return None
    m = dict(models[key])
    try:
        layer_GB = float(m["layer_GB"])
        embedding_GB = float(m["embedding_GB"])
        num_layers = int(m["num_layers"])
        tp_size, pp_size = _resolve_tp_pp_for_solver(
            m, backend_config, model_config
        )
        hidden_size = int(m.get("hidden_size", 4096))
        head_dim = int(m.get("head_dim", 128))
        num_kv_heads = int(m.get("num_kv_heads", 8))
    except (KeyError, TypeError, ValueError):
        return None

    cpu_tw = m.get("cpu_tokenwise_ms", {})
    cpu_lw = {name: float(value) / num_layers for name, value in cpu_tw.items()}
    gpu_tw = m.get("gpu_tokenwise_ms", {})
    gpu_lw = m.get("gpu_layerwise_ms", {})
    cpu_dec = m.get("cpu_decode_ms", {})
    interference = m.get("interference", {})
    if profile.get("interference_lookup", "model_average") == "per_input_length" and interference:
        per_input = interference.get("per_input_length", {})
        if str(int(input_length)) not in per_input:
            raise ValueError(f"Missing measured interference for {key}, input {input_length}")
        factors = per_input[str(int(input_length))]
    else:
        factors = interference
    cpu_multiplier = float(factors.get("cpu_slowdown_x", 1.0))
    loading_multiplier = float(factors.get("loading_slowdown_x", 1.0))
    if not all(math.isfinite(v) and v > 0 for v in (cpu_multiplier, loading_multiplier)):
        raise ValueError("Measured interference multipliers must be finite and positive")
    # The manually specified CPU offset is fixed overhead, not measured AMX work.
    cpu_offset = float(m.get("cpu_offset_ms", 0.0))
    def scale_cpu(line, offset):
        return {**line,
                "intercept": offset + (float(line.get("intercept", 0.0)) - offset) * cpu_multiplier,
                "coef_tokens": float(line.get("coef_tokens", 0.0)) * cpu_multiplier,
                **({'coef_tokens_squared': float(line['coef_tokens_squared']) * cpu_multiplier}
                   if 'coef_tokens_squared' in line else {})}
    cpu_tw = scale_cpu(cpu_tw, cpu_offset)
    cpu_lw = scale_cpu(cpu_lw, cpu_offset / max(1, num_layers))
    hareware_info = profile.get("hardware", {})

    return {
        "fixed_ms": float(m.get("engine_init_ms", 0.0)),
        "pcie_GBps": hareware_info.get("pcie_cpu_to_gpu_GBps", 0.0),
        "rdma_GBps": hareware_info.get("rdma_node_to_node_GBps", 0.0),
        "numa_nodes": hareware_info.get("numa_nodes", 2),
        "embedding_GB": embedding_GB,
        "layer_GB": layer_GB,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "input_length": int(input_length),
        "cpu_tw": cpu_tw,
        "cpu_lw": cpu_lw,
        "gpu_tw": gpu_tw,
        "gpu_lw": gpu_lw,
        "cpu_dec": cpu_dec,
        "cpu_interference_multiplier": cpu_multiplier,
        "weight_loading_multiplier": loading_multiplier,
    }


def _solve_lazy_load_method_cpp(
    inputs: Mapping[str, Any], *, debug: bool = False
) -> Optional[List[Any]]:
    solved = _require_solver().solve_lazy_load_method_cpp(
        float(inputs["fixed_ms"]),
        float(inputs["pcie_GBps"]),
        float(inputs["rdma_GBps"]),
        int(inputs["numa_nodes"]),
        float(inputs["embedding_GB"]),
        float(inputs["layer_GB"]),
        int(inputs["num_layers"]),
        int(inputs["num_kv_heads"]),
        int(inputs["head_dim"]),
        int(inputs["hidden_size"]),
        int(inputs["tp_size"]),
        int(inputs["pp_size"]),
        int(inputs["input_length"]),
        float(inputs["cpu_tw"].get("intercept", 0.0)),
        float(inputs["cpu_tw"].get("coef_tokens", 0.0)),
        float(inputs["gpu_tw"].get("intercept", 0.0)),
        float(inputs["gpu_tw"].get("coef_tokens", 0.0)),
        float(inputs["cpu_lw"].get("intercept", 0.0)),
        float(inputs["cpu_lw"].get("coef_tokens", 0.0)),
        float(inputs["gpu_lw"].get("intercept", 0.0)),
        float(inputs["gpu_lw"].get("coef_tokens", 0.0)),
        float(inputs["cpu_dec"].get("intercept", 0.0)),
        float(inputs["cpu_dec"].get("coef_kv_tokens", 0.0)),
        debug,
        float(inputs.get("transfer_setup_ms", 10.0)),
        float(inputs.get("weight_loading_multiplier", 1.0)),
        int(inputs.get("max_cpu_tokens", -1)),
        bool(inputs.get("allow_cpu_decode", True)),
    )
    if solved is None:
        return None
    if len(solved) == 4:
        method, second, layer_idxes, ranking_ttft_ms = solved
        return [
            str(method),
            int(second),
            list(layer_idxes),
            float(ranking_ttft_ms),
        ]
    method, second, layer_idxes = solved
    return [str(method), int(second), list(layer_idxes)]


def _solve_lazy_load_method_cpp_with_ttft(
    inputs: Mapping[str, Any],
    *,
    debug: bool = False,
    backend_config: Optional[Mapping[str, Any]] = None,
) -> Optional[List[Any]]:
    """Like _solve_lazy_load_method_cpp, but also returns solver ranking TTFT."""
    solved = _require_solver().solve_lazy_load_method_cpp(
        float(inputs["fixed_ms"]),
        float(inputs["pcie_GBps"]),
        float(inputs["rdma_GBps"]),
        int(inputs["numa_nodes"]),
        float(inputs["embedding_GB"]),
        float(inputs["layer_GB"]),
        int(inputs["num_layers"]),
        int(inputs["num_kv_heads"]),
        int(inputs["head_dim"]),
        int(inputs["hidden_size"]),
        int(inputs["tp_size"]),
        int(inputs["pp_size"]),
        int(inputs["input_length"]),
        float(inputs["cpu_tw"].get("intercept", 0.0)),
        float(inputs["cpu_tw"].get("coef_tokens", 0.0)),
        float(inputs["gpu_tw"].get("intercept", 0.0)),
        float(inputs["gpu_tw"].get("coef_tokens", 0.0)),
        float(inputs["cpu_lw"].get("intercept", 0.0)),
        float(inputs["cpu_lw"].get("coef_tokens", 0.0)),
        float(inputs["gpu_lw"].get("intercept", 0.0)),
        float(inputs["gpu_lw"].get("coef_tokens", 0.0)),
        float(inputs["cpu_dec"].get("intercept", 0.0)),
        float(inputs["cpu_dec"].get("coef_kv_tokens", 0.0)),
        debug,
        float(inputs.get("transfer_setup_ms", 10.0)),
        float(inputs.get("weight_loading_multiplier", 1.0)),
        int(inputs.get("max_cpu_tokens", -1)),
        bool(inputs.get("allow_cpu_decode", True)),
    )
    if solved is None:
        return None
    if len(solved) == 4:
        method, second, layer_idxes, ranking_ttft_ms = solved
        layer_idxes = _normalize_layer_idxes_for_vllm_pp(
            str(method), layer_idxes, backend_config
        )
        return [
            str(method),
            int(second),
            list(layer_idxes),
            float(ranking_ttft_ms),
        ]
    method, second, layer_idxes = solved
    layer_idxes = _normalize_layer_idxes_for_vllm_pp(
        str(method), layer_idxes, backend_config
    )
    return [str(method), int(second), list(layer_idxes), None]

def solve_static(
    profile: Mapping[str, Any],
    model_name: str,
    input_length: int,
    backend_config: Optional[Mapping[str, Any]] = None,
    model_config: Optional[Mapping[str, Any]] = None,
) -> Optional[List[Any]]:
    """Ablation (--static): bypass solver; pick best of tokenwise and layerwise.

    Both candidates are derived analytically from the ``engine_init_ms`` window
    without running the C++ solver.

    **Tokenwise candidate**
      CPU processes as many tokens as fit in ``engine_init_ms`` via the linear
      ``cpu_tokenwise_ms`` model (with optional decode extension when the full
      prefill fits within the window). GPU loads all layers in sequential order.
      Estimated TTFT:
        - If CPU covered the full prompt:  ``cpu_tokenwise_ms(input_length)``
        - Otherwise: ``gpu_tokenwise_ms(input_length - cpu_tokens)``

    **Layerwise candidate**
      CPU executes as many layers as fit in ``engine_init_ms`` via the per-layer
      ``cpu_layerwise_ms`` model.  GPU load list puts GPU-compute layers first
      then prefix fill-in (same as
      ``expand_first_stage_load_list_compute_layers_first``).
      Estimated TTFT = ``(num_layers - cpu_layers) * gpu_layerwise_ms(input_length)``.

    The candidate with the lower estimated TTFT is returned.
    """
    inputs = _prepare_solver_inputs(
        profile, model_name, input_length, backend_config, model_config
    )
    if inputs is None:
        return None

    engine_init_ms = float(inputs["fixed_ms"])
    num_layers = int(inputs["num_layers"])
    pp = int(inputs["pp_size"])
    T = float(input_length)

    cpu_tw = inputs["cpu_tw"]
    gpu_tw = inputs["gpu_tw"]
    cpu_lw = inputs["cpu_lw"]
    gpu_lw = inputs["gpu_lw"]
    cpu_dec = inputs["cpu_dec"]

    # ── Tokenwise candidate ───────────────────────────────────────────────── #
    tw_intercept = float(cpu_tw.get("intercept", 0.0))
    tw_coef = float(cpu_tw.get("coef_tokens", 0.0))
    dec_intercept = float(cpu_dec.get("intercept", 0.0))

    if tw_coef > 0:
        prefill_ms = tw_intercept + tw_coef * T
        if prefill_ms > engine_init_ms:
            cpu_tokens = int((engine_init_ms - tw_intercept) / tw_coef)
        else:
            # Full prefill fits; extend with decode steps that fit in the remainder
            cpu_tokens = int(T)
            if dec_intercept > 0:
                cpu_tokens += int((engine_init_ms - prefill_ms) / dec_intercept)
    else:
        cpu_tokens = int(T)
    cpu_tokens = max(0, cpu_tokens)

    if cpu_tokens >= int(T):
        ttft_tw = tw_intercept + tw_coef * T
    else:
        gpu_tw_i = float(gpu_tw.get("intercept", 0.0))
        gpu_tw_c = float(gpu_tw.get("coef_tokens", 0.0))
        ttft_tw = engine_init_ms + gpu_tw_i + gpu_tw_c * (T - cpu_tokens)

    all_layers = layer_idxes_by_pp_stages(num_layers, pp)

    # ── Layerwise candidate ───────────────────────────────────────────────── #
    lw_intercept = float(cpu_lw.get("intercept", 0.0))
    lw_coef = float(cpu_lw.get("coef_tokens", 0.0))
    per_layer_ms = lw_intercept + lw_coef * T
    if per_layer_ms > 0:
        cpu_layers = min(int(engine_init_ms / per_layer_ms), num_layers)
    else:
        cpu_layers = num_layers

    gpu_compute_layers = num_layers - cpu_layers
    gpu_lw_i = float(gpu_lw.get("intercept", 0.0))
    gpu_lw_c = float(gpu_lw.get("coef_tokens", 0.0))
    ttft_lw = engine_init_ms + gpu_compute_layers * (gpu_lw_i + gpu_lw_c * T)

    gpu_load = list(range(cpu_layers, num_layers)) + list(range(0, cpu_layers))
    if pp == 1:
        lw_layer_idxes = [gpu_load]
    else:
        # Static mode must obey the same outer stage shape as the runtime PP backend.
        # Keep contiguous stage ownership; only stage 0 can opportunistically load
        # the prefix after its compute layers.
        lw_layer_idxes = layer_idxes_by_pp_stages(num_layers, pp)
        if lw_layer_idxes and cpu_layers > 0:
            stage0 = lw_layer_idxes[0]
            compute = [i for i in stage0 if i >= cpu_layers]
            prefix = [i for i in stage0 if i < cpu_layers]
            lw_layer_idxes[0] = compute + prefix

    # ── Pick the candidate with lower estimated TTFT ──────────────────────── #
    logger.info(
        "static ablation model=%s input_length=%d: "
        "tokenwise est_ttft=%.2f ms (cpu_tokens=%d) | "
        "layerwise est_ttft=%.2f ms (cpu_layers=%d)",
        model_name, input_length,
        ttft_tw, cpu_tokens,
        ttft_lw, cpu_layers,
    )

    if ttft_tw <= ttft_lw:
        logger.info("static ablation → tokenwise selected")
        return ["tokenwise", cpu_tokens, all_layers]
    else:
        logger.info("static ablation → layerwise selected")
        return ["layerwise", cpu_layers, lw_layer_idxes]

def solve_lazy_load_method(
    profile: Mapping[str, Any],
    model_name: str,
    input_length: int,
    *_unused_backend_args: Any,
    debug: bool = False,
) -> Optional[List[Any]]:
    """
    Returns the same triple as ``SllmController._generate_lazy_load_method`` heuristic,
    or None to signal fallback.

    Implemented only by the bundled C++ solver via an installed pybind11 extension.
    Profile values use ms for time and GB for weight sizes.

    Extra positional arguments (e.g. ``backend_config``, ``model_config``): used to
    override profile ``tp_size``/``pp_size`` and to read ``pipeline_parallel_size`` when
    normalizing ``layer_idxes`` for vLLM.

    If ``debug`` is True, writes a phase-by-phase timing breakdown to stderr.

    When ``tensor_parallel_size`` / ``tp_size`` or ``pipeline_parallel_size`` / ``gpu_pp_size``
    appear in the deployment (``backend_config`` or top-level / nested ``model_config``), those
    values override the bundled profile's ``tp_size`` / ``pp_size``.

    Model profile may include optional ``cpu_decode_ms`` (``intercept``, ``coef_kv_tokens``):
    decode step time is ``intercept + coef_kv_tokens * input_length``. When full-input CPU
    prefill is below ``engine_init_ms`` and the decode profile is non-zero, the solver may
    pick an **engine-overlap** tokenwise plan: ranking TTFT is CPU full-input prefill time,
    ``cpu_compute_length`` can extend to prefill plus decode steps that fit in the init
    window, and GPU preload is the largest layer prefix whose first weight DMA fits in
    ``engine_init_ms``. Otherwise tokenwise ranking uses the classic GPU-handoff TTFT
    (phase1 + KV transfer + phase2). Omit or set both coeffs to 0 to disable overlap.

    For ``pipeline_parallel_size`` > 1, the solver may also choose an **engine-overlap**
    branch that competes with classic PP tokenwise and layer-pipeline plans. That branch
    still returns ``method == "tokenwise"`` (never a third name): it is the same
    cold-start path as tokenwise in the controller/vLLM; only ``second`` (CPU compute
    tokens) and per-stage layer lists differ.
    """
    backend_for_prepare = (
        _unused_backend_args[0] if len(_unused_backend_args) > 0 else None
    )
    model_for_prepare = (
        _unused_backend_args[1] if len(_unused_backend_args) > 1 else None
    )
    inputs = _prepare_solver_inputs(
        profile,
        model_name,
        input_length,
        backend_config=backend_for_prepare,
        model_config=model_for_prepare,
    )
    if inputs is None:
        return None
    raw = _solve_lazy_load_method_cpp(inputs, debug=debug)
    if raw is None:
        return None
    method, second, layer_idxes = raw[0], raw[1], raw[2]
    layer_idxes = _normalize_layer_idxes_for_vllm_pp(
        method, layer_idxes, backend_for_prepare
    )
    return [method, second, list(layer_idxes)]
