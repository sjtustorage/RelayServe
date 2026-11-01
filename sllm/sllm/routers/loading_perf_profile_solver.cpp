#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr double kBetaStd = 0;  // TTFT + kBetaStd * pen_std (test_pp.py)
constexpr int kTokenCpuStep = 32;

// KV / activation volumes are counted in elements; convert to GB for transfer_ms (fp16).
constexpr double kBF16Bytes = 2.0;
constexpr double kBytesToGB = 1e-9;
thread_local double gTransferSetupMs = 10.0;
thread_local double gWeightLoadingMultiplier = 1.0;

double tokenwise_ms(double intercept, double coef_tokens, double tokens) {
    return intercept + coef_tokens * tokens;
}

double layerwise_ms(
    double intercept,
    double coef_tokens,
    double tokens,
    double layers
) {
    return layers * (intercept + coef_tokens * tokens);
}

// Total weight volume (GB) moved for this load plan, per TP rank.
double weights_GB(
    double embedding_GB,
    double layer_GB,
    int layers_to_load,
    int numa_nodes,
    int tp_size
) {
    return (embedding_GB + layer_GB * layers_to_load) / std::min(numa_nodes, tp_size);
}

double kvcache_GB(
    int input_length,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int numa_nodes,
    int tp_size
) {
    // 2 = K + V per layer; * kFp16Bytes => bytes; / min(...) TP/NUMA sharding of heads.
    return 2.0 * static_cast<double>(input_length) * head_dim * num_kv_heads
           * num_layers * kBF16Bytes * kBytesToGB
           / std::min(numa_nodes, tp_size);
}

double hidden_GB(
    int input_length,
    int hidden_size,
    int numa_nodes,
    int tp_size
) {
    // 2 = two activation tensors (e.g. residual stream convention in profile).
    return 2.0 * static_cast<double>(input_length) * hidden_size
           * std::max(tp_size / numa_nodes, 1) * kBF16Bytes * kBytesToGB;
}

double transfer_ms(double transfer_GBps, double volume_GB) {
    if (transfer_GBps <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return 1000.0 * volume_GB / transfer_GBps + gTransferSetupMs;
}

double weight_transfer_ms(double bandwidth, double volume) {
    return gWeightLoadingMultiplier * transfer_ms(bandwidth, volume);
}

// Intra-node equations (2)-(5). Loading continues during initialization,
// CPU computation and state transfer. N_st is a continuous estimate, not a
// separately enumerated preload batch or a PP stage completion time.
double intra_node_remaining_ms(double start_ms, int gpu_layers,
                              double gpu_compute_ms, double gpu_layer_ms,
                              double bandwidth, double layer_GB,
                              int numa_nodes, int tp_size) {
    if (gpu_layers <= 0) return 0.0;
    const double one_layer_ms = weight_transfer_ms(bandwidth,
        weights_GB(0.0, layer_GB, 1, numa_nodes, tp_size));
    const double loaded_layers = one_layer_ms > 0.0
        ? start_ms / one_layer_ms : static_cast<double>(gpu_layers);
    const double remaining_layers = std::max(0.0, gpu_layers - loaded_layers);
    const double remaining_GB = layer_GB * remaining_layers
        / std::min(numa_nodes, tp_size);
    const double limit_ms = remaining_layers > 0.0
        ? weight_transfer_ms(bandwidth, remaining_GB) + gpu_layer_ms : 0.0;
    return std::max(gpu_compute_ms, limit_ms);
}

// Tokenwise (single GPU), classic path: total_ms = gpu_handoff = phase1 + KV xfer + phase2.
//
// Engine-overlap branch (when cpu_decode_* configured and full-input prefill < engine_init):
//   TTFT ranking uses cpu_prefill(input_length) only.
//   cpu_compute_length = input_length + floor((engine_init - prefill_full) / decode_step),
//   decode_step = intercept + coef_kv_tokens * input_length.
//   gpu_load_layers = max L with first-batch weight xfer <= engine_init (prefix layers).
// Otherwise search the classic tokenwise grid (decode coeffs ignored for TTFT).

struct TokenwiseBreakdown {
    double weights1_GB;
    double xfer_first_weights_ms;
    double cpu_prefill_ms;
    double phase1_parallel_ms;
    double kv_GB;
    double kv_transfer_ms;
    double weights2_GB;
    double xfer_remaining_weights_ms;
    int    gpu_tail_tokens;
    double gpu_tail_prefill_ms;
    double phase2_parallel_ms;
    /// Time until GPU lazy-handoff path finishes tail prefill (phase1 + KV + phase2).
    double gpu_handoff_ttft_ms;
    /// Debug: prefill + one decode step at cpu_length KV.
    double cpu_first_token_path_ms;
    double cpu_decode_one_step_ms;
    /// Remaining time in engine-prep window after CPU prefill (0 if prefill >= engine).
    double engine_slack_after_prefill_ms;
    /// floor(slack / decode_step) when modeling CPU decode during engine prep.
    int decodes_during_engine_prep;
    double total_ms;
};

TokenwiseBreakdown compute_tokenwise_breakdown(
    double transfer_GBps,
    double embedding_GB,
    double layer_GB,
    double engine_init_ms,
    int    input_length,
    int    cpu_length,
    int    gpu_load_layers,
    int    num_layers,
    int    hidden_size,
    int    num_kv_heads,
    int    head_dim,
    int    tp_size,
    int    numa_nodes,
    double cpu_tw_intercept,
    double cpu_tw_coef_tokens,
    double gpu_tw_intercept,
    double gpu_tw_coef_tokens,
    double cpu_decode_intercept,
    double cpu_decode_coef_kv
) {
    TokenwiseBreakdown b{};
    b.weights1_GB = weights_GB(embedding_GB, layer_GB, gpu_load_layers, numa_nodes, tp_size);
    b.xfer_first_weights_ms = weight_transfer_ms(transfer_GBps, b.weights1_GB);
    b.cpu_prefill_ms = cpu_length > 0
        ? tokenwise_ms(cpu_tw_intercept, cpu_tw_coef_tokens, static_cast<double>(cpu_length))
        : 0.0;
    b.phase1_parallel_ms = std::max(b.cpu_prefill_ms, engine_init_ms);
    b.kv_GB = kvcache_GB(
        cpu_length, num_layers, num_kv_heads, head_dim, numa_nodes, tp_size
    );
    // A zero-token plan is the pure GPU/no-CPU fallback. It has no state to
    // publish, so it must not pay the transfer API setup cost.
    b.kv_transfer_ms = cpu_length > 0 ? transfer_ms(transfer_GBps, b.kv_GB) : 0.0;
    int rem_layers = num_layers - gpu_load_layers;
    // Embedding is already in the first batch. Empty recovery does no DMA
    // and pays no API setup cost.
    b.weights2_GB = rem_layers > 0
        ? weights_GB(0.0, layer_GB, rem_layers, numa_nodes, tp_size) : 0.0;
    b.xfer_remaining_weights_ms = rem_layers > 0
        ? weight_transfer_ms(transfer_GBps, b.weights2_GB) : 0.0;
    b.gpu_tail_tokens     = std::max(input_length - cpu_length, 0);
    b.gpu_tail_prefill_ms = b.gpu_tail_tokens > 0 ? tokenwise_ms(
        gpu_tw_intercept, gpu_tw_coef_tokens, static_cast<double>(b.gpu_tail_tokens)
    ) : 0.0;
    const double start_ms = b.phase1_parallel_ms + b.kv_transfer_ms;
    b.phase2_parallel_ms = b.gpu_tail_tokens > 0 ? intra_node_remaining_ms(
        start_ms, num_layers, b.gpu_tail_prefill_ms,
        b.gpu_tail_prefill_ms / num_layers, transfer_GBps, layer_GB,
        numa_nodes, tp_size) : 0.0;
    b.gpu_handoff_ttft_ms = b.phase1_parallel_ms + b.kv_transfer_ms + b.phase2_parallel_ms;

    b.cpu_decode_one_step_ms =
        cpu_decode_intercept
        + cpu_decode_coef_kv * static_cast<double>(std::max(cpu_length, 0));
    b.engine_slack_after_prefill_ms = 0.0;
    b.decodes_during_engine_prep  = 0;

    if (b.cpu_prefill_ms < engine_init_ms) {
        b.engine_slack_after_prefill_ms = engine_init_ms - b.cpu_prefill_ms;
        if (b.cpu_decode_one_step_ms > 1e-12) {
            b.decodes_during_engine_prep = static_cast<int>(std::floor(
                b.engine_slack_after_prefill_ms / b.cpu_decode_one_step_ms
            ));
        }
    }
    b.cpu_first_token_path_ms = b.cpu_prefill_ms + b.cpu_decode_one_step_ms;

    b.total_ms = b.gpu_handoff_ttft_ms;
    return b;
}

void log_tokenwise_breakdown(
    const TokenwiseBreakdown& b,
    int    cpu_length,
    int    gpu_load_layers,
    double engine_init_ms,
    bool   debug
) {
    if (!debug) {
        return;
    }
    std::fprintf(
        stderr,
        "\n[loading_perf_solver] tokenwise breakdown (cpu_tokens=%d gpu_preload_layers=%d)\n",
        cpu_length, gpu_load_layers
    );
    std::fprintf(
        stderr,
        "  phase1 parallel = max(cpu_prefill_ms=%.3f, "
        "engine_init_ms=%.3f) = %.3f ms\n",
        b.cpu_prefill_ms, engine_init_ms, b.phase1_parallel_ms
    );
    std::fprintf(
        stderr,
        "  weights_batch1_GB=%.6f  weights_batch2_GB=%.6f\n",
        b.weights1_GB, b.weights2_GB
    );
    std::fprintf(
        stderr,
        "  kv_transfer sequential: kv_GB=%.6f  kv_xfer_ms=%.3f ms\n",
        b.kv_GB, b.kv_transfer_ms
    );
    std::fprintf(
        stderr,
        "  phase2 paper loading/compute overlap (gpu_tail_prefill_ms=%.3f) "
        "= %.3f ms (gpu_tail_tokens=%d)\n",
        b.gpu_tail_prefill_ms,
        b.phase2_parallel_ms,
        b.gpu_tail_tokens
    );
    std::fprintf(
        stderr,
        "  gpu_handoff_TTFT_ms = phase1 + kv + phase2 = %.3f + %.3f + %.3f = %.3f ms\n",
        b.phase1_parallel_ms,
        b.kv_transfer_ms,
        b.phase2_parallel_ms,
        b.gpu_handoff_ttft_ms
    );
    if (b.cpu_decode_one_step_ms > 1e-12) {
        std::fprintf(
            stderr,
            "  engine_slack_after_prefill_ms=%.3f  decode_step_ms=%.3f  "
            "decodes_during_engine_prep=%d (debug)\n",
            b.engine_slack_after_prefill_ms,
            b.cpu_decode_one_step_ms,
            b.decodes_during_engine_prep
        );
        std::fprintf(
            stderr,
            "  cpu prefill + one decode_step (debug) = %.3f + %.3f = %.3f\n",
            b.cpu_prefill_ms,
            b.cpu_decode_one_step_ms,
            b.cpu_first_token_path_ms
        );
    }
    std::fprintf(
        stderr,
        "  TTFT (classic tokenwise ranking) = gpu_handoff = %.3f ms\n\n",
        b.total_ms
    );
}

/// Debug log for engine-overlap tokenwise: TTFT ranks as CPU full-prefill only;
/// cpu_compute_tokens may include extra decode steps (KV length > input_length).
void log_tokenwise_engine_overlap_breakdown(
    int    input_length,
    int    cpu_compute_tokens,
    int    gpu_load_layers,
    double engine_init_ms,
    double cpu_prefill_full_ms,
    double cpu_decode_intercept,
    double cpu_decode_coef_kv,
    double pcie_GBps,
    double embedding_GB,
    double layer_GB,
    int    num_layers,
    int    numa_nodes,
    int    tp_size
) {
    const int k_extra_decode = cpu_compute_tokens - input_length;
    double    decode_step =
        cpu_decode_intercept
        + cpu_decode_coef_kv * static_cast<double>(input_length);
    double wgb = weights_GB(embedding_GB, layer_GB, gpu_load_layers, numa_nodes, tp_size);
    double xfer_first = weight_transfer_ms(pcie_GBps, wgb);
    std::fprintf(
        stderr,
        "\n[loading_perf_solver] tokenwise ENGINE-OVERLAP plan "
        "(cpu_compute_tokens=%d = input_length=%d + decode_steps=%d, "
        "gpu_preload_layers=%d)\n",
        cpu_compute_tokens,
        input_length,
        k_extra_decode,
        gpu_load_layers
    );
    std::fprintf(
        stderr,
        "  ranking TTFT = CPU full prefill only = %.3f ms (not GPU handoff)\n",
        cpu_prefill_full_ms
    );
    std::fprintf(
        stderr,
        "  engine_init_ms=%.3f  cpu_prefill_full_ms=%.3f  decode_step_ms=%.3f "
        "(intercept + coef_kv*input_len)\n",
        engine_init_ms,
        cpu_prefill_full_ms,
        decode_step
    );
    std::fprintf(
        stderr,
        "  first-batch weights: gpu_load_layers=%d (indices [0, %d))  "
        "weights_GB=%.6f  xfer_ms=%.3f (must be <= engine_init for overlap preload)\n",
        gpu_load_layers,
        gpu_load_layers,
        wgb,
        xfer_first
    );
    std::fprintf(
        stderr,
        "  remaining layers on GPU after init: %d\n\n",
        num_layers - gpu_load_layers
    );
}

struct LayerwiseBreakdown {
    double weights1_GB;
    double load_first_ms;
    double cpu_layers_ms;
    double phase1_parallel_ms;
    double kv_GB;
    double act_GB;
    double kv_act_transfer_ms;
    int    gpu_compute_layers;
    double gpu_remaining_layers_ms;
    double total_ms;
};

LayerwiseBreakdown compute_layerwise_breakdown(
    double pcie_GBps,
    double embedding_GB,
    double layer_GB,
    double engine_init_ms,
    int    input_length,
    int    cpu_compute_layers,
    int    gpu_load_layers,
    int    num_layers,
    int    hidden_size,
    int    num_kv_heads,
    int    head_dim,
    int    tp_size,
    int    numa_nodes,
    double cpu_lw_intercept,
    double cpu_lw_coef_tokens,
    double gpu_lw_intercept,
    double gpu_lw_coef_tokens
) {
    LayerwiseBreakdown b{};
    b.weights1_GB = weights_GB(embedding_GB, layer_GB, gpu_load_layers, numa_nodes, tp_size);
    b.load_first_ms = weight_transfer_ms(pcie_GBps, b.weights1_GB);
    b.cpu_layers_ms = layerwise_ms(
        cpu_lw_intercept, cpu_lw_coef_tokens,
        static_cast<double>(input_length),
        static_cast<double>(cpu_compute_layers)
    );
    b.phase1_parallel_ms = std::max(b.cpu_layers_ms, engine_init_ms);
    b.kv_GB = kvcache_GB(
        input_length, cpu_compute_layers, num_kv_heads, head_dim, numa_nodes, tp_size
    );
    // With no CPU prefix there are no intermediate states to publish.
    // This is the same pure-GPU fallback as tokenwise's zero-token plan.
    b.act_GB = cpu_compute_layers > 0
        ? hidden_GB(input_length, hidden_size, numa_nodes, tp_size) : 0.0;
    b.kv_act_transfer_ms = cpu_compute_layers > 0
        ? transfer_ms(pcie_GBps, b.kv_GB + b.act_GB) : 0.0;
    b.gpu_compute_layers = num_layers - cpu_compute_layers;
    b.gpu_remaining_layers_ms = layerwise_ms(
        gpu_lw_intercept, gpu_lw_coef_tokens,
        static_cast<double>(input_length),
        static_cast<double>(b.gpu_compute_layers)
    );
    // Both handoff modes use the paper's intra-node loading limit.
    const double start_ms = b.phase1_parallel_ms + b.kv_act_transfer_ms;
    const double compute = gpu_lw_intercept + gpu_lw_coef_tokens * input_length;
    b.total_ms = b.gpu_compute_layers == 0 ? b.cpu_layers_ms : start_ms
        + intra_node_remaining_ms(start_ms, b.gpu_compute_layers,
            b.gpu_remaining_layers_ms, compute, pcie_GBps, layer_GB,
            numa_nodes, tp_size);
    return b;
}

void log_layerwise_breakdown(
    const LayerwiseBreakdown& b,
    int    cpu_compute_layers,
    int    gpu_preload_layers,
    double engine_init_ms,
    bool   debug
) {
    if (!debug) {
        return;
    }
    std::fprintf(
        stderr,
        "\n[loading_perf_solver] layerwise breakdown (cpu_layers=%d gpu_preload_layers=%d)\n",
        cpu_compute_layers, gpu_preload_layers
    );
    std::fprintf(
        stderr,
        "  phase1 parallel = max(cpu_layers_ms=%.3f, engine_init_ms=%.3f) "
        "= %.3f ms\n",
        b.cpu_layers_ms, engine_init_ms, b.phase1_parallel_ms
    );
    std::fprintf(stderr, "  weights_preload_GB=%.6f\n", b.weights1_GB);
    std::fprintf(
        stderr,
        "  phase2 sequential KV+hidden xfer: kv_GB=%.6f act_GB=%.6f xfer_ms=%.3f\n",
        b.kv_GB, b.act_GB, b.kv_act_transfer_ms
    );
    std::fprintf(
        stderr,
        "  phase3 gpu remaining layers=%d  gpu_layerwise_ms=%.3f\n",
        b.gpu_compute_layers, b.gpu_remaining_layers_ms
    );
    std::fprintf(
        stderr,
        "  TTFT = T_st + max(T_G, T_limit) (all-CPU returns CPU completion) = %.3f ms\n\n",
        b.total_ms
    );
}

// Time to load one transformer layer's weights (no embedding), for pen_std scaling.
double load_ms_per_layer(
    double pcie_GBps,
    double layer_GB,
    int numa_nodes,
    int tp_size
) {
    return weight_transfer_ms(pcie_GBps, weights_GB(0.0, layer_GB, 1, numa_nodes, tp_size));
}

double load_stage_weights_ms(
    double pcie_GBps,
    double embedding_GB,
    double layer_GB,
    int stage_idx,
    int ni,
    int numa_nodes,
    int tp_size
) {
    double emb = (stage_idx == 0) ? embedding_GB : 0.0;
    return weight_transfer_ms(pcie_GBps, weights_GB(emb, layer_GB, ni, numa_nodes, tp_size));
}

// Activation / KV bubble between upstream and this PP stage (ms).
double trans_to_pp_stage_ms(
    int stage_idx,
    int nc,
    double pcie_GBps,
    double rdma_GBps,
    int input_length,
    int num_kv_heads,
    int head_dim,
    int hidden_size,
    int numa_nodes,
    int tp_size
) {
    if (stage_idx == 0) {
        if (nc <= 0) {
            return 0.0;
        }
        double vol = kvcache_GB(input_length, nc, num_kv_heads, head_dim, numa_nodes, tp_size)
                   + hidden_GB(input_length, hidden_size, numa_nodes, tp_size);
        return transfer_ms(pcie_GBps, vol);
    }
    return transfer_ms(rdma_GBps, hidden_GB(input_length, hidden_size, numa_nodes, tp_size));
}

double penalty_std_shortfall(
    int nc,
    const std::vector<int>& n_gpu,
    int num_layers,
    int pp_size,
    int target_l,
    double t_load_per_layer_ms,
    bool layer_pipeline
) {
    int off = nc;
    double total_missing = 0.0;
    for (int i = 0; i < pp_size; ++i) {
        int std_lo = i * target_l;
        int std_hi = (i == pp_size - 1) ? num_layers : (i + 1) * target_l;
        if (std_hi <= std_lo) {
            continue;
        }
        int act_lo = 0;
        int act_hi = 0;
        if (layer_pipeline && i == 0) {
            act_lo = 0;
            act_hi = nc + n_gpu[0];
            off    = act_hi;
        } else {
            act_lo = off;
            act_hi = off + n_gpu[i];
            off    = act_hi;
        }
        int overlap = std::max(0, std::min(std_hi, act_hi) - std::max(std_lo, act_lo));
        int missing_i = (std_hi - std_lo) - overlap;
        total_missing += static_cast<double>(std::max(0, missing_i));
    }
    return total_missing * t_load_per_layer_ms;
}

// Recursively enumerate monotone GPU layer counts (Ni >= 0, sum = remaining_n).
template <typename Visitor>
void visit_partitions_full_n_on_gpu(
    int remaining_n,
    int remaining_p,
    int prev,
    int l_min,
    int l_max,
    std::vector<int>& cur,
    Visitor&& emit
) {
    if (remaining_p < 1 || remaining_n < 0) return;
    // A monotone suffix cannot recover if its minimum sum exceeds the
    // remaining layers, or its maximum sum is too small. Zero is legal only
    // before the first positive stage. Preserve the original traversal order.
    const int minimum = prev > 0 ? std::max(prev, l_min) : 0;
    if (static_cast<long long>(minimum) * remaining_p > remaining_n ||
        static_cast<long long>(l_max) * remaining_p < remaining_n) return;
    if (remaining_p == 1) {
        if (remaining_n == 0) {
            if (prev < 0 || 0 >= prev) {
                cur.push_back(0);
                emit(cur);
                cur.pop_back();
            }
        } else if (remaining_n >= l_min && remaining_n <= l_max) {
            if (prev < 0 || remaining_n >= prev) {
                cur.push_back(remaining_n);
                emit(cur);
                cur.pop_back();
            }
        }
        return;
    }
    const int upper = std::min(l_max, remaining_n / remaining_p);
    // Even filling every later stage to l_max cannot compensate for a
    // smaller first stage. Compute in 64 bits before narrowing: callers may
    // supply large stage counts even though the actual layer count is small.
    const int lower = static_cast<int>(std::max<long long>(minimum,
        static_cast<long long>(remaining_n) -
        static_cast<long long>(remaining_p - 1) * l_max));
    for (int li = lower; li <= upper; ++li) {
        if (li > 0 && (li < l_min || li > l_max)) {
            continue;
        }
        if (prev >= 0 && li < prev) {
            continue;
        }
        cur.push_back(li);
        visit_partitions_full_n_on_gpu(
            remaining_n - li, remaining_p - 1, li, l_min, l_max, cur, emit
        );
        cur.pop_back();
    }
}

// Exact endpoint form of E(j)=max(E(j-1), L(j))+compute for
// constant-sized layers loading sequentially from time zero.
double pp_stage_completion(double ready, int stage, int layers, double compute,
                           double pcie_GBps, double embedding_GB, double layer_GB,
                           int numa_nodes, int tp_size) {
    if (layers == 0) return ready;
    const double layer_load = gWeightLoadingMultiplier * 1000.0
        * weights_GB(0.0, layer_GB, 1, numa_nodes, tp_size) / pcie_GBps;
    const double first_ready = load_stage_weights_ms(
        pcie_GBps, embedding_GB, layer_GB, stage, 1, numa_nodes, tp_size);
    return std::max(ready + layers * compute,
        first_ready + compute + (layers - 1) * std::max(layer_load, compute));
}

// Each stage waits for upstream states and initialization, then executes
// layers as their asynchronously loaded weights become ready. One transfer
// setup is charged per stage, followed by payload time per layer; the final
// layer's computation must remain after its weight readiness time.
double pp_ttft_layer_pipeline(
    int nc,
    const std::vector<int>& n_gpu,
    int pp_size,
    int num_layers,
    double engine_init_ms,
    double pcie_GBps,
    double rdma_GBps,
    double embedding_GB,
    double layer_GB,
    int input_length,
    int num_kv_heads,
    int head_dim,
    int hidden_size,
    int numa_nodes,
    int tp_size,
    double cpu_lw_intercept,
    double cpu_lw_coef_tokens,
    double gpu_lw_intercept,
    double gpu_lw_coef_tokens,
    bool debug = false
) {
    double per_layer_cpu = cpu_lw_intercept + cpu_lw_coef_tokens * input_length;
    double t_cpu         = nc * per_layer_cpu;

    if (debug) {
        std::fprintf(
            stderr,
            "[loading_perf_solver] pp_ttft_layer_pipeline: nc=%d pp_size=%d input_len=%d "
            "per_layer_cpu_ms=%.4f t_cpu_total_ms=%.3f\n",
            nc, pp_size, input_length, per_layer_cpu, t_cpu
        );
        std::fprintf(stderr, "  n_gpu per stage:");
        for (int x : n_gpu) {
            std::fprintf(stderr, " %d", x);
        }
        std::fprintf(stderr, "\n");
    }

    if (nc >= num_layers) {
        double t = std::max(engine_init_ms, t_cpu);
        if (debug) {
            std::fprintf(
                stderr,
                "  all layers on CPU: TTFT = max(engine_init=%.3f, t_cpu=%.3f) = %.3f ms\n",
                engine_init_ms, t_cpu, t
            );
        }
        return t;
    }

    double per_layer_gpu = gpu_lw_intercept + gpu_lw_coef_tokens * input_length;

    std::vector<double> ends;
    ends.reserve(pp_size);
    for (int i = 0; i < pp_size; ++i) {
        const int ni = n_gpu[i];
        if (ni == 0) {
            // An empty stage forwards the upstream completion after its
            // activation transfer; this keeps asymmetric partitions valid.
            const double upstream = i == 0 ? t_cpu : ends.back();
            ends.push_back(upstream + trans_to_pp_stage_ms(
                i, nc, pcie_GBps, rdma_GBps, input_length,
                num_kv_heads, head_dim, hidden_size, numa_nodes, tp_size));
            continue;
        }
        const double upstream = i == 0 ? t_cpu : ends.back();
        const double transfer_in = trans_to_pp_stage_ms(
            i, nc, pcie_GBps, rdma_GBps, input_length,
            num_kv_heads, head_dim, hidden_size, numa_nodes, tp_size);
        const double ready = std::max(upstream + transfer_in, engine_init_ms);
        ends.push_back(pp_stage_completion(ready, i, ni, per_layer_gpu,
            pcie_GBps, embedding_GB, layer_GB, numa_nodes, tp_size));
    }
    if (debug) {
        std::fprintf(
            stderr,
            "  PP layer_pipeline TTFT (last stage end) = %.3f ms\n\n",
            ends.back()
        );
    }
    return ends.back();
}

// Tokenwise CPU prefix with PP GPU suffix, matching the PP0 KV relay path.
double pp_ttft_token_cpu(
    const std::vector<int>& n_gpu,
    int pp_size,
    int num_layers,
    int input_length,
    int s_cpu,
    double engine_init_ms,
    double pcie_GBps,
    double rdma_GBps,
    double embedding_GB,
    double layer_GB,
    int num_kv_heads,
    int head_dim,
    int hidden_size,
    int numa_nodes,
    int tp_size,
    double cpu_tw_intercept,
    double cpu_tw_coef_tokens,
    double gpu_tw_intercept,
    double gpu_tw_coef_tokens,
    bool debug = false
) {
    const int cpu_tokens = std::clamp(s_cpu, 0, input_length);
    const int gpu_tokens = input_length - cpu_tokens;
    // Tokenwise profiles describe the whole model, not one layer.
    const double cpu_done = cpu_tokens > 0
        ? tokenwise_ms(cpu_tw_intercept, cpu_tw_coef_tokens, cpu_tokens) : 0.0;
    if (gpu_tokens == 0) {
        // The CPU has already emitted the first token. Handoff affects
        // subsequent decode latency, not this request's TTFT.
        return cpu_done;
    }
    const double compute = tokenwise_ms(
        gpu_tw_intercept, gpu_tw_coef_tokens, gpu_tokens) / num_layers;

    // Current engine: PP0 imports all prefix KV from host, then forwards
    // downstream layers through its relay buffer before its first forward.
    // All this work is therefore on the initial handoff critical path.
    double handoff = std::max(cpu_done, engine_init_ms);
    if (cpu_tokens > 0) {
        handoff += transfer_ms(pcie_GBps, kvcache_GB(cpu_tokens, num_layers,
            num_kv_heads, head_dim, numa_nodes, tp_size));
        for (int i = 1; i < pp_size; ++i) {
            if (n_gpu[i] > 0) {
                handoff += transfer_ms(rdma_GBps, kvcache_GB(cpu_tokens, n_gpu[i],
                    num_kv_heads, head_dim, numa_nodes, tp_size));
            }
        }
    }
    double end = handoff;
    for (int i = 0; i < pp_size; ++i) {
        if (i > 0) {
            end += transfer_ms(rdma_GBps,
                hidden_GB(gpu_tokens, hidden_size, numa_nodes, tp_size));
        }
        end = pp_stage_completion(end, i, n_gpu[i], compute,
            pcie_GBps, embedding_GB, layer_GB, numa_nodes, tp_size);
    }
    if (debug) {
        std::fprintf(stderr,
            "[loading_perf_solver] PP tokenwise: cpu_tokens=%d gpu_tokens=%d "
            "cpu_done=%.3f handoff=%.3f ttft=%.3f ms\n",
            cpu_tokens, gpu_tokens, cpu_done, handoff, end);
    }
    return end;
}

double ttft_tokenwise(
    double transfer_GBps,
    double embedding_GB,
    double layer_GB,
    double engine_init_ms,
    int input_length,
    int cpu_length,
    int gpu_load_layers,
    int num_layers,
    int hidden_size,
    int num_kv_heads,
    int head_dim,
    int tp_size,
    int numa_nodes,
    double cpu_tw_intercept,
    double cpu_tw_coef_tokens,
    double gpu_tw_intercept,
    double gpu_tw_coef_tokens,
    double cpu_decode_intercept,
    double cpu_decode_coef_kv
) {
    return compute_tokenwise_breakdown(
               transfer_GBps, embedding_GB, layer_GB, engine_init_ms, input_length,
               cpu_length, gpu_load_layers, num_layers, hidden_size, num_kv_heads,
               head_dim, tp_size, numa_nodes, cpu_tw_intercept, cpu_tw_coef_tokens,
               gpu_tw_intercept, gpu_tw_coef_tokens, cpu_decode_intercept,
               cpu_decode_coef_kv
    )
        .total_ms;
}

// Layerwise TTFT (single GPU, non-PP).
double ttft_layerwise(
    double pcie_GBps,
    double embedding_GB,
    double layer_GB,
    double engine_init_ms,
    int input_length,
    int cpu_compute_layers,
    int gpu_load_layers,
    int num_layers,
    int hidden_size,
    int num_kv_heads,
    int head_dim,
    int tp_size,
    int numa_nodes,
    double cpu_lw_intercept,
    double cpu_lw_coef_tokens,
    double gpu_lw_intercept,
    double gpu_lw_coef_tokens
) {
    return compute_layerwise_breakdown(
               pcie_GBps, embedding_GB, layer_GB, engine_init_ms, input_length,
               cpu_compute_layers, gpu_load_layers, num_layers, hidden_size,
               num_kv_heads, head_dim, tp_size, numa_nodes, cpu_lw_intercept,
               cpu_lw_coef_tokens, gpu_lw_intercept, gpu_lw_coef_tokens
    )
        .total_ms;
}

py::object solve_pp_lazy_load(
    double fixed_ms,
    double pcie_GBps,
    double rdma_GBps,
    int numa_nodes,
    double embedding_GB,
    double layer_GB,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int hidden_size,
    int tp_size,
    int pp_size,
    int input_length,
    double cpu_tw_intercept,
    double cpu_tw_coef_tokens,
    double gpu_tw_intercept,
    double gpu_tw_coef_tokens,
    double cpu_lw_intercept,
    double cpu_lw_coef_tokens,
    double gpu_lw_intercept,
    double gpu_lw_coef_tokens,
    double cpu_decode_intercept,
    double cpu_decode_coef_kv,
    bool debug
) {
    if (pp_size <= 1 || num_layers < 1) {
        return py::none();
    }

    // Enumerate every nonnegative monotone PP partition. The previous
    // target_l +/- 4 window dropped valid uneven recovery layouts.
    int target_l       = std::max(1, num_layers / pp_size);
    int l_min          = 0;
    int l_max          = num_layers;
    const int base     = num_layers / pp_size;
    const int rem      = num_layers % pp_size;

    double t_load_layer = load_ms_per_layer(pcie_GBps, layer_GB, numa_nodes, tp_size);

    // --- Scenario A: layer pipeline (Nc, N0..N_{P-1}), Nc + sum Ni = N
    //
    // nc runs 0 .. num_layers-1 (Nc == num_layers is all-CPU handled inside ttft helper).
    // N0 may be 0 ("virtual" stage): first GPU layers can start at PP rank 1+.
    // target0 = base + (rem > 0 ? 1 : 0) is the balanced allocation for PP stage 0.
    //
    // For each nc we evaluate two sets of GPU partitions:
    //  (a) Standard partitions generated within [l_min, l_max] per stage (Ni may be 0).
    //  (b) The "natural" balanced partition: n_gpu[0] = target0 - nc (may be 0),
    //      when >= 0; n_gpu[i] = base + (i<rem?1:0) for i>0.
    //      Included when nat[0] >= 0; penalty_std_shortfall is 0 for that template.
    double best_a_score = std::numeric_limits<double>::infinity();
    int best_nc         = 0;
    std::vector<int> best_n_gpu_a(pp_size, 0);

    const int target0_a = base + (rem > 0 ? 1 : 0);   // stage-0 balanced target

    for (int nc = 0; nc < num_layers; ++nc) {
        int r = num_layers - nc;
        std::vector<int> c2;
        std::vector<int> nat(pp_size);
        nat[0] = target0_a - nc;
        for (int i = 1; i < pp_size; ++i) nat[i] = base + (i < rem ? 1 : 0);
        bool natural_seen = false;
        auto evaluate = [&](const std::vector<int>& n_gpu) {
            natural_seen = natural_seen || n_gpu == nat;
            double ttft = pp_ttft_layer_pipeline(
                nc, n_gpu, pp_size, num_layers, fixed_ms, pcie_GBps, rdma_GBps,
                embedding_GB, layer_GB, input_length, num_kv_heads, head_dim,
                hidden_size, numa_nodes, tp_size,
                cpu_lw_intercept, cpu_lw_coef_tokens,
                gpu_lw_intercept, gpu_lw_coef_tokens
            );
            double pen = penalty_std_shortfall(
                nc, n_gpu, num_layers, pp_size, target_l, t_load_layer, true
            );
            double sc = ttft + kBetaStd * pen;
            if (sc < best_a_score) {
                best_a_score = sc;
                best_nc      = nc;
                best_n_gpu_a = n_gpu;
            }
        };
        visit_partitions_full_n_on_gpu(r, pp_size, -1, l_min, l_max, c2, evaluate);
        // This legacy template need not be monotone. Keep it last so exact
        // ties retain the same winner, without storing/scanning all partitions.
        if (nat[0] >= 0 && !natural_seen) evaluate(nat);
    }

    // --- Scenario B: token split + same GPU partitions as full N
    double best_b_score = std::numeric_limits<double>::infinity();
    int best_s_cpu      = 0;
    std::vector<int> best_n_gpu_b(pp_size, 0);

    // Include the complete CPU prefill even for prompts not divisible by
    // the enumeration step: it may emit the first token before GPU init.
    std::vector<int> cur;
    visit_partitions_full_n_on_gpu(num_layers, pp_size, -1, l_min, l_max, cur,
        [&](const std::vector<int>& n_gpu) {
        const double pen = penalty_std_shortfall(
            0, n_gpu, num_layers, pp_size, target_l, t_load_layer, false);
        for (int split = 0; split <= (input_length + kTokenCpuStep - 1) / kTokenCpuStep; ++split) {
            const int s_cpu = std::min(split * kTokenCpuStep, input_length);
            double ttft = pp_ttft_token_cpu(
                n_gpu, pp_size, num_layers, input_length, s_cpu, fixed_ms,
                pcie_GBps, rdma_GBps, embedding_GB, layer_GB,
                num_kv_heads, head_dim, hidden_size, numa_nodes, tp_size,
                cpu_tw_intercept, cpu_tw_coef_tokens,
                gpu_tw_intercept, gpu_tw_coef_tokens
            );
            double sc = ttft + kBetaStd * pen;
            // The former split-major traversal preferred smaller CPU splits,
            // then lexicographically earlier partitions on an exact tie.
            if (sc < best_b_score || (sc == best_b_score &&
                (s_cpu < best_s_cpu || (s_cpu == best_s_cpu && n_gpu < best_n_gpu_b)))) {
                best_b_score = sc;
                best_s_cpu   = s_cpu;
                best_n_gpu_b = n_gpu;
            }
        }
    });

    // --- Scenario C: engine-overlap tokenwise (same idea as single-GPU block when
    // cpu_decode_* is set and full-input CPU prefill fits in engine init).
    // Ranking score uses TTFT = cpu_prefill_full only; each PP stage's first weight
    // batch must DMA within fixed_ms (per-stage load_stage_weights_ms).
    double                    best_c_score   = std::numeric_limits<double>::infinity();
    int                       best_c_second  = 0;
    std::vector<int>          best_n_gpu_c(pp_size, 0);
    bool                      have_c         = false;
    const double              cpu_prefill_full_pp = tokenwise_ms(
        cpu_tw_intercept, cpu_tw_coef_tokens, static_cast<double>(input_length)
    );
    const bool decode_profile_ok_pp =
        (cpu_decode_intercept != 0.0 || cpu_decode_coef_kv != 0.0);
    if (decode_profile_ok_pp && input_length > 0
        && cpu_prefill_full_pp + 1e-9 < fixed_ms) {
        double decode_step = cpu_decode_intercept + cpu_decode_coef_kv * input_length;
        if (decode_step <= 0.0) {
            decode_step = 1e-6;
        }
        int k_extra = static_cast<int>(
            std::floor((fixed_ms - cpu_prefill_full_pp) / decode_step)
        );
        if (k_extra < 0) {
            k_extra = 0;
        }
        const int cpu_len_ov = input_length + k_extra;
        visit_partitions_full_n_on_gpu(num_layers, pp_size, -1, l_min, l_max, cur,
            [&](const std::vector<int>& n_gpu) {
            bool load_ok = true;
            for (int si = 0; si < pp_size; ++si) {
                double load_i = load_stage_weights_ms(
                    pcie_GBps, embedding_GB, layer_GB, si, n_gpu[si], numa_nodes, tp_size
                );
                if (load_i > fixed_ms + 1e-9) {
                    load_ok = false;
                    break;
                }
            }
            if (!load_ok) {
                return;
            }
            double pen_c = penalty_std_shortfall(
                0, n_gpu, num_layers, pp_size, target_l, t_load_layer, false
            );
            double sc = cpu_prefill_full_pp + kBetaStd * pen_c;
            if (sc < best_c_score - 1e-9) {
                best_c_score  = sc;
                best_c_second = cpu_len_ov;
                best_n_gpu_c  = n_gpu;
                have_c        = true;
            } else if (std::abs(sc - best_c_score) <= 1e-9 && have_c) {
                int s_new = 0;
                int s_old = 0;
                for (int t : n_gpu) {
                    s_new += t;
                }
                for (int t : best_n_gpu_c) {
                    s_old += t;
                }
                if (s_new > s_old) {
                    best_c_second = cpu_len_ov;
                    best_n_gpu_c  = n_gpu;
                }
            }
        });
    }

    // Min score among A, B, C. On exact tie, prefer A over B over C (legacy A vs B: layerwise
    // wins a tie with classic tokenwise).
    double min_score  = best_a_score;
    int    winner     = 0;  // 0=A layer, 1=B token, 2=C engine-overlap token
    if (best_b_score < min_score - 1e-9) {
        min_score = best_b_score;
        winner    = 1;
    }
    if (have_c && best_c_score < min_score - 1e-9) {
        min_score = best_c_score;
        winner    = 2;
    }

    const bool  use_layer = (winner == 0);
    int         second    = 0;
    std::vector<int> n_gpu_win(pp_size, 0);
    if (winner == 0) {
        second   = best_nc;
        n_gpu_win = best_n_gpu_a;
    } else if (winner == 1) {
        second   = best_s_cpu;
        n_gpu_win = best_n_gpu_b;
    } else {
        second   = best_c_second;
        n_gpu_win = best_n_gpu_c;
    }
    const std::vector<int>& n_gpu = n_gpu_win;

    py::list stages;
    int g = use_layer ? best_nc : 0;
    for (int i = 0; i < pp_size; ++i) {
        py::list stage_layers;
        for (int j = 0; j < n_gpu[i]; ++j) {
            stage_layers.append(g++);
        }
        stages.append(stage_layers);
    }

    // B (classic token split) and C (engine-overlap tokenwise) both use vLLM load_method
    // "tokenwise" — only A is "layerwise".
    std::string method = (winner == 0) ? "layerwise" : "tokenwise";

    // Ranking TTFT (pure, no penalty) for the winning plan.
    double ranking_ttft_ms = 0.0;
    if (winner == 0) {
        ranking_ttft_ms = pp_ttft_layer_pipeline(
            second, n_gpu, pp_size, num_layers, fixed_ms, pcie_GBps, rdma_GBps,
            embedding_GB, layer_GB, input_length, num_kv_heads, head_dim,
            hidden_size, numa_nodes, tp_size,
            cpu_lw_intercept, cpu_lw_coef_tokens,
            gpu_lw_intercept, gpu_lw_coef_tokens
        );
    } else if (winner == 1) {
        ranking_ttft_ms = pp_ttft_token_cpu(
            n_gpu, pp_size, num_layers, input_length, second, fixed_ms,
            pcie_GBps, rdma_GBps, embedding_GB, layer_GB,
            num_kv_heads, head_dim, hidden_size, numa_nodes, tp_size,
            cpu_tw_intercept, cpu_tw_coef_tokens,
            gpu_tw_intercept, gpu_tw_coef_tokens
        );
    } else {
        ranking_ttft_ms = cpu_prefill_full_pp;
    }

    if (debug) {
        std::fprintf(stderr, "\n[loading_perf_solver] ========== PP chosen plan ==========\n");
        if (have_c) {
            std::fprintf(
                stderr,
                "  winner=%d method=%s second=%d  score_A=%.3f score_B=%.3f score_C=%.3f  "
                "(C=engine-overlap tokenwise; %.2f*pen)\n",
                winner,
                method.c_str(),
                second,
                best_a_score,
                best_b_score,
                best_c_score,
                kBetaStd
            );
        } else {
            std::fprintf(
                stderr,
                "  winner=%d method=%s second=%d  score_A=%.3f score_B=%.3f score_C=n/a  "
                "(C=engine-overlap tokenwise; %.2f*pen)\n",
                winner,
                method.c_str(),
                second,
                best_a_score,
                best_b_score,
                kBetaStd
            );
        }
        if (use_layer) {
            pp_ttft_layer_pipeline(
                second, n_gpu, pp_size, num_layers, fixed_ms, pcie_GBps, rdma_GBps,
                embedding_GB, layer_GB, input_length, num_kv_heads, head_dim,
                hidden_size, numa_nodes, tp_size, cpu_lw_intercept, cpu_lw_coef_tokens,
                gpu_lw_intercept, gpu_lw_coef_tokens, true
            );
        } else if (winner == 1) {
            pp_ttft_token_cpu(
                n_gpu, pp_size, num_layers, input_length, second, fixed_ms,
                pcie_GBps, rdma_GBps, embedding_GB, layer_GB, num_kv_heads, head_dim,
                hidden_size, numa_nodes, tp_size, cpu_tw_intercept, cpu_tw_coef_tokens,
                gpu_tw_intercept, gpu_tw_coef_tokens, true
            );
        } else if (have_c) {
            std::fprintf(
                stderr,
                "[loading_perf_solver] PP engine-overlap: ranking TTFT = CPU prefill = %.3f ms, "
                "cpu_compute_tokens(second)=%d, per_stage_layer_counts",
                cpu_prefill_full_pp,
                second
            );
            for (int i = 0; i < pp_size; ++i) {
                std::fprintf(stderr, " %d", n_gpu[i]);
            }
            std::fprintf(
                stderr,
                "  (each stage first-batch load must be <= engine_init=%.3f ms)\n\n",
                fixed_ms
            );
        }
    }

    return py::make_tuple(method, second, stages, ranking_ttft_ms);
}

// Static plan evaluated with the accurate (bandwidth-aware) cost model.
//
// For both single-GPU and PP, two candidates are evaluated and the better is returned:
//
//   Tokenwise candidate:
//     cpu_tokens = max tokens CPU can prefill within engine_init_ms.
//     Single GPU : gpu_load_layers = num_layers; TTFT via compute_tokenwise_breakdown.
//     PP         : balanced GPU partition (each stage i gets base+(i<rem?1:0) layers);
//                  TTFT via pp_ttft_token_cpu(s_cpu=cpu_tokens).
//
//   Layerwise candidate:
//     cpu_layers = max layers CPU can process within engine_init_ms.
//     Single GPU : gpu_load_layers = num_layers; TTFT via compute_layerwise_breakdown.
//     PP         : stage 0 gets (target0 - nc) GPU layers where nc = min(cpu_layers, target0-1),
//                  all other stages get their balanced share; TTFT via pp_ttft_layer_pipeline.
//
// Returns 4-tuple (method, second, layer_idxes, ttft_ms).
py::object solve_static_lazy_load_cpp(
    double fixed_ms,
    double pcie_GBps,
    double rdma_GBps,
    int    numa_nodes,
    double embedding_GB,
    double layer_GB,
    int    num_layers,
    int    num_kv_heads,
    int    head_dim,
    int    hidden_size,
    int    tp_size,
    int    pp_size,
    int    input_length,
    double cpu_tw_intercept,
    double cpu_tw_coef_tokens,
    double gpu_tw_intercept,
    double gpu_tw_coef_tokens,
    double cpu_lw_intercept,
    double cpu_lw_coef_tokens,
    double gpu_lw_intercept,
    double gpu_lw_coef_tokens,
    double cpu_decode_intercept,
    double cpu_decode_coef_kv,
    bool   debug = false,
    double transfer_setup_ms = 10.0,
    double weight_loading_multiplier = 1.0
) {
    gTransferSetupMs = std::max(transfer_setup_ms, 0.0);
    if (!std::isfinite(weight_loading_multiplier) || weight_loading_multiplier <= 0.0)
        throw std::invalid_argument("weight_loading_multiplier must be finite and positive");
    gWeightLoadingMultiplier = weight_loading_multiplier;
    if (num_layers < 1) {
        return py::none();
    }

    // ── Shared: compute cpu_tokens (tokenwise) and cpu_layers (layerwise) ─ //

    int cpu_tokens = 0;
    if (input_length > 0) {
        if (cpu_tw_coef_tokens > 1e-12) {
            double prefill_ms =
                cpu_tw_intercept + cpu_tw_coef_tokens * static_cast<double>(input_length);
            if (prefill_ms > fixed_ms) {
                cpu_tokens = std::max(
                    0,
                    static_cast<int>((fixed_ms - cpu_tw_intercept) / cpu_tw_coef_tokens)
                );
            } else {
                cpu_tokens = input_length;
                double decode_step =
                    cpu_decode_intercept
                    + cpu_decode_coef_kv * static_cast<double>(input_length);
                if (decode_step > 1e-12) {
                    cpu_tokens += static_cast<int>(
                        std::floor((fixed_ms - prefill_ms) / decode_step)
                    );
                }
            }
        } else {
            cpu_tokens = input_length;
        }
        cpu_tokens = std::max(0, cpu_tokens);
    }

    double per_layer_ms =
        cpu_lw_intercept + cpu_lw_coef_tokens * static_cast<double>(input_length);
    int cpu_layers = 0;
    if (per_layer_ms > 1e-12) {
        cpu_layers = std::min(
            static_cast<int>(std::floor(fixed_ms / per_layer_ms)), num_layers
        );
    } else {
        cpu_layers = num_layers;
    }
    cpu_layers = std::max(0, cpu_layers);

    // ── PP static: tokenwise + layerwise candidates ────────────────────── //
    if (pp_size > 1) {
        int base = num_layers / pp_size;
        int rem  = num_layers % pp_size;

        // Balanced GPU partition shared by both candidates.
        std::vector<int> n_gpu_balanced(pp_size);
        for (int i = 0; i < pp_size; ++i) {
            n_gpu_balanced[i] = base + (i < rem ? 1 : 0);
        }

        // ── PP tokenwise: each GPU stage gets its balanced share ──────── //
        double ttft_tw = pp_ttft_token_cpu(
            n_gpu_balanced, pp_size, num_layers, input_length, cpu_tokens, fixed_ms,
            pcie_GBps, rdma_GBps, embedding_GB, layer_GB,
            num_kv_heads, head_dim, hidden_size, numa_nodes, tp_size,
            cpu_tw_intercept, cpu_tw_coef_tokens,
            gpu_tw_intercept, gpu_tw_coef_tokens,
            debug
        );

        // ── PP layerwise: stage 0 = (target0 - nc) GPU layers ─────────── //
        int target0 = n_gpu_balanced[0];  // = base + (0 < rem ? 1 : 0)
        int nc      = std::min(cpu_layers, target0 - 1);
        nc          = std::max(0, nc);

        std::vector<int> n_gpu_lw = n_gpu_balanced;
        n_gpu_lw[0] = target0 - nc;

        double ttft_lw = pp_ttft_layer_pipeline(
            nc, n_gpu_lw, pp_size, num_layers, fixed_ms,
            pcie_GBps, rdma_GBps, embedding_GB, layer_GB,
            input_length, num_kv_heads, head_dim, hidden_size, numa_nodes, tp_size,
            cpu_lw_intercept, cpu_lw_coef_tokens,
            gpu_lw_intercept, gpu_lw_coef_tokens,
            debug
        );

        if (debug) {
            std::fprintf(
                stderr,
                "\n[loading_perf_solver] ===== static PP plan =====\n"
                "  tokenwise: s_cpu=%d  ttft=%.3f ms\n"
                "  layerwise: nc=%d  n_gpu[0]=%d  ttft=%.3f ms\n"
                "  winner=%s\n\n",
                cpu_tokens, ttft_tw,
                nc, n_gpu_lw[0], ttft_lw,
                ttft_tw <= ttft_lw ? "tokenwise" : "layerwise"
            );
        }

        py::list stages;
        if (ttft_tw <= ttft_lw) {
            int g = 0;
            for (int i = 0; i < pp_size; ++i) {
                py::list sl;
                for (int j = 0; j < n_gpu_balanced[i]; ++j) sl.append(g++);
                stages.append(sl);
            }
            return py::make_tuple(std::string("tokenwise"), cpu_tokens, stages, ttft_tw);
        } else {
            int g = nc;
            for (int i = 0; i < pp_size; ++i) {
                py::list sl;
                for (int j = 0; j < n_gpu_lw[i]; ++j) sl.append(g++);
                stages.append(sl);
            }
            return py::make_tuple(std::string("layerwise"), nc, stages, ttft_lw);
        }
    }

    // ── Single-GPU static: accurate cost model for both candidates ────── //
    // (cpu_tokens and cpu_layers already computed in the shared block above)

    // Tokenwise candidate: gpu_load = all layers.
    double ttft_tw = std::numeric_limits<double>::infinity();
    if (input_length > 0) {
        TokenwiseBreakdown tb = compute_tokenwise_breakdown(
            pcie_GBps, embedding_GB, layer_GB, fixed_ms,
            input_length, cpu_tokens, num_layers,
            num_layers, hidden_size, num_kv_heads, head_dim, tp_size, numa_nodes,
            cpu_tw_intercept, cpu_tw_coef_tokens,
            gpu_tw_intercept, gpu_tw_coef_tokens,
            cpu_decode_intercept, cpu_decode_coef_kv
        );
        ttft_tw = tb.total_ms;
        if (debug) log_tokenwise_breakdown(tb, cpu_tokens, num_layers, fixed_ms, true);
    }

    // Layerwise candidate: gpu_load = all layers.
    LayerwiseBreakdown lb = compute_layerwise_breakdown(
        pcie_GBps, embedding_GB, layer_GB, fixed_ms,
        input_length, cpu_layers, num_layers,
        num_layers, hidden_size, num_kv_heads, head_dim, tp_size, numa_nodes,
        cpu_lw_intercept, cpu_lw_coef_tokens,
        gpu_lw_intercept, gpu_lw_coef_tokens
    );
    double ttft_lw = lb.total_ms;
    if (debug) log_layerwise_breakdown(lb, cpu_layers, num_layers, fixed_ms, true);

    if (debug) {
        std::fprintf(
            stderr,
            "[loading_perf_solver] static single-GPU: "
            "ttft_tw=%.3f ms (cpu_tokens=%d)  "
            "ttft_lw=%.3f ms (cpu_layers=%d)  winner=%s\n\n",
            ttft_tw, cpu_tokens, ttft_lw, cpu_layers,
            ttft_tw <= ttft_lw ? "tokenwise" : "layerwise"
        );
    }

    py::list lipp;
    if (ttft_tw <= ttft_lw) {
        py::list layers;
        for (int i = 0; i < num_layers; ++i) layers.append(i);
        lipp.append(layers);
        return py::make_tuple(std::string("tokenwise"), cpu_tokens, lipp, ttft_tw);
    } else {
        py::list layers;
        // Compute layers first, then prefix fill-in (matches expand_first_stage convention).
        for (int i = cpu_layers; i < num_layers; ++i) layers.append(i);
        for (int i = 0;          i < cpu_layers;  ++i) layers.append(i);
        lipp.append(layers);
        return py::make_tuple(std::string("layerwise"), cpu_layers, lipp, ttft_lw);
    }
}

}  // namespace

py::object solve_lazy_load_method_cpp(
    double fixed_ms,
    double pcie_GBps,
    double rdma_GBps,
    int numa_nodes,
    double embedding_GB,
    double layer_GB,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int hidden_size,
    int tp_size,
    int pp_size,
    int input_length,
    double cpu_tw_intercept,
    double cpu_tw_coef_tokens,
    double gpu_tw_intercept,
    double gpu_tw_coef_tokens,
    double cpu_lw_intercept,
    double cpu_lw_coef_tokens,
    double gpu_lw_intercept,
    double gpu_lw_coef_tokens,
    double cpu_decode_intercept,
    double cpu_decode_coef_kv,
    bool debug = false,
    double transfer_setup_ms = 10.0,
    double weight_loading_multiplier = 1.0,
    int max_cpu_tokens = -1,
    bool allow_cpu_decode = true
) {
    gTransferSetupMs = std::max(transfer_setup_ms, 0.0);
    if (!std::isfinite(weight_loading_multiplier) || weight_loading_multiplier <= 0.0)
        throw std::invalid_argument("weight_loading_multiplier must be finite and positive");
    gWeightLoadingMultiplier = weight_loading_multiplier;
    if (max_cpu_tokens < -1)
        throw std::invalid_argument("max_cpu_tokens must be nonnegative or -1");
    const int cpu_limit = max_cpu_tokens < 0 ? input_length
                                           : std::min(input_length, max_cpu_tokens);
    if (pp_size > 1) {
        if (max_cpu_tokens >= 0 || !allow_cpu_decode)
            throw std::invalid_argument("Constrained CPU token planning currently requires PP=1");
        return solve_pp_lazy_load(
            fixed_ms, pcie_GBps, rdma_GBps, numa_nodes, embedding_GB, layer_GB,
            num_layers, num_kv_heads, head_dim, hidden_size, tp_size, pp_size,
            input_length, cpu_tw_intercept, cpu_tw_coef_tokens,
            gpu_tw_intercept, gpu_tw_coef_tokens, cpu_lw_intercept, cpu_lw_coef_tokens,
            gpu_lw_intercept, gpu_lw_coef_tokens, cpu_decode_intercept,
            cpu_decode_coef_kv, debug
        );
    }

    double min_tokenwise_ttft = std::numeric_limits<double>::infinity();
    int best_tokenwise_cpu_compute_length = 0;
    int best_tokenwise_gpu_load_layers    = 0;
    bool best_tokenwise_is_engine_overlap = false;

    auto consider_classic_tokenwise = [&](int cc, int gl) {
        double t = ttft_tokenwise(
            pcie_GBps, embedding_GB, layer_GB, fixed_ms, input_length, cc, gl,
            num_layers, hidden_size, num_kv_heads, head_dim, tp_size, numa_nodes,
            cpu_tw_intercept, cpu_tw_coef_tokens, gpu_tw_intercept, gpu_tw_coef_tokens,
            0.0, 0.0
        );
        if (t < min_tokenwise_ttft - 1e-9) {
            min_tokenwise_ttft                 = t;
            best_tokenwise_cpu_compute_length = cc;
            best_tokenwise_gpu_load_layers    = gl;
            best_tokenwise_is_engine_overlap  = false;
        } else if (std::abs(t - min_tokenwise_ttft) <= 1e-9
                   && gl > best_tokenwise_gpu_load_layers) {
            best_tokenwise_cpu_compute_length = cc;
            best_tokenwise_gpu_load_layers    = gl;
            best_tokenwise_is_engine_overlap  = false;
        }
    };

    // Include every legal first-batch GPU layer count for every executable
    // CPU prefix. The transfer schedule may expose a different optimum than
    // the full-model preload, so neither dimension is sampled.
    for (int cc = 0; cc <= cpu_limit; ++cc) {
        for (int gl = 1; gl <= num_layers; ++gl) {
            consider_classic_tokenwise(cc, gl);
        }
    }

    const double cpu_prefill_full = tokenwise_ms(
        cpu_tw_intercept, cpu_tw_coef_tokens, static_cast<double>(input_length)
    );
    const bool decode_profile_ok =
        (cpu_decode_intercept != 0.0 || cpu_decode_coef_kv != 0.0);
    if (decode_profile_ok && input_length > 0 && input_length <= cpu_limit
            && cpu_prefill_full + 1e-9 < fixed_ms) {
        double decode_step = cpu_decode_intercept
            + cpu_decode_coef_kv * static_cast<double>(input_length);
        if (decode_step <= 0.0) {
            decode_step = 1e-6;
        }
        int k = static_cast<int>(
            std::floor((fixed_ms - cpu_prefill_full) / decode_step)
        );
        if (k < 0) {
            k = 0;
        }
        if (max_cpu_tokens >= 0)
            k = std::min(k, max_cpu_tokens - input_length);
        const int cpu_len_ov = input_length + (allow_cpu_decode ? k : 0);
        int       gpu_load_ov = 0;
        for (int L = num_layers; L >= 1; --L) {
            double wgb = weights_GB(embedding_GB, layer_GB, L, numa_nodes, tp_size);
            if (weight_transfer_ms(pcie_GBps, wgb) <= fixed_ms + 1e-9) {
                gpu_load_ov = L;
                break;
            }
        }
        if (gpu_load_ov > 0) {
            const double ttft_ov = cpu_prefill_full;
            if (ttft_ov < min_tokenwise_ttft - 1e-9) {
                min_tokenwise_ttft                 = ttft_ov;
                best_tokenwise_cpu_compute_length = cpu_len_ov;
                best_tokenwise_gpu_load_layers    = gpu_load_ov;
                best_tokenwise_is_engine_overlap  = true;
            } else if (std::abs(ttft_ov - min_tokenwise_ttft) <= 1e-9
                       && gpu_load_ov > best_tokenwise_gpu_load_layers) {
                best_tokenwise_cpu_compute_length = cpu_len_ov;
                best_tokenwise_gpu_load_layers    = gpu_load_ov;
                best_tokenwise_is_engine_overlap  = true;
            }
        }
    }

    py::list best_tokenwise_layers;
    for (int i = 0; i < best_tokenwise_gpu_load_layers; ++i) {
        best_tokenwise_layers.append(i);
    }

    double min_layerwise_ttft = std::numeric_limits<double>::infinity();
    int best_layerwise_cpu_compute_layers = 0;
    int best_layerwise_gpu_load_layers    = 0;

    // Enumerate execution boundaries only. Equation (3) determines loading
    // progress; the returned list submits the suffix asynchronously.
    {
        for (int cpu_compute_layers = 1; cpu_limit > 0 && cpu_compute_layers <= num_layers;
             ++cpu_compute_layers) {
            // Full CPU layerwise is executed via tokenwise takeover. A prompt
            // spanning multiple chunks cannot use this equivalent candidate.
            if (cpu_compute_layers == num_layers && input_length > cpu_limit)
                continue;
            const int gpu_load_layers = num_layers - cpu_compute_layers;
            double current_ttft = ttft_layerwise(
                pcie_GBps, embedding_GB, layer_GB, fixed_ms, input_length,
                cpu_compute_layers, gpu_load_layers, num_layers, hidden_size,
                num_kv_heads, head_dim, tp_size, numa_nodes,
                cpu_lw_intercept, cpu_lw_coef_tokens,
                gpu_lw_intercept, gpu_lw_coef_tokens
            );
            if (current_ttft < min_layerwise_ttft) {
                min_layerwise_ttft                 = current_ttft;
                best_layerwise_cpu_compute_layers = cpu_compute_layers;
                best_layerwise_gpu_load_layers    = gpu_load_layers;
            }
            else if (current_ttft == min_layerwise_ttft && gpu_load_layers > best_layerwise_gpu_load_layers) {
                best_layerwise_gpu_load_layers = gpu_load_layers;
            }
        }
    }

    py::list best_layerwise_layers;
    int start_idx = num_layers - best_layerwise_gpu_load_layers;
    for (int i = 0; i < best_layerwise_gpu_load_layers; ++i) {
        best_layerwise_layers.append(start_idx + i);
    }

    if (!std::isfinite(min_tokenwise_ttft) && !std::isfinite(min_layerwise_ttft)) {
        return py::none();
    }

    if (debug) {
        std::fprintf(
            stderr,
            "\n[loading_perf_solver] ========== single-GPU solve (pp_size=1) ==========\n"
        );
        std::fprintf(
            stderr,
            "  input_length=%d num_layers=%d pcie_GBps=%.3f engine_init_ms=%.3f\n",
            input_length, num_layers, pcie_GBps, fixed_ms
        );
        if (std::isfinite(min_tokenwise_ttft)) {
            if (best_tokenwise_is_engine_overlap) {
                const double cpu_pf_full = tokenwise_ms(
                    cpu_tw_intercept,
                    cpu_tw_coef_tokens,
                    static_cast<double>(input_length)
                );
                log_tokenwise_engine_overlap_breakdown(
                    input_length,
                    best_tokenwise_cpu_compute_length,
                    best_tokenwise_gpu_load_layers,
                    fixed_ms,
                    cpu_pf_full,
                    cpu_decode_intercept,
                    cpu_decode_coef_kv,
                    pcie_GBps,
                    embedding_GB,
                    layer_GB,
                    num_layers,
                    numa_nodes,
                    tp_size
                );
            } else {
                TokenwiseBreakdown tb = compute_tokenwise_breakdown(
                    pcie_GBps, embedding_GB, layer_GB, fixed_ms, input_length,
                    best_tokenwise_cpu_compute_length, best_tokenwise_gpu_load_layers,
                    num_layers, hidden_size, num_kv_heads, head_dim, tp_size, numa_nodes,
                    cpu_tw_intercept, cpu_tw_coef_tokens, gpu_tw_intercept, gpu_tw_coef_tokens,
                    cpu_decode_intercept, cpu_decode_coef_kv
                );
                log_tokenwise_breakdown(
                    tb, best_tokenwise_cpu_compute_length, best_tokenwise_gpu_load_layers,
                    fixed_ms, true
                );
            }
        }
        if (std::isfinite(min_layerwise_ttft)) {
            LayerwiseBreakdown lb = compute_layerwise_breakdown(
                pcie_GBps, embedding_GB, layer_GB, fixed_ms, input_length,
                best_layerwise_cpu_compute_layers, best_layerwise_gpu_load_layers,
                num_layers, hidden_size, num_kv_heads, head_dim, tp_size, numa_nodes,
                cpu_lw_intercept, cpu_lw_coef_tokens, gpu_lw_intercept, gpu_lw_coef_tokens
            );
            log_layerwise_breakdown(
                lb, best_layerwise_cpu_compute_layers, best_layerwise_gpu_load_layers,
                fixed_ms, true
            );
        }
        std::fprintf(
            stderr,
            "[loading_perf_solver] compare: ttft_tokenwise=%.3f ms  ttft_layerwise=%.3f ms  "
            "-> winner: %s\n\n",
            min_tokenwise_ttft,
            min_layerwise_ttft,
            min_tokenwise_ttft < min_layerwise_ttft ? "tokenwise" : "layerwise"
        );
    }

    // vLLM ``layer_idxes_by_pp`` outer length must equal ``pipeline_parallel_size`` (1 here).
    py::list layer_idxes_by_pp;
    if (min_tokenwise_ttft < min_layerwise_ttft) {
        layer_idxes_by_pp.append(best_tokenwise_layers);
        return py::make_tuple(
            std::string("tokenwise"),
            best_tokenwise_cpu_compute_length,
            layer_idxes_by_pp,
            min_tokenwise_ttft
        );
    }
    layer_idxes_by_pp.append(best_layerwise_layers);
    return py::make_tuple(
        std::string("layerwise"),
        best_layerwise_cpu_compute_layers,
        layer_idxes_by_pp,
        min_layerwise_ttft
    );
}

PYBIND11_MODULE(_loading_perf_profile_solver, m) {
    m.doc() = "Pybind11 lazy-load performance solver";
    m.def("evaluate_pp_tokenwise", [](const std::vector<int>& stages,
        int input_length, int cpu_tokens, double init_ms, double pcie, double rdma,
        double embedding, double layer, int heads, int head_dim, int hidden,
        int numa, int tp, double cpu_intercept, double cpu_slope,
        double gpu_intercept, double gpu_slope, double setup, double multiplier) {
        if (stages.empty() || input_length <= 0 || pcie <= 0 || rdma <= 0
            || numa <= 0 || tp <= 0 || !std::isfinite(multiplier) || multiplier <= 0)
            throw std::invalid_argument("positive stages, tokens, bandwidths and multiplier required");
        int layers = 0;
        for (int count : stages) {
            if (count < 0) throw std::invalid_argument("negative stage layer count");
            layers += count;
        }
        if (layers <= 0) throw std::invalid_argument("at least one model layer required");
        gTransferSetupMs = std::max(setup, 0.0);
        gWeightLoadingMultiplier = multiplier;
        return pp_ttft_token_cpu(stages, stages.size(), layers, input_length,
            cpu_tokens, init_ms, pcie, rdma, embedding, layer, heads, head_dim,
            hidden, numa, tp, cpu_intercept, cpu_slope, gpu_intercept, gpu_slope);
    }, "Evaluate a fixed PP tokenwise plan using the production cost model.");
    m.def(
        "solve_lazy_load_method_cpp",
        &solve_lazy_load_method_cpp,
        py::arg("fixed_ms"),
        py::arg("pcie_GBps"),
        py::arg("rdma_GBps"),
        py::arg("numa_nodes"),
        py::arg("embedding_GB"),
        py::arg("layer_GB"),
        py::arg("num_layers"),
        py::arg("num_kv_heads"),
        py::arg("head_dim"),
        py::arg("hidden_size"),
        py::arg("tp_size"),
        py::arg("pp_size"),
        py::arg("input_length"),
        py::arg("cpu_tw_intercept"),
        py::arg("cpu_tw_coef_tokens"),
        py::arg("gpu_tw_intercept"),
        py::arg("gpu_tw_coef_tokens"),
        py::arg("cpu_lw_intercept"),
        py::arg("cpu_lw_coef_tokens"),
        py::arg("gpu_lw_intercept"),
        py::arg("gpu_lw_coef_tokens"),
        py::arg("cpu_decode_intercept"),
        py::arg("cpu_decode_coef_kv"),
        py::arg("debug") = false,
        py::arg("transfer_setup_ms") = 10.0,
        py::arg("weight_loading_multiplier") = 1.0,
        py::arg("max_cpu_tokens") = -1,
        py::arg("allow_cpu_decode") = true
    );
    m.def(
        "solve_static_lazy_load_cpp",
        &solve_static_lazy_load_cpp,
        py::arg("fixed_ms"),
        py::arg("pcie_GBps"),
        py::arg("rdma_GBps"),
        py::arg("numa_nodes"),
        py::arg("embedding_GB"),
        py::arg("layer_GB"),
        py::arg("num_layers"),
        py::arg("num_kv_heads"),
        py::arg("head_dim"),
        py::arg("hidden_size"),
        py::arg("tp_size"),
        py::arg("pp_size"),
        py::arg("input_length"),
        py::arg("cpu_tw_intercept"),
        py::arg("cpu_tw_coef_tokens"),
        py::arg("gpu_tw_intercept"),
        py::arg("gpu_tw_coef_tokens"),
        py::arg("cpu_lw_intercept"),
        py::arg("cpu_lw_coef_tokens"),
        py::arg("gpu_lw_intercept"),
        py::arg("gpu_lw_coef_tokens"),
        py::arg("cpu_decode_intercept"),
        py::arg("cpu_decode_coef_kv"),
        py::arg("debug") = false,
        py::arg("transfer_setup_ms") = 10.0,
        py::arg("weight_loading_multiplier") = 1.0
    );
}
