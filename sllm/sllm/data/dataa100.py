# ---------------------------------------------------------------------------- #
#  Bundled default when no profile file is set.                                #
#  Override: set env SLLM_LOADING_PERF_PROFILE to a .json or .py file whose    #
#  root / LOADING_PERF_PROFILE is the full dict (no merge with this file).     #
# ---------------------------------------------------------------------------- #

LOADING_PERF_PROFILE = {
    "hardware": {
        # Effective CPU host -> GPU device memory bandwidth (GB/s) for weight DMA.
        "pcie_cpu_to_gpu_GBps": 25,
        # Typical NIC / IB line rate between nodes when weights move over RDMA.
        "rdma_node_to_node_GBps": 25,
        "numa_nodes": 2
    },
    # Per logical model id (HuggingFace name or short id). Missing keys fall back to "default".
    "models": {
        "Qwen3-8B": {
            "engine_init_ms": 1000,
            "layer_GB": 0.3861111,
            "embedding_GB": 1.25,
            "num_layers": 36,
            "num_kv_heads": 8,
            "head_dim": 128,
            "hidden_size": 4096,
            "tp_size": 1,
            "pp_size": 1,
            # CPU prefill time for tokenwise path: ms = intercept + coef_tokens * T
            "cpu_tokenwise_ms": {
                "intercept": 140,
                "coef_tokens": 0.51,
            },
            "gpu_tokenwise_ms": {
                "intercept": 17,
                "coef_tokens": 0.075,
            },
            # Optional: CPU decode during engine prep when prefill < engine_init.
            # decode_step_ms = intercept + coef_kv_tokens * cpu_prefill_len;
            # n_decode_in_window = floor((engine_init - prefill) / decode_step_ms).
            # Search requires first weight xfer <= engine_init when the above applies.
            "cpu_decode_ms": {
                "intercept": 78,
                "coef_kv_tokens": 0,
            },
            # CPU time for layerwise path: ms = intercept + coef_tokens*T + coef_layers*L + coef_tokens_layers*T*L
            "cpu_layerwise_ms": {
                "intercept": 9,
                "coef_tokens": 0.01,
            },
            # Optional: GPU prefill term for ranking (same shape as cpu_layerwise_ms).
            "gpu_layerwise_ms": {
                "intercept": 1,
                "coef_tokens": 0.002,
            },
        },
        "Qwen3-14B": {
            "engine_init_ms": 1000,
            "layer_GB": 0.66,
            "embedding_GB": 1.6,
            "num_layers": 40,
            "num_kv_heads": 8,
            "head_dim": 128,
            "hidden_size": 5120,
            "tp_size": 1,
            "pp_size": 1,
            # CPU prefill time for tokenwise path: ms = intercept + coef_tokens * T
            "cpu_tokenwise_ms": {
                "intercept": 258.02,
                "coef_tokens": 0.613,
            },
            "gpu_tokenwise_ms": {
                "intercept": 12.34,
                "coef_tokens": 0.15,
            },
            # Optional: CPU decode during engine prep when prefill < engine_init.
            # decode_step_ms = intercept + coef_kv_tokens * cpu_prefill_len;
            # n_decode_in_window = floor((engine_init - prefill) / decode_step_ms).
            # Search requires first weight xfer <= engine_init when the above applies.
            "cpu_decode_ms": {
                "intercept": 133,
                "coef_kv_tokens": 0,
            },
            # CPU time for layerwise path: ms = intercept + coef_tokens*T + coef_layers*L + coef_tokens_layers*T*L
            "cpu_layerwise_ms": {
                "intercept": 7.2,
                "coef_tokens": 0.04,
            },
            # Optional: GPU prefill term for ranking (same shape as cpu_layerwise_ms).
            "gpu_layerwise_ms": {
                "intercept": 0.31,
                "coef_tokens": 0.00375,
            },
        },
        "Qwen3-32B": {
            "engine_init_ms": 1100,
            "layer_GB": 0.975,
            "embedding_GB": 1.6,
            "num_layers": 64,
            "num_kv_heads": 8,
            "head_dim": 80,
            "hidden_size": 5120,
            "tp_size": 2,
            "pp_size": 1,
            # CPU prefill time for tokenwise path: ms = intercept + coef_tokens * T
            "cpu_tokenwise_ms": {
                "intercept": 160,
                "coef_tokens": 2.139,
            },
            "gpu_tokenwise_ms": {
                "intercept": 20.8,
                "coef_tokens": 0.191,
            },
            # Optional: CPU decode during engine prep when prefill < engine_init.
            # decode_step_ms = intercept + coef_kv_tokens * cpu_prefill_len;
            # n_decode_in_window = floor((engine_init - prefill) / decode_step_ms).
            # Search requires first weight xfer <= engine_init when the above applies.
            "cpu_decode_ms": {
                "intercept": 293,
                "coef_kv_tokens": 0,
            },
            # CPU time for layerwise path: ms = intercept + coef_tokens*T + coef_layers*L + coef_tokens_layers*T*L
            "cpu_layerwise_ms": {
                "intercept": 4.4,
                "coef_tokens": 0.04,
            },
            # Optional: GPU prefill term for ranking (same shape as cpu_layerwise_ms).
            "gpu_layerwise_ms": {
                "intercept": 0.325,
                "coef_tokens": 0.003,
            },
        },
    },
}
