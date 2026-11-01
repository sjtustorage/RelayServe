# ---------------------------------------------------------------------------- #
#  Bundled default when no profile file is set.                                #
#  Override: set env SLLM_LOADING_PERF_PROFILE to a .json or .py file whose    #
#  root / LOADING_PERF_PROFILE is the full dict (no merge with this file).     #
# ---------------------------------------------------------------------------- #

LOADING_PERF_PROFILE = {
    "hardware": {
        # Effective CPU host -> GPU device memory bandwidth (GB/s) for weight DMA.
        "pcie_cpu_to_gpu_GBps": 40,
        # Typical NIC / IB line rate between nodes when weights move over RDMA.
        "rdma_node_to_node_GBps": 25,
        "numa_nodes": 2
    },
    # Per logical model id (HuggingFace name or short id). Missing keys fall back to "default".
    "models": {
        "Qwen3-8B": {
            "engine_init_ms": 900,
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
                "intercept": 350,
                "coef_tokens": 0.58,
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
                "intercept": 85,
                "coef_kv_tokens": 0,
            },
            # CPU time for layerwise path: ms = intercept + coef_tokens*T + coef_layers*L + coef_tokens_layers*T*L
            "cpu_layerwise_ms": {
                "intercept": 9.7,
                "coef_tokens": 0.016,
            },
            # Optional: GPU prefill term for ranking (same shape as cpu_layerwise_ms).
            "gpu_layerwise_ms": {
                "intercept": 0.4,
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
                "intercept": 370,
                "coef_tokens": 1.3,
            },
            "gpu_tokenwise_ms": {
                "intercept": -7,
                "coef_tokens": 0.28,
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
                "intercept": 7.5,
                "coef_tokens": 0.04,
            },
            # Optional: GPU prefill term for ranking (same shape as cpu_layerwise_ms).
            "gpu_layerwise_ms": {
                "intercept": -0.1,
                "coef_tokens": 0.0043,
            },
        },
        "Qwen3-32B": {
            "engine_init_ms": 1100,
            "layer_GB": 0.975,
            "embedding_GB": 1.6,
            "num_layers": 64,
            "num_kv_heads": 8,
            "head_dim": 128,
            "hidden_size": 5120,
            "tp_size": 2,
            "pp_size": 1,
            # CPU prefill time for tokenwise path: ms = intercept + coef_tokens * T
            "cpu_tokenwise_ms": {
                "intercept": 2300,
                "coef_tokens": 1.6,
            },
            "gpu_tokenwise_ms": {
                "intercept": 17,
                "coef_tokens": 0.27,
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
                "intercept": 35,
                "coef_tokens": 0.025,
            },
            # Optional: GPU prefill term for ranking (same shape as cpu_layerwise_ms).
            "gpu_layerwise_ms": {
                "intercept": 0.26,
                "coef_tokens": 0.0042,
            },
        },
        "Qwen2.5-72B-tp": {
            "engine_init_ms": 000,
            "layer_GB": 1.75,
            "embedding_GB": 2.7,
            "num_layers": 80,
            "num_kv_heads": 8,
            "head_dim": 128,
            "hidden_size": 8192,
            "tp_size": 4,
            "pp_size": 1,
            "cpu_tokenwise_ms": {
                "intercept": 2000,
                "coef_tokens": 5.85,
            },
            "gpu_tokenwise_ms": {
                "intercept": 17,
                "coef_tokens": 0.28,
            },
            # Optional: CPU decode during engine prep when prefill < engine_init.
            # decode_step_ms = intercept + coef_kv_tokens * cpu_prefill_len;
            # n_decode_in_window = floor((engine_init - prefill) / decode_step_ms).
            # Search requires first weight xfer <= engine_init when the above applies.
            "cpu_decode_ms": {
                "intercept": 629,
                "coef_kv_tokens": 0,
            },
            # CPU time for layerwise path: ms = intercept + coef_tokens*T + coef_layers*L + coef_tokens_layers*T*L
            "cpu_layerwise_ms": {
                "intercept": 31.25,
                "coef_tokens": 0.09,
            },
            # Optional: GPU prefill term for ranking (same shape as cpu_layerwise_ms).
            "gpu_layerwise_ms": {
                "intercept": 0.2125,
                "coef_tokens": 0.0035,
            },
        },
        "Qwen2.5-72B": {
            "engine_init_ms": 900,
            "layer_GB": 1.75,
            "embedding_GB": 2.7,
            "num_layers": 80,
            "num_kv_heads": 8,
            "head_dim": 128,
            "hidden_size": 8192,
            "tp_size": 1,
            "pp_size": 4,
            "cpu_tokenwise_ms": {
                "intercept": 2000,
                "coef_tokens": 5.85,
            },
            "gpu_tokenwise_ms": {
                "intercept": 100,
                "coef_tokens": 1.04,
            },
            # Optional: CPU decode during engine prep when prefill < engine_init.
            # decode_step_ms = intercept + coef_kv_tokens * cpu_prefill_len;
            # n_decode_in_window = floor((engine_init - prefill) / decode_step_ms).
            # Search requires first weight xfer <= engine_init when the above applies.
            "cpu_decode_ms": {
                "intercept": 629,
                "coef_kv_tokens": 0,
            },
            # CPU time for layerwise path: ms = intercept + coef_tokens*T + coef_layers*L + coef_tokens_layers*T*L
            "cpu_layerwise_ms": {
                "intercept": 31.25,
                "coef_tokens": 0.09,
            },
            # Optional: GPU prefill term for ranking (same shape as cpu_layerwise_ms).
            "gpu_layerwise_ms": {
                "intercept": 1.25,
                "coef_tokens": 0.013,
            },
        }
    },
}
