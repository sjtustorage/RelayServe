# RoPE first-use import in SHM model construction

The controlled Qwen3-32B TP=2 experiment is documented in sllm's
`tests/resident_first_init_results.md`. Its slowest first-model samples spent
197–218 ms in model architecture construction versus approximately 99–109 ms
in typical samples.

A separate three-pool diagnostic run enabled `VLLM_PROFILE_SHM_ARCH` and saved
12 cProfile files under `/tmp/resident-first-init32-arch-profiles/` (first and
reload models, two ranks). All outputs matched original safetensors. One first
architecture profile took 478 ms: `torch.functional.einsum` had 153 ms self time
and importlib `_fill_cache` 152 ms self time. The call path includes
`RotaryEmbedding._compute_cos_sin_cache` and lazy import of
`torch.backends.opt_einsum`. This establishes an avoidable first-use import path,
but does not prove every earlier architecture outlier had this cause. Profiled
latencies are diagnostic and are not performance samples.

Both CPU and GPU engines now use `torch.outer(t, inv_freq)` for this two-vector
outer product, removing einsum dispatch and its backend import from this path.
No weights-loading or warmup synchronization changes are involved.

An eight-pair fresh-process microbenchmark in the GPU environment measured the
complete 40960-position, 128-dimensional BF16 RoPE cache construction, with one
Torch CPU thread. Alternating pair order gave median 32.263 ms for einsum and
31.698 ms for outer. The backend module was imported in all einsum children and
none of the outer children. This is a small median difference, not a large
general speedup claim.

`tests/verify_rotary_cache_initialization.py` checks the actual engine method
against the old expression at (positions, dimension) = (40960,128), (4096,64),
(131072,128). Full FP32/BF16/FP16 caches are bit-exact. It also forbids einsum
dispatch in the new method. The CPU-built checks pass in both engine environments;
the CUDA-built checks also pass in the GPU environment.

The unprofiled five-pool post-change run at
`/tmp/resident-first-init32-tp2-outer/summary.json` used the same workload and GPUs
as the earlier baseline. All first/reload/warm outputs matched independent
original-safetensors inference, with persistent PIDs within each trial.

| Metric | Before | After |
|---|---:|---:|
| First lazy median | 755.1 ms | 723.4 ms |
| First lazy range | 703.0–831.8 ms | 708.7–756.6 ms |
| First creation-to-token median | 952.3 ms | 922.0 ms |
| First creation-to-token range | 915.2–1026.5 ms | 890.1–1354.3 ms |
| Reload lazy median | 172.2 ms | 183.4 ms |
| Reload lazy range | 170.7–289.7 ms | 160.5–220.4 ms |

These are separate small sequential samples on a shared host, not randomized
end-to-end paired trials or proof of a latency guarantee. First architecture
construction is 91.1–140.3 ms after the change. One after-change first request
still spends 459.5 ms in generation before its first output, versus 50–56 ms in
the other first samples; its full creation-to-token time is 1354.3 ms despite
712.1 ms lazy initialization. The source of that remaining delay needs further
investigation. Weight DMA can extend beyond lazy warmup readiness, and host stage
logs do not establish whether this delay is weight readiness or another cause.

To collect Python architecture profiles, set `VLLM_PROFILE_SHM_ARCH` to an output
directory before starting GPU ranks. Each construction writes a separate PID
and UUID-named pstats file. Profiling is disabled by default and adds no CUDA
events or stream synchronization. The controlled benchmark marks profiled runs
as invalid for latency measurement in metadata.
