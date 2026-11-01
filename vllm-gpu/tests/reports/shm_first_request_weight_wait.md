# SHM first-request host weight waits

`VLLM_PROFILE_WEIGHT_WAIT=1` enables host wall-time accumulation around existing
individual weight fences. It creates no CUDA events, performs no additional
readiness queries and adds no stream synchronization. The runner logs model
submission (including host fences) and post-forward-to-token time for the first
two iterations of each SHM model. The latter includes logits, sampling and the
existing token transfer synchronization; it is not pure GPU execution time.

The five-pool Qwen3-32B diagnostic run is saved at
`/tmp/resident-first-init32-tp2-weight-wait/summary.json`. GPU TP=2, devices 0,4,
128 tokens, CPU TP=2 resident weights with no CPU inference, chunk 4096, eager V1,
SHM lazy loading. All first/reload/warm outputs match independently loaded
original safetensors. This diagnostic run is separate from latency baselines.

| Trial | First-model max-rank host weight wait ms | Reload max-rank host weight wait ms | Reload max-rank model submission ms |
|---|---:|---:|---:|
| 0 | 1.031 | 432.964 | 458.691 |
| 1 | 0.994 | 423.318 | 447.638 |
| 2 | 1.091 | 431.836 | 457.555 |
| 3 | 0.917 | 468.338 | 495.460 |
| 4 | 1.088 | 459.709 | 488.926 |

The first-model warmup window hides weight readiness in these five trials. The
shorter reload initialization leaves substantial weights still outstanding at
the first forward: 423–468 ms of host readiness fences, compared with 448–495 ms
model submission. Readiness fences stay per-layer, allowing already queued GPU
work to overlap host waits. Thus host wait is not an additive pure idle metric.

The previously observed 459.5 ms first-model generation-to-output delay was not
reproduced here. Reload waits establish the cause for the reload samples only;
they do not retrospectively prove the cause of that first-model outlier.
Next investigation should measure the native transfer timeline and copy/depack
throughput while preserving overlap with structure construction and warmup.

Validation: all ten shared-memory-loader unit tests pass, including host timer
accumulation over two individual fences without CUDA timing events or additional
readiness checks. Runtime logs include both TP ranks and both first/reload models.

## Qwen3-14B 4096-token diagnostic

Two fresh TP=2 pools with CPU inference disabled completed at
`/tmp/resident-first-init14b-tp2-4096-wait/summary.json`; all cold/warm/reference
outputs match original safetensors. This host-timer diagnostic is not a latency
baseline. First-generation-to-token times are 501.6/480.8 ms versus immediately
warm 461.5/461.3 ms: the earlier 143–156 ms excess was not reproduced.

| Trial | Phase | Max-rank host weight wait ms | Max-rank model submission ms | Max-rank post-forward-to-tokens ms |
|---|---|---:|---:|---:|
| 0 | First | 0.819 | 58.276 | 441.247 |
| 0 | Reload | 0.699 | 220.736 | 441.210 |
| 1 | First | 0.719 | 44.617 | 437.746 |
| 1 | Reload | 182.665 | 226.199 | 263.999 |

First-model samples show little host weight readiness wait. Reload trial 0
instead spends 220.7 ms submitting model work without a significant weight
fence, while reload trial 1 spends 182.7 ms in fences. Post-forward time includes
completion of previously queued forward kernels and does not isolate sampling
or CUDA module loading. These observations do not establish the cause of the
earlier excess or identify the slow submission path in reload trial 0.
