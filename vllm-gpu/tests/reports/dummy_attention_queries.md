# Attention query coverage during lazy warmup

`GPUModelRunner._dummy_run` constructed attention metadata using
`query_start_loc`, but never populated that buffer for the dummy batch. Newly
allocated buffers contain zeros, and reused buffers can retain a prior request's
offsets. Traversing all transformer layers therefore did not prove attention
processed the intended queries. This also affected CUDA graph attention capture.

The runner now writes cumulative scheduled-token offsets before building
attention metadata and copies them through the existing CPU/GPU buffer. No
stream fence is added. The prefill warmup uses one more token than its request
count, bounded by `max_num_batched_tokens`, so at least one multi-token query is
present when the token budget permits it. With the resident benchmark's two
requests, prefill offsets are `[0, 1, 3]`; uniform decode offsets are `[0, 1, 2]`.
Both traverse all model layers. CUDA module copy throttling and the final
compute-stream completion fence remain active.

Six regression cases exercise the real `_dummy_run` metadata construction with
fresh-zero and stale buffers, decode and uneven prefill batches. They reject
empty or stale query ranges. Before the fix, all three original stale-buffer
cases failed at the attention metadata builder. Afterward, these six cases and
the thirteen relay submission/warmup control tests pass (19 total). The CPU
repository receives the same buffer fix and the six cases pass there too.
CPU pytest needs `--confcutdir=tests/v1/worker` because its unrelated root
conftest imports a nonexistent `vllm.config.model` module.

Real Qwen3-14B validation uses CPU TP=2, GPU TP=2 (devices 0,4), tokenwise CPU
prefix 128, actual prompt 4096, chunk 4096, eager V1 SHM lazy loading. Two fresh
GPU pools each reload the model once. All cold/warm eight-token outputs match
independently loaded original safetensors; every handoff imports 128 KV tokens.
Artifacts: `/tmp/resident-first-init14b-tp2-4096-realqueries/summary.json`.

| Trial | Phase | Lazy ms | Creation-to-first-token ms | GPU generation-to-first-token ms |
|---|---|---:|---:|---:|
| 0 | First | 477.9 | 1450.8 | 465.9 |
| 0 | Reload | 202.8 | 1011.1 | 562.9 |
| 1 | First | 358.0 | 999.3 | 457.7 |
| 1 | Reload | 142.6 | 943.2 | 473.8 |

The slower first trial includes 504.1 ms model creation before lazy warmup,
versus 180.7 ms in trial 1; its excess cannot be attributed entirely to this
warmup change. The previous version's identical tokenwise workload had first
lazy 456.1/390.3 ms and creation-to-token 1089.2/1092.6 ms. These small, sequential
shared-host samples validate correctness, but do not establish a speedup or
exclude performance regressions. A full 4096-token dummy traversal is not used.

Qwen3-32B also completed three fresh TP=2 pools with real concurrent CPU TP=2
AMX work (128-token prompt, first 13 CPU layers), each with a reload, immediate
warm request and independent original-safetensors check. All outputs match.
The baseline and fixed-query artifacts are respectively
`/tmp/resident-first-init32-tp2-128-amx13/summary.json` and
`/tmp/resident-first-init32-tp2-128-realqueries/summary.json`.

| Trial | First lazy before/after ms | Reload lazy before/after ms | First creation-to-token before/after ms |
|---|---:|---:|---:|
| 0 | 847.2 / 744.7 | 186.2 / 178.0 | 1047.7 / 945.8 |
| 1 | 726.6 / 717.9 | 186.5 / 194.4 | 952.8 / 929.8 |
| 2 | 735.5 / 805.4 | 209.2 / 170.7 | 954.4 / 1166.7 |

These are sequential trials rather than simultaneous matched hardware states.
The corrected warmup's first lazy range remains 717.9–805.4 ms, and one complete
TTFT sample reaches 1166.7 ms. Query coverage is repaired, but lazy variability
has not been resolved and this must not be reported as a stabilization fix.

## Rejected single-core isolation experiment

Five fresh pools ran the same 32B/128-token/13-layer workload with torchrun
restricted to CPUs 143,191. Existing GPU NUMA binding then restricts each GPU
rank to one local CPU. `/proc` observations confirm the actual rank affinities
143 and 191; CPU AMX retains its usual allocation. Artifact:
`/tmp/resident-first-init32-tp2-128-dedicated/summary.json`.
All cold/warm/original outputs match. First lazy values are
828.9,819.5,798.4,799.5,836.9 ms; reload values are
327.8,285.3,290.6,287.3,254.4 ms. This restricts CPU resources too aggressively:
reload is slower than the normal NUMA-bound 170.7–194.4 ms samples above.
No single-core restriction is added to production.
