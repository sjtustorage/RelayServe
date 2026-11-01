# FA3 warmup host waits on H20

Use `VLLM_PROFILE_LAZY_HOST=/absolute/directory` to capture one cProfile
file per model warmup and rank. The log records wall time and calling-thread
CPU time. This diagnostic adds no CUDA event or synchronization and preserves
the SHM module-loading throttle, including on exceptions. Leave it unset for
latency benchmarks. CUDA tracing/completion diagnostics take precedence.

On five independent Qwen3-32B GPU TP=2 pools, with CPU TP=2 AMX executing
13 layers concurrently and 128 prompt tokens, first warmup profiles spent
418.5–495.8 ms in the two FA3 `get_scheduler_metadata` calls on nine of ten
ranks. Reload calls spent approximately 0.1–0.2 ms combined. Outputs matched
direct original-safetensors inference. These profiled runs are diagnostics,
not latency baselines.

Standalone calls with the same 32B attention dimensions took approximately
0.4 ms in a fresh process after generic runtime prewarm, without concurrent
DMA. Native FA3 scheduler code allocates a small tensor and submits a kernel;
it does not explicitly synchronize. The host profiles cannot establish whether
the overlapped delay comes from allocation, module loading, launch contention,
or GPU work. CUDA API timelines are needed before changing the copy throttle.

An experiment prewarming FA3 scheduler calls in the generic resident runtime
did not improve first lazy initialization and was reverted. Keep FA3 selected;
FA2 did not demonstrate an initialization improvement in the 128-token comparison.
One FA2 first warmup took 74 s; its cause is unresolved and is not evidence of
a general FA2 cost. Both attention versions produced exact output token matches
against their original-safetensors reference.

Validation: `tests/v1/worker/test_relay_weight_submission.py` passes 15 tests,
including successful and failing host-profile runs, valid pstats output, throttle
cleanup, and absence of an extra device synchronization.

## CUDA trace and rejected layer throttle

Two additional real 32B/TP=2/AMX13 pools with CUDA profiling passed exact
original-safetensors output checks. Their FA3 scheduler calls took 0.3–0.5 ms,
but profiler setup itself cost 364–485 ms in the first pool. The tracing run
therefore did not reproduce the unprofiled host delay and cannot identify its
cause. One rank's custom all-reduce kernel waited 123 ms for its peer; another
pool showed a 66 ms wait. These are diagnostic observations, not benchmarks.
The untraced host profiles recorded approximately 518 ms calling-thread CPU
time within 526 ms wall time, motivating investigation of native first-use
initialization rather than assuming PCIe wait time.

Moving the warmup copy fence from each fragment to each layer preserved exact
outputs in three fresh pools, but first lazy times were 741.7/721.0/813.9 ms
versus the existing fragment baseline 744.7/717.9/805.4 ms. First creation-to-token
times were 1340.6/954.8/1112.8 ms. No demonstrated first-model benefit: the
experiment was reverted, preserving fragment synchronization during KV/module
initialization. Native CPU TP=2 to GPU TP=2 and TP=4 shard checks also passed
bit-exact BF16 copy/unpack, reset, and reattachment during the experiment.

The remaining native cleanup borrows immutable task vectors and individual
tasks instead of copying their shape vectors and strings on every submission.
It removes unused local CPU-affinity bookkeeping. This leaves CUDA operations,
event dependencies, copy throttling, and readiness publication unchanged. No
claim of resolved lazy instability or measured end-to-end gain follows from
this allocation cleanup.

The final extension rebuilt successfully. With fragment throttling restored,
`test_direct_shm_h2d.py --async-load --cpu-tp 2` passed both GPU TP=2/rank1
and GPU TP=4/rank3, including initialization throttling, exact tensor comparisons,
pending-load teardown, surviving destinations, and reattachment.

## Native runtime API attribution

A temporary LD_PRELOAD shim measured CUDA runtime call wall time and calling
thread CPU time, without adding CUDA events or synchronization. The two-pool
32B/TP=2/AMX13 host-profile diagnostic reproduced the scheduler delay inside
`cudaLaunchKernel`: 440/465 ms in one pool and 349/383 ms in the other, with
347–462 ms calling-thread CPU time. Exact original-safetensors output checks
passed. These remain diagnostic runs; their latency is not a benchmark.
This locates the delay at submission, but does not distinguish driver lock
contention from module loading or other runtime first-use work.

With dynamic FA3 scheduling (`num_splits=0`) followed by an actual FA3 forward
added to generic preparation, three fresh pools passed exact output checks but
did not establish a creation-to-token benefit: first lazy 737.9/712.5/700.4 ms;
creation-to-token 987.0/949.4/1411.7 ms. The experiment was reverted. In one pool,
architecture construction on one rank grew to 364 ms while the peer took
108 ms; reducing one warmup substage is not sufficient evidence of a win.

A separate experiment replacing each fragment's stream wait with a reusable
blocking event also passed exact output checks. First lazy was
716.1/744.2/823.3 ms and creation-to-token 953.5/968.2/1191.6 ms. Reload lazy
included a 627.8 ms outlier. This did not establish an improvement and was
reverted. Fragment completion waits remain in place.

## Initial bounded staging submission experiment

During structure construction the original code submits every tensor's copy
and unpack work, with GPU stream waits protecting staging-buffer reuse. This
can queue most of the model before KV/module throttling begins. An experiment
also waits on the buffer's previous unpack completion on the host, only during
structure construction, bounding submissions to the existing two buffers.
Blocking, non-timing buffer events allow the submitting thread to sleep. The
GPU stream dependencies and later fragment throttle are preserved.

Three fresh 32B/TP=2/AMX13/128-token pools produced first lazy times
274.9/246.2/233.9 ms and creation-to-token 854.9/845.4/851.6 ms. Reload lazy
was 167.0/178.7/164.1 ms; reload creation-to-token 793.7/786.5/773.3 ms.
Every cold and warm output matched original-safetensors inference. Native
CPU TP=2 to GPU TP=4/rank3 packing/copy/reset checks and the CUDA readiness
invalidation test also passed. First dummy warmup in pool0 fell to 82 ms.

The 4096-token tokenwise comparison (CPU prefix128, TP=2) was inconclusive:

| Configuration | First lazy ms | First creation-to-token ms | Reload creation-to-token ms |
| --- | --- | --- | --- |
| Bounded submission | 909.7 / 355.7 | 2737.3 / 2142.3 | 2405.4 / 2167.0 |
| Original, adjacent control | 742.4 / 765.1 | 2418.1 / 2217.4 | 2343.4 / 2098.9 |

Both configurations passed exact original-safetensors output comparisons;
warm full-GPU prefill was approximately 1076–1081 ms in both. An improvement
in one layerwise configuration is not sufficient to change all transfer paths.
The original native source and binary were restored for adjacent layerwise
control measurements. At this stage the candidate was not retained; the
additional controls and final production policy are recorded below.

Artifacts: `/tmp/resident-first-init32-tp2-fa3-{nativewait,forwardprepared,
blocking-throttle,bounded-submit,bounded-token4096,control-token4096}/`.
The candidate diff and extension are preserved at
`/tmp/shm-bounded-submit.patch` and `/tmp/shm-bounded-submit-extension.so`.

## Adjacent controls and retained layerwise policy

All runs use H20, FA3, V1 eager SHM lazy loading, the real Qwen3-32B
checkpoint, CPU TP=2, and chunked prefill=4096. CPU weights and GPU contexts
and communication groups are prepared before the model-creation timer.
AMX computation, weight transfer/depacking, and GPU initialization overlap.
Every pool creates and releases two model instances and checks both cold and
warm token output against an independently loaded original-safetensors model.
No CUDA or host diagnostic profiler is enabled in these latency runs.

The following numbers are medians in milliseconds. Each layerwise run uses
13 CPU layers. TP=2/128 and TP=4/128 have three fresh pools per implementation;
TP=2/4096 has two. The bounded experiment uses the same transfer pacing that
the final implementation enables for layerwise loading.

| Workload | Original first lazy | Bounded first lazy | Original creation-to-first-token | Bounded creation-to-first-token | Original reload creation-to-first-token | Bounded reload creation-to-first-token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Layerwise TP=2, 128 tokens | 714.5 | 246.2 | 929.3 | 851.6 | 788.8 | 786.5 |
| Layerwise TP=2, 4096 tokens | 846.4 | 368.2 | 3366.3 | 3303.9 | 3191.0 | 3174.6 |
| Layerwise TP=4, 128 tokens | 515.5 | 299.6 | 799.0 | 591.7 | 546.4 | 528.9 |

For TP=2/4096 the original first dummy-model warmup costs 514–585 ms per
rank, versus 111–132 ms with bounded submission. Full warmup, sampler warmup,
and the compute-stream completion fence are still executed for every model.
Long layerwise requests remain CPU-bound: first creation-to-token spans
3315–3418 ms in the control and 3115–3493 ms in the candidate. These small
samples establish a lazy/warmup improvement, not a stable long-request TTFT
gain or a tail-latency guarantee across arbitrary models and workloads.

A new three-pool adjacent tokenwise comparison uses a CPU prefix of 128 and
a 4096-token prompt. The first-lazy median drops from 775.3 to 339.7 ms, but
the creation-to-first-token median is essentially unchanged (2103.5 versus
2102.2 ms), and reload creation-to-token rises from 2187.1 to 2234.4 ms.
The CPU prefix dominates readiness in this configuration, and its execution
time also varies. Combined with the earlier inconclusive tokenwise experiment,
there is insufficient end-to-end evidence to retain tokenwise pacing.

The retained implementation therefore enables the bound only for asynchronous
layerwise structure initialization. The shared C++ copy/depack implementation
uses blocking, non-timing staging events and a host wait before reuse when the
bound is active. The original stream dependencies, per-fragment KV/module
throttle, and readiness publication remain. Turning off architecture
initialization also turns off the bound. Existing single-argument native callers
keep unbounded architecture submission by default; the SHM loader selects
layerwise pacing internally, with no additional user-facing configuration.

The final extension rebuilt successfully. Native CPU TP=2 to GPU TP=2/rank1
and TP=4/rank3 both passed exact BF16 copy/unpack, initialization throttling,
pending-load teardown, destination survival, and SHM reattachment checks with
the bound explicitly active. The CUDA readiness and worker submission/warmup
suite passed all 16 tests.

Two additional fresh pools per workload validated the final policy against
independent original-safetensors inference:

| Final workload | First lazy ms | First creation-to-token ms | Reload creation-to-token ms |
| --- | --- | --- | --- |
| Layerwise TP=4, 128 tokens | 269.5 / 296.3 | 559.0 / 644.1 | 509.7 / 513.8 |
| Tokenwise TP=2, 4096 tokens, CPU prefix128 | 404.2 / 429.2 | 1989.7 / 2049.3 | 2130.7 / 2030.6 |

The final tokenwise path has no host-side architecture submission bound and
shares the non-timing staging events. These final measurements are better than
the earlier original controls, but they are not an isolated event-flags A/B
experiment, and CPU-prefix latency differs between runs. Do not attribute the
entire observed gain to event flags or infer a tail-latency guarantee from two
pools. Real final-policy samples and build metadata are recorded in the
accompanying JSON artifact.

Artifact directories add `control-layer128`, `control-layer4096`,
`bounded-layer4096`, `control-token4096-repeat`, `bounded-token4096-repeat`,
and `final-token4096`
under `/tmp/resident-first-init32-tp2-fa3-`, plus `bounded-layer128`,
`control-layer128`, and `final-layer128` under
`/tmp/resident-first-init32-tp4-fa3-`. Per-sample times and metadata are retained
in `fa3_bounded_submission_results.json`; newer benchmarks record source and
extension SHA-256 hashes because a git revision alone cannot identify an
experimental native build. No real 72B checkpoint was available for this test.
