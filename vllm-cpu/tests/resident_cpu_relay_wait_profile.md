# Real CPU relay compute and feedback timing

`VLLM_PROFILE_RELAY=1` enables CPU engine step timings for scheduling,
executor execution, hidden publication, scheduler metadata publication, GPU
feedback receive, and scheduler cleanup. Worker timings record model execution
and metadata build/SHM-slot wait/write. This adds no barriers, CUDA waits, or
speculative CPU execution. Disabled profiling adds no timing calls in these
new measurement points; the existing model execution timer remains.

Real Qwen3-32B bf16/V1, CPU TP2 -> GPU TP4 (GPUs0/1/4/5), eight CPU layers,
prompt4097/chunk4096, pre-registered sources:

| CPU measurement | First cold | After idle expiry |
| --- | --- | --- |
| Model execution across ranks | 1.5368–1.5373 s | 1.3250–1.3275 s |
| Executor including input preparation/IPC | 1.5391 s | 1.3302 s |
| Hidden publication RPC | 3.313 ms | 2.605 ms |
| Metadata publication RPC | 1.289 ms | 1.138 ms |
| Worker SHM empty-slot wait | 17 us | 5 us |
| CPU waiting for GPU feedback | 0.5306 s | 0.5214 s |
| Producer total (including GPU wait) | 2.0803 s | 1.8587 s |

The GPU's re-cold metadata wait was 0.8268–0.8271 s, necessary suffix weight
wait 0.224–0.237 ms, and executor execution about0.510 s. CPU model computation
starts while GPU create/lazy proceeds. The residual CPU computation after GPU
readiness explains the metadata wait; SHM-slot congestion is not the cause.
Producer total is not pure AMX compute: it includes CPU attention/normalization
and the subsequent GPU feedback wait. CPU model timing is also whole model
prefix execution, not an isolated GEMM measurement.

Create/lazy were0.353/0.587 s first cold and0.216/0.294 s re-cold. Complete
requests were2.256/2.036 s, warm first output0.597 s. Cold, warm, expiry/reload,
and independent safetensors outputs matched all eight tokens. GPU PIDs stayed
stable. The process exited zero; no performance speedup is claimed from these
single diagnostic samples.

37 focused CPU relay tests passed with `--confcutdir=tests/v1`. The repository's
root conftest imports `vllm.config.model`, absent in this checkout, so the normal
root-conftest invocation failed before collecting tests. No runtime failure was
hidden by this test configuration.

Artifacts: `/tmp/resident-cpu-wait-profile32-tp4-artifacts/`; log:
`/tmp/resident-cpu-wait-profile32-tp4.log`.

## Single publication RPC

The layerwise handoff now exposes one `publish_pending_relay_step` executor
RPC. It writes CPU hidden states first and then publishes the scheduler
metadata semaphore; the previous two RPC calls were removed. This preserves
the data dependency while eliminating one collective control round trip.

Focused protocol, publication-order, and failure-preservation tests passed
(39 total). A real Qwen3-32B TP2→TP4 run produced exactly the same eight tokens
for cold, warm, expiry/reload, and independent safetensors reference paths.
The combined state publish measured 3.55–4.22 ms; these samples are not a
controlled before/after benchmark because CPU scheduling noise varied between
runs. The run exited successfully with stable GPU PIDs.

## CPU operator breakdown

`VLLM_CPU_PROFILE_RELAY_OPERATORS=1` installs temporary module timing hooks
only on the first nonempty layerwise iteration. Hooks are removed on success
and failure. Normal execution installs no hooks. Projection measurements
include their TP reductions; these are not isolated AMX kernel costs.

Real Qwen3-32B CPU TP2→GPU TP4 prompt4097/chunk4096, eight CPU layers,
after idle expiry: executor execution1.482 s, projection totals1.23–1.25 s
(about83–84%), attention139–152 ms, normalization25–34 ms across ranks.
This prioritizes GEMM/TP investigation rather than normalization optimization.
The ranks already bind47 physical cores per NUMA domain; simply increasing
OMP_NUM_THREADS does not explain or resolve the observed cost.

42 focused tests passed, including profile-disabled execution and hook cleanup
after model failure. Cold/warm/re-cold and independent safetensors outputs
matched all eight tokens, with stable GPU PIDs and successful verifier exit.
Artifacts: `/tmp/resident-operators32-tp4-artifacts/`; log:
`/tmp/resident-operators32-tp4.log`. Profiling samples guide investigation and
do not establish a production speedup.

## Local GEMM and TP reduction

The optional operator profiler now separates local GEMM and TP all-reduce.
Real Qwen3-32B CPU TP2 to GPU TP4, prompt4097/chunk4096, fixed eight CPU
layers, after idle expiry: the first CPU chunk executed in 1.3476--1.3483 s;
local GEMM totals were 1.0244 s on rank0 and 0.9571 s on rank1. Reduction
totals were 33.68 ms and 160.75 ms respectively. These wall times include
waiting for the other rank and are not pure SHM transfer costs.

GPU weight completion waits were 0.221--0.241 ms across the cold and recold
runs. Lazy initialization was 0.558 s cold and 0.357 s recold; full request
completion was 2.407 s and 2.069 s respectively. Weight preparation is
recorded separately. This test forces eight CPU layers and does not establish
the optimal partition. All eight output tokens matched cold/warm/re-cold and
the independent checkpoint reference, with stable GPU PIDs and exit zero.
Artifacts: `/tmp/resident-gemm-reduce32-tp4-artifacts/`.

Temporary GEMM methods and the reduction symbol are restored on success and
failure; disabled profiling adds no wrappers. Removed unused forward clock
reads and a redundant rank query from RowParallelLinear, preserving numerical
operations and reduction order. The real run preceded this dead-code cleanup;
26 focused profile/publication/protocol tests passed afterward. No controlled
performance improvement is claimed from the cleanup or diagnostic timings.
