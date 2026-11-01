Actual workspace validation on eight H20 GPUs
============================================

CPU repository: /workspace/vllm-cpu at 7ee0ff6.
GPU repository: /workspace/vllm-gpu at 5d7ccd3c4 plus the first-token
timestamp change committed with these logs. Each role uses its own .venv.
Checkpoint: /workspace/models/Qwen3-32B, real 64-layer model and weights.

CPU command, executed first from /workspace/vllm-cpu:

```bash
VLLM_CPU_OMP_THREADS_BIND='0-47|48-95' VLLM_CPU_KVCACHE_SPACE=1 OMP_NUM_THREADS=48 timeout 600 .venv/bin/python /workspace/vllm-gpu/tests/verify_qwen_layerwise_v1.py --role cpu --folder /tmp/relay-workspace-32b-tp4-20261013 --checkpoint /workspace/models/Qwen3-32B --tp 2 --cpu-layers 8 --prompt-tokens 128 --output-tokens 8
```

After cpu-ready.json exists, GPU command from /workspace/vllm-gpu:

```bash
VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_FLASH_ATTN_VERSION=3 CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 450 .venv/bin/python tests/verify_qwen_layerwise_v1.py --role gpu --folder /tmp/relay-workspace-32b-tp4-20261013 --tp 4 --cpu-layers 8 --prompt-tokens 128 --output-tokens 8
```

Both commands exited 0. Lazy initialization call took 1.569517 seconds;
call-start to first returned token took 1.878235 seconds. These are single
observations, exclude process/context startup and CPU weight preparation,
and include concurrent CPU request computation and GPU weight transfer.
Chunked prefill is enabled with a 4096-token budget; this 128-token request
does not test crossing chunks. The 8 generated tokens exactly match a later
full GPU prefill on the same loaded model. That comparison tests relay state
continuity, not an independent safetensors GPU loader or model accuracy.
All rank readiness boundaries were 0 after first GPU execution, so this run
does not exercise continued CPU assistance with incomplete weight loading.

4096-token baseline before initialization optimization
-----------------------------------------------------

GPU revision: 517c87c08; CPU revision: 7ee0ff6. Same TP and model as above.
Commands use folder /tmp/relay-workspace-32b-tp4-4096, --prompt-tokens 4096,
and VLLM_CPU_KVCACHE_SPACE=4; other arguments are unchanged.
Both processes exited 0. Lazy initialization: 1.797868 seconds.
Lazy-init call start to first token: 2.983080 seconds. CPU first 8 layers:
2.249–2.255 seconds, overlapping initialization. Output tokens exactly match
the same-SHM full-GPU reference. This request fills one 4096-token chunk;
it does not cross chunks. The fixed 8-layer split was not planner-selected.
Raw logs: qwen32-tp2-tp4-4096.cpu.log and qwen32-tp2-tp4-4096.gpu.log.

CPU-default construction without redundant DeviceContext
-------------------------------------------------------

CPU revision: 7ee0ff6. GPU: 517c87c08 plus the loader optimization committed
with this result. Commands above use folder
/tmp/relay-init32-tp2-fastcpu-20261013 and GPU --tp 2 with devices 0,1.
CPU TP=2, 128 prompt tokens, CPU 8 layers, 8 output tokens, packed weights,
FA3, complete prefill/decode/sampler warmup; AMX and loading remain concurrent.
No profiling environment variables were enabled. Both processes exited 0.
Structure construction: 0.378843 and 0.384551 seconds.
Lazy initialization: 1.028670 seconds. Call start to first token: 1.122595
seconds. Generated tokens exactly match same-SHM full GPU prefill.
This is one observation, not a stability result or an independent loader
accuracy test. The previous TP=2 observation was 1.112162 seconds; the
difference is not a controlled repeated comparison.
Raw logs: qwen32-tp2-fastcpu.cpu.log and qwen32-tp2-fastcpu.gpu.log.

Affine KV block addresses instead of per-block pointer hash tables
-----------------------------------------------------------------

CPU revision remains 7ee0ff6. GPU baseline: fa72a42cb. Changed native KV
manager stores per-layer GPU bases and per-CPU-rank SHM bases; block addresses
are computed on transfer. No duplicate legacy pointer tables remain.
The GPU native extension was rebuilt using its own .venv.

Same commands/configuration as the 128-token runs above, with
VLLM_PROFILE_RELAY=1 on GPU (phase timestamps, no profiler or extra fences).
Folders match each raw-log stem. All six CPU/GPU processes exited 0.

| GPU TP | Native pointer tables | KV registration per rank | Lazy init |
| --- | --- | --- | --- |
| 4 | Before | 0.197696–0.217060 s | 1.273340 s |
| 4 | Removed | 0.001528–0.009642 s | 1.005579 s |
| 2 | Removed | 0.010087–0.020323 s | 0.785747 s |

Raw logs: relay-init32-tp4-kvprofile-20261013.{cpu,gpu}.log (before),
relay-init32-tp4-kvbase-20261013.{cpu,gpu}.log and
relay-init32-tp2-kvbase-20261013.{cpu,gpu}.log (after).
All relay outputs match the same-SHM full GPU reference exactly. These
are single observations with AMX/loading overlap and unchanged warmup;
they do not establish stable sub-second initialization or planner accuracy.

Native regression passed all 18 TP/rank combinations, including head
replication and CPU TP greater than GPU TP. Added a sparse multilayer test
with nonadjacent pages, separate K/V planes, and replacement GPU allocations
for every TP4 rank; exact BF16 comparisons and untouched-page checks pass.
Three SHM reset/reattachment cycles also pass. Logs:
relay-kv-base-native-test.log and relay-kv-base-lifetime.log.

Additional 32B diagnostic observations (not the 72B TP4 target)
-------------------------------------------------------------

GPU revision a0c1724d2; CPU revision 7ee0ff6. The unmodified command with
folder /tmp/relay-init32-tp4-repeat1-20261013 gave lazy initialization
1.109946 s (external monotonic timestamps) and creation-call to first-token
1.541184 s. Architecture construction per rank: 0.408594–0.537978 s;
dummy model warmup: 0.300–0.314 s. Both roles exited 0, with 8 output
tokens exactly matching the same-SHM reference.

A diagnostic using the same arguments with folder
/tmp/relay-init32-tp4-affinity-20261013 and `taskset -c 96-191` before the
GPU command made initialization slower: 1.902904 s (engine timer),
architecture 0.784474–0.863761 s and dummy warmup 0.471–0.526 s.
Both roles exited 0 and the output comparison passed. This affinity setting
was not adopted in engine code or subsequent benchmark commands.

These are Qwen3-32B TP4 diagnostic measurements. The requested TP4
performance target is Qwen2.5-72B with real BF16 weights and CPU TP2;
the 32B results must not be presented as 72B results.
Raw logs use relay-init32-tp4-{repeat1,affinity}-20261013.{cpu,gpu}.log.


Real Qwen2.5-72B TP4 and bounded ordinary RoPE cache
--------------------------------------------------

Downloaded the official base Qwen2.5-72B checkpoint from ModelScope to
/workspace/models/Qwen2.5-72B: 47 files, 37 safetensors shards, about
145.424 GB. Every downloaded file passed its source-manifest SHA256 check.
The source manifest and completion record are preserved here; weights are
not committed. Architecture: 80 layers, hidden 8192, intermediate 29568,
64 query heads, 8 KV heads, head dimension 128, BF16, context 131072.

SHM Qwen2/Qwen3 workers now create ordinary unscaled text RoPE caches for
max(configured model context, scheduler batch budget), bounded by the HF
maximum. The configured context is unchanged. Scaled RoPE, dual chunk
attention, multimodal models and non-SHM loaders preserve the old length.
Every retained row is bit-exact in FP32/BF16/FP16. This avoids constructing
131072 cache positions when the worker can only use 4096 or 4104 positions.
Complete prefill/decode/sampler warmup and its compute-stream completion
fence are preserved; weight transfer remains asynchronous.

Real-weight CPU TP2 -> GPU TP4 results, FA3, packed CPU weights, eager,
chunked prefill budget 4096, diagnostic fixed CPU split of 8 layers:

| Run | Prompt | Engine lazy initialization | Creation call to first token |
| --- | ---: | ---: | ---: |
| Before cache change | 128 | 1.807344 s | 2.085217 s |
| Bounded cache | 128 | 1.538048 s | 1.844406 s |
| Repeat (minor concurrent pytest activity) | 128 | 1.643 s | 1.959400 s |
| Bounded cache, larger context | 4096 | 1.740 s | 5.742975 s |

The timer begins just before lazy_init, includes model construction RPC and
worker preparation, and ends after readiness. CPU checkpoint preparation
precedes this interval. TTFT includes CPU prefill and GPU continuation.
These observations do not establish stable subsecond initialization. The
repeat is not a clean latency baseline because a 6.46 s pytest invocation
occurred during that run. The 4096 run is one chunk, not cross-chunk coverage.
The fixed split does not validate planner choices.

All runs exited 0 for both roles and produced 8 tokens exactly matching the
same-SHM full-GPU reference. In addition, a fresh standard safetensors TP4
model on GPUs 4-7 produced the same 8 tokens for both 128 and 4096 inputs.
This independent reference ran before the repeat/4096 performance tests;
its loading latency is not an SHM initialization baseline. Its log contains
an engine shutdown message after both reference outputs were saved; the
script exited 0. Output agreement on these prompts is not exhaustive weight
or request validation.

GPU focused regression: 27 passed (RoPE, SHM loader, complete relay warmup).
CPU RoPE regression: 11 passed with --confcutdir=tests/model_executor, since
the existing root conftest imports unavailable vllm.config.model.
Both repositories pass git diff --check. Commands, environments, base
revisions, raw CPU/GPU logs and independent reference outputs accompany this
record. Benchmark revisions have the bounded-cache patch uncommitted.


Blocking host waits during per-task DMA pacing
---------------------------------------------

Replace the per-task cudaStreamSynchronize during KV/module initialization
with a reusable cudaEventBlockingSync | cudaEventDisableTiming event recorded
at the same copy-stream boundary. Submission remains serialized; the event
lives through the manager lifetime. This preserves pacing while sleeping the
loader host thread instead of spending its wait interval spinning a CPU core.
It does not fence all weights or the compute/unpack streams.

Independent 64 MiB pinned H2D wait microbenchmark (100 samples, two rounds):
stream wall median 1.219/1.222 ms, thread CPU median 1.216/1.216 ms;
blocking event wall median 1.261/1.262 ms, thread CPU median 0.0156/0.0169 ms.
The event adds about 40 us in this isolated experiment while reducing host
CPU use. Adoption is based on relay results, not isolated copy speed.

Real 72B TP4, 128 prompt tokens, unchanged 8-layer CPU split:
- GPU 0,1,2,3, same layout as prior 1.538 s run: initialization 1.237 s,
  creation-call to first token 1.495585 s, dummy warmup 0.391–0.402 s.
- GPU 0,1,4,5, balanced across NUMA to match CPU source shards:
  initialization 1.179 s, creation-call to first token 1.450881 s,
  dummy warmup 0.366–0.392 s.
Both roles exit 0 and eight output tokens match the preserved independent
safetensors reference. These remain individual observations, not proof of
stable subsecond initialization. The changed device layout is explicitly
separated from the event-only comparison. The harness now defaults to the
balanced layout and saves tracked diffs plus SHA256 alongside base revisions.
Older runs remain on their recorded GPU layouts.

Host cProfile diagnostic (not a latency baseline) found linear host calls
0.225–0.289 s and console writes 0.046–0.091 s across four ranks. It also
exposed that initial layerwise warmup executes the active suffix, not every
constructed recovery layer; earlier 'entire local stage' unit coverage did
not cover a narrowed execution range. Complete recovery-range warmup still
needs correction and real validation; current measurements must not be
interpreted as proof of that requirement.

## Representative eager Qwen module warmup

The preceding recovery-range diagnostic exposed a narrowed forward range,
but executing all recovery layers is not necessary to load shared CUDA
kernels. The uncommitted full-range prototype was replaced by representative
layer warmup for SHM eager, unquantized Qwen2/Qwen3 without sliding-window,
dual-chunk attention or speculative decoding. One allocated decoder layer
runs both prefill and decode, exercising plain and fused-residual RMSNorm,
attention, projections and MLP. Embedding, final norm and sampler remain
covered by the existing dummy model/sampler paths. Each TP rank runs warmup.
Active relay bounds are restored in finally. Compiled execution, kernel
tuning and CUDA Graph capture retain their original bounds. Other model
paths retain their prior warmup behavior. No whole-weight fence was added.

Real Qwen2.5-72B CPU TP2 -> GPU TP4, GPU 0,1,4,5, FA3, BF16,
chunk budget 4096, fixed diagnostic CPU split of 8 layers (not planner-selected):

| Prompt | Full-range prototype lazy / TTFT | Representative lazy / TTFT |
| --- | --- | --- |
| 128 | 1.211 s / 1.444578 s | 1.023 s / 1.375115 s |
| 4096 | 1.518 s / 5.614894 s | 1.128 s / 5.627565 s |

The 4096 TTFT did not improve despite shorter initialization: this request
is dominated by CPU prefill and relay computation. These are individual
observations; stability and subsecond initialization remain unproven. Both
roles exited 0 in every recorded run; eight output tokens match the preserved
independent safetensors reference. The prompt is repetitive, not exhaustive
output coverage; 4096 is one chunk, not a cross-chunk test.

Raw logs, exact commands, exit codes and patch snapshots are in
`representative_warmup/`. Full-range prototype logs are retained as comparison
evidence, not as the implementation to deploy.

## Initialization stability follow-up

Five fresh-process 128-token runs with the same balanced GPU layout and
48 CPU threads per rank measured lazy initialization 1.023, 0.959, 1.086,
0.972, 0.927 s. Mean 0.9934 s, median 0.972 s, population standard deviation
55.68 ms (5.60%), range 159 ms. No multi-second spike reproduced in this
small sample; stable subsecond readiness is not established. All runs exited
0 and outputs were explicitly compared with the independent safetensors JSON.
Model-loading RPC ranged 0.596–0.661 s; KV/scheduler/warmup 0.330–0.424 s.
Raw samples and calculation are in `representative_warmup/stability-summary.json`.

Added lightweight diagnostics under existing VLLM_PROFILE_RELAY: architecture
wall time vs host-thread CPU time, and final compute-stream completion time.
A real run passed with lazy 0.969 s; architecture wall 0.345–0.457 s versus
host-thread CPU 0.124–0.129 s. Final compute-stream wait was 0.053–0.927 ms,
so the final completion fence was not the dominant cost in that observation.
Wall-minus-thread CPU also includes other threads and native waits; it does
not uniquely prove AMX interference. cProfile diagnostic, excluded from
latency statistics, locates work across module construction rather than a
single new slow operator. Profile summary and exact patch/logs are preserved.
The machine also has unrelated long-running CPU engines; these were observed
and left untouched. Measurements describe this shared environment.

A rejected diagnostic reserved CPU physical cores 44–47 and 92–95 for GPU
hosts, reducing CPU AMX to 44 threads per socket. Lazy initialization rose
to 1.235 s and creation-to-first-token to 1.677438 s. Both roles exited 0
and outputs matched, but this placement was not adopted. Its commands and
patch snapshot are preserved only as diagnostic evidence; the harness default
remains 48 CPU threads and unrestricted GPU-host affinity.
