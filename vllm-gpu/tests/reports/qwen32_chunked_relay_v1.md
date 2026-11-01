# Qwen3-32B chunked layerwise relay

## Configuration and evidence

Checkpoint: `/workspace/models/Qwen3-32B`; V1, BF16, SHM, native CPU
safetensors reader, AMX, eager GPU execution, prefix caching disabled.
`max_num_batched_tokens=4096`, `enable_chunked_prefill=True`, block size 32.
These are correctness checks, not prewarmed lazy-init or TTFT measurements.

## Correction: logical boundaries are not weight-readiness measurements

The reported CPU boundaries `8 -> 5 -> 2 -> 0` do **not** demonstrate
progressive physical weight loading. The harness releases the CPU through
`start-relay` only after GPU `engine.lazy_init` returns. CPU AMX computation
therefore does not overlap GPU lazy initialization in these runs.

For PP1, the first GPU model construction allocates the union of the
requested suffix and the uniform full-model range. The SHM loader also
submits that full range for transfer. However, the runner and engine record
only the requested suffix in `loaded_layers`. Subsequent engine steps add
layers to this logical set using a fixed schedule before launching their
loads. These steps can request layers again even if their original physical
transfers have already completed. This is existing engine bookkeeping,
not an injected mock, and is not evidence of real loading progress.

The native manager's `check_load_state` queries both copy and unpack CUDA
completion events and checks pending submissions. The engine's reported
boundaries above are not derived from that query. None of these runs
recorded the complete physical readiness state at lazy-init return; full
submission alone does not prove that every transfer had finished then.

The token comparisons and exact transfer checks remain results for the
executed logical boundaries, with the previously listed failures. They
do not establish readiness-driven CPU contraction or cold-start overlap.
If all model weights finish loading during initialization, subsequent
iterations should use the complete GPU model rather than artificially
retain a shrinking CPU prefix. The implementation and harness corrections below address PP1 readiness
and overlap; incomplete-weight dynamic budgets and progressive PP still
need live validation.

## Configuration results

On 2026-10-12, CPU TP=2 to GPU TP=2 processed a 4161-token prompt as
4096 + 65 prefill tokens, followed by 12 decode tokens. All 12 matched the
full GPU reference. CPU relay boundaries were 8 -> 5 -> 2 -> 0. Both role
processes returned exit code 0. Logs: `/tmp/relay32-chunk-{cpu,gpu}.log`.

The shared `relay_step.py` module publishes CPU requests in actual input
batch order. GPU transfer spans map that order to its own batch order;
adjacent spans are coalesced. Native hidden/residual copies go directly
from pinned SHM into their final GPU buffers and synchronize once after
all spans. No intermediate tensor is allocated for reordering.

Unit validation: CPU metadata and remote sampled-token tests: 6 passed.
GPU metadata, KV plan and final layerwise KV tests: 9 passed. Native CUDA
validation checked exact BF16 hidden/residual values for 65, 1 and 256
tokens, and a reordered 65 + 1 row layout.

## Unresolved validation

A separate CPU TP=2 to GPU TP=4 run with concurrent 65-token and 5-token
prompts produced 40/40 matching tokens for the first request, but the
second request failed the independent GPU reference assertion. GPU role
returned 1, CPU returned 0. Logs: `/tmp/relay32-spans-{cpu,gpu}.log`.
This run is a failure, not proof of concurrent correctness. The harness
now prints both requests' token arrays before their assertions.

CPU TP=2 to GPU TP=4 with a 4161-token prompt and a concurrent 5-token
request passed: both requests generated 40/40 tokens matching their full
GPU references, and both role processes returned 0. Logs:
`/tmp/relay32-chunk-tp4-{cpu,gpu}.log`. The first CPU batch contained 4096
tokens from the long request; the next batch admitted its 65-token remainder
and the short request. This does not explain or supersede the separate
simultaneous-prefill failure above.

Tokenwise CPU TP=2 to GPU TP=4 also passed for a 4161-token prompt and
4096-token chunks. CPU reported `num_computed_tokens=4161`; GPU reported
`num_cached_tokens=4161`. The CPU first token plus three GPU decode tokens
matched the full GPU reference: `[9707, 11, 358, 1097]`. GPU role returned 0.
Logs: `/tmp/relay32-token-chunk-{cpu,gpu}.log`. This was sequential staging
with no AMX/lazy overlap; its initialization timings do not establish the
paper's TTFT or prewarmed initialization targets.

## Concurrent prefill diagnostic

A clean simultaneous 65 + 5 token prefill run with the default batch limit
4096 passed both 40-token references; CPU and GPU returned 0. Logs:
`/tmp/relay32-concurrent-diag-{cpu,gpu}.log`.

With batch limit 256, the failure reproduced at output index 13 (14th token),
after CPU relay ended. Hybrid output's top logprobs for token 66 and 346
were both -1.3173317909240723; the GPU reference favored 346 by 0.0625
(-1.2302823066711426 vs -1.2927823066711426). An additional full GPU
reference with both prompts batched also selected 346. CPU returned 0;
GPU returned 1. Logs: `/tmp/relay32-concurrent256-{cpu,gpu}.log`.

This provides evidence of a low-margin BF16 greedy decision, but does not
establish that KV/hidden-state transfer is error-free. The test remains
failed; no assertion or numerical tolerance has been weakened. Further
state-level comparisons are required before attributing it to numerical
variation alone.

A fresh CPU TP=2 -> GPU TP=4, PP=1 run with an 8193-token prompt (4096 +
4096 + 1) and 40 decode tokens also produced exact output for the long
request. The same run admitted a second 5-token relay request after the
long request; the long request remained exact, while the second request
diverged from its independent GPU reference at decode token 25. This run
did not capture logits at the divergence, so its cause is unestablished.
The earlier batch-256 logprob evidence cannot establish the cause of this
separate failure. It is retained as a failed concurrent-request case; no
tolerance was weakened. Logs: `/tmp/relay32-8193-tp4-new-{cpu,gpu}.log`.

The diagnostic rerun reproduced token index 24 (the 25th output): relay
logprobs for 346 and 1371 were both -0.7019472122192383; independent GPU
favored 1371 by 0.125 (-0.6411229968070984 vs -0.7661229968070984).
A full GPU run with both requests submitted together also selected 1371.
Recomputing the shared 29-token context (five prompt tokens + 24 identical
outputs) in one GPU prefill produced exactly the independent reference's
top-five logprobs. This demonstrates a persistent difference in the hybrid
execution's state or numerics, but does not identify whether CPU arithmetic
or transfer is responsible. The exact output test remains failed.

The harness now saves `divergence.json` with relay, independent incremental,
batched incremental, and recomputed-context top logprobs. Evidence:
`/tmp/relay32-8193-tp4-replay/divergence.json`, logs
`/tmp/relay32-8193-tp4-replay-{cpu,gpu}.log`. GPU exited 1, CPU exited 0.
The synthetic long prompt repeats four token IDs; its matching outputs are
limited evidence and do not establish arbitrary long-text correctness.

## Exact native TP2 cache transfer with actual CPU writer

The CPU fixture generator now accepts TP, block count, KV-head count and
head dimension, and writes each TP shard through the actual
`cpu_attn_reshape_and_cache(..., 'amx')` operator. On 2026-10-12:

```
# Run in vllm-cpu, using its own environment.
PYTHONPATH=. OMP_NUM_THREADS=8 .venv/bin/python tests/produce_native_kv.py \
  /tmp/relay-native-qwen32-tp2.pt --blocks 257 --heads 8 --dims 128 --cpu-tp 2
# Run in vllm-gpu, using its own environment.
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  tests/test_native_kv_transfer.py --cpu-fixture /tmp/relay-native-qwen32-tp2.pt
```

Both exited 0. The dimensions match Qwen3-32B's real config: 8 KV heads,
head_dim 128, block size 32. 257 pages cover an 8193-token prompt. For
CPU TP2 -> GPU TP2 and TP4, every logical GPU rank's K/V matched the
independent logical source exactly (`rtol=0, atol=0`) after the actual native
H2D and unpack path, using reverse source/destination page mapping. The
unused destination page remained unchanged. Logical ranks were tested
sequentially on physical GPU0; this check does not exercise collectives.

The existing full-cache attention check and three subsequent FlashAttention
decode writes also passed using this larger fixture. Attention comparisons
use rtol=0.03, atol=0.01; cache and decode-write comparisons are exact BF16.
Logs: `/tmp/relay-native-qwen32-fixture.log` and
`/tmp/relay-native-qwen32-tp2-gpu.log`.

This isolates the raw TP mapping and AMX packed-cache conversion for random
BF16 values. It does not validate the live engine's per-request/layer/page
selection, hidden/residual states, or CPU-vs-GPU model arithmetic. The real
concurrent-request output failure remains unresolved.

## Live model KV import audit

`verify_qwen_layerwise_v1.py --audit-kv` installs a test-only worker hook
via local callable RPC. Immediately after each layerwise import completes,
before GPU forward or CPU's next iteration, it independently decodes the
actual mapped CPU SHM pages and compares BF16 bit patterns with the actual
GPU attention cache. It incurs CPU/D2H synchronization overhead and is not
a performance benchmark. PP1 and GPU TP >= CPU TP are currently required.

A real Qwen3-32B CPU TP2 -> GPU TP4, 8193 + 5 token concurrent run passed
every live KV import check on all four physical GPUs: per rank, steps
1/2/3 checked 12,582,912 / 12,632,064 / 8,454,144 BF16 elements exactly.
The final step includes historical KV for the short request's CPU layers
0 and 1. The long request's 40 tokens matched; the short request retained
the exact same index-24 failure and logprobs as the unaudited run.
GPU exited 1, CPU exited 0. Logs:
`/tmp/relay32-live-kv-audit2-{cpu,gpu}.log` and
`/tmp/relay32-live-kv-audit2/divergence.json`.

This proves that the selected source KV pages survived this run's actual
copy/unpack unchanged. It does not independently validate whether the
selected pages/layers represent the correct logical request history, nor
CPU arithmetic or hidden/residual transfer. The first audit attempt failed
before inference because callable RPC serialization was disabled; the
harness now enables it only for the explicit local GPU audit option.

The same explicit audit now checks hidden/residual immediately after their
H2D copy, independently deriving source/destination row offsets from request
counts rather than calling the implementation's `hidden_state_spans`.
A fresh real run passed on all four physical TP ranks: per rank the first
two chunks each checked 41,943,040 BF16 elements; the third chunk (1 long
request token + 5 short request tokens) checked 61,440 elements. Every
hidden/residual bit matched the actual SHM source. All live KV import checks
also passed, with the same element counts as above.

The short request's index-24 divergence and all diagnostic logprobs remained
identical. GPU exited 1, CPU exited 0. Logs:
`/tmp/relay32-live-state-audit-{cpu,gpu}.log`, diagnostic:
`/tmp/relay32-live-state-audit/divergence.json`.
This excludes value corruption in the observed hidden/residual copy and
metadata-derived row rearrangement for this run; it does not independently
prove CPU published the correct logical state or that CPU/GPU arithmetic
matches at each layer. Comparing both devices at the same layer boundary
remains necessary before assigning the numerical difference a cause.

## CPU/GPU comparison at the same layer boundary

The explicit audit retains the short request's CPU hidden/residual after
layers 0 and 1, then a pre-forward hook on decoder layer 2 captures the
independent GPU prefill with exactly the same five token IDs and positions
0..4. CPU state is bit-identical to the transferred state by the live audit.
The hook writes `boundary-rank-{rank}.{json,pt}` under the run directory.

Fresh run `/tmp/relay32-boundary-compare` found identical statistics on all
four GPU ranks (25,600 BF16 elements per tensor): hidden had 18,007 unequal
elements, max absolute difference 0.125, RMS difference 0.0029576153
(GPU RMS 0.6182194, CPU RMS 0.6174480); residual had 16,675 unequal elements,
max difference 0.0625, RMS difference 0.0026694539 (GPU RMS 0.6364203,
CPU RMS 0.6365104). Exact live KV/hidden transfer checks still passed and
the final short-request index-24 logprobs/failure were unchanged. GPU exited
1, CPU exited 0; logs `/tmp/relay32-boundary-compare-{cpu,gpu}.log`.

Thus a discrepancy already exists after the CPU prefix computation, before
H2D transfer. This does not yet establish whether it is expected CPU/GPU
numerical rounding, weight reconstruction, or an operator issue. Both the
initial prompt boundary discrepancy and the final greedy mismatch remain
evidence to investigate, not passing numerical tolerances.

The local explicit audit now records layer-0 and layer-1 hidden/residual
outputs on the actual CPU relay and the independent GPU short-prompt
prefill, matching the five token IDs and positions before recording once.
Files are separate by role/rank/layer to prevent decode overwrites.
A fresh run `/tmp/relay32-layer-capture` completed the captures and reproduced
the exact index-24 output failure (GPU exit 1, CPU exit 0).

For 25,600 elements each, layer 0 hidden differed at 15,897 elements,
max abs 0.0625, RMS 0.0017671192; residual differed at 15,398 elements,
max abs 0.015625, RMS 0.0006876996. Layer 1 reproduced the prior boundary
statistics. Both CPU ranks were bit-identical to each other, as were all
four GPU ranks. These comparisons localize a difference to at least the
first decoder layer, rather than a disagreement among same-device TP ranks.
Embedding/layernorm/attention and weight comparisons are still needed to
identify the earliest operator and its cause.
Evidence: `/tmp/relay32-layer-capture/layer-comparison.json` and
`layer-{0,1}-{cpu,gpu}-rank-*.pt`, logs
`/tmp/relay32-layer-capture-{cpu,gpu}.log`.

## First observed operator difference

The layer-0 audit captures input, input RMSNorm, attention, post-attention
RMSNorm/residual and MLP outputs for the exact short prompt. In fresh run
`/tmp/relay32-operator-capture`, layer input was bit-identical. Input RMSNorm
already differed at 6,487/25,600 elements (max abs 0.000244140625, RMS
0.00003521945). Attention RMS difference was 0.0006658417; MLP output RMS
was 0.0017671192. Exact transfer audits still passed; final output assertion
failed at the same index 24 (GPU exit 1, CPU exit 0).

A standalone check used this captured input and the original safetensors
`model.layers.0.input_layernorm.weight`, with epsilon 1e-6. Computing
`BF16(normalize_fp32(x)) * weight_bf16` matched the captured GPU output
exactly, as did the actual CUDA rms_norm operator. Computing
`BF16(normalize_fp32(x) * weight_fp32)` differed from the captured CPU output
at only two elements, but from GPU at 6,487 elements. Thus this observed
first difference is consistent with differing RMSNorm rounding order.
The CPU live dispatch may be fused/compiled; its source-level native formula
alone is insufficient to establish its executed arithmetic. This does not
prove the entire final greedy failure is caused exclusively by RMSNorm,
nor justify changing numerical semantics without performance/correctness
validation. No production operator has yet been changed.

Evidence: `operators-{cpu,gpu}-rank-*.pt`, `operator-comparison.json`,
`rmsnorm-rounding.json` under `/tmp/relay32-operator-capture`; logs
`/tmp/relay32-operator-capture-{cpu,gpu}.log`.

The standalone CPU custom `_C.rms_norm` with captured layer-0 input and
original checkpoint weight exactly reproduced the live CPU output (zero
unequal elements), establishing that the C++ CPU kernel itself is sufficient
to reproduce the observed RMSNorm discrepancy.

An isolated compiled experiment copies the current C++ kernel and changes
only the register arithmetic to round normalization to BF16 before weight
multiplication. Its output exactly matches captured GPU RMSNorm, but the
candidate is slower. With eight OpenMP threads, interleaved median timing
(12 rounds, 100 calls each) was baseline/candidate: 1 token 3.71/4.07 us,
5 tokens 4.07/4.54 us, 65 tokens 17.50/20.79 us, 4096 tokens 947.47/1141.95 us.
A reproducible-script rerun also showed a regression at 4096 tokens:
1057.16/1234.36 us. These are standalone microbenchmarks on repeated captured
rows, not AMX-overlap or TTFT measurements; they do not justify deploying
this candidate. Production CPU RMSNorm remains unchanged.

Reproducible experiment is in CPU `tests/benchmark_relay_rmsnorm.py`:
```
OMP_NUM_THREADS=8 MAX_JOBS=2 .venv/bin/python tests/benchmark_relay_rmsnorm.py \
  --captures /tmp/relay32-operator-capture \
  --checkpoint /workspace/models/Qwen3-32B --output /tmp/relay-rmsnorm-repro.json
```
The candidate exists only in an isolated test namespace, not as an engine
fallback. Evidence `/tmp/relay-rmsnorm-benchmark.json`,
`/tmp/relay-rmsnorm-repro.json` and `/tmp/relay-rmsnorm-repro.log`.

## Isolated wide SIMD rounding experiment

The CPU benchmark now includes a BF16 16-lane specialization, two independent
variance accumulators, and a 16-element tail. Its normal RMSNorm rounds the
normalized value before multiplying the weight. Its fused residual RMSNorm
also rounds the residual sum before computing variance, matching CUDA's
executed order. Both outputs match actual CUDA references exactly on the
captured real Qwen3-32B layer-0 inputs. The baseline fused norm differs at
6,825 of 25,600 output elements, while its residual is exact.

GPU `tests/produce_relay_norm_reference.py` produces the fused reference
from the **same CPU-captured** attention output and residual, using the
original checkpoint weight and configured epsilon. Its output reproduced
the prior ad-hoc CUDA fixture exactly. CPU benchmark accepts this fixture
through `--gpu-reference`.

Normal RMSNorm improved at 4096 tokens in several eight-thread trials;
for example baseline/wide 1048.85/908.22 us. Fused norm did not meet the
performance requirement: in that same trial baseline/wide was
1280.71/1392.14 us. A 48-thread rerun restricted to CPUs 96..143 with
`KMP_AFFINITY=disabled` also measured a fused regression at 4096 tokens:
382.66/389.40 us. These are isolated repeated-input operator benchmarks,
not engine TTFT or loading-overlap results. Two attempted OpenMP-place
runs were terminated after thread-affinity inspection showed all workers
restricted to one CPU; neither is valid timing evidence. No production
kernel or engine fallback was installed.

The live harness has an explicit `--cpu-norm-experiment /path/to/library.so`
diagnostic override. A fresh actual CPU TP2 -> GPU TP4, PP1 run using the
wide experimental operators processed 8193 prompt tokens as 4096+4096+1
with a concurrent five-token short request. Both requests' 40 output token
IDs matched their independent complete GPU references. CPU/GPU both exited
0. Live KV and hidden/residual bit checks passed on all four GPU ranks.
CPU prefix boundaries remained 8 -> 5 -> 2 -> 0. The former short-request
token-index-24 mismatch was absent in this run; no output assertion was
weakened. This is an experimental pass, not a default-engine fix or proof
for all inputs, TP configurations, or PP stages.

Layer-0 input RMSNorm is now bit-identical across CPU/GPU. Attention RMS
difference remains 0.0005182488 and the layer-2 hidden boundary difference
remains 0.0026609732, so the experiment does not eliminate all device
arithmetic differences. Production operators remain unchanged because the
fused candidate has not satisfied the no-regression requirement.

Evidence: `/tmp/relay32-norm-experiment/operator-comparison.json`,
`operators-*.pt`, `boundary-rank-*.json`, logs
`/tmp/relay32-norm-experiment-{cpu,gpu}.log`; benchmark records
`/tmp/relay-rmsnorm-wide-unroll.json` and `/tmp/relay-rmsnorm-wide-48.json`.
The exact tested extension was retained at `/tmp/relay-norm-wide-unroll.so`.

Reproduce the fused reference in the GPU environment, then pass it to the
CPU isolated benchmark:

```bash
PYTHONPATH=. /workspace/vllm-gpu/.venv/bin/python \
  tests/produce_relay_norm_reference.py \
  --captures /tmp/relay32-operator-capture \
  --checkpoint /workspace/models/Qwen3-32B \
  --output /tmp/relay-fused-reference-repro.pt
```

```bash
OMP_NUM_THREADS=8 MAX_JOBS=2 /workspace/vllm-cpu/.venv/bin/python \
  tests/benchmark_relay_rmsnorm.py \
  --captures /tmp/relay32-operator-capture \
  --checkpoint /workspace/models/Qwen3-32B \
  --gpu-reference /tmp/relay-fused-reference-repro.pt \
  --output /tmp/relay-rmsnorm-wide.json
```

## Varied real-text prompt with default production operators

The harness now accepts `--prompt-token-file`, a JSON list of token IDs;
the actual list length determines `max_model_len`. This avoids limiting
long-request validation to the repeated four-token synthetic prompt.

Tokenizing the attached RelayServe paper with the actual checkpoint tokenizer
produced 12,461 tokens. The first 8,193 contain 1,839 distinct token IDs.
The local fixture is `/tmp/relay-qwen32-paper-8193.json`; no paper text or
token payload is checked into the repository.

A fresh CPU TP2 -> GPU TP2, PP1 run with this input and a concurrent five-token
short request used default production RMSNorm operators (no experimental
override). Chunk budget was 4096, giving 4096+4096+1 prompt tokens. Both
requests' 40 generated token IDs matched their independent complete GPU
references. CPU and GPU both exited 0. CPU prefix progressed 8 -> 5 -> 2 -> 0.
This is an actual varied-text multi-chunk pass, with eager execution and
prefix caching disabled; it does not supersede the prior TP4 failure or
establish CUDA-graph, PP, 72B, or performance correctness.
Logs: `/tmp/relay32-paper-default-{cpu,gpu}.log`.

The same varied-text fixture with default production operators and GPU TP4
also produced 40/40 exact long-request tokens, but the concurrent short
request again diverged at output index 24: relay chose 346 and GPU reference
chose 1371. The top logprobs exactly reproduce the earlier synthetic-long
case: relay 346/1371 both -0.7019472122; reference 1371 -0.6411229968,
346 -0.7661229968. The long-request pass therefore does not establish
concurrent TP4 correctness, even with real-text input. The failing assertion
remains enabled. Logs: `/tmp/relay32-paper-default-tp4-{cpu,gpu}.log`,
diagnostic `/tmp/relay32-paper-default-tp4/divergence.json`.

## Readiness and final KV handoff correction

The PP1 engine no longer adds a fixed number of layers per iteration.
Every completed relay batch queries copy/unpack completion on every TP
rank. It also records CUDA forward time (after suffix weight waits and KV
import) and native per-layer transfer-through-unpack event durations. The
next loading budget is floor(GPU compute seconds / layer transfer seconds),
limited to remaining layers, using the maximum TP durations. Missing timing
samples give zero prediction. Predicted boundaries are kept separate from
completed weights, and executing each layer retains its actual weight fence.
CPU no longer subtracts a hard-coded two layers from the reported boundary.

Before acknowledging complete GPU takeover, the engine waits for complete
weights and imports the final CPU prefix KV history. Omitting that final
import caused a new TP4 long-prompt mismatch; adding it restored both
long and concurrent short token comparisons. This differs from the earlier
TP4 arithmetic mismatch: no production RMSNorm change was used here.

The harness now releases CPU immediately before GPU lazy_init, allowing
AMX to overlap model-specific GPU initialization. Initial process/context
setup remains outside this overlap. Native transfer order is now reversed
for layerwise loading, as described in the paper. PP completion annotation
uses the actual uniform transition flag rather than iteration count; real
progressive PP assistance still needs implementation and live validation.

CPU TP2 -> GPU TP4 with the actual 8193-token paper fixture, 4096 chunk
budget and concurrent five-token request passed 40/40 tokens for each
request after final KV import was added. Both roles exited 0. Four rank
readiness boundaries were zero at first handoff; measured GPU forward was
1.455--1.464 s, mean layer transfer-through-unpack duration 23.35--33.04 ms
across ranks. These timings include competing work in this live run and are
not isolated performance or subsecond initialization evidence. Logs:
`/tmp/relay32-readiness-kv-tp4-{cpu,gpu}.log`.

Earlier CPU TP2 -> GPU TP2 readiness run also passed both 40-token checks
and exited 0, logs `/tmp/relay32-readiness-tp2-{cpu,gpu}.log`. That run
preceded the final-KV correction and therefore cannot establish the safety
of skipping the final import. Focused GPU tests: 21 passed. Native timing
extension rebuilt successfully. The live runs had all weights ready at the
first handoff, so nonzero dynamic loading budgets remain unverified by an
actual incomplete-weight cold-start run.

A fresh rerun of the final revision (including reverse transfer priority,
predicted/completed state separation, and final KV import) also passed both
40-token comparisons and both roles exited 0. Logs:
`/tmp/relay32-readiness-final-{cpu,gpu}.log`. CPU shared metadata tests passed
14/14 when run independently of repository conftest; normal repository
collection still fails on its existing `vllm.config.model` registry import.

## Loading-priority and progress-protocol audit

The prior all-range reverse order was corrected to match §III-B: load the
current GPU execution assignment in increasing order first, then load the
remaining allocated recovery layers in reverse order. Storage allocation
still covers the cold/final PP union. `execution_layer_idxes` now carries
the priority independently of `layer_idxes` storage coverage.

SHM outputs now keep actual completed suffix size in
`shm_gpu_loaded_layers` and the predicted next CPU prefix in the separate
`shm_next_cpu_boundary` field. CPU consumes the explicit planned boundary;
it does not describe that prediction as completed weight storage. The next
boundary cannot increase beyond the current CPU ownership, even if a prior
prediction was optimistic and requires a GPU weight fence.

The loader tracks submitted layers so incremental/uniform recovery does
not write a second copy into weights already queued or used by inference.
Uniform readiness still checks the entire final assignment: skipping a
submission does not imply that its CUDA events have completed.

Actual CPU TP2 -> GPU TP4, PP1, default production operators, 8193-token
paper fixture (4096+4096+1), concurrent five-token short request: both
40-token outputs matched the complete GPU references and both roles
exited 0. Logs `/tmp/relay32-priority-protocol-{cpu,gpu}.log`. All weights
were complete at first handoff, so this again does not validate incomplete
weight budgets. GPU forward times were 0.557--0.571 s and mean per-layer
transfer-through-unpack times 20.41--23.67 ms across ranks. These are live
observations, not a controlled comparison to the prior timings.

Actual CPU TP2 -> GPU TP2/PP2, cold assignment 20/36 layers, 4161-token
prompt (4096+65): uniform recovery completed, all 12 output tokens matched
the full GPU reference, and both roles exited 0. Logs
`/tmp/relay32-priority-pp2-{cpu,gpu}.log`. This validates the current gated
recovery path with deduplicated submissions, not progressive PP assistance.
Focused GPU tests: 30 passed. CPU shared tests: 21 passed when collected
independently of the existing repository conftest import failure.

## Iteration handshake and CPU KV lifetime

User clarification: CPU must wait for the next iteration; do not start the
next CPU batch while GPU executes the current one. The production loop
continues to use CPU publish -> GPU execute -> feedback -> next CPU
iteration. Cross-iteration CPU/GPU batch overlap is not a current execution
requirement, despite the draft paper description in §III-B. GPU weight
loading and model-specific initialization overlap remain requirements.

PP1 final takeover now releases CPU scheduler KV blocks after the GPU's
zero-boundary acknowledgment, which is emitted after final KV import.
A zero predicted boundary with incomplete actual weights is rejected before
CPU ownership is released. Previously, final CPU abort retained KV blocks
indefinitely. PP legacy first-batch cleanup still retains KV because it has
no final-import acknowledgment; its progressive cleanup and restoration
protocol still need work.

Actual Qwen3-32B CPU TP2 -> GPU TP4/PP1 with default production operators,
8193-token paper prompt (4096+4096+1), concurrent five-token request:
both requests matched all 40 output tokens of their complete GPU references;
both roles exited 0. CPU logs show both requests aborted with KV freeing
only after GPU takeover. Logs `/tmp/relay32-kv-release-{cpu,gpu}.log`.
CPU finish tests passed 4/4 using independent collection, covering legacy
PP retention, acknowledged PP1 release, and rejection of premature release.
No cross-iteration lookahead was added.

## CPU queued-execution handshake audit

CPU commit `0ad0ed5` disables the batch queue for lazy SHM layerwise
relay even when an executor advertises concurrent batches (PP or async
scheduling). The ordinary queued path bypassed the SHM publish/feedback
handshake. The serial step now resolves executor futures before publishing
hidden states and scheduler metadata. Nine independent CPU finish tests
passed, including configuration selection and the order CPU completion ->
hidden-state publication -> metadata publication -> GPU feedback.

GPU commit `dc6e66f` makes native initialization flags atomic and unifies
blocking loads with the double-buffered asynchronous copy/unpack pipeline.
Native CPU TP2 -> GPU TP2/TP4 checks passed for all target ranks, including
repeated loads and positive completed-layer timing samples; 30 focused GPU
relay tests passed.

Actual Qwen3-32B CPU TP2 -> GPU TP4/PP1 at these revisions passed again:
8193-token paper prompt with chunk budget 4096 (4096+4096+1), followed by
a second short relay request. Each request's 40 output tokens exactly
matched the independent complete GPU reference; both roles exited 0.
Logs: `/tmp/relay32-serial-v2-{cpu,gpu}.log`. CPU released KV after takeover.
GPU lazy initialization was 5.3105 s. GPU forward times were
0.5418--0.5507 s; completed layer transfer-through-unpack means were
15.77--20.97 ms across ranks. These observations are not a controlled
performance comparison. All weights were ready at first handoff, so this
run does not establish progressive incomplete-weight handoff or the <1 s
initialization target. The live run used normal scheduling; async-executor
ordering was tested separately in the focused tests.

The first attempt stopped during CPU distributed initialization with a
native SHM `EEXIST` error, before model loading or relay. Its GPU waiting
process was terminated, and the fresh second attempt above completed.
Pre-existing processes and SHM objects were preserved.

## Offline KV sizing audit

The existing SHM V1 path already avoids `profile_run()`. It imports
`_MODEL_DEFAULT_NON_KV_CACHE_MEMORY` from
`vllm/model_executor/models/_cached_model_info.py`. That file contains
model-name-only non-KV occupancy values at batch-token limit 8192, rather
than a deployment-specific KV byte budget. Current matching uses the
model directory basename, so renamed SHM model directories miss the table.
The table has no TP/PP or device dimensions. Restoring a reliable offline
file lookup remains required; manual KV-budget configuration is not the
intended default.

A temporary explicit-budget experiment used 4 GiB of KV per GPU with real
Qwen3-32B CPU TP2 -> GPU TP4/PP1. The 64-layer, two-local-KV-head,
head-dim-128 BF16 layout uses 64 KiB per token per GPU, so the expected
capacity is 65,536 tokens (2,048 blocks of 32 tokens). Runtime reported
exactly that capacity. The 8193-token prompt and second short relay each
matched all 40 GPU-reference output tokens; both roles exited 0.
Logs: `/tmp/relay32-explicit-kv-{cpu,gpu}.log`.
Lazy initialization was 5.0212 s, so avoiding the model-name budget
estimate alone does not resolve the warmup cost. This is an uncontrolled
observation, not a speedup claim. The temporary explicit-budget API and
tests were removed after this experiment in favor of continuing the
existing offline-file design; they are not production features.

### TP communicator preparation before lazy initialization (2026-10-12)

The SHM lazy worker now runs the logits TP collective once in `init_device`,
using its actual GroupCoordinator. Every TP rank participates on each PP stage;
TP=1 skips the operation. This prepares only the current group, not the planned
SLLM resident worker's registry of reusable TP/PP groups. The production sampler
and model warmup remain intact.

Fresh-process isolation benchmark using Qwen3-32B config (hidden size 5120,
vocabulary 151936, two requests, H20 TP=4) is reproducible with:

```bash
PYTHONPATH=. .venv/bin/torchrun --standalone --nproc-per-node=4 \
  tests/benchmark_relay_sampler_warmup.py \
  --checkpoint /workspace/models/Qwen3-32B --prepare-communication
```

The benchmark uses synthetic LM-head weights and production dummy sampler
warmup. It is not a model accuracy test. Without preparation the first raw NCCL
all-gather measured 1.466–2.460 s, sampling 0.200–0.208 s. With preparation the
probe cost 1.489–1.513 s before timing model work, first logits all-gather
0.00035–0.03498 s, sampling 0.208–0.243 s. Repeated total calls were about 1 ms.
Logs: `/tmp/relay-sampler-tp4.log`, `/tmp/relay-sampler-tp4-prepared.log`.

Real checkpoint validation: CPU TP=2 → GPU TP=4/PP=1, prompt 8193 tokens
(chunk 4096), followed by a short second relay. Both requests matched all 40
GPU reference tokens and both processes exited 0. Logs:
`/tmp/relay32-prepared-tp-v1-{cpu,gpu}.log`. Measured lazy initialization was
3.8273 s versus the previous unprepared run's 5.3105 s. This is a single-run
comparison, not a controlled throughput/TTFT claim. Current-group preparation
cost 2.123–2.194 s during worker startup, outside lazy initialization. It moves
work into preparation; it does not eliminate process-startup work.

Remaining lazy phases: model allocation/structure/binding RPC 1.046 s,
KV allocation/binding 0.509 s, dummy model warmup 1.519–1.557 s, sampler
warmup 0.531–0.574 s. Offline KV budget/config calculation about 0.06 s.
The sub-second model-specific initialization target remains unmet. Weights were
fully ready at the first handoff; this run does not validate progressive loading.
Focused relay tests: 40 passed, including current-group preparation for TP1/2/4
and both gather/all-gather policies.

### Warmup synchronization diagnosis (2026-10-12)

Added opt-in `--warmup-diagnostics` to the real relay verifier. The test worker
profiles CPU calls, waits the current compute stream before each phase, records
CUDA events around submission, and waits only the ending event. It queries
completed native layer transfer timings before/after each call. It repeats each
warmup call once in the same worker to identify first-call costs. These waits,
CPU profiling and repeated warmup are diagnostic only; production policy is
unchanged. The CUDA event interval includes CPU launch gaps and must not be
interpreted as summed GPU kernel execution time.

Use the usual verifier command with the GPU process PYTHONPATH including this
repository's `tests` directory, and add `--warmup-diagnostics` for GPU only.
The diagnostic worker is restored after warmup, before real inference.

First diagnostic run logs:
`/tmp/relay32-warmup-diagnostic-v1-{cpu,gpu}.log`.
Repeated-call run logs:
`/tmp/relay32-warmup-repeat-v1-{cpu,gpu}.log`.
Both runs used the actual Qwen3-32B checkpoint, CPU TP2 → GPU TP4/PP1,
8193 prompt tokens with chunk4096, plus a second short relay request. All
40 tokens of each request matched independent GPU references; both CPU/GPU
processes exited 0 in both runs.

All four ranks already had 64 completed native layer transfers before model
warmup, and still had 64 afterwards. In the first diagnostic run the prior
compute stream waits were 15–34 microseconds and ending-event waits 9–20
microseconds. Model warmup CPU calls took 1.708–1.731 s; rank0 `torch.linear`
accumulated 0.736 s and embedding path 0.780 s (nested times overlap and must
not be summed). Sampler CPU calls took 0.432–0.578 s, including first sort,
softmax, where, temperature scaling and gather calls. This run provides no
evidence that incomplete weight copies or a preceding compute-stream backlog
caused the long measured warmup.

In the repeated-call run, first model calls took 1.611–1.648 s and second
calls 0.003–0.050 s. First sampler CPU calls took 0.424–0.559 s and second
CPU calls 0.00138–0.00190 s. Second sampler ending-event waits ranged from
7 microseconds to 0.152 s across ranks, so the second total sampler cost must
not be described as universally 1–2 ms. All layer copies remained complete.
The sharp first/repeated difference supports prewarming reusable runtime
paths in resident workers, rather than adding a weight-copy barrier as a
latency fix. It does not identify every CUDA module nor isolate AMX contention.

The historical worker called `torch.cuda.synchronize()` at SHM warmup entry;
the current worker does not. No global barrier was restored based on these
results. Current native per-fragment copy-stream waits during module warmup
remain intact. These instrumented runs are not production latency benchmarks.
SLLM still requires the requested resident worker/model-switch/group registry
implementation; preparing only the current engine TP group is not completion.
