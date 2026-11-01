# Asymmetric PP KV migration validation

On 2026-10-12, fixed the execution path that only transferred KV from
rank r to r-1. All ranks now traverse the full, identical ordered edge
list from the migration plan, including non-adjacent and ascending edges.
The execution caller passes the complete plan, rather than one rank's
local sub-plan. One final barrier completes migration before cleanup.

A collective CPU metadata check verifies matching plans, active block IDs
and presence of every source/destination layer before any blocking receive.
Missing received layers now raise errors rather than being silently skipped.

Real CUDA/NCCL checks were run using `tests/verify_pp_kv_migration.py`:

```
PYTHONPATH=. /workspace/vllm-gpu/.venv/bin/torchrun --standalone \
  --nproc-per-node=4 tests/verify_pp_kv_migration.py
PYTHONPATH=. /workspace/vllm-gpu/.venv/bin/torchrun --standalone \
  --nproc-per-node=8 tests/verify_pp_kv_migration.py --tp 2
```

Both returned exit code 0. They tested cold layer counts 1/1/1/13 and
13/1/1/1, restoring 16 layers to uniform counts 4/4/4/4. Every owned
destination layer was checked for exact BF16 values at active blocks
1/3/5; inactive blocks retained their original value. TP=2 used distinct
values for the two TP shards to detect communication-group mixing.

Fault injection deleted layer 12 at PP rank 3, or gave PP rank 3 a different
active block list. Every rank rejected each fault before P2P communication,
without a hang. Logs: `/tmp/relay-pp-migration.log` and
`/tmp/relay-pp-tp2-migration.log`.

Focused unit regression: 14 passed, including migration plan, relay state,
metadata row extents, tokenwise KV batching and final layerwise KV import.

Scope: these checks verify the actual migration helper and vLLM communication
groups. They do not prove full model startup, uniform-weight readiness, KV
resizing, multi-node recovery, request continuity, or migration performance.
Those parts of asymmetric PP restoration still need end-to-end validation.

## Full Qwen3-32B recovery investigation

The real layerwise harness now accepts `--gpu-pp` and
`--cold-layer-counts`, and asserts uniform-transition completion before
comparing against a full GPU reference.

An initial CPU TP=2 -> GPU TP=2/PP=2 run used CPU prefix 8 and GPU cold
layer counts 12/44. Both roles returned 1. CPU crashed with
`UnboundLocalError: engine_core_outputs` when prefix execution produced no
sampled output. GPU relay and full reference tokens differed. Logs:
`/tmp/relay32-pp2-{cpu,gpu}.log`.

CPU now initializes the output map and returns frontend termination outputs
even when prefix execution has no sampled output (two lifecycle tests passed).
GPU uniform recovery now imports historical CPU-prefix KV and clears the
old relay message so later forwards cannot reuse first-step hidden states.
A focused takeover test verified the native KV call and restored model range.

KV resizing now uses one slab for all retained layers and rejects missing
uniform layers before mutation. CUDA and CPU checks cover expansion,
shrinkage and strided K/V layouts; a focused regression run passed 21 tests.

A fresh full-model rerun is pending in `/tmp/relay32-pp2-recovery` with logs
`/tmp/relay32-pp2-recovery-{cpu,gpu}.log`. End-to-end PP recovery remains
unproven until that run and subsequent coverage pass.

The corrected full Qwen3-32B run completed: CPU TP=2 -> GPU TP=2/PP=2,
CPU prefix 8, cold GPU layer counts 12/44, 65-token prompt, 40 decode
tokens. `pp_transition_complete=true`; all 40 tokens matched the full GPU
reference. CPU and GPU role processes both returned 0. Logs:
`/tmp/relay32-pp2-recovery-{cpu,gpu}.log`. This proves this single-node
configuration's recovery and request continuation, not arbitrary PP or
multi-node correctness.

The 4161-token rerun also passed with the default 4096-token chunk limit,
CPU TP=2 -> GPU TP=2/PP=2, prefix 8 and cold GPU layer counts 12/44.
`pp_transition_complete=true`; all 40 output tokens matched the full GPU
reference and both role processes returned 0. Logs:
`/tmp/relay32-pp2-chunk2-{cpu,gpu}.log`.

This run covers recovery across a prefill chunk boundary. GPU block tables
can already include blocks for the next chunk when CPU historical KV is
imported. The import now pairs the CPU history with the matching prefix of
the GPU block table, rather than requiring both complete tables to have
equal lengths. Exact output assertions remain in place.

An 8193-token prompt (three 4096-token chunks) with a concurrent five-token
relay request initially exposed a scheduling bug: the first chunk completed,
but the next chunk could be admitted before layerwise PP had completed its
uniform KV/weight recovery. The stale first-batch SHM hidden rows then
produced repeated token 9707. The GPU engine now drains the first batch and
completes recovery before admitting a later chunk. The rerun passed with
both requests producing 40/40 reference tokens, `pp_transition_complete=true`,
and both roles returning 0. Logs:
`/tmp/relay32-pp2-chunk3-fixed-{cpu,gpu}.log`.

## Reverse-direction full-model recovery

On 2026-10-12, the opposite asymmetric assignment was tested with actual
Qwen3-32B BF16: CPU TP2 executes prefix8, GPU TP2/PP2 starts with layer
counts44/12 (ranges8–52 and52–64) then restores uniform ranges0–32 and32–64.
Unlike the earlier12/44 assignment, this requires KV movement from PP0 to
PP1 for layers32–51, along with CPU-prefix takeover. V1, chunk limit4096,
8193-token prompt plus a concurrent five-token request,16 output tokens each.

The full run reported pp_transition_complete=true. Both requests matched
all16 output tokens against complete GPU prefill after recovery in the same
engine; CPU and GPU exited0. No artificial copy delay was injected. Logs:
`/tmp/relay32-pp2-reverse-{cpu,gpu}.log`. This covers a second migration
direction with three prefill chunks and request continuation. It is not an
independently reloaded safetensors reference or a multi-node test.

`tests/verify_qwen_pp_recovery.py` now launches both roles, detects early CPU
failure, enforces a bounded run, rejects stale folders and checks the explicit
transition flag and both token comparisons before writing result.json. It
leaves CPU/GPU logs in the selected folder.13 focused migration/takeover/
resize/chunk-gate tests passed; syntax and diff checks passed.

The new single-command script was itself run successfully on the same real
configuration. Its exit code was0 and result.json records both matching
16-token outputs and completed recovery. Artifacts:
`/tmp/relay32-pp2-recovery-script/{cpu.log,gpu.log,result.json}`.

## Validate cold partitions before model execution

Cold range derivation previously used min/max per stage, silently expanding
lists containing holes and discarding ordering/overlap errors. It now requires
one list per PP stage and a flattened sequence covering one contiguous model
suffix exactly once, with integer in-range IDs. Empty stages preserve the
current global cursor instead of inventing the range0–0. Tokenwise initial
PP execution must start at layer0; layerwise may leave a CPU prefix.
EngineCore validates before mutating relay request state or dispatching the
model-executor RPC. PP1 incremental loading is unaffected.

26 focused tests passed, including11 invalid-partition cases rejected before
RPC/state mutation, tokenwise prefix rejection, empty-stage boundary tests,
and existing migration/resize/takeover/chunk tests. After the production change,
the full Qwen3-32B TP2/PP2 44/12 recovery rerun also passed both16-token
requests, explicit recovery completion and clean CPU/GPU exits. Artifacts:
`/tmp/relay32-pp2-validated-partition/{cpu.log,gpu.log,result.json}`.
Empty-stage range tests do not establish full inference support for empty
PP stages; that requires separate end-to-end testing.
