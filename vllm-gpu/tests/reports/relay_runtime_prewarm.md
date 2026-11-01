# Retained CUDA runtime preparation

The SHM lazy worker now prepares common CUDA operations before model-specific
initialization: linear/embedding, vocabulary masks, normalization, SiLU and
native sampling operations. Small scratch tensors are discarded. The helper
preserves CPU and device RNG states and caches successful preparation by device
and dtype. Model-specific warmup, attention metadata, backend selection and
graph capture remain required. Copy/unpack readiness events and fragment waits
during KV/module initialization are unchanged.

sllm's resident communication registry calls the same helper at pool creation.
Its default BF16 preparation is reused when a BF16 SHM worker is constructed
in that process. This moves reusable runtime work to pool startup; it does not
eliminate the work or imply faster total startup for a newly launched process.

## Real model experiment, 2026-10-12

Qwen3-32B checkpoint, CPU TP2 to GPU TP4 on H20, V1/BF16/eager, SHM asynchronous
layerwise loading. Prompt contains 8193 tokens, chunked prefill 4096; a second
five-token request is also checked. CPU AMX starts before GPU lazy init and
weight transfer starts during GPU structure construction. Every request emits
40 tokens and is compared with complete GPU prefill in the same engine.

Existing production run without common-operation preparation:
`/tmp/relay32-arch-overlap-v1-gpu.log`, lazy 3.290724 s; model dummy warmup
1.484-1.534 s and sampler warmup 0.484-0.543 s across ranks.

Diagnostic preparation run: `/tmp/relay32-runtime-prewarm-v2-gpu.log`, lazy
1.860712 s; model dummy warmup 0.224-0.267 s and sampler warmup 0.244-0.328 s.
Preparation cost 0.999-1.324 s per rank before lazy init. Both 40-token outputs
matched; GPU exit 0. CPU `/tmp/relay32-runtime-prewarm-v1-cpu.log`, exit 0.
The first diagnostic GPU launch failed before lazy init due to an incorrect
test-only SiLU API name. It was corrected, and only the GPU was relaunched
against the still-live CPU producer. No model initialization/inference ran in
that failed GPU process. The diagnostic worker was removed after integration.

Production integration rerun, with the standard worker and no diagnostic
override: `/tmp/relay32-runtime-prewarm-production-v1-gpu.log`, lazy 1.985563 s;
model initialization RPC 0.804 s, model dummy warmup 0.227-0.242 s and sampler
warmup 0.277-0.354 s. Preparation 0.547-1.478 s per rank before lazy init.
Both requests again matched all 40 reference tokens. GPU and CPU exited 0;
CPU log `/tmp/relay32-runtime-prewarm-production-v1-cpu.log`.

These are individual runs, not paired repeated performance averages. They
demonstrate reducible first-call overhead, not the sub-second lazy target,
paper TTFT, planner accuracy or incomplete-weight progressive CPU contraction.
All decoder weights were ready at the first handoff in the successful run.

## Additional validation

`tests/verify_relay_runtime_warmup.py` executed on real CUDA with BF16 and FP16:
CPU/device RNG states preserved, repeated calls hit the cache, no new live
allocation on repeat. First calls 0.537/0.092 s; repeats about 1 microsecond.
This microbenchmark has no model correctness or TTFT claim.

GPU worker submission/prepared-group unit checks: 16 passed. sllm rank protocol,
topology, request validation and lifecycle checks: 39 passed. Four-rank resident
group validation passed TP2, TP4 and TP2/PP2, exact collective/PP values, three
reuse cycles, mismatched manifest rejection and clean shutdown. Log:
`/tmp/resident-runtime-prewarm-groups-v1.log`.

Resident model commands also passed real 8B -> 14B -> 8B switching in retained
TP2 PIDs after pool runtime preparation, including generation/model rejection,
exact repeated 8B output and clean exit. Log:
`/tmp/resident-runtime-prewarm-models-v1.log`. This path still uses safetensors
and online profiling; its timings are not SHM lazy performance evidence.
