# Parallel consumer page population before CUDA weight registration

Large SHM producer pages are already resident, but a newly mmap'd GPU consumer
has no page-table entries for them. cudaHostRegister then faults them while
registering the mapping. Large cacheable weight mappings now read one byte per
page with up to eight threads before registration. Threads inherit the caller's
permitted CPU mask. Small sources and noncacheable KV/hidden-state sources keep
the existing registration behavior. No host weight buffer or weight writes are
introduced; cached source reuse bypasses this work.

## Validation (2026-10-12)

The native benchmark used an existing real 32.76 GB Qwen3-32B CPU TP2 SHM shard,
cache retention disabled and a fresh mapping for every sample. Every sample
checks H2D contents on first, middle and last pages against the source. It never
unlinks/writes the producer. Two samples per method:

| Native source attach | First | Second |
| --- | ---: | ---: |
| Direct register | 5.502 s | 3.566 s |
| Parallel population + register | 2.323 s | 2.052 s |

Population itself took 0.197-0.200 s. These are individual measurements, not
latency percentiles. The first samples overlapped another real relay test, so
they include driver contention. A prior standalone runtime diagnostic measured
direct3.570-3.776 s versus population+register1.846-2.103 s. Read-only CUDA
registration was also explored, but is not introduced by this change.

Reproduction: tests/benchmark_shm_registration.py /dev/shm/<existing-source>
with the GPU .venv and repository PYTHONPATH. Native results:
/tmp/gpu-native-prefault-benchmark.log; runtime exploration:
/tmp/shm-register-benchmark.json.

Pinned-weight cache lease reuse, H2D contents, same-name inode replacement,
bounded eviction, active protection and transient mapping tests passed:
/tmp/gpu-prefault-cache-test.log.

Real Qwen3-32B CPU TP2 -> GPU TP2 tokenwise, 8193 varied prompt tokens, chunked
prefill4096 passed exact full-GPU/checkpoint reference outputs, stable PIDs,
producer survival and SHM -> safetensors -> SHM roundtrip. First weight
registration3.634-3.678 s plus population0.229-0.291 s; total initial source
attachment4.704-4.713 s including KV/hidden-state setup. This test overlapped
the native benchmark's registration, so it does not establish a controlled
whole-engine speedup. Lazy init0.9817-0.9820 s, reloadedlazy0.1812-0.1849 s.
CPU preparation11.802-11.896 s with the preceding AMX packing/reader fixes.
Logs: /tmp/resident-prefault32.log and matching-artifacts/{cpu,gpu}.log.

This change does not prove reduced TTFT, unsupported72B inference, asymmetric
PP recovery, or prolonged incomplete layerwise contraction.

CPU TP2 -> GPU TP4 was then run without the large registration benchmark in
parallel. The same8193-token/reference/switch/lifetime checks passed. Weight
population0.200-0.288 s, register2.718-2.774 s, complete firstattach3.445-3.631 s,
lazy init0.6271-0.6404 s. Same-source reloadattach0.669-0.778 s and lazy
init0.2085-0.2096 s. Historical TP4 firstattach5.94-6.29 s was a separate run,
so it is not a controlled statistical comparison. GPU TP4 still registers a
whole CPU TP2 source shard per consumer even when only part of it is copied;
reducing that redundancy needs a separate copy-region/lifetime implementation.
Logs /tmp/resident-prefault32-tp4.log and matching-artifacts/{cpu,gpu}.log.
Native weight alias/offset/arena/async-teardown lifetime checks also passed:
/tmp/gpu-prefault-lifetime.log.
