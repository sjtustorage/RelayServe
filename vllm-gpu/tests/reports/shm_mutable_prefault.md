# Populate mutable relay mappings before CUDA registration

The existing parallel page population now applies to all large relay sources,
including KV and hidden-state SHM, independently of weight-cache eligibility.
It reads and discards one byte per page using up to eight threads inherited
from the worker's CPU affinity. Sources below 64 MiB and workers with fewer
than two available CPUs retain the existing direct registration behavior.
Mutable sources remain transient: no KV contents or mapping registrations are
retained in the weight cache. No producer data is changed or duplicated.

Real Qwen3-32B CPU TP2 -> GPU TP2, physical GPUs 0/4, V1, 8193 prompt tokens,
chunked prefill 4096, 96-token CPU prefix, on 2026-10-12:

| Measurement | Before | After |
| --- | --- | --- |
| KV mapping/register, 4 GiB per rank | 0.436 / 0.459 s | 0.260 / 0.200 s |
| KV parallel population (included above) | 0 s | 0.024 / 0.021 s |
| HTTP create | 2.952 s | 3.023 s |
| HTTP lazy init | 0.744 s | 0.740 s |
| Relay first GPU output | 2.257 s | 2.257 s |
| Warm GPU reference first output | 2.304 s | 2.303 s |

These sequential runs show reduced KV attachment overhead but do not establish
an overall create speedup: weight registration varied from 2.059 s in the
available baseline rank log to 2.225–2.257 s after the change. Baseline logs
were bounded to 200 lines, so one rank's initial weight profiling was lost.
The resident verifier now separately preserves the create log before later
messages evict it. All eight tokens matched full GPU prefill and independent
safetensors loading after switching, PIDs stayed stable, and the producer
survived consumer release. Copy and warmup synchronization was unchanged.

A diagnostic on an existing real 1 GiB KV segment compared direct registration
with population before this change. Three direct/prefault attachment pairs were
1.063/0.078, 0.129/0.065 and 0.241/0.217 s; sampled H2D pages were exact. Fixed
ordering and a large first-sample effect limit this evidence; it is not a
controlled latency distribution. Raw log `/tmp/kv-prefault32-benchmark.log`.

The rebuilt native extension passed cache/lifetime checks, including a new
64 MiB transient segment that crosses the population threshold. All page-start
values remained exact, producer updates after attach were visible to CUDA
copies, and the registration never entered the retention cache. Six loader
tests passed. Runtime artifacts are `/tmp/resident-attach-profile32-artifacts/`
and `/tmp/resident-kv-prefault32-artifacts/`; native log is
`/tmp/kv-prefault-lifetime.log`.

Real CPU TP2 -> GPU TP4 layerwise, CPU prefix 8 layers, the same 8193-token
prompt and physical GPUs 0/1/4/5 also passed all eight tokens against both
references. HTTP create was 3.709 s, lazy init 0.464 s, relay first GPU output
1.195 s and warm GPU reference 1.246 s. The four KV attachments took
0.287–0.326 s including population, while initial weight registration took
2.816–2.854 s per process. PIDs and producer lifetime checks passed and all
processes exited 0. Artifacts:
`/tmp/resident-kv-prefault32-layer-tp4-artifacts/`.
