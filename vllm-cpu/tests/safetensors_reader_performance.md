# Direct safetensors reader diagnostic (2026-10-12)

The reader now clears only padded vocabulary rows. Previously it zeroed the
entire embedding/head shard before the file reader overwrote all valid rows.
The padding/rank-slice regression uses a checkpoint whose last rank contains
padding, initially NaN-filled destinations, and replicated norm weights.
Its direct invocation passed. Repository pytest collection remains broken by
tests/models/registry.py importing the absent vllm.config.model module; the
test body was executed directly, not claimed as a successful pytest run.

Real Qwen3-32B source checkpoint, CPU TP2 shard size 32,762,800,128 bytes,
four native reader threads per process, two concurrent processes:

| Rank | Fresh SHM read | Populated SHM reread |
| --- | ---: | ---: |
| 0 | 10.657 s | 2.458 s |
| 1 | 10.221 s | 2.487 s |

First-layer o_proj, last-layer down_proj, and embedding shards matched the
checkpoint exactly on both ranks. The benchmark uses separate checkpoint
projections and excludes AMX packing, model construction and CPU worker binding.
It is not a full inference or TTFT benchmark and does not prove the old 31 s
engine read time has been fixed. File cache was not dropped; fresh SHM means
new destination pages, not cold disk contents. Owned arenas were unlinked.

Reproduction: run tests/benchmark_safetensors_shm_reader.py with the CPU .venv,
the checkpoint path, and --rank 0 / --rank 1 in concurrent processes. Reports:
/tmp/reader-shm32-rank0.json and /tmp/reader-shm32-rank1.json.

Earlier single-rank measurements using the original reader: 10.549 s fresh
SHM and 2.337 s populated SHM. There is no controlled speedup claim for the
padding change. Local mapped-copy microbenchmarks were faster than row pread,
but no alternative production loading path was introduced on that evidence.

## Reader affinity fix

The engine binds the OpenMP master thread to a single core. std::async reader
threads inherited that same single-core affinity. Reproducing this with
`taskset -c 96` gave 34.003 s for a fresh SHM destination and 10.706 s for a
populated destination. The four nominal workers were sharing one CPU.

CPUWorker now passes its resolved rank-local CPU binding to the shared loader.
The native reader pins each async thread to one of those allocated cores while
preserving the caller's binding and inherited NUMA memory policy. No additional
host weight buffer or alternate I/O path was added. libnuma is explicitly linked
by the extension instead of relying on another module loading its symbols.

With the caller still bound to CPU 96 and reader cores 96-99, the same checkpoint
diagnostic measured 10.840 s fresh / 5.786 s populated and exact sampled values.
The full relay test was running concurrently with this diagnostic, so these
numbers are evidence for the single-core bottleneck rather than a controlled
benchmark of the speedup distribution.

Real Qwen3-32B CPU TP2 -> GPU TP2 tokenwise, 8193 varied prompt tokens and
chunked prefill 4096 passed exact full-GPU and safetensors reference comparisons,
stable GPU PIDs, producer lifetime, and SHM -> safetensors -> SHM switching.
The CPU ranks read in 14.106 / 11.061 s, packed in 10.994 / 8.046 s, and published
in 25.153 / 25.407 s total including the cross-rank readiness barrier. Historical
engine read times were 32.167 / 31.244 s and preparation about 40.7 s; these are
separate runs, not controlled latency percentiles. First lazy init was
0.9602-0.9605 s; repeated lazy init was 0.5695-0.5700 s. GPU latency has varied
between runs; this CPU fix does not prove reduced GPU initialization or TTFT.

Validation: rebuilt native extension and eight targeted tests passed with
`pytest --noconftest tests/test_direct_shared_weights.py
tests/test_safetensors_regions.py -q`. This bypasses the unrelated broken root
conftest import described above. Tests include padded vocab slices, actual
unaligned region contents with an explicitly rebound reader and unchanged
caller affinity, and rejection of invalid CPU bindings before writes.

Logs: /tmp/reader-shm32-singlecore.json,
/tmp/reader-shm32-fixed-affinity.json, /tmp/cpu-reader-affinity-tests.log,
/tmp/resident-affinity32.log and /tmp/resident-affinity32-artifacts/{cpu,gpu}.log.

The same 8193-token tokenwise test also passed CPU TP2 -> GPU TP4, including
exact references, stable PIDs, SHM lifetime and checkpoint-switch roundtrip.
CPU rank reads were 11.515 / 11.303 s, packing 8.404 / 7.650 s, and total
preparation including publication 19.955 / 20.055 s. First lazy init was
0.7108-0.7122 s, reloaded lazy init 0.2280-0.2291 s. Logs:
/tmp/resident-affinity32-tp4.log and matching -artifacts/{cpu,gpu}.log.

Layerwise CPU TP2 -> GPU TP2 with 56 CPU layers, 8 GPU layers and 65 prompt
tokens also passed the 40-token output references, producer lifetime and
checkpoint switch. CPU reads were 11.613 / 11.524 s, packing 8.250 / 7.713 s,
total preparation 19.916 / 20.027 s. Lazy init 0.9173-0.9235 s. The first
actual readiness boundary was zero, so this confirms completed-loading
consolidation, not prolonged incomplete-loading contraction. Logs:
/tmp/resident-affinity-layer32.log and matching -artifacts/{cpu,gpu}.log.
