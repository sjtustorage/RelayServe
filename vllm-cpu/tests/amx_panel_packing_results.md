# AMX panel packing (2026-10-12)

PackWeightNK formerly allocated a full-matrix temporary, gathered all output
panels into it, then copied the entire matrix back from the OpenMP master thread.
Each packed 32-row N panel occupies exactly the original 32-row contiguous
storage region. The new implementation stages one panel per OpenMP thread and
copies it back before processing that thread's next independent panel. Scratch
is O(threads * 32 * K), rather than O(N * K). Original tensor pointers, shapes,
and packed byte order are preserved; the slower full-matrix implementation was
removed. Input CPU device, dimensions, dtype, contiguity and tile divisibility
are validated before writes.

## Real checkpoint microbenchmark

Both baseline and replacement were compiled from this repository using the
same CPU environment/compiler. Qwen3-32B first-layer checkpoint values, CPU
TP2 rank0, 48 threads bound to CPUs 96-143. Timed packing excludes copying the
reference into the destination and correctness comparison. Every result was
byte-identical to PackWeight(original.T.contiguous()).

| TP shard | Baseline first / later | Replacement first / later |
| --- | --- | --- |
| q_proj 4096 x 5120 | 10.006 / 5.993-6.674 ms | 0.768 / 0.168-0.201 ms |
| gate_proj 12800 x 5120 | 26.645 / 26.221-28.635 ms | 1.461 / 0.765-0.923 ms |
| down_proj 5120 x 12800 | 29.033 / 25.249-28.840 ms | 2.349 / 0.824-0.877 ms |

This is a small number of cache-warm repeated matrix measurements, not a
distribution of production TTFT or end-to-end speedup. Baseline logs:
/tmp/pack32-rebuilt-baseline.json; replacement /tmp/pack32-final.json.
Reproduction script: tests/benchmark_amx_weight_packing.py.

## Engine validation

17 tests passed with pytest --noconftest covering packing byte order for
real matrix dimensions, unaligned storage with adjacent sentinels, unchanged
pointer, invalid dimensions rejected before writes, SHM publication after
packing, AMX numerical output and native reader affinity/padding behavior.
The root conftest's unrelated absent vllm.config.model import is bypassed.

Real Qwen3-32B CPU TP2 -> GPU TP4 tokenwise, 8193 varied prompt tokens and
chunked prefill4096: exact full-GPU/checkpoint references, stable GPU PIDs,
producer survival and SHM -> safetensors -> SHM roundtrip all passed. CPU
file reads 10.915-11.150 s, postprocessing 0.676-0.922 s, publication-inclusive
preparation 12.117-12.223 s. Earlier affinity-fixed test preparation was about
20 s and postprocessing 7.650-8.404 s; these are separate engine runs.
First GPU lazy init 0.6151-0.6263 s. This does not prove improved relay TTFT:
the test intentionally computes the full8193-token prefill on CPU, and this
computation still takes about24.3 s.

The first test uses /tmp/cpu-panel-pack-python to explicitly prepend the newly
built repository extension to PYTHONPATH. The environment previously selected
/workspace/vllm-cpu/amx_gemm_bf16's older extension. Production CPU .venv
editable installation was updated successfully to use the tested repository
build; a plain CPU .venv import now resolves /root/vllm-cpu/amx_gemm_bf16's
extension. The same17 tests passed again without the extension PYTHONPATH
override (/tmp/amx-panel-installed-tests.log).
Logs: /tmp/resident-panel32-tp4.log and matching -artifacts/{cpu,gpu}.log,
/tmp/amx-panel-all-tests.log.

Real layerwise Qwen3-32B CPU TP2 -> GPU TP2, 56-layer CPU prefix, 8-layer GPU
suffix, 65 prompt tokens: 40 generated tokens exactly matched both references,
producer survived unload and checkpoint switch passed. Reads 11.353-12.485 s,
postprocessing 0.568-0.746 s, preparation 13.111-13.221 s, firstlazy init
0.9713-0.9717 s. Logs /tmp/resident-panel-layer32.log and matching
-artifacts/{cpu,gpu}.log. This test does not establish prolonged incomplete
layerwise contraction or 72B correctness; the72B checkpoint remains absent.
