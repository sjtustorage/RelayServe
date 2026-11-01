# Native SHM copy-source coverage

Measured 2026-10-12 using the real Qwen3-32B producer manifest at
`/tmp/relay32-layerwise-tp4/shm_weights.json` and its checkpoint config.
The normal Python metadata generator feeds the normal native copy-task
generator. Synthetic page-aligned base addresses avoid source pinning and
weight reads; native staging allocation and CUDA event creation still run.
No source weights are written, detached, or unlinked by the diagnostic.

CPU TP=2, per-rank source segment: 32,762,800,128 bytes.

| GPU TP | Exact touched pages | Exact merged ranges | Logical copied bytes |
| --- | --- | --- | --- |
| 4, ranks 0/2 | 4,017,731 | 20,802 | 16,382,076,928 |
| 4, ranks 1/3 | 4,017,732 | 20,802 | 16,382,076,928 |
| 2, both ranks | 7,998,731 | 1 | 32,762,800,128 |

For GPU TP=2 the source union is the full segment, rounded to
32,762,802,176 bytes. Partial registration cannot reduce its footprint.

With 4096-byte pages, exact registration covers about 16.46 GB. Coalescing
gaps changes the candidate registration coverage for ranks 0/2 as follows:

| Maximum gap merged | Registered coverage, bytes | Registration ranges |
| --- | --- | --- |
| 64 KiB | 16,456,626,176 | 20,802 |
| 256 KiB | 17,762,103,296 | 10,562 |
| 1 MiB | 21,919,707,136 | 322 |

Ranks 1/3 differ by one page. The 1 MiB candidate removes about one third
of the pinned source footprint while avoiding tens of thousands of calls.
Registration duration, H2D throughput, and lifetime/cache correctness have
subsequently been investigated in `shm_region_registration.md`.
The candidate has variable registration duration; production lifetime/cache
correctness has not been established. It remains experimental.

The diagnostic runs only when `VLLM_PROFILE_SHM_ATTACH` is present. Normal
loading retains full-source registration and performs no coverage sorting.
`tests/test_shm_copy_coverage.py` exercises real native task generation with
duplicate contiguous ranges and a strided tensor; it verifies exact page
union and gap coalescing. The GPU extension was rebuilt and this test passed.

Reproduce with the producer segment still matching its manifest:

```
env PYTHONPATH=/root/vllm-gpu /workspace/vllm-gpu/.venv/bin/python \
  tests/measure_shm_copy_coverage.py /tmp/relay32-layerwise-tp4 --gpu-tp 4
```

Raw outputs: `/tmp/shm-coverage32-tp4-v2.log` and
`/tmp/shm-coverage32-tp2.log`. These are structural coverage measurements,
not inference or registration performance benchmarks.
