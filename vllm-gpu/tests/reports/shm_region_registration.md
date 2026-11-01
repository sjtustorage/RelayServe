# Partial SHM registration experiment

On 2026-10-12, the GPU native task generator gained `copy_source_ranges`:
page-aligned offset/length intervals grouped by producer rank, with optional
gap coalescing. The diagnostic does not change default registration.
The GPU extension was rebuilt. Two native tests passed, covering duplicate
pages, strided pages, gap merging, and adjacent distinct producer mappings.

The benchmark uses existing Qwen3-32B CPU TP=2 source objects for GPU TP=4.
Each sample creates a fresh mapping, prefaults the full source identically,
registers it fully or using native 1 MiB gap-coalesced ranges, checks source
bytes against H2D results, unregisters, then unmaps. No producer bytes are
written or objects unlinked. Order alternates between rounds.

Full registration: 32,762,802,176 bytes, 1 call. Partial registration:
21,919,707,136 bytes (rank 0), 322 calls. Rank 3 uses CPU shard 1 with
21,919,711,232 registered bytes; the file ends partway through the last page.

## Measurements

Initial rank 0 registration-only run, three samples per mode:

| Metric | Full | Partial |
| --- | --- | --- |
| Register seconds | 1.477–1.529 | 1.032–1.055 |

Rank 0 bulk run, copying the same 21.92 GB coalesced source ranges in both
modes with an 8 MiB reusable GPU destination buffer:

| Metric | Full | Partial |
| --- | --- | --- |
| Register seconds | 1.547–1.923 | 1.201–1.701 |
| Bulk H2D seconds | 0.4109–0.4120 | 0.4105–0.4139 |

Rank 3 bulk run, two samples per mode:

| Metric | Full | Partial |
| --- | --- | --- |
| Register seconds | 1.310–1.323 | 0.938–0.945 |
| Bulk H2D seconds | 0.4053–0.4082 | 0.4056–0.4066 |

Final rank 0 run checks the **same** first/last page of every partial range
in both modes and additionally times unregister:

| Round | Full register | Partial register | Full unregister | Partial unregister |
| --- | --- | --- | --- | --- |
| 0 | 1.768 | 2.428 | 1.127 | 1.246 |
| 1 | 1.658 | 2.372 | 1.118 | 1.200 |
| 2 | 1.626 | 1.086 | 1.101 | 0.764 |

Bulk H2D in this final run: full 0.4118–0.4138 s; partial 0.4082–0.4104 s.
All sampled page values matched. Prefault is reported separately and excluded
from registration duration; it is deliberately a serial common baseline,
not the production parallel prefault implementation.

Partial registration can save time, but the 322-call candidate has not shown
stable improvement. The slower samples prevent selecting it as a default
under the requirement to avoid slower production paths. Its variability is
not attributed to a cause without further evidence. Default complete-source
registration and its cache/lifetime behavior remain in place.

This benchmark measures 1D bulk copies plus sampled correctness. It does
not prove full inference correctness, native 2D copy behavior with partially
registered sources, real four-worker concurrent attach latency, or cache
reuse/failure cleanup under a production partial-registration implementation.

Reproduce:

```
env PYTHONPATH=/root/vllm-gpu /workspace/vllm-gpu/.venv/bin/python \
  tests/benchmark_shm_region_registration.py /tmp/relay32-layerwise-tp4 \
  --gpu-tp 4 --rank 0 --rounds 3 --bulk-copy
```

Raw outputs: `/tmp/shm-regions32-{benchmark,bulk,rank3,final}.log`.
