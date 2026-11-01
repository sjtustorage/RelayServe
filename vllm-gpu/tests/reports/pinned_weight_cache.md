# Resident pinned weight registration cache

Weight mappings may retain their CUDA host registration after a drained model
unload. Identity comprises fstat device/inode/file size, page-aligned mapping
size and current CUDA device. Reopening the SHM name and fstat precede every
reuse; a retained mapping keeps an unlinked inode alive, preventing replacement
from matching. Content is read from the same shared mapping, not copied into a
second host weight buffer.

References protect active mappings. Zero-reference entries use least-recently
released eviction under an idle-byte budget. KV and hidden-state calls are
noncacheable. Registry operations are serialized with a native mutex. Model
reset still drains copy/unpack before releasing its mapping reference. The
resident pool sets an 80 GiB per-process idle budget; `--weight-cache-bytes`
changes it. Standalone engines retain release-on-unload by default; set
`VLLM_SHM_WEIGHT_CACHE_BYTES` explicitly to configure retention. Pool shutdown
sets the budget to zero to release idle registrations.

The budget limits idle retention, not active sources required for inference.
Multiple GPU processes can register the same physical host pages. Registration
reuse eliminates repeated work, not the first registration. Retained SHM can
outlive a producer unlink until eviction/shutdown. Mutable KV remains outside
this cache. A truncated live producer mapping is unsupported just as before;
producer ownership must remain valid throughout inference.

Validation: native extension rebuilt; `verify_pinned_weight_cache.py` passed
actual SHM/CUDA tests for lease reuse, H2D contents, same-name inode replacement,
budget eviction, active protection and transient mapping cleanup. Existing
weight lifetime checks and six loader tests passed. Sixty resident protocol,
model, topology and lifecycle tests passed.

Real Qwen3-32B tokenwise CPU TP2 -> GPU TP2, 8193 varied-text tokens,
chunked prefill 4096: SHM -> safetensors -> SHM preserved GPU PIDs, 8193 imported
KV tokens and exact output `[356, 15, 16, 1421]` versus full GPU and independent
checkpoint references. All processes exited 0; producer SHM survived unload.
First attachment 4.889–5.394 s; same-source reattachment 0.503–0.507 s. Lazy
initialization 0.866–0.868 s. Logs show no second weight registration; KV and
hidden-state registers remain (about 0.45–0.47 s and 0.012 s respectively).
Individual-run evidence, not a latency percentile or end-to-end TTFT claim.

Logs: `/tmp/resident-shm-cache32-tp2.log` and matching `-artifacts/{cpu,gpu}.log`.
