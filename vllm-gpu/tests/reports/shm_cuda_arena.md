# SHM CUDA weight arena

The production SHM loader now allocates one arena per requested storage range
rather than calling cudaMalloc once per layer. Each layer starts at a 256-byte
aligned address. Copy tasks, per-layer readiness events and fragment waits during
KV/module initialization retain their existing behavior. The single-range native
API delegates to the same allocator; there is no production per-layer fallback.

The manager and all restored tensors share ownership of the arena. Closing or
deleting the manager drains background transfers and releases its ownership;
surviving tensors remain valid. One surviving tensor retains the entire arena,
which is a deliberate tradeoff: normal model switching drops all tensors together.
Allocation validates device, closed state, nonzero ranges and size/alignment
overflow before allocating. Ownership bases are published atomically.

## Validation

- GPU environment: `/workspace/vllm-gpu/.venv`.
- Native extension rebuilt with `shm_manager/setup.py build_ext --inplace`.
- `tests/verify_shm_weight_lifetime.py`: passed actual CUDA allocation, alignment,
  disjoint ranges, aliases after close, allocation release, invalid ranges,
  closed-manager rejection and background-copy teardown.
- Shared-memory loader unit tests: 6 passed. The run also uncovered missing
  submission-set initialization, fixed separately in commit 8430958. The old
  recopy expectation was updated to assert only new recovery layers are submitted.

## Real Qwen3-32B layerwise relay

CPU TP=2, BF16 checkpoint `/workspace/models/Qwen3-32B`, V1, load_format=shm,
chunked prefill 4096, 8193 varied-text prompt tokens, 40 output tokens, resident
processes and precreated groups. GPU memory utilization 0.5. The verifier checks
relay against full GPU prefill and a subsequent independent safetensors load in
the same GPU PIDs, plus producer SHM survival after unload.

TP=2 previous run: lazy 1.282–1.291 s; weight storage initialization
0.262–0.312 s. Arena run: lazy 0.913–0.921 s; storage initialization
0.00639–0.00640 s. Exact 40-token reference and same-PID checkpoint switching
passed; CPU and GPU processes exited successfully.

Logs: `/tmp/resident-shm-arena32-tp2.log` and
`/tmp/resident-shm-arena32-tp2-artifacts/{cpu,gpu}.log`.

These are individual runs rather than a controlled statistical comparison.
Source attachment before lazy remains separate and has not been optimized here.
Full-device fencing and warmup were retained. These results do not prove
progressive incomplete-weight contraction or asymmetric PP recovery.

TP=4 arena run: lazy 0.565–0.586 s (previous run 0.640–0.643 s).
Exact relay and independently loaded checkpoint references passed, producer SHM
survived unload, same GPU PIDs were reused, and processes exited successfully.
Logs: `/tmp/resident-shm-arena32-tp4.log` and matching `-artifacts` directory.

TP=2 repeat: lazy 0.828–0.847 s; storage initialization 0.0124–0.0131 s.
The same relay/reference/switch assertions passed and processes exited 0.
Logs: `/tmp/resident-shm-arena32-tp2-repeat.log` and matching `-artifacts`.
Both TP=2 arena runs were below one second, but two runs are insufficient to
claim a percentile latency guarantee. Warmup and source attachment still vary.

CPU log inspection confirms the first GPU full-takeover acknowledgment terminated
its active relay request and freed CPU KV cache (64/64 layers ready). This does
not validate the incomplete-weight intermediate contraction case.
