# Real Qwen3-32B V1 layerwise relay

Checkpoint: /workspace/models/Qwen3-32B, BF16, real 64-layer model.
CPU TP=2 AMX/native safetensors into SHM; GPU H20 TP=2 or TP=4,
load_format=shm, lazy_load=True, load_async=True, PP=1, eager.
Prompt: 65 tokens. CPU prefix progresses 8 -> 5 -> 2 -> 0.

- CPU TP=2 -> GPU TP=2: 12/12 tokens equal a complete GPU reference.
- CPU TP=2 -> GPU TP=4: 40/40 tokens equal a complete GPU reference;
  decode crosses a 32-token KV block boundary.
- Lifecycle rerun after CPU finish notification: TP=2, 12/12 equal;
  both CPU and GPU command sessions exit 0. CPU hidden-state SHM destroyed.

Logs: /tmp/relay32-zero-fixed-{cpu,gpu}.log,
/tmp/relay32-layerwise-tp4-{cpu,gpu}.log,
/tmp/relay32-lifecycle-{cpu,gpu}.log.

These are correctness observations, not TTFT/performance results. The TP=4
run initially had contention from lingering earlier CPU test processes.
No concurrency, asymmetric PP, or 72B completion is established here.

Fixes exercised: residual fixed capacity offset, remote sampled-token
feedback to CPU worker, historical KV import only for newly acquired layers,
zero-layer handoff without stale hidden-state overrides, clearing completed
CPU step metadata, CPU frontend termination notification.
