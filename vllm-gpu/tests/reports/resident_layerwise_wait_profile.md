# Real resident CPU TP2 to GPU TP4 relay wait profile

`VLLM_PROFILE_RELAY=1` enables per-step wall-clock diagnostics for CPU metadata
receive/update, scheduling, executor execution, takeover/feedback, and scheduler
update. Worker diagnostics separately report KV import, hidden-state import,
and weight completion waits. No CUDA synchronization or transfer policy was
added; GPU compute timing uses the existing relay CUDA events.

Qwen3-32B, bf16/V1, CPU TP2 -> GPU TP4 (physical GPUs 0/1/4/5), first eight
layers on CPU, prompt4097, chunk4096, persistent pre-registered sources:

| Measurement | First cold request | After idle expiry |
| --- | --- | --- |
| Create | 0.400 s | 0.239 s |
| Lazy init | 0.627 s | 0.310 s |
| GPU generation start to first output | 1.249 s | 1.404 s |
| Complete request | 2.437 s | 2.131 s |

Warm first output was 0.597 s. The second cold step's retained log shows:

* CPU scheduler metadata receive/update: 0.860–0.861 s across ranks.
* GPU CUDA event compute: 0.5025–0.5027 s.
* Hidden-state import: 0.00406–0.00420 s.
* Initial KV import: 0.000017–0.000021 s (the CPU prefix's final KV is imported
  during takeover; these values must not be described as total KV transfer).
* Required suffix weight wait: 0.000230–0.000254 s.
* Final takeover plus feedback: 0.0092–0.0121 s, including final KV import.

CPU execution/readiness dominates this sample's additional delay. Copy/unpack
events report all layers ready and the next CPU boundary zero. No artificial
contraction is exercised. All eight tokens matched warm and independent
safetensors references after idle expiry; GPU PIDs remained stable. These are
single samples, not proof of paper/planner performance. The bounded logs lost
the first cold step timings; the resident verifier now saves first/second relay
logs before checkpoint-switch messages evict them.

23 relay step/state tests passed; the real process exited zero. Artifacts
`/tmp/resident-relay-profile32-tp4-artifacts/`; log
`/tmp/resident-relay-profile32-tp4.log`.
