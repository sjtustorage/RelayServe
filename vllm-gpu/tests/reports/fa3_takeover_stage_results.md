# FA3 real32B TP2 takeover stages

H20, Qwen3-32B, CPU TP2/GPU TP2,13 CPU layers,128 input tokens,
chunk4096, V1 eager lazy SHM, FA3 forced. Two fresh pools each load two
instances; full warmup remains and AMX/transfer/init remain concurrent.
CPU weights, CUDA contexts and communication groups are prepared before
model creation. Every cold/reloaded/warm output matches independent original
safetensors. No CUDA profiler or additional synchronization is introduced.

| Instance | Creation-to-first-token ms | Weight wait at final takeover across TP ranks ms | Final KV import ms |
| --- | ---: | --- | --- |
| Pool0 first | 809.6 | 0.063 / 0.083 | 0.873 / 0.930 |
| Pool0 reload | 769.2 | 48.573 / 58.442 | 0.840 / 0.811 |
| Pool1 first | 793.8 | 0.063 / 0.074 | 0.859 / 0.915 |
| Pool1 reload | 766.3 | 47.133 / 55.050 | 0.894 / 0.844 |

TP readiness queries take1.3–1.5ms; leader feedback publication0.177–0.197ms.
These samples identify incomplete weight copies as the main final-takeover wait
on reload. They do not attribute the earlier model forward latency to FA3.
The CPU protocol requires boundary0 to coincide with all weights ready; it
rejects boundary0 with partial loaded-layer count. Moving this wait after
feedback without updating ownership semantics would violate that invariant.

SLLM_PROFILE_RELAY already enabled host timing in this benchmark. The new
optional [relay takeover timing] record splits readiness RPC, weight readiness,
final KV and feedback without changing execution or copy boundaries. It is
suitable for locating host stages, not precise kernel attribution. Raw samples,
artifact metadata and per-rank stage records are preserved in the JSON.
