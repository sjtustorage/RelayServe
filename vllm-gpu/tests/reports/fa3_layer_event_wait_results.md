# Rejected blocking layer-completion event experiment

Real Qwen3-14B/32B, H20, FA3 forced, V1 eager SHM lazy loading,
CPU TP=2, GPU TP=1/2 respectively, CPU layers8/13, input128, chunk4096.
Two fresh pools per configuration, each first instance followed by a reload.
CPU weights, GPU contexts and communication groups were prepared before the
creation timer. Full warmup and concurrent AMX/DMA/unpack/init were preserved.
Controls and candidates were run sequentially without overlapping jobs.

Candidate sets cudaEventBlockingSync on copy/unpack layer completion events,
retaining unpack timing for the layerwise measured transfer budget. It changes
host wait policy, not readiness boundaries. Native CPU TP=2 to GPU TP=4 rank3
async copy, reset and reattach checks passed. All real cold/reloaded/warm outputs
match independent original safetensors, and metadata marks these runs valid.

| Model/config | First lazy median ms | First creation-to-token median ms | Reload lazy median ms | Reload creation-to-token median ms |
| --- | ---: | ---: | ---: | ---: |
| 14B control | 433.1 | 1017.8 | 196.0 | 969.2 |
| 14B blocking events | 371.0 | 896.1 | 87.6 | 812.1 |
| 32B TP2 control | 293.0 | 918.6 | 144.4 | 813.8 |
| 32B TP2 blocking events | 243.3 | 801.8 | 153.6 | 937.2 |

Despite better first-instance medians and14B reloads,32B reload TTFT regresses.
Two-pool sequential samples with variation in creation and CPU compute cannot
establish causal attribution. The experiment is rejected under the requirement
to retain only verified improvements. Source changes were removed and the
original fd8fab3 native binary restored; no blocking layer event policy remains.
Artifact hashes, raw samples, CPU model steps and paths are preserved in
fa3_layer_event_wait_results.json. No planner costs were changed.
