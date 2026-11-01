# FA3 real 8B/14B single-GPU initialization

The retained GPU implementation is fd8fab3. Controls execute 4f063cc from
an isolated checkout with its original native SHM binary, not by replacing
the active checkout. Real Qwen3 checkpoints, H20, CPU TP=2, GPU TP=1,
V1 eager SHM lazy loading, chunk4096, and a 128-token prompt are used.
The 8B request assigns seven CPU layers; 14B assigns eight. CPU weights,
GPU contexts and communication groups are prepared before model creation.
All cold/reloaded/warm output tokens match independently loaded original
safetensors. Three fresh pools per configuration each create two instances.
The measurement is model creation start through first token.

| Model/configuration | First lazy median ms | First creation-to-token median ms | Reload lazy median ms | Reload creation-to-token median ms |
| --- | ---: | ---: | ---: | ---: |
| 8B original | 593.1 | 1097.5 | 77.7 | 471.8 |
| 8B retained bounded layerwise | 357.3 | 637.4 | 68.3 | 537.4 |
| 14B original | 622.2 | 953.9 | 115.1 | 692.4 |
| 14B retained bounded layerwise | 352.5 | 927.3 | 99.5 | 804.6 |

These controls verify the first-lazy improvement on smaller real models,
but contradict a blanket claim that reload TTFT improves. They are sequential
small samples rather than randomized pairs, and original 8B slot creation
also varied substantially. They are insufficient for a production planner
calibration or a causal attribution of all reload differences to queue pacing.

A further experiment remembered successful warmup configurations, keyed by
PID/device, model/parallel/compilation config, execution layers, warmup sizes,
KV block size, all-reduce setting, and attention environment. It disabled
architecture pacing on identical reloads while retaining full model warmup.
Logs verified True/False pacing for first/reload. All real outputs matched
original weights, and 30 warmup/history tests passed. Nevertheless, 8B first
and reload TTFT medians were 676.3/612.1 ms; 14B were 911.3/788.6 ms.
There was no established end-to-end advantage over the retained implementation.
The experiment was removed, including its cache and tests. Kernel-unloading
support remains incomplete; no new cache creates an unloading dependency.

CPU `cpu_seconds` in these responses measures the entire layerwise producer
session including feedback waits, not AMX model execution. In the 14B history
run, later eight-layer model calls cost 90–119 ms while session durations were
hundreds of milliseconds. The retained 14B reload0 logged approximately96 ms
execution and818 ms GPU feedback wait. Planner CPU costs must use per-step
`CPU relay compute timing` (maximum across TP ranks), not the session duration.
The resident first-init benchmark now records those per-step measurements.

The current planner still accepts separate first/reload profiles but has a
single initialization cost shared by tokenwise/layerwise candidates. The new
pacing makes initialization potentially mode-dependent. These measurements
must not be substituted blindly into the default planner: calibrate the
actual mode/partition and execution costs and verify decisions against an
adjacent zero-CPU baseline first. No default planner costs were changed.

Per-sample data and source/binary metadata are in
`fa3_small_model_results.json`. Artifact directories are recorded there.
Only128-token TP=1 workloads were measured here; 4096-token small-model
and mixed-architecture model-switch latency remain unverified.
