# FA3 first-step diagnostics and rejected hidden-state stream experiment

H20, real Qwen3 checkpoints, V1 eager SHM lazy loading, CPU TP=2,
128 input tokens, chunk4096. CPU weights, GPU contexts and groups are prepared
before model creation. Primary latency is creation start through first token.
Every instance retains complete model warmup; AMX, weight DMA/unpack and GPU
initialization remain concurrent.

The hidden-state nonblocking stream candidate passed the native default-stream
independence test and exact original-safetensors output verification. It keeps
copy completion before consumption. Nevertheless it was removed: there is no
established end-to-end improvement across these small sequential samples.
The active native binary was restored from the pre-experiment backup.

| Candidate | First lazy ms | First creation-to-token ms | Reload lazy ms | Reload creation-to-token ms |
| --- | --- | --- | --- | --- |
| 14B GPU TP=1, three pools | 431.4 / 418.4 / 365.8 | 885.1 / 871.7 / 921.6 | 84.3 / 113.0 / 83.1 | 835.2 / 962.7 / 784.2 |
| 32B GPU TP=2, two pools | 236.0 / 242.5 | 871.5 / 805.5 | 161.9 / 197.1 | 803.6 / 788.2 |

Earlier retained 14B medians were927.3ms first and804.6ms reload; candidate
medians885.1/835.2ms do not establish a consistent benefit. Earlier32B TP=2
retained medians851.6/786.5ms versus candidate838.5/795.9ms likewise do not
establish a win. These are neither randomized pairs nor planner calibrations.

Raw candidate metadata incorrectly sets latency_measurement_valid=false:
an intermediate benchmark implementation classified every LD_PRELOAD as a
diagnostic. These runs had only normal tcmalloc/libiomp5 preloads and no
profiling. The raw records are preserved unchanged; the benchmark now marks
explicit --diagnostic-profile and actual profiler options as diagnostic while
recording LD_PRELOAD separately. These samples are grounds for rejecting an
unproven candidate, not a precise speedup claim.

Optional first-step cProfile diagnostics now capture one engine.step per model
generation and rank through SLLM_PROFILE_FIRST_STEP. They add no CUDA profiler,
event or synchronization, but their latency is explicitly diagnostic. Normal
execution instantiates no profiler. Result preservation, generation boundaries,
error propagation and normal-path tests pass (52 resident/prefix tests total).

14B calling-thread profiles attribute first-step242ms to hidden import140.8ms
and torch linear71.9ms in one pool; another first step336ms had linears135.7ms.
Reload steps476/472ms include final lazy_load_model_sync105.2/108.4ms and forward
layer calls323–324ms. These Python call durations do not isolate device kernels.
The native CUDA API shim separately observed a hidden-copy stream wait120.9ms
with main-thread CPU120.2ms, and simultaneous background launch delays.
On reload some main/background launches took about30ms wall time with almost
no thread CPU time. This supports inspecting host wait policy and transfer/compute
contention; it does not prove that FA3 kernels are the cause. Long later memcpy
calls in those logs belong to the independent safetensors reference loader.

Artifacts and full metadata are recorded in fa3_first_step_diagnostics.json.
Native shim source/binary live only in /tmp; it is an investigative tool, not
part of the production preload path. No planner default or CPU implementation
was changed by this experiment.
