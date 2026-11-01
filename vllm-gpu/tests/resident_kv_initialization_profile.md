# Resident KV initialization timing

`VLLM_PROFILE_RELAY=1` splits GPU KV setup into configuration/input-batch
preparation, attention backend construction, tensor allocation/reshape/bind,
and connector/native pointer registration. These are host wall times, with no
new CUDA fence. They do not measure all asynchronous GPU work to completion.
Disabled profiling does not call the new timers.

Real Qwen3-32B CPU TP2 to GPU TP2 (GPUs0/4), prompt8193/chunk4096, automatic
five-layer CPU prefix, pre-registered weights: cold KV setup on each rank
allocated 21,080,571,904 bytes (5026 blocks), with preparation0.50--0.76 ms,
attention0.29--0.37 ms, tensors69.2--75.0 ms, registration27.4--36.7 ms.
Recold tensors65.8--67.0 ms and registration25.3--26.7 ms.

Cold lazy initialization was0.825 s and recold0.233 s. Request completion for
eight tokens was3.616/3.283 s. All eight tokens matched cold/warm/re-cold and
the independent safetensors reference, stable GPU PIDs, verifier exit zero.
The earlier2.516 s cold lazy sample was not reproduced. This run does not
establish a performance improvement or prove the earlier delay was allocation.
It argues against changing allocation based on that isolated sample.

The initial diagnostic run failed because the new log used a CPU-only `rank`
attribute. The corrected log obtains the TP rank; the complete real rerun
above validated that path. The failed GPU pool exited; its CPU producer exited
after its command deadline, and the remaining verifier process was terminated.

Artifacts: `/tmp/resident-kv-init32-retry-artifacts/`; log:
`/tmp/resident-kv-init32-retry.log`.
