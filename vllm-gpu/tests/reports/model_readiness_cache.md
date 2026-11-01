# Cache observed copy/unpack completion

Qwen3 waits for layer weights before every forward, including after loading
has finished. Native ModelManager previously synchronized both CUDA events
on every such call, and readiness polling queried the events repeatedly.
It now remembers completion only after both events succeed. Every new load
reservation invalidates that layer's cache under the readiness mutex before
background submission. Pending submissions still gate first completion.
KV/module warmup fragment waits and staging-buffer dependencies are unchanged.

The rebuilt native extension passed three CUDA unit tests, including new
distinct-value resubmission checks, polling before synchronization, and
blocking/asynchronous load entry points. Existing real AMX-packed SHM tests
passed CPU TP2 to GPU TP2 rank0 and TP4 rank1, with exact CUDA unpacked
weights, reloading, detach/reattach and closed-manager rejection.

An isolated microbenchmark used 64 layers with completed empty copy batches,
then 2000 iterations calling synchronize separately for each layer. Before
rebuild it took 0.421033 s; after rebuild, 0.071584 s. Both include Python
dispatch and cudaSetDevice. This measures repeated wait overhead only;
it does not measure weight transfer, model compute or cold-start TTFT.

Real Qwen3-32B BF16, CPU TP2 to GPU TP4, V1, chunked prefill 4096,
8193-token prompt and planner-selected CPU prefix96 passed all eight tokens
against warm complete GPU prefill and an independently loaded safetensors
checkpoint after switching. HTTP lazy init was 0.456 s, worker first relay
output 1.212 s, and warm full-GPU reference first output 1.245 s. These
single-run timings are not a controlled before/after model performance claim.
PID retention, stale generation rejection and source survival passed.

Artifacts: `/tmp/readiness-cache-{build,h2d,h2d-tp2}.log` and
`/tmp/resident-readiness-cache32-artifacts/`.
