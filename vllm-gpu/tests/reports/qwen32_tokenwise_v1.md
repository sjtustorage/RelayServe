# Qwen3-32B V1 lazy tokenwise relay

2026-10-12, real /workspace/models/Qwen3-32B, CPU TP=2 BF16 AMX,
GPU H20 TP=2 and TP=4, PP=1, load_format=shm, lazy_load=True,
load_async=True, tokenwise, block_size=32, eager. CPU and GPU use their
separate project environments and the edited checkout via PYTHONPATH.

CPU prefill of [151643,9707,11,358,1097] produces token 360 and retains
remote block 1 with num_computed_tokens=5. GPU imports this actual POSIX
shared KV and generates [13,358,2776]. Both GPU TP configurations match a
separate GPU request that recomputes the full prompt. The GPU reports
num_cached_tokens=5. Both processes terminate normally after explicit
GPU shutdown; the producer remains alive until its stop file is created.

Both TP=2 and TP=4 also pass the scenario where an ordinary request is
admitted 0.1 seconds before the designated relay request. Previously the
engine slept inside step(), preventing its own input loop from processing
the later relay. Waiting now occurs in the input loop; aborting the missing
relay releases the gate. Three focused regression tests pass.

KV initialization previously passed the inclusive final weight layer index
63 to a manager expecting a layer count; registering KV layer 63 failed.
The KV manager now receives 64 without changing weight bucket semantics.

The initial sequential tests establish correctness, not relay performance.
An additional concurrent run triggers CPU AMX prefill together with GPU
lazy_init after GPU process/NCCL/SHM preparation. Monotonic timestamps:

- lazy initialization: 6567017.208772081 to 6567019.670650721 (2.462 s)
- CPU request: 6567017.209828082 to 6567018.066451302 (0.857 s)
- measured CPU/lazy overlap: 0.857 s
- subsequent three-token GPU request: 0.245 s
- tokens equal independent full GPU reference; imported KV tokens: 5

Prewarm measured 19.528 s and is EXCLUDED from lazy initialization. This
run uses a newly spawned GPU process, not a process with model-independent
operator modules already warmed. It does not establish the user's <1 s
initialization target or paper TTFT speedup. The first CPU token is already
available before GPU lazy_init completes. Do not add these overlapping
intervals or report the subsequent GPU request as end-to-end TTFT.

Detailed warmup timing in that run: model dummy warmup 0.908/0.913 s,
sampler warmup 0.489/0.497 s, kernel_warmup helper 0 s. Async weight loading
is submitted before initialize_model, and warmup passes warmup=True,
which skips decoder-layer weight waits and executes one decoder layer.
Weight-transfer GPU timestamps have not yet been measured, so complete
three-way overlap remains unverified. Removing warmup without moving its
cost into prewarm would merely move latency to first inference.

Reproduce CPU (fresh directory, CPU checkout):

```bash
OMP_NUM_THREADS=16 VLLM_CPU_OMP_THREADS_BIND='96-111|144-159' \
VLLM_CPU_KVCACHE_SPACE=1 VLLM_CPU_USE_AMX_GEMM_BF16=1 \
VLLM_CPU_NATIVE_SAFETENSORS=1 PYTHONPATH=$PWD \
/workspace/vllm-cpu/.venv/bin/python \
/root/vllm-gpu/tests/verify_qwen_tokenwise_v1.py --role cpu \
--folder /tmp/qwen32-relay-check --checkpoint /workspace/models/Qwen3-32B \
--overlap
```

After cpu-ready.json appears, GPU checkout:

```bash
PYTHONPATH=$PWD /workspace/vllm-gpu/.venv/bin/python \
tests/verify_qwen_tokenwise_v1.py --role gpu \
--folder /tmp/qwen32-relay-check --gpu-tp 2 --overlap
touch /tmp/qwen32-relay-check/stop
```

For gate tests omit --overlap on both sides, wait for handoff.json, and
use --exercise-gate on GPU. TP=4 can then reuse retained CPU weights/KV
until stop. CPU affinity here is a test-host setting, not a portable default.
This does not verify layerwise, PP transitions, multi-request CPU block
reclamation, 72B, or performance against planner formulas.
