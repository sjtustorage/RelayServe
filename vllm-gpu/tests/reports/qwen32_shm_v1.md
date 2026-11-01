# Qwen3-32B full V1 SHM inference verification

2026-10-12, local H20 GPUs, BF16, eager, PP=1, max_model_len=256,
max_num_seqs=2, gpu_memory_utilization=0.5. Standard checkpoint:
/workspace/models/Qwen3-32B (64 layers, hidden 5120, intermediate 25600,
64 query heads, 8 KV heads, head dimension 128, vocabulary 151936).

The CPU producer uses TP=2, OMP_NUM_THREADS=2, AMX BF16 and the native
safetensors reader. It reads directly into the final shared weight arenas,
processes weights and stays alive to retain SHM ownership. No reference.pt
reconstruction or converted SLLM checkpoint is used. CPU complete model
loading measured 18.859/18.863 seconds in this run.

GPU full V1 workers initialize, allocate KV caches, prefill the prompt
[151643,9707,11,358,1097], and generate four greedy tokens. Both SHM runs
and the standard GPU loader produce [360,13,358,2776]. Explicit engine-core
shutdown returns exit code 0 for both SHM runs and the TP=4 baseline.

| GPU TP | Standard loader initialization (s) | SHM initialization (s) | Standard 4-token request (s) | SHM 4-token request (s) |
| --- | ---: | ---: | ---: | ---: |
| 2 | 28.144 | 19.513 | 0.284 | 0.280 |
| 4 | 42.373 | 31.915 | 0.433 | 0.494 |

Single observations, not a controlled repeated benchmark. CPU ingestion is
excluded from SHM GPU initialization. These numbers are not cold-start TTFT
and do not establish speed relative to converted SLLM loading. The TP=2
baseline previously emitted an engine-death message at implicit shutdown;
its generated output was correct. TP=4 SHM model-info initialization alone
cost 13.35–13.67 seconds, including mapping/pinning the CPU arenas. This is
an outstanding performance target. Four-token request variation does not
establish a steady-state throughput difference.

Reproduce from the CPU checkout with its Python environment and PYTHONPATH:

```bash
mkdir -p /tmp/qwen32-v1-check
OMP_NUM_THREADS=2 VLLM_CPU_USE_AMX_GEMM_BF16=1 VLLM_USE_V1=1 \
VLLM_CPU_NATIVE_SAFETENSORS=1 PYTHONPATH=$PWD \
/workspace/vllm-cpu/.venv/bin/python -m torch.distributed.run \
--standalone --nproc-per-node 2 \
/root/vllm-gpu/tests/test_qwen_direct_shm.py \
--produce /tmp/qwen32-v1-check --cpu-tp 2 \
--checkpoint /workspace/models/Qwen3-32B --load-only --keep-alive
```

After the ready file appears, from the GPU checkout:

```bash
PYTHONPATH=$PWD /workspace/vllm-gpu/.venv/bin/python \
tests/verify_qwen_shm_v1.py --model /tmp/qwen32-v1-check --tp 2
PYTHONPATH=$PWD /workspace/vllm-gpu/.venv/bin/python \
tests/verify_qwen_shm_v1.py --model /tmp/qwen32-v1-check --tp 4
touch /tmp/qwen32-v1-check/stop
```

Use a fresh producer directory for each run. This verifies weight loading
and full joint GPU inference, not CPU inference or KV handoff. No 72B
checkpoint is available locally. The existing SHM connector unit suite
currently has four failures because it mocks the removed symbol
init_kv_caches_info; two tests pass. It cannot serve as KV handoff evidence.
