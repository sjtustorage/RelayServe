"""Prepare weight-independent CUDA operations in retained GPU processes.

Model warmup still runs for attention metadata, backend selection and graphs.
This only materializes common PyTorch/CUDA kernels before SHM lazy loading.
Scratch storage is released; runtime kernels and BLAS handles stay resident.
"""
from functools import cache
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@cache
def warmup_relay_runtime(device: torch.device,
                         dtype: torch.dtype = torch.bfloat16) -> None:
    if device.type != 'cuda' or device.index is None:
        raise ValueError('Relay runtime warmup requires an explicit CUDA device')
    start = time.perf_counter()
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        get_masked_input_and_mask)
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    with torch.inference_mode(), torch.random.fork_rng(devices=[device.index]):
        x = torch.ones((2, 256), device=device, dtype=dtype)
        weight = torch.ones((256, 256), device=device, dtype=x.dtype)
        torch.nn.functional.linear(x, weight)
        ids = torch.arange(2, device=device, dtype=torch.int64)
        masked, mask = get_masked_input_and_mask(ids, 0, 128, 0, 128, 128)
        torch.nn.functional.embedding(masked, weight).masked_fill_(mask[:, None], 0)
        # The model runner selects last-token hidden states with advanced
        # indexing, and mixed greedy/random sampling uses int64 where. These
        # launch distinct CUDA kernels from embedding masks or floating-point
        # sampler operations; first-use module loading belongs in preparation.
        x[ids]
        torch.where(mask, ids, masked)
        out = torch.empty_like(x)
        norm_weight = torch.ones(256, device=device, dtype=x.dtype)
        ops.rms_norm(out, x, norm_weight, 1e-6)
        ops.fused_add_rms_norm(x, out, norm_weight, 1e-6)
        # CUDA lazy loading is per kernel. Norm warmup does not materialize
        # RoPE or KV writes; loading them during the first model's warmup
        # competes with its asynchronous DMA. These probes own only scratch.
        query = torch.ones((2, 256), device=device, dtype=dtype)
        key = torch.ones((2, 128), device=device, dtype=dtype)
        cos_sin = torch.ones((2, 128), device=device, dtype=dtype)
        ops.rotary_embedding(ids, query, key, 128, cos_sin, True)
        cache_key = torch.empty((1, 32, 1, 128), device=device, dtype=dtype)
        cache_value = torch.empty_like(cache_key)
        scale = torch.ones((), device=device, dtype=torch.float32)
        ops.reshape_and_cache_flash(key.view(2, 1, 128),
                                    key.view(2, 1, 128), cache_key,
                                    cache_value, ids, 'auto', scale, scale)
        torch.ops._C.silu_and_mul(
            torch.empty((2, 128), device=device, dtype=x.dtype), x)
        # Large vocabulary rows use CUDA sort/reduction kernels different from
        # the small-row path. Materialize those too before model admission.
        for vocabulary in (1024, 65536):
            logits = torch.ones((2, vocabulary), device=device,
                                dtype=torch.float32)
            torch.rand_like(logits)
            apply_top_k_top_p(logits, torch.full((2,), 10, device=device),
                              torch.full((2,), .9, device=device))
            probs = logits.softmax(-1)
            probs.log_softmax(-1)
            noise = torch.empty_like(probs).exponential_()
            probs.div_(noise).argmax(-1)
        torch.cuda.synchronize(device)
    logger.info("Relay CUDA runtime prepared on %s (%s) in %.3fs",
                device, dtype, time.perf_counter() - start)
