"""Real CUDA validation of resident runtime preparation and RNG preservation."""
import json
import time

import torch

from vllm.model_executor.warmup.relay_runtime import warmup_relay_runtime


def main():
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.manual_seed(1234)
    for dtype in (torch.bfloat16, torch.float16):
        cpu_state = torch.get_rng_state().clone()
        cuda_state = torch.cuda.get_rng_state(device).clone()
        start = time.perf_counter()
        warmup_relay_runtime(device, dtype)
        first = time.perf_counter() - start
        assert torch.equal(cpu_state, torch.get_rng_state())
        assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))
        allocated = torch.cuda.memory_allocated(device)
        hits = warmup_relay_runtime.cache_info().hits
        start = time.perf_counter()
        warmup_relay_runtime(device, dtype)
        repeated = time.perf_counter() - start
        assert warmup_relay_runtime.cache_info().hits == hits + 1
        assert torch.cuda.memory_allocated(device) == allocated
        assert torch.equal(cpu_state, torch.get_rng_state())
        assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))
        print(json.dumps(dict(dtype=str(dtype), first_seconds=first,
            repeated_seconds=repeated, rng_preserved=True,
            repeated_allocation_bytes=0)), flush=True)


if __name__ == '__main__':
    main()
