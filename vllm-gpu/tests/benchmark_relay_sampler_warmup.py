"""Measure first/repeated V1 sampler warmup using checkpoint dimensions.

Run in a fresh process (or torchrun for TP). This isolates LM-head GEMM,
NCCL all-gather, and sampling from AMX and native weight transfers. Weights
and hidden states are synthetic; this is not an inference correctness test.
The sampler warmup itself is the production GPUModelRunner method.
"""
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--num-reqs', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--prepare-communication', action='store_true')
    parser.add_argument('--prepare-runtime', action='store_true',
                        help='Run the resident process CUDA prewarm first')
    args = parser.parse_args()
    if args.num_reqs < 1 or args.repeats < 2:
        parser.error('num-reqs must be positive and repeats must be at least 2')
    config = json.loads((args.checkpoint / 'config.json').read_text())
    config = config.get('text_config', config)
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl')
    try:
        if args.prepare_runtime:
            from vllm.model_executor.warmup.relay_runtime import warmup_relay_runtime
            warmup_relay_runtime(device)
        # ParallelLMHead pads vocabulary to a multiple of 64 and TP size.
        vocab = config['vocab_size']
        padded_vocab = ((vocab + 64 * world - 1) // (64 * world)
                        * 64 * world)
        torch.manual_seed(0)
        weight = torch.randn((padded_vocab // world, config['hidden_size']),
                             device=device, dtype=torch.bfloat16)
        hidden = torch.zeros((args.num_reqs, config['hidden_size']),
                             device=device, dtype=torch.bfloat16)
        sampler = Sampler()
        phases = {}

        def timed(name, fn):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            value = fn()
            torch.cuda.synchronize(device)
            phases[name] = time.perf_counter() - start
            return value

        def compute_logits(states, _):
            local = timed('lm_head_seconds', lambda: states @ weight.T)
            if world > 1:
                shards = [torch.empty_like(local) for _ in range(world)]
                timed('gather_seconds', lambda: dist.all_gather(shards, local))
                return torch.cat(shards, dim=-1)[:, :vocab]
            return local[:, :vocab]

        runner = SimpleNamespace(
            device=device, model=SimpleNamespace(compute_logits=compute_logits),
            sampler=lambda **kwargs: timed('sampling_seconds',
                                            lambda: sampler(**kwargs)),
            speculative_config=None)
        if args.prepare_communication and world > 1:
            probe = torch.zeros((1, 1), device=device, dtype=torch.bfloat16)
            gathered = [torch.empty_like(probe) for _ in range(world)]
            timed('prepare_seconds', lambda: dist.all_gather(gathered, probe))
            print(json.dumps(dict(rank=rank, phase='prepare', **phases)),
                  flush=True)
        for iteration in range(args.repeats):
            phases.clear()
            output = timed('total_seconds', lambda:
                           GPUModelRunner._dummy_sampler_run(runner, hidden))
            ids = output.sampled_token_ids
            assert ids.shape == (args.num_reqs, 1)
            assert ((ids >= 0) & (ids < vocab)).all().item()
            print(json.dumps(dict(
                rank=rank, tp=world, iteration=iteration, vocab_size=vocab,
                hidden_size=config['hidden_size'], num_reqs=args.num_reqs,
                gpu=torch.cuda.get_device_name(device),
                backend=sampler.topk_topp_sampler.forward.__name__,
                **phases)), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    with torch.inference_mode():
        main()
