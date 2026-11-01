"""Run with torchrun --standalone --nproc-per-node=4 (four CUDA devices).

Verifies real NCCL KV migration for non-adjacent ranks in both directions.
This exercises migration only, not model startup or inference recovery.
"""
import os
import argparse

import torch
from vllm.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel,
    destroy_model_parallel, destroy_distributed_environment,
    get_pp_group, get_tp_group)
from vllm.v1.worker.cold_start_transition import (
    compute_migration_plan, migrate_kv_between_pp_ranks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tp', type=int, default=1)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    assert world == 4 * args.tp
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world, rank=rank,
        local_rank=local_rank, distributed_init_method='env://', backend='nccl')
    initialize_model_parallel(tensor_model_parallel_size=args.tp,
                              pipeline_model_parallel_size=4)
    rank = get_pp_group().rank_in_group
    shard_offset = 100 * get_tp_group().rank_in_group
    try:
        uniform = [(i * 4, (i + 1) * 4) for i in range(4)]
        for cold in ([(0, 1), (1, 2), (2, 3), (3, 16)],
                     [(0, 13), (13, 14), (14, 15), (15, 16)]):
            blocks = [1, 3, 5]
            caches = {f'model.layers.{layer}.self_attn':
                      torch.full((2, 8, 2, 32, 64), -1,
                                 device='cuda', dtype=torch.bfloat16)
                      for layer in range(16)}
            for layer in range(*cold[rank]):
                caches[f'model.layers.{layer}.self_attn'][:, blocks] = layer + 1 + shard_offset
            plan = compute_migration_plan(cold, uniform)
            migrate_kv_between_pp_ranks(caches, blocks, plan, rank)
            for layer in range(*uniform[rank]):
                value = caches[f'model.layers.{layer}.self_attn']
                assert torch.all(value[:, blocks] == layer + 1 + shard_offset), (rank, layer)
                assert torch.all(value[:, [0, 2, 4, 6, 7]] == -1), (rank, layer)
            print(f'PASS rank={rank} cold={cold} exact KV and untouched blocks',
                  flush=True)
        # One bad rank must cause every rank to fail before entering recv.
        for failure in ('missing-layer', 'block-mismatch', 'block-out-of-range',
                        'negative-block', 'shape-mismatch', 'dtype-mismatch'):
            bad_caches = dict(caches)
            bad_blocks = list(blocks)
            if rank == 3 and failure == 'missing-layer':
                del bad_caches['model.layers.12.self_attn']
            if rank == 3 and failure == 'block-mismatch':
                bad_blocks = [1, 3]
            if failure == 'block-out-of-range':
                bad_blocks = [8]
            if failure == 'negative-block':
                bad_blocks = [-1]
            if rank == 3 and failure in ('shape-mismatch', 'dtype-mismatch'):
                name = 'model.layers.12.self_attn'
                bad_caches[name] = (caches[name][..., :32] if failure == 'shape-mismatch'
                                    else caches[name].float())
            try:
                migrate_kv_between_pp_ranks(bad_caches, bad_blocks, plan, rank)
            except ValueError:
                print(f'PASS rank={rank} collective rejection={failure}', flush=True)
            else:
                raise AssertionError(failure)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == '__main__':
    main()
