"""Run with torchrun --standalone --nproc-per-node=3 on the CPU environment.

Two live native communicators reuse the rendezvous/group identifier. A subgroup
whose first global rank is nonzero also checks the lifetime broadcast source.
"""
import os
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.distributed as dist

import vllm._custom_ops  # Register the native CPU SHM operators.
from vllm.distributed.device_communicators.cpu_communicator import _CPUSHMDistributed


def main():
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    group = dist.new_group([1, 2], backend='gloo')
    os.environ['VLLM_DIST_IDENT'] = 'verify-reused-rendezvous'
    if rank in (1, 2):
        communicator = SimpleNamespace(
            unique_name='reused-tp-group', ranks=[1, 2], rank=rank - 1,
            rank_in_group=rank - 1, world_size=2,
            cpu_group=group, device_group=group)
        first = _CPUSHMDistributed(communicator)
        second = _CPUSHMDistributed(communicator)
        assert first.group_name != second.group_name
        names = [None, None]
        dist.all_gather_object(names, (first.group_name, second.group_name), group)
        assert names[0] == names[1]
        dist.barrier(group)
        for manager in (first, second):
            for local_rank in range(2):
                assert not Path(f'/dev/shm/{manager.group_name}_{local_rank}').exists()
        for manager in (first, second, first):
            value = torch.full((128,), float(rank))
            manager.all_reduce(value)
            torch.testing.assert_close(value, torch.full_like(value, 3.0))
        if rank == 1:
            print('PASS: unlinked live SHM lifetimes, nonzero subgroup source, native allreduce', flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
