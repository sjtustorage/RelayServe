"""Collectives must expose requested TP order, including uneven splits."""
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.parallel_state import GroupCoordinator


@pytest.mark.parametrize('dim', [0, 1, -1])
@pytest.mark.parametrize('sizes', [None, [1, 3]])
def test_rank_chunk_round_trip(dim, sizes):
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.world_size = 2
    shape = [4, 4]
    tensor = torch.arange(16).reshape(shape)
    physical = [2, 5]
    logical = [5, 2]
    reordered = group._reorder_rank_chunks(tensor, dim, logical, physical, sizes)
    expected = torch.cat(list(reversed(tensor.split(sizes, dim=dim)
                        if sizes else tensor.chunk(2, dim=dim))), dim=dim)
    assert torch.equal(reordered, expected)
    restored = group._reorder_rank_chunks(reordered, dim, physical, logical,
                                         sizes[::-1] if sizes else None)
    assert torch.equal(restored, tensor)
    assert group._reorder_rank_chunks(tensor, dim, physical, physical, sizes) is tensor


def test_collectives_translate_destination_and_sizes():
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.ranks = [5, 2]
    group.world_size = 2
    physical_tensor = torch.tensor([20, 21, 50, 51])
    calls = []
    group.device_communicator = SimpleNamespace(
        ranks=[2, 5],
        all_gather=lambda value, dim: physical_tensor,
        gather=lambda value, dst, dim: (calls.append(('gather', dst)) or physical_tensor),
        reduce_scatter=lambda value, dim: (calls.append(('rs', value.tolist())) or value[:2]),
        all_gatherv=lambda value, dim, sizes: (calls.append(('agv', sizes)) or torch.tensor([20, 21, 22, 50])),
        reduce_scatterv=lambda value, dim, sizes: calls.append(('rsv', value.tolist(), sizes)),
    )
    value = torch.tensor([50, 51, 20, 21])
    assert group._all_gather_out_place(value, 0).tolist() == value.tolist()
    assert group.gather(value, dst=0, dim=0).tolist() == value.tolist()
    group._reduce_scatter_out_place(value, 0)
    assert group.all_gatherv(value, sizes=[1, 3]).tolist() == [50, 20, 21, 22]
    group.reduce_scatterv(torch.tensor([50, 20, 21, 22]), dim=0, sizes=[1, 3])
    assert calls == [('gather', 1), ('rs', [20, 21, 50, 51]),
                     ('agv', [3, 1]), ('rsv', [20, 21, 22, 50], [3, 1])]
