"""Native CUDA regression checks for resident SHM weight ownership.

Run with the GPU environment's Python. Uses actual pinned H2D copies and
cudaMalloc allocations, including teardown with a background submission.
"""
import gc

import torch

from shm_manager._C import ModelManager


def manager():
    return ModelManager(4, 1, 0, 1, torch.cuda.current_device())


def free_memory():
    gc.collect()
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0]


def main():
    torch.cuda.init()
    # PP owns a suffix/subset: unassigned layers have no native events.
    # Destruction must not destroy null handles and poison the next model's
    # first CUDA operation. Exercise repeated sparse manager generations.
    for _ in range(6):
        sparse = manager()
        sparse.update_total_layers(36)
        sparse.close()
        del sparse
        gc.collect()
        torch.zeros(1, device='cuda')
        torch.cuda.synchronize()
    # Exercise the deleter before measuring allocations to exclude startup.
    warm = manager()
    warm_base = warm.allocate_cuda_memory(torch.cuda.current_device(), 4096)
    warm_tensor = warm.restore_tensors({
        'warm': (0, [4], [1], 'torch.float32'),
    }, torch.cuda.current_device(), warm_base)['warm']
    warm_tensor.fill_(7)
    torch.testing.assert_close(warm_tensor.cpu(), torch.full((4,), 7.))
    del warm_tensor, warm
    initial = free_memory()
    size = 32 * 1024 * 1024

    # Aliases at offset zero must not each free the same allocation, and an
    # offset tensor must remain valid after all base tensors and manager die.
    for _ in range(6):
        owner = manager()
        base = owner.allocate_cuda_memory(torch.cuda.current_device(), size)
        tensors = owner.restore_tensors({
            'base': (0, [4], [1], 'torch.float32'),
            'alias': (0, [4], [1], 'torch.float32'),
            'tail': (size - 16, [4], [1], 'torch.float32'),
        }, torch.cuda.current_device(), base)
        tail = tensors.pop('tail')
        tail.fill_(7)
        del tensors, owner
        torch.testing.assert_close(tail.cpu(), torch.full((4,), 7.))
        del tail
    assert initial - free_memory() < size // 2, 'Weight allocation leaked'

    # A manager owning allocations that were never restored must free them.
    owner = manager()
    owner.allocate_cuda_memory(torch.cuda.current_device(), size)
    del owner
    assert initial - free_memory() < size // 2, 'Unrestored allocation leaked'

    # Shared arena: alignment, disjoint storage and aliases surviving close.
    owner = manager()
    bases = owner.allocate_cuda_arena(torch.cuda.current_device(), [17, size, 19])
    assert all(base % 256 == 0 for base in bases)
    assert bases[1] - bases[0] == 256
    assert bases[2] - bases[1] == size
    tensors = [owner.restore_tensors({'value': (0, [4], [1], 'torch.float32')},
               torch.cuda.current_device(), base)['value'] for base in bases]
    for index, tensor in enumerate(tensors):
        tensor.fill_(index + 1)
    del tensor
    owner.close()
    for index, tensor in enumerate(tensors):
        torch.testing.assert_close(tensor.cpu(), torch.full((4,), float(index + 1)))
    del tensor, tensors, owner
    assert initial - free_memory() < size // 2, 'Arena allocation leaked'
    owner = manager()
    for sizes in ([0], [2**64 - 1, 1]):
        try:
            owner.allocate_cuda_arena(torch.cuda.current_device(), sizes)
        except (ValueError, OverflowError):
            pass
        else:
            raise AssertionError('Invalid arena accepted')
    assert owner.allocate_cuda_arena(torch.cuda.current_device(), []) == []
    owner.close()
    try:
        owner.allocate_cuda_arena(torch.cuda.current_device(), [16])
    except RuntimeError:
        pass
    else:
        raise AssertionError('Closed manager accepted allocation')
    del owner

    source = torch.full((size // 4,), 11., pin_memory=True)
    owner = manager()
    owner.update_total_layers(0)
    base = owner.allocate_cuda_memory(torch.cuda.current_device(), size)
    tail = owner.restore_tensors({
        'tail': (size - 16, [4], [1], 'torch.float32'),
    }, torch.cuda.current_device(), base)['tail']
    owner.update_manager_info({0: source.data_ptr()}, {0: base})
    owner.generate_tensor_metadata(0, 0, [size // 4], [size // 4],
                                   [0], -1, 0, 0, -1, False, 'weight')
    owner.generate_copy_tasks()
    assert owner.completed_layer_load_records() == []
    owner.load_layers_async([0])
    owner.synchronize([0])
    records = owner.completed_layer_load_records()
    assert len(records) == 1
    layer, payload, seconds = records[0]
    assert layer == 0 and payload == size and seconds > 0
    assert owner.completed_layer_load_seconds() == [seconds]
    torch.testing.assert_close(tail.cpu(), torch.full((4,), 11.))
    # Submit again to retain coverage of teardown during background loading.
    owner.load_layers_async([0])
    # No explicit layer wait: teardown must fence both transfer streams.
    del owner
    torch.testing.assert_close(tail.cpu(), torch.full((4,), 11.))
    del tail, source
    assert initial - free_memory() < size // 2, 'Async destination leaked'
    print('PASS: aliases, surviving offsets, unrestored allocation, shared arena, '
          'validation, async teardown')


if __name__ == '__main__':
    main()
