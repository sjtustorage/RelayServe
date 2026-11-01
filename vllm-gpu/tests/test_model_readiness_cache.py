"""Real CUDA copies must invalidate previously observed layer readiness."""
import torch
import time

from shm_manager._C import ModelManager


def test_completion_cache_is_invalidated_by_every_resubmission():
    source = torch.empty(1024 * 1024, dtype=torch.bfloat16, pin_memory=True)
    destination = torch.empty_like(source, device='cuda')
    manager = ModelManager(2, 1, 0, 1, torch.cuda.current_device())
    manager.update_total_layers(0)
    manager.update_manager_info({0: source.data_ptr()}, {0: destination.data_ptr()})
    manager.generate_tensor_metadata(0, 0, [source.numel()], [source.numel()],
                                     [0], -1, 0, 0, -1, False, 'norm')
    manager.generate_copy_tasks()
    try:
        assert not manager.check_load_state([0])
        for value in (1, 2, 3, 4):
            source.fill_(value)
            manager.load_layers_async([0])
            manager.synchronize([0])
            assert manager.check_load_state([0])
            for _ in range(8):
                manager.synchronize([0])
                assert manager.check_load_state([0])
            assert torch.equal(destination.cpu(), source)
        # A query can also establish completion before synchronize is called.
        source.fill_(5)
        manager.load_layers_async([0])
        deadline = time.monotonic() + 10
        while not manager.check_load_state([0]):
            assert time.monotonic() < deadline
            time.sleep(.001)
        manager.synchronize([0])
        assert manager.check_load_state([0])
        assert torch.equal(destination.cpu(), source)
        source.fill_(6)
        manager.load_layers_sync([0])
        assert torch.equal(destination.cpu(), source)
    finally:
        manager.close()
