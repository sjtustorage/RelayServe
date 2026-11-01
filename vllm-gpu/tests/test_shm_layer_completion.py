"""Native H20 copy/readiness regression, using the built SHM extension."""
import pytest
import subprocess
import sys
import torch
from shm_manager import _C


def test_background_error_wakes_waiters_without_exiting_worker():
    subprocess.run([sys.executable, '-c', r'''
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from shm_manager import _C
torch.cuda.init()
manager = _C.ModelManager(2, 2, 0, 1, 0)
manager.update_total_layers(2)
os.environ['VLLM_SHM_COPY_THROTTLE_TASKS'] = 'invalid'
def wait(layer):
    try:
        manager.synchronize([layer])
    except (RuntimeError, ValueError) as exc:
        assert 'positive integer' in str(exc), str(exc)
        return
    raise AssertionError('Failed weights reported ready')
with ThreadPoolExecutor(max_workers=2) as pool:
    waiters = [pool.submit(wait, layer) for layer in (0, 1)]
    manager.load_layers_async([0])
    for waiter in waiters:
        waiter.result(timeout=10)
for operation in (lambda: manager.check_load_state([0]),
                  lambda: manager.load_layers_async([1]),
                  manager.completed_layer_load_seconds,
                  manager.completed_layer_load_records,
                  manager.close):
    try:
        operation()
    except (RuntimeError, ValueError) as exc:
        assert 'positive integer' in str(exc), str(exc)
    else:
        raise AssertionError('Background failure was hidden')
del manager
os.environ.pop('VLLM_SHM_COPY_THROTTLE_TASKS')
replacement = _C.ModelManager(2, 2, 0, 1, 0)
replacement.update_total_layers(0)
replacement.load_layers_async([0])
replacement.synchronize([0])
assert replacement.check_load_state([0])
replacement.close()
del replacement
torch.cuda.synchronize()
print('background failure propagated; process and CUDA context alive')
'''], check=True, timeout=30)


@pytest.mark.parametrize("gpu_tp", [2, 4])
@pytest.mark.parametrize("sharding_dim", [0, 1])
def test_tp_copy_completion_and_unsubmitted_layers(gpu_tp, sharding_dim):
    # Production TP workers each own one CUDA device. Keep that process model
    # here too; loading several device contexts serially changes the test.
    for rank in range(gpu_tp):
        subprocess.run([sys.executable, __file__, str(gpu_tp), str(rank),
                        str(sharding_dim)],
                       check=True, timeout=60)


def verify_rank(gpu_tp, rank, sharding_dim):
    torch.cuda.init()
    cpu_tp = 2
    full = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128).to(torch.bfloat16)
    host = []
    for part in full.chunk(cpu_tp, dim=sharding_dim):
        n, k = part.shape
        # AMX panels: N32, K32, N16 half, K pair, N16 row, BF16 pair.
        part = part.reshape(n // 32, 2, 16, k // 32, 16, 2).permute(
            0, 3, 1, 4, 2, 5)
        host.append(part.contiguous().pin_memory())
    torch.cuda.set_device(rank)
    gpu_shape = [128, 128]
    gpu_shape[sharding_dim] //= gpu_tp
    with torch.cuda.device(rank):
        output = [torch.empty(gpu_shape, dtype=torch.bfloat16,
                              device=f"cuda:{rank}") for _ in range(2)]
    manager = _C.ModelManager(2, cpu_tp, rank, gpu_tp, rank)
    manager.update_total_layers(2)
    manager.update_manager_info(
        {i: tensor.data_ptr() for i, tensor in enumerate(host)},
        {i: tensor.data_ptr() for i, tensor in enumerate(output)})
    for layer in range(2):
        manager.generate_tensor_metadata(
            0, 0, [128, 128], gpu_shape, [0, 1],
            sharding_dim, layer, layer, -1, True, f"layer{layer}")
    manager.generate_copy_tasks()
    # Created but unrecorded CUDA events report success; they must not
    # cause the manager to publish unsubmitted weights as ready.
    assert not manager.check_load_state([0])
    assert not manager.check_load_state([1])
    manager.load_layers_async([1])
    manager.synchronize([1])
    assert manager.check_load_state([1])
    assert not manager.check_load_state([0, 1])
    manager.load_layers_async([0])
    manager.synchronize([0])
    assert manager.check_load_state([0, 1])
    expected = full.chunk(gpu_tp, dim=sharding_dim)[rank]
    for tensor in output:
        torch.testing.assert_close(tensor.cpu(), expected, rtol=0, atol=0)
    del manager


if __name__ == "__main__":
    verify_rank(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
