"""The last CPU relay output still needs KV import after weight submission."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from vllm.v1.worker import gpu_model_runner as runner_module


@pytest.mark.parametrize('pp_size', [1, 2])
def test_final_weight_submission_does_not_skip_cpu_kv(monkeypatch, pp_size):
    copy = Mock()
    sync = Mock()
    monkeypatch.setattr(runner_module, 'load_kv_cache_to_gpu', copy)
    monkeypatch.setattr(runner_module, 'load_kv_cache_to_gpu_sync', sync)
    runner = SimpleNamespace(
        pp_size=pp_size, pp_rank=0, step_cnt=3,
        load_config=SimpleNamespace(load_format='shm', load_method='layerwise'),
        loaded_layers=set(range(8)), target_layers=set(range(8)),
        _relay_cpu_boundaries={'relay': 1},
        shm_scheduler_output={
            'shm_computed_layers': 0,
            'req_to_block_ids': {'relay': [10, 11, 12]},
            'req_to_num_scheduled_tokens': {'relay': 1},
        },
        requests={'relay': SimpleNamespace(block_ids=([20, 21, 22],))},
    )
    runner_module.GPUModelRunner.recv_shm_kv_cache(runner)
    copy.assert_called_once_with([20, 21, 22], [10, 11, 12], [0])
    sync.assert_called_once_with()


def test_first_cpu_prefix_and_stable_boundary_need_no_transfer(monkeypatch):
    copy = Mock()
    sync = Mock()
    monkeypatch.setattr(runner_module, 'load_kv_cache_to_gpu', copy)
    monkeypatch.setattr(runner_module, 'load_kv_cache_to_gpu_sync', sync)
    runner = SimpleNamespace(
        pp_size=1, pp_rank=0, step_cnt=0,
        load_config=SimpleNamespace(load_format='shm', load_method='layerwise'),
        shm_scheduler_output={'shm_computed_layers': 5,
                              'req_to_block_ids': {'r': [1, 2]}},
        requests={'r': SimpleNamespace(block_ids=([3, 4],))})
    runner_module.GPUModelRunner.recv_shm_kv_cache(runner)
    runner_module.GPUModelRunner.recv_shm_kv_cache(runner)
    copy.assert_not_called()
    sync.assert_not_called()
    runner.shm_scheduler_output['shm_computed_layers'] = 2
    runner_module.GPUModelRunner.recv_shm_kv_cache(runner)
    copy.assert_called_once_with([3, 4], [1, 2], [2, 3, 4])
    sync.assert_called_once_with()
