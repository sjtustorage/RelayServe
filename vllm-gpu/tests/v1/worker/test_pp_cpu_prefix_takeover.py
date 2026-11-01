from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from vllm.v1.worker import gpu_model_runner as runner_module


def test_pp_recovery_imports_cpu_prefix_and_clears_hidden_state_override(monkeypatch):
    copy, sync = Mock(), Mock()
    monkeypatch.setattr(runner_module, 'load_kv_cache_to_gpu', copy)
    monkeypatch.setattr(runner_module, 'load_kv_cache_to_gpu_sync', sync)
    runner = SimpleNamespace(total_layers=64, pp_size=2, pp_rank=0,
        get_used_block_ids=lambda: [20, 21, 22],
        shm_scheduler_output={'shm_computed_layers': 8,
                              'req_to_block_ids': {'r': [10, 11, 12]}},
        requests={'r': SimpleNamespace(block_ids=([20, 21, 22, 23],))},
        start_computing_layer=8, _update_model_layer_range=Mock())
    runner_module.GPUModelRunner.execute_kv_migration(
        runner, [list(range(8, 32)), list(range(32, 64))], 'layerwise')
    copy.assert_called_once_with([20, 21, 22], [10, 11, 12], list(range(8)))
    sync.assert_called_once_with()
    assert runner.shm_scheduler_output is None
    assert runner.start_computing_layer == 0
    runner._update_model_layer_range.assert_called_once_with(0, 32)


def test_downstream_recovery_imports_its_part_of_cpu_prefix(monkeypatch):
    copy, sync = Mock(), Mock()
    monkeypatch.setattr(runner_module, 'load_kv_cache_to_gpu', copy)
    monkeypatch.setattr(runner_module, 'load_kv_cache_to_gpu_sync', sync)
    # Stop immediately at the NCCL migration boundary, after checking the
    # CPU-owned layers were imported into their final downstream owner.
    migration = Mock(side_effect=RuntimeError('migration boundary'))
    monkeypatch.setattr(runner_module, 'migrate_kv_between_pp_ranks', migration)
    runner = SimpleNamespace(total_layers=8, pp_size=2, pp_rank=1, step_cnt=0,
        get_used_block_ids=lambda: [20, 21], shm_scheduler_output=None,
        requests={'r': SimpleNamespace(block_ids=([20, 21, 22],))},
        start_computing_layer=7, _update_model_layer_range=Mock(),
        _build_kv_caches_dict=lambda: {})
    metadata = {'shm_computed_layers': 6, 'req_to_block_ids': {'r': [10, 11]}}
    runner_module.GPUModelRunner.update_shm_scheduler_output(runner, metadata)
    assert runner.shm_scheduler_output is None
    assert runner.start_computing_layer == 7
    runner._update_model_layer_range.assert_not_called()
    with pytest.raises(RuntimeError, match='migration boundary'):
        runner_module.GPUModelRunner.execute_kv_migration(runner, [[6], [7]])
    copy.assert_called_once_with([20, 21], [10, 11], [4, 5])
    sync.assert_called_once_with()
    assert runner._pp_cpu_prefix_metadata is None
    migration.assert_called_once()
