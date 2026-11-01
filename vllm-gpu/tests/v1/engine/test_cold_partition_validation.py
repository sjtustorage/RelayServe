from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.engine.core import EngineCore
from vllm.v1.worker.cold_start_transition import compute_pp_layer_ranges


@pytest.mark.parametrize('partition', [[], [[0, 1]], [[0, 1], [3]],
    [[0, 1, 2], [2, 3]], [[1, 0], [2, 3]], [[0, 1], [2]],
    [[-1, 0, 1], [2, 3]], [[0, 1], [2, 4]], [[False, 1], [2, 3]],
    [[], []], [None, [0, 1, 2, 3]]])
def test_invalid_partition_rejected_before_mutation_or_rpc(partition):
    core = SimpleNamespace(total_layers=4, _cold_start_partition=None,
        cold_start_request_id='unchanged', model_executor=Mock(),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=2),
            load_config=SimpleNamespace(load_format='shm', load_method='layerwise')))
    with pytest.raises(ValueError):
        EngineCore.lazy_load_model(core, partition, request_id='new')
    assert core.cold_start_request_id == 'unchanged'
    core.model_executor.lazy_load_model.assert_not_called()


def test_suffix_and_empty_stages_preserve_global_boundaries():
    assert compute_pp_layer_ranges(8, 3, [[2, 3], [], [4, 5, 6, 7]]) == [(2, 4), (4, 4), (4, 8)]
    assert compute_pp_layer_ranges(8, 2, [[], list(range(8))]) == [(0, 0), (0, 8)]
    assert compute_pp_layer_ranges(64, 2, [list(range(8, 52)), list(range(52, 64))]) == [(8, 52), (52, 64)]


def test_tokenwise_rejects_missing_cpu_prefix_before_rpc():
    core = SimpleNamespace(total_layers=4, _cold_start_partition=None,
        cold_start_request_id='unchanged', model_executor=Mock(),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=2),
            load_config=SimpleNamespace(load_format='shm', load_method='tokenwise')))
    with pytest.raises(ValueError, match='every model layer'):
        EngineCore.lazy_load_model(core, [[1, 2], [3]])
    core.model_executor.lazy_load_model.assert_not_called()
