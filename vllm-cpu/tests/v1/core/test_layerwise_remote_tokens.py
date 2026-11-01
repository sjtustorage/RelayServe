from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from vllm.v1.core.sched.scheduler import Scheduler

@pytest.mark.parametrize('method,expected', [('layerwise', [[9707]]), ('cpu', []), ('tokenwise', [])])
def test_layerwise_sends_gpu_sampled_token_to_cpu_worker(method, expected):
    scheduler = SimpleNamespace(use_pp=False, connector=None,
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(load_format='shm',load_method=method)))
    request = SimpleNamespace(request_id='r',num_computed_tokens=3,all_token_ids=[1,2,3,9707])
    blocks = Mock()
    blocks.get_block_ids.return_value = None
    result = Scheduler._make_cached_request_data(scheduler,[request],[],{'r':1},{},{'r':blocks})
    assert result.new_token_ids == expected
    assert result.num_computed_tokens == [3]
