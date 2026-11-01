"""SHM allocation views must preserve aliasing and isolate independent layers."""
from types import SimpleNamespace

import torch

from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                       KVCacheGroupSpec, KVCacheTensor)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_attention_slab_alignment_sharing_and_isolation():
    spec = FullAttentionSpec(block_size=32, num_kv_heads=1, head_size=64,
                             dtype=torch.bfloat16, use_mla=False)
    config = KVCacheConfig(num_blocks=1,
        kv_cache_tensors=[KVCacheTensor(size=258, shared_by=['a', 'alias']),
                          KVCacheTensor(size=516, shared_by=['b'])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=['a', 'alias', 'b'],
                                          kv_cache_spec=spec)])
    runner = SimpleNamespace(device=torch.device('cpu'),
        load_config=SimpleNamespace(load_format='shm'),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        runner_only_attn_layers=set())
    raw = GPUModelRunner._allocate_kv_cache_tensors(runner, config)
    assert raw['a'] is raw['alias']
    assert raw['a'].numel() == 258 and raw['b'].numel() == 516
    assert raw['b'].data_ptr() - raw['a'].data_ptr() == 512
    assert raw['b'].view(torch.bfloat16).numel() == 258
    raw['a'].fill_(7)
    raw['b'].fill_(9)
    assert raw['a'].eq(7).all() and raw['b'].eq(9).all()
    # PP recovery releases extra layers individually; do not retain their
    # allocations through the remaining layers' shared storage.
    runner.parallel_config.pipeline_parallel_size = 2
    separate = GPUModelRunner._allocate_kv_cache_tensors(runner, config)
    assert separate['a'].untyped_storage() != separate['b'].untyped_storage()
