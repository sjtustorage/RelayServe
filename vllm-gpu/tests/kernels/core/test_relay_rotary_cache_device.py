import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding, rotary_cache_device


def test_cpu_metadata_build_preserves_native_gpu_rotary_cache():
    kwargs = dict(head_size=128, rotary_dim=128, max_position_embeddings=4096,
                  base=1000000., is_neox_style=True, dtype=torch.bfloat16)
    with torch.device('cuda'):
        native = RotaryEmbedding(**kwargs)
    with torch.device('cpu'), rotary_cache_device('cuda'):
        relay = RotaryEmbedding(**kwargs)
    assert relay.cos_sin_cache.device.type == 'cuda'
    assert torch.equal(native.cos_sin_cache, relay.cos_sin_cache)
    with torch.device('cpu'):
        ordinary = RotaryEmbedding(**kwargs)
    assert ordinary.cos_sin_cache.device.type == 'cpu'
