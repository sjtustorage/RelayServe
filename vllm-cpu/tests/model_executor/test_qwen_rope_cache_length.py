"""Configured Qwen RoPE caches retain every usable position exactly."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models import qwen2, qwen3


def config(context=256, batch=4096, multimodal=False, load_format='shm'):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=context,
                                     is_multimodal_model=multimodal),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=batch),
        load_config=SimpleNamespace(load_format=load_format))


@pytest.mark.parametrize('context,batch,required', [
    (256, 4096, 4096), (4104, 4096, 4104), (32768, 4096, 32768),
    (131072, 4096, 131072)])
def test_all_permitted_positions_are_bit_exact(monkeypatch, context, batch, required):
    monkeypatch.setattr(qwen2, 'get_current_vllm_config',
                        lambda: config(context, batch))
    length = qwen2.get_qwen_rope_cache_length(131072, None, None)
    assert length == required

    def cache(size):
        model = SimpleNamespace(base=1000000, rotary_dim=128,
                                max_position_embeddings=size)
        model._compute_inv_freq = MethodType(RotaryEmbedding._compute_inv_freq, model)
        return RotaryEmbedding._compute_cos_sin_cache(model)

    full = cache(131072)
    bounded = cache(length)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        assert torch.equal(bounded.to(dtype), full[:length].to(dtype))


@pytest.mark.parametrize('scaling,dual,multimodal,load_format', [
    ({'rope_type': 'dynamic', 'factor': 2}, None, False, 'shm'),
    ({'rope_type': 'yarn', 'factor': 4}, None, False, 'shm'),
    (None, {}, False, 'shm'), (None, None, True, 'shm'),
    (None, None, False, 'safetensors')])
def test_length_sensitive_position_semantics_are_preserved(
        monkeypatch, scaling, dual, multimodal, load_format):
    monkeypatch.setattr(qwen2, 'get_current_vllm_config',
                        lambda: config(multimodal=multimodal, load_format=load_format))
    assert qwen2.get_qwen_rope_cache_length(131072, scaling, dual) == 131072


@pytest.mark.parametrize('module,attention', [
    (qwen2, qwen2.Qwen2Attention), (qwen3, qwen3.Qwen3Attention)])
def test_real_attention_constructors_use_configured_cache(monkeypatch, module, attention):
    monkeypatch.setattr(qwen2, 'get_current_vllm_config', lambda: config())
    monkeypatch.setattr(module, 'get_tensor_model_parallel_world_size', lambda: 4)
    for name in ('QKVParallelLinear', 'RowParallelLinear', 'Attention', 'RMSNorm'):
        monkeypatch.setattr(module, name, lambda *a, **k: torch.nn.Identity(), raising=False)
    observed = []

    def rope(*args, **kwargs):
        observed.append(kwargs['max_position'])
        return torch.nn.Identity()

    monkeypatch.setattr(module, 'get_rope', rope)
    attention(hidden_size=8192, num_heads=64, num_kv_heads=8,
              max_position=131072, rope_theta=1000000)
    assert observed == [4096]
