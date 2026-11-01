"""Check full CPU-built RoPE caches against the previous einsum expression.

Run in either engine environment with PYTHONPATH pointing to that engine.
"""
from types import MethodType, SimpleNamespace
from unittest.mock import patch
import argparse

import torch

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


def verify():
    torch.set_num_threads(1)
    for positions, dim in ((40960, 128), (4096, 64), (131072, 128)):
        model = SimpleNamespace(base=1000000, rotary_dim=dim,
                                max_position_embeddings=positions)
        model._compute_inv_freq = MethodType(RotaryEmbedding._compute_inv_freq, model)
        inv_freq = model._compute_inv_freq(model.base)
        freqs = torch.einsum('i,j -> ij', torch.arange(positions, dtype=torch.float), inv_freq)
        reference = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        with patch.object(torch, 'einsum', side_effect=AssertionError('Unexpected einsum dispatch')):
            actual = RotaryEmbedding._compute_cos_sin_cache(model)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            assert torch.equal(actual.to(dtype), reference.to(dtype)), (positions, dim, dtype)
    print('PASS: full RoPE caches bit-exact in FP32/BF16/FP16 without einsum dispatch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    with torch.device(args.device):
        verify()
