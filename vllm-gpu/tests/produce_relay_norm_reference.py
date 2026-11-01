"""Create a CUDA fused RMSNorm reference for the isolated CPU benchmark.

Run in the GPU environment with operator captures from a real relay audit.
Both devices must use the same CPU-captured attention output and residual;
using the GPU live attention output would mix attention and norm differences.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _custom_ops as ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.checkpoint / 'model.safetensors.index.json').read_text())
    name = 'model.layers.0.post_attention_layernorm.weight'
    with safe_open(args.checkpoint / index['weight_map'][name], framework='pt') as f:
        weight = f.get_tensor(name).cuda()
    capture = torch.load(args.captures / 'operators-cpu-rank-0.pt', weights_only=True)
    x = capture['self_attn.0'].cuda()
    residual = capture['input'].cuda()
    config = json.loads((args.checkpoint / 'config.json').read_text())
    ops.fused_add_rms_norm(x, residual, weight, config['rms_norm_eps'])
    torch.save(dict(output=x.cpu(), residual=residual.cpu()), args.output)


if __name__ == '__main__':
    main()
