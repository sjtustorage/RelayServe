"""Benchmark in-place packing with real checkpoint values and TP shard shapes."""
import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
import amx_gemm_bf16_ext as amx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cpu-bind", default="96-143")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.rank < args.tp or args.rounds < 1:
        parser.error("require valid TP rank and positive rounds")
    root = args.checkpoint
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    amx.init_cpu_threads_env(args.cpu_bind)
    suffixes = ("self_attn.q_proj.weight", "mlp.gate_proj.weight", "mlp.down_proj.weight")
    for suffix in suffixes:
        name = "model.layers.0." + suffix
        with safe_open(str(root / index[name]), framework="pt", device="cpu") as source:
            full = source.get_tensor(name)
            axis = 1 if "down_proj" in suffix else 0
            if full.shape[axis] % args.tp:
                parser.error(f"{name}: TP dimension must divide exactly")
            width = full.shape[axis] // args.tp
            original = full.narrow(axis, args.rank * width, width).contiguous()
            target = original.clone()
            reference = amx.pack_weight(original.t().contiguous()).flatten()
            for iteration in range(args.rounds):
                target.copy_(original)
                start = time.perf_counter()
                amx.pack_weight_nk(target)
                elapsed = time.perf_counter() - start
                exact = torch.equal(target.flatten(), reference)
                print(json.dumps({"tensor": name, "shape": list(target.shape),
                                  "round": iteration, "seconds": elapsed,
                                  "exact": exact}), flush=True)
                if not exact:
                    raise AssertionError(f"Packing mismatch: {name}")


if __name__ == "__main__":
    main()
