"""Measure checkpoint reading separately from AMX packing and engine startup.

Uses the checkpoint's real shapes and a fresh final SHM arena for one CPU TP
rank. Projections remain separate here; this is an I/O diagnostic, not a relay
inference or fused-model-loader benchmark. Later rounds reuse populated pages.
"""
import argparse
import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from shm_manager.safetensors_reader import load_qwen_regions
from shm_manager.torch import share_cpu_state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--cpu-bind", default="",
                        help="rank-local reader cores, independent of caller affinity")
    args = parser.parse_args()
    if not 0 <= args.rank < args.tp or args.threads < 1 or args.rounds < 1:
        parser.error("require 0 <= rank < tp and positive threads/rounds")
    root = args.checkpoint
    config = SimpleNamespace(**json.loads((root / "config.json").read_text()))
    if config.model_type != "qwen3" or config.tie_word_embeddings:
        parser.error("requires a dense Qwen3 checkpoint with separate embeddings")
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    torch.set_num_threads(args.threads)
    targets = {}
    for name, filename in index.items():
        with safe_open(str(root / filename), framework="pt", device="cpu") as source:
            shape = list(source.get_slice(name).get_shape())
        if "norm" not in name:
            axis = 1 if "o_proj" in name or "down_proj" in name else 0
            if shape[axis] % args.tp:
                parser.error(f"{name}: this diagnostic requires divisible TP dimensions")
            shape[axis] //= args.tp
        targets[name] = torch.empty(shape, dtype=torch.bfloat16)
    arena_id = f"reader-benchmark-{uuid.uuid4().hex}"
    shared, _, shm_path = share_cpu_state_dict(
        targets, f"/tmp/{arena_id}/rank_{args.rank}", alignment=1, copy=False)
    del targets
    try:
        print(json.dumps({"rank": args.rank, "tp": args.tp,
                          "arena_bytes": os.stat(shm_path).st_size}), flush=True)
        for iteration in range(args.rounds):
            start = time.perf_counter()
            load_qwen_regions(root, shared, config, args.rank, args.tp, args.threads,
                              args.cpu_bind)
            print(json.dumps({"round": iteration,
                              "read_seconds": time.perf_counter() - start}), flush=True)
        samples = ("model.layers.0.self_attn.o_proj.weight",
                   f"model.layers.{config.num_hidden_layers - 1}.mlp.down_proj.weight",
                   "model.embed_tokens.weight")
        for name in samples:
            with safe_open(str(root / index[name]), framework="pt", device="cpu") as source:
                axis = 1 if "o_proj" in name or "down_proj" in name else 0
                width = shared[name].shape[axis]
                expected = source.get_tensor(name).narrow(axis, args.rank * width, width)
                exact = torch.equal(shared[name], expected)
                print(json.dumps({"sample_tensor": name, "exact": exact}), flush=True)
                if not exact:
                    raise AssertionError(f"Checkpoint shard mismatch: {name}")
    finally:
        os.unlink(shm_path)


if __name__ == "__main__":
    main()
