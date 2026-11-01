"""Measure native copy-plan page coverage using a published producer manifest.

This runs the real metadata/task generator without reading or pinning source
weights, copying tensors, or allocating model weights. CUDA staging buffers
and events are created by the native generator. Producer segments must still
match the manifest; no producer object is modified or unlinked.
"""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import shm_manager.torch as shm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("producer", type=Path)
    parser.add_argument("--gpu-tp", type=int, required=True)
    args = parser.parse_args()
    folder = args.producer.resolve()
    config = json.loads((folder / "config.json").read_text())
    manifest = json.loads((folder / "shm_weights.json").read_text())
    os.environ["VLLM_PROFILE_SHM_ATTACH"] = "1"
    torch.cuda.set_device(0)

    def source_bases(names, sizes, cacheable):
        print(json.dumps({"source_segment_bytes": sizes}), flush=True)
        # Separated, page-aligned addresses used only for pointer arithmetic.
        return {rank: (rank + 1) << 44 for rank in names}

    with patch.object(shm, "get_pinned_shared_memory_ptrs", side_effect=source_bases):
        for rank in range(args.gpu_tp):
            print(json.dumps({"gpu_tp": args.gpu_tp, "rank": rank}), flush=True)
            shm.init_model_info(str(folder), rank, args.gpu_tp,
                                manifest["cpu_tp_size"], config["num_attention_heads"],
                                config["num_key_value_heads"], config["head_dim"])
            manager = shm.model_info.manager
            manager.update_manager_info(
                shm.model_info.cpu_memory_ptrs,
                {layer: (1 << 48) + (layer << 30)
                 for layer in shm.model_info.per_layer_gpu_sizes})
            manager.generate_copy_tasks()
            manager.close()


if __name__ == "__main__":
    main()
