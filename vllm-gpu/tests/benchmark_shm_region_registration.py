"""Compare full vs coalesced registration using real native TP copy ranges.

Existing producer objects remain untouched. Timings exclude CUDA startup and
prefault, reported separately. Each sample maps and unregisters independently.
This experimental benchmark does not change the production loading path.
"""
import argparse
import ctypes
import json
import mmap
from pathlib import Path
import time
from unittest.mock import patch

import torch
import shm_manager.torch as shm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("producer", type=Path)
    parser.add_argument("--gpu-tp", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--gap-bytes", type=int, default=1048576)
    parser.add_argument("--bulk-copy", action="store_true",
                        help="Copy the same coalesced ranges in both modes using an 8 MiB GPU buffer")
    args = parser.parse_args()
    if args.rounds < 1 or not 0 <= args.rank < args.gpu_tp or args.gap_bytes < 0:
        parser.error("Invalid rounds, rank, or gap")
    folder = args.producer.resolve()
    config = json.loads((folder / "config.json").read_text())
    manifest = json.loads((folder / "shm_weights.json").read_text())
    torch.cuda.init()
    cudart = torch.cuda.cudart()
    with patch.object(shm, "get_pinned_shared_memory_ptrs",
                      side_effect=lambda names, *_: {rank: (rank + 1) << 44 for rank in names}):
        shm.init_model_info(str(folder), args.rank, args.gpu_tp,
                            manifest["cpu_tp_size"], config["num_attention_heads"],
                            config["num_key_value_heads"], config["head_dim"])
    manager = shm.model_info.manager
    manager.update_manager_info(shm.model_info.cpu_memory_ptrs,
                                {layer: (1 << 48) + (layer << 30)
                                 for layer in shm.model_info.per_layer_gpu_sizes})
    manager.generate_copy_tasks()
    plans = manager.copy_source_ranges(args.gap_bytes)
    manager.close()
    for rank, partial in plans.items():
        source = Path("/dev/shm") / (folder.name + f"-rank_{rank}")
        size = source.stat().st_size
        page = mmap.PAGESIZE
        full = [(0, (size + page - 1) // page * page)]
        for iteration in range(args.rounds):
            order = [("full", full), ("partial", partial)]
            if iteration % 2:
                order.reverse()
            for mode, ranges in order:
                registered = []
                with source.open("r+b") as file, mmap.mmap(file.fileno(), size) as mapping:
                    anchor = ctypes.c_ubyte.from_buffer(mapping)
                    pointer = ctypes.addressof(anchor)
                    try:
                        start = time.perf_counter()
                        # Touch all pages equally for both modes; the small
                        # bytes result is one byte per source page, not weights.
                        mapping[::page]
                        prefault = time.perf_counter() - start
                        start = time.perf_counter()
                        for offset, length in ranges:
                            torch.cuda.check_error(cudart.cudaHostRegister(pointer + offset, length, 0))
                            registered.append(pointer + offset)
                        registration = time.perf_counter() - start
                        copy_seconds = None
                        copied_bytes = 0
                        if args.bulk_copy:
                            buffer = torch.empty(8 * 1024 * 1024, device="cuda", dtype=torch.uint8)
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            for offset, length in partial:
                                remaining = min(length, size - offset)
                                while remaining:
                                    count = min(remaining, buffer.numel())
                                    array = (ctypes.c_ubyte * count).from_address(pointer + offset)
                                    host = torch.frombuffer(array, dtype=torch.uint8)
                                    buffer[:count].copy_(host, non_blocking=True)
                                    copied_bytes += count
                                    offset += count
                                    remaining -= count
                            torch.cuda.synchronize()
                            copy_seconds = time.perf_counter() - start
                            del array, host, buffer
                        # Check first and last readable page in each registered
                        # range, including all disjoint selected source areas.
                        for offset, length in partial:
                            for sample in {offset, min(offset + length - page, size - page)}:
                                array = (ctypes.c_ubyte * page).from_address(pointer + sample)
                                host = torch.frombuffer(array, dtype=torch.uint8)
                                result = host.cuda().cpu()
                                expected = torch.frombuffer(bytearray(mapping[sample:sample + page]),
                                                            dtype=torch.uint8)
                                assert torch.equal(result, expected), (mode, sample)
                                del array, host, result, expected
                        start = time.perf_counter()
                        while registered:
                            torch.cuda.check_error(cudart.cudaHostUnregister(registered[-1]))
                            registered.pop()
                        unregister = time.perf_counter() - start
                        print(json.dumps(dict(round=iteration, mode=mode, cpu_rank=rank,
                            source_bytes=size, registered_bytes=sum(length for _, length in ranges),
                            ranges=len(ranges), prefault_seconds=prefault,
                            register_seconds=registration, sample_pages_exact=True,
                            copy_seconds=copy_seconds, copied_bytes=copied_bytes,
                            unregister_seconds=unregister)), flush=True)
                    finally:
                        for address in reversed(registered):
                            torch.cuda.check_error(cudart.cudaHostUnregister(address))
                        del anchor


if __name__ == "__main__":
    main()
