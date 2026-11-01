"""Measure direct vs weight-prefault registration of an existing producer SHM.

Reads the source and copies selected pages to CUDA; never unlinks or writes it.
No cache retention is used so every round measures a fresh consumer mapping.
"""
import argparse
import ctypes
import json
import mmap
import time
from pathlib import Path

import torch
from shm_manager._C import (get_pinned_shared_memory_ptrs,
                            release_pinned_shared_memory,
                            set_pinned_weight_cache_budget)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    path = args.source.resolve()
    if path.parent != Path("/dev/shm") or args.rounds < 1:
        parser.error("source must be an existing /dev/shm object; rounds must be positive")
    size = path.stat().st_size
    if size < 8192:
        parser.error("requires at least two pages")
    torch.cuda.init()
    set_pinned_weight_cache_budget(0)
    with path.open("rb") as file, mmap.mmap(file.fileno(), size, access=mmap.ACCESS_READ) as original:
        for iteration in range(args.rounds):
            for cacheable in (False, True):
                pointer = None
                try:
                    start = time.perf_counter()
                    pointer = get_pinned_shared_memory_ptrs({0: path.name}, {0: size}, cacheable)[0]
                    elapsed = time.perf_counter() - start
                    for offset in (0, (size // 2 // 4096) * 4096, size - 4096):
                        array = (ctypes.c_ubyte * 4096).from_address(pointer + offset)
                        host = torch.frombuffer(array, dtype=torch.uint8)
                        result = host.cuda().cpu()
                        expected = torch.frombuffer(bytearray(original[offset:offset + 4096]),
                                                    dtype=torch.uint8)
                        assert torch.equal(result, expected), f"H2D mismatch at {offset}"
                        del array, host, result, expected
                    print(json.dumps({"round": iteration, "bytes": size,
                                      "mode": "weight-prefault" if cacheable else "direct",
                                      "attach_seconds": elapsed, "sample_pages_exact": True}), flush=True)
                finally:
                    if pointer is not None:
                        release_pinned_shared_memory(pointer, size)


if __name__ == "__main__":
    main()
