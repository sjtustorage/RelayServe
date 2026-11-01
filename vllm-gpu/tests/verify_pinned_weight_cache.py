"""Real CUDA/SHM checks for bounded resident weight registration reuse."""
import ctypes
from multiprocessing import shared_memory
import os

import torch
from shm_manager._C import (get_pinned_shared_memory_ptrs,
                            release_pinned_shared_memory,
                            set_pinned_weight_cache_budget,
                            pinned_weight_cache_stats)


def main():
    torch.cuda.init()
    size = 1024 * 1024
    prefix = 'relay-cache-' + str(os.getpid())
    source = shared_memory.SharedMemory(name=prefix + '-weight', create=True, size=size)
    replacement = None
    try:
        source.buf[0] = 17
        set_pinned_weight_cache_budget(size)
        def acquire(cacheable=True):
            return get_pinned_shared_memory_ptrs({0: '/' + prefix + '/weight'},
                                                {0: size}, cacheable)[0]
        first = acquire()
        second = acquire()
        assert first == second
        release_pinned_shared_memory(first, size)
        assert pinned_weight_cache_stats()['active_bytes'] == size
        release_pinned_shared_memory(second, size)
        assert pinned_weight_cache_stats()['idle_bytes'] == size
        reused = acquire()
        assert reused == first
        assert ctypes.c_ubyte.from_address(reused).value == 17
        # Exercise actual H2D transfer from the retained registration.
        array = (ctypes.c_ubyte * 4).from_address(reused)
        tensor = torch.frombuffer(array, dtype=torch.uint8)
        torch.testing.assert_close(tensor.cuda().cpu(), torch.tensor([17, 0, 0, 0], dtype=torch.uint8))
        del tensor, array
        release_pinned_shared_memory(reused, size)
        source.unlink()
        replacement = shared_memory.SharedMemory(name=prefix + '-weight', create=True, size=size)
        replacement.buf[0] = 31
        current = acquire()
        assert current != first, 'Replaced inode reused stale registration'
        assert ctypes.c_ubyte.from_address(current).value == 31
        release_pinned_shared_memory(current, size)
        assert pinned_weight_cache_stats()['idle_bytes'] == size
        assert pinned_weight_cache_stats()['entries'] == 1, 'Budget failed to evict old inode'
        assert acquire() == current
        set_pinned_weight_cache_budget(0)
        assert pinned_weight_cache_stats()['idle_bytes'] == 0
        assert pinned_weight_cache_stats()['active_bytes'] == size
        assert ctypes.c_ubyte.from_address(current).value == 31
        release_pinned_shared_memory(current, size)
        assert pinned_weight_cache_stats()['entries'] == 0
        # Non-weight mappings never enter the retention cache.
        set_pinned_weight_cache_budget(size)
        transient = acquire(False)
        release_pinned_shared_memory(transient, size)
        assert pinned_weight_cache_stats()['entries'] == 0
        # Cross the parallel population threshold with mutable transient
        # storage. Producer updates remain visible and registration is never
        # retained, even when the weight-cache budget could hold this segment.
        large_size = 64 * 1024 * 1024
        kv = shared_memory.SharedMemory(name=prefix + '-kv', create=True,
                                       size=large_size)
        try:
            pointer = None
            try:
                producer = torch.frombuffer(kv.buf, dtype=torch.uint8)
                producer[::4096] = 23
                set_pinned_weight_cache_budget(large_size)
                pointer = get_pinned_shared_memory_ptrs({0: prefix + '-kv'},
                                                        {0: large_size}, False)[0]
                view = (ctypes.c_ubyte * large_size).from_address(pointer)
                consumer = torch.frombuffer(view, dtype=torch.uint8)
                assert torch.equal(consumer[::4096].cuda().cpu(), producer[::4096])
                producer[::4096] = 41
                assert torch.equal(consumer[::4096].cuda().cpu(), producer[::4096])
                assert pinned_weight_cache_stats()['entries'] == 0
                del consumer, view, producer
            finally:
                if pointer is not None:
                    release_pinned_shared_memory(pointer, large_size)
        finally:
            kv.unlink()
            kv.close()
        print('PASS: lease reuse, H2D, inode replacement, bounded eviction, active retention, transient mapping')
    finally:
        set_pinned_weight_cache_budget(0)
        source.close()
        if replacement is not None:
            replacement.close()
            replacement.unlink()
        else:
            source.unlink()


if __name__ == '__main__':
    main()
