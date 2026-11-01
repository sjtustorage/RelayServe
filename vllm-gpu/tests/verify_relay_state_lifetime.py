"""Actual CUDA/SHM relay state reset and reattachment in one process."""
import os
from pathlib import Path
from multiprocessing import shared_memory

import torch

from shm_manager import torch as relay
from shm_manager._C import set_pinned_weight_cache_budget, pinned_weight_cache_stats
from test_native_kv_transfer import pack


def main():
    torch.cuda.init()
    model = f'relay-state-lifetime-{os.getpid()}'
    blocks, tokens, dims, heads = 3, 16, 64, 4
    cpu_heads, gpu_heads = 2, 1
    logical = torch.arange(blocks * cpu_heads * tokens * dims).reshape(
        blocks, cpu_heads, tokens, dims).bfloat16()
    packed = torch.stack([
        torch.stack([pack(block, True) for block in logical]),
        torch.stack([pack(block + 100, False) for block in logical]),
    ])
    hidden = torch.arange(2 * 6 * dims).reshape(2, 6, dims).bfloat16()
    segments = []
    names = [f'{model}_kv_cache_rank_1', f'{model}_hidden_states']
    try:
        for name, data in zip(names, [packed, hidden]):
            segment = shared_memory.SharedMemory(
                name=name, create=True, size=data.numel() * data.element_size())
            segments.append(segment)
            segment.buf[:] = data.view(torch.int16).numpy().tobytes()
        baseline_maps = [Path('/proc/self/maps').read_text().count(name)
                         for name in names]
        for cycle in range(6):
            # Verify both immediate detach and bounded registration retention.
            # Change producer contents between generations: retention must
            # reuse a live mapping, never stale KV/hidden values or managers.
            retained = cycle >= 3
            set_pinned_weight_cache_budget(
                sum(((data.numel() * 2 + 4095) // 4096) * 4096
                    for data in (packed, hidden)) if retained else 0)
            logical.add_(1)
            hidden.add_(1)
            packed = torch.stack([
                torch.stack([pack(block, True) for block in logical]),
                torch.stack([pack(block + 100, False) for block in logical]),
            ])
            for segment, data in zip(segments, [packed, hidden]):
                segment.buf[:] = data.view(torch.int16).numpy().tobytes()
            relay.total_layers = 0
            block_bytes = cpu_heads * tokens * dims * 2
            relay.init_shm_kv_cache_info(model, 3, 4, 2, blocks,
                block_bytes, packed.numel() * 2, tokens, dims)
            relay.init_hidden_states_info(model, 6, dims, 3, torch.bfloat16)
            output = torch.full((2, blocks, gpu_heads, tokens, dims), -1.,
                                dtype=torch.bfloat16, device='cuda')
            relay.init_gpu_kv_cache_info(
                {0: [output[0].data_ptr(), output[1].data_ptr()]},
                blocks, gpu_heads * tokens * dims * 2, heads)
            states = torch.full((2, 6, dims), -1., dtype=torch.bfloat16,
                                device='cuda')
            relay.load_hidden_states_to_gpu(6, states[0].data_ptr(),
                states[1].data_ptr(), spans=[(3, 0, 2), (0, 4, 2)])
            kv_manager = relay.kv_cache_info.manager
            hidden_manager = relay.hidden_states_info.manager
            relay.load_kv_cache_to_gpu([2, 0], [0, 2], [0])
            # Reset must drain the async KV operation without an explicit wait.
            relay.reset_relay_info()
            torch.testing.assert_close(output[0, [2, 0]].cpu(),
                logical[[0, 2], 1:2], rtol=0, atol=0)
            torch.testing.assert_close(output[1, [2, 0]].cpu(),
                logical[[0, 2], 1:2] + 100, rtol=0, atol=0)
            torch.testing.assert_close(states[:, :2].cpu(), hidden[:, 3:5],
                                       rtol=0, atol=0)
            torch.testing.assert_close(states[:, 4:6].cpu(), hidden[:, :2],
                                       rtol=0, atol=0)
            assert (states[:, 2:4] == -1).all()
            assert [Path('/proc/self/maps').read_text().count(name)
                    for name in names] == [n + int(retained) for n in baseline_maps]
            assert pinned_weight_cache_stats()['entries'] == (2 if retained else 0)
            for submit in (lambda: kv_manager.load_kv_caches_async([0], [0], [0]),
                           lambda: hidden_manager.load_hidden_state_spans([[0, 0, 1]])):
                try:
                    submit()
                except RuntimeError as exc:
                    assert 'closed' in str(exc)
                else:
                    raise AssertionError('Closed manager accepted transfer')
            relay.reset_relay_info()
            assert all((Path('/dev/shm') / name).exists() for name in names)
        set_pinned_weight_cache_budget(0)
        assert [Path('/proc/self/maps').read_text().count(name)
                for name in names] == baseline_maps
        print('PASS: six KV/hidden reset cycles, mutable cached mappings, '
              'exact TP2->TP4 K/V and hidden spans, closed managers, bounded release')
    finally:
        relay.reset_relay_info()
        set_pinned_weight_cache_budget(0)
        for segment in segments:
            segment.close()
            segment.unlink()


if __name__ == '__main__':
    main()
