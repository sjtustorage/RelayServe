"""Verify native AMX cache conversion, TP aggregation and replicated KV heads."""
import torch
import argparse
import time
from shm_manager._C import KVCacheManager


def pack(logical, key):
    heads, tokens, dims = logical.shape
    packed = torch.empty_like(logical)
    for token in range(tokens):
        for dim in range(dims):
            if key:
                index = ((token // 16 * 16 * (dims // 2) + token % 16
                          + dim // 2 * 16) * 2 + dim % 2)
            else:
                index = (token // 2 * 32 + token % 2
                         + dim // 16 * tokens * 16 + dim % 16 * 2)
            packed.view(heads, -1)[:, index] = logical[:, token, dim]
    return packed


def check_cpu_tp_fixture(fixture):
    """Check actual CPU-written shards, including TP contraction to one rank."""
    cpu_tp = fixture['cpu_tp']
    hosts = [shard.pin_memory() for shard in fixture['packed_shards']]
    _, blocks, heads, tokens, dims = fixture['reference'].shape
    for gpu_tp in (1, 2, 4):
        assert heads % gpu_tp == 0
        assert max(gpu_tp, cpu_tp) % min(gpu_tp, cpu_tp) == 0
        gpu_heads = heads // gpu_tp
        for rank in range(gpu_tp):
            if gpu_tp < cpu_tp:
                ratio = cpu_tp // gpu_tp
                source_ranks = list(range(rank * ratio, (rank + 1) * ratio))
            else:
                source_ranks = [rank // (gpu_tp // cpu_tp)]
            manager = KVCacheManager(
                1, rank, gpu_tp, cpu_tp, blocks,
                heads // cpu_tp * tokens * dims * 2, tokens, dims,
                torch.cuda.current_device())
            manager.update_shm_ptr_info(
                source_ranks, {r: hosts[r].data_ptr() for r in source_ranks})
            output = torch.full((2, blocks + 1, gpu_heads, tokens, dims), -1,
                                dtype=torch.bfloat16, device='cuda')
            manager.update_gpu_ptr_info(
                blocks + 1, gpu_heads * tokens * dims * 2, heads,
                {0: [output[0].data_ptr(), output[1].data_ptr()]})
            # Reverse page mapping catches assumptions about adjacent IDs.
            destinations = list(reversed(range(blocks)))
            manager.load_kv_caches_async(destinations, list(range(blocks)), [0])
            manager.synchronize()
            expected = fixture['reference'][:, :,
                rank * gpu_heads:(rank + 1) * gpu_heads]
            torch.testing.assert_close(output[:, destinations].cpu(), expected,
                                       rtol=0, atol=0)
            assert (output[:, blocks] == -1).all()
            print(f'PASS: actual CPU TP={cpu_tp} -> GPU TP={gpu_tp} '
                  f'rank={rank}, {blocks} pages, exact BF16 K/V')


def run(cpu_tp, gpu_tp, gpu_rank, total_heads):
    tokens, dims, blocks = 16, 64, 3
    cpu_heads = total_heads // cpu_tp
    gpu_heads = max(1, total_heads // gpu_tp)
    source = torch.arange(blocks * total_heads * tokens * dims).reshape(
        blocks, total_heads, tokens, dims).to(torch.bfloat16)
    ratio = max(1, cpu_tp // gpu_tp)
    ranks = (list(range(gpu_rank * ratio, (gpu_rank + 1) * ratio)) if cpu_tp > gpu_tp
             else [gpu_rank // (gpu_tp // cpu_tp)])
    hosts = {}
    for rank in ranks:
        shard = source[:, rank * cpu_heads:(rank + 1) * cpu_heads]
        hosts[rank] = torch.stack([torch.stack([pack(block, True) for block in shard]),
                                   torch.stack([pack(block + 100, False) for block in shard])]).pin_memory()
    device = torch.cuda.current_device()
    manager = KVCacheManager(1, gpu_rank, gpu_tp, cpu_tp, blocks,
                             cpu_heads * tokens * dims * 2, tokens, dims, device)
    manager.update_shm_ptr_info(ranks, {rank: tensor.data_ptr() for rank, tensor in hosts.items()})
    output = torch.full((2, blocks, gpu_heads, tokens, dims), -1,
                        dtype=torch.bfloat16, device="cuda")
    manager.update_gpu_ptr_info(blocks, gpu_heads * tokens * dims * 2, total_heads,
                                {0: [output[0].data_ptr(), output[1].data_ptr()]})
    manager.load_kv_caches_async([2, 0], [0, 2], [0])
    for _ in range(16):
        manager.load_kv_caches_async([2, 0], [0, 2], [0])
    deadline = time.monotonic() + 30
    while not manager.check_load_state():
        if time.monotonic() >= deadline:
            raise AssertionError("Completed KV transfers never reported readiness")
        time.sleep(0.001)
    first = gpu_rank // max(1, gpu_tp // total_heads) * gpu_heads
    expected = source[[0, 2], first:first + gpu_heads]
    torch.testing.assert_close(output[0, [2, 0]].cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(output[1, [2, 0]].cpu(), expected + 100, rtol=0, atol=0)
    assert (output[:, 1] == -1).all()
    for destination, source_blocks, layers in [([3], [0], [0]),
                                                ([0], [0, 1], [0]),
                                                ([0, 0], [0, 1], [0]),
                                                ([0], [0], [1])]:
        manager.load_kv_caches_async(destination, source_blocks, layers)
        try:
            manager.synchronize()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid transfer did not propagate its error")
    assert manager.check_load_state()
    # Polling completion must surface worker failures just as synchronize does.
    manager.load_kv_caches_async([blocks], [0], [0])
    deadline = time.monotonic() + 30
    while True:
        try:
            ready = manager.check_load_state()
        except ValueError:
            break
        if ready:
            raise AssertionError("Polling discarded a failed transfer")
        if time.monotonic() >= deadline:
            raise AssertionError("Failed KV transfer never completed")
        time.sleep(0.001)
    assert manager.check_load_state()
    # Register a sparse PP/progressive allocation and then remove it. Missing
    # layers must never acquire null pointers through map operator[].
    sparse = KVCacheManager(3, gpu_rank, gpu_tp, cpu_tp, blocks,
                            cpu_heads * tokens * dims * 2, tokens, dims, device)
    sparse.update_gpu_ptr_info(blocks, gpu_heads * tokens * dims * 2, total_heads,
                               {0: [output[0].data_ptr(), output[1].data_ptr()],
                                2: [output[0].data_ptr(), output[1].data_ptr()]})
    sparse.load_kv_caches_async([0], [0], [1])
    try:
        sparse.synchronize()
    except ValueError:
        pass
    else:
        raise AssertionError("Sparse allocation registered a missing layer")
    sparse.update_gpu_ptr_info(blocks, gpu_heads * tokens * dims * 2, total_heads, {})
    sparse.load_kv_caches_async([0], [0], [0])
    try:
        sparse.synchronize()
    except ValueError:
        pass
    else:
        raise AssertionError("Removed layer retained stale cache pointers")
    print(f"PASS: CPU TP={cpu_tp}, GPU TP={gpu_tp}, rank={gpu_rank}, KV heads={total_heads}")


def check_sparse_multilayer_offsets():
    """Exercise both layer planes and nonadjacent pages after re-registration."""
    layers, blocks, heads, tokens, dims = 3, 4, 8, 16, 64
    logical = torch.randn(layers, 2, blocks, heads, tokens, dims,
                          dtype=torch.bfloat16)
    hosts = {}
    for rank in range(2):
        planes = []
        for layer in range(layers):
            for plane in range(2):
                planes.append(torch.stack([
                    pack(logical[layer, plane, block, rank * 4:(rank + 1) * 4],
                         key=plane == 0) for block in range(blocks)]))
        hosts[rank] = torch.stack(planes).pin_memory()
    for rank in range(4):
        manager = KVCacheManager(layers, rank, 4, 2, blocks,
                                 4 * tokens * dims * 2, tokens, dims,
                                 torch.cuda.current_device())
        source_rank = rank // 2
        manager.update_shm_ptr_info([source_rank],
                                   {source_rank: hosts[source_rank].data_ptr()})
        # Replacing the allocation must replace every saved base pointer.
        for _ in range(2):
            output = torch.full((layers, 2, blocks + 1, 2, tokens, dims), -1,
                                dtype=torch.bfloat16, device='cuda')
            manager.update_gpu_ptr_info(blocks + 1, 2 * tokens * dims * 2, heads,
                {layer: [output[layer, 0].data_ptr(), output[layer, 1].data_ptr()]
                 for layer in (0, 2)})
            manager.load_kv_caches_async([4, 1], [1, 3], [2, 0])
            manager.synchronize()
            for layer in (0, 2):
                for destination, source in ((4, 1), (1, 3)):
                    torch.testing.assert_close(
                        output[layer, :, destination].cpu(),
                        logical[layer, :, source, rank * 2:(rank + 1) * 2],
                        rtol=0, atol=0)
                assert (output[layer, :, [0, 2, 3]] == -1).all()
            assert (output[1] == -1).all()
        manager.close()
    print('PASS: sparse multilayer KV offsets and allocation replacement, TP2 -> TP4')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-fixture")
    args = parser.parse_args()
    if args.cpu_fixture:
        fixture = torch.load(args.cpu_fixture, weights_only=True)
        if fixture.get('cpu_tp', 1) > 1:
            check_cpu_tp_fixture(fixture)
        packed = fixture["packed"].pin_memory()
        _, blocks, heads, tokens, dims = packed.shape
        output = torch.empty_like(packed, device="cuda")
        manager = KVCacheManager(1, 0, 1, 1, blocks, heads * tokens * dims * 2,
                                 tokens, dims, torch.cuda.current_device())
        manager.update_shm_ptr_info([0], {0: packed.data_ptr()})
        manager.update_gpu_ptr_info(blocks, heads * tokens * dims * 2, heads,
                                    {0: [output[0].data_ptr(), output[1].data_ptr()]})
        manager.load_kv_caches_async(list(range(blocks)), list(range(blocks)), [0])
        manager.synchronize()
        torch.testing.assert_close(output.cpu(), fixture["reference"], rtol=0, atol=0)
        query = fixture["query"].cuda().transpose(0, 1).unsqueeze(0)
        key = output[0].transpose(0, 1).reshape(heads, blocks * tokens, dims).unsqueeze(0)
        value = output[1].transpose(0, 1).reshape(heads, blocks * tokens, dims).unsqueeze(0)
        attention = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        attention = attention.squeeze(0).transpose(0, 1).cpu()
        torch.testing.assert_close(attention.float(), fixture["cpu_attention"].float(),
                                   rtol=0.03, atol=0.01)
        print("PASS: actual CPU AMX cache writer to native GPU unpack")
        print("PASS: GPU attention on transferred cache matches CPU AMX attention")
        from types import SimpleNamespace
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        runner = SimpleNamespace(load_config=SimpleNamespace(load_format="shm"))
        order = GPUModelRunner._get_kv_cache_stride_order(runner, FlashAttentionBackend)
        assert order == (0, 1, 3, 2, 4)
        logical_shape = FlashAttentionBackend.get_kv_cache_shape(blocks, tokens, heads, dims)
        backend_cache = output.reshape(tuple(logical_shape[i] for i in order)).permute(
            *[order.index(i) for i in range(len(order))])
        backend = FlashAttentionImpl(heads, dims, dims ** -0.5, heads, None, None, "auto")
        metadata = FlashAttentionMetadata(
            num_actual_tokens=1, max_query_len=1,
            query_start_loc=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            max_seq_len=blocks * tokens,
            seq_lens=torch.tensor([blocks * tokens], device="cuda", dtype=torch.int32),
            block_table=torch.arange(blocks, device="cuda", dtype=torch.int32).view(1, -1),
            slot_mapping=torch.empty(0, device="cuda", dtype=torch.int64),
            use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None,
            prefix_kv_lens=None, suffix_kv_lens=None)
        scale = torch.ones((), device="cuda")
        layer = SimpleNamespace(_q_scale=scale, _k_scale=scale, _v_scale=scale)
        flash_output = torch.empty_like(fixture["query"], device="cuda")
        backend.forward(layer, fixture["query"].cuda(), None, None,
                        backend_cache, metadata, output=flash_output)
        torch.testing.assert_close(flash_output.cpu().float(), fixture["cpu_attention"].float(),
                                   rtol=0.03, atol=0.01)
        print("PASS: vLLM FlashAttention backend reads transferred HND cache")
        # Continue decode in a fresh page, exercising the backend's actual
        # reshape_and_cache_flash writer on the same HND stride arrangement.
        extended = torch.zeros((2, blocks + 1, heads, tokens, dims),
                               dtype=output.dtype, device="cuda")
        extended[:, :blocks].copy_(output)
        extended_cache = extended.transpose(2, 3)
        logical_key = key.clone()
        logical_value = value.clone()
        generator = torch.Generator(device="cuda").manual_seed(87)
        for step in range(3):
            new_query = torch.randn((1, heads, dims), generator=generator,
                                    device="cuda", dtype=torch.bfloat16)
            new_key = torch.randn_like(new_query)
            new_value = torch.randn_like(new_query)
            length = blocks * tokens + step + 1
            metadata.max_seq_len = length
            metadata.seq_lens.fill_(length)
            metadata.block_table = torch.arange(blocks + 1, device="cuda",
                                                dtype=torch.int32).view(1, -1)
            metadata.slot_mapping = torch.tensor([length - 1], device="cuda", dtype=torch.int64)
            backend.forward(layer, new_query, new_key, new_value,
                            extended_cache, metadata, output=flash_output)
            logical_key = torch.cat([logical_key, new_key.transpose(0, 1).unsqueeze(0)], dim=2)
            logical_value = torch.cat([logical_value, new_value.transpose(0, 1).unsqueeze(0)], dim=2)
            expected = torch.nn.functional.scaled_dot_product_attention(
                new_query.transpose(0, 1).unsqueeze(0).float(),
                logical_key.float(), logical_value.float()).squeeze(0).transpose(0, 1)
            torch.testing.assert_close(flash_output.float(), expected, rtol=0.03, atol=0.01)
            torch.testing.assert_close(extended[0, blocks, :, step], new_key[0], rtol=0, atol=0)
            torch.testing.assert_close(extended[1, blocks, :, step], new_value[0], rtol=0, atol=0)
        torch.testing.assert_close(extended[:, :blocks], output, rtol=0, atol=0)
        print("PASS: three FlashAttention decode writes after CPU cache handoff")
    for cpu_tp, gpu_tp, heads in [(1, 1, 4), (2, 1, 4), (1, 2, 4),
                                (2, 2, 8), (2, 4, 8),
                                (1, 4, 2), (2, 4, 2)]:
        for rank in range(gpu_tp):
            run(cpu_tp, gpu_tp, rank, heads)
    check_sparse_multilayer_offsets()
