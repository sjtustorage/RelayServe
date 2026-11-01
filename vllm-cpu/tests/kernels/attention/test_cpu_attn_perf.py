# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import math
import time
import torch

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    print("Skipping CPU-only tests")
    exit(0)

from vllm._custom_ops import (
    cpu_attention_with_kv_cache,
    cpu_attn_get_scheduler_metadata,
    cpu_attn_reshape_and_cache,
)

# 简化的测试配置（专注于性能测试）
NUM_HEADS = (32, 8)  # (num_query_heads, num_kv_heads)
HEAD_SIZE = 128
BLOCK_SIZE = 128
DTYPE = torch.bfloat16
SLIDING_WINDOW = None
SOFT_CAP = None
NUM_BLOCKS = 1024
USE_ALIBI = False
USE_SINK = False

# 测试场景
TEST_CASES = [
    # (name, seq_lens)
    # ("decode_batch", [(1, 1024), (1, 2048), (1, 4096), (1, 512), (1, 8192)]),
    # ("prefill_small", [(128, 128), (256, 256), (512, 512)]),
    # ("prefill_large", [(1024, 1024), (2048, 2048)]),
    # ("mixed_batch", [(1, 1024), (256, 1024), (1, 2048), (512, 2048), (1, 4096)]),
    ("prefill_large", [(1024, 1024)]),
]

ISA = "amx" if torch._C._cpu._is_amx_tile_supported() else "vec"


# rand number generation cache
@functools.lru_cache(maxsize=128, typed=False)
def tensor_cache(elem_num: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(elem_num, dtype=dtype)


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(
            closest_power_of_2, total_num_heads - closest_power_of_2
        )
        extra_powers = torch.arange(
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes.float()


def benchmark_attention(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    num_blocks: int,
    sliding_window: int | None,
    soft_cap: float | None,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    warmup_iters: int = 0,
    bench_iters: int = 1,
) -> dict:
    """运行性能测试并返回时间统计"""
    
    current_platform.seed_everything(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5
    token_num = sum(query_lens)

    alibi_slopes = _get_alibi_slopes(num_query_heads) if use_alibi else None
    s_aux = (
        15 * torch.rand((num_query_heads,), dtype=torch.bfloat16) if use_sink else None
    )

    # 准备输入数据
    query = tensor_cache(
        elem_num=token_num * num_query_heads * head_size,
        dtype=dtype,
    ).view(token_num, num_query_heads, head_size)

    key_value = tensor_cache(
        elem_num=2 * num_blocks * num_kv_heads * block_size * head_size,
        dtype=dtype,
    ).view(2, num_blocks, block_size, num_kv_heads, head_size)
    key_cache, value_cache = key_value.unbind(0)

    # KV cache for CPU attention
    packed_key_cache = torch.empty(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype
    )
    packed_value_cache = torch.empty_like(packed_key_cache)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    # Reshape and cache
    slot_mapping = torch.arange(0, num_blocks * block_size, dtype=torch.int64)
    cpu_attn_reshape_and_cache(
        key=key_cache.view(-1, num_kv_heads, head_size),
        value=value_cache.view(-1, num_kv_heads, head_size),
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        slot_mapping=slot_mapping,
        isa=isa,
    )

    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=num_seqs,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        seq_lens=kv_lens_tensor,
        dtype=dtype,
        query_start_loc=cu_query_lens,
        causal=True,
        sliding_window_size=sliding_window if sliding_window is not None else -1,
        isa=isa,
        enable_kv_split=False,
    )

    output = torch.empty_like(query)

    # Warmup
    for _ in range(warmup_iters):
        cpu_attention_with_kv_cache(
            query=query,
            key_cache=packed_key_cache,
            value_cache=packed_value_cache,
            output=output,
            query_start_loc=cu_query_lens,
            seq_lens=kv_lens_tensor,
            scale=scale,
            causal=True,
            alibi_slopes=alibi_slopes,
            sliding_window=window_size,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            scheduler_metadata=metadata,
            s_aux=s_aux,
        )

    # Benchmark
    times = []
    for _ in range(bench_iters):
        start_time = time.perf_counter()
        cpu_attention_with_kv_cache(
            query=query,
            key_cache=packed_key_cache,
            value_cache=packed_value_cache,
            output=output,
            query_start_loc=cu_query_lens,
            seq_lens=kv_lens_tensor,
            scale=scale,
            causal=True,
            alibi_slopes=alibi_slopes,
            sliding_window=window_size,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            scheduler_metadata=metadata,
            s_aux=s_aux,
        )
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # 转换为毫秒

    # 统计信息
    times = sorted(times)
    return {
        "mean": sum(times) / len(times),
        "median": times[len(times) // 2],
        "min": min(times),
        "max": max(times),
        "p95": times[int(len(times) * 0.95)],
        "std": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        "times": times,
    }


def print_results(case_name: str, seq_lens: list[tuple[int, int]], stats: dict):
    """打印性能测试结果"""
    total_q_tokens = sum(q for q, _ in seq_lens)
    total_kv_tokens = sum(kv for _, kv in seq_lens)
    
    print(f"\n{'='*80}")
    print(f"Test Case: {case_name}")
    print(f"  Batch size: {len(seq_lens)}")
    print(f"  Seq lens: {seq_lens}")
    print(f"  Total Q tokens: {total_q_tokens}")
    print(f"  Total KV tokens: {total_kv_tokens}")
    print(f"-" * 80)
    print(f"  Mean:   {stats['mean']:8.3f} ms")
    print(f"  Median: {stats['median']:8.3f} ms")
    print(f"  Min:    {stats['min']:8.3f} ms")
    print(f"  Max:    {stats['max']:8.3f} ms")
    print(f"  P95:    {stats['p95']:8.3f} ms")
    print(f"  Std:    {stats['std']:8.3f} ms")
    print(f"  Throughput: {total_q_tokens / (stats['mean'] / 1000):.1f} tokens/sec")
    print(f"{'='*80}")


def main():
    print("="*80)
    print("CPU Attention Performance Benchmark")
    print("="*80)
    print(f"ISA: {ISA}")
    print(f"Dtype: {DTYPE}")
    print(f"Num heads: {NUM_HEADS[0]} (Q) / {NUM_HEADS[1]} (KV)")
    print(f"Head size: {HEAD_SIZE}")
    print(f"Block size: {BLOCK_SIZE}")
    print("="*80)

    all_results = {}

    for case_name, seq_lens in TEST_CASES:
        print(f"\nRunning: {case_name}...")
        
        try:
            stats = benchmark_attention(
                seq_lens=seq_lens,
                num_heads=NUM_HEADS,
                head_size=HEAD_SIZE,
                block_size=BLOCK_SIZE,
                dtype=DTYPE,
                num_blocks=NUM_BLOCKS,
                sliding_window=SLIDING_WINDOW,
                soft_cap=SOFT_CAP,
                use_alibi=USE_ALIBI,
                use_sink=USE_SINK,
                isa=ISA,
                warmup_iters=0,
                bench_iters=1,
            )
            
            all_results[case_name] = stats
            print_results(case_name, seq_lens, stats)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # 汇总输出
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"{'Test Case':<20} {'Mean (ms)':<12} {'Throughput (tok/s)':<20}")
    print("-"*80)
    
    for case_name, seq_lens in TEST_CASES:
        if case_name in all_results:
            stats = all_results[case_name]
            total_q_tokens = sum(q for q, _ in seq_lens)
            throughput = total_q_tokens / (stats['mean'] / 1000)
            print(f"{case_name:<20} {stats['mean']:>10.3f}  {throughput:>18.1f}")
    
    print("="*80)


if __name__ == "__main__":
    main()

