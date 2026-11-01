#!/usr/bin/env python3
"""
示例：选择性地将 CPU AMX KV cache 转换并传输到 GPU

场景：
- CPU 上运行 prefill 阶段，生成大量 KV cache (AMX 布局)
- 只需要将活跃请求的 KV cache 传输到 GPU 进行 decode
- 通过选择性转换减少内存和传输开销
"""

import torch
import time
from typing import List, Dict

try:
    import vllm._custom_ops as ops
    HAS_OPS = True
except ImportError:
    print("Warning: vllm._custom_ops not available")
    HAS_OPS = False


class CPUToGPUKVTransfer:
    """管理 CPU AMX KV cache 到 GPU 的选择性传输"""
    
    def __init__(
        self,
        num_blocks: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0"
    ):
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # CPU 端 KV cache (AMX 布局)
        self.key_cache_cpu = torch.empty(
            num_blocks, num_kv_heads, block_size, head_dim,
            dtype=dtype, device='cpu'
        )
        self.value_cache_cpu = torch.empty(
            num_blocks, num_kv_heads, block_size, head_dim,
            dtype=dtype, device='cpu'
        )
        
        # 跟踪哪些 blocks 已被使用
        self.block_usage = torch.zeros(num_blocks, dtype=torch.bool)
        
        print(f"Initialized KV cache manager:")
        print(f"  Total blocks: {num_blocks}")
        print(f"  Cache size: {self._get_cache_size_mb():.2f} MB")
    
    def _get_cache_size_mb(self) -> float:
        """计算 cache 大小（MB）"""
        bytes_per_elem = 2 if self.dtype == torch.bfloat16 else 4
        total_elems = self.num_blocks * self.num_kv_heads * self.block_size * self.head_dim
        return (total_elems * bytes_per_elem * 2) / (1024 ** 2)  # *2 for key+value
    
    def fill_cache_cpu(self, key: torch.Tensor, value: torch.Tensor, 
                       slot_mapping: torch.Tensor, isa: str = "amx"):
        """在 CPU 上填充 KV cache (使用 AMX 优化)"""
        if not HAS_OPS:
            print("Warning: ops not available, skipping cache fill")
            return
        
        ops.cpu_attn_reshape_and_cache(
            key, value,
            self.key_cache_cpu, self.value_cache_cpu,
            slot_mapping, isa
        )
        
        # 更新 block 使用情况
        block_ids = (slot_mapping // self.block_size).unique()
        self.block_usage[block_ids] = True
    
    def transfer_to_gpu(
        self, 
        block_ids: List[int],
        stream: torch.cuda.Stream = None
    ) -> Dict[str, torch.Tensor]:
        """
        选择性地将指定的 blocks 转换并传输到 GPU
        
        Args:
            block_ids: 需要传输的 block ID 列表
            stream: CUDA stream for async transfer
            
        Returns:
            包含 'key' 和 'value' 的字典，形状为 [len(block_ids), num_kv_heads, block_size, head_dim]
        """
        if not HAS_OPS:
            raise RuntimeError("vllm._custom_ops not available")
        
        num_blocks_to_transfer = len(block_ids)
        transfer_size_mb = (num_blocks_to_transfer / self.num_blocks) * self._get_cache_size_mb()
        
        print(f"\nTransferring {num_blocks_to_transfer} blocks to GPU...")
        print(f"  Block IDs: {block_ids[:10]}{'...' if len(block_ids) > 10 else ''}")
        print(f"  Transfer size: {transfer_size_mb:.2f} MB")
        
        # 1. 转换为标准布局
        start = time.time()
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.int64, device='cpu')
        key_standard, value_standard = ops.cpu_amx_cache_to_standard(
            self.key_cache_cpu, self.value_cache_cpu, 
            block_ids_tensor, self.head_dim
        )
        convert_time = (time.time() - start) * 1000
        print(f"  Conversion time: {convert_time:.2f} ms")
        
        # 2. 传输到 GPU
        start = time.time()
        if stream is not None:
            with torch.cuda.stream(stream):
                key_gpu = key_standard.to(self.device, non_blocking=True)
                value_gpu = value_standard.to(self.device, non_blocking=True)
        else:
            key_gpu = key_standard.to(self.device)
            value_gpu = value_standard.to(self.device)
        
        if stream is not None:
            stream.synchronize()
        
        transfer_time = (time.time() - start) * 1000
        print(f"  Transfer time: {transfer_time:.2f} ms")
        print(f"  Total time: {convert_time + transfer_time:.2f} ms")
        
        bandwidth = transfer_size_mb / (transfer_time / 1000)  # MB/s
        print(f"  Bandwidth: {bandwidth:.2f} MB/s")
        
        return {
            'key': key_gpu,
            'value': value_gpu
        }
    
    def get_active_blocks(self) -> List[int]:
        """获取所有活跃的 block IDs"""
        return torch.where(self.block_usage)[0].tolist()


def example_prefill_decode_workflow():
    """示例：Prefill 在 CPU，Decode 在 GPU"""
    
    print("=" * 60)
    print("Example: CPU Prefill -> GPU Decode Workflow")
    print("=" * 60)
    
    # 配置
    num_blocks = 1000
    num_kv_heads = 32
    block_size = 16
    head_dim = 128
    
    # 创建管理器
    manager = CPUToGPUKVTransfer(
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim
    )
    
    # === 阶段 1: CPU Prefill ===
    print("\n" + "=" * 60)
    print("Phase 1: CPU Prefill")
    print("=" * 60)
    
    # 模拟 prefill：生成 KV
    num_prefill_tokens = 100
    key_prefill = torch.randn(
        num_prefill_tokens, num_kv_heads, head_dim, 
        dtype=torch.bfloat16, device='cpu'
    )
    value_prefill = torch.randn(
        num_prefill_tokens, num_kv_heads, head_dim,
        dtype=torch.bfloat16, device='cpu'
    )
    
    # 分配 slots
    slot_mapping = torch.arange(num_prefill_tokens, dtype=torch.int64)
    
    print(f"Prefill tokens: {num_prefill_tokens}")
    print(f"Filling CPU cache with AMX layout...")
    
    if HAS_OPS:
        manager.fill_cache_cpu(key_prefill, value_prefill, slot_mapping, isa="amx")
        print("✓ CPU cache filled")
    else:
        print("⚠ Skipped (ops not available)")
    
    # === 阶段 2: 选择性传输到 GPU ===
    print("\n" + "=" * 60)
    print("Phase 2: Selective Transfer to GPU")
    print("=" * 60)
    
    # 假设只有部分请求需要在 GPU 上 decode
    # 例如：只需要前 20 个 blocks
    active_blocks = list(range(20))
    
    if HAS_OPS and torch.cuda.is_available():
        # 创建 CUDA stream 用于异步传输
        stream = torch.cuda.Stream()
        
        kv_gpu = manager.transfer_to_gpu(active_blocks, stream=stream)
        
        print(f"\n✓ GPU KV cache ready:")
        print(f"  Key shape: {kv_gpu['key'].shape}")
        print(f"  Value shape: {kv_gpu['value'].shape}")
        print(f"  Device: {kv_gpu['key'].device}")
        
        # === 阶段 3: GPU Decode ===
        print("\n" + "=" * 60)
        print("Phase 3: GPU Decode")
        print("=" * 60)
        print("(Use kv_gpu['key'] and kv_gpu['value'] for GPU attention)")
        
    else:
        print("⚠ Skipped (ops not available or CUDA not available)")
    
    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)


def example_batch_transfer():
    """示例：批量处理多个请求的传输"""
    
    print("\n\n" + "=" * 60)
    print("Example: Batch Transfer for Multiple Requests")
    print("=" * 60)
    
    # 配置
    num_blocks = 500
    num_kv_heads = 32
    block_size = 16
    head_dim = 128
    
    manager = CPUToGPUKVTransfer(
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim
    )
    
    # 模拟多个请求，每个请求使用不同的 blocks
    requests = {
        'req_0': list(range(0, 10)),      # 10 blocks
        'req_1': list(range(10, 25)),     # 15 blocks
        'req_2': list(range(25, 50)),     # 25 blocks
    }
    
    print(f"\nTransferring KV cache for {len(requests)} requests:")
    
    if HAS_OPS and torch.cuda.is_available():
        for req_id, block_ids in requests.items():
            print(f"\n{req_id}: {len(block_ids)} blocks")
            kv_gpu = manager.transfer_to_gpu(block_ids)
            print(f"  ✓ Ready on GPU")
    else:
        print("⚠ Skipped (ops not available or CUDA not available)")


if __name__ == "__main__":
    # 运行示例
    example_prefill_decode_workflow()
    example_batch_transfer()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

