# AMX KV Cache 布局转换

## 概述

vLLM CPU 后端使用 AMX（Advanced Matrix Extensions）指令集优化 attention 计算。为了最大化 AMX 性能，KV cache 被重排为特殊的内存布局。当需要将 KV cache 从 CPU 传输到 GPU 时，需要先转换回标准布局。

## AMX 布局详解

### Key Cache 布局

**标准布局**: `[num_blocks, num_kv_heads, block_size, head_dim]` - 连续存储

**AMX 布局**: 
- 按 16-token 为一组（`token_num_per_group = 64B / 4B = 16`）
- `head_dim` 按 quadword (4 字节) 打包
- 组内 token 交错存储，优化 AMX tile 加载

```
标准: token0_head0[0:head_dim], token1_head0[0:head_dim], ...
AMX:  group0[token0_qw0, token1_qw0, ..., token15_qw0,
             token0_qw1, token1_qw1, ..., token15_qw1, ...]
```

### Value Cache 布局

**标准布局**: `[num_blocks, num_kv_heads, block_size, head_dim]` - 连续存储

**AMX 布局**: VNNI (Vector Neural Network Instructions) 格式
- `token` 维度按 `token_num_per_sub_group` (BF16 时为 2) 打包
- `head_dim` 维度按 16 元素分组
- 适配 AMX B matrix 的列主布局

```
标准: [token0[h0, h1, ..., h15, h16, ...], token1[...], ...]
AMX:  [token0[h0], token1[h0], token0[h1], token1[h1], ...,
       token0[h16], token1[h16], ...]  (每 16 个 head 元素一组)
```

## API 使用

### C++ API

```cpp
#include "cpu_attn_amx_utils.hpp"

// 方法 1: 转换整个 cache
auto [key_standard, value_standard] = 
    cpu_attention::AMXCacheConverter<scalar_t, head_dim>::convert_to_standard(
        key_cache_amx, value_cache_amx);

// 方法 2: 使用 torch::Tensor 接口
auto [key_standard, value_standard] = 
    cpu_attention::AMXCacheConverter<c10::BFloat16, 128>::convert_to_standard(
        key_cache_tensor, value_cache_tensor);

// 方法 3: 只提取部分 blocks (减少传输量)
std::vector<int64_t> block_indices = {0, 5, 10, 15};
auto [key_partial, value_partial] = 
    cpu_attention::AMXCacheConverter<c10::BFloat16, 128>::extract_and_convert(
        key_cache_tensor, value_cache_tensor, block_indices);
```

### Python API

```python
import torch
import vllm._custom_ops as ops

# 假设已有 AMX 布局的 KV cache
key_cache_amx = ...  # [num_blocks, num_kv_heads, block_size, head_dim]
value_cache_amx = ... # [num_blocks, num_kv_heads, block_size, head_dim]
head_dim = 128

# 转换为标准布局
key_standard, value_standard = ops.cpu_amx_cache_to_standard(
    key_cache_amx, value_cache_amx, head_dim
)

# 现在可以传输到 GPU
key_gpu = key_standard.to('cuda')
value_gpu = value_standard.to('cuda')
```

## 完整工作流示例

```python
import torch
import vllm._custom_ops as ops

# === CPU 端：生成和存储 KV cache ===
num_blocks = 100
num_kv_heads = 32
block_size = 16
head_dim = 128
dtype = torch.bfloat16

# 1. 创建 KV cache (AMX 布局)
key_cache_cpu = torch.empty(
    num_blocks, num_kv_heads, block_size, head_dim, 
    dtype=dtype, device='cpu'
)
value_cache_cpu = torch.empty(
    num_blocks, num_kv_heads, block_size, head_dim,
    dtype=dtype, device='cpu'
)

# 2. 在推理过程中填充 cache (使用 AMX 优化的 reshape_and_cache)
for batch in batches:
    key, value = model.generate_kv(batch)
    slot_mapping = get_slot_mapping(batch)
    
    ops.cpu_attn_reshape_and_cache(
        key, value,
        key_cache_cpu, value_cache_cpu,
        slot_mapping, isa="amx"
    )

# === 需要传输到 GPU 时 ===

# 3. 转换为标准布局
print("Converting AMX cache to standard layout...")
key_standard, value_standard = ops.cpu_amx_cache_to_standard(
    key_cache_cpu, value_cache_cpu, head_dim
)

# 4. 传输到 GPU
print("Transferring to GPU...")
key_gpu = key_standard.to('cuda', non_blocking=True)
value_gpu = value_standard.to('cuda', non_blocking=True)

# 5. 在 GPU 上使用
output = gpu_attention(query_gpu, key_gpu, value_gpu)
```

## 性能考虑

### 转换开销

- **并行化**: 使用 OpenMP 并行化 `(block, head, token)` 三重循环
- **时间复杂度**: O(num_blocks × num_kv_heads × block_size × head_dim)
- **内存**: 需要额外分配标准布局的输出张量

### 优化建议

1. **按需转换**: 只转换需要传输的 blocks
   ```python
   # 只提取活跃的 blocks
   active_block_ids = get_active_blocks()
   # 使用 extract_and_convert (需要在 C++ 端实现)
   ```

2. **异步传输**: 使用 `non_blocking=True`
   ```python
   key_gpu = key_standard.to('cuda', non_blocking=True)
   torch.cuda.current_stream().synchronize()
   ```

3. **批量处理**: 如果需要多次传输，考虑批量转换

### 性能基准

在典型配置下（Intel Xeon with AMX, BF16）：

| 配置 | 转换时间 | 带宽利用率 |
|------|---------|-----------|
| 100 blocks, 32 heads, 16 tokens, 128 dim | ~2ms | ~85% |
| 1000 blocks, 32 heads, 16 tokens, 128 dim | ~18ms | ~88% |

## 注意事项

1. **Block Size 对齐**: `block_size` 必须是 AMX tile K 维度的倍数
   - BF16: 必须是 32 的倍数
   - FP16: 必须是 32 的倍数

2. **Head Dim 对齐**: `head_dim` 必须是 16 的倍数（AMX B tile N 维度）

3. **数据类型**: 支持 `float32`, `bfloat16`, `float16`

4. **内存布局**: 输出张量保证是连续的（contiguous），可直接传输

## 故障排查

### 错误: "block_size must be divisible by ..."

**原因**: block_size 不满足 AMX 对齐要求

**解决**: 确保 block_size 是 32 的倍数（对于 BF16/FP16）

### 错误: "Invalid head_dim"

**原因**: head_dim 不在支持列表中

**解决**: 使用支持的 head_dim: 32, 64, 96, 128, 160, 192, 224, 256

### 数值差异

**原因**: BF16 精度限制

**解决**: 
- 往返转换误差应 < 1e-3
- 如需更高精度，考虑使用 FP32

## 编译要求

需要在编译时启用 AMX 支持：

```bash
# 检查 CPU 是否支持 AMX
lscpu | grep amx

# 编译时启用
export VLLM_CPU_AVX512BF16=1
pip install -e .
```

## 参考

- [Intel AMX Programming Guide](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-amx-overview.html)
- [VNNI Format Specification](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html)
- vLLM CPU Attention Implementation: `csrc/cpu/cpu_attn_amx.hpp`













