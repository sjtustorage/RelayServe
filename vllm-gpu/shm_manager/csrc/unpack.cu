#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#define AMX_TILE_ROW_BYTES 64
using scalar_t = uint16_t;

__global__ void unpack_weight_vnni_to_nk_kernel(const uint32_t* packed, uint32_t* unpacked, int N, int K) {
    // 每个线程处理一对 K (32-bit dword)
    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15
    int k_tile_idx = blockIdx.x;
    int n_half_tile_idx = blockIdx.y;

    __shared__ uint32_t smem[16][17]; // 解决 Bank Conflict

    int n_panel = n_half_tile_idx / 2;
    int n_half = n_half_tile_idx % 2;

    // 1. 读取 Packed 到 Shared Memory
    int read_n = n_panel * 32 + n_half * 16 + tx;
    int read_k_pair = k_tile_idx * 16 + ty;

    if (read_n < N && (read_k_pair * 2) < K) {
        uint32_t packed_idx = (n_panel * (32 * K) + k_tile_idx * (32 * 32) + n_half * (16 * 32)) / 2 
                              + ty * 16 + tx;
        smem[ty][tx] = packed[packed_idx];
    }
    __syncthreads();

    // 2. 写入 Unpacked (标准 NK 布局)
    int write_n = n_half_tile_idx * 16 + ty;
    int write_k_pair = k_tile_idx * 16 + tx;

    if (write_n < N && (write_k_pair * 2) < K) {
        uint32_t unpacked_idx = write_n * (K / 2) + write_k_pair;
        unpacked[unpacked_idx] = smem[tx][ty];
    }
}

extern "C" void launch_unpack_weight(const void* src, void* dst, int N, int K, cudaStream_t stream, int gpu_rank, int task_type) {
    dim3 block(16, 16);
    dim3 grid((K + 31) / 32, (N + 15) / 16);
    unpack_weight_vnni_to_nk_kernel<<<grid, block, 0, stream>>>(
        (const uint32_t*)src, (uint32_t*)dst, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] VNNI unpacking weight kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
__global__ void unpack_cache_kernel(
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ block_indices,
    int64_t block_size,
    int64_t head_dim,
    int64_t cache_num_blocks_stride,
    int64_t cache_num_heads_stride) 
{
    const int64_t amx_tile_row_size = AMX_TILE_ROW_BYTES; 
    const int64_t amx_b_tile_n_size = AMX_TILE_ROW_BYTES / 4; 
    const int64_t elems_per_qw = 4 / sizeof(scalar_t); 
    const int64_t token_num_per_group = amx_tile_row_size / 4; 
    const int64_t token_num_per_sub_group = elems_per_qw; 

    extern __shared__ int8_t smem[];
    scalar_t* s_key = reinterpret_cast<scalar_t*>(smem);
    scalar_t* s_val = reinterpret_cast<scalar_t*>(smem + block_size * head_dim * sizeof(scalar_t));

    const int64_t list_idx = blockIdx.x;
    const int64_t head_idx = blockIdx.y;
    const int64_t block_idx = block_indices[list_idx];

    scalar_t* k_head_base = key_cache + block_idx * cache_num_blocks_stride + head_idx * cache_num_heads_stride;
    scalar_t* v_head_base = value_cache + block_idx * cache_num_blocks_stride + head_idx * cache_num_heads_stride;
    const int64_t total_elements = block_size * head_dim;

    for (int64_t tid = threadIdx.x; tid < total_elements; tid += blockDim.x) {
        int64_t token_idx = tid / head_dim;
        int64_t dim_idx = tid % head_dim;

        // Key Map
        int64_t quadword_num = head_dim / elems_per_qw;
        int64_t quadword_num_per_group = token_num_per_group * quadword_num;
        int64_t group_idx = token_idx / token_num_per_group;
        int64_t group_offset = token_idx % token_num_per_group;
        int64_t qw_idx = dim_idx / elems_per_qw;
        int64_t scalar_offset = dim_idx % elems_per_qw;
        
        int64_t k_in_idx = (group_idx * quadword_num_per_group + group_offset + qw_idx * token_num_per_group) * elems_per_qw + scalar_offset;
        s_key[tid] = k_head_base[k_in_idx];

        // Value Map
        int64_t i = dim_idx / amx_b_tile_n_size;
        int64_t j = dim_idx % amx_b_tile_n_size;
        int64_t sub_group_idx = token_idx / token_num_per_sub_group;
        int64_t sub_group_offset = token_idx % token_num_per_sub_group;
        
        int64_t v_in_idx = sub_group_idx * (token_num_per_sub_group * amx_b_tile_n_size) + sub_group_offset + i * (block_size * amx_b_tile_n_size) + j * token_num_per_sub_group;
        s_val[tid] = v_head_base[v_in_idx];
    }

    __syncthreads();

    for (int64_t tid = threadIdx.x; tid < total_elements; tid += blockDim.x) {
        k_head_base[tid] = s_key[tid];
        v_head_base[tid] = s_val[tid];
    }
}

extern "C"
void launch_unpack_cache(
    void* d_key_cache,
    void* d_val_cache,
    const int64_t* d_block_indices,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t block_size,
    int64_t head_dim,
    int64_t cache_num_blocks_stride,
    int64_t cache_num_heads_stride,
    cudaStream_t stream = nullptr) 
{
// 强转为明确大小的指针类型
    scalar_t* k_ptr = reinterpret_cast<scalar_t*>(d_key_cache);
    scalar_t* v_ptr = reinterpret_cast<scalar_t*>(d_val_cache);

    // 1. 计算共享内存大小
    size_t smem_size = 2 * block_size * head_dim * sizeof(scalar_t);

    // 2. 配置 Grid 和 Block
    dim3 grid(num_blocks, num_kv_heads);
    dim3 block(256);

    // 3. 解锁 48KB 动态共享内存限制
    cudaFuncSetAttribute(unpack_cache_kernel<scalar_t>, 
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // 4. 启动 Kernel
    unpack_cache_kernel<scalar_t><<<grid, block, smem_size, stream>>>(
        k_ptr, 
        v_ptr, 
        d_block_indices, 
        block_size, 
        head_dim, 
        cache_num_blocks_stride, 
        cache_num_heads_stride
    );

    // 5. 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] VNNI unpacking cache kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}




