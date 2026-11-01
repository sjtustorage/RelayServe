#include <string>
#include <map>
#include <vector>
#include <cuda_runtime.h>
#include <mutex>
#include <queue>
#include <future>
#include <unordered_map>
#include <unordered_set>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>

class KVCacheManager {
public:
    KVCacheManager( const int total_layers,
                    const int gpu_tp_rank,
                    const int gpu_tp_size,
                    const int cpu_tp_size,
                    const int shm_num_blocks,
                    const int shm_block_len, 
                    const int block_size,
                    const int head_dim,
                    const int device);
    ~KVCacheManager();
    void close();

    void update_shm_ptr_info(
        std::vector<int>& cpu_tp_ranks,
        std::unordered_map<int, uint64_t>& shm_caches_ptrs
    );

    void update_gpu_ptr_info(
        const int gpu_num_blocks,
        const int gpu_block_len,
        const int total_num_kv_heads,
        std::unordered_map<int, std::vector<uint64_t>>& gpu_caches_ptrs
    );

    void load_kv_caches_async(
        const std::vector<int>& gpu_block_ids,
        const std::vector<int>& cpu_block_ids,
        const std::vector<int>& layer_ids
    );

    void synchronize();

    bool check_load_state();

private:
    int gpu_tp_size_;
    int cpu_tp_size_;
    int gpu_tp_rank_;
    std::vector<int> cpu_tp_ranks_;
    int total_layers_;
    int gpu_num_blocks_;
    int gpu_block_len_;
    int shm_num_blocks_;
    int shm_block_len_;
    int total_num_kv_heads_;
    int num_streams_;
    int stream_idx_;
    int device_;

    int block_size_;
    int head_dim_;

    int tp_ratio_;
    // If true, all TP ranks read KV from the same SHM offset (replicated);
    // else each rank uses rank_offset. Exposed for future use; not set from Python.
    bool is_kv_replicated_ = false;

    cudaStream_t copy_stream_;
    int64_t* block_indices_ = nullptr;
    int64_t* host_block_indices_ = nullptr;
    size_t block_indices_capacity_ = 0;

    // Block addresses are affine offsets, not separately allocated objects.
    // Keep registration O(layers + CPU ranks), independent of cache capacity.
    std::unordered_map<int, std::vector<uint64_t>> gpu_caches_ptrs_;
    std::unordered_map<int, uint64_t> shm_caches_ptrs_;


    void load_kv_caches(
        const std::vector<int>& gpu_block_ids,
        const std::vector<int>& cpu_block_ids,
        const std::vector<int>& layer_ids);
    
    std::vector<std::future<void>> load_kv_caches_async_futures_;
    std::mutex transfer_mutex_;
    std::mutex lifecycle_mutex_;
    bool closed_ = false;
    void synchronize_pending();

};

extern "C"
void launch_unpack_cache(
    void* __restrict__ key_cache,
    void* __restrict__ value_cache,
    const int64_t* __restrict__ block_indices,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t block_size,
    int64_t head_dim,
    int64_t cache_num_blocks_stride,
    int64_t cache_num_heads_stride,
    cudaStream_t stream = nullptr
);
