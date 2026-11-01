#include "kvcache_manager.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <algorithm>
#include <cassert>

// --- KVCacheTransferManager 实现 ---
KVCacheManager::KVCacheManager(const int total_layers,
                               const int gpu_tp_rank,
                               const int gpu_tp_size,
                               const int cpu_tp_size,
                               const int shm_num_blocks,
                               const int shm_block_len,
                               const int block_size,
                               const int head_dim,
                               const int device)
    : gpu_tp_rank_(gpu_tp_rank),
      gpu_tp_size_(gpu_tp_size),
      cpu_tp_size_(cpu_tp_size),
      shm_num_blocks_(shm_num_blocks),
      shm_block_len_(shm_block_len),
      total_layers_(total_layers),
      block_size_(block_size),
      head_dim_(head_dim),
      device_(device) {
    
    cudaSetDevice(device_);
    tp_ratio_ = gpu_tp_size / cpu_tp_size;
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    cudaStreamCreateWithPriority(&copy_stream_, cudaStreamNonBlocking, least_priority);
}

void KVCacheManager::update_shm_ptr_info(
    std::vector<int>& cpu_tp_ranks,
    std::unordered_map<int, uint64_t>& shm_caches_ptrs
) {
    cpu_tp_ranks_ = cpu_tp_ranks;
    shm_caches_ptrs_ = shm_caches_ptrs;
}

void KVCacheManager::update_gpu_ptr_info(
    const int gpu_num_blocks,
    const int gpu_block_len,
    const int total_num_kv_heads,
    std::unordered_map<int, std::vector<uint64_t>>& gpu_caches_ptrs
) {
    // Registration precedes inference. Drain before replacing reusable index
    // storage, including when PP recovery registers a different cache capacity.
    if (gpu_num_blocks <= 0)
        throw std::invalid_argument("GPU cache capacity must be positive");
    for (const auto& entry : gpu_caches_ptrs) {
        if (entry.first < 0 || entry.first >= total_layers_ || entry.second.size() != 2)
            throw std::invalid_argument("Invalid GPU cache layer pointers");
    }
    synchronize();
    cudaSetDevice(device_);
    if (static_cast<size_t>(gpu_num_blocks) > block_indices_capacity_) {
        if (block_indices_) cudaFree(block_indices_);
        if (host_block_indices_) cudaFreeHost(host_block_indices_);
        block_indices_ = nullptr;
        host_block_indices_ = nullptr;
        auto result = cudaMalloc(reinterpret_cast<void**>(&block_indices_),
                                 size_t(gpu_num_blocks) * sizeof(int64_t));
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        result = cudaMallocHost(reinterpret_cast<void**>(&host_block_indices_),
                                size_t(gpu_num_blocks) * sizeof(int64_t));
        if (result != cudaSuccess) {
            cudaFree(block_indices_);
            block_indices_ = nullptr;
            throw std::runtime_error(cudaGetErrorString(result));
        }
        block_indices_capacity_ = gpu_num_blocks;
    }
    gpu_num_blocks_ = gpu_num_blocks;
    gpu_block_len_ = gpu_block_len;
    total_num_kv_heads_ = total_num_kv_heads;
    is_kv_replicated_ = gpu_tp_size_ > total_num_kv_heads_;
    // Progressive and PP allocations can contain gaps. Only register actual
    // layer pointers, rather than inserting missing entries with null storage.
    gpu_caches_ptrs_ = gpu_caches_ptrs;

}

KVCacheManager::~KVCacheManager() {
    // Workers retain pointers to this manager and its stream until they finish.
    cudaSetDevice(device_);
    for (auto& future : load_kv_caches_async_futures_) if (future.valid()) future.wait();
    cudaStreamSynchronize(copy_stream_);
    if (block_indices_) cudaFree(block_indices_);
    if (host_block_indices_) cudaFreeHost(host_block_indices_);
    cudaStreamDestroy(copy_stream_);
}

void KVCacheManager::load_kv_caches_async(
        const std::vector<int>& gpu_block_ids,
        const std::vector<int>& cpu_block_ids,
        const std::vector<int>& layer_ids) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (closed_) throw std::runtime_error("KV cache manager is closed");
    load_kv_caches_async_futures_.push_back(
        std::async(std::launch::async, [this, gpu_block_ids, cpu_block_ids, layer_ids]() {
            cudaSetDevice(device_);
            load_kv_caches(gpu_block_ids, cpu_block_ids, layer_ids);
        })
    );
}

void KVCacheManager::load_kv_caches(
        const std::vector<int>& gpu_block_ids,
        const std::vector<int>& cpu_block_ids,
        const std::vector<int>& layer_ids) {
    // A transfer includes both copying packed data and unpacking it in place.
    // Interleaving two workers can otherwise unpack a block twice after the
    // second worker overwrites it, corrupting the final cache.
    std::lock_guard<std::mutex> lock(transfer_mutex_);
    if (gpu_block_ids.size() != cpu_block_ids.size())
        throw std::invalid_argument("CPU and GPU cache block counts differ");
    std::unordered_set<int> destinations;
    for (size_t i = 0; i < gpu_block_ids.size(); ++i) {
        if (gpu_block_ids[i] < 0 || gpu_block_ids[i] >= gpu_num_blocks_ ||
            cpu_block_ids[i] < 0 || cpu_block_ids[i] >= shm_num_blocks_)
            throw std::invalid_argument("Cache block index out of range");
        if (!destinations.insert(gpu_block_ids[i]).second)
            throw std::invalid_argument("Duplicate destination cache block");
    }
    for (int layer : layer_ids)
        if (!gpu_caches_ptrs_.count(layer))
            throw std::invalid_argument("Cache layer is not allocated");
    int num_copy_blocks = gpu_block_ids.size();
    for (int i = 0; i < num_copy_blocks; ++i) {
        int gpu_block_id = gpu_block_ids[i];
        int cpu_block_id = cpu_block_ids[i];
        for (auto layer_idx : layer_ids) {
            const auto& gpu_bases = gpu_caches_ptrs_.at(layer_idx);
            const uint64_t gpu_offset = uint64_t(gpu_block_id) * gpu_block_len_;
            uint64_t gpu_key_block_ptr = gpu_bases[0] + gpu_offset;
            uint64_t gpu_value_block_ptr = gpu_bases[1] + gpu_offset;
            const uint64_t cpu_plane_bytes = uint64_t(shm_num_blocks_) * shm_block_len_;
            const uint64_t cpu_offset = uint64_t(layer_idx) * 2 * cpu_plane_bytes
                                        + uint64_t(cpu_block_id) * shm_block_len_;
            if (tp_ratio_ >= 1) {
                int cpu_tp_rank = cpu_tp_ranks_[0];
                // Replicas share a KV head, not necessarily the first head in
                // their CPU shard. Map the global head back to its local offset.
                const int gpu_heads = std::max(1, total_num_kv_heads_ / gpu_tp_size_);
                const int cpu_heads = std::max(1, total_num_kv_heads_ / cpu_tp_size_);
                const int replicas = std::max(1, gpu_tp_size_ / total_num_kv_heads_);
                const int first_head = gpu_tp_rank_ / replicas * gpu_heads;
                uint64_t rank_offset = (first_head % cpu_heads) *
                    static_cast<uint64_t>(block_size_) * head_dim_ * sizeof(uint16_t);
                uint64_t cpu_key_block_ptr = 
                    shm_caches_ptrs_.at(cpu_tp_rank) + cpu_offset + rank_offset;
                uint64_t cpu_value_block_ptr = 
                    cpu_key_block_ptr + cpu_plane_bytes;
                cudaMemcpyAsync(
                    (void*)gpu_key_block_ptr, 
                    (void*)cpu_key_block_ptr, 
                    gpu_block_len_, 
                    cudaMemcpyHostToDevice, 
                    copy_stream_
                );
                cudaMemcpyAsync(
                    (void*)gpu_value_block_ptr, 
                    (void*)cpu_value_block_ptr, 
                    gpu_block_len_, 
                    cudaMemcpyHostToDevice, 
                    copy_stream_
                );
            }
            else {
                uint64_t gpu_loaded_block_len = 0;
                for (int cpu_tp_rank : cpu_tp_ranks_) {
                    uint64_t cpu_key_block_ptr = 
                        shm_caches_ptrs_.at(cpu_tp_rank) + cpu_offset;
                    uint64_t cpu_value_block_ptr = 
                        cpu_key_block_ptr + cpu_plane_bytes;
                    cudaMemcpyAsync(
                        (void*)gpu_key_block_ptr, 
                        (void*)cpu_key_block_ptr, 
                        shm_block_len_, 
                        cudaMemcpyHostToDevice, 
                        copy_stream_
                    );
                    cudaMemcpyAsync(
                        (void*)gpu_value_block_ptr, 
                        (void*)cpu_value_block_ptr, 
                        shm_block_len_, 
                        cudaMemcpyHostToDevice, 
                        copy_stream_
                    );
                    gpu_loaded_block_len += shm_block_len_;
                    if (gpu_loaded_block_len >= gpu_block_len_) break;
                    gpu_key_block_ptr += shm_block_len_;
                    gpu_value_block_ptr += shm_block_len_;
                }
            }
        }
    }

    if (num_copy_blocks > 0) {
        // The transfer lock retains this pinned staging buffer until unpack
        // completes. Copies and unpack share one stream, with no intermediate
        // host fence or per-transfer device allocation.
        std::copy(gpu_block_ids.begin(), gpu_block_ids.end(), host_block_indices_);
        cudaMemcpyAsync(block_indices_, host_block_indices_,
                        num_copy_blocks * sizeof(int64_t),
                        cudaMemcpyHostToDevice, copy_stream_);
        using scalar_t = uint16_t;
        // 3. 计算内存连续性 Stride (以 scalar_t 元素个数为单位)
        // gpu_block_len_ 是字节数，除以 sizeof(scalar_t) 就是元素的 Stride
        int64_t cache_num_blocks_stride = gpu_block_len_ / sizeof(scalar_t); 
        int64_t cache_num_heads_stride = block_size_ * head_dim_;

        // 5. 按 Layer 批量启动 Kernel
        for (auto layer_idx : layer_ids) {
            const auto& gpu_bases = gpu_caches_ptrs_.at(layer_idx);
            scalar_t* d_key_cache = reinterpret_cast<scalar_t*>(gpu_bases[0]);
            scalar_t* d_val_cache = reinterpret_cast<scalar_t*>(gpu_bases[1]);

            // 复用 streams_[0] 来做异步 Kernel 执行
            launch_unpack_cache(
                d_key_cache,
                d_val_cache,
                block_indices_,
                num_copy_blocks,
                std::max(1, total_num_kv_heads_ / gpu_tp_size_),
                block_size_,
                head_dim_,
                cache_num_blocks_stride,
                cache_num_heads_stride,
                copy_stream_
            );
        }

        // 6. 等待转换完成，并清理显存
        cudaStreamSynchronize(copy_stream_);
    }

}

void KVCacheManager::synchronize() {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    synchronize_pending();
}

void KVCacheManager::close() {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    closed_ = true;
    synchronize_pending();
    cudaSetDevice(device_);
    auto result = cudaStreamSynchronize(copy_stream_);
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

void KVCacheManager::synchronize_pending() {
    std::exception_ptr error;
    for (auto& future : load_kv_caches_async_futures_) {
        try { future.get(); }
        catch (...) { if (!error) error = std::current_exception(); }
    }
    load_kv_caches_async_futures_.clear();
    if (error) std::rethrow_exception(error);
}

bool KVCacheManager::check_load_state() {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    // A completed std::future remains in the vector until it is consumed.
    // Merely checking whether the vector is empty therefore reports "not
    // ready" forever for asynchronous transfers.  Poll without blocking and
    // reclaim the futures once every transfer has completed.  synchronize()
    // also calls get(), so worker exceptions are propagated to the caller.
    for (const auto& future : load_kv_caches_async_futures_) {
        if (future.wait_for(std::chrono::seconds(0)) !=
            std::future_status::ready) {
            return false;
        }
    }
    synchronize_pending();
    return true;
}
