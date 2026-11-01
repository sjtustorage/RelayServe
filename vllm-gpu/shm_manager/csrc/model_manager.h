#pragma once
#include <algorithm>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <string>
#include <thread>
#include <queue>
#include <torch/torch.h>
#include <c10/core/ScalarType.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <unordered_map>
#include <filesystem>
#include "tensor_writer.h"
#include "concurrent_queue.h"
#include <future>
#include <atomic>
#include <memory>

// The manager is now a pure execution engine
class ModelManager {
public:
    ModelManager(int element_size, int cpu_world_size, 
            int gpu_tp_rank, int gpu_world_size, int device);
    ~ModelManager();
    // Seal further submissions and drain before source SHM is detached.
    void close();

    void update_manager_info(
        std::unordered_map<int, uint64_t>& cpu_memory_base_ptrs, 
        std::unordered_map<int, uint64_t>& gpu_memory_base_ptrs);
    
    void generate_tensor_metadata( 
        size_t cpu_tensor_offset, 
        size_t gpu_tensor_offset,
        const std::vector<int>& full_shape, 
        const std::vector<int>& gpu_shape,
        const std::vector<int>& cpu_ranks,
        int sharding_dim, int layer_index, 
        int layer_end, int head_dim = -1,
        bool is_packed = true,
        const std::string& tensor_name = "");
    
    void generate_copy_tasks();
    std::unordered_map<int, std::vector<std::pair<size_t, size_t>>>
        copy_source_ranges(size_t max_gap_bytes = 0);
    void load_layers_async(const std::vector<int>& layer_idxes);
    /** Same workload as load_layers, then blocks until copy and unpack streams finish. */
    void load_layers_sync(const std::vector<int>& layer_idxes);
    void synchronize(const std::vector<int>& layer_idxes);
    bool check_load_state(const std::vector<int>& layer_idxes);
    std::vector<double> completed_layer_load_seconds();
    // Nonblocking snapshots: layer id, DMA payload bytes, copy-start to
    // depack-complete seconds. Includes stream dependencies and throttling.
    std::vector<std::tuple<int, uint64_t, double>> completed_layer_load_records();

    std::unordered_map<std::string, torch::Tensor> RestoreTensors(
        const std::unordered_map<
            std::string, std::tuple<uint64_t, std::vector<int64_t>, std::vector<int64_t>,
                                    std::string>>& meta_state_dict,
        const int device, const uint64_t base_address);
    
    uint64_t AllocateCudaMemory(const int device, const size_t tensor_size);
    // Layer starts are 256-byte aligned. A surviving tensor retains the entire
    // arena, including after close; model teardown normally drops them together.
    std::vector<uint64_t> AllocateCudaArena(
        const int device, const std::vector<size_t>& sizes);
    void update_module_loading_state(const bool is_module_loading);
    void update_model_arch_init_state(const bool is_model_arch_init,
                                     const bool bound_submissions = false);
    void update_total_layers(const int total_layers);
private:
    // Internal data structures
    enum class CopyType { COPY_1D, COPY_2D };

    struct CopyTask {
        CopyType type;
        void* src;
        void* dst;
        void* dst_tensor_base;
        size_t total_bytes;
        size_t width_bytes;
        size_t height;
        size_t src_pitch_bytes;
        size_t dst_pitch_bytes;
        int layer_index;
        int layer_end;
        bool is_packed;

        //-----------------------------
        size_t gpu_task_offset;
        int tensor_end;
        std::vector<int> full_shape;
        std::vector<int> gpu_shape;
        cudaEvent_t copy_event = nullptr;
        cudaEvent_t unpack_event = nullptr;

        std::string tensor_name;
    };

    // Metadata for a single tensor, passed from Python to C++
    struct TensorMetadata {
        // The byte offset of the current tensor relative to its rank's base pointer
        size_t cpu_tensor_offset;
        size_t gpu_tensor_offset;

        // General metadata about the tensor
        std::vector<int> full_shape;
        std::vector<int> gpu_shape;
        std::vector<int> cpu_ranks;
        int sharding_dim;
        int layer_index;
        int layer_end;
        int head_dim;
        bool is_packed;

        std::string tensor_name;
    };

    cudaStream_t copy_stream_;
    // Reused only under submission_mutex_; sleep the host while pacing DMA.
    cudaEvent_t copy_throttle_event_;
    std::vector<cudaEvent_t> copy_layer_events_;
    std::vector<cudaEvent_t> layer_start_events_;
    std::vector<std::vector<CopyTask>> copy_tasks_by_layer_;
    std::queue<TensorMetadata> tensor_metadata_queue_;

    std::vector<std::future<void>> load_layers_futures_;
    std::unordered_map<int, uint64_t> cpu_memory_base_ptrs_;
    std::unordered_map<int, uint64_t> gpu_memory_base_ptrs_;
    // Keep destinations alive until background submissions and streams drain.
    // Every restored tensor retains its allocation, including a shared layer arena.
    std::unordered_map<uint64_t, std::shared_ptr<void>> cuda_allocations_;

    int device_;
    bool closed_ = false;
    std::mutex lifecycle_mutex_;
    // Protected by readiness_mutex_; a failed manager cannot publish readiness.
    std::exception_ptr loading_error_;
    void record_loading_error(std::exception_ptr error);

    int element_size_;
    int cpu_world_size_;
    int gpu_tp_rank_;
    int gpu_world_size_;

    int total_layers_;

    std::atomic<bool> is_module_loading_;
    std::atomic<bool> bound_arch_submissions_;
    std::atomic<bool> is_kv_cache_init_;

    bool is_first_load_;

    void load_layers(const std::vector<int>& layer_idxes);
    void record_layer_complete(int layer_idx);
    void reserve_layer_loads(const std::vector<int>& layer_idxes);
    std::mutex submission_mutex_;
    std::mutex readiness_mutex_;
    std::condition_variable readiness_changed_;
    std::vector<bool> layer_events_recorded_;
    // Protected by readiness_mutex_; invalidated before every submission.
    std::vector<bool> layer_completion_observed_;
    std::vector<size_t> pending_layer_loads_;
    void generate_copy_task_for_tensor(const TensorMetadata& meta);

    //----------------------------------------
    cudaStream_t unpack_stream_;
    std::vector<cudaEvent_t> unpack_layer_events_;

    int buffer_idx_;
    size_t buffer_size_;
    void* buffer_ptrs_[2];
    cudaEvent_t copy_buffer_events_[2];
    cudaEvent_t unpack_buffer_events_[2];
    //----------------------------------------
};

std::unordered_map<std::string, size_t> SaveTensors(
    std::vector<std::string> tensor_names,
    std::unordered_map<std::string, std::pair<size_t, size_t>>& tensor_data,
    const std::string& path);

extern "C" void launch_unpack_weight(const void* src, void* dst, int N, int K, cudaStream_t stream);

