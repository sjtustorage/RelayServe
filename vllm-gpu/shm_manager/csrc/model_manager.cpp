#include "model_manager.h"
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <chrono>

#define N_STEP 32

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error in ") + #call + ": " + cudaGetErrorString(err)); \
        } \
    } while (0)

// Mapping from string to at::ScalarType
at::ScalarType stringToScalarType(const std::string& dtype_str) {
    static const std::unordered_map<std::string, at::ScalarType> dtype_map = {
        {"torch.float16", torch::kFloat16},  {"torch.float32", torch::kFloat32},
        {"torch.float64", torch::kFloat64},  {"torch.int16", torch::kInt16},
        {"torch.int32", torch::kInt32},      {"torch.int64", torch::kInt64},
        {"torch.uint8", torch::kUInt8},      {"torch.int8", torch::kInt8},
        {"torch.bfloat16", torch::kBFloat16}};
  
    auto it = dtype_map.find(dtype_str);
    if (it != dtype_map.end()) {
      return it->second;
    } else {
      throw std::invalid_argument("Unknown dtype string: " + dtype_str);
    }
}

// --- ModelManager Implementation ---

ModelManager::ModelManager(
    int element_size,
    int cpu_world_size,
    int gpu_tp_rank,
    int gpu_world_size,
    int device
) {
    cudaSetDevice(device);
    device_ = device;
    element_size_ = element_size;
    cpu_world_size_ = cpu_world_size;
    gpu_tp_rank_ = gpu_tp_rank;
    gpu_world_size_ = gpu_world_size;
    is_module_loading_ = false;
    bound_arch_submissions_ = false;
    is_kv_cache_init_ = false;
    is_first_load_ = true;
    buffer_idx_ = 0;
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

    CUDA_CHECK(cudaStreamCreateWithPriority(&copy_stream_, cudaStreamNonBlocking, least_priority));
    CUDA_CHECK(cudaEventCreateWithFlags(&copy_throttle_event_,
        cudaEventBlockingSync | cudaEventDisableTiming));
    
    //---------------------------------------------
    CUDA_CHECK(cudaStreamCreateWithPriority(&unpack_stream_, cudaStreamNonBlocking, least_priority));
    buffer_size_ = 0;
    for (int i = 0; i < 2; i++) {
        buffer_ptrs_[i] = nullptr;
        CUDA_CHECK(cudaEventCreate(&copy_buffer_events_[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&unpack_buffer_events_[i],
            cudaEventBlockingSync | cudaEventDisableTiming));
        // Record immediately so the event is in a "completed" state.
        // The first cudaStreamWaitEvent on this event will not block.
        CUDA_CHECK(cudaEventRecord(unpack_buffer_events_[i], 0));
    }
    //---------------------------------------------
}

ModelManager::~ModelManager() {
    // Async submissions capture this manager and may still be using its raw
    // destination pointers. Stream destruction alone does not wait for work.
    cudaSetDevice(device_);
    for (auto &future : load_layers_futures_) if (future.valid()) future.wait();
    load_layers_futures_.clear();
    cudaStreamSynchronize(copy_stream_);
    cudaStreamSynchronize(unpack_stream_);
    cudaStreamDestroy(copy_stream_);
    cudaEventDestroy(copy_throttle_event_);
    for (auto e : copy_layer_events_) if (e) cudaEventDestroy(e);
    copy_layer_events_.clear();
    for (auto e : layer_start_events_) if (e) cudaEventDestroy(e);
    layer_start_events_.clear();

    //---------------------------------------------
    cudaStreamDestroy(unpack_stream_);
    for (auto e : unpack_layer_events_) if (e) cudaEventDestroy(e);
    unpack_layer_events_.clear();

    for (int i = 0; i < 2; i++) {
        cudaFree(buffer_ptrs_[i]);
        cudaEventDestroy(copy_buffer_events_[i]);
        cudaEventDestroy(unpack_buffer_events_[i]);
    }
    //----------------------------------------------
}

void ModelManager::update_total_layers(const int total_layers) {
    CUDA_CHECK(cudaSetDevice(device_));
    total_layers_ = total_layers;
    copy_layer_events_.resize(total_layers_ + 1, nullptr);
    layer_start_events_.resize(total_layers_ + 1, nullptr);
    unpack_layer_events_.resize(total_layers_ + 1, nullptr);
    layer_events_recorded_.resize(total_layers_ + 1, false);
    layer_completion_observed_.resize(total_layers_ + 1, false);
    pending_layer_loads_.resize(total_layers_ + 1, 0);
    copy_tasks_by_layer_.resize(total_layers_ + 1, std::vector<CopyTask>());
}

void ModelManager::update_manager_info(
        std::unordered_map<int, uint64_t>& cpu_memory_base_ptrs, 
        std::unordered_map<int, uint64_t>& gpu_memory_base_ptrs) {
    cpu_memory_base_ptrs_ = cpu_memory_base_ptrs;
    gpu_memory_base_ptrs_ = gpu_memory_base_ptrs;
}

void ModelManager::update_module_loading_state(const bool is_module_loading) {
    if (is_module_loading) is_kv_cache_init_ = false;
    is_module_loading_ = is_module_loading;
}

void ModelManager::update_model_arch_init_state(const bool is_model_arch_init,
                                              const bool bound_submissions) {
    if (!is_model_arch_init) is_kv_cache_init_ = true;
    bound_arch_submissions_ = is_model_arch_init && bound_submissions;
}

// --- Standalone Plan Generator ---

void ModelManager::generate_tensor_metadata(
    size_t cpu_tensor_offset, 
    size_t gpu_tensor_offset,
    const std::vector<int>& full_shape, 
    const std::vector<int>& gpu_shape,
    const std::vector<int>& cpu_ranks,
    int sharding_dim, 
    int layer_index,
    int layer_end,
    int head_dim,
    bool is_packed,
    const std::string& tensor_name
) {
    TensorMetadata meta;
    meta.full_shape = full_shape;
    meta.gpu_shape = gpu_shape;
    meta.cpu_ranks = cpu_ranks;
    meta.sharding_dim = sharding_dim;
    meta.cpu_tensor_offset = cpu_tensor_offset;
    meta.gpu_tensor_offset = gpu_tensor_offset;
    meta.layer_index = layer_index;
    meta.layer_end = layer_end;
    meta.head_dim = head_dim;
    meta.is_packed = is_packed;
    meta.tensor_name = tensor_name;
    tensor_metadata_queue_.push(meta);
}

void ModelManager::generate_copy_task_for_tensor(const TensorMetadata& meta) {    
    const int src_tp_degree = cpu_world_size_;
    const int dst_tp_degree = gpu_world_size_;
    const int gpu_tp_rank = gpu_tp_rank_;
    int layer_index = meta.layer_index;

    if (meta.sharding_dim == 0) { // RowParallel
        const size_t In_Dim = meta.full_shape[0];
        const size_t Out_Dim = meta.full_shape[1];
        const size_t row_bytes = Out_Dim * element_size_;
        int cpu_shard_rows = In_Dim / src_tp_degree;
        int gpu_shard_rows = In_Dim / dst_tp_degree;
        if (meta.head_dim != -1) {
            cpu_shard_rows = std::max(meta.head_dim, cpu_shard_rows);
            gpu_shard_rows = std::max(meta.head_dim, gpu_shard_rows);
        }

        uint64_t dst_offset = gpu_memory_base_ptrs_[meta.layer_index] + meta.gpu_tensor_offset;
        void* dst_tensor_base_ptr = reinterpret_cast<void*>(dst_offset);
        void* dst_current_ptr = dst_tensor_base_ptr;
        int start_global_row = gpu_tp_rank * gpu_shard_rows;
        int end_global_row = (gpu_tp_rank + 1) * gpu_shard_rows;

        for (int g_row = start_global_row; g_row < end_global_row; ) {
            int src_shard_idx = g_row / cpu_shard_rows;
            int src_local_row = g_row % cpu_shard_rows;
            int rows_in_src_shard_left = cpu_shard_rows - src_local_row;
            int rows_to_end_of_dst_shard = end_global_row - g_row;
            int contiguous_rows_to_copy = 
                std::min(rows_in_src_shard_left, rows_to_end_of_dst_shard);

            uint64_t src_offset = 
                cpu_memory_base_ptrs_[src_shard_idx] + meta.cpu_tensor_offset;
            void* src_tensor_base_ptr = reinterpret_cast<void*>(src_offset);
            void* src_ptr = src_tensor_base_ptr + (src_local_row * row_bytes);
            uint64_t copy_bytes = contiguous_rows_to_copy * row_bytes;

            int truly_end = -1;
            if (meta.layer_end != -1) {
                if (g_row + contiguous_rows_to_copy >= end_global_row) {
                    truly_end = meta.layer_end;
                }
            }

            //----------------------------------------------------
            cudaEvent_t copy_event = nullptr, unpack_event = nullptr;
            if (truly_end != -1) {
                CUDA_CHECK(cudaEventCreate(&copy_event));
                CUDA_CHECK(cudaEventCreate(&unpack_event));
                copy_layer_events_[meta.layer_index] = copy_event;
                unpack_layer_events_[meta.layer_index] = unpack_event;

            }
            int tensor_end = g_row + contiguous_rows_to_copy >= end_global_row;
            size_t gpu_task_offset = (uint8_t*)dst_current_ptr - (uint8_t*)dst_tensor_base_ptr;
            //----------------------------------------------------

            CopyTask task = {CopyType::COPY_1D, src_ptr, dst_current_ptr, dst_tensor_base_ptr,
                copy_bytes, 0, 0, 0, 0, 
                meta.layer_index, truly_end, meta.is_packed,
                //------------------------------------------------
                gpu_task_offset, tensor_end, meta.full_shape, meta.gpu_shape, copy_event, unpack_event, meta.tensor_name
                //------------------------------------------------
            };

            copy_tasks_by_layer_[meta.layer_index].push_back(task);

            dst_current_ptr += copy_bytes;
            g_row += contiguous_rows_to_copy;
        }
        
    } else if (meta.sharding_dim == 1) { // ColumnParallel
        size_t In_Dim = meta.full_shape[0];  
        const size_t Out_Dim = meta.full_shape[1];  
        const int cpu_shard_cols = Out_Dim / src_tp_degree;  
        const int gpu_shard_cols = Out_Dim / dst_tp_degree;  
        size_t cpu_pitch_bytes = cpu_shard_cols * element_size_;  
        size_t gpu_pitch_bytes = gpu_shard_cols * element_size_;  

        //--------------------------------------
        In_Dim /= N_STEP;
        cpu_pitch_bytes *= N_STEP;
        gpu_pitch_bytes *= N_STEP;
        //--------------------------------------

        uint64_t dst_offset = 
            gpu_memory_base_ptrs_[meta.layer_index] + meta.gpu_tensor_offset;
        void* dst_tensor_base_ptr = reinterpret_cast<void*>(dst_offset);
        void* dst_current_ptr = dst_tensor_base_ptr;
        int start_global_col = gpu_tp_rank * gpu_shard_cols;  
        int end_global_col = (gpu_tp_rank + 1) * gpu_shard_cols;  

        for (int g_col = start_global_col; g_col < end_global_col; ) {
            int src_shard_idx = g_col / cpu_shard_cols;    
            int src_local_col = g_col % cpu_shard_cols;    
            int cols_in_src_shard_left = cpu_shard_cols - src_local_col;   
            int cols_to_end_of_dst_shard = end_global_col - g_col;         
            int contiguous_cols_to_copy = 
                std::min(cols_in_src_shard_left, cols_to_end_of_dst_shard);    
            
            uint64_t src_offset = 
                cpu_memory_base_ptrs_[src_shard_idx] + meta.cpu_tensor_offset;
            void* src_tensor_base_ptr = reinterpret_cast<void*>(src_offset);
            void* src_ptr = src_tensor_base_ptr + (src_local_col * element_size_);
            uint64_t copy_width_bytes = contiguous_cols_to_copy * element_size_;  

            //---------------------------------------- 
            copy_width_bytes *= N_STEP;
            src_ptr += src_local_col * element_size_ * (N_STEP - 1);
            //----------------------------------------
            
            int truly_end = -1;
            if (meta.layer_end != -1) {
                if (g_col + contiguous_cols_to_copy >= end_global_col) {
                    truly_end = meta.layer_end;
                }
            }

            //----------------------------------------------------
            cudaEvent_t copy_event = nullptr, unpack_event = nullptr;
            if (truly_end != -1) {
                CUDA_CHECK(cudaEventCreate(&copy_event));
                CUDA_CHECK(cudaEventCreate(&unpack_event));
                copy_layer_events_[meta.layer_index] = copy_event;
                unpack_layer_events_[meta.layer_index] = unpack_event;

            }
            int tensor_end = g_col + contiguous_cols_to_copy >= end_global_col;
            size_t gpu_task_offset = (uint8_t*)dst_current_ptr - (uint8_t*)dst_tensor_base_ptr;
            //----------------------------------------------------

            CopyTask task = {CopyType::COPY_2D, src_ptr, dst_current_ptr, dst_tensor_base_ptr,
                0, copy_width_bytes, In_Dim, 
                cpu_pitch_bytes, gpu_pitch_bytes,
                meta.layer_index, truly_end, meta.is_packed,
                //------------------------------------------------
                gpu_task_offset, tensor_end, meta.full_shape, meta.gpu_shape, copy_event, unpack_event, meta.tensor_name
                //------------------------------------------------
            };

            copy_tasks_by_layer_[meta.layer_index].push_back(task);

            dst_current_ptr += copy_width_bytes;
            g_col += contiguous_cols_to_copy;
        }
        
    } else if (meta.sharding_dim == -1) {
        const size_t total_bytes = meta.full_shape[0] * element_size_;
        uint64_t src_offset = 
            cpu_memory_base_ptrs_[meta.cpu_ranks[0]] + meta.cpu_tensor_offset;
        void* src_ptr = reinterpret_cast<void*>(src_offset);
        uint64_t dst_offset = 
            gpu_memory_base_ptrs_[meta.layer_index] + meta.gpu_tensor_offset;
        void* dst_ptr = reinterpret_cast<void*>(dst_offset);

        //-------------------------------------------------
        cudaEvent_t copy_event = nullptr, unpack_event = nullptr;
        if (meta.layer_end != -1) {
            CUDA_CHECK(cudaEventCreate(&copy_event));
            CUDA_CHECK(cudaEventCreate(&unpack_event));
            copy_layer_events_[meta.layer_index] = copy_event;
            unpack_layer_events_[meta.layer_index] = unpack_event;
        }
        //-------------------------------------------------

        CopyTask task = {CopyType::COPY_1D, src_ptr, dst_ptr, dst_ptr,
            total_bytes, 0, 0, 0, 0, 
            meta.layer_index, meta.layer_end, meta.is_packed,
            //---------------------------------------------
            0, 1, meta.full_shape, meta.gpu_shape, copy_event, unpack_event, meta.tensor_name
            //---------------------------------------------
        };
        copy_tasks_by_layer_[meta.layer_index].push_back(task);
    } else {
        throw std::invalid_argument("sharding_dim must be 0 or 1 or -1");
    }
}

std::unordered_map<int, std::vector<std::pair<size_t, size_t>>>
ModelManager::copy_source_ranges(size_t max_gap_bytes) {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    std::vector<std::pair<uintptr_t, uintptr_t>> spans;
    for (const auto& tasks : copy_tasks_by_layer_) {
        for (const auto& task : tasks) {
            const auto source = reinterpret_cast<uintptr_t>(task.src);
            const size_t rows = task.type == CopyType::COPY_1D ? 1 : task.height;
            const size_t width = task.type == CopyType::COPY_1D
                ? task.total_bytes : task.width_bytes;
            if (!width) continue;
            for (size_t row = 0; row < rows; ++row) {
                const auto begin = source + row * task.src_pitch_bytes;
                spans.emplace_back(begin / page * page,
                                   (begin + width + page - 1) / page * page);
            }
        }
    }
    std::sort(spans.begin(), spans.end());
    std::unordered_map<int, std::vector<std::pair<size_t, size_t>>> result;
    for (const auto& [begin, end] : spans) {
        // Mappings are disjoint. The greatest preceding base identifies
        // which producer rank owns this task's source bytes.
        int rank = -1;
        uintptr_t base = 0;
        for (const auto& [candidate, address] : cpu_memory_base_ptrs_) {
            if (address <= begin && address >= base) {
                rank = candidate;
                base = address;
            }
        }
        if (rank < 0) throw std::runtime_error("Copy source precedes all SHM mappings");
        auto& ranges = result[rank];
        const size_t offset = begin - base;
        const size_t length = end - begin;
        if (!ranges.empty()) {
            auto& previous = ranges.back();
            const size_t previous_end = previous.first + previous.second;
            if (offset <= previous_end || offset - previous_end <= max_gap_bytes) {
                previous.second = std::max(previous_end, offset + length) - previous.first;
                continue;
            }
        }
        ranges.emplace_back(offset, length);
    }
    return result;
}

void ModelManager::generate_copy_tasks() {
    const bool profile = std::getenv("VLLM_PROFILE_RELAY") != nullptr;
    auto phase_start = std::chrono::steady_clock::now();
    auto mark = [&](const char* phase) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "SHM copy plan: rank=%d phase=%s wall_ms=%.3f\n",
                     gpu_tp_rank_, phase,
                     std::chrono::duration<double, std::milli>(now - phase_start).count());
        phase_start = now;
    };
    try {
        CUDA_CHECK(cudaSetDevice(device_));
        mark("set_device");
        while (!tensor_metadata_queue_.empty()) {
            TensorMetadata meta = tensor_metadata_queue_.front();
            tensor_metadata_queue_.pop();
            if(gpu_memory_base_ptrs_.find(meta.layer_index) == gpu_memory_base_ptrs_.end()) continue;
            generate_copy_task_for_tensor(meta);
        }
        mark("task_construction");
        if (std::getenv("VLLM_PROFILE_SHM_ATTACH")) {
            // Report the actual host ranges consumed by this GPU.  SHM
            // registration currently pins the complete source segment; this
            // diagnostic makes the potentially useful subset measurable
            // before changing that ownership/lifetime contract.
            struct Range { uintptr_t begin, end; };
            std::vector<Range> ranges;
            size_t logical_bytes = 0;
            for (const auto& tasks : copy_tasks_by_layer_) {
                for (const auto& task : tasks) {
                    if (task.type == CopyType::COPY_1D) {
                        if (!task.total_bytes) continue;
                        auto begin = reinterpret_cast<uintptr_t>(task.src);
                        ranges.push_back({begin, begin + task.total_bytes});
                        logical_bytes += task.total_bytes;
                    } else {
                        auto row = reinterpret_cast<uintptr_t>(task.src);
                        for (size_t i = 0; i < task.height; ++i) {
                            ranges.push_back({row, row + task.width_bytes});
                            logical_bytes += task.width_bytes;
                            row += task.src_pitch_bytes;
                        }
                    }
                }
            }
            const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
            for (auto& range : ranges) {
                range.begin = (range.begin / page) * page;
                range.end = ((range.end + page - 1) / page) * page;
            }
            std::sort(ranges.begin(), ranges.end(),
                      [](const Range& a, const Range& b) {
                          return a.begin < b.begin;
                      });
            size_t merged = 0, pages = 0;
            uintptr_t current_begin = 0, current_end = 0;
            for (const auto& range : ranges) {
                if (current_end < range.begin) {
                    if (current_end) {
                        ++merged;
                        pages += (current_end - current_begin) / page;
                    }
                    current_begin = range.begin;
                    current_end = range.end;
                } else {
                    current_end = std::max(current_end, range.end);
                }
            }
            if (current_end) {
                ++merged;
                pages += (current_end - current_begin) / page;
            }
            std::fprintf(stderr,
                         "SHM copy source coverage: raw_ranges=%zu logical_bytes=%zu "
                         "pages=%zu merged_ranges=%zu\n",
                         ranges.size(), logical_bytes, pages, merged);
            std::fflush(stderr);
            for (size_t gap : {size_t(65536), size_t(262144), size_t(1048576)}) {
                size_t count = 0, covered = 0;
                uintptr_t begin = 0, end = 0;
                for (const auto& range : ranges) {
                    if (end && range.begin > end && range.begin - end > gap) {
                        ++count;
                        covered += end - begin;
                        begin = range.begin;
                        end = range.end;
                    } else {
                        if (!end) begin = range.begin;
                        end = std::max(end, range.end);
                    }
                }
                if (end) { ++count; covered += end - begin; }
                std::fprintf(stderr,
                             "SHM source gap coalescing: gap_bytes=%zu "
                             "covered_bytes=%zu ranges=%zu\n", gap, covered, count);
            }
            std::fflush(stderr);
        }
        size_t required_size = 0;
        for (const auto& tasks : copy_tasks_by_layer_) {
            for (const auto& task : tasks) {
                if (!task.is_packed) continue;
                size_t tensor_bytes = element_size_;
                for (int dimension : task.gpu_shape) tensor_bytes *= dimension;
                const size_t copy_extent = task.type == CopyType::COPY_1D
                    ? task.total_bytes
                    : (task.height == 0 ? 0 :
                       (task.height - 1) * task.dst_pitch_bytes + task.width_bytes);
                required_size = std::max(required_size,
                    std::max(tensor_bytes, task.gpu_task_offset + copy_extent));
            }
        }
        mark("staging_extent");
        if (required_size > buffer_size_) {
            // Reconfiguration must not free staging storage still in use.
            for (auto& future : load_layers_futures_) future.wait();
            mark("future_drain");
            CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
            mark("copy_drain");
            CUDA_CHECK(cudaStreamSynchronize(unpack_stream_));
            mark("unpack_drain");
            for (int i = 0; i < 2; ++i) {
                CUDA_CHECK(cudaFree(buffer_ptrs_[i]));
                mark(i == 0 ? "free_staging_0" : "free_staging_1");
                buffer_ptrs_[i] = nullptr;
                CUDA_CHECK(cudaMalloc(&buffer_ptrs_[i], required_size));
                mark(i == 0 ? "allocate_staging_0" : "allocate_staging_1");
            }
            buffer_size_ = required_size;
        }
    }
    catch (const std::exception& e) {
        fprintf(stderr, "\nBACKGROUND THREAD ERROR\n%s\n", e.what());
        fflush(stderr);
        exit(EXIT_FAILURE); 
    }
}

void ModelManager::load_layers(const std::vector<int>& layer_idxes) {  
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    try {
        // Keep DMA asynchronous during model construction and warmup.  A
        // fence for every tensor serializes PCIe with the host and makes
        // initialization sensitive to tensor count and rank scheduling.  A
        // bounded batch fence still yields the bus to CUDA module/KV work
        // without destroying the copy/depack pipeline.
        size_t throttle_batch_tasks = 32;
        if (const char* value = std::getenv("VLLM_SHM_COPY_THROTTLE_TASKS")) {
            std::string text(value);
            if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("VLLM_SHM_COPY_THROTTLE_TASKS must be a positive integer");
            throttle_batch_tasks = std::stoull(text);
            if (throttle_batch_tasks == 0)
                throw std::invalid_argument("VLLM_SHM_COPY_THROTTLE_TASKS must be positive");
        }
        size_t throttle_tasks = 0;
        for (int layer_idx : layer_idxes) {
            if (!layer_start_events_[layer_idx])
                CUDA_CHECK(cudaEventCreate(&layer_start_events_[layer_idx]));
            CUDA_CHECK(cudaEventRecord(layer_start_events_[layer_idx], copy_stream_));
            const auto& copy_tasks = copy_tasks_by_layer_[layer_idx];
            for (const auto& task : copy_tasks) {
                void* task_dst = task.dst;
                //-------------------------------------------------
                if (task.is_packed) {
                    task_dst = (uint8_t*)buffer_ptrs_[buffer_idx_] + task.gpu_task_offset;
                    // Bound host submissions throughout loading, including
                    // inference. Stream waits alone queue the whole recovery
                    // prefix and block inference kernel submissions in CUDA.
                    // Waiting only for this buffer preserves copy/unpack
                    // overlap while limiting the pending work to two tensors.
                    if (std::getenv("VLLM_SHM_UNTHROTTLED_COPY") == nullptr) {
                        CUDA_CHECK(cudaEventSynchronize(unpack_buffer_events_[buffer_idx_]));
                    }
                    CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, unpack_buffer_events_[buffer_idx_], 0));
                }
                //-------------------------------------------------

                if (task.type == CopyType::COPY_1D) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        task_dst, 
                        task.src, 
                        task.total_bytes, 
                        cudaMemcpyHostToDevice, 
                        copy_stream_
                    ));
                } else {
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        task_dst, 
                        task.dst_pitch_bytes, 
                        task.src, 
                        task.src_pitch_bytes, 
                        task.width_bytes, 
                        task.height, 
                        cudaMemcpyHostToDevice, 
                        copy_stream_
                    ));
                }

                //-----------------------------------------------------------------
                if (task.tensor_end && task.is_packed) {
                    CUDA_CHECK(cudaEventRecord(copy_buffer_events_[buffer_idx_], copy_stream_));
                    CUDA_CHECK(cudaStreamWaitEvent(unpack_stream_, copy_buffer_events_[buffer_idx_], 0));
                    launch_unpack_weight(buffer_ptrs_[buffer_idx_], task.dst_tensor_base,
                                            task.gpu_shape[0], task.gpu_shape[1], unpack_stream_);
                    CUDA_CHECK(cudaEventRecord(unpack_buffer_events_[buffer_idx_], unpack_stream_));

                    buffer_idx_ = 1 - buffer_idx_;
                }            
                //------------------------------------------------------------------

                // Structure construction binds storage without consuming
                // weight values. Keep transfers concurrent in that phase;
                // protect KV initialization and CUDA-module warmup below.
                if ((is_kv_cache_init_ || is_module_loading_) &&
                    std::getenv("VLLM_SHM_UNTHROTTLED_COPY") == nullptr) {
                    ++throttle_tasks;
                    if (throttle_tasks >= throttle_batch_tasks) {
                        // Preserve a bounded DMA completion boundary without
                        // waiting once per tensor.
                        CUDA_CHECK(cudaEventRecord(copy_throttle_event_, copy_stream_));
                        CUDA_CHECK(cudaEventSynchronize(copy_throttle_event_));
                        throttle_tasks = 0;
                    }
                }
            }
            record_layer_complete(layer_idx);
        }
        if (throttle_tasks != 0 && (is_kv_cache_init_ || is_module_loading_) &&
            std::getenv("VLLM_SHM_UNTHROTTLED_COPY") == nullptr) {
            CUDA_CHECK(cudaEventRecord(copy_throttle_event_, copy_stream_));
            CUDA_CHECK(cudaEventSynchronize(copy_throttle_event_));
        }
    }
    catch (...) {
        record_loading_error(std::current_exception());
        throw;
    }
}

void ModelManager::load_layers_sync(const std::vector<int>& layer_idxes) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (closed_) throw std::runtime_error("Model manager is closed");
    reserve_layer_loads(layer_idxes);
    CUDA_CHECK(cudaSetDevice(device_));
    // Both entry points use the same double-buffered copy/unpack pipeline.
    load_layers(layer_idxes);
    synchronize(layer_idxes);
}

void ModelManager::record_layer_complete(int layer_idx) {
    if (!copy_layer_events_[layer_idx])
        CUDA_CHECK(cudaEventCreateWithFlags(&copy_layer_events_[layer_idx], cudaEventDisableTiming));
    if (!unpack_layer_events_[layer_idx])
        CUDA_CHECK(cudaEventCreate(&unpack_layer_events_[layer_idx]));
    CUDA_CHECK(cudaEventRecord(copy_layer_events_[layer_idx], copy_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(unpack_stream_, copy_layer_events_[layer_idx], 0));
    CUDA_CHECK(cudaEventRecord(unpack_layer_events_[layer_idx], unpack_stream_));
    {
        std::lock_guard<std::mutex> lock(readiness_mutex_);
        layer_events_recorded_[layer_idx] = true;
        --pending_layer_loads_[layer_idx];
    }
    readiness_changed_.notify_all();
}

void ModelManager::reserve_layer_loads(const std::vector<int>& layer_idxes) {
    std::lock_guard<std::mutex> lock(readiness_mutex_);
    if (loading_error_) std::rethrow_exception(loading_error_);
    // Validate the complete request before changing any readiness state.
    for (int layer_idx : layer_idxes) pending_layer_loads_.at(layer_idx);
    for (int layer_idx : layer_idxes) {
        ++pending_layer_loads_[layer_idx];
        layer_completion_observed_[layer_idx] = false;
    }
}

void ModelManager::load_layers_async(const std::vector<int>& layer_idxes) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (closed_) throw std::runtime_error("Model manager is closed");
    reserve_layer_loads(layer_idxes);
    try {
      load_layers_futures_.push_back(std::async(std::launch::async, [this, layer_idxes]() {
        try {
            CUDA_CHECK(cudaSetDevice(device_));
            load_layers(layer_idxes);
        } catch (...) {
            record_loading_error(std::current_exception());
            throw;
        }
      }));
    } catch (...) {
        record_loading_error(std::current_exception());
        throw;
    }
}

void ModelManager::record_loading_error(std::exception_ptr error) {
    {
        std::lock_guard<std::mutex> lock(readiness_mutex_);
        if (!loading_error_) loading_error_ = error;
    }
    readiness_changed_.notify_all();
}

void ModelManager::close() {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    closed_ = true;
    CUDA_CHECK(cudaSetDevice(device_));
    // Every producer must finish before storage is released, even when an
    // earlier producer failed. Surface that failure only after draining.
    std::exception_ptr error;
    for (auto& future : load_layers_futures_) if (future.valid()) {
        try { future.get(); }
        catch (...) { if (!error) error = std::current_exception(); }
    }
    load_layers_futures_.clear();
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    CUDA_CHECK(cudaStreamSynchronize(unpack_stream_));
    cpu_memory_base_ptrs_.clear();
    gpu_memory_base_ptrs_.clear();
    cuda_allocations_.clear();
    if (error) std::rethrow_exception(error);
}

void ModelManager::synchronize(const std::vector<int>& layer_idxes) {
    CUDA_CHECK(cudaSetDevice(device_));
    for (auto layer_idx : layer_idxes) {
        std::unique_lock<std::mutex> lock(readiness_mutex_);
        if (loading_error_) std::rethrow_exception(loading_error_);
        if (layer_completion_observed_.at(layer_idx)) continue;
        readiness_changed_.wait(lock, [this, layer_idx] {
            return loading_error_ || (layer_events_recorded_.at(layer_idx) && pending_layer_loads_.at(layer_idx) == 0);
        });
        if (loading_error_) std::rethrow_exception(loading_error_);
        if (copy_layer_events_[layer_idx])
            CUDA_CHECK(cudaEventSynchronize(copy_layer_events_[layer_idx]));
        //------------------------------------
        if (unpack_layer_events_[layer_idx])
            CUDA_CHECK(cudaEventSynchronize(unpack_layer_events_[layer_idx]));
        layer_completion_observed_[layer_idx] = true;
        //------------------------------------
    }
}

bool ModelManager::check_load_state(const std::vector<int>& layer_idxes) {
    CUDA_CHECK(cudaSetDevice(device_));
    std::lock_guard<std::mutex> lock(readiness_mutex_);
    if (loading_error_) std::rethrow_exception(loading_error_);
    // 检查layer_idxes中的所有layer是否已经加载完成
    for (auto layer_idx : layer_idxes) {
        if (layer_completion_observed_.at(layer_idx)) continue;
        if (!layer_events_recorded_.at(layer_idx) || pending_layer_loads_.at(layer_idx)) return false;
        for (auto event : {copy_layer_events_[layer_idx], unpack_layer_events_[layer_idx]}) {
            if (!event) continue;
            const auto status = cudaEventQuery(event);
            if (status == cudaErrorNotReady) return false;
            CUDA_CHECK(status);
        }
        layer_completion_observed_[layer_idx] = true;
    }
    return true;
}

std::vector<double> ModelManager::completed_layer_load_seconds() {
    CUDA_CHECK(cudaSetDevice(device_));
    std::lock_guard<std::mutex> lock(readiness_mutex_);
    if (loading_error_) std::rethrow_exception(loading_error_);
    std::vector<double> seconds;
    for (int layer = 0; layer <= total_layers_; ++layer) {
        if (!layer_events_recorded_[layer] || pending_layer_loads_[layer] ||
            !layer_start_events_[layer]) continue;
        auto status = cudaEventQuery(unpack_layer_events_[layer]);
        if (status == cudaErrorNotReady) continue;
        CUDA_CHECK(status);
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, layer_start_events_[layer],
                                       unpack_layer_events_[layer]));
        if (ms > 0) seconds.push_back(ms / 1000.0);
    }
    return seconds;
}

std::vector<std::tuple<int, uint64_t, double>> ModelManager::completed_layer_load_records() {
    CUDA_CHECK(cudaSetDevice(device_));
    std::lock_guard<std::mutex> lock(readiness_mutex_);
    if (loading_error_) std::rethrow_exception(loading_error_);
    std::vector<std::tuple<int, uint64_t, double>> records;
    for (int layer = 0; layer <= total_layers_; ++layer) {
        if (!layer_events_recorded_[layer] || pending_layer_loads_[layer] ||
            !layer_start_events_[layer]) continue;
        auto status = cudaEventQuery(unpack_layer_events_[layer]);
        if (status == cudaErrorNotReady) continue;
        CUDA_CHECK(status);
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, layer_start_events_[layer],
                                       unpack_layer_events_[layer]));
        if (ms > 0) {
            uint64_t bytes = 0;
            for (const auto& task : copy_tasks_by_layer_[layer])
                bytes += task.type == CopyType::COPY_2D
                    ? task.width_bytes * task.height : task.total_bytes;
            records.emplace_back(layer, bytes, ms / 1000.0);
        }
    }
    return records;
}

std::unordered_map<std::string, torch::Tensor> ModelManager::RestoreTensors(
    const std::unordered_map<
        std::string, std::tuple<uint64_t, std::vector<int64_t>, std::vector<int64_t>,
                                std::string>>& meta_state_dict,
    const int device, const uint64_t base_address) {
    auto allocation = cuda_allocations_.at(base_address);
    std::unordered_map<std::string, torch::Tensor> state_dict;
    for (const auto& p : meta_state_dict) {
        std::string name = p.first;
        auto [offset, sizes, strides, type_str] = p.second;
        uint64_t src_ptr = base_address + offset;

        torch::Device tensor_device(torch::kCUDA, device);
        at::ScalarType dtype = stringToScalarType(type_str);
        state_dict[name] = torch::from_blob(
            reinterpret_cast<void*>(src_ptr), sizes, strides,
            [allocation](void*) {},
            torch::TensorOptions().device(tensor_device).dtype(dtype));
    }
    return state_dict;
}

uint64_t ModelManager::AllocateCudaMemory(const int device, const size_t tensor_size) {
    return AllocateCudaArena(device, {tensor_size}).at(0);
}

std::vector<uint64_t> ModelManager::AllocateCudaArena(
    const int device, const std::vector<size_t>& sizes) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (closed_) throw std::runtime_error("Model manager is closed");
    if (device != device_)
        throw std::invalid_argument("Allocation device must match model manager device");
    constexpr size_t alignment = 256;
    const size_t limit = std::numeric_limits<size_t>::max();
    std::vector<uint64_t> offsets;
    offsets.reserve(sizes.size());
    size_t total = 0;
    for (size_t size : sizes) {
        if (!size) throw std::invalid_argument("Arena ranges must be nonempty");
        if (total > limit - (alignment - 1))
            throw std::overflow_error("CUDA arena alignment overflow");
        total = (total + alignment - 1) & ~(alignment - 1);
        offsets.push_back(total);
        if (size > limit - total)
            throw std::overflow_error("CUDA arena size overflow");
        total += size;
    }
    if (sizes.empty()) return {};
    const bool profile = std::getenv("VLLM_PROFILE_RELAY") != nullptr;
    const auto guard_start = std::chrono::steady_clock::now();
    c10::cuda::CUDAGuard device_guard(device);
    const auto allocation_start = std::chrono::steady_clock::now();
    void* memory_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&memory_ptr, total));
    if (profile) {
        const auto allocation_end = std::chrono::steady_clock::now();
        std::fprintf(stderr,
                     "SHM arena: rank=%d bytes=%zu guard_ms=%.3f malloc_ms=%.3f\n",
                     gpu_tp_rank_, total,
                     std::chrono::duration<double, std::milli>(
                         allocation_start - guard_start).count(),
                     std::chrono::duration<double, std::milli>(
                         allocation_end - allocation_start).count());
    }
    auto allocation = std::shared_ptr<void>(memory_ptr, [device](void* ptr) {
        int previous_device = device;
        cudaGetDevice(&previous_device);
        cudaSetDevice(device);
        cudaFree(ptr);
        cudaSetDevice(previous_device);
    });
    // Build the ownership index before publishing any bases. A failed map
    // allocation frees the arena and leaves existing allocations unchanged.
    auto allocations = cuda_allocations_;
    const auto base = reinterpret_cast<uint64_t>(memory_ptr);
    for (auto& offset : offsets) {
        offset += base;
        allocations.emplace(offset, allocation);
    }
    cuda_allocations_.swap(allocations);
    return offsets;
}

std::unordered_map<std::string, uint64_t> SaveTensors(
    std::vector<std::string> tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path) {
  std::string tensor_filename = std::filesystem::path(path) / "tensor.data";
  // make a tensor writer
  TensorWriter writer(tensor_filename);
  // make a tensor index
  std::unordered_map<std::string, uint64_t> tensor_offsets;
  // Some tensors may share the same data, so we need to record the data to
  // avoid duplication
  std::unordered_map<const char*, std::string> data_record;

  int total = tensor_names.size();
  int count = 0;

  for (const auto& name : tensor_names) {
    const auto& [base, size] = tensor_data[name];
    const char* data_ptr = reinterpret_cast<const char*>(base);
    if (data_record.find(data_ptr) != data_record.end()) {
      tensor_offsets[name] = tensor_offsets[data_record[data_ptr]];
      continue;
    }
    data_record[data_ptr] = name;

    uint64_t offset = writer.writeRecord(data_ptr, size);
    tensor_offsets[name] = offset;

    // Update progress bar
    count++;
    float progress = static_cast<float>(count) / total;
    showProgressBar(progress, "Saving tensors: ");
  }

  return tensor_offsets;
}
