#include "hidden_states_manager.h"
#include <stdexcept>

HiddenStatesManager::HiddenStatesManager(const int hidden_size, const int dtype_size) {
    dtype_size_ = dtype_size;
    hidden_size_ = hidden_size;
    cudaGetDevice(&device_);
    cudaStreamCreate(&copy_stream_);
}

HiddenStatesManager::~HiddenStatesManager() {
    cudaSetDevice(device_);
    cudaStreamSynchronize(copy_stream_);
    cudaStreamDestroy(copy_stream_);
}

void HiddenStatesManager::close() {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    closed_ = true;
    cudaSetDevice(device_);
    auto result = cudaStreamSynchronize(copy_stream_);
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

void HiddenStatesManager::update_shm_ptr_info(
    uint64_t shm_ptr, uint64_t residual_offset) {
    shm_ptr_ = shm_ptr;
    residual_offset_ = residual_offset;
}

void HiddenStatesManager::update_gpu_ptr_info(
    uint64_t gpu_hidden_states_ptr,
    uint64_t gpu_residual_ptr
) {
    gpu_hidden_states_ptr_ = gpu_hidden_states_ptr;
    gpu_residual_ptr_ = gpu_residual_ptr;
}

void HiddenStatesManager::load_hidden_states(
    int num_input_tokens
) {
    load_hidden_state_spans({{0, 0, num_input_tokens}});
}

void HiddenStatesManager::load_hidden_state_spans(
    const std::vector<std::vector<int>>& spans) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (closed_) throw std::runtime_error("Hidden state manager is closed");
    cudaSetDevice(device_);
    const size_t row_bytes = size_t(hidden_size_) * dtype_size_;
    for (const auto& span : spans) {
        if (span.size() != 3 || span[0] < 0 || span[1] < 0 || span[2] < 0)
            throw std::invalid_argument("Invalid hidden-state copy span");
    }
    // Coalesce adjacent source/target ranges before enqueueing. Layerwise
    // recovery often produces one span per request; contiguous requests can
    // be transferred with one DMA submission per buffer without changing the
    // layout or synchronization semantics.
    for (size_t index = 0; index < spans.size();) {
        const size_t source_row = size_t(spans[index][0]);
        const size_t target_row = size_t(spans[index][1]);
        size_t count = size_t(spans[index][2]);
        ++index;
        while (index < spans.size()
                && source_row + count == size_t(spans[index][0])
                && target_row + count == size_t(spans[index][1])) {
            count += size_t(spans[index][2]);
            ++index;
        }
        const size_t source = source_row * row_bytes;
        const size_t target = target_row * row_bytes;
        const size_t bytes = count * row_bytes;
        for (int residual = 0; residual < 2; ++residual) {
            const auto destination = residual ? gpu_residual_ptr_ : gpu_hidden_states_ptr_;
            const auto origin = shm_ptr_ + (residual ? residual_offset_ : 0);
            auto result = cudaMemcpyAsync((void*)(destination + target),
                (void*)(origin + source), bytes, cudaMemcpyHostToDevice, copy_stream_);
            if (result != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(result));
        }
    }
    auto result = cudaStreamSynchronize(copy_stream_);
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}
