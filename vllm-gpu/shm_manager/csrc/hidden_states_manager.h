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

class HiddenStatesManager {
public:
    HiddenStatesManager(const int hidden_size, const int dtype_size);
    ~HiddenStatesManager();
    void close();

    void update_shm_ptr_info(
        uint64_t shm_ptr, uint64_t residual_offset
    );

    void update_gpu_ptr_info(
        uint64_t gpu_hidden_states_ptr,
        uint64_t gpu_residual_ptr
    );

    void load_hidden_states(
        int num_input_tokens
    );
    void load_hidden_state_spans(const std::vector<std::vector<int>>& spans);

private:
    int hidden_size_;
    int dtype_size_;
    uint64_t shm_ptr_;
    uint64_t residual_offset_;
    uint64_t gpu_hidden_states_ptr_;
    uint64_t gpu_residual_ptr_;
    cudaStream_t copy_stream_;
    int device_;
    bool closed_ = false;
    std::mutex lifecycle_mutex_;
};
