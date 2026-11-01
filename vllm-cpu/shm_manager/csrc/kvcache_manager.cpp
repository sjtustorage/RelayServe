#include "kvcache_manager.h"

KVCacheManager::KVCacheManager(const std::string& name, size_t total_size)
    : shm_name_(name), total_size_(total_size) {
    mem_ptr_ = GetSharedMemoryPtr(shm_name_, total_size_, true);
}

std::unordered_map<std::string, torch::Tensor> KVCacheManager::RestoreKVCache(
    const std::unordered_map<
        std::string, std::tuple<uint64_t, std::vector<int64_t>, std::vector<int64_t>>>& meta_state_dict) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    const std::string shm_name = shm_name_;
    void* mem_ptr = mem_ptr_;
    size_t total_size = total_size_;
    for (const auto& p : meta_state_dict) {
        std::string name = p.first;
        auto [offset, sizes, strides] = p.second;
        uint64_t src_ptr = reinterpret_cast<uint64_t>(mem_ptr) + offset; 
        torch::Device tensor_device(torch::kCPU);
        at::ScalarType dtype = torch::kInt8;
        if (offset == 0) {
            torch::Tensor real_tensor = torch::from_blob(
                reinterpret_cast<void*>(src_ptr), c10::makeArrayRef(sizes),
                    c10::makeArrayRef(strides), 
                    [shm_name, mem_ptr, total_size](void* ptr) {
                        DestroySharedMemory(shm_name, mem_ptr, total_size);},
                    torch::TensorOptions().device(tensor_device).dtype(dtype));
            state_dict[name] = real_tensor;
        } else {
            torch::Tensor real_tensor = torch::from_blob(
                reinterpret_cast<void*>(src_ptr), sizes, strides, 
                [](void* ptr) {},
                torch::TensorOptions().device(tensor_device).dtype(dtype));
            state_dict[name] = real_tensor;
        }
    }
    return state_dict;
}

void KVCacheManager::DestroyKVCache() {
    DestroySharedMemory(shm_name_, mem_ptr_, total_size_);
    mem_ptr_ = nullptr;
    total_size_ = 0;
}