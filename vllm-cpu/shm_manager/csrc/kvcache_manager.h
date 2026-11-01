#pragma once

#include <condition_variable>
#include <filesystem>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <torch/extension.h>
#include <torch/script.h>  // One-stop header.
#include <errno.h>
#include <numa.h>
#include "shm_extension.h"

class KVCacheManager {
public:
    KVCacheManager(const std::string& name, size_t total_size);

    std::unordered_map<std::string, torch::Tensor> RestoreKVCache(
        const std::unordered_map<
            std::string, std::tuple<uint64_t,std::vector<int64_t>, 
                     std::vector<int64_t>>>& meta_state_dict);

    void DestroyKVCache();
private:
    std::string shm_name_;
    size_t total_size_;
    void* mem_ptr_;
};