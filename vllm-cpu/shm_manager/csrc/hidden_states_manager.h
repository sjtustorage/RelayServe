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

class HiddenStatesManager {
public:
    HiddenStatesManager(const std::string& hidden_states_name, size_t total_size);

    std::unordered_map<std::string, torch::Tensor> RestoreHiddenStates(
        const std::unordered_map<
            std::string, std::tuple<uint64_t,std::vector<int64_t>, 
                     std::vector<int64_t>>>& meta_state_dict);

    void DestroyHiddenStates();
private:
    std::string shm_name_;
    size_t total_size_;
    void* mem_ptr_;
};