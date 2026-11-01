// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
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

void ReadFileRegions(const std::vector<std::tuple<std::string, uint64_t,
                     uint64_t, uint64_t>>& regions, int num_threads,
                     const std::string& cpu_bind);

class ModelManager {
public:
    ModelManager(const std::string& name, size_t model_size, size_t chunk_size, int num_threads);

    std::unordered_map<std::string, torch::Tensor> RestoreTensors(
        const std::unordered_map<
            std::string, std::tuple<uint64_t,std::vector<int64_t>, 
                     std::vector<int64_t>, std::string>>& meta_state_dict);

    int LoadModelToHost();

    void DestroyModel();
private:
    std::string shm_name_;
    std::string data_path_;
    size_t model_size_;
    void* mem_ptr_;
    size_t chunk_size_;
    int num_threads_;
};


std::unordered_map<std::string, uint64_t> SaveTensors(
    std::vector<std::string> tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path);
