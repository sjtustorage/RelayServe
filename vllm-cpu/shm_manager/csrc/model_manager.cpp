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
#include "model_manager.h"

#include "tensor_writer.h"
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <filesystem>
#include <atomic>
#include <sched.h>

#define KB (1024LL)
#define MB (1024LL * KB)
#define GB (1024LL * MB)

#define BUFFER_SIZE 1 << 30
#define PAGE_SIZE 4096

namespace fs = std::filesystem;

// Each task describes a file offset and final destination. Unlike O_DIRECT,
// buffered pread accepts unaligned safetensors headers and TP slice offsets.
void ReadFileRegions(const std::vector<std::tuple<std::string, uint64_t,
                     uint64_t, uint64_t>>& regions, int num_threads,
                     const std::string& cpu_bind) {
    if (num_threads < 1) throw std::invalid_argument("num_threads must be positive");
    std::vector<int> cpu_ids;
    if (!cpu_bind.empty() && cpu_bind != "all") {
        auto* mask = numa_parse_cpustring_all(cpu_bind.c_str());
        if (!mask) throw std::invalid_argument("Invalid reader CPU binding: " + cpu_bind);
        for (unsigned int cpu = 0; cpu < mask->size; ++cpu) {
            if (numa_bitmask_isbitset(mask, cpu)) cpu_ids.push_back(cpu);
        }
        numa_bitmask_free(mask);
        if (cpu_ids.empty() || cpu_ids.back() >= CPU_SETSIZE)
            throw std::invalid_argument("Reader CPU binding is empty or exceeds CPU_SETSIZE");
    }
    std::atomic<size_t> next{0};
    std::vector<std::future<void>> workers;
    for (size_t worker = 0; worker < std::min<size_t>(num_threads, regions.size()); ++worker) {
        workers.emplace_back(std::async(std::launch::async, [&, worker]() {
            // OpenMP binds the caller to one CPU. New std::async threads
            // inherit that affinity unless explicitly placed on rank-local
            // CPUs. Keep the caller's binding and NUMA memory policy intact.
            if (!cpu_ids.empty()) {
                cpu_set_t mask;
                CPU_ZERO(&mask);
                CPU_SET(cpu_ids[worker % cpu_ids.size()], &mask);
                if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
                    throw std::runtime_error("Cannot bind weight reader CPU: " +
                                             std::to_string(errno));
            }
            std::unordered_map<std::string, int> descriptors;
            auto close_all = [&]() { for (auto& entry : descriptors) close(entry.second); };
            try {
                while (true) {
                    size_t index = next.fetch_add(1);
                    if (index >= regions.size()) break;
                    const auto& [path, offset, address, size] = regions[index];
                    if (!address && size) throw std::invalid_argument("Null read destination");
                    auto found = descriptors.find(path);
                    if (found == descriptors.end()) {
                        int fd = open(path.c_str(), O_RDONLY);
                        if (fd < 0) throw std::runtime_error("Cannot open " + path);
                        found = descriptors.emplace(path, fd).first;
                    }
                    uint64_t done = 0;
                    while (done < size) {
                        ssize_t count = pread(found->second,
                            reinterpret_cast<char*>(address) + done,
                            std::min<uint64_t>(size - done, 64 * MB), offset + done);
                        if (count < 0 && errno == EINTR) continue;
                        if (count <= 0) throw std::runtime_error("Short or failed read: " + path);
                        done += count;
                    }
                }
            } catch (...) { close_all(); throw; }
            close_all();
        }));
    }
    std::exception_ptr error;
    for (auto& worker : workers) {
        try { worker.get(); } catch (...) { if (!error) error = std::current_exception(); }
    }
    if (error) std::rethrow_exception(error);
}


std::string format_shm_name_from_path(const std::string& full_path_str) {
    if (full_path_str.empty()) {
        return "/";
    }

    fs::path p(full_path_str);

    fs::path last_part = p.filename();
    if (last_part.empty()) {
        p = p.parent_path();
        last_part = p.filename();
    }
    
    fs::path parent_part = p.parent_path().filename();
    
    std::string combined = parent_part.string() + "-" + last_part.string();

    return "/" + combined;
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

ModelManager::ModelManager(const std::string& name, size_t model_size, size_t chunk_size, int num_threads) {
    model_size_ = model_size;
    data_path_ = name;
    shm_name_ = format_shm_name_from_path(name);
    mem_ptr_ = GetSharedMemoryPtr(shm_name_, model_size_, true);
    chunk_size_ = chunk_size;
    num_threads_ = num_threads;
}

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

std::unordered_map<std::string, torch::Tensor> ModelManager::RestoreTensors(
    const std::unordered_map<
        std::string, std::tuple<uint64_t, std::vector<int64_t>, std::vector<int64_t>,
                                std::string>>& meta_state_dict) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    const std::string shm_name = shm_name_;
    void* mem_ptr = mem_ptr_;
    size_t model_size = model_size_;
    for (const auto& p : meta_state_dict) {
        std::string name = p.first;
        auto [offset, sizes, strides, type_str] = p.second;
        uint64_t src_ptr = reinterpret_cast<uint64_t>(mem_ptr) + offset; 
        torch::Device tensor_device(torch::kCPU);
        at::ScalarType dtype = stringToScalarType(type_str);
        if (offset == 0) {
            torch::Tensor real_tensor = torch::from_blob(
                reinterpret_cast<void*>(src_ptr), c10::makeArrayRef(sizes),
                    c10::makeArrayRef(strides), 
                    [shm_name, mem_ptr, model_size](void* ptr) {
                        DestroySharedMemory(shm_name, mem_ptr, model_size);},
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

int ModelManager::LoadModelToHost() {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<size_t> partition_sizes_;
    std::vector<std::filesystem::path> partition_paths_;
    std::vector<int> file_descriptors;
    // Attempt to read from 0 until the file is not found
    for (int partition_id = 0;; ++partition_id) {
        auto tensor_path = data_path_ + "/tensor.data_" + std::to_string(partition_id);
        if (access(tensor_path.c_str(), F_OK) == -1) {
            std::cout << "Tensor file " << tensor_path << " does not exist.\n";
            break;
        }
        struct stat st;
        if (stat(tensor_path.c_str(), &st) != 0) {
            std::cout << "Failed to get file size of " << tensor_path << ".\n";
            return -1;
        }
        partition_sizes_.push_back(st.st_size);
        partition_paths_.push_back(tensor_path);

        int fd = open(tensor_path.c_str(), O_DIRECT | O_RDONLY);
        if (fd < 0) {
            std::string err = "open failed: " + tensor_path + ", error: " + strerror(errno);
            std::cout << err << ".\n";
            return -1;
        }
        file_descriptors.push_back(fd);
    }

    size_t num_chunks_ = (model_size_ + chunk_size_ - 1) / chunk_size_;

    auto host_buffers = mem_ptr_;
    std::vector<std::future<int>> futures;
    size_t chunk_per_thread = (num_chunks_ + num_threads_ - 1) / num_threads_;
    std::cout << "Loading model " << data_path_ << " to host with "
                        << num_threads_ << " threads, " << num_chunks_ << " chunks, "
                        << chunk_size_ << " chunk size, " << chunk_per_thread
                        << " chunks per thread.\n";

    for (int thread_idx = 0; thread_idx < num_threads_; ++thread_idx) {
        size_t partition_id = 0;
        size_t file_offset = thread_idx * chunk_per_thread * chunk_size_;
        while (partition_id < partition_sizes_.size() &&
                        file_offset >= partition_sizes_.at(partition_id)) {
            file_offset -= partition_sizes_.at(partition_id);
            partition_id += 1;
        }
        if (partition_id >= partition_sizes_.size()) {
            std::cout << "Thread " << thread_idx << " early exits.\n";
            return -1;
        }
        std::cout << "Thread " << thread_idx << " starting from partition "
                            << partition_id << " offset " << file_offset << ".\n";
        for (size_t chunk_idx = thread_idx * chunk_per_thread;
                    chunk_idx < (thread_idx + 1) * chunk_per_thread &&
                    chunk_idx < num_chunks_;
                    ++chunk_idx) {
            size_t size =
                    std::min(chunk_size_, model_size_ - chunk_idx * chunk_size_);
            char* dest_ptr = static_cast<char*>(host_buffers) + chunk_idx * chunk_size_;
            if (dest_ptr == nullptr) {
                std::cout << "Host buffer not allocated.\n";
                return -1;
            }

            int fd = file_descriptors[partition_id];
            ssize_t ret =
                    pread(fd, dest_ptr, size, file_offset);
            if (ret < 0) {
                auto tensor_path = partition_paths_[partition_id];
                std::cout << "pread() failed for file: " << tensor_path
                                        << ", error: " << strerror(errno);
                return -1;
            } else if (ret != size) {
                if (ret < size && partition_id + 1 < file_descriptors.size()) {
                    partition_id += 1;
                    file_offset = 0;
                    size_t remaining_size = size - ret;
                    int fd = file_descriptors[partition_id];
                    ret = pread(fd, dest_ptr + ret, remaining_size, file_offset);
                    if (ret != remaining_size) {
                        auto tensor_path = partition_paths_[partition_id];
                        std::cout << "Failed to read file: " << tensor_path
                                                << " read: " << ret << " expected: " << remaining_size;
                        return -1;
                    }
                } else {
                    auto tensor_path = partition_paths_[partition_id];
                    std::cout << "Failed to read file: " << tensor_path
                                            << " read: " << ret << " expected: " << size;
                    return -1;
                }
            }
            file_offset += ret;
        }
    }

    // close file
    for (int fd : file_descriptors) {
        close(fd);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken to load model: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms.\n";
    return 0;
}
    
void ModelManager::DestroyModel() {
    DestroySharedMemory(shm_name_, mem_ptr_, model_size_);
    mem_ptr_ = nullptr;
    model_size_ = 0;
}
