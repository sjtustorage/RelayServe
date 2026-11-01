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
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "model_manager.h"
#include "kvcache_manager.h"
#include "hidden_states_manager.h"
#include "shm_extension.h"


namespace py = pybind11;

// define pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Public capability marker consumed by production interference audits.
    // Version 2 means async submission uses the copy/depack completion fences
    // exposed by ModelManager::synchronize and check_load_state.
    m.attr("async_load_fence_version") = 2;
    m.def("get_pinned_shared_memory_ptrs", &GetPinnedSharedMemoryPtrs,
          py::arg("name_map"), py::arg("device_memory"), py::arg("cacheable") = false)
      .def("set_pinned_weight_cache_budget", &SetPinnedWeightCacheBudget)
      .def("pinned_weight_cache_stats", &PinnedWeightCacheStats)
      .def("destroy_shared_memory", &DestroySharedMemory, "Destroy shared memory")
      .def("release_pinned_shared_memory", &ReleasePinnedSharedMemory,
           "Detach a consumer mapping after its transfers finish; do not unlink SHM")
      .def("save_tensors", &SaveTensors, "Save a state dict for GPU");
    // ModelManager class binding
    py::class_<ModelManager>(m, "ModelManager")
      .def("close", &ModelManager::close, py::call_guard<py::gil_scoped_release>())
      .def(py::init<int, int, int, int, int>(), 
           py::arg("element_size"), py::arg("cpu_world_size"), 
           py::arg("gpu_tp_rank"), py::arg("gpu_world_size"), py::arg("device"))
      .def("update_total_layers", &ModelManager::update_total_layers,
           py::arg("total_layers"))
      .def("update_manager_info", &ModelManager::update_manager_info,
           py::arg("cpu_memory_base_ptrs"), py::arg("gpu_memory_base_ptrs"))
      .def("load_layers_async", &ModelManager::load_layers_async, 
           py::arg("layer_idxes"), "Loads a range of layers.",
           py::call_guard<py::gil_scoped_release>())
      .def("load_layers_sync", &ModelManager::load_layers_sync,
           py::arg("layer_idxes"),
           "Loads layers and blocks until host-to-device copies and unpack finish.",
           py::call_guard<py::gil_scoped_release>())
      // Other Python threads must be able to submit work or handle aborts
      // while this thread waits for native copy completion.
      .def("synchronize", &ModelManager::synchronize,
           py::arg("layer_idxes"),
           "Waits for an execution to complete.",
           py::call_guard<py::gil_scoped_release>())
      .def("generate_tensor_metadata", &ModelManager::generate_tensor_metadata,
           py::arg("cpu_tensor_offset"), py::arg("gpu_tensor_offset"),
           py::arg("full_shape"), py::arg("gpu_shape"), py::arg("cpu_ranks"), 
           py::arg("sharding_dim"), py::arg("layer_index"), py::arg("layer_end"),
           py::arg("head_dim"), py::arg("is_packed"),
           py::arg("tensor_name"),
           "Generates tensor metadata.")
      .def("generate_copy_tasks", &ModelManager::generate_copy_tasks,
           "Generates copy tasks.", py::call_guard<py::gil_scoped_release>())
      .def("copy_source_ranges", &ModelManager::copy_source_ranges,
           py::arg("max_gap_bytes") = 0,
           "Page-aligned host source offset/length ranges, grouped by CPU rank.")
      .def("allocate_cuda_memory", &ModelManager::AllocateCudaMemory,
           py::arg("device"), py::arg("tensor_size"),
           "Allocates cuda memory.")
      .def("allocate_cuda_arena", &ModelManager::AllocateCudaArena,
           py::arg("device"), py::arg("sizes"),
           "Allocates aligned layer ranges in one shared CUDA allocation.",
           py::call_guard<py::gil_scoped_release>())
      .def("restore_tensors", &ModelManager::RestoreTensors,
           py::arg("meta_state_dict"), py::arg("device"), py::arg("base_address"),
           "Restores a state dict for GPU.")
      .def("update_module_loading_state", &ModelManager::update_module_loading_state,
           py::arg("is_module_loading"),
           "Updates the module loading state.")
      .def("update_model_arch_init_state", &ModelManager::update_model_arch_init_state,
           py::arg("is_model_arch_init"),
           py::arg("bound_submissions") = false,
           "Updates the model architecture initialization state.")
      .def("completed_layer_load_seconds", &ModelManager::completed_layer_load_seconds)
      .def("completed_layer_load_records", &ModelManager::completed_layer_load_records,
           "Nonblocking (layer_id, payload_bytes, copy_to_depack_seconds) snapshots; "
           "includes stream waits and throttling, not isolated PCIe time.")
      .def("check_load_state", &ModelManager::check_load_state,
           "Returns true if all async layer loads have completed.");

    // KVCacheManager class binding
    py::class_<KVCacheManager>(m, "KVCacheManager")
      .def("close", &KVCacheManager::close, py::call_guard<py::gil_scoped_release>())
      .def(py::init<int, int, int, int, int, int, int, int, int>(), 
           py::arg("total_layers"), py::arg("gpu_tp_rank"), py::arg("gpu_tp_size"), py::arg("cpu_tp_size"),
           py::arg("shm_num_blocks"), py::arg("shm_block_len"), py::arg("block_size"), py::arg("head_dim"),
           py::arg("device"),
           "Initializes the KVCacheManager.")
      .def("update_shm_ptr_info", &KVCacheManager::update_shm_ptr_info,
           py::arg("cpu_tp_ranks"), py::arg("shm_caches_ptrs"),
           "Updates the pointer information.")
      .def("update_gpu_ptr_info", &KVCacheManager::update_gpu_ptr_info,
           py::arg("gpu_num_blocks"), py::arg("gpu_block_len"),
           py::arg("total_num_kv_heads"), py::arg("gpu_caches_ptrs"),
           "Updates the pointer information.")
      .def("load_kv_caches_async", &KVCacheManager::load_kv_caches_async,
           py::arg("gpu_block_ids"), py::arg("cpu_block_ids"), py::arg("layer_ids"),
           "Loads KV caches asynchronously.")
      .def("check_load_state", &KVCacheManager::check_load_state, "Checks the load state.")
      .def("synchronize", &KVCacheManager::synchronize, "Synchronizes KV caches.");

    
    py::class_<HiddenStatesManager>(m, "HiddenStatesManager")
      .def("close", &HiddenStatesManager::close, py::call_guard<py::gil_scoped_release>())
      .def(py::init<int, int>(), py::arg("hidden_size"), py::arg("dtype_size"),
           "Initializes the HiddenStatesManager.")
      .def("update_shm_ptr_info", &HiddenStatesManager::update_shm_ptr_info,
           py::arg("shm_ptr"), py::arg("residual_offset"),
           "Updates the pointer information.")
      .def("update_gpu_ptr_info", &HiddenStatesManager::update_gpu_ptr_info,
           py::arg("gpu_hidden_states_ptr"), py::arg("gpu_residual_ptr"),
           "Updates the pointer information.")
      .def("load_hidden_state_spans", &HiddenStatesManager::load_hidden_state_spans)
      .def("load_hidden_states", &HiddenStatesManager::load_hidden_states,
           py::arg("num_input_tokens"),
           "Loads hidden states.");
}
