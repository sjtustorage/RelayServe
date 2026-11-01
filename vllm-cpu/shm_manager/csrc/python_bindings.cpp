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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>
#include <cstdint>

#include "model_manager.h"
#include "kvcache_manager.h"
#include "hidden_states_manager.h"
namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("read_file_regions", &ReadFileRegions, py::arg("regions"),
        py::arg("num_threads") = 4, py::arg("cpu_bind") = "",
        py::call_guard<py::gil_scoped_release>());
  py::class_<ModelManager>(m, "ModelManager")
      .def(py::init<const std::string&, size_t, size_t, int>(),
           py::arg("name"),
           py::arg("model_size"),
           py::arg("chunk_size"),
           py::arg("num_threads"))
      .def("restore_tensors", &ModelManager::RestoreTensors,
           py::arg("meta_state_dict"),
           "Restore tensors from metadata and device offsets.")
      .def("load_model_to_host", &ModelManager::LoadModelToHost,
           "Load model to CPU and return status code.")
      .def("destroy_model", &ModelManager::DestroyModel,
           "Destroy model.");

  py::class_<KVCacheManager>(m, "KVCacheManager")
    .def(py::init<const std::string&, size_t>(),
         py::arg("name"),
         py::arg("total_size"))
    .def("restore_kvcache", &KVCacheManager::RestoreKVCache,
         py::arg("meta_state_dict"),
         "Restore KV cache from metadata and device offsets.")
    .def("destroy_kvcache", &KVCacheManager::DestroyKVCache,
         "Destroy KV cache.");

  py::class_<HiddenStatesManager>(m, "HiddenStatesManager")
    .def(py::init<const std::string&, size_t>(),
         py::arg("name"),
         py::arg("total_size"))
    .def("restore_hidden_states", &HiddenStatesManager::RestoreHiddenStates,
         py::arg("meta_state_dict"),
         "Restore hidden states from metadata and device offsets.")
    .def("destroy_hidden_states", &HiddenStatesManager::DestroyHiddenStates,
         "Destroy hidden states.");

  m.def("save_tensors", &SaveTensors,
        "Save tensors to disk and return device offsets.");

  m.def("get_shared_memory", &GetSharedMemoryPtr,
        "Get shared memory pointer by name and device memory size.");
}
