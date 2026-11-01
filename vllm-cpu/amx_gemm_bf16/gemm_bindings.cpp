#include "gemm.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_weight", [](torch::Tensor w) { return amx_bf16::PackWeight(w); },
          py::arg("weight"));
    m.def("pack_weight_nk", [](torch::Tensor w) { amx_bf16::PackWeightNK(w); },
          py::arg("weight"));
    m.def("gemm_bf16",
          [](torch::Tensor A, torch::Tensor packed_weight, torch::Tensor C,
             py::object bias_obj,
             int TM, int TN, int TK,
             bool prefetch_a, bool prefetch_b, bool prefetch_c,
             int pf_a_dist, int pf_b_dist, int pf_c_dist,
             bool hw_prefetch) {
              torch::Tensor bias;
              if (!bias_obj.is_none()) {
                  bias = bias_obj.cast<torch::Tensor>();
              }
              amx_bf16::GemmBF16(A, packed_weight, C, bias, TM, TN, TK,
                                 prefetch_a, prefetch_b, prefetch_c,
                                 pf_a_dist, pf_b_dist, pf_c_dist,
                                 hw_prefetch);
          },
          "vLLM AMX GEMM (bf16 output, optional bias)",
          py::arg("A"), py::arg("B_packed"), py::arg("C"),
          py::arg("bias") = py::none(),
          py::arg("TM") = -1, py::arg("TN") = -1, py::arg("TK") = 1376,
          py::arg("prefetch_a") = false, py::arg("prefetch_b") = false, py::arg("prefetch_c") = false,
          py::arg("pf_a_dist") = 2, py::arg("pf_b_dist") = 2, py::arg("pf_c_dist") = 1,
          py::arg("hw_prefetch") = true,
          py::call_guard<py::gil_scoped_release>());
    m.def("gemm_profile",
          [](torch::Tensor A, torch::Tensor packed_weight, torch::Tensor C,
             int TM, int TN, int TK,
             bool prefetch_a, bool prefetch_b, bool prefetch_c,
             int pf_a_dist, int pf_b_dist, int pf_c_dist) {
              return amx_bf16::GemmProfile(A, packed_weight, C, TM, TN, TK,
                                           prefetch_a, prefetch_b, prefetch_c,
                                           pf_a_dist, pf_b_dist, pf_c_dist);
          },
          "vLLM AMX GEMM profile counters",
          py::arg("A"), py::arg("B_packed"), py::arg("C"),
          py::arg("TM") = -1, py::arg("TN") = -1, py::arg("TK") = 1376,
          py::arg("prefetch_a") = false, py::arg("prefetch_b") = false, py::arg("prefetch_c") = false,
          py::arg("pf_a_dist") = 2, py::arg("pf_b_dist") = 2, py::arg("pf_c_dist") = 1);
    m.def("init_cpu_threads_env", &init_cpu_threads_env);
    m.def("set_hw_prefetch", &amx_bf16::set_hw_prefetch,
          "Enable/disable CPU hardware prefetchers via MSR 0x1A4",
          py::arg("enable"));
}
