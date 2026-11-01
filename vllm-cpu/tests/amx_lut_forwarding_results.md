# AMX GEMM extension and prefetch distance validation

The resident CPU producer uses the repository's `amx_gemm_bf16_ext`.
With its CPU Python environment and explicit current source PYTHONPATH,
`VLLM_CPU_USE_AMX_GEMM_BF16` resolves to true, and the extension resolves to
`vllm-cpu/amx_gemm_bf16/amx_gemm_bf16_ext.cpython-312-x86_64-linux-gnu.so`.
CPU unquantized linear dispatch selects `ops.amx_gemm_bf16`, which calls
the extension with LUT TM/TN/TK and prefetch enable flags.

The wrapper currently leaves distances at the pybind defaults (A2/B2/C1)
and hardware prefetch enabled. The native interface accepts explicit
distances and hardware prefetch. Changing hardware prefetch writes MSRs on
worker CPUs and is a host policy; this validation keeps it enabled in both
arms.

Verified artifact: `/tmp/amx-lut-forwarding32-verified.json`.
`tests/benchmark_amx_lut_forwarding.py` compares pybind default distances
against LUT distances, holding tile sizes and prefetch enable flags fixed.
Real Qwen3-32B CPU TP2 matrix dimensions, synthetic BF16 values, M4096,
CPU96--142, five warmups and30 randomized adjacent pairs per matrix.
This measures GEMM only, not end-to-end inference or model loading.

| Matrix | N | K | Default median (ms) | LUT median (ms) | LUT/default |
| --- | --- | --- | --- | --- | --- |
| QKV | 5376 | 5120 | 12.6475 | 12.5187 | 0.9898 |
| O | 5120 | 4096 | 11.1921 | 11.1514 | 0.9964 |
| Gate/up | 25600 | 5120 | 57.1012 | 58.3927 | 1.0226 |
| Down | 5120 | 12800 | 30.4353 | 30.4462 | 1.0004 |

Every output matches bit-for-bit between both arms. Gate/up LUT distances
regress by2.26%; the other changes are small and do not establish a uniform
benefit. No production parameter forwarding is changed on this evidence.
The existing production path remains the custom AMX extension. A faster
distance policy requires renewed tuning and an end-to-end relay comparison.
