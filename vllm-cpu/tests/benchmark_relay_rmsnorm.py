"""Isolated CPU RMSNorm rounding experiment; never changes production ops.

Run with CPU .venv. Inputs come from verify_qwen_layerwise_v1 --audit-kv
on both roles; this benchmark requires the operator-capture .pt files.
"""
import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--captures', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--gpu-reference', type=Path)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
source = (repo / 'csrc/cpu/layernorm.cpp').read_text().replace(
    '#include "cpu_types.hpp"', '')
needle = 'vec_op::FP32Vec8 fp32_out = fp32_x * fp32_s_variance * fp32_w;'
assert source.count(needle) == 1
candidate = source.replace(needle,
    'scalar_vec_t rounded(fp32_x * fp32_s_variance);\n'
    'vec_op::FP32Vec8 fp32_out = vec_op::FP32Vec8(rounded) * fp32_w;')
# Specialize only BF16. Other dtype dispatches keep their original kernels.
norm_end = candidate.index('template <typename scalar_t>\nvoid fused_add')
wide_norm = candidate[candidate.index('template <typename scalar_t>'):norm_end]
wide_norm = wide_norm.replace('template <typename scalar_t>', 'template <>')
wide_norm = wide_norm.replace('rms_norm_impl(', 'rms_norm_impl<c10::BFloat16>(')
wide_norm = wide_norm.replace('scalar_t', 'c10::BFloat16')
wide_norm = wide_norm.replace('vec_op::vec_t<c10::BFloat16>', 'vec_op::BF16Vec16')
wide_norm = wide_norm.replace('FP32Vec8', 'FP32Vec16')
wide_norm = wide_norm.replace('variance = variance + fp32_x * fp32_x;',
    'variance.reg = _mm512_fmadd_ps(fp32_x.reg, fp32_x.reg, variance.reg);')
wide = candidate[:norm_end] + wide_norm + candidate[norm_end:]
fused_begin = source.index('template <typename scalar_t>\nvoid fused_add')
fused_end = source.index('}  // namespace')
wide_fused = source[fused_begin:fused_end]
wide_fused = wide_fused.replace('template <typename scalar_t>', 'template <>')
wide_fused = wide_fused.replace('fused_add_rms_norm_impl(',
                              'fused_add_rms_norm_impl<c10::BFloat16>(')
wide_fused = wide_fused.replace('scalar_t', 'c10::BFloat16')
wide_fused = wide_fused.replace('vec_op::vec_t<c10::BFloat16>', 'vec_op::BF16Vec16')
wide_fused = wide_fused.replace('FP32Vec8', 'FP32Vec16')
wide_fused = wide_fused.replace(
    'variance = variance + fp32_x * fp32_x;\n      scalar_vec_t out(fp32_x);',
    'scalar_vec_t out(fp32_x);\n'
    '      fp32_x = vec_op::FP32Vec16(out);\n'
    '      variance = variance + fp32_x * fp32_x;')
wide_fused = wide_fused.replace(
    'vec_op::FP32Vec16 fp32_out = fp32_res * fp32_s_variance * fp32_w;',
    'scalar_vec_t rounded(fp32_res * fp32_s_variance);\n'
    '      vec_op::FP32Vec16 fp32_out = vec_op::FP32Vec16(rounded) * fp32_w;')
wide_fused = wide_fused.replace('variance = variance + fp32_x * fp32_x;',
    'variance.reg = _mm512_fmadd_ps(fp32_x.reg, fp32_x.reg, variance.reg);')


def unroll_variance(kernel):
    """Two independent accumulation chains, with a 16-element tail."""
    begin = kernel.index('    for (int j = 0;')
    end = kernel.index('\n    float s_variance', begin)
    loop = kernel[begin:end]
    body = loop[loop.index('{') + 1:loop.rindex('}')]
    high = body.replace(' + j', ' + j + VEC_ELEM_NUM').replace(
        'variance.reg', 'variance_high.reg')
    replacement = (
        '    vec_op::FP32Vec16 variance_high(0.0);\n'
        '    int j = 0;\n'
        '    for (; j + 2 * VEC_ELEM_NUM <= hidden_size; j += 2 * VEC_ELEM_NUM) {\n'
        '      {' + body + '\n      }\n'
        '      {' + high + '\n      }\n'
        '    }\n'
        '    if (j < hidden_size) {' + body + '\n    }\n'
        '    variance = variance + variance_high;\n')
    return kernel[:begin] + replacement + kernel[end:]


wide = wide.replace(wide_norm, unroll_variance(wide_norm))
wide_fused = unroll_variance(wide_fused)
wide = wide[:wide.index('}  // namespace')] + wide_fused + wide[wide.index('}  // namespace'):]
labels = ('baseline', 'candidate', 'wide')
with tempfile.TemporaryDirectory(prefix='relay-norm-') as directory:
    cpp = Path(directory) / 'compare.cpp'
    cpp.write_text('#include "cpu_types.hpp"\n#include <torch/library.h>\n'
        + 'namespace baseline {\n' + source + '\n}\n'
        + 'namespace candidate {\n' + candidate + '\n}\n'
        + 'namespace wide {\n' + wide + '\n}\n'
        + 'TORCH_LIBRARY(relay_norm_experiment, m) {\n'
        + 'm.def("baseline_fused(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()", baseline::fused_add_rms_norm);\n'
        + 'm.def("wide_fused(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()", wide::fused_add_rms_norm);\n'
        + 'm.def("baseline(Tensor! out, Tensor input, Tensor weight, float epsilon) -> ()", baseline::rms_norm);\n'
        + 'm.def("candidate(Tensor! out, Tensor input, Tensor weight, float epsilon) -> ()", candidate::rms_norm);\n'
        + 'm.def("wide(Tensor! out, Tensor input, Tensor weight, float epsilon) -> ()", wide::rms_norm);\n}\n')
    load(name='relay_norm_experiment', sources=[str(cpp)],
         extra_include_paths=[str(repo / 'csrc/cpu')],
         extra_cflags=['-O2', '-std=c++20', '-fopenmp', '-mf16c',
                       '-mavx512f', '-mavx512vl', '-mavx512bw', '-mavx512dq',
                       '-mavx512bf16', '-DCPU_CAPABILITY_AVX512',
                       '-DVLLM_CPU_EXTENSION'],
         extra_ldflags=['-fopenmp'], is_python_module=False)
cpu = torch.load(args.captures / 'operators-cpu-rank-0.pt', weights_only=True)
gpu = torch.load(args.captures / 'operators-gpu-rank-0.pt', weights_only=True)
index = json.loads((args.checkpoint / 'model.safetensors.index.json').read_text())
name = 'model.layers.0.input_layernorm.weight'
with safe_open(args.checkpoint / index['weight_map'][name], framework='pt') as f:
    weight = f.get_tensor(name)
result = dict(threads=torch.get_num_threads(), accuracy={}, timing_us={}, fused_timing_us={})
for label in labels:
    out = torch.empty_like(cpu['input'])
    getattr(torch.ops.relay_norm_experiment, label)(out, cpu['input'], weight, 1e-6)
    result['accuracy'][label] = {side: int((out != data['input_layernorm.0']).sum())
        for side, data in [('cpu_live', cpu), ('gpu_live', gpu)]}
assert result['accuracy']['baseline']['cpu_live'] == 0
assert result['accuracy']['candidate']['gpu_live'] == 0
assert result['accuracy']['wide']['gpu_live'] == 0
if args.gpu_reference:
    reference = torch.load(args.gpu_reference, weights_only=True)
    name = 'model.layers.0.post_attention_layernorm.weight'
    with safe_open(args.checkpoint / index['weight_map'][name], framework='pt') as f:
        fused_weight = f.get_tensor(name)
    for label in ('baseline', 'wide'):
        x = cpu['self_attn.0'].clone()
        residual = cpu['input'].clone()
        getattr(torch.ops.relay_norm_experiment, label + '_fused')(
            x, residual, fused_weight, 1e-6)
        result['accuracy'][label + '_fused'] = dict(
            output_unequal=int((x != reference['output']).sum()),
            residual_unequal=int((residual != reference['residual']).sum()))
    assert result['accuracy']['wide_fused']['output_unequal'] == 0
    assert result['accuracy']['wide_fused']['residual_unequal'] == 0
for tokens in (1, 5, 65, 4096):
    x = cpu['input'].repeat((tokens + 4) // 5, 1)[:tokens].contiguous()
    out = torch.empty_like(x)
    records = {label: [] for label in labels}
    for _ in range(30):
        for label in records:
            getattr(torch.ops.relay_norm_experiment, label)(out, x, weight, 1e-6)
    for trial in range(12):
        offset = trial % len(labels)
        order = labels[offset:] + labels[:offset]
        for label in order:
            fn = getattr(torch.ops.relay_norm_experiment, label)
            begin = time.perf_counter()
            for _ in range(100):
                fn(out, x, weight, 1e-6)
            records[label].append((time.perf_counter() - begin) * 1e4)
    result['timing_us'][tokens] = {k: statistics.median(v) for k, v in records.items()}
    if args.gpu_reference:
        fused_labels = ('baseline', 'wide')
        records = {label: [] for label in fused_labels}
        for trial in range(14):
            offset = trial % len(fused_labels)
            for label in fused_labels[offset:] + fused_labels[:offset]:
                fn = getattr(torch.ops.relay_norm_experiment, label + '_fused')
                # Each trial starts at the same input. Copies are outside timing.
                data = x.clone()
                residual = x.clone()
                begin = time.perf_counter()
                for _ in range(100):
                    fn(data, residual, fused_weight, 1e-6)
                if trial >= 2:
                    records[label].append((time.perf_counter() - begin) * 1e4)
        result['fused_timing_us'][tokens] = {k: statistics.median(v) for k, v in records.items()}
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
