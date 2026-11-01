# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from collections import defaultdict
import json
import os
import time
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, local_omp_cpuids: str):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        assert self.speculative_config is None, "spec decode is not supported."

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False
        self.local_omp_cpuids = local_omp_cpuids
        self._postprocess_tensors()

    def _postprocess_tensors(self) -> None:
        # Note: replace device tensors with cpu tensors
        def replace_tensor(obj: Any, cpu_attr_name: str, device_attr_name) -> None:
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if cpu_tensor is not None and device_tensor is not None:
                assert isinstance(cpu_tensor, torch.Tensor)
                assert isinstance(device_tensor, torch.Tensor)
                setattr(obj, device_attr_name, cpu_tensor)

        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                v.gpu = v.cpu

        for k, v in vars(self.input_batch).items():
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch, k, k[:-11])

        for block_table in self.input_batch.block_table.block_tables:
            for k, v in vars(block_table).items():
                if k.endswith("_cpu") and isinstance(v, torch.Tensor):
                    replace_tensor(block_table, k, k[:-4])

    def load_model(self, eep_scale_up: bool = False) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.model_config,
                                              self.scheduler_config,
                                              self.lora_config, self.device)

    def get_model(self) -> nn.Module:
        return self.model

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        profile = (os.environ.get('VLLM_CPU_PROFILE_RELAY_OPERATORS') == '1'
                   and (self.load_config.load_method != 'layerwise'
                        or self.execute_iteration_cnt == 0)
                   and scheduler_output.total_num_scheduled_tokens > 1)
        if not profile:
            return super().execute_model(scheduler_output, intermediate_tensors)
        # Disjoint module boundaries include projection TP reductions. Their
        # costs are wall times, not isolated kernel or AMX throughput metrics.
        measured = {'qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj',
                    'q_norm', 'k_norm', 'input_layernorm', 'post_attention_layernorm',
                    'rotary_emb', 'attn', 'act_fn', 'embed_tokens'}
        totals, calls, starts, handles = defaultdict(float), defaultdict(int), {}, []
        from vllm.model_executor.layers import linear
        original_all_reduce = linear.tensor_model_parallel_all_reduce
        local_gemm, reductions, restore_gemm = defaultdict(float), defaultdict(float), []
        projection_kinds = {}
        current_projection = [None]
        def time_gemm(kind, original):
            def run(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    local_gemm[kind] += time.perf_counter() - started
            return run
        def reduce(input_):
            started = time.perf_counter()
            try:
                return original_all_reduce(input_)
            finally:
                reductions[current_projection[0] or 'other'] += time.perf_counter() - started
        def before(module, args):
            starts[module] = time.perf_counter()
            if hasattr(module, 'cpu_linear'):
                current_projection[0] = projection_kinds[module]
        def after(kind):
            def record(module, args, result):
                totals[kind] += time.perf_counter() - starts.pop(module)
                calls[kind] += 1
                if hasattr(module, 'cpu_linear'):
                    current_projection[0] = None
            return record
        try:
            for name, module in self.model.named_modules():
                kind = name.rsplit('.', 1)[-1]
                if kind in measured:
                    if hasattr(module, 'cpu_linear'):
                        original = module.cpu_linear
                        restore_gemm.append((module, original))
                        projection_kinds[module] = kind
                        module.cpu_linear = time_gemm(kind, original)
                    handles.append(module.register_forward_pre_hook(before))
                    handles.append(module.register_forward_hook(after(kind)))
            linear.tensor_model_parallel_all_reduce = reduce
            started = time.perf_counter()
            result = super().execute_model(scheduler_output, intermediate_tensors)
            elapsed = time.perf_counter() - started
            logger.info('[CPU relay operators] %s', json.dumps(dict(
                rank=self.rank, tokens=scheduler_output.total_num_scheduled_tokens,
                load_method=self.load_config.load_method,
                layers=(self.computing_layers if self.load_config.load_method == 'layerwise'
                        else self.model_config.hf_config.num_hidden_layers),
                execute_seconds=elapsed,
                operator_seconds=dict(totals), operator_calls=dict(calls),
                local_gemm_seconds=dict(local_gemm),
                all_reduce_seconds=dict(reductions)), sort_keys=True))
            return result
        finally:
            linear.tensor_model_parallel_all_reduce = original_all_reduce
            for module, original in restore_gemm:
                module.cpu_linear = original
            for handle in handles:
                handle.remove()

    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings(self.vllm_config):
            self._dummy_run(
                min(
                    max(16, self.max_num_reqs),
                    self.scheduler_config.max_num_batched_tokens,
                )
            )

        logger.info("Warming up done.")

    def _init_device_properties(self) -> None:
        pass

    def _sync_device(self) -> None:
        pass

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        return sampled_token_ids.tolist()

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        # Note: For CPU backend, dp padding is not required for now.
        return 0, None


@contextmanager
def _torch_cuda_wrapper():
    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

    cuda_event = torch.cuda.Event
    cuda_stream = torch.cuda.Stream
    try:
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        yield
    finally:
        torch.cuda.Event = cuda_event
        torch.cuda.Stream = cuda_stream


@contextmanager
def _set_global_compilation_settings(config: VllmConfig):
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    # Note: The MKLDNN and CPPGEMM backend requires freezing parameters.
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
