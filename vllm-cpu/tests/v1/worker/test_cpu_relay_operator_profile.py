from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class ProfileRunner(CPUModelRunner):
    def __del__(self):
        # These tests construct no real SHM/model-runner resources.
        pass


@pytest.mark.parametrize('fail', [False, True])
def test_operator_profile_removes_hooks_after_execution_or_failure(monkeypatch, fail):
    from vllm.model_executor.layers import linear
    original_reduce = linear.tensor_model_parallel_all_reduce
    monkeypatch.setenv('VLLM_CPU_PROFILE_RELAY_OPERATORS', '1')
    model = torch.nn.Module()
    model.add_module('qkv_proj', torch.nn.Identity())
    original_gemm = lambda x, *args: x
    model.qkv_proj.cpu_linear = original_gemm
    runner = object.__new__(ProfileRunner)
    runner.model, runner.rank, runner.computing_layers = model, 0, 8
    runner.load_config = SimpleNamespace(load_method='layerwise')
    runner.execute_iteration_cnt = 0
    output = object()
    def execute(self, schedule, intermediate=None):
        assert model.qkv_proj._forward_pre_hooks and model.qkv_proj._forward_hooks
        model.qkv_proj(torch.ones(2))
        assert model.qkv_proj.cpu_linear is not original_gemm
        assert torch.equal(model.qkv_proj.cpu_linear(torch.ones(2)), torch.ones(2))
        if fail:
            raise RuntimeError('execution failed')
        return output
    monkeypatch.setattr(GPUModelRunner, 'execute_model', execute)
    schedule = SimpleNamespace(total_num_scheduled_tokens=4096)
    if fail:
        with pytest.raises(RuntimeError, match='execution failed'):
            runner.execute_model(schedule)
    else:
        assert runner.execute_model(schedule) is output
    assert not model.qkv_proj._forward_pre_hooks and not model.qkv_proj._forward_hooks
    assert model.qkv_proj.cpu_linear is original_gemm
    assert linear.tensor_model_parallel_all_reduce is original_reduce


def test_disabled_operator_profile_installs_no_hooks(monkeypatch):
    monkeypatch.delenv('VLLM_CPU_PROFILE_RELAY_OPERATORS', raising=False)
    runner = object.__new__(ProfileRunner)
    def execute(self, schedule, intermediate=None):
        return 'ordinary execution'
    monkeypatch.setattr(GPUModelRunner, 'execute_model', execute)
    assert runner.execute_model(object()) == 'ordinary execution'
