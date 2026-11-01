"""Full-layer warmup preserves the real relay execution range."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker import gpu_worker


def test_first_copy_policy_survives_runner_recreation_and_keeps_processes_independent(monkeypatch):
    import os
    from vllm.v1.worker import gpu_model_runner as module
    monkeypatch.setattr(module, '_relay_copy_first_model_pids', set())
    monkeypatch.setenv('VLLM_SHM_COPY_THROTTLE_FIRST_ONLY', '1')
    monkeypatch.setenv('VLLM_SHM_UNTHROTTLED_COPY', '1')
    monkeypatch.setattr(module.os, 'getpid', lambda: 101)
    runner = SimpleNamespace()
    module._select_relay_copy_policy(runner)
    assert 'VLLM_SHM_UNTHROTTLED_COPY' not in os.environ
    module._select_relay_copy_policy(runner)
    assert 'VLLM_SHM_UNTHROTTLED_COPY' not in os.environ
    module._select_relay_copy_policy(SimpleNamespace())
    assert os.environ['VLLM_SHM_UNTHROTTLED_COPY'] == '1'
    monkeypatch.setattr(module.os, 'getpid', lambda: 102)
    module._select_relay_copy_policy(SimpleNamespace())
    assert 'VLLM_SHM_UNTHROTTLED_COPY' not in os.environ


@pytest.mark.parametrize('start,end', [(0, 36), (0, 64), (16, 32)])
@pytest.mark.parametrize('warmup', [True, False])
def test_dummy_forward_visits_every_local_layer(start, end, warmup):
    model = Qwen2Model.__new__(Qwen2Model)
    torch.nn.Module.__init__(model)
    model.start_layer, model.end_layer = start, end
    model.aux_hidden_state_layers = ()
    seen = []
    model.get_input_embeddings = lambda ids: torch.ones((len(ids), 4))

    def layer(positions, hidden, residual, idx, mode):
        seen.append((idx, mode))
        return hidden, hidden

    model.layers = [layer] * end
    model.norm = lambda hidden, residual: (hidden, residual)
    group = SimpleNamespace(is_first_rank=start == 0, is_last_rank=True)
    intermediate = dict(hidden_states=torch.ones(2, 4), residual=None)
    with patch('vllm.model_executor.models.qwen2.get_pp_group', return_value=group), patch(
            'vllm.model_executor.models.qwen2.wait_for_weights_loaded') as wait:
        model.forward(torch.zeros(2, dtype=torch.long), torch.zeros(2),
                      intermediate_tensors=intermediate, warmup=warmup)
    assert wait.call_count == int(not warmup and start == 0)
    assert seen == [(i, warmup) for i in range(start, end)]


@pytest.mark.parametrize('model_type', [Qwen2Model, Qwen3Model])
@pytest.mark.parametrize('active,allocated', [
    ((8, 80), (0, 80)),     # intra-node layerwise recovery prefix
    ((12, 24), (8, 32)),    # expanded asymmetric PP ownership
    ((16, 32), (16, 32)),   # uniform PP after removing migrated KV
])
@pytest.mark.parametrize('fails', [False, True])
def test_warmup_visits_all_allocated_layers_and_restores_relay(
        model_type, active, allocated, fails):
    model = model_type.__new__(model_type)
    torch.nn.Module.__init__(model)
    model.start_layer, model.end_layer = active
    model.aux_hidden_state_layers = ()
    seen = []
    model.get_input_embeddings = lambda ids: torch.ones(len(ids), 4)

    def layer(positions, hidden, residual, idx, mode):
        seen.append(idx)
        return hidden, hidden

    model.layers = [layer] * allocated[1]
    model.norm = lambda hidden, residual: (hidden, residual)
    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format='shm'),
        model_config=SimpleNamespace(enforce_eager=True, hf_config=SimpleNamespace()),
        speculative_config=None,
        get_model=lambda: SimpleNamespace(model=model),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(
            layer_names=[f'model.layers.{i}.self_attn'
                         for i in range(*allocated)])]))
    group = SimpleNamespace(is_first_rank=allocated[0] == 0, is_last_rank=True)
    try:
        with GPUModelRunner.warmup_layer_range(runner), patch(
                'vllm.model_executor.models.qwen2.get_pp_group', return_value=group):
            model.forward(torch.zeros(2, dtype=torch.long), torch.zeros(2),
                          intermediate_tensors=dict(hidden_states=torch.ones(2, 4),
                                                    residual=None), warmup=True)
            if fails:
                raise RuntimeError('warmup failed')
    except RuntimeError as exc:
        assert fails and str(exc) == 'warmup failed'
    assert seen == list(range(*allocated))
    assert (model.start_layer, model.end_layer) == active


@pytest.mark.parametrize('layers,hidden,expected_tokens', [
    (36, 4096, 3), (40, 5120, 3), (64, 5120, 3),
    (80, 8192, 3), (12, 768, 3)])
@pytest.mark.parametrize('phase', ['default', 'first_override', 'reload'])
def test_complete_warmup_preserves_compilation_and_graph_execution_range(
        monkeypatch, layers, hidden, expected_tokens, phase):
    import os
    monkeypatch.setattr(gpu_worker, '_relay_warmed_model_pids',
                        {os.getpid()} if phase == 'reload' else set())
    monkeypatch.delenv('VLLM_RELAY_FIRST_WARMUP_PREFILL_TOKENS', raising=False)
    if phase != 'default':
        monkeypatch.setenv('VLLM_RELAY_FIRST_WARMUP_PREFILL_TOKENS', '1280')
        if phase == 'first_override':
            expected_tokens = 1280
    monkeypatch.delenv('VLLM_RELAY_WARMUP_PREFILL_TOKENS', raising=False)
    inner = Qwen2Model.__new__(Qwen2Model)
    torch.nn.Module.__init__(inner)
    inner.start_layer, inner.end_layer = 8, 80
    events = []

    def record(kind):
        events.append((kind, inner.start_layer, inner.end_layer))

    def dummy(num_tokens, **kwargs):
        if kwargs.get('force_attention') and not kwargs.get('uniform_decode'):
            assert num_tokens == expected_tokens
        record('decode' if kwargs.get('uniform_decode') else
               'prefill' if kwargs.get('force_attention') else 'compile')
        return 'hidden', 'last_hidden'

    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format='shm'),
        model_config=SimpleNamespace(enforce_eager=True, hf_config=SimpleNamespace()),
        speculative_config=None,
        get_model=lambda: SimpleNamespace(model=inner),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(
            layer_names=[f'model.layers.{i}.self_attn' for i in range(80)])]),
        _dummy_run=dummy, capture_model=lambda: record('capture'),
        is_pooling_model=False,
        _dummy_sampler_run=lambda **kwargs: record('sampler'))
    runner.model_config.enforce_eager = False
    runner.warmup_layer_range = lambda: GPUModelRunner.warmup_layer_range(runner)
    worker = SimpleNamespace(
        model_runner=runner, device='cuda:0', rank=0,
        model_config=SimpleNamespace(enforce_eager=False, seed=0,
                                     hf_config=SimpleNamespace(
                                         num_hidden_layers=layers,
                                         hidden_size=hidden)),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=4096),
        vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(
            compile_sizes=[8], cudagraph_capture_sizes=[]),
            load_config=SimpleNamespace(load_format='shm')))
    monkeypatch.setattr(gpu_worker, 'kernel_warmup', lambda _: record('kernel'))
    monkeypatch.setattr(gpu_worker, 'get_pp_group',
                        lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(gpu_worker, 'set_random_seed', lambda _: None)
    monkeypatch.setattr(gpu_worker.torch.cuda, 'current_stream',
                        lambda _: SimpleNamespace(synchronize=lambda: record('complete')))
    Worker._compile_or_warm_up_model(worker)
    assert os.getpid() in gpu_worker._relay_warmed_model_pids
    assert events == [('compile', 8, 80), ('kernel', 8, 80), ('capture', 8, 80),
                      ('prefill', 0, 80), ('decode', 0, 80), ('sampler', 8, 80),
                      ('complete', 8, 80)]


@pytest.mark.parametrize("unsupported", ["non_shm", "quantized", "sliding",
                                         "dual_chunk", "speculative"])
def test_full_warmup_keeps_non_shm_and_covers_other_shm_paths(unsupported):
    inner = Qwen2Model.__new__(Qwen2Model)
    torch.nn.Module.__init__(inner)
    inner.start_layer, inner.end_layer = 8, 80
    config = SimpleNamespace(enforce_eager=True, hf_config=SimpleNamespace())
    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format='shm'), model_config=config,
        speculative_config=None, get_model=lambda: SimpleNamespace(model=inner),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(
            layer_names=['model.layers.0.self_attn'])]))
    if unsupported == 'non_shm':
        runner.load_config.load_format = 'auto'
    elif unsupported == 'quantized':
        inner.quant_config = object()
    elif unsupported == 'sliding':
        config.hf_config.use_sliding_window = True
    elif unsupported == 'dual_chunk':
        config.hf_config.dual_chunk_attention_config = {'chunk_size': 1024}
    else:
        runner.speculative_config = object()
    with GPUModelRunner.warmup_layer_range(runner):
        assert (inner.start_layer, inner.end_layer) == ((8, 80) if unsupported == 'non_shm' else (0, 1))
    assert (inner.start_layer, inner.end_layer) == (8, 80)
