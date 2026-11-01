"""Prepared communicators survive model replacement; CUDA streams do not mix."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.worker import gpu_worker as worker_module
from vllm.v1.worker.gpu_worker import Worker


def build(monkeypatch):
    group = SimpleNamespace(world_size=2, gather=Mock(), all_gather=Mock())
    stream = Mock()
    stream_context = Mock()
    stream_context.__enter__ = Mock(return_value=None)
    stream_context.__exit__ = Mock(return_value=None)
    torch = SimpleNamespace(zeros=Mock(), cuda=SimpleNamespace(
        Stream=Mock(return_value=stream), current_stream=Mock(return_value=stream),
        stream=Mock(return_value=stream_context), synchronize=Mock()))
    platform = SimpleNamespace(use_all_gather=Mock(return_value=False))
    monkeypatch.setattr(worker_module, 'get_tp_group', lambda: group)
    monkeypatch.setattr(worker_module, 'torch', torch)
    monkeypatch.setattr(worker_module.current_platform, 'use_all_gather',
                        platform.use_all_gather)
    runner = SimpleNamespace(device='cuda:0', model_config=SimpleNamespace(dtype='bf16'))
    return Worker._warm_up_relay_tp_communication, runner, group, stream, torch, platform


def test_replacement_reuses_only_the_prepared_communicator(monkeypatch):
    warmup, runner, group, stream, torch, platform = build(monkeypatch)
    warmup(runner)
    warmup(runner)
    group.gather.assert_called_once()
    stream.synchronize.assert_called_once()
    torch.cuda.synchronize.assert_not_called()
    runner.model_config.dtype = 'fp16'
    warmup(runner)
    assert group.gather.call_count == 2
    del group._relay_warmup_keys  # A recreated group carries no old marker.
    warmup(runner)
    assert group.gather.call_count == 3
    platform.use_all_gather.return_value = True
    warmup(runner)
    group.all_gather.assert_called_once()


def test_failed_probe_is_not_marked_ready(monkeypatch):
    warmup, runner, group, stream, torch, platform = build(monkeypatch)
    stream.synchronize.side_effect = RuntimeError('probe failed')
    with pytest.raises(RuntimeError, match='probe failed'):
        warmup(runner)
    assert not hasattr(group, '_relay_warmup_keys')
    stream.synchronize.side_effect = None
    warmup(runner)
    assert group.gather.call_count == 2
