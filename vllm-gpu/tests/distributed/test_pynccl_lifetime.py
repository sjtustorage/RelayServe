from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def communicator():
    comm = PyNcclCommunicator.__new__(PyNcclCommunicator)
    comm.comm = object()
    comm.device = 'cuda:0'
    comm.available = True
    comm.disabled = False
    comm.nccl = SimpleNamespace(ncclCommDestroy=Mock())
    return comm


def test_close_drains_then_destroys_once(monkeypatch):
    comm = communicator()
    handle = comm.comm
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append(('drain', device)))
    comm.nccl.ncclCommDestroy.side_effect = lambda value: calls.append(('destroy', value))
    comm.close()
    comm.close()
    assert calls == [('drain', 'cuda:0'), ('destroy', handle)]
    assert comm.comm is None and not comm.available and comm.disabled


def test_singleton_without_native_handle_needs_no_cuda(monkeypatch):
    comm = PyNcclCommunicator.__new__(PyNcclCommunicator)
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    comm.close()
    synchronize.assert_not_called()


def test_failed_drain_preserves_handle_for_retry(monkeypatch):
    comm = communicator()
    handle = comm.comm
    monkeypatch.setattr(torch.cuda, 'synchronize', Mock(side_effect=RuntimeError('drain failed')))
    with pytest.raises(RuntimeError, match='drain failed'):
        comm.close()
    assert comm.comm is handle
    comm.nccl.ncclCommDestroy.assert_not_called()
