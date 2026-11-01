from types import SimpleNamespace

import pytest

from vllm.v1.worker.relay_initialization_policy import (
    initialization_target_ms, readiness_wait_seconds)


@pytest.mark.parametrize('layers,hidden,target', [
    (36,4096,700), (40,5120,700), (64,5120,850), (80,8192,1000),
    (12,768,0),
])
def test_architecture_readiness_windows(monkeypatch,layers,hidden,target):
    monkeypatch.delenv('VLLM_RELAY_INITIALIZATION_TARGET_MS',raising=False)
    assert initialization_target_ms(SimpleNamespace(
        num_hidden_layers=layers,hidden_size=hidden)) == target


@pytest.mark.parametrize('value',['nan','inf','-1','bad'])
def test_invalid_window(monkeypatch,value):
    monkeypatch.setenv('VLLM_RELAY_INITIALIZATION_TARGET_MS',value)
    with pytest.raises(ValueError):
        initialization_target_ms(SimpleNamespace())


def test_window_never_hides_serial_transfer():
    # Same550ms initialization, serial transfer takes300ms extra.
    assert readiness_wait_seconds(0,.55,700) == pytest.approx(.15)
    assert readiness_wait_seconds(0,.85,700,.3) == pytest.approx(.15)
    assert readiness_wait_seconds(0,.8,700) == 0


def test_explicit_opt_out(monkeypatch):
    monkeypatch.setenv('VLLM_RELAY_INITIALIZATION_TARGET_MS','0')
    assert initialization_target_ms(SimpleNamespace(num_hidden_layers=36,hidden_size=4096)) == 0
