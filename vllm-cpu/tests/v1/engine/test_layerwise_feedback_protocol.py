"""Test CPU iteration feedback without requiring the inference runtime."""
import ast
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("boundary,completed", [(0, 7), (0, 16), (5, 7)])
def test_cpu_ownership_uses_boundary_not_completed_weights(boundary, completed):
    path = Path(__file__).resolve().parents[3] / "vllm/v1/engine/core.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "EngineCore")
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == "step")
    step.returns = None
    namespace = dict(os=os, time=time, logger=Mock())
    exec(compile(ast.Module(body=[step], type_ignores=[]), str(path), "exec"),
         namespace)
    events = []
    scheduler = SimpleNamespace(
        has_requests=lambda: True,
        schedule=lambda: SimpleNamespace(total_num_scheduled_tokens=4096),
        update_from_output=lambda *_: {})
    core = SimpleNamespace(
        scheduler=scheduler, step_cnt=0, total_layers=16,
        shm_gpu_loaded_layers=0, pp_cleanup_enabled=False,
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(
            load_format="shm", lazy_load=True, load_method="layerwise")),
        model_executor=SimpleNamespace(execute_model=Mock(),
            publish_pending_relay_step=lambda: events.append("publish")),
        execute_model_with_error_logging=lambda *_: events.append("cpu"),
        recv_shm_model_output=lambda: (events.append("feedback") or
            SimpleNamespace(shm_next_cpu_boundary=boundary,
                            shm_gpu_loaded_layers=completed)),
        _finish_cpu_relay=Mock(), update_load_method=Mock())
    namespace["step"](core)
    assert events == ["cpu", "publish", "feedback"]
    assert core.shm_gpu_loaded_layers == 16-boundary
    assert core.shm_completed_gpu_layers == completed
    if boundary == 0:
        core._finish_cpu_relay.assert_called_once_with({}, release_kv_cache=True)
        core.update_load_method.assert_called_once_with("cpu")
    else:
        core._finish_cpu_relay.assert_not_called()
        # A second chunk uses the promised boundary, not the completed count.
        # Completion advances independently and must survive on the CPU side.
        scheduled = SimpleNamespace(total_num_scheduled_tokens=128)
        scheduler.schedule = lambda: scheduled
        core.recv_shm_model_output = lambda: SimpleNamespace(
            shm_next_cpu_boundary=2, shm_gpu_loaded_layers=12)
        namespace["step"](core)
        assert scheduled.gpu_loaded_layers == 11
        assert core.shm_gpu_loaded_layers == 14
        assert core.shm_completed_gpu_layers == 12
        assert core.step_cnt == 2
        core._finish_cpu_relay.assert_not_called()
