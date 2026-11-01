"""Exercise the actual SHM feedback method without importing CUDA vLLM."""
import ast
import pickle
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def extract_method(path, name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "EngineCore")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == name)
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace[name]


@pytest.mark.parametrize("boundary,completed,keep_feedback", [
    (5, 7, True), (0, 7, False), (0, 16, False)])
def test_feedback_preserves_completion_when_cpu_retires(
        boundary, completed, keep_feedback):
    path = Path(__file__).resolve().parents[3] / "vllm/v1/engine/core.py"
    memory = SimpleNamespace(buf=bytearray(4096), close=Mock())
    send = extract_method(path, "send_shm_model_output", dict(
        pickle=pickle, struct=struct,
        shared_memory=SimpleNamespace(SharedMemory=lambda **_: memory)))
    core = SimpleNamespace(
        vllm_config=SimpleNamespace(
            load_config=SimpleNamespace(lazy_load=True, load_format="shm",
                                        load_method="layerwise"),
            model_config=SimpleNamespace(model="model")),
        need_send_shm_model_output=True, total_layers=16,
        loaded_layers=set(range(16-completed, 16)),
        target_layers=set(range(16)), _next_cpu_boundary=boundary,
        model_output_empty_signal=Mock(), model_output_full_signal=Mock(),
        _unlink_python_shared_memory=Mock())
    send(core, SimpleNamespace())
    length = struct.unpack("Q", memory.buf[:8])[0]
    output = pickle.loads(memory.buf[8:8+length])
    assert output.shm_gpu_loaded_layers == completed
    assert output.shm_next_cpu_boundary == boundary
    assert core.need_send_shm_model_output == keep_feedback
    core.model_output_full_signal.release.assert_called_once()
