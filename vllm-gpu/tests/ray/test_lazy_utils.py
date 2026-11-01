# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import builtins
import sys
from types import SimpleNamespace

import pytest

from vllm.ray.lazy_utils import is_in_ray_actor, is_ray_initialized


def test_unused_ray_is_not_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "ray", raising=False)
    original_import = builtins.__import__

    def reject_ray(name, *args, **kwargs):
        if name == "ray":
            pytest.fail("A status check imported unused Ray")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_ray)
    assert not is_ray_initialized()
    assert not is_in_ray_actor()


@pytest.mark.parametrize("initialized,actor_id", [
    (False, None), (False, "actor"), (True, None), (True, "actor")
])
def test_imported_ray_state_is_checked_each_time(monkeypatch, initialized,
                                              actor_id):
    def context():
        assert initialized, "Uninitialized Ray must not request a context"
        return SimpleNamespace(get_actor_id=lambda: actor_id)

    ray = SimpleNamespace(is_initialized=lambda: initialized,
                          get_runtime_context=context)
    monkeypatch.setitem(sys.modules, "ray", ray)
    assert is_ray_initialized() is initialized
    assert is_in_ray_actor() is (initialized and actor_id is not None)
    # The helper must not cache the initial false state across ray.init().
    ray.is_initialized = lambda: True
    initialized = True
    assert is_ray_initialized()
    assert is_in_ray_actor() is (actor_id is not None)
