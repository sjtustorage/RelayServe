# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys


def is_ray_initialized():
    """Check if Ray is initialized."""
    # Ray initialization requires importing Ray first. Status checks in local
    # executors must not import its runtime just to discover it is unused.
    if "ray" not in sys.modules:
        return False
    try:
        import ray
        return ray.is_initialized()
    except ImportError:
        return False


def is_in_ray_actor():
    """Check if we are in a Ray actor."""

    if "ray" not in sys.modules:
        return False
    try:
        import ray
        return (ray.is_initialized()
                and ray.get_runtime_context().get_actor_id() is not None)
    except ImportError:
        return False
