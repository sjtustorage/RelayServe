# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Best-effort teardown for SLLM `load_format=shm` IPC (shm_open + named semaphores).

C++/shm_manager and posix_ipc objects may be gone while kernel names in
``/dev/shm`` remain; this module unlinks by the same names used at creation.

Uses an explicit ``libc`` (not ``CDLL(None)``) and ``/dev/shm`` file unlinks as
fallback — required on many Linux builds for ``shm_unlink``/``sem_unlink``.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import errno
import logging
import os
import platform
from multiprocessing import shared_memory
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import posix_ipc
except ImportError:
    posix_ipc = None

_LIBC: ctypes.CDLL | None = None


def _libc(use_errno: bool = True) -> ctypes.CDLL:
    global _LIBC
    if _LIBC is not None:
        return _LIBC
    path = ctypes.util.find_library("c")
    if path:
        try:
            _LIBC = ctypes.CDLL(path, use_errno=use_errno)
            return _LIBC
        except OSError:
            pass
    for cand in ("libc.so.6", "libc.dylib"):
        try:
            _LIBC = ctypes.CDLL(cand, use_errno=use_errno)
            return _LIBC
        except OSError:
            continue
    _LIBC = ctypes.CDLL(None, use_errno=use_errno)
    return _LIBC


def _dev_shm_unlink_file(rel_name: str) -> None:
    """Remove ``/dev/shm/<rel_name>`` if present (Linux)."""
    if platform.system() != "Linux":
        return
    base = rel_name.lstrip("/")
    if not base or "/" in base or ".." in base:
        return
    path = os.path.join("/dev/shm", base)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except IsADirectoryError:
        pass
    except PermissionError:
        logger.warning("Permission denied removing %s (try same user as creator)",
                       path)
    except Exception:
        logger.debug("os.unlink(%s) failed", path, exc_info=True)


def unlink_posix_shm_best_effort(shm_name: str) -> None:
    """``shm_open`` names are like ``/Foo``; backing file is ``/dev/shm/Foo``."""
    if not shm_name:
        return
    name = shm_name if shm_name.startswith("/") else f"/{shm_name.lstrip('/')}"
    base = name.lstrip("/")
    libc = _libc()
    fn = libc.shm_unlink
    fn.argtypes = (ctypes.c_char_p,)
    fn.restype = ctypes.c_int
    ctypes.set_errno(0)
    if fn(name.encode("utf-8")) == 0:
        return
    err = ctypes.get_errno()
    if err not in (errno.ENOENT, errno.EINVAL):
        logger.debug("shm_unlink(%s) errno=%s", name, err)
    _dev_shm_unlink_file(base)


def unlink_named_semaphore_best_effort(sem_name: str) -> None:
    if not sem_name:
        return
    name = sem_name if sem_name.startswith("/") else f"/{sem_name.lstrip('/')}"
    base = name.lstrip("/")
    sem_file = f"sem.{base}"
    if posix_ipc is not None:
        unlink = getattr(posix_ipc, "unlink_semaphore", None)
        if callable(unlink):
            try:
                unlink(name)
                return
            except posix_ipc.ExistentialError:
                return
            except Exception:
                logger.debug("posix_ipc.unlink_semaphore(%s) failed", name,
                             exc_info=True)
    libc = _libc()
    fn = libc.sem_unlink
    fn.argtypes = (ctypes.c_char_p,)
    fn.restype = ctypes.c_int
    ctypes.set_errno(0)
    if fn(name.encode("utf-8")) == 0:
        return
    err = ctypes.get_errno()
    if err != errno.ENOENT:
        logger.debug("sem_unlink(%s) errno=%s", name, err)
    _dev_shm_unlink_file(sem_file)


def _unlink_python_shared_memory(name: str) -> None:
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        logger.debug("SharedMemory unlink %s failed", name, exc_info=True)


def cleanup_load_format_shm_ipc_orphans(
    *,
    model_name: str,
    model_root_path: str,
    tensor_parallel_size: int,
    world_size: int,
) -> None:
    """Unlink names used by shm load / disagg CPU--GPU path (idempotent)."""
    stem = Path(model_root_path).name
    tags = {model_name, stem}
    tags.discard("")

    sem_suffixes = (
        "hidden_states_empty_signal",
        "hidden_states_full_signal",
        "scheduler_output_empty_signal",
        "scheduler_output_full_signal",
        "model_output_full_signal",
        "model_output_empty_signal",
    )
    for tag in tags:
        for suf in sem_suffixes:
            unlink_named_semaphore_best_effort(f"/{tag}_{suf}")
        unlink_posix_shm_best_effort(f"/{tag}_hidden_states")
        for r in range(max(1, world_size)):
            unlink_posix_shm_best_effort(f"/{tag}_kv_cache_rank_{r}")
    for r in range(max(1, tensor_parallel_size)):
        unlink_posix_shm_best_effort(f"/{stem}-rank_{r}")

    for tag in tags:
        _unlink_python_shared_memory(f"{tag}_scheduler_output")
        _unlink_python_shared_memory(f"{tag}_model_output")
