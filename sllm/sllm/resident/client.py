"""Ray-free client for a resident GPU rank pool.

The client owns only the control subprocess and a JSON-lines FIFO. Model
lifetime remains in the resident ranks; every command is serialized and replies
are checked for matching sequence numbers before being returned.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import select
import signal
import time
from collections import deque
import subprocess
import tempfile
import threading
import re
from typing import Any, Optional


@dataclass(frozen=True)
class ResidentReply:
    sequence: int
    next_sequence: int
    ranks: list[dict[str, Any]]


class ResidentCommandRejected(ValueError):
    """All ranks rejected validation before executing the command."""

    def __init__(self, op, reply):
        self.op = op
        self.reply = reply
        super().__init__(f"resident command {op!r} rejected before execution: {reply}")

    @property
    def memory_capacity_unavailable(self):
        """Only recognize the joined, nonexecuting Hydra memory preflight.

        Rank sockets include the exception type; WORLD transport emits the
        message alone. Mixed validation errors must remain visible to callers.
        """
        message = 'Insufficient free GPU memory for Hydra staging budget'
        errors = self.reply.get('ranks')
        if (self.op != 'load' or self.reply.get('accepted') is not False
                or not isinstance(errors, list)):
            return False
        failed = [error for error in errors if error is not None]
        def capacity_error(error):
            if not isinstance(error, str):
                return False
            error = error.removeprefix('ValueError: ')
            threshold = 'Insufficient free GPU memory for Hydra borrowing threshold'
            if error in (message, threshold):
                return True
            budget = re.fullmatch(re.escape(message) + r': free=(\d+) required=(\d+)', error)
            if budget:
                free, required = map(int, budget.groups())
                return free < required
            borrowing = re.fullmatch(re.escape(threshold) +
                r': free=(\d+) total=(\d+) ratio=([0-9.eE+\-]+)', error)
            if borrowing:
                free, total = map(int, borrowing.groups()[:2])
                try:
                    ratio = float(borrowing.group(3))
                except ValueError:
                    return False
                return total > 0 and 0 <= free <= total and 0 < ratio <= 1 and free / total < ratio
            return False
        return bool(failed) and all(capacity_error(error) for error in failed)


class ResidentCommandFailed(RuntimeError):
    """A dispatched resident command returned one or more rank errors."""

    def __init__(self, op, reply):
        self.op = op
        self.reply = reply
        errors = [item.get('error') for item in reply.ranks
                  if isinstance(item, dict) and item.get('error')]
        detail = errors[0] if errors else 'unsuccessful rank reply'
        super().__init__(f"resident command {op!r} failed: {detail}")



def is_preflight_rejection(reply, sequence, gpu_count):
    return (isinstance(reply, dict)
            and type(reply.get('sequence')) is int and reply['sequence'] == sequence
            and type(reply.get('next_sequence')) is int and reply['next_sequence'] == sequence
            and reply.get('accepted') is False and reply.get('ok') is False
            and isinstance(reply.get('ranks'), list) and len(reply['ranks']) == gpu_count
            and all(item is None or isinstance(item, str) for item in reply['ranks'])
            and any(isinstance(item, str) and item for item in reply['ranks']))


class ResidentPoolClient:
    def __init__(self, torchrun: str, gpu_count: int,
                 *, timeout: float = 120.0,
                 weight_cache_bytes: int = 80 * 1024**3,
                 concurrent_topologies: bool = False,
                 copy_throttle: Optional[bool] = None,
                 copy_throttle_tasks: int = 32,
                 copy_throttle_first_only: bool = False,
                 gpu_cpu_ids: Optional[list[int]] = None):
        if gpu_cpu_ids is not None:
            if (not isinstance(gpu_cpu_ids, (list, tuple)) or not gpu_cpu_ids
                    or any(type(cpu) is not int or cpu < 0 for cpu in gpu_cpu_ids)
                    or len(set(gpu_cpu_ids)) != len(gpu_cpu_ids)
                    or not set(gpu_cpu_ids) <= os.sched_getaffinity(0)):
                raise ValueError('gpu_cpu_ids must be distinct allowed CPU IDs')
        if type(copy_throttle_first_only) is not bool:
            raise ValueError('copy_throttle_first_only must be a boolean')
        if copy_throttle_first_only and copy_throttle is False:
            raise ValueError('first-only throttle requires first-model throttling')
        if type(copy_throttle_tasks) is not int or copy_throttle_tasks < 1:
            raise ValueError('copy_throttle_tasks must be a positive integer')
        if copy_throttle is not None and type(copy_throttle) is not bool:
            raise ValueError('copy_throttle must be a boolean')
        if not torchrun or gpu_count < 1 or timeout <= 0 or weight_cache_bytes < 0:
            raise ValueError("invalid resident pool client configuration")
        self.torchrun = torchrun
        self.gpu_count = gpu_count
        self.timeout = timeout
        self.weight_cache_bytes = weight_cache_bytes
        self.concurrent_topologies = concurrent_topologies
        self.copy_throttle = copy_throttle
        self.copy_throttle_tasks = copy_throttle_tasks
        self.copy_throttle_first_only = copy_throttle_first_only
        self.gpu_cpu_ids = tuple(sorted(gpu_cpu_ids)) if gpu_cpu_ids is not None else None
        self._rank_transport = None
        self._tmp: Optional[tempfile.TemporaryDirectory] = None
        self._fd: Optional[int] = None
        self._process: Optional[subprocess.Popen[str]] = None
        self._replies: queue.Queue[Optional[dict[str, Any]]] = queue.Queue()
        self._logs: deque[str] = deque(maxlen=200)
        self._reader: Optional[threading.Thread] = None
        self._sequence = 0
        self._failed = False
        self._closed = False
        self._lock = threading.Lock()

    @property
    def pids(self) -> list[int]:
        reply = self.command("status")
        return [int(item["result"]["pid"]) for item in reply.ranks]

    @property
    def failure(self):
        """Host-only health check; never queue a command behind hot inference."""
        if self._failed or (self._rank_transport is not None and self._rank_transport._failed):
            return 'GPU control channel failed'
        if self._process is not None and self._process.poll() is not None:
            return f'GPU launcher exited with code {self._process.returncode}'
        return None

    def replacement(self):
        return type(self)(self.torchrun, self.gpu_count, timeout=self.timeout,
            weight_cache_bytes=self.weight_cache_bytes,
            concurrent_topologies=self.concurrent_topologies,
            copy_throttle=self.copy_throttle, copy_throttle_tasks=self.copy_throttle_tasks,
            copy_throttle_first_only=self.copy_throttle_first_only,
            gpu_cpu_ids=self.gpu_cpu_ids)

    def start(self) -> None:
        if self._process is not None or self._failed or self._closed:
            raise RuntimeError("resident pool is already started or failed")
        self._tmp = tempfile.TemporaryDirectory(prefix="resident-client-")
        fifo = Path(self._tmp.name) / "commands"
        os.mkfifo(fifo, 0o600)
        self._fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
        env = dict(os.environ,
                   VLLM_SHM_WEIGHT_CACHE_BYTES=str(self.weight_cache_bytes),
                   VLLM_SHM_COPY_THROTTLE_TASKS=str(self.copy_throttle_tasks))
        if self.gpu_cpu_ids is not None:
            env['SLLM_RESIDENT_GPU_CPU_IDS'] = ','.join(map(str, self.gpu_cpu_ids))
        # First-model history is rank-local state, never inherited by a new pool.
        env.pop('VLLM_SHM_COPY_FIRST_MODEL_SEEN', None)
        env.pop('VLLM_SHM_COPY_THROTTLE_FIRST_ONLY', None)
        if self.copy_throttle_first_only:
            env['VLLM_SHM_COPY_THROTTLE_FIRST_ONLY'] = '1'
            env.pop('VLLM_SHM_UNTHROTTLED_COPY', None)
        if self.copy_throttle is not None:
            env.pop('VLLM_SHM_UNTHROTTLED_COPY', None)
            if not self.copy_throttle:
                env['VLLM_SHM_UNTHROTTLED_COPY'] = '1'
        # Freeze the actual launch policy, including legacy environment use.
        self.copy_throttle = 'VLLM_SHM_UNTHROTTLED_COPY' not in env
        # CPU AMX launchers preload Intel OpenMP. GPU PyTorch has its own
        # runtime; inheriting libiomp can bind every GPU rank to core 0.
        # Keep other preloads, including the host allocator, intact.
        if 'LD_PRELOAD' in env:
            preloads = env['LD_PRELOAD'].replace(':', ' ').split()
            env['LD_PRELOAD'] = ':'.join(
                item for item in preloads
                if not Path(item).name.startswith('libiomp'))
        try:
            self._process = subprocess.Popen(
                [self.torchrun, "--standalone", f"--nproc-per-node={self.gpu_count}",
                 "-m", "sllm.resident.worker", "--command-file", str(fifo),
                 "--weight-cache-bytes", str(self.weight_cache_bytes),
                 *(['--rank-socket-dir', self._tmp.name] if self.concurrent_topologies else [])],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env=env, start_new_session=True)
        except BaseException:
            os.close(self._fd)
            self._fd = None
            self._tmp.cleanup()
            self._tmp = None
            self._failed = True
            raise
        assert self._process.stdout is not None
        def drain() -> None:
            for line in self._process.stdout:
                self._logs.append(line)
                if line.startswith("RESIDENT_REPLY "):
                    try:
                        self._replies.put(json.loads(line.split(" ", 1)[1]))
                    except (ValueError, TypeError):
                        self._replies.put(None)
            self._replies.put(None)
        self._reader = threading.Thread(target=drain, daemon=True)
        self._reader.start()
        if self.concurrent_topologies:
            from sllm.resident.rank_transport import RankTransport
            paths = [Path(self._tmp.name) / f'rank-{rank}.sock'
                     for rank in range(self.gpu_count)]
            deadline = time.monotonic() + self.timeout
            try:
                while not all(path.exists() for path in paths):
                    if self._process.poll() is not None:
                        raise RuntimeError('Resident rank startup failed:\n' + ''.join(self._logs))
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Resident rank socket startup timed out')
                    time.sleep(.01)
                self._rank_transport = RankTransport(paths, timeout=self.timeout)
            except BaseException:
                self._failed = True
                self.close()
                raise

    def abort_pending(self):
        """Poison control channels and wake waiters without reusing the pool."""
        self._failed = True
        if self._rank_transport is not None:
            self._rank_transport.abort_pending()
        else:
            self._replies.put(None)

    def command(self, op: str, *, expected: bool = True, **kwargs: Any) -> ResidentReply:
        if self._rank_transport is not None:
            return self._rank_transport.command(op, expected=expected, **kwargs)
        if not isinstance(op, str) or not op:
            raise ValueError("op must be nonempty")
        if type(expected) is not bool or {'sequence', 'op'} & kwargs.keys():
            raise ValueError('Client owns protocol fields and expected must be boolean')
        with self._lock:
            if self._process is None or self._fd is None:
                raise RuntimeError("resident pool is not started")
            if self._failed:
                raise RuntimeError("resident pool failed; create a new client")
            sequence = self._sequence
            payload = json.dumps(dict(sequence=sequence, op=op, **kwargs)) + "\n"
            deadline = time.monotonic() + self.timeout
            try:
                data = memoryview(payload.encode())
                while data:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("command write timed out")
                    if self._process.poll() is not None:
                        raise OSError("resident pool exited during write")
                    _, writable, _ = select.select([], [self._fd], [], min(remaining, 0.1))
                    if not writable:
                        continue
                    try:
                        written = os.write(self._fd, data)
                    except BlockingIOError:
                        continue
                    if written <= 0:
                        raise OSError("command write made no progress")
                    data = data[written:]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("command reply timed out")
                reply = self._replies.get(timeout=remaining)
            except (OSError, queue.Empty) as exc:
                self._failed = True
                raise RuntimeError("resident pool command timed out or closed") from exc
            if reply is None:
                self._failed = True
                raise RuntimeError("resident pool exited:\n" + "".join(list(self._logs)[-40:]))
            if not isinstance(reply, dict) or type(reply.get('sequence')) is not int or reply.get("sequence") != sequence:
                self._failed = True
                raise RuntimeError(f"resident reply sequence mismatch: {reply}")
            rejected = is_preflight_rejection(reply, sequence, self.gpu_count)
            if not rejected and (reply.get('accepted') is not True
                    or type(reply.get('next_sequence')) is not int
                    or reply['next_sequence'] != sequence + 1
                    or type(reply.get('ok')) is not bool
                    or not isinstance(reply.get('ranks'), list)
                    or len(reply['ranks']) != self.gpu_count
                    or any(not isinstance(item, dict) for item in reply['ranks'])):
                self._failed = True
                raise RuntimeError(f"Invalid resident reply: {reply}")
            self._sequence = reply['next_sequence']
            parsed_reply = ResidentReply(sequence, self._sequence,
                                         list(reply.get("ranks", [])))
            if any(isinstance(item, dict) and item.get("fatal")
                   for item in reply.get("ranks", [])):
                self._failed = True
            if bool(reply.get("ok")) is not expected:
                if rejected:
                    raise ResidentCommandRejected(op, reply)
                has_errors = any(isinstance(item, dict) and item.get('error')
                                 for item in parsed_reply.ranks)
                if reply.get('ok') is False and has_errors:
                    raise ResidentCommandFailed(op, parsed_reply)
                raise RuntimeError(f"resident command {op!r} unexpected result: {reply}")
            return parsed_reply

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                self.command("shutdown")
                # Group destruction can outlive the acknowledged shutdown
                # command (notably torchrun's NCCL children).  Never let
                # teardown consume the command timeout or contaminate timing
                # of the next resident model instance.
                process.wait(timeout=min(self.timeout, 5.0))
        except (RuntimeError, ResidentCommandRejected, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (AttributeError, ProcessLookupError):
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (AttributeError, ProcessLookupError):
                    process.kill()
                process.wait()
        finally:
            # torchrun may exit before its children. No old CUDA owner may
            # survive into the replacement pool or CPU SHM teardown.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if self._rank_transport is not None:
                self._rank_transport.close()
                self._rank_transport = None
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            if self._reader is not None:
                self._reader.join(timeout=15)
            if process.stdout is not None:
                process.stdout.close()
            self._process = None
            self._closed = True
            if self._tmp is not None:
                self._tmp.cleanup()
                self._tmp = None

    def __enter__(self) -> "ResidentPoolClient":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
