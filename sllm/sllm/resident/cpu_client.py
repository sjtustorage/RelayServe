"""Own a persistent CPU producer in its CPU Python environment.

Consumers must unload their GPU SHM mappings before closing this client. No
uncertain command is retried: timeout or invalid reply poisons the session.
"""
import json
import math
import os
from pathlib import Path
import queue
import select
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque


class CpuProducerClient:
    def __init__(self, python, cpu_root, checkpoint, *, tp=2, timeout=150,
                 max_model_len=40960, kv_space_gib=4, threads=48, log_path=None,
                 max_num_batched_tokens=4096, record_compute=False, record_operators=False,
                 warmup=True):
        if (type(tp) is not int or tp < 1 or not math.isfinite(timeout) or timeout <= 0
                or type(max_model_len) is not int or max_model_len < 1
                or type(threads) is not int or threads < 1
                or type(max_num_batched_tokens) is not int or max_num_batched_tokens < 1
                or type(record_compute) is not bool
                or type(record_operators) is not bool
                or type(warmup) is not bool
                or not math.isfinite(kv_space_gib) or kv_space_gib <= 0):
            raise ValueError('Invalid CPU producer configuration')
        self.python = str(python)
        self.cpu_root = Path(cpu_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.tp, self.timeout = tp, timeout
        self.max_model_len, self.kv_space_gib, self.threads = max_model_len, kv_space_gib, threads
        self.max_num_batched_tokens = max_num_batched_tokens
        self.record_compute = record_compute
        self.record_operators = record_operators
        self.warmup = warmup
        self.log_path = Path(log_path) if log_path else None
        self.process = None
        self.model_path = None
        self.info = None
        self._temporary = None
        self._reader = None
        self._replies = queue.Queue()
        self._logs = deque(maxlen=100)
        self._sequence = 0
        self._failed = False
        self._closed = False
        self._lock = threading.Lock()

    @property
    def pid(self):
        return self.process.pid if self.process else None

    @property
    def busy(self):
        return self._lock.locked()

    @property
    def failure(self):
        if self._failed:
            return 'CPU producer command failed'
        if self.process is not None and self.process.poll() is not None:
            return f'CPU producer exited with code {self.process.returncode}'
        return None

    def replacement(self):
        import time
        log = (self.log_path.with_name(f'{self.log_path.stem}-recovery-{time.time_ns()}.log')
               if self.log_path else None)
        return type(self)(self.python, self.cpu_root, self.checkpoint, tp=self.tp,
            timeout=self.timeout, max_model_len=self.max_model_len,
            kv_space_gib=self.kv_space_gib, threads=self.threads, log_path=log,
            max_num_batched_tokens=self.max_num_batched_tokens,
            record_compute=self.record_compute, record_operators=self.record_operators,
            warmup=self.warmup)

    def start(self):
        if self.process or self._closed or self._failed:
            raise RuntimeError('CPU producer already started or closed')
        if not (self.checkpoint / 'config.json').is_file():
            raise ValueError('Checkpoint requires config.json')
        self._temporary = tempfile.TemporaryDirectory(prefix='resident-cpu-')
        self.model_path = Path(self._temporary.name) / ('model-' + Path(self._temporary.name).name)
        self.model_path.mkdir()
        try:
            for path in self.checkpoint.iterdir():
                if path.is_file() and path.suffix in ('.json', '.safetensors'):
                    (self.model_path / path.name).symlink_to(path)
            root = Path(__file__).resolve().parents[2]
            env = dict(os.environ, PYTHONPATH=f'{root}:{self.cpu_root}', VLLM_USE_V1='1',
                VLLM_CPU_KVCACHE_SPACE=str(self.kv_space_gib), OMP_NUM_THREADS=str(self.threads),
                KMP_AFFINITY='disabled')
            if self.record_compute:
                # CPU timing must not enable GPU scheduling/allocation
                # diagnostics in the parent or another child process.
                env['VLLM_PROFILE_RELAY'] = '1'
            if self.record_operators:
                env['VLLM_CPU_PROFILE_RELAY_OPERATORS'] = '1'
            self.process = subprocess.Popen([self.python, '-m', 'sllm.resident.cpu_producer'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, cwd=root, start_new_session=True)
            os.set_blocking(self.process.stdin.fileno(), False)
            self._reader = threading.Thread(target=self._drain, name='cpu-producer-replies')
            self._reader.start()
            self.info = self.command('start', model=str(self.model_path),
                tensor_parallel_size=self.tp, max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_batched_tokens,
                warmup=self.warmup)
            if (not isinstance(self.info, dict) or self.info.get('model') != str(self.model_path)
                    or self.info.get('tensor_parallel_size') != self.tp):
                self._failed = True
                raise RuntimeError('CPU producer startup identity mismatch')
        except BaseException:
            self.close()
            raise

    def _drain(self):
        log = None
        try:
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                log = self.log_path.open('w')
            for raw in self.process.stdout:
                line = raw.decode(errors='replace')
                self._logs.append(line)
                if log:
                    log.write(line)
                    log.flush()
                if line.startswith('CPU_PRODUCER_REPLY '):
                    try:
                        self._replies.put(json.loads(line.split(' ', 1)[1]))
                    except ValueError:
                        self._replies.put(None)
        finally:
            if log:
                log.close()
            self._replies.put(None)

    def command(self, op, **fields):
        if {'sequence', 'op'} & fields.keys():
            raise ValueError('Client owns protocol fields')
        with self._lock:
            if self.process is None or self._closed or self._failed:
                raise RuntimeError('CPU producer is not available')
            # Compute the sequence inside the serialization lock.
            data = memoryview((json.dumps(dict(sequence=self._sequence, op=op, **fields)) + '\n').encode())
            deadline = time.monotonic() + self.timeout
            try:
                while data:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or self.process.poll() is not None:
                        raise TimeoutError('CPU producer command write timed out or process exited')
                    _, writable, _ = select.select([], [self.process.stdin], [], min(remaining, .1))
                    if not writable:
                        continue
                    try:
                        count = os.write(self.process.stdin.fileno(), data)
                    except BlockingIOError:
                        continue
                    if count <= 0:
                        raise OSError('CPU command write made no progress')
                    data = data[count:]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('CPU producer reply timed out')
                reply = self._replies.get(timeout=remaining)
                if (not isinstance(reply, dict) or type(reply.get('sequence')) is not int
                        or reply['sequence'] != self._sequence or type(reply.get('ok')) is not bool
                        or type(reply.get('fatal', False)) is not bool
                        or (reply.get('fatal', False) and reply['ok'])):
                    raise RuntimeError('Invalid CPU producer reply')
                if self._failed:
                    raise RuntimeError('CPU producer command was interrupted')
            except (OSError, TimeoutError, queue.Empty, RuntimeError) as exc:
                self._failed = True
                raise RuntimeError('CPU producer protocol failed:\n' + ''.join(self._logs)) from exc
            self._sequence += 1
            if not reply['ok']:
                if reply.get('fatal', False):
                    self._failed = True
                    raise RuntimeError(reply.get('error', 'CPU producer execution failed'))
                raise ValueError(reply.get('error', 'CPU producer rejected command'))
            return reply.get('result')

    def abort_pending(self):
        """Stop computation after a failed GPU handoff, retaining SHM ownership.

        This is fault teardown, never idle reclamation or model switching.
        Do not take the command lock: its owner can be awaiting GPU feedback.
        The temporary directory remains alive until close(), after GPU teardown.
        EOF wakes the command only after all producer descendants close stdout.
        """
        self._failed = True
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self):
        if self._closed:
            return
        process = self.process
        try:
            if process is not None and process.poll() is None:
                if not self._failed:
                    try:
                        self.command('shutdown')
                        process.wait(timeout=min(self.timeout, 5.0))
                    except (RuntimeError, ValueError, subprocess.TimeoutExpired):
                        self._failed = True
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        finally:
            if process is not None:
                # EngineCore/TP descendants inherit stdout. The launcher may
                # exit before them, leaving the reader blocked indefinitely
                # when closing its pipe. This session is exclusively owned by
                # this client (start_new_session=True); reap its remaining
                # descendants even when the launcher has already exited.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.stdin.close()
                if self._reader:
                    self._reader.join(timeout=15)
                process.stdout.close()
            if self._temporary:
                self._temporary.cleanup()
            self._closed = True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()
