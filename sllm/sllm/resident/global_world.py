"""Owned Global-WORLD launcher for single-host multi-node demonstrations."""
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import tempfile
import time

from sllm.resident.rank_transport import RankTransport


def _free_port(host='127.0.0.1'):
    with socket.socket() as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _identity(pid):
    """Avoid signaling a PID reused after an owned worker has exited."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


class GlobalWorldOwner:
    """Own and restart a Global WORLD made of logical local node agents.

    This owner is intentionally single-host: ``devices_by_node`` partitions
    visible GPUs into logical nodes. The process and lease lifecycle matches a
    real multi-node owner, while physical network performance is not implied.
    """
    concurrent_topologies = True

    def __init__(self, torchrun, devices_by_node, *, python=None, timeout=180,
                 rank_host='127.0.0.1', master_host='127.0.0.1',
                 worker_module='sllm.resident.worker', log_dir=None):
        if not devices_by_node or any(not isinstance(value, str) or not value
                                      for value in devices_by_node):
            raise ValueError('devices_by_node requires nonempty CUDA device groups')
        if timeout <= 0:
            raise ValueError('Global WORLD timeout must be positive')
        groups = [value.split(',') for value in devices_by_node]
        if (len({len(group) for group in groups}) != 1
                or any(not device.strip() for group in groups for device in group)
                or len({device.strip() for group in groups for device in group})
                != sum(map(len, groups))):
            raise ValueError('Global WORLD requires equal, disjoint CUDA device groups')
        self.torchrun = str(torchrun)
        self.devices_by_node = tuple(devices_by_node)
        self.python = python or os.environ.get('VIRTUAL_ENV_PYTHON') or 'python'
        self.timeout = timeout
        self.rank_host = rank_host
        self.master_host = master_host
        self.worker_module = worker_module
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.gpu_count = sum(len(value.split(',')) for value in self.devices_by_node)
        self._processes = []
        self._rank_identities = {}
        self._key_file = None
        self._transport = None
        self._temporary = None
        self._failed = False
        self._closed = False

    @property
    def failure(self):
        if self._failed:
            return 'Global WORLD owner failed'
        if any(process.poll() is not None for process, _ in self._processes):
            self._failed = True
            return 'Global WORLD rank launcher exited'
        if self._transport is not None and getattr(self._transport, '_failed', False):
            self._failed = True
            return 'Global WORLD rank transport failed'
        return None

    @property
    def pids(self):
        if self._transport is None:
            raise RuntimeError('Global WORLD is not started')
        return [item['result']['pid'] for item in self.command('status').ranks]

    def start(self):
        if self._transport is not None or self._closed:
            raise RuntimeError('Global WORLD owner already started or closed')
        self._temporary = tempfile.TemporaryDirectory(prefix='resident-global-owner-')
        root = Path(self._temporary.name)
        key_file = root / 'control-key'
        self._key_file = key_file
        key_file.write_text(secrets.token_hex(32))
        master_port = _free_port(self.master_host)
        rank_base = _free_port(self.rank_host)
        try:
            for node, devices in enumerate(self.devices_by_node):
                log_path = (self.log_dir / f'node{node}.log'
                            if self.log_dir is not None else root / f'node{node}.log')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open('w')
                command = [self.torchrun, '--nnodes', str(len(self.devices_by_node)),
                    '--nproc-per-node', str(len(devices.split(','))),
                    '--node-rank', str(node), '--master-addr', self.master_host,
                    '--master-port', str(master_port), '-m', self.worker_module,
                    '--timeout-seconds', str(int(self.timeout)), '--rank-tcp-host', self.rank_host,
                    '--rank-tcp-port-base', str(rank_base + node * len(devices.split(','))),
                    '--rank-key-file', str(key_file)]
                try:
                    process = subprocess.Popen(command,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices),
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                self._processes.append((process, log))
            endpoints = []
            for node, devices in enumerate(self.devices_by_node):
                for local_rank in range(len(devices.split(','))):
                    endpoints.append((self.rank_host, rank_base + len(endpoints)))
            self._transport = RankTransport(endpoints, timeout=self.timeout,
                                            key=key_file.read_text().strip())
            status = self._transport.command('status')
            self._rank_identities = {item['result']['pid']: _identity(item['result']['pid'])
                                     for item in status.ranks}
            return self
        except BaseException:
            self.close()
            raise

    def command(self, op, *, expected=True, **arguments):
        if self._transport is None or self.failure:
            raise RuntimeError('Global WORLD owner is not started or failed')
        return self._transport.command(op, expected=expected, **arguments)

    def abort_pending(self):
        self._failed = True
        if self._transport is not None:
            self._transport.abort_pending()

    def replacement(self):
        return type(self)(self.torchrun, self.devices_by_node, python=self.python,
            timeout=self.timeout, rank_host=self.rank_host, master_host=self.master_host,
            worker_module=self.worker_module, log_dir=self.log_dir)

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Startup can fail before status records rank PIDs, or a launcher can
        # exit first. The unique control-key path still identifies owned ranks.
        if self._key_file is not None:
            for entry in Path('/proc').iterdir():
                if not entry.name.isdecimal():
                    continue
                try:
                    arguments = (entry / 'cmdline').read_bytes().split(b'\0')
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                key = b'--rank-key-file'
                if key in arguments and arguments[arguments.index(key) + 1:][:1] == [
                        os.fsencode(self._key_file)]:
                    pid = int(entry.name)
                    self._rank_identities.setdefault(pid, _identity(pid))
        if self._transport is not None:
            try:
                self._transport.abort_pending()
                self._transport.close()
            except Exception:
                pass
            self._transport = None
        # torchrun starts ranks in separate sessions. Killing only the launcher's
        # process group misses them, especially after a launcher has already died.
        def signal_ranks(sig):
            for pid, identity in self._rank_identities.items():
                if identity is not None and _identity(pid) == identity:
                    try:
                        if os.getpgid(pid) == pid:
                            os.killpg(pid, sig)
                        else:
                            os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
        signal_ranks(signal.SIGTERM)
        # Always kill the process group, even when torchrun's parent already
        # exited and rank children were reparented by the kernel.
        for process, _ in self._processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process, log in self._processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            log.close()
        signal_ranks(signal.SIGKILL)
        deadline = time.monotonic() + 10
        while any(identity is not None and _identity(pid) == identity
                  for pid, identity in self._rank_identities.items()):
            if time.monotonic() >= deadline:
                raise TimeoutError('Owned Global WORLD ranks did not exit after SIGKILL')
            time.sleep(.01)
        self._rank_identities.clear()
        self._processes.clear()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
