"""Two-phase local rank routing, independent of GPU/world collectives.

Every target validates before any target executes. Rank locks are acquired in
numeric order; disjoint topologies can run concurrently. Lost replies poison the
transport, since device ownership cannot be safely retried.
"""
import json
import socket
import secrets
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack

from sllm.resident.client import (ResidentReply, ResidentCommandRejected,
                                  ResidentCommandFailed)


def write(stream, value):
    stream.write((json.dumps(value)+'\n').encode())
    stream.flush()


def read(stream):
    line = stream.readline()
    if not line:
        raise RuntimeError('Rank channel closed')
    return json.loads(line)


class RankTransport:
    concurrent_topologies = True

    def __init__(self, paths, *, timeout=120, key=None):
        if any(isinstance(path, tuple) for path in paths) and (not isinstance(key, str) or not key):
            raise ValueError('TCP rank endpoints require a control key')
        self.gpu_count = len(paths)
        self.timeout = timeout
        self._locks = [threading.RLock() for _ in paths]
        self._metadata = threading.Lock()
        self._sequence = 0
        self._failed = False
        self._sockets = []
        self._streams = []
        deadline = time.monotonic() + timeout
        try:
            for path in paths:
                tcp = isinstance(path, tuple)
                connection = socket.socket(socket.AF_INET if tcp else socket.AF_UNIX)
                connection.settimeout(timeout)
                self._sockets.append(connection)
                # bind creates the pathname before listen is ready. Retrying
                # connection establishment is safe: no command was sent yet.
                while True:
                    try:
                        connection.connect(path if tcp else str(path))
                        break
                    except (FileNotFoundError, ConnectionRefusedError, ConnectionAbortedError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Rank socket did not become ready')
                        connection.close()
                        connection = socket.socket(socket.AF_INET if tcp else socket.AF_UNIX)
                        connection.settimeout(timeout)
                        self._sockets[-1] = connection
                        time.sleep(.01)
                stream = connection.makefile('rwb')
                self._streams.append(stream)
                if tcp:
                    write(stream, dict(key=key))
                    if read(stream) != {'authenticated': True}:
                        raise RuntimeError('Rank authentication failed')
        except BaseException:
            self.close()
            raise

    def abort_pending(self):
        """Wake all rank reads after an unrecoverable peer dependency fails."""
        with self._metadata:
            self._failed = True
        # No rank locks: their owners may be blocked in these socket reads.
        for connection in self._sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def command(self, op, *, expected=True, **arguments):
        if {'sequence', 'op'} & arguments.keys():
            raise ValueError('Transport owns protocol fields')
        ranks = (arguments['topology']['ranks'] if 'topology' in arguments
                 else list(range(self.gpu_count)))
        if (not ranks or any(type(rank) is not int or not 0 <= rank < self.gpu_count for rank in ranks)
                or len(set(ranks)) != len(ranks)):
            raise ValueError('Invalid routed ranks')
        with ExitStack() as locks:
            for rank in sorted(ranks):
                locks.enter_context(self._locks[rank])
            with self._metadata:
                if self._failed:
                    raise RuntimeError('Rank transport failed; restart pool')
                sequence = self._sequence
                self._sequence += 1
            command = dict(sequence=sequence, op=op, **arguments)
            def exchange(rank, message):
                write(self._streams[rank], message)
                reply = read(self._streams[rank])
                if reply.get('sequence') != sequence:
                    raise RuntimeError('Rank reply sequence mismatch')
                return rank, reply
            try:
                with ThreadPoolExecutor(len(ranks)) as executor:
                    validation = {}
                    validating = [executor.submit(exchange, rank,
                        dict(phase='validate', command=command)) for rank in ranks]
                    try:
                        for future in as_completed(validating):
                            rank, reply = future.result()
                            validation[rank] = reply
                    except BaseException:
                        # Wake peers before joining the executor. A dead rank
                        # can otherwise leave validation reads blocked until
                        # the full socket deadline.
                        self.abort_pending()
                        raise
                    rejected = any(reply.get('error') for reply in validation.values())
                    outcomes = {}
                    pending = [executor.submit(exchange, rank,
                        dict(phase='cancel' if rejected else 'execute', sequence=sequence))
                        for rank in ranks]
                    try:
                        for future in as_completed(pending):
                            rank, reply = future.result()
                            outcomes[rank] = reply
                            if reply.get('outcome', {}).get('fatal'):
                                # Peers may be stuck in the failed rank's
                                # collective. Seal their sockets before the
                                # executor joins, preserving the fatal reply.
                                self.abort_pending()
                    except Exception as exc:
                        self.abort_pending()
                        # A failed rank can leave its peers inside a collective.
                        # Preserve the primary rank traceback when those peers
                        # subsequently time out instead of reporting only EOF.
                        failures = {rank: reply['outcome']
                                    for rank, reply in outcomes.items()
                                    if reply.get('outcome', {}).get('error')}
                        if failures:
                            raise RuntimeError(
                                f'Rank operation {op} failed before peer replies: '
                                f'{failures}') from exc
                        raise
                if rejected:
                    errors = [validation.get(rank, {}).get('error') for rank in range(self.gpu_count)]
                    rejection = dict(sequence=sequence, next_sequence=sequence,
                                     accepted=False, ok=False, ranks=errors)
                    if expected:
                        raise ResidentCommandRejected(op, rejection)
                    return ResidentReply(sequence, sequence, errors)
                values = [outcomes[rank]['outcome'] if rank in outcomes else
                          dict(result=dict(idle=True), error=None, fatal=False)
                          for rank in range(self.gpu_count)]
                ok = not any(item.get('error') for item in values)
                if any(item.get('fatal') for item in values):
                    self._failed = True
                reply = ResidentReply(sequence, sequence+1, values)
                if ok is not expected:
                    if not ok:
                        raise ResidentCommandFailed(op, reply)
                    raise RuntimeError(f'Rank operation failed: {values}')
                return reply
            except (ResidentCommandRejected, ResidentCommandFailed):
                # Every rank replied: a nonfatal operation error does not
                # invalidate the channel. Fatal outcomes already sealed it.
                raise
            except BaseException:
                self._failed = True
                raise

    def close(self):
        for stream in self._streams:
            stream.close()
        for connection in self._sockets:
            connection.close()
        self._streams.clear()
        self._sockets.clear()


def serve_rank(path, loop, *, key=None):
    """One persistent channel per GPU process; no background model threads."""
    import time
    tcp = isinstance(path, tuple)
    if tcp and (not isinstance(key, str) or not key):
        raise ValueError('TCP rank listener requires a control key')
    listener = socket.socket(socket.AF_INET if tcp else socket.AF_UNIX)
    listener.bind(path if tcp else str(path))
    listener.listen(1)
    try:
        while True:
            connection, _ = listener.accept()
            stream = connection.makefile('rwb')
            if not tcp:
                break
            connection.settimeout(10)
            try:
                supplied = read(stream).get('key')
                valid = isinstance(supplied, str) and secrets.compare_digest(supplied.encode(), key.encode())
                write(stream, dict(authenticated=valid))
                if valid:
                    connection.settimeout(None)
                    break
            except (OSError, ValueError, AttributeError):
                pass
            stream.close()
            connection.close()
        with connection, stream:
            pending = None
            last_sequence = -1
            while True:
                message = read(stream)
                if message['phase'] == 'validate':
                    if pending is not None:
                        raise RuntimeError('Rank already has a pending command')
                    command = message['command']
                    sequence = command.get('sequence')
                    if type(sequence) is not int or sequence <= last_sequence:
                        raise RuntimeError('Rank command sequence must increase')
                    last_sequence = sequence
                    loop.next_sequence = sequence
                    started = time.perf_counter()
                    error = None
                    try:
                        loop.validate(command)
                    except Exception as exc:
                        error = f'{type(exc).__name__}: {exc}'
                    pending = command
                    validation_seconds = time.perf_counter()-started
                    write(stream, dict(sequence=command['sequence'], error=error))
                    continue
                if pending is None or message['sequence'] != pending['sequence']:
                    raise RuntimeError('No matching validated rank command')
                command, pending = pending, None
                if message['phase'] == 'cancel':
                    cancel = getattr(loop, 'cancel', None)
                    if cancel is not None:
                        cancel(command)
                    write(stream, dict(sequence=command['sequence']))
                    continue
                if message['phase'] != 'execute':
                    raise ValueError('Invalid rank phase')
                if error is not None:
                    raise RuntimeError('Cannot execute a rejected rank command')
                started = time.perf_counter()
                result, error = None, None
                failure_traceback = None
                try:
                    result = loop.execute(command)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    failure_traceback = traceback.format_exc()
                write(stream, dict(sequence=command['sequence'], outcome=dict(
                    result=result, error=error, fatal=loop.registry._poisoned,
                    **({'traceback': failure_traceback} if failure_traceback else {}),
                    validation_seconds=validation_seconds,
                    execution_seconds=time.perf_counter()-started)))
                if command['op'] == 'shutdown' or loop.registry._poisoned:
                    return
    except (BrokenPipeError, ConnectionResetError):
        # The controller can seal all channels when a peer rank fails.
        # This rank may finish its collective and attempt a late reply.
        loop.registry._poisoned = True
    finally:
        listener.close()
        if not tcp:
            path.unlink(missing_ok=True)
