"""Exclusive, ordered HTTP control of a node-owned resident GPU pool.

Inference commands are never retried. Explicit worker reconstruction rotates
the controller lease; only that operation has an idempotent retry token.
"""
from dataclasses import asdict
import json
import secrets
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sllm.resident.client import (ResidentReply, ResidentCommandRejected,
                                  ResidentCommandFailed, is_preflight_rejection)


class NodeAPI:
    def __init__(self, pool, key, *, pool_factory=None):
        if not isinstance(key, str) or not key:
            raise ValueError('Node control key is required')
        self.pool = pool
        self.key = key
        self.lease = None
        self.sequence = 0
        self.closed = False
        self.lock = threading.Lock()
        self._owner_lock = threading.Lock()
        self.pool_factory = pool_factory
        self._restart_reply = None

    def dispatch(self, method, path, payload=None):
        if method != 'POST' or path not in ('/v1/open', '/v1/command', '/v1/close',
                                           '/v1/abort', '/v1/restart', '/v1/health'):
            return 404, dict(error='Unknown node endpoint')
        if not isinstance(payload, dict):
            return 400, dict(error='Request must be an object')
        key = payload.get('key')
        if not isinstance(key, str) or not secrets.compare_digest(key.encode(), self.key.encode()):
            return 403, dict(error='Invalid node control key')
        if path in ('/v1/abort', '/v1/health'):
            # Neither operation may wait behind an in-flight GPU command.
            if payload.keys() != {'key', 'lease'}:
                return 400, dict(error='Invalid control fields')
            with self._owner_lock:
                lease = payload.get('lease')
                if (not isinstance(lease, str) or self.lease is None
                        or not secrets.compare_digest(lease.encode(), self.lease.encode())):
                    return 409, dict(error='Invalid controller lease')
                pool = self.pool
                if path == '/v1/health':
                    failure = getattr(pool, 'failure', None)
                    return 200, dict(failed=self.closed or bool(failure), error=str(failure or ''))
                self.closed = True
            abort = getattr(pool, 'abort_pending', None)
            if abort is None:
                return 501, dict(error='Node pool cannot interrupt pending commands')
            abort()
            return 200, dict(aborted=True)
        with self.lock:
            if path == '/v1/restart':
                if (payload.keys() != {'key', 'lease', 'restart_id'}
                        or not isinstance(payload['restart_id'], str)
                        or len(payload['restart_id']) != 64):
                    return 400, dict(error='Invalid restart fields')
                if self.pool_factory is None:
                    return 501, dict(error='Node agent does not own a restart factory')
                identity = (payload['lease'], payload['restart_id'])
                if self._restart_reply is not None and self._restart_reply[0] == identity:
                    return 200, dict(self._restart_reply[1])
                if self.lease is None or payload['lease'] != self.lease:
                    return 409, dict(error='Invalid controller lease')
                self.closed = True
                replacement = None
                try:
                    self.pool.close()
                    replacement = self.pool_factory()
                    if replacement is self.pool:
                        raise ValueError('Restart requires a fresh pool')
                    replacement.command('status')
                except Exception as exc:
                    if replacement is not None and replacement is not self.pool:
                        replacement.close()
                    return 503, dict(error=f'Node reconstruction failed: {exc}')
                with self._owner_lock:
                    self.pool = replacement
                    self.lease, self.sequence, self.closed = secrets.token_hex(32), 0, False
                result = dict(lease=self.lease, gpu_count=self.pool.gpu_count)
                self._restart_reply = identity, result
                return 200, dict(result)
            if self.closed and path != '/v1/close':
                return 503, dict(error='Node pool requires restart')
            if path == '/v1/open':
                if payload.keys() != {'key'}:
                    return 400, dict(error='Invalid open fields')
                if self.lease is not None:
                    return 409, dict(error='Node pool already has a controller')
                self.lease = secrets.token_hex(32)
                return 200, dict(lease=self.lease, gpu_count=self.pool.gpu_count)
            lease = payload.get('lease')
            if not isinstance(lease, str) or self.lease is None or not secrets.compare_digest(lease.encode(), self.lease.encode()):
                return 409, dict(error='Invalid controller lease')
            if path == '/v1/close':
                if payload.keys() != {'key', 'lease'}:
                    return 400, dict(error='Invalid close fields')
                self.closed = True
                self.pool.close()
                return 200, dict(closed=True)
            if payload.keys() != {'key', 'lease', 'sequence', 'op', 'expected', 'arguments'}:
                return 400, dict(error='Invalid command fields')
            sequence = payload['sequence']
            op = payload['op']
            arguments = payload['arguments']
            if type(sequence) is not int or sequence != self.sequence:
                return 409, dict(error='Unexpected node command sequence')
            if (op not in ('status', 'prepare', 'prepare_weights', 'probe',
                           'load', 'lazy_init', 'unload', 'generate',
                           'profile_memory', 'weights_ready', 'submit', 'step',
                           'collect', 'abort')
                    or type(payload['expected']) is not bool or not isinstance(arguments, dict)
                    or {'sequence', 'op', 'expected'} & arguments.keys()):
                return 400, dict(error='Invalid node command')
            try:
                reply = self.pool.command(op, expected=payload['expected'], **arguments)
            except ResidentCommandRejected as exc:
                self.sequence += 1
                return 200, dict(sequence=sequence, rejection=exc.reply)
            except ResidentCommandFailed as exc:
                self.sequence += 1
                fatal = any(isinstance(item, dict) and item.get('fatal')
                            for item in exc.reply.ranks)
                if fatal:
                    self.closed = True
                    return 503, dict(error=str(exc))
                return 200, dict(sequence=sequence, reply=asdict(exc.reply))
            except Exception as exc:
                self.closed = True
                return 503, dict(error=f'{type(exc).__name__}: {exc}')
            if self.closed:
                return 503, dict(error='Node pool interrupted; restart required')
            self.sequence += 1
            if any(isinstance(item, dict) and item.get('fatal') for item in reply.ranks):
                self.closed = True
            return 200, dict(sequence=sequence, reply=asdict(reply))


class RemotePoolClient:
    def __init__(self, url, key, *, timeout=120):
        if not isinstance(url, str) or not url.startswith(('http://', 'https://')) or not key or timeout <= 0:
            raise ValueError('Invalid remote node configuration')
        self.url = url.rstrip('/')
        self.key = key
        self.timeout = timeout
        self.lease = None
        self.gpu_count = None
        self.sequence = 0
        self.failed = False
        self.closed = False
        self.lock = threading.Lock()
        self._lineage = dict(lease=None, pending=None)
        self._restart = False
        self._last_health = 0.

    @property
    def failure(self):
        if self.failed or self.closed:
            return 'Remote pool failed or closed'
        # Host-only liveness; never submit a CUDA/status command on hot ranks.
        if self.lease is not None and time.monotonic() - self._last_health >= 1.:
            self._last_health = time.monotonic()
            try:
                result = self._post('/v1/health', lease=self.lease,
                                    _timeout=min(self.timeout, 5.))
                if type(result.get('failed')) is not bool:
                    raise ValueError('Invalid node health response')
                self.failed = result['failed']
            except Exception:
                self.failed = True
        return 'Remote owner unavailable' if self.failed else None

    def replacement(self):
        value = type(self)(self.url, self.key, timeout=self.timeout)
        value._lineage = self._lineage
        value._restart = True
        return value

    def _post(self, path, *, _timeout=None, **body):
        request = Request(self.url + path, data=json.dumps(dict(key=self.key, **body)).encode(),
                          headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=self.timeout if _timeout is None else _timeout) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get('error', 'Node request failed')
            except (ValueError, AttributeError):
                detail = 'Node request failed'
            raise RuntimeError(f'Node HTTP {exc.code}: {detail}') from exc

    def start(self):
        with self.lock:
            if self.lease is not None or self.failed or self.closed:
                raise RuntimeError('Remote pool already started or unusable')
            try:
                if self._restart:
                    if self._lineage['pending'] is None:
                        self._lineage['pending'] = (self._lineage['lease'], secrets.token_hex(32))
                    previous, token = self._lineage['pending']
                    result = self._post('/v1/restart', lease=previous, restart_id=token)
                else:
                    result = self._post('/v1/open')
                if (not isinstance(result.get('lease'), str) or not result['lease']
                        or type(result.get('gpu_count')) is not int or result['gpu_count'] < 1):
                    raise ValueError('Invalid node open response')
                self.lease = result['lease']
                self.gpu_count = result['gpu_count']
                self._lineage.update(lease=self.lease, pending=None)
            except Exception:
                self.failed = True
                raise

    def abort_pending(self):
        # Do not acquire self.lock: the pending HTTP command owns it.
        self.failed = True
        if self.lease is not None and not self.closed:
            result = self._post('/v1/abort', lease=self.lease,
                                _timeout=min(self.timeout, 5.0))
            if result.get('aborted') is not True:
                raise RuntimeError('Invalid remote abort response')

    def command(self, op, *, expected=True, **arguments):
        with self.lock:
            if self.lease is None or self.failed or self.closed:
                raise RuntimeError('Remote pool is not started or failed')
            try:
                result = self._post('/v1/command', lease=self.lease, sequence=self.sequence,
                                    op=op, expected=expected, arguments=arguments)
                if self.failed:
                    raise RuntimeError('Remote pool interrupted; restart required')
                if type(result.get('sequence')) is not int or result['sequence'] != self.sequence:
                    raise ValueError('Remote reply sequence mismatch')
                if 'rejection' in result:
                    rejection = result['rejection']
                    if not isinstance(rejection, dict) or not is_preflight_rejection(
                            rejection, rejection.get('sequence'), self.gpu_count):
                        raise ValueError('Invalid remote preflight rejection')
                    self.sequence += 1
                    raise ResidentCommandRejected(op, rejection)
                reply = ResidentReply(**result['reply'])
                if len(reply.ranks) != self.gpu_count and op != 'prepare':
                    raise ValueError('Remote reply does not cover pool ranks')
                self.sequence += 1
                fatal = any(isinstance(item, dict) and item.get('fatal')
                            for item in reply.ranks)
                if fatal:
                    self.failed = True
            except ResidentCommandRejected:
                raise
            except Exception:
                self.failed = True
                raise
            errors = [item.get('error') for item in reply.ranks
                      if isinstance(item, dict) and item.get('error')]
            if expected and errors and not fatal:
                # The HTTP exchange and resident sequence completed normally;
                # a nonfatal model/background error should make the controller
                # quarantine this node, while retaining the remote client for
                # diagnostic status and orderly close.
                raise ResidentCommandFailed(op, reply)
            return reply

    @property
    def pids(self):
        return [item['result']['pid'] for item in self.command('status').ranks]

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.lease is not None:
                try:
                    self._post('/v1/close', lease=self.lease)
                except Exception:
                    self.failed = True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()
