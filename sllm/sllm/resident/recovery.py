"""Node-local failure supervision; failed requests are never replayed."""
import logging
import math
import threading
import time


class RecoverySupervisor:
    def __init__(self, controller, pool_factories, *, relay=None, producer_factory=None,
                 interval=.1, retry_initial=1., retry_max=30.):
        if (not pool_factories or set(pool_factories) - set(controller.pools)
                or any(not callable(value) for value in pool_factories.values())):
            raise ValueError('Recovery requires factories for owned controller pools')
        if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in (interval, retry_initial, retry_max)) or retry_initial > retry_max):
            raise ValueError('Invalid recovery polling/backoff settings')
        if relay is not None and (relay.controller is not controller or len(controller.pools) != 1
                                  or not callable(producer_factory)):
            raise ValueError('Local relay recovery requires its controller and a producer factory')
        self.controller, self.relay = controller, relay
        self.factories, self.producer_factory = dict(pool_factories), producer_factory
        self.interval, self.retry_initial, self.retry_max = interval, retry_initial, retry_max
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._threads = []
        self._state = {node: dict(phase='ready', attempts=0, recoveries=0, error=None,
                                 next_retry_at=None) for node in pool_factories}
        self._started = False

    def status(self):
        with self._lock:
            return {node: dict(state) for node, state in self._state.items()}

    def _update(self, node, **values):
        with self._lock:
            self._state[node].update(values)

    @staticmethod
    def _failure(owner):
        return (getattr(owner, 'failure', None)
                or ('Control owner failed' if getattr(owner, '_failed', False)
                    or getattr(owner, 'failed', False) else None))

    def _detect(self, node):
        with self.controller._lock:
            pool = self.controller.pools[node]
            quarantined = node in self.controller._quarantined
        error = self._failure(pool)
        if self.relay is not None:
            error = error or next((self._failure(model.producer) or 'CPU handoff failed'
                for model in self.relay.models.values()
                if model.failed or self._failure(model.producer)), None)
        if error and not quarantined:
            self.controller._quarantine(node, RuntimeError(str(error)))
        return quarantined or bool(error)

    def _seal(self, node, sealed):
        if self.relay is not None:
            self.relay._recovering.set()
        pool = self.controller.pools[node]
        if pool is not sealed:
            abort = getattr(pool, 'abort_pending', None)
            if abort is not None:
                abort()
        if self.relay is not None:
            for model in self.relay.models.values():
                producer = model.producer
                if getattr(producer, 'busy', False) or self._failure(producer):
                    # Do not acquire the producer command or route mutex: its
                    # owner may be waiting for feedback from the failed GPU.
                    model.failed = True
                    producer.abort_pending()
        return pool

    def _drained(self, node):
        c = self.controller
        with c._lock:
            names = {name for name, instance in c._instances.items() if instance.node == node}
            if self.relay is not None:
                names.update(self.relay.prefix + name for name in self.relay.models)
            if c._prewarming or any(c._operations.get(name) or c._instance_admissions.get(name)
                    or c._admitted_requests.get(name) or name in c._pending_creates
                    or name in c._expiring or name in c._tearing_down for name in names):
                return False
            # CPU completion callbacks release the original semaphore. Never
            # replace it and accidentally admit two AMX owners on one node.
            for physical in set(c.physical_nodes[node]):
                lease = c._amx_leases[physical]
                if not lease.acquire(blocking=False):
                    return False
                lease.release()
        return True

    def _recover(self, node):
        if self.relay is None:
            return self.controller.restart_node(node, self.factories[node])
        result = self.relay.recover(self.factories[node], self.producer_factory)
        self._update(node, phase='preparing_weights')
        for model in self.relay.models.values():
            producer = model.producer
            self.controller._command(node, 'prepare_weights', model=str(producer.model_path),
                gpu_tp=model.tp, shm_info=producer.info['shm_info'],
                chunk_tokens=producer.max_num_batched_tokens)
        return result

    def _run(self, node):
        failures, retry_at, sealed = 0, 0., None
        while not self._stop.is_set():
            try:
                if self._detect(node):
                    sealed = self._seal(node, sealed)
                    if time.monotonic() < retry_at:
                        self._update(node, phase='backoff')
                    elif not self._drained(node):
                        self._update(node, phase='draining')
                    else:
                        with self._lock:
                            self._state[node]['attempts'] += 1
                        self._update(node, phase='restarting', next_retry_at=None)
                        self._recover(node)
                        if self.relay is not None:
                            self.relay._recovering.clear()
                        with self._lock:
                            self._state[node]['recoveries'] += 1
                        self._update(node, phase='ready', error=None, next_retry_at=None)
                        failures, retry_at, sealed = 0, 0., None
            except Exception as exc:
                self.controller._quarantine(node, exc)
                failures += 1
                delay = min(self.retry_max, self.retry_initial * 2 ** min(failures - 1, 30))
                retry_at = time.monotonic() + delay
                self._update(node, phase='backoff', error=f'{type(exc).__name__}: {exc}',
                             next_retry_at=retry_at)
                logging.exception('Resident recovery failed on %s; retry in %.3fs', node, delay)
            self._stop.wait(self.interval)

    def start(self):
        if self._started or self._stop.is_set():
            raise RuntimeError('Recovery supervisor already started or closed')
        self._started = True
        for node in self.factories:
            thread = threading.Thread(target=self._run, args=(node,),
                                      name=f'resident-recovery-{node}')
            self._threads.append(thread)
            thread.start()
        return self

    def close(self):
        self._stop.set()
        for thread in self._threads:
            thread.join()
        for node in self.factories:
            self._update(node, phase='stopped')

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
