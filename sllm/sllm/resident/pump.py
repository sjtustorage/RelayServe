"""Single-owner iteration pump for an already initialized resident instance."""
from concurrent.futures import Future, InvalidStateError
from collections import deque
import threading
from sllm.resident.client import ResidentCommandRejected


class InstancePump:
    def __init__(self, controller, instance):
        self.controller = controller
        self.instance = instance
        self._queue = deque()
        self._active = {}
        self._token_callbacks = {}
        self._admissions = {}
        self._condition = threading.Condition()
        self._closing = False
        self._failure = None
        self._recovery = None
        self._recovery_ready = None
        self._background_recoveries = set()
        self._load_threads = set()
        self._load_cleanup_errors = []
        self._first_output = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, prompt_token_ids, *, request_id, max_tokens=8, on_tokens=None):
        future = Future()
        with self._condition:
            if self._closing or self._failure:
                raise RuntimeError('Instance pump is closed') from self._failure
            if request_id in self._active or any(row[0] == request_id for row in self._queue):
                raise ValueError('Duplicate request ID')
            admission = self.controller.instance_admission(self.instance)
            admission.__enter__()
            self._admissions[request_id] = admission
            if on_tokens is not None:
                self._token_callbacks[request_id] = on_tokens
            self._queue.append((request_id, list(prompt_token_ids), max_tokens, future))
            self._condition.notify()
        return future

    def recover_to(self, target):
        """Queue a native recovery on the iteration owner, preserving Futures."""
        future = Future()
        with self._condition:
            if self._closing or self._failure or self._recovery is not None:
                raise RuntimeError('Pump cannot accept recovery')
            admission = self.controller.instance_admission(target)
            admission.__enter__()
            self._recovery = (target, future, admission)
            self._condition.notify()
        return future

    def recover_when_ready(self, native_future, *, release_source=True):
        """Recover after a background native load, then release borrowed ranks.

        Load failure leaves the serving temporary instance untouched. Recovery
        failure follows recover_to's ownership rules; release happens only
        after all ranks acknowledged native ownership.
        """
        completion = Future()
        with self._condition:
            if self._closing or self._failure:
                raise RuntimeError('Instance pump is closed')
            source = self.instance
            admission = self.controller.instance_admission(source)
            admission.__enter__()
        pinned = True

        def unpin():
            nonlocal pinned
            with self._condition:
                if pinned:
                    admission.__exit__(None, None, None)
                    pinned = False

        completion.add_done_callback(lambda future: unpin() if future.cancelled() else None)
        with self._condition:
            # close/fatal iteration failure may have happened after the source
            # admission was acquired. Do not register an orphaned load callback.
            if self._closing or self._failure:
                completion.cancel()
                return completion
            self._background_recoveries.add(completion)

        def forget(future):
            with self._condition:
                self._background_recoveries.discard(future)

        completion.add_done_callback(forget)

        def loaded(future):
            if not completion.set_running_or_notify_cancel():
                unpin()
                return
            try:
                target = future.result()
                with self._condition:
                    if self.instance != source:
                        raise ValueError('Pump instance changed before native load completed')
                    recovery = self.recover_to(target)
                    unpin()
            except BaseException as exc:
                unpin()
                completion.set_exception(exc)
                return

            def recovered(result):
                try:
                    evidence = result.result()
                    if release_source:
                        self.controller.release(source)
                    completion.set_result(evidence)
                except BaseException as exc:
                    completion.set_exception(exc)

            recovery.add_done_callback(recovered)

        native_future.add_done_callback(loaded)
        return completion

    def materialize_and_recover(self, reservation, layers, *,
                                after_first_output=False, **load_options):
        """Load a reserved native slot, poll H2D, then recover between steps.

        Native construction uses normal serialized rank commands. Only native
        transfers continue outside those commands. Cancellation drains loading
        and releases the unused destination; it never releases the serving PP.

        Hydra gives the serving PP its first output before native construction
        can take shared rank locks. This is an output event, not a timer or a
        first-chunk approximation; long prompts may require many iterations.
        """
        if type(after_first_output) is not bool:
            raise ValueError('after_first_output must be boolean')
        completion = Future()
        stop = threading.Event()
        with self._condition:
            if (self._closing or self._failure or self._background_recoveries
                    or self._load_threads or self._recovery is not None):
                raise RuntimeError('Pump cannot start native loading')
            source = self.instance
            source_pin = self.controller.instance_admission(source)
            target_pin = self.controller.instance_admission(reservation)
            source_pin.__enter__()
            try:
                target_pin.__enter__()
            except BaseException:
                source_pin.__exit__(None, None, None)
                raise
            self._background_recoveries.add(completion)
            completion.add_done_callback(lambda value: stop.set() if value.cancelled() else None)

            def load():
                target = reservation
                target_pinned = True
                source_pinned = True
                recovery_queued = False
                try:
                    while after_first_output and not self._first_output.wait(.01):
                        with self._condition:
                            if stop.is_set() or self._closing:
                                completion.cancel()
                                return
                            if self._failure is not None:
                                raise RuntimeError('Serving PP failed before first output') from self._failure
                    if stop.is_set():
                        return
                    target = self.controller.materialize(target, **load_options)
                    if not target.ready:
                        target = self.controller.lazy_init(target, layers, None)
                    while not stop.is_set() and not self.controller.weights_ready(target):
                        stop.wait(.01)
                    with self._condition:
                        if stop.is_set() or self._closing or self._failure:
                            completion.cancel()
                            return
                        if self.instance != source:
                            raise ValueError('Pump instance changed during native loading')
                        if not completion.set_running_or_notify_cancel():
                            return
                        # Transfer the native pin while the iteration owner is
                        # excluded. No recovery can see the loader's extra pin.
                        target_pin.__exit__(None, None, None)
                        target_pinned = False
                        recovery = self.recover_to(target)
                        recovery_queued = True
                        source_pin.__exit__(None, None, None)
                        source_pinned = False

                    def recovered(value):
                        try:
                            try:
                                evidence = value.result()
                            except (ResidentCommandRejected, ValueError):
                                # A rejected migration preserves source
                                # ownership. Its unused native load must not
                                # occupy capacity until a later idle reap.
                                if self.instance == source:
                                    self.controller.release(target)
                                raise
                            self.controller.release(source)
                            completion.set_result(evidence)
                        except BaseException as exc:
                            completion.set_exception(exc)

                    # Drop the source load pin before installing callbacks:
                    # an already finished recovery invokes callbacks inline.
                    recovery.add_done_callback(recovered)
                except BaseException as exc:
                    if not completion.done():
                        completion.set_exception(exc)
                finally:
                    if source_pinned:
                        source_pin.__exit__(None, None, None)
                    if target_pinned:
                        target_pin.__exit__(None, None, None)
                    if not recovery_queued:
                        try:
                            self.controller.release(target)
                        except BaseException as exc:
                            with self._condition:
                                self._load_cleanup_errors.append(exc)
                    with self._condition:
                        self._load_threads.discard(threading.current_thread())
                        if completion.done():
                            self._background_recoveries.discard(completion)

            def forget(value):
                with self._condition:
                    self._background_recoveries.discard(value)

            completion.add_done_callback(forget)
            thread = None
            try:
                thread = threading.Thread(target=load, name='resident-native-load', daemon=True)
                self._load_threads.add(thread)
                thread.start()
            except BaseException as exc:
                self._load_threads.discard(thread)
                source_pin.__exit__(None, None, None)
                target_pin.__exit__(None, None, None)
                try:
                    self.controller.release(reservation)
                except BaseException as cleanup_exc:
                    self._load_cleanup_errors.append(cleanup_exc)
                completion.set_exception(exc)
        return completion

    def _recover(self):
        # Hold the admission mutex while changing generations. Submitted IDs
        # and queued requests keep their original Future, but their lifecycle
        # pins must move together with the scheduler ownership.
        with self._condition:
            if self._recovery is None:
                return
            if ((self._active or self._queue)
                    and self._recovery_ready is not True):
                # Keep advancing the source until CPU exit feedback and the
                # internal PP KV transition have completed. Native H2D may
                # finish earlier; that alone cannot authorize live recovery.
                return
            target, future, target_admission = self._recovery
            source = self.instance
            self._recovery = None
            if not future.set_running_or_notify_cancel():
                target_admission.__exit__(None, None, None)
                return
            target_pin_released = False
            try:
                with self.controller._instance_operation(source), self.controller._instance_operation(target):
                    target_admission.__exit__(None, None, None)
                    target_pin_released = True
                    for admission in self._admissions.values():
                        admission.__exit__(None, None, None)
                    ids = list(self._admissions)
                    self._admissions.clear()
                    try:
                        result = self.controller.recover_pp(source, target)
                    except (ResidentCommandRejected, ValueError) as exc:
                        for request_id in ids:
                            admission = self.controller.instance_admission(source)
                            admission.__enter__()
                            self._admissions[request_id] = admission
                        future.set_exception(exc)
                        return
                    self.instance = target
                    for request_id in ids:
                        admission = self.controller.instance_admission(target)
                        admission.__enter__()
                        self._admissions[request_id] = admission
                future.set_result(result)
            except BaseException as exc:
                if not target_pin_released:
                    target_admission.__exit__(None, None, None)
                if not future.done():
                    future.set_exception(exc)
                raise

    def close(self):
        """Stop admission and drain accepted requests before returning."""
        with self._condition:
            self._closing = True
            for recovery in list(self._background_recoveries):
                recovery.cancel()
            self._condition.notify()
        self._thread.join()
        with self._condition:
            loading = list(self._load_threads)
        for thread in loading:
            thread.join()
        if self._failure:
            raise RuntimeError('Instance pump failed') from self._failure
        if self._load_cleanup_errors:
            raise RuntimeError('Native load cleanup failed') from self._load_cleanup_errors[0]

    def _run(self):
        try:
            while True:
                with self._condition:
                    while not self._queue and not self._active and not self._closing and self._recovery is None:
                        self._condition.wait()
                    if self._closing and not self._queue and not self._active and self._recovery is None:
                        return
                self._recover()
                with self._condition:
                    incoming = list(self._queue)
                    self._queue.clear()
                    for request_id, _, _, future in incoming:
                        self._active[request_id] = future
                for request_id, prompt, maximum, future in incoming:
                    if future.cancelled():
                        with self._condition:
                            self._active.pop(request_id)
                            self._token_callbacks.pop(request_id, None)
                            admission = self._admissions.pop(request_id)
                        admission.__exit__(None, None, None)
                        continue
                    try:
                        self.controller.submit(self.instance, prompt,
                            request_id=request_id, max_tokens=maximum)
                    except ResidentCommandRejected as exc:
                        # Every rank rejected before device execution. Other
                        # requests can safely keep using the current engine.
                        with self._condition:
                            self._active.pop(request_id)
                            self._token_callbacks.pop(request_id, None)
                            admission = self._admissions.pop(request_id)
                        admission.__exit__(None, None, None)
                        try:
                            future.set_exception(exc)
                        except InvalidStateError:
                            if not future.cancelled():
                                raise
                # Only this iteration owner mutates the scheduler. Future.cancel
                # can be called by HTTP/client threads, but distributed aborts
                # run between iterations alongside ordinary admission.
                for request_id, future in list(self._active.items()):
                    if future.cancelled():
                        self.controller.cancel_request(self.instance, request_id)
                        with self._condition:
                            self._active.pop(request_id)
                            self._token_callbacks.pop(request_id, None)
                            admission = self._admissions.pop(request_id)
                        admission.__exit__(None, None, None)
                with self._condition:
                    if not self._active:
                        continue
                result = self.controller.advance(self.instance)
                with self._condition:
                    self._recovery_ready = result.get('recovery_ready', True)
                if result.get('first_output_request_ids') or result['finished_request_ids']:
                    self._first_output.set()
                for request_id, tokens in result.get('token_updates', {}).items():
                    future = self._active[request_id]
                    callback = self._token_callbacks.get(request_id)
                    if callback is not None and not future.cancelled():
                        try:
                            callback(tokens)
                        except Exception as exc:
                            # A failed consumer belongs to one request. Abort
                            # on this iteration owner and keep peers serving.
                            self.controller.cancel_request(self.instance, request_id)
                            with self._condition:
                                self._active.pop(request_id)
                                self._token_callbacks.pop(request_id, None)
                                admission = self._admissions.pop(request_id)
                            admission.__exit__(None, None, None)
                            try:
                                future.set_exception(exc)
                            except InvalidStateError:
                                if not future.cancelled():
                                    raise
                for request_id in result['finished_request_ids']:
                    if request_id not in self._active:
                        continue
                    output = self.controller.collect(self.instance, request_id)
                    with self._condition:
                        future = self._active.pop(request_id)
                        self._token_callbacks.pop(request_id, None)
                    self._admissions.pop(request_id).__exit__(None, None, None)
                    try:
                        future.set_result(output)
                    except InvalidStateError:
                        if not future.cancelled():
                            raise
        except BaseException as exc:
            with self._condition:
                self._failure = exc
                futures = list(self._active.values()) + [row[3] for row in self._queue]
                self._queue.clear()
                self._active.clear()
                self._token_callbacks.clear()
                admissions = list(self._admissions.values())
                self._admissions.clear()
                recovery = self._recovery
                self._recovery = None
                for background in list(self._background_recoveries):
                    background.cancel()
            if recovery is not None:
                _, future, admission = recovery
                admission.__exit__(None, None, None)
                if not future.done():
                    future.set_exception(exc)
            for admission in admissions:
                admission.__exit__(None, None, None)
            for future in futures:
                try:
                    future.set_exception(exc)
                except InvalidStateError:
                    if not future.cancelled():
                        raise
