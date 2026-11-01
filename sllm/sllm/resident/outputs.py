"""Request-scoped engine output accumulation for resident scheduling."""
from dataclasses import dataclass, field


@dataclass
class RequestOutput:
    request_id: str
    token_ids: list = field(default_factory=list)
    num_cached_tokens: int = 0
    first_output_seconds: float | None = None
    submitted_at: float | None = None
    finished_at: float | None = None
    finished: bool = False

    def update(self, output, elapsed_seconds):
        if self.finished:
            raise RuntimeError('output received after request completion')
        if output.new_token_ids and self.first_output_seconds is None:
            self.first_output_seconds = elapsed_seconds
        self.token_ids.extend(output.new_token_ids)
        self.num_cached_tokens = max(self.num_cached_tokens,
                                     output.num_cached_tokens)
        self.finished = bool(getattr(output, 'finished', False))

    def consume(self, batches, elapsed_seconds):
        for batch in batches.values():
            for output in batch.outputs:
                if output.request_id != self.request_id:
                    continue
                self.update(output, elapsed_seconds)


class RequestSessions:
    """Rank-local request state; the serialized worker loop owns mutations."""

    def __init__(self):
        self._requests = {}
        self._cancelled = set()

    def submit(self, request_id, submitted_at):
        self.validate_submission(request_id)
        state = RequestOutput(request_id)
        state.submitted_at = submitted_at
        self._requests[request_id] = (submitted_at, state)
        return state

    def consume(self, batches, now):
        token_updates = {}
        for batch in batches.values():
            for output in batch.outputs:
                submitted_at, state = self._requests[output.request_id]
                state.update(output, now - submitted_at)
                if output.new_token_ids:
                    token_updates.setdefault(output.request_id, []).extend(output.new_token_ids)
                if state.finished:
                    state.finished_at = now
            # Some scheduler paths report completion without an output row.
            for request_id in getattr(batch, 'finished_requests', None) or ():
                if request_id in self._cancelled:
                    self._cancelled.remove(request_id)
                    continue
                state = self._requests[request_id][1]
                state.finished = True
                state.finished_at = now
        return token_updates

    def take_finished(self, request_id):
        _, state = self._requests[request_id]
        if not state.finished:
            raise RuntimeError('request is still running')
        del self._requests[request_id]
        return state

    def discard(self, request_id):
        """Roll back a submission rejected by the engine before scheduling."""
        del self._requests[request_id]

    def cancel(self, request_id, *, expect_completion=True):
        _, state = self._requests.pop(request_id)
        if not state.finished and expect_completion:
            # Scheduler completion is delivered on a later iteration. Keep
            # only an ID until that notification, never the request/KV state.
            self._cancelled.add(request_id)
        return state

    def validate_cancellation(self, request_id):
        if request_id not in self._requests:
            raise ValueError('unknown request ID')

    def validate_submission(self, request_id):
        if request_id in self._requests or request_id in self._cancelled:
            raise ValueError('duplicate request ID')

    def validate_collection(self, request_id):
        if request_id not in self._requests:
            raise ValueError('unknown request ID')
        if not self._requests[request_id][1].finished:
            raise ValueError('request is still running')

    @property
    def first_output_ids(self):
        return sorted(request_id for request_id, (_, state) in self._requests.items()
                      if state.first_output_seconds is not None)

    @property
    def finished_ids(self):
        return sorted(request_id for request_id, (_, state) in self._requests.items()
                      if state.finished)

    @property
    def retained(self):
        """Includes completed results that their caller has not collected yet."""
        return bool(self._requests)

    @property
    def pending(self):
        return any(not state.finished for _, state in self._requests.values())
