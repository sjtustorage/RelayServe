"""Serialized model-slot lifecycle for a resident GPU rank.

The slot deliberately owns admission and generations, while an adapter owns
vLLM engine construction/destruction. A model adapter must never mutate a live
vLLM config in place: build a new engine against the already prepared
communication topology, drain the old one, then commit the new generation.
"""
from dataclasses import dataclass
from enum import Enum
import asyncio
from typing import Any, Optional, Protocol


class SlotState(str, Enum):
    EMPTY = 'empty'
    LOADING = 'loading'
    READY = 'ready'
    DRAINING = 'draining'
    FAILED = 'failed'


class ModelAdapter(Protocol):
    async def load(self, model: str, topology: Any) -> Any: ...
    async def unload(self, engine: Any) -> None: ...


@dataclass(frozen=True)
class Lease:
    model: str
    generation: int
    engine: Any
    ticket: int


class ResidentModelSlot:
    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter
        self.state = SlotState.EMPTY
        self.model: Optional[str] = None
        self.engine: Any = None
        self.topology: Any = None
        self.generation = 0
        self.in_flight = 0
        self.last_error: Optional[str] = None
        self._next_ticket = 0
        self._leases: dict[int, Lease] = {}
        self._poisoned = False
        self._lock = asyncio.Lock()
        self._idle = asyncio.Condition(self._lock)

    async def load(self, model: str, topology: Any) -> int:
        if not model:
            raise ValueError('model must be non-empty')
        async with self._lock:
            if self._poisoned:
                raise RuntimeError('model teardown failed; restart resident rank')
            if self.state in (SlotState.LOADING, SlotState.DRAINING):
                raise RuntimeError(f'slot is {self.state.value}')
            if self.state is SlotState.READY and self.model == model \
                    and self.topology == topology:
                return self.generation
            old = self.engine
            previous_state = self.state
            self.state = SlotState.DRAINING if old is not None else SlotState.LOADING
            try:
                while self.in_flight:
                    await self._idle.wait()
            except BaseException:
                self.state = previous_state
                raise
            self.engine = None
            self.model = None
            self.topology = None
            self.state = SlotState.LOADING
        if old is not None:
            try:
                await self.adapter.unload(old)
            except BaseException as exc:
                async with self._lock:
                    self._poisoned = True
                    self.state = SlotState.FAILED
                    self.last_error = f'{type(exc).__name__}: {exc}'
                raise
        try:
            engine = await self.adapter.load(model, topology)
        except BaseException as exc:
            async with self._lock:
                self.state = SlotState.FAILED
                self.last_error = f'{type(exc).__name__}: {exc}'
            raise
        async with self._lock:
            self.generation += 1
            self.engine = engine
            self.model = model
            self.topology = topology
            self.last_error = None
            self.state = SlotState.READY
            return self.generation

    async def acquire(self, model: Optional[str] = None) -> Lease:
        async with self._lock:
            if self.state is not SlotState.READY or self.engine is None:
                raise RuntimeError(f'slot is not ready: {self.state.value}')
            if model is not None and model != self.model:
                raise RuntimeError(f'active model is {self.model!r}')
            self.in_flight += 1
            self._next_ticket += 1
            lease = Lease(self.model, self.generation, self.engine,
                          self._next_ticket)
            self._leases[lease.ticket] = lease
            return lease

    async def release(self, lease: Lease) -> None:
        async with self._lock:
            if (lease.generation != self.generation or
                    self._leases.get(lease.ticket) is not lease):
                raise RuntimeError('invalid or stale model lease')
            del self._leases[lease.ticket]
            self.in_flight -= 1
            if not self.in_flight:
                self._idle.notify_all()

    async def unload(self) -> None:
        async with self._lock:
            if self._poisoned:
                raise RuntimeError('model teardown failed; restart resident rank')
            if self.state in (SlotState.LOADING, SlotState.DRAINING):
                raise RuntimeError(f'slot is {self.state.value}')
            old = self.engine
            if old is None:
                self.state = SlotState.EMPTY
                return
            self.state = SlotState.DRAINING
            try:
                while self.in_flight:
                    await self._idle.wait()
            except BaseException:
                self.state = SlotState.READY
                raise
            self.engine = None
            self.model = None
            self.topology = None
        try:
            await self.adapter.unload(old)
        except BaseException as exc:
            async with self._lock:
                self._poisoned = True
                self.state = SlotState.FAILED
                self.last_error = f'{type(exc).__name__}: {exc}'
            raise
        async with self._lock:
            self.state = SlotState.EMPTY
            self.last_error = None

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            return dict(state=self.state.value, model=self.model,
                        generation=self.generation, in_flight=self.in_flight,
                        last_error=self.last_error, poisoned=self._poisoned)
