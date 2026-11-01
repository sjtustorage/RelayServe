"""Model-owned bindings for legacy process-global engine state.

Resident rank commands are serialized. CommunicationRegistry is responsible for
fencing device work when changing topology; this binding neither fences nor
stops native asynchronous loaders. Those loaders retain their own managers.
The exported SHM info objects keep their identities because engine modules can
import them directly. Only their attribute dictionaries are exchanged.
"""
from contextlib import contextmanager
import threading


_BINDING_LOCK = threading.RLock()
_ACTIVE_STATE = None


class ModelRuntimeState:
    def __init__(self):
        self._saved = None
        self._depth = 0

    @staticmethod
    def _modules():
        import shm_manager.torch as shm
        from vllm.distributed.kv_transfer import kv_transfer_state as kv
        from vllm.model_executor.layers import rotary_embedding as rotary
        from vllm.model_executor import model_loader
        return shm, kv, rotary, model_loader

    @staticmethod
    def _capture(shm, kv, rotary, model_loader):
        return dict(infos=[dict(info.__dict__) for info in (
            shm.model_info, shm.kv_cache_info, shm.hidden_states_info)],
            total_layers=shm.total_layers, connector=kv._KV_CONNECTOR_AGENT,
            rope=dict(rotary._ROPE_DICT), loader=model_loader._CURRENT_MODEL_LOADER)

    @staticmethod
    def _install(saved, shm, kv, rotary, model_loader):
        for info, attributes in zip((shm.model_info, shm.kv_cache_info,
                                     shm.hidden_states_info), saved['infos']):
            info.__dict__.clear()
            info.__dict__.update(attributes)
        shm.total_layers = saved['total_layers']
        kv._KV_CONNECTOR_AGENT = saved['connector']
        rotary._ROPE_DICT.clear()
        rotary._ROPE_DICT.update(saved['rope'])
        model_loader._CURRENT_MODEL_LOADER = saved['loader']

    @contextmanager
    def activate(self):
        global _ACTIVE_STATE
        if not _BINDING_LOCK.acquire(blocking=False):
            raise RuntimeError('Another thread is executing model runtime state')
        try:
            if _ACTIVE_STATE is not None and _ACTIVE_STATE is not self:
                raise RuntimeError('Cannot nest different model runtime states')
            # Nested teardown must operate on the already selected state.
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            modules = self._modules()
            previous = self._capture(*modules)
            if self._saved is None:
                shm, _, _, _ = modules
                self._saved = dict(infos=[dict(cls().__dict__) for cls in (
                    shm.ModelInfo, shm.KVCacheInfo, shm.HiddenStatesInfo)],
                    total_layers=-1, connector=None, rope={}, loader=None)
            self._install(self._saved, *modules)
            self._depth = 1
            _ACTIVE_STATE = self
            try:
                yield
            finally:
                try:
                    self._saved = self._capture(*modules)
                finally:
                    self._depth = 0
                    _ACTIVE_STATE = None
                    self._install(previous, *modules)
        finally:
            _BINDING_LOCK.release()
