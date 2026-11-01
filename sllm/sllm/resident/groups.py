"""Pre-create vLLM communication groups for a resident GPU process pool.

Every pool rank calls prepare with the same manifest before serving requests.
Activation is process-local and collective-free: the caller dispatches a model
operation to every rank in its topology. This module owns group lifetimes, not
models. It deliberately does not own the pool's WORLD group.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import threading


# vLLM's active groups are process globals, including across registry objects.
_ACTIVE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Topology:
    ranks: tuple[int, ...]
    tp: int
    pp: int = 1

    def __post_init__(self):
        if type(self.tp) is not int or type(self.pp) is not int:
            raise ValueError('TP and PP must be integers')
        if self.tp < 1 or self.pp < 1 or len(self.ranks) != self.tp * self.pp:
            raise ValueError('Topology must contain exactly TP * PP ranks')
        if any(type(rank) is not int or rank < 0 for rank in self.ranks):
            raise ValueError('Ranks must be nonnegative integers')
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError('Topology ranks must be unique')
        if not isinstance(self.ranks, tuple):
            raise ValueError('Topology ranks must be an immutable tuple')

    def partition(self, role, world_size):
        if any(rank >= world_size for rank in self.ranks):
            raise ValueError('Topology rank is outside the resident pool')
        if role in ('tp', 'ep'):
            groups = [self.ranks[i:i + self.tp]
                      for i in range(0, len(self.ranks), self.tp)]
        elif role == 'pp':
            groups = [self.ranks[i::self.tp] for i in range(self.tp)]
        elif role == 'dp':
            groups = [(rank,) for rank in self.ranks]
        else:
            raise ValueError(f'Unknown communication role: {role}')
        # GroupCoordinator requires every pool process to belong to a group.
        groups += [(rank,) for rank in range(world_size)
                   if rank not in self.ranks]
        return tuple(sorted(groups))


class CommunicationRegistry:
    def __init__(self, device):
        import torch.distributed as dist
        from vllm.distributed import parallel_state as state
        if not dist.is_initialized():
            raise RuntimeError('Initialize the resident WORLD before registry')
        from torch import bfloat16
        from vllm.model_executor.warmup.relay_runtime import warmup_relay_runtime
        warmup_relay_runtime(device, bfloat16)
        self.device = device
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.control_group = state.get_world_group().cpu_group
        self._groups = {}
        self._topologies = {}
        self._closed = False
        self._poisoned = False
        self._retained_previous = None

    def memory_profile_inventory(self):
        """Stable pool-wide identity of retained native communication groups."""
        return dict(world_size=self.world_size, groups=[
            dict(role=role, partitions=[list(ranks) for ranks in partition])
            for role, partition in sorted(self._groups)])

    @staticmethod
    def _canonical_partitions(partition, world_size):
        """One communicator per ordered membership, independent of neighbors.

        GroupCoordinator still needs a partition covering every WORLD rank.
        Fill the complement with singleton groups, so the same TP/PP members
        reuse their communicator across different model placements.
        """
        return tuple(sorted({tuple(sorted((members,) + tuple(
            (rank,) for rank in range(world_size) if rank not in members)))
            for members in partition}))

    def _check_open(self):
        if self._closed or self._poisoned:
            raise RuntimeError('Communication registry is closed or failed')

    def prepare(self, topologies):
        """Collectively prepare a manifest, in the same order on every rank.

        Existing entries are reused. A differing manifest is rejected on all
        ranks before any new_group operation, preventing ordering deadlocks.
        Preparation must be dispatched to every pool rank by the controller.
        """
        import torch.distributed as dist
        from vllm.distributed import parallel_state as state
        self._check_open()
        topologies = tuple(topologies)
        error = None
        try:
            for topology in topologies:
                for role in ('tp', 'pp', 'dp', 'ep'):
                    topology.partition(role, self.world_size)
        except (ValueError, AttributeError) as exc:
            error = str(exc)
        locked = _ACTIVE_LOCK.acquire(blocking=False)
        if not locked:
            error = 'Cannot prepare groups during model execution'
        manifests = [None] * self.world_size
        payload = (topologies, tuple(self._topologies), error)
        try:
            dist.all_gather_object(manifests, payload, group=self.control_group)
            if any(item != payload or item[2] is not None for item in manifests):
                raise ValueError(f'Resident group manifests disagree: {manifests}')
        except BaseException:
            if locked:
                _ACTIVE_LOCK.release()
            raise
        try:
            for topology in topologies:
                if topology in self._topologies:
                    continue
                groups = {}
                for role in ('tp', 'pp', 'dp', 'ep'):
                    if role == 'ep':
                        # Dense relay supports DP=1: EP and TP have identical
                        # membership. Reuse TP rather than constructing an
                        # EP communicator whose all-reduce has no pynccl.
                        groups[role] = groups['tp']
                        continue
                    partition = topology.partition(role, self.world_size)
                    local_members = next(members for members in partition if self.rank in members)
                    for canonical in self._canonical_partitions(partition, self.world_size):
                        key = (role, canonical)
                        if key not in self._groups:
                            group = state.GroupCoordinator(
                                group_ranks=[list(ranks) for ranks in canonical],
                                local_rank=self.device.index,
                                torch_distributed_backend='nccl',
                                use_device_communicator=True,
                                use_message_queue_broadcaster=False,
                                group_name=f'resident_{role}')
                            self._groups[key] = group
                            self._warm(group, role)
                    local_partition = self._canonical_partitions((local_members,), self.world_size)[0]
                    groups[role] = self._groups[(role, local_partition)]
                self._topologies[topology] = groups
        except BaseException:
            # Partial creation cannot be retried safely across the pool.
            self._poisoned = True
            raise
        finally:
            _ACTIVE_LOCK.release()

    def _warm(self, group, role):
        import torch
        if group.world_size == 1:
            return
        probe = torch.ones((1, 1), device=self.device, dtype=torch.bfloat16)
        group.all_reduce(probe)
        if role == 'tp':
            # A single BF16 element cannot use custom all-reduce (its byte
            # count must be divisible by 16). Exercise that path before
            # admission too; the scalar above still prepares the NCCL path.
            group.all_reduce(torch.ones((2, 256), device=self.device,
                                        dtype=torch.bfloat16))
        group.all_gather(probe)
        group.gather(probe)
        if role == 'pp':
            # Prepare direct PyNCCL users and the ProcessGroup tensor-dict
            # path used by model pipeline stages. They own different NCCL
            # communicators; warming one does not initialize the other.
            for source in range(group.world_size - 1):
                if group.rank_in_group == source:
                    group.send(probe, dst=source + 1)
                    group.send_tensor_dict({'hidden_states': probe,
                                            'residual': probe}, dst=source + 1)
                elif group.rank_in_group == source + 1:
                    group.recv(probe.size(), probe.dtype, src=source)
                    group.recv_tensor_dict(src=source)
            # Recovery gathers each stage's KV directly at stage zero. Those
            # non-adjacent NCCL peer connections must exist before model/KV
            # allocation, while their device buffers still fit comfortably.
            for source in range(1, group.world_size):
                if group.rank_in_group == source:
                    group.send_tensor_dict({'kv': probe}, dst=0)
                elif group.is_first_rank:
                    group.recv_tensor_dict(src=source)
        torch.cuda.synchronize(self.device)
        if role == 'tp':
            # Both logits collectives above completed during group preparation.
            # Share that readiness with Worker; model switches still perform
            # their model-specific warmup, without repeating this group probe.
            prepared = getattr(group, '_relay_warmup_keys', set())
            device = torch.device(self.device)
            prepared.update((device, torch.bfloat16, all_gather)
                            for all_gather in (False, True))
            group._relay_warmup_keys = prepared

    @contextmanager
    def activate(self, topology, *, retain_active=False):
        """Select prepared groups, optionally retaining them between commands.

        Resident commands retain the selected groups so background weight copies
        can continue after lazy initialization. A change of groups fences before
        replacing globals. Scoped callers still fence and restore on exit.
        Native loader lifetimes must be drained by model teardown before reuse.
        """
        import torch
        from vllm.distributed import parallel_state as state
        self._check_open()
        groups = self._topologies[topology]
        if self.rank not in topology.ranks:
            raise ValueError('Idle rank cannot execute this model topology')
        if not _ACTIVE_LOCK.acquire(blocking=False):
            raise RuntimeError('Another model operation is already active')
        names = ('_TP', '_PP', '_DP', '_EP')
        previous = tuple(getattr(state, name) for name in names)
        selected = tuple(groups[role] for role in ('tp', 'pp', 'dp', 'ep'))
        changed = any(old is not new for old, new in zip(previous, selected))
        failed = True
        try:
            if changed and any(group is not None for group in previous):
                try:
                    torch.cuda.synchronize(self.device)
                except BaseException:
                    self._poisoned = True
                    raise
            if retain_active and self._retained_previous is None:
                self._retained_previous = previous
            for name, group in zip(names, selected):
                setattr(state, name, group)
            yield groups
            failed = False
        finally:
            try:
                if failed or not retain_active:
                    try:
                        torch.cuda.synchronize(self.device)
                    except BaseException:
                        self._poisoned = True
                        raise
                    finally:
                        for name, group in zip(names, previous):
                            setattr(state, name, group)
            finally:
                _ACTIVE_LOCK.release()

    def close(self):
        """Destroy cached groups only when shutting down the resident pool."""
        import torch
        if self._closed:
            return
        if not _ACTIVE_LOCK.acquire(blocking=False):
            raise RuntimeError('Cannot close groups during model execution')
        try:
            torch.cuda.synchronize(self.device)
            for group in reversed(tuple(self._groups.values())):
                group.destroy()
            if self._retained_previous is not None:
                from vllm.distributed import parallel_state as state
                for name, group in zip(('_TP', '_PP', '_DP', '_EP'),
                                       self._retained_previous):
                    setattr(state, name, group)
                self._retained_previous = None
            self._topologies.clear()
            self._groups.clear()
            self._closed = True
        finally:
            _ACTIVE_LOCK.release()
