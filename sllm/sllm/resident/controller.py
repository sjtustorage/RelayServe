"""NUMA-aware model admission and routing over existing resident pool clients.

Each client represents one node's fresh, already-started GPU pool. A controller
owns its pools exclusively; external commands must not load models into them.
Commands drain earlier operations on the same ranks before switching/release.
A pool normally owns one physical node. A global-WORLD pool can declare
physical_nodes_by_rank: TP stages stay node-local, while PP may span nodes.
"""
from dataclasses import dataclass, replace
from contextlib import contextmanager, ExitStack
from itertools import combinations
import threading
import time
import math
from sllm.resident.client import ResidentCommandRejected


class ResidentCapacityUnavailable(RuntimeError):
    """No currently free placement; no device command has executed."""


@dataclass(frozen=True)
class Instance:
    name: str
    model: str
    node: str
    ranks: tuple[int, ...]
    generation: int
    ready: bool
    initialization_phase: str = 'first'
    tp_size: int = 0
    pp_size: int = 1
    slot: str = 'default'
    vllm_initialization_seconds: float | None = None

    @property
    def topology(self):
        topology = dict(ranks=list(self.ranks), tp=self.tp_size or len(self.ranks))
        if self.pp_size != 1:
            topology['pp'] = self.pp_size
        return topology


class ResidentController:
    def __init__(self, pools, numa_by_rank=None, *, idle_timeout=10.0, clock=time.monotonic,
                 physical_nodes_by_rank=None):
        if isinstance(idle_timeout, bool) or not math.isfinite(idle_timeout) or idle_timeout < 0:
            raise ValueError('Idle timeout must be finite and nonnegative')
        self.idle_timeout = idle_timeout
        self._clock = clock
        if numa_by_rank is None:
            numa_by_rank = {}
            for node, pool in pools.items():
                reply = pool.command('status')
                ranks = reply.ranks
                if len(ranks) != pool.gpu_count:
                    raise ValueError('Pool inventory must cover every rank')
                domains = []
                identities = set()
                for rank, item in enumerate(ranks):
                    result = item['result']
                    info = result.get('gpu_inventory') or {}
                    identity = info.get('uuid')
                    if result.get('rank') != rank or not identity or identity in identities:
                        raise ValueError('Invalid or duplicated resident GPU inventory')
                    model_slot = result.get('model_slot') or {}
                    model_slots = result.get('model_slots') or {}
                    if (model_slot.get('model') or any(
                            isinstance(slot, dict) and slot.get('model')
                            for slot in model_slots.values())):
                        raise ValueError('Controller requires a fresh pool with no loaded models')
                    identities.add(identity)
                    domains.append(info.get('numa_node'))
                numa_by_rank[node] = domains
        if not pools or set(pools) != set(numa_by_rank):
            raise ValueError('Each node requires a pool and a NUMA inventory')
        self.pools = dict(pools)
        self.numa = {}
        for node, pool in pools.items():
            domains = tuple(numa_by_rank[node])
            if (len(domains) != pool.gpu_count or
                    any(type(domain) is not int or domain < 0 for domain in domains)):
                raise ValueError('NUMA inventory must cover every visible pool rank')
            self.numa[node] = domains
        if physical_nodes_by_rank is None:
            physical_nodes_by_rank = {node: [node] * pool.gpu_count for node, pool in pools.items()}
        if set(physical_nodes_by_rank) != set(pools):
            raise ValueError('Physical node inventory must cover each pool')
        self.physical_nodes = {}
        for node, pool in pools.items():
            identities = tuple(physical_nodes_by_rank[node])
            if len(identities) != pool.gpu_count or any(not isinstance(value, str) or not value for value in identities):
                raise ValueError('Physical node inventory must cover every rank')
            self.physical_nodes[node] = identities
        self._instances = {}
        self._pending_creates = set()
        self._unloaded_reservations = set()
        self._bounded_prewarm = set()
        self._prewarming = False
        self._last_activity = {}
        self._operations = {}
        self._admitted_requests = {}
        self._submitted_requests = {}
        self._instance_admissions = {}
        self._expiring = set()
        self._tearing_down = set()
        self._owners = {}
        self._quarantined = set()
        self._quarantine_causes = {}
        self._restarting = set()
        self._generation = 0
        self._initialized_ranks = set()
        self._prepared = {node: set() for node in pools}
        self._lock = threading.RLock()
        self._capacity = threading.Condition(self._lock)
        self._node_locks = {node: threading.RLock() for node in pools}
        self._rank_locks = {node: tuple(threading.RLock() for _ in range(pool.gpu_count))
                            for node, pool in pools.items()}
        # Compute admission is separate from GPU ownership/control commands.
        # Contending cold starts proceed on GPU rather than queue for AMX.
        self._amx_leases = {node: threading.BoundedSemaphore(1)
                            for nodes in self.physical_nodes.values() for node in nodes}

    def amx_lease(self, instance, *, physical_node=None):
        """Return the one shared compute slot for an instance's physical node.

        A cross-node pipeline must request each participating node explicitly;
        transport/WORLD names never identify CPU compute ownership.
        """
        with self._lock:
            self._current(instance, allow_pending=True)
            nodes = {self.physical_nodes[instance.node][rank] for rank in instance.ranks}
            if physical_node is None:
                if len(nodes) != 1:
                    raise ValueError('Cross-node AMX admission requires a physical node')
                physical_node = next(iter(nodes))
            if physical_node not in nodes:
                raise ValueError('AMX node is not part of the instance')
            return self._amx_leases[physical_node]

    def _execution_locks(self, instance):
        if getattr(self.pools[instance.node], 'concurrent_topologies', False):
            return [self._rank_locks[instance.node][rank]
                    for rank in sorted(instance.ranks)]
        return [self._node_locks[instance.node]]

    def _slot_name(self, instance):
        return instance.slot or 'default'

    def _drop_reservations(self, instance):
        """Reassign shared rank ownership after removing an instance.

        The caller holds metadata and has already removed the instance from
        _instances. This applies equally to unload and admission rollback.
        """
        for rank in instance.ranks:
            key = instance.node, rank
            if self._owners.get(key) != instance.name:
                continue
            replacement = next((other.name for other in self._instances.values()
                if other.node == instance.node and rank in other.ranks), None)
            if replacement is None:
                del self._owners[key]
            else:
                self._owners[key] = replacement

    @contextmanager
    def _execution(self, instance, *, blocking=True):
        # Globally sorted acquisition prevents cycles between overlapping
        # topologies. Reentrant locks preserve nested relay lifecycle pinning.
        acquired = []
        try:
            for lock in self._execution_locks(instance):
                if not lock.acquire(blocking=blocking):
                    yield False
                    return
                acquired.append(lock)
            yield True
        finally:
            for lock in reversed(acquired):
                lock.release()

    @contextmanager
    def instance_admission(self, instance, *, refresh_activity=True):
        """Pin a specific generation atomically against expiry and teardown."""
        with self._lock:
            self._current(instance)
            if instance.name in self._tearing_down:
                raise ValueError('Instance model teardown is in progress')
            self._instance_admissions[instance.name] = self._instance_admissions.get(instance.name, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                remaining = self._instance_admissions[instance.name] - 1
                if remaining:
                    self._instance_admissions[instance.name] = remaining
                else:
                    del self._instance_admissions[instance.name]
                    if refresh_activity and instance.name in self._instances:
                        self._last_activity[instance.name] = self._clock()

    @contextmanager
    def hydra_lenders(self, native):
        """Pin loaded native slots while selecting/loading temporary staging.

        Enumeration holds only metadata, never another route's mutex or GPU
        execution locks. Unused candidates do not extend the idle deadline.
        The staging slot owns its selected ranks after admission completes.
        """
        with ExitStack() as pins:
            with self._lock:
                self._current(native)
                lenders = tuple(instance for instance in self._instances.values()
                    if instance.node == native.node and instance != native
                    and instance.ready and instance.pp_size == 1
                    and instance.name not in self._pending_creates
                    and instance.name not in self._expiring
                    and instance.name not in self._tearing_down)
                for lender in lenders:
                    pins.enter_context(self.instance_admission(lender, refresh_activity=False))
            yield lenders

    @contextmanager
    def request_admission(self, name):
        """Count accepted route requests before they wait for an execution owner.

        The route can be cold on admission and acquire a model later. Counting
        by route name protects that model through the entire queued lifecycle.
        Explicit release remains a separate control operation.
        """
        with self._lock:
            self._admitted_requests[name] = self._admitted_requests.get(name, 0) + 1
        try:
            with self._capacity:
                # Expiry may already own teardown when an arrival is counted.
                # Wait without holding any route/GPU locks before recreating
                # this name; the reaper completes or quarantines the old slot.
                while name in self._expiring:
                    self._capacity.wait()
            yield
        finally:
            with self._lock:
                remaining = self._admitted_requests[name] - 1
                if remaining:
                    self._admitted_requests[name] = remaining
                else:
                    del self._admitted_requests[name]
                    if name in self._instances:
                        self._last_activity[name] = self._clock()

    @contextmanager
    def _instance_operation(self, instance=None, *, name=None, generation=None,
                            allow_reserved=False, prepare=False):
        # Never wait for a node while holding the metadata lock. Revalidate
        # after taking the node lock: a queued switch may invalidate a handle.
        with self._lock:
            if instance is None:
                instance = self.lookup(name, generation)
            self._current(instance)
            if instance.name in self._unloaded_reservations and not allow_reserved:
                raise ValueError('Instance has reserved capacity but no loaded model')
            self._operations[instance.name] = self._operations.get(instance.name, 0) + 1
        try:
            if prepare:
                self._prepare_topology(instance)
            with self._execution(instance):
                with self._lock:
                    self._current(instance)
                yield instance
        finally:
            with self._lock:
                count = self._operations[instance.name] - 1
                if count:
                    self._operations[instance.name] = count
                else:
                    del self._operations[instance.name]
                if instance.name in self._instances:
                    self._last_activity[instance.name] = self._clock()

    def _quarantine(self, node, cause=None):
        with self._lock:
            self._quarantined.add(node)
            if cause is not None:
                self._quarantine_causes.setdefault(node, cause)
            self._capacity.notify_all()

    def restart_node(self, node, pool_factory):
        """Replace a failed pool after requests have unwound, keeping other nodes.

        Reservations are retained until the old pool has exited and the new
        pool has validated its inventory and prepared the old group manifest.
        Model generations are never reused. Normal idle/switch operations do
        not enter this path and retain their workers.
        """
        with self._lock:
            if node not in self._quarantined or node in self._restarting:
                raise ValueError('Restart requires a quarantined, non-restarting node')
            names = {name for name, item in self._instances.items() if item.node == node}
            if (self._prewarming or any(self._operations.get(name)
                    or self._instance_admissions.get(name)
                    or self._admitted_requests.get(name)
                    or name in self._pending_creates or name in self._expiring
                    or name in self._tearing_down for name in names)):
                raise ValueError('Drain node operations and admissions before restart')
            self._restarting.add(node)
            old = self.pools[node]
            manifest = set(self._prepared[node])
        replacement = None
        try:
            old.close()
            replacement = pool_factory()
            if replacement is old:
                raise ValueError('Restart requires a fresh pool')
            inventory = ResidentController({node: replacement})
            if inventory.numa[node] != self.numa[node]:
                raise ValueError('Replacement pool rank/NUMA inventory differs')
            topologies = []
            for key in sorted(manifest, key=repr):
                if key and isinstance(key[0], tuple):
                    ranks, tp, pp = key
                else:
                    ranks, tp, pp = key, len(key), 1
                topologies.append(dict(ranks=list(ranks), tp=tp, pp=pp))
            if topologies:
                replacement.command('prepare', topologies=topologies)
            with self._lock:
                self.pools[node] = replacement
                for name in names:
                    self._instances.pop(name)
                    self._last_activity.pop(name, None)
                    self._submitted_requests.pop(name, None)
                    self._unloaded_reservations.discard(name)
                self._owners = {key: owner for key, owner in self._owners.items() if key[0] != node}
                self._initialized_ranks = {key for key in self._initialized_ranks if key[0] != node}
                self._quarantined.remove(node)
                self._quarantine_causes.pop(node, None)
                self._capacity.notify_all()
            return dict(node=node, invalidated_instances=sorted(names))
        except BaseException:
            if replacement is not None and replacement is not old:
                replacement.close()
            raise
        finally:
            with self._lock:
                self._restarting.discard(node)

    def _balanced(self, domains, ranks):
        counts = [sum(domains[rank] == domain for rank in ranks)
                  for domain in sorted(set(domains))]
        return max(counts) - min(counts) <= 1

    def prewarm(self, tp_sizes, *, pp_sizes=(1,), max_memberships_per_shape="auto"):
        """Prepare allocator-eligible memberships before model requests.

        Auto mode bounds startup on pools larger than four GPUs. Explicit
        None requests exhaustive prewarming for diagnostics/small pools.
        A bounded manifest caches representative placements; missing topologies
        are prepared at admission under a pool-wide execution fence.

        CommunicationRegistry deduplicates role groups across memberships.
        This method is idempotent; failures quarantine the affected node.
        """
        sizes = tuple(tp_sizes)
        if not sizes or any(type(tp) is not int or tp < 1 for tp in sizes):
            raise ValueError('Prewarm TP sizes must be positive integers')
        pipeline_sizes = tuple(pp_sizes)
        if not pipeline_sizes or any(type(pp) is not int or pp < 1
                                     for pp in pipeline_sizes):
            raise ValueError('Prewarm PP sizes must be positive integers')
        if max_memberships_per_shape == "auto":
            max_memberships_per_shape = (1 if any(len(domains) > 4
                for domains in self.numa.values()) else None)
        if (max_memberships_per_shape is not None and
                (type(max_memberships_per_shape) is not int or
                 max_memberships_per_shape < 1)):
            raise ValueError('max_memberships_per_shape must be a positive integer')
        with self._lock:
            if self._instances or self._prewarming:
                raise ValueError('Prewarm must run before model admission and cannot overlap another prewarm')
            plans = {}
            for node, domains in self.numa.items():
                if node in self._quarantined:
                    raise RuntimeError('Resident node requires pool restart') from self._quarantine_causes.get(node)
                plans[node] = []
                for tp in sorted(set(sizes)):
                    for pp in sorted(set(pipeline_sizes)):
                        shape_count = 0
                        memberships = (combinations(range(len(domains)), tp * pp)
                            if max_memberships_per_shape is None else
                            self._placement_memberships(node, range(len(domains)), tp * pp))
                        for membership in memberships:
                            if (len(set(self.physical_nodes[node])) == 1
                                    and not self._balanced(domains, membership)):
                                continue
                            ranks = self._stage_order(domains, membership, tp, self.physical_nodes[node])
                            if ranks is None:
                                continue
                            orders = {ranks}
                            if pp > 1 and getattr(self.pools[node], 'concurrent_topologies', False):
                                native_memberships = (combinations(membership, tp)
                                    if max_memberships_per_shape is None else
                                    self._placement_memberships(node, membership, tp))
                                for native_ranks in native_memberships:
                                    order = self._hydra_stage_order(domains, membership, tp,
                                        native_ranks, self.physical_nodes[node])
                                    if order is not None:
                                        orders.add(order)
                            for order in sorted(orders):
                                key = order if pp == 1 else (order, tp, pp)
                                if key not in self._prepared[node]:
                                    topology = dict(ranks=list(order), tp=tp)
                                    if pp != 1:
                                        topology['pp'] = pp
                                    plans[node].append((key, topology))
                            shape_count += 1
                            if (max_memberships_per_shape is not None and
                                    shape_count >= max_memberships_per_shape):
                                break
            self._prewarming = True
        # Never hold metadata while waiting for a worker or node lock. Admission
        # is fenced until every preparation command has completed.
        try:
            results = {}
            for node, memberships in plans.items():
                start = time.perf_counter()
                with self._node_locks[node]:
                    if memberships:
                        self._command(node, 'prepare', topologies=[topology
                            for _, topology in memberships])
                        with self._lock:
                            self._prepared[node].update(key for key, _ in memberships)
                            if max_memberships_per_shape is not None:
                                self._bounded_prewarm.add(node)
                    with self._lock:
                        prepared_count = len(self._prepared[node])
                results[node] = dict(new_topologies=len(memberships),
                                     prepare_seconds=time.perf_counter() - start,
                                     prepared_topologies=prepared_count)
            return results
        finally:
            with self._lock:
                self._prewarming = False

    def _placement_memberships(self, node, ranks, size, *, distinguish_owned=False,
                               fixed_ranks=()):
        """Yield one exact representative per placement-equivalent rank count.

        Constraints and load scores depend on physical node / NUMA counts,
        plus borrowed-rank counts for Hydra. Within each class, the smallest
        ranks give the lexicographically first stage order. No feasible class
        is truncated, unlike a candidate-count limit.
        """
        buckets = {}
        for rank in sorted(ranks):
            key = (self.physical_nodes[node][rank], self.numa[node][rank],
                   distinguish_owned and (node, rank) in self._owners)
            buckets.setdefault(key, []).append(rank)
        groups = tuple(buckets.values())
        # All single-host placements require global NUMA balance. Enforce it
        # during generation, including the native prefix of a Hydra placement,
        # rather than constructing an exponential number of invalid leaves.
        balanced = len(set(self.physical_nodes[node])) == 1
        domains = tuple(sorted(set(self.numa[node])))
        low, extra = divmod(size + len(fixed_ranks), len(domains))
        high = low + bool(extra)
        selected = {domain: sum(self.numa[node][rank] == domain for rank in fixed_ranks)
                    for domain in domains}
        group_domains = tuple(self.numa[node][group[0]] for group in groups)
        available = [dict.fromkeys(domains, 0) for _ in range(len(groups) + 1)]
        for index in range(len(groups) - 1, -1, -1):
            available[index] = available[index + 1].copy()
            available[index][group_domains[index]] += len(groups[index])
        suffix = [0] * (len(groups) + 1)
        for index in range(len(groups) - 1, -1, -1):
            suffix[index] = suffix[index + 1] + len(groups[index])

        def visit(index, remaining, chosen):
            if balanced:
                needed = 0
                capacity = 0
                for domain in domains:
                    count = selected[domain]
                    if count > high or count + available[index][domain] < low:
                        return
                    needed += max(0, low - count)
                    capacity += min(available[index][domain], high - count)
                if remaining < needed or remaining > capacity:
                    return
            if remaining == 0:
                yield tuple(sorted(chosen))
                return
            if index == len(groups) or remaining > suffix[index]:
                return
            group = groups[index]
            domain = group_domains[index]
            upper = min(len(group), remaining)
            if balanced:
                upper = min(upper, high - selected[domain])
            for count in range(upper,
                               max(0, remaining - suffix[index + 1]) - 1, -1):
                selected[domain] += count
                yield from visit(index + 1, remaining - count,
                                 chosen + tuple(group[:count]))
                selected[domain] -= count
        if size >= 0:
            yield from visit(0, size, ())

    def _stage_order(self, domains, ranks, tp, physical_nodes=None):
        """Each contiguous PP stage must independently balance its TP ranks."""
        physical_nodes = physical_nodes or ('local',) * len(domains)
        local_domains = {}
        for rank, physical in enumerate(physical_nodes):
            local_domains.setdefault(physical, set()).add(domains[rank])
        def order(remaining):
            if not remaining:
                return ()
            stages = []
            for physical, values in local_domains.items():
                groups = [tuple(rank for rank in remaining
                                if physical_nodes[rank] == physical and domains[rank] == domain)
                          for domain in sorted(values)]
                low, extra = divmod(tp, len(groups))
                total = sum(map(len, groups))
                if not total:
                    continue
                count, remainder = divmod(total, tp)
                # k balanced stages exist exactly when every domain supplies
                # k*low .. k*(low+1) ranks (k*low when extra == 0).
                if remainder or any(not count * low <= len(group) <=
                        count * (low + bool(extra)) for group in groups):
                    return None
                mandatory, optional = [], []
                for index, group in enumerate(groups):
                    choices = [take for take in (low, low + 1)
                               if take <= len(group) and
                               (count - 1) * low <= len(group) - take <=
                               (count - 1) * (low + bool(extra))]
                    if low not in choices:
                        mandatory.append(index)
                    elif low + 1 in choices:
                        optional.append(index)
                needed = extra - len(mandatory)
                if not 0 <= needed <= len(optional):
                    return None
                # The next rank from a domain is the only lexicographic
                # difference between its low and low+1 allocations.
                optional.sort(key=lambda index: groups[index][low])
                larger = set(mandatory + optional[:needed])
                stages.append(tuple(sorted(rank for index, group in enumerate(groups)
                    for rank in group[:low + (index in larger)])))
            if not stages:
                return None
            stage = min(stages)
            members = set(stage)
            return stage + order(tuple(rank for rank in remaining if rank not in members))

        return order(tuple(ranks))

    def _hydra_stage_order(self, domains, ranks, tp, prefix, physical_nodes=None):
        """Order an expanded PP membership with the native TP as stage zero."""
        prefix = tuple(prefix)
        if len(prefix) != tp or not set(prefix) <= set(ranks):
            return None
        if physical_nodes is not None:
            stage_node = physical_nodes[prefix[0]]
            if any(physical_nodes[rank] != stage_node for rank in prefix):
                return None
            local_domains = {domains[rank] for rank in range(len(domains))
                             if physical_nodes[rank] == stage_node}
            counts = [sum(domains[rank] == domain for rank in prefix)
                      for domain in local_domains]
            if max(counts) - min(counts) > 1:
                return None
        elif not self._balanced(domains, prefix):
            return None
        tail = tuple(rank for rank in ranks if rank not in prefix)
        rest = self._stage_order(domains, tail, tp, physical_nodes)
        return prefix + rest if rest is not None else None

    def _select(self, tp, pp=1):
        if type(tp) is not int or tp < 1:
            raise ValueError('TP must be a positive integer')
        if type(pp) is not int or pp < 1:
            raise ValueError('PP must be a positive integer')
        best = None
        for node, domains in self.numa.items():
            if node in self._quarantined:
                continue
            groups = sorted(set(domains))
            active = {domain: sum((node, rank) in self._owners
                                  for rank, value in enumerate(domains) if value == domain)
                      for domain in groups}
            free = [rank for rank in range(len(domains)) if (node, rank) not in self._owners]
            for membership in self._placement_memberships(node, free, tp * pp):
                ranks = self._stage_order(domains, membership, tp, self.physical_nodes[node])
                if ranks is None:
                    continue
                counts = {domain: sum(domains[rank] == domain for rank in ranks) for domain in groups}
                if len(set(self.physical_nodes[node])) == 1 and not self._balanced(domains, ranks):
                    continue
                after = tuple(sorted((active[domain] + counts[domain] for domain in groups), reverse=True))
                candidate = ((sum(active.values()), after, node, ranks), node, ranks)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise ResidentCapacityUnavailable('No free NUMA-balanced TP placement')
        _, node, ranks = best
        return node, ranks

    def _command(self, node, op, **kwargs):
        try:
            return self.pools[node].command(op, **kwargs)
        except ResidentCommandRejected:
            raise
        except BaseException as exc:
            # A failed collective may leave model/native ownership uncertain.
            # Keep every reservation on this node until its pool is restarted.
            self._quarantine(node, exc)
            raise

    def _results(self, instance, reply, *, unloaded=False):
        try:
            values = [reply.ranks[rank]['result'] for rank in instance.ranks]
            if any(value.get('generation') != instance.generation or
                   value.get('model') != (None if unloaded else instance.model) for value in values):
                raise RuntimeError('Resident reply does not match the routed instance')
            return values
        except (IndexError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
            self._quarantine(instance.node, exc)
            raise

    def _current(self, instance, *, allow_pending=False):
        if self._instances.get(instance.name) != instance:
            raise ValueError('Stale or unknown instance handle')
        if instance.node in self._quarantined:
            raise RuntimeError('Resident node requires pool restart') from self._quarantine_causes.get(instance.node)
        if instance.name in self._expiring:
            raise ValueError('Instance is being reclaimed')
        if instance.name in self._pending_creates and not allow_pending:
            raise ValueError('Instance creation is still in progress')
        return instance

    @staticmethod
    def _validate_hydra_memory_ratio(value):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('hydra_min_free_memory_ratio must be a finite number in [0, 1]')

    def _hydra_eligible_lenders(self, native, lenders, ratio):
        self._validate_hydra_memory_ratio(ratio)
        if not ratio or not lenders:
            return None
        reply = self._command(native.node, 'status', include_memory=True)
        eligible = set()
        for lender in lenders:
            for rank in lender.ranks:
                info = reply.ranks[rank]['result']['gpu_memory']
                free, total = info['free_bytes'], info['total_bytes']
                if total <= 0 or not 0 <= free <= total:
                    raise ValueError('Invalid GPU memory inventory')
                if free / total >= ratio:
                    eligible.add(rank)
        return eligible

    def _hydra_placement(self, staging_for, tp, pp, borrow_from, eligible_borrowed=None):
        """Select staging ranks under the metadata lock, without device work."""
        node = staging_for.node
        domains = self.numa[node]
        physical_nodes = self.physical_nodes[node]
        borrowed = set()
        for lender in borrow_from:
            if not isinstance(lender, Instance):
                raise ValueError('Lenders must be Instance handles')
            self._current(lender)
            if lender.node != node or lender == staging_for:
                raise ValueError('Lenders must be other instances in the same pool')
            borrowed.update(rank for rank in lender.ranks
                            if eligible_borrowed is None or rank in eligible_borrowed)
        free = [rank for rank in range(len(domains))
                if ((node, rank) not in self._owners or rank in borrowed)
                and rank not in staging_for.ranks]
        best = None
        for extra in self._placement_memberships(
                node, free, tp * pp - tp, distinguish_owned=True,
                fixed_ranks=staging_for.ranks):
            membership = tuple(staging_for.ranks) + tuple(extra)
            ordered = self._hydra_stage_order(
                domains, membership, tp, staging_for.ranks,
                physical_nodes)
            if ordered is None:
                continue
            if len(set(physical_nodes)) == 1 and not self._balanced(domains, ordered):
                continue
            score = (sum((node, rank) in self._owners and rank not in staging_for.ranks
                         for rank in ordered), ordered)
            if best is None or score < best:
                best = score
        if best is None:
            raise ResidentCapacityUnavailable(
                'No free NUMA-balanced Hydra staging placement')
        # Preserve hot-instance resources whenever idle GPUs can supply the
        # same PP topology. Placement preview and admission share this order.
        return best[1]

    def create(self, name, model, tp, *, pp=1, on_reserved=None,
               admission_timeout=0.0, staging_for=None, borrow_from=(),
               required_free_bytes=None, native_required_free_bytes=0, reserve_only=False,
               hydra_min_free_memory_ratio=0.0, **load_options):
        borrow_from = tuple(borrow_from)
        if (type(native_required_free_bytes) is not int or native_required_free_bytes < 0
                or (native_required_free_bytes and staging_for is None)):
            raise ValueError('Native recovery budget requires Hydra staging')
        if native_required_free_bytes and (
                type(required_free_bytes) is not int or required_free_bytes <= 0):
            raise ValueError('Native recovery budget requires a staging memory budget')
        if 'required_free_bytes_by_rank' in load_options:
            raise ValueError('Controller owns per-rank memory budgets')
        self._validate_hydra_memory_ratio(hydra_min_free_memory_ratio)
        if hydra_min_free_memory_ratio and staging_for is None:
            raise ValueError('Hydra memory threshold requires staging_for')
        eligible_borrowed = self._hydra_eligible_lenders(
            staging_for, borrow_from, hydra_min_free_memory_ratio)
        if type(reserve_only) is not bool:
            raise ValueError('reserve_only must be boolean')
        if reserve_only and (staging_for is not None or on_reserved is not None
                             or load_options or required_free_bytes is not None):
            raise ValueError('Capacity reservation cannot include model loading options')
        if (isinstance(admission_timeout, bool) or
                not math.isfinite(admission_timeout) or admission_timeout < 0):
            raise ValueError('Admission timeout must be finite and nonnegative')
        deadline = time.monotonic() + admission_timeout
        with self._lock:
            borrow_from = tuple(borrow_from)
            if borrow_from and staging_for is None:
                raise ValueError('Borrowing requires a Hydra staging instance')
            if borrow_from and (type(required_free_bytes) is not int or required_free_bytes <= 0):
                raise ValueError('Borrowing requires a positive required_free_bytes budget per rank')
            if required_free_bytes is not None:
                if type(required_free_bytes) is not int or required_free_bytes <= 0:
                    raise ValueError('required_free_bytes must be a positive integer')
                load_options['required_free_bytes'] = required_free_bytes
            if self._prewarming:
                raise ValueError('Communication prewarm is still in progress')
            if not isinstance(name, str) or not name or name in self._instances:
                raise ValueError('Instance name must be nonempty and unique')
            if not isinstance(model, str) or not model:
                raise ValueError('Model path must be nonempty')
            if {'model', 'topology', 'generation'} & load_options.keys():
                raise ValueError('Controller owns model, topology and generation')
            if on_reserved is not None and not callable(on_reserved):
                raise ValueError('Reservation callback must be callable')
            while True:
                try:
                    if staging_for is not None:
                        if not isinstance(staging_for, Instance):
                            raise ValueError('staging_for must be an Instance')
                        self._current(staging_for)
                        if not getattr(self.pools[staging_for.node], 'concurrent_topologies', False):
                            raise ValueError('Hydra staging requires concurrent_topologies')
                        if staging_for.tp_size != tp or staging_for.pp_size != 1:
                            raise ValueError('Hydra staging TP must match the native instance')
                        if model != staging_for.model:
                            raise ValueError('Hydra staging model must match the native instance')
                        if pp <= staging_for.pp_size:
                            raise ValueError('Hydra staging PP must exceed the native PP')
                        node = staging_for.node
                        ranks = self._hydra_placement(staging_for, tp, pp, borrow_from, eligible_borrowed)
                    else:
                        node, ranks = self._select(tp, pp)
                    break
                except ResidentCapacityUnavailable:
                    # Reject impossible placement rather than sleep until timeout.
                    if not any(node not in self._quarantined and any(
                            self._stage_order(domains, members, tp, self.physical_nodes[node]) is not None
                            and (len(set(self.physical_nodes[node])) > 1 or self._balanced(domains, members))
                            for members in self._placement_memberships(
                                node, range(len(domains)), tp * pp))
                            for node, domains in self.numa.items()):
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    # Condition.wait releases metadata; no rank/AMX lease exists yet.
                    self._capacity.wait(remaining)
                    if name in self._instances:
                        raise ValueError('Instance name must be nonempty and unique')
            if hydra_min_free_memory_ratio:
                load_options['hydra_min_free_memory_ratio'] = hydra_min_free_memory_ratio
                load_options['hydra_borrowed_ranks'] = [
                    rank for rank in ranks if any(rank in lender.ranks for lender in borrow_from)]
            if native_required_free_bytes:
                load_options['required_free_bytes_by_rank'] = {
                    str(rank): required_free_bytes + (
                        native_required_free_bytes if rank in staging_for.ranks else 0)
                    for rank in ranks}
            prepared_key = ranks if pp == 1 else (ranks, tp, pp)
            if (getattr(self.pools[node], 'concurrent_topologies', False)
                    and prepared_key not in self._prepared[node]
                    and node not in self._bounded_prewarm):
                raise ValueError('Concurrent pool topology must be prewarmed before admission')
            self._generation += 1
            phase = ('reload' if all((node, rank) in self._initialized_ranks
                                     for rank in ranks) else 'first')
            slot = load_options.pop('slot', None) or (
                f'hydra-{name}' if staging_for is not None else 'default')
            if not isinstance(slot, str) or not slot or len(slot) > 128:
                raise ValueError('slot must be a nonempty string of at most 128 characters')
            if staging_for is None:
                for rank in ranks:
                    self._owners[node, rank] = name
            else:
                if any((node, rank) not in self._owners for rank in staging_for.ranks):
                    raise ValueError('Hydra staging requires every native rank to remain owned')
                for rank in ranks:
                    if (node, rank) not in self._owners:
                        self._owners[node, rank] = name
            instance = Instance(name, model, node, ranks, self._generation, False,
                                phase, tp, pp, slot)
            self._instances[name] = instance
            self._pending_creates.add(name)
            if reserve_only:
                self._pending_creates.discard(name)
                self._unloaded_reservations.add(name)
                self._last_activity[name] = self._clock()
                return instance
        with self._execution(instance):
            with self._lock:
                self._current(instance, allow_pending=True)
            if on_reserved is not None:
                try:
                    # Capacity is reserved and the node is available. Start
                    # CPU work now so it overlaps GPU preparation/loading,
                    # rather than waiting while another node command runs.
                    overrides = on_reserved(instance)
                    if overrides is not None:
                        if (not isinstance(overrides, dict)
                                or overrides.keys() != {'load_method'}
                                or overrides['load_method'] not in ('tokenwise', 'layerwise')):
                            raise ValueError('Reservation callback may only select the relay load method')
                        load_options.update(overrides)
                except BaseException:
                    # No device command has run for this instance yet.
                    with self._lock:
                        del self._instances[name]
                        self._pending_creates.discard(name)
                        self._drop_reservations(instance)
                        self._capacity.notify_all()
                    raise
        try:
            self._prepare_topology(instance)
        except ResidentCommandRejected:
            with self._lock:
                del self._instances[name]
                self._pending_creates.discard(name)
                self._drop_reservations(instance)
                self._capacity.notify_all()
            raise
        with self._execution(instance):
            with self._lock:
                self._current(instance, allow_pending=True)
            try:
                reply = self._command(node, 'load', model=model, topology=instance.topology,
                                      generation=instance.generation, slot=slot, **load_options)
            except ResidentCommandRejected:
                # Rank preflight proves this instance acquired no device state.
                with self._lock:
                    del self._instances[name]
                    self._pending_creates.discard(name)
                    self._drop_reservations(instance)
                    self._capacity.notify_all()
                raise
            values = self._results(instance, reply)
            ready = all(value['ready'] for value in values)
            if any(value['ready'] != ready for value in values):
                self._quarantine(node)
                raise RuntimeError('Resident ranks disagree about model readiness')
            instance = replace(instance, ready=ready)
            with self._lock:
                self._instances[name] = instance
                self._pending_creates.discard(name)
                self._last_activity[name] = self._clock()
                if ready:
                    self._initialized_ranks.update((node, rank) for rank in ranks)
            return instance

    def promote_hydra(self, name, native, planner, input_tokens, *,
                      ttft_slo_ms, tpot_slo_ms, borrow_from=(),
                      hydra_min_free_memory_ratio=0.0, **load_options):
        """Admit the smallest PP meeting SLO, otherwise the largest available PP.

        A PP=1 result is the original native handle, still reserved if it was
        not materialized. Callers then load it directly without PP recovery.
        """
        from sllm.resident.hydra import HydraPlanner
        if not isinstance(planner, HydraPlanner):
            raise ValueError('Hydra promotion requires a measured HydraPlanner')
        if (planner.profile['model'] != native.model
                or planner.profile['tp'] != native.tp_size
                or load_options.get('max_model_len') != planner.profile['max_model_len']):
            raise ValueError('Hydra profile model/TP/context does not match admission')
        if {'pp', 'staging_for', 'required_free_bytes', 'native_required_free_bytes',
                'required_free_bytes_by_rank', 'startup_mode'} & load_options.keys():
            raise ValueError('Hydra planner owns promotion options')
        if 'on_reserved' in load_options:
            raise ValueError('Hydra promotion owns its reservation callback')
        self._validate_hydra_memory_ratio(hydra_min_free_memory_ratio)
        with self.instance_admission(native):
            borrow_from = tuple(borrow_from)
            eligible_borrowed = self._hydra_eligible_lenders(
                native, borrow_from, hydra_min_free_memory_ratio)
            with self._lock:
                phases = {1: native.initialization_phase}
                for entry in planner.profile['pp_profiles']:
                    pp = entry['pp']
                    if pp == 1:
                        continue
                    try:
                        ranks = self._hydra_placement(native, native.tp_size, pp, borrow_from, eligible_borrowed)
                    except ResidentCapacityUnavailable:
                        continue
                    phases[pp] = ('reload' if all(
                        (native.node, rank) in self._initialized_ranks for rank in ranks)
                        else 'first')
            candidates = planner.candidates(input_tokens,
                ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
                initialization_phase=native.initialization_phase,
                placement_phases=phases)
            rejections = []
            for plan in candidates:
                if plan['pp'] == 1:
                    return native, dict(plan, rejected_candidates=rejections)
                def check_phase(stage):
                    if stage.initialization_phase != plan['initialization_phase']:
                        raise ValueError('Hydra placement initialization phase differs from its profile')
                try:
                    with self._lock:
                        native_budget = (plan['recovery_required_free_bytes']
                            if native.name in self._unloaded_reservations else 0)
                    stage_options = dict(load_options)
                    if 'native_num_gpu_blocks' in plan:
                        if 'num_gpu_blocks_override' in stage_options:
                            raise ValueError('Hydra planner owns staging KV block capacity')
                        stage_options['num_gpu_blocks_override'] = plan['native_num_gpu_blocks']
                    stage = self.create(name, native.model, native.tp_size,
                        pp=plan['pp'], staging_for=native, borrow_from=borrow_from,
                        required_free_bytes=plan['resident_required_free_bytes'],
                        native_required_free_bytes=native_budget,
                        hydra_min_free_memory_ratio=hydra_min_free_memory_ratio,
                        startup_mode='hydraserve', on_reserved=check_phase, **stage_options)
                except ResidentCapacityUnavailable as exc:
                    rejections.append(dict(pp=plan['pp'], reason='placement_capacity',
                                           error=str(exc)))
                    continue
                except ResidentCommandRejected as exc:
                    if exc.memory_capacity_unavailable:
                        rejections.append(dict(pp=plan['pp'], reason='memory_capacity',
                            staging_required_free_bytes=plan['resident_required_free_bytes'],
                            native_required_free_bytes=native_budget,
                            rank_errors=exc.reply['ranks']))
                        continue
                    raise
                return stage, dict(plan, rejected_candidates=rejections)
        raise ResidentCapacityUnavailable('No Hydra placement or native fallback')

    def _prepare_topology(self, instance):
        """Prepare a missing WORLD collective before taking instance locks.

        Callers pin the instance through pending creation or an operation.
        Lock order is node fence, then ascending rank locks, then metadata.
        Holding only this instance's rank locks here would deadlock with
        concurrent preparation or allow other ranks to execute during prepare.
        """
        node = instance.node
        key = instance.ranks if instance.pp_size == 1 else (
            instance.ranks, instance.tp_size, instance.pp_size)
        with self._lock:
            self._current(instance, allow_pending=True)
            if key in self._prepared[node]:
                return
        with ExitStack() as fence:
            fence.enter_context(self._node_locks[node])
            for rank_lock in self._rank_locks[node]:
                fence.enter_context(rank_lock)
            with self._lock:
                self._current(instance, allow_pending=True)
                if key in self._prepared[node]:
                    return
            self._command(node, 'prepare', topologies=[instance.topology])
            with self._lock:
                self._prepared[node].add(key)

    def materialize(self, instance, **load_options):
        """Construct a previously reserved native slot without moving its GPUs.

        Rank commands remain serialized with temporary PP iterations. Native
        SHM transfers may continue after construction returns; callers must
        independently initialize and poll weights before recovering requests.
        """
        if {'model', 'topology', 'generation', 'slot'} & load_options.keys():
            raise ValueError('Controller owns native reservation routing fields')
        with self._instance_operation(instance, allow_reserved=True, prepare=True):
            with self._lock:
                if instance.name not in self._unloaded_reservations:
                    raise ValueError('Instance is not an unloaded reservation')
            reply = self._command(instance.node, 'load', model=instance.model,
                topology=instance.topology, generation=instance.generation,
                slot=self._slot_name(instance), **load_options)
            values = self._results(instance, reply)
            if len({value['ready'] for value in values}) != 1:
                self._quarantine(instance.node)
                raise RuntimeError('Resident ranks disagree about model readiness')
            instance = replace(instance, ready=values[0]['ready'])
            with self._lock:
                self._instances[instance.name] = instance
                self._unloaded_reservations.remove(instance.name)
                if instance.ready:
                    self._initialized_ranks.update((instance.node, rank) for rank in instance.ranks)
            return instance

    def lazy_init(self, instance, layers, request_id):
        with self._instance_operation(instance):
            if instance.ready:
                raise ValueError('Instance is already initialized')
            reply = self._command(instance.node, 'lazy_init', topology=instance.topology,
                generation=instance.generation, slot=self._slot_name(instance),
                layer_idxes_by_pp=([list(layers)] if instance.pp_size == 1
                                   else [list(stage) for stage in layers]),
                request_id=request_id)
            values = self._results(instance, reply)
            if not all(value['ready'] for value in values):
                self._quarantine(instance.node)
                raise RuntimeError('Lazy initialization did not make all ranks ready')
            durations = [value.get('vllm_initialization_seconds') for value in values]
            if any(value is not None and (type(value) not in (int, float)
                    or not math.isfinite(value) or value < 0) for value in durations):
                self._quarantine(instance.node)
                raise RuntimeError('Invalid vLLM initialization timing')
            instance = replace(instance, ready=True,
                vllm_initialization_seconds=(max(durations)
                    if all(value is not None for value in durations) else None))
            with self._lock:
                self._instances[instance.name] = instance
                self._initialized_ranks.update((instance.node, rank) for rank in instance.ranks)
            return instance

    def weights_ready(self, instance):
        """Poll native loader events on every rank; never wait for H2D here."""
        with self._instance_operation(instance):
            reply = self._command(instance.node, 'weights_ready',
                topology=instance.topology, generation=instance.generation,
                slot=self._slot_name(instance))
            return all(value['weights_ready'] for value in self._results(instance, reply))

    def generate(self, instance, prompt_token_ids, *, max_tokens=8, on_tokens=None,
                 **request_options):
        with self._instance_operation(instance):
            if on_tokens is not None:
                # Keep the synchronous command for non-streaming callers.
                # Streaming uses the same request/session path as the pump.
                request_id = request_options.pop('request_id', f'stream-{time.monotonic_ns()}')
                self.submit(instance, prompt_token_ids, request_id=request_id,
                            max_tokens=max_tokens, **request_options)
                try:
                    while True:
                        step = self.advance(instance)
                        tokens = step['token_updates'].get(request_id, [])
                        if tokens:
                            on_tokens(tokens)
                        if request_id in step['finished_request_ids']:
                            return self.collect(instance, request_id)
                except BaseException as exc:
                    if instance.node not in self._quarantined:
                        try:
                            self.cancel_request(instance, request_id)
                        except BaseException as cleanup_error:
                            exc.add_note(f'Stream request cleanup failed: {cleanup_error}')
                    raise
            if not instance.ready:
                raise ValueError('Instance needs lazy initialization')
            if {'model', 'topology', 'generation', 'prompt_token_ids', 'max_tokens'} & request_options.keys():
                raise ValueError('Controller owns request routing fields')
            reply = self._command(instance.node, 'generate', model=instance.model,
                topology=instance.topology, generation=instance.generation,
                slot=self._slot_name(instance),
                prompt_token_ids=list(prompt_token_ids), max_tokens=max_tokens, **request_options)
            values = self._results(instance, reply)
            if any(value['token_ids'] != values[0]['token_ids'] for value in values):
                self._quarantine(instance.node)
                raise RuntimeError('TP ranks disagree about generated tokens')
            return values[0]

    def submit(self, instance, prompt_token_ids, *, request_id, max_tokens=8,
               **request_options):
        with self._instance_operation(instance):
            if request_options.keys() - {'kv_transfer_params'}:
                raise ValueError('Unknown incremental request options')
            if not instance.ready:
                raise ValueError('Instance needs lazy initialization')
            if not isinstance(request_id, str) or not request_id:
                raise ValueError('Request ID must be nonempty')
            with self._lock:
                pending = self._submitted_requests.setdefault(instance.name, set())
                if request_id in pending:
                    raise ValueError('Duplicate request ID')
            reply = self._command(instance.node, 'submit', model=instance.model,
                topology=instance.topology, generation=instance.generation,
                slot=self._slot_name(instance),
                prompt_token_ids=list(prompt_token_ids), max_tokens=max_tokens,
                request_id=request_id, **request_options)
            self._results(instance, reply)
            with self._lock:
                pending.add(request_id)

    def step(self, instance):
        return self.advance(instance)['pending']

    def recover_pp(self, source, target):
        """Move live temporary-PP requests to its drained native TP slot."""
        if (source.node != target.node or source.tp_size != target.tp_size
                or source.pp_size <= 1 or target.pp_size != 1
                or source.ranks[:source.tp_size] != target.ranks):
            raise ValueError('Recovery requires native TP at temporary stage zero')
        with self._instance_operation(source), self._instance_operation(target):
            with self._lock:
                if self._submitted_requests.get(target.name):
                    raise ValueError('Drain destination before recovery')
                if (self._instance_admissions.get(source.name)
                        or self._instance_admissions.get(target.name)):
                    raise ValueError('Recovery must run between pump ownership sessions')
                pending = set(self._submitted_requests.get(source.name, ()))
            reply = self._command(source.node, 'recover_pp',
                topology=source.topology, generation=source.generation,
                slot=self._slot_name(source), target_slot=self._slot_name(target),
                target_topology=target.topology, target_generation=target.generation)
            values = [reply.ranks[rank]['result'] for rank in source.ranks]
            if any(set(value['recovered_request_ids']) != pending for value in values):
                self._quarantine(source.node)
                raise RuntimeError('Recovered request ownership mismatch')
            with self._lock:
                self._submitted_requests[source.name] = set()
                self._submitted_requests[target.name] = pending
            return values[0]

    def advance(self, instance):
        """Run one iteration and expose completed IDs without consuming results."""
        with self._instance_operation(instance):
            reply = self._command(instance.node, 'step', topology=instance.topology,
                                  generation=instance.generation, slot=self._slot_name(instance))
            values = self._results(instance, reply)
            finished = [value.get('finished_request_ids', []) for value in values]
            if any(ids != finished[0] for ids in finished):
                self._quarantine(instance.node)
                raise RuntimeError('Ranks disagree about completed requests')
            produced = [value.get('first_output_request_ids', []) for value in values]
            if any(ids != produced[0] for ids in produced):
                self._quarantine(instance.node)
                raise RuntimeError('Ranks disagree about first-output requests')
            updates = [value.get('token_updates', {}) for value in values]
            if any(tokens != updates[0] for tokens in updates):
                self._quarantine(instance.node)
                raise RuntimeError('Ranks disagree about streamed tokens')
            return dict(pending=any(value['pending'] for value in values),
                        token_updates=updates[0],
                        recovery_ready=all(value.get('recovery_ready', True) for value in values),
                        pp_transition_complete=all(value.get('pp_transition_complete', False)
                                                   for value in values),
                        first_output_request_ids=produced[0],
                        finished_request_ids=finished[0])

    def collect(self, instance, request_id):
        with self._instance_operation(instance):
            with self._lock:
                if request_id not in self._submitted_requests.get(instance.name, ()):
                    raise ValueError('Unknown request ID')
            reply = self._command(instance.node, 'collect', topology=instance.topology,
                generation=instance.generation, slot=self._slot_name(instance), request_id=request_id)
            values = self._results(instance, reply)
            if any(value['token_ids'] != values[0]['token_ids'] for value in values):
                self._quarantine(instance.node)
                raise RuntimeError('TP ranks disagree about generated tokens')
            with self._lock:
                self._submitted_requests[instance.name].remove(request_id)
            return values[0]

    def cancel_request(self, instance, request_id):
        """Abort one submitted request without unloading its model or peers."""
        with self._instance_operation(instance):
            with self._lock:
                if request_id not in self._submitted_requests.get(instance.name, ()):
                    raise ValueError('Unknown request ID')
            reply = self._command(instance.node, 'abort', topology=instance.topology,
                generation=instance.generation, slot=self._slot_name(instance),
                request_id=request_id)
            values = self._results(instance, reply)
            if any(value.get('cancelled') is not True for value in values):
                self._quarantine(instance.node)
                raise RuntimeError('Ranks disagree about request cancellation')
            with self._lock:
                self._submitted_requests[instance.name].remove(request_id)
            return values[0]

    def _require_drained(self, instance):
        with self._lock:
            if (self._submitted_requests.get(instance.name) or
                    self._instance_admissions.get(instance.name)):
                raise ValueError('Drain and collect submitted requests before model teardown')

    @contextmanager
    def _model_teardown(self, instance):
        with self._lock:
            self._require_drained(instance)
            self._tearing_down.add(instance.name)
        try:
            yield
        finally:
            with self._lock:
                self._tearing_down.discard(instance.name)

    def release(self, instance):
        with self._instance_operation(instance, allow_reserved=True):
            self._release_locked(instance)

    def _release_locked(self, instance):
        with self._model_teardown(instance):
            self._release_drained(instance)

    def _release_drained(self, instance):
        with self._lock:
            if instance.name in self._unloaded_reservations:
                self._unloaded_reservations.remove(instance.name)
                del self._instances[instance.name]
                self._last_activity.pop(instance.name, None)
                self._drop_reservations(instance)
                self._capacity.notify_all()
                return
        reply = self._command(instance.node, 'unload', topology=instance.topology,
                              generation=instance.generation, slot=self._slot_name(instance))
        values = self._results(instance, reply, unloaded=True)
        # Unload clears the model name, so generation remains authoritative.
        if any(value['ready'] for value in values):
            self._quarantine(instance.node)
            raise RuntimeError('Unload left a rank ready')
        with self._lock:
            del self._instances[instance.name]
            self._last_activity.pop(instance.name, None)
            self._drop_reservations(instance)
            self._capacity.notify_all()

    def reap_idle(self):
        """Unload expired slots without waiting behind active node commands.

        Queueing and execution both hold an operation reservation. Claim expiry
        under the metadata lock, then unload with only the node lock held.
        Failed nodes retain allocations and are quarantined by _command.
        """
        if not self.idle_timeout:
            return []
        with self._lock:
            candidates = list(self._instances.values())
        expired = []
        for instance in candidates:
            with self._execution(instance, blocking=False) as acquired:
                if not acquired:
                    continue
                with self._lock:
                    if (self._instances.get(instance.name) != instance or
                            instance.node in self._quarantined or
                            instance.name in self._pending_creates or
                            self._operations.get(instance.name) or
                            self._admitted_requests.get(instance.name) or
                            self._submitted_requests.get(instance.name) or
                            self._instance_admissions.get(instance.name) or
                            instance.name in self._expiring or
                            self._clock() - self._last_activity[instance.name] < self.idle_timeout):
                        continue
                    self._expiring.add(instance.name)
                try:
                    self._release_locked(instance)
                    expired.append(instance.name)
                except RuntimeError:
                    # Continue independent nodes after a failed unload.
                    self._quarantine(instance.node)
                finally:
                    with self._lock:
                        self._expiring.discard(instance.name)
                        self._capacity.notify_all()
        return expired

    def switch(self, instance, model, **load_options):
        with self._instance_operation(instance), self._model_teardown(instance):
            if not isinstance(model, str) or not model:
                raise ValueError('Model path must be nonempty')
            if {'model', 'topology', 'generation', 'slot'} & load_options.keys():
                raise ValueError('Controller owns model, topology, generation and slot')
            reply = self._command(instance.node, 'unload', topology=instance.topology,
                                  generation=instance.generation, slot=self._slot_name(instance))
            values = self._results(instance, reply, unloaded=True)
            if any(value['ready'] for value in values):
                self._quarantine(instance.node)
                raise RuntimeError('Unload left a rank ready')
            with self._lock:
                self._generation += 1
                phase = ('reload' if all((instance.node, rank) in self._initialized_ranks
                                         for rank in instance.ranks) else 'first')
                instance = replace(instance, model=model, generation=self._generation,
                                   ready=False, initialization_phase=phase)
                self._instances[instance.name] = instance
            try:
                reply = self._command(instance.node, 'load', model=model, topology=instance.topology,
                                      generation=instance.generation, slot=self._slot_name(instance),
                                      **load_options)
            except ResidentCommandRejected:
                # Unload succeeded and the new generation never executed:
                # there is no live model slot left to retain or release.
                with self._lock:
                    del self._instances[instance.name]
                    self._last_activity.pop(instance.name, None)
                    for rank in instance.ranks:
                        if self._owners.get((instance.node, rank)) == instance.name:
                            del self._owners[instance.node, rank]
                    self._capacity.notify_all()
                raise
            values = self._results(instance, reply)
            if len({value['ready'] for value in values}) != 1:
                self._quarantine(instance.node)
                raise RuntimeError('Resident ranks disagree about model readiness')
            instance = replace(instance, ready=values[0]['ready'])
            with self._lock:
                if instance.ready:
                    self._initialized_ranks.update((instance.node, rank) for rank in instance.ranks)
                self._instances[instance.name] = instance
            return instance

    def status(self):
        with self._lock:
            return dict(instances=list(self._instances.values()),
                        quarantined_nodes=sorted(self._quarantined),
                        allocated_gpus=len(self._owners))

    def lookup(self, name, generation):
        """Resolve an external route without silently upgrading its generation."""
        with self._lock:
            instance = self._instances.get(name)
            if (instance is None or type(generation) is not int or
                    generation != instance.generation):
                raise ValueError('Stale or unknown instance generation')
            return self._current(instance)
