# SPDX-License-Identifier: Apache-2.0
"""KV transfer plans shared by tokenwise and layerwise CPU handoff.

Plans contain block indices only; tensor data stays in the native SHM path.
Callers retain responsibility for readiness and synchronization.
"""
from dataclasses import dataclass, field


@dataclass
class RelayKVTransfer:
    gpu_block_ids: list[int] = field(default_factory=list)
    cpu_block_ids: list[int] = field(default_factory=list)

    def add_request(self, local: list[int], remote: list[int]) -> None:
        if len(local) != len(remote):
            raise ValueError("CPU/GPU relay block counts differ")
        self.gpu_block_ids.extend(local)
        self.cpu_block_ids.extend(remote)


class RelayKVPlan:
    """Batch requests that transfer the same layer interval.

    Tokenwise adds the registered layer interval once per request. Layerwise
    adds only layers that changed ownership, including all historical blocks.
    """

    def __init__(self) -> None:
        self.transfers: dict[tuple[int, ...], RelayKVTransfer] = {}

    def add_request(self, layers: tuple[int, ...], local: list[int],
                    remote: list[int]) -> None:
        # Validate even empty intervals before accepting request metadata.
        if len(local) != len(remote):
            raise ValueError("CPU/GPU relay block counts differ")
        if not layers or not local:
            return
        if layers not in self.transfers:
            self.transfers[layers] = RelayKVTransfer()
        self.transfers[layers].add_request(local, remote)
