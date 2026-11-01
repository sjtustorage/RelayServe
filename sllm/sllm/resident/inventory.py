"""Resolve the actual CUDA device UUID to Linux PCI/NUMA topology.

Run inside each resident worker after setting its CUDA device. UUID matching
uses the device selected by CUDA itself, including visible-device reordering.
Unknown NUMA affinity remains unknown; it must never be guessed as node zero.
"""
import csv
import io
import os
from pathlib import Path
import subprocess


def resolve_gpu_inventory(uuid, query_output, sysfs=Path('/sys/bus/pci/devices')):
    uuid = str(uuid).lower().removeprefix('gpu-')
    matches = []
    for row in csv.reader(io.StringIO(query_output)):
        if len(row) != 3:
            raise ValueError('Invalid nvidia-smi GPU inventory row')
        index, identity, bus = (value.strip() for value in row)
        if identity.lower().removeprefix('gpu-') == uuid:
            matches.append((int(index), identity, bus.lower()))
    if len(matches) != 1:
        raise ValueError('CUDA UUID does not uniquely match a physical GPU')
    index, identity, bus = matches[0]
    domain, remainder = bus.split(':', 1)
    pci = f'{int(domain, 16):04x}:{remainder}'
    try:
        numa = int((Path(sysfs) / pci / 'numa_node').read_text().strip())
    except (OSError, ValueError):
        numa = -1
    return dict(physical_index=index, uuid=identity, pci_bus_id=pci,
                numa_node=numa if numa >= 0 else None)


def current_gpu_inventory(device):
    import torch
    output = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id',
         '--format=csv,noheader,nounits'], check=True, capture_output=True,
        text=True, timeout=10).stdout
    uuid = torch.cuda.get_device_properties(device).uuid
    return resolve_gpu_inventory(uuid, output)


def local_cpu_affinity(numa_node, allowed, sysfs=Path('/sys/devices/system/node')):
    """Respect the launcher's CPU allocation while restricting to GPU NUMA."""
    if type(numa_node) is not int or numa_node < 0:
        raise ValueError('GPU NUMA node is unknown; CPU placement cannot be resolved')
    local = set()
    for item in (Path(sysfs) / f'node{numa_node}' / 'cpulist').read_text().strip().split(','):
        bounds = [int(value) for value in item.split('-')]
        if len(bounds) == 1:
            local.add(bounds[0])
        elif len(bounds) == 2 and bounds[0] <= bounds[1]:
            local.update(range(bounds[0], bounds[1] + 1))
        else:
            raise ValueError('Invalid NUMA CPU list')
    selected = sorted(local.intersection(allowed))
    if not selected:
        raise ValueError('GPU NUMA node has no CPUs in the worker launch affinity')
    return selected


def bind_worker_to_gpu_numa(inventory):
    allowed = os.sched_getaffinity(0)
    configured = os.environ.get('SLLM_RESIDENT_GPU_CPU_IDS')
    if configured is not None:
        try:
            values = [int(value) for value in configured.split(',')]
        except ValueError as exc:
            raise ValueError('Invalid SLLM_RESIDENT_GPU_CPU_IDS') from exc
        if (not values or len(set(values)) != len(values)
                or any(cpu < 0 for cpu in values) or not set(values) <= allowed):
            raise ValueError('GPU CPU allocation must contain distinct allowed CPU IDs')
        allowed = set(values)
    selected = local_cpu_affinity(inventory['numa_node'], allowed)
    # torch import may already have created threads. Bind them as well as the
    # main thread, before communication/native loaders create further threads.
    os.sched_setaffinity(0, selected)
    for task in Path('/proc/self/task').iterdir():
        try:
            os.sched_setaffinity(int(task.name), selected)
        except ProcessLookupError:
            pass  # A transient import helper exited during enumeration.
    return dict(inventory, cpu_affinity=sorted(os.sched_getaffinity(0)))
