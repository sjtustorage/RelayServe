# ---------------------------------------------------------------------------- #
#  Ray remote tasks to wipe /dev/shm on *each* cluster node.                    #
#  Lives in a tiny module so workers do not import cli_utils (→ controller,   #
#  transformers, etc.).                                                        #
# ---------------------------------------------------------------------------- #
from __future__ import annotations

import logging
import os
import subprocess
from typing import List

import ray

logger = logging.getLogger(__name__)


def run_dev_shm_rm() -> None:
    """Run ``rm -rf /dev/shm/*`` on this machine (Linux; no-op if ``/dev/shm`` missing).

    Clears the whole tmpfs, not only SLLM; use on dedicated nodes or when safe.
    """
    shm = "/dev/shm"
    if not os.path.isdir(shm):
        return
    try:
        r = subprocess.run(
            ["/bin/sh", "-c", "rm -rf /dev/shm/*"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if r.returncode != 0 and (r.stderr or r.stdout):
            logger.warning(
                "rm /dev/shm/* exited %s: %s%s",
                r.returncode,
                (r.stderr or "").strip(),
                (r.stdout or "").strip(),
            )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        logger.warning("Failed to clean /dev/shm: %s", e)


@ray.remote(num_cpus=0)
def cleanup_dev_shm_on_ray_node() -> None:
    """Runs on a worker: local ``/dev/shm`` cleanup for that node."""
    run_dev_shm_rm()


def cleanup_dev_shm_all_cluster_nodes() -> None:
    """Schedule ``/dev/shm`` cleanup on every live Ray node (before ``ray.shutdown()``)."""
    if not ray.is_initialized():
        return
    try:
        from ray.util.scheduling_strategies import (
            NodeAffinitySchedulingStrategy,
        )
    except ImportError:
        logger.warning(
            "NodeAffinitySchedulingStrategy unavailable; "
            "only cleaning /dev/shm on the local process after shutdown"
        )
        return
    refs: List[ray.ObjectRef] = []
    for node in ray.nodes():
        if not node.get("Alive", True):
            continue
        node_id = node.get("NodeID")
        if node_id is None:
            continue
        try:
            ref = cleanup_dev_shm_on_ray_node.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
            ).remote()
            refs.append(ref)
        except Exception as e:
            logger.warning(
                "Could not schedule /dev/shm cleanup on node %s: %s",
                node_id,
                e,
            )
    if not refs:
        return
    try:
        ray.get(refs, timeout=600)
    except Exception as e:
        logger.warning("Cluster /dev/shm cleanup did not complete cleanly: %s", e)
