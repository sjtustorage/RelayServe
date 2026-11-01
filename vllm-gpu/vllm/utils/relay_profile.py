"""Optional host scheduling diagnostics for concurrent relay initialization."""

import os
from typing import Optional


def thread_runqueue_wait_seconds() -> Optional[float]:
    """Return cumulative Linux runqueue wait for the calling thread."""
    if not os.environ.get("VLLM_PROFILE_RELAY"):
        return None
    try:
        with open("/proc/thread-self/schedstat") as stats:
            return int(stats.read().split()[1]) / 1e9
    except (OSError, ValueError, IndexError):
        return None
