"""Host CPU capacity, including container CPU quotas."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _cgroup_quota_cpus() -> Optional[float]:
    v2 = Path("/sys/fs/cgroup/cpu.max")
    try:
        if v2.is_file():
            quota, period = v2.read_text().split()[:2]
            return None if quota == "max" else int(quota) / int(period)
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return quota / period if quota > 0 else None
    except (OSError, ValueError):
        return None


def effective_cpu_count() -> int:
    """CPUs this process may actually use: affinity mask, capped by a cgroup quota."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    quota = _cgroup_quota_cpus()
    if quota is not None:
        count = min(count, max(1, int(quota)))
    return max(1, count)
