"""How many worker processes this machine can actually run, and setting them up.

``os.cpu_count()`` reports the host's cores, which overstates what a process
may use in a container or under a CPU affinity mask: a build told "64 cores"
inside a 4-CPU container runs 64 workers on 4 CPUs and gets slower, not
faster.  :func:`usable_cpus` therefore takes the smallest of the affinity
mask, the cgroup CPU quota (v2 ``cpu.max`` or v1 ``cpu.cfs_quota_us``) and
``os.cpu_count()``.  Memory caps the worker count too, because each worker
loads its own copy of the lexicons and models: :func:`available_memory`
reads ``MemAvailable`` and the cgroup memory limit where they exist.

On a system that exposes neither (macOS, Windows), the CPU count falls back
to ``os.cpu_count()`` and the memory limit is not applied.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

#: Memory to allow per profile-building worker.  Measured on a 72,000-word
#: novel: the default metrics peaked at 1.74 GB in the worker plus 0.17 GB in a
#: helper subprocess some metrics start; with the parse and sentence-embedding
#: metrics, 2.15 GB plus 1.77 GB.  Each allowance leaves headroom over those.
WORKER_MEMORY_BYTES = 2.5 * 2**30
MODEL_WORKER_MEMORY_BYTES = 4.0 * 2**30

#: Thread-pool variables for the numeric libraries.  Each worker gets its
#: share of the CPUs, so N workers do not each start a pool the size of the
#: machine.
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cgroup_cpu_limit() -> float | None:
    """CPUs allowed by the cgroup quota, or ``None`` when unlimited or unknown."""
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2:
        quota, _, period = v2.partition(" ")
        if quota != "max" and period:
            return int(quota) / int(period)
        return None
    quota, period = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read(
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return int(quota) / int(period)
    return None


def usable_cpus() -> int:
    """CPUs this process may run on: affinity, cgroup quota and core count, lowest wins."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    limit = _cgroup_cpu_limit()
    if limit is not None:
        count = min(count, max(1, math.floor(limit)))
    return max(1, count)


def available_memory() -> int | None:
    """Bytes available to new processes, or ``None`` when the system does not say."""
    candidates = []
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemAvailable:"):
            candidates.append(int(line.split()[1]) * 1024)
    limit = _read("/sys/fs/cgroup/memory.max") or _read(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes")
    usage = _read("/sys/fs/cgroup/memory.current") or _read(
        "/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if limit and limit != "max" and usage and int(limit) < 2**60:
        candidates.append(int(limit) - int(usage))
    return min(candidates) if candidates else None


def auto_jobs(items: int, *, per_worker: float = WORKER_MEMORY_BYTES) -> tuple[int, str]:
    """The worker count for ``items`` independent tasks, and why."""
    cpus = usable_cpus()
    memory = available_memory()
    by_memory = max(1, int(memory // per_worker)) if memory is not None else None
    jobs = min(cpus, items, by_memory if by_memory is not None else cpus)
    jobs = max(1, jobs)
    memory_text = (f"{memory / 2**30:.1f} GB available / {per_worker / 2**30:.1f} GB per worker"
                   if memory is not None else "memory not reported")
    return jobs, f"{cpus} usable CPU(s), {memory_text}, {items} file(s)"


def resolve_jobs(jobs: str | int | None, items: int, *,
                 per_worker: float = WORKER_MEMORY_BYTES) -> tuple[int, str]:
    """``"auto"``/``None`` detects; an integer is used as given (at least 1, at most ``items``)."""
    if jobs in (None, "auto", 0, "0"):
        return auto_jobs(items, per_worker=per_worker)
    count = max(1, min(int(jobs), max(1, items)))
    return count, "set by --jobs"


def limit_worker_threads(threads: int = 1) -> None:
    """Cap numeric-library thread pools, both ones not yet started and ones already running.

    Profile builds run their math single-threaded, serial or parallel.
    float32 matrix products depend on how many threads split them: the same
    chapter's embedding similarities differed in the seventh significant
    digit between a 4-thread and a 1-thread run, so a profile depended on the
    machine's core count.  One thread per process makes a build the same on
    any machine and under any ``--jobs``; it was also faster here (72 s
    against 190 s serially), since these matrices are too small for threads
    to pay for their overhead.
    """
    count = str(max(1, threads))
    for name in _THREAD_VARIABLES:
        os.environ[name] = count
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:  # thread pools a library already started (numpy's BLAS, OpenMP)
        from threadpoolctl import threadpool_limits
        threadpool_limits(max(1, threads))
    except Exception:  # pragma: no cover - optional package
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(max(1, threads))
