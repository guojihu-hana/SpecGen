"""Resource pool shared by every task: GPUs for validation and profiling, CPUs for builds.

Each device has a role:

- ``validator``: up to ``capacity`` concurrent validations (correctness does not need
  an exclusive device);
- ``profiler``: one exclusive job at a time (timing or ncu);
- ``shared`` (``fifo`` allocation): validations up to ``capacity``, or one exclusive job;
- ``builder``: CPU kernel builds, ``capacity`` at a time.

A dispatcher places queued jobs in key order: ``(kind rank, priority, order, arrival)``
for ``fixed`` / ``dynamic`` (timing before ncu, main kernels before spec ones) and
``(kind rank, arrival)`` for ``fifo``.  An exclusive job that finds no idle device
reserves one: no new jobs start there until it has drained.  ``dynamic`` allocation
moves idle GPUs between the validator and profiler roles in proportion to their load.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Optional, Union

from specgen.evaluation.executor import GpuSlot
from specgen.telemetry.recorder import Ewma

log = logging.getLogger(__name__)

ALLOCATIONS = ("fixed", "dynamic", "fifo")
KIND_RANK = {"build": 0, "validate": 0, "benchmark": 0, "ncu": 1}
EXCLUSIVE_KINDS = ("benchmark", "ncu")
PRIORITY_MAIN, PRIORITY_SPEC = 0, 1
# Expected run times before any job of the kind has finished (seconds).
PRIOR_RUN_S = {"build": 50.0, "validate": 20.0, "benchmark": 15.0, "ncu": 60.0}
MIN_REBALANCE_LOAD_S = 60.0  # GPU-seconds of queued and running work before roles move


@dataclass
class PoolJob:
    kind: str  # build | validate | benchmark | ncu
    task_id: str
    epoch: int
    run: Callable[[GpuSlot], Awaitable[Any]]
    priority: int = PRIORITY_MAIN
    label: str = ""
    exclusive: bool = False  # validation alone on its GPU (retry after OOM); timing and ncu always are
    order: float = 0.0  # tie-break within a priority, lower first (e.g. -speedup for ncu)
    sticky: bool = False  # not cancelled by cancel(): e.g. ncu of kernels the next prompt shows
    urgent: bool = False  # ranked with timing instead of after it (e.g. ncu that spec launches wait for)

    @property
    def needs_device(self) -> bool:
        return self.exclusive or self.kind in EXCLUSIVE_KINDS


@dataclass(order=True)
class _Entry:
    key: tuple
    id: int = field(compare=False)
    job: PoolJob = field(compare=False)
    future: asyncio.Future = field(compare=False)
    enqueued_at: float = field(compare=False, default=0.0)


@dataclass(eq=False)
class Device:
    gpu: Optional[str]  # None: CPU
    role: str  # validator | profiler | shared | builder
    capacity: int = 1
    host: Optional[str] = None  # remote worker URL; None: this machine
    running: dict = field(default_factory=dict)  # entry id -> (entry, task, slot)
    reserved_by: Optional[int] = None  # entry id waiting for this device to drain

    @property
    def label(self) -> str:
        where = "cpu" if self.gpu is None else self.gpu
        return f"{self.role}:{where}" + (f"@{self.host}" if self.host else "")

    def free_index(self) -> int:
        used = {slot.index for _, _, slot in self.running.values()}
        return next(i for i in itertools.count() if i not in used)

    def has_exclusive(self) -> bool:
        return any(entry.job.needs_device for entry, _, _ in self.running.values())


@dataclass(frozen=True)
class PoolStats:
    slots: dict[str, int]  # role -> capacity
    busy: dict[str, int]  # role -> running jobs
    queued: dict[str, int]  # role -> queued jobs
    utilization: dict[str, float]  # role -> busy slot-seconds / capacity slot-seconds since start
    slot_busy_s: dict[str, float]  # device label -> busy seconds
    devices: list[dict[str, Any]] = field(default_factory=list)


def fixed_slots(
    gpus: list[str],
    *,
    num_validator: Optional[int] = None,
    num_profiler: Optional[int] = None,
    validations_per_gpu: int = 4,
) -> list[GpuSlot]:
    """Split GPUs into validator and profiler devices (CPU slots when ``gpus`` is empty)."""
    vpg = max(1, validations_per_gpu)
    if not gpus:
        return [GpuSlot(None, "validator", i) for i in range(vpg)] + [GpuSlot(None, "profiler", 0)]
    if len(gpus) == 1:
        log.warning("one GPU: validation and profiling share GPU %s, so timings are not isolated", gpus[0])
        return [GpuSlot(gpus[0], "validator", i) for i in range(vpg)] + [GpuSlot(gpus[0], "profiler", 0)]
    n = len(gpus)
    if num_validator is None and num_profiler is None:
        num_profiler = max(1, n // 2)
    if num_profiler is None:
        num_profiler = n - num_validator
    if num_validator is None:
        num_validator = n - num_profiler
    if num_validator < 1 or num_profiler < 1 or num_validator + num_profiler > n:
        raise ValueError(
            f"need >= 1 validator and >= 1 profiler GPU within {n} GPUs "
            f"(got {num_validator} validator, {num_profiler} profiler)"
        )
    validators, profilers = gpus[:num_validator], gpus[num_validator:num_validator + num_profiler]
    return [GpuSlot(g, "validator", i) for g in validators for i in range(vpg)] + [GpuSlot(g, "profiler", 0) for g in profilers]


def make_devices(
    gpus: list[str],
    *,
    allocation: str = "fixed",
    num_validator: Optional[int] = None,
    num_profiler: Optional[int] = None,
    validations_per_gpu: int = 4,
    build_workers: int = 0,
    host: Optional[str] = None,
) -> list[Device]:
    if allocation not in ALLOCATIONS:
        raise ValueError(f"unknown allocation {allocation!r}")
    vpg = max(1, validations_per_gpu)
    if allocation == "fifo":
        devices = [Device(g, "shared", vpg, host) for g in gpus] or [Device(None, "shared", vpg, host)]
    else:
        devices = devices_from_slots(fixed_slots(gpus, num_validator=num_validator, num_profiler=num_profiler,
                                                 validations_per_gpu=vpg), host=host)
    if build_workers > 0:
        devices.append(Device(None, "builder", build_workers, host))
    return devices


def devices_from_slots(slots: Iterable[GpuSlot], host: Optional[str] = None) -> list[Device]:
    devices: dict[tuple, Device] = {}
    for slot in slots:
        key = (slot.gpu, slot.role, slot.host or host)
        if key in devices:
            devices[key].capacity += 1
        else:
            devices[key] = Device(slot.gpu, slot.role, 1, slot.host or host)
    return list(devices.values())


class ResourcePool:
    def __init__(
        self,
        devices: Union[list[Device], list[GpuSlot]],
        *,
        allocation: str = "fixed",
        on_event: Optional[Callable[..., None]] = None,
        rebalance_interval_s: float = 10.0,
        rebalance_cooldown_s: float = 30.0,
    ):
        devices = list(devices)
        if devices and isinstance(devices[0], GpuSlot):
            devices = devices_from_slots(devices)
        if not devices:
            raise ValueError("a resource pool needs at least one device")
        if allocation not in ALLOCATIONS:
            raise ValueError(f"unknown allocation {allocation!r}")
        self.devices: list[Device] = devices
        self.allocation = allocation
        self._on_event = on_event
        self._queued: dict[int, _Entry] = {}
        self._ids = itertools.count()
        self._started_at: Optional[float] = None
        self._accounted_at: Optional[float] = None
        self._capacity_s: dict[str, float] = {}
        self._busy_s: dict[str, float] = {}
        self._device_busy_s: dict[int, float] = {id(d): 0.0 for d in devices}
        self.run_s: dict[str, Ewma] = {kind: Ewma() for kind in KIND_RANK}
        self._rebalance_interval_s = rebalance_interval_s
        self._rebalance_cooldown_s = rebalance_cooldown_s
        self._last_move = 0.0
        self._rebalancer: Optional[asyncio.Task] = None
        self._closing = False

    # -- compatibility with the slot view -------------------------------------

    @property
    def slots(self) -> list[GpuSlot]:
        return [GpuSlot(d.gpu, d.role, i, d.host) for d in self.devices for i in range(d.capacity)]

    # -- lifecycle -----------------------------------------------------------

    def _emit(self, event: str, **fields: Any) -> None:
        if self._on_event is not None:
            self._on_event(event, **fields)

    async def start(self) -> None:
        if self._started_at is not None:
            return
        self._started_at = self._accounted_at = time.monotonic()
        if self.allocation == "dynamic":
            self._rebalancer = asyncio.create_task(self._rebalance_loop(), name="pool:rebalance")
        self._emit("pool_started", allocation=self.allocation,
                   devices=[{"label": d.label, "gpu": d.gpu, "role": d.role, "capacity": d.capacity} for d in self.devices])

    async def close(self) -> None:
        self._closing = True
        if self._rebalancer is not None:
            self._rebalancer.cancel()
        for entry in list(self._queued.values()):
            entry.future.cancel()
        tasks = [task for d in self.devices for _, task, _ in d.running.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, *(t for t in (self._rebalancer,) if t), return_exceptions=True)

    # -- submission and cancellation -------------------------------------------

    def role_for(self, kind: str) -> str:
        if kind == "build":
            return "builder"
        if self.allocation == "fifo":
            return "shared"
        return "validator" if kind == "validate" else "profiler"

    def has_role(self, role: str) -> bool:
        return any(d.role == role for d in self.devices)

    async def submit(self, job: PoolJob) -> Any:
        """Run ``job`` on a device of its role and return its result.

        Cancelling the caller cancels the job: it is dropped if still queued, and its
        coroutine is cancelled if running.
        """
        if self._started_at is None:
            raise RuntimeError("pool not started (call start())")
        role = self.role_for(job.kind)
        if not self.has_role(role):
            raise RuntimeError(f"pool has no {role} devices for {job.kind} jobs")
        rank = 0 if job.urgent else KIND_RANK[job.kind]
        if self.allocation == "fifo":
            key = (rank, next(self._ids))
        else:
            key = (rank, job.priority, job.order, next(self._ids))
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        entry = _Entry(key, key[-1], job, future, time.monotonic())
        self._queued[entry.id] = entry
        self._emit("job_queued", kind=job.kind, task=job.task_id, epoch=job.epoch, label=job.label)
        self._dispatch()
        try:
            return await future
        finally:
            if self._queued.pop(entry.id, None) is not None:
                self._release(entry.id)
                self._dispatch()

    def cancel(self, task_id: str, before_epoch: int, *, kill_running: bool = True) -> int:
        """Cancel ``task_id``'s non-sticky jobs whose epoch is below ``before_epoch``."""
        def match(job: PoolJob) -> bool:
            return job.task_id == task_id and job.epoch < before_epoch and not job.sticky

        count = self.cancel_where(match, kill_running=kill_running)
        if count:
            self._emit("jobs_cancelled", task=task_id, before_epoch=before_epoch, count=count)
        return count

    def cancel_where(self, predicate: Callable[[PoolJob], bool], *, kill_running: bool = False) -> int:
        count = 0
        for entry in list(self._queued.values()):
            if not entry.future.done() and predicate(entry.job):
                entry.future.cancel()
                count += 1
        if kill_running:
            for device in self.devices:
                for entry, task, _ in list(device.running.values()):
                    if predicate(entry.job):
                        task.cancel()
                        count += 1
        return count

    def load(self, task_id: Optional[str] = None, kinds: Iterable[str] = ("build", "validate", "benchmark")) -> int:
        """Queued plus running jobs of ``kinds`` (of one task, or all)."""
        kinds = set(kinds)
        jobs = [e.job for e in self._queued.values() if not e.future.done()]
        jobs += [e.job for d in self.devices for e, _, _ in d.running.values()]
        return sum(1 for j in jobs if j.kind in kinds and (task_id is None or j.task_id == task_id))

    def queued(self, task_id: Optional[str] = None) -> dict[str, int]:
        """Jobs waiting for a device, by kind (of one task, or all)."""
        counts: dict[str, int] = {}
        for entry in self._queued.values():
            if not entry.future.done() and (task_id is None or entry.job.task_id == task_id):
                counts[entry.job.kind] = counts.get(entry.job.kind, 0) + 1
        return counts

    def exclusive_devices(self) -> int:
        """Devices that run timing and ncu now (dynamic allocation moves them between roles)."""
        role = "shared" if self.allocation == "fifo" else "profiler"
        return sum(1 for d in self.devices if d.role == role)

    # -- dispatching ---------------------------------------------------------

    def _release(self, entry_id: int) -> None:
        for device in self.devices:
            if device.reserved_by == entry_id:
                device.reserved_by = None

    def _ncu_limit(self) -> int:
        exclusive_role = "shared" if self.allocation == "fifo" else "profiler"
        return max(1, sum(1 for d in self.devices if d.role == exclusive_role) - 1)

    def _place(self, entry: _Entry) -> Optional[tuple[Device, int]]:
        job = entry.job
        role = self.role_for(job.kind)
        devices = [d for d in self.devices if d.role == role]
        if not job.needs_device:
            usable = [d for d in devices
                      if d.reserved_by is None and not d.has_exclusive() and len(d.running) < d.capacity]
            if not usable:
                return None
            device = min(usable, key=lambda d: len(d.running) / d.capacity)
            return device, device.free_index()
        if job.kind == "ncu" and sum(
                1 for d in self.devices for e, _, _ in d.running.values() if e.job.kind == "ncu") >= self._ncu_limit():
            return None
        idle = [d for d in devices if not d.running and d.reserved_by in (None, entry.id)]
        if idle:
            device = next((d for d in idle if d.reserved_by == entry.id), idle[0])
            device.reserved_by = None
            return device, 0
        shares_devices = role in ("shared", "validator")
        if shares_devices and not any(d.reserved_by == entry.id for d in devices):
            drainable = [d for d in devices if d.reserved_by is None and not d.has_exclusive()]
            if drainable:
                min(drainable, key=lambda d: len(d.running)).reserved_by = entry.id
        return None

    def _dispatch(self) -> None:
        if self._closing or self._started_at is None:
            return
        for entry in sorted(self._queued.values()):
            if entry.future.done():
                continue
            placement = self._place(entry)
            if placement is not None:
                self._start(entry, *placement)

    def _start(self, entry: _Entry, device: Device, index: int) -> None:
        self._account()
        del self._queued[entry.id]
        self._release(entry.id)
        slot = GpuSlot(device.gpu, device.role, index, device.host)
        concurrency = len(device.running) + 1
        task = asyncio.create_task(self._run(entry, device, slot, concurrency), name=f"pool:{entry.job.label}")
        entry.future.add_done_callback(lambda f, t=task: t.cancel() if f.cancelled() else None)
        device.running[entry.id] = (entry, task, slot)
        self._emit("job_started", kind=entry.job.kind, task=entry.job.task_id, label=entry.job.label,
                   slot=slot.label, concurrency=concurrency, wait_s=round(time.monotonic() - entry.enqueued_at, 3))

    async def _run(self, entry: _Entry, device: Device, slot: GpuSlot, concurrency: int) -> None:
        started = time.monotonic()
        outcome = "done"
        try:
            result = await entry.job.run(slot)
        except asyncio.CancelledError:
            outcome = "cancelled"
            if not entry.future.done():
                entry.future.cancel()
            if self._closing:
                raise
        except Exception as e:  # noqa: BLE001 - delivered to the submitter
            outcome = "error"
            if not entry.future.done():
                entry.future.set_exception(e)
        else:
            if isinstance(result, dict):
                result.setdefault("wait_s", round(started - entry.enqueued_at, 3))
                result.setdefault("concurrency", concurrency)
            if not entry.future.done():
                entry.future.set_result(result)
        finally:
            self._account()
            device.running.pop(entry.id, None)
            seconds = time.monotonic() - started
            self._device_busy_s[id(device)] = self._device_busy_s.get(id(device), 0.0) + seconds
            if outcome == "done":
                self.run_s[entry.job.kind].add(seconds)
            self._emit("job_finished", kind=entry.job.kind, task=entry.job.task_id, label=entry.job.label,
                       slot=slot.label, outcome=outcome, seconds=round(seconds, 3))
            self._dispatch()

    # -- dynamic allocation ----------------------------------------------------

    def expected_run_s(self, kind: str) -> float:
        value = self.run_s[kind].value
        return PRIOR_RUN_S[kind] if value is None else value

    async def _rebalance_loop(self) -> None:
        while True:
            await asyncio.sleep(self._rebalance_interval_s)
            self.rebalance()

    def rebalance(self, now: Optional[float] = None) -> Optional[tuple[str, str, str]]:
        """Move one idle GPU between the validator and profiler roles toward the load ratio."""
        now = time.monotonic() if now is None else now
        gpu_roles: dict[str, set[str]] = {}
        for d in self.devices:
            if d.gpu is not None and d.role in ("validator", "profiler"):
                gpu_roles.setdefault(d.gpu, set()).add(d.role)
        if any(len(roles) > 1 for roles in gpu_roles.values()) or len(gpu_roles) < 3:
            return None
        if now - self._last_move < self._rebalance_cooldown_s:
            return None
        validators = [d for d in self.devices if d.role == "validator" and d.gpu is not None]
        profilers = [d for d in self.devices if d.role == "profiler" and d.gpu is not None]
        vpg = max(d.capacity for d in validators)
        load_v = self.load(kinds=("validate",)) * self.expected_run_s("validate") / vpg
        load_p = self.load(kinds=("benchmark",)) * self.expected_run_s("benchmark")
        if load_v + load_p < MIN_REBALANCE_LOAD_S:  # too little work to tell which role needs the GPU
            return None
        n = len(validators) + len(profilers)
        target = min(n - 1, max(1, round(n * load_p / (load_v + load_p))))
        if target > len(profilers):
            source, role, capacity = validators, "profiler", 1
        elif target < len(profilers):
            source, role, capacity = profilers, "validator", vpg
        else:
            return None
        idle = [d for d in source if not d.running and d.reserved_by is None]
        if not idle:
            return None
        device = idle[-1]
        self._account()
        old = device.role
        device.role, device.capacity = role, capacity
        self._last_move = now
        step = 1 if role == "profiler" else -1
        self._emit("gpu_role_changed", gpu=device.gpu, old=old, new=role, load_validate_s=round(load_v, 1),
                   load_benchmark_s=round(load_p, 1), validators=len(validators) - step,
                   profilers=len(profilers) + step)
        log.info("GPU %s: %s -> %s (validation load %.0fs, timing load %.0fs)", device.gpu, old, role, load_v, load_p)
        self._dispatch()
        return device.gpu, old, role

    # -- statistics ------------------------------------------------------------

    def _account(self) -> None:
        now = time.monotonic()
        if self._accounted_at is None:
            return
        dt = now - self._accounted_at
        self._accounted_at = now
        if dt <= 0:
            return
        for d in self.devices:
            self._capacity_s[d.role] = self._capacity_s.get(d.role, 0.0) + d.capacity * dt
            self._busy_s[d.role] = self._busy_s.get(d.role, 0.0) + len(d.running) * dt

    def stats(self) -> PoolStats:
        self._account()
        roles = sorted({d.role for d in self.devices})
        slots = {r: sum(d.capacity for d in self.devices if d.role == r) for r in roles}
        busy = {r: sum(len(d.running) for d in self.devices if d.role == r) for r in roles}
        queued = {r: 0 for r in roles}
        for entry in self._queued.values():
            if not entry.future.done():
                role = self.role_for(entry.job.kind)
                queued[role] = queued.get(role, 0) + 1
        utilization = {r: round(self._busy_s.get(r, 0.0) / self._capacity_s[r], 4) if self._capacity_s.get(r) else 0.0
                       for r in roles}
        slot_busy: dict[str, float] = {}  # finished jobs only
        for d in self.devices:
            slot_busy[d.label] = round(slot_busy.get(d.label, 0.0) + self._device_busy_s.get(id(d), 0.0), 3)
        devices = [{"label": d.label, "gpu": d.gpu, "role": d.role, "capacity": d.capacity,
                    "running": len(d.running), "reserved": d.reserved_by is not None} for d in self.devices]
        return PoolStats(slots=slots, busy=busy, queued=queued, utilization=utilization, slot_busy_s=slot_busy,
                         devices=devices)
