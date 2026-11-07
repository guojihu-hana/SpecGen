import asyncio

import pytest

from specgen.resources.pool import PRIORITY_SPEC, Device, PoolJob, ResourcePool, fixed_slots, make_devices


def test_fixed_slots_split():
    slots = fixed_slots(["0", "1", "2", "3"], validations_per_gpu=2)
    assert [(s.gpu, s.role) for s in slots] == [
        ("0", "validator"), ("0", "validator"), ("1", "validator"), ("1", "validator"),
        ("2", "profiler"), ("3", "profiler"),
    ]
    assert [s.gpu for s in fixed_slots(["0", "1", "2"], num_validator=2) if s.role == "profiler"] == ["2"]
    assert [s.role for s in fixed_slots([], validations_per_gpu=3)] == ["validator"] * 3 + ["profiler"]
    with pytest.raises(ValueError):
        fixed_slots(["0", "1", "2", "3"], num_validator=3, num_profiler=2)


def run(coro):
    return asyncio.run(coro)


def test_profiler_queue_order():
    async def scenario():
        pool = ResourcePool(fixed_slots(["0", "1"], validations_per_gpu=1))
        await pool.start()
        gate, order = asyncio.Event(), []

        def job(kind, name, priority=0, wait=False):
            async def body(slot):
                if wait:
                    await gate.wait()
                order.append(name)
                return {"name": name}
            return pool.submit(PoolJob(kind, "t", 0, body, priority=priority))

        blocker = asyncio.create_task(job("benchmark", "blocker", wait=True))
        await asyncio.sleep(0.01)
        rest = [asyncio.create_task(job("ncu", "ncu")),
                asyncio.create_task(job("benchmark", "spec", priority=PRIORITY_SPEC)),
                asyncio.create_task(job("benchmark", "main"))]
        await asyncio.sleep(0.01)
        gate.set()
        results = await asyncio.gather(blocker, *rest)
        await pool.close()
        return order, results

    order, results = run(scenario())
    assert order == ["blocker", "main", "spec", "ncu"]
    assert all("wait_s" in r for r in results)


def test_validator_concurrency_per_gpu():
    async def scenario():
        pool = ResourcePool(fixed_slots(["0", "1"], num_validator=1, validations_per_gpu=3))
        await pool.start()
        running, peak = 0, 0

        async def body(slot):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.05)
            running -= 1
            return slot.label

        labels = await asyncio.gather(*(pool.submit(PoolJob("validate", "t", 0, body)) for _ in range(6)))
        await pool.close()
        return peak, labels

    peak, labels = run(scenario())
    assert peak == 3
    assert set(labels) == {"validator:0#0", "validator:0#1", "validator:0#2"}


def test_cancel_by_epoch_drops_queued_and_kills_running():
    async def scenario():
        pool = ResourcePool(fixed_slots(["0", "1"], validations_per_gpu=1))
        await pool.start()
        killed = asyncio.Event()

        async def blocking(slot):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                killed.set()
                raise

        async def quick(slot):
            return "ran"

        running = asyncio.create_task(pool.submit(PoolJob("benchmark", "t", 0, blocking)))
        await asyncio.sleep(0.01)
        stale = asyncio.create_task(pool.submit(PoolJob("benchmark", "t", 0, quick)))
        other_task = asyncio.create_task(pool.submit(PoolJob("benchmark", "u", 0, quick)))
        fresh = asyncio.create_task(pool.submit(PoolJob("benchmark", "t", 1, quick)))
        await asyncio.sleep(0.01)
        assert pool.stats().queued["profiler"] == 3
        cancelled = pool.cancel("t", before_epoch=1)
        outcomes = await asyncio.gather(running, stale, other_task, fresh, return_exceptions=True)
        await pool.close()
        return cancelled, outcomes, killed.is_set()

    cancelled, outcomes, killed = run(scenario())
    assert cancelled == 2 and killed
    assert isinstance(outcomes[0], asyncio.CancelledError) and isinstance(outcomes[1], asyncio.CancelledError)
    assert outcomes[2:] == ["ran", "ran"]


def test_cancelling_the_submitter_cancels_the_job():
    async def scenario():
        pool = ResourcePool(fixed_slots([], validations_per_gpu=1))
        await pool.start()
        started, killed = asyncio.Event(), asyncio.Event()

        async def body(slot):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                killed.set()
                raise

        submitter = asyncio.create_task(pool.submit(PoolJob("validate", "t", 0, body)))
        await started.wait()
        submitter.cancel()
        await asyncio.gather(submitter, return_exceptions=True)
        await asyncio.wait_for(killed.wait(), 1)
        await asyncio.sleep(0.01)  # let the worker observe the cancelled job
        stats = pool.stats()
        await pool.close()
        return stats

    stats = run(scenario())
    assert stats.busy["validator"] == 0 and stats.slots == {"profiler": 1, "validator": 1}
    assert stats.utilization["validator"] > 0


def test_make_devices():
    fixed = make_devices(["0", "1", "2"], validations_per_gpu=3, build_workers=4)
    assert [(d.gpu, d.role, d.capacity) for d in fixed] == [
        ("0", "validator", 3), ("1", "validator", 3), ("2", "profiler", 1), (None, "builder", 4)]
    fifo = make_devices(["0", "1"], allocation="fifo", validations_per_gpu=2)
    assert [(d.gpu, d.role, d.capacity) for d in fifo] == [("0", "shared", 2), ("1", "shared", 2)]
    assert [(d.gpu, d.role) for d in make_devices([], allocation="fifo")] == [(None, "shared")]


class Gate:
    """Jobs that record their order and block until released by name."""

    def __init__(self, pool):
        self.pool, self.order, self.events, self.slots = pool, [], {}, {}

    def submit(self, kind, name, **kwargs):
        self.events[name] = asyncio.Event()

        async def body(slot):
            self.order.append(name)
            self.slots[name] = slot
            await self.events[name].wait()
            return {"name": name}

        return asyncio.create_task(self.pool.submit(PoolJob(kind, kwargs.pop("task", "t"), 0, body, **kwargs)))

    def release(self, name):
        self.events[name].set()


def test_fifo_exclusive_job_reserves_a_device():
    async def scenario():
        pool = ResourcePool([Device("0", "shared", 2)], allocation="fifo")
        await pool.start()
        gate = Gate(pool)
        v1 = gate.submit("validate", "v1")
        await asyncio.sleep(0.01)
        bench = gate.submit("benchmark", "bench")
        v2 = gate.submit("validate", "v2")
        await asyncio.sleep(0.01)
        assert gate.order == ["v1"]  # the device is reserved for the benchmark: v2 may not start
        assert pool.stats().devices[0]["reserved"]
        gate.release("v1")
        await asyncio.sleep(0.01)
        assert gate.order == ["v1", "bench"]
        gate.release("bench")
        gate.release("v2")
        await asyncio.gather(v1, bench, v2)
        await pool.close()
        return gate.order

    assert run(scenario()) == ["v1", "bench", "v2"]


def test_exclusive_validation_drains_a_validator():
    async def scenario():
        pool = ResourcePool([Device("0", "validator", 3), Device("1", "profiler", 1)])
        await pool.start()
        gate = Gate(pool)
        shared = gate.submit("validate", "shared")
        await asyncio.sleep(0.01)
        alone = gate.submit("validate", "alone", exclusive=True)
        await asyncio.sleep(0.01)
        assert gate.order == ["shared"]
        gate.release("shared")
        await asyncio.sleep(0.01)
        late = gate.submit("validate", "late")
        await asyncio.sleep(0.01)
        assert gate.order == ["shared", "alone"]  # nothing joins an exclusive job
        gate.release("alone")
        gate.release("late")
        results = await asyncio.gather(shared, alone, late)
        await pool.close()
        return gate.order, results

    order, results = run(scenario())
    assert order == ["shared", "alone", "late"]
    assert results[0]["concurrency"] == 1


def test_builder_role_runs_builds_on_cpu():
    async def scenario():
        pool = ResourcePool(make_devices(["0", "1"], build_workers=2))
        await pool.start()
        gate = Gate(pool)
        builds = [gate.submit("build", f"b{i}") for i in range(3)]
        await asyncio.sleep(0.01)
        running = list(gate.order)
        for i in range(3):
            gate.release(f"b{i}")
        await asyncio.gather(*builds)
        await pool.close()
        return running, gate.slots

    running, slots = run(scenario())
    assert running == ["b0", "b1"]
    assert {(s.gpu, s.role) for s in slots.values()} == {(None, "builder")}


def test_ncu_leaves_a_profiler_for_timing():
    async def scenario():
        pool = ResourcePool(make_devices(["0", "1", "2", "3"], num_validator=1))
        await pool.start()
        gate = Gate(pool)
        jobs = [gate.submit("ncu", f"n{i}") for i in range(3)]
        await asyncio.sleep(0.01)
        running = list(gate.order)
        bench = gate.submit("benchmark", "bench")
        await asyncio.sleep(0.01)
        after = list(gate.order)
        for name in ("n0", "n1", "n2", "bench"):
            gate.release(name)
        await asyncio.gather(*jobs, bench)
        await pool.close()
        return running, after

    running, after = run(scenario())
    assert running == ["n0", "n1"]  # 3 profilers: at most 2 ncu jobs
    assert after == ["n0", "n1", "bench"]


def test_dynamic_rebalance_moves_an_idle_gpu():
    async def scenario():
        pool = ResourcePool(make_devices(["0", "1", "2"], num_validator=2, validations_per_gpu=2),
                            allocation="dynamic", rebalance_interval_s=3600)
        await pool.start()
        gate = Gate(pool)
        first = gate.submit("benchmark", "b0")
        await asyncio.sleep(0.01)
        assert pool.rebalance() is None  # 15 s of work is too little to move a GPU
        benches = [first] + [gate.submit("benchmark", f"b{i}") for i in range(1, 6)]
        await asyncio.sleep(0.01)
        assert gate.order == ["b0"]
        moved = pool.rebalance()
        await asyncio.sleep(0.01)
        again = pool.rebalance()  # cooldown
        roles = sorted(d.role for d in pool.devices)
        for i in range(6):
            gate.release(f"b{i}")
        await asyncio.gather(*benches)
        await pool.close()
        return moved, again, roles, gate.order

    moved, again, roles, order = run(scenario())
    assert moved[1:] == ("validator", "profiler") and again is None
    assert roles == ["profiler", "profiler", "validator"]
    assert order[:2] == ["b0", "b1"]


def test_sticky_jobs_survive_epoch_cancel_and_load_counts():
    async def scenario():
        pool = ResourcePool(fixed_slots(["0", "1"], validations_per_gpu=1))
        await pool.start()
        gate = Gate(pool)
        sticky = gate.submit("ncu", "sticky", sticky=True)
        plain = gate.submit("validate", "plain")
        other = gate.submit("validate", "other", task="u")
        await asyncio.sleep(0.01)
        load = pool.load("t"), pool.load(kinds=("validate",))
        cancelled = pool.cancel("t", before_epoch=1)
        gate.release("sticky")
        gate.release("other")
        results = await asyncio.gather(sticky, plain, other, return_exceptions=True)
        await pool.close()
        return load, cancelled, results

    load, cancelled, results = run(scenario())
    assert load == (1, 2) and cancelled == 1
    assert results[0] == {"name": "sticky", "wait_s": results[0]["wait_s"], "concurrency": 1}
    assert isinstance(results[1], asyncio.CancelledError) and results[2]["name"] == "other"
