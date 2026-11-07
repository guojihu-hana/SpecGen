import asyncio
import json
import re
import time

import pytest
from conftest import CORRECT_RELU, ModeBackend, make_context, relu_kernel, spec_answer

from specgen.config import TerminationConfig, parse_args
from specgen.control.history import KernelOutcome, TaskHistory
from specgen.control.loop import IterationLoop
from specgen.control.termination import TerminationCriterion
from specgen.control.triggers import TriggerConfig, TriggerGroup
from specgen.llm.events import ContentDelta, ReasoningDelta, StreamEnd
from specgen.tasks.base import KernelBenchTask


def outcome(kernel_id, iteration, speedup):
    return KernelOutcome(kernel_id, iteration, "main", "code", status="success", valid=True, speedup=speedup)


def test_criteria_baselines():
    history = TaskHistory()
    best, hist_avg, first = (TerminationCriterion(n) for n in ("best", "hist_avg", "first_valid"))
    assert best.satisfied(1.01, history) == (True, 1.0) and best.satisfied(0.99, history) == (False, 1.0)
    assert first.satisfied(0.1, history) == (True, None) and not first.satisfied(None, history)[0]
    a, b = outcome("a", 0, 2.0), outcome("b", 1, 1.0)
    history.add(a), history.add(b)
    history.select(0, a), history.select(1, b)
    assert best.satisfied(2.5, history) == (True, 2.0) and not best.satisfied(1.9, history)[0]
    assert hist_avg.satisfied(1.6, history) == (True, 1.5) and not hist_avg.satisfied(1.5, history)[0]
    assert not TerminationCriterion("best", margin=0.3).satisfied(2.5, history)[0]
    with pytest.raises(ValueError):
        TerminationCriterion("fastest")


def test_best_never_ends_an_iteration_below_the_reference():
    history = TaskHistory()
    slow = outcome("slow", 0, 0.41)
    history.add(slow), history.select(0, slow)
    best = TerminationCriterion("best", margin=0.05)
    assert best.satisfied(0.6, history) == (False, 1.0)  # beats 0.41 by far, still slower than PyTorch
    assert best.satisfied(1.06, history) == (True, 1.0)


def test_termination_flags(relu_task):
    base = ["--task", str(relu_task), "--gpus", "none"]
    cfg = parse_args(base + ["--enable-termination", "--termination-criteria", "best", "--termination-margin", "0.1",
                             "--termination-confirm", "2"])
    assert cfg.termination.enabled and cfg.termination.criteria == "best" and cfg.termination.margin == 0.1
    assert cfg.termination.grace_ratio == 0.05 and cfg.termination.confirm == 2
    defaults = parse_args(base)
    assert (defaults.termination.margin, defaults.termination.confirm) == (0.03, 2)
    assert defaults.spec.k_max == 4 and defaults.eval.shift_trial
    assert not parse_args(base + ["--no-validation-shift-trial"]).eval.shift_trial
    for bad in (["--enable-termination"], ["--termination-criteria", "best"], ["--grace-ratio", "-1"],
                ["--termination-confirm", "0"]):
        with pytest.raises(SystemExit):
            parse_args(base + bad)


TRIGGERS = TriggerConfig(groups=(TriggerGroup("kw", (re.compile("tile size"),)),), after_tokens=0,
                         fallback_tokens=10**6, min_gap_tokens=0)


def run_iteration(ctx, task_path):
    async def go():
        await ctx.pool.start()
        loop = IterationLoop(KernelBenchTask.from_path(task_path), ctx)
        try:
            t0 = time.monotonic()
            await loop.run_iteration(0)
            load = ctx.pool.load()
            return loop, time.monotonic() - t0, load
        finally:
            await ctx.pool.close()
            ctx.recorder.close()

    return asyncio.run(go())


def iteration_record(ctx):
    return json.loads((ctx.layout.run_dir / "iterations.jsonl").read_text().splitlines()[0])


def test_a_terminating_spec_kernel_aborts_the_main_generation(tmp_path, relu_task):
    main = [ReasoningDelta("think about the tile size"), 120, ContentDelta("never"), StreamEnd("stop")]
    backend = ModeBackend(main=[main], spec=[spec_answer(CORRECT_RELU)])
    ctx = make_context(tmp_path / "out", backend, triggers=TRIGGERS, k=1, tasks=[relu_task],
                       termination=TerminationConfig(enabled=True, criteria="first_valid", grace_ratio=0.05))
    loop, elapsed, load = run_iteration(ctx, relu_task)
    assert elapsed < 60 and load == 0
    record = iteration_record(ctx)
    assert record["end_reason"] == "terminated" and record["selected"] == "i000-t01-s0"
    assert record["main_status"] == "cancelled" and record["termination"]["criterion"] == "first_valid"
    assert loop.history.selected[0].kernel_id == "i000-t01-s0" and loop.epoch == 1
    events = [json.loads(line)["type"] for line in (ctx.layout.run_dir / "events.jsonl").read_text().splitlines()]
    assert "iteration_terminated" in events and "main_finished" not in events


class TimedEvaluator:
    """Every build and validation passes; each benchmark of a kernel returns its next scripted speedup."""

    def __init__(self, speedups):
        self.speedups = {k: list(v) for k, v in speedups.items()}
        self.benchmarks = []

    async def run(self, job, slot):
        if job.kind != "benchmark":
            return {"ok": True, "status": "success"}
        kernel = job.work_dir.rstrip("/").split("/")[-1]
        self.benchmarks.append((kernel, job.stem))
        speedup = self.speedups[kernel].pop(0)
        return {"ok": True, "status": "success", "speedup": speedup, "latency_ms": 1.0 / speedup, "ref_latency_ms": 1.0}


@pytest.mark.parametrize("retimed, terminated", [(1.02, False), (1.04, True)])
def test_a_criterion_met_once_is_confirmed_by_a_second_timing(tmp_path, relu_task, retimed, terminated):
    main = [ReasoningDelta("think about the tile size"), 1.5, ContentDelta(f"```python\n{CORRECT_RELU}\n```"),
            StreamEnd("stop")]
    evaluator = TimedEvaluator({"i000-t01-s0": [1.05, retimed], "i000-main": [1.0]})
    ctx = make_context(tmp_path / "out", ModeBackend(main=[main], spec=[spec_answer(relu_kernel("x.clamp(min=0)"))]),
                       triggers=TRIGGERS, k=1, tasks=[relu_task], evaluator=evaluator,
                       termination=TerminationConfig(enabled=True, criteria="best", margin=0.03, confirm=2,
                                                     grace_ratio=None))
    run_iteration(ctx, relu_task)
    record = iteration_record(ctx)
    assert ("i000-t01-s0", "benchmark") in evaluator.benchmarks
    assert ("i000-t01-s0", "benchmark_confirm1") in evaluator.benchmarks
    assert (record["end_reason"] == "terminated") == terminated
    assert record["selected"] == "i000-t01-s0" and record["speedup"] == pytest.approx((1.05 + retimed) / 2)
    if terminated:
        assert record["termination"]["timings"] == [1.05, 1.04] and record["main_status"] == "cancelled"
    kernel_dir = ctx.layout.kernel_dir(KernelBenchTask.from_path(relu_task), 0, "i000-t01-s0")
    assert (kernel_dir / "benchmark.json").is_file() and (kernel_dir / "benchmark_confirm1.json").is_file()


def test_grace_period_cuts_slow_candidates(tmp_path, relu_task):
    main = [ReasoningDelta("the tile size"), 1.0, ContentDelta(f"```python\n{CORRECT_RELU}\n```"), StreamEnd("stop")]
    slow_spec = [120, *spec_answer(relu_kernel("torch.clamp_min(x, 0.0) * 1"))]
    backend = ModeBackend(main=[main], spec=[slow_spec])
    ctx = make_context(tmp_path / "out", backend, triggers=TRIGGERS, k=1, tasks=[relu_task],
                       termination=TerminationConfig(grace_ratio=0.5))
    loop, elapsed, _ = run_iteration(ctx, relu_task)
    record = iteration_record(ctx)
    assert record["end_reason"] == "grace_expired" and record["unfinished_spec_tasks"] == 1
    assert record["selected"] == "i000-main" and elapsed < 60


def test_without_grace_every_candidate_finishes(tmp_path, relu_task):
    main = [ReasoningDelta("the tile size"), ContentDelta(f"```python\n{CORRECT_RELU}\n```"), StreamEnd("stop")]
    spec = [0.5, *spec_answer(relu_kernel("torch.clamp_min(x, 0.0) * 1"))]
    ctx = make_context(tmp_path / "out", ModeBackend(main=[main], spec=[spec]), triggers=TRIGGERS, k=1,
                       tasks=[relu_task])
    loop, _, _ = run_iteration(ctx, relu_task)
    record = iteration_record(ctx)
    assert record["end_reason"] == "all_finished" and record["statuses"] == {"success": 2}
