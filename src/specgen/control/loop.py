"""Iteration loop of one task.

Each iteration streams the main generation and launches speculative generations
from prefixes of its reasoning: one at a time whenever the kernels expected at the
profiler fall short, each seeing the test results of the earlier ones
(``control.pacing``), or k at each keyword trigger.  Every candidate kernel (main
or spec) is evaluated as soon as it exists: build on a CPU worker, validation,
then timing.  The fastest benchmarked kernels of the iteration are profiled with
ncu in the background when a profiler is free.  The iteration selects the fastest
benchmarked candidate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from specgen.control.attempts import family
from specgen.control.compliance import CATEGORY_RUNTIME, forbidden_usage, library_kernels
from specgen.control.context import RunContext
from specgen.control.history import KernelOutcome, TaskHistory
from specgen.control.pacing import PassRate, draft_notes, launch_decision
from specgen.control.spec import PacedSpecController, SpecController, code_fingerprint
from specgen.control.termination import TerminationCriterion
from specgen.control.triggers import Trigger, TriggerMonitor
from specgen.evaluation.jobs import EvalJob
from specgen.evaluation.ncu import short_kernel_name
from specgen.generation.extract import has_code
from specgen.generation.generator import PREFILL_CODE_START, GenerationResult, module_from_output
from specgen.generation.prompts import Prompt
from specgen.resources.pool import PRIORITY_MAIN, PRIORITY_SPEC, PoolJob
from specgen.storage.layout import StreamFiles, read_records, write_json
from specgen.tasks.base import Task

log = logging.getLogger(__name__)

_RESULT_SUMMARY_KEYS = (
    "ok", "status", "exec_s", "wait_s", "gpu", "slot", "concurrency", "compile_s", "run_s", "trials",
    "max_abs_diff", "max_rel_err", "shift_trial", "weights_synced", "latency_ms", "ref_latency_ms", "speedup",
    "ncu_s", "returncode",
)


# Pool orders within a priority (lower first): a kernel's confirmation timings go ahead of every
# timing, then the ncu profile spec launches wait for, then demand-paced specs newest first (-seq).
CONFIRM_ORDER = -1e12
URGENT_NCU_ORDER = -1e11
UNFINISHED = ("pending", "evaluating", "cancelled", "grace_truncated")  # cut before a verdict


def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {k: result[k] for k in _RESULT_SUMMARY_KEYS if k in result}


def _error_text(result: Mapping[str, Any]) -> str:
    parts = [str(result.get("error") or "")]
    if result.get("details"):
        parts.append(json.dumps(result["details"], sort_keys=True, default=str))
    if not result.get("error") and result.get("log_tail"):
        parts.append(str(result["log_tail"]))
    return "\n".join(p for p in parts if p).strip()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _recorded_code(kernel_dir: Path, main_dir: Path, source: str) -> str:
    """A kernel's code: ``kernel.py``, or for one stopped before evaluation (a static library call, a duplicate)
    the module in the stream it came from, extracted as the generation did."""
    path = kernel_dir / "kernel.py"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    stream = main_dir if source == "main" else kernel_dir
    read = lambda name: (stream / name).read_text(encoding="utf-8") if (stream / name).is_file() else ""  # noqa: E731
    content = read("content.txt")
    code, _ = module_from_output(content, read("reasoning.txt") if source == "main" else "",
                                 prefilled=source != "main" and content.startswith(PREFILL_CODE_START))
    return code or ""


def _restored_error(record: Mapping[str, Any], kernel_dir: Path, code: str) -> Optional[str]:
    """A finished kernel's error message as the run built it: from its status, or the result of its failed job."""
    status = record.get("status")
    if status in ("success", *UNFINISHED, None):
        return None
    if status == "llm_error":
        return (record.get("generation") or {}).get("error")
    if status == "parse_error":
        return "the module has only comments, no code" if code else "no ```python fence found in the output"
    if status == "duplicate":
        return f"same code as {record.get('duplicate_of')}"
    for stem in ("build", "validate", "benchmark"):
        result = _read_json(kernel_dir / f"{stem}.json")
        if result and not result.get("ok"):
            return _error_text(result)
    if status == "forbidden":
        libraries = library_kernels(k.get("name", "") for k in record.get("forward_kernels") or ())
        return forbidden_usage(code) or (f"{CATEGORY_RUNTIME}: {', '.join(short_kernel_name(n) for n in libraries[:3])}"
                                         if libraries else None)
    return None


def _out_of_memory(result: Mapping[str, Any]) -> bool:
    text = f"{result.get('error') or ''}\n{result.get('log_tail') or ''}".lower()
    return result.get("status") in ("runtime_error", "cuda_error") and "out of memory" in text


@dataclass
class Candidate:
    kernel_id: str
    source: str  # "main" or "spec"
    outcome: KernelOutcome
    record: dict[str, Any]
    kernel_dir: Path
    seq: int = 0  # launch order of a spec; demand-paced specs are evaluated newest first
    upstream: bool = False  # counted in IterationState.upstream until it reaches timing or fails


@dataclass(eq=False)
class SpecItem:
    """One spec generation: waits for a generation slot, generates, then evaluates its kernel."""

    kernel_id: str
    task: Optional[asyncio.Task] = None
    started: bool = False  # holds a generation slot (its LLM request is running or done)
    released: bool = False  # its in-flight reservation is returned
    superseded: bool = False  # replaced by a later trigger before it started


@dataclass
class IterationState:
    iteration: int
    epoch: int
    prompt: Prompt
    started: float = field(default_factory=time.monotonic)
    candidates: list[Candidate] = field(default_factory=list)
    spec_tasks: list[asyncio.Task] = field(default_factory=list)
    spec_items: list[SpecItem] = field(default_factory=list)
    triggers: list[dict[str, Any]] = field(default_factory=list)
    main: Optional[GenerationResult] = None
    terminated: asyncio.Event = field(default_factory=asyncio.Event)
    terminated_by: Optional[Candidate] = None
    termination: dict[str, Any] = field(default_factory=dict)
    # demand-paced specs (control.pacing)
    launches: int = 0
    last_launch_tokens: Optional[int] = None
    upstream: int = 0  # specs generating, building or validating
    pause_kernel: Optional[str] = None  # launches wait for this kernel's ncu profile ...
    pause_started: float = 0.0
    pause_until: float = 0.0  # ... until then at most
    paused_s: float = 0.0


class MainSink:
    """Main generation sink: stream files, the spec controller and progress events."""

    PROGRESS_TOKENS = 1000

    def __init__(self, files: StreamFiles, controller: Optional[SpecController], progress: Callable[[int], None]):
        self.files = files
        self.controller = controller
        self.progress = progress
        self._next_progress = self.PROGRESS_TOKENS

    def on_reasoning(self, text: str, total_tokens: int) -> None:
        self.files.on_reasoning(text, total_tokens)
        if self.controller is not None:
            self.controller.on_reasoning(text, total_tokens)
        if total_tokens >= self._next_progress:
            self._next_progress = (total_tokens // self.PROGRESS_TOKENS + 1) * self.PROGRESS_TOKENS
            self.progress(total_tokens)

    def on_content(self, text: str, total_tokens: int) -> None:
        self.files.on_content(text, total_tokens)
        if self.controller is not None:
            self.controller.on_content(text, total_tokens)


class IterationLoop:
    def __init__(self, task: Task, ctx: RunContext):
        self.task = task
        self.ctx = ctx
        self.history = TaskHistory()
        self.epoch = 0
        self._fingerprints: dict[str, str] = {}  # code fingerprint -> first kernel id
        self._ncu_tasks: dict[str, asyncio.Task] = {}  # kernel id -> background ncu job
        self._demand = ctx.triggers is not None and ctx.cfg.spec.schedule == "demand"
        self.pass_rate = PassRate()  # this task's specs that reached timing
        self.start_iteration = 0  # later when an earlier run is continued (restore)
        term = ctx.cfg.termination
        # the baseline (no spec generation) evaluates one kernel per iteration: nothing to end early
        self.criterion = (TerminationCriterion(term.criteria, term.margin)
                          if term.enabled and term.criteria and ctx.triggers is not None else None)

    async def run(self) -> TaskHistory:
        try:
            for iteration in range(self.start_iteration, self.ctx.cfg.run.max_iterations):
                await self.run_iteration(iteration)
            if self.ctx.cfg.run.max_iterations > 0:  # the final best kernel gets its ncu analysis too
                await self._await_ncu(self.ctx.cfg.run.max_iterations)
        finally:
            pending = list(self._ncu_tasks.values())
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        return self.history

    def _event(self, type: str, state: IterationState, **fields: Any) -> None:
        self.ctx.recorder.event(type, task=self.task.task_id, iteration=state.iteration, **fields)

    def restore(self) -> int:
        """Continue an earlier run of this task in the same run directory: rebuild the history (code, timings, ncu
        profiles, errors, the kernel each iteration selected) from its records, and return the first iteration
        still to run.  An iteration that did not finish runs again."""
        run_dir = self.ctx.layout.run_dir
        finished = {r["iteration"]: r for r in read_records(run_dir / "iterations.jsonl")
                    if r.get("task") == self.task.task_id}
        start = max(finished) + 1 if finished else 0
        restored: dict[str, KernelOutcome] = {}
        for r in read_records(run_dir / "kernels.jsonl"):
            if r.get("task_id") != self.task.task_id or not isinstance(r.get("iteration"), int) or r["iteration"] >= start:
                continue
            kernel_dir = self.ctx.layout.kernel_dir(self.task, r["iteration"], r["kernel_id"])
            code = _recorded_code(kernel_dir, self.ctx.layout.iteration_dir(self.task, r["iteration"]) / "main",
                                  r.get("source", "spec"))
            bench = r.get("benchmark") or {}
            outcome = KernelOutcome(kernel_id=r["kernel_id"], iteration=r["iteration"], source=r.get("source", "spec"),
                                    code=code, status=r.get("status") or "cancelled", valid=bool(r.get("valid")),
                                    speedup=r.get("speedup"), latency_ms=bench.get("latency_ms"),
                                    ref_latency_ms=bench.get("ref_latency_ms"), error=_restored_error(r, kernel_dir, code))
            ncu = _read_json(kernel_dir / "ncu.json")
            if ncu.get("ok"):
                outcome.ncu_profile = ncu.get("profile")
            self.history.add(outcome)
            restored[outcome.kernel_id] = outcome
            if code:
                self._fingerprints.setdefault(code_fingerprint(code), outcome.kernel_id)
            if outcome.source == "spec" and outcome.status not in UNFINISHED:
                self.pass_rate.add(outcome.valid)
        for iteration, r in finished.items():
            if r.get("selected") in restored:
                self.history.select(iteration, restored[r["selected"]])
        self.start_iteration = self.epoch = start
        best = self.history.best()
        log.info("%s: continuing at iteration %d with %d earlier kernels (best %s)", self.task.task_id, start,
                 len(restored), f"{best.speedup:.3f}x {best.kernel_id}" if best else "none")
        return start

    async def run_iteration(self, iteration: int) -> Optional[KernelOutcome]:
        ctx, task = self.ctx, self.task
        run = ctx.cfg.run
        if iteration > 0:
            await self._await_ncu(iteration)
        prompt = ctx.prompts.build(task, self.history.prompt_context(
            iteration, run.feedback_top_k, mode=run.history_prompt, max_approaches=run.summary_max_approaches,
            max_failures=run.summary_max_failures, max_chars=run.summary_max_chars))
        write_json(ctx.layout.iteration_dir(task, iteration) / "prompt.json", prompt.to_dict())
        state = IterationState(iteration, self.epoch, prompt)
        self._event("iteration_started", state, epoch=self.epoch)
        main = asyncio.create_task(self._main(state), name=f"{task.task_id}:i{iteration}:main")
        try:
            end_reason = await self._wait_for_end(state, main)
        finally:
            leftovers = [t for t in (main, *state.spec_tasks) if not t.done()]
            for t in leftovers:
                t.cancel()
            await asyncio.gather(*leftovers, return_exceptions=True)
        for t in state.spec_tasks:
            if not t.cancelled() and t.exception() is not None:
                log.error("%s: spec task failed: %r", task.task_id, t.exception())
        return self._finish(state, end_reason, unfinished=sum(1 for t in leftovers if t is not main))

    async def _wait_for_end(self, state: IterationState, main: asyncio.Task) -> str:
        """Until a kernel meets the termination criterion, or the main kernel is evaluated and the
        other candidates finished or used up the grace period (grace_ratio x main generation time)."""
        terminated = asyncio.create_task(state.terminated.wait())
        try:
            await asyncio.wait({main, terminated}, return_when=asyncio.FIRST_COMPLETED)
            if state.terminated.is_set():
                return "terminated"
            main.result()  # an internal error ends the task
            ratio = self.ctx.cfg.termination.grace_ratio
            deadline = None if ratio is None else time.monotonic() + ratio * (state.main.seconds if state.main else 0.0)
            while True:
                pending = {t for t in state.spec_tasks if not t.done()}
                if not pending:
                    return "all_finished"
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return "grace_expired"
                await asyncio.wait({*pending, terminated}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                if state.terminated.is_set():
                    return "terminated"
        finally:
            terminated.cancel()

    async def _check_termination(self, state: IterationState, candidate: Candidate, priority: int,
                                 bench_params: dict[str, Any]) -> None:
        """End the iteration when the kernel meets the criterion; with ``confirm`` > 1 it is timed again
        and every timing must meet it.  A re-timed kernel keeps the mean of its timings."""
        if self.criterion is None or state.terminated.is_set():
            return
        outcome = candidate.outcome
        met, baseline = self.criterion.satisfied(outcome.speedup, self.history)
        if not met:
            return
        timings = [outcome.speedup]
        confirm = self.ctx.cfg.termination.confirm if baseline is not None else 1  # first_valid has no bar
        for n in range(1, confirm):  # ahead of every other timing: the iteration may end on it
            again = await self._run_job("benchmark", state, candidate, PRIORITY_MAIN, bench_params,
                                        order=CONFIRM_ORDER, name=f"benchmark_confirm{n}")
            timings.append(again.get("speedup") if again.get("ok") else None)
            met, baseline = self.criterion.satisfied(timings[-1], self.history)
            if not met:
                break
        if len(timings) > 1:
            measured = [t for t in timings if t is not None]
            outcome.speedup = sum(measured) / len(measured)
            candidate.record["benchmark_confirmations"] = timings
            self._event("termination_check", state, kernel=candidate.kernel_id, timings=timings, met=met,
                        speedup=outcome.speedup)
        if not met or state.terminated.is_set():
            return
        state.terminated_by = candidate
        state.termination = {"criterion": self.criterion.name, "kernel": candidate.kernel_id,
                             "speedup": outcome.speedup, "baseline": baseline, "timings": timings,
                             "margin": self.criterion.margin, "elapsed_s": round(time.monotonic() - state.started, 3)}
        self._event("iteration_terminated", state, **state.termination)
        state.terminated.set()

    # -- generation ---------------------------------------------------------

    async def _main(self, state: IterationState) -> None:
        ctx = self.ctx
        files = StreamFiles(ctx.layout.iteration_dir(self.task, state.iteration) / "main")
        controller: Optional[SpecController | PacedSpecController] = None
        if self._demand:
            controller = PacedSpecController(lambda tokens, prefix: self._maybe_launch(state, tokens, prefix))
        elif ctx.triggers is not None:
            controller = SpecController(
                TriggerMonitor(ctx.triggers),
                lambda trigger, prefix: self._on_trigger(state, controller, trigger, prefix),
            )
        sink = MainSink(files, controller, lambda tokens: self._event("main_progress", state, tokens=tokens))
        self._event("main_started", state)
        try:
            gen = await ctx.generator.generate_main(state.prompt, sink)
        finally:
            files.close()
            if controller is not None:
                controller.stop()
        state.main = gen
        self._event("main_finished", state, ok=gen.ok, seconds=gen.seconds, reasoning_tokens=gen.reasoning_tokens,
                    content_tokens=gen.content_tokens, finish_reason=gen.finish_reason)
        candidate = self._candidate(state, f"i{state.iteration:03d}-main", "main", gen)
        await self._evaluate(state, candidate, PRIORITY_MAIN)

    def _on_trigger(self, state: IterationState, controller: Optional[SpecController], trigger: Trigger,
                    prefix: str) -> None:
        """Launch this trigger's specs.  With ``--spec-queue replace`` specs beyond the free generation
        slots wait for one, and a newer trigger of the task replaces its waiting ones: the newer
        prefix contains everything the older one did."""
        ctx = self.ctx
        replace = ctx.cfg.spec.queue == "replace"
        superseded = []
        if replace:
            for item in state.spec_items:
                if not item.started and not item.superseded and item.task is not None and not item.task.done():
                    item.superseded = True
                    item.task.cancel()
                    self._release(item)
                    superseded.append(item.kernel_id)
        k, budget = ctx.budget.decide(
            self.task.task_id, pool=ctx.pool, recorder=ctx.recorder, triggers=ctx.triggers,
            tokens_per_s=controller.tokens_per_second() if controller is not None else None, cap=not replace,
        )
        entry = {**trigger.to_dict(), "k": k, "budget": budget, "superseded": len(superseded),
                 "elapsed_s": round(time.monotonic() - state.started, 3)}
        state.triggers.append(entry)
        if superseded:
            self._event("spec_superseded", state, kernels=superseded, by_trigger=trigger.index)
        self._event("spec_triggered", state, **entry)
        if k == 0 and controller is not None:
            controller.monitor.refund()
        for index in range(k):
            item = SpecItem(f"i{state.iteration:03d}-t{trigger.index:02d}-s{index}")
            item.task = asyncio.create_task(self._spec(state, item, trigger, prefix, index, k),
                                            name=f"{self.task.task_id}:{item.kernel_id}")
            item.task.add_done_callback(lambda _, item=item: self._release(item))
            state.spec_items.append(item)
            state.spec_tasks.append(item.task)

    def _maybe_launch(self, state: IterationState, tokens: int, prefix: Callable[[], str]) -> None:
        """Demand pacing (``control.pacing``), on every reasoning delta: launch one spec when the kernels
        this task expects at the profiler fall short, with the test results of the iteration's earlier
        candidates after the reasoning prefix."""
        ctx, spec = self.ctx, self.ctx.cfg.spec
        if state.terminated.is_set() or (spec.max_per_iteration and state.launches >= spec.max_per_iteration):
            return
        if state.last_launch_tokens is not None and tokens - state.last_launch_tokens < spec.min_gap_tokens:
            return
        now = time.monotonic()
        if state.pause_kernel is not None and now >= state.pause_until:
            self._resume(state, state.pause_kernel, "wait cap")
        pool = ctx.pool
        queued = pool.queued(self.task.task_id)
        waiting = queued.get("benchmark", 0) + queued.get("ncu", 0) * (
            pool.expected_run_s("ncu") / max(pool.expected_run_s("benchmark"), 1e-3))
        p_pass = self.pass_rate.value(spec.prior_pass_rate)
        share = pool.exclusive_devices() / max(1, ctx.budget.active_tasks)
        target = max(share + spec.target_extra, share * spec.target_factor)
        launch, _ = launch_decision(tokens=tokens, last_tokens=state.last_launch_tokens, first_tokens=spec.first_tokens,
                                    min_gap_tokens=spec.min_gap_tokens, waiting=waiting, upstream=state.upstream,
                                    p_pass=p_pass, target=target, paused=state.pause_kernel is not None)
        if not launch or not ctx.budget.reserve(self.task.task_id):
            return
        state.launches += 1
        state.last_launch_tokens = tokens
        text = prefix()
        trigger = Trigger(state.launches, "demand", "", tokens, tokens, len(text))
        best = max((c for c in state.candidates if c.outcome.benchmarked), key=lambda c: c.outcome.speedup, default=None)
        notes, items = draft_notes((c.outcome for c in state.candidates), best.kernel_id if best else None)
        entry = {**trigger.to_dict(), "k": 1, "waiting": round(waiting, 2), "upstream": state.upstream,
                 "p_pass": round(p_pass, 3), "target": round(target, 2), "notes_items": items,
                 "notes_chars": len(notes), "elapsed_s": round(now - state.started, 3)}
        state.triggers.append(entry)
        self._event("spec_triggered", state, **entry)
        state.upstream += 1
        item = SpecItem(f"i{state.iteration:03d}-t{trigger.index:02d}-s0")
        item.task = asyncio.create_task(self._spec(state, item, trigger, text, 0, 1, notes=notes, notes_items=items),
                                        name=f"{self.task.task_id}:{item.kernel_id}")
        item.task.add_done_callback(lambda _, item=item: self._release(item))
        state.spec_items.append(item)
        state.spec_tasks.append(item.task)

    def _meets_standard(self, speedup: Optional[float]) -> bool:
        """Whether a timed kernel meets the termination criterion (without one: beats the best so far and the
        reference); only such a kernel makes spec launches wait for its ncu profile."""
        if speedup is None:
            return False
        if self.criterion is not None:
            return self.criterion.satisfied(speedup, self.history)[0]
        best = self.history.best()
        return speedup > max(best.speedup if best is not None else 1.0, 1.0)

    def _resume(self, state: IterationState, kernel_id: str, reason: str) -> None:
        """Spec launches stop waiting for ``kernel_id``'s ncu profile."""
        if state.pause_kernel != kernel_id:
            return
        waited = time.monotonic() - state.pause_started
        state.paused_s += waited
        state.pause_kernel = None
        self._event("spec_resumed", state, kernel=kernel_id, reason=reason, waited_s=round(waited, 3))

    def _leave_upstream(self, state: IterationState, candidate: Candidate) -> None:
        """A demand-paced spec reached timing, or failed before it."""
        if candidate.upstream:
            candidate.upstream = False
            state.upstream -= 1
            self.pass_rate.add(candidate.outcome.valid)

    def _release(self, item: SpecItem) -> None:
        if not item.released:
            item.released = True
            self.ctx.budget.release(self.task.task_id)

    async def _spec(self, state: IterationState, item: SpecItem, trigger: Trigger, prefix: str, index: int,
                    k: int, *, notes: str = "", notes_items: int = 0) -> None:
        ctx = self.ctx
        kernel_id = item.kernel_id
        seed = ctx.cfg.eval.seed * 1_000_003 + state.iteration * 10_007 + trigger.index * 101 + index
        async with ctx.budget.slot():
            item.started = True
            self._event("spec_started", state, kernel=kernel_id, trigger=trigger.index)
            try:
                gen = await ctx.generator.generate_spec(state.prompt, prefix, index=index, seed=seed, notes=notes)
            finally:
                self._release(item)  # only the generation counts as in flight, not the evaluation
        ctx.budget.observe(gen)
        self._event("spec_finished", state, kernel=kernel_id, ok=gen.ok, seconds=gen.seconds,
                    has_code=gen.code is not None)
        extra = {"notes": {"items": notes_items, "chars": len(notes)}} if self._demand else {}
        candidate = self._candidate(state, kernel_id, "spec", gen, seq=trigger.index, trigger=trigger.to_dict(),
                                    batch={"k": k, "index": index}, **extra)
        if notes:
            (candidate.kernel_dir / "spec_notes.txt").write_text(notes.lstrip("\n"), encoding="utf-8")
        candidate.upstream = self._demand  # takes over the count of its launch
        await self._evaluate(state, candidate, PRIORITY_SPEC)

    def _candidate(self, state: IterationState, kernel_id: str, source: str, gen: GenerationResult,
                   seq: int = 0, **extra: Any) -> Candidate:
        ctx = self.ctx
        kernel_dir = ctx.layout.kernel_dir(self.task, state.iteration, kernel_id)
        kernel_dir.mkdir(parents=True, exist_ok=True)
        if source != "main":  # the main stream is already in main/{reasoning,content}.txt
            (kernel_dir / "content.txt").write_text(gen.content, encoding="utf-8")
        record = ctx.recorder.kernel(self.task.task_id, kernel_id, iteration=state.iteration, epoch=state.epoch,
                                     source=source, generation=gen.summary(), **extra)
        outcome = KernelOutcome(kernel_id=kernel_id, iteration=state.iteration, source=source, code=gen.code or "")
        if not gen.ok:
            outcome.status, outcome.error = "llm_error", gen.error
        elif gen.code is None:
            outcome.status, outcome.error = "parse_error", "no ```python fence found in the output"
        elif not has_code(gen.code):
            outcome.status, outcome.error = "parse_error", "the module has only comments, no code"
        elif forbidden_usage(gen.code):
            outcome.status, outcome.error = "forbidden", forbidden_usage(gen.code)
        candidate = Candidate(kernel_id, source, outcome, record, kernel_dir, seq=seq)
        state.candidates.append(candidate)
        return candidate

    # -- evaluation ---------------------------------------------------------

    async def _evaluate(self, state: IterationState, candidate: Candidate, priority: int) -> None:
        try:
            await self._pipeline(state, candidate, priority)
        except Exception as e:  # noqa: BLE001 - one broken candidate must not end the iteration
            log.exception("%s/%s: evaluation failed", self.task.task_id, candidate.kernel_id)
            candidate.outcome.status, candidate.outcome.error = "internal_error", f"{type(e).__name__}: {e}"
        finally:
            if candidate.outcome.status not in ("pending", "evaluating"):  # not when cancelled midway
                self._leave_upstream(state, candidate)

    async def _pipeline(self, state: IterationState, candidate: Candidate, priority: int) -> None:
        outcome, record = candidate.outcome, candidate.record
        if outcome.status != "pending":
            return
        fingerprint = code_fingerprint(outcome.code)
        first = self._fingerprints.setdefault(fingerprint, candidate.kernel_id)
        if first != candidate.kernel_id:
            outcome.status, outcome.error = "duplicate", f"same code as {first}"
            record["duplicate_of"] = first
            return
        (candidate.kernel_dir / "kernel.py").write_text(outcome.code, encoding="utf-8")
        ctx, cfg = self.ctx, self.ctx.cfg.eval
        outcome.status = "evaluating"
        order = -candidate.seq if self._demand and candidate.source == "spec" else 0.0  # newest first
        if ctx.pool.has_role("builder"):
            build = await self._run_job("build", state, candidate, priority, {}, order=order)
            record["build"] = _summary(build)
            if not build.get("ok") and build.get("status") in ("compile_error", "timeout"):
                outcome.status, outcome.error = str(build["status"]), _error_text(build)
                return

        params = {"seed": cfg.seed, "atol": cfg.atol, "rtol": cfg.rtol, "trials": cfg.correctness_trials,
                  "shift_trial": cfg.shift_trial,
                  "ref_cache": str(ctx.layout.task_dir(self.task) / "ref_cache")}
        validation = await self._run_job("validate", state, candidate, priority, params, order=order)
        if not validation.get("ok") and _out_of_memory(validation) and validation.get("concurrency", 1) > 1:
            record["validation_shared"] = _summary(validation)  # retry alone on a GPU
            validation = await self._run_job("validate", state, candidate, priority, params, exclusive=True,
                                             order=order)
        record["validation"] = _summary(validation)
        if not validation.get("ok"):
            outcome.status, outcome.error = str(validation.get("status", "runtime_error")), _error_text(validation)
            return
        record["forward_kernels"] = validation.get("kernels")
        libraries = library_kernels(k.get("name", "") for k in validation.get("kernels") or ())
        if libraries:
            outcome.status = "forbidden"
            outcome.error = f"{CATEGORY_RUNTIME}: {', '.join(short_kernel_name(n) for n in libraries[:3])}"
            return
        outcome.valid = True
        self._leave_upstream(state, candidate)

        bench_params = {"seed": cfg.seed, "warmup": cfg.bench_warmup, "iters": cfg.bench_iters,
                        "rounds": cfg.bench_rounds}
        bench = await self._run_job("benchmark", state, candidate, priority, bench_params, order=order)
        record["benchmark"] = _summary(bench)
        if not bench.get("ok"):
            outcome.status, outcome.error = "benchmark_error", _error_text(bench)
            return
        outcome.status = "success"
        outcome.speedup = bench.get("speedup")
        outcome.latency_ms = bench.get("latency_ms")
        outcome.ref_latency_ms = bench.get("ref_latency_ms")
        self._event("kernel_benchmarked", state, kernel=candidate.kernel_id, source=candidate.source,
                    speedup=outcome.speedup)
        wait = ctx.cfg.spec.best_ncu_wait_s
        urgent = candidate.kernel_id if self._demand and wait > 0 and self._meets_standard(outcome.speedup) else None
        self._schedule_ncu(state, urgent=urgent)
        if urgent is not None and urgent in self._ncu_tasks and not state.terminated.is_set():
            if state.pause_kernel is not None:
                self._resume(state, state.pause_kernel, "newer best")
            state.pause_kernel, state.pause_started = urgent, time.monotonic()
            state.pause_until = state.pause_started + wait
            self._event("spec_paused", state, kernel=urgent, speedup=outcome.speedup, wait_s=wait)
        await self._check_termination(state, candidate, priority, bench_params)

    # -- ncu ---------------------------------------------------------------

    def _ncu_wanted(self, state: IterationState) -> list[Candidate]:
        """This iteration's benchmarked kernels whose ncu profile a later prompt uses, most important
        first: a new best so far; for topk prompts the iteration's fastest; for summary prompts each
        approach's new best (its row shows that kernel's profile).  Without spec generation, the main
        kernel.  At most ``ncu_top_n``."""
        run = self.ctx.cfg.run
        ranked = sorted((c for c in state.candidates if c.outcome.benchmarked),
                        key=lambda c: (-c.outcome.speedup, c.kernel_id))
        if not ranked:
            return []
        best = self.history.best()  # earlier iterations: this one's outcomes join the history at its end
        wanted = [ranked[0]] if best is None or ranked[0].outcome.speedup > best.speedup else []
        if run.history_prompt in ("topk", "topk+summary"):
            wanted += ranked[:run.feedback_top_k + 1]
        if run.history_prompt in ("summary", "topk+summary"):
            earlier = {key: o.speedup for key, o in self.history.approach_bests().items()}
            seen: set[str] = set()
            for c in ranked:
                key = family(c.outcome.code)
                if key not in seen:
                    seen.add(key)
                    if c.outcome.speedup > earlier.get(key, 0.0):
                        wanted.append(c)
        unique = list({c.kernel_id: c for c in wanted}.values())
        return unique[:self.ctx.cfg.eval.ncu_top_n]

    def _schedule_ncu(self, state: IterationState, urgent: Optional[str] = None) -> None:
        """Start ncu jobs for the wanted kernels (``urgent``'s ahead of timing jobs); drop queued ones that
        are no longer wanted."""
        if not self.ctx.ncu_enabled or self.ctx.cfg.eval.ncu_top_n <= 0:
            return
        wanted = self._ncu_wanted(state)
        keep = {c.kernel_id for c in wanted}
        for c in wanted:
            if c.kernel_id not in self._ncu_tasks and c.outcome.ncu_profile is None:
                self._ncu_tasks[c.kernel_id] = asyncio.create_task(
                    self._ncu(state, c, urgent=c.kernel_id == urgent), name=f"ncu:{c.kernel_id}")
        dropped = {f"{self.task.task_id}/{c.kernel_id}/ncu" for c in state.candidates if c.kernel_id not in keep}
        if dropped:
            self.ctx.pool.cancel_where(lambda job: job.kind == "ncu" and job.label in dropped)

    async def _ncu(self, state: IterationState, candidate: Candidate, urgent: bool = False) -> None:
        cfg, outcome = self.ctx.cfg.eval, candidate.outcome
        try:
            try:
                result = await self._run_job("ncu", state, candidate, PRIORITY_SPEC if urgent else PRIORITY_MAIN, {
                    "seed": cfg.seed, "ncu_binary": cfg.ncu_binary, "metrics": list(cfg.ncu_metrics),
                    "sections": list(cfg.ncu_sections), "forward": cfg.ncu_forward,
                    "launch_count": cfg.ncu_launch_count, "timeout_s": cfg.ncu_timeout_s,
                }, order=URGENT_NCU_ORDER if urgent else -(outcome.speedup or 0.0), sticky=True, urgent=urgent)
            except Exception:  # noqa: BLE001 - ncu is best effort
                log.exception("%s/%s: ncu job failed", self.task.task_id, candidate.kernel_id)
                return
            finally:
                self._ncu_tasks.pop(candidate.kernel_id, None)
            candidate.record["ncu"] = {**_summary(result), "kernels": result.get("kernels"),
                                       "rules": result.get("rules"), "degraded": result.get("degraded"),
                                       "selection": result.get("selection"), "urgent": urgent}
            if result.get("ok"):
                outcome.ncu_profile = result.get("profile")
            elif result.get("status") == "ncu_unavailable" and self.ctx.ncu_enabled:
                self.ctx.ncu_enabled = False
                log.warning("%s: %s; ncu disabled for the rest of the run", self.task.task_id, result.get("error"))
            if result.get("degraded") or result.get("status") == "ncu_timeout":
                kind = "timeout" if result.get("status") == "ncu_timeout" else "degraded"
                log.warning("%s/%s: ncu %s after %s s", self.task.task_id, candidate.kernel_id, kind,
                            result.get("ncu_s"))
                self._event("ncu_alert", state, kind=kind, kernel=candidate.kernel_id, ncu_s=result.get("ncu_s"))
            self._event("kernel_ncu", state, kernel=candidate.kernel_id, ok=bool(result.get("ok")),
                        status=result.get("status"), kernels=result.get("kernels"), rules=result.get("rules"),
                        degraded=result.get("degraded"), selection=result.get("selection"), ncu_s=result.get("ncu_s"))
        finally:  # after the profile is attached, so the next launches see it
            self._resume(state, candidate.kernel_id, "ncu finished")

    def _shown_kernels(self, iteration: int) -> set[str]:
        """Kernels whose ncu profile the prompt after ``iteration`` shows: the best so far; for topk
        prompts that iteration's fastest; for summary prompts the best kernel of every approach."""
        run = self.ctx.cfg.run
        best = self.history.best()
        shown = {best.kernel_id} if best is not None else set()
        if run.history_prompt in ("topk", "topk+summary"):
            shown |= {o.kernel_id for o in self.history.shown_from(iteration, run.feedback_top_k)}
        if run.history_prompt in ("summary", "topk+summary"):
            shown |= {o.kernel_id for o in self.history.approach_bests().values()}
        return shown

    async def _await_ncu(self, iteration: int) -> None:
        """Give the ncu jobs of the kernels the next prompt shows up to ``ncu_wait_s`` to finish; one that
        is still running then keeps going and its profile reaches a later prompt."""
        shown = self._shown_kernels(iteration - 1)
        pending = [t for kernel_id, t in self._ncu_tasks.items() if kernel_id in shown and not t.done()]
        wait = self.ctx.cfg.eval.ncu_wait_s
        if not pending or wait <= 0:
            return
        started = time.monotonic()
        done, not_done = await asyncio.wait(pending, timeout=wait)
        seconds = round(time.monotonic() - started, 3)
        self.ctx.recorder.event("ncu_wait", task=self.task.task_id, iteration=iteration, seconds=seconds,
                                finished=len(done), unfinished=len(not_done))
        if not_done:
            log.warning("%s: %d ncu job(s) still running after the %.0f s wait before iteration %d",
                        self.task.task_id, len(not_done), wait, iteration)
            self.ctx.recorder.event("ncu_alert", task=self.task.task_id, iteration=iteration, kind="wait_cap",
                                    seconds=seconds, unfinished=len(not_done))

    async def _run_job(self, kind: str, state: IterationState, candidate: Candidate, priority: int,
                       params: dict[str, Any], *, exclusive: bool = False, order: float = 0.0,
                       sticky: bool = False, urgent: bool = False, name: Optional[str] = None) -> dict[str, Any]:
        kernel_dir = candidate.kernel_dir
        job = EvalJob(kind=kind, task_kind=self.task.kind, task_path=str(self.task.path),
                      kernel_path=str(kernel_dir / "kernel.py"), work_dir=str(kernel_dir), params=params,
                      name=name or "")
        evaluator = self.ctx.evaluator
        result = await self.ctx.pool.submit(PoolJob(
            kind=kind, task_id=self.task.task_id, epoch=state.epoch, priority=priority, exclusive=exclusive,
            order=order, sticky=sticky, urgent=urgent, run=lambda slot: evaluator.run(job, slot),
            label=f"{self.task.task_id}/{candidate.kernel_id}/{job.stem}",
        ))
        write_json(kernel_dir / f"{job.stem}.json", result)
        self.ctx.recorder.observe_job(kind, result)
        return result

    # -- selection ----------------------------------------------------------

    def _finish(self, state: IterationState, end_reason: str, unfinished: int = 0) -> Optional[KernelOutcome]:
        """Select the iteration's kernel, record everything and retire the epoch.

        A terminating kernel is the selection; otherwise the fastest benchmarked one.
        Candidates still in the pipeline were cut by the grace period or the termination.
        """
        ctx, task = self.ctx, self.task
        ranked = sorted((c for c in state.candidates if c.outcome.benchmarked),
                        key=lambda c: (-c.outcome.speedup, c.kernel_id))
        selected = state.terminated_by or (ranked[0] if ranked else None)
        cut = "grace_truncated" if end_reason == "grace_expired" else "cancelled"
        for c in state.candidates:
            if c.outcome.status in ("pending", "evaluating"):
                c.outcome.status, c.outcome.valid = cut, False
            self.history.add(c.outcome)
            c.record.update(status=c.outcome.status, valid=c.outcome.valid, speedup=c.outcome.speedup,
                            selected=c is selected, terminated_iteration=c is state.terminated_by)
            ctx.recorder.finish_kernel(task.task_id, c.kernel_id)
        if selected is not None:
            self.history.select(state.iteration, selected.outcome)
        self.epoch += 1
        ctx.pool.cancel(task.task_id, before_epoch=self.epoch)
        # ncu keeps running only for kernels the next prompt shows; it waits for them (ncu_wait_s)
        shown = self._shown_kernels(state.iteration)
        for kernel_id, ncu_task in list(self._ncu_tasks.items()):
            if kernel_id not in shown:
                ncu_task.cancel()

        if state.pause_kernel is not None:
            self._resume(state, state.pause_kernel, "iteration end")
        main = next((c for c in state.candidates if c.source == "main"), None)
        best = self.history.best()
        statuses: dict[str, int] = {}
        for c in state.candidates:
            statuses[c.outcome.status] = statuses.get(c.outcome.status, 0) + 1
        gen = state.main
        ctx.recorder.finish_iteration({
            "task": task.task_id,
            "iteration": state.iteration,
            "end_reason": end_reason,
            "seconds": round(time.monotonic() - state.started, 3),
            "selected": selected.kernel_id if selected else None,
            "selected_source": selected.source if selected else None,
            "speedup": selected.outcome.speedup if selected else None,
            "best_speedup": best.speedup if best else None,
            "main_status": main.outcome.status if main else "cancelled",
            "main_speedup": main.outcome.speedup if main else None,
            "candidates": len(state.candidates),
            "statuses": statuses,
            "unfinished_spec_tasks": unfinished,
            "termination": state.termination or None,
            "triggers": [{k: t.get(k) for k in ("index", "reason", "keyword", "fired_tokens", "k", "superseded")}
                         for t in state.triggers],
            "superseded_specs": sum(1 for item in state.spec_items if item.superseded),
            "spec_schedule": self.ctx.cfg.spec.schedule if self.ctx.triggers is not None else None,
            "spec_paused_s": round(state.paused_s, 3),
            "reasoning_tokens": gen.reasoning_tokens if gen else None,
            "content_tokens": gen.content_tokens if gen else None,
            "generation_s": gen.seconds if gen else None,
        })
        if selected:
            log.info("%s iter %d: %s %s speedup=%.3fx (%d candidates, %d triggers)", task.task_id, state.iteration,
                     selected.kernel_id, selected.outcome.status, selected.outcome.speedup, len(state.candidates),
                     len(state.triggers))
        else:
            log.info("%s iter %d: no valid kernel (main: %s; %d candidates)", task.task_id, state.iteration,
                     main.outcome.status if main else "cancelled", len(state.candidates))
        return selected.outcome if selected else (main.outcome if main else None)
