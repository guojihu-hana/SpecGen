import asyncio
import json

from conftest import CORRECT_RELU, ModeBackend, make_context, relu_kernel, spec_answer

from specgen.config import EvalConfig
from specgen.control.history import KernelOutcome
from specgen.control.loop import IterationLoop
from specgen.control.pacing import NOTES_HEADER, PassRate, draft_notes, launch_decision
from specgen.control.triggers import TriggerConfig
from specgen.generation.generator import Generator
from specgen.generation.prompts import Prompt
from specgen.llm.events import ContentDelta, ReasoningDelta, StreamEnd
from specgen.llm.profiles import ModelProfile
from specgen.llm.tokens import TokenCounter
from specgen.resources.pool import PRIORITY_MAIN, PRIORITY_SPEC, Device, PoolJob, ResourcePool
from specgen.tasks.base import KernelBenchTask

PROFILE = {"kernels": [{"name": "gemm_kernel", "launches": 1, "share": 1.0, "rules": [],
                        "metrics": {"sm__throughput.avg.pct_of_peak_sustained_elapsed": 61.7}}]}


def test_launch_decision():
    args = dict(first_tokens=500, min_gap_tokens=500, waiting=0.0, upstream=0, p_pass=0.4, target=2.0, paused=False)
    assert launch_decision(tokens=499, last_tokens=None, **args) == (False, "first tokens")
    assert launch_decision(tokens=500, last_tokens=None, **args) == (True, "launch")
    assert launch_decision(tokens=900, last_tokens=500, **args) == (False, "gap")
    assert launch_decision(tokens=1000, last_tokens=500, **args)[0]
    assert launch_decision(tokens=1000, last_tokens=500, **{**args, "upstream": 4})[0]  # 1.6 expected < 2
    assert launch_decision(tokens=1000, last_tokens=500, **{**args, "upstream": 5}) == (False, "profiler fed")
    assert launch_decision(tokens=1000, last_tokens=500, **{**args, "waiting": 2.0})[1] == "profiler fed"
    assert not launch_decision(tokens=1000, last_tokens=500, **{**args, "paused": True})[0]


def test_pass_rate_starts_at_the_prior():
    rate = PassRate()
    assert rate.value(0.4) == 0.4
    rate.add(True)
    rate.add(False)
    assert rate.value(0.4) == (1 + 0.8) / 4


def outcome(kernel_id, status, *, speedup=None, error=None, approach="fp16 mma.sync 128x128x64", profile=None):
    o = KernelOutcome(kernel_id, 0, "spec", f"# APPROACH: {approach}\nimport torch\n", status=status, speedup=speedup,
                      valid=speedup is not None, latency_ms=0.406 if speedup else None, error=error)
    o.ncu_profile = profile
    return o


def test_draft_notes_list_the_finished_drafts_with_the_best_profile():
    outcomes = [
        outcome("a", "compile_error", error='cuda.cu(12): error: identifier "uint32_t" is undefined',
                approach="wmma 64x64 double buffer"),
        outcome("b", "success", speedup=5.16, profile=PROFILE),
        outcome("c", "numerical_error", error='{"reason": "value_mismatch", "rel_err": 0.44}'),
        outcome("d", "forbidden", error="uses a library GEMM or convolution instead of a hand-written kernel: torch.matmul"),
        outcome("e", "evaluating"), outcome("f", "parse_error"), outcome("g", "duplicate"),
    ]
    text, items = draft_notes(outcomes, "b")
    assert items == 4 and text.startswith("\n\n" + NOTES_HEADER)
    lines = text.splitlines()[3:]
    assert lines[0] == '- wmma 64x64 double buffer: does not compile: identifier "uint32_t" is undefined'
    assert lines[1] == "- fp16 mma.sync 128x128x64: correct, 5.16x the reference (0.406 ms), the best so far; " \
                       "Nsight Compute: SM 62%"
    assert lines[2] == "- fp16 mma.sync 128x128x64: wrong results, relative error 0.44"
    assert lines[3] == "- fp16 mma.sync 128x128x64: rejected: uses a library GEMM or convolution instead of a " \
                       "hand-written kernel)"
    assert draft_notes([outcome("e", "evaluating")], None) == ("", 0)


def test_draft_notes_keep_the_best_and_the_latest():
    outcomes = [outcome("best", "success", speedup=3.0)] + [
        outcome(f"k{i}", "runtime_error", error=f"RuntimeError: failure number {i} " + "x" * 300) for i in range(20)]
    text, items = draft_notes(outcomes, "best")
    assert "the best so far" in text and "failure number 19" in text and "failure number 0 " not in text
    assert len(text) <= 2400 + len(NOTES_HEADER) + 8 and items == text.count("\n- ")


def test_urgent_jobs_rank_with_timing_and_queues_are_counted():
    async def go():
        pool = ResourcePool([Device(None, "validator", 2), Device(None, "profiler", 1)])
        await pool.start()
        started, gate = [], asyncio.Event()

        def job(kind, label, **kw):
            async def run(slot):
                started.append(label)
                await gate.wait()
                return {"ok": True}
            return PoolJob(kind=kind, task_id="t", epoch=0, run=run, label=label, **kw)

        first = asyncio.create_task(pool.submit(job("benchmark", "running")))
        await asyncio.sleep(0.01)
        rest = [asyncio.create_task(pool.submit(j)) for j in (
            job("ncu", "ncu-later", order=-5.0), job("benchmark", "spec-1", priority=1, order=-1),
            job("benchmark", "spec-2", priority=1, order=-2), job("ncu", "ncu-urgent", priority=1, order=-1e11,
                                                                   urgent=True))]
        await asyncio.sleep(0.01)
        assert pool.queued("t") == {"ncu": 2, "benchmark": 2} and pool.queued("other") == {}
        assert pool.exclusive_devices() == 1
        gate.set()
        await asyncio.gather(first, *rest)
        await pool.close()
        return started

    assert asyncio.run(go()) == ["running", "ncu-urgent", "spec-2", "spec-1", "ncu-later"]


def test_notes_follow_the_reasoning_inside_the_thinking():
    backend = ModeBackend(spec=[spec_answer(CORRECT_RELU), spec_answer(CORRECT_RELU)], prefill_style="vllm")
    prompt = Prompt("system", (("task", "task text"),))
    notes = "\n\n" + NOTES_HEADER + "\n- x: does not compile)"
    prefill = Generator(backend, ModelProfile("fake"), TokenCounter("generic"))
    asyncio.run(prefill.generate_spec(prompt, "my reasoning", notes=notes))
    assert backend.calls[-1][1][-1]["content"].startswith(f"<think>\nmy reasoning{notes}\n</think>\n\n```python\n")
    reprompt = Generator(backend, ModelProfile("fake"), TokenCounter("generic"), spec_prompt_mode="reprompt")
    asyncio.run(reprompt.generate_spec(prompt, "my reasoning", notes=notes))
    assert f"my reasoning{notes}\n--- end of reasoning ---" in backend.calls[-1][1][-1]["content"]


class PacedEvaluator:
    """Builds and validations pass at once (unless ``fail`` says otherwise); timing and ncu take a while."""

    def __init__(self, *, speedups=None, default_speedup=1.0, bench_s=0.0, ncu_s=0.0, fail=None):
        self.speedups, self.bench_s, self.ncu_s, self.fail = speedups or {}, bench_s, ncu_s, fail or {}
        self.default_speedup = default_speedup
        self.started = []  # (kind, kernel id) in start order

    async def run(self, job, slot):
        kernel_id = job.work_dir.rsplit("/", 1)[-1]
        self.started.append((job.kind, kernel_id))
        if self.fail.get(kernel_id, (None,))[0] == job.kind:
            return self.fail[kernel_id][1]
        if job.kind == "benchmark":
            await asyncio.sleep(self.bench_s)
            return {"ok": True, "status": "success", "speedup": self.speedups.get(kernel_id, self.default_speedup),
                    "latency_ms": 1.0, "ref_latency_ms": 1.0}
        if job.kind == "ncu":
            await asyncio.sleep(self.ncu_s)
            return {"ok": True, "status": "success", "profile": PROFILE, "kernels": 1, "rules": 0}
        return {"ok": True, "status": "success"}


def reasoning(deltas, pause):
    events = []
    for i in range(deltas):
        events += [ReasoningDelta(f"step {i}: weigh the tile shapes against the memory traffic once more. "), pause]
    return events + [ContentDelta(f"```python\n{CORRECT_RELU}\n```"), StreamEnd("stop")]


def prefilled_answer(i):
    """A spec reply continuing the prefilled "```python\\n# APPROACH:"."""
    return [ContentDelta(f" draft {i}\n{relu_kernel(f'torch.clamp_min(x, 0.0) + {i} * 0')}\n```"), StreamEnd("stop")]


def run_paced(tmp_path, relu_task, evaluator, *, deltas, pause, eval_cfg=None, termination=None, history=(),
              **spec_options):
    specs = [prefilled_answer(i) for i in range(200)]
    backend = ModeBackend(main=[reasoning(deltas, pause)], spec=specs, prefill_style="vllm")
    ctx = make_context(tmp_path / "out", backend, triggers=TriggerConfig(), schedule="demand", evaluator=evaluator,
                       eval_cfg=eval_cfg, tasks=[relu_task], spec_options={"first_tokens": 10, "min_gap_tokens": 10,
                                                                           **spec_options},
                       termination=termination, devices=[Device(None, "validator", 8), Device(None, "profiler", 1)])
    submit, ctx.submitted = ctx.pool.submit, []

    async def recording_submit(job):
        ctx.submitted.append(job)
        return await submit(job)

    ctx.pool.submit = recording_submit

    async def go():
        await ctx.pool.start()
        loop = IterationLoop(KernelBenchTask.from_path(relu_task), ctx)
        for earlier in history:
            loop.history.add(earlier)
        try:
            await loop.run_iteration(0)
            await asyncio.gather(*loop._ncu_tasks.values(), return_exceptions=True)
            return loop
        finally:
            await ctx.pool.close()
            ctx.recorder.close()

    loop = asyncio.run(go())
    events = [json.loads(line) for line in (ctx.layout.run_dir / "events.jsonl").read_text().splitlines()]
    return ctx, loop, backend, events


def test_demand_launches_one_spec_at_a_time_while_the_profiler_is_short(tmp_path, relu_task):
    evaluator = PacedEvaluator(bench_s=0.3)
    ctx, loop, backend, events = run_paced(tmp_path, relu_task, evaluator, deltas=40, pause=0.02, best_ncu_wait_s=0)
    launches = [e for e in events if e["type"] == "spec_triggered"]
    assert 2 <= len(launches) <= 20  # paced by the profiler, not by every delta
    assert all(e["reason"] == "demand" and e["k"] == 1 for e in launches)
    assert all(e["waiting"] + e["p_pass"] * e["upstream"] < e["target"] == 2.0 for e in launches)
    tokens = [e["fired_tokens"] for e in launches]
    assert tokens[0] >= 10 and all(b - a >= 10 for a, b in zip(tokens, tokens[1:]))
    assert [e["index"] for e in launches] == list(range(1, len(launches) + 1))
    kernels = [json.loads(line) for line in (ctx.layout.run_dir / "kernels.jsonl").read_text().splitlines()]
    assert {k["kernel_id"] for k in kernels if k["source"] == "spec"} == {f"i000-t{e['index']:02d}-s0" for e in launches}
    assert loop.pass_rate.finished == len(launches) and loop.pass_rate.passed == len(launches)


def test_specs_see_the_results_of_earlier_ones_and_are_timed_newest_first(tmp_path, relu_task):
    fail = {"i000-t01-s0": ("validate", {"ok": False, "status": "compile_error",  # no builder: built in validation
                                         "error": 'cuda.cu(9): error: identifier "uint32_t" is undefined'})}
    evaluator = PacedEvaluator(bench_s=0.1, fail=fail, default_speedup=2.0)
    ctx, loop, backend, events = run_paced(tmp_path, relu_task, evaluator, deltas=40, pause=0.02, best_ncu_wait_s=0,
                                           target_extra=3.0)
    prefixes = [messages[-1]["content"] for mode, messages, _ in backend.calls if mode == "spec"]
    assert NOTES_HEADER not in prefixes[0]
    later = [p for p in prefixes if "2.00x the reference" in p]
    assert later and all(p.index(NOTES_HEADER) < p.index("</think>") for p in later)
    assert any('does not compile: identifier "uint32_t" is undefined' in p for p in later)
    launches = {e["index"]: e for e in events if e["type"] == "spec_triggered"}
    noted = [e for e in launches.values() if e["notes_items"]]
    assert noted and all(e["notes_chars"] > len(NOTES_HEADER) for e in noted)
    kernel_dir = ctx.layout.kernel_dir(KernelBenchTask.from_path(relu_task), 0, f"i000-t{noted[-1]['index']:02d}-s0")
    assert (kernel_dir / "spec_notes.txt").read_text().startswith(NOTES_HEADER)
    for job in ctx.submitted:  # the pool runs lower orders first within a priority: specs newest first
        kernel = job.label.split("/")[-2]
        if job.kind in ("validate", "benchmark") and not job.label.endswith("confirm1"):
            expected = (PRIORITY_MAIN, 0.0) if kernel.endswith("main") else (PRIORITY_SPEC, -int(kernel[6:8]))
            assert (job.priority, job.order) == expected, job.label


def test_only_a_kernel_meeting_the_termination_criterion_pauses_launches(tmp_path, relu_task):
    from specgen.config import TerminationConfig

    evaluator = PacedEvaluator(bench_s=0.02, ncu_s=0.2, speedups={"i000-t01-s0": 1.5, "i000-t05-s0": 3.0})
    earlier = KernelOutcome("old", -1, "main", "", status="success", valid=True, speedup=2.0)  # the bar: 2.0 x 1.03
    termination = TerminationConfig(enabled=True, criteria="best", margin=0.03, confirm=1, grace_ratio=None)
    ctx, loop, backend, events = run_paced(tmp_path, relu_task, evaluator, deltas=40, pause=0.02, history=[earlier],
                                           eval_cfg=EvalConfig(gpu_type="H100", ncu_top_n=2, ncu_wait_s=60),
                                           termination=termination, best_ncu_wait_s=5)
    timed = {e["kernel"]: e["speedup"] for e in events if e["type"] == "kernel_benchmarked"}
    assert timed["i000-t01-s0"] == 1.5  # the iteration's best for a while, below the bar: no pause
    assert [e["kernel"] for e in events if e["type"] == "spec_paused"] == ["i000-t05-s0"]
    assert next(e for e in events if e["type"] == "iteration_terminated")["kernel"] == "i000-t05-s0"
    assert next(e for e in events if e["type"] == "spec_resumed")["reason"] in ("iteration end", "ncu finished")


def test_a_new_best_pauses_launches_until_its_profile_is_in(tmp_path, relu_task):
    evaluator = PacedEvaluator(bench_s=0.02, ncu_s=0.3, speedups={"i000-t01-s0": 1.5})
    eval_cfg = EvalConfig(gpu_type="H100", ncu_top_n=1, ncu_wait_s=60)
    ctx, loop, backend, events = run_paced(tmp_path, relu_task, evaluator, deltas=60, pause=0.02, eval_cfg=eval_cfg,
                                           best_ncu_wait_s=5)
    paused = next(e for e in events if e["type"] == "spec_paused")
    resumed = next(e for e in events if e["type"] == "spec_resumed" and e["kernel"] == paused["kernel"])
    assert paused["kernel"] == "i000-t01-s0" and resumed["reason"] == "ncu finished" and resumed["waited_s"] >= 0.25
    during = [e for e in events if e["type"] == "spec_triggered" and paused["t"] < e["t"] < resumed["t"]]
    assert not during
    after = [p for mode, m, _ in backend.calls if mode == "spec" for p in [m[-1]["content"]]
             if "the best so far; Nsight Compute: SM 62%" in p]
    assert after  # the next specs see the new best's profile
    kernel = next(json.loads(line) for line in (ctx.layout.run_dir / "kernels.jsonl").read_text().splitlines()
                  if json.loads(line)["kernel_id"] == "i000-t01-s0")
    assert kernel["ncu"]["urgent"] is True
    summary = json.loads((ctx.layout.run_dir / "iterations.jsonl").read_text().splitlines()[0])
    assert summary["spec_schedule"] == "demand" and summary["spec_paused_s"] >= 0.25
