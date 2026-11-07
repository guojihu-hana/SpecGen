import asyncio
import json
import re
from types import SimpleNamespace

from conftest import CORRECT_RELU, ModeBackend, answer, make_context

from specgen.control.budget import SpecBudget
from specgen.control.loop import IterationLoop
from specgen.control.orchestrator import resolve_build_arch, resolve_build_workers
from specgen.control.triggers import TriggerConfig, TriggerGroup
from specgen.evaluation.executor import GpuSlot, LocalEvaluator
from specgen.evaluation.jobs import EvalJob
from specgen.generation.generator import GenerationResult
from specgen.resources.pool import Device, PoolStats, make_devices
from specgen.tasks.base import KernelBenchTask


class FakePool:
    allocation = "fixed"

    def __init__(self, slots, run_s, load=0):
        self.devices = [SimpleNamespace(role=r) for r, n in slots.items() for _ in range(n if r == "profiler" else 1)]
        self._slots, self._run_s, self._load = slots, run_s, load

    def stats(self):
        return PoolStats(slots=self._slots, busy={}, queued={}, utilization={}, slot_busy_s={})

    def expected_run_s(self, kind):
        return self._run_s[kind]

    def load(self, task_id=None, kinds=()):
        return self._load


def recorder(ok, total):
    return SimpleNamespace(outcomes={"validate": [ok, total]})


TRIGGERS = SimpleNamespace(min_gap_tokens=500, after_tokens=500)


def test_adaptive_k_follows_pipeline_capacity():
    budget = SpecBudget(k_min=1, k_max=8)
    # 8 validation slots at 20 s, 2 profilers at 10 s, 50% valid, 16 builders at 40 s:
    # capacity = min(0.4, 0.4, 0.4) = 0.4 kernels/s; horizon = 500 / 100 + 30 = 35 s -> 14 kernels
    pool = FakePool({"validator": 8, "profiler": 2, "builder": 16},
                    {"build": 40.0, "validate": 20.0, "benchmark": 10.0})
    k, info = budget.decide("t", pool=pool, recorder=recorder(1, 2), triggers=TRIGGERS, tokens_per_s=100.0)
    assert info["policy"] == "adaptive" and k == 8 and info["raw_k"] > 8
    assert budget.inflight == 8 and budget.task_inflight["t"] == 8

    # what is already in flight counts against the target
    small = FakePool({"validator": 1, "profiler": 1}, {"build": 40.0, "validate": 60.0, "benchmark": 15.0}, load=1)
    k2, info2 = SpecBudget().decide("u", pool=small, recorder=recorder(0, 0), triggers=TRIGGERS, tokens_per_s=100.0)
    assert k2 == 1 and info2["raw_k"] < 1

    for _ in range(8):
        budget.release("t")
    assert budget.inflight == 0 and budget.task_inflight["t"] == 0


def test_more_tasks_share_the_capacity():
    pool = FakePool({"validator": 8, "profiler": 2}, {"build": 40.0, "validate": 20.0, "benchmark": 10.0})
    one, many = SpecBudget(k_max=100), SpecBudget(k_max=100)
    many.active_tasks = 4
    k1, _ = one.decide("t", pool=pool, recorder=recorder(1, 2), triggers=TRIGGERS, tokens_per_s=100.0)
    k4, _ = many.decide("t", pool=pool, recorder=recorder(1, 2), triggers=TRIGGERS, tokens_per_s=100.0)
    assert k4 < k1


def test_spec_statistics():
    budget = SpecBudget(k=3)
    budget.observe(GenerationResult(ok=True, seconds=10.0, code="x"))
    budget.observe(GenerationResult(ok=True, seconds=20.0, code=None))
    assert budget.decide("t")[0] == 3
    assert budget.generation_s.value is not None and 0.5 < budget.parse_rate() < 0.8


def test_build_workers_and_arch():
    assert resolve_build_workers(None, []) == 0 and resolve_build_workers(3, ["0"]) == 3
    assert 1 <= resolve_build_workers(None, ["0"]) <= 64
    devices = make_devices(["0", "1"], build_workers=2)
    assert resolve_build_arch({"0": "9.0a", "1": "9.0a"}, devices) == "9.0a"
    assert resolve_build_arch({}, devices) is None


def test_reference_outputs_are_cached(tmp_path, relu_task):
    kernel = tmp_path / "k" / "kernel.py"
    kernel.parent.mkdir()
    kernel.write_text(CORRECT_RELU)
    params = {"seed": 0, "trials": 2, "ref_cache": str(tmp_path / "ref_cache")}
    evaluator = LocalEvaluator(timeout_s=120)

    def validate():
        job = EvalJob("validate", "kernelbench", str(relu_task), str(kernel), str(kernel.parent), dict(params))
        return asyncio.run(evaluator.run(job, GpuSlot(None, "validator")))

    first, second = validate(), validate()
    assert first["ok"] and second["ok"]
    assert first["ref_cached"] == 0 and second["ref_cached"] == 2
    assert len(list((tmp_path / "ref_cache").glob("cpu_seed0_trial*.pt"))) == 2


def run_loop(ctx, task_path):
    async def go():
        await ctx.pool.start()
        try:
            return await IterationLoop(KernelBenchTask.from_path(task_path), ctx).run()
        finally:
            await ctx.pool.close()
            ctx.recorder.close()

    return asyncio.run(go())


def test_builder_compiles_before_validation(tmp_path, relu_task):
    backend = ModeBackend(main=[answer(CORRECT_RELU)])
    ctx = make_context(tmp_path / "out", backend, devices=make_devices([], build_workers=1), tasks=[relu_task])
    history = run_loop(ctx, relu_task)
    assert history.outcomes[0].status == "success"
    kernel_dir = ctx.layout.kernel_dir(KernelBenchTask.from_path(relu_task), 0, "i000-main")
    assert json.loads((kernel_dir / "build.json").read_text())["status"] == "built"
    record = json.loads((ctx.layout.run_dir / "kernels.jsonl").read_text().splitlines()[0])
    assert record["build"]["slot"] == "builder:cpu#0" and record["validation"]["ok"]


def test_build_failure_skips_the_gpu(tmp_path, relu_task):
    backend = ModeBackend(main=[answer("import torch\nclass Model(:\n    pass")])
    ctx = make_context(tmp_path / "out", backend, devices=make_devices([], build_workers=1), tasks=[relu_task])
    history = run_loop(ctx, relu_task)
    assert history.outcomes[0].status == "compile_error" and "SyntaxError" in history.outcomes[0].error
    kernel_dir = ctx.layout.kernel_dir(KernelBenchTask.from_path(relu_task), 0, "i000-main")
    assert not (kernel_dir / "validate.json").exists()


class OomOnceEvaluator:
    def __init__(self):
        self.calls = []

    async def run(self, job, slot):
        self.calls.append(job.kind)
        if job.kind == "validate" and self.calls.count("validate") == 1:
            return {"ok": False, "status": "runtime_error", "error": "torch.OutOfMemoryError: CUDA out of memory.",
                    "concurrency": 3}
        if job.kind == "validate":
            return {"ok": True, "status": "success"}
        return {"ok": True, "status": "success", "speedup": 1.5, "latency_ms": 1.0, "ref_latency_ms": 1.5}


def test_out_of_memory_validation_is_retried_alone(tmp_path, relu_task):
    evaluator = OomOnceEvaluator()
    ctx = make_context(tmp_path / "out", ModeBackend(main=[answer(CORRECT_RELU)]), evaluator=evaluator,
                       tasks=[relu_task])
    history = run_loop(ctx, relu_task)
    assert evaluator.calls == ["validate", "validate", "benchmark"]
    assert history.outcomes[0].status == "success" and history.outcomes[0].speedup == 1.5
    record = json.loads((ctx.layout.run_dir / "kernels.jsonl").read_text().splitlines()[0])
    assert record["validation_shared"]["status"] == "runtime_error" and record["validation"]["ok"]


def one_kernel_profile(**rule):
    return {"kernels": [{"name": "relu_kernel", "launches": 1, "share": 1.0, "rules": [rule] if rule else [],
                         "metrics": {"sm__throughput.avg.pct_of_peak_sustained_elapsed": 61.7}}]}


class NcuEvaluator:
    def __init__(self, speedups):
        self.speedups, self.ncu = speedups, []

    async def run(self, job, slot):
        kernel_id = job.work_dir.rsplit("/", 1)[-1]
        if job.kind == "ncu":
            self.ncu.append(kernel_id)
            return {"ok": True, "status": "success", "profile": one_kernel_profile(), "kernels": 1, "rules": 0}
        if job.kind == "benchmark":
            return {"ok": True, "status": "success", "speedup": self.speedups[kernel_id], "latency_ms": 1.0,
                    "ref_latency_ms": 1.0}
        return {"ok": True, "status": "success"}


def test_ncu_profiles_the_fastest_kernels(tmp_path, relu_task):
    from specgen.config import EvalConfig
    from specgen.llm.events import ContentDelta, ReasoningDelta, StreamEnd
    from conftest import relu_kernel, spec_answer

    main = [ReasoningDelta("tile size"), ContentDelta(f"```python\n{CORRECT_RELU}\n```"), StreamEnd("stop")]
    specs = [spec_answer(relu_kernel(f"torch.clamp_min(x, 0.0) + {i} * 0")) for i in range(3)]
    triggers = TriggerConfig(groups=(TriggerGroup("kw", (re.compile("tile size"),)),), after_tokens=0,
                             fallback_tokens=10**6, min_gap_tokens=0)
    evaluator = NcuEvaluator({"i000-main": 1.0, "i000-t01-s0": 3.0, "i000-t01-s1": 2.0, "i000-t01-s2": 0.5})
    eval_cfg = EvalConfig(gpu_type="H100", ncu_top_n=2)
    ctx = make_context(tmp_path / "out", ModeBackend(main=[main], spec=specs), triggers=triggers, k=3,
                       evaluator=evaluator, eval_cfg=eval_cfg, tasks=[relu_task],
                       devices=[Device(None, "validator", 4), Device(None, "profiler", 1)])

    async def go():
        await ctx.pool.start()
        loop = IterationLoop(KernelBenchTask.from_path(relu_task), ctx)
        try:
            await loop.run_iteration(0)
            await asyncio.gather(*loop._ncu_tasks.values(), return_exceptions=True)
            return loop
        finally:
            await ctx.pool.close()
            ctx.recorder.close()

    loop = asyncio.run(go())
    assert set(evaluator.ncu) <= {"i000-main", "i000-t01-s0", "i000-t01-s1"}
    assert "i000-t01-s0" in evaluator.ncu and "i000-t01-s2" not in evaluator.ncu
    best = loop.history.best()
    assert best.kernel_id == "i000-t01-s0" and best.ncu_profile == one_kernel_profile()


class SlowNcuEvaluator(NcuEvaluator):
    PROFILE = one_kernel_profile(section="SourceCounters", rule="UncoalescedGlobalAccess", type="OPT", speedup=40.0,
                                 speedup_type="global", text="This kernel has uncoalesced global accesses.")
    delay = 0.5  # still running when the iteration ends

    async def run(self, job, slot):
        if job.kind == "ncu":
            await asyncio.sleep(self.delay)
            self.ncu.append(job.work_dir.rsplit("/", 1)[-1])
            return {"ok": True, "status": "success", "profile": self.PROFILE, "kernels": 1, "rules": 1}
        return await super().run(job, slot)


def test_the_next_prompt_waits_for_the_ncu_analysis_of_the_best_kernel(tmp_path, relu_task):
    from specgen.config import EvalConfig
    from conftest import relu_kernel

    evaluator = SlowNcuEvaluator({"i000-main": 1.5, "i001-main": 1.2})
    eval_cfg = EvalConfig(gpu_type="H100", ncu_top_n=1, ncu_wait_s=60)
    backend = ModeBackend(main=[answer(CORRECT_RELU), answer(relu_kernel("torch.relu(x)"))])
    ctx = make_context(tmp_path / "out", backend, evaluator=evaluator, eval_cfg=eval_cfg, tasks=[relu_task],
                       max_iterations=2, history_prompt="summary",
                       devices=[Device(None, "validator", 4), Device(None, "profiler", 1)])
    run_loop(ctx, relu_task)
    task = KernelBenchTask.from_path(relu_task)
    prompt = json.loads((ctx.layout.iteration_dir(task, 1) / "prompt.json").read_text())
    assert "Nsight Compute profile of one forward pass (1 kernel;" in prompt["sections"]["best"]
    assert "throughput (% of peak): SM 62%" in prompt["sections"]["best"]
    assert "[OPT, est. speedup 40%] UncoalescedGlobalAccess" in prompt["sections"]["best"]
    events = [json.loads(line) for line in (ctx.layout.run_dir / "events.jsonl").read_text().splitlines()]
    waits = [e for e in events if e["type"] == "ncu_wait"]
    assert len(waits) == 1 and waits[0]["finished"] == 1 and waits[0]["iteration"] == 1


def test_the_final_best_kernel_gets_its_ncu_analysis_before_the_run_ends(tmp_path, relu_task):
    from specgen.config import EvalConfig

    evaluator = SlowNcuEvaluator({"i000-main": 1.5})
    ctx = make_context(tmp_path / "out", ModeBackend(main=[answer(CORRECT_RELU)]), evaluator=evaluator,
                       eval_cfg=EvalConfig(gpu_type="H100", ncu_top_n=1, ncu_wait_s=60), tasks=[relu_task],
                       max_iterations=1, devices=[Device(None, "validator", 4), Device(None, "profiler", 1)])
    history = run_loop(ctx, relu_task)
    assert evaluator.ncu == ["i000-main"] and history.best().ncu_profile == SlowNcuEvaluator.PROFILE
    kernel_dir = ctx.layout.kernel_dir(KernelBenchTask.from_path(relu_task), 0, "i000-main")
    assert json.loads((kernel_dir / "ncu.json").read_text())["rules"] == 1
    events = [json.loads(line) for line in (ctx.layout.run_dir / "events.jsonl").read_text().splitlines()]
    assert [e["iteration"] for e in events if e["type"] == "ncu_wait"] == [1]
    assert [e["rules"] for e in events if e["type"] == "kernel_ncu"] == [1]


def test_which_kernels_get_profiled_follows_the_history_prompt(tmp_path, relu_task):
    from conftest import relu_kernel
    from specgen.control.history import KernelOutcome
    from specgen.control.loop import Candidate, IterationState
    from specgen.config import EvalConfig

    vectorized = CORRECT_RELU + '\n_src = "__global__ void k(float4* x) {}"\n'  # another approach (code features)
    earlier = KernelOutcome("i000-main", 0, "main", relu_kernel("torch.relu(x) + 0"), status="success", valid=True,
                            speedup=1.5)

    def wanted(mode, top_n=6):
        ctx = make_context(tmp_path / mode, ModeBackend(), tasks=[relu_task], history_prompt=mode, feedback_top_k=1,
                           eval_cfg=EvalConfig(gpu_type="H100", ncu_top_n=top_n))
        loop = IterationLoop(KernelBenchTask.from_path(relu_task), ctx)
        loop.history.add(earlier)
        state = IterationState(1, 0, None)
        for kid, code, speedup in [("a", vectorized, 3.0), ("b", CORRECT_RELU, 2.0), ("c", CORRECT_RELU, 1.8),
                                   ("d", vectorized, 1.0), ("e", CORRECT_RELU, 1.2)]:
            outcome = KernelOutcome(kid, 1, "spec", code, status="success", valid=True, speedup=speedup)
            state.candidates.append(Candidate(kid, "spec", outcome, {}, tmp_path))
        return [c.kernel_id for c in loop._ncu_wanted(state)]

    # a: new best (and its approach's best); b: beats the plain approach's earlier best (1.5x)
    assert wanted("summary") == ["a", "b"]
    assert wanted("topk") == ["a", "b"]  # the best and the next feedback_top_k
    assert wanted("topk+summary", top_n=1) == ["a"]


def test_modules_calling_a_library_gemm_are_rejected_before_the_build(tmp_path, relu_task):
    cheat = CORRECT_RELU.replace("return torch.clamp_min(x, 0.0)", "return torch.clamp_min(x @ torch.eye(4), 0.0)")
    ctx = make_context(tmp_path / "out", ModeBackend(main=[answer(cheat)]), devices=make_devices([], build_workers=1),
                       tasks=[relu_task])
    outcome = run_loop(ctx, relu_task).outcomes[0]
    assert outcome.status == "forbidden" and outcome.error.endswith(": the @ operator")
    kernel_dir = ctx.layout.kernel_dir(KernelBenchTask.from_path(relu_task), 0, "i000-main")
    assert not (kernel_dir / "build.json").exists()


class LibraryKernelEvaluator(NcuEvaluator):
    async def run(self, job, slot):
        if job.kind == "validate":
            return {"ok": True, "status": "success", "kernels": [
                {"name": "nvjet_sm103_sss_tf32_128x256_32x6_2x2f_2", "calls": 1, "us": 180.0},
                {"name": "void at::native::vectorized_elementwise_kernel<4>(int)", "calls": 1, "us": 3.0}]}
        if job.kind == "benchmark":
            raise AssertionError("a module that runs library kernels must not be timed")
        return await super().run(job, slot)


def test_modules_whose_forward_runs_library_kernels_are_rejected(tmp_path, relu_task):
    ctx = make_context(tmp_path / "out", ModeBackend(main=[answer(CORRECT_RELU)]),
                       evaluator=LibraryKernelEvaluator({}), tasks=[relu_task])
    outcome = run_loop(ctx, relu_task).outcomes[0]
    assert outcome.status == "forbidden" and not outcome.valid and outcome.speedup is None
    assert outcome.error == "runs library kernels instead of hand-written ones: nvjet_sm103_sss_tf32_128x256_32x6_2x2f_2"
    record = json.loads((ctx.layout.run_dir / "kernels.jsonl").read_text().splitlines()[0])
    assert [k["name"][:5] for k in record["forward_kernels"]] == ["nvjet", "void "]


def test_a_prompt_that_cannot_wait_longer_raises_an_alert(tmp_path, relu_task):
    from specgen.config import EvalConfig
    from conftest import relu_kernel

    evaluator = SlowNcuEvaluator({"i000-main": 1.5, "i001-main": 1.2})
    evaluator.delay = 1.0
    ctx = make_context(tmp_path / "out", ModeBackend(main=[answer(CORRECT_RELU), answer(relu_kernel("torch.relu(x)"))]),
                       evaluator=evaluator, eval_cfg=EvalConfig(gpu_type="H100", ncu_top_n=1, ncu_wait_s=0.1),
                       tasks=[relu_task], max_iterations=2,
                       devices=[Device(None, "validator", 4), Device(None, "profiler", 1)])
    run_loop(ctx, relu_task)
    events = [json.loads(line) for line in (ctx.layout.run_dir / "events.jsonl").read_text().splitlines()]
    alerts = [e for e in events if e["type"] == "ncu_alert"]
    assert alerts and alerts[0]["kind"] == "wait_cap" and alerts[0]["iteration"] == 1


def test_a_module_of_comments_only_is_not_evaluated(tmp_path, relu_task):
    from specgen.generation.extract import has_code

    assert not has_code("# APPROACH: FP16 mma.sync, 128x128x32\n\n# more") and has_code("# APPROACH: x\nimport torch")
    backend = ModeBackend(main=[answer("# APPROACH: FP16 mma.sync GEMM, 128x128x32 tile")])
    ctx = make_context(tmp_path / "out", backend, devices=make_devices([], build_workers=1), tasks=[relu_task])
    history = run_loop(ctx, relu_task)
    outcome = history.outcomes[0]
    assert outcome.status == "parse_error" and outcome.error == "the module has only comments, no code"
    kernel_dir = ctx.layout.kernel_dir(KernelBenchTask.from_path(relu_task), 0, "i000-main")
    assert not (kernel_dir / "kernel.py").exists() and not (kernel_dir / "build.json").exists()
