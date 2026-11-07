import asyncio
import json

from conftest import CORRECT_RELU, WRONG_RELU, ScriptedBackend, answer, make_context

from specgen.config import EvalConfig, PoolConfig, RunConfig, SpecConfig, SpecGenConfig
from specgen.control.loop import IterationLoop
from specgen.control.orchestrator import run_async
from specgen.llm.events import ContentDelta, StreamEnd
from specgen.tasks.base import KernelBenchTask

FAST_EVAL = EvalConfig(gpu_type="H100", ncu_top_n=0, correctness_trials=2, bench_warmup=1, bench_iters=3,
                       bench_rounds=2)


def run_with_context(ctx, task_path):
    async def go():
        await ctx.pool.start()
        try:
            return await IterationLoop(KernelBenchTask.from_path(task_path), ctx).run()
        finally:
            await ctx.pool.close()
            ctx.recorder.close()

    return asyncio.run(go())


def run_loop(tmp_path, task_path, scripts, iterations):
    backend = ScriptedBackend(scripts)
    ctx = make_context(tmp_path / "out", backend, max_iterations=iterations, tasks=[task_path])
    return run_with_context(ctx, task_path), backend, ctx.layout


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_two_iterations(tmp_path, relu_task):
    history, backend, layout = run_loop(tmp_path, relu_task, [answer(CORRECT_RELU), answer(WRONG_RELU)], 2)
    first, second = history.outcomes
    assert first.status == "success" and first.speedup > 0 and first.valid
    assert second.status == "numerical_error" and not second.valid and "mismatch" in second.error
    assert history.best() is first

    second_prompt = backend.calls[1][1]["content"]
    assert "Best kernel so far" in second_prompt and "torch.clamp_min(x, 0.0)" in second_prompt
    assert "do not resubmit it unchanged; make every effort to try new optimizations" in second_prompt
    assert "Fastest kernels from the previous iteration" not in second_prompt

    task_dir = layout.run_dir / "level0__1_ReLU"
    kernel_dir = task_dir / "iter_000" / "kernels" / "i000-main"
    for name in ("kernel.py", "validate.json", "benchmark.json", "validate.log"):
        assert (kernel_dir / name).exists(), name
    assert (task_dir / "iter_000" / "main" / "reasoning.txt").read_text() == "plan the kernel"
    assert json.loads((task_dir / "iter_001" / "prompt.json").read_text())["sections"]["best"]

    kernels = read_jsonl(layout.run_dir / "kernels.jsonl")
    assert [k["status"] for k in kernels] == ["success", "numerical_error"]
    assert kernels[0]["validation"]["ok"] and kernels[0]["benchmark"]["speedup"] > 0
    assert [i["iteration"] for i in read_jsonl(layout.run_dir / "iterations.jsonl")] == [0, 1]
    events = {e["type"] for e in read_jsonl(layout.run_dir / "events.jsonl")}
    assert {"iteration_started", "job_queued", "job_started", "job_finished", "kernel_finished"} <= events


def test_parse_error_is_fed_back(tmp_path, relu_task):
    no_code = [ContentDelta("I cannot write this."), StreamEnd("stop")]
    history, backend, _ = run_loop(tmp_path, relu_task, [no_code, answer(CORRECT_RELU)], 2)
    assert [o.status for o in history.outcomes] == ["parse_error", "success"]
    second_prompt = backend.calls[1][1]["content"]
    assert "no correct kernel" in second_prompt and "Result: parse_error" in second_prompt


def test_orchestrator_end_to_end(tmp_path, relu_task, linear_task):
    from specgen.config import TerminationConfig

    cfg = SpecGenConfig(  # the baseline: no spec generation, so the termination criterion is ignored
        pool=PoolConfig(gpus=[], validations_per_gpu=2),
        eval=FAST_EVAL,
        spec=SpecConfig(enabled=False),
        termination=TerminationConfig(enabled=True, criteria="best"),
        run=RunConfig(tasks=[relu_task, linear_task], output_root=tmp_path / "runs", run_id="e2e", max_iterations=1,
                      max_concurrent_tasks=1),
    )
    linear_ok = linear_task.read_text().replace("torch.relu(self.fc(x))", "torch.clamp_min(self.fc(x), 0.0)")
    backend = ScriptedBackend([answer(CORRECT_RELU), answer(linear_ok)])
    results = asyncio.run(run_async(cfg, backend=backend))
    assert all(h.best() is not None for h in results.values()), results
    assert backend.closed
    run_dir = tmp_path / "runs" / "e2e"
    resolved = json.loads((run_dir / "config.json").read_text())["resolved"]
    assert resolved["gpu_type"] == "H100" and resolved["build_jobs"] >= 2 and resolved["termination"] is None
    assert len(read_jsonl(run_dir / "kernels.jsonl")) == 2
    events = read_jsonl(run_dir / "events.jsonl")
    assert not any(e["type"] in ("iteration_terminated", "termination_check") for e in events)
