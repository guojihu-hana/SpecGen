import asyncio
import json

import pytest
from conftest import CORRECT_RELU, ModeBackend, ScriptedBackend, answer, make_context

from specgen.analysis.data import load_run
from specgen.config import EvalConfig, PoolConfig, RunConfig, SpecConfig, SpecGenConfig
from specgen.control.compliance import forbidden_usage
from specgen.control.loop import IterationLoop
from specgen.control.orchestrator import RunSetupError, run_async
from specgen.control.triggers import TriggerConfig
from specgen.tasks.base import KernelBenchTask

FAST_EVAL = EvalConfig(gpu_type="H100", ncu_top_n=0, correctness_trials=2, bench_warmup=1, bench_iters=3,
                       bench_rounds=2)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def baseline_config(tmp_path, tasks, iterations, resume=False):
    return SpecGenConfig(pool=PoolConfig(gpus=[], validations_per_gpu=2), eval=FAST_EVAL, spec=SpecConfig(enabled=False),
                         run=RunConfig(tasks=tasks, output_root=tmp_path / "runs", run_id="r", max_iterations=iterations,
                                       max_concurrent_tasks=1, resume=resume))


def test_a_resumed_run_continues_with_its_history(tmp_path, relu_task, linear_task):
    first = ScriptedBackend([answer(CORRECT_RELU)])
    asyncio.run(run_async(baseline_config(tmp_path, [relu_task], 1), backend=first))
    run = tmp_path / "runs" / "r"
    events = run / "events.jsonl"  # as if the run had ended an hour ago
    events.write_text("".join(json.dumps({**e, "t": e["t"] - 3600}) + "\n" for e in jsonl(events)))
    task_dir = run / KernelBenchTask.from_path(relu_task).dir_name  # and was killed during iteration 1
    (task_dir / "iter_001" / "kernels" / "i001-main").mkdir(parents=True)
    (task_dir / "iter_001" / "kernels" / "i001-main" / "ncu.json").write_text("{}")
    with (run / "kernels.jsonl").open("a") as f:
        f.write(json.dumps({"task_id": "level0/1_ReLU", "kernel_id": "i001-t01-s0", "iteration": 1,
                            "source": "spec", "status": "success", "valid": True, "speedup": 50.0}) + "\n")
        f.write('{"task_id": "level0/1_ReLU", "kernel_id": "i001-t02')  # the line it was writing

    second = ScriptedBackend([answer(CORRECT_RELU.replace("clamp_min(x, 0.0)", "clamp(x, min=0.0)"))])
    results = asyncio.run(run_async(baseline_config(tmp_path, [relu_task], 2, resume=True), backend=second))
    assert results["level0/1_ReLU"].best() is not None and len(second.calls) == 1
    prompt = second.calls[0][1]["content"]
    assert "Best kernel so far" in prompt and "torch.clamp_min(x, 0.0)" in prompt  # iteration 0's kernel
    assert [k["kernel_id"] for k in jsonl(run / "kernels.jsonl")] == ["i000-main", "i001-main"]
    assert [k["kernel_id"] for k in jsonl(run / "kernels.interrupted1.jsonl")] == ["i001-t01-s0"]
    assert (task_dir / "iter_001.interrupted1" / "kernels" / "i001-main" / "ncu.json").is_file()
    assert not (task_dir / "iter_001" / "kernels" / "i001-main" / "ncu.json").exists()
    assert results["level0/1_ReLU"].best().speedup < 50  # the interrupted iteration's kernels are not history
    assert [i["iteration"] for i in jsonl(run / "iterations.jsonl")] == [0, 1]
    assert json.loads((run / "config.json").read_text())["run"]["max_iterations"] == 1  # the original stays
    assert json.loads((run / "config.resume1.json").read_text())["run"]["max_iterations"] == 2
    started = [e for e in jsonl(events) if e["type"] == "run_started"]
    assert [bool(e.get("resumed")) for e in started] == [False, True] and started[1]["previous_end"]
    assert started[1]["interrupted_kernels"] == 1
    data = load_run(run)
    assert len(data.resumes) == 1 and data.resumes[0]["gap_s"] > 3500 and data.t_end - data.t0 < 600
    assert [s["iteration"] for s in data.spans] == [0, 1]

    with pytest.raises(RunSetupError, match="tasks of the run"):
        asyncio.run(run_async(baseline_config(tmp_path, [relu_task, linear_task], 3, resume=True),
                              backend=ScriptedBackend([])))


def test_analysis_ends_the_interrupted_session(tmp_path):
    task = "level0/1_ReLU"

    def ev(t, type, **fields):
        return {"t": t, "type": type, "task": task, **fields}

    def job(t, step, kernel, **fields):
        return ev(t, f"job_{step}", label=f"{task}/{kernel}/validate", kind="validate", **fields)

    events = [
        ev(0, "run_started"), ev(1, "iteration_started", iteration=0), ev(1, "main_started", iteration=0),
        job(2, "queued", "i000-main"), job(3, "started", "i000-main"), job(4, "finished", "i000-main", outcome="done"),
        ev(5, "kernel_benchmarked", iteration=0, kernel="i000-main", source="main", speedup=1.5),
        ev(6, "iteration_finished", iteration=0, selected="i000-main"),
        ev(7, "iteration_started", iteration=1), ev(7, "main_started", iteration=1),
        ev(8, "main_progress", iteration=1, tokens=700),
        ev(8, "spec_started", iteration=1, kernel="i001-t01-s0", trigger=1),
        job(9, "queued", "i001-t01-s0"), job(9, "queued", "i001-t02-s0"), job(10, "started", "i001-t01-s0"),
        ev(11, "kernel_benchmarked", iteration=1, kernel="i001-t03-s0", source="spec", speedup=9.0),
        # killed here, resumed an hour later: iteration 1 runs again
        ev(3611, "run_started", resumed=True, previous_end=11),
        ev(3612, "iteration_started", iteration=1), ev(3612, "main_started", iteration=1),
        job(3613, "queued", "i001-t01-s0"), job(3614, "started", "i001-t01-s0"),
        job(3615, "finished", "i001-t01-s0", outcome="done"),
        ev(3616, "main_finished", iteration=1, ok=True), ev(3617, "iteration_finished", iteration=1, selected="i000-main"),
        ev(3618, "run_finished"),
    ]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    data = load_run(tmp_path)
    assert len(data.resumes) == 1 and data.resumes[0]["gap_s"] > 3500
    validate = [j for j in data.jobs if j["label"].endswith("i001-t01-s0/validate")]
    assert [j["outcome"] for j in validate] == ["interrupted", "done"]
    assert validate[0]["end"] == 11 and validate[1]["end"] - validate[1]["queued"] == 2  # its own queue time
    assert sorted((s["iteration"], s["end"] - s["start"]) for s in data.spans) == [(0, 5), (1, 5)]
    assert data.interrupted == [{"task": task, "iteration": 1, "start": 7, "end": 11, "specs": 1, "main_tokens": 700}]
    assert [b["kernel"] for b in data.benchmarks] == ["i000-main"]  # the discarded iteration's timing is gone
    assert sorted(m["iteration"] for m in data.mains) == [0, 1] and not data.specs


def test_restore_rebuilds_code_results_profiles_and_errors(tmp_path, relu_task):
    ctx = make_context(tmp_path / "out", ModeBackend(), triggers=TriggerConfig(), schedule="demand", tasks=[relu_task])
    task = KernelBenchTask.from_path(relu_task)
    run = ctx.layout.run_dir
    library = CORRECT_RELU.replace("torch.clamp_min(x, 0.0)", "torch.matmul(x, x)")
    records = [
        {"task_id": task.task_id, "kernel_id": "i000-main", "iteration": 0, "source": "main", "status": "success",
         "valid": True, "speedup": 2.0, "benchmark": {"latency_ms": 0.5, "ref_latency_ms": 1.0}},
        {"task_id": task.task_id, "kernel_id": "i000-t01-s0", "iteration": 0, "source": "spec", "status": "forbidden",
         "valid": False},
        {"task_id": task.task_id, "kernel_id": "i000-t02-s0", "iteration": 0, "source": "spec", "status": "compile_error",
         "valid": False},
        {"task_id": task.task_id, "kernel_id": "i001-t01-s0", "iteration": 1, "source": "spec", "status": "success",
         "valid": True, "speedup": 9.0},  # iteration 1 never finished: it runs again
    ]
    (run / "kernels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (run / "iterations.jsonl").write_text(json.dumps({"task": task.task_id, "iteration": 0, "selected": "i000-main"}) + "\n")
    main = ctx.layout.kernel_dir(task, 0, "i000-main")
    main.mkdir(parents=True)
    (main / "kernel.py").write_text(CORRECT_RELU)
    profile = {"kernels": [{"name": "relu_kernel", "share": 1.0, "metrics": {}, "rules": []}]}
    (main / "ncu.json").write_text(json.dumps({"ok": True, "profile": profile}))
    forbidden = ctx.layout.kernel_dir(task, 0, "i000-t01-s0")  # rejected before kernel.py was written
    forbidden.mkdir(parents=True)
    (forbidden / "content.txt").write_text(f"```python\n# APPROACH: library call\n{library}\n```")
    broken = ctx.layout.kernel_dir(task, 0, "i000-t02-s0")
    broken.mkdir(parents=True)
    (broken / "kernel.py").write_text("import torch\nclass Model: pass\n")
    (broken / "build.json").write_text(json.dumps({"ok": False, "status": "compile_error",
                                                    "error": 'cuda.cu(3): error: identifier "x" is undefined'}))

    loop = IterationLoop(task, ctx)
    assert loop.restore() == 1 and loop.start_iteration == 1 and loop.epoch == 1
    outcomes = {o.kernel_id: o for o in loop.history.outcomes}
    assert set(outcomes) == {"i000-main", "i000-t01-s0", "i000-t02-s0"}
    assert outcomes["i000-main"].ncu_profile == profile and outcomes["i000-main"].latency_ms == 0.5
    assert loop.history.best().kernel_id == "i000-main" and loop.history.selected[0].kernel_id == "i000-main"
    assert "torch.matmul" in outcomes["i000-t01-s0"].code
    assert outcomes["i000-t01-s0"].error == forbidden_usage(outcomes["i000-t01-s0"].code)
    assert outcomes["i000-t02-s0"].error == 'cuda.cu(3): error: identifier "x" is undefined'
    assert (loop.pass_rate.passed, loop.pass_rate.finished) == (0, 2)
    ctx.recorder.close()
