import asyncio
import json
import re
from pathlib import Path

import pytest
from conftest import CORRECT_RELU, ModeBackend, make_context, spec_answer

from specgen.analysis.data import RunData, load_run, stage_stats
from specgen.analysis.report import (best_curve, build_payload, compare_payload, main as report_main, render,
                                     split_intro)
from specgen.analysis.summary import format_summary, main as summary_main, summarize
from specgen.config import TerminationConfig
from specgen.control.loop import IterationLoop
from specgen.control.triggers import TriggerConfig, TriggerGroup
from specgen.llm.events import ContentDelta, ReasoningDelta, StreamEnd
from specgen.tasks.base import KernelBenchTask


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("analysis")
    task_path = tmp / "level0" / "1_ReLU.py"
    task_path.parent.mkdir()
    from conftest import RELU_TASK

    task_path.write_text(RELU_TASK)
    main = [ReasoningDelta("Plan: one thread per element, then pick the tile size carefully. "), 60,
            ContentDelta("never"), StreamEnd("stop")]
    triggers = TriggerConfig(groups=(TriggerGroup("tiling", (re.compile("tile size"),)),), after_tokens=0,
                             fallback_tokens=10**6, min_gap_tokens=0)
    ctx = make_context(tmp / "out", ModeBackend(main=[main], spec=[spec_answer(CORRECT_RELU)]), triggers=triggers,
                       k=1, tasks=[task_path],
                       termination=TerminationConfig(enabled=True, criteria="first_valid", grace_ratio=0.05))

    async def go():
        await ctx.pool.start()
        try:
            await IterationLoop(KernelBenchTask.from_path(task_path), ctx).run()
        finally:
            await ctx.pool.close()
            ctx.recorder.close()

    asyncio.run(go())
    return ctx.layout.run_dir


def test_load_run_derives_the_timeline(run_dir):
    data = load_run(run_dir)
    assert data.tasks == ["level0/1_ReLU"]
    assert len(data.mains) == 1 and data.mains[0]["aborted"]
    assert [t["reason"] for t in data.triggers] == ["tiling"]
    assert data.triggers[0]["excerpt"].endswith("tile size carefully. ")
    assert [s["kernel"] for s in data.specs] == ["i000-t01-s0"]
    assert {j["kind"] for j in data.jobs} == {"validate", "benchmark"}
    assert all(j["end"] >= j["start"] >= j["queued"] for j in data.jobs)
    span = data.spans[0]
    assert span["end_reason"] == "terminated" and span["termination"]["kernel"] == "i000-t01-s0"
    stats = stage_stats(data)
    assert stats["validate"]["done"] == 1 and stats["benchmark"]["run_mean_s"] is not None


def test_summary(run_dir, capsys):
    summary = summarize(load_run(run_dir))
    task = summary["tasks"][0]
    assert task["best"]["source"] == "spec" and task["spec"]["selected"] == 1
    assert task["iterations"][0]["end_reason"] == "terminated" and task["iterations"][0]["first_kernel_source"] == "spec"
    assert summary["triggers"] == {"count": 1, "launched": 1, "k": {1: 1}, "reasons": {"tiling": 1}}
    text = format_summary(summary)
    assert "terminated" in text and "i000-t01-s0" in text
    assert summary_main([str(run_dir), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tasks"][0]["task"] == task["task"]


def test_report(run_dir, tmp_path):
    data = load_run(run_dir)
    payload = build_payload(data, "t")
    assert payload["duration"] > 0 and payload["kernels"]["level0/1_ReLU|i000-t01-s0"]["selected"]
    assert payload["kernels"]["level0/1_ReLU|i000-t01-s0"]["failed_step"] is None
    assert payload["mains"][0]["aborted"] and payload["spans"][0]["termination"]["t"] <= payload["duration"]
    page = render(data, title="Demo <run>", intro_html="<p id='intro'>hi</p>")
    assert "Demo &lt;run&gt;" in page and "<p id='intro'>hi</p>" in page and "__DATA__" not in page
    embedded = re.search(r'<script id="run-data" type="application/json">(.*?)</script>', page, re.S).group(1)
    assert json.loads(embedded.replace("<\\/", "</"))["sections"][0]["tasks"] == data.tasks
    out = tmp_path / "r.html"
    assert report_main([str(run_dir), "-o", str(out)]) == 0 and out.stat().st_size > 10000


def test_report_shows_ncu_waits_and_alerts(run_dir, tmp_path):
    import shutil

    run = shutil.copytree(run_dir, tmp_path / "run")
    lines = (run / "events.jsonl").read_text().splitlines()
    t = json.loads(lines[-1])["t"]
    task = json.loads(lines[0]).get("task") or "level0/1_ReLU"
    extra = [{"type": "ncu_alert", "t": t - 1.0, "task": task, "iteration": 0, "kind": "timeout",
              "kernel": "i000-t01-s0", "ncu_s": 120.0},
             {"type": "ncu_wait", "t": t, "task": task, "iteration": 1, "seconds": 4.0, "finished": 1,
              "unfinished": 1}]
    (run / "events.jsonl").write_text("\n".join(lines + [json.dumps(e) for e in extra]) + "\n")
    data = load_run(run)
    assert data.ncu_waits == [{"task": task, "iteration": 1, "finished": 1, "unfinished": 1, "start": t - 4.0,
                               "end": t}]
    assert [(a["kind"], a["kernel"]) for a in data.alerts] == [("timeout", "i000-t01-s0")]
    payload = build_payload(data, "t")
    wait = payload["ncu_waits"][0]
    assert wait["end"] - wait["start"] == pytest.approx(4.0) and payload["alerts"][0]["ncu_s"] == 120.0


def test_intro_can_be_a_standalone_page(run_dir, tmp_path):
    page = ("<!doctype html><html><head><meta charset='utf-8'><style data-standalone>body{color:red}</style>"
            "<style>.intro{color:blue}</style></head><body><p data-standalone>open demo.html</p>"
            "<section class='intro'>hello</section></body></html>")
    html_part, css = split_intro(page)
    assert html_part == "<section class='intro'>hello</section>" and css == ".intro{color:blue}"
    assert split_intro("<section>fragment</section>") == ("<section>fragment</section>", "")
    intro = tmp_path / "intro.html"
    intro.write_text(page)
    out = tmp_path / "with_intro.html"
    assert report_main([str(run_dir), "--intro", str(intro), "-o", str(out)]) == 0
    text = out.read_text()
    assert "<section class='intro'>hello</section>" in text and ".intro{color:blue}" in text
    assert "open demo.html" not in text and "body{color:red}" not in text


def fake_run(points, iteration_ends, aborted=0):
    data = RunData(Path("fake"), {}, [{"type": "run_started", "t": 0.0, "tasks": ["t"]}], [], [], t0=0.0,
                   t_end=max(iteration_ends))
    data.benchmarks = [{"task": "t", "t": t, "speedup": s, "iteration": next(i for i, end in enumerate(iteration_ends)
                                                                              if t <= end)} for t, s in points]
    data.spans = [{"task": "t", "iteration": i, "start": 0.0, "end": end, "termination": {"kernel": "k", "t": end}}
                  for i, end in enumerate(iteration_ends)]
    data.mains = [{"task": "t", "iteration": i, "aborted": i < aborted, "start": 0.0, "end": end, "progress": [[0.0, 0]]}
                  for i, end in enumerate(iteration_ends)]
    return data


def test_compare_runs():
    spec = fake_run([(10, 1.0), (20, 2.0), (30, 1.5), (40, 3.0)], [25, 50], aborted=2)
    plain = fake_run([(30, 1.2), (80, 2.95)], [90])  # its criterion was met too, after the main stream ended
    assert best_curve(spec, "t") == [[10, 1.0], [20, 2.0], [40, 3.0]]
    compare = compare_payload(spec, plain, ("spec", "plain"))
    assert compare["labels"] == ["spec", "plain"] and compare["duration"] == 90
    assert compare["curves"]["t"] == [[[10, 1.0], [20, 2.0], [40, 3.0]], [[30, 1.2], [80, 2.95]]]
    a, b = compare["rows"][0]["a"], compare["rows"][0]["b"]
    tokens = {"in": 0, "in_hit_free": 0, "in_hit_1_50": 0, "out": 0, "main_out": 0, "spec_out": 0}
    assert a == {"iterations": 2, "end": 50, "first": 10, "best": 3.0, "evaluated": 0, "valid": 0, "aborted": 2,
                 "reach": 40, "saved_s": 10.0, "at_reach": tokens, "total": tokens,
                 "profiler_at_reach": {"busy_s": 0.0, "seconds": 40.0, "share": 0.0}}
    assert b == {"iterations": 1, "end": 90, "first": 30, "best": 2.95, "evaluated": 0, "valid": 0, "aborted": 0,
                 "reach": None}  # 2.95 never reaches 3.0
    assert compare["baseline"] and set(compare["criteria"]["t"]) == {"baseline", "recorded"}  # plain has no specs
    assert compare["timeline"]["duration"] == 90 and len(compare["timeline"]["spans"]) == 1  # drawn under the run's


def synthetic_run():
    """Three iterations whose kernels meet first_valid, hist_avg and best at different times, and two specs."""
    data = RunData(Path("fake"), {"termination": {"margin": 0.03, "confirm": 1}},
                   [{"type": "run_started", "t": 0.0, "tasks": ["t"]}], [], [], t0=0.0, t_end=100.0)
    data.spans = [{"task": "t", "iteration": i, "start": s, "end": e} for i, (s, e) in enumerate([(0, 30), (35, 60), (65, 100)])]
    data.benchmarks = [{"task": "t", "iteration": it, "kernel": f"k{n}", "speedup": v, "t": t} for n, (it, t, v) in
                       enumerate([(0, 10, 1.5), (1, 40, 1.2), (1, 50, 1.6), (2, 70, 1.58), (2, 80, 1.62), (2, 90, 1.7)])]
    data.specs = [{"task": "t", "iteration": 1, "kernel": "s1", "trigger": 1, "start": 36.0, "end": 45.0},
                  {"task": "t", "iteration": 1, "kernel": "s2", "trigger": 2, "start": 45.0, "end": 55.0}]
    data.kernels = [
        {"task_id": "t", "kernel_id": "s1", "generation": {"server_prompt_tokens": 1000, "server_cached_prompt_tokens": 900,
                                                           "server_completion_tokens": 300, "seconds": 9.0, "ttft_s": 1.0}},
        {"task_id": "t", "kernel_id": "s2", "generation": {"server_prompt_tokens": 1200, "server_cached_prompt_tokens": 1000,
                                                           "server_completion_tokens": 400, "seconds": 10.0, "ttft_s": 1.0}}]
    return data


def test_other_criteria_cut_the_timeline():
    import statistics

    from specgen.analysis.criteria import simulate, stopped_at

    data = synthetic_run()
    first, best, hist = (simulate(data, name)["t"] for name in ("first_valid", "best", "hist_avg"))
    assert first["per_iteration"] == [1.5, 1.5, 1.58] and first["seconds"] == 30.0 and first["best"] == 1.58
    assert best["per_iteration"] == [1.5, 1.6, 1.7] and best["seconds"] == 60.0 and best["best"] == 1.7
    assert hist["per_iteration"] == [1.5, 1.6, 1.62] and hist["seconds"] == 50.0  # the mean of 1.5 and 1.6, +3%
    rate = statistics.median([300 / 8, 400 / 9])  # output tokens per second of the finished specs
    assert first["tokens"] == {"in": 1000, "in_hit_free": 100, "in_hit_1_50": 118, "out": round(3 * rate),
                               "main_out": 0, "spec_out": round(3 * rate)}  # s1 cut at 40 s, s2 not started
    assert (best["tokens"]["in_hit_free"], best["tokens"]["in_hit_1_50"]) == (300, 338)
    assert best["tokens"]["out"] == round(300 + 4 * rate)  # s1 finished, s2 cut at 50 s
    assert stopped_at(data, "t", 50.0)["tokens"] == best["tokens"]


def test_a_baseline_run_has_one_trajectory():
    from specgen.analysis.criteria import criteria_payload, is_baseline

    data = synthetic_run()
    assert not is_baseline(data)
    data.config["spec"] = {"enabled": False}
    data.specs, data.kernels = [], []
    assert is_baseline(data)
    payload = criteria_payload(data)["t"]
    assert set(payload) == {"baseline", "recorded"}
    assert payload["baseline"]["per_iteration"] == [1.5, 1.6, 1.7] and payload["baseline"]["seconds"] == 100.0


def test_report_with_comparison(run_dir, tmp_path):
    out = tmp_path / "compare.html"
    assert report_main([str(run_dir), "--compare", str(run_dir), "--label", "开 spec", "--compare-label", "baseline",
                        "-o", str(out)]) == 0
    embedded = re.search(r'<script id="run-data" type="application/json">(.*?)</script>', out.read_text(), re.S).group(1)
    payload = json.loads(embedded.replace("<\\/", "</"))["sections"][0]
    compare = payload["compare"]
    assert compare["labels"] == ["开 spec", "baseline"]
    assert payload["slots"] == compare["timeline"]["slots"] and payload["slots"]  # same resources, same rows
    row = compare["rows"][0]
    assert row["a"] == row["b"] and row["a"]["aborted"] == 1 and row["a"]["valid"] == 1
    assert row["a"]["reach"] is not None and row["a"]["saved_s"] >= 0
    first = payload["criteria"]["level0/1_ReLU"]["first_valid"]  # the scripted backend reports no token counts
    assert first["best"] and set(first["tokens"]) == {"in", "in_hit_free", "in_hit_1_50", "out", "main_out", "spec_out"}


def test_compared_runs_get_the_same_resource_rows():
    from specgen.analysis.report import slot_rows

    busy, idle = RunData(Path("a"), {}, [], [], []), RunData(Path("b"), {}, [], [], [])
    busy.jobs = [{"slot": s} for s in ("profiler:7#0", "validator:6#2", "builder:cpu#1", "validator:6#0", "builder:cpu#0")]
    idle.jobs = [{"slot": s} for s in ("builder:cpu#0", "validator:0#0", "profiler:1#0")]
    assert slot_rows(busy, idle) == [
        ["builder:cpu#0", "builder:cpu#1", "validator:6#0", "validator:6#1", "validator:6#2", "profiler:7#0"],
        ["builder:cpu#0", "builder:cpu#1", "validator:0#0", "validator:0#1", "validator:0#2", "profiler:1#0"]]


def test_a_resumed_run_keeps_its_first_sessions_gpu_numbers():
    from specgen.analysis.data import align_resumed_devices

    events = [{"type": "run_started", "t": 0.0, "devices": ["validator:6", "profiler:7", "builder:cpu"]},
              {"type": "run_started", "t": 100.0, "resumed": True, "devices": ["validator:0", "profiler:1", "builder:cpu"]}]
    data = RunData(Path("r"), {}, events, [], [])
    data.jobs = [{"slot": "validator:6#1", "start": 10.0}, {"slot": "validator:0#2", "start": 120.0},
                 {"slot": "profiler:1#0", "start": 130.0}, {"slot": "builder:cpu#3", "start": 140.0}]
    assert align_resumed_devices(data) == {"validator:0": "validator:6", "profiler:1": "profiler:7"}
    assert [j["slot"] for j in data.jobs] == ["validator:6#1", "validator:6#2", "profiler:7#0", "builder:cpu#3"]
    assert align_resumed_devices(RunData(Path("s"), {}, events[:1], [], [])) == {}


def test_profiler_utilization_counts_the_time_kept():
    from specgen.analysis.criteria import simulate, stopped_at

    data = synthetic_run()
    data.jobs = [{"task": "t", "kind": "benchmark", "start": 8.0, "end": 12.0},
                 {"task": "t", "kind": "ncu", "start": 10.0, "end": 14.0},  # overlaps the timing: counted once
                 {"task": "t", "kind": "validate", "start": 20.0, "end": 25.0},  # not profiler work
                 {"task": "t", "kind": "benchmark", "start": 45.0, "end": 55.0}]
    # first_valid keeps 0-10, 30-40 and 60-70; best keeps 0-10, 30-50 and 60-90
    assert simulate(data, "first_valid")["t"]["profiler"] == {"busy_s": 2.0, "seconds": 30.0, "share": round(2 / 30, 4)}
    assert simulate(data, "best")["t"]["profiler"] == {"busy_s": 7.0, "seconds": 60.0, "share": round(7 / 60, 4)}
    assert stopped_at(data, "t", 50.0)["profiler"] == {"busy_s": 11.0, "seconds": 50.0, "share": 0.22}


def test_a_budget_stops_the_cut_timeline():
    from specgen.analysis.criteria import equivalent, simulate

    data = synthetic_run()
    # best keeps 0-10, 30-50 and 60-90; 45 s run out 10 s into iteration 2, after its 1.58 at 70 s
    timed = simulate(data, "best", budget=("seconds", 45.0))["t"]
    assert (timed["best"], timed["seconds"], timed["iterations"], timed["cut"]) == (1.6, 45.0, 2, True)
    assert timed["per_iteration"] == [1.5, 1.6, 1.6]
    full = simulate(data, "best")["t"]
    assert (full["iterations"], full["cut"]) == (3, False)
    assert simulate(data, "best", budget=("seconds", 1000.0))["t"]["best"] == full["best"]  # enough time: the whole run
    # s1 (prompt 1000, 900 cached, 300 out) is all the first 45 s spend; s2's prompt (220 more) starts at 45 s
    s1 = equivalent({"in_hit_1_50": 100 + 900 / 50, "out": 300})
    spent = simulate(data, "best", budget=("tokens", s1 + 200))["t"]
    assert (spent["best"], spent["iterations"], spent["cut"]) == (1.5, 1, True)  # s2's prompt would exceed it
    assert spent["seconds"] == 25.0 and equivalent(spent["tokens"]) <= s1 + 200


def test_a_page_of_several_runs(run_dir):
    from specgen.analysis.report import render_page

    data = load_run(run_dir)
    sections = [{**build_payload(data, f"SpecGen · run {i}"), "anchor": f"r{i}", "note": "n"} for i in range(2)]
    page = render_page(sections, "Two runs", intro_html="<table id='overview'></table>")
    embedded = re.search(r'<script id="run-data" type="application/json">(.*?)</script>', page, re.S).group(1)
    assert [s["title"] for s in json.loads(embedded.replace("<\\/", "</"))["sections"]] == ["SpecGen · run 0",
                                                                                         "SpecGen · run 1"]
    assert "<title>Two runs</title>" in page and "<table id='overview'></table>" in page
    assert not any(marker in page for marker in ("__TITLE__", "__INTRO__", "__EXTRA_STYLE__", "__DATA__"))
