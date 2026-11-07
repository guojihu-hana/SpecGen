"""``specgen summarize RUN_DIR``: what a run produced and where its time went."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from specgen.analysis.data import RunData, load_run, stage_stats


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(data: RunData) -> dict[str, Any]:
    cfg, resolved = data.config, data.config.get("resolved", {})
    started = next((e for e in data.events if e["type"] == "run_started"), {})
    finished = next((e for e in reversed(data.events) if e["type"] == "run_finished"), {})
    spans = {(s["task"], s["iteration"]): s for s in data.spans}
    first_bench: dict[tuple, dict[str, Any]] = {}
    for b in data.benchmarks:
        first_bench.setdefault((b["task"], b["iteration"]), b)

    kernels_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for k in data.kernels:
        kernels_by_task[k["task_id"]].append(k)
    tasks = []
    for task in data.tasks:
        iterations = []
        for it in (r for r in data.iterations if r["task"] == task):
            span = spans.get((task, it["iteration"]), {})
            first = first_bench.get((task, it["iteration"]))
            iterations.append({
                **{k: it.get(k) for k in ("iteration", "end_reason", "seconds", "selected", "selected_source", "speedup",
                                          "main_status", "main_speedup", "candidates", "statuses",
                                          "reasoning_tokens", "generation_s", "unfinished_spec_tasks",
                                          "superseded_specs")},
                "triggers": len(it.get("triggers") or []),
                "spec_generations": sum(t.get("k") or 0 for t in it.get("triggers") or []),
                "first_kernel_s": round(first["t"] - span["start"], 1) if first and span else None,
                "first_kernel_source": first.get("source") if first else None,
            })
        kernels = kernels_by_task[task]
        spec = [k for k in kernels if k.get("source") == "spec"]
        best = max((k for k in kernels if k.get("speedup") is not None and k.get("valid")),
                   key=lambda k: k["speedup"], default=None)
        tasks.append({
            "task": task,
            "iterations": iterations,
            "best": {k: best.get(k) for k in ("kernel_id", "iteration", "source", "speedup")} if best else None,
            "kernels": len(kernels),
            "spec": {
                "kernels": len(spec),
                "parsed": sum(1 for k in spec if (k.get("generation") or {}).get("has_code")),
                "valid": sum(1 for k in spec if k.get("valid")),
                "benchmarked": sum(1 for k in spec if k.get("speedup") is not None),
                "duplicates": sum(1 for k in spec if k.get("status") == "duplicate"),
                "selected": sum(1 for it in iterations if it["selected_source"] == "spec"),
            },
            "statuses": dict(Counter(k.get("status") for k in kernels)),
        })
    ks = [t.get("k") or 0 for t in data.triggers]
    return {
        "run": {
            "dir": str(data.run_dir), "model": cfg.get("llm", {}).get("model"), "endpoint": cfg.get("llm", {}).get("endpoint"),
            "gpu_type": resolved.get("gpu_type"), "allocation": cfg.get("pool", {}).get("allocation"),
            "devices": started.get("devices") or started.get("slots"), "spec_prompt_mode": resolved.get("spec_prompt_mode"),
            "termination": cfg.get("termination"), "duration_s": round(data.t_end - data.t0, 1),
        },
        "tasks": tasks,
        "triggers": {"count": len(ks), "launched": sum(ks), "k": dict(sorted(Counter(ks).items())),
                     "reasons": dict(Counter(t.get("reason") for t in data.triggers).most_common())},
        "stages": stage_stats(data),
        "utilization": (finished.get("pool") or {}).get("utilization"),
        "spec_budget": finished.get("spec"),
    }


def format_summary(summary: dict[str, Any]) -> str:
    run = summary["run"]
    lines = [
        f"run        {run['dir']}",
        f"model      {run['model']} ({run['endpoint']}), GPU {run['gpu_type']}, allocation {run['allocation']}",
        f"spec       prompt mode {run['spec_prompt_mode']}; termination {run['termination']}",
        f"duration   {_fmt(run['duration_s'], 1)} s",
        "",
    ]
    for task in summary["tasks"]:
        best = task["best"]
        lines.append(f"== {task['task']}  best {_fmt(best and best['speedup'], 3)}x "
                     f"({best['kernel_id'] if best else '-'}, {best['source'] if best else '-'})")
        lines.append("  iter  end            secs   selected          src   speedup  main          trig  specs  repl  first")
        for it in task["iterations"]:
            lines.append(
                f"  {it['iteration']:>4}  {_fmt(it['end_reason']):<13} {_fmt(it['seconds'], 0):>5}  "
                f"{_fmt(it['selected']):<17} {_fmt(it['selected_source']):<5} {_fmt(it['speedup'], 3):>7}  "
                f"{_fmt(it['main_status']):<12} {it['triggers']:>5} {it['spec_generations']:>6} "
                f"{it.get('superseded_specs') or 0:>5}  {_fmt(it['first_kernel_s'], 0)}s/{_fmt(it['first_kernel_source'])}")
        spec = task["spec"]
        lines.append(f"  spec kernels {spec['kernels']}: {spec['parsed']} parsed, {spec['valid']} valid, "
                     f"{spec['benchmarked']} timed, {spec['duplicates']} duplicates; selected in "
                     f"{spec['selected']} of {len(task['iterations'])} iterations")
        lines.append(f"  statuses {task['statuses']}")
        lines.append("")
    trig = summary["triggers"]
    lines.append(f"triggers   {trig['count']} fired, {trig['launched']} spec generations; k {trig['k']}")
    lines.append(f"           reasons {trig['reasons']}")
    lines.append("stages     kind       jobs  done  cancel  wait mean/p90 s   run mean/p90 s")
    for kind, s in summary["stages"].items():
        lines.append(f"           {kind:<10} {s['jobs']:>4} {s['done']:>5} {s['cancelled']:>7}  "
                     f"{_fmt(s['wait_mean_s'], 1):>7}/{_fmt(s['wait_p90_s'], 1):<7}  "
                     f"{_fmt(s['run_mean_s'], 1):>7}/{_fmt(s['run_p90_s'], 1)}")
    if summary.get("utilization"):
        lines.append(f"utilization {summary['utilization']}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="specgen summarize", description="Summarize a SpecGen run directory.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    a = p.parse_args(argv)
    summary = summarize(load_run(a.run_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str) if a.json else format_summary(summary))
    return 0
