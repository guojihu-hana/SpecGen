"""``specgen report RUN_DIR``: a self-contained HTML timeline of a run that can be replayed."""

from __future__ import annotations

import argparse
import html
import json
import re
from importlib import resources
from pathlib import Path
from typing import Any, Optional

from specgen.analysis.criteria import Ledger, criteria_payload, is_baseline, stopped_at
from specgen.analysis.data import RunData, load_run
from specgen.analysis.summary import summarize

BUDGET_KEYS = ("policy", "capacity_per_s", "horizon_s", "in_pipeline", "raw_k", "capped_by_inflight")
NOT_EVALUATED = {"duplicate", "cancelled", "grace_truncated", "parse_error", "llm_error"}
_STYLE = re.compile(r"<style(?P<attrs>[^>]*)>(?P<css>.*?)</style>", re.S | re.I)
_BODY = re.compile(r"<body[^>]*>(?P<body>.*)</body>", re.S | re.I)
_STANDALONE = re.compile(r"<(?P<tag>\w+)\b[^>]*\bdata-standalone\b[^>]*>.*?</(?P=tag)>", re.S | re.I)


def split_intro(text: str) -> tuple[str, str]:
    """``(html, css)`` of an intro to embed.  A complete HTML document (so the intro can be
    opened on its own) gives its body and head styles, except elements and styles marked
    ``data-standalone``; a fragment is embedded as it is."""
    body = _BODY.search(text)
    if body is None:
        return text, ""
    css = "\n".join(m.group("css") for m in _STYLE.finditer(text[:body.start()])
                    if "data-standalone" not in m.group("attrs"))
    return _STANDALONE.sub("", body.group("body")), css


def _r(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 3)


ROLE_ORDER = ("builder", "validator", "shared", "profiler")
STEPS = (("build", "build"), ("validation", "validate"), ("benchmark", "benchmark"))  # record key, job kind


def slot_rows(*runs: RunData) -> list[list[str]]:
    """The resource rows of runs shown side by side, one list per run: every role gets as many slots on each of a
    run's devices as the busiest of the runs used, so runs given the same resources show the same rows (a run
    that used fewer slots is not drawn smaller)."""
    used = []
    for data in runs:
        devices: dict[str, dict[str, set[int]]] = {}
        for job in data.jobs:
            prefix, _, index = job["slot"].partition("#")
            role, _, device = prefix.partition(":")
            devices.setdefault(role, {}).setdefault(device, set()).add(int(index or 0))
        used.append(devices)
    width = {role: max(max(indices) + 1 for devices in used for indices in devices.get(role, {}).values())
             for role in {role for devices in used for role in devices}}
    order = lambda role: (ROLE_ORDER.index(role) if role in ROLE_ORDER else len(ROLE_ORDER), role)  # noqa: E731
    numeric = lambda device: (not device.isdigit(), int(device) if device.isdigit() else 0, device)  # noqa: E731
    return [[f"{role}:{device}#{i}" for role in sorted(devices, key=order)
             for device in sorted(devices[role], key=numeric) for i in range(width[role])] for devices in used]


def timeline_payload(data: RunData, *, slots: Optional[list[str]] = None) -> dict[str, Any]:
    """What the page draws of a run's timeline (also of a compared run, on the same time axis).  ``slots``: its
    resource rows (default: the slots its jobs used)."""
    rel = data.rel
    kernels = {}
    for k in data.kernels:
        generation, trigger = k.get("generation") or {}, k.get("trigger") or {}
        kernels[f"{k['task_id']}|{k['kernel_id']}"] = {
            "task": k["task_id"], "kernel": k["kernel_id"], "iteration": k.get("iteration"), "source": k.get("source"),
            "trigger": f"#{trigger['index']} {trigger['reason']}" if trigger else "", "status": k.get("status"),
            "speedup": k.get("speedup"), "valid": k.get("valid"), "selected": k.get("selected"),
            "generation_s": generation.get("seconds"), "build_s": (k.get("build") or {}).get("exec_s"),
            "validate_s": (k.get("validation") or {}).get("exec_s"),
            "benchmark_s": (k.get("benchmark") or {}).get("exec_s"),
            "failed_step": next((kind for key, kind in STEPS if (k.get(key) or {}).get("ok") is False), None),
        }
    return {
        "duration": round(data.t_end - data.t0, 3),
        "tasks": data.tasks,
        "spans": [{**{k: s.get(k) for k in ("task", "iteration", "end_reason", "selected")},
                   "start": rel(s["start"]), "end": rel(s["end"]),
                   "termination": ({**s["termination"], "t": rel(s["termination"]["t"])} if s.get("termination") else None)}
                  for s in data.spans],
        "mains": [{**{k: m.get(k) for k in ("task", "iteration", "ok", "aborted", "reasoning_tokens")},
                   "start": rel(m["start"]), "end": rel(m["end"]),
                   "progress": [[rel(t), tokens] for t, tokens in m["progress"]]} for m in data.mains],
        "triggers": [{**{k: t.get(k) for k in ("task", "iteration", "index", "reason", "keyword", "fired_tokens", "k",
                                               "excerpt")},
                      "t": rel(t["t"]), "budget": {k: t["budget"].get(k) for k in BUDGET_KEYS if k in t["budget"]}}
                     for t in data.triggers],
        "specs": [{**{k: s.get(k) for k in ("task", "iteration", "kernel", "trigger", "ok", "has_code", "cancelled")},
                   "start": rel(s["start"]), "end": rel(s["end"])} for s in data.specs],
        "jobs": [{**{k: j.get(k) for k in ("kind", "task", "kernel", "slot", "outcome", "concurrency")},
                  "queued": rel(j["queued"]), "start": rel(j["start"]), "end": rel(j["end"])} for j in data.jobs],
        "kernels": kernels,
        "roles": [{**{k: r.get(k) for k in ("gpu", "old", "new", "load_validate_s", "load_benchmark_s")}, "t": rel(r["t"])}
                  for r in data.role_changes],
        "ncu_waits": [{**w, "start": rel(w["start"]), "end": rel(w["end"])} for w in data.ncu_waits],
        "resumes": [{**r, "t": rel(r["t"])} for r in data.resumes],
        "alerts": [{**a, "t": rel(a["t"])} for a in data.alerts],
        "slots": slots,
    }


def build_payload(data: RunData, title: str, *, slots: Optional[list[str]] = None) -> dict[str, Any]:
    """The page data of one run: its timeline (``slots``: its resource rows), timings, tables and the
    termination criteria."""
    summary = summarize(data)
    return {
        "title": title,
        "run": {**summary["run"], "dir": data.run_dir.name},  # a shared page should not carry local paths
        **timeline_payload(data, slots=slots),
        "benchmarks": [{**{k: b.get(k) for k in ("task", "iteration", "kernel", "source", "speedup")},
                        "t": data.rel(b["t"])} for b in data.benchmarks if b.get("speedup") is not None],
        "iterations": data.iterations,
        "summary": {k: summary[k] for k in ("triggers", "stages", "utilization")},
        "criteria": criteria_payload(data),
        "criterion": (data.config.get("termination") or {}).get("criteria"),
        "baseline": is_baseline(data),
    }


def best_curve(data: RunData, task: str) -> list[list[float]]:
    """``[t, best speedup so far]`` at each improvement, ``t`` from the run start."""
    curve: list[list[float]] = []
    for b in sorted((b for b in data.benchmarks if b["task"] == task and b.get("speedup") is not None),
                    key=lambda b: b["t"]):
        if not curve or b["speedup"] > curve[-1][1]:
            curve.append([data.rel(b["t"]), b["speedup"]])
    return curve


def _side(data: RunData, task: str, curve: list[list[float]]) -> dict[str, Any]:
    spans = [s for s in data.spans if s["task"] == task]
    evaluated = [k for k in data.kernels if k["task_id"] == task and k.get("status") not in NOT_EVALUATED]
    return {
        "iterations": len(spans),
        "end": max((data.rel(s["end"]) for s in spans), default=None),
        "first": curve[0][0] if curve else None,
        "best": curve[-1][1] if curve else None,
        "evaluated": len(evaluated),
        "valid": sum(1 for k in evaluated if k.get("valid")),
        # a met criterion only cuts an iteration short while the main stream is still running
        "aborted": sum(1 for m in data.mains if m["task"] == task and m.get("aborted")),
    }


def _reach(run: RunData, ledger: Ledger, task: str, side: dict[str, Any], curve: list[list[float]],
           target: Optional[float]) -> None:
    """When ``run`` first measured a kernel at least as fast as ``target``; the time and tokens it would have
    saved by stopping then."""
    side["reach"] = next((t for t, v in curve if target and v >= target), None)
    if side["reach"] is None:
        return
    side["saved_s"] = round(side["end"] - side["reach"], 1) if side["end"] is not None else None
    stopped = stopped_at(run, task, run.t0 + side["reach"], ledger=ledger)
    side["at_reach"], side["profiler_at_reach"] = stopped["tokens"], stopped["profiler"]
    side["total"] = stopped_at(run, task, run.t_end, ledger=ledger)["tokens"]


def compare_payload(data: RunData, baseline: RunData, labels: tuple[str, str], *,
                    slots: Optional[list[str]] = None) -> dict[str, Any]:
    """Two runs of the same tasks side by side: the compared run's timeline (``slots``: its resource rows), best
    speedup over time and per-task totals.  ``reach`` is when a run's best first reached the other run's final
    best, with what stopping then would have saved."""
    curves, rows = {}, []
    ledgers = (Ledger(data), Ledger(baseline))
    for task in data.tasks:
        mine, theirs = best_curve(data, task), best_curve(baseline, task)
        a, b = _side(data, task, mine), _side(baseline, task, theirs)
        _reach(data, ledgers[0], task, a, mine, b["best"])
        _reach(baseline, ledgers[1], task, b, theirs, a["best"])
        curves[task] = [mine, theirs]
        rows.append({"task": task, "a": a, "b": b})
    return {"labels": list(labels), "duration": round(baseline.t_end - baseline.t0, 3), "curves": curves, "rows": rows,
            "criteria": criteria_payload(baseline),
            "criterion": (baseline.config.get("termination") or {}).get("criteria"), "baseline": is_baseline(baseline),
            "timeline": timeline_payload(baseline, slots=slots)}


def render_page(sections: list[dict[str, Any]], title: str, *, intro_html: str = "", extra_style: str = "") -> str:
    """A page of one or more runs (``sections``: ``build_payload`` results, each with an optional ``compare``,
    ``anchor`` and ``note``); a page of several runs opens with ``title`` and ``intro_html``."""
    payload = json.dumps({"title": title, "sections": sections}, ensure_ascii=False, default=str).replace("</", "<\\/")
    template = resources.files("specgen.analysis").joinpath("report_template.html").read_text(encoding="utf-8")
    return (template.replace("__EXTRA_STYLE__", extra_style).replace("__INTRO__", intro_html)
            .replace("__TITLE__", html.escape(title)).replace("__DATA__", payload))


def render(data: RunData, *, title: Optional[str] = None, intro_html: str = "", extra_style: str = "",
           compare: Optional[dict[str, Any]] = None, slots: Optional[list[str]] = None) -> str:
    title = title or f"SpecGen · {data.run_dir.name}"
    return render_page([{**build_payload(data, title, slots=slots), "compare": compare}], title,
                       intro_html=intro_html, extra_style=extra_style)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="specgen report", description="Write an HTML timeline of a SpecGen run.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None, help="default: RUN_DIR/report.html")
    p.add_argument("--title", default=None)
    p.add_argument("--intro", type=Path, default=None,
                   help="HTML inserted above the timeline: a fragment, or a full page (its body and styles)")
    p.add_argument("--style", type=Path, default=None, help="extra CSS")
    p.add_argument("--compare", type=Path, default=None, metavar="RUN_DIR",
                   help="another run of the same tasks, e.g. the baseline (iterative optimization without spec "
                        "generation): adds best speedup over time of both")
    p.add_argument("--label", default=None, help="name of RUN_DIR in the comparison (default: its directory name)")
    p.add_argument("--compare-label", default=None, help="name of the --compare run (default: its directory name)")
    a = p.parse_args(argv)
    data = load_run(a.run_dir)
    intro_html, intro_css = split_intro(a.intro.read_text(encoding="utf-8")) if a.intro else ("", "")
    extra_style = "\n".join(s for s in (intro_css, a.style.read_text(encoding="utf-8") if a.style else "") if s)
    compare, slots = None, None
    if a.compare:
        other = load_run(a.compare)
        labels = (a.label or a.run_dir.resolve().name, a.compare_label or a.compare.resolve().name)
        slots, other_slots = slot_rows(data, other)
        compare = compare_payload(data, other, labels, slots=other_slots)
    page = render(data, title=a.title, intro_html=intro_html, extra_style=extra_style, compare=compare, slots=slots)
    output = a.output or a.run_dir / "report.html"
    output.write_text(page, encoding="utf-8")
    print(f"wrote {output} ({len(page) // 1024} KiB)")
    return 0
