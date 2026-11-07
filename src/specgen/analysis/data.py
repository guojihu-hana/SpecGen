"""Load a run directory into one structure for summaries and reports."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

EXCERPT_CHARS = 360


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a run killed mid-write leaves a partial last line
    return rows


def _task_dir(task_id: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "__", task_id)


@dataclass
class RunData:
    run_dir: Path
    config: dict[str, Any]
    events: list[dict[str, Any]]
    kernels: list[dict[str, Any]]
    iterations: list[dict[str, Any]]
    t0: float = 0.0
    t_end: float = 0.0
    jobs: list[dict[str, Any]] = field(default_factory=list)
    mains: list[dict[str, Any]] = field(default_factory=list)
    specs: list[dict[str, Any]] = field(default_factory=list)
    triggers: list[dict[str, Any]] = field(default_factory=list)
    spans: list[dict[str, Any]] = field(default_factory=list)  # iterations with start / end
    benchmarks: list[dict[str, Any]] = field(default_factory=list)
    role_changes: list[dict[str, Any]] = field(default_factory=list)
    ncu_waits: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    resumes: list[dict[str, Any]] = field(default_factory=list)  # where a resumed run continues, and the idle time
    interrupted: list[dict[str, Any]] = field(default_factory=list)  # iteration spans cut by the end of a session

    @property
    def tasks(self) -> list[str]:
        started = next((e for e in self.events if e["type"] == "run_started"), None)
        if started and started.get("tasks"):
            return list(started["tasks"])
        return sorted({e["task"] for e in self.events if "task" in e})

    def rel(self, t: Optional[float]) -> Optional[float]:
        return None if t is None else round(t - self.t0, 3)


def load_run(run_dir: Path) -> RunData:
    run_dir = Path(run_dir)
    if not (run_dir / "events.jsonl").is_file():
        raise FileNotFoundError(f"{run_dir} has no events.jsonl")
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    data = RunData(run_dir, config, _jsonl(run_dir / "events.jsonl"), _jsonl(run_dir / "kernels.jsonl"),
                   _jsonl(run_dir / "iterations.jsonl"))
    _close_resume_gaps(data)
    if data.events:
        data.t0 = data.events[0]["t"]
        data.t_end = data.events[-1]["t"]
    _derive(data)
    return data


RESUME_SPACING_S = 1.0  # a resumed session starts this long after the previous one ended, on the timeline


def _close_resume_gaps(data: RunData) -> None:
    """A resumed run (``specgen run --resume``) appends a session to the same files, hours later perhaps: shift
    every session back to start right after the previous one ended, and note where and how long the gap was."""
    offset, last = 0.0, None
    for e in data.events:
        if e["type"] == "run_started" and e.get("resumed") and last is not None:
            gap = e["t"] - offset - last - RESUME_SPACING_S
            if gap > 0:
                offset += gap
                data.resumes.append({"t": last + RESUME_SPACING_S, "gap_s": round(gap, 1)})
        e["t"] -= offset
        last = e["t"]


def _devices_by_role(run_started: dict[str, Any]) -> dict[str, list[str]]:
    """``role -> ["role:device", ...]`` from a run_started event, in the order the run listed them."""
    out: dict[str, list[str]] = defaultdict(list)
    for device in run_started.get("devices") or []:
        out[device.split(":")[0]].append(device)
    return out


def align_resumed_devices(data: RunData) -> dict[str, str]:
    """Label the devices of a resumed run's later sessions as in its first one, role by role (``validator:0`` of
    the second session becomes the first session's ``validator:6``), so that a run continued on other GPUs keeps
    one timeline row per slot.  Only the labels of the jobs change.  Returns the relabelling, ``old -> new``."""
    sessions = [e for e in data.events if e["type"] == "run_started"]
    if len(sessions) < 2:
        return {}
    first = _devices_by_role(sessions[0])
    applied: dict[str, str] = {}
    for i, session in enumerate(sessions[1:], start=1):
        end = sessions[i + 1]["t"] if i + 1 < len(sessions) else float("inf")
        mapping = {old: new for role, devices in _devices_by_role(session).items()
                   for old, new in zip(devices, first[role]) if old != new}
        for job in data.jobs:
            prefix, sep, index = job["slot"].partition("#")
            if session["t"] <= job["start"] < end and prefix in mapping:
                job["slot"] = mapping[prefix] + sep + index
        applied.update(mapping)
    return applied


def _derive(data: RunData) -> None:
    queued: dict[str, list[float]] = defaultdict(list)
    open_jobs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mains: dict[tuple, dict[str, Any]] = {}
    specs: dict[tuple, dict[str, Any]] = {}
    spans: dict[tuple, dict[str, Any]] = {}
    last_t = None
    for e in data.events:
        kind, t = e["type"], e["t"]
        key = (e.get("task"), e.get("iteration"))
        if kind == "run_started" and e.get("resumed") and last_t is not None:
            _end_session(data, last_t, queued, open_jobs, mains, specs, spans)
        last_t = t
        if kind == "job_queued":
            queued[e["label"]].append(t)
        elif kind == "job_started":
            q = queued[e["label"]].pop(0) if queued[e["label"]] else t
            job = {"label": e["label"], "kind": e["kind"], "task": e["task"], "slot": e.get("slot"), "queued": q,
                   "start": t, "end": None, "outcome": None, "concurrency": e.get("concurrency")}
            job["kernel"] = e["label"].split("/")[-2] if e["label"].count("/") >= 2 else ""
            open_jobs[e["label"]].append(job)
            data.jobs.append(job)
        elif kind == "job_finished":
            if open_jobs[e["label"]]:
                job = open_jobs[e["label"]].pop(0)
                job.update(end=t, outcome=e.get("outcome"))
        elif kind == "iteration_started":
            spans[key] = {"task": e["task"], "iteration": e["iteration"], "start": t, "end": None}
        elif kind == "iteration_finished":
            span = spans.setdefault(key, {"task": e["task"], "iteration": e["iteration"], "start": t})
            span.update(end=t, end_reason=e.get("end_reason"), selected=e.get("selected"))
        elif kind == "iteration_terminated":
            spans.setdefault(key, {"task": e["task"], "iteration": e["iteration"], "start": t})["termination"] = {
                k: e.get(k) for k in ("criterion", "kernel", "speedup", "baseline")} | {"t": t}
        elif kind == "main_started":
            mains[key] = {"task": e["task"], "iteration": e["iteration"], "start": t, "end": None,
                          "progress": [[t, 0]], "ok": None}
        elif kind == "main_progress" and key in mains:
            mains[key]["progress"].append([t, e["tokens"]])
        elif kind == "main_finished" and key in mains:
            mains[key].update(end=t, ok=e.get("ok"), reasoning_tokens=e.get("reasoning_tokens"),
                              content_tokens=e.get("content_tokens"), seconds=e.get("seconds"))
            mains[key]["progress"].append([t, e.get("reasoning_tokens") or mains[key]["progress"][-1][1]])
        elif kind == "spec_triggered":
            data.triggers.append({**{k: e.get(k) for k in ("task", "iteration", "index", "reason", "keyword",
                                                            "matched_tokens", "fired_tokens", "prefix_chars", "k")},
                                  "t": t, "budget": e.get("budget") or {}})
        elif kind == "spec_started":
            specs[(e["task"], e["kernel"])] = {"task": e["task"], "iteration": e["iteration"], "kernel": e["kernel"],
                                               "trigger": e.get("trigger"), "start": t, "end": None}
        elif kind == "spec_finished" and (e["task"], e["kernel"]) in specs:
            specs[(e["task"], e["kernel"])].update(end=t, ok=e.get("ok"), has_code=e.get("has_code"),
                                                   seconds=e.get("seconds"))
        elif kind == "kernel_benchmarked":
            data.benchmarks.append({k: e.get(k) for k in ("task", "iteration", "kernel", "source", "speedup")} | {"t": t})
        elif kind == "gpu_role_changed":
            data.role_changes.append(e)
        elif kind == "ncu_wait":  # before the prompt of `iteration`, waiting for ncu
            data.ncu_waits.append({**{k: e.get(k) for k in ("task", "iteration", "finished", "unfinished")},
                                   "start": t - (e.get("seconds") or 0.0), "end": t})
        elif kind == "ncu_alert":
            data.alerts.append({**{k: e.get(k) for k in ("task", "iteration", "kind", "kernel", "ncu_s", "seconds")},
                                "t": t})
    for span in spans.values():
        span["end"] = span.get("end") or data.t_end
    for main in mains.values():
        if main["end"] is None:  # aborted by a termination (or the run ended)
            span = spans.get((main["task"], main["iteration"]), {})
            main["end"] = span.get("end") or data.t_end
            main["aborted"] = True
    for spec in specs.values():
        if spec["end"] is None:
            span = spans.get((spec["task"], spec["iteration"]), {})
            spec["end"] = span.get("end") or data.t_end
            spec["cancelled"] = True
    for job in data.jobs:
        if job["end"] is None:
            job["end"], job["outcome"] = data.t_end, "unfinished"
    data.mains = list(mains.values())
    data.specs = list(specs.values())
    data.spans = list(spans.values())
    _attach_excerpts(data)


def _end_session(data: RunData, t: float, queued, open_jobs, mains, specs, spans) -> None:
    """A resumed run starts a new session: the previous one ended at ``t`` (perhaps killed).  Its unfinished jobs
    end there; the iteration it did not finish runs again, so (as in the records) its stream, specs, triggers and
    timings leave the data and its span moves to ``data.interrupted``."""
    for jobs in open_jobs.values():
        for job in jobs:
            job.update(end=t, outcome="interrupted")
    open_jobs.clear()
    queued.clear()
    rerun = {key for key, span in spans.items() if span.get("end") is None}
    cut = {key: [k for k, spec in specs.items() if (spec["task"], spec["iteration"]) == key] for key in rerun}
    data.interrupted += [{**spans.pop(key), "end": t, "specs": len(cut[key]),  # spent, though not in the token counts
                          "main_tokens": mains[key]["progress"][-1][1] if key in mains else 0}
                         for key in sorted(rerun, key=str)]
    for key in rerun:
        mains.pop(key, None)
        for k in cut[key]:
            del specs[k]
    for name in ("triggers", "benchmarks", "ncu_waits", "alerts"):
        setattr(data, name, [x for x in getattr(data, name) if (x.get("task"), x.get("iteration")) not in rerun])


def _attach_excerpts(data: RunData) -> None:
    """The reasoning just before each trigger's cut, from the stored main stream."""
    cache: dict[tuple, str] = {}
    for trigger in data.triggers:
        key = (trigger["task"], trigger["iteration"])
        if key not in cache:
            path = data.run_dir / _task_dir(trigger["task"]) / f"iter_{trigger['iteration']:03d}" / "main" / "reasoning.txt"
            cache[key] = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        text, end = cache[key], trigger.get("prefix_chars") or 0
        trigger["excerpt"] = text[max(0, end - EXCERPT_CHARS):end] if text else ""


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def stage_stats(data: RunData) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in data.jobs:
        by_kind[job["kind"]].append(job)
    for kind, jobs in sorted(by_kind.items()):
        waits = [j["start"] - j["queued"] for j in jobs]
        runs = [j["end"] - j["start"] for j in jobs if j["outcome"] == "done"]
        out[kind] = {
            "jobs": len(jobs), "done": len(runs), "cancelled": sum(1 for j in jobs if j["outcome"] == "cancelled"),
            "wait_mean_s": round(statistics.fmean(waits), 1) if waits else None,
            "wait_p90_s": round(percentile(waits, 0.9), 1) if waits else None,
            "run_mean_s": round(statistics.fmean(runs), 1) if runs else None,
            "run_p90_s": round(percentile(runs, 0.9), 1) if runs else None,
        }
    return out
