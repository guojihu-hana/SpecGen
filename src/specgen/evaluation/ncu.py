"""Nsight Compute profiling of a generated module: every kernel of one steady-state forward.

The harness (``ncu_harness``) wraps each forward in an NVTX range ``forward<i>``; ncu profiles only the
kernels launched inside ``forward<N>`` (``N`` = 3 by default, so caches and lazy setup are warm), each
launch once, collecting the requested sections and metrics.  If NVTX filtering captures nothing, each
distinct kernel (by mangled name) is profiled once at its N-th call instead.

The report is read per launch in base units (``--print-units base``: the second row of the raw page is
the units, then one row per launch) and grouped by kernel: each kernel gets its own metrics, its share
of the profiled GPU time and the rule results (bottlenecks, estimated speedups) of its launches.
Different kernels are never merged.  A run past the timeout is killed and repeated with a minimal set
of sections.
"""

from __future__ import annotations

import csv
import io
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from specgen.evaluation.jobs import EvalJob

HARNESS_ITERATIONS = 5
PROFILED_FORWARD = 3
LAUNCH_CAP = 64  # kernels of the profiled forward, at most
MINIMAL_SECTIONS = ("SpeedOfLight", "LaunchStats", "Occupancy")
MAX_RULE_CHARS = 2000


def _tail(text: str, limit: int = 4000) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "...\n" + text[-limit:]


def _parse_value(cell: str) -> float | str:
    """A raw-page value cell; may carry ``;``-separated instances (averaged)."""
    cell = cell.strip().strip('"')
    if not cell:
        return cell
    parts = [c.strip() for c in cell.split(";") if c.strip()] if ";" in cell else [cell]
    values: list[float] = []
    for part in parts:
        tokens = part.replace(",", "").split()
        try:
            values.append(float(tokens[0]))
        except (IndexError, ValueError):
            return cell
    return sum(values) / len(values) if values else cell


def _clean_rows(text: str) -> list[list[str]]:
    clean = "\n".join(line for line in text.splitlines() if line.strip() and not line.strip().startswith("=="))
    try:
        return [[c.strip().strip('"') for c in row] for row in csv.reader(io.StringIO(clean))]
    except csv.Error:
        return []


def short_kernel_name(name: str) -> str:
    """``void gemm<128, 64>(float const*, ...)`` -> ``gemm<128, 64>``."""
    depth = 0
    for i, ch in enumerate(name):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth <= 0:
            name = name[:i]
            break
    name = name.strip()
    name = name[5:] if name.startswith("void ") else name
    return name if len(name) <= 80 else name[:79] + "…"


def parse_raw_launches(text: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """``(launches, units)`` of ``ncu --import ... --page raw --csv``: one entry per profiled launch with
    its kernel's full name and every numeric metric; the units row (if any) maps metric -> unit."""
    rows = _clean_rows(text)
    start = next((i for i, row in enumerate(rows) if "Kernel Name" in row and "ID" in row), None)
    if start is None:
        return [], {}
    header, body = rows[start], rows[start + 1:]
    column = {name: i for i, name in enumerate(header)}
    units: dict[str, str] = {}
    if body and not body[0][column["ID"]].strip().isdigit():
        units = {name: cell for name, cell in zip(header, body[0]) if cell}
        body = body[1:]
    launches = []
    for row in body:
        if len(row) <= column["Kernel Name"] or not row[column["ID"]].strip().isdigit():
            continue
        metrics = {}
        for name, cell in zip(header, row):
            if "__" in name:
                value = _parse_value(cell)
                if isinstance(value, float):
                    metrics[name] = value
        launches.append({"id": row[column["ID"]].strip(), "kernel": row[column["Kernel Name"]], "metrics": metrics})
    return launches, units


def parse_details_rules(text: str) -> dict[str, list[dict[str, Any]]]:
    """Launch id -> rule results, from ``ncu --import ... --page details --csv``."""
    rows = _clean_rows(text)
    if len(rows) < 2 or "Rule Name" not in rows[0]:
        return {}
    header = rows[0]
    rules: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows[1:]:
        d = dict(zip(header, row))
        if not d.get("ID") or not d.get("Rule Name"):
            continue
        try:
            speedup: Optional[float] = float(d.get("Estimated Speedup") or "")
        except ValueError:
            speedup = None
        rules[d["ID"]].append({
            "section": d.get("Section Name", ""), "rule": d["Rule Name"], "type": d.get("Rule Type", ""),
            "speedup": speedup, "speedup_type": d.get("Estimated Speedup Type", ""),
            "text": (d.get("Rule Description") or "")[:MAX_RULE_CHARS],
        })
    return dict(rules)


def build_profile(launches: list[dict[str, Any]], units: dict[str, str],
                  rules: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Kernels of the profiled launches, most GPU time first, each with its own metrics (averaged over
    its own launches only), its share of the profiled time and its rule results."""
    kernels: dict[str, dict[str, Any]] = {}
    for launch in launches:
        k = kernels.setdefault(launch["kernel"], {"name": short_kernel_name(launch["kernel"]),
                                                  "full_name": launch["kernel"][:400], "launches": 0,
                                                  "time_ns": 0.0, "values": defaultdict(list), "rules": []})
        k["launches"] += 1
        k["time_ns"] += launch["metrics"].get("gpu__time_duration.sum", 0.0)
        for name, value in launch["metrics"].items():
            k["values"][name].append(value)
        if not k["rules"]:
            k["rules"] = rules.get(launch["id"], [])
    total = sum(k["time_ns"] for k in kernels.values())
    ordered = []
    for k in sorted(kernels.values(), key=lambda k: -k["time_ns"]):
        values = k.pop("values")
        k["metrics"] = {name: sum(v) / len(v) for name, v in values.items()}
        k["share"] = round(k["time_ns"] / total, 4) if total > 0 else None
        ordered.append(k)
    used = {name for k in ordered for name in k["metrics"]}
    return {"kernels": ordered, "units": {name: unit for name, unit in units.items() if name in used},
            "time_ns": total}


def profile_command(ncu_binary: str, report: Path, *, sections: list[str], metrics: list[str], nvtx: bool,
                    forward: int, launch_cap: int, kernel_path: str, task_path: str, seed: int,
                    iterations: int) -> list[str]:
    """ncu over the harness; ``nvtx`` selects the kernels of forward ``forward``, otherwise each distinct
    kernel's ``forward``-th call.  The harness launches every kernel itself, so child processes (ninja,
    nvcc when the extension has to be built) are not profiled."""
    if nvtx:
        selection = ["--nvtx", "--nvtx-include", f"forward{forward}/"]
    else:
        selection = ["--kernel-name-base", "mangled", "--kernel-id", f"::regex:.*:{forward}"]
    cmd = [ncu_binary, "-f", "-o", str(report), "--target-processes", "application-only", *selection,
           f"--launch-count={launch_cap}"]
    for section in sections:
        cmd += ["--section", section]
    if metrics:
        cmd += ["--metrics", ",".join(metrics)]
    return cmd + [sys.executable, "-m", "specgen.evaluation.ncu_harness", kernel_path, task_path,
                  str(iterations), str(seed)]


def _run(cmd: list[str], cwd: Path, timeout: Optional[float]) -> tuple[Optional[int], str, str]:
    """``(returncode, stdout, stderr)``; past the timeout the whole process group is killed and the
    return code is None."""
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        return None, out, err


def import_report(ncu_binary: str, report: Path) -> tuple[Optional[dict[str, Any]], str]:
    """``(profile, error)`` from the raw page (base units) and the details page (rules)."""
    raw = subprocess.run([ncu_binary, "--import", str(report), "--page", "raw", "--csv", "--print-units", "base"],
                         capture_output=True, text=True, timeout=600)
    if raw.returncode != 0:
        return None, f"ncu --import --page raw failed: {_tail(raw.stderr or raw.stdout, 2000)}"
    launches, units = parse_raw_launches(raw.stdout)
    details = subprocess.run([ncu_binary, "--import", str(report), "--page", "details", "--csv"],
                             capture_output=True, text=True, timeout=600)
    rules = parse_details_rules(details.stdout) if details.returncode == 0 else {}
    return build_profile(launches, units, rules), ""


def run(job: EvalJob) -> dict[str, Any]:
    p = job.params
    ncu_binary = str(p.get("ncu_binary", "ncu"))
    metrics = list(p.get("metrics") or [])
    sections = list(p.get("sections") or [])
    forward = int(p.get("forward", PROFILED_FORWARD))
    timeout = float(p["timeout_s"]) if p.get("timeout_s") else None
    work = Path(job.work_dir)
    report = work / "profile.ncu-rep"
    t0 = time.perf_counter()
    attempts: list[dict[str, Any]] = []

    def profile(nvtx: bool, secs: list[str], limit: Optional[float]) -> tuple[Optional[int], str]:
        report.unlink(missing_ok=True)
        cmd = profile_command(ncu_binary, report, sections=secs, metrics=metrics, nvtx=nvtx, forward=forward,
                              launch_cap=int(p.get("launch_count", LAUNCH_CAP)), kernel_path=job.kernel_path,
                              task_path=job.task_path, seed=int(p.get("seed", 0)),
                              iterations=int(p.get("harness_iterations", HARNESS_ITERATIONS)))
        started = time.perf_counter()
        rc, out, err = _run(cmd, work, limit)
        attempts.append({"selection": "nvtx" if nvtx else "kernel-id", "sections": len(secs),
                         "seconds": round(time.perf_counter() - started, 3), "timed_out": rc is None})
        return rc, out + err

    def done(**fields: Any) -> dict[str, Any]:
        return {"ok": False, "ncu_s": round(time.perf_counter() - t0, 3), "attempts": attempts, **fields}

    secs, degraded = sections, False
    rc, output = profile(True, secs, timeout)
    if "ERR_NVGPUCTRPERM" in output:
        return done(status="ncu_unavailable", returncode=rc,
                    error="GPU performance counters are not accessible to this user (ERR_NVGPUCTRPERM)")
    if rc is None:  # too slow: the headline sections only
        secs, degraded = list(MINIMAL_SECTIONS), True
        rc, output = profile(True, secs, max(30.0, (timeout or 60.0) / 2))
        if rc is None:
            return done(status="ncu_timeout", error=f"ncu ran past {timeout:.0f} s, and so did the minimal profile")
    result, error = (None, "ncu wrote no report")
    if report.is_file():
        result, error = import_report(ncu_binary, report)
    if not (result and result["kernels"]):  # NVTX selection captured nothing: each kernel's N-th call
        rc, output = profile(False, secs, timeout)
        if rc is None:
            return done(status="ncu_timeout", error="ncu ran past the timeout (kernel-id selection)")
        result, error = (import_report(ncu_binary, report) if report.is_file() else (None, "ncu wrote no report"))
    if not (result and result["kernels"]):
        return done(status="ncu_error", returncode=rc, error=error or "ncu profiled no kernel",
                    stdout_tail=_tail(output))
    kernels = result["kernels"]
    fields = {"profile": result, "kernels": len(kernels), "rules": sum(len(k["rules"]) for k in kernels),
              "selection": attempts[-1]["selection"], "degraded": degraded, "returncode": rc,
              "report_path": str(report),
              # the dominant kernel's metrics, for summaries that want a flat dict
              "metrics": kernels[0]["metrics"]}
    if rc != 0:
        return done(status="ncu_error", error="profiled process exited with a non-zero code",
                    stdout_tail=_tail(output), **fields)
    return {**done(**fields), "ok": True, "status": "success"}
