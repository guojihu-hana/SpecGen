#!/usr/bin/env python3
"""The artifact evaluation's report: SpecGen against CudaForge on every task whose two runs have started.

    python ae/report.py --runs ae_results/runs --prefix main --output ae_results

Writes, for the runs ``<prefix>-<task>-specgen`` / ``<prefix>-<task>-cudaforge``:
  <output>/report.html    (<prefix>_report.html for other prefixes, e.g. start_ for the getting-started test) the
                          page `specgen report` draws, one section per
                          task, under a comparison table that switches between termination criteria and
                          comparison dimensions (same iterations, time or equivalent tokens)
  <output>/summary.md, summary.csv   (<prefix>_summary.*) per task and the geometric mean, as the runs went
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from pathlib import Path

from specgen.analysis.criteria import CRITERIA, Ledger, equivalent, simulate, stopped_at
from specgen.analysis.data import RunData, align_resumed_devices, load_run
from specgen.analysis.report import build_payload, compare_payload, render_page, slot_rows

LABELS = ("SpecGen", "CudaForge")
DIMENSIONS = [  # value, label, formula shown beside the menu (HTML)
    ("time", "Same time", "<i>T</i> = min(<i>T</i><sub>SpecGen</sub>, <i>T</i><sub>CudaForge</sub>)"),
    ("tokens", "Same tokens", "<i>E</i> = min(<i>E</i><sub>SpecGen</sub>, <i>E</i><sub>CudaForge</sub>)"),
    ("iters", "Same iterations", "<i>N</i>: the same number of iterations for both"),
]
FRAC = "<span class=\"frac\"><span>{}</span><span>{}</span></span>"
DEFINITIONS = [
    "<span class=\"math\"><i>T</i><sub>SpecGen</sub></span>, <span class=\"math\"><i>T</i><sub>CudaForge</sub></span>"
    ": each run's time under the chosen termination criterion (<span class=\"math\"><i>E</i></span> likewise, in "
    "equivalent tokens); both runs are cut at <span class=\"math\"><i>T</i></span> or "
    "<span class=\"math\"><i>E</i></span>",
    "Equivalent tokens <span class=\"math\"><i>E</i> = " + FRAC.format("<i>x</i><sub>hit</sub>", "50")
    + " + <i>x</i><sub>miss</sub> + 4<i>y</i></span> (<span class=\"math\"><i>x</i><sub>hit</sub></span>, "
    "<span class=\"math\"><i>x</i><sub>miss</sub></span>: cache-hit and cache-miss input; "
    "<span class=\"math\"><i>y</i></span>: output); M = <span class=\"math\">10<sup>6</sup></span>",
    "GPU utilization = <span class=\"math\">" + FRAC.format("time with jobs running", "time")
    + "</span> (profiler: benchmark, ncu; validator: validation)",
    "Iterations <span class=\"math\"><i>N</i>+</span>: <span class=\"math\"><i>N</i></span> finished, stopped during "
    "the next one",
    "▲ / ▼: SpecGen's final best above / below CudaForge's (hover for the ratio)",
    "Estimate = the actual timeline cut where the criterion would have ended each iteration; stop on reaching "
    "CudaForge's final best: CudaForge in full",
    "CudaForge = SpecGen's baseline mode (one kernel per iteration, no spec, no early termination); both runs use the "
    "same GPUs and CPUs",
]
STYLE = """
#overview th { position: static; }
#overview th.run-a { color: var(--run-a); } #overview th.run-b { color: var(--run-b); }
#overview .up { color: var(--ok); } #overview .down { color: var(--accent); }
.ov-tools { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; color: var(--muted); }
.ov-tools select { background: var(--panel2); color: var(--text); border: 1px solid #2c3240; border-radius: 6px; padding: 4px; }
.ov-tools select:disabled { opacity: .45; }
.ov-tools .sep { width: 16px; }
.ov-tools .math { color: var(--text); margin-left: 6px; }
"""


def iterations(data: RunData) -> int:
    return len({s["iteration"] for s in data.spans})


def task_order(run_dir: Path) -> tuple:
    """T1-T20 in the paper's order, then other tasks by KernelBench level and number."""
    name = run_dir.name.split("-")[1]
    if re.fullmatch(r"T\d+", name):
        return (0, int(name[1:]), 0)
    try:
        task = json.loads((run_dir / "config.json").read_text())["run"]["tasks"][0]
        level, number = re.search(r"level(\d+)/(\d+)_", task).groups()
        return (1, int(level), int(number))
    except (OSError, ValueError, KeyError, IndexError, AttributeError):
        return (2, 0, 0)


def outcome(result: dict) -> dict:
    return {"best": result.get("best"), "seconds": result["seconds"], "tokens": result["tokens"],
            "iterations": result.get("iterations"), "cut": result.get("cut", False),
            "profiler": result["profiler"]["share"], "validator": result["validator"]["share"]}


def cells_of(spec: RunData, base: RunData, compare: dict) -> dict[str, dict]:
    """Both runs per ``criterion|dimension`` (and ``reach``): ``{"spec": ..., "base": ...}``."""
    task = spec.tasks[0]
    spec_ledger, base_ledger = Ledger(spec), Ledger(base)
    base_full = simulate(base, None, ledger=base_ledger)[task]
    cells = {}
    for criterion in CRITERIA:
        spec_full = simulate(spec, criterion, ledger=spec_ledger)[task]
        cells[f"{criterion}|iters"] = {"spec": outcome(spec_full), "base": outcome(base_full)}
        budgets = {"time": ("seconds", min(spec_full["seconds"], base_full["seconds"])),
                   "tokens": ("tokens", min(equivalent(spec_full["tokens"]), equivalent(base_full["tokens"])))}
        for dimension, budget in budgets.items():
            cells[f"{criterion}|{dimension}"] = {
                "spec": outcome(simulate(spec, criterion, ledger=spec_ledger, budget=budget)[task]),
                "base": outcome(simulate(base, None, ledger=base_ledger, budget=budget)[task])}
    a, b = compare["rows"][0]["a"], compare["rows"][0]["b"]
    if a.get("reach") is not None:
        reach_t = spec.t0 + a["reach"]
        stop = stopped_at(spec, task, reach_t, ledger=spec_ledger)
        cells["reach"] = {"spec": {**outcome({**stop, "cut": True}),
                                   "best": next(v for t, v in compare["curves"][task][0] if v >= b["best"]),
                                   "iterations": sum(1 for s in spec.spans if s["task"] == task and s["end"] <= reach_t)},
                          "base": outcome(base_full)}
    return cells


def section(name: str, spec_dir: Path, base_dir: Path, note: str = "") -> dict:
    spec, base = load_run(spec_dir), load_run(base_dir)
    for data in (spec, base):
        align_resumed_devices(data)
    spec_slots, base_slots = slot_rows(spec, base)
    task = spec.tasks[0]
    compare = compare_payload(spec, base, LABELS, slots=base_slots)
    title = f"SpecGen · {task.split('/')[-1]} · {iterations(spec)} iter"
    criterion = (spec.config.get("termination") or {}).get("criteria")
    return {**build_payload(spec, title, slots=spec_slots), "anchor": name, "note": note, "compare": compare,
            "level": task.split("/")[0], "actual": criterion, "overview": cells_of(spec, base, compare),
            "iterations_run": [iterations(spec), iterations(base)]}


def overview(sections: list[dict]) -> str:
    millions = lambda v: f"{v / 1e6:.2f}M"  # noqa: E731
    percent = lambda v: "-" if v is None else f"{v * 100:.0f}%"  # noqa: E731
    actual = sections[0].get("actual") or "best"
    criteria = [(c, f"{c} ({'actual run' if c == actual else 'estimate'})") for c in CRITERIA]
    criteria.sort(key=lambda option: option[0] != actual)
    criteria.append(("reach", "stop on reaching CudaForge's final best"))
    default = f"{actual}|time"

    def arrow(best, other):
        if not best or not other or best == other:
            return ""
        ratio = f"{best / other:.2f}x of CudaForge"
        return f" <span class=\"up\" title=\"{ratio}\">▲</span>" if best > other else f" <span class=\"down\" title=\"{ratio}\">▼</span>"

    def cells(run, other=None):
        best = "-" if run["best"] is None else f"{run['best']:.2f}×" + (arrow(run["best"], other["best"]) if other else "")
        tok = run["tokens"]
        return [best, f"{run['iterations']}{'+' if run['cut'] else ''}", f"{run['seconds'] / 60:.1f} min",
                f"{millions(tok['in'] - tok['in_hit_free'])} / {millions(tok['in_hit_free'])}", millions(tok["out"]),
                millions(equivalent(tok)), f"{percent(run['profiler'])} / {percent(run['validator'])}"]

    keys = [f"{c}|{d}" for c in CRITERIA for d, _, _ in DIMENSIONS] + ["reach"]
    bodies = []
    for key in keys:
        rows = []
        for s in sections:
            pair = s["overview"].get(key)
            spec_cells = cells(pair["spec"], pair["base"]) if pair else ["not reached"] * 7
            base_cells = cells(pair["base"] if pair else s["overview"][f"{actual}|iters"]["base"])
            row = [f"<a href=\"#{s['anchor']}\">{s['anchor']} · {html.escape(s['tasks'][0].split('/')[-1])}</a>",
                   html.escape(s["level"])] + [c for both in zip(spec_cells, base_cells) for c in both]
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
        bodies.append(f"<tbody data-key=\"{key}\"{'' if key == default else ' hidden'}>" + "".join(rows) + "</tbody>")
    groups = ["Final best", "Iterations", "Time", "Input tokens: cache hit / miss", "Output tokens", "Equivalent tokens",
              "GPU utilization: profiler / validator"]
    head = ("<thead><tr><th rowspan=\"2\">Task</th><th rowspan=\"2\">KernelBench</th>"
            + "".join(f"<th colspan=\"2\">{g}</th>" for g in groups) + "</tr><tr>"
            + "<th class=\"run-a\">SpecGen</th><th class=\"run-b\">CudaForge</th>" * len(groups) + "</tr></thead>")
    select = lambda name, options, chosen: (f"<select id=\"{name}\">" + "".join(  # noqa: E731
        f"<option value=\"{o[0]}\"{' selected' if o[0] == chosen else ''}>{html.escape(o[1])}</option>" for o in options)
        + "</select>")
    formulas = "".join(f"<span class=\"math\" data-dim=\"{v}\"{'' if v == 'time' else ' hidden'}>{f}</span>"
                       for v, _, f in DIMENSIONS)
    tools = ("<div class=\"ov-tools\">SpecGen termination criterion " + select("ov-criterion", criteria, actual)
             + "<span class=\"sep\"></span>Compare at " + select("ov-dimension", DIMENSIONS, "time") + formulas + "</div>")
    toggle = """<script>(() => {
  const criterion = document.getElementById("ov-criterion"), dimension = document.getElementById("ov-dimension");
  const show = () => {
    const reach = criterion.value === "reach";
    dimension.disabled = reach;
    const key = reach ? "reach" : `${criterion.value}|${dimension.value}`;
    for (const body of document.querySelectorAll("#overview tbody")) body.hidden = body.dataset.key !== key;
    for (const f of document.querySelectorAll(".ov-tools [data-dim]")) f.hidden = reach || f.dataset.dim !== dimension.value;
  };
  criterion.addEventListener("change", show);
  dimension.addEventListener("change", show);
})();</script>"""
    definitions = "<ul class=\"defs\">" + "".join(f"<li>{d}</li>" for d in DEFINITIONS) + "</ul>"
    return ("<section class=\"panel\" style=\"margin-left:0;margin-right:0\"><h2>Performance comparison</h2>" + tools
            + "<table id=\"overview\">" + head + "".join(bodies) + "</table>" + toggle + definitions + "</section>")


SUMMARY_COLUMNS = ["task", "kernelbench", "iterations", "e2e_min", "time_speedup", "profiled_kernels", "feedback_ratio",
                   "best_speedup", "speedup_ratio", "profiler_util", "validator_util", "input_hit_M", "input_miss_M",
                   "output_M", "equivalent_M"]


def summary(sections: list[dict]) -> tuple[list[dict], dict]:
    """Per task, as the runs went (SpecGen under its own criterion, CudaForge in full), and the geometric means."""
    rows, ratios = [], {"time_speedup": [], "feedback_ratio": [], "speedup_ratio": []}
    for s in sections:
        actual = s.get("actual") or "best"
        pair = s["overview"][f"{actual}|iters"]
        a, b = pair["spec"], pair["base"]
        compare_row = s["compare"]["rows"][0]
        both = lambda key, f: f"{f(a[key])} / {f(b[key])}"  # noqa: E731
        tok = lambda run, kind: (run["tokens"]["in"] - run["tokens"]["in_hit_free"] if kind == "hit" else  # noqa: E731
                                 run["tokens"]["in_hit_free"] if kind == "miss" else run["tokens"]["out"])
        time_speedup = b["seconds"] / a["seconds"] if a["seconds"] else None
        feedback = (compare_row["a"]["valid"], compare_row["b"]["valid"])
        feedback_ratio = feedback[0] / feedback[1] if feedback[1] else None
        speedup_ratio = a["best"] / b["best"] if a["best"] and b["best"] else None
        for key, value in (("time_speedup", time_speedup), ("feedback_ratio", feedback_ratio), ("speedup_ratio", speedup_ratio)):
            if value:
                ratios[key].append(value)
        rows.append({
            "task": s["anchor"], "kernelbench": s["tasks"][0], "iterations": "{} / {}".format(*s["iterations_run"]),
            "e2e_min": both("seconds", lambda v: f"{v / 60:.1f}"),
            "time_speedup": f"{time_speedup:.2f}x" if time_speedup else "-",
            "profiled_kernels": f"{feedback[0]} / {feedback[1]}", "feedback_ratio": f"{feedback_ratio:.2f}x" if feedback_ratio else "-",
            "best_speedup": both("best", lambda v: f"{v:.3f}x" if v else "-"),
            "speedup_ratio": f"{speedup_ratio:.2f}x" if speedup_ratio else "-",
            "profiler_util": both("profiler", lambda v: f"{v * 100:.1f}%" if v is not None else "-"),
            "validator_util": both("validator", lambda v: f"{v * 100:.1f}%" if v is not None else "-"),
            "input_hit_M": f"{tok(a, 'hit') / 1e6:.2f} / {tok(b, 'hit') / 1e6:.2f}",
            "input_miss_M": f"{tok(a, 'miss') / 1e6:.2f} / {tok(b, 'miss') / 1e6:.2f}",
            "output_M": f"{tok(a, 'out') / 1e6:.2f} / {tok(b, 'out') / 1e6:.2f}",
            "equivalent_M": f"{equivalent(a['tokens']) / 1e6:.2f} / {equivalent(b['tokens']) / 1e6:.2f}",
        })
    geomean = {key: math.exp(sum(map(math.log, values)) / len(values)) if values else None for key, values in ratios.items()}
    return rows, geomean


def write_summary(rows: list[dict], geomean: dict, path_md: Path, path_csv: Path) -> str:
    with path_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# SpecGen vs CudaForge", "",
             "Each cell is `SpecGen / CudaForge`, as the runs went: SpecGen with its termination criterion, CudaForge "
             "(SpecGen's baseline mode: one kernel per iteration, no speculative generation, no early termination) in full. "
             "Equivalent tokens = cache-hit input / 50 + cache-miss input + 4 x output. Utilization = busy time / run time.",
             "", "| " + " | ".join(SUMMARY_COLUMNS) + " |", "|" + "---|" * len(SUMMARY_COLUMNS)]
    lines += ["| " + " | ".join(str(row[c]) for c in SUMMARY_COLUMNS) + " |" for row in rows]
    fmt = lambda v: f"{v:.2f}x" if v else "-"  # noqa: E731
    lines += ["", f"Geometric mean over {len(rows)} task(s): E2E time speedup {fmt(geomean['time_speedup'])}, "
                  f"profiling feedback {fmt(geomean['feedback_ratio'])}, best kernel speedup {fmt(geomean['speedup_ratio'])} "
                  "(SpecGen relative to CudaForge)."]
    text = "\n".join(lines) + "\n"
    path_md.write_text(text)
    return text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--prefix", default="main")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default=None, help="of the page (default: SpecGen vs CudaForge and the task count)")
    p.add_argument("--note", nargs=2, action="append", default=[], metavar=("TASK", "HTML"),
                   help="a note under the title of TASK's section")
    a = p.parse_args()
    notes = dict(a.note)
    pairs = []
    for spec_dir in sorted(a.runs.glob(f"{a.prefix}-*-specgen"), key=task_order):
        name = spec_dir.name[len(a.prefix) + 1:-len("-specgen")]
        base_dir = a.runs / f"{a.prefix}-{name}-cudaforge"
        if all((d / "iterations.jsonl").is_file() and (d / "iterations.jsonl").stat().st_size for d in (spec_dir, base_dir)):
            pairs.append((name, spec_dir, base_dir, notes.get(name, "")))
    if not pairs:
        print(f"no {a.prefix} task has both runs with a finished iteration yet")
        return 0
    sections = [section(*pair) for pair in pairs]
    a.output.mkdir(parents=True, exist_ok=True)
    stem = "" if a.prefix == "main" else f"{a.prefix}_"
    title = a.title or f"SpecGen vs CudaForge · {len(sections)} KernelBench task{'s' if len(sections) > 1 else ''}"
    page = render_page(sections, title, intro_html=overview(sections), extra_style=STYLE)
    (a.output / f"{stem}report.html").write_text(page, encoding="utf-8")
    rows, geomean = summary(sections)
    text = write_summary(rows, geomean, a.output / f"{stem}summary.md", a.output / f"{stem}summary.csv")
    print(text)
    print(f"wrote {a.output / f'{stem}report.html'} and {a.output / f'{stem}summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
