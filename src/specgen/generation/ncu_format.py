"""Rendering of an ncu profile (``evaluation.ncu``): one forward pass, every kernel it launches.

Values are in ncu's base units (percent, byte, byte/second, ...).  Durations measured under ncu
(locked clocks, flushed caches) are not shown: the measured speedup already says how fast the module
is; a kernel's share of the forward's GPU time is, when the forward launches more than one kernel.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

# tensor pipe: active cycles, which count tcgen05 (Blackwell) as well as mma.sync / wmma; executed-
# instruction counters see only the latter
TENSOR_PIPE = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
TCGEN05_PIPE = "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active"
THROUGHPUT = (
    ("SM", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("memory", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("tensor pipe", TENSOR_PIPE),
)
MEMORY = (
    ("DRAM", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L1", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
    ("L2", "lts__throughput.avg.pct_of_peak_sustained_active"),
)
OCCUPANCY_LIMITS = (
    ("registers", "launch__occupancy_limit_registers"),
    ("shared memory", "launch__occupancy_limit_shared_mem"),
    ("warps", "launch__occupancy_limit_warps"),
    ("blocks", "launch__occupancy_limit_blocks"),
)
STALL = re.compile(r"^smsp__warp_issue_stalled_(\w+?)_per_warp_active\.pct$")
SUMMARY_RULE = "SOLBottleneck"

# sentences of ncu rule text that point at its GUI or documentation rather than at the kernel
_UI_SENTENCE = re.compile(
    r"(?i)\b(?:see the|check the|kernel profiling guide|best practices guide|programming guide|source page)\b"
    r"|https?://")


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "?"
    return f"{value:.0f}%" if abs(value) >= 10 else f"{value:.1f}%"


def _bandwidth(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value / 1e12:.2f} TB/s" if value >= 1e12 else f"{value / 1e9:.0f} GB/s"


def _int(value: Optional[float]) -> str:
    return "?" if value is None else f"{value:.0f}"


def clean_rule_text(text: str, limit: int) -> str:
    text = re.sub(r"\s*\(https?://[^)]*\)", "", " ".join(text.split()))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(s for s in sentences if not _UI_SENTENCE.search(s)) or sentences[0]
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def occupancy_limiters(m: Mapping[str, float]) -> list[str]:
    """What bounds the blocks per SM (the smallest launch__occupancy_limit_*)."""
    limits = {name: m[key] for name, key in OCCUPANCY_LIMITS if key in m}
    if not limits:
        return []
    low = min(limits.values())
    return [name for name, v in limits.items() if v == low]


def advice(kernel: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """ncu's optimization advice and warnings for a kernel, largest estimated speedup first."""
    rules = [r for r in kernel.get("rules") or () if r.get("type") in ("OPT", "WRN") and r.get("rule") != SUMMARY_RULE]
    return sorted(rules, key=lambda r: -(r.get("speedup") or 0.0))


def _speedup_tag(rule: Mapping[str, Any]) -> str:
    if not rule.get("speedup"):
        return ""
    local = "local " if rule.get("speedup_type") == "local" else ""
    return f"est. {local}speedup {rule['speedup']:.3g}%"


def kernel_lines(kernel: Mapping[str, Any]) -> list[str]:
    """Throughput, memory, occupancy and stall lines of one kernel."""
    m = kernel.get("metrics") or {}
    lines = []
    tp = [f"{label} {_pct(m[key])}" + (" (tcgen05)" if key == TENSOR_PIPE and m.get(TCGEN05_PIPE, 0.0) > 1.0 else "")
          for label, key in THROUGHPUT if key in m]
    if tp:
        lines.append("throughput (% of peak): " + ", ".join(tp))
    mem = [f"{label} {_pct(m[key])}" for label, key in MEMORY if key in m]
    if "dram__bytes.sum.per_second" in m:
        mem.append(f"DRAM bandwidth {_bandwidth(m['dram__bytes.sum.per_second'])}")
    if mem:
        lines.append("memory (% of peak): " + ", ".join(mem))
    occ = []
    if "sm__warps_active.avg.pct_of_peak_sustained_active" in m:
        occ.append(f"achieved {_pct(m['sm__warps_active.avg.pct_of_peak_sustained_active'])}")
    if "sm__maximum_warps_per_active_cycle_pct" in m:
        occ.append(f"theoretical {_pct(m['sm__maximum_warps_per_active_cycle_pct'])}")
    limiters = occupancy_limiters(m)
    if limiters:
        occ.append("limited by " + " and ".join(limiters))
    launch = []
    if "launch__registers_per_thread" in m:
        launch.append(f"{_int(m['launch__registers_per_thread'])} registers/thread")
    smem = sum(m.get(k, 0.0) for k in ("launch__shared_mem_per_block_static", "launch__shared_mem_per_block_dynamic"))
    if any(k in m for k in ("launch__shared_mem_per_block_static", "launch__shared_mem_per_block_dynamic")):
        launch.append(f"{smem / 1024:.1f} KB shared memory/block")
    if "launch__block_size" in m:
        launch.append(f"{_int(m['launch__block_size'])} threads/block")
    if "launch__grid_size" in m:
        launch.append(f"{_int(m['launch__grid_size'])} blocks")
    if occ or launch:
        lines.append("occupancy: " + ", ".join(occ + launch))
    stalls = sorted(((STALL.match(k).group(1).replace("_", " "), v) for k, v in m.items() if STALL.match(k)),
                    key=lambda kv: -kv[1])
    if stalls:
        lines.append("warp stalls (% of active warp cycles): " + ", ".join(f"{n} {v:.1f}" for n, v in stalls[:4]))
    conflicts = m.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
    if conflicts:
        lines.append(f"shared-memory load bank conflicts: {_int(conflicts)}")
    return lines


def format_ncu_profile(profile: Mapping[str, Any], limit: int, *, max_kernels: int = 3, min_share: float = 0.05,
                       max_rules: int = 6, rule_chars: int = 700) -> str:
    """The prompt block: each kernel of the forward with at least ``min_share`` of its GPU time."""
    kernels = list(profile.get("kernels") or ())
    if not kernels:
        return ""
    many = len(kernels) > 1
    lines = [f"Nsight Compute profile of one forward pass ({len(kernels)} kernel{'s' if many else ''}"
             + ("; shares of the forward's GPU time" if many else "") + "; ncu runs with locked clocks):"]
    for i, kernel in enumerate(kernels[:max_kernels]):
        share = kernel.get("share")
        if i > 0 and (share is None or share < min_share):
            break
        lines.append(f"- `{kernel.get('name', '?')}`" + (f" ({share:.0%})" if many and share is not None else "") + ":")
        lines += [f"  - {line}" for line in kernel_lines(kernel)]
        headline = [r for r in kernel.get("rules") or () if r.get("rule") == SUMMARY_RULE][:1]
        rules = (headline + advice(kernel))[:max_rules if i == 0 else 3]
        if rules:
            lines.append("  - ncu analysis:")
            for rule in rules:
                tag = str(rule.get("type") or "INF") + (f", {_speedup_tag(rule)}" if _speedup_tag(rule) else "")
                lines.append(f"    - [{tag}] {rule.get('rule')}: {clean_rule_text(rule.get('text', ''), rule_chars)}")
    while len(lines) > 1 and len("\n".join(lines)) > limit:
        lines.pop()
    return "\n".join(lines)


def ncu_summary_cell(profile: Optional[Mapping[str, Any]], *, max_advice: int = 2) -> str:
    """One table cell: the dominant kernel's key numbers and ncu's top advice."""
    kernels = list((profile or {}).get("kernels") or ())
    if not kernels:
        return "-"
    k = kernels[0]
    m = k.get("metrics") or {}
    parts = []
    if len(kernels) > 1:
        parts.append(f"{len(kernels)} kernels, the main one {k.get('share') or 0:.0%}")
    for label, key in (("SM", THROUGHPUT[0][1]), ("tensor", TENSOR_PIPE), ("DRAM", MEMORY[0][1])):
        if key in m:
            parts.append(f"{label} {_pct(m[key])}")
    if "sm__warps_active.avg.pct_of_peak_sustained_active" in m:
        limiters = occupancy_limiters(m)
        parts.append(f"occupancy {_pct(m['sm__warps_active.avg.pct_of_peak_sustained_active'])}"
                     + (f" (limit: {', '.join(limiters)})" if limiters else ""))
    if "launch__registers_per_thread" in m:
        parts.append(f"{_int(m['launch__registers_per_thread'])} regs")
    top = advice(k)[:max_advice]
    text = ", ".join(parts)
    if top:
        text += "; advice: " + ", ".join(f"{r.get('rule')}" + (f" +{r['speedup']:.3g}%" if r.get("speedup") else "")
                                         for r in top)
    return text or "-"
