"""What the earlier iterations of a task tried, as a compact table for the next prompt.

Rules only: each kernel's self-reported ``# APPROACH:`` line, features read from its code, and its
validation and timing results.  Kernels with the same code features form one approach; its row says
how many were tried and passed and the best measured speedup, and a status that follows from those
numbers, and lists the block tile and pipeline depth variants the kernels report with their own
counts.  Recurring failures are grouped by their normalized error message, with a hint for the
common ones.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping, Optional

from specgen.control.compliance import CATEGORY_LIBRARY, CATEGORY_PRECISION, violations
from specgen.generation.ncu_format import ncu_summary_cell

APPROACH_LINE = re.compile(r"^[ \t]*#[ \t]*APPROACH:[ \t]*(\S[^\n]*?)[ \t]*$", re.M)
# kernels that never reached a verdict (or have no code) say nothing about an approach
NOT_EVALUATED = {"duplicate", "cancelled", "grace_truncated", "llm_error", "parse_error", "pending", "evaluating"}
DEAD_END_TRIES = 5
WELL_BELOW = 0.8  # of the best speedup
MAX_APPROACH_CHARS = 160
MAX_VARIANTS = 4  # tile / stage variants listed per approach in a capped table

PRECISIONS = (
    ("int8", r"mma\.sync[^\"\n]*\.s8|\.s32\.s8\.s8|__dp4a"),
    ("fp8", r"__nv_fp8|\.e4m3|\.e5m2"),
    ("bf16", r"mma\.sync[^\"\n]*\.bf16|__nv_bfloat16|kBFloat16|bfloat16"),
    # before fp16: TF32 WMMA declares wmma:: fragments too
    ("tf32", r"mma\.sync[^\"\n]*\.tf32|wgmma[^\"\n]*\.tf32|precision::tf32|__float_to_tf32|cvt\.rna\.tf32|tfloat32"),
    ("fp16", r"mma\.sync[^\"\n]*\.f16|wgmma[^\"\n]*\.f16|__half2?\b|\bhalf2\b|wmma::fragment<[^;>]*\bhalf\b|kHalf"
             r"|\.half\(\)|float16"),
)
METHODS = (
    ("wgmma", r"wgmma"),
    ("mma.sync", r"mma\.sync"),
    ("wmma", r"wmma::"),
    ("library GEMM", r"torch\.(?:matmul|mm|bmm|addmm|einsum)\b|F\.linear|(?:torch|at)::(?:matmul|mm|addmm)\b|cublas"),
)
TECHNIQUES = (
    ("cp.async", r"cp\.async|__pipeline_memcpy_async|cuda::memcpy_async"),
    ("TMA", r"cp\.async\.bulk|CUtensorMap|cuTensorMap"),
    ("ldmatrix", r"ldmatrix"),
    ("split-K", r"(?i)split[_-]?k"),
    ("warp shuffle", r"__shfl"),
    ("vectorized", r"\b(?:float4|half2|__half2|int4|uint4|__nv_bfloat162)\b"),
    ("atomics", r"\batomic(?:Add|Max|Min|CAS|Exch)\b"),
)
HINTS = (
    (r"no attribute 'Model'", "keep the class name Model"),
    (r"misaligned address", "check alignment before vectorized or cp.async accesses"),
    (r"illegal memory access", "check index bounds and allocation sizes, and call cudaFuncSetAttribute(kernel, "
                               "cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) before launching with more than "
                               "48 KB of dynamic shared memory"),
    (r"invalid argument", "check launch configuration and shared-memory size"),
    (r"is undefined|was not declared|has no member", "include the header or declare it"),
    (r"undefined (?:symbol|reference)", "define every function that is declared or bound, with the same signature"),
    (r"asm operand must be an integral constant", "pass literals or template parameters to \"n\" asm operands"),
    (r"unsupported toolchain|no kernel image is available",
     "do not pass -arch / -gencode flags: kernels are built for the GPU they run on"),
    (r"shifted inputs", "do not assume the input range of get_inputs()"),
    (r"uses too much shared data", "use less shared memory per block, or raise the limit with cudaFuncSetAttribute"),
    (r"unterminated triple-quoted string", "the module was cut off: keep it shorter"),
    (r"library GEMM|library kernels|matmul precision",
     "do all the work in hand-written CUDA kernels: no library call, no fallback to one, no TF32 switch"),
)
# a build that failed when the module was imported for validation (e.g. a second load_inline)
BUILD_FAILED = re.compile(r"Error building extension")
# block tile in a self-report, e.g. 128x128x32 (not a warp tile or a thread layout); pipeline depth
TILE = re.compile(r"(?<![\w.])(\d{2,3})\s*[x×]\s*(\d{2,3})(?:\s*[x×]\s*(\d{1,3}))?(?![\w.])")
STAGES = re.compile(r"(?i)\b([1-9])[- ]?stages?\b|\b(double|triple)[- ]?buffer")


def _code_only(code: str) -> str:
    return re.sub(r"//[^\n]*|#[^\n]*", "", re.sub(r"/\*.*?\*/", "", code, flags=re.S))


@lru_cache(maxsize=8192)
def features(code: str) -> tuple[str, str, tuple[str, ...]]:
    """(precision, compute method, techniques) found in the code (comments ignored); a library call or
    a TF32 switch in the Python code (``control.compliance``) is listed with the techniques."""
    body = _code_only(code)
    precision = next((name for name, pattern in PRECISIONS if re.search(pattern, body)), "fp32")
    method = next((name for name, pattern in METHODS if re.search(pattern, body)), "SIMT")
    techniques = tuple(name for name, pattern in TECHNIQUES if re.search(pattern, body))
    found = {category for category, _ in violations(code)}
    techniques += (("library call",) if CATEGORY_LIBRARY in found else ()) + (
        ("TF32 switch",) if CATEGORY_PRECISION in found else ())
    return precision, method, techniques


def family(code: str) -> str:
    precision, method, techniques = features(code)
    return f"{precision} {method}" + (f" + {', '.join(techniques)}" if techniques else "")


def self_reported(code: str) -> str:
    match = APPROACH_LINE.search(code or "")
    if not match:
        return ""
    text = match.group(1).strip()
    return text if len(text) <= MAX_APPROACH_CHARS else text[:MAX_APPROACH_CHARS - 1] + "…"


def tiling(reported: str) -> str:
    """Block tile and pipeline depth a self-report names, e.g. ``128x128x32, 4 stages`` ("" if neither)."""
    tile = ""
    for match in TILE.finditer(reported):
        dims = [int(d) for d in match.groups() if d]
        near = f"{reported[max(0, match.start() - 10):match.start()]} {reported[match.end():match.end() + 8]}".lower()
        if (min(dims[:2]) >= 16 and max(dims) <= 512 and (len(dims) < 3 or dims[2] >= 4)
                and "warp" not in near and "thread" not in near):
            tile = "x".join(map(str, dims))
            break
    stages = ""
    match = STAGES.search(reported)
    if match:
        n = match.group(1) or {"double": "2", "triple": "3"}[match.group(2).lower()]
        stages = f"{n} stage{'' if n == '1' else 's'}"
    return ", ".join(part for part in (tile, stages) if part)


def failure_kind(status: str, error: Optional[str]) -> str:
    """The status, except that a build failing during validation counts as a compile error."""
    return "compile_error" if BUILD_FAILED.search(error or "") else status


def _compiler_error(text: str) -> str:
    found = re.findall(r"error(?: #\d+-D)?\s*: ([^\n]+)", text)
    if not found or found[0].startswith("ld returned"):
        undefined = re.findall(r"undefined reference to ([^\n]+)", text)
        if undefined:
            return f"undefined reference to {undefined[0]}"
    found = found or re.findall(r"(?:ptxas|nvcc) (?:fatal|error)\s*: ([^\n]+)", text)
    return found[0] if found else ""


def failure(status: str, error: Optional[str]) -> tuple[str, str]:
    """(cluster key, example message) of a failed kernel."""
    text = error or ""
    if status == "numerical_error":
        rel = re.search(r'"rel_err": ([0-9.eE+-]+)', text)
        example = f"relative error {float(rel.group(1)):.2g}" if rel else ""
        if '"trial": "shift"' in text or "shifted inputs" in text:
            return "wrong results on shifted inputs", example
        if '"reason": "relative_error"' in text:
            return "wrong results (small values, relative error too large)", example
        return "wrong results", example
    if status == "timeout":
        return "timed out", ""
    if status == "forbidden":
        category, _, what = text.partition(": ")
        return category, what[:160]
    message = ""
    if status == "compile_error" or BUILD_FAILED.search(text):
        message = _compiler_error(text)
        if not message and BUILD_FAILED.search(text):
            message = "the extension failed to build (no compiler error in the log)"
    if not message:
        cuda = re.findall(r"CUDA error: ([^\n]+)", text)
        message = f"CUDA error: {cuda[0]}" if cuda else ""
    if not message:
        found = re.findall(r"\b(\w+(?:Error|Exception): [^\n]+)", text)
        message = found[-1] if found else status.replace("_", " ")
    message = message.strip()
    key = re.sub(r'"[^"\n]*"|\'[^\'\n]*\'|‘[^’\n]*’|`[^`\n]*\'', "…", message)
    key = re.sub(r"undefined (symbol: |reference to )\S+", r"undefined \1…", key)
    key = re.sub(r"(/[\w.\-]+)+", "<path>", key)
    key = re.sub(r"\b0x[0-9a-fA-F]+\b|\b\d+\b", "N", key)[:120]  # line numbers, sizes
    return key, _readable(message)[:160]


PATH_DIRS = re.compile(r"(?:/[\w.\-]+)+/(?=[\w.\-])")
MANGLED = re.compile(r"(undefined (?:symbol: |reference to `?))(_Z\w+)")


def _readable(message: str) -> str:
    """The message without directories, with undefined C++ symbols demangled (which declaration is unmatched)."""
    return MANGLED.sub(lambda m: m.group(1) + _demangle(m.group(2)), PATH_DIRS.sub("", message))


@lru_cache(maxsize=1024)
def _demangle(symbol: str) -> str:
    try:
        out = subprocess.run(["c++filt", symbol], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return symbol
    return out or symbol


@dataclass
class _Variant:  # one self-reported block tile and pipeline depth within an approach
    tried: int = 0
    passed: int = 0
    best: Optional[float] = None


@dataclass
class _Approach:
    label: str  # code features
    tried: int = 0
    passed: int = 0
    failed: Counter = field(default_factory=Counter)
    best: Optional[float] = None
    best_kernel: str = ""
    best_profile: Optional[Mapping[str, Any]] = None  # ncu profile of the best kernel
    reported: str = ""
    reported_by_best: bool = False
    variants: dict[str, _Variant] = field(default_factory=dict)

    def variants_cell(self, limit: int) -> str:
        """``128x128x32, 4 stages: 40/22 1.456x; ...`` (tried/passed, best); "-" if none reported."""
        if set(self.variants) <= {""}:
            return "-"
        ranked = sorted(self.variants.items(), key=lambda kv: (kv[1].best is None, -(kv[1].best or 0.0),
                                                               -kv[1].tried, kv[0]))
        shown, rest = (ranked[:limit], ranked[limit:]) if limit else (ranked, [])
        parts = [f"{name or 'not reported'}: {v.tried}/{v.passed}" + (f" {v.best:.3f}x" if v.best is not None else "")
                 for name, v in shown]
        if rest:
            parts.append(f"{len(rest)} more: {sum(v.tried for _, v in rest)}/{sum(v.passed for _, v in rest)}")
        return "; ".join(parts)


@dataclass
class _Failure:
    status: str
    key: str
    example: str
    count: int = 0
    approaches: Counter = field(default_factory=Counter)


def summarize(outcomes: Iterable, *, best_kernel: Optional[str] = None, max_approaches: int = 8,
              max_failures: int = 5, max_chars: int = 6000) -> str:
    """The table for the next prompt; a limit of 0 means none.  A capped table lists at most
    ``MAX_VARIANTS`` tile / stage variants per approach, an uncapped one all of them."""
    approaches: dict[str, _Approach] = {}
    failures: dict[tuple[str, str], _Failure] = {}
    iterations, total = set(), 0
    for o in outcomes:
        if o.status in NOT_EVALUATED or not o.code:
            continue
        total += 1
        iterations.add(o.iteration)
        reported = self_reported(o.code)
        label, tile = family(o.code), tiling(reported)
        a = approaches.setdefault(label, _Approach(label))
        v = a.variants.setdefault(tile, _Variant())
        a.tried += 1
        v.tried += 1
        if o.valid and o.speedup is not None:
            a.passed += 1
            v.passed += 1
            v.best = o.speedup if v.best is None else max(v.best, o.speedup)
            if a.best is None or o.speedup > a.best:
                a.best, a.best_kernel = o.speedup, o.kernel_id
                a.best_profile = getattr(o, "ncu_profile", None)
                if reported:
                    a.reported, a.reported_by_best = reported, True
        elif o.status != "success":
            kind = failure_kind(o.status, o.error)
            a.failed[kind] += 1
            key, example = failure(kind, o.error)
            f = failures.setdefault((kind, key), _Failure(kind, key, example))
            f.count += 1
            f.approaches[f"{label} | {tile}" if tile else label] += 1
        if reported and not a.reported_by_best:
            a.reported = reported
    if not total:
        return ""
    top = max((a.best for a in approaches.values() if a.best is not None), default=None)
    rows = sorted(approaches.values(), key=lambda a: (a.best is None, -(a.best or 0.0), -a.tried, a.label))
    shown, rest = (rows[:max_approaches], rows[max_approaches:]) if max_approaches else (rows, [])
    max_variants = MAX_VARIANTS if max_approaches else 0

    def status(a: _Approach) -> str:
        if a.best_kernel and a.best_kernel == best_kernel:
            return "best so far"
        if a.passed == 0:
            parts = ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in a.failed.most_common())
            return f"dead end: all {a.tried} failed ({parts})" if a.tried >= DEAD_END_TRIES else f"all failed ({parts})"
        if a.best < 1.0:
            return "slower than the reference"
        if top and a.best < WELL_BELOW * top:
            return f"well below the best ({a.best / top:.0%} of it)"
        return ""

    span = f"iteration {min(iterations)}" if len(iterations) == 1 else f"iterations {min(iterations)}-{max(iterations)}"
    lines = [f"Approaches tried so far ({span}, {total} kernels evaluated; code features are read from the code, "
             f"tile, stages and design are what the kernels report; counts and speedups are measured; ncu: the "
             f"profile of the approach's best kernel, % of peak, and ncu's advice with its estimated speedup):", "",
             "| Code features | Tile, stages: tried/passed best | Self-reported design | Tried | Passed | Best "
             "| Status | ncu |",
             "|---|---|---|---|---|---|---|---|"]
    for a in shown:
        best = f"{a.best:.3f}x" if a.best is not None else "-"
        lines.append(f"| {a.label} | {a.variants_cell(max_variants)} | {a.reported or '-'} | {a.tried} | {a.passed} | "
                     f"{best} | {status(a)} | {ncu_summary_cell(a.best_profile)} |")
    if rest:
        bests = [a.best for a in rest if a.best is not None]
        lines.append(f"| {len(rest)} other approaches | - | - | {sum(a.tried for a in rest)} | "
                     f"{sum(a.passed for a in rest)} | {f'{max(bests):.3f}x' if bests else '-'} | | |")
    ranked = sorted(failures.values(), key=lambda f: (-f.count, f.status, f.key))
    if ranked:
        lines += ["", "Recurring failures (count, kind, example, approaches):"]
        listed = ranked[:max_failures] if max_failures else ranked
        for f in listed:
            hint = next((h for pattern, h in HINTS if re.search(pattern, f.key + " " + f.example)), "")
            where = "; ".join(name for name, _ in f.approaches.most_common(2))
            example = f" — e.g. {f.example}" if f.example and f.example != f.key else ""
            lines.append(f"- {f.count}x {f.status.replace('_', ' ')}: {f.key}{example}"
                         f"{f' ({hint})' if hint else ''} [{where}]")
        if len(ranked) > len(listed):
            hidden = ranked[len(listed):]
            lines.append(f"- {len(hidden)} more kinds of failure ({sum(f.count for f in hidden)} kernels)")
    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars - 40)
        text = text[:cut if cut > 0 else max_chars - 40] + "\n- ... (table cut to fit the prompt)"
    return text
