"""Demand-paced speculative generation: when a task launches its next spec, and the test results of
the iteration's earlier specs that the new one sees.

Timing and ncu on the profiler GPUs are the bottleneck; builds and validation keep up.  A task
launches one spec whenever the kernels it expects to reach the profiler fall below a target set by
its share of the profiler devices:

    expected = waiting + p_pass * upstream  <  target = max(share + target_extra, share * target_factor)

``share`` is profiler devices over active tasks; ``waiting`` the task's profiler work still queued, in
timing-job units (an ncu job counts as its run time over a timing job's); ``upstream`` its specs still
generating, building or validating; ``p_pass`` the share of its finished specs that reached timing.
Launches are at least ``min_gap_tokens`` of reasoning apart (the first comes after ``first_tokens``)
and pause while the ncu profile of a kernel that meets the termination criterion is pending, for at
most ``best_ncu_wait_s``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from specgen.control.attempts import failure, failure_kind, family, self_reported
from specgen.generation.ncu_format import ncu_summary_cell

PRIOR_WEIGHT = 2.0  # pseudo-observations behind the prior pass rate
NOTES_HEADER = "(Test results of the drafts written from earlier parts of this reasoning, oldest first:"
MAX_NOTE_LINES = 12
MAX_NOTES_CHARS = 2400
STATUS_WORDS = {
    "compile_error": "does not compile", "runtime_error": "runtime error", "cuda_error": "CUDA error",
    "timeout": "timed out", "benchmark_error": "timing failed", "forbidden": "rejected",
    "internal_error": "evaluation failed", "worker_error": "evaluation failed", "numerical_error": "wrong results",
}


@dataclass
class PassRate:
    """Share of a task's finished specs that passed validation and reached timing."""

    passed: int = 0
    finished: int = 0

    def add(self, passed: bool) -> None:
        self.passed += int(passed)
        self.finished += 1

    def value(self, prior: float) -> float:
        return (self.passed + prior * PRIOR_WEIGHT) / (self.finished + PRIOR_WEIGHT)


def launch_decision(*, tokens: int, last_tokens: Optional[int], first_tokens: int, min_gap_tokens: int,
                    waiting: float, upstream: int, p_pass: float, target: float, paused: bool) -> tuple[bool, str]:
    """(launch now?, why) for one reasoning delta."""
    if paused:
        return False, "waiting for the ncu profile of the new best kernel"
    if last_tokens is None:
        if tokens < first_tokens:
            return False, "first tokens"
    elif tokens - last_tokens < min_gap_tokens:
        return False, "gap"
    if waiting + p_pass * upstream >= target:
        return False, "profiler fed"
    return True, "launch"


def _result(outcome, best: bool) -> str:
    if outcome.benchmarked:
        text = f"correct, {outcome.speedup:.2f}x the reference"
        if outcome.latency_ms:
            text += f" ({outcome.latency_ms:.3f} ms)"
        if best:
            text += ", the best so far"
            if outcome.ncu_profile:
                text += f"; Nsight Compute: {ncu_summary_cell(outcome.ncu_profile)}"
        return text
    key, example = failure(outcome.status, outcome.error)
    if outcome.status == "numerical_error":
        return key + (f", {example}" if example else "")
    if outcome.status == "forbidden":
        return f"rejected: {key}"
    word = STATUS_WORDS.get(failure_kind(outcome.status, outcome.error), outcome.status.replace("_", " "))
    return f"{word}: {example}" if example and example != word else word


def draft_notes(outcomes: Iterable, best_kernel: Optional[str]) -> tuple[str, int]:
    """(text to append to the reasoning prefix, drafts it lists) for the iteration's finished candidates:
    the latest ones, always with the best; empty before any finished."""
    finished = [o for o in outcomes if o.benchmarked or o.status in STATUS_WORDS]
    if not finished:
        return "", 0
    keep = finished[-MAX_NOTE_LINES:]
    if best_kernel is not None and all(o.kernel_id != best_kernel for o in keep):
        keep = [o for o in finished if o.kernel_id == best_kernel] + keep[1:]
    lines = [f"- {self_reported(o.code) or family(o.code)}: {_result(o, o.kernel_id == best_kernel)}" for o in keep]
    while len(lines) > 1 and sum(len(line) + 1 for line in lines) > MAX_NOTES_CHARS:
        drop = next(i for i, o in enumerate(keep) if o.kernel_id != best_kernel)
        del lines[drop], keep[drop]
    return "\n\n" + NOTES_HEADER + "\n" + "\n".join(lines) + ")", len(lines)
