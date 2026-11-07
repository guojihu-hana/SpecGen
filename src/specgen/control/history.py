"""Per-task history of evaluated kernels, and the feedback it yields for the next prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from specgen.control.attempts import family, summarize
from specgen.generation.prompts import KernelFeedback, PromptContext


@dataclass
class KernelOutcome:
    kernel_id: str
    iteration: int
    source: str  # "main" or "spec"
    code: str
    status: str = "pending"
    valid: bool = False
    speedup: Optional[float] = None
    latency_ms: Optional[float] = None
    ref_latency_ms: Optional[float] = None
    ncu_profile: Optional[dict[str, Any]] = None  # evaluation.ncu: the kernels of one forward
    error: Optional[str] = None

    @property
    def benchmarked(self) -> bool:
        return self.valid and self.speedup is not None

    def feedback(self) -> KernelFeedback:
        return KernelFeedback(
            source=self.code,
            status=self.status,
            speedup=self.speedup,
            latency_ms=self.latency_ms,
            ref_latency_ms=self.ref_latency_ms,
            ncu_profile=self.ncu_profile,
            error=self.error,
        )


class TaskHistory:
    def __init__(self) -> None:
        self.outcomes: list[KernelOutcome] = []
        self.selected: dict[int, KernelOutcome] = {}  # iteration -> its selected (benchmarked) kernel

    def add(self, outcome: KernelOutcome) -> None:
        self.outcomes.append(outcome)

    def select(self, iteration: int, outcome: KernelOutcome) -> None:
        self.selected[iteration] = outcome

    def winners_mean_speedup(self) -> Optional[float]:
        """Mean speedup of the kernels earlier iterations selected (None before any)."""
        speedups = [o.speedup for o in self.selected.values() if o.speedup is not None]
        return sum(speedups) / len(speedups) if speedups else None

    def iteration(self, index: int) -> list[KernelOutcome]:
        return [o for o in self.outcomes if o.iteration == index]

    def best(self) -> Optional[KernelOutcome]:
        """Fastest benchmarked kernel; ties go to the earlier one."""
        best: Optional[KernelOutcome] = None
        for outcome in self.outcomes:
            if outcome.benchmarked and (best is None or outcome.speedup > best.speedup):
                best = outcome
        return best

    def prompt_context(self, iteration: int, top_k: int, *, mode: str = "topk", max_approaches: int = 8,
                       max_failures: int = 5, max_chars: int = 6000) -> PromptContext:
        """Best kernel so far, plus (``topk``) the fastest kernels of the previous iteration (best
        excluded), or that iteration's main kernel and error if none was valid; (``summary``) a
        table of every approach tried so far and the recurring failures; or (``topk+summary``) both."""
        if iteration == 0:
            return PromptContext()
        best = self.best()
        table = ""
        if mode in ("summary", "topk+summary"):
            table = summarize(self.outcomes, best_kernel=best.kernel_id if best else None,
                              max_approaches=max_approaches, max_failures=max_failures, max_chars=max_chars)
        if mode == "summary":
            return PromptContext(best=best.feedback() if best else None, attempts=table)
        feedback = tuple(o.feedback() for o in self.shown_from(iteration - 1, top_k))
        failure = None
        last = self.iteration(iteration - 1)
        if not feedback and last and not any(o.benchmarked for o in last):
            main = next((o for o in last if o.source == "main"), last[0])
            failure = main.feedback()
        return PromptContext(best=best.feedback() if best else None, feedback=feedback, failure=failure,
                             attempts=table)

    def approach_bests(self) -> dict[str, KernelOutcome]:
        """Approach (code features, ``control.attempts.family``) -> its fastest benchmarked kernel."""
        bests: dict[str, KernelOutcome] = {}
        for outcome in self.outcomes:
            if outcome.benchmarked:
                key = family(outcome.code)
                if key not in bests or outcome.speedup > bests[key].speedup:
                    bests[key] = outcome
        return bests

    def shown_from(self, iteration: int, top_k: int) -> list[KernelOutcome]:
        """The fastest kernels of an iteration that a ``topk`` prompt shows besides the best so far."""
        best = self.best()
        ranked = sorted((o for o in self.iteration(iteration) if o.benchmarked),
                        key=lambda o: (-o.speedup, o.kernel_id))
        return [o for o in ranked if best is None or o.kernel_id != best.kernel_id][:top_k]
