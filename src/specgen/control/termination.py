"""Early termination: a benchmarked kernel that meets the criterion ends its iteration.

- ``first_valid``: the first kernel of the iteration that passes validation and timing;
- ``best``: a speedup above the best of all earlier iterations, and never below the reference
  (a kernel slower than PyTorch does not end an iteration);
- ``hist_avg``: a speedup above the mean of the kernels earlier iterations selected.

The first iteration compares against the reference (1.0x); ``margin`` is the fraction
a kernel must exceed the baseline by.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from specgen.control.history import TaskHistory

CRITERIA = ("first_valid", "best", "hist_avg")


@dataclass(frozen=True)
class TerminationCriterion:
    name: str
    margin: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in CRITERIA:
            raise ValueError(f"unknown termination criterion {self.name!r} (known: {', '.join(CRITERIA)})")

    def baseline(self, history: TaskHistory) -> Optional[float]:
        """Speedup to beat; None when any benchmarked kernel qualifies."""
        if self.name == "first_valid":
            return None
        if self.name == "best":
            best = history.best()
            return max(best.speedup, 1.0) if best is not None else 1.0
        mean = history.winners_mean_speedup()
        return 1.0 if mean is None else mean

    def satisfied(self, speedup: Optional[float], history: TaskHistory) -> tuple[bool, Optional[float]]:
        """``(met, baseline)`` for a kernel of the current iteration (``history`` holds earlier ones)."""
        if speedup is None:
            return False, None
        baseline = self.baseline(history)
        if baseline is None:
            return True, None
        return speedup > baseline * (1.0 + self.margin), baseline
