"""SpecBudget: how many speculative generations each trigger launches (shared by all tasks).

Adaptive policy: the evaluation pipeline absorbs at most

    C = min(G_b / t_b, G_v / t_v, G_p / (r_valid * t_p))   kernels per second

(builder, validation and timing capacity: slot counts over mean run times).  A task
gets a fair share of C.  Over the horizon H until its next trigger can fire plus one
spec generation, the task should therefore have about share * H kernels in the
pipeline; W of them are already there (queued or running jobs, plus in-flight specs
times the parse rate).  Each new spec yields a kernel with probability r_parse:

    k = clip(ceil((share * H - W) / r_parse), k_min, min(k_max, remaining in-flight capacity))
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections import defaultdict
from typing import Any, AsyncIterator, Optional

from specgen.generation.generator import GenerationResult
from specgen.telemetry.recorder import Ewma

PRIOR_SPEC_S = 30.0
PRIOR_TOKENS_PER_S = 50.0
PRIOR_VALID_RATE = 0.5
PRIOR_PARSE_RATE = 0.8
PRIOR_WEIGHT = 2.0  # pseudo-observations behind each prior rate


def _rate(done: int, total: int, prior: float) -> float:
    return (done + prior * PRIOR_WEIGHT) / (total + PRIOR_WEIGHT)


class SpecBudget:
    def __init__(self, *, k: Optional[int] = None, k_min: int = 1, k_max: int = 4, max_inflight: int = 0):
        self.k = k  # None: adaptive
        self.k_min, self.k_max = k_min, k_max
        self.max_inflight = max_inflight  # 0: unlimited
        self._semaphore = asyncio.Semaphore(max_inflight) if max_inflight > 0 else None
        self.inflight = 0  # spec generations decided whose LLM request has not finished (waiting or running)
        self.task_inflight: dict[str, int] = defaultdict(int)
        self.active_tasks = 1
        self.generation_s = Ewma()
        self.parsed = [0, 0]  # [with code, total]

    def remaining(self) -> Optional[int]:
        return None if self.max_inflight <= 0 else max(0, self.max_inflight - self.inflight)

    def decide(self, task_id: str, *, pool: Any = None, recorder: Any = None, triggers: Any = None,
               tokens_per_s: Optional[float] = None, cap: bool = True, **_: Any) -> tuple[int, dict[str, Any]]:
        """k for one trigger of ``task_id``; the caller must ``release(task_id)`` each of the k once its
        generation ends.  With ``cap`` k never exceeds the free generation slots; without it the
        excess waits for a slot (``slot()``)."""
        if self.k is not None or pool is None:
            k = max(self.k_min, min(self.k_max, 2 if self.k is None else self.k))
            info: dict[str, Any] = {"policy": "fixed"}
        else:
            k, info = self._adaptive(task_id, pool, recorder, triggers, tokens_per_s)
        remaining = self.remaining()
        if cap and remaining is not None and remaining < k:
            k = remaining
            info["capped_by_inflight"] = True
        info["inflight"] = self.inflight
        self.inflight += k
        self.task_inflight[task_id] += k
        return k, info

    def _adaptive(self, task_id: str, pool: Any, recorder: Any, triggers: Any,
                  tokens_per_s: Optional[float]) -> tuple[int, dict[str, Any]]:
        stats = pool.stats()
        fifo = getattr(pool, "allocation", "fixed") == "fifo"
        validate_slots = stats.slots.get("shared" if fifo else "validator", 0)
        profile_slots = (sum(1 for d in pool.devices if d.role == "shared") if fifo
                         else stats.slots.get("profiler", 0))
        build_slots = stats.slots.get("builder", 0)
        t_b, t_v, t_p = (pool.expected_run_s(kind) for kind in ("build", "validate", "benchmark"))
        valid = recorder.outcomes["validate"] if recorder is not None else [0, 0]
        r_valid = _rate(valid[0], valid[1], PRIOR_VALID_RATE)
        r_parse = self.parse_rate()
        capacity = [validate_slots / t_v, profile_slots / (r_valid * t_p)]
        if build_slots:
            capacity.append(build_slots / t_b)
        c = min(capacity)
        share = c / max(1, self.active_tasks)
        tps = tokens_per_s or PRIOR_TOKENS_PER_S
        gap = max(getattr(triggers, "min_gap_tokens", 500), getattr(triggers, "after_tokens", 500))
        spec_s = self.generation_s.value or PRIOR_SPEC_S
        horizon = gap / tps + spec_s
        in_pipeline = pool.load(task_id) + self.task_inflight[task_id] * r_parse
        raw = (share * horizon - in_pipeline) / r_parse
        k = max(self.k_min, min(self.k_max, math.ceil(raw)))
        info = {
            "policy": "adaptive", "capacity_per_s": round(c, 4), "share_per_s": round(share, 4),
            "horizon_s": round(horizon, 1), "in_pipeline": round(in_pipeline, 2), "raw_k": round(raw, 2),
            "t_build_s": round(t_b, 1), "t_validate_s": round(t_v, 1), "t_benchmark_s": round(t_p, 1),
            "valid_rate": round(r_valid, 3), "parse_rate": round(r_parse, 3), "tokens_per_s": round(tps, 1),
            "active_tasks": self.active_tasks,
        }
        return k, info

    def reserve(self, task_id: str) -> bool:
        """One generation of ``task_id`` (demand-paced specs), unless every generation slot is taken;
        the caller must ``release(task_id)`` once its generation ends."""
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            return False
        self.inflight += 1
        self.task_inflight[task_id] += 1
        return True

    def release(self, task_id: str) -> None:
        self.inflight -= 1
        self.task_inflight[task_id] -= 1

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hard cap on concurrent spec requests (--max-inflight-spec-gen)."""
        if self._semaphore is None:
            yield
        else:
            async with self._semaphore:
                yield

    def observe(self, result: GenerationResult) -> None:
        if result.ok:
            self.generation_s.add(result.seconds)
        self.parsed[0] += 1 if result.code is not None else 0
        self.parsed[1] += 1

    def parse_rate(self) -> float:
        return _rate(self.parsed[0], self.parsed[1], PRIOR_PARSE_RATE)

    def stats(self) -> dict[str, Any]:
        return {"inflight": self.inflight, "generation_s": self.generation_s.value,
                "parse_rate": round(self.parse_rate(), 4), "spec_generations": self.parsed[1]}
