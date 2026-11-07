"""What a run found and cost under other termination criteria, or had it stopped at a given moment.

The run's own timeline is cut (the criterion the run used keeps its real iteration ends).  Another criterion is
met no later on that timeline than the run's actual end, so each iteration ends at the first timed kernel that
meets it (plus its confirmation timings) or keeps its real end;
later iterations are assumed to unfold as they did, while the history a criterion compares against follows the
cut timeline.  Billed tokens up to a cut: every request started before it pays its whole prompt; output counts
as far as it got (finished requests: the server's counts; the main stream: its progress events times the run's
server/client token ratio; specs cut short: elapsed time times the run's spec output rate).  A prompt a request
never reported (a main stream cut short) is a spec prompt of its iteration less that spec's reasoning prefix,
with the run's cache-hit rate.  Input is priced two ways: cache hits free, or at 1/50 of a miss (the
provider's list price).  Profiler utilization: the time a timing or ncu job of the task was running (overlaps
once) over the time kept (each iteration from its start to the cut, plus the gaps between iterations).
"""

from __future__ import annotations

import bisect
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from specgen.analysis.data import RunData

CRITERIA = ("best", "hist_avg", "first_valid")
CACHE_HIT_PRICE = 1 / 50
OUTPUT_PRICE = 4  # an output token costs 4 input tokens that miss the cache
DEFAULT_RATIO = 1.04  # server / client tokens of a main stream, before any finished one
DEFAULT_RATE = 150.0  # spec output tokens per second, before any finished one


@dataclass
class Tokens:
    main_in: float = 0.0
    main_cached: float = 0.0
    main_out: float = 0.0
    spec_in: float = 0.0
    spec_cached: float = 0.0
    spec_out: float = 0.0

    def add(self, other: "Tokens") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def priced_in(self, hit_price: float) -> float:
        cached = self.main_cached + self.spec_cached
        return self.main_in + self.spec_in - cached + hit_price * cached

    def summary(self) -> dict[str, int]:
        return {"in": round(self.main_in + self.spec_in), "in_hit_free": round(self.priced_in(0.0)),
                "in_hit_1_50": round(self.priced_in(CACHE_HIT_PRICE)), "out": round(self.main_out + self.spec_out),
                "main_out": round(self.main_out), "spec_out": round(self.spec_out)}


class Ledger:
    """Token accounting of one run: what the requests of an iteration cost up to a moment."""

    def __init__(self, data: RunData):
        self.gens = {(k["task_id"], k["kernel_id"]): k.get("generation") or {} for k in data.kernels}
        mains = [g for (_, kid), g in self.gens.items() if kid.endswith("-main")]
        specs = [g for (_, kid), g in self.gens.items() if not kid.endswith("-main")]
        ratios = [g["server_completion_tokens"] / (g.get("reasoning_tokens", 0) + g.get("content_tokens", 0))
                  for g in mains if g.get("server_completion_tokens") and g.get("reasoning_tokens", 0) + g.get("content_tokens", 0)]
        self.ratio = statistics.median(ratios) if ratios else DEFAULT_RATIO
        rates = [g["server_completion_tokens"] / max(0.5, g["seconds"] - (g.get("ttft_s") or 1.0))
                 for g in specs if g.get("server_completion_tokens") and g.get("seconds")]
        self.rate = statistics.median(rates) if rates else DEFAULT_RATE
        self.hit_main, self.hit_spec = self._hit(mains), self._hit(specs)
        self.known_main_in = [g["server_prompt_tokens"] for g in mains if g.get("server_prompt_tokens")]
        self.mains = {(m["task"], m["iteration"]): m for m in data.mains}
        self.specs: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for s in data.specs:
            self.specs[(s["task"], s["iteration"])].append(s)
        self.fired = {(t["task"], t["iteration"], t["index"]): t.get("fired_tokens") for t in data.triggers}

    @staticmethod
    def _hit(gens: list[dict[str, Any]]) -> float:
        prompt = sum(g.get("server_prompt_tokens") or 0 for g in gens)
        return sum(g.get("server_cached_prompt_tokens") or 0 for g in gens) / prompt if prompt else 0.0

    def _main_in(self, task: str, it: int) -> tuple[float, float]:
        g = self.gens.get((task, f"i{it:03d}-main"), {})
        if g.get("server_prompt_tokens"):
            return g["server_prompt_tokens"], g.get("server_cached_prompt_tokens") or 0
        guesses = []
        for s in self.specs[(task, it)]:
            prompt, fired = self.gens.get((task, s["kernel"]), {}).get("server_prompt_tokens"), self.fired.get(
                (task, it, s.get("trigger")))
            if prompt and fired is not None:
                guesses.append(prompt - fired * self.ratio)
        prompt = max(0.0, min(guesses)) if guesses else (statistics.median(self.known_main_in) if self.known_main_in else 0.0)
        return prompt, prompt * self.hit_main

    def _main_out(self, task: str, it: int, t: float) -> float:
        m, g = self.mains.get((task, it)), self.gens.get((task, f"i{it:03d}-main"), {})
        if m is None or t <= m["start"]:
            return 0.0
        if not m.get("aborted") and m["end"] <= t and g.get("server_completion_tokens"):
            return g["server_completion_tokens"]
        times, tokens = [p[0] for p in m["progress"]], [p[1] for p in m["progress"]]
        stop = min(t, m["end"])
        j = bisect.bisect_right(times, stop)
        if j < len(times):
            t0, t1, k0, k1 = times[j - 1], times[j], tokens[j - 1], tokens[j]
            return (k0 + (k1 - k0) * (stop - t0) / max(1e-6, t1 - t0)) * self.ratio
        speed = tokens[-1] / (times[-1] - times[0]) if times[-1] > times[0] else 0.0  # still streaming after the last event
        return (tokens[-1] + speed * (stop - times[-1])) * self.ratio

    def tokens(self, task: str, it: int, end: float) -> Tokens:
        """What the requests of one iteration that started before ``end`` cost up to ``end``."""
        tok = Tokens()
        if (task, it) in self.mains and self.mains[(task, it)]["start"] < end:
            tok.main_in, tok.main_cached = self._main_in(task, it)
            tok.main_out = self._main_out(task, it, end)
        specs = self.specs[(task, it)]
        reported = [self.gens.get((task, s["kernel"]), {}).get("server_prompt_tokens") for s in specs]
        typical = statistics.median([p for p in reported if p] or [0])
        for s in specs:
            if s["start"] >= end:
                continue
            g = self.gens.get((task, s["kernel"]), {})
            prompt = g.get("server_prompt_tokens") or typical
            tok.spec_in += prompt
            tok.spec_cached += (g.get("server_cached_prompt_tokens") or 0) if g.get("server_prompt_tokens") else prompt * self.hit_spec
            if not s.get("cancelled") and s["end"] <= end and g.get("server_completion_tokens"):
                tok.spec_out += g["server_completion_tokens"]
            else:
                tok.spec_out += max(0.0, min(end, s["end"]) - s["start"] - 1.0) * self.rate
        return tok


def _confirm_s(data: RunData) -> float:
    """Extra timings a criterion with a baseline waits for before it ends an iteration."""
    confirm = int((data.config.get("termination") or {}).get("confirm") or 1)
    runs = [j["end"] - j["start"] for j in data.jobs if j["kind"] == "benchmark" and j.get("outcome") == "done"]
    return (confirm - 1) * (statistics.median(runs) if runs else 0.0)


PROFILER_KINDS = ("benchmark", "ncu")
VALIDATOR_KINDS = ("validate",)
# budgets come from rounded results: a run that used exactly its budget must not count as cut
BUDGET_SLACK = {"seconds": 0.06, "tokens": 5.0}


def busy_intervals(data: RunData, task: str, kinds: tuple[str, ...]) -> list[list[float]]:
    """The merged intervals during which a job of ``task`` of one of ``kinds`` was running."""
    merged: list[list[float]] = []
    for start, end in sorted((j["start"], j["end"]) for j in data.jobs
                             if j.get("task") == task and j.get("kind") in kinds):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def profiler_busy(data: RunData, task: str) -> list[list[float]]:
    """The merged intervals during which a timing or ncu job of ``task`` was running."""
    return busy_intervals(data, task, PROFILER_KINDS)


def utilization(busy: list[list[float]], windows: list[tuple[float, float]]) -> dict[str, Optional[float]]:
    """Busy time within ``windows`` and its share of their length."""
    seconds = sum(max(0.0, end - start) for start, end in windows)
    busy_s = sum(max(0.0, min(end, b_end) - max(start, b_start)) for start, end in windows for b_start, b_end in busy)
    return {"busy_s": round(busy_s, 1), "seconds": round(seconds, 1),
            "share": round(busy_s / seconds, 4) if seconds > 0 else None}


def is_baseline(data: RunData) -> bool:
    """A run without spec generation: one kernel per iteration, no early termination."""
    spec = data.config.get("spec") or {}
    return not spec["enabled"] if "enabled" in spec else not data.specs


def equivalent(tokens: dict[str, int]) -> float:
    """Billed tokens in units of an input token that misses the cache: hits at 1/50, output at 4 times."""
    return tokens["in_hit_1_50"] + OUTPUT_PRICE * tokens["out"]


def _budget_cut(budget: tuple[str, float], ledger: Ledger, task: str, span: dict[str, Any], end: float,
                seconds: float, spent: Tokens) -> Optional[float]:
    """When, within an iteration kept from its start to ``end``, a run that already used ``seconds`` and ``spent``
    runs out of ``budget``; None if the budget lasts the iteration."""
    kind, limit = budget
    if kind == "seconds":
        left = limit - seconds
        return span["start"] + max(0.0, left) if end - span["start"] > left + BUDGET_SLACK["seconds"] else None
    used = equivalent(spent.summary())
    within = lambda t: used + equivalent(ledger.tokens(task, span["iteration"], t).summary())  # noqa: E731
    if within(end) <= limit + BUDGET_SLACK["tokens"]:
        return None
    # tokens only grow with time (a request's prompt as it starts): the last moment within the budget
    lo, hi = span["start"], end
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if within(mid) > limit else (mid, hi)
    return lo


def simulate(data: RunData, criterion: Optional[str], *, ledger: Optional[Ledger] = None,
             budget: Optional[tuple[str, float]] = None) -> dict[str, dict[str, Any]]:
    """Per task: the best speedup after each iteration, the final best, the wall time and the tokens had every
    iteration ended under ``criterion`` (``best``, ``hist_avg`` or ``first_valid``; None: as the run went).
    ``budget`` (``("seconds", s)`` or ``("tokens", n)``, n in ``equivalent`` tokens) also stops the run once it has
    used that much on the cut timeline; ``iterations`` counts the iterations it finished, ``cut`` whether the budget
    stopped it."""
    if criterion is not None and criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}")
    ledger = ledger or Ledger(data)
    margin = float((data.config.get("termination") or {}).get("margin") or 0.03)
    confirm_s = _confirm_s(data)
    # the run's own criterion, or none at all: its real iteration ends
    actual = criterion is None or criterion == (data.config.get("termination") or {}).get("criteria")
    out = {}
    for task in data.tasks:
        spans = sorted((s for s in data.spans if s["task"] == task), key=lambda s: s["iteration"])
        tok, seconds, chosen, best, curve, kept, finished, cut = Tokens(), 0.0, [], None, [], [], 0, False
        for i, span in enumerate(spans):
            benches = sorted((b for b in data.benchmarks if b["task"] == task and b["iteration"] == span["iteration"]
                              and b.get("speedup")), key=lambda b: b["t"])
            if criterion is None:
                met = None
            elif criterion == "first_valid":
                met = benches[0] if benches else None
            else:
                bar = (max(best or 1.0, 1.0) if criterion == "best" else
                       statistics.fmean(chosen) if chosen else 1.0) * (1 + margin)
                met = next((b for b in benches if b["speedup"] > bar), None)
            if actual or met is None:
                end = span["end"]
            else:
                end = min(span["end"], met["t"] + (0.0 if criterion == "first_valid" else confirm_s))
            stop = _budget_cut(budget, ledger, task, span, end, seconds, tok) if budget else None
            if stop is not None:
                end, cut = stop, True
            if met is not None and met["t"] > end:  # the budget ran out first
                met = None
            found = [b["speedup"] for b in benches if b["t"] <= end]
            if found:
                chosen.append(met["speedup"] if met is not None else max(found))
                best = max(best or 0.0, max(found))
            curve.append(None if best is None else round(best, 4))
            tok.add(ledger.tokens(task, span["iteration"], end))
            seconds += end - span["start"]
            kept.append((span["start"], end))
            if cut:
                break
            finished += 1
            gap = spans[i + 1]["start"] - span["end"] if i + 1 < len(spans) else 0.0
            if gap > 0 and budget and budget[0] == "seconds" and seconds + gap > budget[1] + BUDGET_SLACK["seconds"]:
                kept.append((span["end"], span["end"] + budget[1] - seconds))
                seconds, cut = budget[1], True
                break
            if gap > 0:
                seconds += gap
                kept.append((span["end"], spans[i + 1]["start"]))
        out[task] = {"criterion": criterion, "best": None if best is None else round(best, 4), "per_iteration": curve,
                     "seconds": round(seconds, 1), "tokens": tok.summary(),
                     "profiler": utilization(profiler_busy(data, task), kept),
                     "validator": utilization(busy_intervals(data, task, VALIDATOR_KINDS), kept),
                     "iterations": finished, "cut": cut}
    return out


def stopped_at(data: RunData, task: str, t: float, *, ledger: Optional[Ledger] = None) -> dict[str, Any]:
    """The tokens spent, the time used and the profiler utilization of ``task`` had the run stopped at absolute
    time ``t``."""
    ledger = ledger or Ledger(data)
    tok = Tokens()
    for span in data.spans:
        if span["task"] == task and span["start"] < t:
            tok.add(ledger.tokens(task, span["iteration"], min(t, span["end"])))
    return {"seconds": round(t - data.t0, 1), "tokens": tok.summary(),
            "profiler": utilization(profiler_busy(data, task), [(data.t0, t)]),
            "validator": utilization(busy_intervals(data, task, VALIDATOR_KINDS), [(data.t0, t)])}


def criteria_payload(data: RunData) -> dict[str, dict[str, Any]]:
    """task -> criterion -> simulate() result (a baseline run: only ``baseline``, the run as it went), plus the
    whole run's tokens under the same accounting."""
    ledger = Ledger(data)
    names = ("baseline",) if is_baseline(data) else CRITERIA
    runs = {name: simulate(data, None if name == "baseline" else name, ledger=ledger) for name in names}
    out: dict[str, dict[str, Any]] = {}
    for task in data.tasks:
        out[task] = {name: runs[name][task] for name in names}
        out[task]["recorded"] = stopped_at(data, task, data.t_end, ledger=ledger)
    return out
