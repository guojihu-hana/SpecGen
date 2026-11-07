"""TriggerMonitor: decide when the main reasoning stream should spawn speculative generations.

A keyword match arms a trigger; the reasoning prefix is cut once ``after_tokens``
more reasoning tokens have arrived.  While a trigger is armed further matches are
ignored, consecutive triggers are at least ``min_gap_tokens`` apart, an iteration
has at most ``max_triggers`` of them, and ``fallback_tokens`` without any trigger
fires one anyway.  The monitor stops when the model starts its final answer.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

FALLBACK = "fallback"


class TriggerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TriggerGroup:
    name: str
    patterns: tuple[re.Pattern, ...]
    after_tokens: Optional[int] = None


@dataclass(frozen=True)
class TriggerConfig:
    groups: tuple[TriggerGroup, ...] = ()
    after_tokens: int = 500
    fallback_tokens: int = 2000
    min_gap_tokens: int = 500
    max_triggers: int = 20

    def with_overrides(self, *, after_tokens: Optional[int] = None,
                       fallback_tokens: Optional[int] = None) -> "TriggerConfig":
        groups = self.groups
        if after_tokens is not None:  # the command line value replaces every group's own value
            groups = tuple(TriggerGroup(g.name, g.patterns, None) for g in groups)
        return TriggerConfig(
            groups=groups,
            after_tokens=self.after_tokens if after_tokens is None else after_tokens,
            fallback_tokens=self.fallback_tokens if fallback_tokens is None else fallback_tokens,
            min_gap_tokens=self.min_gap_tokens,
            max_triggers=self.max_triggers,
        )


def default_triggers_file() -> Path:
    """``$SPECGEN_TRIGGERS_FILE``, else ``./configs/triggers.yaml``, else the repository copy."""
    env = os.environ.get("SPECGEN_TRIGGERS_FILE")
    if env:
        return Path(env)
    local = Path.cwd() / "configs" / "triggers.yaml"
    if local.is_file():
        return local
    return Path(__file__).resolve().parents[3] / "configs" / "triggers.yaml"


def _int(raw: Mapping[str, Any], key: str, default: int, minimum: int = 0) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise TriggerConfigError(f"{key} must be an integer >= {minimum}, got {value!r}")
    return value


def parse_trigger_config(raw: Mapping[str, Any]) -> TriggerConfig:
    if not isinstance(raw, Mapping):
        raise TriggerConfigError("trigger config must be a mapping")
    flags = 0 if raw.get("case_sensitive", False) else re.IGNORECASE
    groups = []
    for i, g in enumerate(raw.get("groups") or []):
        if not isinstance(g, Mapping) or not g.get("name") or not g.get("patterns"):
            raise TriggerConfigError(f"group #{i} needs a name and a list of patterns")
        try:
            patterns = tuple(re.compile(str(p), flags) for p in g["patterns"])
        except re.error as e:
            raise TriggerConfigError(f"group {g['name']!r}: bad pattern: {e}") from None
        after = g.get("after_tokens")
        if after is not None and (not isinstance(after, int) or after < 0):
            raise TriggerConfigError(f"group {g['name']!r}: after_tokens must be an integer >= 0")
        groups.append(TriggerGroup(str(g["name"]), patterns, after))
    return TriggerConfig(
        groups=tuple(groups),
        after_tokens=_int(raw, "default_after_tokens", 500),
        fallback_tokens=_int(raw, "fallback_tokens", 2000, 1),
        min_gap_tokens=_int(raw, "min_gap_tokens", 500),
        max_triggers=_int(raw, "max_triggers_per_iteration", 20),
    )


def load_trigger_config(path: Optional[Path] = None) -> TriggerConfig:
    path = Path(path) if path is not None else default_triggers_file()
    if not path.is_file():
        raise TriggerConfigError(f"trigger config not found: {path}")
    return parse_trigger_config(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


@dataclass(frozen=True)
class Trigger:
    index: int  # 1-based within the iteration
    reason: str  # group name, or "fallback"
    keyword: str  # matched text ("" for the fallback)
    matched_tokens: int  # reasoning tokens when the keyword appeared
    fired_tokens: int  # reasoning tokens in the prefix
    prefix_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "reason": self.reason, "keyword": self.keyword,
                "matched_tokens": self.matched_tokens, "fired_tokens": self.fired_tokens,
                "prefix_chars": self.prefix_chars}


@dataclass
class _Armed:
    reason: str
    keyword: str
    matched_tokens: int
    fire_at: int


class TriggerMonitor:
    WINDOW = 256  # characters kept from earlier deltas so keywords split across deltas still match

    def __init__(self, config: TriggerConfig):
        self.config = config
        self.fired = 0  # triggers so far (their indexes)
        self.counted = 0  # triggers counted against max_triggers
        self.stopped = False
        self._chars = 0
        self._tail = ""
        self._armed: Optional[_Armed] = None
        self._last_fire_tokens = 0

    def stop(self) -> None:
        self.stopped = True
        self._armed = None

    def refund(self) -> None:
        """A trigger that launched nothing (no capacity) does not count against the limit."""
        self.counted = max(0, self.counted - 1)

    def feed(self, text: str, total_tokens: int) -> list[Trigger]:
        """Consume one reasoning delta; ``total_tokens`` counts all reasoning so far."""
        if self.stopped or not text:
            return []
        window = self._tail + text
        new_from = len(self._tail)  # matches must end inside the new text
        self._chars += len(text)
        self._tail = window[-self.WINDOW:]
        cfg = self.config
        if self.counted >= cfg.max_triggers:
            return []
        if self._armed is None:
            match = self._first_match(window, new_from)
            if match is not None:
                group, keyword = match
                after = cfg.after_tokens if group.after_tokens is None else group.after_tokens
                fire_at = total_tokens + after
                if self.fired:
                    fire_at = max(fire_at, self._last_fire_tokens + cfg.min_gap_tokens)
                self._armed = _Armed(group.name, keyword, total_tokens, fire_at)
        if self._armed is not None:
            if total_tokens >= self._armed.fire_at:
                armed, self._armed = self._armed, None
                return [self._fire(armed.reason, armed.keyword, armed.matched_tokens, total_tokens)]
        elif total_tokens - self._last_fire_tokens >= cfg.fallback_tokens:
            return [self._fire(FALLBACK, "", total_tokens, total_tokens)]
        return []

    def _first_match(self, window: str, new_from: int) -> Optional[tuple[TriggerGroup, str]]:
        best: Optional[tuple[int, TriggerGroup, str]] = None
        for group in self.config.groups:
            for pattern in group.patterns:
                for m in pattern.finditer(window):
                    if m.end() <= new_from or not m.group(0):
                        continue
                    if best is None or m.start() < best[0]:
                        best = (m.start(), group, m.group(0))
                    break
        return None if best is None else (best[1], best[2])

    def _fire(self, reason: str, keyword: str, matched: int, tokens: int) -> Trigger:
        self.fired += 1
        self.counted += 1
        self._last_fire_tokens = tokens
        return Trigger(self.fired, reason, keyword, matched, tokens, self._chars)
