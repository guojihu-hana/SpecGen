"""Speculative generation control: watch the main reasoning stream and cut prefixes."""

from __future__ import annotations

import hashlib
import time
from typing import Callable, Optional

from specgen.control.triggers import Trigger, TriggerMonitor


def code_fingerprint(code: str) -> str:
    """Hash of the code with trailing whitespace and blank lines removed, so two
    candidates that differ only in layout are evaluated once."""
    lines = (line.rstrip() for line in code.strip().splitlines())
    return hashlib.sha256("\n".join(line for line in lines if line).encode("utf-8")).hexdigest()[:16]


class SpecController:
    """Generation sink side of speculative generation for one main stream.

    Every reasoning delta goes to the TriggerMonitor; each trigger hands the
    reasoning so far to ``on_trigger``.  Content (the final answer) stops triggering.
    """

    def __init__(self, monitor: TriggerMonitor, on_trigger: Callable[[Trigger, str], None]):
        self.monitor = monitor
        self._on_trigger = on_trigger
        self._parts: list[str] = []
        self.tokens = 0
        self._t0 = time.monotonic()
        self._first_token_at: Optional[float] = None

    def on_reasoning(self, text: str, total_tokens: int) -> None:
        if self._first_token_at is None:
            self._first_token_at = time.monotonic()
        self._parts.append(text)
        self.tokens = total_tokens
        for trigger in self.monitor.feed(text, total_tokens):
            self._on_trigger(trigger, "".join(self._parts))

    def on_content(self, text: str, total_tokens: int) -> None:
        self.monitor.stop()

    def stop(self) -> None:
        self.monitor.stop()

    def tokens_per_second(self) -> Optional[float]:
        """Reasoning rate since the first reasoning token."""
        if self._first_token_at is None or self.tokens <= 0:
            return None
        elapsed = time.monotonic() - self._first_token_at
        return self.tokens / elapsed if elapsed > 0.5 else None


class PacedSpecController:
    """Demand-paced speculative generation for one main stream (``control.pacing``): every reasoning
    delta asks ``on_tokens(total_tokens, prefix)`` whether to launch a spec now, ``prefix()`` being the
    reasoning so far.  Content (the final answer) stops it."""

    monitor = None

    def __init__(self, on_tokens: Callable[[int, Callable[[], str]], None]):
        self._on_tokens = on_tokens
        self._parts: list[str] = []
        self.tokens = 0
        self.stopped = False
        self._first_token_at: Optional[float] = None

    def prefix(self) -> str:
        if len(self._parts) > 1:
            self._parts = ["".join(self._parts)]
        return self._parts[0] if self._parts else ""

    def on_reasoning(self, text: str, total_tokens: int) -> None:
        if self.stopped:
            return
        if self._first_token_at is None:
            self._first_token_at = time.monotonic()
        self._parts.append(text)
        self.tokens = total_tokens
        self._on_tokens(total_tokens, self.prefix)

    def on_content(self, text: str, total_tokens: int) -> None:
        self.stopped = True

    def stop(self) -> None:
        self.stopped = True

    tokens_per_second = SpecController.tokens_per_second
