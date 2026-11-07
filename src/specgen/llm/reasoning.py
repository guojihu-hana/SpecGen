"""Split reasoning that a model inlines in its content between think tags."""

from __future__ import annotations

REASONING = "reasoning"
CONTENT = "content"


class InlineThinkSplitter:
    """Streaming splitter for ``<think>...</think>`` inside content.

    Tags may arrive split across chunks; a possible tag prefix at the end of a
    chunk is held back until the next chunk decides it.  Only the first reasoning
    block is recognized: once it closes, everything else is content.  Set
    ``starts_in_reasoning`` for models whose chat template opens the think tag in
    the prompt, so the output starts directly with reasoning.
    """

    def __init__(self, open_tag: str = "<think>", close_tag: str = "</think>", *, starts_in_reasoning: bool = False):
        self._open = open_tag
        self._close = close_tag
        self._in_reasoning = starts_in_reasoning
        self._closed = False
        self._buf = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        self._buf += text
        out: list[tuple[str, str]] = []
        while self._buf:
            if self._closed:
                out.append((CONTENT, self._buf))
                self._buf = ""
                break
            tag = self._close if self._in_reasoning else self._open
            kind = REASONING if self._in_reasoning else CONTENT
            index = self._buf.find(tag)
            if index >= 0:
                if index:
                    out.append((kind, self._buf[:index]))
                self._buf = self._buf[index + len(tag):]
                if self._in_reasoning:
                    self._closed = True
                self._in_reasoning = not self._in_reasoning
                continue
            hold = _partial_suffix(self._buf, tag)
            emit = self._buf[: len(self._buf) - hold]
            if emit:
                out.append((kind, emit))
            self._buf = self._buf[len(self._buf) - hold:]
            break
        return [(k, t) for k, t in out if t]

    def flush(self) -> list[tuple[str, str]]:
        if not self._buf:
            return []
        kind = REASONING if self._in_reasoning and not self._closed else CONTENT
        out, self._buf = [(kind, self._buf)], ""
        return out


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0
