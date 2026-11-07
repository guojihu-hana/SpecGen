"""Token counting for streamed text.

``generic`` uses tiktoken's ``o200k_base`` (an approximation of any model's
count) and falls back to ``len(text) / 4`` when tiktoken is unavailable; any
other value is loaded as a local Hugging Face tokenizer.
"""

from __future__ import annotations

import functools
import logging
from typing import Callable

log = logging.getLogger(__name__)

GENERIC = "generic"


def _approx(text: str) -> int:
    return (len(text) + 3) // 4


@functools.lru_cache(maxsize=8)
def _load(spec: str) -> tuple[str, Callable[[str], int]]:
    if spec in ("", GENERIC):
        try:
            import tiktoken

            enc = tiktoken.get_encoding("o200k_base")
        except Exception as e:  # noqa: BLE001 - missing package or offline download
            log.warning("generic tokenizer unavailable (%s); counting tokens as len/4", e)
            return "chars/4", _approx
        return "tiktoken:o200k_base", lambda text: len(enc.encode(text, disallowed_special=()))
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(spec, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"cannot load tokenizer {spec!r}: {e}") from e
    return f"hf:{spec}", lambda text: len(tok.encode(text, add_special_tokens=False))


class TokenCounter:
    def __init__(self, tokenizer: str = GENERIC):
        self.name, self._count = _load(tokenizer or GENERIC)

    def count(self, text: str) -> int:
        return self._count(text) if text else 0

    def stream(self) -> "StreamCounter":
        return StreamCounter(self)


def _word_boundary(text: str) -> int:
    """Index of the last space that directly follows a non-space character, or -1.

    BPE pre-tokenizers attach such a space to the next word, so splitting there
    leaves tokenization unchanged; inside whitespace runs (indentation) or at
    newlines (which can join a preceding punctuation token) it would not.
    """
    for i in range(len(text) - 1, 0, -1):
        if text[i] == " " and not text[i - 1].isspace():
            return i
    return -1


class StreamCounter:
    """Running count of text that arrives in pieces.

    Text is committed up to the last word boundary; only the short uncommitted
    tail is re-counted on each read.
    """

    MAX_PENDING = 2048

    def __init__(self, counter: TokenCounter):
        self._counter = counter
        self._committed = 0
        self._pending = ""

    def add(self, text: str) -> int:
        self._pending += text
        cut = _word_boundary(self._pending)
        if cut <= 0 and len(self._pending) > self.MAX_PENDING:
            cut = len(self._pending)
        if cut > 0:
            self._committed += self._counter.count(self._pending[:cut])
            self._pending = self._pending[cut:]
        return self.total

    @property
    def total(self) -> int:
        return self._committed + self._counter.count(self._pending)
