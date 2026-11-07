"""Events a streaming chat backend emits, independent of the provider's wire format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class ContentDelta:
    text: str


@dataclass(frozen=True)
class UsageUpdate:
    """Cumulative usage reported by the server (per chunk with continuous usage stats)."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None  # prompt tokens served from the provider's prefix cache


@dataclass(frozen=True)
class StreamEnd:
    finish_reason: Optional[str] = None


GenEvent = Union[ReasoningDelta, ContentDelta, UsageUpdate, StreamEnd]
