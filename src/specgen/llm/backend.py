"""The streaming chat backend interface and its factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Protocol

from specgen.llm.events import GenEvent
from specgen.llm.profiles import ModelProfile


@dataclass
class ChatParams:
    temperature: float
    top_p: float
    max_tokens: int
    seed: Optional[int] = None
    mode: str = "main"  # selects the profile's request parameters (e.g. thinking on for main, off for spec)
    prefill: bool = False  # the last message is an assistant prefix the model continues
    extra_body: dict[str, Any] = field(default_factory=dict)


class ChatBackend(Protocol):
    # How this endpoint continues an assistant prefix ("deepseek", "vllm"), or None if it cannot.
    prefill_style: Optional[str]

    def stream_chat(self, messages: list[dict[str, Any]], params: ChatParams) -> AsyncIterator[GenEvent]:
        """Stream one completion. Closing the iterator early aborts the request."""
        ...

    async def aclose(self) -> None: ...


def make_backend(profile: ModelProfile, endpoint: str, *, timeout_s: float, max_retries: int) -> ChatBackend:
    resolved = profile.endpoint(endpoint).resolve()
    if resolved.backend == "anthropic":
        from specgen.llm.anthropic import AnthropicBackend as impl
    elif resolved.backend == "gemini":
        from specgen.llm.gemini import GeminiBackend as impl
    elif resolved.backend == "file_queue":
        from specgen.llm.file_queue import FileQueueBackend as impl
    else:  # vllm, sglang, api, together: OpenAI-compatible
        from specgen.llm.openai_compat import OpenAICompatBackend as impl
    return impl(resolved, profile, timeout_s=timeout_s, max_retries=max_retries)
