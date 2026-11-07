"""OpenAI-compatible streaming backend for vLLM, SGLang and OpenAI-style APIs."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from specgen.llm.backend import ChatParams
from specgen.llm.events import ContentDelta, GenEvent, ReasoningDelta, StreamEnd, UsageUpdate
from specgen.llm.profiles import ModelProfile, ResolvedEndpoint
from specgen.llm.reasoning import REASONING, InlineThinkSplitter

_REASONING_FIELDS = ("reasoning_content", "reasoning")
_MESSAGE_KEYS = ("role", "content", "name", "prefix")


class OpenAICompatBackend:
    def __init__(
        self,
        endpoint: ResolvedEndpoint,
        profile: ModelProfile,
        *,
        timeout_s: float = 3600.0,
        max_retries: int = 2,
        client: Any = None,
        prefill_client: Any = None,
    ):
        self._timeout_s, self._max_retries = timeout_s, max_retries
        self._client = client if client is not None else self._make_client(endpoint.base_url, endpoint.api_key)
        self._prefill_client = prefill_client
        self._endpoint = endpoint
        self._profile = profile
        self.prefill_style = endpoint.prefill_style

    def _make_client(self, base_url: str, api_key: str) -> Any:
        from openai import AsyncOpenAI

        return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=self._timeout_s, max_retries=self._max_retries)

    def _client_for_prefill(self) -> Any:
        ep = self._endpoint
        if not ep.prefill_base_url or ep.prefill_base_url == ep.base_url:
            return self._client
        if self._prefill_client is None:
            self._prefill_client = self._make_client(ep.prefill_base_url, ep.api_key)
        return self._prefill_client

    async def stream_chat(self, messages: list[dict[str, Any]], params: ChatParams) -> AsyncIterator[GenEvent]:
        ep, profile = self._endpoint, self._profile
        messages = [{k: v for k, v in m.items() if k in _MESSAGE_KEYS} for m in messages]
        extra_body = {**ep.extra_body, **profile.mode_params(params.mode, ep), **params.extra_body}
        client = self._client
        if params.prefill:
            if ep.prefill_style == "deepseek":
                messages[-1]["prefix"] = True
                client = self._client_for_prefill()
            elif ep.prefill_style == "vllm":
                extra_body.update(continue_final_message=True, add_generation_prompt=False)
            else:
                raise ValueError(f"endpoint {ep.name!r} cannot continue an assistant prefix (no prefill style)")
        stream_options: dict[str, Any] = {"include_usage": True}
        if ep.backend == "vllm":
            stream_options["continuous_usage_stats"] = True
        kwargs: dict[str, Any] = {
            "model": ep.model,
            "messages": messages,
            "stream": True,
            "stream_options": stream_options,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
        }
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if extra_body:
            kwargs["extra_body"] = extra_body

        splitter: Optional[InlineThinkSplitter] = None
        if profile.reasoning in ("inline", "auto"):
            splitter = InlineThinkSplitter(
                *profile.think_tags,
                starts_in_reasoning=profile.reasoning == "inline" and profile.inline_starts_in_reasoning,
            )

        stream = await client.chat.completions.create(**kwargs)
        finish: Optional[str] = None
        try:
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    details = getattr(usage, "completion_tokens_details", None)
                    prompt_details = getattr(usage, "prompt_tokens_details", None)
                    cached = getattr(usage, "prompt_cache_hit_tokens", None)  # DeepSeek
                    if cached is None and prompt_details is not None:
                        cached = getattr(prompt_details, "cached_tokens", None)
                    yield UsageUpdate(
                        prompt_tokens=getattr(usage, "prompt_tokens", None),
                        completion_tokens=getattr(usage, "completion_tokens", None),
                        reasoning_tokens=getattr(details, "reasoning_tokens", None) if details is not None else None,
                        cached_prompt_tokens=cached,
                    )
                for choice in chunk.choices or ():
                    if getattr(choice, "index", 0) != 0:
                        continue
                    if choice.delta is not None:
                        reasoning, content = _split_delta(choice.delta, profile)
                        if reasoning:
                            yield ReasoningDelta(reasoning)
                        if content:
                            pieces = [("content", content)] if splitter is None else splitter.feed(content)
                            for kind, text in pieces:
                                yield ReasoningDelta(text) if kind == REASONING else ContentDelta(text)
                    if choice.finish_reason:
                        finish = str(choice.finish_reason)
            if splitter is not None:
                for kind, text in splitter.flush():
                    yield ReasoningDelta(text) if kind == REASONING else ContentDelta(text)
            yield StreamEnd(finish)
        finally:
            await stream.close()

    async def aclose(self) -> None:
        await self._client.close()
        if self._prefill_client is not None and self._prefill_client is not self._client:
            await self._prefill_client.close()


def _field(delta: Any, name: str) -> Any:
    value = getattr(delta, name, None)
    if value is None:
        extra = getattr(delta, "model_extra", None) or {}
        value = extra.get(name)
    return value


def _split_delta(delta: Any, profile: ModelProfile) -> tuple[str, str]:
    """``(reasoning, content)`` text of one delta."""
    reasoning = ""
    for name in (profile.reasoning_field,) if profile.reasoning_field else _REASONING_FIELDS:
        value = _field(delta, name)
        if isinstance(value, str) and value:
            reasoning = value
            break
    content = getattr(delta, "content", None)
    if isinstance(content, list):  # content parts
        texts, thoughts = [], []
        for part in content:
            part = part if isinstance(part, dict) else getattr(part, "model_dump", lambda: {})()
            text = part.get("text") or part.get("thinking") or ""
            kind = str(part.get("type", "")).lower()
            (thoughts if ("think" in kind or "reason" in kind) else texts).append(text)
        reasoning += "".join(thoughts)
        content = "".join(texts)
    return reasoning, content if isinstance(content, str) else ""
