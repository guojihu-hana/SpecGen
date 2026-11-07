"""Anthropic Messages API backend (streaming, extended thinking, prompt caching).

Thinking blocks become reasoning, text blocks content.  The prompt's stable sections
(task and best kernel) end with cache breakpoints, so iterations of one task reuse
the cached prefix.  A trailing assistant message is continued natively (spec prefill).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

import httpx

from specgen.llm.backend import ChatParams
from specgen.llm.events import ContentDelta, GenEvent, ReasoningDelta, StreamEnd, UsageUpdate
from specgen.llm.profiles import ModelProfile, ResolvedEndpoint
from specgen.llm.sse import iter_sse, loads, open_stream

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"
CACHED_SECTIONS = ("task", "best")
STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}


def _user_blocks(message: dict[str, Any]) -> Any:
    """Text blocks per prompt section, with cache breakpoints after the stable ones."""
    sections = [s for s in message.get("sections") or () if s[1]]
    if not sections:
        return message["content"]
    blocks = []
    for i, (name, text) in enumerate(sections):
        block: dict[str, Any] = {"type": "text", "text": text if i == len(sections) - 1 else f"{text}\n\n"}
        if name in CACHED_SECTIONS:
            block["cache_control"] = {"type": "ephemeral"}
        blocks.append(block)
    return blocks


def build_request(messages: list[dict[str, Any]], params: ChatParams, endpoint: ResolvedEndpoint,
                  profile: ModelProfile) -> dict[str, Any]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = []
    for m in messages:
        if m["role"] == "system":
            continue
        content = _user_blocks(m) if m["role"] == "user" else m["content"]
        if m["role"] == "assistant" and isinstance(content, str):
            content = content.rstrip()  # the API rejects a final assistant turn ending in whitespace
        turns.append({"role": m["role"], "content": content})
    body: dict[str, Any] = {"model": endpoint.model, "max_tokens": params.max_tokens, "messages": turns, "stream": True}
    if system:
        body["system"] = [{"type": "text", "text": system}]
    body.update(endpoint.extra_body)
    body.update(profile.mode_params(params.mode, endpoint))
    body.update(params.extra_body)
    thinking = (body.get("thinking") or {}).get("type") == "enabled"
    if not thinking:  # extended thinking fixes the sampling parameters
        body.setdefault("temperature", params.temperature)
        body.setdefault("top_p", params.top_p)
    return body


class AnthropicBackend:
    prefill_style = "native"

    def __init__(self, endpoint: ResolvedEndpoint, profile: ModelProfile, *, timeout_s: float = 3600.0,
                 max_retries: int = 2, client: Optional[httpx.AsyncClient] = None):
        self._endpoint, self._profile, self._max_retries = endpoint, profile, max_retries
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=30.0))
        self._url = (endpoint.base_url or DEFAULT_BASE_URL).rstrip("/") + "/v1/messages"
        self._headers = {"x-api-key": endpoint.api_key, "anthropic-version": endpoint.options.get("version", API_VERSION),
                         "content-type": "application/json"}
        beta = endpoint.options.get("beta")
        if beta:
            self._headers["anthropic-beta"] = ",".join(beta) if isinstance(beta, (list, tuple)) else str(beta)

    async def stream_chat(self, messages: list[dict[str, Any]], params: ChatParams) -> AsyncIterator[GenEvent]:
        body = build_request(messages, params, self._endpoint, self._profile)
        finish: Optional[str] = None
        prompt_tokens: Optional[int] = None
        async with open_stream(self._client, self._url, json_body=body, headers=self._headers,
                               max_retries=self._max_retries) as response:
            async for event, data in iter_sse(response):
                payload = loads(data) or {}
                kind = payload.get("type", event)
                if kind == "message_start":
                    usage = (payload.get("message") or {}).get("usage") or {}
                    prompt_tokens = sum(usage.get(k) or 0 for k in
                                        ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
                    yield UsageUpdate(prompt_tokens=prompt_tokens)
                elif kind == "content_block_start":
                    block = payload.get("content_block") or {}
                    if block.get("type") == "text" and block.get("text"):
                        yield ContentDelta(block["text"])
                    elif block.get("type") == "thinking" and block.get("thinking"):
                        yield ReasoningDelta(block["thinking"])
                elif kind == "content_block_delta":
                    delta = payload.get("delta") or {}
                    if delta.get("type") == "thinking_delta" and delta.get("thinking"):
                        yield ReasoningDelta(delta["thinking"])
                    elif delta.get("type") == "text_delta" and delta.get("text"):
                        yield ContentDelta(delta["text"])
                elif kind == "message_delta":
                    stop = (payload.get("delta") or {}).get("stop_reason")
                    if stop:
                        finish = STOP_REASONS.get(stop, stop)
                    usage = payload.get("usage") or {}
                    if usage.get("output_tokens") is not None:
                        yield UsageUpdate(prompt_tokens=prompt_tokens, completion_tokens=usage["output_tokens"])
                elif kind == "error":
                    raise RuntimeError(f"anthropic stream error: {payload.get('error') or data}")
        yield StreamEnd(finish)

    async def aclose(self) -> None:
        await self._client.aclose()
