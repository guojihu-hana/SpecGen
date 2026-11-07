"""Google Gemini backend (``streamGenerateContent`` over server-sent events).

Parts flagged ``thought`` (``thinkingConfig.includeThoughts``) become reasoning.
Mode parameters are merged into ``generationConfig``, e.g. ``thinkingConfig``.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

import httpx

from specgen.llm.backend import ChatParams
from specgen.llm.events import ContentDelta, GenEvent, ReasoningDelta, StreamEnd, UsageUpdate
from specgen.llm.profiles import ModelProfile, ResolvedEndpoint
from specgen.llm.sse import iter_sse, loads, open_stream

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
FINISH_REASONS = {"STOP": "stop", "MAX_TOKENS": "length"}


def build_request(messages: list[dict[str, Any]], params: ChatParams, endpoint: ResolvedEndpoint,
                  profile: ModelProfile) -> dict[str, Any]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"]
    generation: dict[str, Any] = {"temperature": params.temperature, "topP": params.top_p,
                                  "maxOutputTokens": params.max_tokens}
    if params.seed is not None:
        generation["seed"] = params.seed
    generation.update(profile.mode_params(params.mode, endpoint))
    generation.update(params.extra_body)
    body: dict[str, Any] = {"contents": contents, "generationConfig": generation, **endpoint.extra_body}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


class GeminiBackend:
    prefill_style = None

    def __init__(self, endpoint: ResolvedEndpoint, profile: ModelProfile, *, timeout_s: float = 3600.0,
                 max_retries: int = 2, client: Optional[httpx.AsyncClient] = None):
        self._endpoint, self._profile, self._max_retries = endpoint, profile, max_retries
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=30.0))
        base = (endpoint.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._url = f"{base}/models/{endpoint.model}:streamGenerateContent?alt=sse"
        self._headers = {"x-goog-api-key": endpoint.api_key, "content-type": "application/json"}

    async def stream_chat(self, messages: list[dict[str, Any]], params: ChatParams) -> AsyncIterator[GenEvent]:
        body = build_request(messages, params, self._endpoint, self._profile)
        finish: Optional[str] = None
        async with open_stream(self._client, self._url, json_body=body, headers=self._headers,
                               max_retries=self._max_retries) as response:
            async for _, data in iter_sse(response):
                payload = loads(data) or {}
                if payload.get("error"):
                    raise RuntimeError(f"gemini stream error: {payload['error']}")
                for candidate in (payload.get("candidates") or [])[:1]:
                    for part in (candidate.get("content") or {}).get("parts") or ():
                        text = part.get("text")
                        if text:
                            yield ReasoningDelta(text) if part.get("thought") else ContentDelta(text)
                    if candidate.get("finishReason"):
                        reason = candidate["finishReason"]
                        finish = FINISH_REASONS.get(reason, reason.lower())
                usage = payload.get("usageMetadata")
                if usage:
                    completion = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
                    yield UsageUpdate(prompt_tokens=usage.get("promptTokenCount"), completion_tokens=completion or None,
                                      reasoning_tokens=usage.get("thoughtsTokenCount"))
        yield StreamEnd(finish)

    async def aclose(self) -> None:
        await self._client.aclose()
