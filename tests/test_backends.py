import asyncio
import json

import httpx
import pytest

from specgen.generation.prompts import Prompt
from specgen.llm.anthropic import AnthropicBackend
from specgen.llm.backend import ChatParams, make_backend
from specgen.llm.events import ContentDelta, ReasoningDelta, StreamEnd, UsageUpdate
from specgen.llm.file_queue import FileQueueBackend
from specgen.llm.gemini import GeminiBackend
from specgen.llm.openai_compat import OpenAICompatBackend
from specgen.llm.profiles import ModelProfile, ResolvedEndpoint, load_profiles

PROMPT = Prompt(system="sys", sections=(("gpu", "H100"), ("task", "reference"), ("best", ""),
                                         ("instruction", "write it")))


def sse(*events):
    lines = []
    for event in events:
        if isinstance(event, tuple):
            lines.append(f"event: {event[0]}")
            event = event[1]
        lines.append(f"data: {json.dumps(event)}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


class Recorder:
    def __init__(self, *bodies, status=None):
        self.bodies = list(bodies)
        self.statuses = list(status or [200] * len(bodies))
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.statuses.pop(0), content=self.bodies.pop(0),
                              headers={"content-type": "text/event-stream"})


def collect(backend, messages, params):
    async def go():
        try:
            return [e async for e in backend.stream_chat(messages, params)]
        finally:
            await backend.aclose()

    return asyncio.run(go())


def endpoint(backend, modes=None, **kwargs):
    return ResolvedEndpoint("api", backend, kwargs.pop("base_url", ""), "model-x", "secret", {}, modes or {}, **kwargs)


ANTHROPIC_STREAM = sse(
    ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 5}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "plan"}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "s"}}),
    ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "code"}}),
    ("ping", {"type": "ping"}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}}),
    ("message_stop", {"type": "message_stop"}),
)


def test_anthropic_stream_and_request():
    handler = Recorder(ANTHROPIC_STREAM)
    modes = {"main": {"thinking": {"type": "enabled", "budget_tokens": 2048}}}
    backend = AnthropicBackend(endpoint("anthropic", modes), ModelProfile("m"),
                               client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    events = collect(backend, PROMPT.messages(), ChatParams(0.7, 0.9, 4096))
    assert events == [UsageUpdate(prompt_tokens=15), ReasoningDelta("plan"), ContentDelta("code"),
                      UsageUpdate(prompt_tokens=15, completion_tokens=7), StreamEnd("stop")]
    request = handler.requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "secret" and request.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(request.content)
    assert body["system"] == [{"type": "text", "text": "sys"}] and body["stream"] is True
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048} and "temperature" not in body
    blocks = body["messages"][0]["content"]
    assert [b["text"] for b in blocks] == ["H100\n\n", "reference\n\n", "write it"]
    assert [("cache_control" in b) for b in blocks] == [False, True, False]


def test_anthropic_spec_prefill_and_retry():
    handler = Recorder(b"overloaded", ANTHROPIC_STREAM, status=[529, 200])
    backend = AnthropicBackend(endpoint("anthropic"), ModelProfile("m"), max_retries=1,
                               client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    messages = [*PROMPT.messages(), {"role": "assistant", "content": "<think>\nplan\n</think>\n\n```python\n"}]
    events = collect(backend, messages, ChatParams(1.0, 0.95, 512, mode="spec", prefill=True))
    assert events[-1] == StreamEnd("stop") and len(handler.requests) == 2
    body = json.loads(handler.requests[1].content)
    assert body["messages"][-1] == {"role": "assistant", "content": "<think>\nplan\n</think>\n\n```python"}
    assert body["temperature"] == 1.0 and body["top_p"] == 0.95
    assert backend.prefill_style == "native"


def test_anthropic_errors_are_raised():
    handler = Recorder(b'{"error": "bad key"}', status=[401])
    backend = AnthropicBackend(endpoint("anthropic"), ModelProfile("m"),
                               client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(RuntimeError, match="HTTP 401"):
        collect(backend, PROMPT.messages(), ChatParams(1.0, 1.0, 16))


def test_gemini_stream_and_request():
    handler = Recorder(sse(
        {"candidates": [{"content": {"parts": [{"text": "think", "thought": True}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "code"}]}, "finishReason": "MAX_TOKENS"}],
         "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 5, "thoughtsTokenCount": 3}},
    ))
    modes = {"spec": {"thinkingConfig": {"thinkingBudget": 0}}}
    backend = GeminiBackend(endpoint("gemini", modes), ModelProfile("m"),
                            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    events = collect(backend, PROMPT.messages(), ChatParams(0.5, 0.9, 100, mode="spec", seed=3))
    assert events == [ReasoningDelta("think"), ContentDelta("code"),
                      UsageUpdate(prompt_tokens=9, completion_tokens=8, reasoning_tokens=3), StreamEnd("length")]
    request = handler.requests[0]
    assert str(request.url).endswith("/v1beta/models/model-x:streamGenerateContent?alt=sse")
    assert request.headers["x-goog-api-key"] == "secret"
    body = json.loads(request.content)
    assert body["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": PROMPT.user}]}]
    assert body["generationConfig"] == {"temperature": 0.5, "topP": 0.9, "maxOutputTokens": 100, "seed": 3,
                                        "thinkingConfig": {"thinkingBudget": 0}}


def test_file_queue_round_trip(tmp_path):
    backend = FileQueueBackend(endpoint("file_queue", options={"dir": str(tmp_path)}), ModelProfile("m"), poll_s=0.01)

    async def responder():
        while not list((tmp_path / "requests").glob("*.json")):
            await asyncio.sleep(0.01)
        request = json.loads(next((tmp_path / "requests").glob("*.json")).read_text())
        out = tmp_path / "responses" / f"{request['id']}.jsonl"
        for event in ({"type": "reasoning", "text": "plan "}, {"type": "reasoning", "text": "more"},
                      {"type": "content", "text": "code"}, {"type": "usage", "completion_tokens": 4},
                      {"type": "end", "finish_reason": "stop"}):
            with out.open("a") as f:
                f.write(json.dumps(event) + "\n")
            await asyncio.sleep(0.02)
        return request

    async def go():
        task = asyncio.create_task(responder())
        events = [e async for e in backend.stream_chat(PROMPT.messages(), ChatParams(0.6, 0.95, 64, mode="main"))]
        return events, await task

    events, request = asyncio.run(go())
    assert events == [ReasoningDelta("plan "), ReasoningDelta("more"), ContentDelta("code"),
                      UsageUpdate(completion_tokens=4), StreamEnd("stop")]
    assert request["model"] == "model-x" and request["params"]["mode"] == "main"
    assert request["messages"][1] == {"role": "user", "content": PROMPT.user}


def test_file_queue_cancellation_leaves_a_marker(tmp_path):
    backend = FileQueueBackend(endpoint("file_queue", base_url=f"file://{tmp_path}"), ModelProfile("m"), poll_s=0.01)

    async def go():
        stream = backend.stream_chat(PROMPT.messages(), ChatParams(0.6, 0.95, 64))
        task = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await stream.aclose()

    asyncio.run(go())
    assert len(list((tmp_path / "cancelled").iterdir())) == 1


def test_make_backend_dispatch(tmp_path, monkeypatch):
    path = tmp_path / "models.yaml"
    path.write_text(
        "m:\n"
        "  endpoints:\n"
        "    claude: {backend: anthropic, base_url: 'https://api.anthropic.com', model: c, api_key_env: K}\n"
        "    gemini: {backend: gemini, model: g, api_key_env: K}\n"
        f"    queue: {{backend: file_queue, base_url: 'file://{tmp_path}/q', model: q}}\n"
        "    together: {backend: together, base_url: 'https://api.together.xyz/v1', model: t, api_key_env: K}\n"
    )
    monkeypatch.setenv("K", "key")
    profile = load_profiles(path)["m"]
    kinds = {name: type(make_backend(profile, name, timeout_s=10, max_retries=0)).__name__
             for name in ("claude", "gemini", "queue", "together")}
    assert kinds == {"claude": "AnthropicBackend", "gemini": "GeminiBackend", "queue": "FileQueueBackend",
                     "together": "OpenAICompatBackend"}


def test_repository_profiles_still_load():
    assert "DeepSeek-V4-Flash" in load_profiles()
