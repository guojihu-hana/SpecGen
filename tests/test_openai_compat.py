import asyncio

from openai.types.chat import ChatCompletionChunk

from specgen.llm.backend import ChatParams
from specgen.llm.events import ContentDelta, ReasoningDelta, StreamEnd, UsageUpdate
from specgen.llm.openai_compat import OpenAICompatBackend
from specgen.llm.profiles import ModelProfile, ResolvedEndpoint


def chunk(delta=None, finish=None, usage=None):
    choices = [] if delta is None and finish is None else [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    return ChatCompletionChunk.model_validate(
        {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": choices, "usage": usage}
    )


class FakeStream:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

    async def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, chunks):
        self.stream = FakeStream(chunks)
        self.kwargs = None
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.stream

    async def close(self):
        pass


def backend_for(chunks, *, backend="vllm", reasoning="auto"):
    client = FakeClient(chunks)
    modes = {"main": {"thinking": {"type": "enabled"}}, "spec": {"thinking": {"type": "disabled"}}}
    endpoint = ResolvedEndpoint("selfhost", backend, "http://x/v1", "served", "EMPTY", {"top_k": 20}, modes)
    return OpenAICompatBackend(endpoint, ModelProfile("m", reasoning=reasoning), client=client), client


def collect(backend, limit=None, mode="main"):
    async def go():
        events = []
        params = ChatParams(0.6, 0.95, 128, mode=mode, extra_body={"x": 1})
        stream = backend.stream_chat([{"role": "user", "content": "hi"}], params)
        async for event in stream:
            events.append(event)
            if limit is not None and len(events) >= limit:
                await stream.aclose()
                break
        return events

    return asyncio.run(go())


def test_reasoning_field_content_and_usage():
    backend, client = backend_for([
        chunk({"role": "assistant"}),
        chunk({"reasoning_content": "tile "}),
        chunk({"reasoning_content": "size 32"}),
        chunk({"content": "```python\nx=1\n```"}),
        chunk({}, finish="stop"),
        chunk(usage={"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16}),
    ])
    events = collect(backend)
    assert events == [
        ReasoningDelta("tile "), ReasoningDelta("size 32"), ContentDelta("```python\nx=1\n```"),
        UsageUpdate(prompt_tokens=7, completion_tokens=9, reasoning_tokens=None), StreamEnd("stop"),
    ]
    assert client.kwargs["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
    assert client.kwargs["extra_body"] == {"top_k": 20, "thinking": {"type": "enabled"}, "x": 1}
    assert client.kwargs["model"] == "served" and client.kwargs["max_tokens"] == 128
    assert client.stream.closed


def test_cached_prompt_tokens_from_deepseek_and_openai_usage():
    deepseek = {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16, "prompt_cache_hit_tokens": 5,
                "prompt_cache_miss_tokens": 2}
    openai = {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16, "prompt_tokens_details": {"cached_tokens": 4}}
    for usage, cached in ((deepseek, 5), (openai, 4)):
        backend, _ = backend_for([chunk({}, finish="stop"), chunk(usage=usage)])
        events = collect(backend)
        assert UsageUpdate(prompt_tokens=7, completion_tokens=9, cached_prompt_tokens=cached) in events


def test_spec_mode_parameters():
    backend, client = backend_for([chunk({}, finish="stop")])
    collect(backend, mode="spec")
    assert client.kwargs["extra_body"]["thinking"] == {"type": "disabled"}


def test_inline_reasoning_is_split():
    backend, _ = backend_for([chunk({"content": "<think>plan</th"}), chunk({"content": "ink>code"}), chunk({}, finish="stop")],
                             reasoning="inline")
    assert collect(backend) == [ReasoningDelta("plan"), ContentDelta("code"), StreamEnd("stop")]


def test_api_backend_requests_final_usage_only():
    backend, client = backend_for([chunk({}, finish="stop")], backend="api")
    collect(backend)
    assert client.kwargs["stream_options"] == {"include_usage": True}


def test_closing_early_closes_the_http_stream():
    backend, client = backend_for([chunk({"reasoning_content": "a"}), chunk({"reasoning_content": "b"})])
    assert collect(backend, limit=1) == [ReasoningDelta("a")]
    assert client.stream.closed
