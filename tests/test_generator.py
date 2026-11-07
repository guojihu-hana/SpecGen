import asyncio

from conftest import ScriptedBackend

from specgen.generation.generator import Generator
from specgen.generation.prompts import Prompt
from specgen.llm.events import ContentDelta, ReasoningDelta, StreamEnd, UsageUpdate
from specgen.llm.profiles import ModelProfile, Sampling
from specgen.llm.tokens import TokenCounter

PROMPT = Prompt(system="sys", sections=(("task", "do it"),))


class ModeRecordingBackend(ScriptedBackend):
    def __init__(self, scripts):
        super().__init__(scripts)
        self.modes = []

    async def stream_chat(self, messages, params):
        self.modes.append(params.mode)
        async for event in super().stream_chat(messages, params):
            yield event


class Sink:
    def __init__(self):
        self.reasoning, self.content, self.last_tokens = [], [], 0

    def on_reasoning(self, text, total):
        self.reasoning.append(text)
        self.last_tokens = total

    def on_content(self, text, total):
        self.content.append(text)


def generate(scripts, **kwargs):
    backend = ModeRecordingBackend(scripts)
    generator = Generator(backend, ModelProfile("m", sampling=Sampling(temperature=0.5)), TokenCounter("generic"), **kwargs)
    sink = Sink()
    return asyncio.run(generator.generate_main(PROMPT, sink)), backend, sink, generator


def test_collects_reasoning_content_and_code():
    result, backend, sink, generator = generate([[
        ReasoningDelta("think "), ReasoningDelta("more"),
        ContentDelta("```python\nimport torch\nclass Model: pass\n```"),
        UsageUpdate(prompt_tokens=10, completion_tokens=20, cached_prompt_tokens=8), StreamEnd("stop"),
    ]])
    assert result.ok and result.finish_reason == "stop"
    assert result.reasoning == "think more" and result.code == "import torch\nclass Model: pass"
    assert result.server_completion_tokens == 20 and result.server_prompt_tokens == 10
    assert result.server_cached_prompt_tokens == 8 and result.summary()["server_cached_prompt_tokens"] == 8
    assert result.reasoning_tokens > 0 and result.content_tokens > 0 and result.ttft_s is not None
    assert sink.reasoning == ["think ", "more"] and sink.last_tokens == result.reasoning_tokens
    assert generator.params.temperature == 0.5
    assert backend.calls[0] == PROMPT.messages()


def test_truncation_triggers_one_continuation():
    result, backend, _, _ = generate([
        [ReasoningDelta("long plan"), UsageUpdate(completion_tokens=100), StreamEnd("length")],
        [ContentDelta("```python\nx = 1\n```"), UsageUpdate(completion_tokens=5), StreamEnd("stop")],
    ])
    assert result.continuations == 1 and result.finish_reason == "stop" and result.code == "x = 1"
    assert result.server_completion_tokens == 105
    assert backend.modes == ["main", "spec"]
    follow_up = backend.calls[1][-1]["content"]
    assert follow_up.startswith("do it") and "long plan" in follow_up and "reasoning so far" in follow_up


def test_backend_error_is_reported():
    result, *_ = generate([[ReasoningDelta("x"), RuntimeError("connection reset")]])
    assert not result.ok and "RuntimeError: connection reset" in result.error
    assert result.reasoning == "x"


def test_code_from_reasoning_fallback():
    result, *_ = generate([[ReasoningDelta("```python\ny = 2\n```"), StreamEnd("stop")]], max_continuations=0)
    assert result.code == "y = 2" and result.code_from_reasoning
