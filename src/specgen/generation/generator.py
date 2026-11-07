"""Generation: the streamed main completion and speculative completions from its reasoning prefix."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional, Protocol

from specgen.generation.extract import extract_python_module
from specgen.generation.prompts import Prompt
from specgen.llm.backend import ChatBackend, ChatParams
from specgen.llm.events import ContentDelta, ReasoningDelta, StreamEnd, UsageUpdate
from specgen.llm.profiles import ModelProfile
from specgen.llm.tokens import StreamCounter, TokenCounter

CONTINUE_INSTRUCTION = (
    "Your previous answer was cut off by the output length limit. Its reasoning so far is below. "
    "Do not continue reasoning; output only the final ```python fence with the complete module.\n\n"
    "--- reasoning so far ---\n{reasoning}\n--- end of reasoning ---"
)
SPEC_INSTRUCTION = (
    "Below is your reasoning on this task so far. Stop reasoning and write the kernel it leads to: "
    "output only the final ```python fence with the complete module.\n\n"
    "--- reasoning so far ---\n{reasoning}\n--- end of reasoning ---"
)
PREFILL_CODE_START = "```python\n"
PREFILL_APPROACH = "# APPROACH:"  # a prefilled spec writes its self-report line (see control.attempts) first
SPEC_PROMPT_MODES = ("auto", "prefill", "reprompt")


def resolve_spec_prompt_mode(requested: str, backend: ChatBackend) -> str:
    """``prefill`` needs an endpoint that continues assistant prefixes; ``auto`` uses it when available."""
    supported = getattr(backend, "prefill_style", None) is not None
    if requested == "auto":
        return "prefill" if supported else "reprompt"
    if requested == "prefill" and not supported:
        raise ValueError("spec prompt mode 'prefill' needs an endpoint with a prefill style in the models file")
    if requested not in SPEC_PROMPT_MODES:
        raise ValueError(f"unknown spec prompt mode {requested!r}")
    return requested


class GenerationSink(Protocol):
    """Receives text as it streams, with the running token count of that kind."""

    def on_reasoning(self, text: str, total_tokens: int) -> None: ...

    def on_content(self, text: str, total_tokens: int) -> None: ...


@dataclass
class GenerationResult:
    kind: str = "main"  # "main" or "spec"
    ok: bool = False
    reasoning: str = ""
    content: str = ""
    code: Optional[str] = None
    code_from_reasoning: bool = False
    prompt_mode: Optional[str] = None  # spec: "prefill" or "reprompt"
    temperature: Optional[float] = None
    finish_reason: Optional[str] = None
    reasoning_tokens: int = 0
    content_tokens: int = 0
    server_completion_tokens: Optional[int] = None
    server_prompt_tokens: Optional[int] = None
    server_cached_prompt_tokens: Optional[int] = None  # of the prompt tokens, those the provider had cached
    started_at: float = 0.0
    seconds: float = 0.0
    ttft_s: Optional[float] = None
    continuations: int = 0
    error: Optional[str] = None

    def summary(self) -> dict:
        out = asdict(self)
        for key in ("reasoning", "content", "code"):
            out.pop(key)
        out["has_code"] = self.code is not None
        return out


@dataclass
class _StreamState:
    t0: float
    reasoning: list[str] = field(default_factory=list)
    content: list[str] = field(default_factory=list)
    reasoning_counter: Optional[StreamCounter] = None
    content_counter: Optional[StreamCounter] = None
    first_token_s: Optional[float] = None
    completion_tokens: int = 0
    prompt_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None


class Generator:
    def __init__(
        self,
        backend: ChatBackend,
        profile: ModelProfile,
        counter: TokenCounter,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_continuations: int = 1,
        spec_prompt_mode: str = "auto",
    ):
        sampling = profile.sampling
        self.backend = backend
        self.profile = profile
        self.counter = counter
        self.max_continuations = max(0, max_continuations)
        self.params = ChatParams(
            temperature=sampling.temperature if temperature is None else temperature,
            top_p=sampling.top_p if top_p is None else top_p,
            max_tokens=sampling.max_tokens if max_tokens is None else max_tokens,
        )
        self.spec_prompt_mode = resolve_spec_prompt_mode(spec_prompt_mode, backend)

    def _new_state(self) -> _StreamState:
        return _StreamState(
            t0=time.perf_counter(), reasoning_counter=self.counter.stream(), content_counter=self.counter.stream()
        )

    @staticmethod
    def _reprompt(prompt: Prompt, instruction: str, reasoning: str) -> list[dict[str, Any]]:
        # Appended after the whole prompt, so the request shares the main request's cached prefix.
        messages = prompt.messages()
        extra = instruction.format(reasoning=reasoning)
        messages[-1] = {"role": "user", "content": f"{prompt.user}\n\n{extra}",
                        "sections": [*messages[-1]["sections"], ["spec", extra]]}
        return messages

    def _prefill(self, prompt: Prompt, reasoning: str, notes: str = "") -> list[dict[str, Any]]:
        open_tag, close_tag = self.profile.think_tags
        prefix = f"{open_tag}\n{reasoning}{notes}\n{close_tag}\n\n{PREFILL_CODE_START}{PREFILL_APPROACH}"
        return [*prompt.messages(), {"role": "assistant", "content": prefix}]

    async def generate_main(self, prompt: Prompt, sink: Optional[GenerationSink] = None) -> GenerationResult:
        result = GenerationResult(kind="main", started_at=time.time(), temperature=self.params.temperature)
        state = self._new_state()
        try:
            finish = await self._consume(prompt.messages(), self.params, state, sink)
            while finish == "length" and result.continuations < self.max_continuations:
                # Finishing from the reasoning so far is a spec-style request: thinking off.
                result.continuations += 1
                messages = self._reprompt(prompt, CONTINUE_INSTRUCTION, "".join(state.reasoning))
                finish = await self._consume(messages, replace(self.params, mode="spec"), state, sink)
            result.finish_reason = finish
            result.ok = True
        except Exception as e:  # noqa: BLE001 - provider errors become a failed generation
            result.error = f"{type(e).__name__}: {e}"
        self._finish(result, state)
        return result

    async def generate_spec(self, prompt: Prompt, reasoning: str, *, index: int = 0,
                            seed: Optional[int] = None, notes: str = "") -> GenerationResult:
        """Write the final module from a prefix of the main reasoning, without thinking.

        ``index`` is the spec's position within its trigger's batch: later specs of
        one prefix sample at a higher temperature.  ``notes`` (test results of earlier
        specs) follow the reasoning, after the prefix every request shares.
        """
        sampling = self.profile.sampling
        prefill = self.spec_prompt_mode == "prefill"
        params = replace(self.params, mode="spec", prefill=prefill, seed=seed,
                         temperature=sampling.spec_temperature_for(index),
                         max_tokens=min(self.params.max_tokens, sampling.spec_max_tokens))
        messages = (self._prefill(prompt, reasoning, notes) if prefill
                    else self._reprompt(prompt, SPEC_INSTRUCTION, reasoning + notes))
        result = GenerationResult(kind="spec", started_at=time.time(), prompt_mode=self.spec_prompt_mode,
                                  temperature=params.temperature)
        state = self._new_state()
        try:
            result.finish_reason = await self._consume(messages, params, state, None)
            result.ok = True
        except Exception as e:  # noqa: BLE001
            result.error = f"{type(e).__name__}: {e}"
        self._finish(result, state, content_prefix=PREFILL_CODE_START + PREFILL_APPROACH if prefill else "")
        return result

    def _finish(self, result: GenerationResult, state: _StreamState, content_prefix: str = "") -> None:
        result.seconds = round(time.perf_counter() - state.t0, 3)
        result.ttft_s = None if state.first_token_s is None else round(state.first_token_s, 3)
        result.reasoning = "".join(state.reasoning)
        result.content = content_prefix + "".join(state.content)
        result.reasoning_tokens = state.reasoning_counter.total
        result.content_tokens = state.content_counter.total
        result.server_completion_tokens = state.completion_tokens or None
        result.server_prompt_tokens = state.prompt_tokens
        result.server_cached_prompt_tokens = state.cached_prompt_tokens
        result.code, result.code_from_reasoning = module_from_output(result.content, result.reasoning,
                                                                     prefilled=bool(content_prefix))

    async def _consume(
        self, messages: list[dict[str, Any]], params: ChatParams, state: _StreamState, sink: Optional[GenerationSink]
    ) -> Optional[str]:
        finish: Optional[str] = None
        base_completion = state.completion_tokens
        stream = self.backend.stream_chat(messages, params)
        try:
            async for event in stream:
                if isinstance(event, (ReasoningDelta, ContentDelta)) and state.first_token_s is None:
                    state.first_token_s = time.perf_counter() - state.t0
                if isinstance(event, ReasoningDelta):
                    state.reasoning.append(event.text)
                    total = state.reasoning_counter.add(event.text)
                    if sink is not None:
                        sink.on_reasoning(event.text, total)
                elif isinstance(event, ContentDelta):
                    state.content.append(event.text)
                    total = state.content_counter.add(event.text)
                    if sink is not None:
                        sink.on_content(event.text, total)
                elif isinstance(event, UsageUpdate):
                    if event.completion_tokens is not None:
                        state.completion_tokens = base_completion + event.completion_tokens
                    if event.prompt_tokens is not None and state.prompt_tokens is None:
                        state.prompt_tokens = event.prompt_tokens
                    if event.cached_prompt_tokens is not None and state.cached_prompt_tokens is None:
                        state.cached_prompt_tokens = event.cached_prompt_tokens
                elif isinstance(event, StreamEnd):
                    finish = event.finish_reason
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
        return finish


def module_from_output(content: str, reasoning: str = "", *, prefilled: bool = False) -> tuple[Optional[str], bool]:
    """(the module, whether it came from the reasoning) in a generation's output: the content's fence; for a
    prefilled spec what the prefill opened inside the fence; else a fence in the reasoning."""
    code = extract_python_module(content)
    if code is None and prefilled:
        code = _unclosed_module(content[len(PREFILL_CODE_START):])
    if code is None:
        code = extract_python_module(reasoning)
        return code, code is not None
    return code, False


def _unclosed_module(text: str) -> Optional[str]:
    """A prefilled fence the model never closed (e.g. cut by max_tokens)."""
    body = text.split("```", 1)[0].strip()
    return body if "import" in body and "Model" in body else None
