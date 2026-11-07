import asyncio
import json
import re

import pytest
from conftest import CORRECT_RELU, WRONG_RELU, ModeBackend, answer, make_context, relu_kernel, spec_answer
from test_openai_compat import FakeClient, chunk

from specgen.control.loop import IterationLoop
from specgen.control.spec import code_fingerprint
from specgen.control.triggers import TriggerConfig, TriggerGroup
from specgen.generation.generator import Generator
from specgen.generation.prompts import Prompt
from specgen.llm.backend import ChatParams
from specgen.llm.events import ContentDelta, ReasoningDelta, StreamEnd
from specgen.llm.openai_compat import OpenAICompatBackend
from specgen.llm.profiles import ModelProfile, ResolvedEndpoint, Sampling
from specgen.llm.tokens import TokenCounter
from specgen.tasks.base import KernelBenchTask

PROMPT = Prompt(system="sys", sections=(("task", "do it"),))


def test_spec_reprompt_appends_the_reasoning_after_the_prompt():
    backend = ModeBackend(spec=[spec_answer("x = 1")])
    sampling = Sampling(spec_temperature=0.8, spec_temperature_step=0.2, spec_temperature_max=1.1,
                        spec_max_tokens=100, max_tokens=1000)
    generator = Generator(backend, ModelProfile("m", sampling=sampling), TokenCounter("generic"))
    assert generator.spec_prompt_mode == "reprompt"
    result = asyncio.run(generator.generate_spec(PROMPT, "my partial plan", index=2, seed=7))
    mode, messages, params = backend.calls[0]
    assert mode == "spec" and not params.prefill and params.seed == 7 and params.max_tokens == 100
    assert params.temperature == 1.1  # 0.8 + 2 * 0.2, capped
    assert messages[0] == PROMPT.messages()[0]
    assert messages[1]["content"].startswith(PROMPT.user) and "my partial plan" in messages[1]["content"]
    assert result.ok and result.code == "x = 1" and result.kind == "spec" and result.prompt_mode == "reprompt"
    assert sampling.spec_temperature_for(0) == 0.8


def test_spec_prefill_continues_an_assistant_prefix():
    backend = ModeBackend(spec=[[ContentDelta(" copy\nimport torch\nclass Model: pass\n```"), StreamEnd("stop")]],
                          prefill_style="deepseek")
    generator = Generator(backend, ModelProfile("m"), TokenCounter("generic"))
    assert generator.spec_prompt_mode == "prefill"
    result = asyncio.run(generator.generate_spec(PROMPT, "plan"))
    _, messages, params = backend.calls[0]
    assert params.prefill and messages[:2] == PROMPT.messages()
    assert messages[2] == {"role": "assistant", "content": "<think>\nplan\n</think>\n\n```python\n# APPROACH:"}
    assert result.code == "# APPROACH: copy\nimport torch\nclass Model: pass" and result.prompt_mode == "prefill"


def test_spec_prefill_keeps_an_unclosed_module():
    backend = ModeBackend(spec=[[ContentDelta(" copy\nimport torch\nclass Model: pass"), StreamEnd("length")]],
                          prefill_style="vllm")
    result = asyncio.run(Generator(backend, ModelProfile("m"), TokenCounter("generic")).generate_spec(PROMPT, "p"))
    assert result.code == "# APPROACH: copy\nimport torch\nclass Model: pass" and result.finish_reason == "length"


def test_prefill_mode_needs_endpoint_support():
    with pytest.raises(ValueError):
        Generator(ModeBackend(), ModelProfile("m"), TokenCounter("generic"), spec_prompt_mode="prefill")
    assert Generator(ModeBackend(prefill_style="vllm"), ModelProfile("m"), TokenCounter("generic"),
                     spec_prompt_mode="reprompt").spec_prompt_mode == "reprompt"


def _prefill_backend(style, prefill_url=None):
    main, beta = FakeClient([chunk({}, finish="stop")]), FakeClient([chunk({}, finish="stop")])
    endpoint = ResolvedEndpoint("api", "api", "https://api.example.com", "m", "k", {},
                                {"spec": {"thinking": {"type": "disabled"}}}, prefill_style=style,
                                prefill_base_url=prefill_url)
    return OpenAICompatBackend(endpoint, ModelProfile("m"), client=main, prefill_client=beta), main, beta


def _send(backend, messages, prefill):
    async def go():
        params = ChatParams(1.0, 0.95, 64, mode="spec", prefill=prefill)
        return [e async for e in backend.stream_chat(messages, params)]

    return asyncio.run(go())


def test_deepseek_prefill_uses_the_beta_client_and_prefix_flag():
    backend, main, beta = _prefill_backend("deepseek", "https://api.example.com/beta")
    messages = [{"role": "user", "content": "u", "cache_breaks": [1]}, {"role": "assistant", "content": "a"}]
    _send(backend, messages, prefill=True)
    assert main.kwargs is None and beta.kwargs["messages"][-1] == {"role": "assistant", "content": "a", "prefix": True}
    assert beta.kwargs["messages"][0] == {"role": "user", "content": "u"}  # unknown keys are dropped
    assert beta.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "prefix" not in messages[-1]  # the caller's messages are not modified


def test_vllm_prefill_continues_the_final_message():
    backend, main, _ = _prefill_backend("vllm")
    _send(backend, [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}], prefill=True)
    assert main.kwargs["extra_body"]["continue_final_message"] is True
    assert main.kwargs["extra_body"]["add_generation_prompt"] is False


def test_prefill_without_support_is_an_error():
    backend, _, _ = _prefill_backend(None)
    with pytest.raises(ValueError):
        _send(backend, [{"role": "assistant", "content": "a"}], prefill=True)


def test_code_fingerprint_ignores_layout():
    assert code_fingerprint("a = 1  \n\n\nb = 2\n") == code_fingerprint("a = 1\nb = 2")
    assert code_fingerprint("a = 1") != code_fingerprint("a = 2")


def keyword_triggers(after=0, fallback=100000):
    group = TriggerGroup("kw", (re.compile(r"\btile size\b", re.IGNORECASE),))
    return TriggerConfig(groups=(group,), after_tokens=after, fallback_tokens=fallback, min_gap_tokens=0)


def run_spec_loop(tmp_path, task_path, backend, *, iterations=1, k=2, **kwargs):
    ctx = make_context(tmp_path / "out", backend, triggers=keyword_triggers(), k=k, max_iterations=iterations,
                       tasks=[task_path], **kwargs)

    async def go():
        await ctx.pool.start()
        try:
            return await IterationLoop(KernelBenchTask.from_path(task_path), ctx).run()
        finally:
            await ctx.pool.close()
            ctx.recorder.close()

    return asyncio.run(go()), ctx


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_spec_candidates_are_evaluated_and_the_fastest_is_selected(tmp_path, relu_task):
    slow_main = relu_kernel("torch.clamp_min(x, 0.0) + 0 * torch.stack([x] * 64).sum(0)")
    main = [ReasoningDelta("First, the tile size. "), ReasoningDelta("Then the rest. "),
            ContentDelta(f"```python\n{slow_main}\n```"), StreamEnd("stop")]
    backend = ModeBackend(main=[main], spec=[spec_answer(CORRECT_RELU), spec_answer(WRONG_RELU)])
    history, ctx = run_spec_loop(tmp_path, relu_task, backend)

    by_id = {o.kernel_id: o for o in history.outcomes}
    assert set(by_id) == {"i000-main", "i000-t01-s0", "i000-t01-s1"}
    assert by_id["i000-t01-s0"].status == "success" and by_id["i000-t01-s1"].status == "numerical_error"
    assert by_id["i000-main"].status == "success"
    spec_calls = [c for c in backend.calls if c[0] == "spec"]
    assert len(spec_calls) == 2 and "First, the tile size." in spec_calls[0][1][-1]["content"]
    assert [c[2].temperature for c in spec_calls] == [0.9, 1.0]

    run_dir = ctx.layout.run_dir
    kernels = {k["kernel_id"]: k for k in read_jsonl(run_dir / "kernels.jsonl")}
    assert kernels["i000-t01-s0"]["trigger"]["reason"] == "kw" and kernels["i000-t01-s0"]["batch"] == {"k": 2, "index": 0}
    assert sum(k["selected"] for k in kernels.values()) == 1
    iteration = read_jsonl(run_dir / "iterations.jsonl")[0]
    assert iteration["triggers"][0]["k"] == 2 and iteration["candidates"] == 3
    assert iteration["selected"] == max(
        (o for o in history.outcomes if o.benchmarked), key=lambda o: o.speedup).kernel_id
    events = {e["type"] for e in read_jsonl(run_dir / "events.jsonl")}
    assert {"spec_triggered", "spec_started", "spec_finished", "main_started", "main_finished"} <= events
    spec_dir = run_dir / "level0__1_ReLU" / "iter_000" / "kernels" / "i000-t01-s0"
    assert (spec_dir / "content.txt").read_text().startswith("```python")


def test_duplicate_candidates_are_not_evaluated_twice(tmp_path, relu_task):
    main = [ReasoningDelta("pick a tile size"), ContentDelta(f"```python\n{CORRECT_RELU}\n```"), StreamEnd("stop")]
    backend = ModeBackend(main=[main], spec=[spec_answer(CORRECT_RELU + "\n\n"), [ContentDelta("no code"), StreamEnd("stop")]])
    history, ctx = run_spec_loop(tmp_path, relu_task, backend)
    statuses = {o.kernel_id: o.status for o in history.outcomes}
    assert statuses["i000-main"] == "success"  # its code was seen first
    assert statuses["i000-t01-s0"] == "duplicate" and statuses["i000-t01-s1"] == "parse_error"
    kernels = {k["kernel_id"]: k for k in read_jsonl(ctx.layout.run_dir / "kernels.jsonl")}
    assert kernels["i000-t01-s0"]["duplicate_of"] == "i000-main"


def test_no_triggers_without_keywords(tmp_path, relu_task):
    backend = ModeBackend(main=[answer(CORRECT_RELU, reasoning="nothing special")])
    history, _ = run_spec_loop(tmp_path, relu_task, backend)
    assert [o.kernel_id for o in history.outcomes] == ["i000-main"]


def _three_triggers(pause):
    return [ReasoningDelta("tile size"), pause, ReasoningDelta(" then tile size"), pause, ReasoningDelta(" so tile size"),
            ContentDelta(f"```python\n{CORRECT_RELU}\n```"), StreamEnd("stop")]


def _variant(n, pause=0.0):
    return [pause, ContentDelta(f"```python\n{relu_kernel(f'torch.clamp_min(x, 0.0) + {n} * 0')}\n```"), StreamEnd("stop")]


def test_in_flight_counts_generation_only(tmp_path, relu_task):
    # each spec generates in 0.05 s but evaluates for seconds: the slot is free again by the next trigger
    backend = ModeBackend(main=[_three_triggers(0.5)], spec=[_variant(i, 0.05) for i in range(3)])
    history, ctx = run_spec_loop(tmp_path, relu_task, backend, k=1, max_inflight=1)
    iteration = read_jsonl(ctx.layout.run_dir / "iterations.jsonl")[0]
    assert [t["k"] for t in iteration["triggers"]] == [1, 1, 1]
    assert ctx.budget.inflight == 0


def test_replace_supersedes_queued_specs(tmp_path, relu_task):
    # the first spec holds the only slot for 1 s; trigger 2 queues behind it and trigger 3 replaces it
    backend = ModeBackend(main=[_three_triggers(0.1)], spec=[_variant(1, 1.0), _variant(3)])
    history, ctx = run_spec_loop(tmp_path, relu_task, backend, k=1, max_inflight=1, spec_queue="replace")
    assert sorted(o.kernel_id for o in history.outcomes if o.source == "spec") == ["i000-t01-s0", "i000-t03-s0"]
    iteration = read_jsonl(ctx.layout.run_dir / "iterations.jsonl")[0]
    assert [t["superseded"] for t in iteration["triggers"]] == [0, 0, 1] and iteration["superseded_specs"] == 1
    events = read_jsonl(ctx.layout.run_dir / "events.jsonl")
    assert [e["kernels"] for e in events if e["type"] == "spec_superseded"] == [["i000-t02-s0"]]
    assert ctx.budget.inflight == 0


def test_drop_launches_nothing_while_the_slot_is_busy(tmp_path, relu_task):
    backend = ModeBackend(main=[_three_triggers(0.1)], spec=[_variant(1, 1.0)])
    history, ctx = run_spec_loop(tmp_path, relu_task, backend, k=1, max_inflight=1)
    iteration = read_jsonl(ctx.layout.run_dir / "iterations.jsonl")[0]
    assert [t["k"] for t in iteration["triggers"]] == [1, 0, 0]


def test_max_inflight_caps_k(tmp_path, relu_task):
    main = [ReasoningDelta("tile size"), 0.01, ReasoningDelta(" and the tile size again"),
            ContentDelta(f"```python\n{CORRECT_RELU}\n```"), StreamEnd("stop")]
    spec = lambda n: [0.2, ContentDelta(f"```python\n{relu_kernel(f'torch.clamp_min(x, 0.0) + {n} * 0')}\n```"),
                      StreamEnd("stop")]
    backend = ModeBackend(main=[main], spec=[spec(i) for i in range(3)])
    history, ctx = run_spec_loop(tmp_path, relu_task, backend, k=3, max_inflight=2)
    assert [o.status for o in history.outcomes if o.source == "spec"] == ["success"] * 2
    iteration = read_jsonl(ctx.layout.run_dir / "iterations.jsonl")[0]
    assert [t["k"] for t in iteration["triggers"]] == [2, 0]  # the second trigger found no capacity left
    assert ctx.budget.inflight == 0
