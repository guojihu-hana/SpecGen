import asyncio
import os
import subprocess
from pathlib import Path
from typing import Optional

import pytest

from specgen.llm.events import ContentDelta, ReasoningDelta, StreamEnd

RELU_TASK = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)


def get_inputs():
    return [torch.randn(64, 128)]


def get_init_inputs():
    return []
"""

LINEAR_TASK = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        return torch.relu(self.fc(x))


def get_inputs():
    return [torch.randn(8, 32)]


def get_init_inputs():
    return [32, 16]
"""


def relu_kernel(expr: str) -> str:
    """A pure-PyTorch 'kernel' for RELU_TASK whose forward returns ``expr``."""
    return RELU_TASK.replace("return torch.relu(x)", f"return {expr}")


CORRECT_RELU = relu_kernel("torch.clamp_min(x, 0.0)")
WRONG_RELU = relu_kernel("torch.abs(x)")


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def relu_task(tmp_path) -> Path:
    return write(tmp_path / "level0" / "1_ReLU.py", RELU_TASK)


@pytest.fixture
def linear_task(tmp_path) -> Path:
    return write(tmp_path / "level0" / "2_Linear.py", LINEAR_TASK)


class ScriptedBackend:
    """ChatBackend that replays one scripted event list per request."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []
        self.closed = False

    async def stream_chat(self, messages, params):
        self.calls.append(messages)
        for event in self.scripts.pop(0):
            if isinstance(event, BaseException):
                raise event
            yield event

    async def aclose(self):
        self.closed = True


class ModeBackend:
    """ChatBackend with one script queue per generation mode ("main" / "spec")."""

    def __init__(self, main=(), spec=(), prefill_style=None):
        self.scripts = {"main": list(main), "spec": list(spec)}
        self.calls = []  # (mode, messages, params)
        self.prefill_style = prefill_style
        self.closed = False

    async def stream_chat(self, messages, params):
        self.calls.append((params.mode, messages, params))
        script = self.scripts[params.mode].pop(0)
        for event in script:
            if isinstance(event, BaseException):
                raise event
            if isinstance(event, (int, float)) and not isinstance(event, bool):
                await asyncio.sleep(event)  # a number in a script is a pause
                continue
            yield event

    async def aclose(self):
        self.closed = True


def answer(code: str, reasoning: str = "plan the kernel") -> list:
    return [ReasoningDelta(reasoning), ContentDelta(f"```python\n{code}\n```"), StreamEnd("stop")]


def spec_answer(code: str) -> list:
    return [ContentDelta(f"```python\n{code}\n```"), StreamEnd("stop")]


def make_context(run_dir_root: Path, backend, *, gpus=(), triggers=None, k=2, max_iterations=1, eval_cfg=None,
                 spec_prompt_mode="auto", validations_per_gpu=2, max_inflight=0, devices=None, allocation="fixed",
                 evaluator=None, termination=None, spec_queue="drop", schedule="triggers", spec_options=None,
                 **run_kwargs):
    """A RunContext over a CPU pool for loop tests (the caller starts and closes the pool).

    Iterations wait for every candidate unless ``termination`` says otherwise: scripted
    main generations take no time, so any grace ratio would cut the spec candidates."""
    from specgen.config import EvalConfig, RunConfig, SpecConfig, SpecGenConfig, TerminationConfig
    from specgen.control.budget import SpecBudget
    from specgen.control.context import RunContext
    from specgen.evaluation.executor import LocalEvaluator
    from specgen.generation.generator import Generator
    from specgen.generation.prompts import PromptBuilder
    from specgen.llm.profiles import ModelProfile
    from specgen.llm.tokens import TokenCounter
    from specgen.resources.pool import ResourcePool, fixed_slots
    from specgen.storage.layout import RunLayout
    from specgen.telemetry.recorder import PerfRecorder

    eval_cfg = eval_cfg or EvalConfig(gpu_type="H100", ncu_top_n=0, correctness_trials=2, bench_warmup=1,
                                      bench_iters=3, bench_rounds=2)
    cfg = SpecGenConfig(eval=eval_cfg, spec=SpecConfig(enabled=triggers is not None, k=k, max_inflight=max_inflight,
                                                       queue=spec_queue, schedule=schedule, **(spec_options or {})),
                        termination=termination or TerminationConfig(grace_ratio=None),
                        run=RunConfig(max_iterations=max_iterations, **run_kwargs))
    layout = RunLayout(run_dir_root, "r1")
    recorder = PerfRecorder(layout.run_dir)
    return RunContext(
        cfg=cfg,
        generator=Generator(backend, ModelProfile("fake"), TokenCounter("generic"), spec_prompt_mode=spec_prompt_mode),
        pool=ResourcePool(devices or fixed_slots(list(gpus), validations_per_gpu=validations_per_gpu),
                          allocation=allocation, on_event=recorder.event),
        evaluator=evaluator or LocalEvaluator(timeout_s=180),
        prompts=PromptBuilder("H100"),
        recorder=recorder,
        layout=layout,
        budget=SpecBudget(k=k, max_inflight=max_inflight),
        triggers=triggers,
        ncu_enabled=eval_cfg.ncu_top_n > 0,
    )


def find_idle_gpu() -> Optional[str]:
    forced = os.environ.get("SPECGEN_TEST_GPU")
    if forced is not None:
        return forced or None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    for line in out.stdout.splitlines():
        index, memory, util = (c.strip() for c in line.split(","))
        if float(memory) < 500 and float(util) < 10:
            return index
    return None


@pytest.fixture
def idle_gpu() -> str:
    gpu = find_idle_gpu()
    if gpu is None:
        pytest.skip("no idle GPU")
    return gpu
