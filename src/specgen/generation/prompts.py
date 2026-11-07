"""Prompt construction.

Sections are ordered from most to least stable, so LLM prefix caches are reused
across tasks (system, GPU block), across iterations of one task (reference) and
while the best kernel is unchanged (best block); per-iteration feedback comes
last.  Nothing volatile (iteration numbers, timestamps, ids) appears before the
feedback, and numbers are formatted deterministically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from specgen.generation.gpu_specs import format_gpu_specs
from specgen.generation.ncu_format import format_ncu_profile
from specgen.tasks.base import Task

TEMPLATES = Path(__file__).with_name("templates")
SECTION_ORDER = ("gpu", "task", "best", "feedback", "attempts", "instruction")
BEST_TITLE = ("Best kernel so far (already measured: do not resubmit it unchanged; make every effort to try new "
              "optimizations that beat it)")


@dataclass(frozen=True)
class KernelFeedback:
    source: str
    status: str = "success"
    speedup: Optional[float] = None
    latency_ms: Optional[float] = None
    ref_latency_ms: Optional[float] = None
    ncu_profile: Optional[Mapping[str, Any]] = None  # evaluation.ncu: the kernels of one forward
    error: Optional[str] = None


@dataclass(frozen=True)
class PromptContext:
    best: Optional[KernelFeedback] = None  # global best so far
    feedback: tuple[KernelFeedback, ...] = ()  # fastest kernels of the last iteration, best excluded
    failure: Optional[KernelFeedback] = None  # last iteration had no valid kernel: the main kernel's error
    attempts: str = ""  # table of everything tried so far (control.attempts)


@dataclass(frozen=True)
class Prompt:
    system: str
    sections: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def user(self) -> str:
        return "\n\n".join(text for _, text in self.sections if text)

    def messages(self) -> list[dict[str, Any]]:
        """Chat messages; the user message also lists its sections, which backends with explicit
        cache breakpoints (Anthropic) use and the others drop."""
        sections = [[name, text] for name, text in self.sections if text]
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": self.user, "sections": sections}]

    def to_dict(self) -> dict[str, Any]:
        return {"system": self.system, "sections": dict(self.sections), "user": self.user}


def fmt_num(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def truncate_middle(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return f"{text[:half]}\n...\n{text[-half:]}"


_SIGNAL_LINE = re.compile(
    r"""(?ix)
    (?:^|\s) error [`':] .+ | error: | not\ declared | \bFAILED: | \bninja:\s+build\s+stopped |
    Error\s+building\ extension | undefined\ reference\ to | \.cu\(\d+\):\s*error | CalledProcessError: |
    \bTraceback\ \( | \bFile\ ".*",\ line\ \d+,
    """
)


def summarize_error(text: str, limit: int) -> str:
    """Keep compiler / linker / traceback lines of a long log, plus its head and tail."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    seen: set[str] = set()
    key_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped not in seen and _SIGNAL_LINE.search(stripped):
            seen.add(stripped)
            key_lines.append(line)
    if len(key_lines) < 2:
        return truncate_middle(text, limit)
    budget = max(200, limit // 5)
    block = truncate_middle("\n".join(key_lines), limit - 2 * budget)
    return f"{text[:budget]}\n--- key lines ---\n{block}\n--- end of log ---\n{text[-budget:]}"


class PromptBuilder:
    def __init__(
        self,
        gpu_type: str,
        *,
        max_ncu_chars: int = 7000,
        max_error_chars: int = 8000,
        templates_dir: Path = TEMPLATES,
    ):
        self.gpu_block = format_gpu_specs(gpu_type)
        self.max_ncu_chars = max_ncu_chars
        self.max_error_chars = max_error_chars
        self.system = (templates_dir / "system.txt").read_text(encoding="utf-8").strip()
        self.task_intro = (templates_dir / "task.txt").read_text(encoding="utf-8").strip()
        self.instruction = (templates_dir / "instruction.txt").read_text(encoding="utf-8").strip()

    def build(self, task: Task, ctx: PromptContext = PromptContext()) -> Prompt:
        task_block = f"{self.task_intro}\n\nReference implementation:\n\n```python\n{task.reference_source().rstrip()}\n```"
        best_block = ""
        if ctx.best is not None:
            best_block = self._kernel_block(BEST_TITLE, ctx.best)
        feedback_block = ""
        if ctx.feedback:
            parts = ["Fastest kernels from the previous iteration:"]
            parts += [self._kernel_block(f"Kernel {i}", fb) for i, fb in enumerate(ctx.feedback, start=1)]
            feedback_block = "\n\n".join(parts)
        elif ctx.failure is not None:
            feedback_block = self._failure_block(ctx.failure)
        sections = (
            ("gpu", self.gpu_block),
            ("task", task_block),
            ("best", best_block),
            ("feedback", feedback_block),
            ("attempts", ctx.attempts),
            ("instruction", self.instruction),
        )
        return Prompt(system=self.system, sections=sections)

    def _kernel_block(self, title: str, fb: KernelFeedback) -> str:
        lines = [f"{title}:", "", f"```python\n{fb.source.rstrip()}\n```", ""]
        if fb.speedup is not None:
            measured = f"Measured speedup over the reference: {fb.speedup:.3f}x"
            if fb.latency_ms is not None and fb.ref_latency_ms is not None:
                measured += f" (kernel {fmt_num(fb.latency_ms)} ms, reference {fmt_num(fb.ref_latency_ms)} ms)"
            lines.append(measured)
        profile = format_ncu_profile(fb.ncu_profile, self.max_ncu_chars) if fb.ncu_profile else ""
        if profile:
            lines.append(profile)
        return "\n".join(lines).rstrip()

    def _failure_block(self, fb: KernelFeedback) -> str:
        lines = ["The previous iteration produced no correct kernel."]
        if fb.source.strip():
            lines += ["Its kernel:", "", f"```python\n{fb.source.rstrip()}\n```", ""]
        lines.append(f"Result: {fb.status}")
        if fb.error:
            lines.append(summarize_error(fb.error, self.max_error_chars))
        return "\n".join(lines).rstrip()
