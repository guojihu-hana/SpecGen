"""Run-wide services shared by every task's iteration loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from specgen.config import SpecGenConfig
from specgen.control.budget import SpecBudget
from specgen.control.triggers import TriggerConfig
from specgen.generation.generator import Generator
from specgen.generation.prompts import PromptBuilder
from specgen.resources.pool import ResourcePool
from specgen.storage.layout import RunLayout
from specgen.telemetry.recorder import PerfRecorder


@dataclass
class RunContext:
    cfg: SpecGenConfig
    generator: Generator
    pool: ResourcePool
    evaluator: Any  # async run(job, slot) -> dict, e.g. LocalEvaluator
    prompts: PromptBuilder
    recorder: PerfRecorder
    layout: RunLayout
    budget: SpecBudget
    triggers: Optional[TriggerConfig] = None  # None: speculative generation is off
    ncu_enabled: bool = True  # cleared for the whole run once ncu reports it cannot profile
