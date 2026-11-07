"""Tasks: what a kernel must reproduce, and which evaluator checks it."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


class Task(Protocol):
    task_id: str  # stable id, e.g. "level1/19_ReLU"
    kind: str  # selects the evaluator implementation in the evaluation child
    path: Path

    @property
    def dir_name(self) -> str: ...

    def reference_source(self) -> str: ...


@dataclass(frozen=True)
class KernelBenchTask:
    """A KernelBench file defining ``Model``, ``get_inputs`` and ``get_init_inputs``."""

    task_id: str
    path: Path
    kind: str = "kernelbench"

    @classmethod
    def from_path(cls, path: Path) -> "KernelBenchTask":
        path = Path(path).resolve()
        return cls(task_id=f"{path.parent.name}/{path.stem}", path=path)

    @property
    def dir_name(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "__", self.task_id)

    def reference_source(self) -> str:
        return self.path.read_text(encoding="utf-8")


def load_tasks(paths: Iterable[Path]) -> list[KernelBenchTask]:
    tasks = [KernelBenchTask.from_path(p) for p in paths]
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise ValueError(f"duplicate task {task.task_id}")
        seen.add(task.task_id)
    return tasks
