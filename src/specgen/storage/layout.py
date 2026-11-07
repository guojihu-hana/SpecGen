"""Run directory layout.

``<output_root>/<run_id>/<task>/iter_<NNN>/`` holds ``prompt.json`` and the main
generation's ``main/{reasoning,content}.txt``; each kernel gets
``kernels/<kernel_id>/`` with ``kernel.py``, job results and build artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from specgen.tasks.base import Task


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_records(path: Path) -> list[dict[str, Any]]:
    """The rows of a JSON-lines record file (a line cut short by a killed run is skipped)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else ():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


class RunLayout:
    def __init__(self, output_root: Path, run_id: str):
        self.run_dir = Path(output_root).resolve() / run_id

    def task_dir(self, task: Task) -> Path:
        return self.run_dir / task.dir_name

    def iteration_dir(self, task: Task, iteration: int) -> Path:
        return self.task_dir(task) / f"iter_{iteration:03d}"

    def kernel_dir(self, task: Task, iteration: int, kernel_id: str) -> Path:
        return self.iteration_dir(task, iteration) / "kernels" / kernel_id


class StreamFiles:
    """Generation sink that writes reasoning and content to two files as they stream."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self._reasoning = (directory / "reasoning.txt").open("w", encoding="utf-8")
        self._content = (directory / "content.txt").open("w", encoding="utf-8")

    def on_reasoning(self, text: str, total_tokens: int) -> None:
        self._reasoning.write(text)

    def on_content(self, text: str, total_tokens: int) -> None:
        self._content.write(text)

    def close(self) -> None:
        self._reasoning.close()
        self._content.close()
