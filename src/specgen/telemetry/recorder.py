"""PerfRecorder: append-only event log, per-kernel records and running statistics."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass
class Ewma:
    alpha: float = 0.3
    value: Optional[float] = None
    count: int = 0

    def add(self, x: float) -> None:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        self.count += 1


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str, sort_keys=True)


class PerfRecorder:
    """Writes ``events.jsonl``, ``kernels.jsonl`` and ``iterations.jsonl`` under the run directory.

    Kernel records are kept in memory while the kernel moves through the pipeline
    and appended to ``kernels.jsonl`` when finished.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._events = (self.run_dir / "events.jsonl").open("a", encoding="utf-8")
        self.kernels: dict[str, dict[str, Any]] = {}
        self.exec_s: dict[str, Ewma] = defaultdict(Ewma)
        self.wait_s: dict[str, Ewma] = defaultdict(Ewma)
        self.outcomes: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # kind -> [ok, total]

    @staticmethod
    def key(task_id: str, kernel_id: str) -> str:
        return f"{task_id}#{kernel_id}"

    def event(self, type: str, **fields: Any) -> None:
        line = _dumps({"t": round(time.time(), 6), "type": type, **fields})
        with self._lock:
            self._events.write(line + "\n")
            self._events.flush()

    def kernel(self, task_id: str, kernel_id: str, **fields: Any) -> dict[str, Any]:
        record = self.kernels.setdefault(self.key(task_id, kernel_id), {"task_id": task_id, "kernel_id": kernel_id})
        record.update(fields)
        return record

    def observe_job(self, kind: str, result: Mapping[str, Any]) -> None:
        if isinstance(result.get("exec_s"), (int, float)):
            self.exec_s[kind].add(float(result["exec_s"]))
        if isinstance(result.get("wait_s"), (int, float)):
            self.wait_s[kind].add(float(result["wait_s"]))
        counts = self.outcomes[kind]
        counts[0] += 1 if result.get("ok") else 0
        counts[1] += 1

    def finish_kernel(self, task_id: str, kernel_id: str) -> None:
        record = self.kernels.pop(self.key(task_id, kernel_id), None)
        if record is not None:
            self._append("kernels.jsonl", record)
            self.event("kernel_finished", task=task_id, kernel=kernel_id, status=record.get("status"))

    def finish_iteration(self, record: Mapping[str, Any]) -> None:
        self._append("iterations.jsonl", dict(record))
        self.event("iteration_finished", **{k: record.get(k) for k in ("task", "iteration", "end_reason", "selected")})

    def stats(self) -> dict[str, dict[str, Any]]:
        kinds = set(self.exec_s) | set(self.outcomes)
        return {
            kind: {
                "exec_s": self.exec_s[kind].value,
                "wait_s": self.wait_s[kind].value,
                "count": self.outcomes[kind][1],
                "pass_rate": self.outcomes[kind][0] / self.outcomes[kind][1] if self.outcomes[kind][1] else None,
            }
            for kind in sorted(kinds)
        }

    def _append(self, name: str, record: Mapping[str, Any]) -> None:
        with self._lock, (self.run_dir / name).open("a", encoding="utf-8") as f:
            f.write(_dumps(record) + "\n")

    def close(self) -> None:
        with self._lock:
            if not self._events.closed:
                self._events.close()
