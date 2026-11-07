"""Evaluation jobs: the request a GPU child process executes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

JOB_KINDS = ("build", "validate", "benchmark", "ncu")


@dataclass
class EvalJob:
    kind: str  # one of JOB_KINDS
    task_kind: str  # selects the evaluator, e.g. "kernelbench"
    task_path: str
    kernel_path: str
    work_dir: str  # per-kernel directory: request/result/log files and build artifacts
    params: dict[str, Any] = field(default_factory=dict)
    name: str = ""  # stem of the request/result/log files; the kind by default (a repeated job needs its own)

    def __post_init__(self) -> None:
        if self.kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {self.kind!r}")

    @property
    def stem(self) -> str:
        return self.name or self.kind

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalJob":
        return cls(**data)
