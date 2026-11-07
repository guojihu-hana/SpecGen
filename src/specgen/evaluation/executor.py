"""Run evaluation jobs in isolated child processes pinned to one GPU slot."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import specgen
from specgen.evaluation.jobs import EvalJob

_PACKAGE_ROOT = str(Path(specgen.__file__).resolve().parent.parent)


@dataclass(frozen=True)
class GpuSlot:
    gpu: Optional[str]  # CUDA device id as the driver sees it; None runs on CPU
    role: str  # "validator", "profiler", "shared" or "builder"
    index: int = 0  # slot number on this device
    host: Optional[str] = None  # remote worker URL; None: this machine

    @property
    def label(self) -> str:
        where = "cpu" if self.gpu is None else self.gpu
        return f"{self.role}:{where}#{self.index}" + (f"@{self.host}" if self.host else "")


def _tail(path: Path, limit: int = 6000) -> str:
    try:
        data = path.read_bytes()[-limit:]
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _descendants(pid: int) -> list[int]:
    """All live descendants of ``pid`` (from /proc; empty where /proc is unavailable)."""
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8", errors="replace") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    out, stack = [], [pid]
    while stack:
        for child in children.get(stack.pop(), ()):
            out.append(child)
            stack.append(child)
    return out


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the child's process group and every descendant: tools such as ncu start the
    profiled program in a session of its own, which a group kill alone would orphan."""
    descendants = _descendants(proc.pid)  # before the kill re-parents them to init
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    for pid in descendants:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


class LocalEvaluator:
    """Each job runs in a fresh process: a bad kernel cannot poison another job's CUDA
    context, and a hung or cancelled job is killed with its whole process group.

    ``cuda_arch`` maps a GPU id to its ``TORCH_CUDA_ARCH_LIST`` value, so kernels are
    built for exactly the device they run on (an inherited multi-arch list slows every
    build and rejects valid architecture-specific code); GPUs without an entry keep
    the inherited environment.  ``build_arch`` is the list CPU builder slots compile for
    (it must equal the validators' list, or validation rebuilds).  ``build_jobs`` becomes
    ``MAX_JOBS`` for the ninja build, so concurrent builds share the CPUs instead of each
    starting one job per core.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 600.0,
        python: str = sys.executable,
        extra_env: Optional[Mapping[str, str]] = None,
        cuda_arch: Optional[Mapping[str, str]] = None,
        build_arch: Optional[str] = None,
        build_jobs: Optional[int] = None,
        fast_includes: bool = True,
        pch: bool = True,
    ):
        self.timeout_s = timeout_s
        self.python = python
        self.extra_env = dict(extra_env or {})
        self.cuda_arch = dict(cuda_arch or {})
        self.build_arch = build_arch
        self.build_jobs = build_jobs
        self.fast_includes = fast_includes
        self.pch = pch

    def _env(self, job: EvalJob, slot: GpuSlot) -> dict[str, str]:
        env = {**os.environ, **self.extra_env}
        env["CUDA_VISIBLE_DEVICES"] = "" if slot.gpu is None else slot.gpu
        # NVIDIA's PyTorch containers set it to 1, which runs FP32 reference matmuls in TF32.
        env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
        if slot.gpu in self.cuda_arch:
            env["TORCH_CUDA_ARCH_LIST"] = self.cuda_arch[slot.gpu]
        elif slot.role == "builder" and self.build_arch:
            env["TORCH_CUDA_ARCH_LIST"] = self.build_arch
        if self.build_jobs is not None:
            env["MAX_JOBS"] = str(self.build_jobs)
        env["SPECGEN_FAST_INCLUDES"] = "1" if self.fast_includes else "0"
        env["SPECGEN_PCH"] = "1" if self.pch else "0"
        env["TORCH_EXTENSIONS_DIR"] = str(Path(job.work_dir) / "torch_extensions")
        env["PYTHONPATH"] = os.pathsep.join(p for p in (_PACKAGE_ROOT, env.get("PYTHONPATH", "")) if p)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    async def run(self, job: EvalJob, slot: GpuSlot) -> dict[str, Any]:
        work = Path(job.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        request, result_path, log = (work / f"{job.stem}.{suffix}" for suffix in ("request.json", "result.json", "log"))
        result_path.unlink(missing_ok=True)
        request.write_text(json.dumps(job.to_dict(), ensure_ascii=False), encoding="utf-8")

        t0 = time.perf_counter()
        timed_out = False
        with open(log, "wb") as log_file:
            proc = await asyncio.create_subprocess_exec(
                self.python, "-m", "specgen.evaluation.child", "--request", str(request), "--result", str(result_path),
                stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
                env=self._env(job, slot), start_new_session=True,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=self.timeout_s)
            except asyncio.TimeoutError:
                timed_out = True
                _kill_tree(proc)
                await proc.wait()
            except asyncio.CancelledError:
                _kill_tree(proc)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=10)
                raise
        exec_s = time.perf_counter() - t0

        if timed_out:
            result: dict[str, Any] = {"ok": False, "status": "timeout", "error": f"killed after {self.timeout_s:g}s"}
        elif result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {"ok": False, "status": "runtime_error",
                      "error": f"evaluation child exited with code {proc.returncode} and no result"}
        if not result.get("ok"):
            result.setdefault("log_tail", _tail(log))
        result.update(exec_s=round(exec_s, 3), gpu=slot.gpu, slot=slot.label, returncode=proc.returncode)
        return result
