"""Evaluation worker: runs jobs sent by a remote ``specgen run`` on this machine's GPUs.

Protocol (JSON over HTTP; ``Authorization: Bearer <token>`` when the worker has one):

- ``GET /info`` -> ``{"gpus", "cuda_arch", "gpu_type", "version"}``
- ``POST /jobs`` with ``{"id", "job", "slot", "files": {"task", "kernel"}, "work_key"}`` runs the
  job and answers with its result (the task and kernel sources are written under the
  worker's root; the work directory, and so the kernel's build, persists per work key)
- ``POST /jobs/<id>/cancel`` kills a running job

Kernels are arbitrary code: run workers only on trusted networks, with a token.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import specgen
from specgen.evaluation.executor import GpuSlot, LocalEvaluator
from specgen.evaluation.jobs import EvalJob

log = logging.getLogger(__name__)
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.\-/]+$")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class WorkerState:
    def __init__(self, root: Path, *, gpus: list[str], cuda_arch: dict[str, str], gpu_type: Optional[str],
                 timeout_s: float = 600.0, build_jobs: Optional[int] = None, token: Optional[str] = None):
        self.root = Path(root).resolve()
        self.gpus = list(gpus)
        self.cuda_arch = dict(cuda_arch)
        self.gpu_type = gpu_type
        self.token = token
        self.evaluator = LocalEvaluator(timeout_s=timeout_s, cuda_arch=cuda_arch, build_jobs=build_jobs,
                                        build_arch=next(iter(cuda_arch.values()), None))
        self._running: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Task]] = {}
        self._lock = threading.Lock()

    def info(self) -> dict[str, Any]:
        return {"gpus": self.gpus, "cuda_arch": self.cuda_arch, "gpu_type": self.gpu_type,
                "version": specgen.__version__, "hostname": socket.gethostname()}

    def run_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = str(payload["id"])
        key = str(payload["work_key"]).strip("/")
        if not _SAFE_KEY.match(key) or ".." in key.split("/"):
            raise ValueError(f"bad work key {key!r}")
        slot_raw = payload["slot"]
        gpu = slot_raw.get("gpu")
        if gpu is not None and gpu not in self.gpus:
            raise ValueError(f"GPU {gpu!r} is not served by this worker (serving {self.gpus})")
        task_src, kernel_src = payload["files"]["task"], payload["files"]["kernel"]
        task_path = self.root / "tasks" / f"{_sha(task_src)}.py"
        if not task_path.is_file():
            _write_atomic(task_path, task_src)
        work = self.root / "work" / key
        kernel_path = work / "kernel.py"
        if not kernel_path.is_file() or kernel_path.read_text(encoding="utf-8") != kernel_src:
            _write_atomic(kernel_path, kernel_src)
        raw = payload["job"]
        params = dict(raw.get("params") or {})
        if params.get("ref_cache"):
            params["ref_cache"] = str(self.root / "ref_cache" / _sha(task_src))
        job = EvalJob(kind=raw["kind"], task_kind=raw["task_kind"], task_path=str(task_path),
                      kernel_path=str(kernel_path), work_dir=str(work), params=params)
        slot = GpuSlot(gpu, str(slot_raw.get("role", "validator")), int(slot_raw.get("index", 0)))

        loop = asyncio.new_event_loop()
        task = loop.create_task(self.evaluator.run(job, slot))
        with self._lock:
            self._running[job_id] = (loop, task)
        try:
            return loop.run_until_complete(task)
        except asyncio.CancelledError:
            return {"ok": False, "status": "cancelled", "error": "cancelled by the client"}
        finally:
            with self._lock:
                self._running.pop(job_id, None)
            loop.close()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            entry = self._running.get(job_id)
        if entry is None:
            return False
        loop, task = entry
        loop.call_soon_threadsafe(task.cancel)
        return True


class _Handler(BaseHTTPRequestHandler):
    server_version = "specgen-worker"
    protocol_version = "HTTP/1.1"

    @property
    def state(self) -> WorkerState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("%s %s", self.address_string(), fmt % args)

    def _reply(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token = self.state.token
        if not token:
            return True
        if self.headers.get("Authorization") == f"Bearer {token}":
            return True
        self._reply(401, {"error": "unauthorized"})
        return False

    def _body(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path == "/info":
            self._reply(200, self.state.info())
        else:
            self._reply(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        try:
            payload = self._body()
        except (ValueError, json.JSONDecodeError) as e:
            self._reply(400, {"error": f"bad JSON: {e}"})
            return
        parts = self.path.strip("/").split("/")
        if parts == ["jobs"]:
            try:
                result = self.state.run_job(payload)
            except (KeyError, TypeError, ValueError) as e:
                self._reply(400, {"error": f"{type(e).__name__}: {e}"})
                return
            self._reply(200, result)
        elif len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
            self._reply(200, {"cancelled": self.state.cancel(parts[1])})
        else:
            self._reply(404, {"error": f"no route {self.path}"})


def make_server(state: WorkerState, host: str = "0.0.0.0", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.daemon_threads = True
    server.state = state  # type: ignore[attr-defined]
    return server
