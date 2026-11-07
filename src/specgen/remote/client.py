"""Client side of remote evaluation workers."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

import httpx

from specgen.evaluation.executor import GpuSlot
from specgen.evaluation.jobs import EvalJob

WORK_KEY_PARTS = 5  # <run_id>/<task>/iter_NNN/kernels/<kernel_id>


class WorkerError(RuntimeError):
    pass


def _headers(token: Optional[str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch_worker_info(url: str, token: Optional[str] = None, timeout_s: float = 30.0) -> dict[str, Any]:
    try:
        response = httpx.get(f"{url.rstrip('/')}/info", headers=_headers(token), timeout=timeout_s)
    except httpx.HTTPError as e:
        raise WorkerError(f"worker {url} is unreachable: {e}") from None
    if response.status_code != 200:
        raise WorkerError(f"worker {url} answered {response.status_code}: {response.text[:500]}")
    return response.json()


class RemoteEvaluator:
    """Sends jobs to one worker; cancelling ``run`` kills the job there."""

    def __init__(self, url: str, *, token: Optional[str] = None, timeout_s: float = 600.0,
                 client: Optional[httpx.AsyncClient] = None):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s + 120.0, connect=30.0))

    async def run(self, job: EvalJob, slot: GpuSlot) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        payload = {
            "id": job_id,
            "job": job.to_dict(),
            "slot": {"gpu": slot.gpu, "role": slot.role, "index": slot.index},
            "files": {"task": Path(job.task_path).read_text(encoding="utf-8"),
                      "kernel": Path(job.kernel_path).read_text(encoding="utf-8")},
            "work_key": "/".join(Path(job.work_dir).parts[-WORK_KEY_PARTS:]),
        }
        try:
            response = await self._client.post(f"{self.url}/jobs", json=payload, headers=_headers(self.token))
        except asyncio.CancelledError:
            await asyncio.shield(self._cancel(job_id))
            raise
        except httpx.HTTPError as e:
            return {"ok": False, "status": "worker_error", "error": f"{type(e).__name__}: {e}", "worker": self.url}
        if response.status_code != 200:
            return {"ok": False, "status": "worker_error", "worker": self.url,
                    "error": f"HTTP {response.status_code}: {response.text[:2000]}"}
        result = response.json()
        result.update(worker=self.url, slot=slot.label)
        return result

    async def _cancel(self, job_id: str) -> None:
        try:
            await self._client.post(f"{self.url}/jobs/{job_id}/cancel", headers=_headers(self.token), timeout=10.0)
        except httpx.HTTPError:
            pass

    async def aclose(self) -> None:
        await self._client.aclose()


class EvaluatorRouter:
    """Runs a job with the evaluator of its slot's host (local when the slot has none)."""

    def __init__(self, local: Any, remotes: Mapping[str, RemoteEvaluator]):
        self.local = local
        self.remotes = dict(remotes)

    async def run(self, job: EvalJob, slot: GpuSlot) -> dict[str, Any]:
        if slot.host is None:
            return await self.local.run(job, slot)
        return await self.remotes[slot.host].run(job, slot)

    async def aclose(self) -> None:
        for remote in self.remotes.values():
            await remote.aclose()
