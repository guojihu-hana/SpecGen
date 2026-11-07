"""File-queue backend: requests and responses are files in a shared directory.

For models served by an external process (a batch system, another cluster, a person):

- the backend writes ``requests/<id>.json`` (model, messages, parameters);
- the responder appends events to ``responses/<id>.jsonl``, one JSON object per line:
  ``{"type": "reasoning" | "content", "text": ...}``, ``{"type": "usage", ...}`` and
  finally ``{"type": "end", "finish_reason": ...}`` (or ``{"type": "error", "message": ...}``);
- an abandoned request gets a ``cancelled/<id>`` marker.

Streaming responses keep triggers working; a responder may also write the whole
file at once.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from specgen.llm.backend import ChatParams
from specgen.llm.events import ContentDelta, GenEvent, ReasoningDelta, StreamEnd, UsageUpdate
from specgen.llm.profiles import ModelProfile, ResolvedEndpoint


def queue_dir(endpoint: ResolvedEndpoint) -> Path:
    raw = endpoint.options.get("dir") or endpoint.base_url
    if raw.startswith("file://"):
        raw = raw[len("file://"):]
    return Path(raw)


class FileQueueBackend:
    def __init__(self, endpoint: ResolvedEndpoint, profile: ModelProfile, *, timeout_s: float = 3600.0,
                 poll_s: Optional[float] = None, **_: Any):
        self._endpoint, self._profile = endpoint, profile
        self.root = queue_dir(endpoint)
        self.timeout_s = timeout_s
        self.poll_s = float(endpoint.options.get("poll_s", 0.2) if poll_s is None else poll_s)
        self.prefill_style = endpoint.prefill_style
        for sub in ("requests", "responses", "cancelled"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def _write_request(self, request_id: str, messages: list[dict[str, Any]], params: ChatParams) -> None:
        payload = {
            "id": request_id, "model": self._endpoint.model, "created": time.time(),
            "messages": [{k: v for k, v in m.items() if k in ("role", "content")} for m in messages],
            "params": {"temperature": params.temperature, "top_p": params.top_p, "max_tokens": params.max_tokens,
                       "seed": params.seed, "mode": params.mode, "prefill": params.prefill,
                       "extra": {**self._profile.mode_params(params.mode, self._endpoint), **params.extra_body}},
        }
        path = self.root / "requests" / f"{request_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    async def stream_chat(self, messages: list[dict[str, Any]], params: ChatParams) -> AsyncIterator[GenEvent]:
        request_id = uuid.uuid4().hex
        self._write_request(request_id, messages, params)
        response = self.root / "responses" / f"{request_id}.jsonl"
        deadline = time.monotonic() + self.timeout_s
        offset, partial, finished = 0, "", False
        try:
            while not finished:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no complete response for request {request_id} in {self.timeout_s:g}s")
                if response.is_file():
                    with response.open("r", encoding="utf-8") as f:
                        f.seek(offset)
                        chunk = f.read()
                        offset = f.tell()
                    lines = (partial + chunk).split("\n")
                    partial = lines.pop()
                    for line in lines:
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        kind = event.get("type")
                        if kind == "reasoning" and event.get("text"):
                            yield ReasoningDelta(event["text"])
                        elif kind == "content" and event.get("text"):
                            yield ContentDelta(event["text"])
                        elif kind == "usage":
                            yield UsageUpdate(prompt_tokens=event.get("prompt_tokens"),
                                              completion_tokens=event.get("completion_tokens"),
                                              reasoning_tokens=event.get("reasoning_tokens"))
                        elif kind == "error":
                            raise RuntimeError(f"file queue responder error: {event.get('message')}")
                        elif kind == "end":
                            yield StreamEnd(event.get("finish_reason") or "stop")
                            finished = True
                            break
                if not finished:
                    await asyncio.sleep(self.poll_s)
        finally:
            if not finished:
                (self.root / "cancelled" / request_id).touch()

    async def aclose(self) -> None:
        pass
