"""Server-sent events over httpx, with retries for requests that have not streamed yet."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, AsyncIterator, Mapping, Optional

import httpx

RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504, 529)


class ProviderError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:2000]}")
        self.status = status


async def iter_sse(response: httpx.Response) -> AsyncIterator[tuple[Optional[str], str]]:
    """``(event, data)`` pairs of an event stream (multi-line data joined with newlines)."""
    event: Optional[str] = None
    data: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = None, []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield event, "\n".join(data)


@contextlib.asynccontextmanager
async def open_stream(client: httpx.AsyncClient, url: str, *, json_body: Mapping[str, Any],
                      headers: Mapping[str, str], max_retries: int) -> AsyncIterator[httpx.Response]:
    """POST and yield the streaming response; retryable statuses back off and retry."""
    attempt = 0
    while True:
        request = client.build_request("POST", url, json=dict(json_body), headers=dict(headers))
        response = await client.send(request, stream=True)
        if response.status_code < 400:
            try:
                yield response
            finally:
                await response.aclose()
            return
        body = (await response.aread()).decode("utf-8", errors="replace")
        await response.aclose()
        if response.status_code not in RETRY_STATUS or attempt >= max_retries:
            raise ProviderError(response.status_code, body)
        await asyncio.sleep(min(30.0, 2.0 ** attempt))
        attempt += 1


def loads(data: str) -> Any:
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None
