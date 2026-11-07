#!/usr/bin/env python3
"""Probe an OpenAI-compatible endpoint to learn how a model streams its output.

Reports, for one streaming chat completion:

* which delta fields carry text (``content``, ``reasoning_content``, ``reasoning``, ...),
  and whether reasoning is inlined in ``content`` between think tags;
* whether token usage is reported on every chunk (vLLM ``continuous_usage_stats``),
  only at the end, or not at all;
* time to first token and throughput;

and optionally whether the server continues a partial assistant message
(prefill continuation), which spec generation's ``prefill`` mode relies on.

Only depends on ``httpx`` (``pyyaml`` to read ``configs/models.yaml``, ``tiktoken``
for the generic token estimate).

Examples::

    python tools/probe_stream.py --model GLM-5.1-FP8 --endpoint selfhost
    python tools/probe_stream.py --base-url http://127.0.0.1:8000/v1 --model-name glm-5.1 \\
        --api-key-env SPECGEN_API_KEY --report probe_glm.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

DEFAULT_PROMPT = (
    "Write a CUDA kernel that computes C = A @ B for two 1024x1024 float32 matrices "
    "using shared-memory tiling. Explain the tile size you choose, then give the code."
)
CONTINUATION_PREFIX_REASONING = (
    "The user wants a tiled matmul. I will use 32x32 tiles in shared memory and one thread "
    "per output element. That is enough planning; time to write the final code."
)
DEFAULT_THINK_TAGS = ("<think>", "</think>")
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# --------------------------------------------------------------------------- SSE
def iter_sse_data(lines: Iterable[str]) -> Iterator[str]:
    """Yield the payload of each ``data:`` line until ``[DONE]``."""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        yield payload


# --------------------------------------------------------------------------- analysis
def _delta_text_fields(delta: dict[str, Any]) -> dict[str, str]:
    """Text carried by one delta, keyed by field name (content parts are split by type)."""
    out: dict[str, str] = {}
    for key, value in delta.items():
        if key in ("role", "tool_calls", "function_call", "refusal", "audio"):
            continue
        if isinstance(value, str) and value:
            out[key] = out.get(key, "") + value
        elif isinstance(value, list):
            for part in value:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    name = f"{key}[{part.get('type', 'text')}]"
                    out[name] = out.get(name, "") + part["text"]
    return out


def _generic_token_count(text: str) -> tuple[int, str]:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text)), "tiktoken:o200k_base"
    except Exception:
        return max(1, len(text) // 4) if text else 0, "chars/4"


def analyze_chunks(
    chunks: list[dict[str, Any]],
    *,
    arrival_s: Optional[list[float]] = None,
    think_tags: tuple[str, str] = DEFAULT_THINK_TAGS,
) -> dict[str, Any]:
    """Summarize a recorded stream. ``arrival_s[i]`` is the receive time of ``chunks[i]``."""
    field_text: dict[str, str] = {}
    first_text_index: Optional[int] = None
    usage_indices: list[int] = []
    last_usage: Optional[dict[str, Any]] = None
    finish_reason: Optional[str] = None

    for index, chunk in enumerate(chunks):
        usage = chunk.get("usage")
        if isinstance(usage, dict) and usage:
            usage_indices.append(index)
            last_usage = usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            pieces = _delta_text_fields(delta)
            if pieces and first_text_index is None:
                first_text_index = index
            for name, text in pieces.items():
                field_text[name] = field_text.get(name, "") + text
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])

    content = field_text.get("content", "")
    open_tag, close_tag = think_tags
    reasoning_fields = sorted(name for name in field_text if name != "content" and field_text[name])
    if reasoning_fields:
        reasoning_mode = "field"
        reasoning_text = "".join(field_text[name] for name in reasoning_fields)
    elif close_tag in content:
        reasoning_mode = "inline"
        start = content.find(open_tag)
        start = start + len(open_tag) if start >= 0 else 0
        reasoning_text = content[start:content.find(close_tag)]
    else:
        reasoning_mode = "none"
        reasoning_text = ""

    text_chunks = sum(1 for c in chunks for ch in (c.get("choices") or []) if _delta_text_fields(ch.get("delta") or {}))
    if not usage_indices:
        usage_mode = "none"
    elif len(usage_indices) > 1 and usage_indices[0] < len(chunks) - 2:
        usage_mode = "continuous"
    else:
        usage_mode = "final"

    reasoning_est, estimator = _generic_token_count(reasoning_text)
    content_est, _ = _generic_token_count(content if reasoning_mode != "inline" else content.split(close_tag, 1)[-1])
    report: dict[str, Any] = {
        "chunks": len(chunks),
        "text_chunks": text_chunks,
        "text_fields": {name: len(text) for name, text in sorted(field_text.items())},
        "reasoning_mode": reasoning_mode,
        "reasoning_fields": reasoning_fields,
        "inline_starts_with_open_tag": content.lstrip().startswith(open_tag) if reasoning_mode == "inline" else None,
        "usage_mode": usage_mode,
        "usage_chunks": len(usage_indices),
        "last_usage": last_usage,
        "finish_reason": finish_reason,
        "generic_token_estimate": {"estimator": estimator, "reasoning": reasoning_est, "content": content_est},
    }
    completion = (last_usage or {}).get("completion_tokens")
    if isinstance(completion, int) and completion > 0:
        report["generic_vs_server_ratio"] = round((reasoning_est + content_est) / completion, 3)
    if arrival_s and first_text_index is not None:
        report["ttft_s"] = round(arrival_s[first_text_index] - arrival_s[0], 3) if len(arrival_s) > 1 else None
        report["stream_s"] = round(arrival_s[-1] - arrival_s[0], 3)
        tokens = completion if isinstance(completion, int) and completion > 0 else reasoning_est + content_est
        span = arrival_s[-1] - arrival_s[first_text_index]
        report["tokens_per_s"] = round(tokens / span, 1) if span > 0 else None
    return report


def analyze_continuation(message: dict[str, Any], think_tags: tuple[str, str] = DEFAULT_THINK_TAGS) -> dict[str, Any]:
    """Judge a non-streaming reply to a prefill-continuation request."""
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    reasoning = "".join(
        message.get(k) or "" for k in ("reasoning_content", "reasoning") if isinstance(message.get(k), str)
    )
    reentered = bool(reasoning.strip()) or think_tags[0] in content or think_tags[1] in content
    has_code = "```" in content or "__global__" in content
    if reentered:
        verdict = "no: model produced new reasoning (prefix ignored or stripped by the template)"
    elif has_code:
        verdict = "yes: model answered directly with code"
    else:
        verdict = "unclear: no new reasoning but no code either"
    return {
        "verdict": verdict,
        "usable": (not reentered) and has_code,
        "new_reasoning_chars": len(reasoning),
        "content_head": content[:300],
    }


# --------------------------------------------------------------------------- endpoint
def _expand_env(value: str, what: str) -> str:
    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in os.environ:
            raise SystemExit(f"{what}: environment variable {name} is not set")
        return os.environ[name]

    out = _ENV_REF.sub(repl, value)
    if out.startswith("<") and out.endswith(">"):
        raise SystemExit(f"{what} is still a placeholder ({out}); fill it in configs/models.yaml")
    return out


def _mode_params(profile: dict[str, Any], endpoint: dict[str, Any], mode: str) -> dict[str, Any]:
    return {**((profile.get("modes") or {}).get(mode) or {}), **((endpoint.get("modes") or {}).get(mode) or {})}


def resolve_endpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Endpoint settings; ``main`` / ``spec`` hold the profile's request parameters for each mode."""
    extra_main = json.loads(args.extra_body) if args.extra_body else {}
    extra_spec = json.loads(args.continuation_extra_body) if args.continuation_extra_body else {}
    if args.base_url:
        return {
            "base_url": args.base_url,
            "model": args.model_name or "",
            "api_key": os.environ.get(args.api_key_env, "") if args.api_key_env else "",
            "think_tags": DEFAULT_THINK_TAGS,
            "kind": args.endpoint,
            "main": extra_main,
            "spec": extra_spec,
        }
    import yaml

    profiles = yaml.safe_load(Path(args.models_file).read_text(encoding="utf-8")) or {}
    if args.model not in profiles:
        raise SystemExit(f"model {args.model!r} not in {args.models_file} (have {sorted(profiles)})")
    profile = profiles[args.model]
    endpoint = (profile.get("endpoints") or {}).get(args.endpoint)
    if not endpoint:
        raise SystemExit(f"model {args.model!r} has no {args.endpoint!r} endpoint")
    key_env = endpoint.get("api_key_env")
    return {
        "base_url": _expand_env(str(endpoint["base_url"]), "base_url"),
        "model": args.model_name or _expand_env(str(endpoint.get("model", "")), "model"),
        "api_key": os.environ.get(key_env, "") if key_env else "",
        "think_tags": tuple(profile.get("think_tags") or DEFAULT_THINK_TAGS),
        "kind": args.endpoint,
        "main": {**_mode_params(profile, endpoint, "main"), **extra_main},
        "spec": {**_mode_params(profile, endpoint, "spec"), **extra_spec},
    }


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def run_stream(client: Any, ep: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[float], bool]:
    """One streaming request; retries without ``continuous_usage_stats`` if the server rejects it."""
    messages = [{"role": "user", "content": args.prompt}]
    for continuous in (True, False):
        payload: dict[str, Any] = {
            "model": ep["model"],
            "messages": messages,
            "stream": True,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream_options": {"include_usage": True, **({"continuous_usage_stats": True} if continuous else {})},
            **ep["main"],
        }
        chunks: list[dict[str, Any]] = []
        arrival: list[float] = []
        with client.stream("POST", f"{ep['base_url'].rstrip('/')}/chat/completions", json=payload, headers=_headers(ep["api_key"])) as resp:
            if resp.status_code == 400 and continuous:
                resp.read()
                continue
            resp.raise_for_status()
            for data in iter_sse_data(resp.iter_lines()):
                arrival.append(time.perf_counter())
                chunks.append(json.loads(data))
        return chunks, arrival, continuous
    raise RuntimeError("unreachable")


def run_continuation(client: Any, ep: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    open_tag, close_tag = ep["think_tags"]
    prefix = f"{open_tag}\n{CONTINUATION_PREFIX_REASONING}\n{close_tag}\n"
    assistant: dict[str, Any] = {"role": "assistant", "content": prefix}
    payload: dict[str, Any] = {
        "model": ep["model"],
        "max_tokens": min(args.max_tokens, 2048),
        "temperature": args.temperature,
        **ep["spec"],
    }
    if args.continuation_style == "deepseek":
        assistant["prefix"] = True
    else:
        payload["continue_final_message"] = True
        payload["add_generation_prompt"] = False
    payload["messages"] = [{"role": "user", "content": args.prompt}, assistant]
    base_url = (args.continuation_base_url or ep["base_url"]).rstrip("/")
    resp = client.post(f"{base_url}/chat/completions", json=payload, headers=_headers(ep["api_key"]))
    if resp.status_code >= 400:
        return {"verdict": f"no: HTTP {resp.status_code}", "usable": False, "error": resp.text[:500]}
    choices = resp.json().get("choices") or [{}]
    return {"style": args.continuation_style, **analyze_continuation(choices[0].get("message") or {}, ep["think_tags"])}


def suggest_profile(stream_report: dict[str, Any], continuation: Optional[dict[str, Any]]) -> dict[str, Any]:
    suggestion: dict[str, Any] = {"reasoning": stream_report["reasoning_mode"]}
    if stream_report["reasoning_fields"]:
        suggestion["reasoning_field"] = stream_report["reasoning_fields"][0]
    if stream_report["reasoning_mode"] == "inline":
        suggestion["inline_starts_in_reasoning"] = not stream_report["inline_starts_with_open_tag"]
    suggestion["usage_stats"] = stream_report["usage_mode"]
    if continuation is not None:
        suggestion["prefix_continuation"] = bool(continuation.get("usable"))
    return suggestion


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="GLM-5.1-FP8", help="profile name in --models-file")
    p.add_argument("--endpoint", default="selfhost", choices=["selfhost", "api"])
    p.add_argument("--models-file", default=str(Path(__file__).resolve().parent.parent / "configs" / "models.yaml"))
    p.add_argument("--base-url", default="", help="bypass --models-file, e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--model-name", default="", help="served model name (overrides the profile)")
    p.add_argument("--api-key-env", default="", help="environment variable holding the API key")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--extra-body", default="", help="JSON merged into the streaming request (after the main mode)")
    p.add_argument("--no-continuation", action="store_true", help="skip the prefill-continuation test")
    p.add_argument("--continuation-style", default="vllm", choices=["vllm", "deepseek"],
                   help="vllm: continue_final_message; deepseek: assistant message with prefix=true")
    p.add_argument("--continuation-base-url", default="", help="e.g. https://api.deepseek.com/beta for DeepSeek")
    p.add_argument("--continuation-extra-body", default="", help="JSON merged into the continuation request")
    p.add_argument("--dump", default="", help="write raw chunks (JSON lines) here")
    p.add_argument("--report", default="", help="write the JSON report here")
    args = p.parse_args(argv)

    import httpx

    ep = resolve_endpoint(args)
    with httpx.Client(timeout=args.timeout) as client:
        chunks, arrival, continuous_requested = run_stream(client, ep, args)
        stream_report = analyze_chunks(chunks, arrival_s=arrival, think_tags=ep["think_tags"])
        stream_report["continuous_usage_stats_accepted"] = continuous_requested
        continuation = None if args.no_continuation else run_continuation(client, ep, args)

    if args.dump:
        Path(args.dump).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    report = {
        "endpoint": {"base_url": ep["base_url"], "model": ep["model"], "kind": ep["kind"]},
        "stream": stream_report,
        "continuation": continuation,
        "suggested_profile": suggest_profile(stream_report, continuation),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
