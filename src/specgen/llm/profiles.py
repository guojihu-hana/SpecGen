"""Model profiles: how each model streams its output and where to reach it.

Endpoint values may reference environment variables as ``${VAR}``; they are
expanded when the endpoint is resolved, so unused endpoints may stay unset.
``<...>`` values are placeholders and refuse to resolve.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

REASONING_MODES = ("auto", "field", "inline")
# vllm / sglang / api / together speak the OpenAI chat API; the others have their own backend.
BACKENDS = ("vllm", "sglang", "api", "together", "anthropic", "gemini", "file_queue")
DEFAULT_URL_BACKENDS = ("anthropic", "gemini", "file_queue")
GENERATION_MODES = ("main", "spec")
# How an endpoint continues a partial assistant message (spec "prefill" prompts):
#   deepseek - last assistant message carries prefix=true (DeepSeek's beta base URL)
#   vllm     - continue_final_message=true, add_generation_prompt=false (check with
#              tools/probe_stream.py that the chat template keeps the final message's reasoning)
#   native   - the API continues a trailing assistant message by itself (Anthropic does)
PREFILL_STYLES = ("deepseek", "vllm", "native")
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.95
    spec_temperature: float = 0.9
    # the j-th spec of one trigger samples at spec_temperature + j * step, capped at max
    spec_temperature_step: float = 0.1
    spec_temperature_max: float = 1.5
    max_tokens: int = 32768
    spec_max_tokens: int = 16384  # a spec only writes the final module

    def spec_temperature_for(self, index: int) -> float:
        return round(min(self.spec_temperature_max, self.spec_temperature + index * self.spec_temperature_step), 4)


@dataclass(frozen=True)
class ResolvedEndpoint:
    name: str
    backend: str
    base_url: str
    model: str
    api_key: str
    extra_body: Mapping[str, Any]
    modes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # mode -> request params
    prefill_style: Optional[str] = None
    prefill_base_url: Optional[str] = None  # None: base_url
    options: Mapping[str, Any] = field(default_factory=dict)  # backend-specific settings


@dataclass(frozen=True)
class Endpoint:
    name: str
    backend: str = "vllm"
    base_url: str = ""
    model: str = ""
    api_key_env: Optional[str] = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    modes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    prefill_style: Optional[str] = None
    prefill_base_url: Optional[str] = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def resolve(self, environ: Optional[Mapping[str, str]] = None) -> ResolvedEndpoint:
        env = os.environ if environ is None else environ
        api_key = env.get(self.api_key_env, "") if self.api_key_env else ""
        prefill_url = None
        if self.prefill_base_url:
            prefill_url = _expand(self.prefill_base_url, f"endpoint {self.name!r} prefill base_url", env)
        base_url = ""  # anthropic / gemini / file_queue (options.dir) have defaults
        if self.base_url or self.backend not in DEFAULT_URL_BACKENDS:
            base_url = _expand(self.base_url, f"endpoint {self.name!r} base_url", env)
        return ResolvedEndpoint(
            name=self.name,
            backend=self.backend,
            base_url=base_url,
            model=_expand(self.model, f"endpoint {self.name!r} model", env),
            api_key=api_key or "EMPTY",
            extra_body=dict(self.extra_body),
            modes={mode: dict(params) for mode, params in self.modes.items()},
            prefill_style=self.prefill_style,
            prefill_base_url=prefill_url,
            options=dict(self.options),
        )


@dataclass(frozen=True)
class ModelProfile:
    name: str
    tokenizer: str = "generic"
    reasoning: str = "auto"
    reasoning_field: Optional[str] = None
    think_tags: tuple[str, str] = ("<think>", "</think>")
    inline_starts_in_reasoning: bool = False
    sampling: Sampling = Sampling()
    modes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # defaults for every endpoint
    endpoints: Mapping[str, Endpoint] = field(default_factory=dict)

    def endpoint(self, name: str) -> Endpoint:
        try:
            return self.endpoints[name]
        except KeyError:
            raise ProfileError(f"model {self.name!r} has no {name!r} endpoint (have {sorted(self.endpoints)})") from None

    def mode_params(self, mode: str, endpoint: ResolvedEndpoint) -> dict[str, Any]:
        """Request parameters of a generation mode: profile defaults, overridden by the endpoint."""
        return {**self.modes.get(mode, {}), **endpoint.modes.get(mode, {})}


def _expand(value: str, what: str, env: Mapping[str, str]) -> str:
    missing = [name for name in _ENV_REF.findall(value) if not env.get(name)]
    if missing:
        raise ProfileError(f"{what}: set environment variable(s) {', '.join(missing)}")
    out = _ENV_REF.sub(lambda m: env[m.group(1)], value).strip()
    if not out or (out.startswith("<") and out.endswith(">")):
        raise ProfileError(f"{what} is a placeholder ({value!r}); fill it in the models file")
    return out


def default_models_file() -> Path:
    """``$SPECGEN_MODELS_FILE``, else ``./configs/models.yaml``, else the repository copy."""
    env = os.environ.get("SPECGEN_MODELS_FILE")
    if env:
        return Path(env)
    local = Path.cwd() / "configs" / "models.yaml"
    if local.is_file():
        return local
    return Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


def _parse_modes(raw: Any, what: str) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping) or not all(isinstance(v, Mapping) for v in raw.values()):
        raise ProfileError(f"{what}: modes must map a mode name to request parameters")
    unknown = set(raw) - set(GENERATION_MODES)
    if unknown:
        raise ProfileError(f"{what}: unknown modes {sorted(unknown)} (known: {GENERATION_MODES})")
    return {str(mode): dict(params) for mode, params in raw.items()}


def _parse_profile(name: str, raw: Mapping[str, Any]) -> ModelProfile:
    if not isinstance(raw, Mapping):
        raise ProfileError(f"model {name!r}: expected a mapping")
    reasoning = str(raw.get("reasoning", "auto"))
    if reasoning not in REASONING_MODES:
        raise ProfileError(f"model {name!r}: reasoning must be one of {REASONING_MODES}, got {reasoning!r}")
    tags = tuple(raw.get("think_tags") or ("<think>", "</think>"))
    if len(tags) != 2 or not all(isinstance(t, str) and t for t in tags):
        raise ProfileError(f"model {name!r}: think_tags must be two non-empty strings")
    sampling_raw = raw.get("sampling") or {}
    unknown = set(sampling_raw) - set(Sampling.__dataclass_fields__)
    if unknown:
        raise ProfileError(f"model {name!r}: unknown sampling keys {sorted(unknown)}")
    endpoints: dict[str, Endpoint] = {}
    for ep_name, ep_raw in (raw.get("endpoints") or {}).items():
        if not isinstance(ep_raw, Mapping):
            raise ProfileError(f"model {name!r} endpoint {ep_name!r}: expected a mapping")
        backend = str(ep_raw.get("backend", "vllm"))
        if backend not in BACKENDS:
            raise ProfileError(f"model {name!r} endpoint {ep_name!r}: backend must be one of {BACKENDS}")
        prefill = ep_raw.get("prefill") or {}
        if not isinstance(prefill, Mapping) or (prefill and prefill.get("style") not in PREFILL_STYLES):
            raise ProfileError(f"model {name!r} endpoint {ep_name!r}: prefill needs a style in {PREFILL_STYLES}")
        options = ep_raw.get("options") or {}
        if not isinstance(options, Mapping):
            raise ProfileError(f"model {name!r} endpoint {ep_name!r}: options must be a mapping")
        endpoints[ep_name] = Endpoint(
            name=ep_name,
            backend=backend,
            base_url=str(ep_raw.get("base_url", "")),
            model=str(ep_raw.get("model", "")),
            api_key_env=ep_raw.get("api_key_env"),
            extra_body=dict(ep_raw.get("extra_body") or {}),
            modes=_parse_modes(ep_raw.get("modes"), f"model {name!r} endpoint {ep_name!r}"),
            prefill_style=prefill.get("style"),
            prefill_base_url=prefill.get("base_url"),
            options=dict(options),
        )
    return ModelProfile(
        name=name,
        tokenizer=str(raw.get("tokenizer", "generic")),
        reasoning=reasoning,
        reasoning_field=raw.get("reasoning_field"),
        think_tags=(tags[0], tags[1]),
        inline_starts_in_reasoning=bool(raw.get("inline_starts_in_reasoning", False)),
        sampling=Sampling(**sampling_raw),
        modes=_parse_modes(raw.get("modes"), f"model {name!r}"),
        endpoints=endpoints,
    )


def load_profiles(path: Optional[Path] = None) -> dict[str, ModelProfile]:
    path = Path(path) if path is not None else default_models_file()
    if not path.is_file():
        raise ProfileError(f"models file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ProfileError(f"{path}: expected a mapping of model names to profiles")
    return {str(name): _parse_profile(str(name), raw) for name, raw in data.items()}
