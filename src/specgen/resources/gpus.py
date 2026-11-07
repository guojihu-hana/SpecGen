"""GPU discovery through nvidia-smi (does not create a CUDA context)."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Optional

from specgen.generation.gpu_specs import normalize_gpu_name

_ARCH_TOKEN = re.compile(r"^(\d+)\.(\d+)(a?)(\+PTX)?$")
ARCH_EXAMPLES = "9.0a for H100/H200, 10.0a for B200, 10.3a for B300"


def _nvidia_smi(fields: str) -> list[list[str]]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [[c.strip() for c in line.split(",")] for line in proc.stdout.splitlines() if line.strip()]


def discover_gpus() -> list[str]:
    """``CUDA_VISIBLE_DEVICES`` if set, else every GPU nvidia-smi lists."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [g.strip() for g in visible.split(",") if g.strip()]
    return [row[0] for row in _nvidia_smi("index")]


def compute_capabilities() -> dict[str, str]:
    """GPU index -> compute capability, e.g. ``{"0": "9.0"}``."""
    return {row[0]: row[1] for row in _nvidia_smi("index,compute_cap") if len(row) >= 2}


def arch_for(capability: str) -> str:
    """``TORCH_CUDA_ARCH_LIST`` entry for one device; sm_90 and newer get the
    arch-specific ``a`` variant so wgmma / TMA style instructions compile."""
    major = int(capability.split(".")[0])
    return f"{capability}a" if major >= 9 else capability


def parse_arch_list(value: str) -> list[str]:
    """Split a ``TORCH_CUDA_ARCH_LIST`` value (spaces or semicolons) and validate each entry."""
    tokens = [t for t in re.split(r"[;\s]+", value.strip()) if t]
    bad = [t for t in tokens if not _ARCH_TOKEN.match(t)]
    if not tokens or bad:
        raise ValueError(f"invalid CUDA arch list {value!r} (expected entries like {ARCH_EXAMPLES})")
    return tokens


def arch_runs_on(tokens: list[str], capability: str) -> bool:
    """Whether code built for ``tokens`` runs on a device of ``capability``.

    Plain SASS runs on the same major version with an equal or newer minor; the
    arch-specific ``a`` variants only on that exact version; ``+PTX`` JIT-compiles
    for any newer device.
    """
    cap_major, cap_minor = (int(x) for x in capability.split("."))
    for token in tokens:
        major, minor, specific, ptx = _ARCH_TOKEN.match(token).groups()
        major, minor = int(major), int(minor)
        if (major, minor) == (cap_major, cap_minor):
            return True
        if not specific and major == cap_major and minor <= cap_minor:
            return True
        if ptx and not specific and (major, minor) <= (cap_major, cap_minor):
            return True
    return False


def detect_gpu_type(gpus: list[str]) -> Optional[str]:
    """Prompt GPU key of the first GPU, if nvidia-smi knows it and it is in the spec table."""
    if not gpus:
        return None
    names = {row[0]: row[1] for row in _nvidia_smi("index,name") if len(row) >= 2}
    name = names.get(gpus[0])
    return normalize_gpu_name(name) if name else None
