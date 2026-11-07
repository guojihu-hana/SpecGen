"""Extract the generated Python module from LLM output."""

from __future__ import annotations

import re
from typing import Optional

_FENCED_PYTHON = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def extract_python_module(text: str) -> Optional[str]:
    """The last ```python fence (models often put the final file after earlier
    snippets); else the last generic fence that looks like a module defining ``Model``."""
    if not text or not text.strip():
        return None
    blocks = [b.strip() for b in _FENCED_PYTHON.findall(text) if b.strip()]
    if blocks:
        return blocks[-1]
    generic = [b.strip() for b in _ANY_FENCE.findall(text) if "import" in b and "Model" in b]
    return generic[-1] if generic else None


def has_code(module: str) -> bool:
    """False for a module of comments and blank lines only (e.g. a lone ``# APPROACH:`` line)."""
    return any(line.strip() and not line.lstrip().startswith("#") for line in module.splitlines())
