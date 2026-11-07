"""Faster ``load_inline`` builds of generated kernels.

A ``load_inline`` build is nvcc on ``cuda.cu`` (the critical path) in parallel with g++
on ``main.cpp`` (the pybind bindings).  Inside evaluation children the wrapper below
tries cheaper ways to build the same code, in order:

0. ``bare``: CUDA sources that use nothing from torch (kernels and launchers taking raw
   pointers, with every tensor-facing wrapper in ``cpp_sources``) get only CUDA and C++
   standard headers.  Measured on a B300 host for the same kernels: nvcc 14.9 s slim,
   1.9 s bare, identical SASS.
1. ``slim``: the CUDA source gets a small prelude instead of the ``<torch/types.h>``
   ``load_inline`` adds (all of ATen) and of any ``<torch/extension.h>``-style umbrella
   the code includes: ``at::Tensor``, the common tensor factories and the ``torch::``
   dtype aliases, everything generated kernels use.  The prelude is a subset of
   ``<torch/types.h>``, so whatever builds with it also builds as written.
2. ``trimmed``: the implicit ``<torch/types.h>``, without ``<torch/extension.h>``,
   ``<torch/torch.h>`` and ``<torch/all.h>``.
3. ``original``: the source as written.

A tier that does not build falls through to the next.  The tier that built is recorded
next to the build, so every process loading the kernel (build, validation, timing,
ncu) compiles the same source and reuses the same build.

Independently, ``main.cpp`` is compiled against a precompiled ``<torch/extension.h>``
kept in a private cache (one per compiler flag set; torch's own ``use_pch`` writes
into the shared torch installation).  Measured on an H100 host: nvcc 44 s as written,
25 s trimmed, 15 s slim; ``main.cpp`` 16 s, 9 s with the PCH at -O3.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import hashlib
import inspect
import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

ENV_FLAG = "SPECGEN_FAST_INCLUDES"
PCH_FLAG = "SPECGEN_PCH"
PCH_DIR_ENV = "SPECGEN_PCH_DIR"
MARKER = "specgen_fastbuild.json"
TIERS = ("bare", "slim", "trimmed", "original")

TORCH_REFERENCE = re.compile(
    r"\b(?:torch|at|c10|caffe2|pybind11)::|\b(?:Tensor|TensorOptions|ScalarType)\b|\b(?:TORCH|AT|C10)_[A-Z0-9_]+\b"
    r"|#[ \t]*include[ \t]*[<\"](?:torch|ATen|c10|pybind11)/")
COMMENT_OR_STRING = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"', re.S)
# What torch-free CUDA code commonly uses without including it (torch's headers used to bring these in).
# They add about 0.5 s to nvcc, still shorter than main.cpp, the other half of the build.
BARE_PRELUDE = "\n".join([
    "#include <cuda.h>",
    "#include <cuda_runtime.h>",
    "#include <cuda_fp16.h>",
    "#include <cuda_bf16.h>",
    *(f"#include <{h}>" for h in ("algorithm", "array", "cfloat", "climits", "cmath", "cstddef", "cstdint", "cstdio",
                                  "cstdlib", "cstring", "limits", "stdexcept", "string", "type_traits", "utility",
                                  "vector")),
    "",
])
HEAVY_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]torch/(?:extension|torch|all)\.h[>"][^\n]*$', re.M)
UMBRELLA_INCLUDE = re.compile(
    r'^[ \t]*#[ \t]*include[ \t]*[<"](?:torch/(?:types|extension|torch|all)\.h|ATen/ATen\.h)[>"][^\n]*$', re.M)
SLIM_OPS = ("empty", "empty_like", "empty_strided", "zeros", "zeros_like", "ones", "ones_like", "full", "full_like")
# Must stay a subset of <torch/types.h>: no CUDA-specific torch headers here.
SLIM_PRELUDE = "\n".join([
    "#include <cuda.h>",
    "#include <cuda_runtime.h>",
    "#include <ATen/core/Tensor.h>",
    *(f"#include <ATen/ops/{op}.h>" for op in SLIM_OPS),
    "#include <optional>",
    "namespace torch {",
    "using namespace at;  // NOLINT",
    "using std::nullopt;  // NOLINT",
    "using std::optional;  // NOLINT",
    "using Dtype = at::ScalarType;",
    "constexpr auto kUInt8 = at::kByte;",
    "constexpr auto kInt8 = at::kChar;",
    "constexpr auto kInt16 = at::kShort;",
    "constexpr auto kInt32 = at::kInt;",
    "constexpr auto kInt64 = at::kLong;",
    "constexpr auto kUInt16 = at::kUInt16;",
    "constexpr auto kUInt32 = at::kUInt32;",
    "constexpr auto kUInt64 = at::kUInt64;",
    "constexpr auto kFloat16 = at::kHalf;",
    "constexpr auto kFloat32 = at::kFloat;",
    "constexpr auto kFloat64 = at::kDouble;",
    "constexpr auto kU8 = kUInt8;",
    "constexpr auto kU16 = kUInt16;",
    "constexpr auto kU32 = kUInt32;",
    "constexpr auto kU64 = kUInt64;",
    "constexpr auto kI8 = kInt8;",
    "constexpr auto kI16 = kInt16;",
    "constexpr auto kI32 = kInt32;",
    "constexpr auto kI64 = kInt64;",
    "constexpr auto kF16 = kFloat16;",
    "constexpr auto kF32 = kFloat32;",
    "constexpr auto kF64 = kFloat64;",
    "}  // namespace torch",
    "",
])


def _sub(pattern: re.Pattern, sources: Any) -> Any:
    if isinstance(sources, str):
        return pattern.sub("", sources)
    if isinstance(sources, (list, tuple)):
        return type(sources)(pattern.sub("", s) if isinstance(s, str) else s for s in sources)
    return sources


def strip_heavy_includes(sources: Any) -> Any:
    """``cuda_sources`` (a string or a list of strings) without the whole-API torch includes."""
    return _sub(HEAVY_INCLUDE, sources)


def _as_list(sources: Any) -> list:
    if sources is None:
        return []
    return [sources] if isinstance(sources, str) else list(sources)


def torch_free(sources: Any) -> bool:
    """Whether CUDA sources reference nothing from torch, ignoring comments and string literals."""
    text = "\n".join(s for s in _as_list(sources) if isinstance(s, str))
    return bool(text.strip()) and not TORCH_REFERENCE.search(COMMENT_OR_STRING.sub(" ", text))


def tier_overrides(arguments: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``(tier, load_inline arguments to override)`` from cheapest to as written."""
    cuda = arguments.get("cuda_sources")
    tiers: list[tuple[str, dict[str, Any]]] = []
    if not arguments.get("no_implicit_headers"):
        cpp = ["#include <torch/extension.h>", *_as_list(arguments.get("cpp_sources"))]
        if torch_free(cuda):
            tiers.append(("bare", {"cuda_sources": [BARE_PRELUDE, *_as_list(cuda)], "cpp_sources": cpp,
                                   "no_implicit_headers": True}))
        tiers.append(("slim", {
            "cuda_sources": [SLIM_PRELUDE, *_as_list(_sub(UMBRELLA_INCLUDE, cuda))],
            "cpp_sources": cpp,
            "no_implicit_headers": True,
        }))
        trimmed = strip_heavy_includes(cuda)
        if trimmed != cuda:
            tiers.append(("trimmed", {"cuda_sources": trimmed}))
    tiers.append(("original", {}))
    return tiers


def _marker_path(build_directory: Optional[str]) -> Optional[Path]:
    root = build_directory or os.environ.get("TORCH_EXTENSIONS_DIR")
    return Path(root) / MARKER if root else None


def _read(path: Optional[Path]) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path is not None and path.is_file() else {}
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def _record(path: Optional[Path], name: str, choice: str) -> None:
    if path is None:
        return
    data = _read(path)
    if data.get(name) != choice:
        data[name] = choice
        with contextlib.suppress(OSError):
            _write_json(path, data)


# -- precompiled <torch/extension.h> for main.cpp -----------------------------------

def main_cpp_flags(extra_cflags: Any, extra_include_paths: Any, with_cuda: bool) -> tuple[str, list[str]]:
    """``(compiler, flags)`` torch uses for ``main.cpp``, without the per-extension name."""
    from torch.utils import cpp_extension

    with tempfile.TemporaryDirectory(prefix="specgen_pch_probe_") as tmp:
        ninja = Path(tmp) / "build.ninja"
        cpp_extension._write_ninja_file_to_build_library(
            path=str(ninja), name="specgen_pch_probe", sources=[str(Path(tmp) / "main.cpp")],
            extra_cflags=list(extra_cflags or []), extra_cuda_cflags=[], extra_sycl_cflags=[], extra_ldflags=[],
            extra_include_paths=list(extra_include_paths or []), with_cuda=with_cuda, with_sycl=False,
            is_standalone=False)
        values = dict(re.findall(r"^(\w+) = (.*)$", ninja.read_text(), re.M))
    flags = [f for f in shlex.split(values.get("cflags", "")) if not f.startswith("-DTORCH_EXTENSION_NAME=")]
    return values.get("cxx", "c++"), flags


def pch_root() -> Path:
    return Path(os.environ.get(PCH_DIR_ENV) or Path.home() / ".cache" / "specgen" / "pch")


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=600)


def _compiler_version(compiler: str) -> str:
    try:
        return _run([compiler, "--version"]).stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        return "?"


def pch_include(extra_cflags: Any, extra_include_paths: Any, with_cuda: bool) -> list[str]:
    """``["-include", header]`` when a working PCH for these flags exists (built here if nobody is
    building it now); ``[]`` otherwise, which just compiles main.cpp the normal way."""
    try:
        import torch

        compiler, flags = main_cpp_flags(extra_cflags, extra_include_paths, with_cuda)
        key = hashlib.sha256(json.dumps([compiler, _compiler_version(compiler), torch.__version__, flags]).encode())
        directory = pch_root() / key.hexdigest()[:16]
        header, gch = directory / "specgen_torch_ext.h", directory / "specgen_torch_ext.h.gch"
        ok, failed = directory / "ok", directory / "failed"
        if ok.is_file() and gch.is_file():
            return ["-include", str(header)]
        if failed.is_file():
            return []
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return []  # another build is making it; do not wait
            if ok.is_file() and gch.is_file():
                return ["-include", str(header)]
            header.write_text("#include <torch/extension.h>\n", encoding="utf-8")
            tmp = directory / f"specgen_torch_ext.h.gch.{os.getpid()}.tmp"
            build = _run([compiler, "-x", "c++-header", *flags, str(header), "-o", str(tmp)])
            if build.returncode != 0:
                failed.write_text(build.stderr[-4000:], encoding="utf-8")
                return []
            os.replace(tmp, gch)
            probe = directory / "probe.cpp"
            probe.write_text("#include <torch/extension.h>\nint specgen_probe() { return 0; }\n", encoding="utf-8")
            check = _run([compiler, *flags, "-DTORCH_EXTENSION_NAME=specgen_probe", "-include", str(header),
                          "-Winvalid-pch", "-H", "-fsyntax-only", str(probe)])
            if check.returncode != 0 or f"! {gch}" not in check.stderr:
                failed.write_text("precompiled header not used:\n" + check.stderr[-4000:], encoding="utf-8")
                return []
            ok.write_text(" ".join(flags), encoding="utf-8")
            return ["-include", str(header)]
    except Exception:  # noqa: BLE001 - the PCH only saves time; never fail a build over it
        return []


# -- the wrapper ---------------------------------------------------------------------

def install(*, tiers: bool = True, pch: bool = True) -> bool:
    """Wrap ``torch.utils.cpp_extension.load_inline``; call before importing a kernel module."""
    from torch.utils import cpp_extension

    original = cpp_extension.load_inline
    if getattr(original, "_specgen_fastbuild", False):
        return False
    signature = inspect.signature(original)

    @functools.wraps(original)
    def load_inline(*args: Any, **kwargs: Any) -> Any:
        arguments = dict(signature.bind(*args, **kwargs).arguments)
        name, cuda = arguments.get("name"), arguments.get("cuda_sources")
        if pch:
            include = pch_include(arguments.get("extra_cflags"), arguments.get("extra_include_paths"),
                                  bool(cuda) if arguments.get("with_cuda") is None else bool(arguments["with_cuda"]))
            if include:
                arguments["extra_cflags"] = [*_as_list(arguments.get("extra_cflags")), *include]
        if not tiers or not cuda:
            return original(**arguments)
        marker = _marker_path(arguments.get("build_directory"))
        candidates = tier_overrides(arguments)
        recorded = _read(marker).get(name)
        if recorded in TIERS:
            candidates = [c for c in candidates if c[0] == recorded] or candidates[-1:]
        for index, (tier, overrides) in enumerate(candidates):
            try:
                module = original(**{**arguments, **overrides})
            except RuntimeError as e:
                if "Error building extension" not in str(e) or index == len(candidates) - 1:
                    raise
                # Retry under the same name, so the build directory and module name stay those
                # that later processes use.
                cpp_extension.JIT_EXTENSION_VERSIONER.entries.pop(name, None)
                continue
            _record(marker, name, tier)
            return module
        raise AssertionError("unreachable")

    load_inline._specgen_fastbuild = True  # type: ignore[attr-defined]
    cpp_extension.load_inline = load_inline
    return True


def install_if_enabled() -> bool:
    tiers = os.environ.get(ENV_FLAG, "1") != "0"
    pch = os.environ.get(PCH_FLAG, "1") != "0"
    return (tiers or pch) and install(tiers=tiers, pch=pch)
