"""Kernels must do the work in hand-written CUDA: no library GEMM / convolution, not even as a fallback,
and no change to torch's matmul precision (TF32) settings.

Two checks: ``forbidden_usage`` reads the module before it is built (Python calls and assignments via
``ast``, plus ATen / cuBLAS / cuDNN calls in the C++ / CUDA source strings); ``library_kernels`` names
the vendor-library kernels a forward actually ran (from the profiler trace of a validation run), which
also catches calls the source check cannot see.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from typing import Iterable, Optional

LIBRARY_OPS = {"matmul", "mm", "bmm", "addmm", "addbmm", "baddbmm", "addmv", "mv", "einsum", "tensordot",
               "linear", "bilinear", "chain_matmul", "conv1d", "conv2d", "conv3d", "conv_transpose1d",
               "conv_transpose2d", "conv_transpose3d"}
TENSOR_METHODS = {"matmul", "mm", "bmm", "addmm", "baddbmm", "mv"}  # x.matmul(y) on a tensor
TORCH_ROOTS = {"torch", "F", "functional", "nn", "_C", "_VF", "aten", "ops"}
PRECISION_CALLS = {"set_float32_matmul_precision"}
PRECISION_FLAGS = re.compile(r"allow_tf32|allow_fp16_reduced_precision_reduction|allow_bf16_reduced_precision_reduction")
CPP_LIBRARY = re.compile(
    r"\b(?:at|torch)::(?:" + "|".join(sorted(LIBRARY_OPS)) + r"|_convolution|convolution)\s*\(|\bcublas\w*\s*\(|\bcudnn\w*\s*\(")
LIBRARY_KERNEL = re.compile(r"(?i)cutlass|cublas|nvjet|xmma|cudnn|^(?:void\s+)?(?:ampere|volta|turing|hopper|sm\d+)_\w*gemm"
                            r"|^(?:void\s+)?(?:gemv2[nt]|gemmsn|gemmk1|splitkreduce)")

CATEGORY_LIBRARY = "uses a library GEMM or convolution instead of a hand-written kernel"
CATEGORY_PRECISION = "changes torch's matmul precision (TF32) settings"
CATEGORY_RUNTIME = "runs library kernels instead of hand-written ones"


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


class _Finder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.extensions: set[str] = set()  # names bound to load_inline(...) / load(...) results
        self.from_torch: set[str] = set()  # names imported from torch modules (from torch import matmul)
        self.torch_aliases: set[str] = set(TORCH_ROOTS)
        self.found: list[tuple[str, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] == "torch":
                self.torch_aliases.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if (node.module or "").split(".")[0] == "torch":
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name in LIBRARY_OPS:
                    self.from_torch.add(name)
                else:
                    self.torch_aliases.add(name)  # e.g. from torch.nn import functional as F

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call) and _dotted(node.value.func).split(".")[-1] in ("load_inline", "load"):
            for target in node.targets:
                if isinstance(target, (ast.Name, ast.Attribute)):
                    self.extensions.add(_dotted(target))
        for target in node.targets:
            if isinstance(target, ast.Attribute) and PRECISION_FLAGS.search(_dotted(target)):
                self.found.append((CATEGORY_PRECISION, _dotted(target)))
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult):
            self.found.append((CATEGORY_LIBRARY, "the @ operator"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.from_torch:
            self.found.append((CATEGORY_LIBRARY, f"{func.id}()"))
        elif isinstance(func, ast.Attribute):
            dotted = _dotted(func)
            root, base = dotted.split(".")[0], dotted.rsplit(".", 1)[0]
            if func.attr in PRECISION_CALLS:
                self.found.append((CATEGORY_PRECISION, dotted))
            elif base in self.extensions:
                pass  # a function of the module's own extension
            elif func.attr in LIBRARY_OPS and root in self.torch_aliases:
                self.found.append((CATEGORY_LIBRARY, f"{dotted}()"))
            elif func.attr in TENSOR_METHODS and root not in self.torch_aliases:
                self.found.append((CATEGORY_LIBRARY, f"{dotted}()"))
        self.generic_visit(node)


@lru_cache(maxsize=8192)
def violations(code: str) -> tuple[tuple[str, str], ...]:
    """``(category, what)`` for each library call or precision switch found, in source order.
    Unparsable code has none: the build reports the syntax error."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()
    finder = _Finder()
    finder.visit(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            match = CPP_LIBRARY.search(node.value)
            if match:
                finder.found.append((CATEGORY_LIBRARY, f"{match.group(0).rstrip('( ')} in the C++/CUDA source"))
    return tuple(finder.found)


def forbidden_usage(code: str) -> Optional[str]:
    """``"<category>: <what>"`` for the module's violations (library calls first), or None."""
    found = violations(code)
    if not found:
        return None
    parts = []
    for category in (CATEGORY_LIBRARY, CATEGORY_PRECISION):
        what = list(dict.fromkeys(w for c, w in found if c == category))
        if what:
            parts.append((category, what))
    (category, what), rest = parts[0], parts[1:]
    message = f"{category}: {', '.join(what[:3])}"
    return message + "".join(f"; also {c}: {', '.join(w[:3])}" for c, w in rest)


def library_kernels(names: Iterable[str]) -> list[str]:
    """The vendor-library (cuBLAS, cuBLASLt, CUTLASS, cuDNN) kernels among executed kernel names."""
    return sorted({name for name in names if LIBRARY_KERNEL.search(name.strip())})
