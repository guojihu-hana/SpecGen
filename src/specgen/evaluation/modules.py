"""Loading task and kernel modules inside an evaluation child."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

REFERENCE_MODULE = "specgen_reference"
KERNEL_MODULE = "specgen_kernel"


def import_module_from_path(path: Path, name: str) -> ModuleType:
    path = Path(path).resolve()
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def move_to_device(obj: Any, device: Any) -> Any:
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def make_inputs(reference: ModuleType, device: Any) -> Any:
    """``get_inputs()`` with tensors created on ``device``: a multi-GB input is generated
    by the GPU instead of the CPU, and never copied (the seed still fixes the values)."""
    import torch

    if getattr(device, "type", "cpu") == "cuda":
        with torch.device(device):
            return move_to_device(reference.get_inputs(), device)
    return move_to_device(reference.get_inputs(), device)


def call_model(model: Any, inputs: Any) -> Any:
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    return model(inputs)


def is_cuda_error(exc: BaseException) -> bool:
    text = str(exc)
    return any(s in text for s in ("CUDA error", "illegal memory access", "device-side assert", "misaligned address"))


class EvalFailure(Exception):
    """Ends an evaluation with a structured result."""

    def __init__(self, status: str, error: str, **extra: Any):
        super().__init__(error)
        self.result = {"ok": False, "status": status, "error": error, **extra}


@dataclass
class LoadedPair:
    reference: ModuleType
    model_ref: Any
    model_gen: Any
    device: Any
    compile_s: float
    weights_synced: bool


def load_pair(task_path: str, kernel_path: str, seed: int) -> LoadedPair:
    """Import the reference and the generated module and build both models with equal weights.

    Both models are built from the same seed; the reference state dict is then copied
    into the generated model when the parameter structure matches.  A job pinned to a
    GPU always runs there: without CUDA it fails instead of validating on the CPU.
    """
    import torch

    from specgen.evaluation import fastbuild

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if device.type != "cuda" and visible:
        raise EvalFailure("cuda_unavailable", f"assigned GPU {visible}, but CUDA is not available in the evaluation process")
    try:
        reference = import_module_from_path(Path(task_path), REFERENCE_MODULE)
    except Exception:
        raise EvalFailure("reference_error", f"failed to load the reference:\n{traceback.format_exc()}")
    fastbuild.install_if_enabled()
    t0 = time.perf_counter()
    try:
        generated = import_module_from_path(Path(kernel_path), KERNEL_MODULE)
    except Exception:
        raise EvalFailure(
            "compile_error",
            f"failed to import the kernel:\n{traceback.format_exc()}",
            compile_s=round(time.perf_counter() - t0, 3),
        )
    compile_s = round(time.perf_counter() - t0, 3)
    try:
        init_args = list(reference.get_init_inputs())
        torch.manual_seed(seed)
        model_ref = reference.Model(*init_args)
        torch.manual_seed(seed)
        model_gen = generated.Model(*init_args)
        try:
            model_gen.load_state_dict(model_ref.state_dict(), strict=True)
            weights_synced = True
        except Exception:
            weights_synced = False
        model_ref = model_ref.to(device)
        model_gen = model_gen.to(device)
    except Exception as e:
        status = "cuda_error" if is_cuda_error(e) else "runtime_error"
        raise EvalFailure(status, f"model construction failed:\n{traceback.format_exc()}", compile_s=compile_s)
    return LoadedPair(reference, model_ref, model_gen, device, compile_s, weights_synced)
