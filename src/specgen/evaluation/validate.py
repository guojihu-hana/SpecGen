"""Correctness: compare the generated model against the reference on several inputs.

Each output tensor must be elementwise close (``atol``, ``rtol``) and, as a whole, within
``rtol`` relative error (L2).  The second check matters when the outputs are tiny: a softmax
over 393216 columns has values around 3e-6, so ``atol = 1e-2`` alone accepts a kernel that
leaves most rows unwritten.  Before the generated model runs, blocks the size of its outputs
are filled with NaN and freed, so elements it never writes do not pass as stale data.

With ``shift_trial`` one more trial feeds floating inputs mapped to 2x - mean(|x|)/2: twice the
range, and negative values where get_inputs() makes none.  A kernel that hard-codes the range of
the task's inputs (e.g. an INT8 GEMM quantizing x as round(255 x) - 128 for x in [0, 1)) fails it.
It is skipped when the reference itself fails or gives NaN/Inf on such inputs (log, sqrt, ...).

With ``ref_cache`` set, the reference outputs of each (seed, trial) are computed once
per task and reused by later validations.
"""

from __future__ import annotations

import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from specgen.evaluation.jobs import EvalJob
from specgen.evaluation.modules import EvalFailure, call_model, is_cuda_error, load_pair, make_inputs, move_to_device

MISMATCH_SAMPLE = 20
COMPARE_CHUNK = 1 << 24  # elements per comparison chunk
# Larger outputs are recomputed: on the GPU that is faster than reading them back from disk.
REF_CACHE_MAX_BYTES = 64 * 2**20


def _nbytes(obj: Any) -> int:
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(x) for x in obj)
    return 0


def _tensors(obj: Any) -> list[Any]:
    import torch

    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for x in obj for t in _tensors(x)]
    return []


def poison_freed_memory(like: Any) -> None:
    """Allocate blocks shaped like the reference outputs, fill them with NaN (integers with
    their minimum) and free them: the caching allocator hands them to the generated model's
    outputs, so elements it never writes cannot match by accident."""
    import torch

    blocks = []
    for t in _tensors(like):
        if t.is_cuda and t.dtype != torch.bool:
            block = torch.empty_like(t)
            block.fill_(float("nan") if block.is_floating_point() or block.is_complex()
                        else torch.iinfo(block.dtype).min)
            blocks.append(block)
    del blocks


SHIFT_SCALE, SHIFT_OFFSET = 2.0, 0.5  # x -> SHIFT_SCALE * x - SHIFT_OFFSET * mean(|x|)


def shift_inputs(obj: Any) -> Any:
    """Floating tensors mapped in place to 2x - mean(|x|)/2; other inputs are kept."""
    import torch

    for t in _tensors(obj):
        if t.is_floating_point() and t.numel():
            mean_abs = float(torch.linalg.vector_norm(t.detach(), ord=1)) / t.numel()  # no |x| copy
            t.mul_(SHIFT_SCALE).sub_(SHIFT_OFFSET * mean_abs)
    return obj


def _all_finite(obj: Any) -> bool:
    import torch

    return all(bool(torch.isfinite(t).all()) for t in _tensors(obj) if t.is_floating_point() or t.is_complex())


def _detached_cpu(obj: Any) -> Any:
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_detached_cpu(x) for x in obj)
    return obj


def reference_output(model_ref: Any, inputs: Any, device: Any, cache: Optional[Path]) -> tuple[Any, bool]:
    """``(output, from_cache)``; a fresh output is stored when it is small enough."""
    import torch

    if cache is not None and cache.is_file():
        try:
            return move_to_device(torch.load(cache, map_location="cpu", weights_only=True), device), True
        except Exception:  # noqa: BLE001 - a damaged entry is recomputed
            pass
    out = call_model(model_ref, inputs)
    if cache is not None and _nbytes(out) <= REF_CACHE_MAX_BYTES:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
            torch.save(_detached_cpu(out), tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return out, False


def _compare_tensors(a: Any, b: Any, atol: float, rtol: float) -> tuple[bool, dict[str, Any]]:
    """Elementwise isclose and the relative L2 error of ``b`` against ``a``, on the tensors'
    device and in chunks: outputs of several GB never leave the GPU and temporaries stay small."""
    import torch

    flat_a, flat_b = a.detach().reshape(-1), b.detach().reshape(-1).to(a.device)
    max_abs, mismatches, sample = 0.0, 0, []
    diff_sq, ref_sq = 0.0, 0.0
    for start in range(0, flat_a.numel(), COMPARE_CHUNK):
        ca = flat_a[start:start + COMPARE_CHUNK].float()
        cb = flat_b[start:start + COMPARE_CHUNK].float()
        diff = (ca - cb).abs()
        chunk_max = float(diff.max().item())
        max_abs = chunk_max if chunk_max != chunk_max else max(max_abs, chunk_max)  # keep a NaN visible
        diff_sq += float(diff.square().sum(dtype=torch.float64).item())
        ref_sq += float(ca.square().sum(dtype=torch.float64).item())
        bad = ~torch.isclose(ca, cb, atol=atol, rtol=rtol, equal_nan=False)
        count = int(bad.sum().item())
        if count and len(sample) < MISMATCH_SAMPLE:
            for index in torch.nonzero(bad, as_tuple=True)[0][:MISMATCH_SAMPLE - len(sample)].tolist():
                ref, gen = float(ca[index]), float(cb[index])
                sample.append({"flat_index": start + index, "ref": ref, "gen": gen, "abs_diff": abs(ref - gen)})
        mismatches += count
    # an all-zero reference leaves only the elementwise check
    rel_err = math.sqrt(diff_sq / ref_sq) if ref_sq > 0 else None
    info: dict[str, Any] = {"max_abs_diff": max_abs, "rel_err": rel_err, "atol": atol, "rtol": rtol}
    if mismatches:
        info.update(reason="value_mismatch", mismatches=mismatches, sample=sample)
        return False, info
    if rel_err is not None and not rel_err <= rtol:  # also catches NaN
        info.update(reason="relative_error")
        return False, info
    return True, info


def compare_outputs(out_ref: Any, out_gen: Any, atol: float, rtol: float) -> tuple[bool, dict[str, Any]]:
    import torch

    if type(out_ref) is not type(out_gen):
        return False, {"reason": "type_mismatch", "ref": type(out_ref).__name__, "gen": type(out_gen).__name__}
    if isinstance(out_ref, torch.Tensor):
        if out_ref.shape != out_gen.shape:
            return False, {"reason": "shape_mismatch", "ref": list(out_ref.shape), "gen": list(out_gen.shape)}
        if out_ref.dtype != out_gen.dtype:
            return False, {"reason": "dtype_mismatch", "ref": str(out_ref.dtype), "gen": str(out_gen.dtype)}
        return _compare_tensors(out_ref, out_gen, atol, rtol)
    if isinstance(out_ref, (list, tuple)):
        if len(out_ref) != len(out_gen):
            return False, {"reason": "length_mismatch", "ref": len(out_ref), "gen": len(out_gen)}
        worst, worst_rel = 0.0, None
        for i, (a, b) in enumerate(zip(out_ref, out_gen)):
            ok, info = compare_outputs(a, b, atol, rtol)
            if not ok:
                return False, {**info, "output_index": i}
            worst = max(worst, info.get("max_abs_diff", 0.0))
            if info.get("rel_err") is not None:
                worst_rel = max(worst_rel or 0.0, info["rel_err"])
        return True, {"max_abs_diff": worst, "rel_err": worst_rel}
    equal = out_ref == out_gen
    return bool(equal), {"reason": "non_tensor_output", "equal": bool(equal)}


def run(job: EvalJob) -> dict[str, Any]:
    import torch

    p = job.params
    seed, trials = int(p.get("seed", 0)), int(p.get("trials", 3))
    atol, rtol = float(p.get("atol", 1e-2)), float(p.get("rtol", 1e-2))
    try:
        pair = load_pair(job.task_path, job.kernel_path, seed)
    except EvalFailure as failure:
        return failure.result

    on_cuda = pair.device.type == "cuda"
    cache_dir = Path(p["ref_cache"]) if p.get("ref_cache") else None
    worst, worst_rel = 0.0, None
    cached = 0
    details: Optional[dict[str, Any]] = None
    inputs = out_ref = out_gen = None
    t0 = time.perf_counter()
    for trial in range(trials):
        try:
            inputs = out_ref = out_gen = None  # free the previous trial's tensors first
            torch.manual_seed(seed + trial)
            inputs = make_inputs(pair.reference, pair.device)
            cache = cache_dir / f"{pair.device.type}_seed{seed}_trial{trial}.pt" if cache_dir else None
            with torch.no_grad():
                out_ref, hit = reference_output(pair.model_ref, inputs, pair.device, cache)
                cached += hit
                if on_cuda:
                    poison_freed_memory(out_ref)
                out_gen = call_model(pair.model_gen, inputs)
            if on_cuda:
                torch.cuda.synchronize()
            ok, details = compare_outputs(out_ref, out_gen, atol, rtol)
        except Exception as e:
            status = "cuda_error" if is_cuda_error(e) else "runtime_error"
            return {"ok": False, "status": status, "error": traceback.format_exc(), "trial": trial,
                    "compile_s": pair.compile_s}
        if not ok:
            return {"ok": False, "status": "numerical_error", "error": f"output mismatch in trial {trial}",
                    "details": {**details, "trial": trial}, "compile_s": pair.compile_s}
        worst = max(worst, details.get("max_abs_diff", 0.0))
        if details.get("rel_err") is not None:
            worst_rel = max(worst_rel or 0.0, details["rel_err"])

    shift: dict[str, Any] = {"status": "off"}
    if p.get("shift_trial", True):
        try:
            inputs = out_ref = out_gen = None
            torch.manual_seed(seed + trials)
            with torch.no_grad():
                inputs = shift_inputs(make_inputs(pair.reference, pair.device))
                try:
                    out_ref = call_model(pair.model_ref, inputs)
                    if on_cuda:
                        torch.cuda.synchronize()
                    usable = _all_finite(out_ref)
                except Exception:  # noqa: BLE001 - inputs outside what the task accepts
                    usable = False
                if not usable:
                    shift = {"status": "skipped", "reason": "the reference fails or gives NaN/Inf on shifted inputs"}
                else:
                    if on_cuda:
                        poison_freed_memory(out_ref)
                    out_gen = call_model(pair.model_gen, inputs)
                    if on_cuda:
                        torch.cuda.synchronize()
                    ok, details = compare_outputs(out_ref, out_gen, atol, rtol)
                    if not ok:
                        return {"ok": False, "status": "numerical_error",
                                "error": "output mismatch on shifted inputs (2x - mean|x|/2)",
                                "details": {**details, "trial": "shift"}, "compile_s": pair.compile_s}
                    shift = {"status": "passed", "max_abs_diff": details.get("max_abs_diff"),
                             "rel_err": details.get("rel_err")}
        except Exception as e:
            status = "cuda_error" if is_cuda_error(e) else "runtime_error"
            return {"ok": False, "status": status, "error": traceback.format_exc(), "trial": "shift",
                    "compile_s": pair.compile_s}
    result = {
        "ok": True,
        "status": "success",
        "trials": trials,
        "max_abs_diff": worst,
        "max_rel_err": worst_rel,
        "shift_trial": shift,
        "weights_synced": pair.weights_synced,
        "compile_s": pair.compile_s,
        "run_s": round(time.perf_counter() - t0, 3),
        "ref_cached": cached,
        "device": str(pair.device),
    }
    if on_cuda:
        try:
            result["kernels"] = forward_kernels(pair.model_gen, inputs)
        except Exception as e:  # noqa: BLE001 - the trace is informative only
            result["kernels_error"] = f"{type(e).__name__}: {e}"
    return result


def forward_kernels(model: Any, inputs: Any) -> list[dict[str, Any]]:
    """The CUDA kernels one forward runs (profiler activity trace, no counters needed): name, calls and
    GPU time, most time first."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad():
        call_model(model, inputs)  # warm
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            call_model(model, inputs)
            torch.cuda.synchronize()
    kernels: dict[str, dict[str, Any]] = {}
    for event in prof.events():
        if event.device_type.name != "CUDA" or event.name.startswith(("Memcpy", "Memset")):
            continue
        k = kernels.setdefault(event.name, {"name": event.name[:300], "calls": 0, "us": 0.0})
        k["calls"] += 1
        k["us"] += float(getattr(event, "device_time_total", 0.0) or 0.0)
    return sorted(kernels.values(), key=lambda k: -k["us"])
