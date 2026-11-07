"""Timing: per-iteration latency of the generated model and the reference.

Each round times ``iters`` back-to-back forwards with CUDA events (wall clock on
CPU); reference and kernel rounds alternate, and the median round is reported.
"""

from __future__ import annotations

import statistics
import time
import traceback
from typing import Any, Callable

from specgen.evaluation.jobs import EvalJob
from specgen.evaluation.modules import EvalFailure, call_model, is_cuda_error, load_pair, make_inputs


def _round_timer(on_cuda: bool) -> Callable[[Callable[[], Any], int], float]:
    import torch

    def cuda_round(fn: Callable[[], Any], iters: int) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iters

    def cpu_round(fn: Callable[[], Any], iters: int) -> float:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) * 1e3 / iters

    return cuda_round if on_cuda else cpu_round


def run(job: EvalJob) -> dict[str, Any]:
    import torch

    p = job.params
    seed = int(p.get("seed", 0))
    warmup, iters, rounds = int(p.get("warmup", 10)), int(p.get("iters", 20)), int(p.get("rounds", 5))
    try:
        pair = load_pair(job.task_path, job.kernel_path, seed)
    except EvalFailure as failure:
        return failure.result

    on_cuda = pair.device.type == "cuda"
    time_round = _round_timer(on_cuda)
    try:
        torch.manual_seed(seed)
        inputs = make_inputs(pair.reference, pair.device)
        run_ref = lambda: call_model(pair.model_ref, inputs)  # noqa: E731
        run_gen = lambda: call_model(pair.model_gen, inputs)  # noqa: E731
        ref_ms, gen_ms = [], []
        with torch.no_grad():
            for _ in range(warmup):
                run_ref()
                run_gen()
            if on_cuda:
                torch.cuda.synchronize()
            for _ in range(rounds):
                ref_ms.append(time_round(run_ref, iters))
                gen_ms.append(time_round(run_gen, iters))
    except Exception as e:
        status = "cuda_error" if is_cuda_error(e) else "runtime_error"
        return {"ok": False, "status": status, "error": traceback.format_exc(), "compile_s": pair.compile_s}

    latency, ref_latency = statistics.median(gen_ms), statistics.median(ref_ms)
    return {
        "ok": True,
        "status": "success",
        "latency_ms": latency,
        "ref_latency_ms": ref_latency,
        "speedup": ref_latency / latency if latency > 0 else None,
        "rounds_ms": gen_ms,
        "ref_rounds_ms": ref_ms,
        "warmup": warmup,
        "iters": iters,
        "device": str(pair.device),
        "device_name": torch.cuda.get_device_name(0) if on_cuda else "cpu",
        "compile_s": pair.compile_s,
    }
