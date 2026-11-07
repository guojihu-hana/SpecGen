import asyncio
import json
import time

import pytest
from conftest import CORRECT_RELU, WRONG_RELU, write

from specgen.evaluation.executor import GpuSlot, LocalEvaluator
from specgen.evaluation.jobs import EvalJob

CPU = GpuSlot(None, "validator", 0)


def run_job(kind, task, kernel_source, work_dir, *, timeout=180, task_kind="kernelbench", **params):
    kernel = write(work_dir / "kernel.py", kernel_source)
    job = EvalJob(kind=kind, task_kind=task_kind, task_path=str(task), kernel_path=str(kernel),
                  work_dir=str(work_dir), params=params)
    return asyncio.run(LocalEvaluator(timeout_s=timeout).run(job, CPU))


def test_validate_correct_kernel(tmp_path, relu_task):
    result = run_job("validate", relu_task, CORRECT_RELU, tmp_path / "k", trials=2)
    assert result["ok"] and result["status"] == "success", result
    assert result["trials"] == 2 and result["device"] == "cpu" and result["gpu"] is None
    assert json.loads((tmp_path / "k" / "validate.result.json").read_text())["status"] == "success"
    assert (tmp_path / "k" / "validate.log").exists()


def test_weights_are_shared(tmp_path, linear_task):
    result = run_job("validate", linear_task, linear_task.read_text().replace("torch.relu(", "torch.clamp_min(").replace(
        "self.fc(x))", "self.fc(x), 0.0)"), tmp_path / "k")
    assert result["ok"] and result["weights_synced"], result


def test_wrong_kernel_is_a_numerical_error(tmp_path, relu_task):
    result = run_job("validate", relu_task, WRONG_RELU, tmp_path / "k")
    assert result["status"] == "numerical_error"
    assert result["details"]["reason"] == "value_mismatch" and result["details"]["mismatches"] > 0


UNIT_TASK = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return OP


def get_inputs():
    return [torch.rand(64, 128)]


def get_init_inputs():
    return []
"""


@pytest.mark.parametrize("op, kernel_expr, status, shift", [
    ("x * 2", "x * 2", "success", "passed"),
    # quantizes x as if it were always in [0, 1): within tolerance on get_inputs(), wrong on shifted inputs
    ("x * 2", "torch.round(x.clamp(0, 1) * 255) / 255 * 2", "numerical_error", None),
    ("torch.sqrt(x)", "torch.sqrt(x)", "success", "skipped"),  # sqrt of the shifted inputs is NaN
])
def test_shifted_inputs(tmp_path, op, kernel_expr, status, shift):
    task = write(tmp_path / "level0" / "3_Unit.py", UNIT_TASK.replace("OP", op))
    result = run_job("validate", task, UNIT_TASK.replace("OP", kernel_expr), tmp_path / "k", trials=2)
    assert result["status"] == status, result
    if shift:
        assert result["shift_trial"]["status"] == shift
    else:
        assert result["details"]["trial"] == "shift"
    off = run_job("validate", task, UNIT_TASK.replace("OP", kernel_expr), tmp_path / "k2", trials=2, shift_trial=False)
    assert off["status"] == "success" and off["shift_trial"] == {"status": "off"}


def test_tiny_outputs_need_a_small_relative_error():
    """Softmax over 393216 columns: values near 3e-6, far below atol.  Leaving most rows
    unwritten passes allclose; the relative error does not."""
    import torch

    from specgen.evaluation.validate import compare_outputs

    torch.manual_seed(0)
    ref = torch.softmax(torch.rand(64, 1 << 14), dim=1)
    partial = ref.clone()
    partial[4:] = 0
    ok, info = compare_outputs(ref, partial, 1e-2, 1e-2)
    assert not ok and info["reason"] == "relative_error" and info["rel_err"] > 0.9
    assert torch.allclose(ref, partial, atol=1e-2, rtol=1e-2)
    ok, info = compare_outputs(ref, ref * (1 + 1e-3 * torch.randn_like(ref)), 1e-2, 1e-2)
    assert ok and info["rel_err"] < 2e-3
    ok, info = compare_outputs(torch.zeros(8), torch.full((8,), 1e-3), 1e-2, 1e-2)
    assert ok and info["rel_err"] is None  # an all-zero reference is checked elementwise only
    gen = ref.clone()
    gen[0, 0] = float("nan")
    ok, info = compare_outputs(ref, gen, 1e-2, 1e-2)
    assert not ok and info["reason"] == "value_mismatch"
    ok, info = compare_outputs((ref, ref[:2]), (ref, partial[:2]), 1e-2, 1e-2)
    assert ok and info["rel_err"] == 0.0


def test_compile_error(tmp_path, relu_task):
    result = run_job("validate", relu_task, "def broken(:\n", tmp_path / "k")
    assert result["status"] == "compile_error" and "SyntaxError" in result["error"]


def test_benchmark_on_cpu(tmp_path, relu_task):
    result = run_job("benchmark", relu_task, CORRECT_RELU, tmp_path / "k", warmup=2, iters=5, rounds=3)
    assert result["ok"], result
    assert result["speedup"] > 0 and result["latency_ms"] > 0 and len(result["rounds_ms"]) == 3


def test_timeout_kills_the_child(tmp_path, relu_task):
    hanging = CORRECT_RELU.replace("return torch.clamp_min(x, 0.0)", "import time; time.sleep(120)")
    t0 = time.monotonic()
    result = run_job("validate", relu_task, hanging, tmp_path / "k", timeout=15)
    assert result["status"] == "timeout" and time.monotonic() - t0 < 60


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_timeout_kills_detached_grandchildren(tmp_path, relu_task):
    escaping = (
        "import os, subprocess, time\n"
        "p = subprocess.Popen(['sleep', '300'], start_new_session=True)\n"
        "open(os.path.join(os.path.dirname(__file__), 'grandchild.pid'), 'w').write(str(p.pid))\n"
        "time.sleep(300)\n"
    )
    result = run_job("validate", relu_task, escaping, tmp_path / "k", timeout=15)
    assert result["status"] == "timeout"
    pid = int((tmp_path / "k" / "grandchild.pid").read_text())
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(pid)


def test_unsupported_task_kind(tmp_path, relu_task):
    result = run_job("validate", relu_task, CORRECT_RELU, tmp_path / "k", task_kind="triton")
    assert result["status"] == "unsupported"
