"""Evaluation on a real GPU with a load_inline CUDA kernel (skipped without an idle GPU)."""

import asyncio
import shutil

import pytest
from conftest import write

from specgen.control.orchestrator import resolve_cuda_arch
from specgen.evaluation.executor import GpuSlot, LocalEvaluator
from specgen.evaluation.jobs import EvalJob

pytestmark = pytest.mark.gpu

TASK = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)


def get_inputs():
    return [torch.randn(4096, 4096)]


def get_init_inputs():
    return []
"""

CUDA_KERNEL = r'''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ y, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = BODY;
}
torch::Tensor relu_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    long n = x.numel();
    relu_kernel<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

ext = load_inline(name="NAME", cpp_sources="torch::Tensor relu_cuda(torch::Tensor x);",
                  cuda_sources=cuda_source, functions=["relu_cuda"], verbose=False)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.relu_cuda(x.contiguous())


def get_inputs():
    return [torch.randn(4096, 4096)]


def get_init_inputs():
    return []
'''


SPLIT_KERNEL = r'''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ y, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(x[i], 0.f);
}
void relu_launch(const float* x, float* y, long n, cudaStream_t stream) {
    relu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
}
"""

cpp_source = """
#include <ATen/cuda/CUDAContext.h>
void relu_launch(const float* x, float* y, long n, cudaStream_t stream);
torch::Tensor relu_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    relu_launch(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), at::cuda::getCurrentCUDAStream());
    return y;
}
"""

ext = load_inline(name="specgen_test_bare", cpp_sources=cpp_source, cuda_sources=cuda_source,
                  functions=["relu_cuda"], verbose=False)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.relu_cuda(x.contiguous())


def get_inputs():
    return [torch.randn(4096, 4096)]


def get_init_inputs():
    return []
'''


CP_ASYNC_KERNEL = r'''
__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ y, long n) {
    __shared__ float buf[256];
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(&buf[threadIdx.x]));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(x + i) : "memory");
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    if (i < n) y[i] = fmaxf(buf[threadIdx.x], 0.f);
}
'''


def kernel(body: str, name: str) -> str:
    return CUDA_KERNEL.replace("BODY", body).replace("NAME", name)


def run_job(kind, task, work_dir, gpu, role, **params):
    job = EvalJob(kind=kind, task_kind="kernelbench", task_path=str(task), kernel_path=str(work_dir / "kernel.py"),
                  work_dir=str(work_dir), params=params)
    evaluator = LocalEvaluator(timeout_s=900, cuda_arch=resolve_cuda_arch("auto", [gpu]))
    return asyncio.run(evaluator.run(job, GpuSlot(gpu, role)))


@pytest.fixture
def gpu_task(tmp_path):
    return write(tmp_path / "level1" / "relu.py", TASK)


def test_validate_and_benchmark_cuda_kernel(tmp_path, gpu_task, idle_gpu):
    work = tmp_path / "k"
    write(work / "kernel.py", kernel("x[i] > 0.f ? x[i] : 0.f", "specgen_test_relu"))
    validation = run_job("validate", gpu_task, work, idle_gpu, "validator", trials=2)
    assert validation["ok"], validation
    assert validation["device"] == "cuda" and validation["gpu"] == idle_gpu
    bench = run_job("benchmark", gpu_task, work, idle_gpu, "profiler", warmup=5, iters=20, rounds=3)
    assert bench["ok"], bench
    assert bench["speedup"] > 0.1 and bench["device"] == "cuda"
    assert bench["compile_s"] < validation["compile_s"]  # the build from validation is reused


def test_unwritten_output_is_rejected(tmp_path, gpu_task, idle_gpu):
    """The kernel covers half of the elements; the rest of torch::empty_like must not pass as
    stale data (the validator fills the blocks the output will reuse with NaN)."""
    work = tmp_path / "k"
    write(work / "kernel.py", kernel("fmaxf(x[i], 0.f)", "specgen_test_half").replace(
        "relu_kernel<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n)",
        "relu_kernel<<<(n / 2 + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n / 2)"))
    validation = run_job("validate", gpu_task, work, idle_gpu, "validator", trials=2)
    assert validation["status"] == "numerical_error", validation
    assert validation["details"]["reason"] == "value_mismatch"
    assert all(s["gen"] != s["gen"] for s in validation["details"]["sample"])  # NaN: the poisoned block


def test_cpu_build_is_reused_by_validation(tmp_path, gpu_task, idle_gpu):
    """A builder slot compiles without a GPU; validation on the GPU then loads that build."""
    work = tmp_path / "k"
    write(work / "kernel.py", kernel("fmaxf(x[i], 0.f)", "specgen_test_prebuilt"))
    arch = resolve_cuda_arch("auto", [idle_gpu])
    evaluator = LocalEvaluator(timeout_s=900, cuda_arch=arch, build_arch=arch.get(idle_gpu), build_jobs=4)
    job = lambda kind, **params: EvalJob(kind=kind, task_kind="kernelbench", task_path=str(gpu_task),  # noqa: E731
                                         kernel_path=str(work / "kernel.py"), work_dir=str(work), params=params)
    build = asyncio.run(evaluator.run(job("build"), GpuSlot(None, "builder")))
    assert build["ok"] and build["status"] == "built", build
    validation = asyncio.run(evaluator.run(job("validate", trials=1), GpuSlot(idle_gpu, "validator")))
    assert validation["ok"], validation
    assert validation["compile_s"] < build["compile_s"] / 3, (build["compile_s"], validation["compile_s"])


def test_fast_builds_and_fallback(tmp_path, gpu_task, idle_gpu):
    """Torch-free cuda_sources (wrappers in cpp_sources) build bare, typical ones with the slim prelude,
    main.cpp with the PCH; code that needs the full headers falls back to the source as written, and
    validation reuses whichever build the builder made."""
    import json

    arch = resolve_cuda_arch("auto", [idle_gpu])
    pch_dir = tmp_path / "pch"
    evaluator = LocalEvaluator(timeout_s=900, cuda_arch=arch, build_arch=arch.get(idle_gpu), build_jobs=4,
                               extra_env={"SPECGEN_PCH_DIR": str(pch_dir)})
    plain = kernel("fmaxf(x[i], 0.f)", "specgen_test_slim").replace(
        "#include <torch/extension.h>", "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>")
    needs_nn = kernel("fmaxf(x[i], 0.f)", "specgen_test_fallback").replace(
        "auto y = torch::empty_like(x);", "auto y = torch::nn::functional::relu(x) * 0 + torch::empty_like(x);")
    for name, source, choice in (("specgen_test_bare", SPLIT_KERNEL, "bare"), ("specgen_test_slim", plain, "slim"),
                                 ("specgen_test_fallback", needs_nn, "original")):
        work = tmp_path / name
        write(work / "kernel.py", source)
        job = lambda kind, **params: EvalJob(kind=kind, task_kind="kernelbench", task_path=str(gpu_task),  # noqa: E731
                                             kernel_path=str(work / "kernel.py"), work_dir=str(work), params=params)
        build = asyncio.run(evaluator.run(job("build"), GpuSlot(None, "builder")))
        assert build["status"] == "built", build
        marker = json.loads((work / "torch_extensions" / "specgen_fastbuild.json").read_text())
        assert marker == {name: choice}
        validation = asyncio.run(evaluator.run(job("validate", trials=1), GpuSlot(idle_gpu, "validator")))
        assert validation["ok"] and validation["compile_s"] < 5, validation
    assert [p.name for p in pch_dir.glob("*/ok")] == ["ok"]  # built once, checked, used by both builds
    shutil.rmtree(pch_dir)  # about 0.7 GB


def test_architecture_specific_code_compiles(tmp_path, gpu_task, idle_gpu):
    """cp.async needs sm_80+; an inherited multi-arch list including sm_75 rejects it."""
    work = tmp_path / "k"
    source = kernel("0.f", "specgen_test_cp_async")
    start = source.index("__global__")
    end = source.index("torch::Tensor relu_cuda")
    write(work / "kernel.py", source[:start] + CP_ASYNC_KERNEL.strip() + "\n" + source[end:])
    result = run_job("validate", gpu_task, work, idle_gpu, "validator", trials=1)
    assert result["ok"], result


def test_illegal_address_is_contained(tmp_path, gpu_task, idle_gpu):
    bad = tmp_path / "bad"
    write(bad / "kernel.py", kernel("((const float*)0)[i]", "specgen_test_bad"))
    result = run_job("validate", gpu_task, bad, idle_gpu, "validator", trials=1)
    # some drivers (B300, 590) do not fault on the null read: the output is garbage instead
    assert not result["ok"] and result["status"] in ("cuda_error", "runtime_error", "numerical_error"), result
    good = tmp_path / "good"
    write(good / "kernel.py", kernel("fmaxf(x[i], 0.f)", "specgen_test_good"))
    assert run_job("validate", gpu_task, good, idle_gpu, "validator", trials=1)["ok"]


@pytest.mark.ncu
@pytest.mark.skipif(shutil.which("ncu") is None, reason="ncu not on PATH")
def test_ncu_profiles_each_kernel_of_one_forward(tmp_path, gpu_task, idle_gpu):
    from specgen.config import DEFAULT_NCU_METRICS, DEFAULT_NCU_SECTIONS
    from specgen.generation.ncu_format import format_ncu_profile

    work = tmp_path / "k"
    two_kernels = kernel("x[i] > 0.f ? x[i] : 0.f", "specgen_test_relu_ncu").replace(
        "return ext.relu_cuda(x.contiguous())", "return ext.relu_cuda(x.contiguous()) * 1.0")  # + a torch mul kernel
    write(work / "kernel.py", two_kernels)
    result = run_job("ncu", gpu_task, work, idle_gpu, "profiler", metrics=list(DEFAULT_NCU_METRICS),
                     sections=list(DEFAULT_NCU_SECTIONS), timeout_s=300)
    if result["status"] == "ncu_unavailable":
        pytest.skip(result["error"])
    assert result["ok"] and result["selection"] == "nvtx" and not result["degraded"], result
    kernels = result["profile"]["kernels"]
    assert sorted(k["launches"] for k in kernels) == [1, 1]  # one forward: each kernel once
    assert any(k["name"] == "relu_kernel" for k in kernels) and abs(sum(k["share"] for k in kernels) - 1) < 1e-3
    for k in kernels:
        assert k["metrics"]["sm__throughput.avg.pct_of_peak_sustained_elapsed"] >= 0 and k["rules"]
    assert result["profile"]["units"]["gpu__time_duration.sum"].startswith("n")  # base units: nanoseconds
    assert "- `relu_kernel` (" in format_ncu_profile(result["profile"], 7000)


def test_validation_lists_the_kernels_a_forward_runs(tmp_path, gpu_task, idle_gpu):
    from specgen.control.compliance import library_kernels

    own = tmp_path / "own"
    write(own / "kernel.py", kernel("x[i] > 0.f ? x[i] : 0.f", "specgen_test_relu_trace"))
    result = run_job("validate", gpu_task, own, idle_gpu, "validator", trials=1)
    assert result["ok"] and [k["name"] for k in result["kernels"]][0].startswith("relu_kernel")
    assert library_kernels(k["name"] for k in result["kernels"]) == []
    cheat = tmp_path / "cheat"
    write(cheat / "kernel.py", kernel("x[i]", "specgen_test_relu_cheat").replace(
        "return ext.relu_cuda(x.contiguous())",
        "return torch.relu(ext.relu_cuda(x.contiguous()) @ torch.eye(x.shape[1], device=x.device))"))
    result = run_job("validate", gpu_task, cheat, idle_gpu, "validator", trials=1)
    assert result["ok"] and library_kernels(k["name"] for k in result["kernels"]), result.get("kernels")
