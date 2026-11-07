import asyncio

import pytest
from conftest import CORRECT_RELU, ScriptedBackend, answer

from specgen.config import EvalConfig, PoolConfig, RunConfig, SpecGenConfig
from specgen.control import orchestrator
from specgen.control.orchestrator import (
    RunSetupError, resolve_build_jobs, resolve_cuda_arch, resolve_gpu_type, run_async,
)
from specgen.evaluation.executor import GpuSlot, LocalEvaluator
from specgen.evaluation.jobs import EvalJob
from specgen.generation.gpu_specs import normalize_gpu_name
from specgen.resources.gpus import arch_for, arch_runs_on, parse_arch_list


def test_arch_for():
    assert [arch_for(c) for c in ("8.0", "8.9", "9.0", "10.0", "10.3", "12.0")] == [
        "8.0", "8.9", "9.0a", "10.0a", "10.3a", "12.0a"]


def test_parse_arch_list():
    assert parse_arch_list("9.0a;10.3a") == ["9.0a", "10.3a"]
    assert parse_arch_list(" 8.0  9.0+PTX ") == ["8.0", "9.0+PTX"]
    for bad in ("", "sm_90", "9", "9.0b", "9.0a 10"):
        with pytest.raises(ValueError):
            parse_arch_list(bad)


@pytest.mark.parametrize("tokens, cap, ok", [
    (["9.0a"], "9.0", True),
    (["9.0a"], "10.0", False),
    (["10.3a"], "9.0", False),
    (["10.0a"], "10.3", False),
    (["10.0"], "10.3", True),
    (["8.0"], "8.6", True),
    (["8.6"], "8.0", False),
    (["9.0+PTX"], "10.3", True),
    (["9.0a+PTX"], "10.0", False),
    (["9.0a", "10.3a"], "10.3", True),
])
def test_arch_runs_on(tokens, cap, ok):
    assert arch_runs_on(tokens, cap) is ok


def test_resolve_cuda_arch(monkeypatch):
    monkeypatch.setattr(orchestrator, "compute_capabilities", lambda: {"0": "9.0", "1": "8.6"})
    assert resolve_cuda_arch("auto", ["0", "1", "7"]) == {"0": "9.0a", "1": "8.6"}
    assert resolve_cuda_arch("inherit", ["0"]) == {}
    assert resolve_cuda_arch(None, []) == {}
    assert resolve_cuda_arch("8.0;9.0a", ["0", "1"]) == {"0": "8.0 9.0a", "1": "8.0 9.0a"}
    assert resolve_cuda_arch("10.3a", ["7"]) == {"7": "10.3a"}  # capability unknown: trust the setting
    with pytest.raises(RunSetupError, match="use 9.0a"):
        resolve_cuda_arch("10.3a", ["0"])
    with pytest.raises(RunSetupError, match="GPU 1"):
        resolve_cuda_arch("9.0a", ["0", "1"])


def test_resolve_gpu_type(monkeypatch):
    assert resolve_gpu_type("B300", ["0"]) == "B300"
    with pytest.raises(RunSetupError, match="unknown"):
        resolve_gpu_type("X100", ["0"])
    monkeypatch.setattr(orchestrator, "detect_gpu_type", lambda gpus: None)
    with pytest.raises(RunSetupError, match="--gpu-type"):
        resolve_gpu_type("auto", ["0"])
    monkeypatch.setattr(orchestrator, "detect_gpu_type", lambda gpus: "H100")
    assert resolve_gpu_type("auto", ["0"]) == "H100"


def test_gpu_names():
    assert normalize_gpu_name("NVIDIA B300 SXM6 AC") == "B300"
    assert normalize_gpu_name("NVIDIA GB300") == "B300"
    assert normalize_gpu_name("NVIDIA B200") == "B200"
    assert normalize_gpu_name("NVIDIA H100 80GB HBM3") == "H100"
    assert normalize_gpu_name("NVIDIA H800") == "H800"
    assert normalize_gpu_name("NVIDIA H200") == "H200"
    assert normalize_gpu_name("NVIDIA A800-SXM4-80GB") == "A800"
    assert normalize_gpu_name("NVIDIA A100-SXM4-80GB") == "A100-80GB"


def test_resolve_build_jobs(monkeypatch):
    monkeypatch.setattr(orchestrator, "effective_cpu_count", lambda: 84)
    assert resolve_build_jobs(None, 10) == 8
    assert resolve_build_jobs(None, 1) == orchestrator.MAX_AUTO_BUILD_JOBS
    assert resolve_build_jobs(None, 100) == 2
    assert resolve_build_jobs(5, 10) == 5


def test_child_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.5 8.0 9.0+PTX")
    monkeypatch.setenv("MAX_JOBS", "99")
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")  # as NVIDIA's PyTorch containers set it
    evaluator = LocalEvaluator(cuda_arch={"3": "9.0a"}, build_jobs=6)
    job = EvalJob("validate", "kernelbench", "t.py", "k.py", str(tmp_path))
    pinned = evaluator._env(job, GpuSlot("3", "validator"))
    assert pinned["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] == "0"  # FP32 references stay FP32
    assert pinned["CUDA_VISIBLE_DEVICES"] == "3" and pinned["TORCH_CUDA_ARCH_LIST"] == "9.0a"
    assert pinned["TORCH_EXTENSIONS_DIR"] == str(tmp_path / "torch_extensions")
    assert pinned["PYTHONDONTWRITEBYTECODE"] == "1" and pinned["MAX_JOBS"] == "6"
    assert evaluator._env(job, GpuSlot("4", "validator"))["TORCH_CUDA_ARCH_LIST"] == "7.5 8.0 9.0+PTX"
    assert evaluator._env(job, GpuSlot(None, "validator"))["CUDA_VISIBLE_DEVICES"] == ""
    assert LocalEvaluator()._env(job, GpuSlot("3", "validator"))["MAX_JOBS"] == "99"


def config(tmp_path, task, **pool):
    return SpecGenConfig(
        pool=PoolConfig(gpus=[], validations_per_gpu=1, **pool),
        eval=EvalConfig(gpu_type="H100", ncu_top_n=0, correctness_trials=1, bench_warmup=1, bench_iters=2,
                        bench_rounds=1),
        run=RunConfig(tasks=[task], output_root=tmp_path / "runs", run_id="same", max_iterations=1),
    )


def test_run_directory_is_never_shared(tmp_path, relu_task):
    cfg = config(tmp_path, relu_task)
    asyncio.run(run_async(cfg, backend=ScriptedBackend([answer(CORRECT_RELU)])))
    with pytest.raises(RunSetupError, match="already exists"):
        asyncio.run(run_async(cfg, backend=ScriptedBackend([answer(CORRECT_RELU)])))


def test_bad_gpu_split_is_a_setup_error(tmp_path, relu_task):
    cfg = config(tmp_path, relu_task)
    cfg.pool.gpus, cfg.pool.num_validator_gpus, cfg.pool.num_profiler_gpus = ["0", "1"], 2, 1
    cfg.eval.cuda_arch = "inherit"
    with pytest.raises(RunSetupError, match="validator"):
        asyncio.run(run_async(cfg, backend=ScriptedBackend([])))
    assert not (tmp_path / "runs" / "same").exists()


def test_gpus_need_a_cuda_arch(tmp_path, relu_task):
    cfg = config(tmp_path, relu_task)
    cfg.pool.gpus = ["0", "1"]
    with pytest.raises(RunSetupError, match="cuda_arch"):
        asyncio.run(run_async(cfg, backend=ScriptedBackend([])))
    assert not (tmp_path / "runs" / "same").exists()


def test_unknown_gpu_type_is_a_setup_error(tmp_path, relu_task):
    cfg = config(tmp_path, relu_task)
    cfg.eval.gpu_type = "auto"
    with pytest.raises(RunSetupError, match="--gpu-type"):
        asyncio.run(run_async(cfg, backend=ScriptedBackend([])))
    assert not (tmp_path / "runs" / "same").exists()
