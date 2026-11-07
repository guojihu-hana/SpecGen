import importlib.util
import json
import sys
from pathlib import Path

import pytest

from specgen.config import parse_args
from specgen.llm.profiles import load_profiles

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ae():
    spec = importlib.util.spec_from_file_location("ae_driver", ROOT / "ae" / "ae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


def test_every_task_is_in_kernelbench(ae):
    assert [f"T{i}" for i in range(1, 21)] == [name for name in ae.TASKS if name.startswith("T")]
    missing = [name for name, path in ae.TASKS.items() if not (ROOT / "KernelBench" / path).is_file()]
    assert not missing
    for stage, (_, tasks, _, _) in ae.STAGES.items():
        assert set(tasks) <= set(ae.TASKS), stage


def test_cpu_ranges(ae):
    assert ae.cpu_ranges([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert ae.cpu_ranges([3]) == "3"


@pytest.mark.parametrize("model", ["deepseek", "glm-api", "glm"])
@pytest.mark.parametrize("method", ["specgen", "cudaforge"])
def test_the_runs_flags_parse(ae, method, model, tmp_path):
    label, flags = ae.METHODS[method]
    profile, endpoint = ae.MODELS[model]
    cfg = parse_args(["--task", str(ROOT / "KernelBench" / ae.TASKS["T1"]), "--models-file", str(ROOT / "ae" / "models.yaml"),
                      "--model", profile, "--endpoint", endpoint, "--gpus", "0,1", "--gpu-type", "auto", "--cuda-arch",
                      "auto", "--gpu-allocation", "dynamic", "--validations-per-gpu", "8", "--max-concurrent-tasks", "1",
                      "--max-iterations", "2", "--output-root", str(tmp_path), "--run-id", "r", *flags],
                     discover_gpus=lambda: ["0", "1"])
    assert cfg.spec.enabled == (method == "specgen")
    if method == "specgen":
        assert cfg.termination.enabled and cfg.termination.criteria == "hist_avg"


def test_the_glm_api_profile_stays_within_the_api_limits():
    profile = load_profiles(ROOT / "ae" / "models.yaml")["GLM-5.1"]
    endpoint = profile.endpoint("api").resolve({"SPECGEN_GLM_API_URL": "https://api.z.ai/api/paas/v4",
                                                "SPECGEN_GLM_API_KEY": "k", "SPECGEN_GLM_API_MODEL": "glm-5.1"})
    assert (endpoint.model, endpoint.prefill_style) == ("glm-5.1", None)  # spec generations reprompt
    assert profile.mode_params("spec", endpoint) == {"thinking": {"type": "disabled"}}
    assert max(profile.sampling.spec_temperature_for(i) for i in range(8)) <= 1.0


def test_the_report_orders_the_tasks(tmp_path):
    spec = importlib.util.spec_from_file_location("ae_report", ROOT / "ae" / "report.py")
    report = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = report
    spec.loader.exec_module(report)
    dirs = []
    for name, task in [("mlp", "level3/1_MLP.py"), ("T10", ""), ("sqmm", "level1/1_Square_matrix_multiplication_.py"),
                       ("T2", ""), ("rmsnorm", "level1/36_RMSNorm_.py")]:
        run_dir = tmp_path / f"ref-{name}-specgen"
        run_dir.mkdir()
        if task:
            (run_dir / "config.json").write_text(json.dumps({"run": {"tasks": [f"/x/KernelBench/{task}"]}}))
        dirs.append(run_dir)
    assert [d.name.split("-")[1] for d in sorted(dirs, key=report.task_order)] == ["T2", "T10", "sqmm", "rmsnorm", "mlp"]
