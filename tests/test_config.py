import json

import pytest

from specgen.config import DEFAULT_NCU_METRICS, parse_args


def test_defaults(relu_task):
    cfg = parse_args(["--task", str(relu_task), "--cuda-arch", "9.0a"], discover_gpus=lambda: ["0", "1"])
    assert cfg.pool.gpus == ["0", "1"]
    assert cfg.eval.cuda_arch == "9.0a" and cfg.eval.build_jobs is None
    assert cfg.eval.atol == cfg.eval.rtol == 1e-2
    assert cfg.pool.allocation == "dynamic" and cfg.pool.build_workers is None
    assert cfg.spec.enabled and cfg.spec.k is None and cfg.spec.prompt_mode == "auto"
    assert cfg.spec.schedule == "demand" and cfg.spec.target_extra == 1.0 and cfg.spec.max_per_iteration == 0
    assert cfg.spec.target_factor == 1.5 and not cfg.run.resume
    assert parse_args(["--task", str(relu_task), "--gpus", "none", "--resume", "--run-id", "r1"]).run.resume
    assert cfg.spec.min_gap_tokens == cfg.spec.first_tokens == 500 and cfg.spec.best_ncu_wait_s == 30.0
    assert cfg.pool.validations_per_gpu == 8
    assert cfg.llm.model == "GLM-5.1-FP8" and cfg.llm.endpoint == "selfhost"
    assert cfg.run.tasks == [relu_task.resolve()]
    assert cfg.run.max_iterations == 10 and cfg.run.feedback_top_k == 3
    assert cfg.eval.ncu_metrics == list(DEFAULT_NCU_METRICS)
    assert cfg.run.run_id
    json.dumps(cfg.to_dict())


def test_gpu_lists(relu_task):
    assert parse_args(["--task", str(relu_task), "--gpus", "none"]).pool.gpus == []
    assert parse_args(["--task", str(relu_task), "--gpus", "2, 3", "--cuda-arch", "9.0a"]).pool.gpus == ["2", "3"]


def test_level_dir(tmp_path, relu_task, linear_task):
    cfg = parse_args(["--level-dir", str(relu_task.parent), "--gpus", "none"])
    assert [p.name for p in cfg.run.tasks] == ["1_ReLU.py", "2_Linear.py"]


@pytest.mark.parametrize("extra", [
    ["--gpu-allocation", "greedy"], ["--max-iterations", "0"], ["--gpu-type", "X100"],
    ["--cuda-arch", "sm_90"], ["--cuda-arch", "9"], ["--build-jobs", "0"], ["--build-jobs", "many"],
    ["--spec-k", "x"], ["--build-workers", "-1"], ["--spec-k-min", "3", "--spec-k-max", "2"],
    ["--spec-prompt-mode", "both"], ["--fallback-trigger-tokens", "0"], ["--spec-schedule", "eager"],
    ["--spec-min-gap-tokens", "-1"], ["--best-ncu-wait-s", "-5"], ["--spec-max-per-iteration", "-1"],
    ["--resume"],  # needs the --run-id of the run to continue
])
def test_rejected(relu_task, extra):
    with pytest.raises(SystemExit):
        parse_args(["--task", str(relu_task), "--gpus", "none", *extra])


def test_cuda_arch_required_with_gpus(relu_task):
    with pytest.raises(SystemExit):
        parse_args(["--task", str(relu_task), "--gpus", "0"])
    assert parse_args(["--task", str(relu_task), "--gpus", "none"]).eval.cuda_arch is None
    for arch in ("auto", "inherit", "10.3a", "9.0a;10.0a", "8.0 9.0+PTX"):
        assert parse_args(["--task", str(relu_task), "--gpus", "0", "--cuda-arch", arch]).eval.cuda_arch == arch


def test_build_jobs(relu_task):
    assert parse_args(["--task", str(relu_task), "--gpus", "none", "--build-jobs", "8"]).eval.build_jobs == 8


def test_spec_and_pool_flags(relu_task):
    cfg = parse_args(["--task", str(relu_task), "--gpus", "none", "--spec-k", "3", "--build-workers", "0",
                      "--gpu-allocation", "fifo", "--no-enable-spec-gen", "--trigger-after-tokens", "200",
                      "--max-inflight-spec-gen", "6"])
    assert cfg.spec.k == 3 and not cfg.spec.enabled and cfg.spec.after_tokens == 200 and cfg.spec.max_inflight == 6
    assert cfg.pool.build_workers == 0 and cfg.pool.allocation == "fifo"
    cfg = parse_args(["--task", str(relu_task), "--gpus", "none", "--spec-schedule", "triggers", "--spec-target-extra",
                      "2", "--spec-min-gap-tokens", "300", "--spec-first-tokens", "800", "--best-ncu-wait-s", "0",
                      "--spec-max-per-iteration", "40", "--spec-target-factor", "2"])
    assert (cfg.spec.schedule, cfg.spec.target_extra, cfg.spec.min_gap_tokens, cfg.spec.first_tokens,
            cfg.spec.best_ncu_wait_s, cfg.spec.max_per_iteration, cfg.spec.target_factor) == (
        "triggers", 2.0, 300, 800, 0.0, 40, 2.0)


def test_missing_task(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--task", str(tmp_path / "nope.py"), "--gpus", "none"])
    with pytest.raises(SystemExit):
        parse_args(["--gpus", "none"])
