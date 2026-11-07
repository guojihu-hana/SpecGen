"""Run configuration: one typed object built from the command line."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from specgen.generation.gpu_specs import GPU_SPECS
from specgen.resources.gpus import ARCH_EXAMPLES, parse_arch_list

ALLOCATIONS = ("fixed", "dynamic", "fifo")
CRITERIA = ("first_valid", "best", "hist_avg")
HISTORY_PROMPTS = ("topk", "summary", "topk+summary")

# ncu sections whose rules (bottlenecks with estimated speedups) go into the feedback
DEFAULT_NCU_SECTIONS: tuple[str, ...] = (
    "SpeedOfLight",
    "SpeedOfLight_RooflineChart",
    "ComputeWorkloadAnalysis",
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Tables",
    "LaunchStats",
    "Occupancy",
    "SchedulerStats",
    "WarpStateStats",
    "InstructionStats",
    "SourceCounters",
)

# collected on top of the sections (so the minimal fallback profile has them too); generation.ncu_format
# shows them per kernel
DEFAULT_NCU_METRICS: tuple[str, ...] = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__block_size",
    "launch__grid_size",
    "sm__inst_executed.sum",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
)


@dataclass
class LLMConfig:
    model: str = "GLM-5.1-FP8"
    endpoint: str = "selfhost"
    models_file: Optional[Path] = None  # None: see llm.profiles.default_models_file
    tokenizer: Optional[str] = None  # None: the profile's tokenizer
    temperature: Optional[float] = None  # None: the profile's sampling
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    request_timeout_s: float = 3600.0
    max_retries: int = 2
    max_continuations: int = 1


@dataclass
class PoolConfig:
    gpus: list[str] = field(default_factory=list)  # empty: evaluate on CPU
    allocation: str = "dynamic"
    num_validator_gpus: Optional[int] = None  # fixed: the split; dynamic: the initial split
    num_profiler_gpus: Optional[int] = None
    validations_per_gpu: int = 8
    build_workers: Optional[int] = None  # CPU build slots; None: from the CPU count; 0: build on the GPU slots
    job_timeout_s: float = 600.0
    remote_workers: list[str] = field(default_factory=list)  # URLs of `specgen worker` servers


@dataclass
class EvalConfig:
    gpu_type: str = "auto"
    # TORCH_CUDA_ARCH_LIST for kernel builds: an explicit list (e.g. "9.0a"), "auto" (each GPU's
    # compute capability) or "inherit" (keep the environment); required when evaluating on GPUs
    cuda_arch: Optional[str] = None
    build_jobs: Optional[int] = None  # MAX_JOBS per kernel build; None: CPUs / concurrent build slots
    fast_includes: bool = True  # build cuda_sources with lighter torch headers first (see evaluation.fastbuild)
    build_pch: bool = True  # compile main.cpp against a cached precompiled <torch/extension.h>
    seed: int = 0
    atol: float = 1e-2
    rtol: float = 1e-2
    correctness_trials: int = 3
    # one more trial on shifted, rescaled floating inputs (2x - mean|x|/2); a kernel that only works for
    # the range get_inputs() happens to produce fails it
    shift_trial: bool = True
    bench_warmup: int = 10
    bench_iters: int = 20
    bench_rounds: int = 5
    # kernels profiled per iteration, at most: the ones later prompts show (control.loop._ncu_wanted)
    ncu_top_n: int = 6
    ncu_binary: str = "ncu"
    ncu_metrics: list[str] = field(default_factory=lambda: list(DEFAULT_NCU_METRICS))
    ncu_sections: list[str] = field(default_factory=lambda: list(DEFAULT_NCU_SECTIONS))  # empty: metrics only
    ncu_forward: int = 3  # the forward (1-based) whose kernels ncu profiles, once each
    ncu_launch_count: int = 64  # kernels of that forward profiled, at most
    # an ncu run past this is killed and repeated with the minimal sections (evaluation.ncu)
    ncu_timeout_s: float = 120.0
    # before building a prompt, wait this long for the ncu jobs of the kernels it shows
    ncu_wait_s: float = 300.0


@dataclass
class SpecConfig:
    enabled: bool = True
    trigger_config: Optional[Path] = None  # None: see control.triggers.default_triggers_file
    after_tokens: Optional[int] = None  # override the trigger config
    fallback_tokens: Optional[int] = None
    k: Optional[int] = None  # spec generations per trigger; None: adaptive (SpecBudget)
    k_min: int = 1
    k_max: int = 4  # also caps a fixed k; later specs of a batch sample hotter and rarely pass beyond 4
    max_inflight: int = 0  # spec LLM requests in flight across all tasks; 0: unlimited
    prompt_mode: str = "auto"  # auto | prefill | reprompt
    # when every generation slot is busy: "drop" launches fewer specs (possibly none); "replace" queues
    # them, and a newer trigger of the task replaces its queued (not yet started) specs
    queue: str = "drop"
    # "demand": one spec whenever the kernels expected at the profiler fall short (control.pacing);
    # "triggers": k specs at each keyword trigger of the reasoning
    schedule: str = "demand"
    # demand: expected kernels at the profiler, max(devices + target_extra, devices * target_factor), where
    # devices is the task's share of the profiler devices
    target_extra: float = 1.0
    target_factor: float = 1.5
    min_gap_tokens: int = 500  # demand: reasoning tokens between two launches
    first_tokens: int = 500  # demand: reasoning tokens before the first launch
    # demand: launches pause this long, at most, for the ncu profile of a kernel that meets the termination
    # criterion (without one: beats the best so far)
    best_ncu_wait_s: float = 30.0
    max_per_iteration: int = 0  # demand: spec launches per iteration and task, at most; 0: no limit
    prior_pass_rate: float = 0.4  # demand: share of specs reaching timing, before any finished


@dataclass
class TerminationConfig:
    enabled: bool = False
    criteria: Optional[str] = None  # first_valid | best | hist_avg
    margin: float = 0.03
    # a kernel that meets the criterion is timed this many times in all and must meet it each time
    confirm: int = 2
    # after the main kernel is evaluated, other candidates get at most this fraction of the
    # main generation time to finish; None waits for all of them
    grace_ratio: Optional[float] = 0.05


@dataclass
class RunConfig:
    tasks: list[Path] = field(default_factory=list)
    output_root: Path = Path("runs")
    run_id: str = ""
    max_iterations: int = 10
    feedback_top_k: int = 3  # the "topk" kernels: how many of the previous iteration's fastest
    # what earlier iterations add to the prompt besides the best kernel's code: "topk" the code of the
    # previous iteration's fastest kernels; "summary" a table of every approach tried (control.attempts);
    # "topk+summary" both
    history_prompt: str = "topk+summary"
    summary_max_approaches: int = 0  # 0: no limit
    summary_max_failures: int = 0
    summary_max_chars: int = 0
    max_concurrent_tasks: int = 4
    resume: bool = False  # continue the existing run run_id up to max_iterations


@dataclass
class SpecGenConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    spec: SpecConfig = field(default_factory=SpecConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        def plain(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value

        return plain(asdict(self))


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _auto_or_int(parser: argparse.ArgumentParser, flag: str, value: str, *, minimum: int) -> Optional[int]:
    if value == "auto":
        return None
    if not value.isdigit() or int(value) < minimum:
        parser.error(f"{flag} must be 'auto' or an integer >= {minimum}")
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="specgen run", description="Iteratively generate and optimize kernels.")

    g = p.add_argument_group("tasks")
    g.add_argument("--task", action="append", type=Path, default=[], help="task file (repeatable)")
    g.add_argument("--level-dir", type=Path, default=None, help="add every .py task in this directory")
    g.add_argument("--output-root", type=Path, default=Path("runs"))
    g.add_argument("--run-id", default="", help="default: current timestamp")
    g.add_argument("--resume", action="store_true",
                   help="continue the run --run-id up to --max-iterations, with the history it recorded (same tasks; "
                        "give the flags of the original run)")
    g.add_argument("--max-iterations", type=int, default=10)
    g.add_argument("--feedback-top-k", type=int, default=3,
                   help="the previous iteration's fastest kernels shown with topk prompts (the k of topk)")
    g.add_argument("--history-prompt", default="topk+summary", choices=list(HISTORY_PROMPTS),
                   help="besides the best kernel's code: topk = code of the previous iteration's fastest kernels; "
                        "summary = table of every approach tried so far and the recurring failures; "
                        "topk+summary = both (default)")
    g.add_argument("--summary-max-approaches", type=int, default=0, help="rows of the summary table (0: no limit)")
    g.add_argument("--summary-max-failures", type=int, default=0, help="failure kinds listed (0: no limit)")
    g.add_argument("--summary-max-chars", type=int, default=0, help="length of the summary (0: no limit)")
    g.add_argument("--max-concurrent-tasks", type=int, default=4)

    g = p.add_argument_group("llm")
    g.add_argument("--model", default="GLM-5.1-FP8", help="profile name in the models file")
    g.add_argument("--endpoint", default="selfhost", choices=["selfhost", "api"])
    g.add_argument("--models-file", type=Path, default=None)
    g.add_argument("--tokenizer", default=None, help="'generic' or a local tokenizer path (default: profile)")
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--top-p", type=float, default=None)
    g.add_argument("--max-tokens", type=int, default=None)
    g.add_argument("--request-timeout", type=float, default=3600.0)
    g.add_argument("--max-retries", type=int, default=2)
    g.add_argument("--max-continuations", type=int, default=1)

    g = p.add_argument_group("resources")
    g.add_argument("--gpus", default="auto", help="'auto', 'none' (CPU only) or a list such as 0,1,2")
    g.add_argument("--gpu-allocation", default="dynamic", choices=ALLOCATIONS,
                   help="fixed: fixed validator/profiler split; dynamic: move idle GPUs between the roles by "
                        "load; fifo: every GPU takes any job in arrival order")
    g.add_argument("--num-validator-gpus", type=int, default=None)
    g.add_argument("--num-profiler-gpus", type=int, default=None)
    g.add_argument("--validations-per-gpu", type=int, default=8)
    g.add_argument("--build-workers", default="auto",
                   help="concurrent kernel builds on CPU before validation ('auto': from the CPU count; "
                        "0: build inside validation)")
    g.add_argument("--job-timeout", type=float, default=600.0)
    g.add_argument("--remote-worker", action="append", default=[], metavar="URL",
                   help="add the GPUs of a `specgen worker` (repeatable; token in $SPECGEN_WORKER_TOKEN)")

    g = p.add_argument_group("evaluation")
    g.add_argument("--gpu-type", default="auto", choices=["auto", *sorted(GPU_SPECS)],
                   help="GPU spec injected into prompts (default: detect from the first GPU)")
    g.add_argument("--cuda-arch", default=None,
                   help=f"TORCH_CUDA_ARCH_LIST for kernel builds, required with GPUs: {ARCH_EXAMPLES}; "
                        "'auto' detects each GPU's compute capability, 'inherit' keeps the environment")
    g.add_argument("--build-jobs", default="auto",
                   help="parallel compile jobs per kernel build (MAX_JOBS); 'auto' divides the CPUs "
                        "among the slots that may build at once")
    g.add_argument("--fast-cuda-includes", action=argparse.BooleanOptionalAction, default=True,
                   help="build cuda_sources with a slim torch prelude, then without <torch/extension.h>, then as "
                        "written (the first that builds is used); nvcc 44 s -> 15 s on typical kernels")
    g.add_argument("--build-pch", action=argparse.BooleanOptionalAction, default=True,
                   help="compile main.cpp against a precompiled <torch/extension.h> cached in $SPECGEN_PCH_DIR "
                        "(default ~/.cache/specgen/pch, about 0.7 GB per compiler flag set)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--atol", type=float, default=1e-2)
    g.add_argument("--rtol", type=float, default=1e-2)
    g.add_argument("--correctness-trials", type=int, default=3)
    g.add_argument("--validation-shift-trial", action=argparse.BooleanOptionalAction, default=True,
                   help="validate once more on shifted, rescaled floating inputs (skipped when the reference "
                        "gives NaN/Inf or fails on them)")
    g.add_argument("--bench-warmup", type=int, default=10)
    g.add_argument("--bench-iters", type=int, default=20)
    g.add_argument("--bench-rounds", type=int, default=5)
    g.add_argument("--ncu-top-n", type=int, default=6,
                   help="kernels profiled per iteration, at most: the ones later prompts show (0 disables ncu)")
    g.add_argument("--ncu-binary", default="ncu")
    g.add_argument("--ncu-metrics", default="", help="comma-separated (default: built-in list)")
    g.add_argument("--ncu-sections", default="",
                   help="comma-separated ncu sections whose rule results go into the feedback "
                        "(default: built-in list; 'none': metrics only)")
    g.add_argument("--ncu-forward", type=int, default=3,
                   help="profile every kernel of this forward pass (1-based), once each")
    g.add_argument("--ncu-launch-count", type=int, default=64, help="kernels of that forward profiled, at most")
    g.add_argument("--ncu-timeout-s", type=float, default=120.0,
                   help="an ncu run past this is killed and repeated with a minimal set of sections (0: no limit)")
    g.add_argument("--ncu-wait-s", type=float, default=300.0,
                   help="before a prompt is built, wait up to this long for the ncu jobs of the kernels it shows "
                        "(0: do not wait)")

    g = p.add_argument_group("speculative generation")
    g.add_argument("--enable-spec-gen", action=argparse.BooleanOptionalAction, default=True,
                   help="launch spec generations from triggers in the main reasoning (default: on)")
    g.add_argument("--trigger-config", type=Path, default=None, help="default: configs/triggers.yaml")
    g.add_argument("--trigger-after-tokens", type=int, default=None,
                   help="reasoning tokens between a keyword and the prefix cut (overrides the config)")
    g.add_argument("--fallback-trigger-tokens", type=int, default=None,
                   help="reasoning tokens without a trigger before one fires anyway (overrides the config)")
    g.add_argument("--spec-k", default="auto",
                   help="spec generations per trigger: 'auto' sizes it to keep the evaluation pipeline busy, "
                        "or a fixed number")
    g.add_argument("--spec-k-min", type=int, default=1)
    g.add_argument("--spec-k-max", type=int, default=4, help="also caps a fixed --spec-k")
    g.add_argument("--max-inflight-spec-gen", type=int, default=0,
                   help="spec LLM requests in flight across all tasks (0: unlimited; set it for self-hosted vLLM)")
    g.add_argument("--spec-queue", default="drop", choices=["drop", "replace"],
                   help="when all spec slots are busy: drop launches fewer specs; replace queues them and lets "
                        "a newer trigger replace the task's queued, not yet started specs")
    g.add_argument("--spec-prompt-mode", default="auto", choices=["auto", "prefill", "reprompt"],
                   help="prefill continues the main reasoning as an assistant prefix; reprompt appends it to the "
                        "prompt; auto uses prefill when the endpoint supports it")
    g.add_argument("--spec-schedule", default="demand", choices=["demand", "triggers"],
                   help="demand: launch one spec whenever the kernels expected at the profiler fall short, with the "
                        "test results of the iteration's earlier specs; triggers: k specs at each keyword trigger")
    g.add_argument("--spec-target-extra", type=float, default=1.0,
                   help="demand: the target of expected kernels at the profiler is max(devices + this, devices x "
                        "--spec-target-factor), devices being the task's share of the profiler GPUs")
    g.add_argument("--spec-target-factor", type=float, default=1.5, help="demand: see --spec-target-extra")
    g.add_argument("--spec-min-gap-tokens", type=int, default=500, help="demand: reasoning tokens between launches")
    g.add_argument("--spec-first-tokens", type=int, default=500,
                   help="demand: reasoning tokens before the first launch")
    g.add_argument("--best-ncu-wait-s", type=float, default=30.0,
                   help="demand: when a kernel meets the termination criterion (without one: beats the best so far), "
                        "launches pause up to this long for its ncu profile (0: never)")
    g.add_argument("--spec-max-per-iteration", type=int, default=0,
                   help="demand: spec launches per iteration and task, at most (0: no limit)")

    g = p.add_argument_group("iteration end")
    g.add_argument("--enable-termination", action=argparse.BooleanOptionalAction, default=False,
                   help="end an iteration as soon as a kernel meets --termination-criteria")
    g.add_argument("--termination-criteria", default=None, choices=CRITERIA,
                   help="first_valid: first kernel that passes timing; best: beats the best so far; "
                        "hist_avg: beats the mean of earlier iterations' selected kernels")
    g.add_argument("--termination-margin", type=float, default=0.03,
                   help="fraction a kernel must exceed the baseline by (best / hist_avg)")
    g.add_argument("--termination-confirm", type=int, default=2,
                   help="time a kernel that meets the criterion this many times in all; every timing must meet "
                        "it (2 keeps one noisy timing from ending an iteration)")
    g.add_argument("--grace-ratio", type=float, default=0.05,
                   help="after the main kernel is evaluated, wait at most this fraction of the main generation "
                        "time for the other candidates")
    return p


def parse_args(argv: Optional[list[str]] = None, *, discover_gpus=None) -> SpecGenConfig:
    p = build_parser()
    a = p.parse_args(argv)

    tasks = list(a.task)
    if a.level_dir is not None:
        if not a.level_dir.is_dir():
            p.error(f"--level-dir {a.level_dir} is not a directory")
        tasks += sorted(path for path in a.level_dir.glob("*.py") if path.is_file())
    if not tasks:
        p.error("give at least one --task or --level-dir")
    for path in tasks:
        if not path.is_file():
            p.error(f"task file not found: {path}")
    for name in ("max_iterations", "validations_per_gpu", "correctness_trials", "bench_iters", "bench_rounds",
                 "max_concurrent_tasks", "spec_k_max"):
        if getattr(a, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    for name in ("spec_k_min", "max_inflight_spec_gen", "spec_target_extra", "spec_target_factor", "spec_min_gap_tokens",
                 "spec_first_tokens", "best_ncu_wait_s", "spec_max_per_iteration"):
        if getattr(a, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be >= 0")
    spec_k = _auto_or_int(p, "--spec-k", a.spec_k, minimum=0)
    build_workers = _auto_or_int(p, "--build-workers", a.build_workers, minimum=0)
    if a.spec_k_min > a.spec_k_max:
        p.error("--spec-k-min must not exceed --spec-k-max")
    for name in ("trigger_after_tokens", "fallback_trigger_tokens"):
        if getattr(a, name) is not None and getattr(a, name) < (1 if name.startswith("fallback") else 0):
            p.error(f"--{name.replace('_', '-')} is out of range")
    if a.resume and not a.run_id:
        p.error("--resume needs the --run-id of the run to continue")
    if a.enable_termination and a.termination_criteria is None:
        p.error("--enable-termination needs --termination-criteria (first_valid, best or hist_avg)")
    if a.termination_criteria is not None and not a.enable_termination:
        p.error("--termination-criteria needs --enable-termination")
    if a.termination_margin < 0 or a.grace_ratio < 0:
        p.error("--termination-margin and --grace-ratio must be >= 0")
    if a.termination_confirm < 1:
        p.error("--termination-confirm must be >= 1")

    gpus_arg = a.gpus.strip().lower()
    if gpus_arg == "auto":
        if discover_gpus is None:
            from specgen.resources.gpus import discover_gpus
        gpus = discover_gpus()
    elif gpus_arg in ("", "none", "cpu"):
        gpus = []
    else:
        gpus = _csv(a.gpus)

    if gpus and a.cuda_arch is None:
        p.error(f"--cuda-arch is required when evaluating on GPUs ({ARCH_EXAMPLES}; or 'auto')")
    if a.cuda_arch not in (None, "auto", "inherit"):
        try:
            parse_arch_list(a.cuda_arch)
        except ValueError as e:
            p.error(str(e))
    build_jobs = _auto_or_int(p, "--build-jobs", a.build_jobs, minimum=1)

    return SpecGenConfig(
        llm=LLMConfig(
            model=a.model,
            endpoint=a.endpoint,
            models_file=a.models_file,
            tokenizer=a.tokenizer,
            temperature=a.temperature,
            top_p=a.top_p,
            max_tokens=a.max_tokens,
            request_timeout_s=a.request_timeout,
            max_retries=a.max_retries,
            max_continuations=a.max_continuations,
        ),
        pool=PoolConfig(
            gpus=gpus,
            allocation=a.gpu_allocation,
            num_validator_gpus=a.num_validator_gpus,
            num_profiler_gpus=a.num_profiler_gpus,
            validations_per_gpu=a.validations_per_gpu,
            build_workers=build_workers,
            job_timeout_s=a.job_timeout,
            remote_workers=[url.rstrip("/") for url in a.remote_worker],
        ),
        eval=EvalConfig(
            gpu_type=a.gpu_type,
            cuda_arch=a.cuda_arch,
            build_jobs=build_jobs,
            fast_includes=a.fast_cuda_includes,
            build_pch=a.build_pch,
            seed=a.seed,
            atol=a.atol,
            rtol=a.rtol,
            correctness_trials=a.correctness_trials,
            shift_trial=a.validation_shift_trial,
            bench_warmup=a.bench_warmup,
            bench_iters=a.bench_iters,
            bench_rounds=a.bench_rounds,
            ncu_top_n=max(0, a.ncu_top_n),
            ncu_binary=a.ncu_binary,
            ncu_metrics=_csv(a.ncu_metrics) or list(DEFAULT_NCU_METRICS),
            ncu_sections=[] if a.ncu_sections.strip().lower() == "none" else (
                _csv(a.ncu_sections) or list(DEFAULT_NCU_SECTIONS)),
            ncu_forward=max(1, a.ncu_forward),
            ncu_launch_count=max(1, a.ncu_launch_count),
            ncu_timeout_s=max(0.0, a.ncu_timeout_s),
            ncu_wait_s=max(0.0, a.ncu_wait_s),
        ),
        spec=SpecConfig(
            enabled=a.enable_spec_gen,
            trigger_config=a.trigger_config,
            after_tokens=a.trigger_after_tokens,
            fallback_tokens=a.fallback_trigger_tokens,
            k=spec_k,
            k_min=a.spec_k_min,
            k_max=a.spec_k_max,
            max_inflight=a.max_inflight_spec_gen,
            prompt_mode=a.spec_prompt_mode,
            queue=a.spec_queue,
            schedule=a.spec_schedule,
            target_extra=a.spec_target_extra,
            target_factor=a.spec_target_factor,
            min_gap_tokens=a.spec_min_gap_tokens,
            first_tokens=a.spec_first_tokens,
            best_ncu_wait_s=a.best_ncu_wait_s,
            max_per_iteration=a.spec_max_per_iteration,
        ),
        termination=TerminationConfig(
            enabled=a.enable_termination,
            criteria=a.termination_criteria,
            margin=a.termination_margin,
            confirm=a.termination_confirm,
            grace_ratio=a.grace_ratio,
        ),
        run=RunConfig(
            tasks=[path.resolve() for path in tasks],
            output_root=a.output_root,
            run_id=a.run_id or time.strftime("%Y%m%d-%H%M%S"),
            max_iterations=a.max_iterations,
            feedback_top_k=max(1, a.feedback_top_k),
            history_prompt=a.history_prompt,
            summary_max_approaches=max(0, a.summary_max_approaches),
            summary_max_failures=max(0, a.summary_max_failures),
            summary_max_chars=max(0, a.summary_max_chars),
            max_concurrent_tasks=a.max_concurrent_tasks,
            resume=a.resume,
        ),
    )
