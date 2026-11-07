"""Wire the components together and run every task's iteration loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from specgen.config import SpecGenConfig
from specgen.control.budget import SpecBudget
from specgen.control.context import RunContext
from specgen.control.history import TaskHistory
from specgen.control.loop import IterationLoop
from specgen.control.triggers import TriggerConfig, TriggerConfigError, load_trigger_config
from specgen.evaluation.executor import LocalEvaluator
from specgen.generation.generator import Generator
from specgen.generation.gpu_specs import GPU_SPECS
from specgen.generation.prompts import PromptBuilder
from specgen.llm.backend import ChatBackend, make_backend
from specgen.llm.profiles import ProfileError, load_profiles
from specgen.llm.tokens import TokenCounter
from specgen.resources.gpus import arch_for, arch_runs_on, compute_capabilities, detect_gpu_type, parse_arch_list
from specgen.resources.host import effective_cpu_count
from specgen.remote.client import EvaluatorRouter, RemoteEvaluator, WorkerError, fetch_worker_info
from specgen.resources.pool import Device, ResourcePool, make_devices
from specgen.storage.layout import RunLayout, read_records, write_json, write_records
from specgen.tasks.base import Task, load_tasks
from specgen.telemetry.recorder import PerfRecorder

log = logging.getLogger(__name__)
WORKER_TOKEN_ENV = "SPECGEN_WORKER_TOKEN"
MAX_AUTO_BUILD_JOBS = 32
MAX_AUTO_BUILD_WORKERS = 64


class RunSetupError(RuntimeError):
    pass


def resolve_gpu_type(setting: str, gpus: list[str]) -> str:
    if setting != "auto":
        if setting not in GPU_SPECS:
            raise RunSetupError(f"unknown --gpu-type {setting!r} (known: {', '.join(sorted(GPU_SPECS))})")
        return setting
    detected = detect_gpu_type(gpus)
    if detected is None:
        raise RunSetupError(f"cannot detect the GPU type of {gpus or 'a CPU-only run'}; pass --gpu-type "
                            f"(one of {', '.join(sorted(GPU_SPECS))})")
    return detected


def resolve_cuda_arch(setting: Optional[str], gpus: list[str]) -> dict[str, str]:
    """GPU id -> TORCH_CUDA_ARCH_LIST for kernel builds (empty: keep the environment).

    An explicit list is checked against each GPU's compute capability, so a list for
    another machine (e.g. B300's 10.3a on an H100) fails here instead of in every kernel.
    """
    if not gpus or setting in (None, "inherit"):
        return {}
    caps = compute_capabilities()
    if setting == "auto":
        missing = [gpu for gpu in gpus if gpu not in caps]
        if missing:
            log.warning("no compute capability for GPU(s) %s; their builds keep TORCH_CUDA_ARCH_LIST from the environment", missing)
        return {gpu: arch_for(caps[gpu]) for gpu in gpus if gpu in caps}
    try:
        tokens = parse_arch_list(setting)
    except ValueError as e:
        raise RunSetupError(str(e)) from None
    for gpu in gpus:
        cap = caps.get(gpu)
        if cap is not None and not arch_runs_on(tokens, cap):
            raise RunSetupError(f"--cuda-arch {setting} cannot run on GPU {gpu} (compute capability {cap}); "
                                f"use {arch_for(cap)} on this machine")
    return {gpu: " ".join(tokens) for gpu in gpus}


def resolve_build_jobs(setting: Optional[int], build_slots: int) -> int:
    """Compile jobs per kernel build: the CPUs shared by every slot that may build at once."""
    if setting is not None:
        return setting
    return max(2, min(MAX_AUTO_BUILD_JOBS, effective_cpu_count() // max(1, build_slots)))


def resolve_build_workers(setting: Optional[int], gpus: list[str]) -> int:
    """CPU build slots; none for CPU-only runs, whose kernels have nothing to compile ahead."""
    if not gpus:
        return 0
    if setting is not None:
        return setting
    # A load_inline build compiles two units in parallel (main.cpp and cuda.cu): two CPUs per build.
    return min(MAX_AUTO_BUILD_WORKERS, max(1, effective_cpu_count() // 2))


def resolve_build_arch(cuda_arch: dict[str, str], devices: list[Device]) -> Optional[str]:
    """The arch list CPU builds use: the validators' (their builds are then reused as they are)."""
    for device in devices:
        if device.role in ("validator", "shared") and device.gpu in cuda_arch:
            return cuda_arch[device.gpu]
    return next(iter(cuda_arch.values()), None)


def resolve_remote_workers(urls: list[str]) -> dict[str, dict[str, Any]]:
    token = os.environ.get(WORKER_TOKEN_ENV) or None
    workers = {}
    for url in urls:
        try:
            workers[url] = fetch_worker_info(url, token)
        except WorkerError as e:
            raise RunSetupError(str(e)) from None
    return workers


def check_resumable(run_dir: Path, tasks: list[str]) -> Optional[float]:
    """The time of the last event of the run to continue at ``run_dir``; errors unless it ran the same tasks
    (compared as ``level/file``, so a moved KernelBench checkout still matches)."""
    config = run_dir / "config.json"
    if not config.is_file():
        raise RunSetupError(f"cannot resume {run_dir}: it has no config.json")
    task_name = lambda path: "/".join(Path(path).parts[-2:])  # noqa: E731
    before = sorted(map(task_name, json.loads(config.read_text(encoding="utf-8"))["run"]["tasks"]))
    if before != sorted(map(task_name, tasks)):
        raise RunSetupError(f"--resume needs the tasks of the run it continues: {before}")
    return next((e["t"] for e in reversed(read_records(run_dir / "events.jsonl")) if "t" in e), None)


def set_aside_interrupted(layout: RunLayout, tasks: list[Task], session: int) -> list[str]:
    """Clear the way for rerunning each task's unfinished iteration: its directory becomes
    ``iter_NNN.interrupted<session>`` and its kernel records move to ``kernels.interrupted<session>.jsonl``.
    Returns the kernel ids moved."""
    run_dir = layout.run_dir
    start = {task.task_id: 0 for task in tasks}
    for r in read_records(run_dir / "iterations.jsonl"):
        if r.get("task") in start:
            start[r["task"]] = max(start[r["task"]], r["iteration"] + 1)
    for task in tasks:
        for directory in sorted(layout.task_dir(task).glob("iter_[0-9][0-9][0-9]")):
            if int(directory.name[len("iter_"):]) >= start[task.task_id]:
                directory.rename(directory.with_name(f"{directory.name}.interrupted{session}"))
    rows = read_records(run_dir / "kernels.jsonl")
    stale = [r for r in rows if r.get("task_id") in start and r.get("iteration", -1) >= start[r["task_id"]]]
    if stale:
        write_records(run_dir / f"kernels.interrupted{session}.jsonl", stale)
        write_records(run_dir / "kernels.jsonl", [r for r in rows if not any(r is s for s in stale)])
    return [f"{r['task_id']}|{r['kernel_id']}" for r in stale]


def resolve_triggers(cfg: SpecGenConfig) -> Optional[TriggerConfig]:
    if not cfg.spec.enabled:
        return None
    try:
        triggers = load_trigger_config(cfg.spec.trigger_config)
    except TriggerConfigError as e:
        raise RunSetupError(f"trigger config: {e}") from None
    return triggers.with_overrides(after_tokens=cfg.spec.after_tokens, fallback_tokens=cfg.spec.fallback_tokens)


async def run_async(cfg: SpecGenConfig, *, backend: Optional[ChatBackend] = None) -> dict[str, TaskHistory | BaseException]:
    profiles = load_profiles(cfg.llm.models_file)
    if cfg.llm.model not in profiles:
        raise ProfileError(f"model {cfg.llm.model!r} not in the models file (have {sorted(profiles)})")
    profile = profiles[cfg.llm.model]
    if backend is None:  # resolves the endpoint, so placeholders fail before anything is written
        backend = make_backend(profile, cfg.llm.endpoint, timeout_s=cfg.llm.request_timeout_s,
                               max_retries=cfg.llm.max_retries)
    tasks = load_tasks(cfg.run.tasks)
    triggers = resolve_triggers(cfg)
    try:
        generator = Generator(
            backend, profile, TokenCounter(cfg.llm.tokenizer or profile.tokenizer),
            temperature=cfg.llm.temperature, top_p=cfg.llm.top_p, max_tokens=cfg.llm.max_tokens,
            max_continuations=cfg.llm.max_continuations, spec_prompt_mode=cfg.spec.prompt_mode,
        )
    except ValueError as e:
        raise RunSetupError(str(e)) from None
    workers = resolve_remote_workers(cfg.pool.remote_workers)
    remote_gpu_type = next((info.get("gpu_type") for info in workers.values() if info.get("gpu_type")), None)
    if cfg.eval.gpu_type == "auto" and not cfg.pool.gpus and remote_gpu_type:
        gpu_type = remote_gpu_type
    else:
        gpu_type = resolve_gpu_type(cfg.eval.gpu_type, cfg.pool.gpus)
    if cfg.pool.gpus and cfg.eval.cuda_arch is None:
        raise RunSetupError("cuda_arch is required when evaluating on GPUs")
    cuda_arch = resolve_cuda_arch(cfg.eval.cuda_arch, cfg.pool.gpus)
    build_workers = resolve_build_workers(cfg.pool.build_workers, cfg.pool.gpus)
    try:
        devices = []
        if cfg.pool.gpus or not workers:
            devices = make_devices(cfg.pool.gpus, allocation=cfg.pool.allocation,
                                   num_validator=cfg.pool.num_validator_gpus, num_profiler=cfg.pool.num_profiler_gpus,
                                   validations_per_gpu=cfg.pool.validations_per_gpu, build_workers=build_workers)
        for url, info in workers.items():
            devices += make_devices(list(info.get("gpus") or []), allocation=cfg.pool.allocation,
                                    validations_per_gpu=cfg.pool.validations_per_gpu, host=url)
    except ValueError as e:
        raise RunSetupError(str(e)) from None
    validation_slots = sum(d.capacity for d in devices if d.role in ("validator", "shared"))
    build_jobs = resolve_build_jobs(cfg.eval.build_jobs, build_workers or validation_slots)
    build_arch = resolve_build_arch(cuda_arch, devices)

    layout = RunLayout(cfg.run.output_root, cfg.run.run_id)
    layout.run_dir.parent.mkdir(parents=True, exist_ok=True)
    previous_end, interrupted = None, []
    if cfg.run.resume:
        previous_end = check_resumable(layout.run_dir, [str(path) for path in cfg.run.tasks])
        session = len(list(layout.run_dir.glob("config.resume*.json"))) + 1
        interrupted = set_aside_interrupted(layout, tasks, session)
        config_path = layout.run_dir / f"config.resume{session}.json"
    else:
        try:
            layout.run_dir.mkdir()  # atomic: two runs can never share a directory
        except FileExistsError:
            raise RunSetupError(f"run directory {layout.run_dir} already exists; choose another --run-id") from None
        config_path = layout.run_dir / "config.json"
    if cfg.termination.enabled and triggers is None:
        log.warning("--enable-termination is ignored without spec generation: the baseline evaluates one kernel "
                    "per iteration")
    resolved = {"termination": cfg.termination.criteria if cfg.termination.enabled and triggers is not None else None,
                "gpu_type": gpu_type, "cuda_arch": cuda_arch, "build_arch": build_arch, "build_workers": build_workers,
                "build_jobs": build_jobs, "effective_cpus": effective_cpu_count(),
                "spec_prompt_mode": generator.spec_prompt_mode if triggers else None,
                "remote_workers": {url: {k: info.get(k) for k in ("gpus", "gpu_type", "cuda_arch", "hostname")}
                                   for url, info in workers.items()}}
    write_json(config_path, {**cfg.to_dict(), "resolved": resolved})
    recorder = PerfRecorder(layout.run_dir)
    pool = ResourcePool(devices, allocation=cfg.pool.allocation, on_event=recorder.event)
    evaluator: Any = LocalEvaluator(timeout_s=cfg.pool.job_timeout_s, cuda_arch=cuda_arch, build_arch=build_arch,
                                    build_jobs=build_jobs, fast_includes=cfg.eval.fast_includes,
                                    pch=cfg.eval.build_pch)
    if workers:
        token = os.environ.get(WORKER_TOKEN_ENV) or None
        evaluator = EvaluatorRouter(evaluator, {url: RemoteEvaluator(url, token=token, timeout_s=cfg.pool.job_timeout_s)
                                                for url in workers})
    ctx = RunContext(
        cfg=cfg,
        generator=generator,
        pool=pool,
        evaluator=evaluator,
        prompts=PromptBuilder(gpu_type),
        recorder=recorder,
        layout=layout,
        budget=SpecBudget(k=cfg.spec.k, k_min=cfg.spec.k_min, k_max=cfg.spec.k_max,
                          max_inflight=cfg.spec.max_inflight),
        triggers=triggers,
        ncu_enabled=cfg.eval.ncu_top_n > 0,
    )
    recorder.event("run_started", tasks=[t.task_id for t in tasks], model=profile.name, endpoint=cfg.llm.endpoint,
                   gpu_type=gpu_type, allocation=cfg.pool.allocation, devices=[d.label for d in devices],
                   spec=triggers is not None, spec_prompt_mode=resolved["spec_prompt_mode"],
                   resumed=cfg.run.resume, previous_end=previous_end, interrupted_kernels=len(interrupted))

    semaphore = asyncio.Semaphore(cfg.run.max_concurrent_tasks)
    ctx.budget.active_tasks = 0

    async def run_task(task) -> TaskHistory:
        async with semaphore:
            ctx.budget.active_tasks += 1
            try:
                loop = IterationLoop(task, ctx)
                if cfg.run.resume:
                    loop.restore()
                return await loop.run()
            finally:
                ctx.budget.active_tasks -= 1

    await pool.start()
    try:
        results = await asyncio.gather(*(run_task(t) for t in tasks), return_exceptions=True)
    finally:
        recorder.event("run_finished", pool=pool.stats().__dict__, stats=recorder.stats(),
                       spec=ctx.budget.stats())
        await pool.close()
        await backend.aclose()
        if isinstance(evaluator, EvaluatorRouter):
            await evaluator.aclose()
        recorder.close()
    out: dict[str, TaskHistory | BaseException] = {}
    for task, result in zip(tasks, results):
        if isinstance(result, BaseException):
            log.error("task %s failed: %r", task.task_id, result)
        out[task.task_id] = result
    return out


def run(cfg: SpecGenConfig) -> int:
    try:
        results = asyncio.run(run_async(cfg))
    except (ProfileError, RunSetupError) as e:
        print(f"specgen: {e}")
        return 2
    failed = 0
    for task_id, result in results.items():
        if isinstance(result, BaseException):
            failed += 1
            print(f"{task_id}: failed ({type(result).__name__}: {result})")
            continue
        best = result.best()
        print(f"{task_id}: best speedup {best.speedup:.3f}x (iteration {best.iteration})" if best else f"{task_id}: no valid kernel")
    return 1 if failed else 0
