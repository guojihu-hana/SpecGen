"""Spec quality as a function of the reasoning prefix length.

Takes the main reasoning recorded by a run, cuts it after N tokens for several N,
generates spec kernels from each cut exactly as a trigger would, and evaluates them
(build, validation, timing).  Answers: does a later trigger (a longer prefix) give
better kernels?

  python tools/prefix_sweep.py --run runs/<id> --task level1/88_MinGPTNewGelu --iteration 0 \
      --cuts 500,2000,4000,8000 --samples 4 --model DeepSeek-V4-Flash --endpoint api \
      --gpus 2,3 --cuda-arch 9.0a --output runs/<sweep-id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from specgen.control.orchestrator import resolve_cuda_arch
from specgen.evaluation.executor import LocalEvaluator
from specgen.evaluation.jobs import EvalJob
from specgen.generation.generator import Generator
from specgen.generation.prompts import SECTION_ORDER, Prompt
from specgen.llm.backend import make_backend
from specgen.llm.profiles import load_profiles
from specgen.llm.tokens import TokenCounter
from specgen.resources.pool import PoolJob, ResourcePool, make_devices
from specgen.tasks.base import KernelBenchTask


def load_prompt(path: Path) -> Prompt:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sections = tuple((name, raw["sections"].get(name, "")) for name in SECTION_ORDER)
    return Prompt(system=raw["system"], sections=sections)


def prefix_of(text: str, tokens: int, counter: TokenCounter) -> str:
    """The longest prefix of ``text`` with at most ``tokens`` tokens."""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text[:mid]) <= tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


async def evaluate(pool: ResourcePool, evaluator: LocalEvaluator, task: KernelBenchTask, kernel_dir: Path,
                   code: str) -> dict[str, Any]:
    (kernel_dir / "kernel.py").write_text(code, encoding="utf-8")

    async def job(kind: str, params: dict[str, Any]) -> dict[str, Any]:
        spec = EvalJob(kind, task.kind, str(task.path), str(kernel_dir / "kernel.py"), str(kernel_dir), params)
        result = await pool.submit(PoolJob(kind, task.task_id, 0, lambda slot: evaluator.run(spec, slot),
                                           label=f"{task.task_id}/{kernel_dir.name}/{kind}"))
        (kernel_dir / f"{kind}.json").write_text(json.dumps(result, default=str), encoding="utf-8")
        return result

    out: dict[str, Any] = {}
    if pool.has_role("builder"):
        build = await job("build", {})
        out["build_s"] = build.get("exec_s")
        if not build.get("ok") and build.get("status") in ("compile_error", "timeout"):
            return {**out, "status": build["status"]}
    validation = await job("validate", {"seed": 0, "trials": 3, "atol": 1e-2, "rtol": 1e-2})
    out["validate_s"] = validation.get("exec_s")
    if not validation.get("ok"):
        return {**out, "status": validation.get("status")}
    bench = await job("benchmark", {"seed": 0, "warmup": 10, "iters": 20, "rounds": 5})
    if not bench.get("ok"):
        return {**out, "status": "benchmark_error"}
    return {**out, "status": "success", "speedup": bench.get("speedup")}


async def sweep(a: argparse.Namespace) -> list[dict[str, Any]]:
    profile = load_profiles()[a.model]
    backend = make_backend(profile, a.endpoint, timeout_s=600, max_retries=2)
    generator = Generator(backend, profile, TokenCounter("generic"), spec_prompt_mode=a.spec_prompt_mode)
    counter = TokenCounter("generic")
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    cuda_arch = resolve_cuda_arch(a.cuda_arch, gpus)
    pool = ResourcePool(make_devices(gpus, allocation="fixed", validations_per_gpu=4, build_workers=a.build_workers))
    evaluator = LocalEvaluator(timeout_s=600, cuda_arch=cuda_arch, build_arch=next(iter(cuda_arch.values()), None),
                               build_jobs=4)
    llm_slots = asyncio.Semaphore(a.max_inflight)
    rows: list[dict[str, Any]] = []
    await pool.start()
    try:
        jobs = []
        for task_id in a.task:
            task = KernelBenchTask.from_path(Path(a.kernelbench) / f"{task_id}.py")
            iteration_dir = Path(a.run) / task.dir_name / f"iter_{a.iteration:03d}"
            prompt = load_prompt(iteration_dir / "prompt.json")
            reasoning = (iteration_dir / "main" / "reasoning.txt").read_text(encoding="utf-8")
            total = counter.count(reasoning)
            for cut in a.cuts:
                if cut > total:
                    print(f"{task_id}: only {total} reasoning tokens, skipping cut {cut}")
                    continue
                prefix = prefix_of(reasoning, cut, counter)
                for sample in range(a.samples):
                    jobs.append((task, prompt, prefix, cut, sample))

        async def one(task: KernelBenchTask, prompt: Prompt, prefix: str, cut: int, sample: int) -> None:
            kernel_dir = Path(a.output) / task.dir_name / f"cut{cut:05d}-s{sample}"
            kernel_dir.mkdir(parents=True, exist_ok=True)
            async with llm_slots:
                gen = await generator.generate_spec(prompt, prefix, index=sample, seed=1000 + sample)
            (kernel_dir / "content.txt").write_text(gen.content, encoding="utf-8")
            row = {"task": task.task_id, "cut_tokens": cut, "sample": sample, "temperature": gen.temperature,
                   "gen_s": gen.seconds, "has_code": gen.code is not None, "ok": gen.ok}
            row.update(await evaluate(pool, evaluator, task, kernel_dir, gen.code) if gen.code
                       else {"status": "parse_error" if gen.ok else "llm_error"})
            rows.append(row)
            print(json.dumps(row), flush=True)

        await asyncio.gather(*(one(*j) for j in jobs))
    finally:
        await pool.close()
        await backend.aclose()
    return rows


def report(rows: list[dict[str, Any]]) -> str:
    lines = ["task                     cut    n  code  valid  speedup median / max (valid only)"]
    keys = sorted({(r["task"], r["cut_tokens"]) for r in rows})
    for task, cut in keys:
        group = [r for r in rows if r["task"] == task and r["cut_tokens"] == cut]
        speedups = [r["speedup"] for r in group if r.get("speedup")]
        lines.append(f"{task.split('/')[-1][:24]:24s} {cut:>5} {len(group):>3} {sum(r['has_code'] for r in group):>5} "
                     f"{len(speedups):>6}  " + (f"{statistics.median(speedups):6.3f} / {max(speedups):6.3f}"
                                                 if speedups else "     - /      -"))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="run directory with the recorded main reasoning")
    p.add_argument("--task", action="append", required=True, help="task id, e.g. level1/88_MinGPTNewGelu")
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--cuts", type=lambda s: [int(x) for x in s.split(",")], default=[500, 2000, 4000, 8000])
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--model", default="DeepSeek-V4-Flash")
    p.add_argument("--endpoint", default="api")
    p.add_argument("--spec-prompt-mode", default="auto")
    p.add_argument("--gpus", required=True)
    p.add_argument("--cuda-arch", required=True)
    p.add_argument("--build-workers", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=8)
    p.add_argument("--kernelbench", default="KernelBench")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    Path(a.output).mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    rows = asyncio.run(sweep(a))
    (Path(a.output) / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    text = report(rows)
    (Path(a.output) / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"{len(rows)} specs in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
