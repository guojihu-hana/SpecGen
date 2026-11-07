#!/usr/bin/env python3
"""SpecGen artifact evaluation: from a GPU machine and an LLM endpoint to the results in one command.

    bash ae/run.sh getting-started  # about 30 minutes: check, GPU tests, 1_Square_matrix_multiplication_ x 1 iteration
    bash ae/run.sh e1               # T1-T10 x 10 iterations
    bash ae/run.sh full             # T1-T10 x 100 iterations (continues the e1 runs)
    bash ae/run.sh                  # all: getting-started, then e1
    bash ae/run.sh check            # hardware, software and LLM endpoint only
    bash ae/run.sh report           # the reports of the runs so far

LLM, one of: DEEPSEEK_API_KEY (DeepSeek-V4-Pro through the DeepSeek API); GLM_API_KEY (GLM-5.1 through an
OpenAI-compatible API, Z.ai's unless GLM_API_URL / GLM_API_MODEL say otherwise); GLM_URL (GLM-5.1-FP8 served by
ae/serve_glm51.sh, e.g. http://host:8000/v1, with GLM_SERVER_KEY if it needs one).  AE_MODEL=deepseek|glm-api|glm
picks one when several are set.  GPUs for validation and profiling: the first 8 visible (AE_GPUS=0,1,... to
choose); H800, H100, H200, B200 and B300 are detected and matched automatically.  CPUs: half of those available.
Every task runs on its own, SpecGen first and then CudaForge (SpecGen's baseline mode), each with all the GPUs;
--parallel-methods runs the two at once on half of the GPUs each (the getting-started test does when it has 4
GPUs or more).  AE_EXTRA_ARGS adds flags to every `specgen run`.  Running a command again resumes what stopped
and skips what finished.  Results: ae_results/ (AE_OUTPUT), with report.html and summary.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("AE_OUTPUT") or ROOT / "ae_results")
POOL_GPUS = 8
SUPPORTED_GPUS = ("H800", "H100", "H200", "B200", "B300")
LARGE_GPU_MIB = 140_000  # H200, B200, B300: 8 validations per GPU; 80 GB parts: 4
TASKS = {  # the paper's names (Tables 1 and 3) -> KernelBench files
    "T1": "level1/100_HingeLoss.py",
    "T2": "level1/10_3D_tensor_matrix_multiplication.py",
    "T3": "level1/11_4D_tensor_matrix_multiplication.py",
    "T4": "level1/12_Matmul_with_diagonal_matrices_.py",
    "T5": "level1/13_Matmul_for_symmetric_matrices.py",
    "T6": "level1/14_Matmul_for_upper_triangular_matrices.py",
    "T7": "level1/15_Matmul_for_lower_triangular_matrices.py",
    "T8": "level1/16_Matmul_with_transposed_A.py",
    "T9": "level1/17_Matmul_with_transposed_B.py",
    "T10": "level1/18_Matmul_with_transposed_both.py",
    "T11": "level2/12_Gemm_Multiply_LeakyReLU.py",
    "T12": "level2/14_Gemm_Divide_Sum_Scaling.py",
    "T13": "level2/33_Gemm_Scale_BatchNorm.py",
    "T14": "level2/52_Conv2d_Activation_BatchNorm.py",
    "T15": "level2/56_Matmul_Sigmoid_Sum.py",
    "T16": "level2/73_Conv2d_BatchNorm_Scaling.py",
    "T17": "level2/76_Gemm_Add_ReLU.py",
    "T18": "level2/99_Matmul_GELU_Softmax.py",
    "T19": "level3/1_MLP.py",
    "T20": "level3/50_ReLUSelfAttention.py",
    "sqmm": "level1/1_Square_matrix_multiplication_.py",  # the getting-started test
}
LEVEL1 = [f"T{i}" for i in range(1, 11)]
STAGES = {  # stage: run-id prefix, default tasks, iterations, SpecGen and CudaForge at once
    "getting-started": ("start", ["sqmm"], 1, True),
    "e1": ("main", LEVEL1, 10, False),
    "full": ("main", LEVEL1, 100, False),
}
METHODS = {  # method: label, flags on top of the common ones
    "specgen": ("SpecGen", ["--enable-termination", "--termination-criteria", "hist_avg"]),
    "cudaforge": ("CudaForge", ["--no-enable-spec-gen"]),
}
MODELS = {"deepseek": ("DeepSeek-V4-Pro", "api"), "glm-api": ("GLM-5.1", "api"), "glm": ("GLM-5.1-FP8", "selfhost")}
GLM_API_URL, GLM_API_MODEL = "https://api.z.ai/api/paas/v4", "glm-5.1"
RETRIES = 2  # a run that crashed resumes this many times (not after a setup error, exit 2)


def log(message: str) -> None:
    line = time.strftime("%H:%M:%S ") + message
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ae.log").open("a") as f:
        f.write(line + "\n")


def cpu_ranges(cpus: list[int]) -> str:
    """``0-63,128-191`` for a sorted CPU list (taskset syntax)."""
    parts, start = [], cpus[0]
    for prev, cur in zip(cpus, cpus[1:] + [None]):
        if cur != prev + 1:
            parts.append(f"{start}-{prev}" if prev > start else str(start))
            start = cur
    return ",".join(parts)


def cuda_bin() -> Path | None:
    for home in (os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
        if home and (Path(home) / "bin" / "nvcc").exists():
            return Path(home) / "bin"
    nvcc = shutil.which("nvcc")
    return Path(nvcc).parent if nvcc else None


def detect_gpus() -> list[dict]:
    from specgen.generation.gpu_specs import normalize_gpu_name

    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    gpus = []
    for line in out.strip().splitlines():
        index, name, total, used = (part.strip() for part in line.split(","))
        gpus.append({"index": index, "name": name, "type": normalize_gpu_name(name), "total_mib": int(total),
                     "used_mib": int(used)})
    return gpus


def ncu_status() -> tuple[str | None, bool]:
    """(ncu path, whether this user may read the GPU performance counters)."""
    bin_dir = cuda_bin()
    ncu = shutil.which("ncu") or next((str(p) for p in ([bin_dir / "ncu"] if bin_dir else []) if p.exists()), None)
    params = Path("/proc/driver/nvidia/params")
    admin_only = params.is_file() and "RmProfilingAdminOnly: 1" in params.read_text()
    cap_sys_admin = False
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("CapEff:"):
            cap_sys_admin = bool(int(line.split()[1], 16) & (1 << 21))
    return ncu, (not admin_only) or cap_sys_admin


def http_json(url: str, headers: dict[str, str], body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={**headers, **({"Content-Type": "application/json"} if data else {})})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


@dataclass
class Setup:
    gpus: list[str]
    gpu_type: str
    validations_per_gpu: int
    cpus: str
    model: str
    ncu: bool
    env: dict[str, str] = field(default_factory=dict)


def check() -> Setup:
    """Every requirement, with a report; exits with the list of problems if one is not met."""
    problems, rows = [], []
    rows.append(("Python", sys.version.split()[0]))
    try:
        import torch
        rows.append(("PyTorch", f"{torch.__version__} (CUDA {torch.version.cuda})"))
        if not torch.cuda.is_available():
            problems.append("PyTorch does not see a CUDA GPU")
    except ImportError:
        problems.append("PyTorch is not installed (pip install torch, or use ae/Dockerfile)")
    try:
        import specgen  # noqa: F401
    except ImportError:
        problems.append("SpecGen is not installed (pip install -e . in the repository)")
    bin_dir = cuda_bin()
    if bin_dir is None:
        problems.append("nvcc not found (set CUDA_HOME or put the CUDA toolkit's bin on PATH)")
    else:
        version = subprocess.run([str(bin_dir / "nvcc"), "--version"], capture_output=True, text=True).stdout
        rows.append(("nvcc", version.strip().splitlines()[-1] if version else "?"))

    gpus = detect_gpus() if shutil.which("nvidia-smi") else []
    if not gpus:
        problems.append("no NVIDIA GPU (nvidia-smi found none)")
        raise SystemExit("\n".join(["problems:"] + [f"  - {p}" for p in problems]))
    wanted = os.environ.get("AE_GPUS")
    chosen = [g for g in gpus if g["index"] in wanted.split(",")] if wanted else gpus[:POOL_GPUS]
    types = {g["type"] for g in chosen}
    gpu_type = next(iter(types)) if len(types) == 1 else None
    rows.append(("GPUs", f"{len(chosen)} x {chosen[0]['name']} (indices {','.join(g['index'] for g in chosen)})"))
    if len(types) > 1:
        problems.append(f"the GPUs chosen are of different types: {sorted(map(str, types))}")
    elif gpu_type not in SUPPORTED_GPUS:
        problems.append(f"{chosen[0]['name']} is not one of the supported GPUs ({', '.join(SUPPORTED_GPUS)})")
    if len(chosen) < POOL_GPUS:
        rows.append(("note", f"{len(chosen)} GPUs instead of the standard {POOL_GPUS}: runs take longer"))
    busy = [g["index"] for g in chosen if g["used_mib"] > 2048]
    if busy:
        problems.append(f"GPUs {busy} are in use: free them or choose others with AE_GPUS")
    validations = int(os.environ.get("AE_VALIDATIONS_PER_GPU")
                      or (8 if min(g["total_mib"] for g in chosen) >= LARGE_GPU_MIB else 4))
    rows.append(("validation", f"{validations} at once per GPU (AE_VALIDATIONS_PER_GPU to change)"))

    cpus = sorted(os.sched_getaffinity(0))
    half = cpus[:max(2, len(cpus) // 2)]
    rows.append(("CPUs", f"{len(half)} of {len(cpus)} ({cpu_ranges(half)}); {len(half) // 2} build workers"))

    ncu, counters = ncu_status()
    rows.append(("ncu", f"{ncu or 'not found'}; performance counters {'accessible' if counters else 'NOT accessible'}"))
    if not ncu or not counters:
        rows.append(("note", "without ncu SpecGen still runs; the prompts lack the ncu profile "
                             "(enable: run as root with CAP_SYS_ADMIN, or NVreg_RestrictProfilingToAdminUsers=0)"))

    env = dict(os.environ)
    path = ([str(bin_dir)] if bin_dir else []) + [str(Path(sys.executable).parent), env.get("PATH", "")]
    env["PATH"] = ":".join(path)  # the CUDA toolkit, then this Python environment's tools (pip's ninja)
    if bin_dir:
        env["CUDA_HOME"] = str(bin_dir.parent)
    ninja = shutil.which("ninja", path=env["PATH"])
    rows.append(("ninja", ninja or "not found"))
    if not ninja:
        problems.append("ninja not found: PyTorch needs it to build the kernels (pip install ninja)")
    model = os.environ.get("AE_MODEL") or ("deepseek" if os.environ.get("DEEPSEEK_API_KEY") else
                                            "glm" if os.environ.get("GLM_URL") else
                                            "glm-api" if os.environ.get("GLM_API_KEY") else "")
    if model == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            problems.append("AE_MODEL=deepseek needs DEEPSEEK_API_KEY")
        else:
            env["SPECGEN_DEEPSEEK_API_KEY"] = key
            try:
                ids = [m["id"] for m in http_json("https://api.deepseek.com/models",
                                                  {"Authorization": f"Bearer {key}"}).get("data", [])]
                rows.append(("LLM", f"DeepSeek API reachable; deepseek-v4-pro {'listed' if 'deepseek-v4-pro' in ids else 'NOT listed: ' + ', '.join(ids)}"))
            except (urllib.error.URLError, OSError, ValueError) as e:
                problems.append(f"DeepSeek API not reachable ({e}); behind a proxy, export https_proxy")
    elif model == "glm-api":
        key = os.environ.get("GLM_API_KEY", "")
        url = os.environ.get("GLM_API_URL", GLM_API_URL).rstrip("/")
        name = os.environ.get("GLM_API_MODEL", GLM_API_MODEL)
        if not key:
            problems.append("AE_MODEL=glm-api needs GLM_API_KEY")
        else:
            env.update(SPECGEN_GLM_API_URL=url, SPECGEN_GLM_API_KEY=key, SPECGEN_GLM_API_MODEL=name)
            try:
                reply = http_json(f"{url}/chat/completions", {"Authorization": f"Bearer {key}"},
                                  {"model": name, "messages": [{"role": "user", "content": "Reply with OK."}],
                                   "max_tokens": 8, "thinking": {"type": "disabled"}})
                if reply.get("choices"):
                    rows.append(("LLM", f"{name} at {url} answers"))
                else:
                    problems.append(f"{url} did not answer a request to {name}: {str(reply)[:200]}")
            except (urllib.error.URLError, OSError, ValueError) as e:
                problems.append(f"GLM API {url} refused or did not answer a request to {name} ({e})")
    elif model == "glm":
        url = os.environ.get("GLM_URL", "").rstrip("/")
        if not url:
            problems.append("AE_MODEL=glm needs GLM_URL (e.g. http://host:8000/v1 from ae/serve_glm51.sh)")
        else:
            env["SPECGEN_GLM_SELFHOST_URL"] = url
            env["SPECGEN_GLM_SELFHOST_KEY"] = os.environ.get("GLM_SERVER_KEY", "EMPTY")
            try:
                ids = [m["id"] for m in http_json(f"{url}/models", {"Authorization": f"Bearer {env['SPECGEN_GLM_SELFHOST_KEY']}"})
                       .get("data", [])]
                if "GLM-5.1-FP8" not in ids:
                    problems.append(f"{url} serves {ids}, not GLM-5.1-FP8 (start it with ae/serve_glm51.sh)")
                else:
                    rows.append(("LLM", f"GLM-5.1-FP8 at {url}"))
            except (urllib.error.URLError, OSError, ValueError) as e:
                problems.append(f"GLM endpoint {url} not reachable ({e})")
    else:
        problems.append("no LLM: export DEEPSEEK_API_KEY (DeepSeek-V4-Pro), GLM_API_KEY (GLM-5.1 API) "
                        "or GLM_URL (GLM-5.1-FP8 served by ae/serve_glm51.sh)")
    rows.append(("model", f"{MODELS[model][0]} ({MODELS[model][1]})" if model in MODELS else "-"))
    rows.append(("results", str(OUT)))

    width = max(len(k) for k, _ in rows)
    log("environment:\n" + "\n".join(f"  {k.ljust(width)}  {v}" for k, v in rows))
    if problems:
        raise SystemExit("problems:\n" + "\n".join(f"  - {p}" for p in problems))
    return Setup([g["index"] for g in chosen], gpu_type, validations, cpu_ranges(half), model, bool(ncu and counters), env)


def gpu_tests(setup: Setup) -> None:
    """The GPU tests of the repository: build, validation, timing (and ncu when it may run)."""
    marker = "gpu" if setup.ncu else "gpu and not ncu"
    log(f"GPU tests ({marker}) on GPU {setup.gpus[0]}, about 5 minutes")
    env = {**setup.env, "SPECGEN_TEST_GPU": setup.gpus[0]}  # found idle by check(); the tests skip on a busy one
    with (OUT / "logs").joinpath("gpu_tests.log").open("w") as out:
        code = subprocess.call(["taskset", "-c", setup.cpus, sys.executable, "-m", "pytest", "-q", "-rs", "-p",
                                "no:cacheprovider", "-m", marker, "tests/test_gpu.py"], cwd=ROOT, env=env, stdout=out,
                               stderr=subprocess.STDOUT)
    lines = (OUT / "logs" / "gpu_tests.log").read_text().strip().splitlines()
    tail = lines[-1] if lines else ""
    passed = code == 0 and "skipped" not in tail
    log(f"GPU tests: {'passed' if passed else 'FAILED'} ({tail}); log: {OUT / 'logs' / 'gpu_tests.log'}")
    if not passed:
        raise SystemExit("the GPU tests failed or were skipped: see the log above")


def finished(run_dir: Path) -> int:
    path = run_dir / "iterations.jsonl"
    done = set()
    for line in path.read_text().splitlines() if path.is_file() else ():
        try:
            done.add(json.loads(line)["iteration"])
        except (ValueError, KeyError):
            continue
    return len(done)


def best_so_far(run_dir: Path) -> float | None:
    path = run_dir / "kernels.jsonl"
    best = None
    for line in path.read_text().splitlines() if path.is_file() else ():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("valid") and record.get("speedup"):
            best = max(best or 0.0, record["speedup"])
    return best


def run_one(setup: Setup, prefix: str, task: str, method: str, iterations: int, gpus: list[str]) -> int:
    label, flags = METHODS[method]
    run_id = f"{prefix}-{task}-{method}"
    run_dir = OUT / "runs" / run_id
    if finished(run_dir) >= iterations:
        log(f"{task} {label}: {iterations} iterations done before, skipped")
        return 0
    if run_dir.exists() and not (run_dir / "config.json").is_file():
        shutil.rmtree(run_dir)  # stopped before it wrote anything to continue from
    profile, endpoint = MODELS[setup.model]
    common = ["--task", str(ROOT / "KernelBench" / TASKS[task]), "--models-file", str(ROOT / "ae" / "models.yaml"),
              "--model", profile, "--endpoint", endpoint, "--gpus", ",".join(gpus), "--gpu-type", "auto",
              "--cuda-arch", "auto", "--gpu-allocation", "dynamic", "--validations-per-gpu",
              str(setup.validations_per_gpu), "--max-concurrent-tasks", "1", "--max-iterations", str(iterations),
              "--output-root", str(OUT / "runs"), "--run-id", run_id, *flags,
              *shlex.split(os.environ.get("AE_EXTRA_ARGS", ""))]
    code = 1
    for attempt in range(1 + RETRIES):
        resume = (run_dir / "config.json").is_file()
        if attempt:
            subprocess.call(["pkill", "-9", "-f", f"/runs/{run_id}/"])  # job processes of the crashed run
            log(f"{task} {label}: resuming (retry {attempt} of {RETRIES})")
            time.sleep(30)
        cmd = ["taskset", "-c", setup.cpus, sys.executable, "-m", "specgen.cli", "run", *common,
               *(["--resume"] if resume else [])]
        log(f"{task} {label}: {'continuing' if resume else 'starting'} on GPUs {','.join(gpus)} "
            f"({finished(run_dir)}/{iterations} iterations done)")
        started, shown = time.time(), -1
        with (OUT / "logs" / f"{run_id}.log").open("a") as out:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=setup.env, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while proc.poll() is None:
                time.sleep(30)
                done = finished(run_dir)
                if done != shown:
                    shown, best = done, best_so_far(run_dir)
                    log(f"{task} {label}: {done}/{iterations} iterations, best "
                        f"{f'{best:.2f}x' if best else '-'}, {(time.time() - started) / 60:.0f} min")
        code = proc.returncode
        if code in (0, 2):
            break
    log(f"{task} {label}: {'finished' if code == 0 else f'stopped with exit {code}'}; log: {OUT / 'logs' / f'{run_id}.log'}")
    return code


def run_stage(setup: Setup, stage: str, tasks: list[str], iterations: int, parallel: bool) -> None:
    prefix = STAGES[stage][0]
    parallel = parallel and len(setup.gpus) >= 4  # each run keeps a validation and a profiling GPU of its own
    log(f"{stage}: {', '.join(tasks)} x {iterations} iterations, one task at a time, "
        f"{'SpecGen and CudaForge at once on half of the GPUs each' if parallel else 'SpecGen then CudaForge'}")
    for task in tasks:
        if parallel:
            half = len(setup.gpus) // 2
            threads = [threading.Thread(target=run_one, args=(setup, prefix, task, method, iterations, gpus))
                       for method, gpus in (("specgen", setup.gpus[:half]), ("cudaforge", setup.gpus[half:]))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for method in METHODS:
                run_one(setup, prefix, task, method, iterations, setup.gpus)
        report(prefix)


def report(prefix: str) -> None:
    code = subprocess.call([sys.executable, str(ROOT / "ae" / "report.py"), "--runs", str(OUT / "runs"),
                            "--prefix", prefix, "--output", str(OUT)], cwd=ROOT)
    if code != 0:
        log(f"report for {prefix} runs failed (exit {code})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", nargs="?", default="all", choices=["all", "check", "gpu-tests", *STAGES, "report"])
    p.add_argument("--tasks", default=None, help="comma-separated, e.g. T1,T4 or T11-T20 names (default: the stage's)")
    p.add_argument("--iterations", type=int, default=None, help="per task (default: the stage's)")
    p.add_argument("--parallel-methods", action="store_true",
                   help="run SpecGen and CudaForge of a task at once, on half of the GPUs each")
    p.add_argument("--skip-gpu-tests", action="store_true")
    a = p.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    if a.stage == "report":
        for prefix in ("start", "main"):
            report(prefix)
        return 0
    setup = check()
    if a.stage == "check":
        return 0
    if a.stage in ("all", "gpu-tests", "getting-started") and not a.skip_gpu_tests:
        gpu_tests(setup)
    stages = ["getting-started", "e1"] if a.stage == "all" else [a.stage] if a.stage in STAGES else []
    for stage in stages:
        _, tasks, iterations, parallel = STAGES[stage]
        if stage != "getting-started":  # --tasks and --iterations are for the experiments
            tasks = a.tasks.split(",") if a.tasks else tasks
            iterations = a.iterations or iterations
        unknown = [t for t in tasks if t not in TASKS]
        if unknown:
            raise SystemExit(f"unknown tasks {unknown}; known: {', '.join(TASKS)}")
        run_stage(setup, stage, tasks, iterations, a.parallel_methods or parallel)
    if stages:
        log(f"done: {OUT}/start_report.html and start_summary.md (getting-started); "
            f"report.html and summary.md (e1, full)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
