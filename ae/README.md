# SpecGen: artifact evaluation

This artifact accompanies *SpecGen: Accelerating Agentic Kernel Optimization with Speculative Generation*
(PPoPP 2027, [arXiv:2606.17518](https://arxiv.org/abs/2606.17518)). It contains SpecGen, scripts that run SpecGen
and CudaForge on KernelBench tasks with the same LLM and GPUs and compare them, and our reference results.

Badges requested: **Artifacts Evaluated – Reusable** and **Artifacts Available** (archived at
[doi.org/10.5281/zenodo.24049999](https://doi.org/10.5281/zenodo.24049999)).

## Contents

| Path | What it is |
|---|---|
| `ae/run.sh` | the entry point: installs missing Python dependencies, then runs `ae/ae.py` |
| `ae/ae.py` | checks the machine and the LLM, runs the GPU tests and the experiments, resumes interrupted runs |
| `ae/report.py` | the comparison page (`report.html`) and tables (`summary.md`, `summary.csv`) |
| `ae/models.yaml` | the LLM profiles used here |
| `ae/serve_glm51.sh`, `ae/Dockerfile` | GLM-5.1-FP8 served with vLLM; the software environment as a container |
| `ae/reference/` | our reference results: report, summary tables and the run records behind them |
| `src/specgen/`, `tests/` | SpecGen and its tests |
| `KernelBench/` | the tasks ([KernelBench](https://github.com/ScalingIntelligence/KernelBench), MIT License) |
| `docs/architecture.md` | how the code is organized and how to extend it |

## Requirements

- **GPUs for validation and profiling:** 8 GPUs of one type among H800, H100, H200, B200 and B300; the standard
  setup is 8 H200 GPUs. The type is detected automatically. `AE_GPUS=0,1,...` chooses the GPUs; fewer than 8
  work, more slowly.
- **LLM**, one of:
  - DeepSeek-V4-Pro through the DeepSeek API: `export DEEPSEEK_API_KEY=...`. The paper used the 0424 preview
    release; the API now serves the 0813 release, which is priced higher.
  - GLM-5.1 through an OpenAI-compatible API: `export GLM_API_KEY=...`. The default is Z.ai's API
    (`https://api.z.ai/api/paas/v4`, model `glm-5.1`); `GLM_API_URL` and `GLM_API_MODEL` select another provider,
    e.g. `GLM_API_URL=https://open.bigmodel.cn/api/paas/v4`.
  - GLM-5.1-FP8 served with vLLM on 8 more H200 GPUs: `GLM_MODEL=<weights> bash ae/serve_glm51.sh` on those
    GPUs, then `export GLM_URL=http://<host>:8000/v1` where `ae/run.sh` runs.

  `AE_MODEL=deepseek|glm-api|glm` picks one when several are set.
- **CPU:** half of the CPUs available to the command are used.
- **Software** (or the container): Linux, NVIDIA driver, CUDA toolkit 12.9 or later (`nvcc`), Python 3.10+,
  PyTorch with CUDA, Nsight Compute (`ncu`). `ae/run.sh` installs SpecGen and its Python dependencies
  (`pip install -e ".[tokens,dev]"`) into the Python it runs with (`PYTHON=...`, default `python3`). Tested with
  NGC PyTorch 26.03 (PyTorch 2.11, CUDA 13.2), the base of `ae/Dockerfile`.
- **ncu permission:** ncu needs the GPU performance counters (root with `CAP_SYS_ADMIN`, e.g. `docker run
  --cap-add SYS_ADMIN`, or the driver option `NVreg_RestrictProfilingToAdminUsers=0`). Without it the runs go on
  without ncu profiles.
- **Network:** the LLM API (behind a proxy: `export https_proxy=...`), or the vLLM endpoint.
- **Disk:** about 10 GB for `e1`, 100 GB for `full`.

## Getting started (about 30 minutes)

```bash
cd SpecGen                         # the unpacked artifact
export DEEPSEEK_API_KEY=...        # or GLM_API_KEY / GLM_URL (see Requirements)
bash ae/run.sh getting-started
```

The command installs what is missing, checks the machine and the LLM endpoint, runs the GPU tests (about 5
minutes), then optimizes `KernelBench/level1/1_Square_matrix_multiplication_.py` for one iteration with SpecGen
and CudaForge at once on 4 GPUs each (one after the other with fewer than 4 GPUs), and writes the report. It took
about 20 minutes on 8 B300 GPUs with DeepSeek-V4-Pro.

Or in the container:

```bash
docker build -f ae/Dockerfile -t specgen-ae .
docker run --rm --gpus all --cap-add SYS_ADMIN --ipc host -e DEEPSEEK_API_KEY \
    -v "$PWD/ae_results:/specgen/ae_results" specgen-ae getting-started
```

Expected output (abridged):

```text
installing SpecGen and its test dependencies into /path/to/venv
environment:
  Python      3.12.3
  PyTorch     2.11.0a0+a6c236b9fd.nv26.03.46836102 (CUDA 13.2)
  nvcc        Build cuda_13.2.r13.2/compiler.37434383_0
  GPUs        8 x NVIDIA B300 SXM6 AC (indices 0,1,2,3,4,5,6,7)
  validation  8 at once per GPU (AE_VALIDATIONS_PER_GPU to change)
  CPUs        16 of 32 (200-215); 8 build workers
  ncu         /usr/local/bin/ncu; performance counters NOT accessible
  note        without ncu SpecGen still runs; the prompts lack the ncu profile (enable: ...)
  ninja       /usr/local/bin/ninja
  LLM         DeepSeek API reachable; deepseek-v4-pro listed
  model       DeepSeek-V4-Pro (api)
  results     /path/to/SpecGen/ae_results
GPU tests (gpu and not ncu) on GPU 0, about 5 minutes
GPU tests: passed (7 passed, 1 deselected in ...); log: .../logs/gpu_tests.log
getting-started: sqmm x 1 iterations, one task at a time, SpecGen and CudaForge at once on half of the GPUs each
sqmm SpecGen: starting on GPUs 0,1,2,3 (0/1 iterations done)
sqmm CudaForge: starting on GPUs 4,5,6,7 (0/1 iterations done)
...
sqmm SpecGen: 1/1 iterations, best 0.95x, 14 min
sqmm CudaForge: 1/1 iterations, best -, 14 min
done: .../start_report.html and start_summary.md (getting-started); report.html and summary.md (e1, full)
```

`ae_results/start_summary.md` then holds one row, `SpecGen / CudaForge` in each cell; ours (abridged):

| task | iterations | e2e_min | profiled_kernels | best_speedup | profiler_util | validator_util | input_hit_M | input_miss_M | output_M | equivalent_M |
|---|---|---|---|---|---|---|---|---|---|---|
| sqmm | 1 / 1 | 13.5 / 13.2 | 12 / 0 | 0.949x / - | 5.8% / 0.0% | 42.3% / 1.2% | 2.85 / 0.00 | 0.16 / 0.00 | 0.31 / 0.05 | 1.45 / 0.22 |

`ae_results/start_report.html` is the page described under [Results](#results). The artifact works if the check
and the GPU tests pass, both runs finish their iteration and the summary is written. In our run SpecGen evaluated
104 candidate kernels (12 valid) and CudaForge one; SpecGen's validation GPUs were busy 42% of the time,
CudaForge's 1%. Single-iteration results vary from run to run.

## Experiments and the paper's claims

| Experiment | Command | Time on 8 H200 (DeepSeek-V4-Pro / GLM-5.1) | Token budget, SpecGen + CudaForge (DeepSeek-V4-Pro) |
|---|---|---|---|
| Getting started | `bash ae/run.sh getting-started` | about 30 min | about 3M input (95% cache hits), 0.4M output |
| E1: T1-T10 x 10 iterations | `bash ae/run.sh e1` | about 26 h / 33 h (16 h / 21 h with `--parallel-methods`) | input up to 195M, output up to 30M |
| T1-T10 x 100 iterations | `bash ae/run.sh full` (continues `e1`) | about 11 / 14 days (7 / 9 days with `--parallel-methods`) | input up to 3B, output up to 0.3B |
| T11-T20 (Level 2/3) | `bash ae/run.sh e1 --tasks T11,T12,...,T20` (or `full`) | longer than T1-T10 | more than T1-T10 |

Tasks run one at a time, SpecGen and then CudaForge, each with all 8 GPUs (`--parallel-methods`: both at once on 4
GPUs each). `bash ae/run.sh` alone runs the getting-started test and then `e1`. After an interruption, run the same
command again: it resumes. Use a separate results directory per LLM, e.g.
`AE_OUTPUT=ae_results_glm AE_MODEL=glm-api GLM_API_KEY=... bash ae/run.sh e1`.

Times are from the paper's per-iteration measurements on H200 (Table 9); token budgets are upper estimates, and
`start_summary.md` shows the rate on your setup. At the provider's list price, the cost is about `cache-hit input
/ 50 + cache-miss input + 4 x output` input-token prices ("equivalent tokens" in the results).

### Claims, figures and tables

`ae_results/summary.md` gives each metric per task and the geometric means, SpecGen / CudaForge.

| Paper | Claim | Paper's value (DeepSeek-V4-Pro; GLM-5.1) | In the results |
|---|---|---|---|
| Fig. 10 | SpecGen shortens end-to-end time | 1.88x; 1.50x (geometric mean, T1-T10, 100 iterations) | `e2e_min`, `time_speedup` |
| Fig. 11 | SpecGen gets more profiling feedback | 66.4 vs 40.7; 70.3 vs 36.3 profiled kernels per task | `profiled_kernels`, `feedback_ratio` |
| Table 4 | SpecGen keeps validation and profiling GPUs busy | 96.1% vs 11.3%; 88.2% vs 4.7% | `validator_util`, `profiler_util` |
| Table 6, Fig. 13 | SpecGen finds faster kernels | the best final kernel on all ten tasks | `best_speedup`, `speedup_ratio`; the report's speedup-over-time charts |
| Table 7 | SpecGen's token use stays close to CudaForge's | GLM-5.1: 23.14M vs 23.66M | `input_hit_M`, `input_miss_M`, `output_M`, `equivalent_M` |
| Table 8 | the same on Level 2/3 tasks | DeepSeek-V4-Pro: time 1.57x, feedback 54.6 vs 26.1, utilization 88.2% vs 15.7% | the same columns, with `--tasks T11,...,T20` |
| Table 9 | the termination criterion trades end-to-end time for kernel speed | first valid, historical average, historical best | the report's termination-criterion menu |

- Figures 10 and 11 and Tables 4 and 6 also compare with AlphaEvolve and KernelAgent, which are not included; run
  them with the configuration under [Systems compared](#systems-compared).
- The report estimates the other termination criteria from the recorded run. To run SpecGen with one:
  `AE_OUTPUT=ae_results_best AE_EXTRA_ARGS="--termination-criteria best" bash ae/run.sh e1`.

### Interpreting differences from the published results

- Runs vary with LLM sampling: compare the geometric means and the direction of each effect.
- `e1` runs 10 iterations, the paper 100: compare ratios; final kernel speedups need `full`.
- GPUs other than H200, and runs without ncu permission, change speedups and times.

## Reference results: DeepSeek-V4-Flash on B300

`ae/reference/` holds our runs of SpecGen and CudaForge on four KernelBench tasks, 20 iterations each, with
DeepSeek-V4-Flash through the DeepSeek API. Each run used 2 B300 GPUs, one for validation (8 at once) and one for
profiling; SpecGen ran with `--termination-criteria best`.

| Task | E2E time, min | Profiled kernels | Best speedup | Profiler utilization | Validator utilization | Equivalent tokens, M |
|---|---|---|---|---|---|---|
| level1/1_Square_matrix_multiplication_ | 117.8 / 115.2 | 164 / 17 | 23.02x / 5.95x | 52% / 7% | 54% / 2% | 21.6 / 4.8 |
| level1/36_RMSNorm_ | 106.0 / 95.7 | 244 / 19 | 4.58x / 3.94x | 70% / 7% | 25% / 2% | 12.4 / 3.7 |
| level2/99_Matmul_GELU_Softmax | 111.1 / 123.4 | 190 / 13 | 20.67x / 6.24x | 63% / 5% | 38% / 2% | 20.2 / 5.2 |
| level3/1_MLP | 128.9 / 157.7 | 155 / 15 | 11.52x / 9.58x | 75% / 8% | 31% / 12% | 16.0 / 6.0 |

Each cell is SpecGen / CudaForge. Geometric means, SpecGen relative to CudaForge: end-to-end time 1.05x, profiled
kernels 11.7x, best kernel speedup 2.06x. The report's estimates for `hist_avg`: end-to-end time 2.34x, best kernel
speedup 1.75x; for `first_valid`: 3.85x and 1.68x. In 1_Square_matrix_multiplication_, SpecGen's 21.94x and 23.02x
(iterations 18 and 19) are FP8 (e4m3) kernels; without FP8, its best is 15.87x.

- `ae/reference/report.html`: the page `ae/report.py` writes; `ae/reference/summary.md`, `summary.csv`: the tables.
- `ae/reference/records/`: the records of the eight runs (`config.json`, `events.jsonl`, `iterations.jsonl`,
  `kernels.jsonl`). `python ae/report.py --runs ae/reference/records --prefix ref --output /tmp/ref` rebuilds the
  tables and the page from them (without the reasoning excerpts).

## Results

`ae_results/` (`AE_OUTPUT`):

- `report.html` (`start_report.html` for the getting-started test): a comparison table across SpecGen's termination
  criteria and comparison dimensions; then, per task, a replayable timeline of each run, best speedup over time,
  termination criteria, iterations and kernels.
- `summary.md`, `summary.csv` (`start_summary.*`): per task and the geometric means: E2E time, profiled kernels,
  best kernel speedup over the PyTorch reference, profiler and validator GPU utilization, input (cache hit / miss),
  output and equivalent tokens.
- `runs/<stage>-<task>-<method>/`: everything a run recorded; `logs/`: the logs; `ae.log`: what `ae/run.sh` did.
  `bash ae/run.sh report` rebuilds the reports.

## Systems compared

- **SpecGen**, this repository (`specgen run`): speculative generations launched from the reasoning prefix,
  validation and profiling GPUs reallocated by load, early termination when a kernel beats the mean speedup of the
  kernels selected in earlier iterations (`--termination-criteria hist_avg`).
- **CudaForge**: we recommend SpecGen's baseline mode (`specgen run --no-enable-spec-gen`, what `ae/run.sh`
  runs). It follows CudaForge's workflow (one reasoning generation per iteration, its kernel validated and then
  profiled, the hardware feedback in the next prompt) in the same harness as SpecGen, so both use the same model,
  prompts, tasks, GPUs, timing and correctness checks. Beyond the original it supports more GPU configurations,
  gives richer profiling information, handles validation and profiling exceptions and errors, and checks
  correctness more robustly (several trials and a trial with inputs from a shifted distribution). The original
  implementation is <https://github.com/OptimAI-Lab/CudaForge> (set its parallel k as needed).
- **KernelAgent and AlphaEvolve (OpenEvolve)**, not included. The paper ran their latest versions as of April 2026
  with their default configurations, early termination and the other pruning strategies enabled, parallel search
  enabled, the same LLM endpoints, tasks T1-T10 and 100 iterations per task:
  - KernelAgent: <https://github.com/meta-pytorch/KernelAgent>, commit
    [`2d8ed0d`](https://github.com/meta-pytorch/KernelAgent/commit/2d8ed0d518e2c5ec834a1acd1bc41607d7591a7f)
    (2026-04-29);
  - AlphaEvolve, OpenEvolve implementation: <https://github.com/algorithmicsuperintelligence/openevolve>, commit
    [`80945ed`](https://github.com/algorithmicsuperintelligence/openevolve/commit/80945ed82886d5c4ff2f3d22436765d50cb61266)
    (2026-03-18, v0.2.27).

## Reusing SpecGen

`docs/architecture.md` describes the code and where to extend it; `specgen run --help` lists every option.

- **Another configuration of the experiments:** `AE_EXTRA_ARGS` adds flags to every `specgen run` of `ae/run.sh`,
  e.g. another termination criterion (`--termination-criteria first_valid`), fixed GPU roles
  (`--gpu-allocation fixed --num-validator-gpus 4 --num-profiler-gpus 4`) or a cap on concurrent LLM requests
  (`--max-inflight-spec-gen 4`). Give each configuration its own `AE_OUTPUT`.
- **Another task:** any Python file in KernelBench's format (`class Model(nn.Module)`, `get_inputs()`,
  `get_init_inputs()`), with SpecGen's command line:

  ```bash
  export SPECGEN_DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY
  specgen run --task my_task.py --models-file ae/models.yaml --model DeepSeek-V4-Pro --endpoint api \
      --gpus 0,1,2,3 --cuda-arch auto --max-iterations 10 --enable-termination --termination-criteria hist_avg \
      --run-id mine-specgen
  specgen run --task my_task.py --models-file ae/models.yaml --model DeepSeek-V4-Pro --endpoint api \
      --gpus 4,5,6,7 --cuda-arch auto --max-iterations 10 --no-enable-spec-gen --run-id mine-baseline
  specgen report runs/mine-specgen --compare runs/mine-baseline --label SpecGen --compare-label CudaForge
  ```

  To run it through `ae/run.sh`, add it to `TASKS` in `ae/ae.py` and pass `--tasks NAME`.
- **Another LLM:** a profile in a models file (see `ae/models.yaml`; OpenAI-compatible endpoints need no code):
  where the reasoning streams (`reasoning: field` with `reasoning_field`, or inline think tags), sampling, request
  parameters per generation mode (`modes`), and whether the endpoint continues a partial assistant message
  (`prefill`). `tools/probe_stream.py` shows how an endpoint streams. Anthropic and Gemini backends are included.
- **Another GPU type:** an entry in `src/specgen/generation/gpu_specs.py` and in `SUPPORTED_GPUS` of `ae/ae.py`.
- **Tests:** `pytest -m "not gpu"` runs anywhere in a few minutes; `pytest -m gpu` needs an idle GPU.

## Tasks

T1-T10 (Table 1, KernelBench Level 1): `100_HingeLoss`, `10_3D_tensor_matrix_multiplication`,
`11_4D_tensor_matrix_multiplication`, `12_Matmul_with_diagonal_matrices_`, `13_Matmul_for_symmetric_matrices`,
`14_Matmul_for_upper_triangular_matrices`, `15_Matmul_for_lower_triangular_matrices`,
`16_Matmul_with_transposed_A`, `17_Matmul_with_transposed_B`, `18_Matmul_with_transposed_both`.
T11-T20 (Table 3, Level 2/3): `12_Gemm_Multiply_LeakyReLU`, `14_Gemm_Divide_Sum_Scaling`,
`33_Gemm_Scale_BatchNorm`, `52_Conv2d_Activation_BatchNorm`, `56_Matmul_Sigmoid_Sum`,
`73_Conv2d_BatchNorm_Scaling`, `76_Gemm_Add_ReLU`, `99_Matmul_GELU_Softmax`, Level 3 `1_MLP` and
`50_ReLUSelfAttention`. The getting-started test uses `level1/1_Square_matrix_multiplication_`.

## Troubleshooting

- *A run stopped*: run the same command again; it resumes from the last finished iteration.
- *`ncu` performance counters not accessible*: see Requirements.
- *API errors or rate limits*: persistent errors stop the run; run the command again to resume. With a low rate
  limit: `AE_EXTRA_ARGS="--max-inflight-spec-gen 4"`.
- *Out of GPU memory in validation*: run fewer at once per GPU, e.g. `AE_VALIDATIONS_PER_GPU=2`.
- *Progress*: `tail -f ae_results/ae.log`.

## License and citation

SpecGen is released under the MIT License (`LICENSE`); the KernelBench tasks keep their MIT License
(`KernelBench/LICENSE`). To cite SpecGen, see `CITATION.cff`.
