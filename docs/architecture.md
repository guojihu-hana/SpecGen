# SpecGen architecture

SpecGen optimizes GPU kernels with a reasoning LLM, iteration by iteration. While the main generation reasons,
SpecGen forks speculative (non-reasoning) generations from the reasoning prefix, sends every kernel through a
build, validation and profiling pipeline on a pool of GPUs, and can end the iteration as soon as a kernel meets the
termination criterion.

## 1. One iteration

```text
IterationLoop (one per task; the tasks of a run share the resource pool, the spec budget and the LLM endpoint)
  1. PromptBuilder: the task, the best kernel so far, the fastest kernels of the last iteration, the table of
     approaches tried (timings, ncu summaries, errors)
  2. Generator.generate_main (streaming; the model profile separates reasoning from content)
       reasoning deltas -> spec scheduler: demand (default) or keyword triggers -> Generator.generate_spec
  3. main and spec kernels -> dedup -> Builder (CPU pool) -> Validator (several per GPU)
       valid -> Profiler: timing (exclusive GPU) -> TerminationController
                          the kernels the next prompts show -> ncu (exclusive GPU, when one is free)
  4. end: as soon as a kernel meets the termination criterion; otherwise once the main kernel is evaluated,
     after waiting at most grace_ratio x the main generation's duration for the other candidates

PerfRecorder: every event goes to events.jsonl, with running statistics (durations, pass rates, utilization)
```

Orchestration is `asyncio`: LLM streams, spec requests, subprocesses and cancellation are tasks, and ending an
iteration cancels its group. LLM calls run in-process (for token-level events); GPU jobs run in isolated
subprocesses with a fixed `CUDA_VISIBLE_DEVICES`, a private `TORCH_EXTENSIONS_DIR` per kernel and a timeout that
kills the process group.

## 2. Code map

| Package | What it holds |
|---|---|
| `specgen/cli.py`, `specgen/config.py` | `specgen run / summarize / report / worker`; every option and its default (`specgen run --help`) |
| `specgen/llm/` | model profiles (`profiles.py`), streaming backends: OpenAI-compatible (`openai_compat.py`, also vLLM, SGLang, DeepSeek, Together), Anthropic, Gemini, file queue; reasoning parsing, token counting |
| `specgen/generation/` | prompts and their templates, main and spec generation (`generator.py`), code extraction, GPU specification tables (`gpu_specs.py`), ncu summaries for prompts |
| `specgen/control/` | the iteration loop (`loop.py`), spec scheduling (`pacing.py` demand, `triggers.py` + `budget.py` keyword triggers), termination criteria, history and the table of approaches (`attempts.py`), compliance checks |
| `specgen/evaluation/` | build (with fast header handling), validation, timing, ncu, run in child processes (`child.py`, `executor.py`) |
| `specgen/resources/` | the GPU/CPU resource pool and its allocation policies (`pool.py`), GPU discovery |
| `specgen/remote/` | `specgen worker`: evaluation on GPUs of other machines over HTTP |
| `specgen/telemetry/`, `specgen/storage/` | the event recorder and the run directory layout |
| `specgen/analysis/` | loading a run (`data.py`), summaries, termination-criterion estimates (`criteria.py`), the HTML report |
| `specgen/tasks/` | task loading: a KernelBench-style file with `Model`, `get_inputs()` and `get_init_inputs()` |

## 3. Generation

- **Model profiles** (`configs/models.yaml`, `ae/models.yaml`): per model, where the reasoning arrives (`field`, a
  separate delta field such as `reasoning_content`, or `inline` between think tags), the tokenizer for counting,
  sampling settings, and its endpoints (`backend`, `base_url`, `model`, `api_key_env`). `${VAR}` in a value comes
  from the environment; keys are only read from environment variables. `modes` adds request parameters per
  generation (`main` / `spec`), e.g. DeepSeek's `thinking` switch.
- **Spec prompts** (`--spec-prompt-mode`): `prefill` continues a partial assistant message that holds the
  reasoning prefix and the start of the code block (endpoints declare how: DeepSeek's `/beta` prefix completion or
  vLLM's `continue_final_message`); `reprompt` restates the prefix in a new prompt; `auto` picks prefill when the
  endpoint supports it.
- **Spec scheduling** (`--spec-schedule`):
  - `demand` (default): one spec at a time whenever the kernels expected at the profiling GPUs fall short of a
    target (`--spec-target-extra`, `--spec-target-factor`); the prefix of each new spec ends with the test results
    of the iteration's earlier candidates. At least `--spec-min-gap-tokens` reasoning tokens apart.
  - `triggers`: keyword triggers in the reasoning (`configs/triggers.yaml`) launch k specs, k sized by the global
    `SpecBudget` from the capacity of the evaluation pipeline (`--spec-k auto`, or a fixed k).
- Spec jobs are queued last-in first-out (newer prefixes first); the main kernel and confirmation timings go first.

## 4. Evaluation

- **Build**: a CPU pool (`--build-workers`) builds every kernel's `load_inline` extension ahead of validation, for
  the validation GPUs' architecture (`--cuda-arch auto`). Builds try progressively fuller torch headers and use a
  private precompiled header for the C++ side.
- **Validation**: several random input sets against the reference (`--atol`, `--rtol`, also a relative L2 bound),
  outputs pre-filled with NaN so that unwritten elements fail, and one more trial on shifted inputs
  (`--validation-shift-trial`). Several validations share a GPU (`--validations-per-gpu`); out-of-memory jobs retry
  alone.
- **Timing**: CUDA events, `bench_warmup` warm-up runs, then `bench_rounds` x `bench_iters` timed runs; the median,
  on an exclusive GPU. Speedup = reference time / kernel time.
- **ncu**: only the kernels the next prompts will show (`--ncu-top-n`), on an exclusive GPU when one is free; the
  third forward pass of the module, all its CUDA kernels. Without permission for the GPU performance counters the
  run continues without ncu.
- **Compliance**: kernels must not call library GEMM/convolution (cuBLAS, cuDNN, CUTLASS, `torch.matmul`, ...) or
  change torch's precision settings; checked on the source before the build and on the kernels a forward pass
  actually launches.

## 5. Resource pool

A central dispatcher places jobs on devices by role and capacity; priorities: timing, validation, ncu; within a
kind, main before spec. Exclusive jobs reserve a device and wait for it to drain. Allocation (`--gpu-allocation`):
`fixed` (`--num-validator-gpus`, `--num-profiler-gpus`), `dynamic` (default: idle GPUs move between the validation
and profiling roles by load, one at a time, with hysteresis), `fifo` (any GPU takes any job in arrival order).
`--remote-worker URL` adds the GPUs of a `specgen worker` on another machine.

## 6. Ending an iteration

- **Baseline mode** (`--no-enable-spec-gen`): one kernel per iteration from the main generation, evaluated, then the
  next iteration; no spec, no early termination. The artifact evaluation uses it as CudaForge.
- **Early termination** (`--enable-termination --termination-criteria ...`): `first_valid` (the first kernel that
  passes validation and timing), `best` (beats the best so far), `hist_avg` (beats the mean speedup of the kernels
  earlier iterations selected); iteration 0 compares with the reference (1.0x). `--termination-margin` (default
  0.03) and `--termination-confirm` (default 2 timings) guard against timing noise. Ending cancels the main stream,
  the specs in flight and the iteration's jobs (each task has an epoch number; stale results are only recorded).
- **Natural end**: after the main kernel, the other candidates get `--grace-ratio` x the main generation's duration.
- **Resume** (`--resume --run-id ID --max-iterations N`): continues a stopped run from its first unfinished
  iteration, rebuilding the history from the records.

## 7. Prompts

Ordered from the most to the least stable part so that prefix caches hit: system rules; GPU specification;
task and reference code; the best kernel so far; then what changes every iteration (the last iteration's fastest
kernels or the main kernel's error, `--history-prompt`, and the table of approaches tried, built by rules from the
kernels' self-reported `# APPROACH:` lines, their code and their errors).

## 8. Run directory

```text
<output_root>/<run_id>/
  config.json  events.jsonl  kernels.jsonl  iterations.jsonl
  <task>/iter_000/prompt.json  main/{reasoning.txt, content.txt}
  <task>/iter_000/kernels/<kernel_id>/{kernel.py, build.json, validate.json, benchmark.json, ncu.json, *.log}
```

Kernel ids: `i000-main` is iteration 0's main kernel, `i000-t03-s1` a spec of iteration 0 (trigger 03, spec 1 of
that trigger). `events.jsonl` is the source of truth; `kernels.jsonl` and `iterations.jsonl` are per-kernel and
per-iteration records derived from it.

## 9. Analysis

- `specgen summarize RUN_DIR`: per iteration and per task, end reasons, selected kernels, pass rates, queueing and
  run times, GPU utilization.
- `specgen report RUN_DIR [--compare OTHER_RUN]`: a self-contained HTML page with a replayable timeline (main and
  spec generations, every job on its GPU slot), speedup over time, the comparison with another run, and estimates
  of the other termination criteria (`analysis/criteria.py` cuts the recorded timeline where a criterion would have
  ended each iteration, optionally within a time or token budget).
- `ae/report.py`: the artifact evaluation's page and summary tables for pairs of runs.

## 10. Extending SpecGen

- **A task**: any Python file in KernelBench's format (`class Model(nn.Module)`, `get_inputs()`,
  `get_init_inputs()`); `specgen run --task my_task.py`.
- **A model or provider**: a profile in a models file (`--models-file`); OpenAI-compatible endpoints need no code.
  `tools/probe_stream.py` shows how an endpoint streams reasoning and reports usage, and whether it can continue a
  partial assistant message. Other protocols implement the backend interface in `specgen/llm/backend.py`.
- **A GPU**: an entry in `generation/gpu_specs.py` (the table the prompts show, name patterns, compute
  capability).
- **A termination criterion**: `control/termination.py` (the run) and `analysis/criteria.py` (the estimates).
- **An allocation policy**: `resources/pool.py`.
- **A spec scheduler**: `control/pacing.py` (demand) or `control/triggers.py` and `configs/triggers.yaml`.

The tests (`pytest -m "not gpu"` on any machine, `pytest -m gpu` on a GPU machine) cover each of these parts.
