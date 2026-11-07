from specgen.config import parse_args
from specgen.control.attempts import failure, failure_kind, family, features, self_reported, summarize, tiling
from specgen.control.compliance import forbidden_usage
from specgen.control.history import KernelOutcome, TaskHistory
from specgen.generation.prompts import PromptBuilder
from specgen.tasks.base import KernelBenchTask

MMA = '''# APPROACH: FP16 mma.sync 128x128x32, 3-stage cp.async, fused epilogue
cuda = r"""
__global__ void k(const __half* a) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...};");
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;");
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {...};");
}
"""
'''
INT8 = '# APPROACH: INT8 GEMM\ncuda = "asm(\\"mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32\\")"'
SIMT = '# APPROACH: fp32 tiled GEMM\ncuda = "__global__ void k(float* x) { float4 v; __shfl_xor_sync(0xffffffff, v.x, 1); }"'
LIBRARY = "# APPROACH: cuBLAS then fused epilogue\ny = torch.matmul(x, w)"
COMMENTED = '// mma.sync would be faster\n# int8 is tempting\ncuda = "__global__ void k(float* x) {}"'
TF32_WMMA = ('# APPROACH: TF32 WMMA 128x128x32\ncuda = "wmma::fragment<wmma::matrix_a, 16, 16, 8, '
             'wmma::precision::tf32, wmma::row_major> a; a.x[0] = wmma::__float_to_tf32(v);"')
FP16_WMMA = '# APPROACH: FP16 WMMA\ncuda = "wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;"'
TF32_MMA = 'cuda = "asm(\\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\\")"'
NAMED_TF32 = 'cuda = "__global__ void gemm_tf32_kernel(float* x) { int half = 2; }"'


def outcome(kernel_id, iteration, code, status="success", speedup=None, error=None):
    return KernelOutcome(kernel_id, iteration, "spec", code, status=status, error=error, speedup=speedup,
                         valid=status == "success" and speedup is not None)


def test_features_come_from_the_code_not_the_comments():
    assert features(MMA) == ("fp16", "mma.sync", ("cp.async", "ldmatrix"))
    assert features(INT8)[:2] == ("int8", "mma.sync")
    assert features(SIMT) == ("fp32", "SIMT", ("warp shuffle", "vectorized"))
    assert features(LIBRARY)[1] == "library GEMM"
    assert features(COMMENTED) == ("fp32", "SIMT", ())
    assert family(MMA) == "fp16 mma.sync + cp.async, ldmatrix"
    assert self_reported(MMA) == "FP16 mma.sync 128x128x32, 3-stage cp.async, fused epilogue"
    assert self_reported("import torch") == "" and self_reported("# APPROACH:\nimport torch") == ""


def test_precision_comes_from_instructions_and_types():
    assert features(TF32_WMMA)[:2] == ("tf32", "wmma")  # its wmma:: fragments do not make it fp16
    assert features(FP16_WMMA)[:2] == ("fp16", "wmma")
    assert features(TF32_MMA)[:2] == ("tf32", "mma.sync")
    assert features(NAMED_TF32)[:2] == ("fp32", "SIMT")  # a name or a variable called half is no evidence


def test_tile_and_stages_from_the_self_report():
    assert tiling("FP16 mma.sync 128x128x32, 3-stage cp.async, fused epilogue") == "128x128x32, 3 stages"
    assert tiling("FP16 mma.sync GEMM, 128x128x64 tile, 8 warps/64x32 warp tile, 4-stage cp.async") == "128x128x64, 4 stages"
    assert tiling("4x2 warps, warp tile 64x32, block tile 128 × 256, double buffering") == "128x256, 2 stages"
    assert tiling("16x16 threads per block, 5 stages") == "5 stages"
    assert tiling("SIMT, 4096x4096 problem") == "" and tiling("") == ""


def test_failures_are_grouped_by_their_normalized_message():
    nvcc = 'Error building extension\n/x/y/cuda.cu(57): error: identifier "__cvta_generic_to_shared" is undefined\n'
    assert failure("compile_error", nvcc) == ("identifier … is undefined",
                                             'identifier "__cvta_generic_to_shared" is undefined')
    cuda = "Traceback\ntorch.AcceleratorError: CUDA error: misaligned address\nCUDA kernel errors might be reported"
    assert failure("cuda_error", cuda)[0] == "CUDA error: misaligned address"
    missing = "Traceback\nAttributeError: module 'specgen_kernel' has no attribute 'Model'"
    assert failure("runtime_error", missing)[0] == "AttributeError: module … has no attribute …"
    shifted = 'output mismatch on shifted inputs (2x - mean|x|/2)\n{"rel_err": 0.28, "trial": "shift"}'
    assert failure("numerical_error", shifted) == ("wrong results on shifted inputs", "relative error 0.28")
    plain = 'output mismatch in trial 0\n{"reason": "value_mismatch", "rel_err": 1.1, "trial": 0}'
    assert failure("numerical_error", plain) == ("wrong results", "relative error 1.1")
    assert failure("timeout", None) == ("timed out", "")
    syntax = "SyntaxError: unterminated triple-quoted string literal (detected at line 421)"
    assert failure("compile_error", syntax)[0] == "SyntaxError: unterminated triple-quoted string literal (detected at line N)"


def test_builds_failing_during_validation_show_the_compiler_error():
    late = ("Traceback\nRuntimeError: Error building extension 'gemm_tf32_ext': [1/2] /usr/local/cuda/bin/nvcc "
            "-DTORCH_EXTENSION_NAME=gemm_tf32_ext -c cuda.cu\nFAILED: cuda.cuda.o\n"
            "/tmp/ext/cuda.cu(88): error: identifier \"wm\" is undefined\n")
    assert failure_kind("runtime_error", late) == "compile_error"
    assert failure_kind("cuda_error", "CUDA error: misaligned address") == "cuda_error"
    assert failure("runtime_error", late) == ("identifier … is undefined", 'identifier "wm" is undefined')
    other = late.replace("gemm_tf32_ext", "fused_linear_relu").replace('"wm"', '"acc"')
    assert failure("runtime_error", other)[0] == failure("runtime_error", late)[0]
    silent = "RuntimeError: Error building extension 'x': [1/2] /usr/local/cuda/bin/nvcc -c cuda.cu"
    assert failure("runtime_error", silent)[0] == "the extension failed to build (no compiler error in the log)"
    link = ("main.o: in function `bind':\nmain.cpp:(.text+0x1c): undefined reference to `launch_gemm(float const*, "
            "float*, int)'\ncollect2: error: ld returned 1 exit status")
    assert failure("compile_error", link)[0] == "undefined reference to …"
    symbol = "ImportError: /tmp/x/gemm_ext.so: undefined symbol: _Z18launch_transpose_bPKfPviiP11CUstream_st"
    assert failure("runtime_error", symbol) == (
        "ImportError: <path>: undefined symbol: …",
        "ImportError: gemm_ext.so: undefined symbol: launch_transpose_b(float const*, void*, int, int, CUstream_st*)")


def test_import_errors_from_unmatched_declarations_fail_the_build(tmp_path, monkeypatch):
    from specgen.evaluation import build
    from specgen.evaluation.jobs import EvalJob

    kernel = tmp_path / "kernel.py"
    kernel.write_text("raise ImportError('/w/torch_extensions/ext/ext.so: undefined symbol: _Z6launchPvi')\n")
    monkeypatch.setenv("SPECGEN_FAST_INCLUDES", "0")
    monkeypatch.setenv("SPECGEN_PCH", "0")
    result = build.run(EvalJob("build", "kernelbench", "t.py", str(kernel), str(tmp_path)))
    assert result["status"] == "compile_error" and not result["ok"]
    kernel.write_text("raise RuntimeError('no CUDA GPUs are available')\n")
    assert build.run(EvalJob("build", "kernelbench", "t.py", str(kernel), str(tmp_path)))["status"] == "build_skipped"


def test_summary_table():
    outcomes = [outcome("a", 0, MMA, speedup=1.5), outcome("b", 0, MMA, speedup=1.2),
                outcome("c", 1, LIBRARY, speedup=0.7),
                *[outcome(f"s{i}", 1, SIMT, "cuda_error", error="CUDA error: misaligned address") for i in range(5)],
                outcome("d", 1, INT8, "numerical_error", error='mismatch on shifted inputs {"rel_err": 0.3, "trial": "shift"}'),
                outcome("x", 1, MMA, "cancelled")]
    text = summarize(outcomes, best_kernel="a")
    assert text.startswith("Approaches tried so far (iterations 0-1, 9 kernels evaluated")
    rows = [line for line in text.splitlines() if line.startswith("| ") and "---" not in line][1:]
    assert rows == [
        "| fp16 mma.sync + cp.async, ldmatrix | 128x128x32, 3 stages: 2/2 1.500x | FP16 mma.sync 128x128x32, 3-stage "
        "cp.async, fused epilogue | 2 | 2 | 1.500x | best so far | - |",
        "| fp32 library GEMM + library call | - | cuBLAS then fused epilogue | 1 | 1 | 0.700x | slower than the reference "
        "| - |",
        "| fp32 SIMT + warp shuffle, vectorized | - | fp32 tiled GEMM | 5 | 0 | - | dead end: all 5 failed (5 cuda error) "
        "| - |",
        "| int8 mma.sync | - | INT8 GEMM | 1 | 0 | - | all failed (1 numerical error) | - |",
    ]
    assert ("- 5x cuda error: CUDA error: misaligned address (check alignment before vectorized or cp.async accesses) "
            "[fp32 SIMT + warp shuffle, vectorized]") in text
    assert ("- 1x numerical error: wrong results on shifted inputs — e.g. relative error 0.3 "
            "(do not assume the input range of get_inputs()) [int8 mma.sync]") in text
    assert summarize([outcome("x", 0, MMA, "cancelled")]) == ""
    crashed = summarize([outcome("i", 0, MMA, "cuda_error", error="CUDA error: an illegal memory access was encountered")])
    assert "48 KB of dynamic shared memory" in crashed and "cudaFuncAttributeMaxDynamicSharedMemorySize" in crashed


def test_summary_shows_each_approachs_ncu_profile_and_forbidden_modules():
    best = outcome("a", 0, MMA, speedup=1.5)
    best.ncu_profile = {"kernels": [{"name": "gemm", "launches": 1, "share": 1.0, "metrics": {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 62.0,
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": 70.0,
        "sm__warps_active.avg.pct_of_peak_sustained_active": 12.0,
        "launch__occupancy_limit_registers": 1.0, "launch__occupancy_limit_shared_mem": 2.0},
        "rules": [{"rule": "TheoreticalOccupancy", "type": "OPT", "speedup": 43.9}]}]}
    cheat = "# APPROACH: hand-written\nimport torch\ndef f(A, B):\n    return torch.matmul(A, B)\n"
    outcomes = [best, outcome("b", 0, MMA, speedup=1.2), outcome("c", 1, cheat, "forbidden", error=forbidden_usage(cheat))]
    text = summarize(outcomes, best_kernel="a")
    assert ("| best so far | SM 62%, tensor 70%, occupancy 12% (limit: registers); advice: TheoreticalOccupancy "
            "+43.9% |") in text
    assert "| fp32 library GEMM + library call | - | hand-written | 1 | 0 | - | all failed (1 forbidden) | - |" in text
    assert ("- 1x forbidden: uses a library GEMM or convolution instead of a hand-written kernel — e.g. torch.matmul() "
            "(do all the work in hand-written CUDA kernels: no library call, no fallback to one, no TF32 switch) "
            "[fp32 library GEMM + library call]") in text


def test_summary_lists_tile_variants_and_counts_late_builds_as_compile_errors():
    wide = MMA.replace("128x128x32, 3-stage", "128x256x32, 4-stage")
    late = "RuntimeError: Error building extension 'e': [1/2] nvcc\n/t/cuda.cu(9): error: expected a \";\"\n"
    outcomes = [outcome("a", 0, MMA, speedup=1.5), outcome("b", 0, wide, speedup=1.1),
                *[outcome(f"w{i}", 1, wide, "runtime_error", error=late) for i in range(2)]]
    text = summarize(outcomes, best_kernel="a")
    assert ("| fp16 mma.sync + cp.async, ldmatrix | 128x128x32, 3 stages: 1/1 1.500x; 128x256x32, 4 stages: 3/1 "
            "1.100x | FP16 mma.sync 128x128x32, 3-stage cp.async, fused epilogue | 4 | 2 | 1.500x | best so far |") in text
    assert ("- 2x compile error: expected a … — e.g. expected a \";\" "
            "[fp16 mma.sync + cp.async, ldmatrix | 128x256x32, 4 stages]") in text


def test_capped_tables_list_four_tile_variants():
    codes = [MMA.replace("128x128x32", f"{m}x128x32") for m in (16, 32, 64, 128, 256, 512)]
    outcomes = [outcome(f"k{i}", 0, code, speedup=1.0 + i / 10) for i, code in enumerate(codes)]
    capped = summarize(outcomes, best_kernel="k5")
    assert ("512x128x32, 3 stages: 1/1 1.500x; 256x128x32, 3 stages: 1/1 1.400x; 128x128x32, 3 stages: 1/1 1.300x; "
            "64x128x32, 3 stages: 1/1 1.200x; 2 more: 2/2 |") in capped
    uncapped = summarize(outcomes, best_kernel="k5", max_approaches=0, max_failures=0, max_chars=0)
    assert "16x128x32, 3 stages: 1/1 1.000x |" in uncapped and "more:" not in uncapped


def test_summary_limits():
    kinds = ["", "__half x;", "__nv_bfloat16 x;", "precision::tf32", "wmma::fragment f;", "atomicAdd(p, 1);",
             "split_k = 4;", "__shfl_down_sync(m, v, 1);", "float4 v;", "torch.matmul(a, b)"]
    outcomes = [outcome(f"k{i}", 0, f'cuda = "{body}"', speedup=1.0 + i / 10) for i, body in enumerate(kinds)]
    outcomes += [outcome(f"f{c}", 0, 'cuda = ""', "runtime_error", error=f"ValueError: case {c}") for c in "abcdefg"]
    capped = summarize(outcomes, max_approaches=8, max_failures=5, max_chars=0)
    assert "| 2 other approaches |" in capped  # 10 kinds (the failing kernels share the first): 8 shown
    assert sum(line.startswith("- 1x runtime error") for line in capped.splitlines()) == 5
    assert "- 2 more kinds of failure (2 kernels)" in capped
    unlimited = summarize(outcomes, max_approaches=0, max_failures=0, max_chars=0)
    assert "other approaches" not in unlimited and "more kinds" not in unlimited
    assert sum(line.startswith("- 1x runtime error") for line in unlimited.splitlines()) == 7
    short = summarize(outcomes, max_approaches=0, max_failures=0, max_chars=400)
    assert len(short) <= 400 and short.endswith("(table cut to fit the prompt)")


def test_summary_prompt_keeps_only_the_best_kernel_code(relu_task):
    history = TaskHistory()
    best, other = outcome("i000-t01-s0", 0, MMA, speedup=1.5), outcome("i000-t01-s1", 0, SIMT, speedup=1.1)
    for o in (best, other):
        history.add(o)
    history.select(0, best)
    context = history.prompt_context(1, 3, mode="summary", max_approaches=8, max_failures=5, max_chars=6000)
    assert context.best.source == MMA and context.feedback == () and context.failure is None
    sections = dict(PromptBuilder("H100").build(KernelBenchTask.from_path(relu_task), context).sections)
    assert sections["feedback"] == "" and "| fp32 SIMT + warp shuffle, vectorized |" in sections["attempts"]
    assert SIMT not in sections["attempts"] and MMA in sections["best"]
    topk = history.prompt_context(1, 3)
    assert topk.attempts == "" and topk.feedback[0].source == SIMT
    both = history.prompt_context(1, 3, mode="topk+summary", max_approaches=0, max_failures=0, max_chars=0)
    assert both.best.source == MMA and both.feedback[0].source == SIMT and "| fp32 SIMT" in both.attempts
    sections = dict(PromptBuilder("H100").build(KernelBenchTask.from_path(relu_task), both).sections)
    assert SIMT in sections["feedback"] and "Approaches tried so far" in sections["attempts"]


def test_history_prompt_flags(relu_task):
    base = ["--task", str(relu_task), "--gpus", "none"]
    defaults = parse_args(base).run  # top-3 kernels plus the full table
    assert (defaults.history_prompt, defaults.feedback_top_k, defaults.summary_max_approaches,
            defaults.summary_max_failures, defaults.summary_max_chars) == ("topk+summary", 3, 0, 0, 0)
    run = parse_args(base + ["--history-prompt", "summary", "--summary-max-approaches", "8",
                             "--summary-max-failures", "5", "--summary-max-chars", "6000"]).run
    assert (run.history_prompt, run.summary_max_approaches, run.summary_max_failures, run.summary_max_chars) == (
        "summary", 8, 5, 6000)
    assert parse_args(base + ["--history-prompt", "topk", "--feedback-top-k", "5"]).run.feedback_top_k == 5
