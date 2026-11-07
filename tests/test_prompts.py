from specgen.generation.prompts import (
    SECTION_ORDER,
    KernelFeedback,
    PromptBuilder,
    PromptContext,
    summarize_error,
)
from specgen.tasks.base import KernelBenchTask


def fb(speedup, source="kernel", profile=None):
    return KernelFeedback(source=source, speedup=speedup, latency_ms=1.0 / speedup, ref_latency_ms=1.0,
                          ncu_profile=profile)


def one_kernel(metrics):
    return {"kernels": [{"name": "k", "launches": 1, "share": 1.0, "metrics": metrics, "rules": []}]}


def test_first_iteration(relu_task):
    prompt = PromptBuilder("H100").build(KernelBenchTask.from_path(relu_task))
    assert [name for name, _ in prompt.sections] == list(SECTION_ORDER)
    assert prompt.user.startswith("Target GPU: H100")
    assert "Reference implementation" in prompt.user and "torch.relu(x)" in prompt.user
    assert "Best kernel so far" not in prompt.user
    assert prompt.messages()[0] == {"role": "system", "content": prompt.system}


def test_gpu_block_names_the_compute_capability_and_builds_pick_the_arch():
    from specgen.generation.gpu_specs import GPU_SPECS, format_gpu_specs

    assert "- GPU Architecture: Blackwell Ultra\n- Compute capability: 10.3 (sm_103)\n" in format_gpu_specs("B300")
    assert all("- Compute capability: " in format_gpu_specs(gpu) for gpu in GPU_SPECS)
    assert "Do not pass `-arch`, `-gencode` or `-code` flags" in PromptBuilder("B300").system
    assert "cudaFuncAttributeMaxDynamicSharedMemorySize" in PromptBuilder("B300").system


def test_stable_prefix_across_iterations(relu_task):
    builder, task = PromptBuilder("H100"), KernelBenchTask.from_path(relu_task)
    first = builder.build(task)
    best = fb(1.5, "best kernel")
    second = builder.build(task, PromptContext(best=best, feedback=(fb(1.2, "k2"),)))
    third = builder.build(task, PromptContext(best=best, feedback=(fb(1.1, "k3"),)))
    static = "\n\n".join(text for name, text in first.sections if name in ("gpu", "task"))
    assert second.user.startswith(static) and third.user.startswith(static)
    best_block = dict(second.sections)["best"]
    assert second.user.startswith(f"{static}\n\n{best_block}") and third.user.startswith(f"{static}\n\n{best_block}")
    assert first.system == second.system == third.system


def test_deterministic_metrics(relu_task):
    builder, task = PromptBuilder("H100"), KernelBenchTask.from_path(relu_task)
    sm, tensor = "sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
    a = builder.build(task, PromptContext(best=fb(2.0, profile=one_kernel({sm: 61.7, tensor: 7.25}))))
    b = builder.build(task, PromptContext(best=fb(2.0, profile=one_kernel({tensor: 7.25, sm: 61.7}))))
    assert a.user == b.user
    assert "- `k`:\n  - throughput (% of peak): SM 62%, tensor pipe 7.2%" in a.user
    assert "2.000x" in a.user


def test_failure_feedback(relu_task):
    log = "\n".join(f"noise line {i}" for i in range(2000)) + "\nkernel.cu(12): error: identifier \"x\" is undefined\n"
    ctx = PromptContext(failure=KernelFeedback(source="bad code", status="compile_error", error=log))
    prompt = PromptBuilder("H100", max_error_chars=2000).build(KernelBenchTask.from_path(relu_task), ctx)
    feedback = dict(prompt.sections)["feedback"]
    assert "no correct kernel" in feedback and "Result: compile_error" in feedback
    assert 'error: identifier "x" is undefined' in feedback
    assert len(feedback) < 3000


def test_summarize_error_short_text_unchanged():
    assert summarize_error("short", 100) == "short"
