# SpecGen

SpecGen optimizes GPU kernels with a reasoning LLM and speculative generation. An agent improves a CUDA kernel
iteration by iteration; while the LLM reasons, SpecGen forks non-reasoning generations from the reasoning prefix,
validates and profiles their kernels on a pool of GPUs as the reasoning goes on, feeds the results back, and ends
an iteration early when a kernel meets the termination criterion.

Paper: *SpecGen: Accelerating Agentic Kernel Optimization with Speculative Generation*, PPoPP 2027
([arXiv:2606.17518](https://arxiv.org/abs/2606.17518)).

- **How the code works and how to extend it:** [docs/architecture.md](docs/architecture.md).

## Quick start

Requirements: Linux, NVIDIA GPUs, the CUDA toolkit (`nvcc`), Python 3.10+ with PyTorch for CUDA, and an LLM
endpoint (profiles in `configs/models.yaml`).

```bash
pip install -e ".[tokens,dev]"
export SPECGEN_DEEPSEEK_API_KEY=...
specgen run --task KernelBench/level1/1_Square_matrix_multiplication_.py --model DeepSeek-V4-Pro --endpoint api \
    --gpus 0,1 --cuda-arch auto --max-iterations 5 --enable-termination --termination-criteria hist_avg \
    --run-id first
specgen report runs/first          # an HTML timeline of the run: runs/first/report.html
```

`specgen run --help` lists the options; `pytest -m "not gpu"` runs the tests that need no GPU.

## License

SpecGen is released under the MIT License ([LICENSE](LICENSE)). The task files in `KernelBench/` come from
[KernelBench](https://github.com/ScalingIntelligence/KernelBench) and keep their MIT License
([KernelBench/LICENSE](KernelBench/LICENSE)).

## Citation

```bibtex
@article{guo2026specgen,
  title   = {SpecGen: Accelerating Agentic Kernel Optimization with Speculative Generation},
  author  = {Guo, Jihu and Lu, Sitian and Ma, Tenghui and Gao, Wei and Ye, Zhisheng and Zhang, Xingcheng and Lin, Dahua},
  journal = {arXiv preprint arXiv:2606.17518},
  year    = {2026}
}
```
