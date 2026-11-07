from specgen.control.compliance import forbidden_usage, library_kernels

HEADER = '''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
__global__ void gemm(const float* a, const float* b, float* c, int n) {
    // matmul tile: c = a @ b, then at::matmul-free epilogue
    asm volatile("@p bra DONE;");
}
"""
_ext = load_inline(name="g", cpp_sources="torch::Tensor matmul(torch::Tensor a, torch::Tensor b);",
                   cuda_sources=cuda_src, functions=["matmul"])
'''


def module(forward_body: str, extra: str = "") -> str:
    return HEADER + extra + f'''
class Model(nn.Module):
    def forward(self, A, B):
{forward_body}
'''


def test_hand_written_modules_pass():
    assert forbidden_usage(module("        return _ext.matmul(A, B)")) is None  # the extension's own matmul
    holder = module("        return self.ext.mm(A, B)", extra="").replace(
        "class Model(nn.Module):\n", "class Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
        "        self.ext = load_inline(name='h', cpp_sources='', functions=['mm'])\n")
    assert forbidden_usage(holder) is None
    assert forbidden_usage(module("        C = torch.empty_like(A)\n        return C.relu_()")) is None
    assert forbidden_usage("def broken(:\n") is None  # the build reports syntax errors


def test_library_calls_and_precision_switches_are_found():
    tf32 = module('        prev = torch.get_float32_matmul_precision()\n'
                  '        torch.set_float32_matmul_precision("high")\n'
                  '        try:\n            return torch.matmul(A, B)\n'
                  '        finally:\n            torch.set_float32_matmul_precision(prev)')
    assert forbidden_usage(tf32) == ("uses a library GEMM or convolution instead of a hand-written kernel: "
                                     "torch.matmul(); also changes torch's matmul precision (TF32) settings: "
                                     "torch.set_float32_matmul_precision")
    fallback = module("        if A.shape[0] % 256:\n            return A @ B\n        return _ext.matmul(A, B)")
    assert forbidden_usage(fallback).endswith(": the @ operator")
    assert "F.linear()" in forbidden_usage(module("        return F.linear(A, B)"))
    assert "A.mm()" in forbidden_usage(module("        return A.mm(B)"))
    assert "torch.backends.cuda.matmul.allow_tf32" in forbidden_usage(
        module("        torch.backends.cuda.matmul.allow_tf32 = True\n        return _ext.matmul(A, B)"))
    imported = module("        return mm(A, B)").replace("import torch\n", "import torch\nfrom torch import mm\n", 1)
    assert "mm()" in forbidden_usage(imported)
    cpp = module("        return _ext.matmul(A, B)").replace(
        'cpp_sources="torch::Tensor matmul', 'cpp_sources="torch::Tensor f(torch::Tensor a) { return at::matmul(a, a); }'
        ' torch::Tensor matmul')
    assert "at::matmul in the C++/CUDA source" in forbidden_usage(cpp)


def test_library_kernels_by_name():
    names = ["nvjet_sm103_sss_tf32_128x256_32x6_2x2f_2", "cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x128x8_1",
             "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_nn_align4>(Params)",
             "ampere_sgemm_128x64_nn", "sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize128x128x32",
             "void gemv2T_kernel_val<int, int, float>(...)", "cudnn::detail::implicit_convolve_sgemm",
             "gemm_f16_kernel<4>", "convert_f32_to_f16_kernel",
             "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>>(int)"]
    assert library_kernels(names) == sorted(names[:7])
