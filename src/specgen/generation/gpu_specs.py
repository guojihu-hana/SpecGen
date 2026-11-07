"""GPU parameters injected into prompts."""

from __future__ import annotations

from typing import Optional

_BLACKWELL_SM_LIMITS = {
    "Register File Size": "64K 32-bit registers per SM",
    "Maximum number of registers per thread": "255",
    "Maximum number of thread blocks per SM": "32",
    "Shared memory capacity per SM": "228 KB",
    "Maximum shared memory per thread block": "227 KB",
}

# Blackwell figures are per GPU from NVIDIA's HGX B200 / B300 specifications.
GPU_SPECS: dict[str, dict[str, str]] = {
    "B300": {
        "GPU Architecture": "Blackwell Ultra",
        "GPU Memory": "288 GB HBM3e",
        "Memory Bandwidth": "8 TB/s",
        "FP64 TFLOPS": "1.2",
        "FP32 TFLOPS": "75",
        "TF32 Tensor Core TFLOPS": "2200 with sparsity (1100 dense)",
        "BFLOAT16 Tensor Core TFLOPS": "4500 with sparsity (2250 dense)",
        "FP16 Tensor Core TFLOPS": "4500 with sparsity (2250 dense)",
        "FP8 Tensor Core TFLOPS": "9000 with sparsity (4500 dense)",
        "FP4 Tensor Core TFLOPS": "18000 with sparsity (13500 dense)",
        **_BLACKWELL_SM_LIMITS,
    },
    "B200": {
        "GPU Architecture": "Blackwell",
        "GPU Memory": "180 GB HBM3e",
        "Memory Bandwidth": "8 TB/s",
        "FP64 TFLOPS": "37",
        "FP32 TFLOPS": "75",
        "TF32 Tensor Core TFLOPS": "2250 with sparsity (1125 dense)",
        "BFLOAT16 Tensor Core TFLOPS": "4500 with sparsity (2250 dense)",
        "FP16 Tensor Core TFLOPS": "4500 with sparsity (2250 dense)",
        "FP8 Tensor Core TFLOPS": "9000 with sparsity (4500 dense)",
        "FP4 Tensor Core TFLOPS": "18000 with sparsity (9000 dense)",
        **_BLACKWELL_SM_LIMITS,
    },
    "L40S": {
        "GPU Architecture": "Ada",
        "GPU Memory": "48GB GDDR6 with ECC",
        "Memory Bandwidth": "864 GB/s",
        "FP32 TFLOPS": "91.6",
        "TF32 Tensor Core TFLOPS": "183.2 (366 with sparsity)",
        "FP16 Tensor Core TFLOPS": "362.05 (733 with sparsity)",
        "FP8 Tensor Core TFLOPS": "733 (1466 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "24",
        "Shared memory capacity per SM": "100 KB",
        "Maximum shared memory per thread block": "99 KB",
    },
    "H100": {
        "GPU Architecture": "Hopper",
        "GPU Memory": "80GB",
        "Memory Bandwidth": "3.35 TB/s",
        "FP64 TFLOPS": "34",
        "FP64 Tensor Core TFLOPS": "67",
        "FP32 TFLOPS": "67",
        "TF32 Tensor Core TFLOPS": "989 with sparsity",
        "BFLOAT16 Tensor Core TFLOPS": "1979 with sparsity",
        "FP16 Tensor Core TFLOPS": "1979 with sparsity",
        "FP8 Tensor Core TFLOPS": "3958 with sparsity",
        "INT8 Tensor Core TOPS": "3958 with sparsity",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "228 KB",
        "Maximum shared memory per thread block": "227 KB",
    },
    "H800": {
        "GPU Architecture": "Hopper",
        "GPU Memory": "80GB HBM3",
        "Memory Bandwidth": "3.35 TB/s",
        "NVLink Bandwidth": "400 GB/s",
        "FP64 TFLOPS": "1",
        "FP64 Tensor Core TFLOPS": "1",
        "FP32 TFLOPS": "67",
        "TF32 Tensor Core TFLOPS": "989 with sparsity",
        "BFLOAT16 Tensor Core TFLOPS": "1979 with sparsity",
        "FP16 Tensor Core TFLOPS": "1979 with sparsity",
        "FP8 Tensor Core TFLOPS": "3958 with sparsity",
        "INT8 Tensor Core TOPS": "3958 with sparsity",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "228 KB",
        "Maximum shared memory per thread block": "227 KB",
    },
    "H200": {
        "GPU Architecture": "Hopper",
        "GPU Memory": "141 GB HBM3e",
        "Memory Bandwidth": "4.8 TB/s",
        "FP64 TFLOPS": "34",
        "FP64 Tensor Core TFLOPS": "67",
        "FP32 TFLOPS": "67",
        "TF32 Tensor Core TFLOPS": "989",
        "BFLOAT16 Tensor Core TFLOPS": "1979",
        "FP16 Tensor Core TFLOPS": "1979",
        "FP8 Tensor Core TFLOPS": "3958",
        "INT8 Tensor Core TOPS": "3958",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "228 KB",
        "Maximum shared memory per thread block": "227 KB",
    },
    "A100": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "40GB",
        "Memory Bandwidth": "1555 GB/s",
        "FP64 TFLOPS": "9.7",
        "FP64 Tensor Core TFLOPS": "19.5",
        "FP32 TFLOPS": "19.5",
        "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
        "BFLOAT16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "A100-80GB": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "80GB",
        "Memory Bandwidth": "1935 GB/s",
        "FP64 TFLOPS": "9.7",
        "FP64 Tensor Core TFLOPS": "19.5",
        "FP32 TFLOPS": "19.5",
        "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
        "BFLOAT16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "A800": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "80GB",
        "Memory Bandwidth": "2039 GB/s",
        "NVLink Bandwidth": "400 GB/s",
        "FP64 TFLOPS": "9.7",
        "FP64 Tensor Core TFLOPS": "19.5",
        "FP32 TFLOPS": "19.5",
        "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
        "BFLOAT16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "L4": {
        "GPU Architecture": "Ada",
        "GPU Memory": "24GB",
        "Memory Bandwidth": "300 GB/s",
        "FP32 TFLOPS": "30.3",
        "TF32 Tensor Core TFLOPS": "120 with sparsity",
        "BFLOAT16 Tensor Core TFLOPS": "242 with sparsity",
        "FP8 Tensor Core TFLOPS": "485 with sparsity",
        "INT8 Tensor Core TOPS": "485 with sparsity",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "24",
        "Shared memory capacity per SM": "100 KB",
        "Maximum shared memory per thread block": "99 KB",
    },
    "Quadro RTX 6000": {
        "GPU Architecture": "Turing",
        "GPU Memory": "24GB GDDR6 with ECC",
        "Memory Bandwidth": "624 GB/s",
        "FP32 TFLOPS": "16.3",
        "FP16 Tensor Core TFLOPS": "261 (522 with sparsity)",
        "INT8 Tensor Core TOPS": "261 (522 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "64 KB",
        "Maximum shared memory per thread block": "64 KB",
    },
    "T4": {
        "GPU Architecture": "Turing",
        "GPU Memory": "16 GB GDDR6",
        "Memory Bandwidth": "300 GB/s",
        "Single-Precision TFLOPS": "8.1",
        "Mixed-Precision (FP16/FP32) TFLOPS": "65",
        "INT8 TOPS": "130",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "16",
        "Shared memory capacity per SM": "64 KB",
    },
    "A10G": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "24GB GDDR6",
        "Memory Bandwidth": "600 GB/s",
        "FP32 TFLOPS": "31.2",
        "TF32 Tensor Core TFLOPS": "62.5 (125 with sparsity)",
        "BFLOAT16 Tensor Core TFLOPS": "125 (250 with sparsity)",
        "FP16 Tensor Core TFLOPS": "125 (250 with sparsity)",
        "INT8 Tensor Core TOPS": "250 (500 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
}

# Checked in order: "L40S" before "L4", "A100 80GB" before "A100".
_NAME_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("B300",), "B300"),
    (("B200",), "B200"),
    (("H200",), "H200"),
    (("H100",), "H100"),
    (("H800",), "H800"),
    (("A100", "80GB"), "A100-80GB"),
    (("A100",), "A100"),
    (("A800",), "A800"),
    (("L40S",), "L40S"),
    (("L4",), "L4"),
    (("A10G",), "A10G"),
    (("RTX 6000",), "Quadro RTX 6000"),
    (("T4",), "T4"),
)


def normalize_gpu_name(name: str) -> Optional[str]:
    """Map a device name (e.g. ``NVIDIA H100 80GB HBM3``) to a key of :data:`GPU_SPECS`."""
    upper = name.upper()
    for needles, key in _NAME_PATTERNS:
        if all(n.upper() in upper for n in needles):
            return key
    return None


COMPUTE_CAPABILITY: dict[str, str] = {
    "B300": "10.3", "B200": "10.0", "L40S": "8.9", "H100": "9.0", "H800": "9.0", "H200": "9.0", "A100": "8.0",
    "A100-80GB": "8.0",
    "A800": "8.0", "L4": "8.9", "Quadro RTX 6000": "7.5", "T4": "7.5", "A10G": "8.6",
}


def format_gpu_specs(gpu_type: str) -> str:
    if gpu_type not in GPU_SPECS:
        raise KeyError(f"unknown gpu_type {gpu_type!r}; known: {', '.join(sorted(GPU_SPECS))}")
    lines = [f"Target GPU: {gpu_type}", "", "Hardware parameters:"]
    for key, value in GPU_SPECS[gpu_type].items():
        lines.append(f"- {key}: {value}")
        if key == "GPU Architecture" and gpu_type in COMPUTE_CAPABILITY:
            cc = COMPUTE_CAPABILITY[gpu_type]
            lines.append(f"- Compute capability: {cc} (sm_{cc.replace('.', '')})")
    return "\n".join(lines)
