"""Build: import the kernel module on a CPU worker so its extensions compile before validation.

The child runs with no visible GPU and the target ``TORCH_CUDA_ARCH_LIST``; the build
lands in the kernel's ``TORCH_EXTENSIONS_DIR``, where the validation child finds it
up to date.  Only definite build failures fail the kernel: a module that needs a GPU
at import time is left to the validation step.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

from specgen.evaluation.jobs import EvalJob
from specgen.evaluation.modules import KERNEL_MODULE, import_module_from_path

BUILD_FAILURE_MARKERS = (
    "Error building extension",
    "ninja: build stopped",
    "subprocess.CalledProcessError",
    "nvcc fatal",
    ".so: undefined symbol:",  # the extension built but a declared function has no matching definition
)


def run(job: EvalJob) -> dict[str, Any]:
    from specgen.evaluation import fastbuild

    fastbuild.install_if_enabled()
    t0 = time.perf_counter()
    try:
        import_module_from_path(Path(job.kernel_path), KERNEL_MODULE)
    except SyntaxError:
        return {"ok": False, "status": "compile_error", "error": traceback.format_exc(),
                "compile_s": round(time.perf_counter() - t0, 3)}
    except Exception:  # noqa: BLE001
        error = traceback.format_exc()
        compile_s = round(time.perf_counter() - t0, 3)
        if any(marker in error for marker in BUILD_FAILURE_MARKERS):
            return {"ok": False, "status": "compile_error", "error": error, "compile_s": compile_s}
        return {"ok": True, "status": "build_skipped", "reason": error[-2000:], "compile_s": compile_s}
    return {"ok": True, "status": "built", "compile_s": round(time.perf_counter() - t0, 3)}
