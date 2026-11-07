"""Process profiled by ncu: build the generated model and run its forward repeatedly, each forward in
an NVTX range ``forward<i>`` (1-based) so ncu can profile exactly the kernels of one of them.

Usage: ``python -m specgen.evaluation.ncu_harness KERNEL TASK [ITERATIONS] [SEED]``
"""

from __future__ import annotations

import sys
import traceback


def main(argv: list[str]) -> int:
    import torch

    from specgen.evaluation.modules import EvalFailure, call_model, load_pair, make_inputs

    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    kernel, task = argv[0], argv[1]
    iterations = int(argv[2]) if len(argv) > 2 else 5
    seed = int(argv[3]) if len(argv) > 3 else 0
    if not torch.cuda.is_available():
        print("ncu_harness: CUDA is required", file=sys.stderr)
        return 1
    try:
        pair = load_pair(task, kernel, seed)
        torch.manual_seed(seed)
        inputs = make_inputs(pair.reference, pair.device)
        with torch.no_grad():
            for i in range(1, iterations + 1):
                torch.cuda.nvtx.range_push(f"forward{i}")
                call_model(pair.model_gen, inputs)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
    except EvalFailure as failure:
        print(f"ncu_harness: {failure.result['status']}: {failure.result['error']}", file=sys.stderr)
        return 1
    except Exception:
        print(f"ncu_harness: forward failed:\n{traceback.format_exc()}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
