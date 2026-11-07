"""Evaluation child process: run one job and write its result as JSON.

Usage: ``python -m specgen.evaluation.child --request REQ.json --result RES.json``

The result file is the only channel back to the parent (stdout/stderr go to a
log), and it is always written unless the process is killed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

from specgen.evaluation.jobs import EvalJob

SUPPORTED_TASK_KINDS = ("kernelbench",)


def execute(job: EvalJob) -> dict[str, Any]:
    if job.task_kind not in SUPPORTED_TASK_KINDS:
        return {"ok": False, "status": "unsupported", "error": f"no evaluator for task kind {job.task_kind!r}"}
    if job.kind == "build":
        from specgen.evaluation import build as impl
    elif job.kind == "validate":
        from specgen.evaluation import validate as impl
    elif job.kind == "benchmark":
        from specgen.evaluation import benchmark as impl
    else:
        from specgen.evaluation import ncu as impl
    return impl.run(job)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run one SpecGen evaluation job.")
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        result = execute(EvalJob.from_dict(json.loads(args.request.read_text(encoding="utf-8"))))
    except BaseException:  # noqa: BLE001 - the parent needs a result for every failure
        result = {"ok": False, "status": "runtime_error", "error": traceback.format_exc()}
    tmp = args.result.with_suffix(args.result.suffix + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, args.result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
