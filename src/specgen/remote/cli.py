"""``specgen worker``: serve this machine's GPUs to a remote ``specgen run``."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from specgen.resources.gpus import ARCH_EXAMPLES, detect_gpu_type, discover_gpus

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="specgen worker", description="Run evaluation jobs for a remote specgen run.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--root", type=Path, default=Path("worker_runs"), help="kernels, builds and caches go here")
    p.add_argument("--gpus", default="auto", help="'auto', 'none' (CPU only) or a list such as 0,1")
    p.add_argument("--cuda-arch", default=None, help=f"required with GPUs: {ARCH_EXAMPLES}; or 'auto'")
    p.add_argument("--job-timeout", type=float, default=600.0)
    p.add_argument("--build-jobs", type=int, default=None, help="MAX_JOBS per build (default: from the CPU count)")
    p.add_argument("--token-env", default="SPECGEN_WORKER_TOKEN",
                   help="environment variable holding the shared token clients must send")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    from specgen.control.orchestrator import RunSetupError, resolve_build_jobs, resolve_cuda_arch
    from specgen.remote.server import WorkerState, make_server

    p = build_parser()
    a = p.parse_args(argv)
    gpus_arg = a.gpus.strip().lower()
    if gpus_arg == "auto":
        gpus = discover_gpus()
    elif gpus_arg in ("", "none", "cpu"):
        gpus = []
    else:
        gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    if gpus and a.cuda_arch is None:
        p.error(f"--cuda-arch is required when serving GPUs ({ARCH_EXAMPLES}; or 'auto')")
    try:
        cuda_arch = resolve_cuda_arch(a.cuda_arch, gpus)
    except RunSetupError as e:
        p.error(str(e))
    token = os.environ.get(a.token_env) or None
    if token is None:
        log.warning("no token in $%s: any client that reaches %s:%d can run code here", a.token_env, a.host, a.port)
    state = WorkerState(a.root, gpus=gpus, cuda_arch=cuda_arch, gpu_type=detect_gpu_type(gpus),
                        timeout_s=a.job_timeout, build_jobs=resolve_build_jobs(a.build_jobs, max(1, 4 * len(gpus))),
                        token=token)
    server = make_server(state, a.host, a.port)
    log.info("specgen worker on %s:%d serving GPUs %s (root %s)", a.host, server.server_address[1], gpus or "none",
             state.root)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
