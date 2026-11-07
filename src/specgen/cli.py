"""``specgen`` command line."""

from __future__ import annotations

import logging
import sys
from typing import Optional

USAGE = """usage: specgen <command> [options]

commands:
  run        iteratively generate and optimize kernels (specgen run --help)
  worker     serve this machine's GPUs to a remote run (specgen worker --help)
  summarize  print a run's results (specgen summarize RUN_DIR)
  report     write an HTML timeline of a run (specgen report RUN_DIR -o report.html)
"""


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if command == "run":
        from specgen.config import parse_args
        from specgen.control.orchestrator import run

        return run(parse_args(rest))
    if command == "worker":
        from specgen.remote.cli import main as worker_main

        return worker_main(rest)
    if command == "summarize":
        from specgen.analysis.summary import main as summary_main

        return summary_main(rest)
    if command == "report":
        from specgen.analysis.report import main as report_main

        return report_main(rest)
    print(f"unknown command {command!r}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
