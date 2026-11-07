#!/usr/bin/env bash
# SpecGen artifact evaluation in one command (details: ae/README.md).
#
#   export DEEPSEEK_API_KEY=...                 # DeepSeek-V4-Pro through the DeepSeek API, or
#   export GLM_API_KEY=...                      # GLM-5.1 through an OpenAI-compatible API (Z.ai by default), or
#   export GLM_URL=http://<host>:8000/v1        # GLM-5.1-FP8 served by ae/serve_glm51.sh
#   bash ae/run.sh getting-started              # about 30 minutes: check, GPU tests, one task x 1 iteration
#   bash ae/run.sh e1|full|check|report [--tasks T1,T4] [--iterations N] [--parallel-methods]
#
# Run it again after an interruption: finished runs are skipped and stopped ones resume.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
if ! "$PYTHON" -c "import specgen, tiktoken, pytest, yaml, ninja" >/dev/null 2>&1; then
  echo "installing SpecGen and its test dependencies into $("$PYTHON" -c 'import sys; print(sys.prefix)')"
  "$PYTHON" -m pip install --quiet -e ".[tokens,dev]"
fi
exec "$PYTHON" ae/ae.py "$@"
