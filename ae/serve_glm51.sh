#!/usr/bin/env bash
# Serve GLM-5.1-FP8 with vLLM for `ae/run.sh`, on 8 GPUs other than those that validate and profile kernels:
#
#   GLM_MODEL=/path/to/GLM-5.1-FP8 CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 bash ae/serve_glm51.sh
#   export GLM_URL=http://<this host>:${PORT:-8000}/v1      # then, where ae/run.sh runs
#
# GLM_MODEL: a local directory or a Hugging Face id.  TP, PORT, MAX_MODEL_LEN and VLLM_ARGS (extra flags) can
# be set the same way.
set -euo pipefail
: "${GLM_MODEL:?set GLM_MODEL to the GLM-5.1-FP8 weights (a local directory or a Hugging Face id)}"
# shellcheck disable=SC2086  # VLLM_ARGS holds several flags
exec vllm serve "$GLM_MODEL" --served-model-name GLM-5.1-FP8 --tensor-parallel-size "${TP:-8}" \
  --host 0.0.0.0 --port "${PORT:-8000}" --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --enable-prefix-caching --trust-remote-code ${VLLM_ARGS:-}
