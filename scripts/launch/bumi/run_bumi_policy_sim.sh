#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_DIR="${KITOV_BUMI_MODEL_DIR:-models/bumi/kitov_fb_bumi_action_scale_0.5}"
HZ="${KITOV_HZ:-50}"

exec python "${REPO_ROOT}/scripts/launch/g1/xrobot_policy_infer.py" \
  --robot bumi \
  --model-dir "${MODEL_DIR}" \
  --hz "${HZ}" \
  --offset-to-ground \
  --quiet-gmr \
  --viewer \
  --gmr-viewer \
  --show-human \
  "$@"
