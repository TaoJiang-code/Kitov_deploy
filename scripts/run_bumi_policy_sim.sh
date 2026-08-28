#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_DIR="${KITOV_BUMI_MODEL_DIR:-models/bumi/kitov_fb_bumi_action_scale_0.5}"
HZ="${KITOV_HZ:-50}"

exec python scripts/xrobot_policy_infer.py \
  --robot bumi \
  --model-dir "${MODEL_DIR}" \
  --hz "${HZ}" \
  --offset-to-ground \
  --quiet-gmr \
  --viewer \
  --gmr-viewer \
  --show-human \
  "$@"
