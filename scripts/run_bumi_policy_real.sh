#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_DIR="${KITOV_BUMI_RGMT_MODEL_DIR:-models/bumi/rgmt}"
HZ="${KITOV_HZ:-50}"

exec uv run python scripts/xrobot_bumi_rgmt_policy_real.py \
  --model-dir "${MODEL_DIR}" \
  --hz "${HZ}" \
  --offset-to-ground \
  --quiet-gmr \
  --send \
  "$@"
