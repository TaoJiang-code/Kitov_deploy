#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_DIR="${KITOV_G1_MODEL_DIR:-models/g1/kitov_fb_g1}"
HZ="${KITOV_HZ:-50}"
ELASTIC_LENGTH="${KITOV_G1_ELASTIC_LENGTH:-1.5}"

exec python scripts/xrobot_policy_infer.py \
  --robot g1 \
  --model-dir "${MODEL_DIR}" \
  --hz "${HZ}" \
  --offset-to-ground \
  --quiet-gmr \
  --viewer \
  --gmr-viewer \
  --show-human \
  --elastic-band \
  --elastic-length "${ELASTIC_LENGTH}" \
  "$@"
