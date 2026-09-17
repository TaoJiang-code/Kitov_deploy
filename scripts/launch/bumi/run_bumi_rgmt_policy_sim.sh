#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_DIR="${KITOV_BUMI_RGMT_MODEL_DIR:-models/bumi/rgmt}"
HZ="${KITOV_HZ:-50}"
BODY_SOURCE="${KITOV_BUMI_BODY_SOURCE:-local}"
BODY_BIND_ADDRESS="${KITOV_BUMI_BODY_BIND_ADDRESS:-0.0.0.0}"
BODY_PORT="${KITOV_BUMI_BODY_PORT:-47001}"
BODY_SOURCE_HOST="${KITOV_BUMI_BODY_SOURCE_HOST:-}"
RELAY_CONFIG="${KITOV_XROBOT_RELAY_CONFIG:-${REPO_ROOT}/workspace/xrobot_relay.env}"
ENV_BODY_PORT="${KITOV_BUMI_BODY_PORT-}"
ENV_SOURCE_HOST="${KITOV_BUMI_BODY_SOURCE_HOST-}"
if [[ -f "${RELAY_CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${RELAY_CONFIG}"
  [[ -n "${ENV_BODY_PORT}" ]] && KITOV_BUMI_BODY_PORT="${ENV_BODY_PORT}"
  [[ -n "${ENV_SOURCE_HOST}" ]] && KITOV_BUMI_BODY_SOURCE_HOST="${ENV_SOURCE_HOST}"
  BODY_PORT="${KITOV_BUMI_BODY_PORT:-${BODY_PORT}}"
  BODY_SOURCE_HOST="${KITOV_BUMI_BODY_SOURCE_HOST:-${BODY_SOURCE_HOST}}"
fi
BODY_SOURCE_ARGS=()
if [[ -n "${BODY_SOURCE_HOST}" ]]; then
  BODY_SOURCE_ARGS=(--body-source-host "${BODY_SOURCE_HOST}")
fi

exec uv run python "${REPO_ROOT}/scripts/launch/bumi/xrobot_rgmt_policy_infer.py" \
  --model-dir "${MODEL_DIR}" \
  --hz "${HZ}" \
  --body-source "${BODY_SOURCE}" \
  --body-bind-address "${BODY_BIND_ADDRESS}" \
  --body-port "${BODY_PORT}" \
  "${BODY_SOURCE_ARGS[@]}" \
  --offset-to-ground \
  --quiet-gmr \
  --viewer \
  --gmr-viewer \
  --show-human \
  "$@"
