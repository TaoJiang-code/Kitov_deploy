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
SERVICE_SCRIPT="${KITOV_XROBOT_SERVICE_SCRIPT:-/opt/apps/roboticsservice/runService.sh}"
SERVICE_PATTERN="${KITOV_XROBOT_SERVICE_PATTERN:-[R]oboticsServiceProcess}"
STOP_SERVICE_ON_EXIT="${KITOV_STOP_ROBOTICS_SERVICE_ON_EXIT:-1}"

PY_PID=""
SERVICE_SHOULD_STOP=0

# Allow both the documented environment variable and a direct CLI override.
EXPECT_BODY_SOURCE=0
for arg in "$@"; do
  if [[ "${EXPECT_BODY_SOURCE}" == "1" ]]; then
    BODY_SOURCE="${arg}"
    EXPECT_BODY_SOURCE=0
  elif [[ "${arg}" == "--body-source" ]]; then
    EXPECT_BODY_SOURCE=1
  elif [[ "${arg}" == --body-source=* ]]; then
    BODY_SOURCE="${arg#--body-source=}"
  fi
done

cleanup() {
  local status=$?
  trap - INT TERM EXIT

  if [[ -n "${PY_PID}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
    echo "[run_bumi_policy_real] stopping xrobot_bumi_rgmt_policy_real.py"
    kill -INT "${PY_PID}" 2>/dev/null || true
    wait "${PY_PID}" 2>/dev/null || true
  fi

  if [[ "${STOP_SERVICE_ON_EXIT}" != "0" && "${SERVICE_SHOULD_STOP}" == "1" ]]; then
    if pgrep -f "${SERVICE_PATTERN}" >/dev/null 2>&1; then
      echo "[run_bumi_policy_real] stopping XRoboToolkit PC Service"
      pkill -f "${SERVICE_PATTERN}" 2>/dev/null || true
    fi
  fi

  exit "${status}"
}

trap cleanup INT TERM EXIT

ensure_xrobot_service() {
  if [[ "${BODY_SOURCE}" == "udp" ]]; then
    echo "[run_bumi_policy_real] remote body mode: skipping local XRoboToolkit PC Service"
    return
  fi
  if pgrep -f "${SERVICE_PATTERN}" >/dev/null 2>&1; then
    echo "[run_bumi_policy_real] XRoboToolkit PC Service already running"
    return
  fi

  if [[ ! -x "${SERVICE_SCRIPT}" ]]; then
    echo "[run_bumi_policy_real] service script not found or not executable: ${SERVICE_SCRIPT}" >&2
    exit 1
  fi

  echo "[run_bumi_policy_real] starting XRoboToolkit PC Service"
  "${SERVICE_SCRIPT}"
  sleep 1

  if ! pgrep -f "${SERVICE_PATTERN}" >/dev/null 2>&1; then
    echo "[run_bumi_policy_real] XRoboToolkit PC Service did not start" >&2
    exit 1
  fi
  SERVICE_SHOULD_STOP=1
}

ensure_xrobot_service

set +e
uv run python "${REPO_ROOT}/scripts/launch/bumi/xrobot_bumi_rgmt_policy_real.py" \
  --model-dir "${MODEL_DIR}" \
  --hz "${HZ}" \
  --body-source "${BODY_SOURCE}" \
  --body-bind-address "${BODY_BIND_ADDRESS}" \
  --body-port "${BODY_PORT}" \
  "${BODY_SOURCE_ARGS[@]}" \
  --offset-to-ground \
  --quiet-gmr \
  --send \
  "$@"
PY_STATUS=$?
set -e
exit "${PY_STATUS}"
