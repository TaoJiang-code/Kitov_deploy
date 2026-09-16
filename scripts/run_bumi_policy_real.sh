#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_DIR="${KITOV_BUMI_RGMT_MODEL_DIR:-models/bumi/rgmt}"
HZ="${KITOV_HZ:-50}"
SERVICE_SCRIPT="${KITOV_XROBOT_SERVICE_SCRIPT:-/opt/apps/roboticsservice/runService.sh}"
SERVICE_PATTERN="${KITOV_XROBOT_SERVICE_PATTERN:-[R]oboticsServiceProcess}"
STOP_SERVICE_ON_EXIT="${KITOV_STOP_ROBOTICS_SERVICE_ON_EXIT:-1}"

PY_PID=""
SERVICE_SHOULD_STOP=0

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
uv run python scripts/xrobot_bumi_rgmt_policy_real.py \
  --model-dir "${MODEL_DIR}" \
  --hz "${HZ}" \
  --offset-to-ground \
  --quiet-gmr \
  --send \
  "$@"
PY_STATUS=$?
set -e
exit "${PY_STATUS}"
