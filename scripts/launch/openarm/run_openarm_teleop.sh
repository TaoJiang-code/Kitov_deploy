#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

HZ="${KITOV_HZ:-50}"
CAN_INTERFACES_TEXT="${KITOV_OPENARM_CAN_INTERFACES:-can0 can1}"
CAN_BITRATE="${KITOV_OPENARM_CAN_BITRATE:-1000000}"
CAN_DBITRATE="${KITOV_OPENARM_CAN_DBITRATE:-5000000}"
SERVICE_SCRIPT="${KITOV_XROBOT_SERVICE_SCRIPT:-/opt/apps/roboticsservice/runService.sh}"
SERVICE_PATTERN="${KITOV_XROBOT_SERVICE_PATTERN:-[R]oboticsServiceProcess}"
STOP_SERVICE_ON_EXIT="${KITOV_STOP_ROBOTICS_SERVICE_ON_EXIT:-1}"

PY_PID=""
SERVICE_SHOULD_STOP=0

cleanup() {
  local status=$?
  trap - INT TERM EXIT

  if [[ -n "${PY_PID}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
    echo "[run_openarm_teleop] stopping xrobot_openarm_control.py"
    kill -INT "${PY_PID}" 2>/dev/null || true
    wait "${PY_PID}" 2>/dev/null || true
  fi

  if [[ "${STOP_SERVICE_ON_EXIT}" != "0" && "${SERVICE_SHOULD_STOP}" == "1" ]]; then
    if pgrep -f "${SERVICE_PATTERN}" >/dev/null 2>&1; then
      echo "[run_openarm_teleop] stopping XRoboToolkit PC Service"
      pkill -f "${SERVICE_PATTERN}" 2>/dev/null || true
    fi
  fi

  exit "${status}"
}

trap cleanup INT TERM EXIT

find_openarm_can_cli() {
  if command -v openarm-can-cli >/dev/null 2>&1; then
    command -v openarm-can-cli
    return
  fi

  local local_cli="${REPO_ROOT}/third_party/openarm_can/build/openarm-can-cli"
  if [[ -x "${local_cli}" ]]; then
    printf '%s\n' "${local_cli}"
    return
  fi

  echo "[run_openarm_teleop] cannot find openarm-can-cli" >&2
  echo "Install openarm_can or build third_party/openarm_can first." >&2
  exit 1
}

can_is_configured() {
  local iface="$1"
  local details
  details="$(ip -details link show "${iface}" 2>/dev/null || true)"
  [[ -n "${details}" ]] || return 1
  grep -q "state UP" <<<"${details}" || return 1
  grep -q " bitrate ${CAN_BITRATE} " <<<"${details}" || return 1
  grep -q " dbitrate ${CAN_DBITRATE} " <<<"${details}" || return 1
  grep -q " fd on" <<<"${details}" || return 1
}

configure_can_if_needed() {
  local cli="$1"
  shift

  local iface
  for iface in "$@"; do
    if can_is_configured "${iface}"; then
      echo "[run_openarm_teleop] ${iface} already configured"
    else
      echo "[run_openarm_teleop] configuring ${iface}"
      "${cli}" -i "${iface}" can_configure
    fi
  done
}

ensure_xrobot_service() {
  if pgrep -f "${SERVICE_PATTERN}" >/dev/null 2>&1; then
    echo "[run_openarm_teleop] XRoboToolkit PC Service already running"
    return
  fi

  if [[ ! -x "${SERVICE_SCRIPT}" ]]; then
    echo "[run_openarm_teleop] service script not found or not executable: ${SERVICE_SCRIPT}" >&2
    exit 1
  fi

  echo "[run_openarm_teleop] starting XRoboToolkit PC Service"
  "${SERVICE_SCRIPT}"
  sleep 1

  if ! pgrep -f "${SERVICE_PATTERN}" >/dev/null 2>&1; then
    echo "[run_openarm_teleop] XRoboToolkit PC Service did not start" >&2
    exit 1
  fi
}

read -r -a CAN_INTERFACES <<<"${CAN_INTERFACES_TEXT}"
if [[ "${#CAN_INTERFACES[@]}" -eq 0 ]]; then
  echo "[run_openarm_teleop] no CAN interfaces configured" >&2
  exit 1
fi

OPENARM_CAN_CLI="$(find_openarm_can_cli)"
configure_can_if_needed "${OPENARM_CAN_CLI}" "${CAN_INTERFACES[@]}"
ensure_xrobot_service
SERVICE_SHOULD_STOP=1

echo "[run_openarm_teleop] starting OpenArm XRobot teleop"
uv run python "${REPO_ROOT}/scripts/launch/openarm/xrobot_openarm_control.py" \
  --hz "${HZ}" \
  --quiet-gmr \
  --send \
  --enable-motors \
  --enable-ee-control \
  --viewer \
  --show-human \
  "$@" &

PY_PID=$!
set +e
wait "${PY_PID}"
PY_STATUS=$?
set -e
PY_PID=""
exit "${PY_STATUS}"
