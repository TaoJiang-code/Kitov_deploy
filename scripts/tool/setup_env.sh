#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON_VERSION="${KITOV_PYTHON_VERSION:-3.10}"
INSTALL_TARGET="${KITOV_INSTALL_TARGET:-}"
TORCH_MODE="${KITOV_TORCH_MODE:-auto}"
JETSON_TORCH_WHEEL="${KITOV_JETSON_TORCH_WHEEL:-}"
ONNXRUNTIME_MODE="${KITOV_ONNXRUNTIME_MODE:-auto}"
JETSON_ONNXRUNTIME_WHEEL="${KITOV_JETSON_ONNXRUNTIME_WHEEL:-}"
XROBOT_SETUP="${KITOV_XROBOT_SETUP:-}"
RECREATE_VENV="${KITOV_RECREATE_VENV:-0}"
FORCE_XROBOT_SERVICE_INSTALL="${KITOV_FORCE_XROBOT_SERVICE_INSTALL:-${KITOV_FORCE_XROBOT_SERVICE_DEB:-0}}"
JETSON_ONNXRUNTIME_JP6_CU126_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"
JETSON_ONNXRUNTIME_VERSION="${KITOV_JETSON_ONNXRUNTIME_VERSION:-1.23.0}"

log() {
  printf '[setup_env] %s\n' "$*"
}

die() {
  printf '[setup_env] ERROR: %s\n' "$*" >&2
  exit 1
}

version_ge() {
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

choose_install_target() {
  if [ -n "${INSTALL_TARGET}" ]; then
    return
  fi
  if [ $# -gt 0 ]; then
    die "positional install targets are no longer supported. Run scripts/tool/setup_env.sh and choose from the menu, or set KITOV_INSTALL_TARGET."
  fi

  cat <<'EOF'
[setup_env] Select optional runtime install:
  1) all          ONNX Runtime + PyTorch
  2) onnxruntime  ONNX Runtime only
  3) torch        PyTorch only
  4) skip         uv sync only
EOF

  local choice
  while true; do
    printf '[setup_env] choice [1-4]: '
    read -r choice
    case "${choice}" in
      1)
        INSTALL_TARGET="all"
        return
        ;;
      2)
        INSTALL_TARGET="onnxruntime"
        return
        ;;
      3)
        INSTALL_TARGET="torch"
        return
        ;;
      4)
        INSTALL_TARGET="skip"
        return
        ;;
      *)
        printf '[setup_env] invalid choice: %s\n' "${choice}" >&2
        ;;
    esac
  done
}

choose_xrobot_setup() {
  if [ -n "${XROBOT_SETUP}" ]; then
    return
  fi

  cat <<'EOF'
[setup_env] Select XRobot setup:
  1) skip        do not install XRobot SDK or PC Service
  2) sdk         build/install xrobotoolkit_sdk into this .venv
  3) service     install XRoboToolkit PC Service from .deb or source
  4) all         SDK + PC Service
EOF

  local choice
  while true; do
    printf '[setup_env] choice [1-4]: '
    read -r choice
    case "${choice}" in
      1)
        XROBOT_SETUP="skip"
        return
        ;;
      2)
        XROBOT_SETUP="sdk"
        return
        ;;
      3)
        XROBOT_SETUP="service"
        return
        ;;
      4)
        XROBOT_SETUP="all"
        return
        ;;
      *)
        printf '[setup_env] invalid choice: %s\n' "${choice}" >&2
        ;;
    esac
  done
}

install_torch_cpu() {
  log "installing PyTorch CPU wheel"
  uv pip install torch --index-url https://download.pytorch.org/whl/cpu
}

install_torch_cuda() {
  local cuda_tag="$1"
  log "installing PyTorch CUDA wheel: ${cuda_tag}"
  uv pip install torch --index-url "https://download.pytorch.org/whl/${cuda_tag}"
}

install_torch_jetson() {
  if [ -z "${JETSON_TORCH_WHEEL}" ]; then
    cat >&2 <<'EOF'
[setup_env] ERROR: Jetson detected, but KITOV_JETSON_TORCH_WHEEL is not set.

Jetson PyTorch wheels are tied to JetPack/L4T and cannot use the x86 CUDA
wheel index. Download or choose the matching NVIDIA Jetson PyTorch wheel, then:

  KITOV_JETSON_TORCH_WHEEL=/path/to/torch-xxx-linux_aarch64.whl scripts/tool/setup_env.sh

or:

  KITOV_JETSON_TORCH_WHEEL=https://.../torch-xxx-linux_aarch64.whl scripts/tool/setup_env.sh
EOF
    exit 2
  fi
  log "installing Jetson PyTorch wheel: ${JETSON_TORCH_WHEEL}"
  uv pip install "${JETSON_TORCH_WHEEL}"
}

install_onnxruntime_cpu() {
  log "installing ONNX Runtime CPU wheel"
  uv pip install --force-reinstall onnxruntime
}

install_onnxruntime_jetson_gpu() {
  local wheel="${JETSON_ONNXRUNTIME_WHEEL}"
  local use_index=0
  if [ -z "${wheel}" ]; then
    if [ -f /etc/nv_tegra_release ] && grep -q "R36" /etc/nv_tegra_release; then
      use_index=1
      log "using default JetPack 6 ONNX Runtime GPU index: ${JETSON_ONNXRUNTIME_JP6_CU126_INDEX}"
    else
      cat >&2 <<'EOF'
[setup_env] ERROR: Jetson ONNX Runtime GPU was requested, but no wheel is configured.

Set KITOV_JETSON_ONNXRUNTIME_WHEEL to a wheel matching this JetPack/L4T, for example:

  KITOV_JETSON_ONNXRUNTIME_WHEEL=/path/to/onnxruntime_gpu-...-linux_aarch64.whl \
    KITOV_INSTALL_TARGET=onnxruntime scripts/tool/setup_env.sh

For JetPack 6 / Python 3.10, the script has a built-in default wheel URL.
EOF
      exit 2
    fi
  fi

  log "replacing CPU onnxruntime with Jetson GPU package"
  uv pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime_gpu >/dev/null 2>&1 || true
  if [ "${use_index}" = "1" ]; then
    uv pip install \
      --index-url "${JETSON_ONNXRUNTIME_JP6_CU126_INDEX}" \
      "onnxruntime-gpu==${JETSON_ONNXRUNTIME_VERSION}" \
      "numpy<2"
  else
    uv pip install "${wheel}" "numpy<2"
  fi
}

install_onnxruntime_auto() {
  local arch
  arch="$(uname -m)"
  log "detected arch=${arch}"

  case "${arch}" in
    x86_64)
      install_onnxruntime_cpu
      ;;
    aarch64)
      if [ -f /etc/nv_tegra_release ]; then
        log "detected Jetson: $(head -n 1 /etc/nv_tegra_release)"
        install_onnxruntime_jetson_gpu
      else
        install_onnxruntime_cpu
      fi
      ;;
    *)
      install_onnxruntime_cpu
      ;;
  esac
}

detect_x86_cuda_tag() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found. Install/check NVIDIA driver first, or set KITOV_TORCH_MODE=cpu."
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi cannot talk to the NVIDIA driver. Fix the driver first, or set KITOV_TORCH_MODE=cpu."
  fi

  local cuda_version
  cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)"
  if [ -z "${cuda_version}" ]; then
    die "could not parse CUDA Version from nvidia-smi."
  fi

  if version_ge "${cuda_version}" "12.8"; then
    printf 'cu128\n'
  elif version_ge "${cuda_version}" "12.6"; then
    printf 'cu126\n'
  else
    die "driver reports CUDA Version ${cuda_version}; expected at least 12.6 for the default GPU wheels."
  fi
}

install_torch_auto() {
  local arch
  arch="$(uname -m)"
  log "detected arch=${arch}"

  case "${arch}" in
    x86_64)
      install_torch_cuda "$(detect_x86_cuda_tag)"
      ;;
    aarch64)
      if [ -f /etc/nv_tegra_release ]; then
        log "detected Jetson: $(head -n 1 /etc/nv_tegra_release)"
        install_torch_jetson
      else
        die "aarch64 host is not detected as Jetson. Set KITOV_TORCH_MODE=cpu if CPU PyTorch is intended."
      fi
      ;;
    *)
      die "unsupported arch=${arch}. Set KITOV_TORCH_MODE=cpu to install CPU PyTorch."
      ;;
  esac
}

verify_torch() {
  uv run --no-sync python - <<'PY'
import torch
print("[setup_env] torch:", torch.__version__)
print("[setup_env] cuda available:", torch.cuda.is_available())
print("[setup_env] torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("[setup_env] gpu:", torch.cuda.get_device_name(0))
PY
}

verify_onnxruntime() {
  uv run --no-sync python - <<'PY'
import onnxruntime as ort
import numpy as np
print("[setup_env] numpy:", np.__version__)
print("[setup_env] onnxruntime:", ort.__version__)
print("[setup_env] onnxruntime providers:", ort.get_available_providers())
PY
}

install_selected_onnxruntime() {
  case "${ONNXRUNTIME_MODE}" in
    auto)
      install_onnxruntime_auto
      ;;
    cpu)
      install_onnxruntime_cpu
      ;;
    jetson|jetson-gpu|gpu)
      install_onnxruntime_jetson_gpu
      ;;
    skip)
      log "skipping ONNX Runtime install because KITOV_ONNXRUNTIME_MODE=skip"
      return 0
      ;;
    *)
      die "unknown KITOV_ONNXRUNTIME_MODE=${ONNXRUNTIME_MODE}. Use auto, cpu, jetson-gpu, or skip."
      ;;
  esac
  verify_onnxruntime
}

install_selected_torch() {
  case "${TORCH_MODE}" in
    auto)
      install_torch_auto
      ;;
    skip)
      log "skipping PyTorch install because KITOV_TORCH_MODE=skip"
      return 0
      ;;
    cpu)
      install_torch_cpu
      ;;
    cu128|cu126)
      install_torch_cuda "${TORCH_MODE}"
      ;;
    jetson)
      install_torch_jetson
      ;;
    *)
      die "unknown KITOV_TORCH_MODE=${TORCH_MODE}. Use auto, cpu, cu128, cu126, jetson, or skip."
      ;;
  esac
  verify_torch
}

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"
  if [ -d "${target_dir}/.git" ]; then
    log "repository already exists: ${target_dir}"
    return
  fi
  log "cloning ${repo_url} -> ${target_dir}"
  git clone "${repo_url}" "${target_dir}"
}

ensure_xrobot_service_repo() {
  local service_repo="$1"
  local workspace
  workspace="$(dirname "${service_repo}")"

  mkdir -p "${workspace}"
  clone_if_missing "https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git" "${service_repo}"
}

install_xrobot_python_sdk() {
  command -v git >/dev/null 2>&1 || die "git not found; install git before XRobot SDK setup."

  local workspace="workspace/xrobot_toolkit"
  local service_repo="${workspace}/XRoboToolkit-PC-Service"
  local pybind_repo="${workspace}/XRoboToolkit-PC-Service-Pybind"

  mkdir -p "${workspace}"
  ensure_xrobot_service_repo "${service_repo}"
  clone_if_missing "https://github.com/Axellwppr/XRoboToolkit-PC-Service-Pybind" "${pybind_repo}"

  local sdk_dir="${service_repo}/RoboticsService/PXREARobotSDK"
  log "building XRoboToolkit PXREARobotSDK"
  (cd "${sdk_dir}" && bash build.sh)

  log "copying XRoboToolkit C++ SDK artifacts into Python binding project"
  mkdir -p "${pybind_repo}/lib" "${pybind_repo}/include/nlohmann"
  cp "${sdk_dir}/PXREARobotSDK.h" "${pybind_repo}/include/"
  cp -a "${sdk_dir}/nlohmann/." "${pybind_repo}/include/nlohmann/"
  cp "${sdk_dir}/build/libPXREARobotSDK.so" "${pybind_repo}/lib/"

  log "installing xrobotoolkit_sdk into current uv environment"
  uv pip install "${pybind_repo}"

  uv run --no-sync python - <<'PY'
import xrobotoolkit_sdk as xrt
print("[setup_env] xrobotoolkit_sdk import ok")
print("[setup_env] init:", hasattr(xrt, "init"))
print("[setup_env] callback:", hasattr(xrt, "register_frame_callback"))
print("[setup_env] polling:", hasattr(xrt, "is_body_data_available"))
PY
}

install_xrobot_pc_service_from_source() {
  command -v git >/dev/null 2>&1 || die "git not found; install git before XRoboToolkit PC Service source setup."

  local service_repo="workspace/xrobot_toolkit/XRoboToolkit-PC-Service"
  ensure_xrobot_service_repo "${service_repo}"

  local build_script="${service_repo}/RoboticsService/qt-gcc.sh"
  local bin_dir="${service_repo}/RoboticsService/bin"
  if [ ! -f "${build_script}" ]; then
    die "XRoboToolkit PC Service build script not found: ${build_script}"
  fi

  log "building XRoboToolkit PC Service from source"
  log "source path: ${service_repo}"
  if ! (cd "${service_repo}" && bash RoboticsService/qt-gcc.sh); then
    cat >&2 <<'EOF'
[setup_env] ERROR: XRoboToolkit PC Service source build failed.

This build depends on Qt. Install the Qt version expected by XRoboToolkit
PC Service on this machine, then rerun setup_env.sh.
EOF
    exit 2
  fi

  if [ ! -x "${bin_dir}/RoboticsServiceProcess" ]; then
    die "XRoboToolkit PC Service build finished, but RoboticsServiceProcess was not found in ${bin_dir}"
  fi

  log "installing source-built XRoboToolkit PC Service to /opt/apps/roboticsservice"
  sudo mkdir -p /opt/apps/roboticsservice
  sudo cp -a "${bin_dir}/." /opt/apps/roboticsservice/
  sudo tee /opt/apps/roboticsservice/runService.sh >/dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="${APP_DIR}:${APP_DIR}/lib:${APP_DIR}/SDK/x64:${APP_DIR}/SDK/linux/64:${APP_DIR}/SDK/linux_aarch64/64:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="${APP_DIR}/plugins:${QT_PLUGIN_PATH:-}"
export QML2_IMPORT_PATH="${APP_DIR}/qml:${QML2_IMPORT_PATH:-}"

cd "${APP_DIR}"
"${APP_DIR}/RoboticsServiceProcess" "$@" &
EOF
  sudo chmod +x /opt/apps/roboticsservice/runService.sh
  log "XRoboToolkit PC Service source install complete: /opt/apps/roboticsservice/runService.sh"
}

install_xrobot_pc_service() {
  if ! command -v dpkg >/dev/null 2>&1; then
    log "dpkg not found; falling back to XRoboToolkit PC Service source build"
    install_xrobot_pc_service_from_source
    return
  fi

  local arch
  arch="$(dpkg --print-architecture)"

  if [ -x "/opt/apps/roboticsservice/runService.sh" ] && [ "${FORCE_XROBOT_SERVICE_INSTALL}" != "1" ]; then
    log "XRoboToolkit PC Service already appears installed; set KITOV_FORCE_XROBOT_SERVICE_INSTALL=1 to reinstall"
    return
  fi

  if [ ! -r /etc/os-release ]; then
    log "cannot detect Ubuntu version because /etc/os-release is not readable; falling back to source build"
    install_xrobot_pc_service_from_source
    return
  fi

  local deb_path=""
  if [ "${arch}" = "amd64" ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    case "${VERSION_ID:-}" in
      20.04)
        deb_path="packages/xrobotoolkit_pc_service/XRoboToolkit_PC_Service_1.0.0_ubuntu_20.04_amd64.deb"
        ;;
      22.04)
        deb_path="packages/xrobotoolkit_pc_service/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb"
        ;;
      *)
        log "no bundled XRoboToolkit PC Service .deb for Ubuntu VERSION_ID=${VERSION_ID:-unknown}; falling back to source build"
        ;;
    esac
  else
    log "no bundled XRoboToolkit PC Service .deb for arch=${arch}; falling back to source build"
  fi

  if [ -z "${deb_path}" ]; then
    install_xrobot_pc_service_from_source
    return
  fi

  if [ ! -f "${deb_path}" ]; then
    log "XRoboToolkit PC Service package not found: ${deb_path}; falling back to source build"
    install_xrobot_pc_service_from_source
    return
  fi

  log "installing XRoboToolkit PC Service package: ${deb_path}"
  if ! sudo dpkg -i "${deb_path}"; then
    log "dpkg reported missing dependencies; running sudo apt-get install -f -y"
    sudo apt-get install -f -y
    sudo dpkg -i "${deb_path}"
  fi
}

install_selected_xrobot() {
  case "${XROBOT_SETUP}" in
    skip)
      log "skipping XRobot SDK / PC Service setup"
      ;;
    sdk)
      install_xrobot_python_sdk
      ;;
    service)
      install_xrobot_pc_service
      ;;
    all)
      install_xrobot_python_sdk
      install_xrobot_pc_service
      ;;
    *)
      die "unknown KITOV_XROBOT_SETUP=${XROBOT_SETUP}. Use skip, sdk, service, or all."
      ;;
  esac
}

command -v uv >/dev/null 2>&1 || die "uv not found. Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
choose_install_target "$@"
log "install target=${INSTALL_TARGET}"
choose_xrobot_setup
log "xrobot setup=${XROBOT_SETUP}"

if [ -d ".venv" ]; then
  if [ "${RECREATE_VENV}" = "1" ]; then
    log "recreating uv environment with Python ${PYTHON_VERSION}"
    uv venv --python "${PYTHON_VERSION}" --clear
  else
    log "reusing existing .venv; set KITOV_RECREATE_VENV=1 to recreate it"
  fi
else
  log "creating uv environment with Python ${PYTHON_VERSION}"
  uv venv --python "${PYTHON_VERSION}"
fi

log "syncing project dependencies"
uv sync --inexact

case "${INSTALL_TARGET}" in
  all)
    install_selected_onnxruntime
    install_selected_torch
    ;;
  skip)
    log "skipping optional ONNX Runtime/PyTorch installs; project dependencies are synced"
    ;;
  onnxruntime|onnx|ort)
    install_selected_onnxruntime
    ;;
  torch|pytorch)
    install_selected_torch
    ;;
  *)
    die "unknown install target=${INSTALL_TARGET}. Use all, onnxruntime, torch, or skip."
    ;;
esac

install_selected_xrobot

log "done"
