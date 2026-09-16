#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON_VERSION="${KITOV_PYTHON_VERSION:-3.10}"
INSTALL_TARGET="${KITOV_INSTALL_TARGET:-}"
TORCH_MODE="${KITOV_TORCH_MODE:-auto}"
JETSON_TORCH_WHEEL="${KITOV_JETSON_TORCH_WHEEL:-}"
ONNXRUNTIME_MODE="${KITOV_ONNXRUNTIME_MODE:-auto}"
JETSON_ONNXRUNTIME_WHEEL="${KITOV_JETSON_ONNXRUNTIME_WHEEL:-}"
RECREATE_VENV="${KITOV_RECREATE_VENV:-0}"
JETSON_ONNXRUNTIME_JP6_CU126_WHEEL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/e1e/9e3dc2f4d5551/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl"

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
  if [ -z "${wheel}" ]; then
    if [ -f /etc/nv_tegra_release ] && grep -q "R36" /etc/nv_tegra_release; then
      wheel="${JETSON_ONNXRUNTIME_JP6_CU126_WHEEL}"
      log "using default JetPack 6 ONNX Runtime GPU wheel"
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

  log "replacing CPU onnxruntime with Jetson GPU wheel"
  uv pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime_gpu >/dev/null 2>&1 || true
  uv pip install "numpy<2"
  uv pip install "${wheel}"
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
  uv run python - <<'PY'
import torch
print("[setup_env] torch:", torch.__version__)
print("[setup_env] cuda available:", torch.cuda.is_available())
print("[setup_env] torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("[setup_env] gpu:", torch.cuda.get_device_name(0))
PY
}

verify_onnxruntime() {
  uv run python - <<'PY'
import onnxruntime as ort
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

command -v uv >/dev/null 2>&1 || die "uv not found. Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
choose_install_target "$@"
log "install target=${INSTALL_TARGET}"

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

log "done"
