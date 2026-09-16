#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON_VERSION="${KITOV_PYTHON_VERSION:-3.10}"
TORCH_MODE="${KITOV_TORCH_MODE:-auto}"
JETSON_TORCH_WHEEL="${KITOV_JETSON_TORCH_WHEEL:-}"
RECREATE_VENV="${KITOV_RECREATE_VENV:-0}"

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

command -v uv >/dev/null 2>&1 || die "uv not found. Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"

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

case "${TORCH_MODE}" in
  auto)
    install_torch_auto
    ;;
  skip)
    log "skipping PyTorch install because KITOV_TORCH_MODE=skip"
    exit 0
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
log "done"
