#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-gpu"
PYTHON_VERSION="3.12"
RECREATE="0"
INSTALL_WARP="0"

usage() {
  cat <<'EOF'
Usage: scripts/setup_gpu_mjx_env.sh [options]

Options:
  --venv-dir <path>      Virtualenv path (default: .venv-gpu under repo root)
  --python <version>     Python version for uv venv (default: 3.12)
  --recreate             Delete and recreate the virtualenv before install
  --with-warp            Also install editable mujoco_warp and comfree_warp packages
  -h, --help             Show this help

This script creates an isolated GPU MJX environment and validates:
- jax backend is gpu
- at least one CudaDevice is visible
- mujoco.mjx imports successfully
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --recreate)
      RECREATE="1"
      shift
      ;;
    --with-warp)
      INSTALL_WARP="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$RECREATE" == "1" && -d "$VENV_DIR" ]]; then
  rm -rf "$VENV_DIR"
fi

cd "$ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[gpu-setup] creating venv at $VENV_DIR (python=$PYTHON_VERSION)"
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
else
  echo "[gpu-setup] reusing existing venv at $VENV_DIR"
fi

echo "[gpu-setup] installing GPU MJX core stack"
uv pip install --python "$VENV_DIR/bin/python" \
  "mujoco==3.6.0" "mujoco-mjx==3.6.0" "jax[cuda12]==0.9.2"

echo "[gpu-setup] installing project in editable mode (no dependency overrides)"
uv pip install --python "$VENV_DIR/bin/python" --no-deps -e .

echo "[gpu-setup] installing phase1 runtime dependencies"
uv pip install --python "$VENV_DIR/bin/python" \
  "matplotlib>=3.8" "imageio>=2.34" "pillow>=10.0" "tyro>=0.8.5" "pyyaml>=6.0.2"

if [[ "$INSTALL_WARP" == "1" ]]; then
  echo "[gpu-setup] installing warp-based backends (editable)"
  uv pip install --python "$VENV_DIR/bin/python" -e external/mujoco_warp -e external/comfree_warp
fi

echo "[gpu-setup] validating GPU backend and MJX imports"
"$VENV_DIR/bin/python" - <<'PY'
import jax
import mujoco
import mujoco.mjx as mjx

backend = jax.default_backend()
devices = jax.devices()
if backend != "gpu":
    raise SystemExit(f"Expected jax backend 'gpu', got '{backend}'")
if not any(getattr(d, "platform", "") == "gpu" for d in devices):
    raise SystemExit(f"Expected at least one CUDA device, got: {devices}")

print("jax", jax.__version__)
print("backend", backend)
print("devices", devices)
print("mujoco", mujoco.__version__)
print("mjx", mjx.__name__)
PY

echo "[gpu-setup] done"
