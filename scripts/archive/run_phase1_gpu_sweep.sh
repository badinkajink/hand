#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-gpu"
OUT_DIR="${ROOT_DIR}/results/phase1/run_gpu_mjx_sweep"
ITERATIONS="12"
SEED="0"
Y_VALUES="0.0250 0.0300 0.0350"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_phase1_gpu_sweep.sh [options] [-- <extra args to phase1_optimize_grasp.py>]

Options:
  --venv-dir <path>      GPU virtualenv path (default: .venv-gpu under repo root)
  --output-dir <path>    Sweep output directory
  --iterations <int>     Optimizer iterations (default: 12)
  --seed <int>           Random seed (default: 0)
  --y-values "vals"      Space-separated y half-sizes (default: "0.0250 0.0300 0.0350")
  -h, --help             Show this help

The script validates that JAX backend is GPU, then runs MJX-autodiff prism y-sweep
with GIF rendering enabled.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --y-values)
      Y_VALUES="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
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

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "GPU python not found at $VENV_DIR/bin/python" >&2
  echo "Run scripts/setup_gpu_mjx_env.sh first." >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$OUT_DIR"

echo "[gpu-sweep] validating GPU backend"
"$VENV_DIR/bin/python" - <<'PY'
import jax
backend = jax.default_backend()
if backend != "gpu":
    raise SystemExit(f"Expected jax backend 'gpu', got '{backend}'")
if not any(getattr(d, "platform", "") == "gpu" for d in jax.devices()):
  raise SystemExit(f"Expected at least one CUDA device, got: {jax.devices()}")
print("backend", backend)
print("devices", jax.devices())
PY

for y in $Y_VALUES; do
  tag="y$(echo "$y" | tr '.' 'd')"
  echo "[gpu-sweep] running y=$y tag=$tag"
  MUJOCO_GL=egl "$VENV_DIR/bin/python" scripts/phase1_optimize_grasp.py \
    --scene-xml "assets/mjcf/generated/scene_prism_x0.0200_y${y}_z0.0200.xml" \
    --optimizer mjx-autodiff \
    --iterations "$ITERATIONS" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    --tag "$tag" \
    "${EXTRA_ARGS[@]}"
done

echo "[gpu-sweep] done: $OUT_DIR"
