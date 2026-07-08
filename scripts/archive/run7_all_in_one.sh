#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

SAMPLES="${SAMPLES:-1200}"
SEED="${SEED:-7}"
RUN_FOUNDATIONAL="${RUN_FOUNDATIONAL:-1}"
FP_REFRESH_INTERVAL="${FP_REFRESH_INTERVAL:-40}"
TOPK_GIFS="${TOPK_GIFS:-8}"

SCENE_XML="assets/mjcf/scene_screwdriver_medium.xml"
CAPSULE_SCENE_XML="assets/mjcf/scene_screwdriver_medium_capsuletips.xml"
KEYFRAMES=(open_flat open_vertical open_90vertical)

FOUNDATIONAL_ROOT="results/phase1/run7_foundational"
OUTPUT_ROOT="results/phase1"
BASELINE_TAG="${BASELINE_TAG:-run7_strict_spheres}"
CAPSULE_TAG="${CAPSULE_TAG:-run7_strict_capsules}"

common_args=(
  --keyframes "${KEYFRAMES[@]}"
  --samples "$SAMPLES"
  --seed "$SEED"
  --window 100
  --fp-refresh-interval "$FP_REFRESH_INTERVAL"
  --morph-sort distance
  --top-k-gifs "$TOPK_GIFS"
  --output-dir "$OUTPUT_ROOT"
  --foundational-root "$FOUNDATIONAL_ROOT"
  --max-cube-xy-drift 0.018
  --max-cube-yaw-drift 0.26
  --max-cube-axis-tilt 0.20
  --max-cube-ang-drift 0.44
  --vertical-keyframes open_vertical open_90vertical
  --vertical-max-cube-xy-drift 0.010
  --vertical-max-cube-yaw-drift 0.11
  --vertical-max-cube-axis-tilt 0.09
  --vertical-max-cube-ang-drift 0.20
  --objective-weight-cube-yaw-drift-penalty 4.0
  --objective-weight-cube-axis-tilt-penalty 6.0
  --objective-weight-cube-ang-drift-penalty 2.0
  --objective-weight-finger-yaw-drift-penalty 1.1
  --objective-weight-finger-flex-drift-penalty 0.55
  --interval-adapt-iterations 14
  --interval-adapt-population 32
  --interval-adapt-elite-fraction 0.25
  --interval-adapt-sigma-init 0.08
  --sparse-adapt-iterations 2
  --sparse-adapt-population 10
  --sparse-adapt-elite-fraction 0.25
  --sparse-adapt-sigma-init 0.055
)

if [[ "$RUN_FOUNDATIONAL" == "1" ]]; then
  echo "[run7] Running foundational optimization for all keyframes"
  for keyframe in "${KEYFRAMES[@]}"; do
    for seed_i in 0 1 2; do
      out_dir="$FOUNDATIONAL_ROOT/$keyframe/seed_${seed_i}"
      "$PYTHON_BIN" scripts/phase1_optimize_grasp.py \
        --scene-xml "$SCENE_XML" \
        --keyframe "$keyframe" \
        --output-dir "$out_dir" \
        --iterations 96 \
        --population 48 \
        --elite-fraction 0.25 \
        --sigma-init 0.10 \
        --seed "$seed_i" \
        --skip-gif \
        --objective-weight-cube-yaw-drift-penalty 4.0 \
        --objective-weight-cube-axis-tilt-penalty 6.0 \
        --objective-weight-cube-ang-drift-penalty 2.0
    done
  done
fi

echo "[run7] Ensuring capsule-tip scene exists"
"$PYTHON_BIN" scripts/generate_capsule_tip_scene.py \
  --input-scene "$SCENE_XML" \
  --output-scene "$CAPSULE_SCENE_XML" \
  --tip-radius 0.005 \
  --tip-half-length 0.006

echo "[run7] Running strict baseline (sphere tips)"
"$PYTHON_BIN" scripts/run6_combined_multitask.py \
  --scene-xml "$SCENE_XML" \
  "${common_args[@]}" \
  --tag "$BASELINE_TAG"

echo "[run7] Running strict capsule-tip variant"
"$PYTHON_BIN" scripts/run6_combined_multitask.py \
  --scene-xml "$CAPSULE_SCENE_XML" \
  "${common_args[@]}" \
  --tag "$CAPSULE_TAG"

if [[ -f "$OUTPUT_ROOT/$BASELINE_TAG/all_candidates_multitask.csv" ]]; then
  echo "[run7] Analysis: baseline + capsule run comparison"
  "$PYTHON_BIN" scripts/run6_analysis.py \
    --run-dirs "$OUTPUT_ROOT/$BASELINE_TAG" "$OUTPUT_ROOT/$CAPSULE_TAG" \
    --output-subdir analysis \
    --metrics cube_xy_drift cube_yaw_drift cube_axis_tilt cube_ang_drift finger_flex_drift
fi

echo "[run7] Complete: $OUTPUT_ROOT/$BASELINE_TAG and $OUTPUT_ROOT/$CAPSULE_TAG"
