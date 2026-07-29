#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

SAMPLES="${SAMPLES:-800}"
SEED="${SEED:-8}"
RUN_FOUNDATIONAL="${RUN_FOUNDATIONAL:-1}"
RUN_SHORT_PHALANGES="${RUN_SHORT_PHALANGES:-1}"
FP_REFRESH_INTERVAL="${FP_REFRESH_INTERVAL:-40}"
TOPK_GIFS="${TOPK_GIFS:-8}"

SCENE_XML="assets/mjcf/baseline/scenes/scene_power_drill.xml"
KEYFRAMES=(open_flat)

FOUNDATIONAL_ROOT="results/phase1/run8_power_drill_foundational"
OUTPUT_ROOT="results/phase1"
BASE_TAG="${BASE_TAG:-run8_power_drill_forward_to_down}"
SHORT_TAG="${SHORT_TAG:-run8_power_drill_forward_to_down_shortlen}"

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
  --vertical-keyframes none
  --lift-delta-z 0.060
  --lift-ramp-steps 100
  --pivot-steps 180
  --pivot-ramp-steps 120
  --pivot-delta-ry 1.20
  --max-cube-xy-drift 0.030
  --max-cube-yaw-drift 0.55
  --max-cube-axis-tilt 0.65
  --max-cube-ang-drift 1.40
  --objective-weight-cube-yaw-drift-penalty 3.0
  --objective-weight-cube-axis-tilt-penalty 2.0
  --objective-weight-cube-ang-drift-penalty 1.0
  --objective-weight-finger-yaw-drift-penalty 1.0
  --objective-weight-finger-flex-drift-penalty 0.45
  --interval-adapt-iterations 16
  --interval-adapt-population 36
  --interval-adapt-elite-fraction 0.25
  --interval-adapt-sigma-init 0.09
  --sparse-adapt-iterations 2
  --sparse-adapt-population 10
  --sparse-adapt-elite-fraction 0.25
  --sparse-adapt-sigma-init 0.06
)

if [[ "$RUN_FOUNDATIONAL" == "1" ]]; then
  echo "[run8-drill] Running foundational optimization for open_flat"
  for seed_i in 0 1 2; do
    out_dir="$FOUNDATIONAL_ROOT/open_flat/seed_${seed_i}"
    "$PYTHON_BIN" scripts/phase1_optimize_grasp.py \
      --scene-xml "$SCENE_XML" \
      --keyframe open_flat \
      --output-dir "$out_dir" \
      --iterations 100 \
      --population 52 \
      --elite-fraction 0.25 \
      --sigma-init 0.10 \
      --seed "$seed_i" \
      --skip-gif \
      --lift-delta-z 0.060 \
      --lift-ramp-steps 100 \
      --pivot-steps 180 \
      --pivot-ramp-steps 120 \
      --pivot-delta-ry 1.20 \
      --objective-weight-cube-yaw-drift-penalty 3.0 \
      --objective-weight-cube-axis-tilt-penalty 2.0 \
      --objective-weight-cube-ang-drift-penalty 1.0
  done
fi

echo "[run8-drill] Running baseline power drill sampler (forward -> down pivot)"
"$PYTHON_BIN" scripts/run6_combined_multitask.py \
  --scene-xml "$SCENE_XML" \
  "${common_args[@]}" \
  --tag "$BASE_TAG"

if [[ "$RUN_SHORT_PHALANGES" == "1" ]]; then
  echo "[run8-drill] Running short-phalanx follow-up sampler"
  "$PYTHON_BIN" scripts/run6_combined_multitask.py \
    --scene-xml "$SCENE_XML" \
    "${common_args[@]}" \
    --len-max 0.024 \
    --len-perturb 0.008 \
    --tag "$SHORT_TAG"
fi

if [[ -f "$OUTPUT_ROOT/$BASE_TAG/all_candidates_multitask.csv" ]]; then
  run_dirs=("$OUTPUT_ROOT/$BASE_TAG")
  if [[ "$RUN_SHORT_PHALANGES" == "1" && -f "$OUTPUT_ROOT/$SHORT_TAG/all_candidates_multitask.csv" ]]; then
    run_dirs+=("$OUTPUT_ROOT/$SHORT_TAG")
  fi

  echo "[run8-drill] Running analysis"
  "$PYTHON_BIN" scripts/run6_analysis.py \
    --run-dirs "${run_dirs[@]}" \
    --output-subdir analysis \
    --metrics cube_xy_drift cube_yaw_drift cube_axis_tilt cube_ang_drift finger_flex_drift cube_lift
fi

if [[ "$RUN_SHORT_PHALANGES" == "1" ]]; then
  echo "[run8-drill] Complete: $OUTPUT_ROOT/$BASE_TAG and $OUTPUT_ROOT/$SHORT_TAG"
else
  echo "[run8-drill] Complete: $OUTPUT_ROOT/$BASE_TAG"
fi
