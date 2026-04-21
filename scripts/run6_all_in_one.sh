#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-6}"
TAG="${TAG:-run6_combined_1000}"
RUN_FOUNDATIONAL="${RUN_FOUNDATIONAL:-0}"
FP_REFRESH_INTERVAL="${FP_REFRESH_INTERVAL:-50}"
TOPK_GIFS="${TOPK_GIFS:-5}"

SCENE_XML="assets/mjcf/scene_screwdriver_medium.xml"
KEYFRAMES=(open_flat open_vertical open_90vertical)
FOUNDATIONAL_ROOT="results/phase1/run6_foundational"
OUTPUT_ROOT="results/phase1"

if [[ "$RUN_FOUNDATIONAL" == "1" ]]; then
  echo "[run6] Running foundational optimization for all keyframes"
  for keyframe in "${KEYFRAMES[@]}"; do
    for seed_i in 0 1; do
      out_dir="$FOUNDATIONAL_ROOT/$keyframe/seed_${seed_i}"
      "$PYTHON_BIN" scripts/phase1_optimize_grasp.py \
        --scene-xml "$SCENE_XML" \
        --keyframe "$keyframe" \
        --output-dir "$out_dir" \
        --iterations 80 \
        --population 36 \
        --elite-fraction 0.25 \
        --sigma-init 0.12 \
        --seed "$seed_i"
    done
  done
fi

echo "[run6] Running combined multitask sampler"
"$PYTHON_BIN" scripts/run6_combined_multitask.py \
  --scene-xml "$SCENE_XML" \
  --keyframes "${KEYFRAMES[@]}" \
  --foundational-root "$FOUNDATIONAL_ROOT" \
  --samples "$SAMPLES" \
  --seed "$SEED" \
  --window 100 \
  --fp-refresh-interval "$FP_REFRESH_INTERVAL" \
  --morph-sort distance \
  --top-k-gifs "$TOPK_GIFS" \
  --output-dir "$OUTPUT_ROOT" \
  --tag "$TAG"

RUN_DIR="$OUTPUT_ROOT/$TAG"

if [[ -f "$RUN_DIR/all_candidates_multitask.csv" ]]; then
  echo "[run6] Running analysis on combined run outputs"
  "$PYTHON_BIN" scripts/run6_analysis.py \
    --run-dirs "$RUN_DIR" \
    --output-subdir analysis
fi

echo "[run6] Complete: $RUN_DIR"
