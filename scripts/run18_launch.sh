#!/usr/bin/env bash
# One-shot run18 launcher: sweep -> filter -> analysis -> render -> summary.
# All steps run sequentially; this script is the single entry point so the
# user doesn't have to chain them.
#
# Defaults reflect the "current best" objective:
#   contact_target_reward = 3.0   (down from 10 — proximity must not dominate)
#   min_finger_persistence = 14.0 (up from 6 — real contact must dominate)
#   adapt = interval-initial-fp + sparse-per-morph (run6's two-stage adapt)
#
# Pass any overrides as `KEY=VALUE` env vars before invocation, e.g.:
#   SAMPLES=500 TAG=run18_test ./scripts/run18_launch.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"

TAG="${TAG:-run18_final}"
SAMPLES="${SAMPLES:-2000}"
TOP_K="${TOP_K:-5}"

CONTACT_TARGET_REWARD="${CONTACT_TARGET_REWARD:-3.0}"
CONTACT_TARGET_DISTANCE_PENALTY="${CONTACT_TARGET_DISTANCE_PENALTY:-20.0}"
MIN_FINGER_PERSISTENCE="${MIN_FINGER_PERSISTENCE:-14.0}"

ADAPT_MODE="${ADAPT_MODE:-interval-initial-fp}"
FP_REFRESH_INTERVAL="${FP_REFRESH_INTERVAL:-40}"
INTERVAL_ADAPT_ITERATIONS="${INTERVAL_ADAPT_ITERATIONS:-16}"
INTERVAL_ADAPT_POPULATION="${INTERVAL_ADAPT_POPULATION:-36}"

SPARSE_ADAPT_MODE="${SPARSE_ADAPT_MODE:-sparse-per-morph}"
SPARSE_ADAPT_ITERATIONS="${SPARSE_ADAPT_ITERATIONS:-2}"
SPARSE_ADAPT_POPULATION="${SPARSE_ADAPT_POPULATION:-10}"

FOUNDATIONAL_ITERATIONS="${FOUNDATIONAL_ITERATIONS:-100}"
FOUNDATIONAL_POPULATION="${FOUNDATIONAL_POPULATION:-52}"

OUTPUT_DIR="$ROOT_DIR/results/phase1/$TAG"

echo "[$TAG] === Stage 1/5: sweep ==="
"$PYTHON" scripts/run18_multi_object_sweep.py \
  --tag "$TAG" \
  --samples "$SAMPLES" \
  --foundational-iterations "$FOUNDATIONAL_ITERATIONS" \
  --foundational-population "$FOUNDATIONAL_POPULATION" \
  --adapt-mode "$ADAPT_MODE" \
  --fp-refresh-interval "$FP_REFRESH_INTERVAL" \
  --interval-adapt-iterations "$INTERVAL_ADAPT_ITERATIONS" \
  --interval-adapt-population "$INTERVAL_ADAPT_POPULATION" \
  --sparse-adapt-mode "$SPARSE_ADAPT_MODE" \
  --sparse-adapt-iterations "$SPARSE_ADAPT_ITERATIONS" \
  --sparse-adapt-population "$SPARSE_ADAPT_POPULATION" \
  --contact-target-reward "$CONTACT_TARGET_REWARD" \
  --contact-target-distance-penalty "$CONTACT_TARGET_DISTANCE_PENALTY" \
  --min-finger-persistence "$MIN_FINGER_PERSISTENCE"

echo
echo "[$TAG] === Stage 2/5: filter physics blowups ==="
"$PYTHON" scripts/run18_filter_blowups.py --run-dir "$OUTPUT_DIR"

echo
echo "[$TAG] === Stage 3/5: analysis (filtered) ==="
"$PYTHON" scripts/run18_analysis.py \
  --run-dir "$OUTPUT_DIR" \
  --cross-csv all_candidates_multi_filtered.csv \
  --per-task-csv all_candidates_filtered.csv \
  --output-subdir analysis_filtered \
  --top-k "$TOP_K"

echo
echo "[$TAG] === Stage 4/5: render top-$TOP_K videos ==="
"$PYTHON" scripts/run18_render_top.py \
  --run-dir "$OUTPUT_DIR" \
  --cross-csv all_candidates_multi_filtered.csv \
  --video-subdir videos_filtered \
  --top-k "$TOP_K" \
  --adapt-before-render \
  --adapt-iterations "$INTERVAL_ADAPT_ITERATIONS" \
  --adapt-population "$INTERVAL_ADAPT_POPULATION"

echo
echo "[$TAG] === Stage 5/5: plain-text summary ==="
"$PYTHON" scripts/run18_text_summary.py --run-dir "$OUTPUT_DIR" \
  | tee "$OUTPUT_DIR/SUMMARY.txt"

echo
echo "[$TAG] DONE. Artifacts under $OUTPUT_DIR"
