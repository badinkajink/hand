#!/usr/bin/env bash
# Run 13: Power-drill pivot-to-down morphology sweep with correct pivot deltas
# and a side-by-side contact-map / patch comparison.
#
# See RUN13.md for the full diff against run8 (last full morphology sweep on
# the drill). Short version of what changed:
#
#   1. Pivot fully specified to drive wrist (rx, ry, rz) from the open_flat
#      starting pose (1.6, 1.4, -0.031) to (0, 0, pi/2 ~ 1.5708). Run8/run12
#      drove only one axis each (run8: ry only; run12: rz only) so rx and ry
#      were stuck at the initial 1.6/1.4 values — confirmed by inspection of
#      assets/mjcf/scene_power_drill_short_proximal.xml @ open_flat ctrl.
#   2. Scene switched to scene_power_drill_short_proximal.xml (matches the
#      contact_targets YAML authored at the short-proximal grip).
#   3. Two parallel morphology sweeps:
#         a) baseline  (no contact targets, run8 objective shape)
#         b) contact_map (assets/contact_targets/power_drill_short_proximal.yaml
#            patches enabled with reward=10.0 / distance_penalty=20.0,
#            matching scripts/compare_methods.py + scripts/eval_suite.py)
#   4. Frozen-scene protocol enforced: phase1_optimize_grasp.py freezes the
#      scene before constructing the evaluator (see
#      morphohand.sampling.scene.freeze_scene_for_eval), and the morphology
#      sweep already emits rigid scenes per candidate via
#      write_rigid_scene_with_object_size. No raw morph-jointed scene reaches
#      Phase1GraspEvaluator anywhere in this run.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

SAMPLES="${SAMPLES:-800}"
SEED="${SEED:-13}"
RUN_FOUNDATIONAL="${RUN_FOUNDATIONAL:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_CONTACT_MAP="${RUN_CONTACT_MAP:-1}"
FP_REFRESH_INTERVAL="${FP_REFRESH_INTERVAL:-40}"
TOPK_GIFS="${TOPK_GIFS:-8}"

SCENE_XML="assets/mjcf/scene_power_drill_short_proximal.xml"
CONTACT_TARGETS_YAML="assets/contact_targets/power_drill_short_proximal.yaml"
KEYFRAMES=(open_flat)

# Pivot maths: open_flat wrist ctrl = (0.052, -0.038, -0.051, 1.6, 1.4, -0.031),
# pose_actuator_ids[3:6] = (a_palm_rx, a_palm_ry, a_palm_rz). Targets:
#   rx: 1.6   -> 0       => delta_rx = -1.6
#   ry: 1.4   -> 0       => delta_ry = -1.4
#   rz: -0.031 -> 1.5708 => delta_rz =  1.6018
PIVOT_DELTA_RX="-1.6"
PIVOT_DELTA_RY="-1.4"
PIVOT_DELTA_RZ="1.6018"
PIVOT_STEPS="${PIVOT_STEPS:-180}"
PIVOT_RAMP_STEPS="${PIVOT_RAMP_STEPS:-120}"

# Contact-target weights match compare_methods / eval_suite contact_map method.
CT_REWARD="${CT_REWARD:-10.0}"
CT_DIST_PENALTY="${CT_DIST_PENALTY:-20.0}"

FOUNDATIONAL_ROOT="results/phase1/run13_power_drill_pivot_to_down_foundational"
OUTPUT_ROOT="results/phase1"
BASELINE_TAG="${BASELINE_TAG:-run13_drill_short_proximal_pivot_to_down_baseline}"
CONTACT_MAP_TAG="${CONTACT_MAP_TAG:-run13_drill_short_proximal_pivot_to_down_contact_map}"

# Args shared by both morphology sweep variants. Drift caps + lift settings
# carried over from run8; pivot deltas are the corrected 3-axis version.
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
  --pivot-steps "$PIVOT_STEPS"
  --pivot-ramp-steps "$PIVOT_RAMP_STEPS"
  --pivot-delta-rx "$PIVOT_DELTA_RX"
  --pivot-delta-ry "$PIVOT_DELTA_RY"
  --pivot-delta-rz "$PIVOT_DELTA_RZ"
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

foundational_common_args=(
  --scene-xml "$SCENE_XML"
  --keyframe open_flat
  --iterations 100
  --population 52
  --elite-fraction 0.25
  --sigma-init 0.10
  # No --skip-gif: render the foundational best grasp so issues (finger
  # drift, drill-ground collision, etc.) surface before we burn a sweep.
  --lift-delta-z 0.060
  --lift-ramp-steps 100
  --pivot-steps "$PIVOT_STEPS"
  --pivot-ramp-steps "$PIVOT_RAMP_STEPS"
  --pivot-delta-rx "$PIVOT_DELTA_RX"
  --pivot-delta-ry "$PIVOT_DELTA_RY"
  --pivot-delta-rz "$PIVOT_DELTA_RZ"
  --objective-weight-cube-yaw-drift-penalty 3.0
  --objective-weight-cube-axis-tilt-penalty 2.0
  --objective-weight-cube-ang-drift-penalty 1.0
)

run_foundational_pair () {
  # Foundational CEM is the "seed pose" the morphology sweep adapts from. We
  # produce one set of foundational poses per (variant, seed) so the two
  # morphology sweeps don't have to share an initial control vector.
  local variant_tag="$1"; shift
  local extra_args=("$@")
  for seed_i in 0 1 2; do
    local out_dir="$FOUNDATIONAL_ROOT/$variant_tag/open_flat/seed_${seed_i}"
    # phase1_optimize_grasp.py writes to $out_dir/<tag>/summary.json
    # (auto-timestamped tag when --tag is omitted), so resume = any nested
    # summary.json exists.
    if compgen -G "$out_dir/*/summary.json" >/dev/null 2>&1; then
      echo "[run13-drill] foundational $variant_tag seed=$seed_i already present, skipping"
      continue
    fi
    echo "[run13-drill] foundational $variant_tag seed=$seed_i"
    "$PYTHON_BIN" scripts/phase1_optimize_grasp.py \
      "${foundational_common_args[@]}" \
      --output-dir "$out_dir" \
      --seed "$seed_i" \
      "${extra_args[@]}"
  done
}

if [[ "$RUN_FOUNDATIONAL" == "1" ]]; then
  echo "[run13-drill] === foundational optimization (open_flat, 3 seeds per variant) ==="
  if [[ "$RUN_BASELINE" == "1" ]]; then
    run_foundational_pair baseline
  fi
  if [[ "$RUN_CONTACT_MAP" == "1" ]]; then
    run_foundational_pair contact_map \
      --contact-targets-yaml "$CONTACT_TARGETS_YAML" \
      --objective-weight-contact-target-reward "$CT_REWARD" \
      --objective-weight-contact-target-distance-penalty "$CT_DIST_PENALTY"
  fi
fi

# Each sweep variant points at its own foundational seed folder via
# --foundational-root, so it picks up the right initial control vector.
if [[ "$RUN_BASELINE" == "1" ]]; then
  echo "[run13-drill] === morphology sweep: baseline (no contact targets) ==="
  "$PYTHON_BIN" scripts/run6_combined_multitask.py \
    --scene-xml "$SCENE_XML" \
    "${common_args[@]}" \
    --foundational-root "$FOUNDATIONAL_ROOT/baseline" \
    --tag "$BASELINE_TAG"
fi

if [[ "$RUN_CONTACT_MAP" == "1" ]]; then
  echo "[run13-drill] === morphology sweep: contact_map (patch reward/distance) ==="
  "$PYTHON_BIN" scripts/run6_combined_multitask.py \
    --scene-xml "$SCENE_XML" \
    "${common_args[@]}" \
    --foundational-root "$FOUNDATIONAL_ROOT/contact_map" \
    --contact-targets-yaml "$CONTACT_TARGETS_YAML" \
    --objective-weight-contact-target-reward "$CT_REWARD" \
    --objective-weight-contact-target-distance-penalty "$CT_DIST_PENALTY" \
    --tag "$CONTACT_MAP_TAG"
fi

# Side-by-side analysis if both sweeps produced their CSVs. (Includes
# variants from prior runs too — useful when resuming with RUN_BASELINE=0
# after the baseline sweep already finished.)
analysis_dirs=()
if [[ -f "$OUTPUT_ROOT/$BASELINE_TAG/all_candidates_multitask.csv" ]]; then
  analysis_dirs+=("$OUTPUT_ROOT/$BASELINE_TAG")
fi
if [[ -f "$OUTPUT_ROOT/$CONTACT_MAP_TAG/all_candidates_multitask.csv" ]]; then
  analysis_dirs+=("$OUTPUT_ROOT/$CONTACT_MAP_TAG")
fi
if [[ ${#analysis_dirs[@]} -gt 0 ]]; then
  echo "[run13-drill] === analysis across variants ==="
  "$PYTHON_BIN" scripts/run6_analysis.py \
    --run-dirs "${analysis_dirs[@]}" \
    --output-subdir analysis \
    --metrics cube_xy_drift cube_yaw_drift cube_axis_tilt cube_ang_drift finger_flex_drift cube_lift
fi

echo "[run13-drill] complete:"
for d in "${analysis_dirs[@]}"; do
  echo "  - $d"
done
