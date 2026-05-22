#!/usr/bin/env bash
# Run 15: Debug iteration on drill pivot-to-down, building on run14 with the
# paper-proposed trajectory force-closure metric.
#
# Diagnosis from run14: anchor penalty pulled the chosen ctrl back toward
# the keyframe, but `all_finger_contact_persistence` stayed at 0 across all
# seeds. The keyframe pose has fingertips tangent to the drill (not gripping),
# so the moment the wrist accelerates the contacts break and don't reform.
# Existing FC term was sampled ONLY at settle (paper proposed throughout
# trajectory) and weight=0 by default, so the optimizer had no signal that
# mid-trajectory contact loss was the problem.
#
# Changes vs run14:
#   1. Trajectory FC sampling enabled: q1_distance (Ferrari-Canny FC residual)
#      and fingers_engaged sampled at 8 points across lift+pivot+hold.
#      Penalty weight 2.0 on worst-case q1; reward weight 1.0 on min fingers.
#   2. Anchor weight reduced 2.0 -> 0.5 so the optimizer can find grips
#      meaningfully closer than the (open) keyframe.
#   3. min_finger_persistence weight boosted 2.0 -> 6.0 to give a stronger
#      contact-throughout signal alongside FC.
#   4. sigma_init bumped 0.10 -> 0.15, iterations 100 -> 120 to widen search
#      and give the optimizer more room to leave the bad keyframe basin.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

SCENE_XML="${SCENE_XML:-assets/mjcf/scene_power_drill_short_proximal_capsuletips.xml}"
CONTACT_TARGETS_YAML="assets/contact_targets/power_drill_short_proximal.yaml"
SEEDS="${SEEDS:-3}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_CONTACT_MAP="${RUN_CONTACT_MAP:-1}"

# Pivot deltas (same as run14, correct 3-axis spec).
PIVOT_DELTA_RX="-1.6"
PIVOT_DELTA_RY="-1.4"
PIVOT_DELTA_RZ="1.6018"
PIVOT_STEPS="${PIVOT_STEPS:-180}"
PIVOT_RAMP_STEPS="${PIVOT_RAMP_STEPS:-120}"

# Lift (same as run14).
LIFT_DELTA_Z="${LIFT_DELTA_Z:-0.110}"
LIFT_RAMP_STEPS="${LIFT_RAMP_STEPS:-180}"

# Anchor: relaxed (was 2.0 in run14 and clamped the optimizer to keyframe).
CTRL_ANCHOR_WEIGHT="${CTRL_ANCHOR_WEIGHT:-0.5}"

# Trajectory force-closure. q1 penalty is the worst-case Ferrari-Canny
# residual seen during lift/pivot/hold; fingers reward is min count of
# tips in contact at any sampled instant.
TRAJ_FC_Q1_WEIGHT="${TRAJ_FC_Q1_WEIGHT:-2.0}"
TRAJ_FC_FINGERS_WEIGHT="${TRAJ_FC_FINGERS_WEIGHT:-1.0}"
TRAJ_FC_SAMPLES="${TRAJ_FC_SAMPLES:-8}"

# Boost min_finger_persistence to amplify the "contact throughout" signal.
MIN_FINGER_PERSIST_WEIGHT="${MIN_FINGER_PERSIST_WEIGHT:-6.0}"

# Search budget bumped from run14 (100/52, sigma 0.10).
ITERATIONS="${ITERATIONS:-120}"
POPULATION="${POPULATION:-52}"
SIGMA_INIT="${SIGMA_INIT:-0.15}"

# Contact-target weights match compare_methods / eval_suite contact_map.
CT_REWARD="${CT_REWARD:-10.0}"
CT_DIST_PENALTY="${CT_DIST_PENALTY:-20.0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/phase1/run15_drill_pivot_traj_fc}"

base_args=(
  --scene-xml "$SCENE_XML"
  --keyframe open_flat
  --iterations "$ITERATIONS"
  --population "$POPULATION"
  --elite-fraction 0.25
  --sigma-init "$SIGMA_INIT"
  --lift-delta-z "$LIFT_DELTA_Z"
  --lift-ramp-steps "$LIFT_RAMP_STEPS"
  --pivot-steps "$PIVOT_STEPS"
  --pivot-ramp-steps "$PIVOT_RAMP_STEPS"
  --pivot-delta-rx "$PIVOT_DELTA_RX"
  --pivot-delta-ry "$PIVOT_DELTA_RY"
  --pivot-delta-rz "$PIVOT_DELTA_RZ"
  --objective-weight-cube-yaw-drift-penalty 3.0
  --objective-weight-cube-axis-tilt-penalty 2.0
  --objective-weight-cube-ang-drift-penalty 1.0
  --objective-weight-finger-ctrl-anchor "$CTRL_ANCHOR_WEIGHT"
  --objective-weight-min-finger-persistence "$MIN_FINGER_PERSIST_WEIGHT"
  --trajectory-fc-sample-count "$TRAJ_FC_SAMPLES"
  --objective-weight-trajectory-fc-q1-penalty "$TRAJ_FC_Q1_WEIGHT"
  --objective-weight-trajectory-fc-min-fingers-reward "$TRAJ_FC_FINGERS_WEIGHT"
)

run_variant () {
  local variant_tag="$1"; shift
  local extra_args=("$@")
  for seed_i in $(seq 0 $((SEEDS - 1))); do
    local out_dir="$OUTPUT_ROOT/$variant_tag/seed_${seed_i}"
    if compgen -G "$out_dir/*/summary.json" >/dev/null 2>&1; then
      echo "[run15] $variant_tag seed=$seed_i already present, skipping"
      continue
    fi
    echo "[run15] === $variant_tag seed=$seed_i ==="
    "$PYTHON_BIN" scripts/phase1_optimize_grasp.py \
      "${base_args[@]}" \
      --output-dir "$out_dir" \
      --seed "$seed_i" \
      "${extra_args[@]}"
  done
}

if [[ "$RUN_BASELINE" == "1" ]]; then
  run_variant baseline
fi

if [[ "$RUN_CONTACT_MAP" == "1" ]]; then
  run_variant contact_map \
    --contact-targets-yaml "$CONTACT_TARGETS_YAML" \
    --objective-weight-contact-target-reward "$CT_REWARD" \
    --objective-weight-contact-target-distance-penalty "$CT_DIST_PENALTY"
fi

echo
echo "[run15] === summary ==="
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
root = Path("results/phase1/run15_drill_pivot_traj_fc")
if not root.exists():
    print("(no output dir)")
    raise SystemExit
print(
    f"{'variant':12s} {'seed':6s} {'score':>8s} {'lift':>6s} {'tilt':>6s} "
    f"{'anchor':>6s} {'fc_max_q1':>9s} {'fc_min_fing':>11s} {'all_persist':>11s} {'min_persist':>11s}"
)
for sp in sorted(root.rglob("summary.json")):
    s = json.load(sp.open())
    variant = sp.parts[-4]
    seed = sp.parts[-3]
    m = s["best_metrics"]
    print(
        f"{variant:12s} {seed:6s} {s['best_score']:+8.3f} "
        f"{m.get('cube_lift', 0):+6.3f} {m.get('cube_axis_tilt', 0):+6.3f} "
        f"{m.get('finger_ctrl_anchor_dist', 0):+6.3f} "
        f"{m.get('trajectory_fc_max_q1', 0):+9.3f} "
        f"{m.get('trajectory_fc_min_fingers', 0):+11.1f} "
        f"{m.get('all_finger_contact_persistence', 0):+11.3f} "
        f"{m.get('min_finger_contact_persistence', 0):+11.3f}"
    )
PY
