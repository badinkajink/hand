#!/usr/bin/env bash
# Run 17: Drill pivot-to-down with the new `open_flat_gripping` keyframe.
#
# Runs 13/14/15 all converged to the same failure mode:
# all_finger_contact_persistence = 0 across every seed. Root cause was the
# `open_flat` keyframe geometry — fingertips tangent to the barrel, no
# inward press. Lightning Grasp (run16) hit the same failure family, which
# confirmed the keyframe was the bottleneck rather than the optimizer.
#
# Candidate fix #1 from project_drill_pivot_open_question.md: author a new
# keyframe whose fingers are already curled around the grip so the keyframe
# itself is a closing pose. That keyframe now exists:
#   scene: assets/mjcf/baseline/scenes/scene_power_drill_short_proximal_rigid_capsuletips.xml
#   key:   open_flat_gripping
#     thumb_pip = -1.7  (was -1.2 in open_flat)
#     index_pip = +1.5  (was +1.1)
#     middle_pip = +2.0 (was +1.4)
#     mcps and palm wrist match open_flat (1.6, 1.4, -0.031), so the same
#     3-axis pivot deltas apply.
#
# This is a debug iteration, NOT a morphology sweep. Same objective shape as
# run15 (trajectory FC + min-finger-persistence + anchor) so the only
# variable vs run15 is the keyframe geometry.
#
# Capsule fingertips are already in the scene name; the scene is pre-rigid
# (no morph joints), so freeze_scene_for_eval will copy it verbatim.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

SCENE_XML="${SCENE_XML:-assets/mjcf/baseline/scenes/scene_power_drill_short_proximal_rigid_capsuletips.xml}"
KEYFRAME="${KEYFRAME:-open_flat_gripping}"
CONTACT_TARGETS_YAML="assets/contact_targets/power_drill_short_proximal.yaml"
SEEDS="${SEEDS:-3}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_CONTACT_MAP="${RUN_CONTACT_MAP:-1}"

# Wrist start (palm_rx, palm_ry, palm_rz) at open_flat_gripping = (1.6, 1.4,
# -0.031) — identical to open_flat — so target (0, 0, pi/2) gives:
PIVOT_DELTA_RX="-1.6"
PIVOT_DELTA_RY="-1.4"
PIVOT_DELTA_RZ="1.6018"
PIVOT_STEPS="${PIVOT_STEPS:-180}"
PIVOT_RAMP_STEPS="${PIVOT_RAMP_STEPS:-120}"

# Lift carried over from run14/15 (head clearance during pivot).
LIFT_DELTA_Z="${LIFT_DELTA_Z:-0.110}"
LIFT_RAMP_STEPS="${LIFT_RAMP_STEPS:-180}"

# Anchor pull: keep run15's relaxed 0.5 so this is a clean one-variable test
# of the new keyframe. (Higher anchor is now safe since the keyframe is a
# closing grip; revisit upward if first pass already shows closure.)
CTRL_ANCHOR_WEIGHT="${CTRL_ANCHOR_WEIGHT:-0.5}"

# Trajectory force-closure objective from run15.
TRAJ_FC_Q1_WEIGHT="${TRAJ_FC_Q1_WEIGHT:-2.0}"
TRAJ_FC_FINGERS_WEIGHT="${TRAJ_FC_FINGERS_WEIGHT:-1.0}"
TRAJ_FC_SAMPLES="${TRAJ_FC_SAMPLES:-8}"

MIN_FINGER_PERSIST_WEIGHT="${MIN_FINGER_PERSIST_WEIGHT:-6.0}"

ITERATIONS="${ITERATIONS:-120}"
POPULATION="${POPULATION:-52}"
SIGMA_INIT="${SIGMA_INIT:-0.15}"

CT_REWARD="${CT_REWARD:-10.0}"
CT_DIST_PENALTY="${CT_DIST_PENALTY:-20.0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/phase1/run17_drill_pivot_gripping_keyframe}"

base_args=(
  --scene-xml "$SCENE_XML"
  --keyframe "$KEYFRAME"
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
      echo "[run17] $variant_tag seed=$seed_i already present, skipping"
      continue
    fi
    echo "[run17] === $variant_tag seed=$seed_i ==="
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
echo "[run17] === summary ==="
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
root = Path("results/phase1/run17_drill_pivot_gripping_keyframe")
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
