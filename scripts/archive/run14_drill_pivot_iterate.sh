#!/usr/bin/env bash
# Run 14: Debug iteration on drill pivot-to-down. NOT a morphology sweep.
#
# Three coupled changes relative to run13 foundational:
#   1. Capsule fingertips
#      (assets/mjcf/scene_power_drill_short_proximal_capsuletips.xml,
#      generated via scripts/generate_capsule_tip_scene.py).
#   2. Lift bumped 0.060 -> 0.110 (head clearance during pivot); ramp scaled
#      100 -> 180 to preserve the per-step lift rate.
#   3. NEW anchor penalty `--objective-weight-finger-ctrl-anchor` on
#      mean(|finger_ctrl - keyframe_finger_ctrl|). Closes the CEM loophole
#      where finger_yaw/flex_drift only penalize qpos motion during
#      rollout, so the optimizer could set a ctrl that opened the fingers
#      from the start and pay zero drift cost.
#
# Renders video on every foundational seed (per
# feedback_prelim_render_videos.md).

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

# Pivot deltas (same as run13, correct 3-axis spec).
PIVOT_DELTA_RX="-1.6"
PIVOT_DELTA_RY="-1.4"
PIVOT_DELTA_RZ="1.6018"
PIVOT_STEPS="${PIVOT_STEPS:-180}"
PIVOT_RAMP_STEPS="${PIVOT_RAMP_STEPS:-120}"

# Lift bumped from 0.060 to 0.110; ramp scaled proportionally.
LIFT_DELTA_Z="${LIFT_DELTA_Z:-0.110}"
LIFT_RAMP_STEPS="${LIFT_RAMP_STEPS:-180}"

# Anchor penalty weight. Mean |ctrl-keyframe| ~0.25 on the run13 winner;
# at weight 2.0 that's ~0.5 score penalty, comparable in size to existing
# drift terms but smaller than contact_target_distance_penalty.
CTRL_ANCHOR_WEIGHT="${CTRL_ANCHOR_WEIGHT:-2.0}"

# Contact-target weights match compare_methods / eval_suite contact_map.
CT_REWARD="${CT_REWARD:-10.0}"
CT_DIST_PENALTY="${CT_DIST_PENALTY:-20.0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/phase1/run14_drill_pivot_iterate}"

base_args=(
  --scene-xml "$SCENE_XML"
  --keyframe open_flat
  --iterations 100
  --population 52
  --elite-fraction 0.25
  --sigma-init 0.10
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
)

run_variant () {
  local variant_tag="$1"; shift
  local extra_args=("$@")
  for seed_i in $(seq 0 $((SEEDS - 1))); do
    local out_dir="$OUTPUT_ROOT/$variant_tag/seed_${seed_i}"
    if compgen -G "$out_dir/*/summary.json" >/dev/null 2>&1; then
      echo "[run14] $variant_tag seed=$seed_i already present, skipping"
      continue
    fi
    echo "[run14] === $variant_tag seed=$seed_i ==="
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
echo "[run14] === summary ==="
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
root = Path("results/phase1/run14_drill_pivot_iterate")
if not root.exists():
    print("(no output dir)")
    raise SystemExit
print(f"{'variant':12s} {'seed':5s} {'score':>8s} {'lift':>7s} {'tilt':>7s} {'anchor':>7s} {'mean_d_to_patch':>16s}")
for sp in sorted(root.rglob("summary.json")):
    s = json.load(sp.open())
    variant = sp.parts[-4]
    seed = sp.parts[-3]
    m = s["best_metrics"]
    print(
        f"{variant:12s} {seed:5s} {s['best_score']:+8.3f} "
        f"{m.get('cube_lift', 0):+7.3f} {m.get('cube_axis_tilt', 0):+7.3f} "
        f"{m.get('finger_ctrl_anchor_dist', 0):+7.3f} "
        f"{m.get('contact_target_mean_distance', 0):+16.3f}"
    )
PY
