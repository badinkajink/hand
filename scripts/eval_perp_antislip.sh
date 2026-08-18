#!/usr/bin/env bash
# Evaluate the r6 anti-slip sweep against r4, as DISTRIBUTIONS, plus a render of each.
#
# Single rollouts of this stack are not reproducible (three deterministic rollouts of one r4
# checkpoint ended three different ways -- parallel contact solves do not reduce in a fixed
# order on GPU), so every headline number here is over N envs. The render exists to be LOOKED
# at, not to be measured: it is one draw, and on this topology a shaft standing upright on the
# floor reads cos +1.000 while being a total failure.
#
# The comparison that matters is hold_steps (aligned AND physically held), NOT hold_rate at a
# fixed step -- a fixed-step rate ranks a SLOWER policy higher when the horizon truncates
# before it finishes failing, which is the trap r2-vs-r4 already fell into.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

N="${N:-64}"
STEPS="${STEPS:-600}"
OUT="${OUT:-$ROOT/docs/rl/videos/20260817_perp_review}"
EXP="$ROOT/docs/experiments"
mkdir -p "$OUT"

# r4 evaluates on the excludes scene it was NOT trained on (zero-shot); the r6 runs evaluate on
# the same scene they trained on. Both are the current base scene, so the reorient is compared
# like for like -- the only asymmetry is r4's zero-shot transfer, already measured as intact.
EVAL_RUN_R4="$ROOT/results/phase1/perp/eval_shipped_excludes"

run_eval() {                       # $1 label  $2 policy  $3 morph-run  $4 slug
  echo "=== eval $1"
  export WARP_CACHE_PATH="$(mktemp -d)"
  MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/policy_eval_suite.py \
      --policy "$2" --morphology-run "$3" \
      --closed-ctrl-from-keyframe closed --open-finger-from-keyframe \
      --lift-delta 0.14 --steps "$STEPS" --n "$N" \
      --json-out "$EXP/20260817-$4.json" \
      --plot "$OUT/$4_eval.png" --label "$1"
  rm -rf "$WARP_CACHE_PATH"
}

run_render() {                     # $1 run-dir  $2 slug
  echo "=== render $2"
  export WARP_CACHE_PATH="$(mktemp -d)"
  MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/rl_render_reorient.py \
      --run "$1" --checkpoint model_338.pt \
      --output "$OUT/$2.mp4" --steps "$STEPS" --width 960 --height 720
  MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/policy_filmstrip.py \
      --run "$1" --video "$OUT/$2.mp4" --out "$OUT/$2_filmstrip.png"
  rm -rf "$WARP_CACHE_PATH"
}

for W in ${WEIGHTS:--1000 -3000}; do
  RUN="$(ls -dt "$ROOT/results/rl/"*"perp_single_r6_slip${W}" 2>/dev/null | head -1)"
  [[ -n "$RUN" && -f "$RUN/tensorboard/model_338.pt" ]] || { echo "SKIP $W (no model_338)"; continue; }
  # The r6 runs trained on the shipped morphology re-baked from the current base scene; the
  # eval needs a morphology-run dir (summary.json + frozen_scene.xml), so mirror perp_v1 with
  # that scene substituted -- the same construction the r4 excludes eval used.
  EV="$ROOT/results/phase1/perp/eval_r6_slip${W}"
  rm -rf "$EV"; cp -r "$ROOT/results/phase1/perp/perp_v1" "$EV"
  cp "$ROOT/results/rl/perp_compact_queue/t0.00_x0.00_y0.00/frozen_scene.xml" "$EV/frozen_scene.xml"
  run_eval "r6 axial-slip w=$W" "$RUN/tensorboard/model_338.pt" "$EV" "perp_r6_slip${W}"
  run_render "$RUN" "r6_slip${W}"
done

echo
echo "=== summary (r4 baseline is docs/experiments/20260817-perp_r4_on_excludes_scene.json) ==="
uv run python - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("docs/experiments/20260817-perp_r*.json")):
    if "envelope" in p or "compliance" in p or "control" in p:
        continue
    d = json.load(open(p))
    rows.append((d.get("label", os.path.basename(p)), d))
hdr = f"{'policy':34s} {'align':>7s} {'peak_cos':>9s} {'t_align':>9s} {'hold_steps':>12s} {'drop_step':>10s}"
print(hdr); print("-" * len(hdr))
for label, d in rows:
    print(f"{label[:34]:34s} {d['align_rate']*100:6.1f}% {d['peak_cos_mean']:9.3f} "
          f"{d.get('t_align_mean') or float('nan'):9.1f} "
          f"{d['hold_steps_mean']:7.0f}±{d['hold_steps_sd']:<4.0f} "
          f"{d.get('drop_step_mean') or float('nan'):10.1f}")
PY
