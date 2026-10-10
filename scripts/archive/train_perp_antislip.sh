#!/usr/bin/env bash
# r6: r4 + the axial-slip penalty, swept over weight. Sequential, one 16 GB GPU.
#
# WHY. r4 reorients 100% of rollouts to peak cos 0.994 and then loses every one of them by
# ~step 400. Measured in the PALM frame (scripts/probe_joint_envelope.py's sibling calibration,
# docs/rl/perp_topology.md §2026-08-17), the shaft slides 44.8 +/- 0.8 mm DOWN through the
# pinch over the hold -- far more than the ~12 mm the world-frame plot suggests, because the
# palm is moving too. Nothing in the reward saw it: object_lateral_drift and object_xy_drift
# are both xy, and object_lift_height charges ~12 mm of height for what ends as a total loss.
# New term `object_axial_slip` charges downward palm-frame motion per step, one-sided.
#
# WHY IT IS NOT ALIGN-GATED, unlike the grip catch. The slip splits 23.8 mm before cos 0.7 (in
# only ~44 steps) and 20.4 mm after (over ~349 steps). More than half of it IS the rotation --
# the policy walks the shaft down through the pinch to turn it. So a gate at cos 0.7 would miss
# the faster half and still let the pad be spent before the hold begins.
#
# WHICH MEANS THE WEIGHT IS THE WHOLE EXPERIMENT, and it is genuinely two-sided:
#   too low  -> nothing changes, r4's drop reappears;
#   too high -> the cheapest way to stop slipping is to stop rotating, which is exactly how
#               revision 3 died (reward 883 -> 430 as it traded rotation for survival).
# Hence a SWEEP, not a guess. Integrated over an r4 episode at deadband 1e-4 the term is
# ~0.056 m, so weight -1000 costs ~-56 against an alignment reward that reaches ~+100.
#
#   bash scripts/train_perp_antislip.sh              # run/resume both weights
#   WEIGHTS="-1000" bash scripts/train_perp_antislip.sh
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# The shipped morphology baked from the CURRENT base scene (12 palm excludes). This is the same
# scene the fixed compact queue uses for t0.00_x0.00_y0.00, so r6 and the queue are comparable.
SCENE="${SCENE:-$ROOT/results/rl/perp_compact_queue/t0.00_x0.00_y0.00/frozen_scene.xml}"
MORPH_RUN="$ROOT/results/phase1/perp/perp_v1"
WEIGHTS="${WEIGHTS:--1000 -3000}"
DEADBAND="${DEADBAND:-0.0001}"
# NOT in the recipe -- lift is a run knob and the trainer default is 0.05. Omitting it is what
# voided the entire r5 queue; r2/r4 both trained at 0.14. See CLAUDE.md launcher-parity gotcha.
LIFT_DELTA="${LIFT_DELTA:-0.14}"
# r4 is the reference the whole experiment is measured against, so its config.yaml is the
# authority on every knob r6 is NOT deliberately changing. These three values are read off it.
REF_RUN="${REF_RUN:-$ROOT/results/rl/20260731-1300-perp_single_r4}"
NUM_ENVS="${NUM_ENVS:-3072}"
TOTAL_STEPS="${TOTAL_STEPS:-25000000}"
LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/perp_antislip_$(date +%Y%m%d-%H%M).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

wait_for_gpu() {
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [[ "$used" -lt 2500 ]] && return 0
    sleep 10
  done
  say "WARN: GPU still at ${used} MiB after 10 min; launching anyway"
}

[[ -f "$SCENE" ]] || { say "FATAL: scene missing: $SCENE"; exit 1; }
say "r6 anti-slip sweep: weights [$WEIGHTS] deadband $DEADBAND lift $LIFT_DELTA -> $QLOG"

for W in $WEIGHTS; do
  TAG="perp_single_r6_slip${W}"
  DEST="$ROOT/results/rl/antislip_${W}"; mkdir -p "$DEST"
  if [[ -f "$DEST/.DONE" ]]; then say "SKIP $W (.DONE)"; continue; fi

  # NaN ladder. The recipe pins init_noise_std 0.05, which was tuned on r4's scene; the shipped
  # morphology re-baked from the CURRENT base scene NaNs at iteration 0 under it at lift 0.14
  # and trains clean at 0.02. Verified NOT to be the new reward term: the same launch with
  # --axial-slip-weight 0 NaNs identically, which is the control the compact queue's comment
  # demands before touching the noise. Record whichever value actually ran -- it is a real
  # difference from r4's exploration and any comparison has to carry it.
  RC=1
  for NOISE in "" 0.02 0.01; do
    if [[ -n "$NOISE" ]]; then
      say "=== $TAG: RETRY at init_noise_std=$NOISE ==="
      NOISE_ARG=(--init-noise-std "$NOISE")
    else
      say "=== train $TAG (recipe noise) ==="
      NOISE_ARG=()
    fi
    export WARP_CACHE_PATH="$(mktemp -d)"    # a shared Warp cache races and NaNs
    # Flags are NOT a matter of taste -- they are copied from REF's config.yaml, and the
    # pre-flight below re-checks that claim against the file the trainer actually wrote.
    # --open-finger-from-keyframe and --num-envs/--total-timesteps are here because omitting
    # them is silent: the first gives the wrong open pose (gotcha #5), the second inherits the
    # 200M default and a 6 h run. The recipe pinned open_finger_from_keyframe when r5 ran and
    # does not today, which is precisely why it is passed explicitly rather than trusted.
    MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/rl_train_cube.py \
        --recipe perp_single \
        --morphology-run "$MORPH_RUN" \
        --frozen-scene-xml "$SCENE" \
        --tag "$TAG" \
        --lift-target-z-above-init "$LIFT_DELTA" --lift-delta-z "$LIFT_DELTA" \
        --open-finger-from-keyframe \
        --num-envs "$NUM_ENVS" \
        --total-timesteps "$TOTAL_STEPS" \
        --axial-slip-weight "$W" \
        --axial-slip-rate-deadband "$DEADBAND" \
        "${NOISE_ARG[@]}" \
        >>"$DEST/train.log" 2>&1 &
    TRAIN_PID=$!

    # Pre-flight: the trainer dumps config.yaml seconds after launch and long before it starts
    # stepping, so a divergent run can be killed while it is still free. Anything intended has
    # to be declared with --allow, which puts the intended delta in the launcher where it is
    # reviewable, instead of in a config file nobody reads until the run is over.
    NEW_RUN="$ROOT/results/rl/$(ls -t "$ROOT/results/rl" | grep -m1 "$TAG" || true)"
    sleep 20
    NEW_RUN="$(ls -dt "$ROOT/results/rl/"*"$TAG" 2>/dev/null | head -1)"
    if [[ -n "$NEW_RUN" ]]; then
      if ! uv run python scripts/assert_config_parity.py \
             --run "$NEW_RUN" --reference "$REF_RUN" \
             --allow env.axial_slip_weight --allow env.axial_slip_rate_deadband \
             --allow ppo.init_noise_std 2>&1 | tee -a "$QLOG" | grep -q "^\[parity\] OK"; then
        say "ABORT $W: config parity check failed — killing $TRAIN_PID"
        kill "$TRAIN_PID" 2>/dev/null; wait "$TRAIN_PID" 2>/dev/null
        rm -rf "$WARP_CACHE_PATH"; wait_for_gpu; RC=99; break
      fi
    fi

    wait "$TRAIN_PID"
    RC=$?
    rm -rf "$WARP_CACHE_PATH"
    [[ $RC -eq 0 ]] && break
    grep -q "contains NaN values" "$DEST/train.log" || { say "FAIL $W (rc=$RC, not a NaN) — no retry"; break; }
    wait_for_gpu
  done

  if [[ $RC -eq 0 ]]; then
    date -Is > "$DEST/.DONE"; say "DONE $W"
  else
    say "FAIL $W (rc=$RC) — tail:"; tail -5 "$DEST/train.log" | tee -a "$QLOG"
  fi
  wait_for_gpu
done

say "r6 sweep finished"
