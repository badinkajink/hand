#!/usr/bin/env bash
# r8: train the three-finger CHUCK on the 25 mm-proximal opposed hand.
#
# What is different from r7, which failed. r7 asked a policy to brace with a thumb that was
# STOWED at yaw 0.6 on the 50 mm-proximal hand, and the reward read 0.0000 for 339 iterations.
# Two things have changed since, both measured (docs/rl/perp_topology.md, 2026-08-19):
#
#   * At that geometry, touching WAS dropping. Six scripted engages put 8-11 N of thumb on the
#     shaft and all six ejected it, because after the swing the pair sits at +-90 deg and its
#     two normals are collinear -- nothing reacts the thumb's push but friction. PPO declining
#     to touch was correct behaviour. On the 25 mm-proximal hand with palm-frame hold targets
#     the same engage HOLDS: 100% of the window vertical, three fingers loaded 100%.
#   * The thumb can now reach the brace pose INSIDE the policy's own authority. The scripted
#     chuck's thumb sits at (yaw, mcp, pip) = (-0.005, 1.661, -1.219) against the `closed_manual`
#     set-point (0.333, 2.019, -1.281): deltas -0.34 / -0.36 / +0.06 rad, all inside the +-0.5
#     rad residual. On r7's hand the same pose was outside it.
#
# So this run is a genuine test of the brace reward rather than a repeat. The tell is
# `Episode_Reward/thumb_brace_force`: if it stays at 0.0000 the way it did in r7, the reward
# cannot find the chuck on its own and the next step is imitation from the scripted
# demonstration (the reference trajectory recorded at results/phase1/perp_thumb_engage/
# sp25_manual/best_rollout.npz IS that demonstration).
#
# Scene contract, all three parts load-bearing:
#   frozen scene       -- base scenes let the mounts slide under load and the hand absorbs the
#                         thumb's push; the same maneuver reads 100% base and 0% frozen.
#   keyframe open_manual -- the user's authored grasp, pair seated near the shaft's COM.
#   closed_manual      -- the LIGHT symmetric pinch (0.2 mm commanded depth at x = +23 mm). The
#                         CEM grip clamps at ~21 N, outside the 4-9 N window the pitch needs.
#
#   bash scripts/train_perp_sp25_chuck.sh
#   WEIGHTS="25" bash scripts/train_perp_sp25_chuck.sh
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

MORPH_RUN="${MORPH_RUN:-$ROOT/results/phase1/perp_thumb_engage/sp25_manual}"
SCENE="${SCENE:-$MORPH_RUN/frozen_scene.xml}"
WEIGHTS="${WEIGHTS:-8 25}"
BRACE_THRESH="${BRACE_THRESH:-0.7}"     # same gate the grip catch uses -- protects the swing
BRACE_MAXN="${BRACE_MAXN:-6.0}"         # the scripted chuck runs the thumb at 20 N; 6 saturates early
REF_RUN="${REF_RUN:-$ROOT/results/rl/20260731-1300-perp_single_r4}"
LIFT_DELTA="${LIFT_DELTA:-0.14}"
NUM_ENVS="${NUM_ENVS:-3072}"
TOTAL_STEPS="${TOTAL_STEPS:-25000000}"

LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/perp_sp25_chuck_$(date +%Y%m%d-%H%M).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

# The GPU is single and shared. Wait for whatever is on it -- a stomped run is two experiments
# lost, not one -- and after killing a Warp process wait for memory to actually drain.
wait_for_gpu() {
  for _ in $(seq 1 "${1:-360}"); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [[ "$used" -lt 2500 ]] && return 0
    sleep 20
  done
  say "WARN: GPU still at ${used} MiB; NOT launching"
  return 1
}

[[ -f "$SCENE" ]] || { say "FATAL: scene missing: $SCENE"; exit 1; }
say "r8 sp25 chuck: weights [$WEIGHTS] gate cos>=$BRACE_THRESH sat ${BRACE_MAXN}N -> $QLOG"
say "waiting for the GPU to free before the first launch"
wait_for_gpu 540 || exit 1

for W in $WEIGHTS; do
  TAG="perp_sp25_chuck_w${W}"
  DEST="$ROOT/results/rl/sp25_chuck_${W}"; mkdir -p "$DEST"
  [[ -f "$DEST/.DONE" ]] && { say "SKIP $W (.DONE)"; continue; }

  RC=1
  for NOISE in 0.02 0.01; do
    say "=== train $TAG at init_noise_std=$NOISE ==="
    export WARP_CACHE_PATH="$(mktemp -d)"
    MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/rl_train_cube.py \
        --recipe perp_single \
        --morphology-run "$MORPH_RUN" \
        --frozen-scene-xml "$SCENE" \
        --closed-ctrl-from-keyframe closed_manual \
        --open-finger-from-keyframe \
        --tag "$TAG" \
        --lift-target-z-above-init "$LIFT_DELTA" --lift-delta-z "$LIFT_DELTA" \
        --num-envs "$NUM_ENVS" --total-timesteps "$TOTAL_STEPS" \
        --thumb-brace-weight "$W" \
        --thumb-brace-align-thresh "$BRACE_THRESH" \
        --thumb-brace-max-force "$BRACE_MAXN" \
        --init-noise-std "$NOISE" \
        --watchdog-collapse-z 0.030 --watchdog-from-iter 50 \
        --watchdog-sentinel "$DEST/train.log.COLLAPSED" \
        >>"$DEST/train.log" 2>&1 &
    TRAIN_PID=$!

    sleep 25
    NEW_RUN="$(ls -dt "$ROOT/results/rl/"*"$TAG" 2>/dev/null | head -1)"
    if [[ -n "$NEW_RUN" ]]; then
      if ! uv run python scripts/assert_config_parity.py \
             --run "$NEW_RUN" --reference "$REF_RUN" \
             --allow env.thumb_brace_weight --allow env.thumb_brace_align_thresh \
             --allow env.thumb_brace_max_force --allow ppo.init_noise_std \
             --allow env.frozen_scene_xml --allow env.keyframe_name \
             --allow env.foundational_run_dir --allow env.finger_default_ctrl \
             2>&1 | tee -a "$QLOG" | grep -q "^\[parity\] OK"; then
        say "ABORT $W: config parity failed — killing $TRAIN_PID"
        kill "$TRAIN_PID" 2>/dev/null; wait "$TRAIN_PID" 2>/dev/null
        rm -rf "$WARP_CACHE_PATH"; RC=99; wait_for_gpu; break
      fi
    fi

    wait "$TRAIN_PID"; RC=$?
    rm -rf "$WARP_CACHE_PATH"
    [[ $RC -eq 0 ]] && break
    grep -q "contains NaN values" "$DEST/train.log" || { say "FAIL $W (rc=$RC, not a NaN)"; break; }
    wait_for_gpu
  done

  if [[ $RC -eq 0 ]]; then date -Is > "$DEST/.DONE"; say "DONE $W"
  else say "FAIL $W (rc=$RC)"; tail -5 "$DEST/train.log" | tee -a "$QLOG"; fi
  wait_for_gpu
done
say "r8 sp25 chuck queue finished"
