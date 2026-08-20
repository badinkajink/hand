#!/usr/bin/env bash
# r9: reward the whole-hand HOLD CONFIGURATION, not the thumb's contact force.
#
# r8 (scripts/train_perp_sp25_chuck.sh) is the thumb-brace reward on the hand where the scripted
# chuck holds. If its `Episode_Reward/thumb_brace_force` ends flat at 0.0000 the way r7's did,
# the answer is not another weight: the reward is asking for a motion whose immediate consequence
# is a drop. A thumb press against a pair still sitting at +-90 deg ejects the shaft, measured six
# times out of six, so a policy that keeps the thumb stowed is behaving correctly. What the hold
# needs is all three contacts moving TOGETHER, and no thumb-only term can express that.
#
# `chuck_pose_match` states the target as the object-frame position of all three fingertips,
# recorded off the scripted maneuver (results/phase1/perp_thumb_engage/sp25_manual/
# chuck_pose.npz), reduced over the WORST-placed finger so two correct fingers cannot pay for a
# third that never moved, and gated on alignment so it cannot buy a clamp during the swing.
#
# THE BINDING CONSTRAINT IS REACH, NOT REWARD. Measured off the demo's own hold (not from an
# IK guess): the chuck sits 1.296 rad away at thumb_pip, 0.654 at middle_pip and 0.559 at
# thumb_mcp from the closed set-point, while the policy's action is a +-0.5 rad residual around
# that set-point. Three joints are outside the budget, so the pose is not unexplored, it is
# unreachable, and r7/r8's flat 0.0000 is what an unreachable target looks like from inside the
# reward table. That is why this sweep moves --finger-residual-scale and holds the weight fixed.
# It breaks train/deploy parity (gotcha #13) on purpose, so evaluate with the SAME scale:
#   scripts/policy_eval_suite.py --finger-residual-scale <same value>
#
# Static, not time-indexed, deliberately: the learned rotation takes ~4x longer than the gravity
# swing the demonstration was recorded from, so the existing imitation term -- which samples a
# trajectory at (step - onset) * dt -- would aim at the wrong phase of the manoeuvre. Over the
# scripted hold the configuration is static to 1.46 mm per axis, so there is nothing to index.
#
#   bash scripts/train_perp_sp25_chuckpose.sh
#   RESIDUALS="1.5" bash scripts/train_perp_sp25_chuckpose.sh
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

MORPH_RUN="${MORPH_RUN:-$ROOT/results/phase1/perp_thumb_engage/sp25_manual}"
SCENE="${SCENE:-$MORPH_RUN/frozen_scene.xml}"
POSE="${POSE:-$MORPH_RUN/chuck_pose.npz}"
WEIGHT="${WEIGHT:-30}"
# The sweep axis is the RESIDUAL SCALE, not the reward weight. Measured on the demo's own hold,
# the chuck needs joint excursions of 1.296 rad (thumb_pip), 0.654 (middle_pip) and 0.559
# (thumb_mcp) away from the closed set-point, against a policy whose action is a +-0.5 rad
# residual around it. Three joints are outside the budget, so at 0.5 the pose is not merely
# unexplored -- it is UNREACHABLE, and no weight on any term can buy it. 0.5 is the control that
# proves the point; 1.5 is the first scale that covers the excursion.
RESIDUALS="${RESIDUALS:-0.5 1.5}"
POSE_THRESH="${POSE_THRESH:-0.7}"
POSE_ALPHA="${POSE_ALPHA:-2000}"
REF_RUN="${REF_RUN:-$ROOT/results/rl/20260731-1300-perp_single_r4}"
LIFT_DELTA="${LIFT_DELTA:-0.14}"
NUM_ENVS="${NUM_ENVS:-3072}"
TOTAL_STEPS="${TOTAL_STEPS:-25000000}"

LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/perp_sp25_chuckpose_$(date +%Y%m%d-%H%M).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

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
[[ -f "$POSE"  ]] || { say "FATAL: chuck pose missing: $POSE (probe --save-chuck-pose)"; exit 1; }
say "r9 chuck-pose: weight $WEIGHT, residual scales [$RESIDUALS], gate cos>=$POSE_THRESH -> $QLOG"
wait_for_gpu 720 || exit 1

for FRS in $RESIDUALS; do
  W="$WEIGHT"
  TAG="perp_sp25_chuckpose_frs${FRS}"
  DEST="$ROOT/results/rl/sp25_chuckpose_frs${FRS}"; mkdir -p "$DEST"
  [[ -f "$DEST/.DONE" ]] && { say "SKIP $FRS (.DONE)"; continue; }

  RC=1
  for NOISE in 0.02 0.01; do
    say "=== train $TAG (residual $FRS) at init_noise_std=$NOISE ==="
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
        --chuck-pose-npz "$POSE" \
        --chuck-pose-weight "$W" \
        --chuck-pose-align-thresh "$POSE_THRESH" \
        --chuck-pose-alpha "$POSE_ALPHA" \
        --finger-residual-scale "$FRS" \
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
             --allow env.chuck_pose_weight --allow env.chuck_pose_align_thresh \
             --allow env.chuck_pose_alpha --allow ppo.init_noise_std \
             --allow env.chuck_pose_npz --allow env.finger_residual_scale \
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

  if [[ $RC -eq 0 ]]; then date -Is > "$DEST/.DONE"; say "DONE $FRS"
  else say "FAIL $FRS (rc=$RC)"; tail -5 "$DEST/train.log" | tee -a "$QLOG"; fi
  wait_for_gpu
done
say "r9 chuck-pose queue finished"
