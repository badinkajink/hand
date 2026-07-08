#!/bin/bash
# Train a normal-lift-compatible Policy B (so the A->B handoff is seamless, no reset),
# then auto-render the continuous handoff video with it.
#
# B is finetuned from the signed+critic reorient policy but in the NORMAL-LIFT env
# (skip_lift_phase=False, real lift to 0.10): the cylinder spawns flat, the scripted
# grasp+palm lift it, and reorient rewards gate AFTER the lift (reorient_start_step=35).
# So B learns to hold through the real lift (closing the distribution gap that made it
# drop the cylinder at handoff) then reorient. critic warmstart ON (default).
#
# Detached:
#   nohup setsid bash scripts/train_normallift_B_and_handoff.sh > normallift.bg.log 2>&1 < /dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
WARMSTART="$ROOT/results/rl/b03_20260602-1636-policyB_abl_signed/tensorboard/model_405.pt"
TOTAL_TS=${TOTAL_TS:-15000000}
TAG="policyB_normallift"
LOG="$ROOT/normallift_train.log"

echo "[nl] training normal-lift B (warmstart signed+critic, real lift to 0.10)..."
WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
  --morphology-run "$MORPH" --object-body-name screwdriver_medium \
  --num-envs 1024 --total-timesteps "$TOTAL_TS" --init-actor-checkpoint "$WARMSTART" \
  --episode-length-s 5.0 \
  --lift-target-z-above-init 0.10 --lift-delta-z 0.10 \
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad \
  --object-xy-drift-weight=-3.0 --object-orientation-drift-weight=0.0 --finger-drift-weight=-0.3 \
  --contact-gate-stability-rewards \
  --enable-lift-terminations --lift-phase-start-step 35 \
  --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0 --term-tip-lost-steps 10 --term-finger-slip 100.0 \
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 --reorient-start-step 35 \
  --finger-residual-active-from-step 35 \
  --contact-min-weight 15.0 --target-axis-progress-weight 300.0 --target-axis-alpha-curriculum-iters 0 \
  --enable-floor-proximity-termination --object-min-z 0.05 --floor-proximity-phase-start-step 35 \
  --action-rate-weight=-0.1 --object-ang-acc-weight=-0.05 --object-ang-acc-phase-start-step 35 \
  --init-noise-std 0.05 --no-wandb --tag "$TAG" \
  > "$LOG" 2>&1
echo "[nl] training done (rc=$?)"

DIR=$(ls -dt "$ROOT/results/rl/"*-"$TAG" 2>/dev/null | head -1)
CK=$(ls -t "$DIR"/tensorboard/model_*.pt 2>/dev/null | head -1)
echo "[nl] rendering continuous handoff with $CK ..."
# This B is normal-lift trained, so a SINGLE-env continuous rollout (no reset) should hold:
# Policy A lifts, then this B takes over the same physics state and reorients.
WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_demo_handoff_continuous.py" \
  --policy-b "$CK" \
  --output "$ROOT/docs/rl/videos/reorient/handoff_continuous_seamless.mp4" \
  --handoff-step 45 --total-steps 260 \
  >> "$LOG" 2>&1
echo "[nl] DONE. video: docs/rl/videos/reorient/handoff_continuous_seamless.mp4"
echo "[nl] check min-z line in $LOG to confirm it stayed aloft post-handoff."
