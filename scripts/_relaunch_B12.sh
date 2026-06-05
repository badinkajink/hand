#!/bin/bash
# NaN-resilient relaunch of B12 (smoothness finetune of B10): retry on nonzero exit up to 4x.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
B10="$ROOT/results/rl/20260604-1642-policyB_holdonlyws_repro/tensorboard/model_541.pt"
LOG="$ROOT/handoff_iter2_B12_smooth_train.log"
for attempt in 1 2 3 4; do
  echo "[B12] attempt $attempt $(date '+%T')" >> "$LOG"
  WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
    --morphology-run $MORPH --object-body-name screwdriver_medium \
    --num-envs 3072 --total-timesteps 40000000 --init-actor-checkpoint "$B10" \
    --episode-length-s 5.0 --lift-target-z-above-init 0.10 --lift-delta-z 0.10 \
    --finger-residual-scale 0.5 --finger-close-easing ease_out_quad \
    --object-xy-drift-weight=-3.0 --object-orientation-drift-weight=0.0 --finger-drift-weight=-0.3 \
    --contact-gate-stability-rewards --enable-lift-terminations --lift-phase-start-step 50 \
    --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0 --term-tip-lost-steps 10 --term-finger-slip 100.0 \
    --finger-residual-active-from-step 35 --contact-min-weight 15.0 \
    --enable-floor-proximity-termination --object-min-z 0.05 --floor-proximity-phase-start-step 50 \
    --lateral-drift-weight=-8.0 --lateral-drift-deadband 0.01 --lateral-drift-power 2.0 \
    --object-ang-acc-phase-start-step 50 --init-noise-std 0.05 --no-wandb \
    --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 \
    --reorient-start-step 50 --target-axis-progress-weight 300.0 --target-axis-alpha-curriculum-iters 0 \
    --action-rate-weight=-0.1 --action-rate-weight-final=-0.4 \
    --object-ang-acc-weight=-0.05 --object-ang-acc-weight-final=-0.25 \
    --smoothness-curriculum-start-iter 40 --smoothness-curriculum-iters 150 \
    --tag policyB_handoff_B12_smooth >> "$LOG" 2>&1
  rc=$?
  echo "[B12] attempt $attempt exited rc=$rc $(date '+%T')" >> "$LOG"
  # rc 0 = finished cleanly; if a model_541 exists we're done regardless.
  D=$(ls -dt "$ROOT"/results/rl/*policyB_handoff_B12_smooth 2>/dev/null | head -1)
  [ -f "$D/tensorboard/model_541.pt" ] && { echo "[B12] model_541 present, done." >> "$LOG"; break; }
  [ $rc -eq 0 ] && break
  sleep 10
done
