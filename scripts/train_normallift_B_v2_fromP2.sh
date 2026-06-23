#!/bin/bash
# Normal-lift Policy B v2 — the handoff fix, properly resourced.
#
# DIAGNOSIS (2026-06-03): the A->B seam drop is an OBSERVATION-DISCONTINUITY shock,
# not a grip problem. Skip-lift-trained B (incl. P1 handoff-DR and P3 statebank) sees
# a lift-command phase / ref_object_pose at the seam that it never saw in training, and
# collapses the grip within 3-5 steps. Training B in the NORMAL-LIFT deploy env (residual
# gated to activate at the handoff step) removes that shock: the 15M/1024-env normallift B
# held ~10-15 steps past the seam before dropping (undertrained), vs instant collapse.
#
# This run combines the two things that each worked:
#   (a) normal-lift training (fixes the seam shock), and
#   (b) warmstart from P2 = the new best reorienter (held-cos 0.988, jerk 25.8, the
#       signed+critic recipe + lateral-drift -8 which acts as a smoothing regularizer).
# Critic warmstart ON (default). lateral-drift -8 kept so the warmstarted critic stays
# valid (P2's reward included it). Scaled to 40M ts / 3072 envs for full convergence.
#
# Detached (gotcha #5):
#   nohup setsid bash scripts/train_normallift_B_v2_fromP2.sh > normallift_v2.bg.log 2>&1 < /dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
WARMSTART="$ROOT/results/rl/b04_20260603-1746-policyB_p2_lateral_only/tensorboard/model_541.pt"
TOTAL_TS=${TOTAL_TS:-40000000}
NUM_ENVS=${NUM_ENVS:-3072}
TAG="policyB_normallift_v2_fromP2"
LOG="$ROOT/normallift_v2_train.log"

echo "[nl2] training normal-lift B v2 (warmstart P2 model_541, real lift to 0.10, 40M/3072)..."
WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
  --morphology-run "$MORPH" --object-body-name screwdriver_medium \
  --num-envs "$NUM_ENVS" --total-timesteps "$TOTAL_TS" --init-actor-checkpoint "$WARMSTART" \
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
  --lateral-drift-weight=-8.0 --lateral-drift-deadband 0.01 --lateral-drift-power 2.0 \
  --init-noise-std 0.05 --no-wandb --tag "$TAG" \
  > "$LOG" 2>&1
echo "[nl2] training done (rc=$?)"

DIR=$(ls -dt "$ROOT/results/rl/"*-"$TAG" 2>/dev/null | head -1)
CK=$(ls -t "$DIR"/tensorboard/model_*.pt 2>/dev/null | head -1)
echo "[nl2] rendering continuous handoff with $CK ..."
WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_demo_handoff_continuous.py" \
  --policy-b "$CK" \
  --output "$ROOT/docs/rl/videos/reorient/handoff_seamless_v2.mp4" \
  --handoff-step 45 --total-steps 260 \
  >> "$LOG" 2>&1
echo "[nl2] DONE. video: docs/rl/videos/reorient/handoff_seamless_v2.mp4"
echo "[nl2] grep 'min object' $LOG to confirm it stayed aloft post-handoff (want min-z > 0.05)."
