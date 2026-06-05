#!/bin/bash
# Branch B — un-freeze Policy A toward B10's initiation set.
#
# Diagnosis (RESEARCH_STATE.md + this session): the A->B10 seam drops because at
# the handoff (~step 35) B10's OBJECT state already matches A's flat delivery, but
# A's GRIP (finger qpos) differs from B10's holding grip (up to 0.16 rad/joint;
# A scores 0.347 vs B10-self 1.0 at tol 0.05). Even handing off early still drops.
# So: keep B10 frozen, fine-tune A (warmstart) in its DEPLOY env (normal lift to
# 0.10, 65-d A space) with a seam-gated dense reward (handoff_target_proximity)
# pulling A's delivered grip onto B10's recorded step-35 grip. A's native
# grasp/lift/centering rewards keep the object pose good (clean separation).
#
# Detached: nohup setsid bash scripts/train_handoff_branchB_unfreezeA.sh > branchB_unfreezeA.log 2>&1 </dev/null & disown
# SMOKE=1 -> 1M ts (plumbing/NaN check).  WEIGHT env overrides the proximity weight.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
A_CKPT="${A_CKPT:-$ROOT/results/rl/20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt}"
BANK="${BANK:-$ROOT/results/rl/b10_initiation_bank_s35.npz}"
WEIGHT="${WEIGHT:-4.0}"
TOTAL_TS=${TOTAL_TS:-15000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-policyA_unfreezeA_gripw${WEIGHT}}"
for f in "$A_CKPT" "$BANK" "$ROOT/$MORPH/best_rollout.npz"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

ARGS=(
  --morphology-run "$MORPH" --object-body-name screwdriver_medium
  --num-envs 3072 --total-timesteps "$TOTAL_TS"
  --init-actor-checkpoint "$A_CKPT" --init-noise-std 0.05
  # deploy-matched lift (B10/continuous-handoff env lifts the cylinder to 0.10):
  --episode-length-s 5.0 --lift-target-z-above-init 0.10 --lift-delta-z 0.10
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad
  # A's grasp discipline (keep the working grip healthy):
  --object-xy-drift-weight=-30.0 --object-orientation-drift-weight=-20.0 --finger-drift-weight=-10.0
  --contact-gate-stability-rewards --contact-min-weight 15.0
  --cube-spawn-x-jitter 0.003 --cube-spawn-y-jitter 0.003 --cube-spawn-yaw-jitter 0.26
  --enable-lift-terminations --lift-phase-start-step 40
  --term-object-slip-xy 0.015 --term-object-slip-yaw 0.5 --term-object-drop 0.02
  --term-tip-lost-steps 3 --term-finger-slip 0.3
  # Branch-B grip-proximity reward (NO --enable-target-axis-reward => 65-d A space):
  --handoff-target-bank "$BANK" --handoff-target-weight "$WEIGHT"
  --handoff-target-seam-lo 33 --handoff-target-seam-hi 37 --handoff-target-qpos-tol 0.05
  --tag "$TAG" --no-wandb
)
c=$(mktemp -d)
echo "[branchB] TAG=$TAG ts=$TOTAL_TS weight=$WEIGHT bank=$BANK WARP_CACHE=$c"
WARP_CACHE_PATH="$c" MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" "${ARGS[@]}"
echo "[branchB] DONE rc=$?"
