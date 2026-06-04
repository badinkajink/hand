#!/bin/bash
# Normal-lift Policy B v3 — fix the training COLLAPSE that broke v2.
#
# v2 (warmstart P2, normallift) collapsed: standalone held-cos 0.029, drop 100%, reward
# fell 12->3 at iter ~5. Cause: at step 35 EVERYTHING fired at once — B's residual
# activates, lift+floor terminations engage, AND the full reorient reward (100 + 300
# progress). The warmstart actor (skip-lift prior) is OOD on the post-scripted-lift
# state, fumbles, terminations kill the episode short -> reward collapse -> never learns.
#
# v3 gives B a GRACE WINDOW: it takes over at step 35 but only has to HOLD until step 50
# (terminations + reorient pressure off). Then the pressure turns on. Launches two runs:
#   G (grace)     : grace window + reorient (the candidate fix).
#   H (hold-only) : grace window, reorient DISABLED — control to test if B can even
#                   survive the scripted-lift->B takeover at all (isolates the cause).
# Both warmstart P2, 40M/3072, per-process WARP_CACHE, staggered (gotcha #2).
#
# Detached:
#   nohup setsid bash scripts/train_normallift_B_v3_gracewindow.sh > normallift_v3.bg.log 2>&1 < /dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
WARMSTART="$ROOT/results/rl/20260603-1746-policyB_p2_lateral_only/tensorboard/model_541.pt"
TOTAL_TS=${TOTAL_TS:-40000000}
NUM_ENVS=${NUM_ENVS:-3072}
GRACE_TERM=50   # terminations + reorient engage here; B takes over (residual) at 35

common_args() {
  echo "--morphology-run $MORPH --object-body-name screwdriver_medium \
    --num-envs $NUM_ENVS --total-timesteps $TOTAL_TS --init-actor-checkpoint $WARMSTART \
    --episode-length-s 5.0 \
    --lift-target-z-above-init 0.10 --lift-delta-z 0.10 \
    --finger-residual-scale 0.5 --finger-close-easing ease_out_quad \
    --object-xy-drift-weight=-3.0 --object-orientation-drift-weight=0.0 --finger-drift-weight=-0.3 \
    --contact-gate-stability-rewards \
    --enable-lift-terminations --lift-phase-start-step $GRACE_TERM \
    --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0 --term-tip-lost-steps 10 --term-finger-slip 100.0 \
    --finger-residual-active-from-step 35 \
    --contact-min-weight 15.0 \
    --enable-floor-proximity-termination --object-min-z 0.05 --floor-proximity-phase-start-step $GRACE_TERM \
    --action-rate-weight=-0.1 --object-ang-acc-weight=-0.05 --object-ang-acc-phase-start-step $GRACE_TERM \
    --lateral-drift-weight=-8.0 --lateral-drift-deadband 0.01 --lateral-drift-power 2.0 \
    --init-noise-std 0.05 --no-wandb"
}

# Run G: grace window + reorient (the candidate fix)
LOGG="$ROOT/normallift_v3_grace_train.log"
echo "[v3] launching G (grace+reorient)..."
nohup setsid env WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl \
  uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" $(common_args) \
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 \
  --reorient-start-step $GRACE_TERM --target-axis-progress-weight 300.0 --target-axis-alpha-curriculum-iters 0 \
  --tag policyB_normallift_v3_grace > "$LOGG" 2>&1 < /dev/null & disown
echo "[v3] G pid family launched -> $LOGG"

sleep 90  # stagger so kernel compiles don't collide (gotcha #2)

# Run H: hold-only control (no reorient)
LOGH="$ROOT/normallift_v3_holdonly_train.log"
echo "[v3] launching H (hold-only control, no reorient)..."
nohup setsid env WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl \
  uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" $(common_args) \
  --tag policyB_normallift_v3_holdonly > "$LOGH" 2>&1 < /dev/null & disown
echo "[v3] H pid family launched -> $LOGH"
echo "[v3] both launched. Evaluate with rl_eval_reorient_metrics.py (held-cos, drop) + handoff demo."
