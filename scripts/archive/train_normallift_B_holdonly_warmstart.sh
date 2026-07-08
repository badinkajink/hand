#!/bin/bash
# Handoff fix: normal-lift Policy B grace-window finetune, WARMSTARTED FROM THE HOLD-ONLY
# CONTROL instead of the skip-lift reorienter (B4 / p2_lateral).
#
# WHY (the v3b verdict + the branch analysis): branch A (make B robust to A's delivery by
# training it in the normal-lift env with a grace window) hit a ceiling — v3b (B9) ran to
# completion but collapsed to a degenerate plateau: B holds during the grace window, then
# DROPS the instant the reorient phase engages. Root cause: it warmstarts the *skip-lift*
# reorienter (B4), which is OUT OF DISTRIBUTION on the normal-lift delivery, so the
# grace->reorient transition starts OOD and never crosses into a working reorient.
#
# Branch D confirmed the same thing from the other side: B4's *critic* reads +4.75 at step 0
# then collapses to ~-5000 on the normal-lift delivery — its value function is garbage there.
#
# THE FIX (this script): warmstart from the v3 HOLD-ONLY control
# (20260603-2349-policyB_normallift_v3_holdonly/model_541), which already PROVED it survives
# A's scripted-lift -> takeover in the normal-lift env (tip_lost humped to ~44 then recovered
# to ~1-4). So the grace->reorient transition begins IN-DISTRIBUTION. The hold-only policy is
# 65-dim (no target_axis_misalign obs / no reorient); this run is 66-dim with reorient on, so
# the warmstart is a PARTIAL load: rl_train_cube.py zero-inits the new obs column for both the
# actor and the critic (gotcha #7), keeping std_param init for exploration.
#
# Everything else = the v3b grace-window config (B takes over at sim step 35; terminations +
# reorient engage at step 50). Two parallel variants for NaN-resilience (gotcha #2), staggered,
# per-process WARP_CACHE. Registry: B10 (repro) / B11 (soft).
#
# Detached:
#   nohup setsid bash scripts/train_normallift_B_holdonly_warmstart.sh > holdonlyws.bg.log 2>&1 < /dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
WARMSTART="$ROOT/results/rl/bx_20260603-2349-policyB_normallift_v3_holdonly/tensorboard/model_541.pt"
TOTAL_TS=${TOTAL_TS:-40000000}
NUM_ENVS=${NUM_ENVS:-3072}
GRACE_TERM=50   # terminations + reorient engage here; B takes over (residual) at step 35

if [ ! -f "$WARMSTART" ]; then echo "MISSING warmstart: $WARMSTART" >&2; exit 1; fi

common_args() {
  local resid=$1
  echo "--morphology-run $MORPH --object-body-name screwdriver_medium \
    --num-envs $NUM_ENVS --total-timesteps $TOTAL_TS --init-actor-checkpoint $WARMSTART \
    --episode-length-s 5.0 \
    --lift-target-z-above-init 0.10 --lift-delta-z 0.10 \
    --finger-residual-scale $resid --finger-close-easing ease_out_quad \
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

# Run R (B10): repro grace, finger-residual-scale 0.5, hard reorient onset — primary candidate.
LOGR="$ROOT/holdonlyws_repro_train.log"
echo "[holdonlyws] launching R/B10 (repro grace, warmstart hold-only)..."
nohup setsid env WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl \
  uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" $(common_args 0.5) \
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 \
  --reorient-start-step $GRACE_TERM --target-axis-progress-weight 300.0 --target-axis-alpha-curriculum-iters 0 \
  --tag policyB_holdonlyws_repro > "$LOGR" 2>&1 < /dev/null & disown
echo "[holdonlyws] R launched -> $LOGR"

sleep 90  # stagger so kernel compiles don't collide (gotcha #2)

# Run S (B11): soft onset, finger-residual-scale 0.4 + basin-width curriculum alpha 0.5->4.0/150it.
LOGS="$ROOT/holdonlyws_soft_train.log"
echo "[holdonlyws] launching S/B11 (soft grace, warmstart hold-only)..."
nohup setsid env WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl \
  uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" $(common_args 0.4) \
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 \
  --target-axis-alpha-start 0.5 --target-axis-alpha-curriculum-iters 150 \
  --reorient-start-step $GRACE_TERM --target-axis-progress-weight 300.0 \
  --tag policyB_holdonlyws_soft > "$LOGS" 2>&1 < /dev/null & disown
echo "[holdonlyws] S launched -> $LOGS"
echo "[holdonlyws] both launched. Want: post-handoff min-z > 0.05 at held-cos near B4's 0.988."
echo "[holdonlyws] Eval: rl_eval_reorient_metrics.py (held-cos/drop) + rl_demo_handoff_continuous.py --blend-steps 8 (min-z>0.05)."
