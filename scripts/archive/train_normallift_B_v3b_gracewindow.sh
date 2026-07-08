#!/bin/bash
# Normal-lift Policy B v3b — RUN THE GRACE WINDOW TO COMPLETION (NaN-resilient relaunch).
#
# v3 grace (train_normallift_B_v3_gracewindow.sh) was the candidate fix and it WORKED:
# reward stayed healthy (~10, flat) for 60 iters — i.e. the grace window prevented the
# v2 collapse (v2 had reward fall 12->3 because step-35 fired residual+terminations+
# reorient simultaneously). v3 grace then NaN-CRASHED at iter ~60 (only model_50 saved,
# undertrained). The NaN is a transient env-level physics blowup (a single warp env going
# unstable); rsl_rl's check_nan raises on ANY env NaN and kills the whole run — there is no
# retry. So the approach is sound, it just died to bad luck at 60/750.
#
# v3b = relaunch the SAME grace config to completion, NaN-resilient by running TWO parallel
# variants (so a stochastic single-env NaN doesn't waste the whole 70-min effort):
#   R (repro) : byte-identical to v3 grace (finger-residual-scale 0.5).
#   S (soft)  : finger-residual-scale 0.4 + basin-width curriculum (alpha 0.5->4.0 over 150
#               iters) so the reorient reward turns on GENTLY at the seam — lowers both the
#               physics-blowup (NaN) risk and the step-50 OOD shock; an ablation on top.
# Both warmstart P2 (the best reorienter, 0.988 held-cos), 40M/3072, per-process WARP_CACHE,
# staggered (gotcha #2). B takes over (residual) at sim step 35; terminations + reorient
# engage at step 50 (the grace window is 35..50).
#
# Detached:
#   nohup setsid bash scripts/train_normallift_B_v3b_gracewindow.sh > normallift_v3b.bg.log 2>&1 < /dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
WARMSTART="$ROOT/results/rl/b04_20260603-1746-policyB_p2_lateral_only/tensorboard/model_541.pt"
TOTAL_TS=${TOTAL_TS:-40000000}
NUM_ENVS=${NUM_ENVS:-3072}
GRACE_TERM=50   # terminations + reorient engage here; B takes over (residual) at 35

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

# Run R: repro (finger-residual-scale 0.5, hard reorient onset) — the primary candidate.
LOGR="$ROOT/normallift_v3b_repro_train.log"
echo "[v3b] launching R (repro grace)..."
nohup setsid env WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl \
  uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" $(common_args 0.5) \
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 \
  --reorient-start-step $GRACE_TERM --target-axis-progress-weight 300.0 --target-axis-alpha-curriculum-iters 0 \
  --tag policyB_normallift_v3b_repro > "$LOGR" 2>&1 < /dev/null & disown
echo "[v3b] R launched -> $LOGR"

sleep 90  # stagger so kernel compiles don't collide (gotcha #2)

# Run S: soft onset (residual 0.4 + basin-width curriculum alpha 0.5->4.0 over 150 iters).
LOGS="$ROOT/normallift_v3b_soft_train.log"
echo "[v3b] launching S (soft grace)..."
nohup setsid env WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl \
  uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" $(common_args 0.4) \
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 \
  --target-axis-alpha-start 0.5 --target-axis-alpha-curriculum-iters 150 \
  --reorient-start-step $GRACE_TERM --target-axis-progress-weight 300.0 \
  --tag policyB_normallift_v3b_soft > "$LOGS" 2>&1 < /dev/null & disown
echo "[v3b] S launched -> $LOGS"
echo "[v3b] both launched. Eval: rl_eval_reorient_metrics.py (held-cos/drop) + rl_demo_handoff_continuous.py (min-z>0.05)."
