#!/bin/bash
# Handoff iteration 2 — tame the VIOLENT transition that B10 shows, keep the tilt B11 lost.
#
# CONTEXT (eyeball + metrics from B10/B11, the hold-only-warmstart grace runs):
#   B10 (holdonlyws_repro, residual 0.5, HARD reorient onset alpha=4 @ step 50):
#       picks up AND reorients through the handoff (FIRST policy to do both!) — but the
#       transition looks VIOLENT (large action magnitudes at the seam / reorient onset).
#   B11 (holdonlyws_soft, residual 0.4, alpha curriculum 0.5->4 over 150 iters):
#       picks up + holds but DOES NOT TILT — the 150-iter basin curriculum was so dilute
#       the policy settled into a hold-only optimum before the reorient signal sharpened.
#   => The hold-only warmstart fixed the collapse (both survive the seam). Now: smooth B10,
#      and find a soft-but-committing onset between B10 (too hard) and B11 (too soft).
#
# Two parallel candidates (NaN-resilient, staggered, per-process WARP_CACHE; ~70 min):
#   B12 = smoothness finetune OF B10. Warmstart B10 (already reorients), then ramp the
#         action-rate + object-angular-accel penalties IN LATE (curriculum starts iter 40)
#         so it learns to tilt first, then smooths the seam — the same "learn it then make
#         it smooth" recipe that worked for the v2 smoothness finetunes (B2). Targets the
#         violence directly. Risk (gotcha #10): smoothness can re-break the tilt; gentle ramp.
#   B13 = soft-but-COMMITTING onset from the hold-only control. Same warmstart as B10/B11,
#         residual 0.5 (full authority, unlike B11's 0.4), basin curriculum alpha 0.5->4 over
#         just 40 iters (commits fast, unlike B11's 150). Eases the first few iters of the
#         reorient onset without diluting it into a hold-only optimum.
#
# Detached:
#   nohup setsid bash scripts/train_handoff_iter2.sh > handoff_iter2.bg.log 2>&1 < /dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
B10="$ROOT/results/rl/b10_20260604-1642-policyB_holdonlyws_repro/tensorboard/model_541.pt"
HOLDONLY="$ROOT/results/rl/bx_20260603-2349-policyB_normallift_v3_holdonly/tensorboard/model_541.pt"
TOTAL_TS=${TOTAL_TS:-40000000}
NUM_ENVS=${NUM_ENVS:-3072}
GRACE_TERM=50
for f in "$B10" "$HOLDONLY"; do [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }; done

common_args() {  # $1 = finger-residual-scale, $2 = init checkpoint
  echo "--morphology-run $MORPH --object-body-name screwdriver_medium \
    --num-envs $NUM_ENVS --total-timesteps $TOTAL_TS --init-actor-checkpoint $2 \
    --episode-length-s 5.0 \
    --lift-target-z-above-init 0.10 --lift-delta-z 0.10 \
    --finger-residual-scale $1 --finger-close-easing ease_out_quad \
    --object-xy-drift-weight=-3.0 --object-orientation-drift-weight=0.0 --finger-drift-weight=-0.3 \
    --contact-gate-stability-rewards \
    --enable-lift-terminations --lift-phase-start-step $GRACE_TERM \
    --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0 --term-tip-lost-steps 10 --term-finger-slip 100.0 \
    --finger-residual-active-from-step 35 \
    --contact-min-weight 15.0 \
    --enable-floor-proximity-termination --object-min-z 0.05 --floor-proximity-phase-start-step $GRACE_TERM \
    --lateral-drift-weight=-8.0 --lateral-drift-deadband 0.01 --lateral-drift-power 2.0 \
    --object-ang-acc-phase-start-step $GRACE_TERM \
    --init-noise-std 0.05 --no-wandb \
    --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 \
    --reorient-start-step $GRACE_TERM --target-axis-progress-weight 300.0"
}

# B12: smoothness finetune of B10 — ramp action-rate + ang-accel penalties in LATE.
LOG12="$ROOT/handoff_iter2_B12_smooth_train.log"
echo "[iter2] launching B12 (smoothness finetune of B10)..."
nohup setsid env WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl \
  uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" $(common_args 0.5 "$B10") \
  --target-axis-alpha-curriculum-iters 0 \
  --action-rate-weight=-0.1 --action-rate-weight-final=-0.4 \
  --object-ang-acc-weight=-0.05 --object-ang-acc-weight-final=-0.25 \
  --smoothness-curriculum-start-iter 40 --smoothness-curriculum-iters 150 \
  --tag policyB_handoff_B12_smooth > "$LOG12" 2>&1 < /dev/null & disown
echo "[iter2] B12 launched -> $LOG12"

sleep 90  # stagger (gotcha #2)

# B13: soft-but-committing onset from hold-only — residual 0.5, alpha 0.5->4 over 40 iters.
LOG13="$ROOT/handoff_iter2_B13_softcommit_train.log"
echo "[iter2] launching B13 (soft-committing from hold-only)..."
nohup setsid env WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl \
  uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" $(common_args 0.5 "$HOLDONLY") \
  --action-rate-weight=-0.1 --object-ang-acc-weight=-0.05 \
  --target-axis-alpha-start 0.5 --target-axis-alpha-curriculum-iters 40 \
  --tag policyB_handoff_B13_softcommit > "$LOG13" 2>&1 < /dev/null & disown
echo "[iter2] B13 launched -> $LOG13"
echo "[iter2] both launched. Want: post-handoff min-z>0.05 AND held-cos near B4's 0.988 AND obj_jerk DOWN (smoother seam)."
