#!/bin/bash
# Policy B v2 "smooth & quick" finetune sweep (parallel).
#
# Finetunes Policy B v1 (warmstart from model_2033.pt) with the smooth-&-quick
# curriculum, sweeping the smoothness-ramp target: 5x vs 10x. Both runs enable
# signed progress + all three quick mechanisms (success-hold termination +
# bonus, per-step time cost, early-crossing speed bonus).
#
# PARALLEL FIX (vs queue_overnight_v2.sh, which failed):
#   1. Each run gets its OWN Warp kernel-cache dir (WARP_CACHE_PATH=$(mktemp -d)).
#      The old queue shared ~/.cache/warp with a 30s stagger -> the second run
#      raced the kernel-compile cache and crashed with actor-returns-NaN.
#   2. --num-envs 1024 each (not 2048): two 1024-env runs fit in 16 GB; two
#      2048-env runs OOM'd (rc=143 SIGTERM).
#   3. Run B launches only AFTER run A logs "starting PPO" (kernel compile done).
#   4. Runs are decoupled — one failing does not kill the other.
#
# Usage:
#   scripts/queue_reorient_smooth_sweep.sh            # full sweep (30M ts each)
#   SMOKE=1 scripts/queue_reorient_smooth_sweep.sh    # 2-run ~2min smoke (1M ts)

set -u
ROOT=/home/humanoid/Programs/hand
cd "$ROOT"

MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
# Warmstart checkpoint. Default = Policy B v1 (Stage 1). Override via env for
# Stage 2 (warmstart from the best Stage-1 checkpoint).
WARMSTART="${WARMSTART:-$ROOT/results/rl/20260601-1033-policyB_v1/tensorboard/model_2033.pt}"
TOTAL_TS=${TOTAL_TS:-30000000}
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then TOTAL_TS=1000000; fi

for f in "$WARMSTART" "$ROOT/$MORPH/best_rollout.npz"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

# ---- Policy B base args + the smooth&quick curriculum (shared) ----
COMMON_ARGS=(
  --morphology-run "$MORPH"
  --object-body-name screwdriver_medium
  --num-envs 1024
  --total-timesteps "$TOTAL_TS"
  --init-actor-checkpoint "$WARMSTART"
  --episode-length-s 4.0
  --lift-target-z-above-init 0.0 --lift-delta-z 0.1
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad
  --object-xy-drift-weight=-3.0 --object-orientation-drift-weight=0.0 --finger-drift-weight=-0.3
  --contact-gate-stability-rewards
  --enable-lift-terminations --lift-phase-start-step 10
  --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0 --term-tip-lost-steps 10 --term-finger-slip 100.0
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 --reorient-start-step 10
  --contact-min-weight 15.0 --target-axis-progress-weight 300.0
  --target-axis-alpha-curriculum-iters 300 --target-axis-alpha-start 0.5
  --enable-floor-proximity-termination --object-min-z 0.05 --floor-proximity-phase-start-step 10
  --skip-lift-phase --skip-lift-drop-offset 0.005
  --action-rate-weight=-0.1 --object-ang-acc-weight=-0.05 --object-ang-acc-phase-start-step 10
  --init-noise-std 0.05 --no-wandb
  # Stage 1 (default): smoothness ramp + signed progress (default), NO quick
  # mechanisms — keep the warmstart grip stable while we dial up smoothness.
  --smoothness-curriculum-start-iter 200 --smoothness-curriculum-iters 400
)

# Stage 2 quick-mechanism args, appended only when QUICK=1. Run after a
# Stage-1 policy has converged (warmstart Stage-2 from its checkpoint).
QUICK=${QUICK:-0}
QUICK_ARGS=()
if [ "$QUICK" = "1" ]; then
  QUICK_ARGS=(
    --enable-alignment-success-termination --success-align-thresh 0.9 --success-hold-steps 10
    --success-bonus-weight 50.0
    --time-cost-weight=-0.05
    --speed-bonus-weight 20.0 --speed-bonus-align-thresh 0.9
  )
fi
COMMON_ARGS+=("${QUICK_ARGS[@]}")

LAST_PID=""
launch() {  # $1=tag  $2=action_rate_final  $3=ang_acc_final  $4=logfile
  # Must run in the MAIN shell (not in $(...)), otherwise the backgrounded
  # process is spawned inside a command-substitution subshell, gets reparented
  # when that subshell exits, and `wait` later fails with "not a child of this
  # shell". So we set the global LAST_PID instead of echoing the pid.
  local tag="$1" ar="$2" aa="$3" logf="$4"
  local cache; cache=$(mktemp -d)
  echo "[launch] $tag  WARP_CACHE_PATH=$cache  log=$logf"
  WARP_CACHE_PATH="$cache" MUJOCO_GL=egl \
    uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
      "${COMMON_ARGS[@]}" --tag "$tag" \
      --action-rate-weight-final="$ar" --object-ang-acc-weight-final="$aa" \
      > "$logf" 2>&1 &
  LAST_PID=$!
}

wait_for_ppo() {  # $1=logfile  $2=pid — block until "starting PPO" or the proc dies
  local logf="$1" pid="$2"
  for _ in $(seq 1 600); do          # up to ~20 min for kernel compile
    grep -q "starting PPO" "$logf" 2>/dev/null && { echo "  -> compiled, PPO started"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "  -> process $pid died before PPO start (see $logf)"; return 1; }
    sleep 2
  done
  echo "  -> timed out waiting for PPO start"; return 1
}

LOG5="$ROOT/policyB_v2_5x.log"
LOG10="$ROOT/policyB_v2_10x.log"

echo "================ Policy B v2 smooth&quick sweep (SMOKE=$SMOKE, ts=$TOTAL_TS) ================"
SUF=${TAG_SUFFIX:-}
launch "policyB_v2_smooth5x${SUF}" -0.5 -0.25 "$LOG5"; PID5=$LAST_PID
echo "[run A] policyB_v2_smooth5x${SUF} pid=$PID5; waiting for kernel compile before launching run B..."
wait_for_ppo "$LOG5" "$PID5"

launch "policyB_v2_smooth10x${SUF}" -1.0 -0.5 "$LOG10"; PID10=$LAST_PID
echo "[run B] policyB_v2_smooth10x${SUF} pid=$PID10"

echo "[sweep] waiting for both runs..."
wait "$PID5"; RC5=$?
wait "$PID10"; RC10=$?
echo "[sweep] DONE  5x rc=$RC5  10x rc=$RC10"
echo "  5x  log: $LOG5"
echo "  10x log: $LOG10"
