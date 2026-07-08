#!/bin/bash
# Policy B v3 — bracing sweep (workstream C / Phase 3).
#
# Warmstarts the de-centering winner (Policy B v2.1) and adds the bracing
# rewards: palm<->cylinder normal force (push the end flat into the palm) and
# fingertip grip force (pinch-to-power), the brace reward GATED on alignment so
# the policy reorients first, then braces. Trains in the SAME skip-lift env as
# v2 (object spawns lifted+gripped) so the warmstart is in-distribution — the
# separate A->B handoff distribution gap (normal-lift retrain) is NOT addressed
# here on purpose.
#
# Config vs the de-centering winner: keeps smoothness flat at the converged 10x
# level (-1.0/-0.5), signed progress, target_axis, and the lateral-drift penalty
# (LAT_W); DROPS the near-inert "quick" mechanisms (they risked threshold-gaming);
# ADDS brace + grip force. Sweeps brace strength.
#
# Run detached:
#   WARMSTART=<decenter_winner_ckpt> LAT_W=-15 \
#   nohup setsid bash scripts/queue_reorient_brace.sh > brace_sweep.bg.log 2>&1 < /dev/null & disown
# SMOKE=1 -> 1M ts each.
set -u
ROOT=/home/humanoid/Programs/hand
cd "$ROOT"

MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
# Default warmstart = recommended v2 (override with the de-centering winner).
WARMSTART="${WARMSTART:-$ROOT/results/rl/b02_20260602-0024-policyB_v2_smooth10x_quick/tensorboard/model_1219.pt}"
LAT_W="${LAT_W:--15.0}"           # lateral-drift weight carried from the de-centering winner
TOTAL_TS=${TOTAL_TS:-30000000}
SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && TOTAL_TS=1000000
# Sweep the DENSE brace-distance weight (the ingredient that makes bracing
# discoverable; diagnostic showed the gripped cylinder sits ~8cm from the palm,
# so the sparse force reward never fires on its own). Sparse force + grip fixed.
BRACE_A=${BRACE_A:-8.0}
BRACE_B=${BRACE_B:-20.0}
GRIP_W=${GRIP_W:-5.0}             # fingertip grip-force reward (both runs)
BRACE_FORCE_W=${BRACE_FORCE_W:-15.0}  # sparse palm-contact-force reward (both runs)

for f in "$WARMSTART" "$ROOT/$MORPH/best_rollout.npz"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

COMMON_ARGS=(
  --morphology-run "$MORPH" --object-body-name screwdriver_medium
  --num-envs 1024 --total-timesteps "$TOTAL_TS" --init-actor-checkpoint "$WARMSTART"
  --episode-length-s 4.0
  --lift-target-z-above-init 0.0 --lift-delta-z 0.1
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad
  --object-xy-drift-weight=-3.0 --object-orientation-drift-weight=0.0 --finger-drift-weight=-0.3
  --contact-gate-stability-rewards
  --enable-lift-terminations --lift-phase-start-step 10
  --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0 --term-tip-lost-steps 10 --term-finger-slip 100.0
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 --reorient-start-step 10
  --contact-min-weight 15.0 --target-axis-progress-weight 300.0 --target-axis-alpha-curriculum-iters 0
  --enable-floor-proximity-termination --object-min-z 0.05 --floor-proximity-phase-start-step 10
  --skip-lift-phase --skip-lift-drop-offset 0.005
  --action-rate-weight=-1.0 --object-ang-acc-weight=-0.5 --object-ang-acc-phase-start-step 10
  --smoothness-curriculum-iters 0
  --lateral-drift-weight="$LAT_W" --lateral-drift-deadband 0.01 --lateral-drift-power 2.0
  # bracing (gated on alignment): reorient first, then push end into palm.
  # dense distance shaping (swept) + sparse force + grip force.
  --grip-force-weight="$GRIP_W" --grip-force-max 3.0 --grip-force-reduce mean
  --brace-align-thresh 0.7 --brace-max-force 3.0
  --brace-force-weight="$BRACE_FORCE_W" --brace-distance-scale 0.04
  --init-noise-std 0.05 --no-wandb
)

LAST_PID=""
launch() {  # $1=tag $2=brace_distance_weight $3=logfile
  local tag="$1" bw="$2" logf="$3"; local cache; cache=$(mktemp -d)
  echo "[launch] $tag brace_dist=$bw brace_force=$BRACE_FORCE_W grip=$GRIP_W lat=$LAT_W WARP_CACHE=$cache log=$logf"
  WARP_CACHE_PATH="$cache" MUJOCO_GL=egl \
    uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
      "${COMMON_ARGS[@]}" --tag "$tag" --brace-distance-weight="$bw" \
      > "$logf" 2>&1 &
  LAST_PID=$!
}
wait_for_ppo() { local logf="$1" pid="$2"; for _ in $(seq 1 600); do
    grep -q "starting PPO" "$logf" 2>/dev/null && { echo "  -> PPO started"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "  -> $pid died (see $logf)"; return 1; }; sleep 2; done; return 1; }

LOGA="$ROOT/brace_b${BRACE_A%.*}.log"; LOGB="$ROOT/brace_b${BRACE_B%.*}.log"
echo "============ bracing sweep (SMOKE=$SMOKE ts=$TOTAL_TS brace $BRACE_A/$BRACE_B grip $GRIP_W) ============"
echo "warmstart: $WARMSTART"
launch "policyB_v3_brace${BRACE_A%.*}" "$BRACE_A" "$LOGA"; PIDA=$LAST_PID
echo "[run A] pid=$PIDA; waiting for compile..."; wait_for_ppo "$LOGA" "$PIDA"
launch "policyB_v3_brace${BRACE_B%.*}" "$BRACE_B" "$LOGB"; PIDB=$LAST_PID
echo "[run B] pid=$PIDB"
echo "[sweep] waiting for both..."; wait "$PIDA"; RCA=$?; wait "$PIDB"; RCB=$?
echo "[sweep] DONE  brace$BRACE_A rc=$RCA  brace$BRACE_B rc=$RCB"
echo "  A log: $LOGA"; echo "  B log: $LOGB"
