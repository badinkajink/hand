#!/bin/bash
# Policy B v2.1 — de-centering fix sweep (workstream A).
#
# Warmstarts the recommended v2 policy (s2-10x-quick model_1219) with its OWN
# converged reward config held fixed, and adds ONLY the palm-frame lateral-drift
# penalty (object_lateral_drift), sweeping its weight. Isolates the de-centering
# fix: everything else matches the warmstart so any change is attributable to the
# lateral penalty.
#
# Differences from the s2-10x-quick training config, and why:
#   - action_rate_weight/object_ang_acc held FLAT at the converged 10x level
#     (-1.0 / -0.5), smoothness curriculum OFF — we warmstart an already-smooth
#     policy, so there's nothing to ramp.
#   - target_axis_alpha curriculum OFF (alpha=4.0 flat) — avoids re-softening the
#     alignment basin at iter 0 of a finetune (the curriculum restarts per run).
#   - quick mechanisms kept ON (as in s2-10x-quick, where they're near-inert) so
#     the only *new* signal is the lateral penalty.
#
# Parallel-safe (per-process Warp cache, 1024 envs each). Run detached:
#   nohup setsid bash scripts/queue_reorient_decenter.sh > decenter_sweep.log 2>&1 < /dev/null & disown
# SMOKE=1 → 1M ts each (~2 min) to validate.
set -u
ROOT=/home/humanoid/Programs/hand
cd "$ROOT"

MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
WARMSTART="$ROOT/results/rl/20260602-0024-policyB_v2_smooth10x_quick/tensorboard/model_1219.pt"
TOTAL_TS=${TOTAL_TS:-20000000}
SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && TOTAL_TS=1000000
# Lateral-drift weights to sweep (one run each). Default: moderate vs strong.
W_A=${W_A:--15.0}
W_B=${W_B:--40.0}

for f in "$WARMSTART" "$ROOT/$MORPH/best_rollout.npz"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

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
  --target-axis-alpha-curriculum-iters 0
  --enable-floor-proximity-termination --object-min-z 0.05 --floor-proximity-phase-start-step 10
  --skip-lift-phase --skip-lift-drop-offset 0.005
  --action-rate-weight=-1.0 --object-ang-acc-weight=-0.5 --object-ang-acc-phase-start-step 10
  --smoothness-curriculum-iters 0
  --enable-alignment-success-termination --success-align-thresh 0.9 --success-hold-steps 10
  --success-bonus-weight 30.0 --time-cost-weight=-0.02 --speed-bonus-weight 15.0 --speed-bonus-align-thresh 0.9
  --lateral-drift-deadband 0.01 --lateral-drift-power 2.0
  --init-noise-std 0.05 --no-wandb
)

LAST_PID=""
launch() {  # $1=tag $2=lateral_weight $3=logfile  (background in MAIN shell)
  local tag="$1" w="$2" logf="$3"
  local cache; cache=$(mktemp -d)
  echo "[launch] $tag  lateral_weight=$w  WARP_CACHE_PATH=$cache  log=$logf"
  WARP_CACHE_PATH="$cache" MUJOCO_GL=egl \
    uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
      "${COMMON_ARGS[@]}" --tag "$tag" --lateral-drift-weight="$w" \
      > "$logf" 2>&1 &
  LAST_PID=$!
}
wait_for_ppo() { local logf="$1" pid="$2"; for _ in $(seq 1 600); do
    grep -q "starting PPO" "$logf" 2>/dev/null && { echo "  -> PPO started"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "  -> $pid died (see $logf)"; return 1; }; sleep 2; done; return 1; }

LOGA="$ROOT/decenter_w${W_A#-}.log"; LOGB="$ROOT/decenter_w${W_B#-}.log"
echo "================ de-centering sweep (SMOKE=$SMOKE, ts=$TOTAL_TS, weights $W_A / $W_B) ================"
launch "policyB_v21_decenter_w${W_A#-}" "$W_A" "$LOGA"; PIDA=$LAST_PID
echo "[run A] pid=$PIDA; waiting for compile..."; wait_for_ppo "$LOGA" "$PIDA"
launch "policyB_v21_decenter_w${W_B#-}" "$W_B" "$LOGB"; PIDB=$LAST_PID
echo "[run B] pid=$PIDB"
echo "[sweep] waiting for both..."
wait "$PIDA"; RCA=$?; wait "$PIDB"; RCB=$?
echo "[sweep] DONE  A($W_A) rc=$RCA  B($W_B) rc=$RCB"
echo "  A log: $LOGA"; echo "  B log: $LOGB"
