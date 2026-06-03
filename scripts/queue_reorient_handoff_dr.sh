#!/bin/bash
# Handoff-robustness DR sweep (workstream B fix). Two PARALLEL 2048-env runs
# (a single 2048 run is only ~3.9GB/44%, so two fit easily in 16GB — the earlier
# parallel failure was the Warp-cache race, now fixed with per-process caches).
#
# Both finetune the signed+critic reorienter in skip-lift mode but with spawn DR
# (roll/pitch tilt + height jitter), so B tolerates the varied lifted pose a real
# Policy-A lift hands off. Then the existing no-reset continuous handoff script
# should hold + articulate instead of freaking out. Sweep DR strength: lo vs hi.
#
# Detached:  nohup setsid bash scripts/queue_reorient_handoff_dr.sh > hdr.bg.log 2>&1 < /dev/null & disown
# SMOKE=1 -> 1M ts each.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
WARMSTART="${WARMSTART:-$ROOT/results/rl/20260602-1636-policyB_abl_signed/tensorboard/model_405.pt}"
TOTAL_TS=${TOTAL_TS:-20000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
for f in "$WARMSTART" "$ROOT/$MORPH/best_rollout.npz"; do [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

COMMON_ARGS=(
  --morphology-run "$MORPH" --object-body-name screwdriver_medium
  --num-envs 2048 --total-timesteps "$TOTAL_TS" --init-actor-checkpoint "$WARMSTART"
  --episode-length-s 4.0 --lift-target-z-above-init 0.0 --lift-delta-z 0.1
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad
  --object-xy-drift-weight=-3.0 --object-orientation-drift-weight=0.0 --finger-drift-weight=-0.3
  --contact-gate-stability-rewards --enable-lift-terminations --lift-phase-start-step 10
  --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0 --term-tip-lost-steps 10 --term-finger-slip 100.0
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 --reorient-start-step 10
  --contact-min-weight 15.0 --target-axis-progress-weight 300.0 --target-axis-alpha-curriculum-iters 0
  --enable-floor-proximity-termination --object-min-z 0.05 --floor-proximity-phase-start-step 10
  --skip-lift-phase --skip-lift-drop-offset 0.005
  --action-rate-weight=-0.1 --object-ang-acc-weight=-0.05 --smoothness-curriculum-iters 0
  --object-ang-acc-phase-start-step 10 --init-noise-std 0.05 --no-wandb
)
LAST_PID=""
launch(){ local tag="$1" tilt="$2" zj="$3" logf="$4"; local c; c=$(mktemp -d)
  echo "[launch] $tag tilt=$tilt zjit=$zj WARP_CACHE=$c log=$logf"
  WARP_CACHE_PATH="$c" MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
    "${COMMON_ARGS[@]}" --tag "$tag" --skip-lift-spawn-tilt-jitter="$tilt" --skip-lift-spawn-z-jitter="$zj" \
    > "$logf" 2>&1 &
  LAST_PID=$!; }
wfp(){ local l="$1" p="$2"; for _ in $(seq 1 600); do grep -q "starting PPO" "$l" 2>/dev/null && return 0; kill -0 "$p" 2>/dev/null || return 1; sleep 2; done; }

LOGLO="$ROOT/hdr_lo.log"; LOGHI="$ROOT/hdr_hi.log"
echo "===== handoff-DR sweep (2048 envs x2, ts=$TOTAL_TS) ====="
launch "policyB_hdr_lo" 0.12 0.02 "$LOGLO"; PLO=$LAST_PID; echo "[lo] pid=$PLO compiling..."; wfp "$LOGLO" "$PLO"
launch "policyB_hdr_hi" 0.25 0.04 "$LOGHI"; PHI=$LAST_PID; echo "[hi] pid=$PHI"
echo "[hdr] waiting both..."; wait "$PLO"; RLO=$?; wait "$PHI"; RHI=$?
echo "[hdr] DONE lo rc=$RLO hi rc=$RHI"
