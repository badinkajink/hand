#!/bin/bash
# Branch B weight sweep — un-freeze Policy A grip-proximity finetune.
# The grip-proximity reward (seam window 33-37) competes with A's native
# track_finger_qpos / finger_drift, which pull the fingers back to A's own
# grasp. So sweep how hard we pull toward B10's grip: weights 4 / 8 / 16.
# Three 3072-env runs fit in 16 GB (~11 GB total); separate Warp caches +
# staggered launches (gotcha #2). 20M ts each (~35 min/run, parallel).
#
# Detached: nohup setsid bash scripts/sweep_branchB_unfreezeA.sh > branchB_sweep.bg.log 2>&1 </dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
export TOTAL_TS=${TOTAL_TS:-20000000}
PIDS=()
for W in 4 8 16; do
  LOG="$ROOT/branchB_w${W}.log"
  echo "[sweep] launching weight=$W -> $LOG"
  WEIGHT=$W TAG="policyA_unfreezeA_gripw${W}" \
    bash "$ROOT/scripts/train_handoff_branchB_unfreezeA.sh" > "$LOG" 2>&1 &
  PIDS+=($!)
  # stagger so kernel compiles don't pile up
  for _ in $(seq 1 45); do grep -q "starting PPO\|Learning iteration" "$LOG" 2>/dev/null && break; sleep 2; done
done
echo "[sweep] pids=${PIDS[*]} ; waiting..."
RC=0; for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
echo "[sweep] ALL DONE rc=$RC"
