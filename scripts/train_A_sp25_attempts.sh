#!/bin/bash
# Policy A on the 25 mm-proximal hand (sp25), best-of-N attempts.
# =============================================================================
# A from-scratch A is a DRAW, not a deterministic outcome: the morphology sweeps see
# 40-50% of A runs collapse to object-on-floor regardless of design, which is why
# --a-attempts 3 became the sweep default. Attempt 1 here lifted cleanly for 9 iters
# (object_height 0.058 -> 0.124) and then parked the tool on the floor (0.0123) from
# iter 10, tripping the watchdog at 40. That is the known draw, not a verdict on the
# hand -- the open-loop CEM grip lifts this design fine (cube_lift 0.0497).
#
# Terminations are pinned to a10's OWN config, not the a_lift recipe defaults: a10
# trained with the slip guards effectively off (finger_slip 100.0, object_slip_yaw
# 10.0, object_slip_xy 0.05) and the recipe ships 0.3 / 0.5 / 0.015. Training under
# the strict values is a different task and would have read as "the short hand
# cannot lift".
#
# Launch: nohup setsid bash scripts/train_A_sp25_attempts.sh > logs/sp25/A_attempts.run.log 2>&1 </dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH_RUN="$ROOT/results/phase1/shortprox25/20260819-sp25_ik_cem"
ATTEMPTS=${ATTEMPTS:-3}

for t in $(seq 1 "$ATTEMPTS"); do
  TAG="policyA_sp25_t${t}"
  LOG="$ROOT/logs/sp25/A_sp25_t${t}.trainer.log"
  rm -f "${LOG}.COLLAPSED"                      # stale sentinel = false abort
  echo "[sp25-A] === attempt $t/$ATTEMPTS -> $TAG"
  MORPH_RUN="$MORPH_RUN" WARMSTART=none INIT_NOISE_STD=0.1 LIFT_DELTA_A=0.10 \
  NUM_ENVS=2048 TOTAL_TS=30000000 \
  EXTRA_ARGS="--open-finger-from-keyframe --lift-phase-start-step 60 --term-finger-slip 100.0 --term-object-slip-xy 0.05 --term-object-slip-yaw 10.0" \
  TAG="$TAG" LOG="$LOG" bash scripts/train_A_on_morph.sh

  if [ -e "${LOG}.COLLAPSED" ]; then
    echo "[sp25-A] attempt $t COLLAPSED — retrying"
    # let the GPU drain before the next Warp process (CLAUDE.md: ~1 GB)
    for _ in $(seq 1 40); do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$u" -lt 2000 ] && break; sleep 15
    done
    continue
  fi
  echo "[sp25-A] attempt $t SURVIVED — keeping $TAG"
  echo "$TAG" > "$ROOT/logs/sp25/A_KEPT"
  break
done
echo "[sp25-A] done. kept=$(cat "$ROOT/logs/sp25/A_KEPT" 2>/dev/null || echo NONE)"
