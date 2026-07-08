#!/bin/bash
# Autonomous follow-on: wait for the cont40M live-A run to finish, then eval the
# final checkpoint on the continuous A->B handoff and write the answers to
# LIVEA_CONT_RESULTS.txt. Tests BOTH issues flagged 2026-06-10:
#   (1) reorientation quality  -> post-handoff held-cos (was 0.75; 40M should raise it)
#   (2) jitter/smoothness      -> A/B the deploy action low-pass (0.5 vs off)
# Launch detached: nohup setsid bash scripts/liveA_cont_eval_trigger.sh > liveA_cont_eval.log 2>&1 </dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
OUT="$ROOT/LIVEA_CONT_RESULTS.txt"
VID=docs/rl/videos/reorient

echo "[trigger] waiting for cont40M trainer to exit..."
while pgrep -f "rl_train_cube.py.*policyB_liveAreset_cont40M" >/dev/null 2>&1; do sleep 60; done
echo "[trigger] trainer gone; locating final checkpoint..."
RUN=$(ls -1dt "$ROOT"/results/rl/*policyB_liveAreset_cont40M 2>/dev/null | head -1)
CKPT=$(ls -1t "$RUN"/tensorboard/model_*.pt 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "[trigger] FATAL: no checkpoint in $RUN"; exit 1; }

# Scale-0.2 lineage (model_270 native scale) — MUST match training (gotcha #13).
COMMON=(--policy-b "$CKPT" --finger-residual-scale 0.2 --finger-close-easing linear
        --no-contact-gate --handoff-step 40 --total-steps 240)

{
  echo "=== live-A cont40M handoff eval  $(date) ==="
  echo "ckpt: $CKPT"
  echo "(scale-0.2 lineage; metric = POST-HANDOFF min-z [hold>0.05] + held-vertical cos)"
  echo
} > "$OUT"

run_eval () {  # $1 label  $2... extra args
  local label="$1"; shift
  local c; c=$(mktemp -d)
  echo "[trigger] eval: $label"
  local log; log=$(mktemp)
  WARP_CACHE_PATH="$c" MUJOCO_GL=egl uv run --extra rl --extra gpu python \
    scripts/rl_demo_handoff_continuous.py "${COMMON[@]}" "$@" > "$log" 2>&1
  {
    echo "--- $label ---"
    grep -E "min object-center z (over|POST)|held-vertical" "$log" || echo "  (eval failed; see $log)"
    echo
  } >> "$OUT"
}

run_eval "baseline (no filter)"      --output "$VID/handoff_cont40M.mp4"
run_eval "action low-pass 0.5"       --output "$VID/handoff_cont40M_lp05.mp4" --action-lowpass 0.5
run_eval "action low-pass 0.3"       --output "$VID/handoff_cont40M_lp03.mp4" --action-lowpass 0.3

echo "[trigger] DONE. Results:"; cat "$OUT"
