#!/bin/bash
# EVAL-ON-DONE for the 2026-06-22 smooth/low-force runs. Waits for both trainers to finish,
# then evals each at its OWN matched config (gotcha #13) and writes LOWFORCE_EVAL_RESULTS.txt.
#   - gentleB (Policy B, b34 lineage @ scale 0.2): the clean case — post-handoff min-z + JERK +
#     fingertip FORCE + held-cos via the continuous-handoff demo. Compare to b34_t20 (6.6 N / jerk
#     76 / cos 0.78): WIN = holds at lower force AND lower ang-jerk, even at a lower cos.
#   - lowforceA (Policy A, A's lineage @ scale 0.5): run the handoff demo with the NEW A as
#     --policy-a and frozen b34_t20 as --policy-b, at A's scale 0.5. The PRE-handoff phase (A
#     holding, steps 0..handoff) shows A's delivery fingertip FORCE — compare to the a01 baseline
#     eval'd identically. (The post-handoff B portion is scale-mismatched here — read it only as
#     "did the gentler A still deliver a holdable state", not as a B-quality number.)
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
OUT="$ROOT/LOWFORCE_EVAL_RESULTS.txt"
GB_DIR="${GB_DIR:-results/rl/bx_20260622-1738-policyB_gentleLowforce}"
A_DIR="${A_DIR:-results/rl/bx_20260622-1740-policyA_lowforce}"
A_BASE="results/rl/a01_20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt"
B34_T20="results/rl/bx_20260613-0909-policyB_b33iter_forcefloor_t20/tensorboard/model_405.pt"
GB_LOG="$ROOT/gentleB_policyB_gentleLowforce.trainer.log"
A_LOG="$ROOT/lowforceA_policyA_lowforce.trainer.log"

latest_ckpt () { ls -t "$1"/tensorboard/model_*.pt 2>/dev/null | head -1; }
done_or_dead () { # $1=log  -> 0 (true) when the trainer process is gone
  local log="$1"
  [ -e "${log}.COLLAPSED" ] && return 0
  # the train scripts print "DONE rc=" / "ABORTED" at the very end
  grep -qE "DONE rc=|ABORTED" "$ROOT"/$(basename "$log" .trainer.log).run.log 2>/dev/null && return 0
  # fallback: no python rl_train writing this run for >2 min  (pgrep by run dir is racey; use log mtime)
  return 1
}

echo "[eval] waiting for both trainers to finish ..." | tee "$OUT"
while true; do
  gbdone=0; adone=0
  ls "$GB_DIR"/tensorboard/model_*.pt >/dev/null 2>&1 && done_or_dead "$GB_LOG" && gbdone=1
  ls "$A_DIR"/tensorboard/model_*.pt  >/dev/null 2>&1 && done_or_dead "$A_LOG"  && adone=1
  [ "$gbdone" = 1 ] && [ "$adone" = 1 ] && break
  sleep 60
done
echo "[eval] both done $(date)" | tee -a "$OUT"

run_eval () { # $1=label  $2..=args to the demo
  local label="$1"; shift
  echo "" | tee -a "$OUT"; echo "=== $label ===" | tee -a "$OUT"
  local c; c=$(mktemp -d)
  WARP_CACHE_PATH="$c" MUJOCO_GL=egl uv run --extra rl --extra gpu \
    python "$ROOT/scripts/rl_demo_handoff_continuous.py" "$@" 2>&1 \
    | grep -iE "\[hc\]|\[diag\]|min-z|held-cos|FORCE|JITTER|VERDICT|drift" | tee -a "$OUT"
}

# --- gentleB: clean, scale 0.2 (matched) ---
GB_CKPT=$(latest_ckpt "$GB_DIR")
if [ -e "${GB_LOG}.COLLAPSED" ]; then echo "" | tee -a "$OUT"; echo "=== gentleB COLLAPSED (watchdog) — skipped ===" | tee -a "$OUT"
else run_eval "gentleB  $(basename "$GB_CKPT")  (vs b34_t20: 6.6N / jerk76 / cos0.78)" \
       --policy-b "$GB_CKPT" --finger-residual-scale 0.2 --finger-close-easing linear --no-contact-gate \
       --handoff-step 45 --total-steps 240 --output "$ROOT/docs/rl/videos/reorient/gentleLowforceB_cont.mp4"; fi

# --- lowforceA: A's delivery force @ scale 0.5; pre-handoff phase = A's grip. Plus the a01 baseline. ---
A_CKPT=$(latest_ckpt "$A_DIR")
if [ -e "${A_LOG}.COLLAPSED" ]; then echo "" | tee -a "$OUT"; echo "=== lowforceA COLLAPSED (watchdog) — grasp dropped under the force penalty ===" | tee -a "$OUT"
else
  run_eval "lowforceA $(basename "$A_CKPT")  (NEW A delivery force, scale 0.5; read PRE-handoff)" \
       --policy-a "$A_CKPT" --policy-b "$B34_T20" --finger-residual-scale 0.5 \
       --finger-close-easing ease_out_quad --handoff-step 45 --total-steps 240 \
       --output "$ROOT/docs/rl/videos/reorient/lowforceA_delivery.mp4"
  run_eval "a01 BASELINE (frozen A delivery force, scale 0.5; reference)" \
       --policy-a "$A_BASE" --policy-b "$B34_T20" --finger-residual-scale 0.5 \
       --finger-close-easing ease_out_quad --handoff-step 45 --total-steps 240 \
       --output "$ROOT/docs/rl/videos/reorient/a01_baseline_delivery.mp4"
fi
echo "" | tee -a "$OUT"; echo "[eval] DONE — results in $OUT" | tee -a "$OUT"
