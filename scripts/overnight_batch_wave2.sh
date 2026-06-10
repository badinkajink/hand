#!/bin/bash
# Overnight batch WAVE 2 (2026-06-09 ~21:00) — proper co-adaptation B.
# =============================================================================
# Wave-1 findings that shape this:
#   - VELOCITY injection is HARMFUL (teleport + real finger qvel = grip-perturbing
#     transient); STATIC (zero-vel) injection is the best B-side variant (0.0081).
#     last_action override is ~neutral. -> use INJECT_VEL=0 INJECT_LASTACT=0.
#   - The co-adapt pairing Atol20×Badapt=0.0114 is the best result, but the proper
#     co-adapt TRAINING (run 1) was sabotaged twice: warmstart Badapt (overfit to
#     frozen-A, can't catch the MIGRATED delivery) AND complete-state inject (harmful
#     velocity). Wave 2 fixes both: warmstart B10 + STATIC inject of Atol20's delivery.
# GOAL: a B that catches the MIGRATED A (Atol20) -> eval Atol20×NEW should beat 0.0114.
# (If it does, wave 3 = the alternating loop: record this B's catch, re-migrate A to it.)
#
# Launch detached:
#   nohup setsid bash scripts/overnight_batch_wave2.sh > overnight_batch_wave2.run.log 2>&1 </dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
RES="$ROOT/BATCH_RESULTS.md"
B10=results/rl/20260604-1642-policyB_holdonlyws_repro/tensorboard/model_541.pt
BADAPT=results/rl/20260609-1113-policyB_onsetInject_bankA_s40/tensorboard/model_270.pt
A_TOL20=results/rl/20260605-1609-policyA_unfreezeA_v2_w2_tol20/tensorboard/model_270.pt
BANK_ATOL20="$ROOT/results/rl/handoff_state_bank_Atol20_s40_full.npz"

echo "" >> "$RES"; echo "## Batch run $(date '+%Y-%m-%d %H:%M') — wave 2 (proper co-adapt, STATIC inject)" >> "$RES"
echo "| run | warmstart | eval pairing | min-z (bar .05) | z@handoff | status |" >> "$RES"
echo "|---|---|---|---|---|---|" >> "$RES"

wait_for_gpu () { for _ in $(seq 1 40); do
    pgrep -f "rl_train_cube.py" >/dev/null 2>&1 || {
      M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      [ -n "$M" ] && [ "$M" -lt 3000 ] 2>/dev/null && return 0; }
    sleep 15; done; return 0; }
latest_ckpt () { ls -t "$1"/tensorboard/model_*.pt 2>/dev/null | head -1; }
rundir_for () { ls -dt "$ROOT"/results/rl/*-"$1" 2>/dev/null | head -1; }

eval_seam () { local c; c=$(mktemp -d)
  WARP_CACHE_PATH="$c" MUJOCO_GL=egl uv run --extra rl --extra gpu python \
    scripts/rl_demo_handoff_continuous.py --policy-a "$2" --policy-b "$3" \
    --output "docs/rl/videos/reorient/batch_$1.mp4" --handoff-step 40 --total-steps 240 2>&1 \
    | grep -E "object z at handoff|min object-center" ; }

# do_run TAG WS_LABEL EVAL_A -- <train command...>   (eval = EVAL_A × NEW ckpt)
do_run () {
  local tag="$1" ws="$2" evala="$3"; shift 3; [ "$1" = "--" ] && shift
  echo "[w2] === $tag START $(date '+%H:%M') ==="; wait_for_gpu
  ( "$@" ) >/dev/null 2>&1
  local rd ck; rd=$(rundir_for "$tag"); ck=$(latest_ckpt "$rd")
  local st="ok"; ls "$ROOT"/*"$tag"*.COLLAPSED >/dev/null 2>&1 && st="COLLAPSED"
  echo "[w2] $tag -> ck=$ck $st"; wait_for_gpu
  if [ -z "$ck" ]; then echo "| $tag | $ws | — | NO CKPT | — | $st |" >> "$RES";
  else local out mz zh; out=$(eval_seam "$tag" "$evala" "$ck")
    mz=$(echo "$out" | grep -oE "min object-center z over rollout: [-0-9.]+" | grep -oE "[-0-9.]+$")
    zh=$(echo "$out" | grep -oE "object z at handoff[^:]*: [-0-9.]+" | grep -oE "[-0-9.]+$")
    echo "| $tag | $ws | Atol20×NEW | ${mz:-?} | ${zh:-?} | $st |" >> "$RES"
    echo "[w2] $tag RESULT min-z=$mz z@handoff=$zh $st"; fi
  echo "[w2] === $tag DONE $(date '+%H:%M') ==="; }

# (1) KEY: B10 warmstart + STATIC inject of the MIGRATED A's (Atol20) delivery.
do_run coadapt_B10_Atol20_static B10 "$A_TOL20" -- \
  env BANK="$BANK_ATOL20" B_CKPT="$ROOT/$B10" INJECT_VEL=0 INJECT_LASTACT=0 \
  TAG=coadapt_B10_Atol20_static bash scripts/train_handoff_onset_inject.sh

# (2) Badapt warmstart + STATIC inject of Atol20 (does static rescue what complete-state
#     broke in wave-1 run 1? isolates warmstart vs inject-mode for the migrated delivery).
do_run coadapt_Badapt_Atol20_static Badapt "$A_TOL20" -- \
  env BANK="$BANK_ATOL20" B_CKPT="$ROOT/$BADAPT" INJECT_VEL=0 INJECT_LASTACT=0 \
  COLLAPSE_Z=0.012 GUARD_FROM_ITER=100 TAG=coadapt_Badapt_Atol20_static \
  bash scripts/train_handoff_onset_inject.sh

echo "[w2] ===== WAVE 2 COMPLETE $(date) ===== see $RES"
