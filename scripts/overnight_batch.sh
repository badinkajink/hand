#!/bin/bash
# Overnight handoff-seam batch runner (2026-06-09) — CO-ADAPTATION wave.
# =============================================================================
# Runs a QUEUE of training experiments ONE AT A TIME (safest unattended: no OOM,
# eval always has the GPU free), auto-evaluates each on the decisive continuous-
# handoff min-z (handoff@40), and appends a row to BATCH_RESULTS.md. Robust:
#   - set -u, NOT -e: one run's NaN/collapse is logged and SKIPPED.
#   - GPU-release wait between runs (gotcha #8/#12).
#   - unique TAG per run -> run dir found by TAG; eval handles missing ckpts.
#
# WHY (state @2026-06-09): the B-side injection paradigm is SATURATED (skip-lift
# bank 0.0028, static onset 0.0081, complete-state onset 0.0027 — Markov-complete
# injection did NOT help, so the seam is not missing-state-info). The NEW signal
# is CO-ADAPTATION: pairing the independently-migrated A (Atol20, A->B10 grip) with
# the independently-adapted B (Badapt, B->frozenA delivery) gives min-z 0.0114 — a
# new best, beating either-side-alone (A-alone -0.0001, B-alone 0.0075). Moving
# BOTH toward each other is the lever (Lee 2021 / Röstel 2025). Wave 1 runs a
# proper co-adaptation ROUND + pushes A-migration + ablates the complete-state vars.
# (Bar = 0.05; prior best 0.0081; co-adapt cross-pairing already 0.0114.)
#
# Launch detached:
#   nohup setsid bash scripts/overnight_batch.sh > logs/overnight_batch.run.log 2>&1 </dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
RES="$ROOT/BATCH_RESULTS.md"
B10=results/rl/b10_20260604-1642-policyB_holdonlyws_repro/tensorboard/model_541.pt
A_FROZEN=results/rl/a01_20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt
A_TOL20=results/rl/a07_20260605-1609-policyA_unfreezeA_v2_w2_tol20/tensorboard/model_270.pt
BADAPT=results/rl/b15_20260609-1113-policyB_onsetInject_bankA_s40/tensorboard/model_270.pt
BANK_FROZENA="$ROOT/results/rl/handoff_state_bank_A_s40_full.npz"
BANK_ATOL20="$ROOT/results/rl/handoff_state_bank_Atol20_s40_full.npz"
BANK_B10GRIP="$ROOT/results/rl/b10_initiation_bank_s35.npz"
BANK_BADAPTGRIP="$ROOT/results/rl/badapt_initiation_s48.npz"

echo "" >> "$RES"; echo "## Batch run $(date '+%Y-%m-%d %H:%M') — co-adaptation wave 1" >> "$RES"
echo "| run | warmstart | eval pairing | min-z (bar .05) | z@handoff | status |" >> "$RES"
echo "|---|---|---|---|---|---|" >> "$RES"

wait_for_gpu () { for _ in $(seq 1 40); do
    pgrep -f "rl_train_cube.py" >/dev/null 2>&1 || {
      M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      [ -n "$M" ] && [ "$M" -lt 3000 ] 2>/dev/null && return 0; }
    sleep 15; done; return 0; }
latest_ckpt () { ls -t "$1"/tensorboard/model_*.pt 2>/dev/null | head -1; }
rundir_for () { ls -dt "$ROOT"/results/rl/*-"$1" 2>/dev/null | head -1; }
resolve () { [ "$1" = "NEW" ] && echo "$2" || echo "$1"; }  # NEW -> the run's own ckpt

eval_seam () {  # name, policy_a, policy_b
  local c; c=$(mktemp -d)
  WARP_CACHE_PATH="$c" MUJOCO_GL=egl uv run --extra rl --extra gpu python \
    scripts/rl_demo_handoff_continuous.py --policy-a "$2" --policy-b "$3" \
    --output "docs/rl/videos/reorient/batch_$1.mp4" --handoff-step 40 --total-steps 240 2>&1 \
    | grep -E "object z at handoff|min object-center" ; }

row () {  # name, ws, pairing-label, A, B, status
  local out mz zh; out=$(eval_seam "$1" "$4" "$5")
  mz=$(echo "$out" | grep -oE "min object-center z over rollout: [-0-9.]+" | grep -oE "[-0-9.]+$")
  zh=$(echo "$out" | grep -oE "object z at handoff[^:]*: [-0-9.]+" | grep -oE "[-0-9.]+$")
  echo "| $1 | $2 | $3 | ${mz:-?} | ${zh:-?} | $6 |" >> "$RES"
  echo "[batch] $1 RESULT min-z=$mz z@handoff=$zh ($3) $6"; }

# do_run TAG WS_LABEL EVALA EVALB -- <train command...>
do_run () {
  local tag="$1" ws="$2" evala="$3" evalb="$4"; shift 4; [ "$1" = "--" ] && shift
  echo "[batch] === $tag START $(date '+%H:%M') ==="; wait_for_gpu
  ( "$@" ) >/dev/null 2>&1
  local rd ck; rd=$(rundir_for "$tag"); ck=$(latest_ckpt "$rd")
  local st="ok"; ls "$ROOT"/*"$tag"*.COLLAPSED >/dev/null 2>&1 && st="COLLAPSED"
  echo "[batch] $tag -> rd=$rd ck=$ck $st"; wait_for_gpu
  if [ -z "$ck" ]; then echo "| $tag | $ws | — | NO CKPT | — | $st |" >> "$RES";
  else row "$tag" "$ws" "$(basename "$(resolve "$evala" NEWMARK)"|sed 's/.pt//')×$(basename "$(resolve "$evalb" NEWMARK)"|sed 's/.pt//')" \
            "$(resolve "$evala" "$ck")" "$(resolve "$evalb" "$ck")" "$st"; fi
  echo "[batch] === $tag DONE $(date '+%H:%M') ==="; }

# ===================== WAVE 1: CO-ADAPTATION ROUND =====================
# The weak link is B CATCHING (A's migrated delivery holds through 0-40; B drops at
# takeover). So co-adaptation = specialize B to the migrated A's delivery. (Note:
# Badapt has NO stable post-seam holding grip — it drops at ~step 48 — so migrating A
# toward "Badapt's grip" is not viable; the A->B10-grip migration is the A-side lever.)
#
# (1) KEY: re-adapt B (warmstart Badapt) to the MIGRATED A's (Atol20) REAL delivery
#     (complete-state bank). Pairs with Atol20 -> should beat the 0.0114 cross-pairing.
do_run coadapt_B_toAtol20 Badapt "$A_TOL20" NEW -- \
  env BANK="$BANK_ATOL20" B_CKPT="$ROOT/$BADAPT" TAG=coadapt_B_toAtol20 \
  bash scripts/train_handoff_onset_inject.sh

# (2) Converged complete-state B (the 0.0027 run NaN'd @221): warmstart Badapt (not
#     B10) on frozen-A's full bank -> a clean complete-state B. Pairs with frozen-A.
do_run B_complete_fromBadapt Badapt "$A_FROZEN" NEW -- \
  env BANK="$BANK_FROZENA" B_CKPT="$ROOT/$BADAPT" TAG=B_complete_fromBadapt \
  bash scripts/train_handoff_onset_inject.sh

# (3,4) PUSH A-migration toward B10's grip (more-migrated A variants). Eval vs Badapt
#       (the catch that helped in the free cross-pairing).
do_run branchB_w6_tol20 frozenA "NEW" "$BADAPT" -- \
  env WEIGHT=6 QPOS_TOL=0.20 TAG=branchB_w6_tol20 \
  bash scripts/train_handoff_branchB_v2.sh
do_run branchB_w4_tol15 frozenA "NEW" "$BADAPT" -- \
  env WEIGHT=4 QPOS_TOL=0.15 TAG=branchB_w4_tol15 \
  bash scripts/train_handoff_branchB_v2.sh

# (5,6) Complete-state injection 2x2 ablation (explain 0.0027<0.0081). frozen-A pairing.
do_run inject_velOnly frozenA "$A_FROZEN" NEW -- \
  env BANK="$BANK_FROZENA" INJECT_LASTACT=0 TAG=inject_velOnly \
  bash scripts/train_handoff_onset_inject.sh
do_run inject_lastactOnly frozenA "$A_FROZEN" NEW -- \
  env BANK="$BANK_FROZENA" INJECT_VEL=0 TAG=inject_lastactOnly \
  bash scripts/train_handoff_onset_inject.sh

echo "[batch] ===== WAVE 1 COMPLETE $(date) ===== see $RES"
