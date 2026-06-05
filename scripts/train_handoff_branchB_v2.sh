#!/bin/bash
# Branch B v2 — un-freeze Policy A toward B10's grip, WITHOUT destabilizing A.
# =============================================================================
# WHY v1 FAILED (runs 20260604-22{05,06,10}_unfreezeA_gripw{4,8,16}):
#   v1 stripped ALL of A's lift-phase terminations (never passed
#   --enable-lift-terminations) on the theory that lift_height + track_object
#   rewards would hold the grasp. They did NOT. With no drop termination and no
#   drop penalty, "object resting on the floor" is a stable non-terminating
#   state; PPO discovered it within ~25 iters (object_height 0.09 -> 0.013,
#   contact_min 6.6 -> 0.008) and never recovered. The grip-proximity reward
#   stayed at ~0.02 the whole time (fingers never near B10's grip once the object
#   was on the floor), so v1 NEVER ACTUALLY TESTED the grip-match hypothesis.
#   All three were also OOM-killed (rc=137) at ~15 min — three 3072-env runs at
#   once over-subscribed memory. So: run ONE at a time.
#
# THE FIX (change exactly one thing vs A's own training env):
#   Policy A was TRAINED with enable_lift_terminations:true, term_object_drop
#   0.02, term_finger_slip 0.3, lift_phase_start_step 40. We restore ALL of that
#   (so the floor can never become an attractor), and relax ONLY term_finger_slip
#   (0.3 -> 2.0) so A is allowed to migrate its grip ~0.16 rad/joint toward B10's
#   holding pose (~0.48 rad L2 would otherwise trip the 0.3 finger-slip term —
#   the one real tension v1 spotted but "fixed" by throwing out every guardrail).
#   The drop/tip-loss terminations still forbid actually losing the object, so
#   the worst case is "A keeps its own grip, proximity stays low" (safe &
#   informative), never "A learns to drop" (the v1 collapse).
#   Start the proximity weight MODEST (2.0, vs v1's 4/8/16) — the structural
#   contact/lift rewards must stay in charge; sweep up only after a run holds.
#
# A baked-in watchdog kills the run if object_height collapses (<0.045 at iter>=24)
# so we never waste compute on a dead run.
#
# Launch (single run, detached):
#   nohup setsid bash scripts/train_handoff_branchB_v2.sh > branchB_v2.log 2>&1 </dev/null & disown
# Knobs: WEIGHT (proximity, default 2.0), FINGER_SLIP_TOL (default 2.0),
#        TOTAL_TS (default 20M), SMOKE=1 (1M ts plumbing/NaN check).
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
A_CKPT="${A_CKPT:-$ROOT/results/rl/20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt}"
BANK="${BANK:-$ROOT/results/rl/b10_initiation_bank_s35.npz}"
WEIGHT="${WEIGHT:-2.0}"
FINGER_SLIP_TOL="${FINGER_SLIP_TOL:-2.0}"   # relaxed (A trained at 0.3) so grip can migrate
# v2 attempt-1 (gripw2.0, 20260605-1530) collapsed @iter25 — but NOT v1's floor-drop:
# --enable-lift-terminations also turned on A's TIGHT object-slip guards
# (term_object_slip_xy 0.015 m, term_object_slip_yaw 0.5 rad). Migrating the grip
# nudges the object past 1.5 cm, so object_slip fired 100+/iter from iter 1 and
# killed every episode before A could learn. FIX: keep the DROP guard tight (the
# real "don't lose it"), relax the precision OBJECT-SLIP guards so grip migration
# is allowed. The drop term (0.02 m) still forbids actually losing the object.
SLIP_XY_TOL="${SLIP_XY_TOL:-0.05}"          # was A's 0.015 — allow object motion during re-grip
SLIP_YAW_TOL="${SLIP_YAW_TOL:-1.0}"         # was A's 0.5  — allow some yaw during re-grip
# v2 attempt-2 (gripw2_slip5) FIXED the collapse (object_height held ~0.088 to iter 55) but the
# grip-proximity reward stayed FLAT at ~0.002: with qpos_tol 0.05 and A's grip ~0.16 rad/joint off
# B10's, A sits ~3σ out in the Gaussian tail where reward≈0 AND gradient≈0 — nothing pulls A toward
# B10's grip. FIX: WIDEN the basin so there's a gradient at the real gap (tol 0.15 -> 0.57/step).
QPOS_TOL="${QPOS_TOL:-0.15}"                 # was 0.05 (too sharp; no gradient at the 0.16-rad gap)
TOTAL_TS=${TOTAL_TS:-20000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-policyA_unfreezeA_v2_gripw${WEIGHT}}"
for f in "$A_CKPT" "$BANK" "$ROOT/$MORPH/best_rollout.npz"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

# Deploy-matched A-side env (normal lift to 0.10, 65-d A space, contact-gate),
# but with A's OWN training guardrails RESTORED (the v1 regression undone):
#   --enable-lift-terminations  -> drop / tip-loss / finger-slip terms back on
#   --term-finger-slip 2.0      -> ONLY the finger-slip term relaxed (grip migration)
#   (term_object_drop 0.02, term_tip_lost_steps 3, lift_phase_start_step 40 = A's defaults)
ARGS=(
  --morphology-run "$MORPH" --object-body-name screwdriver_medium
  --num-envs 3072 --total-timesteps "$TOTAL_TS"
  --init-actor-checkpoint "$A_CKPT" --init-noise-std 0.05
  --episode-length-s 5.0 --lift-target-z-above-init 0.10 --lift-delta-z 0.10
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad
  --contact-gate-stability-rewards --contact-min-weight 15.0
  --object-xy-drift-weight=-30.0 --object-orientation-drift-weight=-20.0
  # --- GUARDRAILS RESTORED (this is the v1 fix) ---
  --enable-lift-terminations
  --term-object-drop 0.02 --term-tip-lost-steps 3 --term-finger-slip "$FINGER_SLIP_TOL"
  --term-object-slip-xy "$SLIP_XY_TOL" --term-object-slip-yaw "$SLIP_YAW_TOL"
  --lift-phase-start-step 40
  # --- the ONE intended change: nudge A's delivered grip onto B10's (65-d A space) ---
  --handoff-target-bank "$BANK" --handoff-target-weight "$WEIGHT"
  --handoff-target-seam-lo 33 --handoff-target-seam-hi 37 --handoff-target-qpos-tol "$QPOS_TOL"
  --tag "$TAG" --no-wandb
)
c=$(mktemp -d)
LOG="${LOG:-$ROOT/branchB_v2_${TAG}.trainer.log}"
echo "[branchB-v2] TAG=$TAG ts=$TOTAL_TS weight=$WEIGHT finger_slip_tol=$FINGER_SLIP_TOL"
echo "[branchB-v2] WARP_CACHE=$c  trainer-log=$LOG"

# Trainer in its own process group so the watchdog can kill the whole tree.
WARP_CACHE_PATH="$c" MUJOCO_GL=egl setsid uv run --extra rl --extra gpu \
  python "$ROOT/scripts/rl_train_cube.py" "${ARGS[@]}" > "$LOG" 2>&1 &
RUNPID=$!
echo "[branchB-v2] trainer pid(group)=$RUNPID"

# --- collapse watchdog: A must keep lifting (object_height) or we abort ---------
COLLAPSE_Z=0.045        # healthy run holds ~0.09; v1 collapse sat at ~0.014
GUARD_FROM_ITER=24      # only judge after the scripted lift + warmup settles
while kill -0 "$RUNPID" 2>/dev/null; do
  sleep 30
  tail -c 400 "$LOG" 2>/dev/null | grep -q . || true
  IT=$(grep -oE "Learning iteration [0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
  OH=$(grep "lift_height/object_height" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
  [ -n "$IT" ] && [ -n "$OH" ] || continue
  if [ "$IT" -ge "$GUARD_FROM_ITER" ] 2>/dev/null && awk "BEGIN{exit !($OH < $COLLAPSE_Z)}"; then
    echo "[watchdog] COLLAPSE: object_height=$OH < $COLLAPSE_Z at iter $IT — A stopped lifting. Killing."
    touch "${LOG}.COLLAPSED"
    kill -TERM -"$RUNPID" 2>/dev/null; sleep 5; kill -KILL -"$RUNPID" 2>/dev/null
    break
  fi
done
wait "$RUNPID" 2>/dev/null; RC=$?
if [ -e "${LOG}.COLLAPSED" ]; then
  echo "[branchB-v2] ABORTED — grasp collapsed (see ${LOG}.COLLAPSED). Lower WEIGHT or raise FINGER_SLIP_TOL and retry."
else
  echo "[branchB-v2] DONE rc=$RC — A held the grasp through training."
  echo "[branchB-v2] NEXT: eval the seam with the FROZEN B10:"
  echo "  WARP_CACHE_PATH=\$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python \\"
  echo "    scripts/rl_demo_handoff_continuous.py --policy-a results/rl/${TAG}/tensorboard/model_<N>.pt \\"
  echo "    --policy-b <B10_ckpt> --output docs/rl/videos/reorient/handoff_branchB_v2.mp4 \\"
  echo "    --handoff-step 45 --total-steps 240   # SUCCESS = min object-center z > 0.05"
fi
