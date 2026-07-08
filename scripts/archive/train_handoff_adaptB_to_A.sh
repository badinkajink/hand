#!/bin/bash
# Adapt B to A's delivery — the symmetric move after branch B (move-A) stalled.
# =============================================================================
# Branch B (un-freeze A toward B10's grip) trained clean but A only migrated its
# grip PARTWAY and the seam stayed open (min-z 0.005-0.007 vs frozen B10). A
# resists fully adopting B10's grip. So instead: LEAVE A FROZEN (it's a good
# grasper) and make B robust to A's ACTUAL delivered grip+pose.
#
# Mechanism: B's env consumes a handoff-state-bank only in SKIP-LIFT mode
# (env_cfg.py: `if cfg.skip_lift_phase and cfg.handoff_state_bank`). The bank
# (recorded by rl_record_handoff_states.py from FROZEN A model_500 in the
# normal-lift env) holds A's real object pose+vel AND robot qpos (the grip). B
# resets from sampled bank states each episode -> B trains to reorient starting
# from A's real grip, in-distribution by construction.
#
# This is "B6 (P3 statebank) done right": warmstart the BEST reorienter B4
# (lateral-only, held-cos 0.988) instead of B6's signed+critic 0.979 base, and a
# freshly-recorded + verified bank. Mode-consistent (B4 is skip-lift native).
# KNOWN RISK (RESEARCH_STATE): skip-lift obs schedule (lift-command/ref_object_pose)
# can differ from the normal-lift continuous deploy even when the STATE matches —
# that's the obs-OOD that hurt B6. If this drops at the seam too, the next step is
# a normal-lift onset-grip injection (needs new code). This run tests cheaply
# whether the better warmstart + real bank is enough.
#
# Launch (detached): nohup setsid bash scripts/train_handoff_adaptB_to_A.sh > adaptB.log 2>&1 </dev/null & disown
# Knobs: TOTAL_TS (20M), SMOKE=1 (1M), BANK, B_CKPT.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
B_CKPT="${B_CKPT:-$ROOT/results/rl/b04_20260603-1746-policyB_p2_lateral_only/tensorboard/model_541.pt}"  # B4
BANK="${BANK:-$ROOT/results/rl/handoff_state_bank_A_s40.npz}"
TOTAL_TS=${TOTAL_TS:-20000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-policyB_adaptToA_bankA_s40}"
for f in "$B_CKPT" "$BANK" "$ROOT/$MORPH/best_rollout.npz"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

# B4's EXACT reorient recipe (skip-lift, target-axis 100 / progress 300, lateral -8),
# warmstart B4 actor+critic, + the A-delivery bank (activates bank reset in skip-lift).
ARGS=(
  --morphology-run "$MORPH" --object-body-name screwdriver_medium
  --num-envs 3072 --total-timesteps "$TOTAL_TS"
  --init-actor-checkpoint "$B_CKPT" --warmstart-critic
  --episode-length-s 4.0 --lift-target-z-above-init 0.0
  --skip-lift-phase --skip-lift-drop-offset 0.005
  --handoff-state-bank "$BANK"
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0
  --target-axis-progress-weight 300.0 --reorient-start-step 10
  --finger-residual-active-from-step 0 --contact-min-weight 15.0
  --lateral-drift-weight=-8.0 --lateral-drift-deadband 0.01 --lateral-drift-power 2.0
  --tag "$TAG" --no-wandb
)
c=$(mktemp -d)
LOG="${LOG:-$ROOT/logs/adaptB_${TAG}.trainer.log}"
echo "[adaptB] TAG=$TAG ts=$TOTAL_TS bank=$BANK warmstart=B4 WARP_CACHE=$c trainer-log=$LOG"
WARP_CACHE_PATH="$c" MUJOCO_GL=egl setsid uv run --extra rl --extra gpu \
  python "$ROOT/scripts/rl_train_cube.py" "${ARGS[@]}" > "$LOG" 2>&1 &
RUNPID=$!
echo "[adaptB] trainer pid(group)=$RUNPID"

# collapse watchdog. NB SKIP-LIFT: lift_target_z_above_init=0, so object_height is
# measured ABOVE the (already-lifted) spawn — it reads ~0.00 when held and goes
# NEGATIVE (~ -0.10) only if B drops the object toward the floor. So the collapse
# signal here is object_height < -0.06 (NOT the normal-lift <0.045). Grep keeps the sign.
COLLAPSE_Z=-0.06; GUARD_FROM_ITER=30
while kill -0 "$RUNPID" 2>/dev/null; do
  sleep 30
  IT=$(grep -oE "Learning iteration [0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
  OH=$(grep "lift_height/object_height" "$LOG" 2>/dev/null | tail -1 | grep -oE "\-?[0-9.]+$")
  [ -n "$IT" ] && [ -n "$OH" ] || continue
  if [ "$IT" -ge "$GUARD_FROM_ITER" ] 2>/dev/null && awk "BEGIN{exit !($OH < $COLLAPSE_Z)}"; then
    echo "[watchdog] COLLAPSE: object_height=$OH < $COLLAPSE_Z at iter $IT — B dropped. Killing."
    touch "${LOG}.COLLAPSED"; kill -TERM -"$RUNPID" 2>/dev/null; sleep 5; kill -KILL -"$RUNPID" 2>/dev/null; break
  fi
done
wait "$RUNPID" 2>/dev/null; RC=$?
[ -e "${LOG}.COLLAPSED" ] && echo "[adaptB] ABORTED — B dropped the object." \
  || echo "[adaptB] DONE rc=$RC. NEXT: continuous-handoff eval vs FROZEN A (min-z>0.05 = seam closed)."
