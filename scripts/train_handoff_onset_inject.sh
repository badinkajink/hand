#!/bin/bash
# Normal-lift ONSET-grip injection — the one untried combination.
# =============================================================================
# WHY: branch-B (move A->B's grip) and adapt-B (skip-lift bank into B) both train
# clean but leave the seam open. adapt-B PROVED the binding constraint is the
# OBSERVATION schedule, not the state: its bank only fires in skip-lift, so B still
# trained under the skip-lift obs trajectory (OOD vs the normal-lift deploy). This
# run closes that gap by matching BOTH at once:
#   - train B in the NORMAL-lift env  -> obs schedule == deploy
#   - at the handoff onset, inject A's REAL delivered state (object pose+vel + grip)
#     from the bank via a step-mode event (mjlab_terms.inject_handoff_bank_at_onset)
#                                       -> state == deploy
# Warmstart B10 (holdonlyws_repro, normal-lift native, already reorients 0.977 but
# DROPS in the continuous handoff ONLY because it never saw A's real onset state).
# The injection targets exactly that gap; no obs-frame mismatch, no dim surgery.
#
# Mechanism is NEW code (committed alongside this script):
#   env_cfg.handoff_onset_bank / handoff_onset_step  +  the step-mode event.
# Distinct from --handoff-state-bank (skip-lift reset-time spawn).
#
# SCHEDULE (tunable): onset/inject at step 40 (matches the s40 bank + deploy
# handoff@40); B's residual active from 40 (B controls at the seam, like deploy);
# reorient reward from 45 (5-step grace to stabilize the catch before tilting).
#
# Launch (detached): nohup setsid bash scripts/train_handoff_onset_inject.sh > onset.run.log 2>&1 </dev/null & disown
# Knobs: TOTAL_TS (20M), SMOKE=1 (1M), BANK, B_CKPT, ONSET_STEP, REORIENT_START.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
B_CKPT="${B_CKPT:-$ROOT/results/rl/20260604-1642-policyB_holdonlyws_repro/tensorboard/model_541.pt}"  # B10
BANK="${BANK:-$ROOT/results/rl/handoff_state_bank_A_s40.npz}"
ONSET_STEP=${ONSET_STEP:-40}; REORIENT_START=${REORIENT_START:-45}
TOTAL_TS=${TOTAL_TS:-20000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-policyB_onsetInject_bankA_s40}"
for f in "$B_CKPT" "$BANK" "$ROOT/$MORPH/best_rollout.npz"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

# B10's reorient recipe (target-axis 100 / progress 300, lateral -8, contact-min 15),
# NORMAL-lift (no --skip-lift-phase; lift to z+0.1, episode 5 s), B10's RELAXED slip
# guards (object_drop tight 0.02; slip/finger relaxed so the migrating grip survives),
# warmstart B10 actor+critic, + the NEW onset-injection of A's real delivery bank.
ARGS=(
  --morphology-run "$MORPH" --object-body-name screwdriver_medium
  --num-envs 3072 --total-timesteps "$TOTAL_TS"
  --init-actor-checkpoint "$B_CKPT" --warmstart-critic
  --episode-length-s 5.0 --lift-target-z-above-init 0.1
  --handoff-onset-bank "$BANK" --handoff-onset-step "$ONSET_STEP"
  --lift-phase-start-step "$ONSET_STEP"
  --finger-residual-active-from-step "$ONSET_STEP"
  --reorient-start-step "$REORIENT_START"
  --enable-lift-terminations
  --term-object-drop 0.02 --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0
  --term-finger-slip 100.0
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0
  --target-axis-progress-weight 300.0 --contact-min-weight 15.0
  --lateral-drift-weight=-8.0 --lateral-drift-deadband 0.01 --lateral-drift-power 2.0
  --tag "$TAG" --no-wandb
)
c=$(mktemp -d)
LOG="${LOG:-$ROOT/onset_${TAG}.trainer.log}"
echo "[onset] TAG=$TAG ts=$TOTAL_TS onset_step=$ONSET_STEP reorient_start=$REORIENT_START"
echo "[onset] bank=$BANK warmstart=B10 WARP_CACHE=$c trainer-log=$LOG"
WARP_CACHE_PATH="$c" MUJOCO_GL=egl setsid uv run --extra rl --extra gpu \
  python "$ROOT/scripts/rl_train_cube.py" "${ARGS[@]}" > "$LOG" 2>&1 &
RUNPID=$!
echo "[onset] trainer pid(group)=$RUNPID"

# NORMAL-lift collapse watchdog (gotcha #10): object_height is height ABOVE init, ~0.09
# when held after the lift; a sustained < 0.045 past the warmup = B dropped to the floor.
COLLAPSE_Z=0.045; GUARD_FROM_ITER=30
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
[ -e "${LOG}.COLLAPSED" ] && echo "[onset] ABORTED — B dropped the object." \
  || echo "[onset] DONE rc=$RC. NEXT: continuous-handoff eval vs FROZEN A (min-z>0.05 = seam closed)."
