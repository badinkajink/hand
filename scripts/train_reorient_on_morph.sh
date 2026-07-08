#!/bin/bash
# Skip-lift REORIENTER on a NEW morphology — does the balanced grasp give a balanced grip?
# =============================================================================
# The thumb-opposition sweep found a morphology (thumb +0.02 x, +0.02 len) whose CEM grasp is a
# BALANCED 3-finger tripod (contacts 3.0, persistence 1.0/1.0/1.0, imbalance 0.0) vs the baseline's
# degenerate 2-finger pinch (middle persistence 0.0). This trains a skip-lift reorienter ON that
# morphology, replicating B4's exact recipe (the best skip-lift reorienter, 0.988) + warmstart B4.
# Skip-lift spawns the object pre-gripped in the morphology's CEM grasp, so NO Policy A retrain is
# needed — the cheap, direct test of "balanced grasp -> balanced, low-force reorient grip".
#
# Judge: probe_grip_balance.py on the trained policy's own (faithful) skip-lift env -> per-finger
# force. WIN = all three fingers share load (thumb no longer idle ~1.6 N) at lower per-finger force
# than the old-morphology gentleB (thumb 1.6 / index 8.0 / mid 6.4 N).
#
# Launch (detached): nohup setsid bash scripts/train_reorient_on_morph.sh > logs/reorientMorph.run.log 2>&1 </dev/null & disown
# SMOKE=1 -> 1M ts. Knobs: MORPH_RUN, WARMSTART, TOTAL_TS, NUM_ENVS, TAG.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH_RUN="${MORPH_RUN:-$ROOT/results/phase1/morph_winner/thumb_opposed_balanced}"
WARMSTART="${WARMSTART:-$ROOT/results/rl/b04_20260603-1746-policyB_p2_lateral_only/tensorboard/model_541.pt}"
TOTAL_TS=${TOTAL_TS:-20000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-policyB_reorient_thumbBalanced}"
for f in "$MORPH_RUN/best_rollout.npz" "$MORPH_RUN/frozen_scene.xml" "$WARMSTART"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

# B4's exact skip-lift recipe (signed progress + lateral-only smoother), on the new morphology.
ARGS=(
  --morphology-run "$MORPH_RUN" --object-body-name screwdriver_medium
  --num-envs "${NUM_ENVS:-2048}" --total-timesteps "$TOTAL_TS"
  --init-actor-checkpoint "$WARMSTART" --warmstart-critic --init-noise-std 0.05
  --skip-lift-phase --skip-lift-drop-offset 0.005
  --episode-length-s 4.0 --lift-target-z-above-init 0.0
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad --contact-gate-stability-rewards
  --enable-target-axis-reward --target-axis-weight "${ALIGN_W:-100.0}" --target-axis-alpha "${ALPHA:-4.0}"
  --target-axis-progress-weight "${PROG_W:-300.0}" --reorient-start-step 10 --contact-min-weight 15.0
  --lateral-drift-weight="${LATERAL_W:--8.0}" --lateral-drift-deadband 0.01 --lateral-drift-power 2.0
  --enable-lift-terminations
  --term-object-drop 0.02 --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0
  --term-finger-slip 100.0 --term-tip-lost-steps 10
  --enable-floor-proximity-termination --object-min-z 0.05 --floor-proximity-phase-start-step 10
  --tag "$TAG" --no-wandb
)
read -r -a _extra <<< "${EXTRA_ARGS:-}"   # gentle+spread overrides etc.
ARGS+=("${_extra[@]}")
c=$(mktemp -d)
LOG="${LOG:-$ROOT/logs/reorientMorph_${TAG}.trainer.log}"
echo "[reorientMorph] TAG=$TAG ts=$TOTAL_TS  MORPH=$MORPH_RUN"
echo "[reorientMorph] warmstart-B4=$WARMSTART  WARP_CACHE=$c  log=$LOG"
WARP_CACHE_PATH="$c" MUJOCO_GL=egl setsid uv run --extra rl --extra gpu \
  python "$ROOT/scripts/rl_train_cube.py" "${ARGS[@]}" > "$LOG" 2>&1 &
RUNPID=$!
echo "[reorientMorph] trainer pid(group)=$RUNPID"

# skip-lift watchdog: object_height is ~0.0 held, NEGATIVE on a drop (gotcha #10) -> drop < -0.06.
COLLAPSE_Z=${COLLAPSE_Z:--0.060}; GUARD_FROM_ITER=${GUARD_FROM_ITER:-60}
while kill -0 "$RUNPID" 2>/dev/null; do
  sleep 30
  IT=$(grep -oE "Learning iteration [0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
  OH=$(grep "lift_height/object_height" "$LOG" 2>/dev/null | tail -1 | grep -oE "\-?[0-9.]+$")
  [ -n "$IT" ] && [ -n "$OH" ] || continue
  if [ "$IT" -ge "$GUARD_FROM_ITER" ] 2>/dev/null && awk "BEGIN{exit !($OH < $COLLAPSE_Z)}"; then
    echo "[watchdog] COLLAPSE: object_height=$OH < $COLLAPSE_Z at iter $IT — dropped. Killing."
    touch "${LOG}.COLLAPSED"; kill -TERM -"$RUNPID" 2>/dev/null; sleep 5; kill -KILL -"$RUNPID" 2>/dev/null; break
  fi
done
wait "$RUNPID" 2>/dev/null; RC=$?
[ -e "${LOG}.COLLAPSED" ] && echo "[reorientMorph] ABORTED — dropped." \
  || echo "[reorientMorph] DONE rc=$RC. NEXT: probe_grip_balance.py <run>:model_<N>.pt — per-finger force vs old-morphology gentleB."
