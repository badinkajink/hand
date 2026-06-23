#!/bin/bash
# Retrain Policy A (lift + deliver) on a NEW morphology — for the full A->B handoff.
# =============================================================================
# Replicates a01's lift recipe (the frozen Policy A) on a given morphology, warmstart a01.
# Needed because changing the hand geometry changes the grasp, so A must relearn the lift/deliver.
# Pairs with train_reorient_on_morph.sh (B); together they give a full handoff on the new design.
#
# Launch (detached): nohup setsid bash scripts/train_A_on_morph.sh > A_morph.run.log 2>&1 </dev/null & disown
# SMOKE=1 -> 1M ts. Knobs: MORPH_RUN, WARMSTART, TOTAL_TS, NUM_ENVS, TAG.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH_RUN="${MORPH_RUN:-$ROOT/results/phase1/morph_winner/thumb_opposed_balanced}"
WARMSTART="${WARMSTART:-$ROOT/results/rl/a01_20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt}"
TOTAL_TS=${TOTAL_TS:-30000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-policyA_thumbBalanced}"
for f in "$MORPH_RUN/best_rollout.npz" "$MORPH_RUN/frozen_scene.xml" "$WARMSTART"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

# a01's lift recipe (normal lift, no reorient), on the new morphology.
ARGS=(
  --morphology-run "$MORPH_RUN" --object-body-name screwdriver_medium
  --num-envs "${NUM_ENVS:-2048}" --total-timesteps "$TOTAL_TS"
  --init-actor-checkpoint "$WARMSTART" --warmstart-critic --init-noise-std 0.05
  --episode-length-s 1.4 --lift-target-z-above-init 0.05 --lift-delta-z 0.05
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad
  --contact-gate-stability-rewards --contact-min-weight 15.0
  --enable-lift-terminations --lift-phase-start-step 40
  --term-object-drop 0.02 --term-tip-lost-steps 3 --term-finger-slip 0.3
  --term-object-slip-xy 0.015 --term-object-slip-yaw 0.5
  --tag "$TAG" --no-wandb
)
read -r -a _extra <<< "${EXTRA_ARGS:-}"; ARGS+=("${_extra[@]}")
c=$(mktemp -d)
LOG="${LOG:-$ROOT/A_morph_${TAG}.trainer.log}"
echo "[A_morph] TAG=$TAG ts=$TOTAL_TS  MORPH=$MORPH_RUN  warmstart-a01=$WARMSTART  WARP_CACHE=$c"
WARP_CACHE_PATH="$c" MUJOCO_GL=egl setsid uv run --extra rl --extra gpu \
  python "$ROOT/scripts/rl_train_cube.py" "${ARGS[@]}" > "$LOG" 2>&1 &
RUNPID=$!
echo "[A_morph] trainer pid(group)=$RUNPID"

# normal-lift watchdog: held object_height ~0.09; drop < 0.045 (gotcha #10).
COLLAPSE_Z=${COLLAPSE_Z:-0.045}; GUARD_FROM_ITER=${GUARD_FROM_ITER:-40}
while kill -0 "$RUNPID" 2>/dev/null; do
  sleep 30
  IT=$(grep -oE "Learning iteration [0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
  OH=$(grep "lift_height/object_height" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
  [ -n "$IT" ] && [ -n "$OH" ] || continue
  if [ "$IT" -ge "$GUARD_FROM_ITER" ] 2>/dev/null && awk "BEGIN{exit !($OH < $COLLAPSE_Z)}"; then
    echo "[watchdog] COLLAPSE: object_height=$OH < $COLLAPSE_Z at iter $IT — A stopped lifting. Killing."
    touch "${LOG}.COLLAPSED"; kill -TERM -"$RUNPID" 2>/dev/null; sleep 5; kill -KILL -"$RUNPID" 2>/dev/null; break
  fi
done
wait "$RUNPID" 2>/dev/null; RC=$?
[ -e "${LOG}.COLLAPSED" ] && echo "[A_morph] ABORTED — A failed to lift on the new morphology." \
  || echo "[A_morph] DONE rc=$RC — A retrained on the new morphology."
