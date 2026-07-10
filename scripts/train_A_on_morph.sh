#!/bin/bash
# Retrain Policy A (lift + deliver) on a NEW morphology — for the full A->B handoff.
# =============================================================================
# Replicates a01's lift recipe (the frozen Policy A) on a given morphology, warmstart a01.
# Needed because changing the hand geometry changes the grasp, so A must relearn the lift/deliver.
# Pairs with train_reorient_on_morph.sh (B); together they give a full handoff on the new design.
#
# Launch (detached): nohup setsid bash scripts/train_A_on_morph.sh > logs/A_morph.run.log 2>&1 </dev/null & disown
# SMOKE=1 -> 1M ts. Knobs: MORPH_RUN, WARMSTART, TOTAL_TS, NUM_ENVS, TAG.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH_RUN="${MORPH_RUN:-$ROOT/results/phase1/morph_winner/thumb_opposed_balanced}"
WARMSTART="${WARMSTART:-$ROOT/results/rl/a01_20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt}"
TOTAL_TS=${TOTAL_TS:-30000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-policyA_thumbBalanced}"
# WARMSTART=none => train from scratch (no --init-actor-checkpoint). Needed on a new
# morphology whose grip differs enough that a01's baseline-tuned residual EJECTS the object
# (the open-loop CEM grip+scripted-lift already lifts, so residual~0 from scratch lifts too).
FROM_SCRATCH=0; [ "$WARMSTART" = "none" ] && FROM_SCRATCH=1
CHECK=("$MORPH_RUN/best_rollout.npz" "$MORPH_RUN/frozen_scene.xml")
[ "$FROM_SCRATCH" = "0" ] && CHECK+=("$WARMSTART")
for f in "${CHECK[@]}"; do [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

# a01's lift recipe (normal lift, no reorient), on the new morphology. The pinned
# block lives in configs/recipes/a_lift.yaml (--recipe); only run-specific knobs here.
# Explicit flags override recipe values, so the env-var knobs behave as before.
ARGS=(
  --recipe "${RECIPE:-a_lift}"
  --morphology-run "$MORPH_RUN"
  --num-envs "${NUM_ENVS:-2048}" --total-timesteps "$TOTAL_TS"
  --init-noise-std "${INIT_NOISE_STD:-0.05}")
[ "$FROM_SCRATCH" = "0" ] && ARGS+=(--init-actor-checkpoint "$WARMSTART" --warmstart-critic)
ARGS+=(
  --lift-target-z-above-init "${LIFT_DELTA_A:-0.05}" --lift-delta-z "${LIFT_DELTA_A:-0.05}"
  --tag "$TAG"
)
read -r -a _extra <<< "${EXTRA_ARGS:-}"; ARGS+=("${_extra[@]}")
c=$(mktemp -d)
LOG="${LOG:-$ROOT/logs/A_morph_${TAG}.trainer.log}"
# normal-lift collapse watchdog (gotcha #10), now TRAINER-SIDE (morphohand.rl.watchdog):
# held object_height ~0.09; drop < 0.045 aborts and writes ${LOG}.COLLAPSED.
ARGS+=(
  --watchdog-collapse-z "${COLLAPSE_Z:-0.045}" --watchdog-from-iter "${GUARD_FROM_ITER:-40}"
  --watchdog-sentinel "${LOG}.COLLAPSED"
)
echo "[A_morph] TAG=$TAG ts=$TOTAL_TS  MORPH=$MORPH_RUN  warmstart-a01=$WARMSTART  WARP_CACHE=$c"
WARP_CACHE_PATH="$c" MUJOCO_GL=egl uv run --extra rl --extra gpu \
  python "$ROOT/scripts/rl_train_cube.py" "${ARGS[@]}" > "$LOG" 2>&1
RC=$?
if [ -e "${LOG}.COLLAPSED" ]; then
  echo "[A_morph] ABORTED — A failed to lift on the new morphology."
else
  echo "[A_morph] DONE rc=$RC — A retrained on the new morphology."
  # BAKED-IN trajectory-health gate: flag a degenerate lift (late finger / idle finger /
  # drop / jitter / de-centering / over-clamp) that aggregate reward hides. Non-fatal.
  CKPT=$(ls -t "$ROOT"/results/rl/*"$TAG"*/tensorboard/model_*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then
    OFK=""; case "${EXTRA_ARGS:-}" in *open-finger-from-keyframe*) OFK="--open-finger-from-keyframe";; esac
    echo "[A_morph] running trajectory-health gate on $CKPT ..."
    WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu \
      python "$ROOT/scripts/policy_healthcheck.py" --policy "$CKPT" \
      --morphology-run "$MORPH_RUN" --lift-delta "${LIFT_DELTA_A:-0.05}" $OFK \
      --title "$TAG" 2>&1 | grep -E "policy health|PASS|WARN|FAIL|health ->" || true
  fi
fi
