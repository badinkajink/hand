#!/bin/bash
# B->A CO-REFINEMENT — nudge Policy A's lift so it delivers a grip Policy B reorients better.
# =============================================================================
# The user's "slow gradient updates from B to A": rather than a hand-crafted grip-match
# reward (the old Branch-B proxy), let A be trained by B's ACTUAL downstream reorient reward.
# Mechanism (live_a_runner drive_post): the LEARNER is Policy A, driving the lift 0..onset;
# a FROZEN reorienter B drives >= onset; B-driven steps are masked from the PPO update, but
# the whole reward stream (incl. B's post-onset reorient reward) feeds the return, so A's
# lift steps get discounted DOWNSTREAM credit via GAE. Low LR + fixed schedule = gentle
# ("slow") so A's lift is nudged toward better-for-B delivery, not overwritten.
#
# Order (frozen-first, then co-refine): run AFTER a frozen-A live-reset B exists, so B is a
# competent reorienter of A's delivery; co-refinement then closes the last gap by moving A.
#
# Launch: nohup setsid bash scripts/train_corefine_BtoA.sh > corefine.run.log 2>&1 </dev/null & disown
# SMOKE=1 -> 1M ts (~code-path validation). Knobs: A_CKPT, B_CKPT, MORPH, LIFT_DELTA, LR, ONSET, TAG.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH="${MORPH:-$ROOT/results/phase1/landscape/m05_ik_cem}"
A_CKPT="${A_CKPT:?set A_CKPT to the native Policy A checkpoint (the learner to nudge)}"
B_CKPT="${B_CKPT:?set B_CKPT to the frozen reorienter (drives post-onset)}"
LIFT_DELTA="${LIFT_DELTA:-0.05}"   # MUST match A's training lift height (m05 A = 0.05)
ONSET="${ONSET:-40}"; LR="${LR:-5e-5}"; TOTAL_TS=${TOTAL_TS:-15000000}
SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-corefine_BtoA_m05}"
for f in "$A_CKPT" "$B_CKPT" "$MORPH/best_rollout.npz" "$MORPH/frozen_scene.xml"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

# Normal-lift handoff env (obs==deploy), reorient reward ON (== the post-onset signal that
# gives A downstream credit). Learner=A (warmstart A_CKPT); frozen reorienter=B_CKPT drives
# >= ONSET via --live-a-drive-post. Reorient recipe mirrors the live-A reset (B4/B_m05 recipe).
ARGS=(
  --morphology-run "$MORPH" --object-body-name screwdriver_medium
  --num-envs "${NUM_ENVS:-3072}" --total-timesteps "$TOTAL_TS"
  --init-actor-checkpoint "$A_CKPT" --warmstart-critic
  --live-a-checkpoint "$B_CKPT" --live-a-onset "$ONSET" --live-a-drive-post
  --learning-rate "$LR" --lr-schedule fixed
  --episode-length-s 5.0 --lift-target-z-above-init "$LIFT_DELTA" --lift-delta-z "$LIFT_DELTA"
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad
  --lift-phase-start-step "$ONSET" --reorient-start-step "$((ONSET + 5))"
  --enable-lift-terminations
  --term-object-drop 0.02 --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0
  --term-finger-slip 100.0 --term-tip-lost-steps 3
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0
  --target-axis-progress-weight 300.0 --contact-min-weight 15.0
  --lateral-drift-weight=-8.0 --lateral-drift-deadband 0.01 --lateral-drift-power 2.0
  --contact-gate-stability-rewards
  --tag "$TAG" --no-wandb
)
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=( $EXTRA_ARGS )
c=$(mktemp -d)
LOG="${LOG:-$ROOT/corefine_${TAG}.trainer.log}"
echo "[corefine] TAG=$TAG ts=$TOTAL_TS onset=$ONSET LR=$LR lift=$LIFT_DELTA"
echo "[corefine] learner-A=$A_CKPT  frozen-B=$B_CKPT  MORPH=$MORPH  WARP_CACHE=$c"
WARP_CACHE_PATH="$c" MUJOCO_GL=egl setsid uv run --extra rl --extra gpu \
  python "$ROOT/scripts/rl_train_cube.py" "${ARGS[@]}" > "$LOG" 2>&1 &
RUNPID=$!
echo "[corefine] trainer pid(group)=$RUNPID"
wait "$RUNPID" 2>/dev/null; RC=$?
echo "[corefine] DONE rc=$RC. NEXT: eval A_corefined vs A_frozen on continuous handoff (min-z, held-cos)."
