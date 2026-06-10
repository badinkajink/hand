#!/bin/bash
# LIVE-A RESET — train Policy B from Policy A's ORGANIC delivery (zero teleport).
# =============================================================================
# WHY: every teleport-based B-side approach saturated at continuous-handoff
# min-z ~0.002-0.012 (skip-lift bank / normal-lift onset inject, static OR
# Markov-complete — all confirmed below the 0.05 "held" bar). The injection
# paradigm is CAPPED. The one remaining train/deploy difference is the teleport
# itself: B's training reset jumps to a snapshot, while at deploy A runs LIVE
# into the seam (real contact-solver warmstart / contact-force ramp / last_action
# that NO instantaneous teleport reproduces).
#
# This run removes the teleport. Frozen Policy A drives the env LIVE for steps
# 0..ONSET of every B episode (real physics), then B's PPO rollout begins at the
# organic seam; the A-driven pre-onset steps are MASKED from the PPO update
# (advantage-masking — see src/morphohand/rl/live_a_runner.py). Env is NORMAL-lift
# to 0.10 (obs schedule AND lift height == the continuous-handoff deploy).
#
# Single-variable vs train_handoff_onset_inject.sh: same warmstart (B10), same
# reorient recipe, same schedule — ONLY the reset changes (live-A vs inject-bank).
#
# CONFIG-PARITY (load-bearing — this bit the first run): the TRAINING env must
# match the continuous-handoff DEPLOY env (rl_demo_handoff_continuous.py), or B
# is OOD at eval. The trainer defaults finger_residual_scale=0.2 / easing=linear
# / contact_gate=off, but B10 AND the deploy demo use 0.5 / ease_out_quad / on.
# The first run trained at 0.2 -> B held in training but the 0.5 eval applied its
# residuals 2.5x too large -> instant seam collapse (an artifact, not a failure;
# re-eval at 0.2 held post-handoff min-z 0.110, cos 0.75). So pin all three here.
#
# Launch (detached): nohup setsid bash scripts/train_handoff_liveA_reset.sh > liveA.run.log 2>&1 </dev/null & disown
# SMOKE=1 -> 1M ts (~13 iters, ~3 min) supervised sanity. Knobs: TOTAL_TS, B_CKPT,
# A_CKPT, ONSET_STEP, REORIENT_START, TAG.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
A_CKPT="${A_CKPT:-$ROOT/results/rl/20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt}"
B_CKPT="${B_CKPT:-$ROOT/results/rl/20260604-1642-policyB_holdonlyws_repro/tensorboard/model_541.pt}"  # B10
ONSET_STEP=${ONSET_STEP:-40}; REORIENT_START=${REORIENT_START:-45}
TOTAL_TS=${TOTAL_TS:-20000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-policyB_liveAreset_fromB10}"
for f in "$A_CKPT" "$B_CKPT" "$ROOT/$MORPH/best_rollout.npz"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

# B10's reorient recipe (target-axis 100 / progress 300, lateral -8, contact-min 15),
# NORMAL-lift to 0.10 (lift-delta-z 0.1 so A lifts to B's in-distribution height,
# matching the continuous-handoff deploy), B10's relaxed slip guards, warmstart
# B10 actor+critic, + LIVE Policy A driving steps 0..ONSET (residual active from 0
# so A's action lifts; reorient reward gated to REORIENT_START as a catch grace).
ARGS=(
  --morphology-run "$MORPH" --object-body-name screwdriver_medium
  --num-envs 3072 --total-timesteps "$TOTAL_TS"
  --init-actor-checkpoint "$B_CKPT" --warmstart-critic
  --live-a-checkpoint "$A_CKPT" --live-a-onset "$ONSET_STEP"
  --episode-length-s 5.0 --lift-target-z-above-init 0.1 --lift-delta-z 0.1
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad --contact-gate-stability-rewards
  --lift-phase-start-step "$ONSET_STEP"
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
LOG="${LOG:-$ROOT/liveA_${TAG}.trainer.log}"
echo "[liveA] TAG=$TAG ts=$TOTAL_TS onset=$ONSET_STEP reorient_start=$REORIENT_START"
echo "[liveA] A=$A_CKPT warmstart=B10 WARP_CACHE=$c trainer-log=$LOG"
WARP_CACHE_PATH="$c" MUJOCO_GL=egl setsid uv run --extra rl --extra gpu \
  python "$ROOT/scripts/rl_train_cube.py" "${ARGS[@]}" > "$LOG" 2>&1 &
RUNPID=$!
echo "[liveA] trainer pid(group)=$RUNPID"

# NORMAL-lift collapse watchdog (gotcha #10): object_height is height ABOVE init,
# reported as the EPISODE AVERAGE. Steps 0..onset are A's live lift (ramping
# 0->0.10), so the mean sits ~0.07-0.09 when B holds post-seam; a true floor drop
# gives ~0.01. Threshold well below the operating band. Default 0.030 / from iter 50.
COLLAPSE_Z=${COLLAPSE_Z:-0.030}; GUARD_FROM_ITER=${GUARD_FROM_ITER:-50}
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
[ -e "${LOG}.COLLAPSED" ] && echo "[liveA] ABORTED — B dropped the object." \
  || echo "[liveA] DONE rc=$RC. NEXT: continuous-handoff eval vs FROZEN A (min-z>0.05 = seam closed)."
