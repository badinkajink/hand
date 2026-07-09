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
# Launch (detached): nohup setsid bash scripts/train_handoff_liveA_reset.sh > logs/liveA.run.log 2>&1 </dev/null & disown
# SMOKE=1 -> 1M ts (~13 iters, ~3 min) supervised sanity. Knobs: TOTAL_TS, B_CKPT,
# A_CKPT, ONSET_STEP, REORIENT_START, TAG.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH="${MORPH:-results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259}"
A_CKPT="${A_CKPT:-$ROOT/results/rl/a01_20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt}"
B_CKPT="${B_CKPT:-$ROOT/results/rl/b10_20260604-1642-policyB_holdonlyws_repro/tensorboard/model_541.pt}"  # B10
# BLEND>0 = SEAM RAMP-IN (breaks the B4 catch-22): after onset, blend
# alpha*B+(1-alpha)*A over BLEND steps (alpha 0->1) so a reorient-from-step-0
# policy (B4) eases into A's grip instead of shocking it and dropping before the
# reorient reward fires. Those blend steps are masked from PPO too (B trains only
# on fully-B steps >= onset+BLEND). With BLEND set, reorient grading should begin
# at/after B takes full control: REORIENT_START defaults to onset+BLEND+5.
ONSET_STEP=${ONSET_STEP:-40}; BLEND=${BLEND:-0}
# Terminations (tip_lost/drop/floor) engage at LIFT_TERM_START. With a seam
# ramp-in, engaging at onset kills every episode via tip_lost at ~onset+10 —
# BEFORE B reaches full control (keep_from=onset+BLEND) — so no trainable steps
# ever accumulate (mask frac stuck at 1.0). Default it to keep_from so A's lift +
# the ramp run termination-free (the object stays up there) and terminations
# engage exactly when B takes full control. Legacy hard-onset (BLEND=0) => onset.
LIFT_TERM_START=${LIFT_TERM_START:-$(( ONSET_STEP + BLEND ))}
REORIENT_START=${REORIENT_START:-$(( ONSET_STEP + BLEND + 5 ))}
TOTAL_TS=${TOTAL_TS:-20000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-policyB_liveAreset_fromB10}"
for f in "$A_CKPT" "$B_CKPT" "$ROOT/$MORPH/best_rollout.npz"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

# B10's reorient recipe (target-axis 100 / progress 300, lateral -8, contact-min 15),
# NORMAL-lift to 0.10 (lift-delta-z 0.1 so A lifts to B's in-distribution height,
# matching the continuous-handoff deploy), B10's relaxed slip guards, warmstart
# B10 actor+critic, + LIVE Policy A driving steps 0..ONSET (residual active from 0
# so A's action lifts; reorient reward gated to REORIENT_START as a catch grace).
# The pinned block lives in configs/recipes/b_liveA.yaml (--recipe; RECIPE=b_liveA_imit
# adds the design-neutral imitation prior). Explicit flags override recipe values,
# so the env-var knobs behave as before.
ARGS=(
  --recipe "${RECIPE:-b_liveA}"
  --morphology-run "$MORPH"
  --num-envs "${NUM_ENVS:-3072}" --total-timesteps "$TOTAL_TS"
  --init-actor-checkpoint "$B_CKPT"
  --live-a-checkpoint "$A_CKPT" --live-a-onset "$ONSET_STEP" --live-a-blend-steps "$BLEND"
  --lift-target-z-above-init "${LIFT_DELTA:-0.1}" --lift-delta-z "${LIFT_DELTA:-0.1}"
  --finger-residual-scale "${RESID_SCALE:-0.5}" --finger-close-easing "${EASING:-ease_out_quad}"
  --lift-phase-start-step "$LIFT_TERM_START"
  --reorient-start-step "$REORIENT_START"
  --term-tip-lost-steps "${TIP_LOST_STEPS:-3}"
  --target-axis-weight "${TAXIS_W:-100.0}"
  --target-axis-progress-weight "${TPROG_W:-300.0}"
  --tag "$TAG"
)
# EXTRA_ARGS: space-separated trainer flags appended verbatim (e.g. a commit bonus
# --success-bonus-weight / --speed-bonus-weight, or --target-axis-alpha-curriculum-iters).
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=( $EXTRA_ARGS )
# Contact-gate stability rewards: ON via the recipe (deploy/B10 parity). Set CONTACT_GATE=0
# to match a policy trained without it (the scale-0.2 live-A lineage from model_270).
[ "${CONTACT_GATE:-1}" = "1" ] || ARGS+=( --no-contact-gate-stability-rewards )
c=$(mktemp -d)
LOG="${LOG:-$ROOT/logs/liveA_${TAG}.trainer.log}"
echo "[liveA] TAG=$TAG ts=$TOTAL_TS onset=$ONSET_STEP blend=$BLEND lift_term_start=$LIFT_TERM_START reorient_start=$REORIENT_START"
echo "[liveA] A=$A_CKPT warmstart-B=$B_CKPT WARP_CACHE=$c trainer-log=$LOG"
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
