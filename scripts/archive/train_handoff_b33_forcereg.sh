#!/bin/bash
# B33 — FORCE-REGULARIZE b32 (is the 11 N death-grip necessary, or learned laziness?)
# =============================================================================
# WHY: b32 is the firmest+smoothest HANDOFF yet (post-handoff min-z 0.1085, held-cos
# 0.891, ang-jerk 113) but holds with an ~11 N fingertip death-grip — the source of
# both the thumb-into-screwdriver penetration AND the residual jitter (high-freq
# corrections of a tense grip). B3 holds the SAME scene gently (low force → seated →
# smooth) but DROPS the continuous handoff. This run asks the one decisive question:
# can b32 hold at 2-3 N if we PENALIZE force above threshold?
#   - still holds at 2-3 N  => jitter + penetration fall out together; B3-like
#     gentleness on a policy that survives the handoff. WIN.
#   - drops                 => the fingertip grip is fundamentally marginal; the real
#     lever is A's delivery/centering, not the grip weights. Clean negative result.
#
# SINGLE VARIABLE vs b32: warmstart from b32/model_405 (actor+critic), keep EVERY b32
# reward/term/curriculum (grip_force +6, brace, smoothness action_rate/ang_acc -0.05,
# live-A reset @ scale 0.2, contact-gate OFF) and ADD ONE term: grip_force_excess —
# a quadratic penalty on fingertip force above PEN_THRESH N (mjlab_terms.grip_force_excess).
# The grip_force REWARD saturates at 3 N, so the two only overlap above PEN_THRESH:
# below it grip is still rewarded, above it extra force now COSTS.
#
# Judge on the FULL diagnostics (rl_demo_handoff_continuous.py): hold (min-z) + smoothness
# (lin/ang jerk) + the new contact-force readout — NOT cos/min-z alone (b31 lesson).
#
# Launch (detached):
#   nohup setsid bash scripts/train_handoff_b33_forcereg.sh > logs/b33.run.log 2>&1 </dev/null & disown
# SMOKE=1 -> 1M ts (~13 iters) sanity. Knobs: PEN_WEIGHT, PEN_THRESH, PEN_SCALE, TOTAL_TS, TAG.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
A_CKPT="${A_CKPT:-$ROOT/results/rl/a01_20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt}"
B_CKPT="${B_CKPT:-$ROOT/results/rl/b32_20260612-1224-policyB_b30iter_gripSmooth_w4/tensorboard/model_405.pt}"  # b32 final

# live-A reset wiring (b32 lineage: onset 40, no blend, scale 0.2, contact-gate OFF)
ONSET_STEP=${ONSET_STEP:-40}; BLEND=${BLEND:-0}
LIFT_TERM_START=$(( ONSET_STEP + BLEND ))          # 40 (== b32 lift_phase_start_step)
REORIENT_START=$(( ONSET_STEP + BLEND + 5 ))       # 45 (== b32 reorient_start_step)
TOTAL_TS=${TOTAL_TS:-20000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-policyB_b32iter_forcereg_w${PEN_WEIGHT_TAG:-6}}"

# ---- the ONLY new lever: over-grip penalty (force above PEN_THRESH N, quadratic) ----
PEN_WEIGHT=${PEN_WEIGHT:--6.0}     # negative; the penalty weight
PEN_THRESH=${PEN_THRESH:-4.0}      # N above which force is penalized (grip reward saturates at 3)
PEN_SCALE=${PEN_SCALE:-4.0}        # N normaliser: penalty = ((force-thresh)/scale)^2

for f in "$A_CKPT" "$B_CKPT" "$ROOT/$MORPH/best_rollout.npz"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

# Full b32 recipe, explicit (independent of trainer defaults), + the new penalty.
ARGS=(
  --morphology-run "$MORPH" --object-body-name screwdriver_medium
  --num-envs "${NUM_ENVS:-2048}" --total-timesteps "$TOTAL_TS"
  --init-actor-checkpoint "$B_CKPT" --warmstart-critic
  --live-a-checkpoint "$A_CKPT" --live-a-onset "$ONSET_STEP" --live-a-blend-steps "$BLEND"
  --episode-length-s 5.0 --lift-target-z-above-init 0.1 --lift-delta-z 0.1
  --finger-residual-scale 0.2 --finger-close-easing linear
  --lift-phase-start-step "$LIFT_TERM_START" --reorient-start-step "$REORIENT_START"
  --enable-lift-terminations
  --term-object-drop 0.02 --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0
  --term-finger-slip 100.0 --term-tip-lost-steps 3
  # stability / drift (b32 values; xy/orient/finger == trainer defaults, explicit for clarity)
  --object-xy-drift-weight=-3.0 --object-orientation-drift-weight=-3.0 --finger-drift-weight=-2.0
  --lateral-drift-weight=-8.0 --lateral-drift-deadband 0.01 --lateral-drift-power 2.0
  # reorient
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0
  --target-axis-progress-weight 300.0 --contact-min-weight 15.0
  # smoothness (b32: gated at onset/reorient step 45)
  --action-rate-weight=-0.05 --object-ang-acc-weight=-0.05 --object-ang-acc-phase-start-step 45
  # grip + brace (b32) — REWARD untouched; the new PENALTY is the single change
  --grip-force-weight 6.0 --grip-force-max 3.0 --grip-force-reduce mean
  --brace-force-weight 4.0 --brace-align-thresh 0.7 --brace-max-force 3.0
  --brace-distance-weight 12.0 --brace-distance-scale 0.025
  # >>> the new lever <<<
  --grip-force-penalty-weight="$PEN_WEIGHT" --grip-force-penalty-thresh "$PEN_THRESH"
  --grip-force-penalty-scale "$PEN_SCALE" --grip-force-penalty-reduce mean
  --tag "$TAG" --no-wandb
)
c=$(mktemp -d)
LOG="${LOG:-$ROOT/logs/b33_${TAG}.trainer.log}"
echo "[b33] TAG=$TAG ts=$TOTAL_TS  PENALTY w=$PEN_WEIGHT thresh=$PEN_THRESH scale=$PEN_SCALE"
echo "[b33] warmstart-B(b32)=$B_CKPT  A=$A_CKPT  onset=$ONSET_STEP reorient_start=$REORIENT_START"
echo "[b33] WARP_CACHE=$c trainer-log=$LOG"
WARP_CACHE_PATH="$c" MUJOCO_GL=egl setsid uv run --extra rl --extra gpu \
  python "$ROOT/scripts/rl_train_cube.py" "${ARGS[@]}" > "$LOG" 2>&1 &
RUNPID=$!
echo "[b33] trainer pid(group)=$RUNPID"

# Collapse watchdog (same as liveA: object_height is episode-avg above init; A's live
# lift 0..onset keeps the mean ~0.07-0.09 when B holds, ~0.01 on a true floor drop).
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
[ -e "${LOG}.COLLAPSED" ] && echo "[b33] ABORTED — B dropped the object (grip was marginal: the lever is A's delivery, not grip weights)." \
  || echo "[b33] DONE rc=$RC. NEXT: continuous-handoff eval (rl_demo_handoff_continuous.py) — hold + jerk + contact force vs b32."
