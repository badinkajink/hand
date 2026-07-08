#!/bin/bash
# GENTLE LOW-FORCE REORIENT (Policy B) — the verticality-relaxed regime (2026-06-22)
# =============================================================================
# USER DIRECTIVE (2026-06-22): "i don't care about a close-to-vertical reorientation,
# i just want a SMOOTH, LOW-FORCE grasp and reorient." This is the first run to act on
# that reprioritisation.
#
# WHY THIS IS GENUINELY UNTRIED: the ENTIRE b32->b34 arc held target_axis_alignment at
# +100 (alpha 4.0, sharp) — i.e. it MAXIMISED verticality. The b34 force-floor sweep then
# proved that, while *pushing to vertical*, fingertip force has a hard floor ~6.6 N and the
# corrective jitter is load-bearing (it stabilises a marginal grip fighting to the last ~17°).
# Nobody has asked: if we STOP demanding vertical, does the grip relax + smooth out on its own?
# A gentle partial reorient (~45-60 deg) should not need the tense corrective clamp that the
# last degrees to vertical require — so force AND jitter may both fall for free.
#
# SINGLE COHERENT CHANGE-SET vs the b34_t20 warmstart (the gentlest seam-survivor so far:
# 6.6 N, ang-jerk 76, held-cos 0.78), all toward "gentle + smooth + low-force":
#   - RELAX VERTICALITY:   target-axis-weight 100 -> 40, alpha 4.0 -> 1.5 (WIDE basin =
#                          partial tilt is already rewarding = policy can stop early),
#                          progress-weight 300 -> 120 (less drive to keep rotating up).
#   - LOWER FORCE:         grip-force REWARD 6 -> 2 (stop rewarding a hard press); KEEP the
#                          over-grip PENALTY and push it toward the floor (thresh 2.5, w -6).
#   - SMOOTHER:            lateral-drift -8 -> -12 (the PROVEN smoother: P2/B4's lateral
#                          penalty halved object-jerk; it acts as a regulariser, not a
#                          de-centerer). Keep action-rate/ang-acc at the b32 level.
# EVERYTHING else == the b34/live-A lineage: live-A reset @ onset 40 / scale 0.2 / no blend /
# contact-gate OFF (gotcha #13 — eval at the SAME scale 0.2), all grasp guardrails kept.
#
# JUDGE on the FULL diagnostics (rl_demo_handoff_continuous.py --finger-residual-scale 0.2
# --finger-close-easing linear --no-contact-gate): POST-HANDOFF min-z (hold), lin/ang JERK
# (smooth), fingertip CONTACT FORCE (low), and held-cos (we EXPECT this to drop — that's the
# point; we're buying smooth+gentle WITH it). NOT cos/min-z alone (b31 lesson).
#   WIN  = holds (min-z > 0.05) at materially lower force AND lower ang-jerk than b34_t20,
#          even at a lower held-cos. That is exactly what the user asked for.
#
# Launch (detached): nohup setsid bash scripts/train_gentle_lowforce_B.sh > logs/gentleB.run.log 2>&1 </dev/null & disown
# SMOKE=1 -> 1M ts (~13 iters) plumbing/NaN/term-fires check. Knobs: ALIGN_W, ALPHA, PROG_W,
#            GRIP_REW, PEN_THRESH, PEN_WEIGHT, LATERAL_W, TOTAL_TS, NUM_ENVS, TAG.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
A_CKPT="${A_CKPT:-$ROOT/results/rl/a01_20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt}"
# warmstart = b34_t20 (gentlest + lowest-force live-A seam-survivor)
B_CKPT="${B_CKPT:-$ROOT/results/rl/bx_20260613-0909-policyB_b33iter_forcefloor_t20/tensorboard/model_405.pt}"

# live-A reset wiring (b34 lineage: onset 40, no blend, scale 0.2, contact-gate OFF)
ONSET_STEP=${ONSET_STEP:-40}; BLEND=${BLEND:-0}
LIFT_TERM_START=$(( ONSET_STEP + BLEND ))          # 40
REORIENT_START=$(( ONSET_STEP + BLEND + 5 ))       # 45
TOTAL_TS=${TOTAL_TS:-20000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-policyB_gentleLowforce}"

# --- the verticality-relaxation + low-force + smoothing knobs ---
ALIGN_W=${ALIGN_W:-40.0}        # was 100 — relax verticality
ALPHA=${ALPHA:-1.5}             # was 4.0 — WIDE basin: partial tilt already rewards
PROG_W=${PROG_W:-120.0}         # was 300 — less drive to keep rotating up
GRIP_REW=${GRIP_REW:-2.0}       # was 6   — stop rewarding a hard press (saturates at 3 N)
PEN_THRESH=${PEN_THRESH:-2.5}   # over-grip penalty knee (push force toward the floor)
PEN_WEIGHT=${PEN_WEIGHT:--6.0}
PEN_SCALE=${PEN_SCALE:-4.0}
LATERAL_W=${LATERAL_W:--12.0}   # was -8 — the proven smoothing regulariser, raised

for f in "$A_CKPT" "$B_CKPT" "$ROOT/$MORPH/best_rollout.npz"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

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
  # stability / drift  (lateral raised = the smoother)
  --object-xy-drift-weight=-3.0 --object-orientation-drift-weight=-3.0 --finger-drift-weight=-2.0
  --lateral-drift-weight="$LATERAL_W" --lateral-drift-deadband 0.01 --lateral-drift-power 2.0
  # reorient — RELAXED verticality (wide basin, lower weights)
  --enable-target-axis-reward --target-axis-weight "$ALIGN_W" --target-axis-alpha "$ALPHA"
  --target-axis-progress-weight "$PROG_W" --contact-min-weight 15.0
  # smoothness (gated at reorient onset, b32 level)
  --action-rate-weight=-0.05 --object-ang-acc-weight=-0.05 --object-ang-acc-phase-start-step 45
  # grip REWARD lowered + the over-grip PENALTY pushed toward the floor
  --grip-force-weight "$GRIP_REW" --grip-force-max 3.0 --grip-force-reduce mean
  --brace-force-weight 4.0 --brace-align-thresh 0.7 --brace-max-force 3.0
  --brace-distance-weight 12.0 --brace-distance-scale 0.025
  --grip-force-penalty-weight="$PEN_WEIGHT" --grip-force-penalty-thresh "$PEN_THRESH"
  --grip-force-penalty-scale "$PEN_SCALE" --grip-force-penalty-reduce "${PEN_REDUCE:-mean}"
  --tag "$TAG" --no-wandb
)
# EXTRA_ARGS: append experiment-specific flags (e.g. --grip-force-spread-weight,
# --frozen-scene-xml for the hard-contact variant). B_CKPT overrides the warmstart.
read -r -a _extra <<< "${EXTRA_ARGS:-}"
ARGS+=("${_extra[@]}")
c=$(mktemp -d)
LOG="${LOG:-$ROOT/logs/gentleB_${TAG}.trainer.log}"
echo "[gentleB] TAG=$TAG ts=$TOTAL_TS  ALIGN_W=$ALIGN_W ALPHA=$ALPHA PROG_W=$PROG_W"
echo "[gentleB] GRIP_REW=$GRIP_REW  PENALTY thresh=$PEN_THRESH w=$PEN_WEIGHT reduce=${PEN_REDUCE:-mean}  LATERAL=$LATERAL_W"
echo "[gentleB] warmstart-B=$B_CKPT  A=$A_CKPT  onset=$ONSET_STEP reorient_start=$REORIENT_START"
echo "[gentleB] EXTRA_ARGS=${EXTRA_ARGS:-<none>}"
echo "[gentleB] WARP_CACHE=$c trainer-log=$LOG"
WARP_CACHE_PATH="$c" MUJOCO_GL=egl setsid uv run --extra rl --extra gpu \
  python "$ROOT/scripts/rl_train_cube.py" "${ARGS[@]}" > "$LOG" 2>&1 &
RUNPID=$!
echo "[gentleB] trainer pid(group)=$RUNPID"

# Collapse watchdog (object_height is episode-avg above init; A's live lift 0..onset keeps
# the mean ~0.07-0.09 when B holds, ~0.01 on a true floor drop).
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
[ -e "${LOG}.COLLAPSED" ] && echo "[gentleB] ABORTED — B dropped the object." \
  || echo "[gentleB] DONE rc=$RC. NEXT: rl_demo_handoff_continuous.py --policy-b results/rl/${TAG}/tensorboard/model_<N>.pt --finger-residual-scale 0.2 --finger-close-easing linear --no-contact-gate  (hold + JERK + FORCE vs b34_t20)."
