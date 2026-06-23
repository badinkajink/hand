#!/bin/bash
# LOWER-FORCE GRASP (re-open Policy A) — force-penalise A's delivery (2026-06-22)
# =============================================================================
# USER DIRECTIVE (2026-06-22): "i still don't love our best grasp ... i just want a
# SMOOTH, LOW-FORCE grasp and reorient." The user chose to RE-OPEN Policy A this round
# (branch-B territory) to chase a gentler delivery, rather than leave A frozen.
#
# OBJECTIVE (different from branch-B v1/v2): NOT migrating A's grip onto B's pose — instead,
# ask the direct question: can A lift + deliver the screwdriver with LESS fingertip force?
# A was trained force-UNAWARE (no grip term, 1.4 s episodes). The b34 sweep found the
# *fingertip* hold floors ~6.6 N; A's delivery force is unmeasured. We add ONLY the over-grip
# penalty (grip_force_excess, thresh 5 N) and keep A's full lift recipe, on a longer episode
# (5 s) so the penalty bites during the HOLD, not just the catch.
#
# SAFETY (lesson #7 / gotcha #4 — the v1 collapse): NEVER strip A's grasp guardrails when
# adding a finger-perturbing reward (a force penalty pushes fingers to RELEASE). So:
#   - KEEP --enable-lift-terminations + term_object_drop 0.02 + term_tip_lost 3
#     (object-on-floor can never become a non-terminating attractor).
#   - RELAX ONLY the precision guards that fight a grip re-shape: finger-slip 0.3 -> 2.0,
#     object-slip-xy 0.015 -> 0.05, object-slip-yaw 0.5 -> 1.0 (the exact v2 relaxations
#     that gave a clean-training A-finetune).
#   - Baked-in watchdog kills the run if object_height < 0.045 at iter >= 24.
# Worst case = "A keeps its grip, force-penalty does little" (safe, == b34's floor finding on
# the A side); best case = a measurably gentler delivery. Either way it's informative.
#
# JUDGE: (1) does A still lift+hold (object_height ~0.09, no collapse)?  (2) re-measure A's
# delivered fingertip force (scripts/probe_grip_force.py on A's standalone lift) vs the frozen
# a01 baseline — did it drop?  (3) does the gentler A still hand off cleanly to a frozen
# reorienter (rl_demo_handoff_continuous.py --policy-a <new A> --policy-b b34_t20)?
#
# Launch (single run, detached): nohup setsid bash scripts/train_lowforce_A.sh > lowforceA.run.log 2>&1 </dev/null & disown
# SMOKE=1 -> 1M ts plumbing/NaN/penalty-fires check. Knobs: PEN_THRESH, PEN_WEIGHT,
#            FINGER_SLIP_TOL, TOTAL_TS, NUM_ENVS, TAG.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
A_CKPT="${A_CKPT:-$ROOT/results/rl/a01_20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt}"

PEN_THRESH=${PEN_THRESH:-5.0}      # penalise A's fingertip force above 5 N (start here; lower later)
PEN_WEIGHT=${PEN_WEIGHT:--6.0}
PEN_SCALE=${PEN_SCALE:-4.0}
FINGER_SLIP_TOL=${FINGER_SLIP_TOL:-2.0}    # was A's 0.3 — allow the grip to re-shape gentler
SLIP_XY_TOL=${SLIP_XY_TOL:-0.05}           # was 0.015
SLIP_YAW_TOL=${SLIP_YAW_TOL:-1.0}          # was 0.5
TOTAL_TS=${TOTAL_TS:-20000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
TAG="${TAG:-policyA_lowforce}"

[ -e "$A_CKPT" ] && [ -e "$ROOT/$MORPH/best_rollout.npz" ] || { echo "FATAL missing A ckpt or morph"; exit 1; }

# A's deploy-matched lift env (normal lift to 0.10, 65-d A space, contact-gate, residual 0.5),
# guardrails restored, ONLY the precision slip guards relaxed + the over-grip penalty added.
ARGS=(
  --morphology-run "$MORPH" --object-body-name screwdriver_medium
  --num-envs "${NUM_ENVS:-3072}" --total-timesteps "$TOTAL_TS"
  --init-actor-checkpoint "$A_CKPT" --warmstart-critic --init-noise-std 0.05
  --episode-length-s 5.0 --lift-target-z-above-init 0.10 --lift-delta-z 0.10
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad
  --contact-gate-stability-rewards --contact-min-weight 15.0
  --object-xy-drift-weight=-30.0 --object-orientation-drift-weight=-20.0
  # --- GUARDRAILS RESTORED; only precision-slip relaxed (lesson #7) ---
  --enable-lift-terminations
  --term-object-drop 0.02 --term-tip-lost-steps 3 --term-finger-slip "$FINGER_SLIP_TOL"
  --term-object-slip-xy "$SLIP_XY_TOL" --term-object-slip-yaw "$SLIP_YAW_TOL"
  --lift-phase-start-step 40
  # --- the ONE intended change: penalise over-grip force during lift+hold ---
  --grip-force-penalty-weight="$PEN_WEIGHT" --grip-force-penalty-thresh "$PEN_THRESH"
  --grip-force-penalty-scale "$PEN_SCALE" --grip-force-penalty-reduce mean
  --tag "$TAG" --no-wandb
)
c=$(mktemp -d)
LOG="${LOG:-$ROOT/lowforceA_${TAG}.trainer.log}"
echo "[lowforceA] TAG=$TAG ts=$TOTAL_TS  PENALTY thresh=$PEN_THRESH w=$PEN_WEIGHT  finger_slip_tol=$FINGER_SLIP_TOL"
echo "[lowforceA] warmstart-A=$A_CKPT"
echo "[lowforceA] WARP_CACHE=$c trainer-log=$LOG"
WARP_CACHE_PATH="$c" MUJOCO_GL=egl setsid uv run --extra rl --extra gpu \
  python "$ROOT/scripts/rl_train_cube.py" "${ARGS[@]}" > "$LOG" 2>&1 &
RUNPID=$!
echo "[lowforceA] trainer pid(group)=$RUNPID"

# collapse watchdog: A must keep lifting (object_height ~0.09) or abort (v1 collapse sat ~0.014).
COLLAPSE_Z=${COLLAPSE_Z:-0.045}; GUARD_FROM_ITER=${GUARD_FROM_ITER:-24}
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
if [ -e "${LOG}.COLLAPSED" ]; then
  echo "[lowforceA] ABORTED — grasp collapsed. Raise PEN_THRESH or FINGER_SLIP_TOL, or lower |PEN_WEIGHT|."
else
  echo "[lowforceA] DONE rc=$RC — A held the grasp. NEXT: probe_grip_force.py on the new A vs a01 baseline; then seam-eval with frozen b34_t20."
fi
