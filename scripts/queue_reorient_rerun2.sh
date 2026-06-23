#!/bin/bash
# Batch-2 re-runs WITH the critic fix, on the clean signed+critic base (best
# reorienter, 0.978 held; NO smoothness penalty since that proved counterproductive).
#   run A: + "quick" mechanisms (success term+bonus, time cost, speed bonus) — re-test
#          the Stage-2 hypothesis now that the critic isn't sabotaging it.
#   run B: + de-centering lateral penalty (-40) — re-test v2.1 (note: deterministic root
#          drift was ~0 for all policies, so this likely no-ops; confirming).
# Both warmstart the signed+critic checkpoint, critic warmstart ON (default),
# alpha-curriculum OFF, no smoothness ramp.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
WARMSTART="${WARMSTART:-$ROOT/results/rl/b03_20260602-1636-policyB_abl_signed/tensorboard/model_405.pt}"
TOTAL_TS=${TOTAL_TS:-15000000}; SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
for f in "$WARMSTART" "$ROOT/$MORPH/best_rollout.npz"; do [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

COMMON_ARGS=(
  --morphology-run "$MORPH" --object-body-name screwdriver_medium
  --num-envs 1024 --total-timesteps "$TOTAL_TS" --init-actor-checkpoint "$WARMSTART"
  --episode-length-s 4.0 --lift-target-z-above-init 0.0 --lift-delta-z 0.1
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad
  --object-xy-drift-weight=-3.0 --object-orientation-drift-weight=0.0 --finger-drift-weight=-0.3
  --contact-gate-stability-rewards --enable-lift-terminations --lift-phase-start-step 10
  --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0 --term-tip-lost-steps 10 --term-finger-slip 100.0
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 --reorient-start-step 10
  --contact-min-weight 15.0 --target-axis-progress-weight 300.0 --target-axis-alpha-curriculum-iters 0
  --enable-floor-proximity-termination --object-min-z 0.05 --floor-proximity-phase-start-step 10
  --skip-lift-phase --skip-lift-drop-offset 0.005
  --action-rate-weight=-0.1 --object-ang-acc-weight=-0.05 --smoothness-curriculum-iters 0
  --object-ang-acc-phase-start-step 10 --init-noise-std 0.05 --no-wandb
)
LAST_PID=""
launch(){ local tag="$1" extra="$2" logf="$3"; local c; c=$(mktemp -d)
  echo "[launch] $tag extra='$extra' log=$logf"
  WARP_CACHE_PATH="$c" MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
    "${COMMON_ARGS[@]}" --tag "$tag" $extra > "$logf" 2>&1 &
  LAST_PID=$!; }
wfp(){ local l="$1" p="$2"; for _ in $(seq 1 600); do grep -q "starting PPO" "$l" 2>/dev/null && return 0; kill -0 "$p" 2>/dev/null || return 1; sleep 2; done; }

QUICK="--enable-alignment-success-termination --success-align-thresh 0.9 --success-hold-steps 10 --success-bonus-weight 30.0 --time-cost-weight=-0.02 --speed-bonus-weight 15.0 --speed-bonus-align-thresh 0.9"
LAT="--lateral-drift-weight=-40.0 --lateral-drift-deadband 0.01 --lateral-drift-power 2.0"
LOGQ="$ROOT/rerun2_quick.log"; LOGL="$ROOT/rerun2_lateral.log"
echo "===== Batch-2 re-run (ts=$TOTAL_TS) warmstart=$WARMSTART ====="
launch "policyB_rr_quick" "$QUICK" "$LOGQ"; PQ=$LAST_PID; echo "[quick] pid=$PQ; compiling..."; wfp "$LOGQ" "$PQ"
launch "policyB_rr_lateral" "$LAT" "$LOGL"; PL=$LAST_PID; echo "[lateral] pid=$PL"
echo "[rerun2] waiting both..."; wait "$PQ"; RQ=$?; wait "$PL"; RL=$?
echo "[rerun2] DONE quick rc=$RQ lateral rc=$RL"
