#!/bin/bash
# Verticality ablation from v1 (held-cos gated).
#
# v1 PolicyB holds the cylinder near-vertical (deterministic held cos 0.97); every
# v2 finetune dropped to ~0.83-0.87. v2 stacked several changes at once and was judged
# on reward SUMS that masked the regression. This isolates the cause by finetuning v1
# with ONE change at a time, warmstarting v1 model_2033, v1's exact config otherwise,
# with the alpha curriculum DISABLED (flat sharp alpha, so the warmstart basin isn't
# re-softened). Headline metric is post-hoc deterministic held-cos (scripts/cmp), NOT
# reward sums.
#
# Round 1 (this script): CONTROL vs SIGNED-progress.
#   - control: non-negative progress (v1-faithful) -> should reproduce ~0.97, proving
#     finetuning itself doesn't cost verticality.
#   - signed:  signed progress (penalises backward dcos) -> tests the leading hypothesis
#     that this makes the policy "park" sub-vertical.
# (Round 2, if control holds 0.97: smoothness-only, via SMOOTH=1.)
#
# Run detached:
#   nohup setsid bash scripts/queue_reorient_ablation.sh > ablation_sweep.bg.log 2>&1 < /dev/null & disown
# SMOKE=1 -> 1M ts each.
set -u
ROOT=/home/humanoid/Programs/hand
cd "$ROOT"
MORPH=results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259
WARMSTART="$ROOT/results/rl/20260601-1033-policyB_v1/tensorboard/model_2033.pt"
TOTAL_TS=${TOTAL_TS:-20000000}
SMOKE=${SMOKE:-0}; [ "$SMOKE" = "1" ] && TOTAL_TS=1000000
SMOOTH=${SMOOTH:-0}   # round 2: add 5x smoothness ramp to both runs

for f in "$WARMSTART" "$ROOT/$MORPH/best_rollout.npz"; do [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }; done

# v1-faithful config, alpha curriculum OFF (flat sharp), no lateral/quick/smoothness-curriculum.
COMMON_ARGS=(
  --morphology-run "$MORPH" --object-body-name screwdriver_medium
  --num-envs 1024 --total-timesteps "$TOTAL_TS" --init-actor-checkpoint "$WARMSTART"
  --episode-length-s 4.0 --lift-target-z-above-init 0.0 --lift-delta-z 0.1
  --finger-residual-scale 0.5 --finger-close-easing ease_out_quad
  --object-xy-drift-weight=-3.0 --object-orientation-drift-weight=0.0 --finger-drift-weight=-0.3
  --contact-gate-stability-rewards
  --enable-lift-terminations --lift-phase-start-step 10
  --term-object-slip-xy 0.5 --term-object-slip-yaw 10.0 --term-tip-lost-steps 10 --term-finger-slip 100.0
  --enable-target-axis-reward --target-axis-weight 100.0 --target-axis-alpha 4.0 --reorient-start-step 10
  --contact-min-weight 15.0 --target-axis-progress-weight 300.0 --target-axis-alpha-curriculum-iters 0
  --enable-floor-proximity-termination --object-min-z 0.05 --floor-proximity-phase-start-step 10
  --skip-lift-phase --skip-lift-drop-offset 0.005
  --object-ang-acc-phase-start-step 10 --init-noise-std 0.05 --no-wandb
)
if [ "$SMOOTH" = "1" ]; then
  COMMON_ARGS+=( --action-rate-weight=-0.1 --object-ang-acc-weight=-0.05
                 --action-rate-weight-final=-0.5 --object-ang-acc-weight-final=-0.25
                 --smoothness-curriculum-start-iter 200 --smoothness-curriculum-iters 400 )
else
  COMMON_ARGS+=( --action-rate-weight=-0.1 --object-ang-acc-weight=-0.05 --smoothness-curriculum-iters 0 )
fi

LAST_PID=""
launch() {  # $1=tag $2=extra_flag(may be empty) $3=logfile
  local tag="$1" extra="$2" logf="$3"; local cache; cache=$(mktemp -d)
  echo "[launch] $tag extra='$extra' WARP_CACHE=$cache log=$logf"
  WARP_CACHE_PATH="$cache" MUJOCO_GL=egl \
    uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
      "${COMMON_ARGS[@]}" --tag "$tag" $extra > "$logf" 2>&1 &
  LAST_PID=$!
}
wait_for_ppo(){ local logf="$1" pid="$2"; for _ in $(seq 1 600); do
  grep -q "starting PPO" "$logf" 2>/dev/null && { echo "  -> PPO started"; return 0; }
  kill -0 "$pid" 2>/dev/null || { echo "  -> $pid died"; return 1; }; sleep 2; done; return 1; }

SFX=""; [ "$SMOOTH" = "1" ] && SFX="_smooth"
LOGC="$ROOT/ablation_control${SFX}.log"; LOGS="$ROOT/ablation_signed${SFX}.log"
echo "===== verticality ablation (SMOKE=$SMOKE SMOOTH=$SMOOTH ts=$TOTAL_TS) ====="
# CONTROL: non-negative progress (v1-faithful) -> pass the clamp flag (True)
launch "policyB_abl_control${SFX}" "--target-axis-progress-clamp-negative" "$LOGC"; PC=$LAST_PID
echo "[control] pid=$PC; waiting compile..."; wait_for_ppo "$LOGC" "$PC"
# SIGNED: signed progress -> omit the flag (default False = signed)
launch "policyB_abl_signed${SFX}" "" "$LOGS"; PS=$LAST_PID
echo "[signed] pid=$PS"
echo "[ablation] waiting both..."; wait "$PC"; RC=$?; wait "$PS"; RS=$?
echo "[ablation] DONE control rc=$RC signed rc=$RS"
echo "  control log: $LOGC"; echo "  signed log: $LOGS"
echo "  Next: scripts/cmp held-cos on the two run dirs' final ckpts."
