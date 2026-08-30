#!/bin/bash
# Real-hardware-family observation transfer: sighted vs hidden-object actor.
#
# This is the corrected successor to train_blind_actor_2x2.sh.  That launcher is
# intentionally legacy-only (m05); every morphology below is generated from
# assets/mjcf/real_v1/real_hand.xml and lies inside REAL_V1_WORKSPACE.
#
# Scope is deliberately narrow: the nominal task is already close to open-loop, so
# only the informative jittered pair is trained.  "Blind" here means object-state
# blind, NOT the final 18-D hardware interface: joint velocity, trajectory references,
# and last action remain.  Servo load is not yet an RL observation.  Do not describe
# these checkpoints as hardware-ready.
#
# Launch:
#   nohup setsid bash scripts/train_real_v1_blind_pair.sh \
#     > logs/real_v1_blind_pair.log 2>&1 </dev/null &
#
# The queue is resumable via per-arm DONE files.  Each 5M run writes a rendered eval
# video at iteration 50.  Set DESIGNS="rv05_manual" or ARMS="S1_sighted_jitter" to
# narrow a rerun; SMOKE=1 uses 1M timesteps.
set -u
ROOT=/home/humanoid/Programs/hand
cd "$ROOT"

STATE=$ROOT/logs/real_v1_blind_pair
OUT=$ROOT/docs/experiments/20260830-real_v1-obs-transfer
mkdir -p "$STATE" "$OUT"

TOTAL_TS=${TOTAL_TS:-5000000}
SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && TOTAL_TS=1000000
SEED=${SEED:-42}
DESIGNS=${DESIGNS:-"rv05_manual rv03_narrowy"}
ARMS=${ARMS:-"S1_sighted_jitter B1_blind_jitter"}

BLIND=(--actor-blind-terms object_pos object_pose_actual target_axis_misalign)
JITTER=(--cube-spawn-xy-jitter 0.005 --cube-spawn-yaw-jitter 0.087 --dr-anneal-iters 50)

design_vars () {
  case "$1" in
    rv05_manual)
      MORPH=results/phase1/real_v1/rv05_manual_stored
      A_CKPT=$ROOT/results/rl/20260828-0550-policyA_rv05_manual_t0/tensorboard/model_609.pt
      B_CKPT=$ROOT/results/rl/20260828-1215-policyB_rv05_anchor_t0/tensorboard/model_270.pt
      REF=$ROOT/results/rl/20260828-1215-policyB_rv05_anchor_t0
      AXIS_MIN=0.0
      ;;
    rv03_narrowy)
      MORPH=results/phase1/real_v1/rv03_narrowy_sp40
      A_CKPT=$ROOT/results/rl/20260828-0000-policyA_rv03_narrowy_t0/tensorboard/model_609.pt
      # Iteration 150 of the lift-gated run is the measured best checkpoint:
      # held-cos 0.645 +/- 0.007, 3/3 kept.  Iteration 270 is worse.
      B_CKPT=$ROOT/results/rl/20260828-1327-policyB_rv03_gated_t0/tensorboard/model_150.pt
      REF=$ROOT/results/rl/20260828-1327-policyB_rv03_gated_t0
      AXIS_MIN=0.06
      ;;
    *) echo "FATAL unknown or non-real_v1 design '$1'"; return 1 ;;
  esac
}

common_args () {
  COMMON=(
    --recipe b_liveA --morphology-run "$MORPH" --seed "$SEED"
    --num-envs 3072 --total-timesteps "$TOTAL_TS"
    --live-a-checkpoint "$A_CKPT" --live-a-onset 58
    --init-actor-checkpoint "$B_CKPT"
    --lift-target-z-above-init 0.1 --lift-delta-z 0.1
    --finger-residual-scale 0.5 --finger-close-easing ease_out_quad
    --lift-phase-start-step 58 --reorient-start-step 58
    --term-tip-lost-steps 15 --open-finger-from-keyframe
    --hold-ctrl-from-keyframe hold_ik --hold-switch-from-sim-step 600
    --hold-switch-steps 550 --hold-switch-min-z 0.08
    --object-orientation-drift-weight 0.0 --target-axis-min-lift "$AXIS_MIN"
  )
}

parity_gate () { # design, arm, then arm-specific args
  local design=$1 arm=$2
  shift 2
  local dry tag="20260830-parity-${design}-${arm}"
  dry=$(mktemp -d)
  WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu \
    python "$ROOT/scripts/rl_train_cube.py" "${COMMON[@]}" "$@" \
      --tag "$tag" --output-root "$dry" --dry-run > "$dry/dry.log" 2>&1 || return 1
  local allow=(--allow ppo.total_timesteps --allow env.cube_spawn_xy_jitter
               --allow env.cube_spawn_yaw_jitter --allow env.dr_anneal_iters)
  [ "$arm" = B1_blind_jitter ] && allow+=(--allow env.actor_blind_terms)
  uv run python "$ROOT/scripts/assert_config_parity.py" \
    --run "$dry/$tag" --reference "$REF" --wait 5 "${allow[@]}" || {
      echo "FATAL parity failed; dry-run log: $dry/dry.log"
      return 1
    }
  rm -rf "$dry"
}

run_arm () { # design, arm, then arm-specific args
  local design=$1 arm=$2
  shift 2
  local tag="20260830-${design}-${arm}_s${SEED}"
  local done="$STATE/${design}-${arm}_s${SEED}.DONE"
  local log="$STATE/${design}-${arm}_s${SEED}.trainer.log"
  [ -e "$done" ] && { echo "[real_v1] SKIP $design $arm (done)"; return 0; }

  parity_gate "$design" "$arm" "$@" || return 1
  echo "[real_v1] === $design $arm  seed=$SEED ts=$TOTAL_TS ==="
  WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu \
    python "$ROOT/scripts/rl_train_cube.py" "${COMMON[@]}" "$@" \
      --tag "$tag" --watchdog-collapse-z 0.030 --watchdog-from-iter 60 \
      --watchdog-sentinel "${log}.COLLAPSED" > "$log" 2>&1
  local rc=$?
  if [ -e "${log}.COLLAPSED" ]; then
    echo "[real_v1] $design $arm collapsed (recorded; not retried)"
  elif [ $rc -ne 0 ]; then
    echo "[real_v1] $design $arm FAILED rc=$rc; see $log"
    return $rc
  fi
  touch "$done"
  echo "[real_v1] $design $arm done -> results/rl/$tag"
}

for design in $DESIGNS; do
  design_vars "$design" || exit 1
  common_args
  for f in "$A_CKPT" "$B_CKPT" "$REF/config.yaml" "$ROOT/$MORPH/best_rollout.npz"; do
    [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }
  done
  for arm in $ARMS; do
    case "$arm" in
      S1_sighted_jitter) run_arm "$design" "$arm" "${JITTER[@]}" || exit 1 ;;
      B1_blind_jitter)   run_arm "$design" "$arm" "${BLIND[@]}" "${JITTER[@]}" || exit 1 ;;
      *) echo "FATAL unknown arm '$arm'"; exit 1 ;;
    esac
  done
done

touch "$STATE/QUEUE_s${SEED}.DONE"
echo "[real_v1] queue complete; rendered videos are under each run's eval_videos/."

