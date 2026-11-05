#!/bin/bash
# Evaluate the hardware-family sighted/object-blind training pair on both distributions.
set -u
ROOT=/home/humanoid/Programs/hand
cd "$ROOT"
OUT=$ROOT/docs/experiments/20260830-real_v1-obs-transfer
SEED=${SEED:-42}
BLIND_TERMS="object_pos object_pose_actual target_axis_misalign"
mkdir -p "$OUT"

latest_ckpt () {
  ls "$1"/tensorboard/model_*.pt 2>/dev/null \
    | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-
}

design_vars () {
  case "$1" in
    rv05_manual)
      MORPH=results/phase1/real_v1/rv05_manual_stored
      A_CKPT=$ROOT/results/rl/20260828-0550-policyA_rv05_manual_t0/tensorboard/model_609.pt ;;
    rv03_narrowy)
      MORPH=results/phase1/real_v1/rv03_narrowy_sp40
      A_CKPT=$ROOT/results/rl/20260828-0000-policyA_rv03_narrowy_t0/tensorboard/model_609.pt ;;
    *) echo "FATAL unknown design $1"; return 1 ;;
  esac
}

for design in rv05_manual rv03_narrowy; do
  design_vars "$design" || exit 1
  for arm in S1_sighted_jitter B1_blind_jitter; do
    run=$ROOT/results/rl/20260830-${design}-${arm}_s${SEED}
    ck=$(latest_ckpt "$run")
    [ -n "$ck" ] || { echo "[eval] missing checkpoint: $run"; continue; }
    for dist in nominal jitter; do
      args=(--policy-a "$A_CKPT" --policy-b "$ck" --morphology-run "$MORPH"
            --open-finger-from-keyframe --hold-ctrl-from-keyframe hold_ik
            --hold-switch-from-sim-step 600 --hold-switch-steps 550
            --hold-switch-min-z 0.08 --num-envs 32 --conditions none)
      [ "$arm" = B1_blind_jitter ] && args+=(--actor-blind-terms $BLIND_TERMS)
      [ "$dist" = jitter ] && args+=(--spawn-jitter 0.005 --spawn-yaw-jitter 0.087)
      echo "[eval] $design $arm test=$dist ck=$(basename "$ck")"
      WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu \
        python "$ROOT/scripts/probe_obs_ablation.py" "${args[@]}" \
        --output "$OUT/EVAL_${design}_${arm}_test-${dist}.json" \
        2>&1 | grep -E '^\[abl\] (none|->|spawn)'
    done
  done
done

echo "[eval] complete -> $OUT/EVAL_*.json"

