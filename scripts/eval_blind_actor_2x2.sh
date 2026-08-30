#!/bin/bash
# Evaluate the blind-actor 2x2 (scripts/train_blind_actor_2x2.sh) on ONE protocol.
# =============================================================================
# Every arm is read out through the SAME continuous A->B handoff as b33 itself
# (probe_obs_ablation.py --conditions none, 32 envs), on BOTH test distributions:
# nominal, and the 5 mm / 5 deg jitter the training queue's jittered arms saw. A
# policy is only interesting if it beats the others on the distribution it will
# actually meet, so every policy is scored on both.
#
# The blind arms are evaluated WITH their blinding applied. That is not a detail:
# an actor trained on zeros for object_pos/object_pose_actual/target_axis_misalign
# and then handed live values is out of distribution in exactly the way gotcha #13
# describes for finger_residual_scale, and it would read as "the blind policy is
# fine" or "the blind policy is broken" depending only on that mistake.
#
# Usage:  bash scripts/eval_blind_actor_2x2.sh [checkpoint_basename]
# Default checkpoint is the last one each run wrote.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
MORPH=results/phase1/landscape/m05_ik_cem
A_CKPT=$ROOT/results/rl/a10_20260702-1256-policyA_m05_ik10/tensorboard/model_609.pt
OUT=$ROOT/docs/experiments/20260830-obs_ablation; mkdir -p "$OUT"
SEED=${SEED:-42}
BLIND_TERMS="object_pos object_pose_actual target_axis_misalign"

latest_ckpt () {  # newest model_N.pt by N, not by mtime
  ls "$1"/tensorboard/model_*.pt 2>/dev/null \
    | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-
}

eval_one () {  # $1=arm name, $2=blind? (0/1), $3=test distribution (nominal|jitter)
  local arm=$1 blind=$2 dist=$3
  local run=$ROOT/results/rl/20260830-${arm}_s${SEED}
  [ -d "$run" ] || { echo "[eval] MISSING run $run — arm did not train, skipping"; return 0; }
  local ck; ck=$(latest_ckpt "$run")
  [ -n "$ck" ] || { echo "[eval] no checkpoint in $run"; return 0; }
  local args=( --policy-a "$A_CKPT" --policy-b "$ck" --morphology-run "$MORPH"
               --open-finger-from-keyframe --num-envs 32 --total-steps 240
               --conditions none )
  [ "$blind" = "1" ] && args+=( --actor-blind-terms $BLIND_TERMS )
  [ "$dist" = "jitter" ] && args+=( --spawn-jitter 0.005 --spawn-yaw-jitter 0.087 )
  echo "=========== $arm  |  test=$dist  |  $(basename "$ck") ==========="
  local c; c=$(mktemp -d)
  WARP_CACHE_PATH="$c" MUJOCO_GL=egl uv run --extra rl --extra gpu \
    python "$ROOT/scripts/probe_obs_ablation.py" "${args[@]}" \
    --output "$OUT/EVAL_${arm}_test-${dist}.json" 2>&1 | grep -E "^\[abl\]" | grep -v WARNING
}

for dist in nominal jitter; do
  eval_one S0_sighted_nominal 0 "$dist"
  eval_one B0_blind_nominal   1 "$dist"
  eval_one S1_sighted_jitter  0 "$dist"
  eval_one B1_blind_jitter    1 "$dist"
done

echo
echo "=================== 2x2 SUMMARY ==================="
uv run python - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
arms = ["S0_sighted_nominal", "B0_blind_nominal", "S1_sighted_jitter", "B1_blind_jitter"]
print(f"{'arm':22s} {'test':8s} {'hold':>6s} {'min_z':>7s} {'cos|held':>9s} {'peak':>7s}  n_held")
for a in arms:
    for d in ("nominal", "jitter"):
        f = out / f"EVAL_{a}_test-{d}.json"
        if not f.exists():
            print(f"{a:22s} {d:8s}      -       -         -       -  (not run)")
            continue
        r = json.loads(f.read_text())["rows"][0]
        print(f"{a:22s} {d:8s} {r['hold_rate']:6.2f} {r['min_z_post_mean']:7.3f} "
              f"{r['final_cos_held_mean']:+9.3f} {r['peak_cos_mean']:+7.3f}  "
              f"{r['n_held']}/{r['n']}")
print()
print("Read S1 vs B1 on test=jitter: that gap is the cost of blinding the actor on the")
print("distribution where feedback can matter at all. S0 vs B0 on test=nominal says")
print("whether the nominal task ever needed the object state.")
PY
