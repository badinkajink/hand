#!/bin/bash
# BLIND-ACTOR 2x2 — what does the reorienter lose when the actor cannot see the object?
# =============================================================================
# WHY THIS SHAPE. `scripts/probe_obs_ablation.py` on b33 (docs/experiments/
# 20260830-obs_ablation) found that in the nominal task the reference reorienter is
# effectively FEED-FORWARD: replaying another env's observations over the whole
# 66-dim actor input leaves hold 0.97 and held-cos 0.900 against a 0.895 baseline.
# There is no closed-loop content there to distill. The same probe also found the
# policy collapses on its own (hold 0.41) under 5 mm / 5 deg of spawn jitter.
#
# So the observability question is only meaningful under perturbation, and it needs
# both factors crossed:
#
#              nominal                     jittered (5mm xy, 5deg yaw)
#   sighted    S0: reproduces b33          S1: the ORACLE — solvable WITH object state?
#   blind      B0: is the task open-loop?  B1: solvable WITHOUT it? = deployability
#
# S1 - B1 is the plan's oracle-vs-AAC gate (docs/rl/partial_observation_transfer.md
# §7), measured instead of assumed. S0 is the control that isolates finetune drift;
# B0 says whether the nominal task ever needed the object at all.
#
# "Blind" is genuine asymmetric actor-critic: the actor's object_pos /
# object_pose_actual / target_axis_misalign are forced to zero, the critic keeps
# them. Those three are exactly what the real_v1 bench cannot measure — it has no
# object tracker — so whatever B1 learns is deployable by construction.
#
# All four warmstart b33's actor AND critic (gotcha #8) and are otherwise identical:
# `assert_config_parity.py` reports S0 at EXACT parity with b33's own config, and the
# other three differing only by the four intended keys. Warmstarting is what makes
# this affordable at one seed per cell — from-scratch reorient draws vary by sd
# 0.3-0.5 across seeds, warmstarted ones by 0.032 (project_b33_seed_band).
#
# Launch (detached):
#   nohup setsid bash scripts/train_blind_actor_2x2.sh > logs/blind2x2.log 2>&1 </dev/null &
# Resumable: each arm writes logs/blind2x2/<arm>.DONE and is skipped if present.
# SMOKE=1 -> 1M timesteps per arm (~3 min each) for a supervised sanity pass.
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"

MORPH=results/phase1/landscape/m05_ik_cem
A_CKPT=$ROOT/results/rl/a10_20260702-1256-policyA_m05_ik10/tensorboard/model_609.pt
B_CKPT=$ROOT/results/rl/b33_20260702-1353-policyB_m05_reorient_ik10/tensorboard/model_270.pt
REF=$ROOT/results/rl/b33_20260702-1353-policyB_m05_reorient_ik10
STATE=$ROOT/logs/blind2x2; mkdir -p "$STATE"

TOTAL_TS=${TOTAL_TS:-20000000}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = "1" ] && TOTAL_TS=1000000
SEED=${SEED:-42}

for f in "$A_CKPT" "$B_CKPT" "$ROOT/$MORPH/best_rollout.npz"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

BLIND=(--actor-blind-terms object_pos object_pose_actual target_axis_misalign)
JIT=(--cube-spawn-xy-jitter 0.005 --cube-spawn-yaw-jitter 0.087 --dr-anneal-iters 50)

# One arm. $1=name, rest=arm-specific flags.
run_arm () {
  local name=$1; shift
  local tag="20260830-${name}_s${SEED}"
  if [ -e "$STATE/$name.DONE" ]; then echo "[2x2] SKIP $name (done)"; return 0; fi
  local log="$ROOT/logs/blind2x2/$name.trainer.log"
  local c; c=$(mktemp -d)
  echo "[2x2] === $name  tag=$tag  ts=$TOTAL_TS  seed=$SEED ==="
  # The collapse watchdog engages after the jitter curriculum has finished ramping
  # (anneal ends at iter 50), so an anneal transient cannot be read as a collapse.
  WARP_CACHE_PATH="$c" MUJOCO_GL=egl uv run --extra rl --extra gpu \
    python "$ROOT/scripts/rl_train_cube.py" \
      --recipe b_liveA --morphology-run "$MORPH" --tag "$tag" --seed "$SEED" \
      --num-envs 3072 --total-timesteps "$TOTAL_TS" \
      --live-a-checkpoint "$A_CKPT" --live-a-onset 58 \
      --init-actor-checkpoint "$B_CKPT" \
      --lift-target-z-above-init 0.1 --lift-delta-z 0.1 \
      --finger-residual-scale 0.5 --finger-close-easing ease_out_quad \
      --lift-phase-start-step 58 --reorient-start-step 58 \
      --term-tip-lost-steps 10 --open-finger-from-keyframe \
      --watchdog-collapse-z 0.030 --watchdog-from-iter 80 \
      --watchdog-sentinel "${log}.COLLAPSED" \
      "$@" > "$log" 2>&1
  local rc=$?
  if [ -e "${log}.COLLAPSED" ]; then
    echo "[2x2] $name ABORTED by the collapse watchdog — recorded, not retried."
  elif [ $rc -ne 0 ]; then
    # A crash is NOT a result. Leave no DONE sentinel so a rerun retries this arm,
    # and do not let the queue silently report 4/4 on 3 real runs.
    echo "[2x2] $name FAILED rc=$rc — see $log"; return 1
  fi
  touch "$STATE/$name.DONE"
  echo "[2x2] $name finished rc=$rc -> results/rl/$tag"
}

# Parity gate: S0 must reproduce b33's env exactly, or every comparison below is
# against a moving reference. Checked on the dumped config, before the GPU time.
echo "[2x2] parity-checking the control arm against b33 ..."
DRY=/tmp/blind2x2_dry_$$; mkdir -p "$DRY"
WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu \
  python "$ROOT/scripts/rl_train_cube.py" --recipe b_liveA --morphology-run "$MORPH" \
    --tag 20260830-S0_parity --output-root "$DRY" --num-envs 3072 --total-timesteps "$TOTAL_TS" \
    --lift-target-z-above-init 0.1 --lift-delta-z 0.1 \
    --finger-residual-scale 0.5 --finger-close-easing ease_out_quad \
    --lift-phase-start-step 58 --reorient-start-step 58 \
    --term-tip-lost-steps 10 --open-finger-from-keyframe --dry-run > "$DRY/dry.log" 2>&1
# A shortened budget is the ONLY delta a smoke pass is allowed to introduce; every
# other divergence still aborts, including on a smoke run.
PARITY_ALLOW=()
[ "$TOTAL_TS" = "20000000" ] || PARITY_ALLOW+=( --allow ppo.total_timesteps )
uv run python "$ROOT/scripts/assert_config_parity.py" --run "$DRY/20260830-S0_parity" \
  --reference "$REF" --wait 5 "${PARITY_ALLOW[@]+"${PARITY_ALLOW[@]}"}" \
  || { echo "[2x2] FATAL parity broken — not launching."; exit 1; }
rm -rf "$DRY"

run_arm S0_sighted_nominal
run_arm B0_blind_nominal   "${BLIND[@]}"
run_arm S1_sighted_jitter  "${JIT[@]}"
run_arm B1_blind_jitter    "${BLIND[@]}" "${JIT[@]}"

echo "[2x2] queue complete. NEXT: evaluate all four with the SAME continuous-handoff"
echo "[2x2] protocol (scripts/probe_obs_ablation.py --conditions none, 32 envs), and"
echo "[2x2] evaluate the blind arms with their blinding APPLIED — a blind-trained actor"
echo "[2x2] read out in a sighted env is out of distribution in the gotcha-#13 way."
touch "$STATE/QUEUE.DONE"
