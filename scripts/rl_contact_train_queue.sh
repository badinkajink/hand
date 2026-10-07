#!/usr/bin/env bash
# RL contact comparison (owner-approved 2026-10-06 18:40): D6 reorientation from scratch on the working plant, the TPU
# block as one convex mesh (point contact) against 1 mm sphere pads, two seeds each, 20 M steps at 2,048 envs, one GPU
# job at a time. Flags copied from results/rl/20260917-1141-d6_cal_reorient_gp025_60M_s0/config.yaml (dry-run diff:
# only num_envs, total_timesteps and save_interval differ). Scenes: scripts/make_work_plant_runs.py.
# Resumable: a run whose final checkpoint exists is skipped. Each run goes through resguard (10 GB, 8 cores).
#
#   nohup setsid bash scripts/rl_contact_train_queue.sh > logs/20261006-rl_contact/train_queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
LOGS=logs/20261006-rl_contact
mkdir -p "$LOGS"
RUNS=results/phase1/real_v1
for job in "tpu2.7mesh 0" "tpu2.7pads1 0" "tpu2.7mesh 1" "tpu2.7pads1 1"; do
  set -- $job
  variant=$1; seed=$2
  tag="20261006-d6_work_${variant/./}_20M_s${seed}"
  out=results/rl/$tag
  if ls "$out"/tensorboard/model_405.pt >/dev/null 2>&1; then
    echo "$(date +%H:%M) skip $tag (final checkpoint present)"; continue
  fi
  until ~/.claude/bin/resguard.sh status | grep -q "launch gate: OPEN"; do sleep 60; done
  echo "$(date +%H:%M) start $tag"
  WARP_CACHE_PATH=$(mktemp -d) ~/.claude/bin/resguard.sh run --mem 10G --cpu 800 -- \
    uv run --extra rl --extra gpu python scripts/rl_train_cube.py \
      --morphology-run "$RUNS/20261006-sv1_u0308_b050_work_tip_${variant}" --recipe b_liveA --tag "$tag" \
      --seed "$seed" --num-envs 2048 --total-timesteps 20000000 --save-interval 41 \
      --closed-ctrl-from-keyframe open_ik --open-finger-from-keyframe \
      --lift-target-z-above-init 0.1 --lift-delta-z 0.1 --finger-residual-scale 0.5 \
      --finger-close-easing ease_out_quad --lift-phase-start-step 58 --reorient-start-step 58 \
      --finger-residual-active-from-step 58 --term-tip-lost-steps 15 --friction-dr \
      --grip-force-penalty-weight 0.25 > "$LOGS/train_${tag}.log" 2>&1
  echo "$(date +%H:%M) end $tag exit $?"
  sleep 90                     # let the GPU memory of the finished Warp process drop
done
echo "$(date +%H:%M) queue done"
