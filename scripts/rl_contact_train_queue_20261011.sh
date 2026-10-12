#!/usr/bin/env bash
# RL contact-model comparison on the servo-limited plant, 2026-10-11 (hand-off docs/handoff/20261011-servo_plant_policies.md):
# D6 reorientation from scratch with the TPU block as one convex mesh (point contact) and as 1 mm sphere pads, two seeds
# each, one GPU job at a time. Dated copy of scripts/rl_contact_train_queue_20261008.sh with two changes, both after the
# 10-08 policies gripped the 26 g tool with 44-96 N:
#   plant  kp0.72_kv0_fr0.18_fl0_dp0.0144 (scripts/make_work_plant_runs.py --plant; scenes 20261011-sv1_u0308_b050_servo_tip*):
#          the SCS0009 stalls at 0.23 N m at 6 V and its overload protection trips above 80 % (0.18 N m); the load
#          register reaches full drive at 18 deg of error, so 0.23 N m / 0.314 rad = 0.72 N m/rad; tau = 0.02 s as fitted
#   reward --grip-force-penalty-weight -2 (the 2026-09-19/20 tranches' value): a penalty above 4 N per fingertip; the
#          10-06 and 10-08 queues passed +0.25, which paid a quadratic bonus for squeezing
#
# Jobs come from a file, one per line: <tag> <morphology run dir> <seed> <total timesteps>; '#' starts a comment. The
# file is re-read after every run, so jobs can be added or reordered while the queue runs. A job whose final checkpoint
# (model_<iters-1>.pt, iters = timesteps // (2,048 x 24)) exists is skipped. A run that ends without it is moved to
# <tag>_failed<k> and retried, at most twice; its log is kept as train_<tag>_failed<k>.log. Each run goes through
# resguard (10 GB, 8 cores) under /usr/bin/time -v; GPU memory is sampled every 30 s into gpu_mem.tsv. After each run
# the queue runs $LOGS/after_run.sh <tag> <run> <timesteps> if that file exists.
#
#   nohup setsid bash scripts/rl_contact_train_queue_20261011.sh logs/20261011-servo_plant_policies/jobs.txt \
#       > logs/20261011-servo_plant_policies/train_queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl           # the trainer renders eval videos; a detached shell has no DISPLAY
JOBS=${1:-logs/20261011-servo_plant_policies/jobs.txt}
LOGS=logs/20261011-servo_plant_policies
mkdir -p "$LOGS"
STEPS_PER_IT=$((2048 * 24))
MAX_FAIL=2

final_ckpt() {  # run dir, timesteps -> path of the final checkpoint
  echo "$1/tensorboard/model_$(( $2 / STEPS_PER_IT - 1 )).pt"
}

next_job() {  # first job without its final checkpoint and with fewer than MAX_FAIL+1 attempts
  grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | while read -r tag run seed steps; do
    [ -e "$(final_ckpt "results/rl/$tag" "$steps")" ] && continue
    nfail=$(ls -d "results/rl/${tag}_failed"* 2>/dev/null | wc -l)
    [ "$nfail" -gt "$MAX_FAIL" ] && continue
    echo "$tag $run $seed $steps"; break
  done
}

while true; do
  job=$(next_job)
  [ -z "$job" ] && break
  set -- $job
  tag=$1; run=$2; seed=$3; steps=$4
  out=results/rl/$tag
  if [ -d "$out" ]; then                       # a partial run left by a crash or a stop: keep it as a failure
    k=$(( $(ls -d "${out}_failed"* 2>/dev/null | wc -l) + 1 ))
    mv "$out" "${out}_failed$k"; [ -f "$LOGS/train_${tag}.log" ] && mv "$LOGS/train_${tag}.log" "$LOGS/train_${tag}_failed$k.log"
    echo "$(date '+%F %H:%M') moved partial $tag to ${tag}_failed$k"
  fi
  until ~/.claude/bin/resguard.sh status | grep -q "launch gate: OPEN"; do sleep 60; done
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  until [ "$free" -gt 13000 ]; do sleep 30; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); done
  echo "$(date '+%F %H:%M:%S') start $tag ($run, seed $seed, $steps steps)"
  ( while sleep 30; do
      echo -e "$(date +%s)\t$tag\t$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    done >> "$LOGS/gpu_mem.tsv" ) &
  sampler=$!
  WARP_CACHE_PATH=$(mktemp -d) ~/.claude/bin/resguard.sh run --mem 10G --cpu 800 -- \
    /usr/bin/time -v uv run --extra rl --extra gpu python scripts/rl_train_cube.py \
      --morphology-run "$run" --recipe b_liveA --tag "$tag" \
      --seed "$seed" --num-envs 2048 --total-timesteps "$steps" --save-interval 41 \
      --closed-ctrl-from-keyframe open_ik --open-finger-from-keyframe \
      --lift-target-z-above-init 0.1 --lift-delta-z 0.1 --finger-residual-scale 0.5 \
      --finger-close-easing ease_out_quad --lift-phase-start-step 58 --reorient-start-step 58 \
      --finger-residual-active-from-step 58 --term-tip-lost-steps 15 --friction-dr \
      --grip-force-penalty-weight -2 > "$LOGS/train_${tag}.log" 2>&1
  rc=$?
  kill "$sampler" 2>/dev/null
  if [ -e "$(final_ckpt "$out" "$steps")" ]; then
    echo "$(date '+%F %H:%M:%S') end $tag exit $rc, final checkpoint present"
  else
    k=$(( $(ls -d "${out}_failed"* 2>/dev/null | wc -l) + 1 ))
    [ -d "$out" ] && mv "$out" "${out}_failed$k"
    mv "$LOGS/train_${tag}.log" "$LOGS/train_${tag}_failed$k.log"
    echo "$(date '+%F %H:%M:%S') FAILED $tag exit $rc (attempt $k): $(grep -i -m1 -o '.\{0,80\}nan.\{0,40\}\|error.\{0,120\}' "$LOGS/train_${tag}_failed$k.log" | head -1)"
  fi
  sleep 90                     # let the GPU memory of the finished Warp process drop
  # optional per-run hook (checkpoint evaluations, step 2), read at call time so it can be written while the queue runs
  if [ -f "$LOGS/after_run.sh" ]; then
    echo "$(date '+%F %H:%M:%S') hook $tag"
    bash "$LOGS/after_run.sh" "$tag" "$run" "$steps" >> "$LOGS/after_run.log" 2>&1
    echo "$(date '+%F %H:%M:%S') hook $tag exit $?"
    sleep 30
  fi
done
echo "$(date '+%F %H:%M:%S') queue done"
