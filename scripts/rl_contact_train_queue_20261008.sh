#!/usr/bin/env bash
# RL contact-model comparison, 2026-10-08 (hand-off docs/handoff/20261007-contact_model_policy_training.md): D6
# reorientation from scratch on the working plant, five fingertip contact models, three seeds each, one GPU job at a
# time. Dated copy of scripts/rl_contact_train_queue.sh (2026-10-06) with the same trainer flags (copied from
# results/rl/20260917-1141-d6_cal_reorient_gp025_60M_s0/config.yaml; num_envs, total_timesteps, save_interval differ).
#
# Jobs come from a file, one per line: <tag> <morphology run dir> <seed> <total timesteps>; '#' starts a comment. The
# file is re-read after every run, so jobs can be added or reordered while the queue runs. A job whose final checkpoint
# (model_<iters-1>.pt, iters = timesteps // (2,048 x 24)) exists is skipped. A run that ends without it is moved to
# <tag>_failed<k> and retried, at most twice; its log is kept as train_<tag>_failed<k>.log. Each run goes through
# resguard (10 GB, 8 cores) under /usr/bin/time -v; GPU memory is sampled every 30 s into gpu_mem.tsv.
#
#   nohup setsid bash scripts/rl_contact_train_queue_20261008.sh logs/20261008-contact_model_policies/jobs.txt \
#       > logs/20261008-contact_model_policies/train_queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
JOBS=${1:-logs/20261008-contact_model_policies/jobs.txt}
LOGS=logs/20261008-contact_model_policies
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
      --grip-force-penalty-weight 0.25 > "$LOGS/train_${tag}.log" 2>&1
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
done
echo "$(date '+%F %H:%M:%S') queue done"
