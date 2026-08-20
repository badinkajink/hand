#!/usr/bin/env bash
# Why the GH200 came in SLOWER than the workstation, and whether num_envs fixes it.
#
# Measured 2026-08-20, both at num_envs=3072, num_steps_per_env=24 (73,728 steps
# per PPO iteration):
#     RTX 4070 Ti SUPER  7.00 s/iter  -> 10,530 steps/s
#     GH200              11.36 s/iter ->  6,490 steps/s   (0.62x)
#
# The GH200 has ~2x the SMs and 6x the memory, so a raw compute deficit does not
# explain it. The likely cause is CPU-side: mjwarp launches many small kernels per
# step and the Grace cores are slower per-thread than the desktop x86, so launch
# overhead — not the solve — sets the rate. If that is the mechanism, it is fixed
# by amortising over more envs, which the 95 GiB makes free. If throughput is flat
# here too, the deficit is real and every cluster GPU-hour estimate must be
# divided by 0.62.
#
# All three configs run inside ONE reservation, deliberately: a job appears to
# bill at hour granularity (a 6:37 job took the balance 92 -> 91), so three
# separate submissions could cost three hours to learn the same thing.
#
#SBATCH --job-name=mh_envscale
#SBATCH --account=bgcd-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=logs/envscale_%j.log
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module purge
module load cuda
source .venv/bin/activate

export WARP_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/warp_envscale_$SLURM_JOB_ID"
mkdir -p "$WARP_CACHE_PATH"
tar -C "$WARP_CACHE_PATH" -xf "$HOME/warp_cache.tar"

MORPH_RUN=results/phase1/perp_thumb_engage/sp25_manual
STEPS="${STEPS:-500000}"

for NE in 3072 8192 16384; do
  echo "######## num_envs=$NE ########"
  python scripts/rl_train_cube.py \
      --recipe perp_single --morphology-run "$MORPH_RUN" \
      --frozen-scene-xml "$MORPH_RUN/frozen_scene.xml" \
      --closed-ctrl-from-keyframe closed_manual \
      --tag "envscale_${NE}" --num-envs "$NE" --total-timesteps "$STEPS" \
      --lift-target-z-above-init 0.14 --lift-delta-z 0.14 \
      --open-finger-from-keyframe --no-wandb --no-record-videos --seed 0 \
      2>&1 | grep -E "Iteration time|Total timesteps|out of memory|Error" | tail -6
  echo
done

echo "steps/iteration = num_envs * 24; divide by the steady-state iteration time."
rm -rf "$WARP_CACHE_PATH"
