#!/usr/bin/env bash
# BULK TRAINING (runbook steps 5-6, our version): one Slurm array task per
# (design, seed) pair.
#
# This is the whole reason to be on ACCESS. Locally every launcher in scripts/ is
# built around wait_for_gpu() — a spin loop that serialises runs onto the one
# 4070 Ti SUPER, so a 5-design queue at n=4 seeds is 20 runs x ~40 min = 13 h of
# strictly sequential wall clock. Here the same 20 runs are 20 array tasks, each
# on its own GH200, and the wall clock is one run plus queue time.
#
# It also fixes the measurement problem, not just the speed one: per-design
# reorient quality is seed-dominated (per-draw sd 0.3-0.5), so a single run per
# design cannot resolve morphology differences at all. n=1 was a budget
# concession. Here it does not have to be.
#
# Usage:
#   1. write a manifest (one line per run, TAB-separated):
#        TAG <TAB> MORPH_RUN <TAB> SEED <TAB> EXTRA_ARGS...
#      e.g.  scripts/cluster/make_seed_manifest.py --designs A,B,C --seeds 4 > manifest.tsv
#   2. sbatch --array=1-$(wc -l < manifest.tsv)%32 \
#        --account=<yours> scripts/cluster/deltaai_seed_array.slurm manifest.tsv
#
# %32 caps concurrency at 32 GPUs. Raise it if the queue is empty, lower it if
# you want to keep credits in reserve — GPU-hours bill on RESERVED time.
#
#SBATCH --job-name=mh_seeds
#SBATCH --account=bgcd-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/seeds_%A_%a.log
set -euo pipefail

MANIFEST="${1:?usage: sbatch ... deltaai_seed_array.slurm <manifest.tsv>}"
cd "$SLURM_SUBMIT_DIR"
module purge
module load cuda
source .venv/bin/activate

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[ -n "$LINE" ] || { echo "manifest line $SLURM_ARRAY_TASK_ID is empty"; exit 1; }
IFS=$'\t' read -r TAG MORPH_RUN SEED EXTRA <<<"$LINE"
echo "== task $SLURM_ARRAY_TASK_ID: tag=$TAG morph=$MORPH_RUN seed=$SEED extra='$EXTRA'"
echo "== node $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Private, node-local Warp cache from the primed tar. Both halves matter: private
# because a shared cache races and NaNs, node-local because 32 tasks hammering a
# kernel cache on Lustre is a self-inflicted outage.
export WARP_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/warp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"
[ -f "$HOME/warp_cache.tar" ] && tar -C "$WARP_CACHE_PATH" -xf "$HOME/warp_cache.tar" \
  || echo "!! no primed cache; run deltaai_smoke.slurm first or this task pays nvcc"
trap 'rm -rf "$WARP_CACHE_PATH"' EXIT

# Same convention as the local launchers: DEST holds the log + sentinels, while
# the trainer writes its own results/rl/<timestamp>-<tag> run dir.
TAG="${TAG}_s${SEED}"
DEST="results/rl/${TAG}"
mkdir -p "$DEST"
if [ -f "$DEST/.DONE" ]; then echo "SKIP (.DONE) — array is resumable"; exit 0; fi

# FLAG PARITY. Recipes carry reward shaping only — the timestep budget, num_envs
# and the lift height live on the launcher, and omitting them is silent. Two local
# 5-design queues died 0/5 in one day this way, and a perp run at the default lift
# instead of 0.14 aligns 0.0% while looking healthy in every log line. Anything
# run-specific (--recipe, --lift-delta-z, --frozen-scene-xml, ...) belongs in the
# manifest's EXTRA column; diff a task's dumped config.yaml against the last
# working run before trusting an array of 48.
#
# --no-wandb: compute nodes have no outbound network (this is the analogue of the
#   ACT runbook's push_to_hub trap — a default that tries to phone home and dies).
# --no-record-videos: EGL on the compute node is an untested risk and eval video
#   is GPU time; render locally from the checkpoint instead, with policy-eyes.
# The collapse watchdog still runs — a design that drops the object should stop
# burning its 40 minutes.
python scripts/rl_train_cube.py \
    --morphology-run "$MORPH_RUN" \
    --tag "$TAG" \
    --seed "$SEED" \
    --num-envs "${NUM_ENVS:-3072}" \
    --total-timesteps "${TOTAL_STEPS:-25000000}" \
    --open-finger-from-keyframe \
    --no-wandb --no-record-videos \
    --watchdog-collapse-z 0.030 --watchdog-from-iter 50 \
    --watchdog-sentinel "$DEST/train.log.COLLAPSED" \
    $EXTRA \
    2>&1 | tee -a "$DEST/train.log"

touch "$DEST/.DONE"
echo "== done: $DEST"
