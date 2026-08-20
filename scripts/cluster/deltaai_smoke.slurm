#!/usr/bin/env bash
# Runbook step 4 for our stack: prove the GPU path works, prime the Warp kernel
# cache, and MEASURE steps/s on a GH200 — before spending anything in bulk.
#
# The ACT runbook's smoke test decoded one video frame. Ours has to do more,
# because our failure modes are further in: warp compiling for sm_90, mjwarp
# stepping 3072 envs in 96 GB, and the throughput number that decides whether
# bulk training is worth doing at all.
#
#   sbatch scripts/cluster/deltaai_smoke.slurm
#   tail -f logs/deltaai_smoke_*.log
#
#SBATCH --job-name=mh_smoke
#SBATCH --account=bgcd-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/deltaai_smoke_%j.log
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module purge
module load cuda
source .venv/bin/activate          # NOT `uv run` — compute nodes have no internet,
                                   # and uv would try to re-resolve the lockfile.

echo "== node $(hostname)  arch $(uname -m)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Warp cache: private to this process (a shared one races and NaNs) and on
# NODE-LOCAL NVMe, never Lustre. First run compiles cold; we tar the result so
# array tasks never pay it again.
export WARP_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/warp_cache_$SLURM_JOB_ID"
mkdir -p "$WARP_CACHE_PATH"
if [ -f "$HOME/warp_cache.tar" ]; then
  echo "== reusing primed warp cache"
  tar -C "$WARP_CACHE_PATH" -xf "$HOME/warp_cache.tar"
  PRIMED=1
else
  echo "== cold Warp cache — this run pays the kernel compile (several minutes)"
  PRIMED=0
fi

python - <<'PY'
import torch, warp as wp
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch cannot see the GPU — wrong wheel arch or missing `module load cuda`"
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
wp.init()
print("warp devices:", wp.get_devices())
PY

# ---- a real (short) training run on the real scene ---------------------------
# 2M timesteps is ~26 PPO iterations: enough to time honestly, cheap enough to
# throw away. Videos and wandb are OFF — compute nodes have no outbound network,
# and eval video is GPU time we would rather spend on steps.
MORPH_RUN="${MORPH_RUN:-results/phase1/perp_thumb_engage/sp25_manual}"
SCENE="${SCENE:-$MORPH_RUN/frozen_scene.xml}"
NUM_ENVS="${NUM_ENVS:-3072}"
STEPS="${STEPS:-2000000}"

if [ ! -d "$MORPH_RUN" ]; then
  echo "!! MORPH_RUN=$MORPH_RUN not on the cluster."
  echo "   Push it from your laptop:  scripts/cluster/deltaai_push.sh $MORPH_RUN"
  exit 1
fi

# FLAG PARITY (this is how the r5 queue was voided): the recipe carries the reward
# shaping but NOT the timestep budget, NOT num_envs, and NOT the lift height. A
# perp run at the default lift instead of 0.14 trains clean and aligns 0.0%, and
# nothing in the log says so. Every one of these is load-bearing.
LIFT="${LIFT:-0.14}"
ARGS=(--recipe perp_single --morphology-run "$MORPH_RUN"
      --tag deltaai_smoke --num-envs "$NUM_ENVS" --total-timesteps "$STEPS"
      --lift-target-z-above-init "$LIFT" --lift-delta-z "$LIFT"
      --open-finger-from-keyframe --no-wandb --no-record-videos --seed 0)
[ -n "$SCENE" ] && ARGS+=(--frozen-scene-xml "$SCENE" --closed-ctrl-from-keyframe closed_manual)

START=$SECONDS
python scripts/rl_train_cube.py "${ARGS[@]}"
ELAPSED=$((SECONDS - START))

echo "=============================================================="
echo " $STEPS timesteps in ${ELAPSED}s  ->  $((STEPS / (ELAPSED>0?ELAPSED:1))) steps/s"
echo " local RTX 4070 Ti SUPER reference: ~10,400 steps/s"
echo " (includes ~60-90 s of startup; the cold-cache run also includes nvcc)"
echo "=============================================================="

# Bank the primed cache for every later array task.
if [ "$PRIMED" = "0" ]; then
  tar -C "$WARP_CACHE_PATH" -cf "$HOME/warp_cache.tar" .
  echo "== warp cache banked -> $HOME/warp_cache.tar ($(du -sh "$HOME/warp_cache.tar" | cut -f1))"
fi
rm -rf "$WARP_CACHE_PATH"
