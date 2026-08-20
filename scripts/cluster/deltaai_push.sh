#!/usr/bin/env bash
# Push the repo (code + scenes + the morphology runs a job needs) to DeltaAI.
#
# Runbook step 2. Unlike the ACT case there is no dataset — what has to go up is
# the code, the frozen scenes, and any checkpoints a run warmstarts from. results/
# is gitignored locally AND is the thing training reads (morphology runs live
# there), so it is synced selectively, not wholesale.
#
#   scripts/cluster/deltaai_push.sh                 # code only
#   scripts/cluster/deltaai_push.sh results/rl/20260819-2201-perp_sp25_holdthumb_z_z
#   scripts/cluster/deltaai_push.sh results/phase1/perp_sp25 results/rl/<ckpt-run>
#
# Open the Duo-authenticated socket first:  ssh deltaai exit
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="${REMOTE:-deltaai}"
# Plain relative path: rsync resolves it against the remote home ($HOME is not
# expanded on the far side, it is passed through literally and mkdir fails).
DEST="${DEST:-hand}"

cd "$ROOT"

# The repo is 28 GB on disk and almost none of it is needed to train. What must go
# up: src/, scripts/, configs/, assets/mjcf, and the two editable path deps
# (external/mujoco_warp + external/comfree_warp) — without those `uv sync` fails.
# What must not: docs/uhas (11 GB of renders), external/GraspGenX + lightning-grasp
# (17 GB of unrelated submodules), every venv, every video. ~134 MB / 2,100 files goes up (measured 2026-08-20).
rsync -avz --progress \
  --exclude '.git/' --exclude '.venv*/' --exclude 'wandb/' --exclude 'logs/' \
  --exclude 'results/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude 'external/GraspGenX/' --exclude 'external/lightning-grasp/' \
  --exclude 'external/035_power_drill/' \
  --exclude 'docs/uhas/' --exclude 'docs/videos/' --exclude 'docs/experiments/' \
  --exclude 'relevant_papers/' --exclude 'webpaper/' --exclude 'paper/' \
  --exclude '*.mp4' --exclude '*.gif' --exclude '*.pdf' \
  --exclude 'MUJOCO_LOG.TXT' --exclude 'MJDATA.TXT' \
  ./ "$REMOTE:$DEST/"

# Extra paths (morphology runs, warmstart checkpoints) named on the command line.
for p in "$@"; do
  [ -e "$p" ] || { echo "!! no such path: $p" >&2; exit 1; }
  echo "--- pushing $p"
  rsync -avz --progress --relative \
    --exclude 'eval_videos/' --exclude '*.mp4' --exclude '*.gif' \
    "./$p" "$REMOTE:$DEST/"
done

echo
echo "OK. On the cluster: cd ~/hand && bash scripts/cluster/deltaai_env_setup.sh"
