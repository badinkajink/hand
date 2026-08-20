#!/usr/bin/env bash
# Pull finished runs back down (runbook step 7). Checkpoints + configs + logs
# only — the videos get rendered here, on a machine with a display stack, from
# the checkpoint.
#
#   scripts/cluster/deltaai_pull.sh                    # every run under results/rl
#   scripts/cluster/deltaai_pull.sh 'H06_04_s*'        # one design's seeds
#
# Then, before believing any of it (CLAUDE.md lesson 2):
#   uv run python scripts/policy_filmstrip.py --run results/rl/<run> --width 960 --height 720
#   uv run python scripts/policy_eval_suite.py --run results/rl/<run>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="${REMOTE:-deltaai}"
SRC="${SRC:-hand/results/rl}"
PAT="${1:-*}"

mkdir -p "$ROOT/results/rl"
rsync -avz --progress \
  --include='*/' \
  --include='*.pt' --include='config.yaml' --include='*.json' \
  --include='train.log' --include='.DONE' --include='.COLLAPSED' \
  --include='events.out.tfevents.*' \
  --exclude='*' \
  "$REMOTE:$SRC/$PAT" "$ROOT/results/rl/"

echo
echo "Pulled. Seeds of one design, side by side:"
echo "  ls -d results/rl/<design>_s* | xargs -I{} uv run python scripts/policy_eval_suite.py --run {}"
