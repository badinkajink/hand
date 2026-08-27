#!/usr/bin/env bash
# CEM grasp for each candidate link-length hand (2026-08-27 study).
# Config is pinned to the sp25 run's, so the scores sit on the same scale as
# docs/experiments/SHORTPROX25.txt §1 rather than on the script's own defaults.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/phase1/20260827-linklen
for cfg in "$@"; do
  scene=assets/mjcf/experimental/20260827-linklen/${cfg}/scene.xml
  echo "=== ${cfg} $(date -Is) ==="
  MUJOCO_GL=egl WARP_CACHE_PATH=$(mktemp -d) \
  uv run --extra rl python scripts/phase1_optimize_grasp.py \
    --scene-xml "${scene}" --keyframe open_ik \
    --iterations 200 --population 80 --elite-fraction 0.2 --sigma-init 0.2 --seed 0 \
    --output-dir "${OUT}" --tag "${cfg}" \
    || echo "!!! ${cfg} FAILED"
done
echo "=== done $(date -Is) ==="
