#!/usr/bin/env bash
# The 2026-10-06 grid of scripts/reorient_backends.py: eight bench hands, seeds 0-8 (2 mm / 2 deg jitter), plant
# `fast` unless named, friction 2.4 unless named. One MuJoCo queue (single core) and four Drake queues (two
# hands each), every rollout through resguard. Rows: docs/experiments/20261006-fingertip_backends/reorient_*.jsonl
#
#   scripts/reorient_backends_grid.sh mujoco      # or: drake
set -u
cd "$(dirname "$0")/.."
PY=logs/20261001-hom_contact/venv/bin/python
OUT=docs/experiments/20261006-fingertip_backends
LOG=logs/20261006-reorient_backends
mkdir -p "$LOG"
SEEDS="0 1 2 3 4 5 6 7 8"
RG="env RESGUARD_MAX_SWAP_GB=${RESGUARD_MAX_SWAP_GB:-16.5} $HOME/.claude/bin/resguard.sh run --mem 1G --cpu 100 --"
ENVS="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl"
R="$PY scripts/reorient_backends.py run --seeds $SEEDS"

case "${1:-}" in
mujoco)
  O="--out $OUT/reorient_mujoco.jsonl"
  $RG $ENVS $R $O --tips legacy --models pt --plants fast --numerics scene --mu scene
  $RG $ENVS $R $O --tips legacy sphere tpu6 tpu2.7 --models pt --plants fast --numerics bed --ir 100 --mu scene
  $RG $ENVS $R $O --tips tpu6 tpu2.7 --models pads --plants fast --numerics bed --ir 100 --mu scene
  $RG $ENVS $R $O --tips tpu6 tpu2.7 --models pads --plants fast --numerics bed --ir 10000 --mu scene
  $RG $ENVS $R $O --tips legacy --models pt --plants fast --numerics bed --ir 100 --mu 1.0
  $RG $ENVS $R $O --tips tpu6 --models pads --plants fast --numerics bed --ir 100 --mu 1.0
  $RG $ENVS $R $O --tips legacy --models pt --plants rigid fast_fl --numerics bed --ir 100 --mu scene
  $RG $ENVS $R $O --tips tpu6 --models pads --plants rigid fast_fl --numerics bed --ir 100 --mu scene
  ;;
drake)
  for H in "D1 D2 D3" "D4 D5 D6" "D7 D8"; do
    tag=$(echo $H | tr -d ' ')
    $RG $ENVS $R --sim drake --hands $H --tips legacy sphere tpu6 tpu2.7 --plants fast --mu scene \
      --out $OUT/reorient_drake_$tag.jsonl > "$LOG/drake_$tag.log" 2>&1 &
  done
  wait
  ;;
*) echo "usage: $0 mujoco|drake"; exit 2;;
esac
