#!/usr/bin/env bash
# Films of the 2026-10-06 reorientation comparison: per hand, the five MuJoCo panels in one process, each Drake panel
# in a process of its own (a Drake diagram is not freed inside one process), then the 3x3 tile. One core, 1 GB each.
#   scripts/reorient_backends_films.sh D7 D2 D5
set -u
cd "$(dirname "$0")/.."
PY=logs/20261001-hom_contact/venv/bin/python
RG="env RESGUARD_MAX_SWAP_GB=${RESGUARD_MAX_SWAP_GB:-16.5} $HOME/.claude/bin/resguard.sh run --mem 1G --cpu 100 --"
F="env OMP_NUM_THREADS=1 MUJOCO_GL=egl $PY scripts/reorient_backends_films.py --seed 0"
for H in "$@"; do
  $RG $F --hand "$H" --only 0 1 2 3 4
  for i in 5 6 7 8; do $RG $F --hand "$H" --only $i; done
  $RG $F --hand "$H" --tile-only
done
