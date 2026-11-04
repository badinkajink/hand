#!/bin/bash
# Shared forces and contact modes on the SCS0009 (autopilot job hand-20261104-hom_shared_controller; owner, 2026-11-03:
# "is it possible to co-design HOM? are there hyperparameters / parameters that we can change within it?"; HOM fixed
# "gives us a fixed reference to evaluate morphologies"). One GPU job at a time, each through resguard:
#   1. level shared of scripts/hom_grasp_refine.py per task: CEM over the forces and mode weights, each candidate on 40
#      layouts of the sv25 sample (30 best by S1, 10 of the random ones) with S1's grasps at seeds 0-2; 24 candidates,
#      8 iterations, candidate 0 of iteration 0 = HOM's defaults (about 15-20 min a task);
#   2. each task's grid at the default object rescored with its pick (hom_gpu.py run --prm; rows grid2dff_v4b_sv25_shared,
#      HOM_ROWS_TAG _sv25_shared; about 11-17 min a task, as the grids of 11-02);
#   3. the pick on the 60 sample layouts at seeds 20-24, with S1's grasp and with the rescored grid's best grasp;
#   4. the report rows/sv25/<date>-shared_levels.json.
cd /home/humanoid/Programs/hand
export SCALE_HAND=cad_v2 NUMPY_MADVISE_HUGEPAGE=0
L=logs/20261104-hom_shared_controller
ROWS=docs/experiments/20261029-foundational_pose_review/rows
T="twist roll pinpush press"
declare -A D=([twist]=40 [roll]=40 [pinpush]=25 [press]=25)
declare -A V=([twist]=base [roll]=base [pinpush]=hold [press]=hold)
declare -A B=([twist]=2048 [roll]=2048 [pinpush]=1536 [press]=1536)
echo "$(date +%T) start"
job() {   # one resguard-wrapped job (MEM, CPU): waits and retries while resguard refuses it (exit 3)
  local mem=$1 cpu=$2 rc; shift 2
  while true; do
    WARP_CACHE_PATH=$(mktemp -d) $HOME/.claude/bin/resguard.sh run --mem $mem --cpu $cpu -- "$@"; rc=$?
    [ $rc -ne 3 ] && return $rc
    echo "$(date +%T) refused by resguard; retry in 120 s"
    sleep 120
  done
}
done_or_run() {   # name, then the command; a finished step leaves Q_<name>.done
  local name=$1 rc; shift
  if [ -e $L/Q_$name.done ]; then echo "$(date +%T) $name done before"; return 0; fi
  echo "$(date +%T) $name"
  job 5G 500 /usr/bin/time -f "peak %M kB, %e s" "$@" > $L/Q_$name.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then touch $L/Q_$name.done; else echo "$(date +%T) $name failed (exit $rc)"; fi
  return $rc
}
for t in $T; do
  done_or_run shared_$t .venv/bin/python scripts/hom_grasp_refine.py run --task $t --level shared
done
for t in $T; do
  prm=$(.venv/bin/python scripts/hom_grasp_refine.py prm --task $t 2>/dev/null)
  if [ -z "$prm" ]; then echo "$(date +%T) grid_$t: no pick of level shared; skipped"; continue; fi
  echo "$(date +%T) grid_$t prm $prm"
  HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25_shared \
    done_or_run grid_$t .venv/bin/python scripts/hom_gpu.py run --task $t --layouts grid2dff_size --d ${D[$t]} \
    --seeds 0 1 2 --variants ${V[$t]} --batch ${B[$t]} --out ${t}_d${D[$t]} --rows-dir $ROWS/grid2dff_v4b_sv25_shared \
    --prm "$prm"
done
for t in $T; do
  done_or_run val_$t .venv/bin/python scripts/hom_grasp_refine.py validate --task $t --shared
done
done_or_run report .venv/bin/python scripts/hom_grasp_refine.py report --shared
echo "$(date +%T) shared controller queue done"
