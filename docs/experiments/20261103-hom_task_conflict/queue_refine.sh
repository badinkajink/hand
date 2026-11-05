#!/bin/bash
# Task-conflict study, step 3 (hand-off docs/handoff/20261104-hom_task_conflict.md): the grasp (S2), then the grasp,
# forces and contact modes (S3) refined by hom_grasp_refine.py's CEM on the 10 shared layouts of the SCS0009 plant
# (scripts/hom/hom_task_conflict.py refine; samples rows_refine/sv25/<task>_layouts.json), each task each layout reaches,
# then the S1, S2 and S3 picks at seeds 20-24 and the report. The search scores its start as candidate 0. Per task about
# 5,800 rollouts at level grasp and 7,700 at level full, one GPU batch per iteration; the batch child peaked at 2.44 GB
# in the score-refine queue. Waits for queue.sh (the validation) to end: one GPU job at a time.
cd /home/humanoid/Programs/hand
export SCALE_HAND=cad_v2 NUMPY_MADVISE_HUGEPAGE=0
L=logs/20261103-hom_task_conflict
T="twist roll pinpush press"
mkdir -p $L
until grep -q "validation queue done" $L/queue.log 2>/dev/null; do sleep 30; done
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
step() {   # name, then hom_task_conflict.py's arguments; a finished step leaves Q_<name>.done
  local name=$1 rc; shift
  if [ -e $L/Q_$name.done ]; then echo "$(date +%T) $name done before"; return 0; fi
  echo "$(date +%T) $name"
  job 5G 500 /usr/bin/time -f "peak %M kB, %e s" .venv/bin/python scripts/hom/hom_task_conflict.py "$@" > $L/Q_$name.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then touch $L/Q_$name.done; else echo "$(date +%T) $name failed (exit $rc)"; fi
  return $rc
}
for t in $T; do
  step refine_grasp_$t refine run --task $t --level grasp --plant sv25
  step refine_full_$t refine run --task $t --level full --plant sv25
  step refine_val_$t refine validate --task $t --plant sv25
done
step refine_report refine report --plant sv25
echo "$(date +%T) refine queue done"
