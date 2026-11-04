#!/bin/bash
# Task-conflict study, validation queue (hand-off docs/handoff/20261104-hom_task_conflict.md, step 1): every layout of
# each plant's lattice grid on every task it reaches, with its S1 grasp, at seeds 20-24 (scripts/hom_task_conflict.py
# validate). 1,860 layout-task pairs and 9,300 rollouts per plant, about 8 GPU batches; the batch child peaked at
# 2.43 GB in the score-refine queue's validation steps. One GPU job at a time.
cd /home/humanoid/Programs/hand
export SCALE_HAND=cad_v2 NUMPY_MADVISE_HUGEPAGE=0
L=logs/20261103-hom_task_conflict
mkdir -p $L
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
  job 5G 500 /usr/bin/time -f "peak %M kB, %e s" .venv/bin/python scripts/hom_task_conflict.py "$@" > $L/Q_$name.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then touch $L/Q_$name.done; else echo "$(date +%T) $name failed (exit $rc)"; fi
  return $rc
}
for p in sv25 k21 k21mu0p4; do step val_$p validate --plant $p; done
echo "$(date +%T) validation queue done"
