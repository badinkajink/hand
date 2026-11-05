#!/bin/bash
# The grasp-genome EA against hom_ea's layout EA at 6,000 rollouts on the SCS0009 (autopilot job
# hand-20261104-hom_grasp_genome_ea; owner, 2026-11-03: "can't we score and refine a grasp based off how well the grasp
# (+morphology) executes the HOM control? from what i understand we are doing pretty random sampling"). One GPU job at a
# time, each through resguard, per task:
#   1. scripts/hom/hom_ea_grasp.py run: search seeds 0-2 in lockstep, population 64, budget 6,000 rollouts each;
#   2. scripts/hom/hom_ea.py run at seeds 0-2, the same budget, on the same plant (rows ea_live_sv25);
#   3. the six incumbents at seeds 20-24 (validate), then on the CPU plant at seed 20, one core (cpu);
# then the report rows/ea_grasp/sv25/<date>-ea_grasp.json.
cd /home/humanoid/Programs/hand
export SCALE_HAND=cad_v2 NUMPY_MADVISE_HUGEPAGE=0
L=logs/20261104-hom_grasp_genome_ea
T="twist roll pinpush press"
declare -A D=([twist]=40 [roll]=40 [pinpush]=25 [press]=25)
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
done_or_run() {   # name, memory, CPU, then the command; a finished step leaves Q_<name>.done
  local name=$1 mem=$2 cpu=$3 rc; shift 3
  if [ -e $L/Q_$name.done ]; then echo "$(date +%T) $name done before"; return 0; fi
  echo "$(date +%T) $name"
  job $mem $cpu /usr/bin/time -f "peak %M kB, %e s" "$@" > $L/Q_$name.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then touch $L/Q_$name.done; else echo "$(date +%T) $name failed (exit $rc)"; fi
  return $rc
}
for t in $T; do
  done_or_run eag_$t 5G 500 .venv/bin/python scripts/hom/hom_ea_grasp.py run --task $t
  for s in 0 1 2; do
    HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 \
      done_or_run ea_${t}_s$s 5G 500 .venv/bin/python scripts/hom/hom_ea.py run --task $t --d ${D[$t]} --seed $s \
      --budget 6000
  done
  done_or_run val_$t 5G 500 .venv/bin/python scripts/hom/hom_ea_grasp.py validate --task $t
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    done_or_run cpu_$t 3G 100 .venv/bin/python scripts/hom/hom_ea_grasp.py cpu --task $t
done
done_or_run report 3G 200 .venv/bin/python scripts/hom/hom_ea_grasp.py report
echo "$(date +%T) ea grasp queue done"
