#!/bin/bash
# Worst-case score over a band of scene frictions (owner, 2026-11-03: "robustness to a band of physical parameters as
# a scoring parameter in hand selection"; "worst-case sampling makes sense"). scripts/hom/hom_grasp_refine.py --band: each
# candidate runs at seeds 0-2 on the STS3032 at kp 2.1 with scene friction 0.4, 0.7 and 1.0, and scores its lowest
# three-seed mean. On the k21 sample (30 layouts per task: 20 best by S1 at friction 1.0, 10 at random), as the 11-03
# queue refined at 1.0 and at 0.4 (logs/20261103-hom_score_refine/queue.sh):
#   1. the friction-1.0 and friction-0.4 picks of 11-03 at friction 0.7 (their other frictions' rows exist);
#   2. levels grasp and full with the worst-case score, 6 iterations, the start as candidate 0 of iteration 0;
#   3. the worst-case picks at seeds 20-24 on each friction; the report rows/band_k21mu/<date>-band_levels.json.
# Budget from the 11-03 k21 steps (both levels 89 min a plant): about 4.5 h for step 2, 15 min for steps 1 and 3; one
# GPU job at a time. Smoke test 04:1x on 11-04: logs/20261104-hom_worst_case/smoke.log and check_band.py.
cd /home/humanoid/Programs/hand
export SCALE_HAND=cad_v2 NUMPY_MADVISE_HUGEPAGE=0
L=logs/20261104-hom_worst_case
T="twist roll pinpush press"
B=k21mu0p4,k21mu0p7,k21
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
step() {   # name, then hom_grasp_refine.py's arguments; a finished step leaves Q_<name>.done
  local name=$1 rc; shift
  if [ -e $L/Q_$name.done ]; then echo "$(date +%T) $name done before"; return 0; fi
  echo "$(date +%T) $name"
  job 5G 500 /usr/bin/time -f "peak %M kB, %e s" .venv/bin/python scripts/hom/hom_grasp_refine.py "$@" > $L/Q_$name.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then touch $L/Q_$name.done; else echo "$(date +%T) $name failed (exit $rc)"; fi
  return $rc
}
for t in $T; do
  step val07_k21_$t validate --task $t --band $B --picks-from k21
  step val07_k21mu0p4_$t validate --task $t --band $B --picks-from k21mu0p4
done
for t in $T; do step band_grasp_$t run --task $t --level grasp --band $B --layouts-from k21 --iters 6; done
for t in $T; do step band_full_$t run --task $t --level full --band $B --layouts-from k21 --iters 6; done
for t in $T; do step band_val_$t validate --task $t --band $B; done
step band_report report --band $B
echo "$(date +%T) worst-case queue done"
