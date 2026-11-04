#!/bin/bash
# Owner, 11-03 18:1x: "yea, let's send it on these tasks ... we can try out multiple scores". The three scores of
# scripts/hom_grasp_refine.py (S1 the best of 16 grasps at default HOM; S2 the grasp refined by CEM; S3 the grasp, the
# commanded forces and the contact modes refined together by CEM):
#   1. the SCS0009 plant with pads at 2.5 MPa (sv25): 60 layouts per task (30 best by S1, 30 at random), 8 iterations;
#   2. the STS3032 at kp 2.1 with scene friction 1.0 (k21), then 0.4 (k21mu0p4, TPU on bare PLA) on the same 30 layouts
#      (20 best by k21's S1, 10 at random), 6 iterations;
#   3. each plant's picks at seeds 20-24, and the STS3032 picks of each friction run at the other friction.
# Smoke test 18:22: the batch child peaks at 2.43 GB, the parent and four draw workers well under 1 GB together. Budget:
# about 5.2 h for sv25, 2 h each for k21 and k21mu0p4, 20 min of cross runs; one GPU job at a time.
cd /home/humanoid/Programs/hand
export SCALE_HAND=cad_v2 NUMPY_MADVISE_HUGEPAGE=0
L=logs/20261103-hom_score_refine
T="twist roll pinpush press"
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
  job 5G 500 /usr/bin/time -f "peak %M kB, %e s" .venv/bin/python scripts/hom_grasp_refine.py "$@" > $L/Q_$name.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then touch $L/Q_$name.done; else echo "$(date +%T) $name failed (exit $rc)"; fi
  return $rc
}
for t in $T; do step sv25_grasp_$t run --task $t --level grasp --plant sv25; done
for t in $T; do step sv25_full_$t run --task $t --level full --plant sv25; done
for t in $T; do step sv25_val_$t validate --task $t --plant sv25; done
step sv25_report report --plant sv25
for p in k21 k21mu0p4; do
  base=""; [ $p = k21mu0p4 ] && base="--layouts-from k21"
  for t in $T; do step ${p}_grasp_$t run --task $t --level grasp --plant $p --top 20 --rand 10 --iters 6 $base; done
  for t in $T; do step ${p}_full_$t run --task $t --level full --plant $p --top 20 --rand 10 --iters 6 $base; done
  for t in $T; do step ${p}_val_$t validate --task $t --plant $p; done
done
for t in $T; do
  step k21mu0p4_from_k21_$t validate --task $t --plant k21mu0p4 --picks-from k21
  step k21_from_k21mu0p4_$t validate --task $t --plant k21 --picks-from k21mu0p4
done
step k21_report report --plant k21
step k21mu0p4_report report --plant k21mu0p4
echo "$(date +%T) score refine queue done"
