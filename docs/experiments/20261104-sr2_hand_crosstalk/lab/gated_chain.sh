#!/usr/bin/env bash
# The rest of plan_layouts_s1_from32 after run4 (from step 105), one layout at a time, each layout's poses gated on a
# look at the gantries. The coordinator, 13:5x on 11-05: in cad_v2 a stepper axis that skips its layout move brings
# fingers into each other at four of the five remaining layouts (to -9.3 mm), and nothing on the hand can see a skip
# (open-loop steppers; joint 4 skipped one at 13:34). So each layout's mounts chunk (zero, layout move, 30 s baseline)
# runs, then the chain waits until OUT/GO_<layout> exists: the owner, or the lab from a camera frame, confirms that every
# finger whose mount changed moved. OUT/STOP stops the chain, also inside a chunk. Every chunk runs with --keep-home,
# so a stop keeps the home and nothing needs homing.
# Any chunk folder of lab/chunk_plan.py runs the same way, its layouts in the order of their numbers:
#   nohup setsid bash docs/experiments/20261104-sr2_hand_crosstalk/lab/gated_chain.sh [CHUNK_DIR OUT [FIRST]] >> OUT.out 2>&1 &
# (defaults: plan/20261105-from32_chunks and logs/20261105-sr2_crosstalk/chain5; FIRST, the number of the first layout
# to run, default 1). MAX_ERR_YAW in the environment sets the yaw joints' limit (default 8 deg; 10 from 15:5x on 11-05,
# after the owner watched the middle's yaw stick at 29.6 deg for 5 s and free itself with nothing touching).
# MIN_CLEARANCE (mm) turns on run_plan.py's --min-clearance: a pose whose measured joints clear less than that in the
# CAD hand is backed out (the coordinator, 16:3x on 11-05: two neighbours both short toward each other at the yaw
# limit can reach contact on poses the plan clears by 12 mm).
set -u
cd /home/will/workspace/hand || exit 1
D=docs/experiments/20261104-sr2_hand_crosstalk
CH=${1:-$D/plan/20261105-from32_chunks}
OUT=${2:-logs/20261105-sr2_crosstalk/chain5}
FIRST=${3:-1}
mkdir -p $OUT
run() {
  /usr/bin/python3 -u $D/lab/run_plan.py "$1" "$2" --token-file ~/.sr2_station_token \
    --boards logs/20261105-sr2_crosstalk/boards --board-channels 0,1,3 --max-err-yaw ${MAX_ERR_YAW:-8} --max-blocked 1 --keep-home \
    ${MIN_CLEARANCE:+--min-clearance $MIN_CLEARANCE}
}
for MF in $(ls $CH/*_mounts.json | sort -V); do
  L=$(basename $MF _mounts.json)
  [ "${L%%_*}" -lt "$FIRST" ] && continue
  [ -e $OUT/STOP ] && { echo "$(date +%H:%M:%S) STOP before $L"; exit 3; }
  echo "$(date +%H:%M:%S) mounts $L"
  run $CH/${L}_mounts.json $OUT/${L}_mounts || { echo "$(date +%H:%M:%S) mounts $L ended early"; exit 3; }
  [ -e $CH/${L}_poses.json ] || break
  echo "$(date +%H:%M:%S) GATE $L: waiting for $OUT/GO_$L"
  until [ -e $OUT/GO_$L ] || [ -e $OUT/STOP ]; do sleep 1; done
  [ -e $OUT/STOP ] && { echo "$(date +%H:%M:%S) STOP at the gate of $L"; exit 3; }
  echo "$(date +%H:%M:%S) poses $L"
  run $CH/${L}_poses.json $OUT/${L}_poses || { echo "$(date +%H:%M:%S) poses $L ended early"; exit 3; }
done
echo "$(date +%H:%M:%S) chain done"
