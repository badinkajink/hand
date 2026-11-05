#!/usr/bin/env bash
# The rest of plan_layouts_s1_from32 after run4 (from step 105), one layout at a time, each layout's poses gated on a
# look at the gantries. The coordinator, 13:5x on 11-05: in cad_v2 a stepper axis that skips its layout move brings
# fingers into each other at four of the five remaining layouts (to -9.3 mm), and nothing on the hand can see a skip
# (open-loop steppers; joint 4 skipped one at 13:34). So each layout's mounts chunk (zero, layout move, 30 s baseline)
# runs, then the chain waits until OUT/GO_<layout> exists: the owner, or the lab from a camera frame, confirms that every
# finger whose mount changed moved. OUT/STOP stops the chain, also inside a chunk. Every chunk runs with --keep-home,
# so a stop keeps the home and nothing needs homing.
#   nohup setsid bash docs/experiments/20261104-sr2_hand_crosstalk/lab/gated_chain.sh >> logs/20261105-sr2_crosstalk/chain5.out 2>&1 &
set -u
cd /home/will/workspace/hand || exit 1
D=docs/experiments/20261104-sr2_hand_crosstalk
CH=$D/plan/20261105-from32_chunks
OUT=logs/20261105-sr2_crosstalk/chain5
mkdir -p $OUT
run() {
  /usr/bin/python3 -u $D/lab/run_plan.py "$1" "$2" --token-file ~/.sr2_station_token \
    --boards logs/20261105-sr2_crosstalk/boards --board-channels 0,1,3 --max-err-yaw 8 --max-blocked 1 --keep-home
}
for L in 1_sv1_w6689 2_rv04_mid 3_sv1_w0099 4_sv1_u0308 5_home; do
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
