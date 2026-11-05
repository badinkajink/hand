#!/usr/bin/env bash
# The night's crosstalk sweep of 2026-11-04 on the SR2 hand: plan_home, random s1, plan_home, random s2 (about 3.3 h),
# each in its own folder under logs/20261104-sr2_crosstalk/. The chain stops at the first plan that does not end
# normally; that run has already put the hand at the zero pose and disabled the motors. The last plan disables them
# when it ends. touch logs/20261104-sr2_crosstalk/STOP stops the chain after the current step.
#   nohup setsid bash docs/experiments/20261104-sr2_hand_crosstalk/lab/night_chain.sh > logs/20261104-sr2_crosstalk/chain.out 2>&1 &
set -u
cd /home/will/workspace/hand || exit 1
D=docs/experiments/20261104-sr2_hand_crosstalk
OUT=logs/20261104-sr2_crosstalk
run() {
  echo "$(date +%H:%M:%S) start $1 -> $OUT/$2"
  /usr/bin/python3 -u $D/lab/run_plan.py $D/plan/$1 $OUT/$2 --token-file ~/.sr2_station_token "${@:3}"
  local rc=$?
  echo "$(date +%H:%M:%S) end $2: exit $rc"
  return $rc
}
run 20261104-plan_home.json run1_home --no-disable || exit 3
run 20261104-plan_random_s1.json run2_random_s1 --no-disable || exit 3
run 20261104-plan_home.json run3_home --no-disable || exit 3
run 20261104-plan_random_s2.json run4_random_s2 || exit 3
echo "$(date +%H:%M:%S) chain done"
