#!/usr/bin/env bash
# The night chain after run1b_home_rest stopped at 19:43:59 on 11-04: the index stalled at mcp 50 + pip 90 with both
# servos pushing, on something outside the CAD model. The plans keep the index's and middle's mcp + pip at 120 deg at
# most (plan_home_rest2, the _v3 plans), and run_plan.py backs out of a pose it cannot reach and stops at the third.
# The rest of the structured plan, random s1, the structured plan and random s2: about 2 h,
# each in its own folder under logs/20261104-sr2_crosstalk/. The chain stops at the first plan that does not end
# normally; that run has already put the hand at the zero pose and disabled the motors. The last plan disables them
# when it ends. touch logs/20261104-sr2_crosstalk/STOP stops the chain after the current step. With the argument
# resume, each run starts after its last finished step (progress.json), and a finished run is skipped.
#   nohup setsid bash docs/experiments/20261104-sr2_hand_crosstalk/lab/night_chain3.sh [resume] >> logs/20261104-sr2_crosstalk/chain.out 2>&1 &
set -u
cd /home/will/workspace/hand || exit 1
D=docs/experiments/20261104-sr2_hand_crosstalk
OUT=logs/20261104-sr2_crosstalk
RESUME=()
[ "${1:-}" = resume ] && RESUME=(--resume)
run() {
  echo "$(date +%H:%M:%S) start $1 -> $OUT/$2"
  /usr/bin/python3 -u $D/lab/run_plan.py $D/plan/$1 $OUT/$2 --token-file ~/.sr2_station_token "${RESUME[@]}" "${@:3}"
  local rc=$?
  echo "$(date +%H:%M:%S) end $2: exit $rc"
  return $rc
}
run 20261104-plan_home_rest2.json run1c_home_rest2 --no-disable || exit 3
run 20261104-plan_random_s1_v3.json run2_random_s1 --no-disable || exit 3
run 20261104-plan_home_v3.json run3_home --no-disable || exit 3
run 20261104-plan_random_s2_v3.json run4_random_s2 || exit 3
echo "$(date +%H:%M:%S) chain done"
