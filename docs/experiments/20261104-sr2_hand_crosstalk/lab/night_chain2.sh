#!/usr/bin/env bash
# The night chain after run1_home stopped at 19:31 on 11-04, its index mcp stopped near 55 deg (its EEPROM angle limit,
# +58.6 deg, less droop, found on 11-06; not the protective torque as first thought): the rest of the structured plan, then random s1, the structured plan and random s2, all with the index mcp at
# 50 deg at most and the middle mcp at 62 (plan_*_v2, plan_home_rest; about 2.2 h),
# each in its own folder under logs/20261104-sr2_crosstalk/. The chain stops at the first plan that does not end
# normally; that run has already put the hand at the zero pose and disabled the motors. The last plan disables them
# when it ends. touch logs/20261104-sr2_crosstalk/STOP stops the chain after the current step. With the argument
# resume, each run starts after its last finished step (progress.json), and a finished run is skipped.
#   nohup setsid bash docs/experiments/20261104-sr2_hand_crosstalk/lab/night_chain2.sh [resume] >> logs/20261104-sr2_crosstalk/chain.out 2>&1 &
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
run 20261104-plan_home_rest.json run1b_home_rest --no-disable || exit 3
run 20261104-plan_random_s1_v2.json run2_random_s1 --no-disable || exit 3
run 20261104-plan_home_v2.json run3_home --no-disable || exit 3
run 20261104-plan_random_s2_v2.json run4_random_s2 || exit 3
echo "$(date +%H:%M:%S) chain done"
