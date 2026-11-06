#!/usr/bin/env bash
# The free-space servo identification of 11-06 on the SR2 hand, from lab/make_sysid_plan.py's plans in OUT/plan. The
# blocks a1 b1 a2 b2 c c40 run in that order, each as its own run_plan.py run in OUT/<block>, with --ramp-deg 0,
# --min-clearance 5, --max-blocked 1 and --keep-home; --max-err-yaw is 10 for a and b, 15 for c and c40. After each
# block, lab/sysid_tables.py writes the tables and lab/push_text.py commits them as plain .tsv to the branch
# sr2-sysid-20261106. It pushes once; if the push fails, the data stay on corvus and the chain goes on. A block that
# does not end with its plan done ends the chain, since a blocked pose waits for the owner's look. OUT/STOP ends the
# chain, also inside a block. BLOCKS in the environment picks the blocks (default all six); STATION points the runner at
# another station (a dry run on lab's mock), and NOPUSH=1 skips the push.
#   nohup setsid bash docs/experiments/20261104-sr2_hand_crosstalk/lab/sysid_chain.sh [OUT] >> OUT/chain.out 2>&1 &
set -u
cd /home/will/workspace/hand || exit 1
D=docs/experiments/20261104-sr2_hand_crosstalk
S=docs/experiments/20261106-sr2_servo_sysid
OUT=${1:-logs/20261106-sr2_sysid}
for B in ${BLOCKS:-a1 b1 a2 b2 c c40}; do
  [ -e $OUT/STOP ] && { echo "$(date +%T) STOP before $B"; exit 3; }
  case $B in c*) YAW=15 ;; *) YAW=10 ;; esac
  echo "$(date +%T) block $B"
  /usr/bin/python3 -u $D/lab/run_plan.py $OUT/plan/20261106-sysid_$B.json $OUT/$B --token-file ~/.sr2_station_token \
    --ramp-deg 0 --min-clearance 5 --max-blocked 1 --keep-home --max-err-yaw $YAW ${STATION:+--station $STATION}
  rc=$?
  echo "$(date +%T) block $B ended (exit $rc): $(tail -1 $OUT/$B/events.jsonl | cut -c1-240)"
  nice -n 10 /usr/bin/python3 $D/lab/sysid_tables.py $OUT $OUT/tables && [ -z "${NOPUSH:-}" ] && \
    nice -n 10 /usr/bin/python3 $D/lab/push_text.py --branch sr2-sysid-20261106 --base origin/main \
      --message "$(printf 'SR2 servo sysid of 11-06: block %s ended %s (exit %s); the dwells and holds tables\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>' $B "$(date +%H:%M:%S)" $rc)" \
      $OUT/tables/dwells.tsv=$S/data/dwells.tsv $OUT/tables/holds_c.tsv=$S/data/holds_c.tsv
  [ $rc -eq 0 ] || { echo "$(date +%T) chain stopped after $B"; exit 3; }
done
echo "$(date +%T) chain done"
