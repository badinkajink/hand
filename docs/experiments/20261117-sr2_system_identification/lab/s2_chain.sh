#!/bin/bash
# S2 of the identification queue (QUEUE.md), 2026-11-17: the free-air servo identification of 11-06 on the SR2 hand at
# 7.5 V. The blocks a1 b1 a2 b2 c c40 of 20261104-sr2_hand_crosstalk/lab/make_sysid_plan.py, built by
# make_sysid_plan_at.py and checked in the CAD hand at the station's layout (the roll press plan's (85, 170) layout:
# 34-51 mm least in the moves, c40 10.0 mm; at (70, 110) c40 collided, -6.3 mm). Each block is one run_plan.py run with
# the flags of sysid_chain.sh (--ramp-deg 0, --min-clearance 5, --max-blocked 1, --keep-home, --max-err-yaw 10 or 15)
# and --max-temp 55. The queue's rules: lab/hom_setup.py diag before and after each block; when a servo is at 50 C or
# more, wait until every servo is under 40 C. A block that does not end with its plan done ends the chain. OUT/STOP
# ends it between blocks. The tables (dwells.tsv, holds_c.tsv) by sysid_tables.py after each block.
#     nohup setsid bash docs/experiments/20261117-sr2_system_identification/lab/s2_chain.sh OUT >> OUT/chain.out 2>&1 &
set -u
cd /home/will/workspace/hand || exit 1
D=docs/experiments/20261104-sr2_hand_crosstalk
H=docs/experiments/20261115-hom_bench_trials
OUT=${1:?OUT}
PY=/usr/bin/python3
log() { echo "[$(date +%H:%M:%S)] $*"; }
diag() { (cd $H && $PY lab/hom_setup.py diag 2>&1 | grep -v -i token | grep -E "volts|temperature|status|rebooted"); }
tmax() { (cd $H && $PY lab/hom_setup.py diag 2>&1 | grep temperature | $PY -c "import sys,re; print(max(float(x) for x in re.findall(r\"'\d+': ([0-9.]+)\", sys.stdin.read())))"); }
cool() {
  local t; t=$(tmax)
  if awk "BEGIN{exit !(${t:-99} >= 50)}"; then
    log "a servo at $t C: waiting for every servo under 40 C"
    for k in $(seq 1 45); do sleep 60; t=$(tmax); awk "BEGIN{exit !(${t:-99} < 40)}" && { log "cooled to $t C"; return 0; }; done
    log "still $t C after 45 min: stopping"; exit 4
  fi
}
log "S2 start, layout: $(cd $H && $PY lab/hom_setup.py status 2>&1 | grep -o "mounts {[^}]*}[^}]*}[^}]*}" | head -1)"
for B in ${BLOCKS:-a1 b1 a2 b2 c c40}; do
  [ -e $OUT/STOP ] && { log "STOP before $B"; exit 3; }
  diag; cool
  case $B in c*) YAW=15 ;; *) YAW=10 ;; esac
  log "block $B"
  $PY -u $D/lab/run_plan.py $OUT/plan/20261106-sysid_$B.json $OUT/$B --token-file ~/.sr2_station_token \
    --ramp-deg 0 --min-clearance 5 --max-blocked 1 --keep-home --max-err-yaw $YAW --max-temp 55
  rc=$?
  log "block $B ended (exit $rc): $(tail -1 $OUT/$B/events.jsonl | cut -c1-240)"
  nice -n 10 $PY $D/lab/sysid_tables.py $OUT $OUT/tables > $OUT/tables_$B.log 2>&1 || log "tables failed (see tables_$B.log)"
  diag
  [ $rc -eq 0 ] || { log "chain stopped after $B"; exit 3; }
done
log "S2 done"
