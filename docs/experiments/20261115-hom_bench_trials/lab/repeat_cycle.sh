#!/bin/bash
# Unattended HOM twist repeats on the SR2 hand, tips off and on alternating (owner, 2026-11-17 18:0x, choosing the
# overnight runs: "HOM twist repeats", "we can be a little aggressive with trials"). trial_cycle.sh's steps for N cycles,
# with the lessons of 2026-11-17's evening: the putdown levels fully the first time (--level-frac 1 --level-max 30; at
# 14-16 deg the seat check held three twists' spheres), ready may move 70 mm sideways (the putdown leaves the sphere up
# to ~20 mm off the table's centre), and a putdown that fails leaves the fingers straightened when the model's path to
# zero is clear of the object (zero's own check). Before every cycle hom_setup.py diag; when a servo is at 50 C or more
# the cycle waits until every servo is under 40 C (the identification queue's rule). Stops at the first step that fails,
# with the hand where it is; V/STOP stops before the next cycle.
# Run from docs/experiments/20261115-hom_bench_trials, with the HOM servers up (OFF and ON ports, --tips off / on):
#     lab/repeat_cycle.sh V [N]       (env: P, OFF, ON, DRIFT, LABEL)
set -u
V=${1:?session folder}; N=${2:-6}
P=${P:-plans/20261117-film_x70_y110_twist_gr_mcpstop_plan.json}
OFF=${OFF:-47051}; ON=${ON:-47052}; DRIFT=${DRIFT:-25}; LABEL=${LABEL:-g0x70y110}
M=maps/20261116-sphere_fitted_map_O.json
PY=/usr/bin/python3
mkdir -p "$V"
log() { echo "[$(date +%H:%M:%S)] $*"; }
step() {            # step NAME CMD...: runs CMD, keeps its output in OUT (and in V), its exit status in RC
  local name=$1; shift
  log "$name"
  OUT=$("$@" 2>&1 | grep -v -i token)
  RC=${PIPESTATUS[0]}
  echo "$OUT" >> "$V/$(date +%Y%m%d)_${name// /_}.log"
}
tmax() { $PY lab/hom_setup.py diag 2>&1 | grep temperature | $PY -c "import sys,re; print(max(float(x) for x in re.findall(r\"'\d+': ([0-9.]+)\", sys.stdin.read())))"; }
cool() {
  local t; t=$(tmax)
  log "servos at most ${t:-?} C"
  if awk "BEGIN{exit !(${t:-99} >= 50)}"; then
    log "waiting for every servo under 40 C"
    for k in $(seq 1 45); do sleep 60; t=$(tmax); awk "BEGIN{exit !(${t:-99} < 40)}" && { log "cooled to $t C"; return 0; }; done
    log "still $t C after 45 min: stopping"; exit 6
  fi
}
safe_end() {        # after a failed putdown: straighten the fingers if the sphere is clear of them, then stop
  step "zero after a failure" $PY lab/hom_setup.py zero --plan $P --object sphere --map $M --go
  echo "$OUT" | tail -2 | cut -c1-200
  log "$1: stopping"; exit "$2"
}
step layout $PY lab/hom_setup.py layout --plan $P --go
echo "$OUT" | tail -1 | cut -c1-160
[ $RC -eq 0 ] || { log "LAYOUT FAILED: $(echo "$OUT" | tail -2)"; exit 1; }
for k in $(seq 1 $N); do
  [ -e "$V/STOP" ] && { log "STOP before cycle $k"; exit 0; }
  if [ $((k % 2)) -eq 1 ]; then PORT=$OFF; TIPS=off; else PORT=$ON; TIPS=on; fi
  log "cycle $k of $N: tips $TIPS (port $PORT)"
  cool
  for j in 1 2; do
    step "place" $PY lab/hom_setup.py place --plan $P --map $M --go --max-level 35 --video-dir $V
    echo "$OUT" | grep -v "^   video" | tail -1 | cut -c1-160
  done
  for guard in "" "--watch-mm 6 --watch-deg 15"; do
    step "ready" $PY lab/hom_setup.py ready --plan $P --object sphere --map $M --go --video-dir $V --max-lat 70 $guard
    echo "$OUT" | grep -E "READY|STOPPED|too far|refused" | tail -2 | cut -c1-200
    echo "$OUT" | grep -q "READY for the trial" && break
  done
  echo "$OUT" | grep -q "READY for the trial" || { log "NOT READY: stopping"; exit 2; }
  step "trial" $PY lab/hom_trial.py trial --task twist --controller 100.65.73.59:$PORT --object sphere --map $M \
      --station http://10.99.99.2:8765 --live --arm --ft --out-dir $V --video-dir $V --label ${LABEL}_tips$TIPS \
      --max-drift-mm $DRIFT --plan $P
  echo "$OUT" | grep -E "STOP:|trial done|fingertips:" | tail -3 | cut -c1-200
  echo "$OUT" | grep -q "trial done" || { log "TRIAL FAILED: $(echo "$OUT" | tail -2 | cut -c1-200)"; safe_end "trial failed" 3; }
  step "putdown" $PY lab/hom_setup.py putdown --plan $P --object sphere --map $M --go --video-dir $V --touch-lat 3 \
      --level-frac 1 --level-max 30
  echo "$OUT" | grep -E -i "carried|touchdown:|  down |not seated|flex back|LET GO|STOPPED|refus|tracker does not" | tail -5 | cut -c1-200
  if [ $RC -ne 0 ] && echo "$OUT" | grep -q "is not seated"; then
    step "lift 25 mm" $PY lab/hom_setup.py place --plan $P --map $M --clear 25 --no-level --go
    step "putdown again" $PY lab/hom_setup.py putdown --plan $P --object sphere --map $M --go --video-dir $V \
        --touch-lat 3 --level-frac 1 --level-max 30
    echo "$OUT" | grep -E -i "touchdown:|  down |not seated|flex back|LET GO|STOPPED|refus" | tail -4 | cut -c1-200
  fi
  { [ $RC -eq 0 ] && echo "$OUT" | grep -q "LET GO"; } || safe_end "PUTDOWN FAILED" 4
  log "cycle $k done"
done
$PY lab/hom_setup.py diag 2>&1 | grep -E "volts|temperature|status" | cut -c1-200
log "repeats done: $N cycles"
