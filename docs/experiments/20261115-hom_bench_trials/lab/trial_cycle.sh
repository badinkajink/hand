#!/bin/bash
# One blind-trial cycle on corvus (hand-4a, 2026-11-16 20:1x; owner: "LETS go for the full runs for conclusive data. i
# want to end the presentation strong, with some preliminary evidence that we can get approximately close to real in
# simulation first"). The plan loaded; the hand placed and levelled (twice: a large levelling turn outruns place's
# timeout); the tracked approach, with wider tracker guards on a second try (off the table's centre the tracker sees 2
# tags and its pose jumps); the live trial under the HOM server at PORT with the drift guard at MAX_DRIFT_MM; the putdown
# that carries the sphere to the table's centre, and when its seat check refuses the sphere's tilt, a lift of 25 mm and
# a second putdown that levels it fully. Stops at the first step that fails, with the hand where it is.
# Run from docs/experiments/20261115-hom_bench_trials:
#     lab/trial_cycle.sh PLAN PORT LABEL [MAX_DRIFT_MM]
set -u
P=$1; PORT=$2; LABEL=$3; DRIFT=${4:-35}
M=maps/20261116-sphere_fitted_map_O.json
V=../../../logs/20261116-hom_trials/20261116-budget_twists
PY=/usr/bin/python3
log() { echo "[$(date +%H:%M:%S)] $*"; }
step() {            # step NAME CMD...: runs CMD, keeps its output in OUT and its exit status in RC
  local name=$1; shift
  log "$name"
  OUT=$("$@" 2>&1 | grep -v -i token)
  RC=${PIPESTATUS[0]}
}
step layout $PY lab/hom_setup.py layout --plan $P --go
echo "$OUT" | tail -1 | cut -c1-120
[ $RC -eq 0 ] || { log "LAYOUT FAILED: $(echo "$OUT" | tail -2)"; exit 1; }
for k in 1 2; do
  step "place $k" $PY lab/hom_setup.py place --plan $P --map $M --go --max-level 35 --video-dir $V
  echo "$OUT" | grep -v "^   video" | tail -1
done
for guard in "" "--watch-mm 6 --watch-deg 15"; do
  step "ready $guard" $PY lab/hom_setup.py ready --plan $P --object sphere --map $M --go --video-dir $V $guard
  echo "$OUT" | grep -E "READY|STOPPED" | tail -2
  echo "$OUT" | grep -q "READY for the trial" && break
done
echo "$OUT" | grep -q "READY for the trial" || { log "NOT READY: stopping"; exit 2; }
step "trial $LABEL" $PY lab/hom_trial.py trial --task twist --controller 100.65.73.59:$PORT --object sphere --map $M \
    --station http://10.99.99.2:8765 --live --arm --ft --out-dir $V --video-dir $V --label $LABEL --max-drift-mm $DRIFT
echo "$OUT" | tail -3
step putdown $PY lab/hom_setup.py putdown --plan $P --object sphere --map $M --go --video-dir $V --touch-lat 3
echo "$OUT" | grep -E "carried|touchdown:|  down |not seated|LET GO|STOPPED|tracker does not" | tail -5
if [ $RC -ne 0 ] && echo "$OUT" | grep -q "is not seated"; then
  step "lift 25 mm" $PY lab/hom_setup.py place --plan $P --map $M --clear 25 --no-level --go
  echo "$OUT" | tail -1
  step "putdown, levelled fully" $PY lab/hom_setup.py putdown --plan $P --object sphere --map $M --go --video-dir $V \
      --touch-lat 3 --level-frac 1 --level-max 30
  echo "$OUT" | grep -E "touchdown:|  down |not seated|LET GO|STOPPED" | tail -4
fi
{ [ $RC -eq 0 ] && echo "$OUT" | grep -q "LET GO"; } || { log "PUTDOWN FAILED: stopping"; exit 3; }
log "cycle done"
