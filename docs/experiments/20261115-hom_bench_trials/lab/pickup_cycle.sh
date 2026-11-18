#!/bin/bash
# The pick-up of the sphere lying on its side, by controlled slip, as one command on corvus (owner, 2026-11-17 18:1x:
# "hopefully this task can be automated for next runs, without losing this interpretability. seems pretty robust at this
# point anyway"). The steps are those of README "State on 2026-11-16 15:51", each a run of lab/pinch_pickup.py or
# lab/hom_setup.py: plan (with the wrist turn), grasp, lift, slip, putdown (tracked touchdown, seat check, release,
# straighten, lift), unturn. Each step prints its key lines here and keeps its whole output in SESSION_DIR. The cycle
# stops at the first step that fails, with the hand where it is: look before going on.
# Run from docs/experiments/20261115-hom_bench_trials:
#     lab/pickup_cycle.sh SESSION_DIR [go]
# Without "go" only the plan runs (it moves nothing) and its checks are printed.
# Settings from the environment; the defaults are those of the 18:05 run of 2026-11-17 at 7.5 V:
#     SQUEEZE 5.3 (deg on the PIPs past touch; 8 at 5 V), STEP 0.35 (deg a slip step; 0.5 at 5 V), RESQUEEZE 0.7,
#     GAP 8 (mm the open pads stand off the ball), LAT 3 (deg above the equator), TABLE_MIN 7.5 (mm: the least
#     tip-to-table clearance at touch, on the conservative tip-end ball), MARGIN_MIN 3.5 (deg: the least joint margin),
#     BASE (a plan at the station's layout; the planner's default is at (105, 70)).
set -u
V=${1:?usage: lab/pickup_cycle.sh SESSION_DIR [go]}; GO=${2:-}
M=maps/20261116-sphere_fitted_map_O.json
PY=/usr/bin/python3
SQ=${SQUEEZE:-5.3}; STEP=${STEP:-0.35}; RESQ=${RESQUEEZE:-0.7}; GAP=${GAP:-8}; LAT=${LAT:-3}
TABLE_MIN=${TABLE_MIN:-7.5}; MARGIN_MIN=${MARGIN_MIN:-3.5}
mkdir -p "$V"
STAMP=$(date +%Y%m%d-%H%M%S)
P=$V/${STAMP}_pickup_plan.json
log() { echo "[$(date +%H:%M:%S)] $*"; }
step() {            # step NAME PATTERN CMD...: runs CMD, keeps its output in V, prints the lines matching PATTERN
  local name=$1 pat=$2; shift 2
  log "$name"
  OUT=$("$@" 2>&1 | grep -v -i token)
  RC=${PIPESTATUS[0]}
  echo "$OUT" > "$V/${STAMP}_${name}.log"
  echo "$OUT" | grep -E "$pat" | grep -v "^   video" | cut -c1-230
}
step diag_before "volts|temperature|status" $PY lab/hom_setup.py diag
step plan "object:|turns|pinch:|at touch|coming down|out of the way|moves:|plan ->|turn that|not lying|off across" \
    $PY lab/pinch_pickup.py plan --map $M --gap $GAP --squeeze $SQ --lat-deg $LAT --turn --out "$P" ${BASE:+--base-plan $BASE}
[ -f "$P" ] || { log "PLAN REFUSED: $(echo "$OUT" | tail -1)"; exit 1; }
CHECK=$($PY - "$P" "$TABLE_MIN" "$MARGIN_MIN" <<'EOF'
import json, sys
pp = json.load(open(sys.argv[1]))["meta"]["hom"]["pinch"]
table = pp["table_mm"]                  # the least tip-to-table clearance at touch (plan_pinch's table_mm)
ok = table >= float(sys.argv[2]) and pp["margin_deg"] >= float(sys.argv[3])
print(("OK" if ok else "BAD") + f" table {table:.1f} mm at touch (least {sys.argv[2]}), joint margin {pp['margin_deg']:.1f} deg "
      f"(least {sys.argv[3]}), wrist turn {pp.get('turn_deg', 0.0):+.1f} deg")
EOF
)
log "checks: $CHECK"
case $CHECK in OK*) ;; *) log "PLAN REFUSED by the checks"; exit 1 ;; esac
[ "$GO" = go ] || { log "plan only (add go to run it): $P"; exit 0; }
step grasp "at the grasp|at touch:|gripped:|over the close|GRASPED|STOPPED|refuse" \
    $PY lab/pinch_pickup.py grasp --plan "$P" --map $M --squeeze $SQ --go --video-dir "$V"
echo "$OUT" | grep -q "^GRASPED" || { log "GRASP FAILED: stopping"; exit 2; }
step lift "after the lift|LIFTED|LIFT STOPPED|STOPPED|refuse" \
    $PY lab/pinch_pickup.py lift --plan "$P" --map $M --go --video-dir "$V"
echo "$OUT" | grep -q "^LIFTED" || { log "LIFT FAILED: stopping"; exit 3; }
step slip "squeeze .* deg: tilt|slides|squeeze back|SLIP DONE|STOPPED|refuse" \
    $PY lab/pinch_pickup.py slip --plan "$P" --map $M --step-deg $STEP --resqueeze $RESQ --go --video-dir "$V"
echo "$OUT" | grep -q "^SLIP DONE" || { log "SLIP FAILED: stopping"; exit 4; }
step putdown "carried:|touchdown:|  down |not seated|released:|LET GO|STOPPED|refus|tracker does not" \
    $PY lab/hom_setup.py putdown --plan "$P" --object sphere --map $M --level-max 0 --go --video-dir "$V"
if ! echo "$OUT" | grep -q "^LET GO"; then
  if echo "$OUT" | grep -q "tracker does not"; then       # after a wrist turn the cameras may see 2 tags only (11-16)
    step setdown "touchdown|seated|STOPPED|refus|bottom" \
        $PY lab/pinch_pickup.py setdown --plan "$P" --map $M --go --video-dir "$V"
    [ $RC -eq 0 ] || { log "SETDOWN FAILED: stopping"; exit 5; }
    step letgo "released:|path to zero|LET GO|STOPPED|refus" \
        $PY lab/hom_setup.py letgo --plan "$P" --object sphere --map $M --go --video-dir "$V"
  fi
  echo "$OUT" | grep -q "^LET GO" || { log "PUTDOWN FAILED: stopping, the sphere may still be in the hand"; exit 5; }
fi
step unturn "UNTURNED|nothing to undo|STOPPED|refus" \
    $PY lab/pinch_pickup.py unturn --plan "$P" --map $M --go --video-dir "$V"
echo "$OUT" | grep -q -E "^UNTURNED|nothing to undo" || { log "UNTURN FAILED: stopping"; exit 6; }
step diag_after "volts|temperature|status" $PY lab/hom_setup.py diag
log "pick-up done: plan $P, logs ${STAMP}_*.log in $V"
