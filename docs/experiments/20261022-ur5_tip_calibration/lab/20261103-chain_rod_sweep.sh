#!/bin/bash
# The rod's sweeps across the tip (rod_deg 90), seats 2, 3 and 1, after the owner's swap to the rod and the owner's go: the
# workstation's sweep plans of c867f7d4 (copies here: lab/20261103-plan_probe_table_rod_seat{2,3,1}_v3id{5,4,6}_sweep.json;
# 20 touches 2.68 mm or more from the tip's body, 31 or 32 sweeps between two reference presses; chained 2, 3, 1 from seat
# 2's hub they replay with no flags). Each seat's session is seeded (probe_seed_session.py, c_M set to the rod's) from a
# sphere session of that seat: seat 2 from $1 (the sweep test's session or s2sph2's), seat 3 from s3v4 (11-02, the 5 mm
# sphere, two-key frame; --frame only acts in homing, which a seeded run skips), seat 1 from s1sph2b. Stages
# preflight,pad,register,grid; a run stopped by a HEX-E loss resumes once hexe_probe.py finds the sensor streaming, and
# anything else stops the chain. About 1.3 h a seat. Each step goes to logs/20261103-chain_rod_sweep.log.
# Usage: bash logs/20261103-chain_rod_sweep.sh SEAT2_SOURCE_SESSION
cd /home/will/workspace/hand || exit 1
L=docs/experiments/20261022-ur5_tip_calibration/lab
FRAME=docs/experiments/20261030-inverted_rig_seat_block/data/20261102-strip_frame_pads.json
HB=/dev/serial/by-id/usb-Adafruit_QT_Py_ESP32-S3__4MB_Flash_2MB_PSRAM__b4:3a:45:b0:d9:94-if00
OUT=logs/20261103-chain_rod_sweep.log
STAGES=preflight,pad,register,grid
say() { echo "$(date +%H:%M:%S) $*" | tee -a $OUT; }

wait_exit() {                       # the run's printout ends with "exit N at HH:MM:SS"
    until grep -q "^exit [0-9]* at " "$1" 2>/dev/null; do sleep 10; done
}

clean() {                           # $1 printout, $2 the grid's action count
    grep -q "^exit 0 at " "$1" && ! grep -q -E "three aborts|Traceback" "$1" && grep -q "($2/$2)" "$1"
}

hexe_loss() {                       # three aborts in a row, the last three ABORT lines all on the HEX-E's stream
    grep -q "three aborts" "$1" && ! grep -q "Traceback" "$1" \
        && [ "$(grep "ABORT" "$1" | tail -n 3 | grep -c -E "HEX-E")" = 3 ]
}

n_grid() {
    /usr/bin/python3 -c "import json; print(sum(a['stage'] == 'grid' for a in json.load(open('$1'))['actions']))"
}

start() {                           # $1 name, $2 plan, $3... more program options; prints the printout's path
    local name=$1 plan=$2
    shift 2
    PLAN=$plan SEAT=$L/poses.jsonl:strip_key1_seated_4 bash $L/tipcal_run.sh "$name" --held-board $HB \
        --frame $FRAME "$@" >> $OUT 2>&1
    sleep 5
    ls -t logs/*-tipcal-"$name".log | head -1
}

finish() {                          # $1 printout, $2 name, $3 plan: resume after HEX-E losses; 0 when clean
    local log=$1 name=$2 plan=$3 n k=0 dir
    n=$(n_grid "$plan")
    wait_exit "$log"
    while ! clean "$log" "$n"; do
        if ! hexe_loss "$log" || [ $k -ge 3 ]; then
            say "$log did not end cleanly ($(grep '^exit' "$log" | tail -1); resumes $k); stopping the chain"
            return 1
        fi
        k=$((k + 1))
        dir=$(sed -n 's/^session in //p' "$log" | tail -1)
        say "$log stopped on a HEX-E loss; waiting for the HEX-E to stream (resume $k of $name on $dir)"
        local t0=$SECONDS
        until /usr/bin/python3 $L/hexe_probe.py >> $OUT 2>&1; do
            if [ $((SECONDS - t0)) -gt 10800 ]; then
                say "the HEX-E did not come back in 3 h; stopping the chain"
                return 1
            fi
            sleep 60
        done
        log=$(start "${name}r$k" "$plan" --resume "$dir" --stages $STAGES)
        say "resumed $name as ${name}r$k, printing to $log"
        wait_exit "$log"
    done
    return 0
}

T=logs/20261103-tip_calibration
SRC2=${1:?the seat 2 source session}
[ -f "$SRC2/registration.json" ] || { echo "no registration in $SRC2"; exit 1; }
for row in "2 5 $SRC2" "3 4 logs/20261102-tip_calibration/220337-s3v4" "1 6 logs/20261103-tip_calibration/122020-s1sph2b"; do
    set -- $row
    PLAN=$L/20261103-plan_probe_table_rod_seat$1_v3id$2_sweep.json
    SEED=$T/$(date +%H%M%S)-rodsw$1_seed
    /usr/bin/python3 $L/probe_seed_session.py "$3" $PLAN $SEED >> $OUT 2>&1 || { say "seat $1's seed failed; stopping"; exit 1; }
    say "starting rodsw$1 ($PLAN, $(n_grid $PLAN) grid actions) on $SEED, seeded from $3"
    log=$(start rodsw$1 $PLAN --resume $SEED --stages $STAGES)
    say "rodsw$1 printing to $log"
    finish "$log" rodsw$1 $PLAN || exit 1
    say "rodsw$1 ended cleanly"
done
say "the rod's three seats are done; the arm waits over seat 1"
