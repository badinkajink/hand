#!/bin/bash
# The owner's go of 15:0x on 11-02 ("you have my go ahead to run the remaining runs"): after seat 1's run (s1v6b) ends
# cleanly, seat 2 (v3 ID 5), then seat 3 (v3 ID 4), each from where the previous one ends (over its seat, joint 5 in the
# work window: the program goes by the next seat's hub). A run is clean when it exits 0 with its last grid action
# printed, no "three aborts" and no traceback.
# A run stopped by a HEX-E link loss (three aborts, each on the HEX-E's stream) is resumed on its own session folder once
# the HEX-E streams again (lab/hexe_probe.py every 60 s, for up to 3 h), at most 3 times per seat; the robot then backs
# off along the stem and comes in by the seat's hub, as at a seam. Anything else stops the chain, and the arm waits where
# the run left it. Each step goes to logs/20261102-chain_seats23.log.
cd /home/will/workspace/hand || exit 1
L=docs/experiments/20261022-ur5_tip_calibration/lab
HB=/dev/serial/by-id/usb-Adafruit_QT_Py_ESP32-S3__4MB_Flash_2MB_PSRAM__b4:3a:45:b0:d9:94-if00
OUT=logs/20261102-chain_seats23.log
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

start() {                           # $1 name, $2 plan, $3... more program options; prints the printout's path
    local name=$1 plan=$2
    shift 2
    PLAN=$plan SEAT=$L/poses.jsonl:strip_key1_seated_4 bash $L/tipcal_run.sh "$name" --stages preflight,homing,pad,register,grid \
        --held-board $HB --frame $L/20261102-strip_frame_two_keys.json "$@" >> $OUT 2>&1
    sleep 5
    ls -t logs/*-tipcal-"$name".log | head -1
}

finish() {                          # $1 printout, $2 grid count, $3 name, $4 plan: resume after HEX-E losses; 0 when clean
    local log=$1 n=$2 name=$3 plan=$4 k=0 dir
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
        log=$(start "${name}r$k" "$plan" --resume "$dir")
        say "resumed $name as ${name}r$k, printing to $log"
        wait_exit "$log"
    done
    return 0
}

prev=logs/20261102-145009-tipcal-s1v6b.log
say "waiting for $prev to end"
finish "$prev" 294 s1v6b $L/20261102-plan_probe_table_seat1_v3id6.json || exit 1
for spec in "s2v5 $L/20261102-plan_probe_table_seat2_v3id5.json" "s3v4 $L/20261102-plan_probe_table_seat3_v3id4.json"; do
    set -- $spec
    n=$(/usr/bin/python3 -c "import json; print(sum(a['stage'] == 'grid' for a in json.load(open('$2'))['actions']))")
    say "starting $1 ($2, $n grid actions)"
    log=$(start "$1" "$2")
    say "$1 printing to $log"
    finish "$log" "$n" "$1" "$2" || exit 1
    say "$1 ended cleanly"
done
say "the chain is done"
