#!/bin/bash
# chain_sweep2 as changed at 16:0x on 11-03 (it was stopped by its PID): the sweep test on seat 2 (v3 ID 5) runs as a new
# session with its own pass 1, not on a seed of s2sph2's fit. s2sph2's reference press (probe-ref-00 to -07, the same point
# every ~25 presses) stepped +0.45 mm along its push between presses 108 and 130 (15:36 to 15:43), and the same ids pressed
# 0.6 to 0.9 mm deeper at 0.5 N than on seat 1 after it (0.04 to 0.29 before): the sphere on its neck, the tip in its seat
# or the strip moved, and s2sph2's fit no longer stands. Pass 1 is s2sph2's 19 touches as they ran at 14:48; then the
# 12 grid actions as replayed (lab/20261103-plan_probe_table_sphere2_seat2_v3id5_sweep_test_pass1.json). The owner, 15:4x:
# "relays count yes, we can go on the sweep test after 16:37 since i doubt we're getting very good info right now".
# Wait for s2sph2 (resuming it after a HEX-E loss), then s2sweep with stages preflight,homing,pad,register,grid; the
# register stops the session before the grid if its fit is over 0.15 mm rms. Then the arm waits over seat 2 for the rod.
# Each step goes to logs/20261103-chain_sweep2b.log.
cd /home/will/workspace/hand || exit 1
L=docs/experiments/20261022-ur5_tip_calibration/lab
FRAME=docs/experiments/20261030-inverted_rig_seat_block/data/20261102-strip_frame_pads.json
HB=/dev/serial/by-id/usb-Adafruit_QT_Py_ESP32-S3__4MB_Flash_2MB_PSRAM__b4:3a:45:b0:d9:94-if00
OUT=logs/20261103-chain_sweep2b.log
STAGES=preflight,homing,pad,register,grid
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

TEST=$L/20261103-plan_probe_table_sphere2_seat2_v3id5_sweep_test_pass1.json
say "waiting for logs/20261103-144807-tipcal-s2sph2.log to end"
finish logs/20261103-144807-tipcal-s2sph2.log s2sph2 $L/20261103-plan_probe_table_sphere2_seat2_v3id5_quick_neck_fast.json || exit 1
say "s2sph2 ended cleanly; $(sed -n 's/^session in //p' logs/*-tipcal-s2sph2*.log | tail -1)"
say "starting s2sweep ($TEST: pass 1 and $(n_grid $TEST) grid actions), a new session"
log=$(start s2sweep $TEST --stages $STAGES)
say "s2sweep printing to $log"
finish "$log" s2sweep $TEST || exit 1
say "s2sweep ended cleanly; the arm waits over seat 2 for the rod"
