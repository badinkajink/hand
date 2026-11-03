#!/bin/bash
# The owner, 15:4x on 11-03 in the lab's session: "relays count yes, we can go on the sweep test after 16:37 since i doubt
# we're getting very good info right now". In place of seat 3's 2 mm grid (chain_sphere2e stopped by its PID at 15:42; s2sph2
# runs on): wait for s2sph2 (seat 2, v3 ID 5) to end, resuming it after a HEX-E loss as the chains did; then the sweep test
# on seat 2 (lab/20261103-plan_probe_table_sphere2_seat2_v3id5_sweep_test.json: the workstation's 8189c5ff less pass 1;
# 10 face sweeps between two reference presses, about 22 min) on a --keep-probe seed of s2sph2's session, stages
# preflight,grid, on the program with the sweep action (a2f44c8f). Then the chain ends and the arm waits over seat 2 for the
# rod's swap. Each step goes to logs/20261103-chain_sweep2.log.
cd /home/will/workspace/hand || exit 1
L=docs/experiments/20261022-ur5_tip_calibration/lab
FRAME=docs/experiments/20261030-inverted_rig_seat_block/data/20261102-strip_frame_pads.json
HB=/dev/serial/by-id/usb-Adafruit_QT_Py_ESP32-S3__4MB_Flash_2MB_PSRAM__b4:3a:45:b0:d9:94-if00
OUT=logs/20261103-chain_sweep2.log
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

T=logs/20261103-tip_calibration
TEST=$L/20261103-plan_probe_table_sphere2_seat2_v3id5_sweep_test.json
say "waiting for logs/20261103-144807-tipcal-s2sph2.log to end"
finish logs/20261103-144807-tipcal-s2sph2.log s2sph2 $L/20261103-plan_probe_table_sphere2_seat2_v3id5_quick_neck_fast.json || exit 1
SRC=$(sed -n 's/^session in //p' logs/*-tipcal-s2sph2*.log | tail -1)
say "s2sph2 ended cleanly; seat 2's session is $SRC"
SEED=$T/$(date +%H%M%S)-s2sweep_seed
/usr/bin/python3 $L/probe_seed_session.py "$SRC" $TEST $SEED --keep-probe >> $OUT 2>&1 || { say "the seed failed; stopping"; exit 1; }
say "starting s2sweep ($TEST, $(n_grid $TEST) grid actions) on $SEED"
log=$(PLAN=$TEST start s2sweep $TEST --resume $SEED --stages preflight,grid)
say "s2sweep printing to $log"
STAGES=preflight,grid finish "$log" s2sweep $TEST || exit 1
say "s2sweep ended cleanly; the arm waits over seat 2 for the rod"
