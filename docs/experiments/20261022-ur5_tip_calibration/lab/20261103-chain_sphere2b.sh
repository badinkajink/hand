#!/bin/bash
# The 2 mm sphere's chain again after s1sph2 (stopped 11:25 on 11-03; NOTES, "s1sph2: a touch on the tip's body and a bad
# fit"), on the program whose register leaves out a touch over 0.5 mm off and stops over 0.15 mm rms, and whose runs stop
# at their third early contact or third press with no level. Runs only on the owner's go in the lab's session after the
# owner's look at v3 ID 6. A new session per seat with its own pass 1, on the pad-fitted frame (--frame): seat 1 (v3 ID 6)
# and seat 2 (v3 ID 5) less pad-tip-349-o35-a270 (_no349), then seat 3 (v3 ID 4) as before, each from where the one
# before ends (over its seat, joint 5 in the work window: the program goes by the next seat's hub). Seat 1 starts from
# where s1sph2 stopped over seat 1 (that first travel checked by the workstation's replay code from the robot's joints).
# A run is clean when it exits 0 with its last grid action printed, no "three aborts" and no traceback. A run stopped by
# a HEX-E link loss (three aborts, each on the HEX-E's stream) is resumed on its session once the HEX-E streams again
# (lab/hexe_probe.py every 60 s, up to 3 h), at most 3 times a run. Anything else stops the chain, and the arm waits where
# the run left it. After seat 3 the chain ends: the rod needs the owner's tool change and go. Each step goes to
# logs/20261103-chain_sphere2b.log.
cd /home/will/workspace/hand || exit 1
L=docs/experiments/20261022-ur5_tip_calibration/lab
FRAME=docs/experiments/20261030-inverted_rig_seat_block/data/20261102-strip_frame_pads.json
HB=/dev/serial/by-id/usb-Adafruit_QT_Py_ESP32-S3__4MB_Flash_2MB_PSRAM__b4:3a:45:b0:d9:94-if00
OUT=logs/20261103-chain_sphere2b.log
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

for spec in "1 s1sph2b $L/20261103-plan_probe_table_sphere2_seat1_v3id6_no349.json" \
            "2 s2sph2 $L/20261103-plan_probe_table_sphere2_seat2_v3id5_no349.json" \
            "3 s3sph2 $L/20261103-plan_probe_table_sphere2_seat3_v3id4.json"; do
    set -- $spec
    say "starting $2 on seat $1 ($3, $(n_grid $3) grid actions)"
    log=$(start "$2" "$3" --stages $STAGES)
    say "$2 printing to $log"
    finish "$log" "$2" "$3" || exit 1
    say "$2 ended cleanly; seat $1's session is $(sed -n 's/^session in //p' logs/*-tipcal-"$2"*.log | tail -1)"
done
say "the 2 mm sphere's three seats are done; the arm waits over seat 3 for the rod"
