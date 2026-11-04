#!/bin/bash
# 11-04's rod points before the SR2 mounting (the owner, relayed at 10:44: "if we just lost density of coverage why dont
# we just add points now. i know i said we were done earlier, but we should have a few hours left to get that density").
# The night's chain (logs/20261103-chain_rod_night.sh) with the day's times: no step starts after LAST_START, and
# logs/20261104-deadline_stop.sh, started here, stops the chain and then the program right after a finished action at
# STOP_AT. Each step is one robot session, in order:
#   seed SEAT PLAN SOURCE  a new session seeded from SOURCE (probe_seed_session.py, c_M set to the rod's), stages
#                          preflight,pad,register,grid (pass 1, the rod's fit, then the plan's grid)
#   more SEAT PLAN         the same seat's session again (the last seed step's), stages preflight,grid
#   resume SEAT PLAN DIR   the session DIR (of an earlier chain) again, stages preflight,grid
# A run stopped by a HEX-E loss resumes once hexe_probe.py finds the sensor streaming; anything else stops the chain.
# Each step goes to logs/20261104-chain_rod_points.log; the runs are named pts<seat><step>.
# Usage: bash logs/20261104-chain_rod_points.sh STEPS_FILE LAST_START STOP_AT   (HH:MM, today)
cd /home/will/workspace/hand || exit 1
L=docs/experiments/20261022-ur5_tip_calibration/lab
FRAME=docs/experiments/20261030-inverted_rig_seat_block/data/20261102-strip_frame_pads.json
HB=/dev/serial/by-id/usb-Adafruit_QT_Py_ESP32-S3__4MB_Flash_2MB_PSRAM__b4:3a:45:b0:d9:94-if00
OUT=logs/20261104-chain_rod_points.log
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

T=logs/20261104-tip_calibration
STEPS=${1:?the steps file}
LAST_START=${2:?the last start, HH:MM}
STOP_AT=${3:?the hard stop, HH:MM}
pgrep -f "[2]0261104-deadline_stop.sh" > /dev/null || bash logs/20261104-deadline_stop.sh $STOP_AT $$ > /dev/null 2>&1 < /dev/null &
say "chain $$ from $STEPS; no step starts after $LAST_START; the hard stop at $STOP_AT (logs/20261104-deadline_stop.log)"
late() {                            # past LAST_START today
    [ "$(date +%H%M)" -ge "${LAST_START/:/}" ]
}
declare -A DIR                      # each seat's session folder
mapfile -t ROWS < "$STEPS"         # read whole: no command in a step can take the file's lines
k=0
for row in "${ROWS[@]}"; do
    read -r kind seat plan src <<< "$row"
    [ -z "$kind" ] || [ "${kind:0:1}" = "#" ] && continue
    k=$((k + 1))
    if late; then
        say "step $k ($kind seat $seat) not started: past $LAST_START; the chain ends"
        exit 0
    fi
    name=pts$seat$k
    if [ "$kind" = seed ]; then
        SEED=$T/$(date +%H%M%S)-${name}_seed
        /usr/bin/python3 $L/probe_seed_session.py "$src" $plan $SEED >> $OUT 2>&1 || { say "step $k: the seed failed; stopping"; exit 1; }
        say "step $k: $name ($plan, $(n_grid $plan) grid actions) on $SEED, seeded from $src"
        log=$(start $name $plan --resume $SEED --stages $STAGES)
        STG=$STAGES
    else
        [ "$kind" = resume ] && DIR[$seat]=$src     # resume SEAT PLAN DIR: a session of an earlier chain
        [ -n "${DIR[$seat]}" ] || { say "step $k: no session of seat $seat yet; stopping"; exit 1; }
        say "step $k: $name ($plan, $(n_grid $plan) grid actions) on seat $seat's session ${DIR[$seat]}"
        log=$(start $name $plan --resume ${DIR[$seat]} --stages preflight,grid)
        STG=preflight,grid
    fi
    say "$name printing to $log"
    STAGES=$STG finish "$log" $name $plan || exit 1
    d=$(sed -n 's/^session in //p' logs/*-tipcal-$name.log logs/*-tipcal-${name}r[0-9]*.log 2>/dev/null | tail -n 1)
    [ "$kind" = seed ] && DIR[$seat]=$d
    say "$name ended cleanly ($d)"
done
say "all steps done; the arm waits at the last action's rest"
