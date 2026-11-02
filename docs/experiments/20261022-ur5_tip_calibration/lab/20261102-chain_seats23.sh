#!/bin/bash
# The owner's go of 15:0x on 11-02 ("you have my go ahead to run the remaining runs", and more runs if the data shows more
# data pays). After seat 1's run (s1v6b) ends cleanly: seat 2 (v3 ID 5), then seat 3 (v3 ID 4), each from where the one
# before ends (over its seat, joint 5 in the work window: the program goes by the next seat's hub). Then the overnight
# runs, each on its seat's own session (--resume, stages preflight,grid), in an order whose seams join one seat to the
# next: the list in logs/20261102-overnight.txt (lines "NAME SEAT PLAN"). An overnight run starts only if its marker
# logs/20261102-ok-NAME exists (the lab writes it once the run's plan passed the simulator and the workstation's replay);
# without it the list stops there, so that no seam crosses two seats.
# A run is clean when it exits 0 with its last grid action printed, no "three aborts" and no traceback. A run stopped by
# a HEX-E link loss (three aborts, each on the HEX-E's stream) is resumed on its session once the HEX-E streams again
# (lab/hexe_probe.py every 60 s, up to 3 h), at most 3 times a run. Anything else stops the chain, and the arm waits where
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

n_grid() {
    /usr/bin/python3 -c "import json; print(sum(a['stage'] == 'grid' for a in json.load(open('$1'))['actions']))"
}

start() {                           # $1 name, $2 plan, $3... more program options; prints the printout's path
    local name=$1 plan=$2
    shift 2
    PLAN=$plan SEAT=$L/poses.jsonl:strip_key1_seated_4 bash $L/tipcal_run.sh "$name" --held-board $HB \
        --frame $L/20261102-strip_frame_two_keys.json "$@" >> $OUT 2>&1
    sleep 5
    ls -t logs/*-tipcal-"$name".log | head -1
}

finish() {                          # $1 printout, $2 name, $3 plan, $4 stages: resume after HEX-E losses; 0 when clean
    local log=$1 name=$2 plan=$3 stages=$4 n k=0 dir
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
        log=$(start "${name}r$k" "$plan" --resume "$dir" --stages "$stages")
        say "resumed $name as ${name}r$k, printing to $log"
        wait_exit "$log"
    done
    return 0
}

declare -A SESSION                  # seat -> its session folder
SESSION[1]=logs/20261102-tip_calibration/145010-s1v6b
say "waiting for logs/20261102-145009-tipcal-s1v6b.log to end"
finish logs/20261102-145009-tipcal-s1v6b.log s1v6b $L/20261102-plan_probe_table_seat1_v3id6.json \
    preflight,homing,pad,register,grid || exit 1
for spec in "2 s2v5 $L/20261102-plan_probe_table_seat2_v3id5.json" "3 s3v4 $L/20261102-plan_probe_table_seat3_v3id4.json"; do
    set -- $spec
    say "starting $2 ($3, $(n_grid $3) grid actions)"
    log=$(start "$2" "$3" --stages preflight,homing,pad,register,grid)
    say "$2 printing to $log"
    finish "$log" "$2" "$3" preflight,homing,pad,register,grid || exit 1
    SESSION[$1]=$(sed -n 's/^session in //p' logs/*-tipcal-"$2"*.log | tail -1)
    say "$2 ended cleanly; seat $1's session is ${SESSION[$1]}"
done
say "seats 1 to 3 done; the overnight runs (logs/20261102-overnight.txt, read again before each run)"
declare -A TRIED
while true; do
    next=""
    while read -r name seat plan; do  # the first entry not yet tried, from the list as it stands now
        [ -z "$name" ] || [ "${name:0:1}" = "#" ] || [ -n "${TRIED[$name]}" ] && continue
        next="$name $seat $plan"
        break
    done < logs/20261102-overnight.txt
    [ -z "$next" ] && break
    set -- $next
    TRIED[$1]=1
    if [ ! -f "logs/20261102-ok-$1" ]; then            # stop there: a skipped run would leave a seam across two seats
        say "stopping before $1: no logs/20261102-ok-$1"
        break
    fi
    say "starting $1 on seat $2's session ${SESSION[$2]} ($3, $(n_grid $3) actions)"
    log=$(start "$1" "$3" --resume "${SESSION[$2]}" --stages preflight,grid)
    say "$1 printing to $log"
    finish "$log" "$1" "$3" preflight,grid || exit 1
    say "$1 ended cleanly"
done
say "the chain is done"
