#!/bin/bash
# The hard stop of 11-04's rod points (the owner, relayed at 10:44: "we should have a few hours left"; the SR2 hand goes on after).
# Usage: bash logs/20261104-deadline_stop.sh HH:MM [CHAIN_PID]
# At HH:MM: stop every chain then running (each "bash .../20261104-chain_rod_points.sh ...", by its PID), so none
# starts or resumes a run, then send the robot program its Ctrl-C (SIGINT to its PID) right after its printout's next
# line of a finished action ("(k/n): ..." or a touch's "onset"), when the tool is back at that action's rest; after 10 min
# without such a line, send it anyway. The chains are found at HH:MM, not at the start: the chain restarted at 18:01 on
# 11-03 under a new PID while this stop held the first chain's, and a chain waiting out a HEX-E loss at HH:MM would have
# resumed the run once the sensor came back. CHAIN_PID (the chain's guard passes its own) is not used. Each step goes to
# logs/20261104-deadline_stop.log.
cd /home/will/workspace/hand || exit 1
OUT=logs/20261104-deadline_stop.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a $OUT; }
DL=${1:?HH:MM}
target=$(date -d "$DL" +%s)
[ "$target" -le "$(date +%s)" ] && target=$(date -d "tomorrow $DL" +%s)
say "armed (pid $$): at $(date -d @$target '+%m-%d %H:%M'), stop the chains then running, then the program after its next finished action"
sleep $((target - $(date +%s)))
for c in $(ps -eo pid,args | awk '$2 ~ /bash$/ && $3 ~ /chain_rod_points\.sh$/ {print $1}'); do
    kill "$c" 2>/dev/null && say "chain $c stopped"
done
sleep 15                            # a program that a stopped chain was starting is running by now
P=$(ps -eo pid,args | awk '$2 == "/usr/bin/python3" && /ur5_tip_calibration\.py --plan/ && !/--sim/ {print $1}' | head -n 1)
if [ -z "$P" ]; then
    say "no robot program runs; done"
    exit 0
fi
F=$(ls -t logs/*-tipcal-*.log | head -n 1)
PAT='\([0-9]+/[0-9]+\): |: onset '
n=$(grep -cE "$PAT" "$F")
t0=$(date +%s)
until [ "$(grep -cE "$PAT" "$F")" -gt "$n" ] || [ $(($(date +%s) - t0)) -gt 600 ]; do sleep 0.2; done
kill -INT "$P" && say "SIGINT to the program $P ($F), after: $(grep -E "$PAT" "$F" | tail -n 1 | cut -c1-100)"
sleep 10
ps -o pid= -p "$P" > /dev/null && say "the program $P still runs 10 s later" || say "the program ended: $(tail -n 1 "$F")"
