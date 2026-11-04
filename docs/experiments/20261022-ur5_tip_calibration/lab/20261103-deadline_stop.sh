#!/bin/bash
# The night's hard stop (the owner, 16:3x on 11-03: the SR2 hand goes on early tomorrow morning; the robot must be free).
# Usage: bash logs/20261103-deadline_stop.sh HH:MM CHAIN_PID
# At HH:MM: stop the chain by its PID (it starts nothing more), then send the robot program its Ctrl-C (SIGINT to its PID)
# right after its printout's next line of a finished action ("(k/n): ..." or a touch's "onset"), when the tool is back at
# that action's rest; after 10 min without such a line, send it anyway. Each step goes to logs/20261103-deadline_stop.log.
cd /home/will/workspace/hand || exit 1
OUT=logs/20261103-deadline_stop.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a $OUT; }
DL=${1:?HH:MM}
CHAIN=${2:?the chain PID}
target=$(date -d "$DL" +%s)
[ "$target" -le "$(date +%s)" ] && target=$(date -d "tomorrow $DL" +%s)
say "armed: at $(date -d @$target '+%m-%d %H:%M'), stop chain $CHAIN and the program after its next finished action"
sleep $((target - $(date +%s)))
kill "$CHAIN" 2>/dev/null && say "chain $CHAIN stopped"
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
