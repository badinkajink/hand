#!/bin/bash
# Starts one run of the calibration program in the tmux session tipcal, with the plan, the DAQ's address and the
# recorded seating of this cell; the printout also goes to logs/<date>-<time>-tipcal-NAME.log. Refuses while the
# program runs. The pane stays open after the program exits, so its printout can be captured.
#
#   bash docs/experiments/20261022-ur5_tip_calibration/lab/tipcal_run.sh NAME [program options ...]
#   e.g. tipcal_run.sh first --stages preflight,homing,gravity,mount
set -u
cd "$(dirname "$0")/../../../.." || exit 1
NAME=$1
shift
if pgrep -af "^/usr/bin/python3 -u src/morphohand/driver/reskin5x/host/ur5_tip_calibration\.py" | grep -qv -- "--sim"; then
    echo "the calibration program is running: not starting $NAME"
    exit 1
fi
PY=/usr/bin/python3
P=src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py
PLAN=${PLAN:-docs/experiments/20261022-ur5_tip_calibration/lab/20261022-plan_sphere_lab_wrist_twist.json}   # the plan of 20:23 (wrist, port, twisted touches)
SEAT=docs/experiments/20261022-ur5_tip_calibration/lab/poses.jsonl:left_indenter_seated
for f in "$PLAN" "${SEAT%:*}"; do                 # a pull or rebase in a clone that skips LFS leaves pointers
    if head -c 40 "$f" | grep -q "git-lfs"; then
        echo "$f is a git-lfs pointer: git -c lfs.fetchexclude= lfs pull --include=$f"
        exit 1
    fi
done
LOG=logs/$(date +%Y%m%d-%H%M%S)-tipcal-$NAME.log
CMD="$PY -u $P --plan $PLAN --ft 192.168.0.6 --home-from $SEAT --name $NAME $* 2>&1 | tee $LOG"
CMD="$CMD; echo \"exit \${PIPESTATUS[0]} at \$(date +%H:%M:%S)\" | tee -a $LOG; sleep infinity"
tmux kill-session -t tipcal 2>/dev/null
tmux new-session -d -s tipcal -x 200 -y 60 "bash -c '$CMD'"
echo "started $NAME in tmux session tipcal; printout in $LOG"
