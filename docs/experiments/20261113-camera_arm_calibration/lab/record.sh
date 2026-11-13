#!/usr/bin/env bash
# Record one pose with preview.py running: pause the preview (it frees the cameras), record, resume.
#   bash record.sh "LABEL"
set -u
cd /home/will/workspace/hand || exit 1
O=logs/20261113-camera_arm_calibration
L=docs/experiments/20261113-camera_arm_calibration/lab
touch $O/PAUSE
timeout 2 tail -f /dev/null
timeout 90 /usr/bin/python3 $L/record_pose.py $O --label "$1" 2>&1 | grep -v "^$" | tail -6
rc=${PIPESTATUS[0]}
rm -f $O/PAUSE
d=$(ls -d $O/p[0-9]* | tail -1); [ -z "$(ls -A $d)" ] && rmdir $d && echo "empty $d removed"
exit $rc
