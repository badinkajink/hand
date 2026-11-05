#!/bin/bash
# Serial render queue: each job through resguard (--mem 3G, 2 cores); a refusal (rc 3, e.g. swap > 15 GB) is retried
# after the status check passes again (the check itself logs no event). One line per job outcome on stdout.
D=/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts
PY=/home/humanoid/Programs/hand/.venv/bin/python
cd /home/humanoid/Programs/hand
run() {
  local name=$1; shift
  while true; do
    # wait on the status check (logs nothing) so that a closed gate does not fill the event log with refusals
    until ~/.claude/bin/resguard.sh status > /dev/null 2>&1; do sleep 30; done
    ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- env MUJOCO_GL=egl PYTHONPATH=/home/humanoid/Programs/hand/src "$PY" "$@" > "$D/log_$name.txt" 2>&1
    rc=$?
    if [ $rc -eq 3 ]; then sleep 30; continue; fi
    echo "$(date +%H:%M:%S) $name rc=$rc $(tail -1 $D/log_$name.txt | cut -c1-160)"
    return
  done
}
run overview $D/r3_overview.py
run gantry $D/r4_gantry.py
run sensors $D/r5_sensors.py
run finger2 $D/r6_finger.py "Finger v3 <2>"
run finger3 $D/r6_finger.py "Finger v3 <3>"
echo "$(date +%H:%M:%S) queue done"
