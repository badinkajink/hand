#!/bin/bash
# home: home the manta gantry (fast if the step counts are trusted, else full).
# See ~/manta-hand/sysid/home.py. The timeout is a hard wall-clock backstop;
# on any failure every axis is disabled.
timeout -s TERM 330 python3 -u "$HOME/manta-hand/sysid/home.py" "$@"
rc=$?
if [ $rc -ne 0 ] && [ "$1" != "--check" ]; then
    python3 -c "
from manta_hand import MantaHandDriver
d = MantaHandDriver('/dev/ttyACM0')
for j in range(6): d.joints[j].disable()
d.close()" && echo "home failed (exit $rc) -- all axes disabled" >&2
fi
exit $rc
