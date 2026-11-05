"""Read-only: the nine servos' positions (deg from each calibrated zero), torque state, temperature, load and alarms,
on the CB1's own manta_hand. Writes no register."""
import time
from manta_hand.servos import FINGER_JOINTS, ServoBus

NAMES = {0: "thumb", 1: "index", 2: "middle"}
bus = ServoBus("/dev/ttyUSB0")
t0 = time.time()
pos = bus.sync_read_joint_positions()
torque = bus.read_torque_all()
temp = bus.read_field("temperature")
load = bus.read_field("load")
alarms = bus.read_alarms()
print(f"read in {time.time() - t0:.2f} s")
for f, joints in FINGER_JOINTS.items():
    for j, (sid, zero, rng) in joints.items():
        print(f"{NAMES[f]:6s} {j:3s} id {sid}: {pos[f][j]:8.2f} deg from zero (range {rng[0]:7.2f} to {rng[1]:6.2f})  "
              f"torque {torque.get(sid)}  T {temp.get(sid) if temp else None}  load {load.get(sid) if load else None}  "
              f"alarm {alarms.get(sid) if alarms else None}")
