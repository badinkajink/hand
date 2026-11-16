"""Every servo's goal speed set slow and its goal set to its present position (torque stays off), so that the station's
torque-on at start holds the pose. Run on the CB1 with the station stopped."""
import math

from manta_hand.servos import FINGER_JOINTS, ServoBus

bus = ServoBus("/dev/ttyUSB0")
for fid, joints in FINGER_JOINTS.items():
    for name, (sid, zero, lim) in joints.items():
        sv = bus.servo(sid)
        sv._call(bus._c.write_goal_speed, 40)
        p = float(sv._call(bus._c.read_present_position)[0])
        g0 = float(sv._call(bus._c.read_goal_position)[0])
        try:
            sv._call_verified(bus._c.write_goal_position, bus._c.read_goal_position, p)
            note = ""
        except RuntimeError as e:
            note = f"  (not taken: {e})"
        g1 = float(sv._call(bus._c.read_goal_position)[0])
        print(f"servo {sid} finger {fid} {name}: present {math.degrees(p):+7.2f}, goal {math.degrees(g0):+7.2f} -> "
              f"{math.degrees(g1):+7.2f} deg, would move {math.degrees(g1 - p):+.2f}{note}", flush=True)
