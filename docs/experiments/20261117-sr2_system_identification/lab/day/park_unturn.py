"""Park after the turned-hand work of 2026-11-19: straight up to the park height, the turn about z_P through the palm
origin back to the orientation of the morning's park, then level over the morning's park position. Each leg is guarded
by the HEX-E (hom_setup.Arm.move; a turn adds 0.9 N/deg to the limit, capped at 12 N). Without --go it prints the legs.

    /usr/bin/python3 day/park_unturn.py [--start PLAN] [--go]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
START = Path("/home/will/workspace/hand/logs/20261119-sr2_s3_press/20261119-121731_approach_index_yaw_pad.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", default=str(START), help="a plan whose T_BP0 is the park to return to")
    ap.add_argument("--stop-n", type=float, default=5.0)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    sys.argv = sys.argv[:1]
    import s3_approach as SA
    import s3_press as S3
    FT, HS, SL = S3.imports()
    import s1b_tags as S1
    start = np.array(json.loads(Path(a.start).read_text())["T_BP0"], float)
    T, mode = S1.arm_T_BP(FT, HS)
    if mode != 7:
        raise SystemExit(f"the arm is not running (robot mode {mode})")
    R = T[:3, :3].T @ start[:3, :3]                    # the start's orientation in the present palm frame
    deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    tilt = math.degrees(math.acos(float(np.clip(T[:3, 2] @ start[:3, 2], -1, 1))))
    print(f"palm origin B {np.round(T[:3, 3], 1).tolist()}; park {np.round(start[:3, 3], 1).tolist()}; turn {deg:+.1f} deg "
          f"about z_P (the z axes differ by {tilt:.2f} deg)")
    if tilt > 0.5:
        raise SystemExit("the hand's z_P is not the park's: this script only turns about z_P")
    up = max(0.0, start[2, 3] - T[2, 3])
    T_up = SA.shifted(T, [0.0, 0.0, up])
    T_turn = SA.turned(T_up, deg)
    d = start[:3, 3] - T_turn[:3, 3]
    d[2] = 0.0
    T_lev = SA.shifted(T_turn, d)
    print(f"legs: up {up:.1f} mm; turn {deg:+.1f} deg; level {np.round(d[:2], 1).tolist()} mm")
    if not a.go:
        return 0
    arm = HS.Arm(False, 0.4)
    try:
        arm.connect(True)
        for name, Tg, v, max_s in (("up", T_up, 0.012, up / 12.0 + 6.0),
                                   ("turn", T_turn, 0.015, 6.0 + 0.6 * abs(deg)),
                                   ("level", T_lev, 0.015, float(np.linalg.norm(d)) / 15.0 + 6.0)):
            if name == "up" and up < 0.5 or name == "turn" and abs(deg) < 0.2 or name == "level" and np.linalg.norm(d) < 0.5:
                continue
            why, moved = arm.move(Tg @ np.linalg.inv(HS.T_EP), v, 0.05, a.stop_n, max_s, f"park: {name}")
            if why:
                raise SystemExit(f"park: {name} stopped: {why}; the arm holds there")
        print(f"parked: palm origin B {np.round(arm.T_BP()[:3, 3], 1).tolist()}", flush=True)
    finally:
        arm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
