"""Lower travels for the table's probe plans (owner, 11-02 14:4x: "you don't need to go this high at all... we should avoid
these super large motions in general"): the back-off before a joint move (travel.probe.safe_mm) 15 mm instead of 45, and
the hub with the sphere's centre 40 mm over the apex instead of 90, both replayed clean by the workstation on the strip's
two-key frame with --real --chain (15:3x, data/20261102-strip_table_2key_run_check_chain_s15_h40.json). The home path's
last waypoint becomes the new hub's joints. Seat numbers 1 to 3.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/probe_low_travel.py PLAN SEAT [OUT]
"""
import json
import math
import sys

SAFE_MM = 15.0
HUBS = {   # flange T_BE (mm) and joints (rad) with the sphere's centre 40 mm over the apex, the probe straight down
    1: dict(T_BE=[[-0.74703, 0.664718, 0.009774, -285.084826], [0.664782, 0.747012, 0.006165, -484.666213],
                  [-0.003203, 0.011103, -0.999933, 270.157238], [0, 0, 0, 1]],
            q=[0.843259, -1.344614, 1.736551, -1.951629, -1.567597, -0.000366]),
    2: dict(T_BE=[[-0.821753, 0.569768, 0.009258, -205.151793], [0.56983, 0.821735, 0.006678, -487.685224],
                  [-0.003802, 0.010763, -0.999935, 270.572525], [0, 0, 0, 1]],
            q=[0.964206, -1.420236, 1.827765, -1.96756, -1.566997, -0.000327]),
    3: dict(T_BE=[[-0.892637, 0.450694, 0.008595, -125.222929], [0.450754, 0.892619, 0.007185, -490.702615],
                  [-0.004433, 0.010288, -0.999937, 270.987794], [0, 0, 0, 1]],
            q=[1.102967, -1.472835, 1.887038, -1.974709, -1.566365, -0.00026]),
}


def main():
    src, seat = sys.argv[1], int(sys.argv[2])
    dst = sys.argv[3] if len(sys.argv) > 3 else src
    plan = json.load(open(src))
    pr = plan["travel"]["probe"]
    old = (pr.get("safe_mm"), pr["hub"]["T_BE"][2][3])
    pr["safe_mm"] = SAFE_MM
    pr["hub"] = dict(HUBS[seat])
    if pr.get("home_path_q"):
        pr["home_path_q"][-1] = [round(math.degrees(v), 3) for v in HUBS[seat]["q"]]
    plan["notes"].insert(1, f"2026-11-02 15:4x, the lab: lower travels (probe_low_travel.py): safe_mm {SAFE_MM:g} (was {old[0]:g}) "
                            f"and the hub 40 mm over the apex (flange at z_B {HUBS[seat]['T_BE'][2][3]:.0f} mm, was "
                            f"{old[1]:.0f}), as the workstation replayed them clean on the two-key frame (--real --chain).")
    open(dst, "w").write(json.dumps(plan, indent=1))
    print(f"{dst}: seat {seat}, safe_mm {old[0]:g} -> {SAFE_MM:g}, hub flange z {old[1]:.0f} -> {HUBS[seat]['T_BE'][2][3]:.0f} mm")


if __name__ == "__main__":
    main()
