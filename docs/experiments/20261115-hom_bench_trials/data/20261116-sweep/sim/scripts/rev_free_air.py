"""The corvus session's free-air sequence of the middle finger's yaw (2026-11-16 15:1x, at (105, 70) mm) in the simulator's
servo plant (hand-4a, for the corvus session: "run exactly this free-air sequence in the simulator's servo plant ... and
report the steady errors next to the hand's 0.9/2.1/2.8/3.7/4.3 deg").

The plant of sr2_hom_replay (the reference cell's pads scene with the layout's mounts and the scene's link masses), the
object's collisions off and the post removed, so the fingers move in free air. Targets in servo degrees: the thumb at 0,
the index's yaw at -30 with its mcp and pip at 0, the middle's mcp and pip at 0; the middle's yaw stepped through 0, -10,
-20, -30, -40 and -45, 2 s per step. Reported per step: the measured middle yaw at the step's end and its error (command
minus measured, in the sense the lab gives), and the index yaw's. Run for each servo gain given.

    OMP_NUM_THREADS=1 .venv/bin/python rev_free_air.py [--kp 0.55 0.25 0.2 0.15]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D))
import rev_rescore as RR  # noqa: E402

STEPS = (0.0, -10.0, -20.0, -30.0, -40.0, -45.0)
HAND = {-10.0: -9.1, -20.0: -17.9, -30.0: -27.2, -40.0: -36.3, -45.0: -40.7}       # the lab's measurement


def run(kp):
    import sr2_hom_replay as RP
    from sr2hand.control import hom_bench as HB
    import mujoco
    import hand_object_scale_sim as S
    import hom_tasks as T
    setup, mxy, _ = RP.toy_setup("twist", 40.0, 329, 10, "grid2dff")
    p_pads, p_pt = RP.ref_scene("sphere", 40.0)
    m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
    RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
    q_open, _ = T.open_pose(setup, m_pt)
    S.set_palm(m_pt, setup["palm"])
    plant = RP.SR2Plant(p_pads, setup, mxy, {**RR.SERVER, "kp": kp, "mu": 1.0}, 0.0)
    plant.reset(q_open, 0)
    m = plant.m
    for g in range(m.ngeom):                       # the object out of the way: no collisions, no post
        if m.geom_bodyid[g] == plant.tb or g == plant.post:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
    servo = {"thumb": {"aa": 0.0, "fe1": 0.0, "fe2": 0.0}, "index": {"aa": -30.0, "fe1": 0.0, "fe2": 0.0},
             "middle": {"aa": 0.0, "fe1": 0.0, "fe2": 0.0}}
    t, out = 0.0, []
    for x in STEPS:
        servo["middle"]["aa"] = x
        plant.set_targets(HB.sim_from_servo(servo))
        t += 2.0
        plant.advance(t)
        meas = HB.servo_from_sim(plant.state()["q"])
        out.append({"cmd": x, "middle_aa": round(float(meas["middle"]["aa"]), 2),
                    "err": round(float(x - meas["middle"]["aa"]), 2),
                    "index_aa": round(float(meas["index"]["aa"]), 2),
                    "middle_fe1": round(float(meas["middle"]["fe1"]), 2), "middle_fe2": round(float(meas["middle"]["fe2"]), 2)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", type=float, nargs="+", default=[0.55, 0.25, 0.2, 0.15])
    ap.add_argument("--json", default=str(D / "free_air_middle_yaw.json"))
    a = ap.parse_args()
    res = {}
    for kp in a.kp:
        rows = run(kp)
        res[str(kp)] = rows
        print(f"kp {kp}: " + "; ".join(f"cmd {r['cmd']:+.0f} -> {r['middle_aa']:+.1f} (err {r['err']:+.2f}"
                                       + (f", hand {r['cmd'] - HAND[r['cmd']]:+.1f})" if r['cmd'] in HAND else ")")
                                       for r in rows) + f"; index yaw {rows[-1]['index_aa']:+.1f}", flush=True)
    Path(a.json).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
