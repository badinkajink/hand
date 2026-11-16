"""The corvus session's per-joint free-air sag protocol (lab/joint_sag.py, data/20261116-sweep/20261116_joint_sag.json) in
the simulator's servo plant (hand-4a, 2026-11-16, for the corvus session: "run the same steps in the sim's servo plant
(server plant and the work plant) with the scene's link masses").

One joint at a time from the straight hand, 2 s per step, at the (105, 70) layout, the object's collisions and the post
off: the thumb's yaw +10..+40 and -10..-40, the thumb's MCP 15/30/45/55, the index's yaw -10..-40 and +10..+40 (the
middle's yaw at +30 while the index turns toward it, +), the index's MCP, the middle's yaw -10..-40 (the index's yaw at
-30) and +10..+40, the middle's MCP. Shortfall = |command| - |measured| at the step's end, servo degrees, as the lab
reports the hand. Plants: the server's (kp 0.55, damping 0.011, tau_max 0.18) and the work plant (kp 4, damping 0.08,
tau_max 1.0).

    OMP_NUM_THREADS=1 .venv/bin/python rev_joint_sag.py [--json joint_sag_sim.json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D))
import rev_rescore as RR  # noqa: E402

PLANTS = {"server": {"kp": 0.55, "damping": 0.011, "tau_max": 0.18},
          "work": {"kp": 4.0, "damping": 0.08, "tau_max": 1.0}}
YAW = (10.0, 20.0, 30.0, 40.0)
MCP = (15.0, 30.0, 45.0, 55.0)
# (finger, joint, signed steps, neighbour moved clear first as {finger: {joint: deg}})
SEQS = [("thumb", "aa", YAW, {}), ("thumb", "aa", tuple(-x for x in YAW), {}), ("thumb", "fe1", MCP, {}),
        ("index", "aa", tuple(-x for x in YAW), {}), ("index", "aa", YAW, {"middle": {"aa": 30.0}}),
        ("index", "fe1", MCP, {}),
        ("middle", "aa", tuple(-x for x in YAW), {"index": {"aa": -30.0}}), ("middle", "aa", YAW, {}),
        ("middle", "fe1", MCP, {})]
HAND = {("thumb", "aa", 1): (0.6, 1.8, 1.6, 2.2), ("thumb", "aa", -1): (0.3, 1.0, 1.3, 1.6),
        ("thumb", "fe1", 1): (0.6, 0.4, 1.1, 1.4), ("index", "aa", -1): (1.2, 1.2, 1.3, 1.3),
        ("index", "aa", 1): (1.5, 2.1, 3.0, 4.0), ("index", "fe1", 1): (0.4, 0.7, 0.5, 0.8),
        ("middle", "aa", -1): (1.2, 1.8, 2.8, 4.0), ("middle", "aa", 1): (1.5, 3.3, 3.6, 4.0),
        ("middle", "fe1", 1): (0.1, 1.3, 0.8, 1.7)}


def plant_for(params):
    import sr2_hom_replay as RP
    import mujoco
    import hand_object_scale_sim as S
    import hom_tasks as T
    setup, mxy, _ = RP.toy_setup("twist", 40.0, 329, 10, "grid2dff")
    p_pads, p_pt = RP.ref_scene("sphere", 40.0)
    m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
    RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
    q_open, _ = T.open_pose(setup, m_pt)
    S.set_palm(m_pt, setup["palm"])
    p = RP.SR2Plant(p_pads, setup, mxy, {**params, "mu": 1.0}, 0.0)
    p.reset(q_open, 0)
    m = p.m
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == p.tb or g == p.post:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
    return p


def run_seq(params, finger, joint, steps, clear):
    from sr2hand.control import hom_bench as HB
    p = plant_for(params)
    servo = {f: {"aa": 0.0, "fe1": 0.0, "fe2": 0.0} for f in ("thumb", "index", "middle")}
    for f, js in clear.items():
        servo[f].update(js)
    t = 2.0
    p.set_targets(HB.sim_from_servo(servo))
    p.advance(t)
    out = []
    for x in steps:
        servo[finger][joint] = x
        p.set_targets(HB.sim_from_servo(servo))
        t += 2.0
        p.advance(t)
        meas = float(HB.servo_from_sim(p.state()["q"])[finger][joint])
        out.append(round(abs(x) - abs(meas), 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(D / "joint_sag_sim.json"))
    a = ap.parse_args()
    res = {}
    for name, prm in PLANTS.items():
        params = {**RR.SERVER, **prm}
        for f, j, steps, clear in SEQS:
            sign = 1 if steps[0] > 0 else -1
            sim = run_seq(params, f, j, steps, clear)
            hand = HAND.get((f, j, sign))
            res.setdefault(name, []).append({"finger": f, "joint": j, "steps": list(steps), "sim": sim,
                                             "hand": list(hand) if hand else None})
            print(f"{name:6s} {f:6s} {j:3s} {'+' if sign > 0 else '-'}: sim " + "/".join(f"{v:.2f}" for v in sim)
                  + (" | hand " + "/".join(f"{v:.1f}" for v in hand) if hand else ""), flush=True)
    Path(a.json).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
