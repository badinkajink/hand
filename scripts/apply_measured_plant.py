#!/usr/bin/env python3
"""Rewrite a real_v1 scene so its actuators and masses are the ones the bench measured.

    python3 scripts/apply_measured_plant.py --scene <in.xml> --out <out.xml> --kp 16

WHAT IT CHANGES, AND WHERE EACH NUMBER COMES FROM

  kp            Every real_v1 scene ships `<position kp="4000">`, a near-rigid position source.
                The SCS0009 is a P controller with `p_coefficient 15` and, decisively,
                `i_coefficient 0` -- no integral, so a standing load torque leaves a
                proportional error forever.  That is the same law MuJoCo's position actuator
                implements, so the fix is the gain, not the model.  The value is CALIBRATED, not
                transcribed: `calibrate_plant_kp.py` sweeps it until the simulated `ctrl - qpos`
                at the hold reproduces the deficits measured on rv05_manual_b85.

  forcerange    Ships as +-1000, i.e. no ceiling.  The servo drops its output to
                `protective_torque` (20 %) after sustained overload, which on the bench pins the
                reported load at exactly 200 and roughly doubles both deflection and scatter.

  frictionloss  Absent from every MJCF in the repo.  The free-hanging staircase measures a
                friction cone of 0.70-1.50 deg on five joints.  NOTE this is known to be an
                UNDER-estimate: under a real grasp the same joint's cone is 12.26 deg, and a
                constant `frictionloss` cannot express both.  See
                docs/experiments/20260902-servo-sysid/.

  masses        The shipped 12.93 / 12.93 / 14.22 g are numerically the capsule volumes in cm^3
                -- MuJoCo's default 1000 kg/m^3 applied to a collision shape, never a model of
                anything.  Replaced with servo (13.2 g, biased low) + PLA shell at 30 % infill +
                the 10 g of boards on the yaw link, per the builder.

The scene is rewritten rather than the base template edited, because the base propagates to
57k generated scenes and every earlier result was produced against it.  A corrected scene is a
new artifact that can be compared with the old one, not a retroactive change to the record.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

FINGERS = ("thumb", "index", "middle")
SERVO_G = 13.2          # measured, per the builder
BOARDS_G = 10.0         # mounted on the yaw link only
PLA_DENSITY = 560.0     # kg/m^3: solid PLA 1240 discounted for 30% infill + perimeters/skins
SERVO_COM_FRAC = 0.80   # servo sits low in its link; 0 = joint, 1 = link tip


def link_geometry(model_path: Path) -> dict[str, tuple[float, float]]:
    """body name -> (capsule volume m^3, link length m), read from the scene itself."""
    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(str(model_path))
    out = {}
    for gi in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gi]) or ""
        if not bn.endswith(("_yaw_frame", "_mcp_frame", "_pip_frame")):
            continue
        if m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        r, hl = float(m.geom_size[gi][0]), float(m.geom_size[gi][1])
        out[bn] = (np.pi * r ** 2 * (2 * hl) + 4 / 3 * np.pi * r ** 3, 2 * (hl + r))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kp", type=float, default=16.0)
    ap.add_argument("--kv", type=float, default=0.6)
    ap.add_argument("--forcerange", type=float, default=0.35,
                    help="N*m ceiling per finger actuator; the protective_torque cliff")
    ap.add_argument("--frictionloss", type=float, default=0.0035,
                    help="N*m. 0 disables. The free-hanging cone; an under-estimate under load")
    ap.add_argument("--no-mass", action="store_true", help="leave the shipped masses alone")
    a = ap.parse_args()

    geom = link_geometry(a.scene)
    tree = ET.parse(a.scene)
    root = tree.getroot()

    changes = []
    for d in root.iter("default"):
        if d.get("class") != "ctrl":
            continue
        for pos in d.findall("position"):
            changes.append(f"ctrl actuator: kp {pos.get('kp')} -> {a.kp:g}, "
                           f"forcerange {pos.get('forcerange')} -> +-{a.forcerange:g}")
            pos.set("kp", f"{a.kp:g}")
            pos.set("kv", f"{a.kv:g}")
            pos.set("forcerange", f"-{a.forcerange:g} {a.forcerange:g}")
        if a.frictionloss > 0:
            for jt in d.findall("joint"):
                jt.set("frictionloss", f"{a.frictionloss:g}")
                changes.append(f"ctrl joint: frictionloss -> {a.frictionloss:g}")

    if not a.no_mass:
        for body in root.iter("body"):
            name = body.get("name") or ""
            if name not in geom:
                continue
            vol, length = geom[name]
            pla = vol * PLA_DENSITY
            boards = BOARDS_G / 1000.0 if name.endswith("_yaw_frame") else 0.0
            servo = SERVO_G / 1000.0
            mass = pla + servo + boards
            # COM: shell uniform over the link, servo low, boards at the servo
            com = -(pla * (length / 2) + (servo + boards) * (SERVO_COM_FRAC * length)) / mass
            inertia = mass * (length ** 2) / 12.0
            for old in body.findall("inertial"):
                body.remove(old)
            ET.SubElement(body, "inertial", {
                "pos": f"0 0 {com:.6f}", "mass": f"{mass:.6f}",
                "diaginertia": f"{inertia:.9f} {inertia:.9f} {inertia * 0.35:.9f}"})
            changes.append(f"{name}: mass -> {mass*1000:.2f} g, com z -> {com*1000:.1f} mm")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(a.out, encoding="unicode")
    print(f"wrote {a.out}")
    for c in changes:
        print("  " + c)

    import mujoco
    m = mujoco.MjModel.from_xml_path(str(a.out))
    print(f"\nloads: nq={m.nq} nu={m.nu} nbody={m.nbody}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
