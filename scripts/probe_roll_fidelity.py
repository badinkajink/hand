#!/usr/bin/env python3
"""Does a sphere-packed cylinder still ROLL like a cylinder?

    uv run python scripts/probe_roll_fidelity.py --eps 0.05 0.02 0.01 0.005

WHY THIS IS THE GATING TEST FOR ARM B. S0 showed that packing the object -- not the hand -- is
what produces a contact map, because contact resolution is set by the flatter of the two
surfaces. But a sphere-packed cylinder is a SCALLOPED cylinder, and the reorient this whole
program is chasing is a ROLL. If scalloping meaningfully changes rolling, arm B buys legibility
by breaking the exact behaviour we want to study, and the branch is dead.

Rolling is also the known-fragile axis here: the sim2real contact-hardening work found grasp
transfers across stiffness changes but reorient-by-rolling does not (align 13 vs 48 under
harder solimp). So "the grasp still works" is NOT evidence that the roll survived.

THREE MEASUREMENTS, because scalloping shows up differently in each:

  bob      - a smooth cylinder rolling on a plane holds its centre height constant. A scalloped
             one rides up over each sphere and drops between them. Peak-to-peak centre height
             IS the scallop depth, made physical. This is the most direct read.
  distance - rolling resistance. A scalloped cylinder loses energy to the micro-collisions at
             every facet crossing, so it stops sooner from the same launch velocity.
  spin     - total angular displacement. Distance and spin should stay locked by the rolling
             constraint (distance ~= radius * angle); if they decouple, the packed cylinder is
             SLIPPING rather than rolling, which is a qualitatively different failure than
             merely rolling less far.

The probe runs on the bare object -- no hand, no policy -- so nothing else can explain a
difference.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from morphohand.geometry.sphere_pack import pack_cylinder

ROOT = Path(__file__).resolve().parents[1]

# A bare rolling rig: floor + one screwdriver-sized cylinder, axis along +Y so it rolls in X.
# Friction/solref/solimp copied from the perp scene so the contact model matches the real one.
RIG = """<mujoco model="roll_probe">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default><geom friction="1.8 0.15 0.01" solref="0.006 1" solimp="0.97 0.995 0.0005"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="obj" pos="0 0 {z0}" quat="0.70711 0.70711 0 0">
      <freejoint/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {iyy} {izz}"/>
{geoms}
    </body>
  </worldbody>
</mujoco>
"""


def _cyl_inertia(radius: float, half_len: float, density: float) -> tuple[float, ...]:
    mass = density * np.pi * radius**2 * (2 * half_len)
    # principal axes of a solid cylinder, local +Z = axis
    it = mass * (3 * radius**2 + (2 * half_len) ** 2) / 12.0
    ia = mass * radius**2 / 2.0
    return mass, it, it, ia


def _build(radius: float, half_len: float, density: float, eps: float | None) -> str:
    mass, ixx, iyy, izz = _cyl_inertia(radius, half_len, density)
    if eps is None:
        geoms = f'      <geom type="cylinder" size="{radius} {half_len}" friction="2.4 0.2 0.02"/>'
    else:
        spheres = pack_cylinder(radius, half_len, eps)
        geoms = "\n".join(
            f'      <geom type="sphere" size="{s.radius:.9g}" '
            f'pos="{s.pos[0]:.9g} {s.pos[1]:.9g} {s.pos[2]:.9g}" friction="2.4 0.2 0.02"/>'
            for s in spheres
        )
    return RIG.format(z0=radius, mass=mass, ixx=ixx, iyy=iyy, izz=izz, geoms=geoms)


def roll(xml: str, v0: float, steps: int, radius: float) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
        fh.write(xml)
        path = fh.name
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)

    # settle onto the floor before launching, so the initial drop is not counted as bob
    for _ in range(300):
        mujoco.mj_step(m, d)
    d.qvel[:] = 0.0
    d.qvel[0] = v0  # linear +X, world frame

    # A freejoint's qvel[3:6] is angular velocity in the BODY-LOCAL frame, and this body is
    # rotated 90 deg about X so its local +Z (the cylinder axis) points along world -Y.
    # Rolling in +X needs omega = v/r about world +Y, which is local -Z => qvel[5].
    # Getting this wrong makes the cylinder SKID, and a skid measures friction, not rolling.
    d.qvel[5] = -v0 / radius
    x0 = float(d.qpos[0])

    zs, ncons = [], []
    ang = 0.0
    for _ in range(steps):
        mujoco.mj_step(m, d)
        zs.append(float(d.qpos[2]))
        ncons.append(int(d.ncon))
        ang += abs(float(d.qvel[5])) * m.opt.timestep

    z = np.array(zs)
    dist = float(d.qpos[0] - x0)
    return {
        "distance_m": dist,
        "spin_rad": ang,
        "roll_ratio": dist / (radius * ang) if ang > 1e-9 else 0.0,
        "bob_pp_mm": float(z.max() - z.min()) * 1000.0,
        "bob_std_mm": float(z.std()) * 1000.0,
        "mean_ncon": float(np.mean(ncons)),
        "max_ncon": int(np.max(ncons)),
        "final_speed": float(abs(d.qvel[0])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=float, default=0.0125)
    ap.add_argument("--half-length", type=float, default=0.05)
    ap.add_argument("--density", type=float, default=500.0)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.02, 0.01, 0.005])
    ap.add_argument("--v0", type=float, default=0.5, help="launch speed, m/s")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    rows = [("smooth cylinder", None)] + [(f"packed eps={e}", e) for e in args.eps]
    print(f"{'object':<20}{'dist(m)':>9}{'roll_ratio':>12}{'bob_pp(mm)':>12}"
          f"{'ncon':>7}{'v_end':>8}")
    out = {}
    base = None
    for label, eps in rows:
        r = roll(_build(args.radius, args.half_length, args.density, eps), args.v0, args.steps, args.radius)
        out[label] = r
        if base is None:
            base = r
        print(f"{label:<20}{r['distance_m']:>9.4f}{r['roll_ratio']:>12.3f}"
              f"{r['bob_pp_mm']:>12.4f}{r['mean_ncon']:>7.1f}{r['final_speed']:>8.3f}"
              f"   ({r['distance_m'] / base['distance_m']:.2f}x)")

    print("\nroll_ratio ~1.0 = rolling without slip; <1 = slipping. "
          "bob_pp is the scallop made physical.")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
