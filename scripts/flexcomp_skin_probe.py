#!/usr/bin/env python3
"""Compile probe: a MuJoCo flexcomp skin whose vertices play the role of the 1 mm pad spheres.

Planning check for the hardware-tip note in 20261004-contact_research_asides.html (not a study):
which flexcomp ``dof`` options the installed MuJoCo accepts, whether a small skin on the real_v1
fingertip sphere (R 10.55 mm) collides with a box when pressed at 1 N, and the wall time per step
of that tiny scene. Each vertex's normal (Winkler) spring is the stiffness of its own slider, set
on the compiled model from the hydroelastic field E/R (E 10 MPa, Drake's default) times the
vertex's share of the sphere area; damping is 2 ms of that stiffness so the explicit spring is
stable at the 1 ms step; the integrator is 'discrete' unless a case says implicitfast.

    logs/20261004-contact-current/venv/bin/python scripts/flexcomp_skin_probe.py
writes docs/experiments/20261004-codex/data/flexcomp_skin_probe.json, one row per case, rewritten
after every case.
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/20261004-codex/data/flexcomp_skin_probe.json"
R_TIP = 0.01055          # real_v1 fingertip sphere radius, m
E_HYDRO = 10e6           # Drake hydroelastic modulus default, Pa (not a measured TPU value)
PRESS = 1.0              # N pressing the tip onto the box
DT = 1e-3                # s, the task benchmark's physics step
# MuJoCo's contact stiffness scales with the contacting body's mass (K = 1/(tc^2 (1-d) diagApprox)),
# and a 5 mg vertex makes a 2 ms contact ~12 N/m: the first probe fell through the box. d = 0.9999
# gives ~10^4 N/m per vertex contact, stiffer than the vertex's own slider spring.
SOLIMP = "0.9999 0.9999 0.001"   # set on the box as well: solmix averages the two geoms otherwise
SETTLE, TIMED = 1000, 2000

SCENE = """
<mujoco>
  <option timestep="{dt}" gravity="0 0 0" integrator="{integrator}"/>
  <worldbody>
    <geom name="box" type="box" size="0.02 0.02 0.005" pos="0 0 0" solref="0.002 1" solimp="{solimp}"/>
    <body name="tip" pos="0 0 {z0}">
      <joint name="press" type="slide" axis="0 0 1" damping="2"/>
      <geom name="core" type="sphere" size="0.007" mass="0.01" contype="0" conaffinity="0"/>
      {mount_open}<flexcomp name="skin" type="{ftype}" dim="{dim}" count="{n} {n} {n}" spacing="{sp} {sp} {sp}"
                radius="0.0005" mass="0.001" dof="{dof}">
        {elasticity}
        <contact condim="3" solref="0.002 1" solimp="{solimp}" selfcollide="none" internal="false"/>
      </flexcomp>{mount_close}
    </body>
  </worldbody>
  <actuator><motor name="push" joint="press" gear="1"/></actuator>
</mujoco>
"""

CASES = [
    # (label, dof, type, dim, count, elasticity); integrator 'discrete' unless the label says otherwise
    ("full", "full", "ellipsoid", 2, 7, ""),
    ("radial", "radial", "ellipsoid", 2, 7, ""),
    ("radial, implicitfast", "radial", "ellipsoid", 2, 7, ""),
    ("radial, 11-grid", "radial", "ellipsoid", 2, 11, ""),
    ("radial + edge springs", "radial", "ellipsoid", 2, 7, '<edge stiffness="{ke}" damping="{be}"/>'),
    ("radial + shell stretch", "radial", "ellipsoid", 2, 7,
     '<elasticity young="3e6" poisson="0.3" thickness="0.001" elastic2d="stretch"/>'),
    ("radial + shell stretch, implicitfast", "radial", "ellipsoid", 2, 7,
     '<elasticity young="3e6" poisson="0.3" thickness="0.001" elastic2d="stretch"/>'),
    ("radial + shell bending", "radial", "ellipsoid", 2, 7,
     '<elasticity young="3e6" poisson="0.3" thickness="0.001" elastic2d="bend"/>'),
    ("radial + shell bending, jointless mount", "radial", "ellipsoid", 2, 7,
     '<elasticity young="3e6" poisson="0.3" thickness="0.001" elastic2d="bend"/>'),
    ("trilinear", "trilinear", "ellipsoid", 2, 7, ""),
    ("quadratic", "quadratic", "ellipsoid", 2, 7, ""),
    ("nonexistent keyword", "bogus", "ellipsoid", 2, 7, ""),
]


def run_case(label, dof, ftype, dim, n, elasticity):
    row = {"label": label, "dof": dof, "type": ftype, "dim": dim, "count": n,
           "elasticity": elasticity or None}
    # flexcomp sizes an ellipsoid or box by spacing * (count - 1) / 2 (scale is not applied):
    # a sphere of the tip radius, or a cube of half-size R_TIP, 0.1 mm above the box
    sp = R_TIP / ((n - 1) / 2)
    k_v = E_HYDRO / R_TIP * 4 * math.pi * R_TIP ** 2 / (6 * (n - 1) ** 2 + 2)  # ~ per-vertex share
    elasticity = elasticity.format(ke=round(k_v, 1), be=round(2e-3 * k_v, 3))
    row["elasticity"] = elasticity or None
    xml = SCENE.format(dt=DT, z0=0.005 + R_TIP + 1e-4, ftype=ftype, dim=dim, n=n, sp=sp,
                       dof=dof, elasticity=elasticity, solimp=SOLIMP,
                       integrator="implicitfast" if "implicitfast" in label else "discrete",
                       mount_open='<body name="mount">' if "mount" in label else "",
                       mount_close="</body>" if "mount" in label else "")
    row["integrator"] = "implicitfast" if "implicitfast" in label else "discrete"
    try:
        m = mujoco.MjModel.from_xml_string(xml)
    except Exception as exc:  # noqa: BLE001 - the compile error text is the result
        row.update(compiled=False, error=str(exc).strip().splitlines()[0][:300])
        return row
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")
    press = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "press")
    skin_joints = [j for j in range(m.njnt) if j != press]
    tip_body = m.jnt_bodyid[press]
    nvert = int(m.flex_vertnum[0])
    verts = d.flexvert_xpos[:nvert] - d.xpos[tip_body]
    extent = float(np.linalg.norm(verts, axis=1).max())
    # Winkler anchor: k = (E/R) * area per vertex on each vertex slider (full: each of x, y, z);
    # an interpolated skin gets the whole-sphere stiffness shared over its control bodies.
    k_total = E_HYDRO / R_TIP * 4 * math.pi * R_TIP ** 2
    n_anchor = nvert if dof in ("full", "radial") else len({int(m.jnt_bodyid[j]) for j in skin_joints})
    k_j = k_total / max(n_anchor, 1)
    m.jnt_stiffness[skin_joints] = k_j
    m.dof_damping[[m.jnt_dofadr[j] for j in skin_joints]] = 2e-3 * k_j
    row.update(compiled=True, mujoco=mujoco.__version__, nq=int(m.nq), nv=int(m.nv),
               nbody=int(m.nbody), nflexvert=nvert, nflexelem=int(m.flex_elemnum[0]),
               skin_joints=len(skin_joints),
               joint_types=sorted({int(m.jnt_type[j]) for j in skin_joints}),
               vertex_extent_mm=round(extent * 1e3, 3), k_joint_N_per_m=round(k_j, 1))
    d.ctrl[0] = -PRESS
    f6 = np.zeros(6)
    try:
        for _ in range(SETTLE):
            mujoco.mj_step(m, d)
        t0 = time.perf_counter()
        for _ in range(TIMED):
            mujoco.mj_step(m, d)
        us = (time.perf_counter() - t0) / TIMED * 1e6
    except Exception as exc:  # noqa: BLE001
        row.update(stepped=False, error=str(exc)[:300])
        return row
    n_box, f_box, fz = 0, 0.0, 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if box in (int(c.geom[0]), int(c.geom[1])) and 0 in (int(c.flex[0]), int(c.flex[1])):
            mujoco.mj_contactForce(m, d, i, f6)
            n_box += 1
            f_box += f6[0]
            fz += float(c.frame.reshape(3, 3).T[2] @ f6[:3])  # world z of the contact force
    defl = 0.0
    if dof == "radial":
        defl = float(-np.min(d.qpos[[m.jnt_qposadr[j] for j in skin_joints]]))
    row.update(stepped=True, finite=bool(np.isfinite(d.qpos).all()), us_per_step=round(us, 2),
               ncon_total=int(d.ncon), ncon_skin_box=n_box, normal_force_N=round(f_box, 4),
               vertical_force_N=round(abs(fz), 4),
               tip_speed_mm_s=round(abs(float(d.qvel[m.jnt_dofadr[press]])) * 1e3, 3),
               tip_drop_mm=round(-float(d.qpos[m.jnt_qposadr[press]]) * 1e3, 4),
               max_inward_slider_mm=round(defl * 1e3, 4) if dof == "radial" else None)
    return row


def main():
    rows = []
    meta = {"mujoco": mujoco.__version__, "python": platform.python_version(),
            "host": platform.node(), "timestep_s": DT, "press_N": PRESS, "settle_steps": SETTLE,
            "timed_steps": TIMED, "E_Pa": E_HYDRO, "R_tip_m": R_TIP, "contact_solref": "0.002 1",
            "contact_solimp": SOLIMP,
            "threads": "single (MjData stepping is single-threaded)",
            "note": "compile probe for the hardware-tip planning note; not a study"}
    for case in CASES:
        row = run_case(*case)
        rows.append(row)
        print(json.dumps(row), flush=True)
        OUT.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
        with open(OUT, "rb+") as fh:
            os.fsync(fh.fileno())


if __name__ == "__main__":
    main()
