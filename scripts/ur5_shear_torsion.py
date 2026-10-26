#!/usr/bin/env python3
"""Friction torque of the 1 mm pads about the contact normal under the 5 mm sphere of the UR5 cell, at the friction
coefficient measured in the slides (scripts/ur5_shear_slides.py), against the torsion of the UR5 twists.

Rig: the 812 pads of the UR5 pad (ur5_tip_calibration_pads.Pads, builder v2) fixed, at 0.90 MPa (the modulus fitted to
the sphere presses, sphere2). The 5 mm sphere, 1 g, on a slide along the contact normal through a twist target of cb2t,
pushed with F = 1, 2.5 or 5 N, and on a hinge about the normal through its centre. After 0.4 s of settling a torque about
the normal is ramped at 2 N mm/s until the sphere turns at 30 deg/s, the spin-onset threshold of task 2 on the test bed;
the torque there is the friction torque at spin onset. Friction 0.35, condim 3, elliptic cone, 1 ms, 100 iterations,
impratio 100. Reference: Hertz's patch under the same sphere at the same modulus, tau = mu F (3 pi / 16) a.

The measured torsion of cb2t's twists is half the difference between the way out and the way back at the same angle
(ur5_tip_calibration_cb2t_checks.py; data/20261025-152920-cb2t/checks.json): medians 1.88, 1.57 and 2.75 N mm at 1, 2.5
and 5 N.

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- .venv/bin/python scripts/ur5_shear_torsion.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
OUT = ROOT / "docs/experiments/20261026-shear_friction_impratio/data"
UR5 = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
MU, E_SPHERE, TC, R_S, M_IND, NU = 0.35, 0.90, 0.01, 5.0, 1e-3, 0.45
RAMP = 2.0e-3                   # N m/s: 2 N mm/s
OMEGA_ON = math.radians(30.0)   # rad/s: spin onset (task 2 of the test bed)
SCRIPT = "scripts/ur5_shear_torsion.py"


def hertz_torque_Nmm(F, E=E_SPHERE, R=R_S, mu=MU):
    Es = E / (1 - NU ** 2)
    a = (3 * F * R / (4 * Es)) ** (1 / 3)
    return mu * F * 3 * math.pi / 16 * a, a


def rig(pads, p, n, ir):
    import mujoco
    n = np.asarray(n, float) / np.linalg.norm(n)
    c = (np.asarray(p, float) + R_S * n) * 1e-3                # the sphere touching the surface point p
    ax = lambda u: f"{u[0]:.7f} {u[1]:.7f} {u[2]:.7f}"  # noqa: E731
    geoms = "\n".join(f'<geom name="pad{i}" type="sphere" size="{0.75e-3:.6g}" pos="{q[0] * 1e-3:.7f} {q[1] * 1e-3:.7f} '
                      f'{q[2] * 1e-3:.7f}" condim="3" priority="1" friction="{MU:g} 0 0"/>' for i, q in enumerate(pads.C))
    xml = f"""<mujoco><option timestep="0.001" integrator="implicitfast" gravity="0 0 0" cone="elliptic" impratio="{ir:g}"/>
<worldbody>{geoms}
<body name="ind" pos="{ax(c)}"><joint name="sn" type="slide" axis="{ax(-n)}"/><joint name="tw" type="hinge" axis="{ax(n)}"/>
<geom name="ind" type="sphere" size="{R_S * 1e-3:g}" mass="{M_IND:g}" condim="3"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    A = float(m.body_invweight0[m.body("ind").id, 0])
    K = E_SPHERE * pads.k * 1e3
    d0 = 1.0 - 1.0 / (TC * TC * K * A)
    ng = len(K)
    m.geom_solref[:ng] = [TC, 1.0]
    m.geom_solimp[:ng, 0] = d0
    m.geom_solimp[:ng, 1] = d0
    m.geom_solimp[:ng, 2:] = [0.001, 0.5, 2.0]
    return m


def onset(m, F, T_max=None):
    import mujoco
    d = mujoco.MjData(m)
    js, jt = m.joint("sn").id, m.joint("tw").id
    ds, dt_ = m.jnt_dofadr[js], m.jnt_dofadr[jt]
    d.qfrc_applied[ds] = F
    for _ in range(400):
        mujoco.mj_step(m, d)
    t0 = d.time
    T_max = T_max or 3.0 * MU * F * 3e-3 / RAMP + 1.0
    rec = []
    while d.time - t0 < T_max:
        tau = RAMP * (d.time - t0)
        d.qfrc_applied[dt_] = tau
        mujoco.mj_step(m, d)
        w = float(d.qvel[dt_])
        rec.append((d.time - t0, tau * 1e3, math.degrees(float(d.qpos[m.jnt_qposadr[jt]])), math.degrees(w)))
        if w > OMEGA_ON:
            fn = 0.0
            f6 = np.zeros(6)
            for i in range(d.ncon):
                mujoco.mj_contactForce(m, d, i, f6)
                fn += f6[0]
            return dict(tau_on_Nmm=tau * 1e3, angle_on_deg=rec[-1][2], fn_N=fn, ncon=int(d.ncon),
                        creep_deg_s_at_half=_creep_at(rec, 0.5 * tau * 1e3)), rec
    return dict(tau_on_Nmm=None), rec


def _creep_at(rec, tau_half):
    r = np.array(rec)
    k = int(np.argmin(np.abs(r[:, 1] - tau_half)))
    lo, hi = max(0, k - 50), min(len(r) - 1, k + 50)
    return float((r[hi, 2] - r[lo, 2]) / max(r[hi, 0] - r[lo, 0], 1e-9))


def main():
    import ur5_tip_calibration_pads as UP
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--forces", nargs="+", type=float, default=[1.0, 2.5, 5.0])
    ap.add_argument("--ir", type=float, default=100.0)
    a = ap.parse_args()
    H = np.load(UR5 / "data/20261025-152920-cb2t/holds.npz", allow_pickle=False)
    tw = H["kind"] == "twist"
    targets = {}
    for i in np.nonzero(tw)[0]:
        key = tuple(np.round(H["target"][i], 3))
        targets.setdefault(key, (H["region"][i], H["p"][i], H["n"][i]))
    pads = UP.Pads()
    checks = json.loads((UR5 / "data/20261025-152920-cb2t/checks.json").read_text())["twists"]
    out = OUT / "20261026-torsion_pads.jsonl"
    rows = []
    for key, (region, p, n) in sorted(targets.items()):
        m = rig(pads, p, n, a.ir)
        for F in a.forces:
            t0 = time.time()
            res, rec = onset(m, F)
            th, aH = hertz_torque_Nmm(F)
            meas = checks.get(f"{F:g}N", {})
            row = dict(target_P=list(key), region=str(region), F_N=F, ir=a.ir, mu=MU, E_MPa=E_SPHERE, **res,
                       arm_on_mm=(res["tau_on_Nmm"] / (MU * F)) if res.get("tau_on_Nmm") else None,
                       hertz=dict(tau_Nmm=th, a_mm=aH, arm_mm=3 * math.pi / 16 * aH),
                       measured_cb2t=meas, script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"),
                       wall_s=round(time.time() - t0, 1))
            rows.append(row)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"{region:5s} {key} F {F:g}: onset {res.get('tau_on_Nmm')} N mm (arm {row['arm_on_mm']}), Hertz {th:.2f} N mm, "
                  f"measured median {meas.get('median_Nmm')} N mm, {row['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
