#!/usr/bin/env python3
"""The sprung skin against the UR5 slides of the printed fingertip: the 812 pads of the UR5 rig (ur5_tip_calibration_pads)
on a skin body sized from the slides themselves, replaying the measured slides, against plain pads. The base follows the
measured travel and the sphere the measured normal force, both smoothed over 0.7 s.

Sizing (scripts/ur5_shear_slides.py, step 1 of hand-off 20261026-shear_friction_impratio): each sideways spring is the
measured give before slip, k_t = 1 / give, 266 um/N at 1 N on tip 2 (3.76 kN/m). The twist hinge takes Lubkin's ratio to
Mindlin's tangential stiffness, k_theta / k_t = (2/3)(2 - nu) a^2, with a the Hertz radius of the 5 mm sphere on the pad
at the modulus fitted to the sphere presses (0.90 MPa), nu 0.45; the slides do not measure torsion. The skin is a 2 g body
on two slides tangent to the face at the slide's contact point and a hinge about its normal, with 18 g of armature on the
slides (the bed's) and critical damping on mass plus armature.

Rig: the pads, at the sphere presses' 0.90 MPa (builder v2, d0 from the compiled inverse weights as in
ur5_tip_calibration_pads.Rig.set_E), on the skin, or fixed to the base for plain pads. The base rides on two slides in
the tangent plane, driven by stiff position actuators (1e6 N/m, 10 kg of armature, critical kv) along the measured travel
of the pad: each direction k of a slide moves it by d_k(t) v_k from where direction k-1 ended, so the eight directions
replay in sequence with the skin's stored shear carried over. The 5 mm sphere, 1 g, slides along the contact normal only
and is pushed by the measured normal force f_n(t). Friction 0.35, the slides' measured coefficient; condim 3, elliptic
cone, 1 ms, 100 iterations, at impratio 10, 100 and 1000. The shear is the contact force on the sphere along v_k. Each
direction's simulated loop is fitted with ur5_shear_slides.fit_epp, as the measured one was, and compared with the
measured shear (less its fitted offset) by RMS.

    .venv/bin/python scripts/ur5_shear_skin.py size
    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- .venv/bin/python scripts/ur5_shear_skin.py slides
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
import ur5_shear_slides as U  # noqa: E402

OUT = U.OUT
NU = 0.45
MU = 0.35                       # the slides' friction coefficient (step 1, tip 2 at 1 and 2 N)
E_SPHERE = 0.90                 # MPa, the pads fitted to the sphere presses (sphere2)
TC = 0.01                       # s, the pads' solref time constant
M_SKIN, ARM_SKIN, I_SKIN = 0.002, 0.018, 5e-8
ARM_T = 5e-7                    # kg m^2 on the hinge (the bed's)
M_IND = 1e-3
BASE_KP, BASE_ARM = 1e6, 10.0
SCRIPT = "scripts/ur5_shear_skin.py"
SMOOTH_S = 0.7                  # s: moving average of the measured travel and normal force that drive the rig


def smooth(x, k):
    """Centred moving average over k samples, the ends padded with the end values."""
    if k <= 1:
        return np.asarray(x, float)
    xp = np.r_[np.full(k // 2, x[0]), x, np.full(k - 1 - k // 2, x[-1])]
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def sizing(F=1.0):
    """k_t (N/m) from the slides' give at F (the power law of step 1, tip 2) and k_theta (N m/rad) by Lubkin's ratio."""
    S = json.loads((OUT / "20261026-ur5_slides_summary.json").read_text())["cb2t"]
    pl = S["give_power_law"]
    give = pl["at_1N_um_N"] * F ** pl["exponent"]
    lv = S["by_level"].get(f"{F:g}")
    if lv:                                             # a press level the slides measured: its median
        give = lv["give_um_N"]["median"]
    kt = 1e6 / give
    Es = E_SPHERE * 1e6 / (1 - NU ** 2)
    a = (3 * F * 5e-3 / (4 * Es)) ** (1 / 3)
    kth = kt * (2.0 / 3.0) * (2 - NU) * a ** 2
    return dict(F_N=F, give_um_N=give, kt_N_m=kt, a_mm=a * 1e3, kth_N_m_rad=kth)


def frame(n, v):
    n = np.asarray(n, float) / np.linalg.norm(n)
    v = np.asarray(v, float)
    v = v - (v @ n) * n
    v /= np.linalg.norm(v)
    return n, v, np.cross(n, v)


def build(pads, row, skin, ir, mu=None):
    """MJCF of the rig for one slide (its first direction's frame: contact point, normal and the tangent plane)."""
    import mujoco
    n, t1, t2 = frame(row["n_P"], row["v_P"])
    pc = np.asarray(row["p_P"], float) * 1e-3
    cP = np.asarray(row["c_P"], float) * 1e-3
    geoms = "\n".join(f'<geom name="pad{i}" type="sphere" size="{0.75e-3:.6g}" pos="{c[0] * 1e-3:.7f} {c[1] * 1e-3:.7f} '
                      f'{c[2] * 1e-3:.7f}" condim="3" priority="1" friction="{mu or MU:g} 0 0"/>' for i, c in enumerate(pads.C))
    ax = lambda u: f"{u[0]:.7f} {u[1]:.7f} {u[2]:.7f}"  # noqa: E731
    if skin:
        kt, kth = skin["kt_N_m"], skin["kth_N_m_rad"]
        ct = 2 * math.sqrt(kt * (M_SKIN + ARM_SKIN))
        cth = 2 * math.sqrt(kth * (I_SKIN + ARM_T))
        pad_body = (f'<body name="skin" pos="0 0 0"><inertial pos="{ax(pc)}" mass="{M_SKIN:g}" diaginertia="{I_SKIN:g} {I_SKIN:g} {I_SKIN:g}"/>'
                    f'<joint name="sk1" type="slide" axis="{ax(t1)}" stiffness="{kt:.6g}" damping="{ct:.6g}" armature="{ARM_SKIN:g}"/>'
                    f'<joint name="sk2" type="slide" axis="{ax(t2)}" stiffness="{kt:.6g}" damping="{ct:.6g}" armature="{ARM_SKIN:g}"/>'
                    f'<joint name="skt" type="hinge" pos="{ax(pc)}" axis="{ax(n)}" stiffness="{kth:.6g}" damping="{cth:.6g}" armature="{ARM_T:g}"/>'
                    f'{geoms}</body>')
    else:
        pad_body = geoms
    kv = 2 * math.sqrt(BASE_KP * BASE_ARM)
    xml = f"""<mujoco><option timestep="0.001" integrator="implicitfast" gravity="0 0 0" cone="elliptic" impratio="{ir:g}"/>
<worldbody>
<body name="base"><inertial pos="0 0 0" mass="0.1" diaginertia="1e-5 1e-5 1e-5"/>
<joint name="b1" type="slide" axis="{ax(t1)}" armature="{BASE_ARM:g}"/><joint name="b2" type="slide" axis="{ax(t2)}" armature="{BASE_ARM:g}"/>
{pad_body}</body>
<body name="ind" pos="{ax(cP)}"><joint name="sn" type="slide" axis="{ax(-n)}"/>
<geom name="ind" type="sphere" size="{5e-3:g}" mass="{M_IND:g}" condim="3"/></body>
</worldbody>
<actuator><position name="a1" joint="b1" kp="{BASE_KP:g}" kv="{kv:.6g}"/><position name="a2" joint="b2" kp="{BASE_KP:g}" kv="{kv:.6g}"/></actuator>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    # pads' stiffness: K_i = E a_i / h_i through d0 with the pad body's and the sphere's inverse weights
    pb = m.body("skin").id if skin else m.body("base").id
    A = float(m.body_invweight0[pb, 0] + m.body_invweight0[m.body("ind").id, 0])
    K = E_SPHERE * pads.k * 1e3
    d0 = 1.0 - 1.0 / (TC * TC * K * A)
    if d0.min() < 0.05:
        raise ValueError(f"pad d0 {d0.min():.3f} at A {A:.3g}")
    ng = len(K)
    m.geom_solref[:ng] = [TC, 1.0]
    m.geom_solimp[:ng, 0] = d0
    m.geom_solimp[:ng, 1] = d0
    m.geom_solimp[:ng, 2:] = [0.001, 0.5, 2.0]
    return m, dict(t1=t1, t2=t2, n=n, A=A, d0=[float(d0.min()), float(d0.max())])


def replay(m, fr, dirs):
    """Replay the directions of one slide in sequence. dirs: (row, trace) per direction with the trace's t (s from the
    direction's start), d (mm), fn (N). Returns per direction the simulated t, d, s, fn and the wall time per step."""
    import mujoco
    d = mujoco.MjData(m)
    sn = m.jnt_qposadr[m.joint("sn").id]
    gi = m.geom("ind").id
    t1, t2, n = fr["t1"], fr["t2"], fr["n"]
    mujoco.mj_forward(m, d)
    # settle the press at the first direction's starting normal force
    f6 = np.zeros(6)
    p0 = np.zeros(2)                        # base target in (t1, t2), m
    out, W = [], []
    first = True
    for row, tr in dirs:
        v = np.asarray(row["v_P"], float)
        v = v - (v @ n) * n
        v /= np.linalg.norm(v)
        vt = np.array([v @ t1, v @ t2])
        tt, dd, ff = tr["t"], tr["d_drive"] * 1e-3, tr["fn_drive"]
        if first:
            d.qfrc_applied[m.jnt_dofadr[m.joint("sn").id]] = float(ff[0])
            for _ in range(500):
                mujoco.mj_step(m, d)
            first = False
        T0 = d.time
        rec = dict(t=[], d=[], s=[], fn=[])
        k_rec = int(round(U.BIN_S / m.opt.timestep))
        nsteps = int(round(tt[-1] / m.opt.timestep))
        for k in range(nsteps):
            t = k * m.opt.timestep
            dk = float(np.interp(t, tt, dd))
            d.ctrl[:2] = p0 + dk * vt
            d.qfrc_applied[m.jnt_dofadr[m.joint("sn").id]] = float(np.interp(t, tt, ff))
            w0 = time.perf_counter()
            mujoco.mj_step(m, d)
            W.append(time.perf_counter() - w0)
            if (k + 1) % k_rec == 0:
                F = np.zeros(3)
                for i in range(d.ncon):
                    c = d.contact[i]
                    if gi not in (c.geom[0], c.geom[1]):
                        continue
                    mujoco.mj_contactForce(m, d, i, f6)
                    fw = c.frame.reshape(3, 3).T @ f6[:3]            # on geom2 from geom1, world
                    F += fw if c.geom[1] == gi else -fw
                base = np.array([d.qpos[m.jnt_qposadr[m.joint("b1").id]], d.qpos[m.jnt_qposadr[m.joint("b2").id]]])
                rec["t"].append(t)
                rec["d"].append(float((base - p0) @ vt) * 1e6)
                rec["s"].append(float(F @ v))
                rec["fn"].append(float(F @ n))
        p0 = p0 + float(dd[-1]) * vt
        out.append({k_: np.asarray(v_) for k_, v_ in rec.items()})
    return out, float(np.median(W) * 1e6), int(np.isfinite(d.qpos).all())


def cmd_size(a):
    for F in (1.0, 2.0, 5.0):
        print(sizing(F))


def cmd_slides(a):
    import ur5_tip_calibration_pads as UP
    rows = [r for r in U.load_rows("cb2t") if r["region"] in a.regions and r["press_N"] in a.levels]
    Z = np.load(OUT / "20261026-ur5_slide_traces_cb2t.npz")
    meta = json.loads(str(Z["meta"]))
    idx = {(m_["action"], m_["dir_index"]): i for i, m_ in enumerate(meta)}
    actions = sorted({r["action"] for r in rows})[: a.max_slides]
    pads = UP.Pads()
    skin = sizing(1.0)
    out = OUT / (a.out or "20261026-skin_slides.jsonl")
    have = {(r["action"], r["model"], r["ir"]) for r in (json.loads(l) for l in open(out))} if out.exists() else set()
    for aid in actions:
        R = sorted([r for r in rows if r["action"] == aid], key=lambda r: r["dir_index"])
        dirs = []
        for r in R:
            i = idx[(aid, r["dir_index"])]
            tr = {k: Z[f"{k}_{i}"].astype(float) for k in ("t", "d", "s", "fn")}
            # the robot's pose is logged at 10 um, so the binned travel has reversals of a few um between bins that rigid
            # Coulomb friction turns into full reversals of the shear: drive the base along the travel smoothed over SMOOTH_S
            tr["d_drive"] = smooth(tr["d"], int(round(SMOOTH_S / U.BIN_S)))
            tr["fn_drive"] = smooth(tr["fn"], int(round(SMOOTH_S / U.BIN_S)))
            # each trace starts at its direction's start hold, which is the previous direction's return hold
            dirs.append((r, tr))
        for model in a.models:
            for ir in a.ir:
                if (aid, model, ir) in have:
                    continue
                t0 = time.time()
                mu = a.mu_by_level.get(R[0]["press_N"], MU) if a.mu_by_level else MU
                m, fr = build(pads, R[0], skin if model == "skin" else None, ir, mu)
                sims, us, ok = replay(m, fr, dirs)
                res = []
                for (r, tr), sm in zip(dirs, sims):
                    fit = U.fit_epp(sm["d"] * 1e-3, sm["s"], np.maximum(sm["fn"], 0.05))
                    n = min(len(sm["s"]), len(tr["s"]))
                    meas = tr["s"][:n] - r["epp"]["s_off_N"]
                    rms = float(np.sqrt(np.mean((sm["s"][:n] - meas) ** 2)))
                    back = float(sm["s"][-3:].mean())
                    res.append(dict(dir_deg=r["dir_deg"], dir_index=r["dir_index"], fit=fit, rms_vs_measured_N=rms,
                                    s_back_N=back, measured=dict(give_um_N=r["epp"]["give_um_N"], mu=r["epp"]["mu"],
                                                                 z_back_N=r["z_back_N"], rms_N=r["epp"]["rms_N"]),
                                    trace=dict(t=np.round(sm["t"], 3).tolist(), d=np.round(sm["d"], 2).tolist(),
                                               s=np.round(sm["s"], 4).tolist())))
                row = dict(action=aid, region=R[0]["region"], press_N=R[0]["press_N"], model=model, ir=ir, mu=mu,
                           E_MPa=E_SPHERE, skin=skin if model == "skin" else None, rig=dict(A=fr["A"], d0=fr["d0"]),
                           us_per_step_median=us, finite=bool(ok), directions=res, script=SCRIPT,
                           when=time.strftime("%Y-%m-%d %H:%M"))
                with open(out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                g = [x["fit"]["give_um_N"] for x in res if x["fit"]]
                print(f"{aid} {model} ir {ir:g}: give {np.median(g):.0f} um/N (measured "
                      f"{np.median([x['measured']['give_um_N'] for x in res]):.0f}), mu "
                      f"{np.median([x['fit']['mu'] for x in res if x['fit']]):.3f}, rms against measured "
                      f"{np.median([x['rms_vs_measured_N'] for x in res]):.3f} N, {us:.0f} us/step, {time.time() - t0:.0f} s",
                      flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("size")
    s = sub.add_parser("slides")
    s.add_argument("--models", nargs="+", default=["pads", "skin"])
    s.add_argument("--ir", nargs="+", type=float, default=[10.0, 100.0, 1000.0])
    s.add_argument("--regions", nargs="+", default=["face"])
    s.add_argument("--levels", nargs="+", type=float, default=[1.0, 2.0])
    s.add_argument("--max-slides", type=int, default=1000)
    s.add_argument("--out", default=None, help="rows file name under the study's data/ (one per process)")
    s.add_argument("--mu", nargs="*", default=None, help="friction per press level, as level=mu (e.g. 1=0.37 2=0.44)")
    a = ap.parse_args()
    if a.cmd == "slides":
        a.mu_by_level = {float(x.split("=")[0]): float(x.split("=")[1]) for x in (a.mu or [])}
    {"size": cmd_size, "slides": cmd_slides}[a.cmd](a)


if __name__ == "__main__":
    main()
