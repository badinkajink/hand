#!/usr/bin/env python3
"""Replay double-pendulum recordings in MuJoCo with BAM (Better Actuator Models) fitted on the single
pendulum (servo 20 only). Both joints use the same fitted model: joint 2 (servo 21) is a different
unit, so this also tests unit-to-unit transfer. Nothing here is fitted to the double-pendulum data,
except optionally the two joint zero offsets (--fit-offsets, on M1 and the P=15 logs only).

    ../.venv-bam/bin/python double_replay.py [--fits fits_best] [--sets double_v2 double_v2_elbow_up]
        [--fit-offsets] [--plot fits_best/double_replay.png]

Conventions: logs from double_pendulum_record.py, degrees. q = 0 hanging, tip
x = L1 sin q1 + L2 sin(q1+q2), y = -L1 cos q1 - L2 cos(q1+q2). Servo 21 is mounted mirrored, so
for logs without "raw_sign" (elbow-up set, recorded before the fix) the physical q2 = -logged q2.
Zero offsets: q_true = q_logged + offset. Default offset = -(hanging reading) per joint, read
2026-10-29 with the 5-hole link 2: servo 20 -3.81 deg, servo 21 raw -0.59 deg (= +0.59 IK).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import mujoco
import numpy as np

import bam_scs0009

bam_scs0009.register()
from bam.model import load_model  # noqa: E402
from bam.mujoco import MujocoController  # noqa: E402

L1, L2 = 0.060, 0.090
HANG_DEG = (-3.81, 0.59)
DT = 0.002


def mass_props():
    """(mass kg, com m, I_com kg m^2) per body about each joint axis, from the logged rig parts."""
    # link 1 + servo 2 (servo 2 ~13 g at 55 mm, ~23 x 12 mm box)
    m_l1, c_l1, i_l1 = 11.9, 20.0, 12690.0 - 11.9 * 20.0**2
    m_s2, c_s2, i_s2 = 13.0, 55.0, 13.0 * (23**2 + 12**2) / 12
    m1 = m_l1 + m_s2
    c1 = (m_l1 * c_l1 + m_s2 * c_s2) / m1
    i1 = i_l1 + m_l1 * (c_l1 - c1) ** 2 + i_s2 + m_s2 * (c_s2 - c1) ** 2
    # link 2: 5-hole arm + steel cylinder 19.88 x 19.88 mm at 90 mm
    m_a, c_a, i_a = 10.8, 37.0, 24088.0 - 10.8 * 37.0**2
    m_c, c_c, i_c = 48.08, 90.0, 48.08 * (19.88 / 2) ** 2 / 2
    m2 = m_a + m_c
    c2 = (m_a * c_a + m_c * c_c) / m2
    i2 = i_a + m_a * (c_a - c2) ** 2 + i_c + m_c * (c_c - c2) ** 2
    return (m1 * 1e-3, c1 * 1e-3, i1 * 1e-9), (m2 * 1e-3, c2 * 1e-3, i2 * 1e-9)


def mjcf(off1: float, off2: float) -> str:
    (m1, c1, i1), (m2, c2, i2) = mass_props()
    # hinge axis -y: rotating (0,0,-1) by q gives (sin q, 0, -cos q), the logged convention.
    # ref = offset (rad) so qpos stays in the logged frame while the geometry is at qpos + offset.
    return f"""
<mujoco>
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="{DT}" gravity="0 0 -9.80665" integrator="implicitfast" noslip_iterations="1"/>
  <worldbody>
    <body name="link1">
      <joint name="j1" type="hinge" axis="0 -1 0" ref="{off1}"/>
      <inertial pos="0 0 {-c1}" mass="{m1}" diaginertia="{i1} {i1} {i1}"/>
      <body name="link2" pos="0 0 {-L1}">
        <joint name="j2" type="hinge" axis="0 -1 0" ref="{off2}"/>
        <inertial pos="0 0 {-c2}" mass="{m2}" diaginertia="{i2} {i2} {i2}"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="j1" joint="j1"/>
    <motor name="j2" joint="j2"/>
  </actuator>
</mujoco>"""


def load_double(path: str):
    d = json.load(open(path))
    E = d["entries"]
    t = np.array([e["timestamp"] for e in E])
    q = np.radians(np.array([e["position"] for e in E], dtype=float))
    g = np.radians(np.array([e["goal_position"] for e in E], dtype=float))
    if "raw_sign" not in d:          # elbow-up set: physical q2 = -logged
        q[:, 1] *= -1
        g[:, 1] *= -1
    vins = [v for e in E if e["input_volts"] for v in e["input_volts"]]
    phase = np.array([e["phase"] for e in E])
    return {"name": os.path.basename(path), "kp": d["kp"], "kd": d.get("kd", 15), "t": t, "q": q, "goal": g,
            "vin": float(np.mean(vins)), "path_mask": phase == "path", "set": os.path.basename(os.path.dirname(path))}


def rollout(model, log, off):
    """Simulate; returns simulated q (rad, logged frame) at the log timestamps."""
    model.reset()
    mm = mujoco.MjModel.from_xml_string(mjcf(*off))
    md = mujoco.MjData(mm)
    act = model.actuator
    act.kp, act.vin, act.kd = log["kp"], log["vin"], log["kd"]
    ctrl = MujocoController(model, ["j1", "j2"], mm, md)
    t, q, goal = log["t"] - log["t"][0], log["q"], log["goal"]
    md.qpos[:] = q[0]
    mujoco.mj_forward(mm, md)
    ctrl.reset()
    delay = model.command_delay.value
    ts = np.arange(0.0, t[-1], DT)
    out = np.empty((len(ts), 2))
    for k, tt in enumerate(ts):
        tg = max(tt - delay, 0.0)
        for j, name in enumerate(("j1", "j2")):
            ctrl.set_q_target(name, np.interp(tg, t, goal[:, j]))
        ctrl.update()
        mujoco.mj_step(mm, md)
        out[k] = md.qpos
    return np.stack([np.interp(t, ts, out[:, j]) for j in range(2)], axis=1)


def tip(q):
    return np.stack([L1 * np.sin(q[:, 0]) + L2 * np.sin(q.sum(1)),
                     -L1 * np.cos(q[:, 0]) - L2 * np.cos(q.sum(1))], axis=1)


def score(sim, log, off):
    m = log["path_mask"]
    jerr = np.degrees(np.mean(np.abs(sim[m] - log["q"][m]), axis=0))
    off = np.asarray(off)
    terr = 1000 * np.mean(np.linalg.norm(tip(sim[m] + off) - tip(log["q"][m] + off), axis=1))
    return jerr, terr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", default="fits_best")
    ap.add_argument("--sets", nargs="+", default=["double_v2", "double_v2_elbow_up"])
    ap.add_argument("--fit-offsets", action="store_true")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--kd", nargs="+", type=int, default=[0], help="replay logs at these firmware D gains")
    ap.add_argument("--d-gain", type=float, default=None, help="override the D-term scale (bam_scs0009.D_GAIN)")
    a = ap.parse_args()

    logs = [load_double(f) for s in a.sets for f in sorted(glob.glob(f"data_raw/{s}/*.json"))]
    logs = [l for l in logs if l["kd"] in a.kd]
    fits = {os.path.basename(f)[:-5]: f for f in sorted(glob.glob(f"{a.fits}/m*.json"))}
    off = np.radians([-HANG_DEG[0], -HANG_DEG[1]])

    if a.fit_offsets:
        from scipy.optimize import minimize
        m1 = load_model(fits["m1"])
        tune = [l for l in logs if l["kp"] == 15 and l["set"] == "double_v2" and l["name"].split("_")[2] in ("circle", "triangle")]

        def cost(x):
            return sum(np.mean(np.abs(rollout(m1, l, x)[l["path_mask"]] - l["q"][l["path_mask"]])) for l in tune)
        r = minimize(cost, off, method="Nelder-Mead", options={"xatol": np.radians(0.2), "fatol": 1e-5, "maxfev": 60})
        off = r.x
        print(f"fitted zero offsets (M1, {len(tune)} P=15 logs): {np.degrees(off).round(2)} deg "
              f"(hanging-reading default {[-HANG_DEG[0], -HANG_DEG[1]]})")
    print(f"zero offsets used: {np.degrees(off).round(2)} deg")

    res = {}
    for name, path in fits.items():
        model = load_model(path)
        if a.d_gain is not None:
            model.d_gain.value = a.d_gain
        for l in logs:
            sim = rollout(model, l, off)
            res[(name, l["name"], l["set"])] = (*score(sim, l, off), sim)

    sets = sorted({l["set"] for l in logs})
    print(f"mean absolute error during the path, D in {a.kd}:  joint 1 / joint 2 (deg),  tip (mm)")
    for s in sets:
        for kp, kd in sorted({(l["kp"], l["kd"]) for l in logs if l["set"] == s}, reverse=True):
            ls = [l for l in logs if l["set"] == s and l["kp"] == kp and l["kd"] == kd]
            if not ls:
                continue
            row = []
            for name in fits:
                j = np.mean([res[(name, l["name"], s)][0] for l in ls], axis=0)
                tp = np.mean([res[(name, l["name"], s)][1] for l in ls])
                row.append(f"{name.upper()} {j[0]:4.2f}/{j[1]:4.2f} {tp:4.1f}")
            print(f"{s:20s} P={kp:<2d} D={kd:<3d} | " + " | ".join(row))
    allj = {n: np.mean([res[(n, l['name'], l['set'])][1] for l in logs]) for n in fits}
    print("overall tip error (mm): " + ", ".join(f"{n.upper()} {v:.2f}" for n, v in allj.items()))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        picks = [l for l in logs if l["set"] == a.sets[0] and l["kp"] == min(x["kp"] for x in logs if x["set"] == a.sets[0])]
        fig, axes = plt.subplots(len(picks), 2, figsize=(13, 2.4 * len(picks)), sharex=True)
        for row, l in zip(axes, picks):
            t = l["t"] - l["t"][0]
            for j in range(2):
                ax = row[j]
                ax.plot(t, np.degrees(l["goal"][:, j]), "k:", lw=1, label="goal")
                ax.plot(t, np.degrees(l["q"][:, j]), "k", lw=1.6, label="real")
                for n, c in (("m1", "tab:orange"), ("m6", "tab:blue")):
                    ax.plot(t, np.degrees(res[(n, l["name"], l["set"])][2][:, j]), c, lw=1.1, label=f"sim {n.upper()}")
                ax.set_title(f"{l['name'][16:-5]}  joint {j + 1} (servo {20 + j})", fontsize=9)
                ax.set_ylabel("deg")
        axes[0, 0].legend(fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=100)
        print("plot ->", a.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
