#!/usr/bin/env python3
"""Tool mass against the 1 mm sphere-pad calibration: fixed Lambda versus a load-time per-object Lambda.

Each pad sphere gets its solimp d0 from the target stiffness K_s through d0 = 1 - 1/(tc^2 K_s Lambda),
with Lambda the inverse effective mass that the solver puts in the constraint regulariser
(efc_diagApprox; in MuJoCo 3.6 the sum of the two bodies' body_invweight0, the same at every pose).
The chain fixes Lambda at 47.2 kg^-1 (thumb tip 6.47 + nominal tool 40.74), so a heavier or lighter
tool changes the realised stiffness by 47.2 / Lambda(m).

Rig: the 10-01 two-pad pinch (scripts/hom_contact_rig.py) with the chain's pad, s 1 mm, rs 0.75 mm,
cap 45 deg, E 10 MPa, relaxation 20 ms (tc 10 ms), impratio 100. Each pad's mass on its rail is set
to 1/6.4712 kg so its inverse weight equals the chain thumb tip's and Lambda(1x) = 47.21 kg^-1.
Tool mass 0.25x, 1x and 4x nominal (24.5 g; inertia scaled with it), pinch 0.5 and 3 N per pad.

  fixed     d0 from Lambda = 47.2 at every mass (the chain's current calibration)
  loadtime  d0 per pad from efc_diagApprox of that pad's contact rows, read once after mj_forward
            with the pads 0.2 mm into the tool (no runtime cost)

pinch (gravity off): 1 s settle, then per-sphere force/overlap, spheres in contact, contact area,
friction arm rbar = sum(r_i f_i)/sum(f_i) about the pinch axis, and the friction torque while the tool
is driven to spin about the pinch axis at 1 rad/s (rbar_tau = tau / (2 mu N), mu 1).
hold (gravity on): 0.3 s weight-compensated close, then a 1 s hold under gravity at physics step
1 ms and 5 ms.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/pad_mass_calibration.py   # -> docs/experiments/20261005-pad_calibration/mass_calibration.jsonl
    $PY scripts/pad_mass_calibration.py --only hold --calibs loadtime --masses 4,2 --dts 2,2.5,3,4,5   # step limit
    $PY scripts/pad_mass_calibration.py --only hold --calibs loadtime --masses 4 --dts 5 --tc 0.0115  # longer tc

Hold stability bound: with many contacts on one pad the pad moves as one mode whose reference damping rate is
2 / (d0 tc), so the explicit velocity update needs dt < d0 tc. A heavier tool lowers Lambda and d0.
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
import hom_contact_rig as H  # noqa: E402
import mujoco  # noqa: E402

SPEC = "mj:spheres:s1:rs0.75:tr0.02:ir100:cap45"
TC = 0.01                       # solref time constant = relaxation / 2
PAD_INV = 6.4712                # chain thumb_tip body_invweight0 (index_tip 6.4644), kg^-1
LAMBDA_FIXED = 47.2             # the chain's calibration Lambda, kg^-1
FIT_C, FIT_P = 0.996e-3, 0.25   # hydroelastic torsion law rbar = c N^(1/4), Drake E 10 MPa, 10-01 rig
MU = 1.0
OUT = ROOT / "docs/experiments/20261005-pad_calibration"
M0, IT0, IA0 = H.M_TOOL, H.I_TOOL_T, H.I_TOOL_A
MASSES, FORCES = (0.25, 1.0, 4.0), (0.5, 3.0)


def build(mscale, gravity):
    """Compiled rig model with the tool mass scaled; returns (model, info, {pad sphere geom: side})."""
    saved = H.M_PAD, H.M_TOOL, H.I_TOOL_T, H.I_TOOL_A
    H.M_PAD, H.M_TOOL, H.I_TOOL_T, H.I_TOOL_A = 1.0 / PAD_INV, M0 * mscale, IT0 * mscale, IA0 * mscale
    try:
        xml, info = H.mj_xml(H.parse_spec(SPEC), 0.0, gravity, pad_stiff=(TC, 0.5))
    finally:
        H.M_PAD, H.M_TOOL, H.I_TOOL_T, H.I_TOOL_A = saved
    m = mujoco.MjModel.from_xml_string(xml)
    side = {g: m.geom(g).name[3] for g in range(m.ngeom) if m.geom(g).name.startswith(("padL_s", "padR_s"))}
    return m, info, side


def _diag(d):
    return d.efc_diagApprox if hasattr(d, "efc_diagApprox") else d.efc_diagA


def lambda_at_touch(m, side, depth=2e-4):
    """Mean, spread and count of the normal-row efc_diagApprox of each pad's contacts at the touch pose."""
    d = mujoco.MjData(m)
    d.qpos[m.jnt_qposadr[m.joint("railL").id]] = depth
    d.qpos[m.jnt_qposadr[m.joint("railR").id]] = -depth
    mujoco.mj_forward(m, d)
    dA = _diag(d)
    lam = {"L": [], "R": []}
    for i in range(d.ncon):
        c = d.contact[i]
        s = side.get(int(c.geom[0])) or side.get(int(c.geom[1]))
        if s and c.efc_address >= 0:
            lam[s].append(float(dA[c.efc_address]))
    return {s: {"mean": float(np.mean(v)), "ptp": float(np.ptp(v)), "rows": len(v)} for s, v in lam.items()}


def calibrate(m, side, lam_by_side, K_s):
    """Write d0 (solimp d0 = dmax) into each pad's sphere geoms; pad priority so the tool's values are not mixed in."""
    d0s = {}
    for s, lam in lam_by_side.items():
        d0 = 1.0 - 1.0 / (TC ** 2 * K_s * lam)
        if d0 < 0.05:
            raise ValueError(f"Lambda {lam:.3g} gives d0 {d0:.3g}, below the 0.05 floor")
        d0s[s] = d0
    for g, s in side.items():
        m.geom_solimp[g, 0] = m.geom_solimp[g, 1] = d0s[s]
        m.geom_priority[g] = 1
    return d0s


def contact_rows(m, d, side):
    """Per pad: (normal force, overlap, distance of the contact point from the pinch axis) for each sphere."""
    f6 = np.zeros(6)
    rows = {"L": [], "R": []}
    for i in range(d.ncon):
        c = d.contact[i]
        s = side.get(int(c.geom[0])) or side.get(int(c.geom[1]))
        if s is None:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        p = c.pos
        rows[s].append((float(f6[0]), float(-c.dist), math.hypot(p[1], p[2])))   # pinch axis = world x
    return rows


def summarize(rows, A_s):
    out = {}
    for s, r in rows.items():
        a = np.array(r, float).reshape(-1, 3)
        f, dl, rr = a[:, 0], a[:, 1], a[:, 2]
        on = f > 1e-6
        ok = on & (dl > 1e-6)
        N = float(f.sum())
        out[s] = {"N": N, "n": int(on.sum()), "area_mm2": float(on.sum() * A_s * 1e6),
                  "k_ls": float((f[ok] * dl[ok]).sum() / max((dl[ok] ** 2).sum(), 1e-30)),
                  "k_med": float(np.median(f[ok] / dl[ok])) if ok.any() else float("nan"),
                  "rbar_mm": float((rr * f).sum() / N * 1e3) if N > 0 else float("nan"),
                  "overlap_max_mm": float(dl.max() * 1e3) if len(dl) else 0.0}
    return out


class Tool:
    def __init__(self, m):
        self.b = m.body("tool").id
        ja = m.body_jntadr[self.b]
        self.qa, self.va = m.jnt_qposadr[ja], m.jnt_dofadr[ja]
        self.prev, self.unwrap = None, 0.0

    def R(self, d):
        R9 = np.zeros(9)
        mujoco.mju_quat2Mat(R9, d.qpos[self.qa + 3:self.qa + 7])
        return R9.reshape(3, 3)

    def theta(self, d):
        a = self.R(d)[:, 2]
        th = math.atan2(a[2], a[1])
        if self.prev is not None:
            self.unwrap += (th - self.prev + math.pi) % (2 * math.pi) - math.pi
        else:
            self.unwrap = th
        self.prev = th
        return self.unwrap

    def omega_x(self, d):
        return float((self.R(d) @ d.qvel[self.va + 3:self.va + 6])[0])


def setup(mscale, calib, gravity, dt):
    m, info, side = build(mscale, gravity)
    lam = lambda_at_touch(m, side)
    used = {s: LAMBDA_FIXED for s in "LR"} if calib == "fixed" else {s: lam[s]["mean"] for s in "LR"}
    d0s = calibrate(m, side, used, info["K_sphere"])
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, info, side, lam, used, d0s


def pinch(mscale, N, calib, dt=1e-3, T_settle=1.0, T_spin=0.8, omega=1.0):
    t0w = time.time()
    m, d, info, side, lam, used, d0s = setup(mscale, calib, False, dt)
    K_s, A_s = info["K_sphere"], info["area_per_sphere_mm2"] * 1e-6
    d.ctrl[:] = [N, -N]
    mujoco.mj_step(m, d, nstep=int(round(T_settle / dt)))
    pads = summarize(contact_rows(m, d, side), A_s)
    tool = Tool(m)
    th0, ts, kp, kd, taus = tool.theta(d), d.time, 0.5, 0.005, []
    for _ in range(int(round(T_spin / dt))):
        tt = d.time - ts
        tau = kp * (th0 + omega * tt - tool.theta(d)) + kd * (omega - tool.omega_x(d))
        d.xfrc_applied[tool.b, 3:] = [tau, 0.0, 0.0]
        mujoco.mj_step(m, d)
        if tt > 0.4:
            taus.append(tau)
    lam_act = lam["L"]["mean"]
    mean = {k: float(np.mean([pads[s][k] for s in "LR"])) for k in pads["L"]}
    tau = float(np.mean(taus))
    _, rbar_w, w = H.winkler_law(N, H.parse_spec(SPEC)["E"])
    return {"exp": "pinch", "calib": calib, "mscale": mscale, "m_tool_g": M0 * mscale * 1e3, "N_cmd": N,
            "dt_ms": dt * 1e3, "tc": TC, "Lambda_touch": lam_act, "Lambda_touch_ptp": lam["L"]["ptp"],
            "Lambda_used": used["L"], "d0": d0s["L"], "K_target": K_s,
            "k_pred": K_s * used["L"] / lam_act, "k_ls": mean["k_ls"], "k_med": mean["k_med"],
            "k_ratio": mean["k_ls"] / K_s, "N_meas": mean["N"], "n_contact": mean["n"],
            "area_mm2": mean["area_mm2"], "overlap_max_mm": mean["overlap_max_mm"], "rbar_mm": mean["rbar_mm"],
            "rbar_hydro_mm": FIT_C * N ** FIT_P * 1e3, "rbar_winkler_mm": rbar_w * 1e3,
            "winkler_area_mm2": math.pi * w["half_y"] * w["half_z"] * 1e6,
            "tau_spin_mNm": tau * 1e3, "tau_spin_sd_mNm": float(np.std(taus)) * 1e3,
            "rbar_tau_mm": tau / (2 * MU * N) * 1e3, "pads": pads, "wall_s": time.time() - t0w}


def hold(mscale, N, calib, dt, T_close=0.3, T_hold=1.0):
    t0w = time.time()
    m, d, info, side, lam, used, d0s = setup(mscale, calib, True, dt)
    tool = Tool(m)
    W = float(m.body_mass[tool.b]) * 9.81
    d.ctrl[:] = [N, -N]
    d.xfrc_applied[tool.b, 2] = W
    mujoco.mj_step(m, d, nstep=int(round(T_close / dt)))
    d.xfrc_applied[tool.b, :] = 0.0
    p0, th0 = d.qpos[tool.qa:tool.qa + 3].copy(), tool.theta(d)
    vmax, n_hold, finite = 0.0, int(round(T_hold / dt)), True
    for k in range(n_hold):
        mujoco.mj_step(m, d)
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            finite = False
            break
        if k * dt >= T_hold - 0.5:
            vmax = max(vmax, float(np.linalg.norm(d.qvel[tool.va:tool.va + 3])))
    pads = summarize(contact_rows(m, d, side), info["area_per_sphere_mm2"] * 1e-6) if finite else {}
    warn = {str(i): int(d.warning[i].number) for i in range(len(d.warning)) if d.warning[i].number}
    disp = float(np.linalg.norm(d.qpos[tool.qa:tool.qa + 3] - p0) * 1e3) if finite else float("nan")
    Nm = float(np.mean([pads[s]["N"] for s in "LR"])) if finite else float("nan")
    stable = finite and not warn and abs(Nm / N - 1.0) < 0.10 and vmax < 0.010 and disp < 1.0
    return {"exp": "hold", "calib": calib, "mscale": mscale, "m_tool_g": M0 * mscale * 1e3, "N_cmd": N,
            "dt_ms": dt * 1e3, "tc": TC, "dt_limit_ms": d0s["L"] * TC * 1e3, "d0": d0s["L"], "Lambda_used": used["L"], "Lambda_touch": lam["L"]["mean"],
            "weight_N": W, "finite": finite, "warnings": warn, "N_meas": Nm,
            "n_contact": float(np.mean([pads[s]["n"] for s in "LR"])) if finite else float("nan"),
            "k_ratio": float(np.mean([pads[s]["k_ls"] for s in "LR"])) / info["K_sphere"] if finite else float("nan"),
            "rbar_mm": float(np.mean([pads[s]["rbar_mm"] for s in "LR"])) if finite else float("nan"),
            "tool_disp_mm": disp, "tool_rot_deg": math.degrees(tool.theta(d) - th0) if finite else float("nan"),
            "tool_vmax_last05_mmps": vmax * 1e3, "stable": stable, "wall_s": time.time() - t0w}


def append(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "mass_calibration.jsonl"))
    ap.add_argument("--only", choices=("pinch", "hold"), default=None)
    ap.add_argument("--dts", default="1,5", help="hold physics steps, ms")
    ap.add_argument("--calibs", default="loadtime,fixed", help="hold calibrations")
    ap.add_argument("--masses", default=",".join(f"{v:g}" for v in MASSES), help="hold tool mass scales")
    ap.add_argument("--tc", type=float, default=None, help="solref time constant override, s (K_s unchanged)")
    a = ap.parse_args()
    global TC
    if a.tc is not None:
        TC = a.tc
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"mujoco {mujoco.__version__}; {SPEC}; tc {TC}; Lambda fixed {LAMBDA_FIXED}", flush=True)
    if a.only in (None, "pinch"):
        for calib in ("fixed", "loadtime"):
            for ms in MASSES:
                for N in FORCES:
                    r = pinch(ms, N, calib)
                    append(out, r)
                    print(f"pinch {calib:8s} m {ms:4.2f} N {N:3.1f}  Lam {r['Lambda_used']:6.2f}/{r['Lambda_touch']:6.2f} "
                          f"d0 {r['d0']:.3f}  k/K {r['k_ratio']:.3f} (pred {r['k_pred'] / r['K_target']:.3f})  "
                          f"n {r['n_contact']:4.1f}  A {r['area_mm2']:5.1f} mm2  rbar {r['rbar_mm']:.3f} mm "
                          f"(hydro {r['rbar_hydro_mm']:.3f})  rbar_tau {r['rbar_tau_mm']:.3f}  {r['wall_s']:.1f} s",
                          flush=True)
    if a.only in (None, "hold"):
        for calib in a.calibs.split(","):
            for dt in (float(v) * 1e-3 for v in a.dts.split(",")):
                for ms in (float(v) for v in a.masses.split(",")):
                    for N in FORCES:
                        r = hold(ms, N, calib, dt)
                        append(out, r)
                        print(f"hold  {calib:8s} tc {TC * 1e3:g} ms dt {dt * 1e3:g} ms (limit {r['dt_limit_ms']:.2f}) m {ms:4.2f} N {N:3.1f}  stable {r['stable']}  "
                              f"N {r['N_meas']:.3f}  k/K {r['k_ratio']:.3f}  rbar {r['rbar_mm']:.3f}  "
                              f"disp {r['tool_disp_mm']:.3f} mm  rot {r['tool_rot_deg']:.2f} deg  "
                              f"vmax {r['tool_vmax_last05_mmps']:.2f} mm/s  warn {r['warnings']}", flush=True)


if __name__ == "__main__":
    main()
