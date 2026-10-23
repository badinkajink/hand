#!/usr/bin/env python3
"""The calibrated pad as the simulations' 1 mm sphere pads, against a calibration session's press holds: the pads'
normal force on the indenter at the registered positions, in MuJoCo and in the pads' spring law, compared with the
HEX-E's force and with the FE pad's (ur5_tip_calibration_fe.FEPad); and where each model puts the contact.

The pads are those of the FE basis (scripts/fingertip_wrench_fe.py --surface, docs/experiments/
20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz), which are also the plan's pad surface: the
surface points p_i of tip_codesign_geometry's 1 mm pads on the pad's free surface, with normals n_i and areas a_i. Each
pad is a sphere of radius r_s = 0.75 mm centred r_s inside along n_i. Builder v2 (reorient_backends.PAD_BUILDER) gives
it the stiffness K_i = E a_i / h_i, with h_i = n_i . (p_i - c_V) (at least 1 mm) its depth below the volume centroid
c_V. Here c_V is the centroid of the pad's own solid (the FE's node centroid); --centroid shape takes the whole tip
shape's centroid in place of it (reorient_backends.replace_tips for a tcd_* tip).

Spring law: the pads push the indenter (a sphere of radius R, or a rod: the distance to its axis) with
sum_i K_i max(0, r_s + R - d_i), d_i the distance from pad i's centre to the indenter's centre or axis. MuJoCo reaches
K_i through solimp d0_i = 1 - 1 / (tc^2 K_i A), with A the indenter's inverse weight, as replace_tips does. At rest each
contact row then pushes with K_i times its penetration, so the law and the simulator agree at equilibrium.

MuJoCo rig: the pads are fixed in the world and the indenter rides on one slide joint along the hold's contact normal
through its registered position. A generalized force equal to the measured normal force pushes it toward the pad for
0.4 s. The slide's final position is how far the indenter stands outside its registered position: the indentation
that the pads do not account for (delta of ur5_tip_calibration_fe.py, there found by bisection on the FE pad).

Moduli: E_FE fits the pads to the FE pad (s = 1) at the registered positions by least squares of the log force ratio
(both forces are linear in their modulus at a fixed position); E_meas fits them to the HEX-E's force by the median
relative error (as s of ur5_tip_calibration_fe.py); E_TPU = 10 MPa is the simulations' default (reorient_backends).
At E_FE the extra indentation is fitted as a + c F, as the FE's is. --fe-json (ur5_tip_calibration_fe.py's output for the
same holds) adds the recipe M3 of docs/experiments/20261018-fe_contact_calibration: pads at a local modulus E_loc in
series with a spring k_s, fitted so that their depth at the measured force equals the FE pad's (least squares).

Shear (--shear): the indenter on three slide joints, friction mu (condim 3, elliptic cone, impratio as given), starts at
the pads' equilibrium depth and takes the measured force vector, normal and tangential, for 1 s. Its tangential speed
over the last 0.5 s is the pads' creep under the hold's shear; the robot held the real sphere still.

Contact location: the pads' centre is the force-weighted mean of the points where the pad spheres meet the indenter;
the FE's is the force-weighted mean of its loaded surface nodes; the geometric contact is the pad surface's point
nearest the indenter's centre (Geometry.surface_distance).

    .venv/bin/python scripts/ur5_tip_calibration_pads.py <holds.npz> --plan <session>/plan.json [--out pads.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration as C  # noqa: E402
import ur5_tip_calibration_fe as FE  # noqa: E402

RS = 0.75                       # pad sphere radius, mm (reorient_backends.PAD_RS)
TC = 0.01                       # solref time constant, s (reorient_backends.PAD_TR / 2)
E_TPU = 10.0                    # MPa (reorient_backends.E_TPU)
M_IND = 1e-3                    # indenter mass in the rig, kg
T_SETTLE = 0.4                  # s
ROD_HALF = 15.0                 # half-length of the rod's capsule in the rig, mm


class Pads:
    def __init__(self, basis=FE.BASIS, centroid="pad"):
        B = np.load(basis)
        self.S = B["pts"].astype(float)                     # surface points, mm, pad frame P
        self.N = B["normals"].astype(float)
        self.A = B["pad_area"].astype(float)                # mm^2
        self.C = self.S - RS * self.N                       # sphere centres
        if centroid == "pad":
            cV = B["node_centroid"].astype(float)
        else:
            import tip_codesign_geometry as TG
            c = TG.shape(str(B["key"])).centroid()
            cV = np.array([(c[0] - (TG.REACH - 0.0085)) * 1e3, c[1] * 1e3, (TG.Z_MOUNT - c[2]) * 1e3])
        self.cV = cV
        self.h = np.maximum(np.einsum("ij,ij->i", self.N, self.S - cV), 1.0)
        self.k = self.A / self.h                            # K_i / E, mm: K_i in N/mm for E in MPa

    def law(self, c, R, axis=None):
        """Force on the indenter per MPa of E (N, P frame) and its contact centre (mm), indenter centre c (mm) or, with
        axis, a rod whose axis passes through c."""
        v = c - self.C
        if axis is not None:
            a = np.asarray(axis, float) / np.linalg.norm(axis)
            v = v - (v @ a)[:, None] * a[None]
        d = np.linalg.norm(v, axis=1)
        o = RS + R - d
        m = o > 0
        if not m.any():
            return np.zeros(3), None, 0
        u = v[m] / d[m, None]
        f = self.k[m] * o[m]
        P = self.C[m] + RS * u
        return (f[:, None] * u).sum(0), (f[:, None] * P).sum(0) / f.sum(), int(m.sum())


class Rig:
    """The pads fixed in the world, the indenter on one slide joint (MuJoCo, metres)."""

    def __init__(self, pads, R, kind="sphere", mu=1.0, shear=False, impratio=1.0):
        cd = 3 if shear else 1
        geoms = "\n".join(f'<geom name="pad{i}" type="sphere" size="{RS * 1e-3:.6g}" pos="{c[0] * 1e-3:.7f} {c[1] * 1e-3:.7f} '
                          f'{c[2] * 1e-3:.7f}" condim="{cd}" priority="1" friction="{mu:g} 0 0"/>' for i, c in enumerate(pads.C))
        ind = (f'<geom name="ind" type="sphere" size="{R * 1e-3:.6g}" mass="{M_IND}" condim="{cd}"/>' if kind == "sphere" else
               f'<geom name="ind" type="capsule" size="{R * 1e-3:.6g} {ROD_HALF * 1e-3:.6g}" mass="{M_IND}" condim="{cd}"/>')
        joints = ('<joint name="sx" type="slide" axis="1 0 0"/><joint name="sy" type="slide" axis="0 1 0"/>'
                  '<joint name="sz" type="slide" axis="0 0 1"/>' if shear else '<joint name="s" type="slide" axis="1 0 0"/>')
        xml = f"""<mujoco><option timestep="0.001" integrator="implicitfast" gravity="0 0 0" cone="elliptic" impratio="{impratio:g}"/>
<worldbody>{geoms}<body name="ind">{joints}{ind}</body></worldbody></mujoco>"""
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.pads = pads
        self.b = self.m.body("ind").id
        self.g = self.m.geom("ind").id
        if hasattr(self.m, "geom_sameframe"):      # compiled at the body's frame: kinematics would ignore geom_quat
            self.m.geom_sameframe[self.g] = 0
        self.A = float(self.m.body_invweight0[self.b, 0])

    def set_E(self, E):
        K = E * self.pads.k * 1e3                           # N/m
        d0 = 1.0 - 1.0 / (TC * TC * K * self.A)
        if d0.min() < 0.05 or d0.max() > 0.9999:
            raise ValueError(f"pad d0 {d0.min():.4f}..{d0.max():.6f} out of range at E {E:g} MPa")
        self.m.geom_solref[:len(K)] = [TC, 1.0]
        self.m.geom_solimp[:len(K), 0] = d0
        self.m.geom_solimp[:len(K), 1] = d0
        self.m.geom_solimp[:len(K), 2:] = [0.001, 0.5, 2.0]

    def settle(self, c, n, F, axis=None):
        """Indenter at c (mm) free along n, pushed toward the pad with F (N): its final offset along n (mm, outward
        positive) and the summed normal force of its contacts (N)."""
        m, d = self.m, self.d
        m.body_pos[self.b] = np.asarray(c) * 1e-3
        m.jnt_axis[0] = n
        if axis is not None:
            q = np.zeros(4)
            mujoco.mju_quatZ2Vec(q, np.asarray(axis, float) / np.linalg.norm(axis))
            m.geom_quat[self.g] = q
        mujoco.mj_resetData(m, d)
        d.qfrc_applied[0] = -F
        for _ in range(int(T_SETTLE / m.opt.timestep)):
            mujoco.mj_step(m, d)
        fn = 0.0
        for i in range(d.ncon):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            fn += f6[0]
        return float(d.qpos[0]) * 1e3, fn, float(abs(d.qvel[0])) * 1e3


    def creep(self, c, f, n, axis=None, T=1.0):
        """Indenter (three slides) starting at c (mm), pushed with the force vector f (N, on the pad, so on the indenter
        by the robot) for T s: its displacement (mm) and its speed over the last half (mm/s) in the plane normal to the
        contact normal n."""
        m, d = self.m, self.d
        m.body_pos[self.b] = np.asarray(c) * 1e-3
        if axis is not None:
            q = np.zeros(4)
            mujoco.mju_quatZ2Vec(q, np.asarray(axis, float) / np.linalg.norm(axis))
            m.geom_quat[self.g] = q
        mujoco.mj_resetData(m, d)
        d.qfrc_applied[:3] = f
        nT = int(T / m.opt.timestep)
        for k in range(nT):
            mujoco.mj_step(m, d)
            if k == nT // 2 - 1:
                q_half = d.qpos[:3].copy()
        u = np.asarray(n, float) / np.linalg.norm(n)
        dq = (d.qpos[:3] - q_half) * 1e3
        return d.qpos[:3] * 1e3, float(np.linalg.norm(dq - (dq @ u) * u)) / (T / 2)


def line_fit(x, y):
    return FE.line_fit(np.asarray(x), np.asarray(y))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("holds")
    ap.add_argument("--plan", required=True, help="the session's plan.json (indenter and pad surface)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-N", type=float, default=0.2)
    ap.add_argument("--centroid", choices=("pad", "shape"), default="pad")
    ap.add_argument("--mujoco-every", type=int, default=1, help="run the MuJoCo rig on every k-th hold")
    ap.add_argument("--shear", default="", help="impratios of the shear test, comma-separated (empty: no test)")
    ap.add_argument("--mu", type=float, default=1.0, help="pad friction in the shear test")
    ap.add_argument("--fe-json", default=None, help="ur5_tip_calibration_fe.py's output for these holds (fit M3)")
    a = ap.parse_args()
    t0 = time.time()
    H = np.load(a.holds, allow_pickle=False)
    plan = json.loads(Path(a.plan).read_text())
    geo = C.Geometry(plan)
    ind = plan["geometry"]["indenter"]
    R, kind = float(ind["radius"]), ind.get("kind", "sphere")
    idx = np.nonzero((H["kind"] == "press") & np.isfinite(H["f_n"]) & (H["f_n"] > a.min_N))[0]
    AX = [H["axis_P"][i] if kind == "rod" else None for i in idx]
    pads = Pads(centroid=a.centroid)
    fe = FE.FEPad()
    n_h = len(idx)
    meas = H["f_n"][idx]
    nrm = H["n"][idx]
    cP = H["c_P"][idx]
    reg = H["region"][idx]
    G = np.zeros(n_h)                                       # pads' push along n per MPa at the registered centre
    Fe = np.zeros(n_h)                                      # FE's at s = 1
    loc = dict(pads=np.full((n_h, 3), np.nan), fe=np.full((n_h, 3), np.nan), geo=np.full((n_h, 3), np.nan))
    n_pads = np.zeros(n_h, int)
    for j, i in enumerate(idx):
        f, cp, n_pads[j] = pads.law(cP[j], R, AX[j])
        G[j] = float(f @ nrm[j])
        F1, _, cf = fe.force(cP[j], 1.0, R, AX[j], patch=True)
        Fe[j] = float(-F1 @ nrm[j])
        if cp is not None:
            loc["pads"][j] = cp
        if cf is not None:
            loc["fe"][j] = cf
        if kind == "sphere":
            loc["geo"][j] = geo.surface_distance(cP[j])[1]
    ok = (G > 0) & (Fe > 0)
    lnE = np.log(Fe[ok] / G[ok])
    E_fe = float(np.exp(lnE.mean()))
    grid = np.geomspace(0.05, 50.0, 400)
    cost = [np.median(np.abs(E * G[G > 0] / meas[G > 0] - 1.0)) for E in grid]
    E_meas = float(grid[int(np.argmin(cost))])
    out = dict(session_holds=str(a.holds), plan=str(a.plan), indenter=dict(kind=kind, R_mm=R), centroid=a.centroid,
               c_V_mm=pads.cV.tolist(), n_pads=int(len(pads.C)), h_mm=dict(min=float(pads.h.min()), median=float(np.median(pads.h)),
               max=float(pads.h.max())), n=int(n_h), n_both=int(ok.sum()), E_FE_MPa=E_fe,
               pads_vs_fe_log_rms_pct=float(100 * np.sqrt(np.mean((lnE - lnE.mean()) ** 2))),
               pads_vs_fe_by_region={r: dict(n=int((ok & (reg == r)).sum()),
                                             ratio_median=float(np.median(E_fe * G[ok & (reg == r)] / Fe[ok & (reg == r)])))
                                     for r in sorted(set(reg.tolist())) if (ok & (reg == r)).any()},
               E_meas_MPa=E_meas, median_rel_err_meas=float(min(cost)),
               ratio_meas={f"{nm}": dict(E_MPa=E, by_region={r: float(np.median(meas[reg == r] / (E * G[reg == r])))
                                                             for r in sorted(set(reg.tolist()))})
                           for nm, E in (("E_TPU", E_TPU), ("E_FE", E_fe), ("E_meas", E_meas))})
    # depth law of each model along n at the measured force: bisection on the spring law (pads) as for the FE in fe.py
    def delta_law(E, j):
        c, n, f_m = cP[j], nrm[j], meas[j]
        lo, hi = -1.5, float(H["depth"][idx[j]]) + 0.5
        if E * float(pads.law(c + lo * n, R, AX[j])[0] @ n) < f_m:
            return np.nan
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if E * float(pads.law(c + mid * n, R, AX[j])[0] @ n) > f_m:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    rig = Rig(pads, R, kind)
    out["rig"] = dict(m_ind_kg=M_IND, invweight=rig.A, tc_s=TC, settle_s=T_SETTLE, timestep_s=float(rig.m.opt.timestep))
    for nm, E in (("E_FE", E_fe), ("E_meas", E_meas), ("E_TPU", E_TPU)):
        dl = np.array([delta_law(E, j) for j in range(n_h)])
        rig.set_E(E)
        dm, fm, vm = np.full(n_h, np.nan), np.full(n_h, np.nan), np.full(n_h, np.nan)
        t1 = time.time()
        for j in range(0, n_h, a.mujoco_every):
            dm[j], fm[j], vm[j] = rig.settle(cP[j], nrm[j], meas[j], AX[j])
        okd = np.isfinite(dl) & np.isfinite(dm)
        okl = np.isfinite(dl)
        fit = line_fit(meas[okl], dl[okl])
        pred = fit["fit"][0] + fit["fit"][1] * meas[okl]
        out[f"delta_{nm}"] = dict(
            E_MPa=E, law=dict(n=int(okl.sum()), median_mm=float(np.median(dl[okl])), fit=fit,
                              rms_mm=float(np.sqrt(np.mean((pred - dl[okl]) ** 2))),
                              by_region={r: line_fit(meas[okl & (reg == r)], dl[okl & (reg == r)])
                                         for r in sorted(set(reg.tolist())) if (okl & (reg == r)).sum() >= 3}),
            mujoco=dict(n=int(okd.sum()), law_minus_mujoco_um=dict(
                median=float(np.median(np.abs(dl[okd] - dm[okd])) * 1e3), max=float(np.max(np.abs(dl[okd] - dm[okd])) * 1e3)),
                force_over_measured=dict(median=float(np.median(fm[okd] / meas[okd])),
                                         max_dev=float(np.max(np.abs(fm[okd] / meas[okd] - 1.0)))),
                speed_end_mm_s=float(np.nanmax(vm)), wall_s=time.time() - t1),
            per_hold=dict(law_mm=[None if not np.isfinite(x) else round(float(x), 5) for x in dl],
                          mujoco_mm=[None if not np.isfinite(x) else round(float(x), 5) for x in dm]))
        print(f"{nm} {E:.3f} MPa: extra indentation (spring law) {fit['fit'][0]:.3f} mm + {fit['fit'][1]:.3f} mm/N x F, rms "
              f"{out[f'delta_{nm}']['law']['rms_mm'] * 1e3:.0f} um over {okl.sum()}; MuJoCo against the law median "
              f"{out[f'delta_{nm}']['mujoco']['law_minus_mujoco_um']['median']:.1f} um, max "
              f"{out[f'delta_{nm}']['mujoco']['law_minus_mujoco_um']['max']:.1f} um; contact force over measured "
              f"{out[f'delta_{nm}']['mujoco']['force_over_measured']['median']:.4f} ({time.time() - t1:.0f} s)", flush=True)
    if a.fe_json:
        fj = json.loads(Path(a.fe_json).read_text())
        if fj["holds"]["id"] != H["id"][idx].tolist():
            raise SystemExit("the FE output is for other holds")
        dfe = np.array([np.nan if x is None else x for x in fj["holds"]["delta_mm"]], float)
        best = None
        for E in np.geomspace(2.0, 200.0, 121):
            dp = np.array([delta_law(E, j) for j in range(n_h)])
            okm = np.isfinite(dp) & np.isfinite(dfe)
            y, x = dp[okm] - dfe[okm], meas[okm]               # = F / k_s for the pads in series with k_s
            cs = max(float(x @ y / (x @ x)), 0.0)
            rms = float(np.sqrt(np.mean((y - cs * x) ** 2)))
            if best is None or rms < best[2]:
                best = (float(E), cs, rms, int(okm.sum()), dp)
        E3, c3, r3, n3, dp3 = best
        okm = np.isfinite(dp3) & np.isfinite(dfe)
        res = (dp3[okm] - dfe[okm]) - c3 * meas[okm]
        out["M3"] = dict(E_loc_MPa=E3, k_s_N_mm=(1.0 / c3 if c3 > 0 else None), rms_mm=r3, n=n3,
                         by_level={f"{L:g}": float(np.sqrt(np.mean(res[H["level_N"][idx][okm] == L] ** 2)))
                                   for L in sorted(set(H["level_N"][idx][okm].tolist()))})
        print(f"M3 against the FE pad: E_loc {E3:.2f} MPa in series with k_s {1 / c3 if c3 > 0 else float('inf'):.2f} N/mm, depth rms "
              f"{r3 * 1e3:.0f} um over {n3} (one modulus E_FE: see the by-level force ratios)")
    lv = H["level_N"][idx]
    out["by_level"] = {f"{L:g}": dict(n=int((lv == L).sum()),
                                      pads_over_fe=float(np.median(E_fe * G[ok & (lv == L)] / Fe[ok & (lv == L)])) if (ok & (lv == L)).any() else None,
                                      meas_over_pads_E_meas=float(np.median(meas[lv == L] / (E_meas * G[lv == L]))),
                                      meas_over_pads_E_FE=float(np.median(meas[lv == L] / (E_fe * G[lv == L]))))
                       for L in sorted(set(lv[np.isfinite(lv)].tolist()))}
    print("by level (N): pads over FE at E_FE; measured over pads at E_meas, at E_FE")
    for L, v in out["by_level"].items():
        print(f"  {L:>4s} n={v['n']:3d}: {v['pads_over_fe']:.3f}; {v['meas_over_pads_E_meas']:.3f}, {v['meas_over_pads_E_FE']:.3f}")
    if a.shear:
        fv = H["f"][idx]
        ft = np.linalg.norm(fv - (np.einsum("ij,ij->i", fv, nrm))[:, None] * nrm, axis=1)
        dur = H["n_rec"][idx] / 100.0 if "n_rec" in H.files else np.full(n_h, np.nan)
        out["shear"] = dict(mu=a.mu, tangential_N=dict(median=float(np.median(ft)), p90=float(np.percentile(ft, 90))),
                            tangential_over_normal=dict(median=float(np.median(ft / meas)), p90=float(np.percentile(ft / meas, 90)),
                                                        max=float(np.max(ft / meas))),
                            hold_s=dict(median=float(np.nanmedian(dur))), runs={})
        print(f"shear of the holds: tangential {np.median(ft):.3f} N median, {np.percentile(ft, 90):.3f} p90; over normal "
              f"{np.median(ft / meas):.3f} median, {np.max(ft / meas):.3f} max; hold {np.nanmedian(dur):.1f} s median")
        for ir in (float(x) for x in a.shear.split(",") if x):  # noqa: B007
            for nm, E in (("E_FE", E_fe), ("E_meas", E_meas)):
                rs3 = Rig(pads, R, kind, mu=a.mu, shear=True, impratio=ir)
                rs3.set_E(E)
                dl = out[f"delta_{nm}"]["per_hold"]["law_mm"]
                v = np.full(n_h, np.nan)
                t1 = time.time()
                for j in range(0, n_h, a.mujoco_every):
                    if dl[j] is None:
                        continue
                    v[j] = rs3.creep(cP[j] + dl[j] * nrm[j], H["f"][idx[j]], nrm[j], AX[j])[1]
                okv = np.isfinite(v)
                key = f"impratio{ir:g}_{nm}"
                out["shear"]["runs"][key] = dict(impratio=ir, E_MPa=E, n=int(okv.sum()), creep_mm_s=dict(
                    median=float(np.median(v[okv])), p90=float(np.percentile(v[okv], 90)), max=float(v[okv].max())),
                    per_hold=[None if not np.isfinite(x) else round(float(x), 4) for x in v])
                print(f"  impratio {ir:g}, {nm} {E:.3f} MPa: creep under the measured shear {np.median(v[okv]):.3f} mm/s median, "
                      f"{np.percentile(v[okv], 90):.3f} p90, {v[okv].max():.3f} max over {okv.sum()} holds ({time.time() - t1:.0f} s)",
                      flush=True)
    dd = {}
    for p, q in (("pads", "fe"), ("pads", "geo"), ("fe", "geo")):
        e = np.linalg.norm(loc[p] - loc[q], axis=1)
        e = e[np.isfinite(e)]
        if len(e):
            dd[f"{p}_vs_{q}"] = dict(n=int(len(e)), median_mm=float(np.median(e)), p90_mm=float(np.percentile(e, 90)),
                                     max_mm=float(e.max()),
                                     by_region={r: float(np.nanmedian(np.linalg.norm(loc[p][reg == r] - loc[q][reg == r], axis=1)))
                                                for r in sorted(set(reg.tolist()))})
    out["location"] = dd
    out["holds"] = dict(id=H["id"][idx].tolist(), region=reg.tolist(), level_N=H["level_N"][idx].tolist(), f_n=meas.tolist(),
                        depth_mm=H["depth"][idx].tolist(), pads_N_per_MPa=G.tolist(), fe_s1_N=Fe.tolist(), n_pads=n_pads.tolist(),
                        loc_pads=np.round(loc["pads"], 4).tolist(), loc_fe=np.round(loc["fe"], 4).tolist(),
                        loc_geo=np.round(loc["geo"], 4).tolist())
    print(f"{n_h} press holds over {a.min_N} N; pads {len(pads.C)}, h {pads.h.min():.2f}-{pads.h.max():.2f} mm (median "
          f"{np.median(pads.h):.2f}), c_V {np.round(pads.cV, 2).tolist()} ({a.centroid})")
    print(f"pads against the FE pad (s = 1) at the registered positions: E_FE {E_fe:.3f} MPa, log rms {out['pads_vs_fe_log_rms_pct']:.1f} %; "
          + ", ".join(f"{r} {v['ratio_median']:.3f}" for r, v in out["pads_vs_fe_by_region"].items()))
    print(f"pads against the HEX-E: E_meas {E_meas:.3f} MPa (median force error {min(cost) * 100:.1f} %); measured over pads "
          + "; ".join(f"{nm} " + ", ".join(f"{r} {v:.3f}" for r, v in d["by_region"].items()) for nm, d in out["ratio_meas"].items()))
    for k, v in dd.items():
        print(f"contact location {k}: median {v['median_mm'] * 1e3:.0f} um, p90 {v['p90_mm'] * 1e3:.0f} um, max {v['max_mm'] * 1e3:.0f} um; "
              + ", ".join(f"{r} {x * 1e3:.0f}" for r, x in v["by_region"].items()))
    print(f"{time.time() - t0:.0f} s")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
