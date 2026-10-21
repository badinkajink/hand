#!/usr/bin/env python3
"""Spherical N52 magnets in place of the 3 x 2 mm cylinders of pad B00 (2026-10-21): the field at the ReSkin 5X chips, the
error of the point-dipole model, the force resolution with spheres of 2-5 mm, and the torque between neighbouring magnets.

Owner, 2026-10-21: "my coworker suggested using spherical n52 magnets ... we can get sizes from 2 to 20mm diameter. does it
make magnetic field sensing easier?"

A uniformly magnetized sphere's field outside it is exactly that of a point dipole at its centre with moment Br V / mu0
(Petruska and Abbott, IEEE Trans. Magn. 49(2):811, 2013); a cylinder's field is a dipole plus higher multipoles that decay
faster with distance. The exact cylinder field here is a converged volume quadrature of point dipoles (fingertip_wrench_model
.cylinder_quadrature at 16 x 64 x 16 points, checked against 24 x 96 x 24). The force resolution is tpu_tip_magnet_sensing
.analyse (contact position and a uniform background unknown) with the gain chosen as v2_tip_fe_checks.sense chooses it. The
sphere rows reuse the cylinders' FE rigid motions at the same centres (sense_fe.jsonl, pad b, depths 6.5-8.5 mm): no FE was
run with spherical inclusions.

    .venv/bin/python scripts/spherical_magnet_check.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_wrench_model as FW  # noqa: E402
import mlx90393_model as MLX  # noqa: E402
import tpu_tip_magnet_sensing as MS  # noqa: E402

OUT = ROOT / "docs/experiments/20261021-spherical_magnets"
SENSE_GAP_B = 3.32 - 6.45            # v2_tip_fe_checks.SENSE_GAP for concept b's pad
MU0 = 4e-7 * math.pi
CASES = ("bm_five_d65", "bm_five_d70", "bm_five_d75", "bm_five_d85", "bm_four_d70", "bm_four_d75")
SPHERES = (2.0, 2.5, 3.0, 4.0, 5.0)
NOISE_MEASURED = np.array([2.57, 1.65, 1.65])   # tip x (chip normal), y, z: GAIN_SEL 1, RES 1, DIG_FILT 5, 2026-10-20


def rows():
    return {r["case"]: r for r in (json.loads(l) for l in open(FW.FE_ROWS) if l.strip())}


def gain_for(bmax_mT):
    """v2_tip_fe_checks.sense's rule: the first (RES, GAIN_SEL) whose range is 1.25 times the largest field component."""
    for rr in (0, 1):
        for gg in range(7, -1, -1):
            rxy, rz = MLX.range_mT(gg, rr)
            if min(rxy, rz, 4.8 * MLX.SENS[gg][rr][0] / MLX.SENS[7][0][0]) >= 1.25 * bmax_mT:
                return gg, rr
    return None, None


def centres_mm(row):
    return np.array([[MS.TD.X_FACE - row["depth_mm"] - t / 2, y, z] for d, t, y, z in row["magnets"]])


def chips_mm():
    return np.array(MLX.board_points(SENSE_GAP_B))


def field_quad(P, C, m_mag, d, t, n, moves=None):
    """Field (T, (n_chips, 3)) of axially magnetized cylinders (axis x) at centres C mm with moment magnitude m_mag by an
    (n_r, n_a, n_z) quadrature; moves = [(t_um, w_mrad)] applies each magnet's rigid motion."""
    off, w = FW.cylinder_quadrature(d, t, *n)
    pts, mom = [], []
    for j, c in enumerate(C):
        mv = np.array([m_mag[j], 0.0, 0.0])
        o = off
        cc = c
        if moves is not None:
            tr, rot = np.array(moves[j][0]) * 1e-3, np.array(moves[j][1]) * 1e-3
            o = off + np.cross(rot[None], off)
            cc = c + tr
            mv = mv + np.cross(rot, mv)
        pts.append(cc[None] + o)
        mom.append(w[:, None] * mv[None])
    return FW.dipole_field(P, np.vstack(pts), np.vstack(mom))


def model_error(row):
    """Point dipole and the estimator's 72-point quadrature against the converged cylinder, at rest and for the 1 N normal
    press at the contact point (the field change), per chip."""
    P, C = chips_mm(), centres_mm(row)
    d, t = row["magnets"][0][:2]
    m = np.full(len(C), FW.BR * math.pi * (d / 2e3) ** 2 * (t / 1e3) / MU0)
    mv = row["per_force"]["1"]["magnets"]
    out = {}
    for tag, n in (("exact", (16, 64, 16)), ("check", (24, 96, 24)), ("quad72", (3, 8, 3)), ("dipole", (1, 1, 1))):
        if tag == "dipole":
            B0 = FW.dipole_field(P, C, m[:, None] * np.array([1.0, 0, 0]))
            mm = [np.array([m[j], 0, 0]) + np.cross(np.array(mv[j][1]) * 1e-3, [m[j], 0, 0]) for j in range(len(C))]
            B1 = FW.dipole_field(P, C + np.array([x[0] for x in mv]) * 1e-3, np.array(mm))
        else:
            B0 = field_quad(P, C, m, d, t, n)
            B1 = field_quad(P, C, m, d, t, n, mv)
        out[tag] = (B0 * 1e6, (B1 - B0) * 1e6)                    # uT
    ex0, ex1 = out["exact"]
    res = dict(case=row["case"], chips_mm=np.round(P, 3).tolist(),
               magnet_to_U1_mm=round(float(np.linalg.norm(C - P[0], axis=1).min()), 3),
               bounding_radius_mm=round(math.hypot(d / 2, t / 2), 3),
               rest_uT=np.round(ex0, 1).tolist(), press_uT_per_N=np.round(ex1, 2).tolist())
    for tag in ("check", "quad72", "dipole"):
        b0, b1 = out[tag]
        res[tag] = dict(rest_err_uT_max=round(float(np.abs(b0 - ex0).max()), 2),
                        rest_err_pct_per_chip=np.round(100 * np.linalg.norm(b0 - ex0, axis=1) / np.linalg.norm(ex0, axis=1), 3).tolist(),
                        press_err_uT_max=round(float(np.abs(b1 - ex1).max()), 3),
                        press_err_pct=round(float(100 * np.linalg.norm(b1 - ex1) / np.linalg.norm(ex1)), 3))
    return res


def sense(row, m_scale=1.0, sphere_d=None, noise=None, centre_shift=0.0, mix=None):
    """tpu_tip_magnet_sensing.analyse of the row with each magnet's moment replaced: a sphere of diameter sphere_d (mm) at the
    cylinder's centre, times m_scale; centre_shift moves every centre along x (mm, + toward the face); mix = (row2, a)
    blends the FE motions as (1 - a) row + a row2. Returns the largest field component, the gain and the force bounds."""
    d, t = row["magnets"][0][:2]
    V = (math.pi / 6 * sphere_d ** 3 if sphere_d else math.pi * (d / 2) ** 2 * t) * 1e-9
    mag = m_scale * FW.BR * V / MU0
    C = centres_mm(row) + np.array([centre_shift, 0, 0])
    if mix is not None:
        row2, a = mix
        row = json.loads(json.dumps(row))
        for p, p2 in zip(row["patches"], row2["patches"]):
            for k in ("normal", "shear_y", "shear_z"):
                p[k] = [[list((1 - a) * np.array(u) + a * np.array(u2)) for u, u2 in zip(mv, mv2)]
                        for mv, mv2 in zip(p[k], p2[k])]
    saved_state, saved_noise = MS.mag_state, MS.NOISE
    MS.mag_state = lambda r, br=FW.BR: [(c * 1e-3, np.array([mag, 0.0, 0.0])) for c in C]
    try:
        ch = MS.chips(row, "board", SENSE_GAP_B)
        bmax = float(np.abs(MS.B0(row, ch)).max() * 1e3)
        g, res = gain_for(bmax)
        if noise is None and g is not None:
            nxy, nz = MLX.noise_uT(g, res, 0, 5)
            MS.NOISE = np.array([nz, nxy, nxy])
        elif noise is not None:
            MS.NOISE = noise
        a = MS.analyse(row, "board", SENSE_GAP_B)
    finally:
        MS.mag_state, MS.NOISE = saved_state, saved_noise
    both = [p["sigF_n_unk_bg"] for p in a["points"]]
    return dict(B_max_component_mT=round(bmax, 2), gain_sel=g, res=res, moment_mA_m2=round(1e3 * mag, 2),
                centre_force_N=round(a["centre"]["sigF_n_unk_bg"], 4), median_force_N=round(float(np.median(both)), 4),
                centre_sig_y_mm=round(a["centre"]["sig_y_mm_bg"], 3), centre_sig_z_mm=round(a["centre"]["sig_z_mm_bg"], 3))


def sense_cylinder_exact(row, noise=None):
    """sense() with each magnet an exact cylinder (the estimator's 72-point quadrature, within 0.01 % of the converged
    field at these chips) in place of tpu_tip_magnet_sensing's point dipole."""
    d, t = row["magnets"][0][:2]
    off, w = FW.cylinder_quadrature(d, t)
    m = FW.BR * math.pi * (d / 2e3) ** 2 * (t / 1e3) / MU0
    C = centres_mm(row)

    def b0(r, ch, br=FW.BR):
        return field_quad(np.array(ch) * 1e3, C, np.full(len(C), m), d, t, (3, 8, 3))

    def signal(r, motions, ch, br=FW.BR):
        B1 = field_quad(np.array(ch) * 1e3, C, np.full(len(C), m), d, t, (3, 8, 3), motions)
        return ((B1 - b0(r, ch)) * 1e6).ravel()

    saved = MS.B0, MS.signal, MS.NOISE
    MS.B0, MS.signal = b0, signal
    try:
        ch = MS.chips(row, "board", SENSE_GAP_B)
        bmax = float(np.abs(b0(row, ch)).max() * 1e3)
        g, res = gain_for(bmax)
        if noise is None and g is not None:
            nxy, nz = MLX.noise_uT(g, res, 0, 5)
            MS.NOISE = np.array([nz, nxy, nxy])
        elif noise is not None:
            MS.NOISE = noise
        a = MS.analyse(row, "board", SENSE_GAP_B)
    finally:
        MS.B0, MS.signal, MS.NOISE = saved
    both = [p["sigF_n_unk_bg"] for p in a["points"]]
    return dict(B_max_component_mT=round(bmax, 2), gain_sel=g, res=res, centre_force_N=round(a["centre"]["sigF_n_unk_bg"], 4),
                median_force_N=round(float(np.median(both)), 4), centre_sig_y_mm=round(a["centre"]["sig_y_mm_bg"], 3),
                centre_sig_z_mm=round(a["centre"]["sig_z_mm_bg"], 3))


def plan_gap_mm(row, D):
    """Least plan-view gap between neighbouring pockets of diameter D (mm), the wall rule being 1.8 mm."""
    yz = np.array([[y, z] for _, _, y, z in row["magnets"]])
    dd = np.linalg.norm(yz[:, None] - yz[None], axis=2)
    return round(float(dd[dd > 0].min() - D), 2)


def neighbour_torque(row, D):
    """Field (mT) at each magnet's centre from the others, all moments +x, and the torque per radian of tilt (N mm/rad) it
    exerts on a sphere of diameter D: the parallel arrangement is an equilibrium the neighbours' field destabilizes."""
    C = centres_mm(row)
    m = FW.BR * math.pi / 6 * (D * 1e-3) ** 3 / MU0
    out = []
    for j in range(len(C)):
        others = np.delete(C, j, 0)
        B = FW.dipole_field(C[j:j + 1], others, np.tile([m, 0.0, 0.0], (len(others), 1)))[0]
        out.append(dict(B_mT=np.round(B * 1e3, 2).tolist(), torque_N_mm_per_rad=round(m * float(np.linalg.norm(B)) * 1e3, 3)))
    return dict(moment_mA_m2=round(1e3 * m, 2), magnets=out)


def main():
    t0 = time.time()
    R = rows()
    out = dict(script="scripts/spherical_magnet_check.py", when=time.strftime("%Y-%m-%d %H:%M"))
    out["model_error"] = [model_error(R[c]) for c in ("bm_five_d75", "bm_five_d85")]
    for me in out["model_error"]:
        print(f"{me['case']}: U1 {me['magnet_to_U1_mm']} mm from the nearest centre ({me['bounding_radius_mm']} mm bounding radius); "
              f"dipole rest error per chip {me['dipole']['rest_err_pct_per_chip']} %, press {me['dipole']['press_err_pct']} %; "
              f"quad72 rest {me['quad72']['rest_err_pct_per_chip']} %, press {me['quad72']['press_err_pct']} %; "
              f"convergence {me['check']['rest_err_uT_max']} uT", flush=True)
    sph = []
    for c in CASES:
        row = R[c]
        base = sense(row)
        rec = dict(case=c, depth_mm=row["depth_mm"], n_magnets=len(row["magnets"]), cylinder_3x2=base,
                   cylinder_3x2_measured_noise=sense(row, noise=NOISE_MEASURED),
                   cylinder_3x2_exact=sense_cylinder_exact(row))
        # the best any magnet volume does at this depth: the moment scaled until the largest component is the lowest
        # gain's limit (GAIN_SEL 0, RES 1: 48.06 mT / 1.25)
        lim = min(min(MLX.range_mT(0, 1)), 4.8 * MLX.SENS[0][1][0] / MLX.SENS[7][0][0]) / 1.25
        rec["range_limited"] = sense(row, m_scale=0.999 * lim / base["B_max_component_mT"])
        for D in SPHERES:
            s = sense(row, sphere_d=D)
            s.update(plan_gap_mm=plan_gap_mm(row, D), face_side_extra_mm=round((D - row["magnets"][0][1]) / 2, 2))
            rec[f"sphere_{D:g}"] = s
        sph.append(rec)
        print(c, json.dumps({k: (v["B_max_component_mT"], v["gain_sel"], v["res"], v["centre_force_N"], v["median_force_N"])
                             for k, v in rec.items() if isinstance(v, dict)}), flush=True)
    out["sensing"] = sph
    # a 3 mm sphere 0.5 mm deeper than the d75 cylinder (its top where the cylinder's top was): motions halfway to d85
    out["sphere_3_top_at_cylinder_top"] = sense(R["bm_five_d75"], sphere_d=3.0, centre_shift=-0.5, mix=(R["bm_five_d85"], 0.5))
    print("3 mm sphere, top at the cylinder's top:", out["sphere_3_top_at_cylinder_top"], flush=True)
    out["neighbour_torque"] = {f"{D:g}": neighbour_torque(R["bm_five_d75"], D) for D in (3.0,)}
    print("torque on 3 mm spheres:", out["neighbour_torque"]["3"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "20261021-spherical_magnet_check.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)} in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
