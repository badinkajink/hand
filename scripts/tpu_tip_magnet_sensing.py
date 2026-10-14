#!/usr/bin/env python3
"""What one or four magnets let an MLX90393 array measure (2026-10-13): Cramer-Rao bounds on contact force and position
from the finite-element rows of scripts/tpu_tip_magnet_layouts.py.

Field: each magnet a point dipole (Br 1.45 T for N52, axis normal to the face, north toward it), moved by the rigid
motion the finite elements give per newton (translation and small rotation). Chips: MLX90393 sensing points `gap` mm
behind the back face; noise per 100 Hz sample 2.3 uT on the chip's in-plane axes and 3.5 uT normal to it
(scripts/tpu_tip_print_plan.py). Every chip sees every magnet.

Estimates at a load state of 1 N normal at a patch position (y, z), no shear:
  known position   force only: normal, shear across (y) and along (z) the finger;
  unknown position the same three forces and the position (y, z), from the change of the field with position
                   (central differences over the patch grid);
  bias             a one-position estimator (it assumes the contact at the contact point) reads F_hat/F = S(y,z).S0/|S0|^2.
The bound is the inverse Fisher information J^T Sigma^-1 J; a near-singular J (one magnet, position unknown) gives a large
bound, which is the result and not a numerical failure.

    .venv/bin/python scripts/tpu_tip_magnet_sensing.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_design as TD  # noqa: E402

D = ROOT / "docs/experiments/20261013-magnet_layouts"
ROWS = D / "fem.jsonl"
import mlx90393_model as MLX  # noqa: E402
_ST = MLX.setting("high_gain_141Hz")        # GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5: a T+XYZ burst every 7.1 ms
NOISE = np.array([_ST["noise_z_uT"], _ST["noise_xy_uT"], _ST["noise_xy_uT"]])   # uT per sample, tip x (chip normal), y, z
B_RANGE_MT = 4.8
BR = 1.45


def rows():
    return {r["layout"]: r for r in (json.loads(l) for l in open(ROWS))} if ROWS.exists() else {}


def dipole(ps, pm, m):
    r = ps - pm
    d = np.linalg.norm(r)
    h = r / d
    return 1e-7 * (3 * h * (h @ m) - m) / d ** 3


def mag_state(row, br=BR):
    """Centre (m) and moment (A m^2) of each magnet."""
    out = []
    for d, t, y, z in row["magnets"]:
        xc = TD.X_FACE - row["depth_mm"] - t / 2
        V = math.pi * (d / 2e3) ** 2 * (t / 1e3)
        out.append((np.array([xc, y, z]) * 1e-3, np.array([br * V / (4e-7 * math.pi), 0.0, 0.0])))
    return out


def chips(row, mode, gap):
    xs = TD.X_BACK - gap
    if mode == "under":
        pts = [(xs, y, z) for _, _, y, z in row["magnets"]]
    elif mode == "grid4":
        pts = [(xs, sy * 3.0, TD.Z_C + sz * 3.5) for sy in (-1, 1) for sz in (-1, 1)]
    else:
        pts = [(xs, 0.0, TD.Z_C)]
    return [np.array(p) * 1e-3 for p in pts]


def B0(row, ch, br=BR):
    M = mag_state(row, br)
    return np.array([sum(dipole(c, pm, m) for pm, m in M) for c in ch])        # T, (n_chips, 3)


def signal(row, motions, ch, br=BR):
    """Field change per newton at every chip (uT), stacked (3 n_chips,), for one load's magnet motions."""
    M = mag_state(row, br)
    out = []
    for c in ch:
        s = np.zeros(3)
        for (pm, m), (t_um, w_mrad) in zip(M, motions):
            t, w = np.array(t_um) * 1e-6, np.array(w_mrad) * 1e-3
            s += dipole(c, pm + t, m + np.cross(w, m)) - dipole(c, pm, m)
        out.append(s * 1e6)
    return np.concatenate(out)


def patch_grid(row):
    ys = sorted({p["y"] for p in row["patches"]})
    zs = sorted({round(p["z"], 6) for p in row["patches"]})
    G = {(p["y"], round(p["z"], 6)): p for p in row["patches"]}
    return ys, zs, G


def deriv(vals, coords, i):
    """Central difference inside the grid, one-sided at its edges."""
    if 0 < i < len(coords) - 1:
        return (vals[i + 1] - vals[i - 1]) / (coords[i + 1] - coords[i - 1])
    if i == 0:
        return (vals[1] - vals[0]) / (coords[1] - coords[0])
    return (vals[i] - vals[i - 1]) / (coords[i] - coords[i - 1])


def crb(J, sig):
    """Square roots of the diagonal of the inverse Fisher information; infinite when the readings cannot separate the
    parameters (fewer readings than parameters, or a singular value under 1e-8 of the largest)."""
    Jw = J / sig[:, None]
    sv = np.linalg.svd(Jw, compute_uv=False)
    if Jw.shape[0] < Jw.shape[1] or sv[-1] < 1e-8 * sv[0]:
        return np.full(J.shape[1], np.inf)
    return np.sqrt(np.diag(np.linalg.inv(Jw.T @ Jw)))


def analyse(row, mode="under", gap=1.0, br=BR):
    ch = chips(row, mode, gap)
    sig = np.tile(NOISE, len(ch))
    U = np.tile(np.eye(3), (len(ch), 1))                  # a uniform background field (Earth's), the same at every chip
    ys, zs, G = patch_grid(row)
    S = {k: np.array([[signal(row, G[(y, z)][k], ch, br) for z in zs] for y in ys]) for k in ("normal", "shear_y", "shear_z")}
    b0 = np.linalg.norm(B0(row, ch, br), axis=1) * 1e3
    res = []
    i0, j0 = ys.index(0.0), min(range(len(zs)), key=lambda j: abs(zs[j] - TD.Z_C))
    S0 = S["normal"][i0, j0]
    for i, y in enumerate(ys):
        for j, z in enumerate(zs):
            Sn, Sy, Sz = S["normal"][i, j], S["shear_y"][i, j], S["shear_z"][i, j]
            dy = deriv(S["normal"][:, j], ys, i)
            dz = deriv(S["normal"][i, :], zs, j)
            known = crb(np.stack([Sn, Sy, Sz], 1), sig)
            unknown = crb(np.stack([Sn, Sy, Sz, dy, dz], 1), sig)
            backg = crb(np.concatenate([np.stack([Sn, Sy, Sz], 1), U], 1), sig)
            res.append(dict(y=y, z=z, sigF_n=known[0], sigF_y=known[1], sigF_z=known[2], sigF_n_unk=unknown[0],
                            sigF_y_unk=unknown[1], sigF_z_unk=unknown[2], sig_y_mm=unknown[3], sig_z_mm=unknown[4],
                            sigF_n_bg=backg[0],
                            bias=float(Sn @ S0 / (S0 @ S0)), signal_n_uT=float(np.linalg.norm(Sn))))
    return dict(layout=row["layout"], chips=mode, n_chips=len(ch), gap_mm=gap, B0_max_mT=float(b0.max()),
                saturated=bool(b0.max() > B_RANGE_MT), points=res,
                centre=[r for r in res if r["y"] == 0.0 and abs(r["z"] - TD.Z_C) < 1e-6][0])


def table():
    R = rows()
    out = []
    for name in ("one_d4.76x1.59_d5", "four_d3.18x1.59_d5", "four_d3x1_d5", "four_d3.18x1.59_d3", "four_d4.76x1.59_d5"):
        if name not in R:
            continue
        for mode in (("under", "grid4") if name.startswith("one") else ("under",)):
            a = analyse(R[name], mode)
            pts = a["points"]
            med = lambda k: float(np.median([p[k] for p in pts]))  # noqa: E731
            c = a["centre"]
            bias = [p["bias"] for p in pts]
            out.append(dict(layout=name, chips=mode, n_chips=a["n_chips"], B0_max_mT=a["B0_max_mT"], saturated=a["saturated"],
                            centre=c, median=dict((k, med(k)) for k in ("sigF_n", "sigF_y", "sigF_z", "sigF_n_unk", "sig_y_mm",
                                                                         "sig_z_mm", "signal_n_uT", "sigF_n_bg")),
                            bias_range=[min(bias), max(bias)], E_best_MPa=R[name]["E_best_MPa"],
                            approach_um=R[name]["per_force"]["1"]["delta_um"]))
    return out


if __name__ == "__main__":
    for r in table():
        c, m = r["centre"], r["median"]
        print(f"{r['layout']:22s} chips {r['chips']:5s} x{r['n_chips']}  B0max {r['B0_max_mT']:.2f} mT{' SAT' if r['saturated'] else ''}  "
              f"E* {r['E_best_MPa']:.2f}  | centre: sigFn {c['sigF_n']:.3f} sigFy {c['sigF_y']:.3f} sigFz {c['sigF_z']:.3f} N; "
              f"unknown pos: sigFn {c['sigF_n_unk']:.3f} N sig_y {c['sig_y_mm']:.2f} sig_z {c['sig_z_mm']:.2f} mm | "
              f"median sigFn {m['sigF_n']:.3f} unk {m['sigF_n_unk']:.3f}, sig_z {m['sig_z_mm']:.2f} | bg {c['sigF_n_bg']:.3f} | bias {r['bias_range'][0]:.2f}-{r['bias_range'][1]:.2f}")
