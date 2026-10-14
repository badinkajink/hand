#!/usr/bin/env python3
"""MLX90393 readout model from the Melexis datasheet (revision 012, 18 Nov 2025), and the force each printed fingertip
resolves with it (2026-10-13).

Owner, 2026-10-13: "can we revisit the mlx driver code and whatever analysis exists and see if the drastically reduced
sensitivity in the gyroid is potentially still usable? ... how did you make those computations? is that just the material
analysis?"

Datasheet quantities used (logs/20261013-mlx/mlx90393.pdf):
  conversion time per axis   TCONVM = 67 + 64 * 2^OSR * (2 + 2^DIG_FILT) us; temperature TCONVT = 67 + 192 * 2^OSR2 us;
                             a burst T+XYZ without delay takes 3 TCONVM + TCONVT + TCONV_END (100 us)  (section 12)
  noise                      Figures 2 and 3: RMS noise of X/Y and Z against the total XYZ conversion time, GAIN_SEL 7,
                             digitized per (OSR, DIG_FILT) below, 1 mGauss = 0.1 uT  (section 13.1)
  resolution                 Table 17: uT per LSB by GAIN_SEL and RES, HALLCONF 0xC, 25 degC
  range                      at least +-4.8 mT at the highest gain (note 8); 16-bit output; saturation onset 50 mT
  thermal                    offset drift < +-1000 LSB from 25 degC over the temperature range, characterized at GAIN_SEL 7,
                             RES 0 (note 7); sensitivity drift +-3 %
The datasheet gives noise at GAIN_SEL 7 only; at a lower gain this model keeps the same field noise and adds the
quantization LSB/sqrt(12), which makes it a lower bound there.

eFlesh's firmware (~/wxie_workspace/eFlesh/arduino/5X_eflesh_stream/5X_eflesh_stream.ino with the bundled
arduino-MLX90393 library): HALLCONF 0xC, GAIN_SEL 1, RES 2 on all axes, OSR 0, DIG_FILT 4, temperature compensation off,
burst T+XYZ, five chips at 0x0C-0x10 read in turn at 400 kHz I2C, each chip in its own burst.

The board (owner, 2026-10-13: "we are using the 5-magnetometer board from anyskin/visuoskin, reused in eflesh"): ReSkin's
5X board, which eFlesh's README names; its Gerbers are raunaqbhirangi/reskin_sensor circuits/5X_ReSkin.Zip. The pick-and-place
file places five MLX90393ELW-ABA-011 in a plus, U1 at the centre of a 20 x 20 mm square and U2-U5 7.00 mm from it on the two
axes, with a 10 x 9 mm tab carrying the four-pin connector (board 20 x 29 mm, rigid FR-4 0.6 mm). Datasheet Figure 26 puts
each chip's sensitive spot at the package centre, 0.42 +- 0.10 mm under its top face. The owner's printed v17 eFlesh finger
(~/wxie_workspace/fingers/merged_finger_v17.3mf) held the board in a 2 mm recess on the bed side of finger_back_v5_2.stl, under
2.5 mm of solid TPU.

    .venv/bin/python scripts/mlx90393_model.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SENS = {  # Table 17, uT/LSB (XY, Z) for RES 0..3, HALLCONF 0xC, 25 degC
    0: [(0.751, 1.210), (1.502, 2.420), (3.004, 4.840), (6.009, 9.680)],
    1: [(0.601, 0.968), (1.202, 1.936), (2.403, 3.872), (4.840, 7.744)],
    2: [(0.451, 0.726), (0.901, 1.452), (1.803, 2.904), (3.605, 5.808)],
    3: [(0.376, 0.605), (0.751, 1.210), (1.502, 2.420), (3.004, 4.840)],
    4: [(0.300, 0.484), (0.601, 0.968), (1.202, 1.936), (2.403, 3.872)],
    5: [(0.250, 0.403), (0.501, 0.807), (1.001, 1.613), (2.003, 3.227)],
    6: [(0.200, 0.323), (0.401, 0.645), (0.801, 1.291), (1.602, 2.581)],
    7: [(0.150, 0.242), (0.300, 0.484), (0.601, 0.968), (1.202, 1.936)],
}
# Figures 2 and 3, read off the plots: (OSR, DIG_FILT) -> (X/Y, Z) RMS noise in mGauss at GAIN_SEL 7
NOISE_MG = {
    (0, 0): (51, 77), (0, 1): (41, 61.5), (0, 2): (30.5, 46.5), (0, 3): (23, 35.5), (0, 4): (16, 25), (0, 5): (11.5, 17.5),
    (0, 6): (8, 12.5),
    (1, 0): (34.5, 52.5), (1, 1): (28, 44.5), (1, 2): (22, 33.5), (1, 3): (15.5, 24.5), (1, 4): (11, 18), (1, 5): (8, 12),
    (1, 6): (6, 9), (1, 7): (4, 6.5),
    (2, 0): (24.5, 37.5), (2, 1): (20.5, 30.5), (2, 2): (15.5, 24), (2, 3): (11, 17), (2, 4): (8, 12), (2, 5): (5.5, 9),
    (2, 6): (4, 6.5), (2, 7): (3, 4.5),
    (3, 0): (17, 25.5), (3, 1): (14, 22.5), (3, 2): (10.5, 16.5), (3, 3): (8, 12), (3, 4): (6, 9), (3, 5): (4, 6.5),
    (3, 6): (3, 5), (3, 7): (2, 4),
}
OFFSET_DRIFT_LSB = 1000          # band from 25 degC over the full range, GAIN_SEL 7, RES 0
T_RANGE_C = 125.0                # -40 to 85 degC

SETTINGS = {   # name: (GAIN_SEL, RES, OSR, DIG_FILT)
    "eflesh_firmware": (1, 2, 0, 4),
    "page_until_now": (7, 0, 0, 3),
    "high_gain_141Hz": (7, 0, 0, 5),
    "high_gain_75Hz": (7, 0, 0, 6),
}
SETTING_LBL = {"eflesh_firmware": "Firmware of eFlesh: GAIN_SEL 1, RES 2, OSR 0, DIG_FILT 4",
               "page_until_now": "Noise the pages used until now: GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 3",
               "high_gain_141Hz": "GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5",
               "high_gain_75Hz": "GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 6"}


def t_axis_us(osr, df):
    return 67 + 64 * 2 ** osr * (2 + 2 ** df)


def burst_ms(osr, df, osr2=0, axes=3):
    return (axes * t_axis_us(osr, df) + 67 + 192 * 2 ** osr2 + 100) / 1000.0


def lsb_uT(gain, res):
    return SENS[gain][res]


def noise_uT(gain, res, osr, df):
    """Per-sample RMS noise (X/Y, Z) in uT: the datasheet's field noise and the quantization of one LSB."""
    nxy, nz = (v / 10.0 for v in NOISE_MG[(osr, df)])
    lxy, lz = lsb_uT(gain, res)
    return math.hypot(nxy, lxy / math.sqrt(12)), math.hypot(nz, lz / math.sqrt(12))


def range_mT(gain, res):
    lxy, lz = lsb_uT(gain, res)
    return min(50.0, 32768 * lxy / 1000), min(50.0, 32768 * lz / 1000)


def setting(name):
    g, r, o, d = SETTINGS[name]
    nxy, nz = noise_uT(g, r, o, d)
    T = burst_ms(o, d)
    rxy, rz = range_mT(g, r)
    lxy, lz = lsb_uT(g, r)
    return dict(name=name, label=SETTING_LBL[name], gain=g, res=r, osr=o, dig_filt=d, burst_ms=T, rate_Hz=1000.0 / T,
                noise_xy_uT=nxy, noise_z_uT=nz, lsb_xy_uT=lxy, lsb_z_uT=lz, range_xy_mT=rxy, range_z_mT=rz,
                offset_drift_z_uT_per_10C=10 * OFFSET_DRIFT_LSB * SENS[7][0][1] / T_RANGE_C)


# --------------------------------------------------------------------------------- the fingertips

def tips():
    """Magnet motion per newton of normal load at the contact point for each tip (4.76 x 1.59 mm N52, top 5 mm deep)."""
    out = []
    M = [json.loads(l) for l in open(ROOT / "docs/experiments/20261009-tpu_tip_fem/magnet_runs.jsonl")]
    for rec, lbl in (("w95_20", "Gyroid 95A 20 %, homogenized model"), ("w95_10", "Gyroid 95A 10 %, homogenized model")):
        r = next(x for x in M if x["recipe"] == rec and x["magnet"] == "d4.76x1.59" and x["depth_mm"] == 5.0)
        out.append(dict(tip=f"{rec}_model", label=lbl, t_um=r["per_force"]["1"]["magnet_t_um"], w_mrad=r["per_force"]["1"]["magnet_w_mrad"],
                        approach_um=r["per_force"]["1"]["delta_um"], E_star=r["E_best_MPa"]))
    F = {}
    for l in open(ROOT / "docs/experiments/20261013-eflesh_printability/fem.jsonl"):
        r = json.loads(l)
        F[r["tip"]] = r
    for k, lbl in (("gyroid20", "Gyroid 95A 20 %, as printed"), ("eflesh_top", "eFlesh lattice, palmar skin, as printed"),
                   ("eflesh_open", "eFlesh lattice, open, as printed")):
        r = F[k]
        out.append(dict(tip=k, label=lbl, t_um=r["per_force"]["1"]["magnet_t_um"], w_mrad=r["per_force"]["1"]["magnet_w_mrad"],
                        approach_um=r["per_force"]["1"]["delta_um"], E_star=r["E_best_MPa"]))
    return out


def dipole(ps, pm, m):
    r = ps - pm
    d = np.linalg.norm(r)
    h = r / d
    return 1e-7 * (3 * h * (h @ m) - m) / d ** 3


def field_terms(t_um, w_mrad, gap=1.0, depth=5.0, d=4.76, t=1.59, br=1.45):
    """Static field at the chip (uT, chip axes: z along the tip's normal) and its change per newton (uT/N)."""
    import tpu_tip_design as TD
    pm = np.array([TD.X_FACE - depth - t / 2, 0.0, TD.Z_C]) * 1e-3
    ps = np.array([TD.X_BACK - gap, 0.0, TD.Z_C]) * 1e-3
    m = np.array([br * math.pi * (d / 2e3) ** 2 * (t / 1e3) / (4e-7 * math.pi), 0.0, 0.0])
    B0 = dipole(ps, pm, m)
    tt, ww = np.array(t_um) * 1e-6, np.array(w_mrad) * 1e-3
    S = (dipole(ps, pm + tt, m + np.cross(ww, m)) - B0) * 1e6
    return B0 * 1e6, S


def disc_axis_mT(z_mm, d=4.76, t=1.59, br=1.45):
    """On-axis field of a uniformly magnetized disc at z_mm from its centre, for the dipole's error at U1."""
    R, h = d / 2, t / 2
    return 1e3 * br / 2 * ((z_mm + h) / math.hypot(R, z_mm + h) - (z_mm - h) / math.hypot(R, z_mm - h))


def neighbour_field_uT(dist_mm, d=4.76, t=1.59, br=1.45):
    """On-axis field of the same magnet at dist_mm (a facing fingertip's magnet seen by this tip's chip)."""
    m = br * math.pi * (d / 2e3) ** 2 * (t / 1e3) / (4e-7 * math.pi)
    return 2e-7 * m / (dist_mm * 1e-3) ** 3 * 1e6


# --------------------------------------------------------------------------------- the five-chip board

BOARD_CHIPS = {"U1": (0.0, 0.0), "U2": (0.0, 7.0), "U3": (7.0, 0.0), "U4": (0.0, -7.0), "U5": (-7.0, 0.05)}   # mm from U1
BOARD_OUTLINE = dict(u=(-10.0, 10.0), v=(-10.10, 10.05), tab_u=(-5.0, 5.0), tab_v=(-19.10, -10.10))   # PCB.Outline, mm from U1
BOARD_HOLE = (-8.0, 8.05, 1.2)   # mounting hole: centre (u, v) and radius, mm
BOARD_T_MM, PKG_H_MM, SPOT_MM = 0.6, 0.9, 0.42
V17_COVER_MM = 2.5
GAPS = {"touch": SPOT_MM, "v17": V17_COVER_MM + SPOT_MM}   # sensing plane behind the tip's back face, mm
GAP_LBL = {"touch": "chips against the back face", "v17": "under 2.5&#8202;mm of solid TPU, as in v17"}
DRIFT_LSB_PER_10C = 10 * OFFSET_DRIFT_LSB / T_RANGE_C


def board_points(gap, z0=None):
    """Sensing points (mm, tip frame) of the five chips, U1 first: board u along the tip's width y, v along its length z with U2
    toward the tip end and the connector tab toward the mount, U1 under the contact point, `gap` mm behind the back face."""
    import tpu_tip_design as TD
    z0 = TD.Z_C if z0 is None else z0
    return [np.array([TD.X_BACK - gap, u, z0 + v]) for u, v in BOARD_CHIPS.values()]


def board_terms(t_um, w_mrad, gap, depth=5.0, d=4.76, t=1.59, br=1.45):
    """Static field (uT) and its change per newton (uT/N) at the five chips, rows in BOARD_CHIPS order, columns tip x, y, z
    (chip Z, X, Y)."""
    import tpu_tip_design as TD
    pm = np.array([TD.X_FACE - depth - t / 2, 0.0, TD.Z_C]) * 1e-3
    m = np.array([br * math.pi * (d / 2e3) ** 2 * (t / 1e3) / (4e-7 * math.pi), 0.0, 0.0])
    tt, ww = np.array(t_um) * 1e-6, np.array(w_mrad) * 1e-3
    B0, S = [], []
    for p in board_points(gap):
        ps = p * 1e-3
        b = dipole(ps, pm, m)
        B0.append(b * 1e6)
        S.append((dipole(ps, pm + tt, m + np.cross(ww, m)) - b) * 1e6)
    return np.array(B0), np.array(S)


def noise_vec(st, n):
    return np.tile([st["noise_z_uT"], st["noise_xy_uT"], st["noise_xy_uT"]], n)


def background_basis(points_mm, kind="uniform"):
    """Columns of a uniform field (3) at n chips and, for "gradient", of a field gradient that is free of curl and divergence
    (5) about U1."""
    n = len(points_mm)
    cols = [np.tile(np.eye(3)[k], n) for k in range(3)]
    if kind == "gradient":
        c = points_mm[0]
        Gs = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            G = np.zeros((3, 3))
            G[i, j] = G[j, i] = 1.0
            Gs.append(G)
        for i, j in ((0, 1), (1, 2)):
            G = np.zeros((3, 3))
            G[i, i], G[j, j] = 1.0, -1.0
            Gs.append(G)
        cols += [np.concatenate([G @ (p - c) for p in points_mm]) for G in Gs]
    return np.array(cols).T


def readout(S, st, background=None, points_mm=None):
    """Weighted least-squares force from the stacked readings of n chips, S (n, 3) in uT/N. Returns the force noise (N) and the
    weights g with F_hat = g . readings. With background "uniform" a uniform field (the Earth's) is estimated alongside and drops
    out of F_hat; "gradient" adds its gradient about U1. Too few readings for the unknowns: infinite noise."""
    n = len(S)
    A = S.reshape(-1, 1)
    if background:
        Bk = background_basis(points_mm if points_mm is not None else [np.zeros(3)] * n, background)
        if A.shape[0] < 1 + Bk.shape[1]:
            return math.inf, None
        A = np.concatenate([A, Bk], 1)
    W = 1.0 / noise_vec(st, n) ** 2
    C = np.linalg.inv(A.T @ (A * W[:, None]))
    return math.sqrt(C[0, 0]), (C @ (A * W[:, None]).T)[0]


def facing_change_uT(points_mm, x_from=25.0, x_to=30.0, d=4.76, t=1.59, br=1.45):
    """Field change at each sensing point when an identical magnet facing this tip (on the normal through U1, moment toward the
    board) moves from x_from to x_to mm in front of the sensing plane."""
    m = np.array([-br * math.pi * (d / 2e3) ** 2 * (t / 1e3) / (4e-7 * math.pi), 0.0, 0.0])
    c = points_mm[0]
    out = []
    for p in points_mm:
        a = dipole(p * 1e-3, (c + [x_from, 0, 0]) * 1e-3, m)
        b = dipole(p * 1e-3, (c + [x_to, 0, 0]) * 1e-3, m)
        out.append((b - a) * 1e6)
    return np.array(out)


def disturbances(g, n, pts, st):
    """Force read from 50 uT of uniform field (worst direction), from offset drift over 10 degC (independent on every axis of every
    chip, RMS) and from a facing magnet moving 25 -> 30 mm, through the weights g."""
    U = np.tile(np.eye(3), (n, 1))
    drift = DRIFT_LSB_PER_10C * np.tile([SENS[7][0][1], SENS[7][0][0], SENS[7][0][0]], n)
    return dict(earth_N=50.0 * float(np.linalg.norm(g @ U)), drift10C_N=float(np.sqrt(((g * drift) ** 2).sum())),
                neighbour_N=abs(float(g @ facing_change_uT(pts).ravel())))


def table(gaps=("touch", "v17")):
    rows = []
    for tp in tips():
        for gname in gaps:
            gap = GAPS[gname]
            B0, S = board_terms(tp["t_um"], tp["w_mrad"], gap)
            pts = board_points(gap)
            for name in SETTINGS:
                st = setting(name)
                n50 = max(1, int(st["rate_Hz"] // 50))
                s1, g1 = readout(S[:1], st)
                s5, g5 = readout(S, st)
                s5b, g5b = readout(S, st, "uniform")
                s5g, g5g = readout(S, st, "gradient", pts)
                d1 = disturbances(g1, 1, pts[:1], st)
                d5b = disturbances(g5b, 5, pts, st)
                d5g = disturbances(g5g, 5, pts, st)
                rows.append(dict(tip=tp["tip"], tip_label=tp["label"], setting=name, gap=gname, gap_mm=gap, rate_Hz=st["rate_Hz"],
                                 S_uT_per_N=S.tolist(), S_norm=float(np.linalg.norm(S[0])), S_norm_board=float(np.linalg.norm(S)),
                                 B0_uT=float(np.linalg.norm(B0[0])), B0_max_uT=float(np.abs(B0).max()),
                                 sigF_N=s1, sigF_board_N=s5, sigF_board_bg_N=s5b, sigF_50Hz_N=s1 / math.sqrt(n50),
                                 sigF_board_bg_50Hz_N=s5b / math.sqrt(n50), lsb_force_N=st["lsb_z_uT"] / abs(S[0, 0]),
                                 saturated=bool(np.abs(B0).max() > 1e3 * min(st["range_z_mT"], st["range_xy_mT"])),
                                 earth_N=d1["earth_N"], drift10C_N=d1["drift10C_N"], neighbour_N=d1["neighbour_N"],
                                 earth_board_N=d5b["earth_N"], drift10C_board_N=d5b["drift10C_N"],
                                 neighbour_board_N=d5b["neighbour_N"], sigF_board_grad_N=s5g,
                                 drift10C_board_grad_N=d5g["drift10C_N"], neighbour_board_grad_N=d5g["neighbour_N"],
                                 approach_um=tp["approach_um"], E_star=tp["E_star"], travel_um=-tp["t_um"][0]))
    return rows


def mount_table(st_name="high_gain_141Hz"):
    """The board's readout of the 20 % gyroid tip against the sensing gap and the magnet's depth: the as-printed tip at 5 mm, and
    the earlier homogenized model's runs at 1.5, 3 and 5 mm (docs/experiments/20261009-tpu_tip_fem/magnet_runs.jsonl)."""
    st = setting(st_name)
    M = [json.loads(l) for l in open(ROOT / "docs/experiments/20261009-tpu_tip_fem/magnet_runs.jsonl")]
    T = {t["tip"]: t for t in tips()}
    cases = [("gyroid20", 5.0, "touch"), ("gyroid20", 5.0, "v17"), ("w95_20", 1.5, "touch"), ("w95_20", 3.0, "touch"),
             ("w95_20", 5.0, "touch"), ("w95_20", 5.0, "v17")]
    out = []
    for tip, depth, gname in cases:
        if tip in T and tip == "gyroid20":
            t_um, w, E = T[tip]["t_um"], T[tip]["w_mrad"], T[tip]["E_star"]
        else:
            r = next(x for x in M if x["recipe"] == tip and x["magnet"] == "d4.76x1.59" and x["depth_mm"] == depth)
            t_um, w, E = r["per_force"]["1"]["magnet_t_um"], r["per_force"]["1"]["magnet_w_mrad"], r["E_best_MPa"]
        B0, S = board_terms(t_um, w, GAPS[gname], depth=depth)
        s1, _ = readout(S[:1], st)
        s5, _ = readout(S, st, "uniform")
        out.append(dict(tip=tip, depth=depth, gap=gname, gap_mm=GAPS[gname], travel_um=-t_um[0], S_U1=float(np.linalg.norm(S[0])),
                        B0_U1_mT=float(np.linalg.norm(B0[0])) / 1e3, sigF_U1=s1, sigF_board_bg=s5, E_star=E,
                        S_chips=S.tolist()))
    return out


if __name__ == "__main__":
    for name in SETTINGS:
        s = setting(name)
        print(f"{name:16s} {s['rate_Hz']:6.0f} Hz  noise xy {s['noise_xy_uT']:.2f} z {s['noise_z_uT']:.2f} uT  "
              f"LSB xy {s['lsb_xy_uT']:.3f} z {s['lsb_z_uT']:.3f} uT  range z {s['range_z_mT']:.1f} mT")
    print()
    for r in table():
        if r["setting"] not in ("high_gain_141Hz",):
            continue
        print(f"{r['tip']:14s} {r['gap']:5s} travel {r['travel_um']:6.1f} um/N  S(U1) {r['S_norm']:6.1f} uT/N  B0max {r['B0_max_uT'] / 1e3:.2f} mT  "
              f"sigF U1 {r['sigF_N']:.3f} board {r['sigF_board_N']:.3f} board+bg {r['sigF_board_bg_N']:.3f} N  "
              f"| U1: earth {r['earth_N']:.2f} drift {r['drift10C_N']:.2f} facing {r['neighbour_N']:.2f}  "
              f"| board+bg: earth {r['earth_board_N']:.2g} drift {r['drift10C_board_N']:.2f} facing {r['neighbour_board_N']:.2f} N"
              f"{'  SAT' if r['saturated'] else ''}")
