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
burst T+XYZ, five chips read in turn at 400 kHz I2C.

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


def neighbour_field_uT(dist_mm, d=4.76, t=1.59, br=1.45):
    """On-axis field of the same magnet at dist_mm (a facing fingertip's magnet seen by this tip's chip)."""
    m = br * math.pi * (d / 2e3) ** 2 * (t / 1e3) / (4e-7 * math.pi)
    return 2e-7 * m / (dist_mm * 1e-3) ** 3 * 1e6


def table(gap=1.0):
    rows = []
    for tp in tips():
        B0, S = field_terms(tp["t_um"], tp["w_mrad"], gap)
        for name in SETTINGS:
            st = setting(name)
            sig = np.array([st["noise_z_uT"], st["noise_xy_uT"], st["noise_xy_uT"]])     # chip z along the tip's x
            sigF = 1.0 / math.sqrt(float(((S / sig) ** 2).sum()))
            n50 = max(1, int(st["rate_Hz"] // 50))
            rows.append(dict(tip=tp["tip"], tip_label=tp["label"], setting=name, rate_Hz=st["rate_Hz"], S_uT_per_N=S.tolist(),
                             S_norm=float(np.linalg.norm(S)), B0_uT=float(np.linalg.norm(B0)), sigF_N=sigF,
                             sigF_50Hz_N=sigF / math.sqrt(n50), lsb_force_N=st["lsb_z_uT"] / abs(S[0]),
                             saturated=bool(abs(B0[0]) > 1e3 * st["range_z_mT"]),
                             earth_N=50.0 / float(np.linalg.norm(S)),
                             drift10C_N=st["offset_drift_z_uT_per_10C"] / abs(S[0]),
                             neighbour_N=(neighbour_field_uT(25.0) - neighbour_field_uT(30.0)) / float(np.linalg.norm(S)),
                             approach_um=tp["approach_um"], E_star=tp["E_star"], travel_um=-tp["t_um"][0]))
    return rows


if __name__ == "__main__":
    for name in SETTINGS:
        s = setting(name)
        print(f"{name:16s} {s['rate_Hz']:6.0f} Hz  noise xy {s['noise_xy_uT']:.2f} z {s['noise_z_uT']:.2f} uT  "
              f"LSB xy {s['lsb_xy_uT']:.3f} z {s['lsb_z_uT']:.3f} uT  range z {s['range_z_mT']:.1f} mT")
    print()
    for r in table():
        print(f"{r['tip']:14s} {r['setting']:16s} travel {r['travel_um']:6.1f} um/N  S {r['S_norm']:6.1f} uT/N  "
              f"sigF {r['sigF_N']:.3f} N @ {r['rate_Hz']:.0f} Hz, {r['sigF_50Hz_N']:.3f} N @50 Hz  LSB {r['lsb_force_N']:.3f} N  "
              f"earth {r['earth_N']:.2f} N  drift/10C {r['drift10C_N']:.2f} N  neighbour 25->30mm {r['neighbour_N']:.1f} N"
              f"{'  SAT' if r['saturated'] else ''}")
