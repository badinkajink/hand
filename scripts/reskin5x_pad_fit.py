"""Polarities and position of a pad's magnets, fitted to the static field the ReSkin 5X board reads.

The model is tpu_tip_magnet_sensing's: point dipoles of N52 magnets (Br 1.45 T) at an FE row's positions, moments along the
chip normal, and the board's five chips 7 mm apart in a plus. For each hypothesis, the polarity of each corner magnet
relative to the centre one (16), the sign of the chips' z axis relative to the tip's normal (2) and which of the stream's
chips 1-4 (0x11, 0x12, 0x13, 0x10 on the workstation's board) sits at U2-U5 (24), a least-squares fit gives the distance
added between the magnets and the sensing plane (dx, mm) and the pad's offset across (dy) and along (dz) the finger. The
fit compares quantities that do not depend on a chip's in-plane rotation: the normal component and the in-plane magnitude
of the field at each chip (mT), with Earth's field and the chips' offsets (under 0.1 mT) left in.

    ~/miniconda3/bin/python scripts/reskin5x_pad_fit.py logs/20261020-reskin5x/20261020-pad_as_found.csv [--case b_v2]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_magnet_sensing as MS  # noqa: E402

FE_ROWS = ROOT / "docs/experiments/20261019-v2_tip_fe_checks/sense_fe.jsonl"
GAP_B = 3.32 - 6.45            # concept b's sensing plane behind the whole tip's back face, mm (v2_tip_fe_checks.SENSE_GAP)
U_NAMES = ("U2", "U3", "U4", "U5")


def fe_row(case):
    for line in open(FE_ROWS):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("case") == case:
            return r
    raise SystemExit(f"no FE row {case} in {FE_ROWS}")


def model(row, signs, dx, dy, dz, k=1.0):
    """Field (mT, tip frame: x normal, y across, z along) at U1-U5 with magnet i's moment times signs[i] and k, the sensing
    plane dx mm farther from the magnets and the pad shifted by (dy, dz) mm."""
    M = MS.mag_state(row)
    ch = [c + np.array([-dx, -dy, -dz]) * 1e-3 for c in MS.chips(row, "board", GAP_B)]
    return k * np.array([sum(s * MS.dipole(c, pm, m) for s, (pm, m) in zip(signs, M)) for c in ch]) * 1e3


def resolution(row, signs, dx, dy, dz, noise_xy, noise_z):
    """Normal-force resolution (N, one sample) at the contact point and its median over the pad's patch grid, with the contact
    position and a uniform background field unknown (tpu_tip_magnet_sensing.analyse, sigF_n_unk_bg), for the given poles
    and position, at the given noise (uT)."""
    saved = MS.mag_state, MS.chips, MS.NOISE
    state, chips = MS.mag_state, MS.chips
    MS.mag_state = lambda r, br=MS.BR: [(pm, s * m) for s, (pm, m) in zip(signs, state(r, br))]
    MS.chips = lambda r, mode, gap: [c + np.array([0.0, -dy, -dz]) * 1e-3 for c in chips(r, mode, gap)]
    MS.NOISE = np.array([noise_z, noise_xy, noise_xy])
    try:
        a = MS.analyse(row, "board", GAP_B + dx)
    finally:
        MS.mag_state, MS.chips, MS.NOISE = saved
    return a["centre"]["sigF_n_unk_bg"], float(np.median([p["sigF_n_unk_bg"] for p in a["points"]]))


def pattern_class(signs):
    flipped = [i for i in (0, 1, 3, 4) if signs[i] != signs[2]]
    if len(flipped) == 2:
        return "2 corners, " + ("diagonal" if flipped in ([0, 4], [1, 3]) else "same row" if flipped in ([0, 1], [3, 4])
                                else "same side")
    return f"{len(flipped)} corners"


def features(B, sz=1.0):
    """Normal component (times the chips' z sign) and in-plane magnitude per chip, (2 n,)."""
    return np.concatenate([sz * B[:, 0], np.hypot(B[:, 1], B[:, 2])])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture", help="reskin5x_check.py --out file of the board with the pad on, at rest")
    ap.add_argument("--case", default="b_v2", help="FE row whose magnet layout the pad has")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--noise", type=float, nargs=2, default=(1.65, 2.57), metavar=("XY", "Z"),
                    help="noise per sample (uT) for the force resolution; default: the board's at GAIN_SEL 1, RES 1")
    a = ap.parse_args()
    row = fe_row(a.case)
    d = np.loadtxt(a.capture).reshape(-1, 5, 4)[:, :, 1:].mean(axis=0) / 1e3      # mT, chip frames, streaming order
    meas = {i: (d[i, 2], np.hypot(d[i, 0], d[i, 1])) for i in range(5)}

    fits = []
    for corner in itertools.product((1, -1), repeat=4):
        signs = (corner[0], corner[1], 1, corner[2], corner[3])     # magnets in the FE row's order; the centre one is third
        for sz in (1.0, -1.0):
            for perm in itertools.permutations(range(1, 5)):           # stream index at U2, U3, U4, U5
                order = (0,) + perm
                y = np.array([meas[i][0] for i in order] + [meas[i][1] for i in order])
                res = lambda p: features(model(row, signs, *p), sz) - y
                best = min((least_squares(res, x0, bounds=([-3, -4, -4], [8, 4, 4])) for x0 in
                            ([0, 0, 0], [3, 0, 0], [6, 0, 0])), key=lambda r: r.cost)
                fits.append(dict(signs=signs, sz=sz, perm=perm, p=best.x, rms=float(np.sqrt(np.mean(best.fun ** 2)))))
    fits.sort(key=lambda f: f["rms"])
    print(f"{len(fits)} hypotheses, FE row {a.case}; magnets (y, z) mm:", [tuple(m[2:]) for m in row["magnets"]])
    print("rms (mT)   signs (centre third)   chips z   U2 U3 U4 U5 = stream index   dx dy dz (mm)")
    for f in fits[:a.top]:
        print(f"  {f['rms']:.3f}    {str(f['signs']):22s} {f['sz']:+.0f}        {f['perm']}        "
              f"{f['p'][0]:+.2f} {f['p'][1]:+.2f} {f['p'][2]:+.2f}")
    print("best fit per class (corner magnets opposite to the centre one):")
    for cls in sorted({pattern_class(f["signs"]) for f in fits}):
        g = next(f for f in fits if pattern_class(f["signs"]) == cls)
        print(f"  {cls:22s} rms {g['rms']:.3f} mT, dx {g['p'][0]:+.2f} mm, signs {g['signs']}")
    print("top fits refitted with the magnets' strength k free (0.3-1.5):")
    for f in fits[:4]:
        order = (0,) + f["perm"]
        y = np.array([meas[i][0] for i in order] + [meas[i][1] for i in order])
        r = least_squares(lambda p: features(model(row, f["signs"], *p), f["sz"]) - y, list(f["p"]) + [1.0],
                          bounds=([-3, -4, -4, 0.3], [8, 4, 4, 1.5]))
        print(f"  {str(f['signs']):22s} rms {np.sqrt(np.mean(r.fun ** 2)):.3f} mT, dx {r.x[0]:+.2f} mm, k {r.x[3]:.2f}")
    f = fits[0]
    same_way = (1, 1, 1, 1, 1)
    print(f"normal-force resolution at noise {a.noise[0]}/{a.noise[1]} uT (contact point, median over the pad), N:")
    for lbl, s, p in (("design: poles the same way, design position", same_way, (0, 0, 0)),
                      ("poles the same way, fitted position", same_way, f["p"]),
                      ("fitted poles, fitted position", f["signs"], f["p"])):
        c, m = resolution(row, s, *p, *a.noise)
        print(f"  {lbl:46s} {c:.4f}  {m:.4f}")
    order = (0,) + f["perm"]
    B = model(row, f["signs"], *f["p"])
    print("best fit per chip, normal / in-plane (mT), measured vs model:")
    for k, i in enumerate(order):
        print(f"  {'U' + str(k + 1)} = stream {i}: {meas[i][0]:+.2f} / {meas[i][1]:.2f}  vs  "
              f"{f['sz'] * B[k, 0]:+.2f} / {np.hypot(B[k, 1], B[k, 2]):.2f}")


if __name__ == "__main__":
    main()
