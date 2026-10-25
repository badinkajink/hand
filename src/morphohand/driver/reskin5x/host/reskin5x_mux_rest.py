#!/usr/bin/env python3
"""Each board's reading at rest over a log of reskin5x_mux.py (several boards behind an I2C mux): the change of its rest
from a reference window, corrected for the chips' temperature, split into a uniform field (the board turned in the
Earth's field) and the rest of it (a magnet nearby, such as another fingertip's), with the steps between still periods.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_mux_rest.py LOG.bin --ref 11:22:00 11:25:00
        [--boards 3] [--names ch0 ch1 ch3] [--bin-s 10] [--since 11:22] [--until 12:30] [--out rest.json]

uniform    least-squares fit of one field vector read by every chip through its rotation (the board's chip frames of
           wrench_viewer.py's --board-cal); 'left' is what it leaves, in uT over the 15 components
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_wrench_model as W  # noqa: E402

BOARD_CAL = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-calib_pad_as_found.json"
CHIPS = 5


def clock(s, day):
    h, m, *sec = (int(v) for v in s.split(":"))
    lt = time.localtime(day)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, sec[0] if sec else 0, 0, 0, -1))


def fields(r, kappa):
    """Temperature-corrected fields (n, 15) uT of one board's records (n, 5, 4), each chip's die temperature smoothed
    over the 450 records (about 3 s) before it."""
    T = r[:, :, 0]
    c = np.cumsum(np.r_[np.zeros((1, CHIPS)), T], axis=0)
    j1 = np.arange(1, len(T) + 1)
    j0 = np.maximum(j1 - 450, 0)
    Tm = (c[j1] - c[j0]) / (j1 - j0)[:, None]
    return W.thermal_correct(r[:, :, 1:], Tm, kappa).reshape(-1, 15)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log")
    ap.add_argument("--ref", nargs=2, required=True, metavar=("FROM", "TO"))
    ap.add_argument("--boards", type=int, default=3)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--bin-s", type=float, default=10.0)
    ap.add_argument("--since", default=None)
    ap.add_argument("--until", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names = a.names or [f"board {b}" for b in range(a.boards)]
    dt = np.dtype([("t", "<f8"), ("r", "<f4", (a.boards * CHIPS * 4,))])
    z = np.fromfile(a.log, dt)
    t, R = z["t"], z["r"].reshape(len(z), a.boards, CHIPS, 4).astype(float)
    day = t[0]
    m = np.ones(len(t), bool)
    if a.since:
        m &= t >= clock(a.since, day)
    if a.until:
        m &= t <= clock(a.until, day)
    kappa = W.load_thermal()
    board = W.Calib.load(BOARD_CAL)
    Rot = np.asarray(W.Model(board).R)                              # (5, 3, 3) chip frames
    U = np.concatenate(list(Rot), 0)                                # (15, 3): reading of a uniform field
    r0, r1 = clock(a.ref[0], day), clock(a.ref[1], day)
    out = dict(log=str(a.log), ref=a.ref, boards={})
    edges = np.arange(t[m][0], t[m][-1], a.bin_s)
    for b, name in enumerate(names):
        B = fields(R[:, b], kappa)
        ref = np.median(B[(t >= r0) & (t <= r1)], axis=0)
        rows = []
        for e0 in edges:
            s = (t >= e0) & (t < e0 + a.bin_s)
            if s.sum() < 50:
                continue
            Bs = B[s]
            d = np.median(Bs, axis=0) - ref
            g = np.linalg.lstsq(U, d, rcond=None)[0]
            left = d - U @ g
            spread = float(np.linalg.norm(Bs.std(0)))
            rows.append(dict(t=time.strftime("%H:%M:%S", time.localtime(e0)), change_uT=round(float(np.linalg.norm(d)), 1),
                             uniform_uT=round(float(np.linalg.norm(g)), 1), left_uT=round(float(np.linalg.norm(left)), 1),
                             chip_uT=[round(float(v), 1) for v in np.linalg.norm(d.reshape(5, 3), axis=1)],
                             spread_uT=round(spread, 1), die_C=round(float(R[s, b, :, 0].mean()), 2)))
        out["boards"][name] = dict(ref_norm_mT=round(float(np.linalg.norm(ref)) / 1e3, 3), rows=rows)
        print(f"{name}: rest {np.linalg.norm(ref) / 1e3:.3f} mT over the 15 components in {a.ref[0]}-{a.ref[1]}")
        print("   time      change  uniform  left  spread  die C   per chip (uT)")
        for rw in rows:
            print(f"   {rw['t']}  {rw['change_uT']:7.1f}  {rw['uniform_uT']:6.1f}  {rw['left_uT']:6.1f}  {rw['spread_uT']:6.1f}"
                  f"  {rw['die_C']:5.2f}   {rw['chip_uT']}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
