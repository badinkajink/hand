"""Offsets that presses leave on the ReSkin 5X pad: the reading of every rest in a recording relative to the first rest,
and the rigid motion of the whole pad (translation, tilt) and the uniform field that reproduce each offset.

A pad that returns to its seat leaves rests within a few uT of each other (the noise of a 1 s mean is about 0.1 uT); a
pad that rocks or slides in its seat leaves offsets of tens of uT that a rigid motion of 10-20 um and a few mrad fits
(the recording of 2026-10-20 19:39: 61 uT after the fourth press, 104 uT after the fifth). The model is that of
scripts/fingertip_wrench_model.py, calibrated on the first rest with the board's chip frames kept.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/pad_shift_check.py logs/20261020-contact_estimation/<file>.csv
    ... logs/20261020-contact_estimation/stream/<start>-stream.bin --t0 120 --t1 180     a stretch of the viewer's stream log

Record with key r in the contact viewer (or reskin5x_plot.py --out): 2 s untouched, then presses of under a second,
each followed by at least 2 s untouched. Stationary stretches of --min-rest seconds count as rests; a press held that
long is listed too, and its pattern fits a rigid motion poorly.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_wrench_model as W  # noqa: E402

BOARD_CAL = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-calib_pad_as_found.json"
BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_patch_basis.npz"


def load(path, t0=None, t1=None):
    """(time from the file's start (s), readings (n, 15) uT, start as epoch s) of a recording (CSV of reskin5x_plot.py
    --out or of the viewer's key r) or of the viewer's stream log (.bin), cut to [t0, t1) seconds from the start."""
    if str(path).endswith(".bin"):
        a = np.fromfile(path, np.dtype([("t", "<f8"), ("r", "<f4", (20,))]))
        th, r = a["t"], a["r"].reshape(-1, 5, 4).astype(float)
    else:
        d = np.loadtxt(path)
        th, r = d[:, 0], d[:, 1:].reshape(-1, 5, 4)
    t = th - th[0]
    sel = np.ones(len(t), bool) if t0 is None else t >= t0
    if t1 is not None:
        sel &= t < t1
    return t[sel], r[sel][:, :, 1:].reshape(-1, 15), float(th[0])


def rests(t, r, min_rest, rate_uT_s):
    """Stationary stretches: the reading changes by less than rate_uT_s (norm over the 15 components, 0.1 s means)."""
    edges = np.arange(t[0], t[-1], 0.1)
    idx = np.searchsorted(t, edges)
    m = np.array([r[a:b].mean(0) if b > a else np.full(15, np.nan) for a, b in zip(idx[:-1], idx[1:])])
    rate = np.r_[np.inf, np.linalg.norm(np.diff(m, axis=0), axis=1) / 0.1]
    still = rate < rate_uT_s
    out, k = [], 0
    while k < len(still):
        if still[k]:
            j = k
            while j < len(still) and still[j]:
                j += 1
            t0, t1 = edges[k] + 0.1, edges[j]            # drop the first bin: it may still be settling
            if t1 - t0 >= min_rest:
                out.append((t0, t1))
            k = j
        else:
            k += 1
    return out


def rigid_jacobian(model):
    """Reading change (uT per mm, per rad) for a rigid translation and rotation (about the magnets' centroid) of the
    whole magnet set, and U (uT per uT) for a uniform field; columns tx ty tz wx wy wz gx gy gz."""
    mg, mu = model.mc.copy(), model.mu.copy()
    c0 = mg.mean(0)

    def reading(v):
        tr, w = v[:3], v[3:]
        return (np.einsum("cij,cj->ci", model.R, model.field(mg + tr[None] + np.cross(w[None], mg - c0),
                                                             mu + np.cross(w[None], mu), np.repeat(w[None], len(mg), 0)))
                * 1e6).ravel()
    r0, h = reading(np.zeros(6)), 1e-4
    J = np.array([(reading(np.eye(6)[i] * h) - r0) / h for i in range(6)]).T
    return np.c_[J, np.concatenate(list(model.R), 0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("--min-rest", type=float, default=1.0, help="shortest stationary stretch counted (s)")
    ap.add_argument("--rate", type=float, default=40.0, help="stationary below this rate of change (uT/s)")
    ap.add_argument("--board-cal", default=str(BOARD_CAL))
    ap.add_argument("--t0", type=float, default=None, help="start of the stretch (s from the file's start)")
    ap.add_argument("--t1", type=float, default=None, help="end of the stretch (s from the file's start)")
    a = ap.parse_args()
    t, r, start = load(a.csv, a.t0, a.t1)
    print(f"file starts {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    segs = rests(t, r, a.min_rest, a.rate)
    if not segs:
        raise SystemExit("no stationary stretch found")
    s0 = (t >= segs[0][0]) & (t < segs[0][1])
    ref = r[s0].mean(0)
    noise = float(np.linalg.norm(r[s0].std(0)) / np.sqrt(max(1, s0.sum())))
    cal = W.calibrate_frames_fixed(ref.reshape(5, 3), W.Calib.load(a.board_cal))
    model = W.Model(cal)
    A = rigid_jacobian(model)
    est = W.Estimator(model, W.load_basis(BASIS))
    est.set_noise_scale(1 / np.sqrt(0.05 * 140))
    print(f"{Path(a.csv).name}: {t[0]:.1f}-{t[-1]:.1f} s; first rest {segs[0][0]:.1f}-{segs[0][1]:.1f} s, its mean known to "
          f"{noise:.2f} uT; rest fit {cal.rms_mT:.2f} mT RMS, magnet strength {cal.k:.2f} of N52")
    print("  rest (s)        offset   per chip (largest component, uT)        rigid fit: translation (um)  tilt (mrad)"
          "   residual   viewer reads")
    for t0, t1 in segs:
        sel = (t >= t0) & (t < t1)
        P = r[sel].mean(0) - ref
        x, *_ = np.linalg.lstsq(A, P, rcond=None)
        res = np.linalg.norm(P - A @ x) / max(np.linalg.norm(P), 1e-9)
        big = [f"{v:+5.0f}" for v in P.reshape(5, 3)[np.arange(5), np.abs(P.reshape(5, 3)).argmax(1)]]
        e = est.estimate(P)
        reads = (f"{e['f'][0]:.2f} N at ({e['y']:+.1f}, {e['z']:.1f})" if e["contact"] and e["e"] > 25 else "no contact")
        print(f"  {t0:5.1f}-{t1:5.1f}  {np.linalg.norm(P):6.1f} uT  {' '.join(big)}   "
              f"({1e3 * x[0]:+5.1f}, {1e3 * x[1]:+5.1f}, {1e3 * x[2]:+5.1f})  ({1e3 * x[3]:+4.1f}, {1e3 * x[4]:+4.1f}, "
              f"{1e3 * x[5]:+4.1f})   {100 * res:4.0f} %    {reads}")
    print("  A pad that returns to its seat gives offsets of a few uT; 26 uT is the model's change for 1 N on the face.")


if __name__ == "__main__":
    main()
