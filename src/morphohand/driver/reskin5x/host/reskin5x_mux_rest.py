#!/usr/bin/env python3
"""Each board's reading at rest over a log of reskin5x_mux.py (several boards behind an I2C mux): the change of its rest
from a reference window, corrected for the chips' temperature, split into a uniform field (the board turned in the
Earth's field) and the rest of it (a magnet nearby, such as another fingertip's), with the steps between still periods.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_mux_rest.py LOG.bin --ref 11:22:00 11:25:00
        [--boards 3] [--names ch0 ch1 ch3] [--bin-s 10] [--since 11:22] [--until 12:30] [--out rest.json]
        [--no-thermal] [--tempfit] [--series rest_1s.npz]

uniform    least-squares fit of one field vector read by every chip through its rotation (the board's chip frames of
           wrench_viewer.py's --board-cal); 'left' is what it leaves, in uT over the 15 components
tempfit    each chip's gain against its die temperature left in the change (2026-10-27): the change of every bin less a
           uniform field of its own is fitted by dk (T - T_ref) times the chip's reference field, dk one coefficient a
           chip (%/C) and T_ref the chip's mean temperature in the reference window. With the thermal correction on, dk
           is what the correction leaves; with --no-thermal it is the chip's whole coefficient (the correction's is
           -0.65 to -0.70 %/C, scripts/fingertip_contact_estimation_study.py thermal, 2026-10-21). A slow drift of
           the pad in its seat that runs alongside the night's cooling enters this fit as well, so it is repeated on
           the swings about the running mean over --hp-min minutes (changes and temperatures alike)
series     the bins' times, changes (15), uniform fields (3), die temperatures (5) and record counts as an npz, board by
           board (keys <k>_t, <k>_d, <k>_g, <k>_T, <k>_n with k the board's index), for 1 s bins and other studies

Clock times after midnight fall on the day after the log's start. The log is read as a memory map, one board at a
time, and the records the viewer drops (reskin5x_mux_check.valid) are left out.
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
sys.path.insert(0, str(HERE))
import fingertip_wrench_model as W  # noqa: E402
from reskin5x_mux_check import at, clock, valid, window  # noqa: E402,F401

BOARD_CAL = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-calib_pad_as_found.json"
CHIPS = 5
CHUNK = 1 << 20


def smoothed_T(T):
    """Each chip's die temperature (n, 5) averaged over the 450 records (about 3 s) up to and including each record."""
    c = np.cumsum(np.r_[np.zeros((1, CHIPS)), np.asarray(T, float)], axis=0)
    j1 = np.arange(1, len(T) + 1)
    j0 = np.maximum(j1 - 450, 0)
    return (c[j1] - c[j0]) / (j1 - j0)[:, None]


def fields(r, kappa):
    """Temperature-corrected fields (n, 15) uT of one board's records (n, 5, 4), each chip's die temperature smoothed
    over the 450 records (about 3 s) before it; kappa None leaves the readings as they are."""
    if kappa is None:
        return np.asarray(r[:, :, 1:], float).reshape(-1, 15)
    Tm = smoothed_T(r[:, :, 0])
    B = np.empty((len(r), 15))
    for i in range(0, len(r), CHUNK):
        B[i:i + CHUNK] = W.thermal_correct(r[i:i + CHUNK, :, 1:], Tm[i:i + CHUNK], kappa).reshape(-1, 15)
    return B


def uniform_basis():
    """(15, 3): the reading of a uniform field of 1 uT along x, y and z of the tip frame through every chip's rotation."""
    Rot = np.asarray(W.Model(W.Calib.load(BOARD_CAL)).R)
    return np.concatenate(list(Rot), 0)


def board(t, r, b, kappa):
    """Host times (m,) and fields (m, 15) of board b's valid records, with the die temperatures (m, 5)."""
    rb = np.array(r[:, b], np.float32)
    ok = valid(rb)
    rb = rb[ok]
    return t[ok], fields(rb, kappa), rb[:, :, 0].astype(float)


def bins(t, B, T, edges, width, ref, U, min_n=50):
    """Median change (k, 15) from ref, uniform field (k, 3), mean die temperature (k, 5), spread (k,) and record count
    (k,) of the bins [e, e + width) that hold at least min_n records."""
    ts = np.maximum.accumulate(t)                      # the splitter's stamps step back by a few ms inside a batch
    i0 = np.searchsorted(ts, edges)
    i1 = np.searchsorted(ts, edges + width)
    keep = (i1 - i0) >= min_n
    D, Tb, S, N = [], [], [], []
    for a, z in zip(i0[keep], i1[keep]):
        D.append(np.median(B[a:z], axis=0) - ref)
        Tb.append(T[a:z].mean(0))
        S.append(float(np.linalg.norm(B[a:z].std(0))))
        N.append(z - a)
    D = np.array(D).reshape(-1, 15)
    G = D @ np.linalg.pinv(U).T
    return edges[keep], D, G, np.array(Tb).reshape(-1, CHIPS), np.array(S), np.array(N)


def tempfit(D, Tb, T_ref, ref, U):
    """Each chip's gain coefficient dk (5,) per C left in the changes D (k, 15), with a uniform field of its own in every
    bin, and the rms (uT) of the residual before and after."""
    P = np.eye(15) - U @ np.linalg.pinv(U)
    y = (D @ P.T).ravel()
    X = np.zeros((len(D), 15, CHIPS))
    for c in range(CHIPS):
        X[:, 3 * c:3 * c + 3, c] = ref[3 * c:3 * c + 3][None, :] * (Tb[:, c] - T_ref[c])[:, None]
    X = np.einsum("ij,kjc->kic", P, X).reshape(-1, CHIPS)
    dk = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ dk
    return dk, float(np.sqrt(np.mean(y ** 2))), float(np.sqrt(np.mean(res ** 2)))


def highpass(x, w):
    """x (k, ...) less its centred running mean over w bins, for the bins with a full window (k - w + 1 rows)."""
    c = np.cumsum(np.r_[np.zeros((1,) + x.shape[1:]), x], axis=0)
    m = (c[w:] - c[:-w]) / w
    h = w // 2
    return x[h:h + len(m)] - m


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
    ap.add_argument("--no-thermal", action="store_true", help="leave the readings uncorrected for the chips' temperature")
    ap.add_argument("--tempfit", action="store_true", help="fit each chip's gain left against its die temperature")
    ap.add_argument("--hp-min", type=float, default=60.0, help="running mean removed for the fit on the swings (min)")
    ap.add_argument("--series", default=None, help="write the bins as an npz")
    ap.add_argument("--rows", type=int, default=30, help="print at most this many rows a board")
    a = ap.parse_args()
    names = a.names or [f"board {b}" for b in range(a.boards)]
    since = a.since and min(a.since, a.ref[0], key=lambda s: clock_at(a.log, a.boards, s))
    t, r = window(a.log, a.boards, since, a.until)
    kappa = None if a.no_thermal else W.load_thermal()
    U = uniform_basis()
    r0, r1 = at(a.ref[0], t[0]), at(a.ref[1], t[0])
    t_from = at(a.since, t[0]) if a.since else t[0]
    edges = np.arange(t_from, t[-1], a.bin_s)
    out = dict(log=str(a.log), ref=a.ref, bin_s=a.bin_s, thermal=kappa is not None, boards={})
    series = {}
    for b, name in enumerate(names):
        tb, B, T = board(t, r, b, kappa)
        m = (tb >= r0) & (tb <= r1)
        ref, T_ref = np.median(B[m], axis=0), T[m].mean(0)
        e, D, G, Tb, S, N = bins(tb, B, T, edges, a.bin_s, ref, U)
        left = D - G @ U.T
        rows = [dict(t=time.strftime("%H:%M:%S", time.localtime(e[k])), change_uT=round(float(np.linalg.norm(D[k])), 1),
                     uniform_uT=round(float(np.linalg.norm(G[k])), 1), left_uT=round(float(np.linalg.norm(left[k])), 1),
                     chip_uT=[round(float(v), 1) for v in np.linalg.norm(D[k].reshape(5, 3), axis=1)],
                     spread_uT=round(float(S[k]), 1), die_C=round(float(Tb[k].mean()), 2))
                for k in range(len(e))]
        res = dict(ref_norm_mT=round(float(np.linalg.norm(ref)) / 1e3, 3), records=int(len(tb)), dropped=int(len(t) - len(tb)),
                   die_C_ref=np.round(T_ref, 2).tolist(), die_C_min=np.round(Tb.min(0), 2).tolist(),
                   die_C_max=np.round(Tb.max(0), 2).tolist(), rows=rows)
        print(f"{name}: rest {np.linalg.norm(ref) / 1e3:.3f} mT over the 15 components in {a.ref[0]}-{a.ref[1]}; "
              f"{len(e)} bins of {a.bin_s:g} s; die temperature {np.round(Tb.min(0), 2)} to {np.round(Tb.max(0), 2)} C "
              f"(range {np.round(Tb.max(0) - Tb.min(0), 2)})")
        if a.tempfit:
            dk, rms0, rms1 = tempfit(D, Tb, T_ref, ref, U)
            w = max(3, int(round(a.hp_min * 60 / a.bin_s)) | 1)
            Dh, Th = highpass(D, w), highpass(Tb, w)
            dkh, rmsh0, rmsh1 = tempfit(Dh, Th + T_ref, T_ref, ref, U)
            res.update(tempfit_pct_per_C=np.round(100 * dk, 4).tolist(), tempfit_rms_uT=[round(rms0, 2), round(rms1, 2)],
                       tempfit_hp_pct_per_C=np.round(100 * dkh, 4).tolist(), tempfit_hp_rms_uT=[round(rmsh0, 2), round(rmsh1, 2)],
                       tempfit_hp_min=a.hp_min, die_swing_C=np.round(Th.std(0), 3).tolist())
            print(f"   gain left against die temperature {np.round(100 * dk, 3)} %/C; rms without the uniform field "
                  f"{rms0:.2f} uT a component, {rms1:.2f} after the fit")
            print(f"   on the swings about the {a.hp_min:g}-min running mean (die sd {np.round(Th.std(0), 3)} C): "
                  f"{np.round(100 * dkh, 3)} %/C; rms {rmsh0:.2f} -> {rmsh1:.2f} uT")
        print("   time      change  uniform  left  spread  die C   per chip (uT)")
        for rw in rows[::max(1, -(-len(rows) // a.rows))]:
            print(f"   {rw['t']}  {rw['change_uT']:7.1f}  {rw['uniform_uT']:6.1f}  {rw['left_uT']:6.1f}  {rw['spread_uT']:6.1f}"
                  f"  {rw['die_C']:5.2f}   {rw['chip_uT']}")
        out["boards"][name] = res
        series.update({f"{b}_t": e, f"{b}_d": D.astype(np.float32), f"{b}_g": G.astype(np.float32),
                       f"{b}_T": Tb.astype(np.float32), f"{b}_n": N.astype(np.int32)})
        del B
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    if a.series:
        Path(a.series).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.series, names=np.array(names), U=U, **series)


def clock_at(log, n_boards, s):
    """The time of clock reading s in the log (for picking the earlier of two readings)."""
    dt = np.dtype([("t", "<f8"), ("r", "<f4", (n_boards * CHIPS * 4,))])
    t0 = float(np.memmap(log, dt, mode="r", shape=(1,))["t"][0])
    return at(s, t0)


if __name__ == "__main__":
    main()
