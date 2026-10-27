#!/usr/bin/env python3
"""A pad's rest over the hours after hard presses: the change of each board's reading at rest from a reference rest
taken before the presses, from the viewer's stream logs (88-byte records, wrench_viewer.read_stream) and then a
reskin5x_mux.py log, corrected for the chips' temperature and split as reskin5x_mux_rest.py splits it into a uniform
field and the part it leaves (owner's presses of 2026-10-26 13:49-14:00 on ID 1 and ID 2, up to the estimator's
25 N cap).

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_mux_recovery.py --names "ID 1" "ID 2"
        --streams logs/.../20261026-134212-id_1-stream.bin,logs/.../20261026-141207-id_1-stream.bin
                  logs/.../20261026-134212-id_2-stream.bin,logs/.../20261026-141207-id_2-stream.bin
        --mux logs/20261026-three_finger_mux/20261026-193314-mux.bin --mux-boards 3 --mux-index 0 1 --mux-until 04:00
        --ref 13:42:30 13:44:00 --presses 13:49 14:00 [--bin-s 60] [--out recovery.json]

still   a bin whose spread (norm over the 15 components of the records' standard deviation) is under --still-uT; only
        still bins enter the hourly means, so the minutes of handling are left out
hourly  the mean of the uniform field and of the part left over the still bins of each hour after the presses ended;
        a board that was moved (the boards were rearranged when ID 3 was plugged in, about 19:30) changes its uniform
        field and the gradient across it, so across such a move only the part left compares
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
from reskin5x_mux_check import at, valid, window  # noqa: E402
from reskin5x_mux_rest import board, fields, uniform_basis  # noqa: E402

STREAM = np.dtype([("t", "<f8"), ("r", "<f4", (20,))])


def stream(path, kappa):
    """Host times (n,), corrected fields (n, 15) and die temperatures (n, 5) of a viewer's stream log."""
    z = np.memmap(path, STREAM, mode="r")
    t = np.array(z["t"])
    r = np.array(z["r"]).reshape(-1, 5, 4)
    ok = valid(r)
    return t[ok], fields(r[ok], kappa), r[ok][:, :, 0].astype(float)


def binned(t, B, T, width, ref, U):
    """60 s (width) bins of one source: start, median change (k, 15), uniform field (k, 3), spread (k,), die (k,)."""
    ts = np.maximum.accumulate(t)
    edges = np.arange(ts[0], ts[-1], width)
    i0, i1 = np.searchsorted(ts, edges), np.searchsorted(ts, edges + width)
    keep = (i1 - i0) >= 50
    D = np.array([np.median(B[a:z], 0) - ref for a, z in zip(i0[keep], i1[keep])]).reshape(-1, 15)
    S = np.array([np.linalg.norm(B[a:z].std(0)) for a, z in zip(i0[keep], i1[keep])])
    Tm = np.array([T[a:z].mean() for a, z in zip(i0[keep], i1[keep])])
    return edges[keep], D, D @ np.linalg.pinv(U).T, S, Tm


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--streams", nargs="+", required=True, help="per board, its stream logs in order, comma-separated")
    ap.add_argument("--mux", default=None)
    ap.add_argument("--mux-boards", type=int, default=3)
    ap.add_argument("--mux-index", type=int, nargs="*", default=None, help="each board's index in the mux log")
    ap.add_argument("--mux-since", default=None)
    ap.add_argument("--mux-until", default=None)
    ap.add_argument("--ref", nargs=2, required=True)
    ap.add_argument("--presses", nargs=2, required=True, help="start and end of the presses")
    ap.add_argument("--bin-s", type=float, default=60.0)
    ap.add_argument("--still-uT", type=float, default=12.0)
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kappa = W.load_thermal()
    U = uniform_basis()
    out = dict(ref=a.ref, presses=a.presses, bin_s=a.bin_s, still_uT=a.still_uT, boards={})
    mux = window(a.mux, a.mux_boards, a.mux_since, a.mux_until) if a.mux else None
    for k, name in enumerate(a.names):
        parts = []
        for p in a.streams[k].split(","):
            parts.append(("stream " + Path(p).name, *stream(p, kappa)))
        t_first = parts[0][1][0]
        r0, r1 = at(a.ref[0], t_first), at(a.ref[1], t_first)
        p_end = at(a.presses[1], t_first)
        tt, BB = parts[0][1], parts[0][2]
        m = (tt >= r0) & (tt <= r1)
        ref = np.median(BB[m], 0)
        if mux is not None:
            tm, Bm, Tm = board(mux[0], mux[1], a.mux_index[k], kappa)
            parts.append(("mux " + Path(a.mux).name, tm, Bm, Tm))
        rows, src = [], []
        for label, t, B, T in parts:
            e, D, G, S, Td = binned(t, B, T, a.bin_s, ref, U)
            L = D - G @ U.T
            for j in range(len(e)):
                rows.append(dict(t=float(e[j]), clock=time.strftime("%H:%M:%S", time.localtime(e[j])), source=label,
                                 change_uT=round(float(np.linalg.norm(D[j])), 1), uniform_uT=round(float(np.linalg.norm(G[j])), 1),
                                 left_uT=round(float(np.linalg.norm(L[j])), 1), spread_uT=round(float(S[j]), 1),
                                 still=bool(S[j] < a.still_uT), die_C=round(float(Td[j]), 2),
                                 g_uT=np.round(G[j], 2).tolist(), left_chip_uT=np.round(np.linalg.norm(L[j].reshape(5, 3), axis=1), 1).tolist()))
            src.append(dict(source=label, start=time.strftime("%H:%M:%S", time.localtime(t[0])),
                            end=time.strftime("%H:%M:%S", time.localtime(t[-1])), records=int(len(t))))
        hourly = []
        for h in range(0, 24):
            lo, hi = p_end + 3600 * h, p_end + 3600 * (h + 1)
            sel = [rw for rw in rows if lo <= rw["t"] < hi and rw["still"]]
            if not sel:
                continue
            srcs = sorted({rw["source"] for rw in sel})
            hourly.append(dict(hour=h, start=time.strftime("%H:%M", time.localtime(lo)), bins=len(sel), sources=srcs,
                               left_uT=round(float(np.mean([rw["left_uT"] for rw in sel])), 1),
                               uniform_uT=round(float(np.mean([rw["uniform_uT"] for rw in sel])), 1),
                               change_uT=round(float(np.mean([rw["change_uT"] for rw in sel])), 1),
                               die_C=round(float(np.mean([rw["die_C"] for rw in sel])), 2)))
        out["boards"][name] = dict(ref_norm_mT=round(float(np.linalg.norm(ref)) / 1e3, 3), sources=src, hourly=hourly, rows=rows)
        print(f"{name}: rest {np.linalg.norm(ref) / 1e3:.3f} mT in {a.ref[0]}-{a.ref[1]}; " +
              "; ".join(f"{s['source']} {s['start']}-{s['end']} ({s['records']} records)" for s in src))
        print("   hour  from   still bins  left  uniform  change  die C")
        for hr in hourly:
            print(f"   {hr['hour']:4d}  {hr['start']}  {hr['bins']:5d}  {hr['left_uT']:6.1f}  {hr['uniform_uT']:6.1f}  "
                  f"{hr['change_uT']:6.1f}  {hr['die_C']:5.2f}  {','.join(s.split()[0] for s in hr['sources'])}")
        print("   time      source  change  uniform  left  spread  die C   left per chip")
        for rw in rows[::max(1, -(-len(rows) // a.rows))]:
            print(f"   {rw['clock']}  {rw['source'][:6]}  {rw['change_uT']:7.1f}  {rw['uniform_uT']:6.1f}  {rw['left_uT']:6.1f}  "
                  f"{rw['spread_uT']:6.1f}  {rw['die_C']:5.2f}   {rw['left_chip_uT']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
