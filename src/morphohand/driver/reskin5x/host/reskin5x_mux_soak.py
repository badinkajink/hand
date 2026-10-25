#!/usr/bin/env python3
"""Fingers left at rest behind the mux (a soak): each board's frames replayed through the contact viewer's logic
(scripts/fingertip_foreign_field.py replay: 50 ms window, re-zero after 3 s at rest, a contact shown above a chi-square
of 25 and 0.25 N) over a window of a reskin5x_mux.py log, with the contacts shown at rest, the size of the change the
viewer reads, the re-zeros and the drift of the rest.

    .venv/bin/python src/morphohand/driver/reskin5x/host/reskin5x_mux_soak.py LOG.bin --since 11:39:30 [--until 12:30]
        --calib ch0=<calib.json> ch1=<calib.json> ch3=<calib.json> [--channels 0 1 3] [--out soak.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
import fingertip_wrench_model as W  # noqa: E402
import fingertip_foreign_field as FF  # noqa: E402
from reskin5x_mux_rest import clock, fields, CHIPS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log")
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", default=None)
    ap.add_argument("--channels", type=int, nargs="+", default=[0, 1, 3])
    ap.add_argument("--calib", nargs="+", required=True, help="ch<k>=<calibration json> for every channel")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cals = dict(s.split("=", 1) for s in a.calib)
    n = len(a.channels)
    dt = np.dtype([("t", "<f8"), ("r", "<f4", (n * CHIPS * 4,))])
    z = np.fromfile(a.log, dt)
    t, R = z["t"], z["r"].reshape(len(z), n, CHIPS, 4).astype(float)
    kappa = W.load_thermal()
    t0 = clock(a.since, t[0])
    t1 = clock(a.until, t[0]) if a.until else t[-1]
    m = (t >= t0 - 5.0) & (t <= t1)
    out = dict(log=str(a.log), since=a.since, until=a.until or time.strftime("%H:%M:%S", time.localtime(t[-1])), boards={})
    for b, ch in enumerate(a.channels):
        B = fields(R[:, b], kappa)
        tb, Bb = t[m], B[m]
        keep = tb >= t0
        cal = W.Calib.load(cals[f"ch{ch}"])
        F = FF.replay([(tb[keep], Bb[keep])], cal, fps=a.fps)
        shown = F[:, 4] > 0
        hours = (F[-1, 0] - F[0, 0]) / 3600
        # re-zeros of the replay: the viewer's RestZero takes a new base after 3 s at rest when the rest moved > 5 uT
        rz = W.RestZero(hold_s=3.0)
        base, nre = np.median(Bb[keep][:300], axis=0), 0
        for tf in np.arange(tb[keep][0] + 2.0, tb[keep][-1], 1.0):
            i1 = np.searchsorted(tb, tf)
            j0 = np.searchsorted(tb, tf - rz.span_s)
            nb = rz.update(tb[j0:i1], Bb[j0:i1], base)
            if nb is not None:
                base, nre = nb, nre + 1
        first, last = np.median(Bb[keep][:1500], axis=0), np.median(Bb[keep][-1500:], axis=0)
        runs = np.flatnonzero(np.diff(np.r_[0, shown.astype(int), 0]))
        events = [dict(t=time.strftime("%H:%M:%S", time.localtime(F[s, 0])), s=round((e - s) / a.fps, 1),
                       d_max_uT=round(float(F[s:e, 1].max()), 1), f_max_N=round(float(F[s:e, 3].max()), 2))
                  for s, e in zip(runs[::2], runs[1::2])]
        T = R[m][keep][:, b, :, 0]
        res = dict(calib=Path(cals[f"ch{ch}"]).name, frames=int(len(F)), hours=round(hours, 3),
                   shown=int(shown.sum()), events=events,
                   d_median_uT=round(float(np.median(F[:, 1])), 1), d_p99_uT=round(float(np.percentile(F[:, 1], 99)), 1),
                   d_max_uT=round(float(F[:, 1].max()), 1), rezeros=nre,
                   rest_drift_uT=round(float(np.linalg.norm(last - first)), 1),
                   die_C=[round(float(T[:1500].mean()), 2), round(float(T[-1500:].mean()), 2)])
        out["boards"][f"ch{ch}"] = res
        print(f"ch{ch} ({res['calib']}): {res['frames']} frames over {hours:.2f} h, {res['shown']} with a contact shown in "
              f"{len(events)} events; viewer's change median {res['d_median_uT']} uT, p99 {res['d_p99_uT']}, max "
              f"{res['d_max_uT']}; {nre} re-zeros; rest drift {res['rest_drift_uT']} uT; die {res['die_C'][0]} -> "
              f"{res['die_C'][1]} C")
        for ev in events[:12]:
            print(f"    {ev['t']} {ev['s']} s, change up to {ev['d_max_uT']} uT, up to {ev['f_max_N']} N")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
