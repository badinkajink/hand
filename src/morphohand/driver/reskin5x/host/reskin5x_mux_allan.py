#!/usr/bin/env python3
"""Noise against averaging time for each board of a reskin5x_mux.py log (fingers at rest): the overlapping Allan
deviation of every component of the temperature-corrected field, from one record to --max-h hours.

    .venv/bin/python src/morphohand/driver/reskin5x/host/reskin5x_mux_allan.py LOG.bin --since 19:35 --until 04:00
        [--boards 3] [--names "ID 1" "ID 2" "ID 3"] [--max-h 2] [--no-thermal] [--out allan.json]

Allan deviation  at averaging time tau = m tau0, the root of half the mean square difference between the means of
                 adjacent stretches of m records, over every start (overlapping estimator); tau0 is the mean spacing of
                 the records (the splitter's stamps jitter by a few ms, so the records are taken as evenly spaced).
                 White noise falls as tau^-1/2; a drift makes it rise again. The record rate exceeds a chip's rate of
                 new conversions (163 against 140 a second), so near tau0 consecutive records repeat a value and the
                 deviation lies below a conversion's noise.
turning point    the tau at which a component's deviation is least, past which averaging longer no longer helps

The log is read as a memory map one board at a time; the records the viewer drops (reskin5x_mux_check.valid) are left
out, and clock times after midnight fall on the day after the log's start.
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
from reskin5x_mux_check import window  # noqa: E402
from reskin5x_mux_rest import board  # noqa: E402

AXES = ("x", "y", "z")


def adev(x, ms):
    """Overlapping Allan deviation of the evenly spaced series x (n,) at the stretch lengths ms (records)."""
    S = np.r_[0.0, np.cumsum(x - x.mean())]
    out = []
    for m in ms:
        A = (S[m:] - S[:-m]) / m
        d = A[m:] - A[:-m]
        out.append(float(np.sqrt(0.5 * np.mean(d * d))))
    return np.array(out)


def at_tau(tau, s, x):
    """s (taus,) interpolated in log-log at tau x."""
    return float(np.exp(np.interp(np.log(x), np.log(tau), np.log(s))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log")
    ap.add_argument("--since", default=None)
    ap.add_argument("--until", default=None)
    ap.add_argument("--boards", type=int, default=3)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--max-h", type=float, default=2.0)
    ap.add_argument("--points", type=int, default=64, help="averaging times, log-spaced")
    ap.add_argument("--no-thermal", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names = a.names or [f"board {b}" for b in range(a.boards)]
    t, r = window(a.log, a.boards, a.since, a.until)
    kappa = None if a.no_thermal else W.load_thermal()
    out = dict(log=str(a.log), since=a.since, until=a.until, thermal=kappa is not None, boards={})
    for b, name in enumerate(names):
        tc = time.time()
        tb, B, _ = board(t, r, b, kappa)
        n = len(tb)
        tau0 = (tb[-1] - tb[0]) / (n - 1)
        m_max = int(min(a.max_h * 3600 / tau0, (n - 1) // 3))
        ms = np.unique(np.round(np.logspace(0, np.log10(m_max), a.points)).astype(int))
        tau = ms * tau0
        S = np.array([adev(B[:, j], ms) for j in range(15)])            # (15, taus)
        i_min = S.argmin(1)
        res = dict(records=int(n), tau0_s=round(float(tau0), 5), tau_s=np.round(tau, 4).tolist(),
                   adev_uT=np.round(S, 4).tolist(),
                   median_xy_uT=np.round(np.median(S.reshape(5, 3, -1)[:, :2].reshape(10, -1), 0), 4).tolist(),
                   median_z_uT=np.round(np.median(S.reshape(5, 3, -1)[:, 2], 0), 4).tolist(),
                   tau_min_s=np.round(tau[i_min], 2).tolist(), adev_min_uT=np.round(S[np.arange(15), i_min], 4).tolist(),
                   at_50ms_uT=[round(at_tau(tau, s, 0.05), 3) for s in S], at_3s_uT=[round(at_tau(tau, s, 3.0), 3) for s in S],
                   at_max_uT=np.round(S[:, -1], 3).tolist())
        out["boards"][name] = res
        lab = [f"{c}{ax}" for c in range(5) for ax in AXES]
        print(f"{name}: {n} records, tau0 {tau0 * 1e3:.2f} ms, {len(ms)} averaging times to {tau[-1] / 3600:.2f} h "
              f"({time.time() - tc:.0f} s)")
        print(f"   median over x and y: {res['median_xy_uT'][0]:.2f} uT at tau0, "
              f"{at_tau(tau, np.array(res['median_xy_uT']), 0.05):.2f} at 50 ms, "
              f"{at_tau(tau, np.array(res['median_xy_uT']), 3.0):.2f} at 3 s, least "
              f"{min(res['median_xy_uT']):.3f} at {tau[int(np.argmin(res['median_xy_uT']))]:.1f} s, "
              f"{res['median_xy_uT'][-1]:.2f} at {tau[-1]:.0f} s")
        print(f"   median over z:       {res['median_z_uT'][0]:.2f} uT at tau0, "
              f"{at_tau(tau, np.array(res['median_z_uT']), 0.05):.2f} at 50 ms, "
              f"{at_tau(tau, np.array(res['median_z_uT']), 3.0):.2f} at 3 s, least "
              f"{min(res['median_z_uT']):.3f} at {tau[int(np.argmin(res['median_z_uT']))]:.1f} s, "
              f"{res['median_z_uT'][-1]:.2f} at {tau[-1]:.0f} s")
        print("   least deviation by component: " + ", ".join(
            f"{lab[j]} {res['adev_min_uT'][j]:.2f} uT at {res['tau_min_s'][j]:.0f} s" for j in range(15)))
        del B
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
