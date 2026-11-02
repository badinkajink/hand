#!/usr/bin/env python3
"""Smoothness metrics for gain_sweep.py raw output: python3 smoothness.py RAW.txt [...]"""
import collections
import sys

import numpy as np

for f in sys.argv[1:]:
    print(f)
    rows = [l.strip().split(",") for l in open(f) if l[:1].isdigit() and l.count(",") == 5]
    d = collections.defaultdict(list)
    order = []
    for t, tag, ph, tg, p, ld in rows:
        if ph in ("sine", "hold"):
            key = (tag, ph) if (tag, ph) not in d or not order or order[-1][0] == tag else (tag + "'", ph)
            if key[0] not in [o[0] for o in order]:
                order.append(key)
            d[key].append((float(t), float(tg), float(p), float(ld)))
    for tag in dict.fromkeys(o[0] for o in order):
        a = np.array(d[(tag, "sine")]); t, tg, p, ld = a.T
        good = np.abs(np.diff(p, prepend=p[0])) < 5          # drop single-sample read tears
        t, tg, p = t[good], tg[good], p[good]
        e = tg - p
        tt = np.arange(t[0], t[-1], 0.01)
        v = np.gradient(np.interp(tt, t, p), tt)
        vt = np.gradient(np.interp(tt, t, tg), tt)
        mv = np.abs(vt) > 10
        vs = np.convolve(v, np.ones(25) / 25, "same")
        print(f"  {tag:10s} track rms {np.sqrt(np.mean(e**2)):.2f} max {np.abs(e).max():.2f} deg | "
              f"stalled while moving {100 * np.mean(np.abs(v[mv]) < 2):.0f}% | "
              f"velocity roughness {np.sqrt(np.mean((v - vs)[50:-50] ** 2)):.1f} deg/s | "
              f"velocity sign flips {np.sum(np.diff(np.sign(v[mv])) != 0) / (mv.sum() * 0.01):.1f}/s")
