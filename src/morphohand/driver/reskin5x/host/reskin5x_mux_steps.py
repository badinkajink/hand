#!/usr/bin/env python3
"""Field changes shared by the boards of a reskin5x_mux.py log at rest: steps of each board's uniform field (the fit of
reskin5x_mux_rest.py, one field vector read by every chip through its rotation) between 1 s bins, and the steps that
every board shows within a few seconds of the others. A field from outside the pads changes all boards at once; a pad
that shifts in its seat changes one board, mostly in the part that the uniform field leaves.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_mux_rest.py LOG.bin --ref 19:35:00 19:40:00
        --since 19:35 --until 04:00 --bin-s 1 --series rest_1s.npz
    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_mux_steps.py rest_1s.npz [--min-uT 3]
        [--within-s 2] [--out steps.json]

step    the change of a board's uniform field across two bins, |g(k+1) - g(k-1)| (uT), above --min-uT; consecutive bins
        over it make one step, placed at the bin of the largest change, whose size is that change; 'net' is the
        difference of the medians over the 10 s after and before it (2 s left out on each side), which is near zero for a
        spike that returns
shared  a step of every board within --within-s of the first board's
left    the same steps in the part of the change that the uniform field leaves (norm over the 15 components), above
        --left-min-uT: the chips' noise alone changes that part by about 2.2 uT over 2 s (median, 1 s bins)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def grid(t, x, dt):
    """x (k, ...) at times t on a regular grid of spacing dt from t[0], NaN where a bin is missing."""
    k = np.round((t - t[0]) / dt).astype(int)
    g = np.full((k[-1] + 1,) + x.shape[1:], np.nan)
    g[k] = x
    return t[0] + dt * np.arange(k[-1] + 1), g


def steps(x, min_uT, dt):
    """Steps of the series x (k, d) on a regular grid: (index, size, net) of each run of |x(k+1) - x(k-1)| > min_uT."""
    c = np.full(len(x), np.nan)
    c[1:-1] = np.linalg.norm(x[2:] - x[:-2], axis=1)
    over = np.nan_to_num(c, nan=0.0) > min_uT
    edges = np.flatnonzero(np.diff(np.r_[0, over.astype(int), 0]))
    out = []
    w, gap = int(round(10 / dt)), int(round(2 / dt))
    for s, e in zip(edges[::2], edges[1::2]):
        k = s + int(np.nanargmax(c[s:e]))
        before = x[max(0, s - gap - w):max(0, s - gap)]
        after = x[e + gap:e + gap + w]
        net = (float(np.linalg.norm(np.nanmedian(after, 0) - np.nanmedian(before, 0)))
               if len(before) and len(after) and np.isfinite(before).any() and np.isfinite(after).any() else float("nan"))
        out.append((k, float(c[k]), net))
    return out, c


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("series")
    ap.add_argument("--min-uT", type=float, default=3.0)
    ap.add_argument("--within-s", type=float, default=2.0)
    ap.add_argument("--left-min-uT", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = np.load(a.series)
    names = [str(v) for v in z["names"]]
    U = z["U"]
    nb = len(names)
    t0 = max(float(z[f"{b}_t"][0]) for b in range(nb))
    t1 = min(float(z[f"{b}_t"][-1]) for b in range(nb))
    dt = float(np.median(np.diff(z["0_t"])))
    hours = (t1 - t0) / 3600
    per, cs, out = [], [], dict(series=str(a.series), min_uT=a.min_uT, within_s=a.within_s, bin_s=dt, left_min_uT=a.left_min_uT,
                                since=time.strftime("%H:%M:%S", time.localtime(t0)),
                                until=time.strftime("%H:%M:%S", time.localtime(t1)), hours=round(hours, 3), boards={})
    for b, name in enumerate(names):
        tb = z[f"{b}_t"]
        m = (tb >= t0 - 0.5) & (tb <= t1 + 0.5)
        tg, G = grid(tb[m], z[f"{b}_g"][m].astype(float), dt)
        _, Dg = grid(tb[m], z[f"{b}_d"][m].astype(float), dt)
        L = Dg - G @ U.T
        st, c = steps(G, a.min_uT, dt)
        lst, cl = steps(L, a.left_min_uT, dt)
        per.append((tg, st))
        cs.append(c)
        sizes = np.array([s for _, s, _ in st]) if st else np.zeros(0)
        res = dict(bins=int(np.isfinite(G[:, 0]).sum()), steps=len(st), per_hour=round(len(st) / hours, 2),
                   size_median_uT=round(float(np.median(sizes)), 2) if len(sizes) else None,
                   size_max_uT=round(float(sizes.max()), 2) if len(sizes) else None,
                   change_2s_median_uT=round(float(np.nanmedian(c)), 3), change_2s_p99_uT=round(float(np.nanpercentile(c, 99)), 3),
                   left_steps=len(lst), left_change_2s_median_uT=round(float(np.nanmedian(cl)), 3),
                   left_change_2s_p99_uT=round(float(np.nanpercentile(cl, 99)), 3),
                   left_step_times=[time.strftime("%H:%M:%S", time.localtime(tg[k])) for k, _, _ in lst[:50]])
        out["boards"][name] = res
        print(f"{name}: {res['bins']} bins of {dt:g} s; uniform field over 2 s changes by median {res['change_2s_median_uT']} uT, "
              f"p99 {res['change_2s_p99_uT']}; {len(st)} steps over {a.min_uT:g} uT ({res['per_hour']} an hour), size median "
              f"{res['size_median_uT']} uT, max {res['size_max_uT']}; the part left: median {res['left_change_2s_median_uT']} "
              f"uT, p99 {res['left_change_2s_p99_uT']}, {len(lst)} steps over {a.left_min_uT:g} uT {res['left_step_times'][:8]}")
    # shared: a step of the first board matched by a step of every other board within within_s
    shared = []
    for k0, s0, n0 in per[0][1]:
        tk = per[0][0][k0]
        match = [(tk, s0, n0)]
        for tg, st in per[1:]:
            near = [(tg[k], s, n) for k, s, n in st if abs(tg[k] - tk) <= a.within_s]
            if not near:
                break
            match.append(min(near, key=lambda v: abs(v[0] - tk)))
        if len(match) == nb:
            shared.append(dict(t=time.strftime("%H:%M:%S", time.localtime(tk)),
                               size_uT=[round(s, 2) for _, s, _ in match], net_uT=[round(n, 2) for _, _, n in match],
                               lag_s=[round(v[0] - tk, 1) for v in match]))
    S = np.array([e["size_uT"] for e in shared]).reshape(-1, nb)
    out["shared"] = dict(count=len(shared), per_hour=round(len(shared) / hours, 2),
                         size_median_uT=np.round(np.median(S, 0), 2).tolist() if len(S) else None,
                         size_max_uT=np.round(S.max(0), 2).tolist() if len(S) else None, events=shared)
    print(f"shared by all {nb} boards within {a.within_s:g} s: {len(shared)} steps over {hours:.2f} h "
          f"({len(shared) / hours:.2f} an hour)" + (f"; size median {np.round(np.median(S, 0), 1)} uT, max "
                                                    f"{np.round(S.max(0), 1)}" if len(S) else ""))
    for e in shared[:20]:
        print(f"   {e['t']} size {e['size_uT']} uT, net {e['net_uT']} uT, lag {e['lag_s']} s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
