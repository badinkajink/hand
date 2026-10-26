#!/usr/bin/env python3
"""Location error against the density of press targets on one tip (owner, 2026-10-26, through the lab session: "seems
like increased grid points really helped. do you think even more would be beneficial? would be good to know where we
begin to see diminishing returns").

A session's press holds are grouped by point: actions whose median contact points lie within 0.75 mm of each other (a
grid target with its fast-grid and rate copies, the references) form one group. Each seed holds out a fifth of the
groups and trains a GP (scripts/ur5_tip_calibration_cb1r_checks.py) on growing, nested fractions of the others.

  fraction  median location error (mm, along the surface) at the held-out groups against the fraction of the other
            groups in training, at 1.5 N and over and under 1.5 N, with the median distance from a held-out hold to
            the nearest training hold and the spread of the seeds' medians.
  distance  the same errors binned by that distance, over every seed and fraction.
  floor     the error at a distance of about 0: the fast-grid presses (0.6 mm/s) predicted by a GP trained on every other
            press, which includes the grid's presses at the same targets (0.15 mm/s) hours earlier.
  thin      the share of the FE surface's points on the face, edges and flanks, by side, farther than 1.5 and 2 mm from
            every press hold of the session.

    .venv/bin/python scripts/ur5_tip_calibration_density_curve.py [--session 20261025-152920-cb2t] [--seeds 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration_cb1r_checks as K  # noqa: E402

FRACTIONS = (0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
BINS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 99.0)


def groups(H, tol=0.75):
    ids = H["id"].astype(str)
    acts = np.unique(ids)
    c = np.array([np.median(H["p"][ids == a], axis=0) for a in acts])
    parent = list(range(len(acts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    d = np.linalg.norm(c[:, None] - c[None], axis=2)
    for i, j in zip(*np.nonzero(np.triu(d < tol, 1))):
        parent[find(i)] = find(j)
    g = {a: find(k) for k, a in enumerate(acts)}
    return np.array([g[a] for a in ids])


def nearest(P, Q):
    return np.array([np.min(np.linalg.norm(Q - x, axis=1)) for x in P])


def bands(e, hi):
    return dict(over_1p5N=K.med(e[hi]), under_1p5N=K.med(e[~hi]))


def run(a):
    H = K.load(K.EXP / "data" / a.session)
    P = K.sub(H, H["kind"] == "press")
    g = groups(P)
    ug = np.unique(g)
    hi = P["f_n"] >= 1.5
    rows = []
    for s in range(a.seeds):
        perm = np.random.default_rng(s).permutation(ug)
        nt = len(ug) // 5
        test, pool = perm[:nt], perm[nt:]
        te = np.isin(g, test)
        T = K.sub(P, te)
        for f in FRACTIONS:
            k = max(3, int(round(f * len(pool))))
            tr = np.isin(g, pool[:k])
            e = K.loc_err(T, K.gp(K.sub(P, tr)).predict(T["d_board"])[:, :3])
            d = nearest(T["p"], P["p"][tr])
            rows += [(s, f, k, x, y, h) for x, y, h in zip(e, d, hi[te])]
    R = np.array([r[:5] for r in rows], float)
    rh = np.array([r[5] for r in rows], bool)
    frac = {}
    for f in FRACTIONS:
        m = R[:, 1] == f
        per_seed = [K.med(R[m & (R[:, 0] == s) & rh, 3]) for s in range(a.seeds)]
        frac[f"{f:g}"] = dict(n_groups=float(np.mean(R[m, 2])), nearest_mm=K.med(R[m, 4]), **bands(R[m, 3], rh[m]),
                              over_1p5N_seed_range=[float(np.nanmin(np.array(per_seed, float))), float(np.nanmax(np.array(per_seed, float)))])
    dist = {}
    for lo, up in zip(BINS[:-1], BINS[1:]):
        m = (R[:, 4] >= lo) & (R[:, 4] < up)
        if m.any():
            dist[f"{lo:g}-{up:g}"] = dict(n_over=int((m & rh).sum()), n_under=int((m & ~rh).sum()), **bands(R[m, 3], rh[m]))
    fg = np.char.startswith(P["id"].astype(str), "fast-grid")
    F = K.sub(P, fg)
    e = K.loc_err(F, K.gp(K.sub(P, ~fg)).predict(F["d_board"])[:, :3])
    floor = dict(n=int(fg.sum()), nearest_mm=K.med(nearest(F["p"], P["p"][~fg])), **bands(e, hi[fg]))
    import ur5_tip_calibration_r24_pass3 as P3
    b = np.load(P3.BASIS)
    pts, nrm = b["pts"], b["normals"]
    reg = np.array([P3.region(n) for n in nrm], dtype=object)
    dp = nearest(pts, P["p"])
    thin = {}
    for r in ("face", "edge", "flank"):
        for side, m in (("left", pts[:, 1] < -1.5), ("mid", np.abs(pts[:, 1]) <= 1.5), ("right", pts[:, 1] > 1.5)):
            k = (reg == r) & m
            if k.any():
                thin[f"{r}/{side}"] = dict(n=int(k.sum()), over_1p5mm=float(np.mean(dp[k] > 1.5)), over_2mm=float(np.mean(dp[k] > 2.0)),
                                           max_mm=float(dp[k].max()))
    res = dict(session=a.session, n_press_holds=int(len(P["id"])), n_groups=int(len(ug)), seeds=a.seeds, fraction=frac,
               distance=dist, floor=floor, thin=thin)
    out = K.EXP / "data" / a.session / "density_curve.json"
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", default="20261025-152920-cb2t")
    ap.add_argument("--seeds", type=int, default=8)
    run(ap.parse_args())
