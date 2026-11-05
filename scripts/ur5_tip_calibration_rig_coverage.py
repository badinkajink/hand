#!/usr/bin/env python3
"""Can the inverted rig's presses widen the calibration of v3 IDs 1 to 3 (owner, 2026-11-04 23:1x: "can we see if the
inverted rig presses, while they did not help the ids 4-6, can be used to improve the coverage of ids 1-3?").

Part 1, v3 ID 3, the one tip pressed on both rigs (tip mode on 10-30; seat 1 of the inverted rig on 11-04, session
20261104-145215-id3pts11). Coverage: the rig's contact points against the nearest tip-mode point, and its loads' tilt
|f_t| / |f_n|. Then the board GP (ur5_tip_calibration_board_gp) with a fifth of the joint point groups held out (4 seeds),
trained on the tip-mode holds alone, on both rigs' holds, and on both with the rig's holds moved by the rigs' offset
(the tip-mode GP's mean error on the training groups' rig holds), tested on held-out tip-mode holds, rig holds at shared
points (within 1 mm of a tip-mode point) and rig holds at new points or tilted loads.

Part 2, IDs 4 to 6 for IDs 1 to 3. Transfer: a GP on one tip's holds predicting another tip's. Pooling: tip k's training
groups with IDs 4 to 6's holds, each tip's holds marked by a one-hot input of weight lam (in units of the board's
standardized components), tested on tip k's held-out tip-mode holds and, for ID 3, on its rig holds.

Location is the error along the surface, direction the angle between predicted and measured force, magnitude the force
error over the force, all medians over holds of 1 N and over.

    .venv/bin/python scripts/ur5_tip_calibration_rig_coverage.py [--part 1|2|3|123]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration_board_gp as BG  # noqa: E402
import ur5_tip_calibration_density_curve as DC  # noqa: E402
import ur5_tip_calibration_v3_data_curve as V  # noqa: E402

HD = "docs/experiments/20261030-inverted_rig_seat_block/data/holds/"
K4 = ["press", "slide", "roll", "twist"]
K6 = ["press", "sweep", "offset", "slide", "roll", "twist"]
RIG = {3: (HD + "20261104-145215-id3pts11",),
       4: (HD + "20261102-220337-s3v4", HD + "20261103-202017-rod32_dense"),
       5: (HD + "20261102-182710-s2v5", HD + "20261103-144808-s2sph2", HD + "20261103-172647-rod21_dense"),
       6: (HD + "20261102-145010-s1v6b", HD + "20261103-122020-s1sph2b", HD + "20261103-214753-rod13_dense")}
OUT = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261104-v3_rig_coverage_ids123"
FLOOR = V.PR.floors(0)[:15]


def load(sessions, kinds, src):
    V.KINDS[:] = kinds
    H = V.load(list(sessions))
    H["src"] = np.full(len(H["p"]), src)
    H["id"] = np.char.add(f"{src}|", H["id"].astype(str))
    return H


def cat(*Hs):
    keys = ("d_board", "p", "n", "f", "tn", "kind", "src", "id")
    H = {k: np.concatenate([h[k] for h in Hs]) for k in keys}
    H["group"] = DC.groups(H)
    return H


def tilt(H):
    fm = np.linalg.norm(H["f"], axis=1)
    fn = np.abs(np.sum(H["f"] * H["n"], 1))
    return np.sqrt(np.maximum(fm ** 2 - fn ** 2, 0)) / (fn + 1e-9)


def errs(H, P, k):
    """Medians over the holds k of 1 N and over: location mm, direction deg, magnitude %, and the holds counted."""
    fm = np.linalg.norm(H["f"][k], axis=1)
    hi = fm >= 1.0
    dp = P[:, :3] - H["p"][k]
    n = H["n"][k]
    loc = np.linalg.norm(dp - np.sum(dp * n, 1)[:, None] * n, axis=1)
    c = np.sum(P[:, 3:6] * H["f"][k], 1) / (np.linalg.norm(P[:, 3:6], axis=1) * fm + 1e-12)
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    mag = np.abs(np.linalg.norm(P[:, 3:6], axis=1) - fm) / (fm + 1e-12)
    if not hi.any():
        return dict(n=0)
    return dict(n=int(hi.sum()), location_mm=round(float(np.median(loc[hi])), 3),
                direction_deg=round(float(np.median(ang[hi])), 2), magnitude_pct=round(float(np.median(mag[hi])) * 100, 1))


def Y_of(H):
    return np.c_[H["p"], H["f"], H["tn"]]


def fit(X, Y, floor=FLOOR):
    return BG.fit(X, Y, floor)


def part1(out):
    T = load(V.TIPS[3], K4, "tip")
    I = load(RIG[3], K6, "rig")
    H = cat(T, I)
    tip, rig = H["src"] == "tip", H["src"] == "rig"
    tree = cKDTree(np.unique(np.round(H["p"][tip], 3), axis=0))
    dist = tree.query(H["p"])[0]
    tl = tilt(H)
    new = rig & ((dist >= 1.0) | (tl >= 0.3))
    shared = rig & ~new
    out["coverage"] = dict(
        tip_holds=int(tip.sum()), rig_holds=int(rig.sum()), groups=int(len(np.unique(H["group"]))),
        rig_points_over_1mm_from_tip_points=int((rig & (dist >= 1.0)).sum()),
        rig_dist_mm=dict(median=round(float(np.median(dist[rig])), 2), p90=round(float(np.percentile(dist[rig], 90)), 2)),
        tilt_over_0p3=dict(tip=round(float((tl[tip] >= 0.3).mean()), 3), rig=round(float((tl[rig] >= 0.3).mean()), 3)),
        rig_new=int(new.sum()), rig_shared=int(shared.sum()))
    print("coverage", json.dumps(out["coverage"]), flush=True)
    Y = Y_of(H)
    G0 = np.unique(H["group"])
    rows = {k: {t: [] for t in ("tip", "rig_shared", "rig_new")} for k in ("tip_only", "both", "both_shifted")}
    for seed in range(4):
        rng = np.random.default_rng(700 + seed)
        G = rng.permutation(G0)
        test = np.isin(H["group"], G[: len(G) // 5])
        tr_t, tr_r = ~test & tip, ~test & rig
        m = fit(H["d_board"][tr_t], Y[tr_t])
        off = np.zeros(Y.shape[1])
        Pr = BG.predict(m, H["d_board"][tr_r])
        off[:3] = (Pr[:, :3] - Y[tr_r, :3]).mean(0)                       # the rigs' offset seen by the tip-mode GP
        Ys = Y.copy()
        Ys[rig, :3] += off[:3]
        models = {"tip_only": m, "both": fit(H["d_board"][~test], Y[~test]),
                  "both_shifted": fit(H["d_board"][~test], Ys[~test])}
        for name, mm in models.items():
            for tag, sel in (("tip", test & tip), ("rig_shared", test & shared), ("rig_new", test & new)):
                if sel.any():
                    P = BG.predict(mm, H["d_board"][sel])
                    if name == "both_shifted" and tag != "tip":
                        P[:, :3] -= off[:3]                                 # back into the rig's own frame
                    rows[name][tag].append(errs(H, P, sel))
        print(f"part 1 seed {seed}: offset {np.round(off[:3], 3).tolist()} mm; " + "; ".join(
            f"{n} {t} {rows[n][t][-1].get('location_mm')}" for n in rows for t in rows[n] if rows[n][t]), flush=True)
    out["id3"] = {n: {t: {k: (float(np.median([r[k] for r in v])) if v else None)
                          for k in ("n", "location_mm", "direction_deg", "magnitude_pct")} for t, v in d.items()}
                  for n, d in rows.items()}


def part2(out, lams=(0.5, 1.0, 2.0), n_other=1500):
    rng = np.random.default_rng(0)
    tipsH = {k: load(V.TIPS[k], K4, f"tip{k}") for k in (1, 2, 3)}
    rigH = {k: load(RIG[k], K6, f"rig{k}") for k in (4, 5, 6)}
    for k, h in rigH.items():                                               # a subsample of each rig tip's holds
        i = rng.choice(len(h["p"]), min(n_other, len(h["p"])), replace=False)
        rigH[k] = {f: v[i] for f, v in h.items()}
    rig3 = load(RIG[3], K6, "rig3")
    tr = {}
    for j, hj in list(rigH.items()) + list(tipsH.items()):                  # transfer: GP on tip j, tested on tip k
        i = rng.choice(len(hj["p"]), min(3000, len(hj["p"])), replace=False)
        m = fit(hj["d_board"][i], Y_of(hj)[i])
        for k, hk in tipsH.items():
            if k == j:
                continue
            tr[f"{j}->{k}"] = errs(hk, BG.predict(m, hk["d_board"]), np.ones(len(hk["p"]), bool))
        print(f"transfer from ID {j}: " + "; ".join(f"{a} {b.get('location_mm')}" for a, b in tr.items()
                                                    if a.startswith(f"{j}->")), flush=True)
    out["transfer"] = tr
    pool = {}
    others = [rigH[j] for j in (4, 5, 6)]
    for k, hk in tipsH.items():
        H = cat(hk, *others, *( [rig3] if k == 3 else []))
        src = H["src"]
        own, rig3m = src == f"tip{k}", src == "rig3"
        Y = Y_of(H)
        X0 = H["d_board"]
        G0 = np.unique(H["group"][own])
        res = {}
        for seed in range(2):
            g = np.random.default_rng(800 + seed).permutation(G0)
            test = np.isin(H["group"], g[: len(g) // 5])
            te_own = test & own
            for lam in (None,) + tuple(lams):
                if lam is None:
                    trm = ~test & own
                    X = X0
                    fl = FLOOR
                else:
                    trm = ~test & ~rig3m
                    srcs = ["tip%d" % k, "rig4", "rig5", "rig6"]
                    oh = np.stack([(src == s).astype(float) for s in srcs], 1) * lam
                    sx = V.PR.scale(X0[own], FLOOR)
                    X = np.c_[X0 / sx, oh]                                 # board in its own tip's units, then tips apart
                    fl = np.r_[np.full(15, 1e-9), np.ones(4)]
                m = BG.fit(X[trm], Y[trm], fl)
                key = "own" if lam is None else f"pooled_lam{lam:g}"
                res.setdefault(key, {"tip": []})["tip"].append(errs(H, BG.predict(m, X[te_own]), te_own))
                if k == 3:
                    res[key].setdefault("rig3", []).append(errs(H, BG.predict(m, X[rig3m]), rig3m))
                print(f"pool ID {k} seed {seed} {key}: tip {res[key]['tip'][-1].get('location_mm')}"
                      + (f", rig3 {res[key]['rig3'][-1].get('location_mm')}" if k == 3 else ""), flush=True)
        pool[str(k)] = {key: {t: {m_: float(np.median([r[m_] for r in v])) for m_ in
                                  ("location_mm", "direction_deg", "magnitude_pct")} for t, v in d.items()}
                        for key, d in res.items()}
    out["pool"] = pool


def part3(out, lams=(0.5, 1.0), n_other=1500):
    """ID 3's tip-mode training groups with IDs 4 to 6's rig holds (none of ID 3's own rig holds), tested on ID 3's rig
    holds at new points (1 mm or more from every tip-mode point, or tilt 0.3 and over) and at shared points: what the
    other tips' rig presses alone give the pad's mount end."""
    rng = np.random.default_rng(0)
    T = load(V.TIPS[3], K4, "tip3")
    others = []
    for j in (4, 5, 6):
        h = load(RIG[j], K6, f"rig{j}")
        i = rng.choice(len(h["p"]), min(n_other, len(h["p"])), replace=False)
        others.append({f: v[i] for f, v in h.items()})
    R3 = load(RIG[3], K6, "rig3")
    H = cat(T, *others, R3)
    src = H["src"]
    own, rig3m = src == "tip3", src == "rig3"
    dist = cKDTree(np.unique(np.round(H["p"][own], 3), axis=0)).query(H["p"])[0]
    new = rig3m & ((dist >= 1.0) | (tilt(H) >= 0.3))
    shared = rig3m & ~new
    Y, X0 = Y_of(H), H["d_board"]
    sx = V.PR.scale(X0[own], FLOOR)
    srcs = ["tip3", "rig4", "rig5", "rig6"]
    res = {}
    for seed in range(2):
        g = np.random.default_rng(800 + seed).permutation(np.unique(H["group"][own]))
        test = np.isin(H["group"], g[: len(g) // 5])
        for lam in (None,) + tuple(lams):
            if lam is None:
                trm, X, fl = ~test & own, X0, FLOOR
            else:
                trm = ~test & ~rig3m
                X = np.c_[X0 / sx, np.stack([(src == s).astype(float) for s in srcs], 1) * lam]
                fl = np.r_[np.full(15, 1e-9), np.ones(4)]
            m = BG.fit(X[trm], Y[trm], fl)
            key = "own" if lam is None else f"pooled_lam{lam:g}"
            for tag, sel in (("tip", test & own), ("rig3_shared", shared), ("rig3_new", new)):
                res.setdefault(key, {}).setdefault(tag, []).append(errs(H, BG.predict(m, X[sel]), sel))
            print(f"part 3 seed {seed} {key}: " + "; ".join(f"{t} {v[-1].get('location_mm')}" for t, v in res[key].items()),
                  flush=True)
    out["id3_from_ids456"] = {key: {t: {m_: float(np.median([r[m_] for r in v])) for m_ in
                                        ("n", "location_mm", "direction_deg", "magnitude_pct")} for t, v in d.items()}
                              for key, d in res.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", default="12")
    a = ap.parse_args()
    out = {"what": __doc__.split("\n\n")[1]}
    p = OUT.with_suffix(".json")
    if p.exists():
        out.update(json.loads(p.read_text()))
    t0 = time.time()
    if "1" in a.part:
        part1(out)
        p.write_text(json.dumps(out, indent=1))
    if "2" in a.part:
        part2(out)
        p.write_text(json.dumps(out, indent=1))
    if "3" in a.part:
        part3(out)
        p.write_text(json.dumps(out, indent=1))
    print(f"done in {time.time() - t0:.0f} s ->", p)


if __name__ == "__main__":
    main()
