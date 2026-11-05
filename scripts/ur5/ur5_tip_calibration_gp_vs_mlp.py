#!/usr/bin/env python3
"""The contact viewer's board GP against an MLP ensemble on all six v3 tips' holds (owner, 2026-11-05 11:5x: "with this
much data, should we also revisit the MLP? i do like the uncertainty measurement of the GP, which enables combination
with the physical model").

Data: the presses and slides of v3 IDs 1 to 3 in tip mode and of IDs 3 to 6 on the inverted rig, 21,138 holds. The GP
(ur5_tip_calibration_board_gp, the viewer's recipe) needs O(N^2) memory, so it trains on at most 1,500 holds of each of
the seven sets, as the viewer's model of 2026-11-05 does. The MLP ensemble (ur5_tip_calibration_predict.MLP: five
members, two hidden layers of `hidden` SiLU units, steps chosen on a random fifth of the training rows) trains on every
training hold. Neither model is told which tip a reading comes from.

Splits, each tested on held-out holds of 1 N and over:
  random s0, s1   a fifth of every set's point groups held out
  gaps            the far end of IDs 4 to 6 (contact point z over 14 mm) and the mount end of ID 3's rig holds (1 mm or
                  more from every ID 3 tip-mode point) held out: parts of a pad that its own training lacks
Uncertainty: the GP's weight w (wrench_viewer.BoardGP: 1 - the posterior variance from the 256 holds of largest kernel
weight) and the MLP members' spread (the norm of the standard deviation of their predicted contact points, mm), each
scored by its Spearman correlation with the location error over the test holds.

    ~/.claude/bin/resguard.sh run --mem 6G --cpu 500 -- env OPENBLAS_NUM_THREADS=4 taskset -c 4-7 .venv/bin/python \\
        scripts/ur5/ur5_tip_calibration_gp_vs_mlp.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401
import ur5_tip_calibration_board_gp as BG  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402
import ur5_tip_calibration_rig_coverage as RC  # noqa: E402
import ur5_tip_calibration_v3_data_curve as V  # noqa: E402

KINDS = ["press", "slide"]
OUT = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261105-v3_gp_vs_mlp.json"


def cat(hs):
    return {f: np.concatenate([h[f] for h in hs]) for f in ("d_board", "p", "n", "f", "src", "group")}


def gp_weight(m, X, k=256):
    """The viewer's weight w for each row of X (wrench_viewer.BoardGP.predict)."""
    Z, sq = m["Z"], (m["Z"] ** 2).sum(1)
    out = np.empty(len(X))
    for i, x in enumerate(X):
        q = (x - m["mx"]) / m["sx"]
        kv = np.exp(-0.5 * ((Z - q) ** 2).sum(1) / m["ell"] ** 2)
        s = np.argpartition(-kv, k)[:k]
        Ks = np.exp(-0.5 * np.maximum(sq[s][:, None] + sq[s][None] - 2 * Z[s] @ Z[s].T, 0) / m["ell"] ** 2)
        out[i] = np.clip(kv[s] @ cho_solve(cho_factor(Ks + m["nz"] ** 2 * np.eye(k), lower=True), kv[s]), 0, 1)
    return out


def mlp_members(mlp, X):
    import torch
    Z = torch.as_tensor((X - mlp.mx) / mlp.sx, dtype=torch.float32)
    with torch.no_grad():
        return np.stack([net(Z).numpy() * mlp.sy + mlp.my for net in mlp.nets])


def loc_err(H, P, sel):
    dp = P[:, :3] - H["p"][sel]
    n = H["n"][sel]
    return np.linalg.norm(dp - np.sum(dp * n, 1)[:, None] * n, axis=1)


def main():
    sets = {f"tip{k}": RC.load(V.TIPS[k], KINDS, f"tip{k}") for k in (1, 2, 3)}
    sets.update({f"rig{j}": RC.load(RC.RIG[j], KINDS, f"rig{j}") for j in (3, 4, 5, 6)})
    for s, h in sets.items():
        h["group"] = np.char.add(f"{s}|", h["group"].astype(str))
    H = cat(list(sets.values()))
    N = len(H["p"])
    d3 = cKDTree(np.unique(np.round(sets["tip3"]["p"], 3), axis=0)).query(H["p"])[0]
    src = H["src"]
    gaps = (np.isin(src, ["rig4", "rig5", "rig6"]) & (H["p"][:, 2] > 14.0)) | ((src == "rig3") & (d3 >= 1.0))
    splits = []
    for seed in range(2):
        rng = np.random.default_rng(1100 + seed)
        test = np.zeros(N, bool)
        for s in sets:
            g = rng.permutation(np.unique(H["group"][src == s]))
            test |= np.isin(H["group"], g[: len(g) // 5])
        splits.append((f"random s{seed}", test))
    splits.append(("gaps", gaps))
    Y = np.c_[H["p"], H["f"]]
    out = {"what": __doc__.split("\n\n")[1], "holds": int(N), "rows": []}
    t0 = time.time()
    for name, test in splits:
        rng = np.random.default_rng(7)
        tr_all = np.flatnonzero(~test)
        tr_gp = np.concatenate([rng.permutation(tr_all[src[tr_all] == s])[:1500] for s in sets])
        te = np.flatnonzero(test)
        row = {"split": name, "n_test": int(len(te)), "n_train_gp": int(len(tr_gp)), "n_train_mlp": int(len(tr_all))}
        preds = {}
        t = time.time()
        m = BG.fit(H["d_board"][tr_gp], Y[tr_gp], RC.FLOOR)
        preds["gp"] = (BG.predict(m, H["d_board"][te]), 1.0 - gp_weight(m, H["d_board"][te]), time.time() - t)
        for hidden in (64, 128):
            t = time.time()
            perm = rng.permutation(tr_all)                     # the step count is chosen on the last fifth of the rows
            mlp = PR.MLP(hidden=hidden).fit(H["d_board"][perm], Y[perm], RC.FLOOR)
            M = mlp_members(mlp, H["d_board"][te])
            preds[f"mlp{hidden}"] = (M.mean(0), np.linalg.norm(M[:, :, :3].std(0), axis=1), time.time() - t)
            row[f"mlp{hidden}_steps"] = int(mlp.steps)
        for model, (P, unc, sec) in preds.items():
            e = loc_err(H, P, te)
            fm = np.linalg.norm(H["f"][te], axis=1)
            hi = fm >= 1.0
            r = {"seconds": round(sec, 1), "all": RC.errs({k: v[te] for k, v in H.items()}, P, np.ones(len(te), bool)),
                 "uncertainty_spearman_vs_location": round(float(spearmanr(unc[hi], e[hi])[0]), 3)}
            for s in sets:
                k = src[te] == s
                if (k & hi).sum() >= 20:
                    r[s] = RC.errs({f: v[te] for f, v in H.items()}, P[k], k)
            row[model] = r
            print(f"{name} {model}: all {r['all'].get('location_mm')} mm {r['all'].get('direction_deg')} deg "
                  f"{r['all'].get('magnitude_pct')} %, uncertainty rho {r['uncertainty_spearman_vs_location']}; "
                  + ", ".join(f"{s} {r[s].get('location_mm')}" for s in sets if s in r)
                  + f"; {sec:.0f} s (total {time.time() - t0:.0f} s)", flush=True)
        out["rows"].append(row)
        OUT.write_text(json.dumps(out, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
