#!/usr/bin/env python3
"""How well the three v1 tips sense, on their UR5 presses (owner, 2026-10-26 19:2x: "run a thorough set of analysis on
the v1 id 3 data that we did get ... and see how good (ideally) the sensing for the three v1 fingertips are"). ID 1:
cb1t and cb1r; ID 2: cb2t; ID 3: cb3t (passes 1 and 2a, the 24 parity actions and R26; its passes 2b to 6 did not run).

For each tip's sphere presses, the location error (mm, along the surface) and the force error (|F_est - F|, N) of:

  physical   the viewer's estimator (magnets fitted to the tip's own rest, the board's chip frames kept);
  own        a GP (scripts/ur5_tip_calibration_cb1r_checks.py) trained on the tip's own presses, five folds grouped by
             action; with too few targets it is limited by density, not by the tip (density curve: 17 groups 1.15 mm,
             84 groups 0.36 mm on ID 2);
  own, thinned  the same with every tip's training thinned to ID 3's number of actions, so that the tips compare at one
             density (eight draws);
  others     a GP trained on the other two tips' presses alone (transfer to a tip it has not seen);
  pooled     the other tips' presses plus this tip's other folds: the best the data allow.

Errors are medians, at 1.5 N and over and under 1.5 N. With --by-force, also over 5 N (ID 3's R26 presses reach 8 N).

    /usr/bin/python3 scripts/ur5_tip_calibration_v1_tips.py [--out data/20261026-v1_tips.json]
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

D = K.EXP / "data"
TIPS = {"ID 1": ("20261023-203938-cb1t", "20261025-114653-cb1r"), "ID 2": ("20261025-152920-cb2t",),
        "ID 3": ("20261026-134522-cb3t",)}


def presses(sessions):
    H = K.cat(*[K.load(D / s) for s in sessions])
    m = (H["kind"] == "press") & np.isfinite(H["d_board"]).all(1) & np.isfinite(H["p"]).all(1) & np.isfinite(H["f"]).all(1)
    return K.sub(H, m)


def folds(H, k=5, rng=None):
    a = H["id"].astype(str)
    u = np.unique(a)
    if rng is not None:
        u = rng.permutation(u)
    f = {x: i % k for i, x in enumerate(u)}
    return np.array([f[x] for x in a])


def score(H, Y):
    e = K.loc_err(H, Y[:, :3])
    ef = np.linalg.norm(Y[:, 3:] - H["f"], axis=1)
    hi, top = H["f_n"] >= 1.5, H["f_n"] >= 5.0
    out = dict(over_1p5N=K.med(e[hi]), under_1p5N=K.med(e[~hi]), force_err_N_over_1p5N=K.med(ef[hi]),
               force_err_N_under_1p5N=K.med(ef[~hi]), n_over=int(hi.sum()), n_under=int((~hi).sum()))
    if top.any():
        out.update(over_5N=K.med(e[top]), force_err_N_over_5N=K.med(ef[top]), n_over_5N=int(top.sum()))
    return out


def cv(train, H, keep=None, rng=None):
    """Predictions for H, five folds over its actions, each trained on train plus H's other folds (or a random keep of
    H's other actions)."""
    fk = folds(H, rng=rng)
    Y = np.full((len(H["id"]), 6), np.nan)
    for k in range(5):
        m = fk == k
        if not m.any():
            continue
        tr = K.sub(H, ~m)
        if keep is not None:
            acts = np.unique(tr["id"].astype(str))
            pick = set(rng.choice(acts, size=min(keep, len(acts)), replace=False))
            tr = K.sub(tr, np.isin(tr["id"].astype(str), list(pick)))
        Y[m] = K.gp(K.cat(train, tr) if train is not None else tr).predict(H["d_board"][m])
    return Y


def run(a):
    P = {t: presses(s) for t, s in TIPS.items()}
    n3 = int(len(np.unique(P["ID 3"]["id"].astype(str))) * 4 // 5)    # ID 3's training actions in a five-fold split
    res = {"n_actions": {t: int(len(np.unique(H["id"].astype(str)))) for t, H in P.items()},
           "n_holds": {t: int(len(H["id"])) for t, H in P.items()}, "thinned_to_actions": n3, "tips": {}}
    for t, H in P.items():
        others = K.cat(*[P[o] for o in P if o != t])
        r = dict(physical=score(H, np.c_[H["est_p"], H["est_f"]]), own=score(H, cv(None, H)),
                 others=score(H, K.gp(others).predict(H["d_board"])), pooled=score(H, cv(others, H)))
        draws = [score(H, cv(None, H, keep=n3, rng=np.random.default_rng(s))) for s in range(a.draws)]
        r["own_thinned"] = {k: K.med([d[k] for d in draws if k in d]) for k in draws[0]}
        res["tips"][t] = r
        print(t, json.dumps({k: {q: round(float(v), 2) for q, v in d.items()} for k, d in r.items()}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--out", default=str(D / "20261026-v1_tips.json"))
    run(ap.parse_args())
