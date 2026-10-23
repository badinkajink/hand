#!/usr/bin/env python3
"""The physical model with a learned correction whose inputs are the physical estimate itself (contact point and normal
force), on the real presses of the UR5 run sphere2: four folds, each holding out every fourth grid press at one offset,
against the GP and the MLP on the raw board change. Such a correction carries over to other magnets in simulation
(scripts/fingertip_magnet_transfer_sim.py); this measures what it gives on real data (owner, 2026-10-23: "the GP/MLP can
just be used to fine-tune the accuracy/precision").

    .venv/bin/python scripts/fingertip_hybrid_cv.py <analysis output folder> [--out hybrid_cv.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import fingertip_wrench_model as W  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402


def run(folder, out):
    H, _, cal0 = PR.load(folder)
    basis = W.load_basis(PR.BASIS)
    ph0 = PR.Physical(cal0, basis)
    est = ph0.estimator()
    allidx = np.arange(len(H["id"]))
    Pv, Fv = ph0.predict(H, allidx)                         # the untrained physical estimate of every hold
    ok = np.all(np.isfinite(Pv), axis=1) & np.isfinite(Fv)
    ids, first = np.unique(H["id"], return_index=True)
    order = ids[np.argsort(H["t"][first])]
    grid = [i for i in order if H["stage"][H["id"] == i][0] == "grid" and H["role"][H["id"] == i][0] != "ref"]
    Xb, Y = PR.features(H, 0), np.c_[H["p"], H["f_n"]]
    Xh = np.c_[Pv, Fv]
    fl_b, fl_h = PR.floors(0), np.array([0.2, 0.2, 0.2, 0.05])
    errs = {m: dict(loc=[], f=[], lv=[]) for m in ("physical", "gp raw", "mlp raw", "physical+gp", "physical+mlp")}
    for k in range(4):
        te = np.nonzero(np.isin(H["id"], grid[k::4]))[0]
        tr = np.nonzero(~np.isin(H["id"], grid[k::4]))[0]
        trh = tr[ok[tr]]
        preds = {"physical": (Pv[te], Fv[te])}
        for m, model in (("gp raw", PR.GP()), ("mlp raw", PR.MLP())):
            Yp = model.fit(Xb[tr], Y[tr], fl_b).predict(Xb[te])
            preds[m] = (PR.snap(est, Yp[:, :3]), Yp[:, 3])
        Rres = np.c_[H["p"][trh] - Pv[trh], H["f_n"][trh] - Fv[trh]]
        for m, model in (("physical+gp", PR.GP()), ("physical+mlp", PR.MLP())):
            C = np.full((len(te), 4), np.nan)
            okt = ok[te]
            C[okt] = model.fit(Xh[trh], Rres, fl_h).predict(Xh[te][okt])
            preds[m] = (PR.snap(est, np.where(np.isfinite(Pv[te]), Pv[te], 0.0) + np.nan_to_num(C[:, :3])), Fv[te] + C[:, 3])
        for m, (P, F) in preds.items():
            d = P - H["p"][te]
            n = H["n"][te]
            errs[m]["loc"] += np.linalg.norm(d - (d * n).sum(1)[:, None] * n, axis=1).tolist()
            errs[m]["f"] += np.abs(F - H["f_n"][te]).tolist()
            errs[m]["lv"] += H["level_N"][te].tolist()
        print(f"fold {k}: {len(te)} held-out holds", flush=True)
    res = {}
    for m, e in errs.items():
        loc, f, lv = np.array(e["loc"]), np.array(e["f"]), np.array(e["lv"])
        res[m] = dict(n=int(len(loc)), loc_median=float(np.nanmedian(loc)), loc_p90=float(np.nanpercentile(loc, 90)),
                      force_median=float(np.nanmedian(f)), loc_median_from_1N=float(np.nanmedian(loc[lv >= 0.99])),
                      loc_median_below_1N=float(np.nanmedian(loc[lv < 0.99])))
        print(f"{m:14s} location {res[m]['loc_median']:.2f} mm (from 1 N {res[m]['loc_median_from_1N']:.2f}, below "
              f"{res[m]['loc_median_below_1N']:.2f}), force {res[m]['force_median']:.3f} N", flush=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.folder, a.out or str(Path(a.folder) / "predict/hybrid_cv.json"))
