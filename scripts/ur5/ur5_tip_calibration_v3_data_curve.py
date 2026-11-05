#!/usr/bin/env python3
"""Held-out error of the board GP on the v3 tips against the number of training holds (lab, 11-02 14:59, for the owner's
go on more runs: "if it's possible to get twists/wrenches, e.g. we see some kind of predictive performance scaling with
data collection, then you have my go ahead to send off more runs on that end too").

The data are the tip-mode sessions of v3 ID 1 to ID 3 that the pooled GP takes (scripts/ur5/ur5_tip_calibration_board_gp.py:
the sphere and rod sessions, presses, slides, rolls and twists). For each tip the holds are grouped by point as in
scripts/ur5/ur5_tip_calibration_density_curve.py (median contact points within 0.75 mm). Each seed holds out a fifth of the
groups (a fifth of those with twists among them) and fits the GP (board_gp.fit: the 15 board components to the contact point in mm, the force in N and the torsion
about the normal in N mm, in the pad's frame) on growing, nested fractions of the other groups.

  location   median distance along the surface from the predicted to the measured contact point, holds of 1 N and over
  direction  median angle between the predicted and the measured force, holds of 1 N and over
  magnitude  median of |predicted - measured| / measured force, holds of 1 N and over
  torsion    median |predicted - measured| torsion about the normal, the held-out twists; against the median measured
             |torsion| there, the error of predicting none

--tip ID=DIR[,DIR ...] (11-04, the inverted rig's v3 IDs 4 to 6): the holds folders of another tip, in place of IDs 1
to 3 (repeat for more tips; --out then required). The probe sessions' holds (scripts/ur5/ur5_probe_calibration_analysis.py)
add the kinds sweep (a sweep's levels) and offset (its tangential loads inside the friction cone); --kinds picks them.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 400 -- env OPENBLAS_NUM_THREADS=4 taskset -c 4-7 .venv/bin/python \\
        scripts/ur5/ur5_tip_calibration_v3_data_curve.py [--seeds 4] [--tip ID=DIR,DIR ... --kinds KIND ... --out OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import ur5_tip_calibration_board_gp as BG  # noqa: E402
import ur5_tip_calibration_density_curve as DC  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402

TIPS = {1: ("logs/20261028-spherev3id1_twists", "logs/20261028-rodv3id1"),
        2: ("logs/20261029-spherev3id2", "logs/20261029-rodv3id2"),
        3: ("logs/20261030-spherev3id3", "logs/20261030-rodv3id3")}
KINDS = ["press", "slide", "roll", "twist"]
FRACTIONS = (0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
OUT = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261102-v3_data_curve.json"


def load(sessions):
    """The holds of a tip's sessions, grouped by point across the sessions (a sphere's and a rod's press at one point
    fall in one group, so neither predicts the other from the training side)."""
    H = {}
    for k, s in enumerate(sessions):
        z = np.load(ROOT / s / "holds.npz", allow_pickle=True)
        ok = (np.isin(z["kind"].astype(str), KINDS) & np.isfinite(z["d_board"]).all(1) & np.isfinite(z["p"]).all(1)
              & np.isfinite(z["f"]).all(1) & np.isfinite(z["tn"]))
        h = {f: z[f][ok] for f in ("d_board", "p", "n", "f", "tn", "kind")}
        h["id"] = np.char.add(f"{k}:", z["id"][ok].astype(str))
        for f, v in h.items():
            H[f] = np.concatenate([H[f], v]) if f in H else v
    H["group"] = DC.groups(H)
    return H


def errors(H, P, test):
    p, f, n, tn = H["p"][test], H["f"][test], H["n"][test], H["tn"][test]
    dp = P[:, :3] - p
    loc = np.linalg.norm(dp - np.sum(dp * n, 1)[:, None] * n, axis=1)
    fm = np.linalg.norm(f, axis=1)
    c = np.sum(P[:, 3:6] * f, 1) / (np.linalg.norm(P[:, 3:6], axis=1) * fm + 1e-12)
    ang = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    mag = np.abs(np.linalg.norm(P[:, 3:6], axis=1) - fm) / (fm + 1e-12)
    hi = fm >= 1.0
    tw = H["kind"][test].astype(str) == "twist"
    med = (lambda x: float(np.median(x)) if len(x) else float("nan"))
    return dict(location_mm=med(loc[hi]), direction_deg=med(ang[hi]), magnitude=med(mag[hi]),
                torsion_Nmm=med(np.abs(P[tw, 6] - tn[tw])), torsion_zero_Nmm=med(np.abs(tn[tw])),
                n_test=int(len(p)), n_twist=int(tw.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--tip", action="append", default=[], help="ID=DIR[,DIR ...]: another tip's holds folders")
    ap.add_argument("--kinds", nargs="*", default=None, help="hold kinds used (default press slide roll twist)")
    a = ap.parse_args()
    if a.tip:
        if not a.out:
            raise SystemExit("--tip needs --out")
        TIPS.clear()
        for t in a.tip:
            k, v = t.split("=", 1)
            TIPS[int(k)] = tuple(x for x in v.split(",") if x)
    if a.kinds:
        KINDS[:] = a.kinds
    a.out = a.out or str(OUT)
    t0 = time.time()
    floor = PR.floors(0)[:15]
    out = dict(what=__doc__.split("\n\n")[1], tips={}, fractions=FRACTIONS, seeds=a.seeds, kinds=list(KINDS))
    for tip, sessions in TIPS.items():
        H = load(sessions)
        Y = np.c_[H["p"], H["f"], H["tn"]]
        G = np.unique(H["group"])
        Gt = np.unique(H["group"][H["kind"].astype(str) == "twist"])      # the points with twists, held out a fifth
        Go = np.setdiff1d(G, Gt)                                           # at a time as the others
        rows = []
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            gt, go = rng.permutation(Gt), rng.permutation(Go)
            kt, ko = max(1, len(gt) // 5), len(go) // 5
            test = np.isin(H["group"], np.r_[gt[:kt], go[:ko]])
            pool = rng.permutation(np.r_[gt[kt:], go[ko:]])
            for fr in FRACTIONS:
                train = np.isin(H["group"], pool[:max(2, int(round(fr * len(pool))))])
                m = BG.fit(H["d_board"][train], Y[train], floor)
                e = errors(H, BG.predict(m, H["d_board"][test]), test)
                rows.append(dict(seed=seed, fraction=fr, n_train=int(train.sum()), n_groups=int(round(fr * len(pool))),
                                 ell=m["ell"], noise=m["nz"], **e))
                print(f"ID {tip} seed {seed} fraction {fr:.2f}: {train.sum()} holds; location {e['location_mm']:.2f} mm, "
                      f"direction {e['direction_deg']:.1f} deg, magnitude {e['magnitude']:.3f}, torsion "
                      f"{e['torsion_Nmm']:.2f} N mm ({e['n_twist']} twists); {time.time() - t0:.0f} s", flush=True)
        summ = {}
        for fr in FRACTIONS:
            r = [x for x in rows if x["fraction"] == fr]
            summ[f"{fr:g}"] = {k: float(np.median([x[k] for x in r])) for k in
                               ("n_train", "location_mm", "direction_deg", "magnitude", "torsion_Nmm", "torsion_zero_Nmm")}
        out["tips"][str(tip)] = dict(sessions=sessions, holds=int(len(Y)), groups=int(len(G)), rows=rows, median=summ)
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for tip, d in out["tips"].items():
        print(f"ID {tip} ({d['holds']} holds, {d['groups']} points), median over seeds:")
        for fr, s in d["median"].items():
            print(f"  {fr:>4}: {s['n_train']:6.0f} holds  location {s['location_mm']:.2f} mm  direction "
                  f"{s['direction_deg']:.1f} deg  magnitude {s['magnitude']:.3f}  torsion {s['torsion_Nmm']:.2f} N mm (none "
                  f"predicted: {s['torsion_zero_Nmm']:.2f})")
    print("->", a.out)


if __name__ == "__main__":
    main()
