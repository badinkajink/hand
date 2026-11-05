#!/usr/bin/env python3
"""Can the tip-mode calibration of v3 IDs 1 to 3 improve that of IDs 4 to 6, which were pressed only on the inverted
rig (owner, 2026-11-05 11:4x: "can we improve the v3 ids 4-6 calibration with the calibration from ids 1-3 ... maybe we
use the combined GP on them too").

The model is the contact viewer's (ur5_tip_calibration_board_gp: one GP shared by every board, no input marks the tip;
inputs the 15 board components, outputs the contact point in mm and the force in N; presses and slides). For each of
IDs 4 to 6 (tip j), "own" trains on tip j's rig holds alone and "combined" adds 1,000 holds of each other set: the
tip-mode holds of IDs 1, 2 and 3, the rig holds of the other two of IDs 4 to 6, and 500 of ID 3's rig holds. Tip j's
holds are subsampled to 3,000 at most. Tests, each against tip j's own ground truth:
  random    a fifth of tip j's point groups held out (2 seeds): points between its own
  tilted    every hold of tip j with a tilt |f_t| / |f_n| of 0.3 or more held out: loads its training lacks
  region    tip j's points split into 5 regions by k-means on the contact point, one region held out at a time: a part
            of the pad its training lacks, as the rig lacked the mount end of IDs 1 to 3
Location is the error along the surface, direction the angle between predicted and measured force, magnitude the force
error over the force, medians over the held-out holds of 1 N and over.

    ~/.claude/bin/resguard.sh run --mem 6G --cpu 500 -- env OPENBLAS_NUM_THREADS=4 taskset -c 4-7 .venv/bin/python \\
        scripts/ur5/ur5_tip_calibration_rig_pool_ids456.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401
import ur5_tip_calibration_board_gp as BG  # noqa: E402
import ur5_tip_calibration_rig_coverage as RC  # noqa: E402
import ur5_tip_calibration_v3_data_curve as V  # noqa: E402

KINDS = ["press", "slide"]
OUT = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261105-v3_ids456_pooled_with_ids123.json"


def sub(h, n, rng):
    if len(h["p"]) <= n:
        return h
    i = np.sort(rng.choice(len(h["p"]), n, replace=False))
    return {f: v[i] for f, v in h.items()}


def XY(hs):
    return (np.concatenate([h["d_board"] for h in hs]), np.concatenate([np.c_[h["p"], h["f"]] for h in hs]))


def main():
    rng = np.random.default_rng(0)
    sets = {f"tip{k}": RC.load(V.TIPS[k], KINDS, f"tip{k}") for k in (1, 2, 3)}
    sets.update({f"rig{j}": RC.load(RC.RIG[j], KINDS, f"rig{j}") for j in (3, 4, 5, 6)})
    out = {"what": __doc__.split("\n\n")[1], "rows": []}
    t0 = time.time()
    for j in (4, 5, 6):
        own = sub(sets[f"rig{j}"], 3000, rng)
        others = [sub(sets[s], 500 if s == "rig3" else 1000, rng) for s in sets if s != f"rig{j}"]
        tl = RC.tilt(own)
        _, lab = kmeans2(own["p"], 5, seed=1, minit="++")
        splits = []
        for seed in range(2):
            g = np.random.default_rng(1000 + seed).permutation(np.unique(own["group"]))
            splits.append((f"random s{seed}", np.isin(own["group"], g[: len(g) // 5])))
        splits.append(("tilted", tl >= 0.3))
        for r in range(5):
            splits.append((f"region {r}", lab == r))
        for name, test in splits:
            tr = {f: v[~test] for f, v in own.items()}
            te = {f: v[test] for f, v in own.items()}
            row = dict(tip=j, test=name, n_test=int(test.sum()),
                       region_centre_mm=np.round(own["p"][test].mean(0), 1).tolist() if name.startswith("region") else None)
            for model, hs in (("own", [tr]), ("combined", [tr] + others)):
                X, Y = XY(hs)
                m = BG.fit(X, Y, RC.FLOOR)
                row[model] = RC.errs(te, BG.predict(m, te["d_board"]), np.ones(len(te["p"]), bool))
                row[model]["n_train"] = int(len(X))
            out["rows"].append(row)
            print(f"ID {j} {name} ({row['n_test']} held out): own {row['own'].get('location_mm')} mm "
                  f"{row['own'].get('direction_deg')} deg, combined {row['combined'].get('location_mm')} mm "
                  f"{row['combined'].get('direction_deg')} deg; {time.time() - t0:.0f} s", flush=True)
            OUT.write_text(json.dumps(out, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
