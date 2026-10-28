#!/usr/bin/env python3
"""Press coverage of the pad by region (owner, 2026-10-28: "it seems the density is not that good even on the easily
reachable space"): for each set of sessions, the share of the FE surface points (20261020-b_v2_surface_basis.npz, regions
of ur5_tip_calibration_r24_pass3.region) within 1.0 and 1.5 mm of a press target, and the median and 90th percentile
distance from a point to its nearest target. A target is the mean contact point of one press action's holds.

    .venv/bin/python scripts/ur5_tip_calibration_coverage.py
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
from scipy.spatial import cKDTree
import ur5_tip_calibration_cb1r_checks as K
import ur5_tip_calibration_r24_pass3 as P3
D = K.EXP / "data"
b = np.load(P3.BASIS)
pts, reg_of = b["pts"], np.array([P3.region(v) or "" for v in b["normals"]])
def targets(path, kinds=("press",)):
    H = K.load(path)
    m = np.isin(H["kind"].astype(str), kinds) & np.isfinite(H["p"]).all(1)
    ids = H["id"][m].astype(str)
    P = H["p"][m]
    # one target per action: the mean contact point of its holds
    out = np.array([P[ids == i].mean(axis=0) for i in np.unique(ids)])
    return out
sets = {
    "cb2t (10-25)": [D / "20261025-152920-cb2t"],
    "v2 ID 1 sphere": [D / "20261026-192158-v2id1"],
    "v2 ID 1 sphere + tall": [D / "20261026-192158-v2id1", D / "20261027-143529-tallv2id1"],
    "v2 ID 1 sphere + tall + rod": [D / "20261026-192158-v2id1", D / "20261027-143529-tallv2id1", D / "20261027-095303-rodv2id1"],
    "v3 ID 1 sphere so far": [K.Path("logs/20261028-spherev3id1_grid")],
}
res = {}
for name, paths in sets.items():
    T = np.vstack([targets(p) for p in paths])
    tree = cKDTree(T)
    row = {"targets": int(len(T))}
    for r in ("face", "edge", "flank", "tip"):
        P = pts[reg_of == r]
        if not len(P):
            continue
        d = tree.query(P)[0]
        row[r] = dict(n=int(len(P)), within_1mm=round(float(np.mean(d <= 1.0)), 3), within_1p5mm=round(float(np.mean(d <= 1.5)), 3),
                      median_mm=round(float(np.median(d)), 2), p90_mm=round(float(np.percentile(d, 90)), 2))
    res[name] = row
    print(name, json.dumps(row), flush=True)
out = K.Path("docs/experiments/20261027-v3_tip_calibration_and_crosstalk/data/20261028-press_coverage_by_region.json")
out.write_text(json.dumps(res, indent=1) + "\n")
print("->", out)
