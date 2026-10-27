#!/usr/bin/env python3
"""Contact direction and location by region on the four checkerboard tips (owner, 2026-10-27: "are we any better at
contact direction? ... the larger errors on those areas", the edges and flanks). For each tip's sphere presses, a GP
trained on the tip's own presses (five folds grouped by action) gives the angle between the estimated and the measured
force (deg) and the location error (mm, along the surface), as medians by region (face, edge, flank) at 1.5 N and over
and under 1.5 N, with the number of holds and press actions and the share of the region's FE points within 1.5 mm of a
press.

    .venv/bin/python scripts/ur5_tip_calibration_direction.py [--out data/20261027-direction_by_region.json]
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
import ur5_tip_calibration_r24_pass3 as P3  # noqa: E402

D = K.EXP / "data"
TIPS = {"ID 1": ("20261023-203938-cb1t", "20261025-114653-cb1r"), "ID 2": ("20261025-152920-cb2t",),
        "ID 3": ("20261026-134522-cb3t",), "v2 ID 1": ("20261026-192158-v2id1",)}


def presses(sessions):
    H = K.cat(*[K.load(D / s) for s in sessions])
    m = (H["kind"] == "press") & np.isfinite(H["d_board"]).all(1) & np.isfinite(H["p"]).all(1) & np.isfinite(H["f"]).all(1)
    return K.sub(H, m)


def cv(H, k=5):
    a = H["id"].astype(str)
    u = np.random.default_rng(0).permutation(np.unique(a))
    f = np.array([{x: i % k for i, x in enumerate(u)}[x] for x in a])
    Y = np.full((len(a), 6), np.nan)
    for q in range(k):
        m = f == q
        Y[m] = K.gp(K.sub(H, ~m)).predict(H["d_board"][m])
    return Y


def run(a):
    b = np.load(P3.BASIS)
    pts, reg_of = b["pts"], np.array([P3.region(v) or "" for v in b["normals"]])
    res = {}
    for t, ss in TIPS.items():
        H = presses(ss)
        Y = cv(H)
        ang = K.angle(Y[:, 3:], H["f"])
        loc = K.loc_err(H, Y[:, :3])
        hi = H["f_n"] >= 1.5
        r = {}
        for region in ("face", "edge", "flank"):
            m = H["region"].astype(str) == region
            P = pts[reg_of == region]
            near = np.array([np.min(np.linalg.norm(H["p"] - p, axis=1)) for p in P]) if len(H["p"]) else np.array([])
            r[region] = dict(n_holds=int(m.sum()), n_actions=int(len(np.unique(H["id"][m]))),
                             covered_1p5mm=float(np.mean(near <= 1.5)),
                             angle_over_1p5N=K.med(ang[m & hi]), angle_under_1p5N=K.med(ang[m & ~hi]),
                             loc_over_1p5N=K.med(loc[m & hi]), loc_under_1p5N=K.med(loc[m & ~hi]))
        res[t] = r
        print(t, json.dumps({g: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for g, x in r.items()}),
              flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(D / "20261027-direction_by_region.json"))
    run(ap.parse_args())
