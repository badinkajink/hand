#!/usr/bin/env python3
"""The magnets' strength k against their depth in the rest fit of a UR5 session's tip (owner, 2026-10-25, on the first
checkerboard tip: "the magnets are the exact same as the past tip. so it doesnt make sense that itd that much stronger").

For the session's fitted pole pattern, the rest fit is repeated with k held at each value of a grid and the offsets and
the uniform field free (the residual profile over k), and with k held at a known strength (default sphere2's 0.336): all
32 patterns are then ranked again, and the physical model of scripts/ur5_tip_calibration_predict.py is scored with that
calibration, untrained (the viewer's) and fitted to the training presses, against the session's own rest fit.

    .venv/bin/python scripts/ur5_tip_calibration_k_check.py <analysis output folder> [--k 0.336] [--out k_check.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import fingertip_wrench_model as W  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402

GRID = (0.25, 0.30, 0.336, 0.40, 0.45, 0.50, 0.568, 0.65, 0.80)
STARTS = ([0, 0, 0], [1.5, 0, 0], [3, 0, 0], [-1.5, 0, 0], [0, 1.5, 1.5], [0, -1.5, -1.5])


def fit_at_k(c, b, row, poles, k):
    """Offsets (mm) and uniform field (uT) fitted to the rest with k held; returns (rms mT, x)."""
    best = None
    for st in STARTS:
        r = least_squares(lambda q: W._resid(b, row, poles, c["s"], tuple(c["perm"]), [k, *q[:3], *q[3:]],
                                             theta0=c["theta0"]),
                          st + [0, 0, 0], bounds=([-3, -4, -4, -100, -100, -100], [8, 4, 4, 100, 100, 100]))
        if best is None or r.cost < best.cost:
            best = r
    return float(np.sqrt(np.mean(best.fun ** 2))), best.x


def run(folder, k_known, out):
    folder = Path(folder)
    c = json.loads((folder / "calib_rest.json").read_text())
    b, row, poles = np.asarray(c["b_rest_uT"], float), W.fe_row(c["case"]), tuple(c["poles"])
    prof = []
    for k in sorted(set(GRID) | {round(c["k"], 3)}):
        rms, x = fit_at_k(c, b, row, poles, k)
        prof.append(dict(k=k, rms_mT=rms, d_mm=x[:3].tolist()))
        print(f"  k {k:.3f}: {rms:.3f} mT, d ({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}) mm", flush=True)
    ranked = sorted(((fit_at_k(c, b, row, pl, k_known)[0], pl) for pl in itertools.product((1, -1), repeat=5)),
                    key=lambda t: t[0])
    print(f"  patterns at k {k_known}: " + ", ".join(f"{p} {r:.3f}" for r, p in ranked[:3]), flush=True)
    rms, x = fit_at_k(c, b, row, poles, k_known)
    H, _, cal0 = PR.load(folder)
    calk = replace(cal0, k=k_known, dx=float(x[0]), dy=float(x[1]), dz=float(x[2]), g_uT=tuple(float(v) for v in x[3:6]),
                   rms_mT=rms)
    basis = W.load_basis(PR.BASIS)
    train, test = PR.split(H)
    te, tr = (np.nonzero(np.isin(H["id"], s))[0] for s in (test, train))
    allp = np.arange(len(H["id"]))
    scores = {}
    for name, cal in (("rest fit", cal0), ("k held", calk)):
        ph = PR.Physical(cal, basis)
        fit = PR.Physical(cal, basis).fit(H, tr)
        scores[name] = dict(k=cal.k, d_mm=[cal.dx, cal.dy, cal.dz], heldout=PR.scores(H, te, *ph.predict(H, te)),
                            all=PR.scores(H, allp, *ph.predict(H, allp)), fitted=PR.scores(H, te, *fit.predict(H, te)),
                            fitted_params=dict(k=fit.theta[0], dx=fit.theta[1], dy=fit.theta[2], dz=fit.theta[3],
                                               c=math.exp(fit.theta[4])))
        s = scores[name]
        print(f"  {name} (k {cal.k:.3f}): untrained {s['heldout']['loc_mm']['median']:.2f} mm, "
              f"{s['heldout']['force_N']['median']:.3f} N; fitted {s['fitted']['loc_mm']['median']:.2f} mm, d_x "
              f"{s['fitted_params']['dx']:.2f} mm, c {s['fitted_params']['c']:.2f}", flush=True)
    res = dict(folder=str(folder), poles=list(poles), k_known=k_known, profile=prof,
               patterns_at_k=[dict(poles=list(p), rms_mT=r) for r, p in ranked],
               k_held=dict(rms_mT=rms, d_mm=x[:3].tolist(), g_uT=x[3:6].tolist()), scores=scores)
    Path(out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--k", type=float, default=0.336, help="the known strength (sphere2's rest fit: 0.336)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.folder, a.k, a.out or str(Path(a.folder) / "k_check.json"))
