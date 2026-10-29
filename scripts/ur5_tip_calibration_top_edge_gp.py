#!/usr/bin/env python3
"""The calibrated GP on the tip's end (2026-10-29): how well the direction GP trained on the sphere's grid locates the
presses of v3 ID 2's top-edge pass (10-28, 19:21 to 19:30: nine presses on the tip's end at 0.25 to 2 N and one on its
edge to 3 N), and how much the pass's own presses improve it.

The direction GP is scripts/ur5_tip_calibration_hard_extrapolation.py's fit_gp_direction: from the direction of the
board's change, d/|d|, to the contact point p and to f/|d|. It is trained on the session's press holds (f_n over 0.05 N,
as scripts/ur5_tip_calibration_four_vs_five.py takes them) in two ways:
  grid only       every press except the pass's ten; it then predicts each of the ten;
  with the pass   every press except the one predicted, so the pass's other nine are in training.
The grid's own score is its five folds grouped by action, without the pass, as four_vs_five scores it. Scores by group
and by force bin: loc_mm, the median location error across the contact normal (mm); force_rel_err, the median of
| |f_est| - |f| | / |f|; angle_deg, the median angle between the estimated and the measured force.

    .venv/bin/python scripts/ur5_tip_calibration_top_edge_gp.py logs/20261029-spherev3id2 [--top sph-top-] [--out FILE]
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
import ur5_tip_calibration_cb1r_checks as K  # noqa: E402
import ur5_tip_calibration_four_vs_five as FV  # noqa: E402
import ur5_tip_calibration_gain_by_region as GR  # noqa: E402
import ur5_tip_calibration_hard_extrapolation as HX  # noqa: E402

BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.5))


def scores(H, P, F, m):
    E = HX.errors(H, P, F)
    out = {"all": HX.stats(E, m)}
    for lo, hi in BINS:
        mb = m & HX.in_bin(H, lo, hi)
        if mb.any():
            out[f"{lo:g}-{hi:g}"] = HX.stats(E, mb)
    return out, E


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", help="an output folder of scripts/ur5_tip_calibration_analysis.py (holds.npz)")
    ap.add_argument("--top", default="sph-top-", help="the ids of the top-edge pass start with this")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    H = FV.holds(a.session)
    ids = H["id"].astype(str)
    centre = (H["role"] == "ref") & (np.linalg.norm(H["target"] - GR.CENTRE_P, axis=1) < 0.01)
    top = np.char.startswith(ids, a.top) & ~centre
    res = dict(script="scripts/ur5_tip_calibration_top_edge_gp.py", session=a.session, press_holds=int(len(ids)),
               pass_presses=int(len(np.unique(ids[top]))), pass_holds=int(top.sum()))

    P, F = np.full((len(ids), 3), np.nan), np.full((len(ids), 3), np.nan)
    pred, _ = HX.fit_gp_direction(K.sub(H, ~top), {})
    P[top], F[top], _ = pred(K.sub(H, top))
    res["grid_only"], E1 = scores(H, P, F, top)

    P2, F2 = np.full((len(ids), 3), np.nan), np.full((len(ids), 3), np.nan)
    for i in np.unique(ids[top]):
        m = ids == i
        pred, _ = HX.fit_gp_direction(K.sub(H, ~m), {})
        P2[m], F2[m], _ = pred(K.sub(H, m))
    res["with_the_pass"], E2 = scores(H, P2, F2, top)

    G = K.sub(H, ~top)
    Pg, Fg = FV.cv(G, HX.fit_gp_direction)
    res["grid_five_folds"], _ = scores(G, Pg, Fg, np.ones(len(G["id"]), bool))

    rows = []
    for i in dict.fromkeys(ids[top]):
        m = top & (ids == i)
        rows.append(dict(id=i, region=str(H["region"][m][0]), f_n=[round(float(x), 2) for x in H["f_n"][m]],
                         loc_grid_only_mm=[round(float(x), 2) for x in E1["loc"][m]],
                         loc_with_the_pass_mm=[round(float(x), 2) for x in E2["loc"][m]]))
    res["per_press"] = rows

    for name in ("grid_five_folds", "grid_only", "with_the_pass"):
        for key, r in res[name].items():
            print(f"{name:16s} {key:6s} n {r['n']:4d}  loc {r['loc_mm']:.2f} mm (p90 {r['loc_p90_mm']:.2f})  "
                  f"force {100 * r['force_rel_err']:.1f} %  angle {r['angle_deg']:.1f} deg")
    for r in rows:
        print(f"  {r['id']:20s} {r['region']:5s} " + "  ".join(f"{f:4.2f} N {g:5.2f}/{w:5.2f}" for f, g, w in
                                                          zip(r["f_n"], r["loc_grid_only_mm"], r["loc_with_the_pass_mm"])))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print("->", a.out)


if __name__ == "__main__":
    main()
