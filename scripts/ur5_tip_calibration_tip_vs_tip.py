#!/usr/bin/env python3
"""Two tips of one design on the same press actions (2026-10-29: v3 ID 2 against v3 ID 1, the two four-magnet tips on the
UR5's low sphere). Each tip's press holds (f_n over 0.05 N, as scripts/ur5_tip_calibration_four_vs_five.py takes them)
are kept for the actions that every tip ran, matched by action id, and each tip is scored on its own holds under the same
five folds grouped by action: the board GP and the direction GP of scripts/ur5_tip_calibration_hard_extrapolation.py,
with four_vs_five's columns (loc_mm, force_rel_err, angle_deg, normal_deg) by force bin and region, and the board's gain
(median |d_board| / f_n, uT/N). four_vs_five scores each four-magnet tip against v2 ID 1 on the actions the two share;
the speed variants that v3 ID 1 ran and v3 ID 2 did not re-press grid points, so those sets score differently.

    .venv/bin/python scripts/ur5_tip_calibration_tip_vs_tip.py "v3 ID 2=logs/20261029-spherev3id2" \\
        "v3 ID 1=logs/20261028-spherev3id1_twists" [--out FILE]
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
import ur5_tip_calibration_hard_extrapolation as HX  # noqa: E402


def score(H):
    out = {}
    for name, fitter in (("gp", HX.fit_gp), ("gp_direction", HX.fit_gp_direction)):
        P, F = FV.cv(H, fitter)
        E = HX.errors(H, P, F)
        nd = FV.normal_err(H, P)

        def st(m):
            row = HX.stats(E, m)
            x = nd[m & np.isfinite(nd)]
            row["normal_deg"] = float(np.median(x)) if len(x) else None
            return row
        rows = {"all": st(np.ones(len(H["id"]), bool))}
        for lo, hi in FV.BINS:
            if HX.in_bin(H, lo, hi).any():
                rows[f"{lo:g}-{hi:g}"] = st(HX.in_bin(H, lo, hi))
        for r in FV.REGIONS:
            if (H["region"] == r).any():
                rows[r] = st(H["region"] == r)
        out[name] = rows
    gain = np.linalg.norm(H["d_board"], axis=1) / H["f_n"]
    out["gain_uT_per_N"] = {f"{lo:g}-{hi:g}": float(np.median(gain[HX.in_bin(H, lo, hi)]))
                            for lo, hi in FV.BINS if HX.in_bin(H, lo, hi).any()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tips", nargs="+", help="LABEL=DIR, DIR an output folder of scripts/ur5_tip_calibration_analysis.py")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tips = []
    for s in a.tips:
        label, d = s.split("=", 1)
        tips.append((label, d, FV.holds(d)))
    ids = sorted(set.intersection(*(set(H["id"].astype(str)) for _, _, H in tips)))
    res = dict(script="scripts/ur5_tip_calibration_tip_vs_tip.py", actions=len(ids), tips={})
    for label, d, H in tips:
        H = K.sub(H, np.isin(H["id"].astype(str), ids))
        res["tips"][label] = dict(holds_dir=d, holds=int(len(H["id"])), **score(H))
    labels = [t[0] for t in tips]
    print(f"{len(ids)} actions; holds " + ", ".join(f"{k} {res['tips'][k]['holds']}" for k in labels))
    for name in ("gp", "gp_direction"):
        for key in res["tips"][labels[0]][name]:
            rows = [res["tips"][k][name].get(key) for k in labels]
            if all(rows):
                print(f"{name:13s} {key:6s} loc " + " / ".join(f"{r['loc_mm']:.3f}" for r in rows) + " mm  force "
                      + " / ".join(f"{100 * r['force_rel_err']:.1f}" for r in rows) + " %  normal "
                      + " / ".join(f"{r['normal_deg']:.1f}" for r in rows) + f" deg   ({' / '.join(labels)})")
    for key in res["tips"][labels[0]]["gain_uT_per_N"]:
        print(f"gain {key:6s} " + " / ".join(f"{res['tips'][k]['gain_uT_per_N'].get(key, float('nan')):.1f}" for k in labels)
              + " uT/N")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print("->", a.out)


if __name__ == "__main__":
    main()
