#!/usr/bin/env python3
"""The calibrated GP's location error at the face centre, four magnets (v3 ID 1) against five (v2 ID 1), on the sphere
presses both tips ran (owner, 2026-10-28: "do you think we can improve this error with greater coverage across surface
area, force, speed on the nominal center area?").

The face centre is the apex over the board's centre chip U1, (y, z) = (0.5, 10.36) mm in the pad frame
(scripts/fingertip_physical_model_fe.py APEX_YZ); a press within 2 mm of it, along y and z, counts as centre. Folds group
the holds by target point (rounded to 0.1 mm; --folds action groups them by action, as
scripts/ur5_tip_calibration_four_vs_five.py does, which leaves sibling actions on the same point in training). Per tip
and fitter (gp, gp_direction of scripts/ur5_tip_calibration_hard_extrapolation.py):
  by_force                   median and 90th percentile of the location error along the surface (mm) for the centre, the
                             rest of the face, and the edge and flank, under 1 N, 1 to 2 N and over 2 N;
                             loc_one_record_mm is the same with one board record's noise (1.77 and 1.83 uT, the two tips'
                             measured noise) added to the held-out inputs; gain_uT_per_N the board's change per newton;
  by_nearest_training_press  the median error by the distance from the held-out target to the nearest training target.

    .venv/bin/python scripts/ur5_tip_calibration_centre_coverage.py V3_HOLDS_DIR [--folds target|action] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration_four_vs_five as F  # noqa: E402
import ur5_tip_calibration_hard_extrapolation as HX  # noqa: E402
import ur5_tip_calibration_cb1r_checks as K  # noqa: E402

APEX = np.array([0.5, 10.36])
NOISE_UT = dict(four=1.77, five=1.83)


def fold_ids(H, mode):
    if mode == "action":
        return HX.folds(H)
    key = [tuple(np.round(x, 1)) for x in np.asarray(H["target"], float)]
    uk = sorted(set(key))
    perm = np.random.default_rng(1).permutation(len(uk))
    fk = {u: int(perm[i]) % HX.FOLDS for i, u in enumerate(uk)}
    return np.array([fk[k] for k in key])


def score(H, sig, mode, rng):
    face = H["region"].astype(str) == "face"
    dc = np.hypot(H["p"][:, 1] - APEX[0], H["p"][:, 2] - APEX[1])
    groups = (("centre", face & (dc <= 2.0)), ("rest_of_face", face & (dc > 2.0)), ("edge_and_flank", ~face))
    fo, tg = fold_ids(H, mode), np.asarray(H["target"], float)
    res = {}
    for name, fitter in (("gp", HX.fit_gp), ("gp_direction", HX.fit_gp_direction)):
        P, Pn, dnear = np.full((len(fo), 3), np.nan), np.full((len(fo), 3), np.nan), np.full(len(fo), np.nan)
        for k in range(HX.FOLDS):
            te = fo == k
            pred, _ = fitter(K.sub(H, ~te), {})
            Ht = K.sub(H, te)
            P[te] = pred(Ht)[0]
            Hn = dict(Ht)
            Hn["d_board"] = Ht["d_board"] + rng.normal(0.0, sig, Ht["d_board"].shape)
            Pn[te] = pred(Hn)[0]
            dnear[te] = cKDTree(tg[~te]).query(tg[te])[0]
        e, en = K.loc_err(H, P), K.loc_err(H, Pn)
        gain = np.linalg.norm(H["d_board"], axis=1) / H["f_n"]
        rows, cov = {}, {}
        for gname, g in groups:
            for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 8.0)):
                m = g & (H["f_n"] > lo) & (H["f_n"] <= hi)
                if m.sum() >= 5:
                    rows[f"{gname} {lo:g}-{hi:g} N"] = dict(
                        n=int(m.sum()), loc_mm=float(np.median(e[m])), loc_p90_mm=float(np.percentile(e[m], 90)),
                        loc_one_record_mm=float(np.median(en[m])), gain_uT_per_N=float(np.median(gain[m])))
            for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 99.0)):
                m = g & (dnear > lo) & (dnear <= hi)
                if m.sum() >= 5:
                    cov[f"{gname} nearest {lo:g}-{hi:g} mm"] = dict(n=int(m.sum()), loc_mm=float(np.median(e[m])))
        res[name] = dict(by_force=rows, by_nearest_training_press=cov)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("v3")
    ap.add_argument("--folds", choices=("target", "action"), default="target")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    H3, H2 = F.holds(a.v3), F.holds(F.V2)
    ids = sorted(set(H3["id"].astype(str)) & set(H2["id"].astype(str)))
    H3, H2 = (K.sub(H, np.isin(H["id"].astype(str), ids)) for H in (H3, H2))
    rng = np.random.default_rng(0)
    out = dict(script="scripts/ur5_tip_calibration_centre_coverage.py", v3=str(a.v3), v2=str(F.V2.relative_to(ROOT)),
               folds=a.folds, actions=len(ids), apex_yz_mm=APEX.tolist(),
               four_v3id1=score(H3, NOISE_UT["four"], a.folds, rng), five_v2id1=score(H2, NOISE_UT["five"], a.folds, rng))
    for fit in ("gp_direction", "gp"):
        for k, r3 in out["four_v3id1"][fit]["by_force"].items():
            r2 = out["five_v2id1"][fit]["by_force"].get(k)
            if r2:
                print(f"{fit:12s} {k:26s} n {r3['n']:3d}  loc {r3['loc_mm']:.2f} / {r2['loc_mm']:.2f} mm  one record "
                      f"{r3['loc_one_record_mm']:.2f} / {r2['loc_one_record_mm']:.2f}  gain {r3['gain_uT_per_N']:.1f} / "
                      f"{r2['gain_uT_per_N']:.1f} uT/N (four / five)")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print("->", a.out)


if __name__ == "__main__":
    main()
