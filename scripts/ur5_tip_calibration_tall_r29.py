#!/usr/bin/env python3
"""v2 ID 1's presses on the tall indenter (R29, 2026-10-27) against its sphere presses (192158-v2id1). Both indenters
are spheres of radius 5 mm; the tall one stands 160 mm above the 2x4, so the pad reaches flank points the sphere
fixture does not. Four questions, each as medians by region (face, edge, flank) at 1.5 N and over and under 1.5 N:

  transfer   a GP trained on the sphere presses alone, on the tall presses: location error (mm, along the surface),
             the angle between the estimated and the measured force (deg) and the force error (N, vector norm);
  loto       the same with the other tall presses added to the sphere presses, leaving out one tall press action;
  sphere     the sphere presses, five folds grouped by action, with and without every tall press in each fold's
             training (does the new density help the points the sphere already had?);
  coverage   the share of the region's FE points within 1.5 mm of a press, sphere alone and with the tall presses.

A fifth, by_distance, bins the errors by the distance from a hold to the nearest training hold: the tall presses under
the sphere's GP, and sphere presses held out with every sphere press within r mm of them left out of training (r 1, 1.5,
2 and 3 mm), so the two indenters are compared at the same distance. A sixth, shear, gives the shear part of the measured force per newton in the pad frame on the three indenters
(sphere, rod and tall), and the transfer's angle error binned by the shear along the finger. It also compares the board
change of the tall indenter's first reference press (tall-ref-a, at the face's centre) with
the sphere's reference presses at the same point and level, and gives the transfer's mean signed location offset in the
pad frame (a registration offset between the two indenters would show there).

    .venv/bin/python scripts/ur5_tip_calibration_tall_r29.py [--tall 20261027-143529-tallv2id1] [--out ...]
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
import ur5_tip_calibration_cb1r_checks as K  # noqa: E402
import ur5_tip_calibration_r24_pass3 as P3  # noqa: E402

D = K.EXP / "data"
SPH = "20261026-192158-v2id1"
REGIONS = ("face", "edge", "flank")


def presses(s):
    H = K.load(D / s)
    m = (H["kind"] == "press") & np.isfinite(H["d_board"]).all(1) & np.isfinite(H["p"]).all(1) & np.isfinite(H["f"]).all(1)
    return K.sub(H, m)


def table(H, Y):
    loc = K.loc_err(H, Y[:, :3])
    ang = K.angle(Y[:, 3:], H["f"])
    fe = np.linalg.norm(Y[:, 3:] - H["f"], axis=1)
    hi = H["f_n"] >= 1.5
    out = {}
    for r in REGIONS:
        m = H["region"].astype(str) == r
        row = dict(n_holds=int(m.sum()), n_actions=int(len(np.unique(H["id"][m]))))
        for q, x in (("loc_mm", loc), ("angle_deg", ang), ("force_err_N", fe)):
            for b, s in (("over_1p5N", hi), ("under_1p5N", ~hi)):
                row[f"{q}_{b}"] = K.med(x[m & s])
        out[r] = row
    return out


def offset(H, Y):
    """Mean signed location error (estimate minus registration) in the pad frame at 1.5 N and over, by region."""
    hi = H["f_n"] >= 1.5
    return {r: (np.mean(Y[m, :3] - H["p"][m], 0).round(3).tolist() if (m := (H["region"].astype(str) == r) & hi).any()
                else None) for r in REGIONS}


def folds(ids, k=5):
    u = np.random.default_rng(0).permutation(np.unique(ids))
    f = {x: i % k for i, x in enumerate(u)}
    return np.array([f[x] for x in ids])


def ref_compare(S, T, tall_ref="tall-ref-a"):
    """Board change of the tall indenter's reference press against the sphere's reference presses at the same target
    and level: |d_tall - mean d_sphere| / |mean d_sphere|, with the forces of both."""
    t = T["id"].astype(str) == tall_ref
    if not t.any():
        return None
    tgt = T["target"][t][0]
    s = (S["role"].astype(str) == "ref") & (np.linalg.norm(S["target"] - tgt, axis=1) < 0.3)
    rows = []
    for lv in np.unique(T["level_N"][t]):
        a, b = t & (T["level_N"] == lv), s & (S["level_N"] == lv)
        if not b.any():
            continue
        da, db = T["d_board"][a].mean(0), S["d_board"][b].mean(0)
        rows.append(dict(level_N=float(lv), f_n_tall=float(T["f_n"][a].mean()), f_n_sphere=float(S["f_n"][b].mean()),
                         n_sphere=int(b.sum()), board_uT_tall=float(np.linalg.norm(da)),
                         board_uT_sphere=float(np.linalg.norm(db)),
                         rel_diff=float(np.linalg.norm(da - db) / np.linalg.norm(db)),
                         sphere_spread=float(np.median([np.linalg.norm(x - db) for x in S["d_board"][b]])
                                             / np.linalg.norm(db))))
    return dict(target_P_mm=np.round(tgt, 2).tolist(), levels=rows)


BINS = (1.0, 1.25, 1.5, 2.0, 3.0, 99.0)


def matched(S, T, Yt, n_actions=60, radii=(1.0, 1.5, 2.0, 3.0)):
    """Errors against the distance from a hold to the nearest training hold (d_nn, mm). Tall: the transfer's errors with
    d_nn to the sphere presses. Sphere: a held-out sphere action predicted by a GP of the sphere presses farther than r
    from all of its holds, for n_actions actions drawn at random and each r; d_nn is then measured as for the tall."""
    rows = dict(tall=[], sphere=[])
    tree = cKDTree(S["p"])
    d = tree.query(T["p"])[0]
    loc, ang = K.loc_err(T, Yt[:, :3]), K.angle(Yt[:, 3:], T["f"])
    fe = np.linalg.norm(Yt[:, 3:] - T["f"], axis=1)
    rows["tall"] = np.c_[d, loc, ang, fe, T["f_n"]]
    ids = S["id"].astype(str)
    acts = np.random.default_rng(1).choice(np.unique(ids), size=min(n_actions, len(np.unique(ids))), replace=False)
    out = []
    for x in acts:
        m = ids == x
        dd = cKDTree(S["p"][m]).query(S["p"])[0]
        for r in radii:
            tr = (dd > r) & ~m
            Y = K.gp(K.sub(S, tr)).predict(S["d_board"][m])
            H = K.sub(S, m)
            dn = cKDTree(S["p"][tr]).query(H["p"])[0]
            out.append(np.c_[dn, K.loc_err(H, Y[:, :3]), K.angle(Y[:, 3:], H["f"]),
                             np.linalg.norm(Y[:, 3:] - H["f"], axis=1), H["f_n"]])
    rows["sphere"] = np.concatenate(out)
    res = {}
    for k, A in rows.items():
        tab = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            for band, sel in (("over_1p5N", A[:, 4] >= 1.5), ("under_1p5N", A[:, 4] < 1.5)):
                m = (A[:, 0] >= lo) & (A[:, 0] < hi) & sel
                tab.append(dict(d_lo=lo, d_hi=hi, band=band, n=int(m.sum()), loc_mm=K.med(A[m, 1]),
                                angle_deg=K.med(A[m, 2]), force_err_N=K.med(A[m, 3])))
        res[k] = tab
    return res


def shear(S, T, Yt, rod="20261027-095303-rodv2id1"):
    """The shear part of the measured force per newton of normal force, in the pad frame (z_P along the finger), as
    medians by region at 1.5 N and over, for the sphere, rod and tall sessions; and the transfer's angle error binned
    by the tall presses' shear along z_P (within 1.6 mm of a sphere press), with the GP's own z_P shear."""
    out = {}
    for name, H in (("sphere", S), ("rod", presses(rod)), ("tall", T)):
        hi = H["f_n"] >= 1.5
        ft = K.shear(H["f"], H["n"]) / H["f_n"][:, None]
        out[name] = {r: np.median(ft[m], 0).round(3).tolist() for r in REGIONS
                     if (m := (H["region"].astype(str) == r) & hi).any()}
    ftz = K.shear(T["f"], T["n"])[:, 2] / T["f_n"]
    ang, loc = K.angle(Yt[:, 3:], T["f"]), K.loc_err(T, Yt[:, :3])
    near = (T["f_n"] >= 1.5) & (cKDTree(S["p"]).query(T["p"])[0] < 1.6)
    out["transfer_by_shear"] = [dict(lo=lo, hi=up, n=int(m.sum()), angle_deg=K.med(ang[m]), loc_mm=K.med(loc[m]))
                                for lo, up in ((-1, 0.06), (0.06, 0.12), (0.12, 0.2), (0.2, 2))
                                if (m := near & (ftz >= lo) & (ftz < up)).any() or True]
    hi = T["f_n"] >= 1.5
    out["gp_vs_measured_z_shear_per_N"] = [K.med((K.shear(Yt[:, 3:], T["n"])[:, 2] / T["f_n"])[hi]), K.med(ftz[hi])]
    return out


def run(a):
    S, T = presses(SPH), presses(a.tall)
    print(f"sphere {len(S['id'])} press holds ({len(np.unique(S['id']))} actions), "
          f"tall {len(T['id'])} ({len(np.unique(T['id']))} actions)", flush=True)
    res = dict(sphere_session=SPH, tall_session=a.tall)

    Y = K.gp(S).predict(T["d_board"])
    res["transfer"] = table(T, Y)
    res["transfer_offset_mm"] = offset(T, Y)
    fs = np.linalg.norm(Y[:, 3:], axis=1) - np.linalg.norm(T["f"], axis=1)
    res["transfer_force_bias_N"] = {r: K.med(fs[(T["region"].astype(str) == r) & (T["f_n"] >= 1.5)]) for r in REGIONS}
    print("transfer", json.dumps(res["transfer"]), json.dumps(res["transfer_offset_mm"]),
          json.dumps(res["transfer_force_bias_N"]), flush=True)
    res["reference"] = ref_compare(S, T)
    print("reference", json.dumps(res["reference"]), flush=True)
    res["shear"] = shear(S, T, Y)
    print("shear", json.dumps(res["shear"]), flush=True)
    res["by_distance"] = matched(S, T, Y)
    for k, tab in res["by_distance"].items():
        for row in tab:
            if row["n"]:
                print(k, json.dumps(row), flush=True)

    Y = np.full((len(T["id"]), 6), np.nan)
    tid = T["id"].astype(str)
    for x in np.unique(tid):
        m = tid == x
        Y[m] = K.gp(K.cat(S, K.sub(T, ~m))).predict(T["d_board"][m])
    res["loto"] = table(T, Y)
    res["loto_offset_mm"] = offset(T, Y)
    print("loto", json.dumps(res["loto"]), flush=True)

    f = folds(S["id"].astype(str))
    Y0, Y1 = np.full((len(f), 6), np.nan), np.full((len(f), 6), np.nan)
    for q in range(5):
        m = f == q
        Y0[m] = K.gp(K.sub(S, ~m)).predict(S["d_board"][m])
        Y1[m] = K.gp(K.cat(K.sub(S, ~m), T)).predict(S["d_board"][m])
    res["sphere_alone"], res["sphere_with_tall"] = table(S, Y0), table(S, Y1)
    print("sphere alone", json.dumps(res["sphere_alone"]), flush=True)
    print("sphere with tall", json.dumps(res["sphere_with_tall"]), flush=True)

    b = np.load(P3.BASIS)
    pts, reg_of = b["pts"], np.array([P3.region(v) or "" for v in b["normals"]])
    cov = {}
    for r in REGIONS:
        P = pts[reg_of == r]
        d0 = cKDTree(S["p"]).query(P)[0]
        d1 = cKDTree(np.r_[S["p"], T["p"]]).query(P)[0]
        cov[r] = dict(n_points=int(len(P)), sphere=float(np.mean(d0 <= 1.5)), sphere_and_tall=float(np.mean(d1 <= 1.5)))
    res["coverage_1p5mm"] = cov
    print("coverage", json.dumps(cov), flush=True)

    out = Path(a.out or D / a.tall / "r29_tall_vs_sphere.json")
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tall", default="20261027-143529-tallv2id1")
    ap.add_argument("--out", default=None)
    run(ap.parse_args())
