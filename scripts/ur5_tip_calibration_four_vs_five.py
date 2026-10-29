#!/usr/bin/env python3
"""Four magnets (v3 ID 1, pad 05R4) against five (v2 ID 1, the b_v2 checkerboard) on the UR5's low sphere: the same
press actions of the sphere stage, scored on each tip's own holds (owner, 2026-10-27: v3 ID 1's suite decides between
the two layouts).

Data: the holds of v2 ID 1's sphere session (data/20261026-192158-v2id1/holds.npz) and of v3 ID 1's (the output of
scripts/ur5_tip_calibration_analysis.py SESSION --case four_05R4), press holds only, from the actions that both ran,
matched by action id (v3 ID 1's stage 2 is v2 ID 1's sphere session with the suite's cuts). f_n is the HEX-E's normal
force at the hold (N), p the registered contact point (mm, pad frame), f the 3D force (N), d_board the board's change
(15 components, uT).

Scores, per tip, under five folds grouped by action (the same actions in the same folds for both tips):
  gp            a Gaussian process from the board's 15 components to p and f (scripts/ur5_tip_calibration_cb1r_checks.gp);
  gp_direction  a GP from the change's direction d/|d| to p and f/|d| (ur5_tip_calibration_hard_extrapolation);
  loc_mm        the median location error along the surface (mm); force_rel_err the median of |f_est| - |f| over |f|;
  angle_deg     the median angle between the estimated and the measured force (deg);
  normal_deg    the median angle between the pad's surface normal at the estimated point and at the registered contact
                point (deg), each interpolated from the four nearest FE surface nodes: the error of the contact's
                direction (normal_p90_deg its 90th percentile);
  gain_uT_per_N the median of |d_board| / f_n over the holds of a bin: the board's signal per newton;
  noise_uT      the board's single-record noise at the holds: the standard deviation of each of the 15 components over
                the central 0.4 s of a hold after a linear trend is removed, the median over components and holds;
  nef_N         noise_uT / gain_uT_per_N: the force whose signal equals one record's noise.
Slides (when both sessions have at least five matched slides): the turns' holds of each slide at 1 N, scored on the
shear, the part of the measured 3D force across the contact normal. Each GP is trained on the tip's matched presses
(presses) and, one slide left out at a time, on those and the tip's other slides (presses_and_other_slides), as
scripts/ur5_tip_calibration_cb1r_checks.slides does: shear_err_N the median norm of the estimated shear less the
measured, angle_deg the median angle between the estimated and the measured force.
Twists (when both sessions have at least five matched twists): the torsion about the contact normal, tn (N mm), scored
as scripts/ur5_tip_calibration_cb1r_checks.twists does on half the difference between the way out and the way back at
every angle a twist reached twice (the torsional friction). hexe_* is the HEX-E's by force (1, 2.5 and 5 N); a GP from
the board's 15 components to tn, one twist left out at a time, is trained on the tip's other twists (gp_twists) or on
those and its matched presses (gp_presses_and_twists): r its correlation with the HEX-E's half differences, rms_Nmm the
RMS of its error, hexe_rms_Nmm the RMS of the HEX-E's.
Bins of f_n: (0, 1], (1, 2], (2, 3.5] and (3.5, 8] N; regions face, edge and flank.

    .venv/bin/python scripts/ur5_tip_calibration_four_vs_five.py V3_HOLDS_DIR --v3-session SESSION [--out FILE]
    the rod: ... V3_ROD_HOLDS_DIR --v3-session SESSION --v2 data/20261027-095303-rodv2id1 --v2-session lab/20261027-095303-rodv2id1
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
import ur5_tip_calibration_hard_extrapolation as HX  # noqa: E402

V2 = K.EXP / "data/20261026-192158-v2id1"
V2_SESSION = K.EXP / "lab/20261026-192158-v2id1"
BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 8.0))
REGIONS = ("face", "edge", "flank")


def holds(folder, kind="press"):
    H = K.load(Path(folder))
    m = ((H["kind"] == kind) & np.isfinite(H["d_board"]).all(1) & np.isfinite(H["p"]).all(1)
         & np.isfinite(H["f"]).all(1) & np.isfinite(H["f_n"]) & (H["f_n"] > 0.05))
    return K.sub(H, m)


def shear(F, n):
    return F - np.sum(F * n, axis=1)[:, None] * n


def slide_scores(Hp, Hs):
    ft = shear(Hs["f"], Hs["n"])

    def sc(F):
        e = np.linalg.norm(shear(F, Hs["n"]) - ft, axis=1)
        return dict(shear_err_N=float(np.nanmedian(e)), shear_est_N=float(np.nanmedian(np.linalg.norm(shear(F, Hs["n"]), axis=1))),
                    angle_deg=float(np.nanmedian(K.angle(F, Hs["f"]))))
    out = dict(n_holds=int(len(Hs["id"])), n_slides=int(len(np.unique(Hs["id"]))),
               hexe_shear_N=float(np.median(np.linalg.norm(ft, axis=1))))
    for name, fitter in (("gp", HX.fit_gp), ("gp_direction", HX.fit_gp_direction)):
        pred, _ = fitter(Hp, {})
        out[f"{name}_presses"] = sc(pred(Hs)[1])
        F = np.full((len(Hs["id"]), 3), np.nan)
        for sid in np.unique(Hs["id"]):
            m = Hs["id"] == sid
            pred, _ = fitter(K.cat(Hp, K.sub(Hs, ~m)), {})
            F[m] = pred(K.sub(Hs, m))[1]
        out[f"{name}_presses_and_other_slides"] = sc(F)
    return out


def twist_scores(Hp, T):
    pr = K.pairs(T)
    out = dict(n_holds=int(len(T["id"])), n_twists=int(len(np.unique(T["id"]))), n_pairs=len(pr))
    for lo, hi, name in ((0.0, 1.5, "1N"), (1.5, 3.5, "2.5N"), (3.5, 9.0, "5N")):
        h = [abs(p[3]) for p in pr if lo <= p[2] < hi]
        out[f"hexe_{name}"] = dict(n_pairs=len(h), median_Nmm=K.med(h))
    x = np.array([p[3] for p in pr])
    for name, base in (("gp_twists", None), ("gp_presses_and_twists", Hp)):
        pred = np.full(len(T["id"]), np.nan)
        for aid in np.unique(T["id"]):
            m = T["id"] == aid
            Tr = K.sub(T, ~m) if base is None else K.cat(base, K.sub(T, ~m))
            pred[m] = K.PR.GP().fit(Tr["d_board"], Tr["tn"][:, None], K.PR.floors(0)[:15]).predict(T["d_board"][m])[:, 0]
        y = np.array([(pred[p[4][0]] - pred[p[4][1]]) / 2 for p in pr])
        out[name] = dict(r=float(np.corrcoef(x, y)[0, 1]), rms_Nmm=float(np.sqrt(np.mean((y - x) ** 2))),
                         hexe_rms_Nmm=float(np.sqrt(np.mean(x ** 2))))
    return out


def normal_err(H, P, k=4):
    """Angle (deg) between the pad's surface normal at the estimated point and at the registered contact point, each
    the inverse-distance mean of the k nearest FE surface nodes' normals (the nodes lie about 1 mm apart)."""
    from scipy.spatial import cKDTree
    b = np.load(HX.PR.BASIS)
    tree, nrm = cKDTree(b["pts"]), b["normals"]

    def smooth(X):
        d, idx = tree.query(X, k=k)
        v = np.einsum("nk,nkj->nj", 1.0 / np.maximum(d, 1e-6), nrm[idx])
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    ok = np.isfinite(P).all(1)
    out = np.full(len(P), np.nan)
    c = np.clip(np.sum(smooth(P[ok]) * smooth(H["p"][ok]), axis=1), -1.0, 1.0)
    out[ok] = np.degrees(np.arccos(c))
    return out


def cv(H, fitter):
    fo = HX.folds(H)
    P, F = np.full((len(fo), 3), np.nan), np.full((len(fo), 3), np.nan)
    for k in range(HX.FOLDS):
        te = fo == k
        if not te.any() or te.all():
            continue
        pred, _ = fitter(K.sub(H, ~te), {})
        P[te], F[te], _ = pred(K.sub(H, te))
    return P, F


def noise(H, session, half_s=0.2):
    """Each hold's single-record noise (uT): the median over the 15 components of their standard deviations over
    t +- half_s, a linear trend removed (the board's records, chip columns x, y, z; column 0 is the die temperature)."""
    b = np.memmap(Path(session) / "board.bin", dtype=[("t", "<f8"), ("r", "<f4", (20,))], mode="r")
    t = b["t"]
    out = np.full(len(H["id"]), np.nan)
    for i, tc in enumerate(H["t"]):
        i0, i1 = np.searchsorted(t, tc - half_s), np.searchsorted(t, tc + half_s)
        if i1 - i0 < 20:
            continue
        r = np.asarray(b["r"][i0:i1], float).reshape(-1, 5, 4)[:, :, 1:].reshape(-1, 15)
        x = np.asarray(t[i0:i1]) - tc
        A = np.c_[np.ones_like(x), x]
        res = r - A @ np.linalg.lstsq(A, r, rcond=None)[0]
        out[i] = float(np.median(res.std(axis=0)))
    return out


def score(H, session):
    out = {}
    sig = noise(H, session)
    gain = np.linalg.norm(H["d_board"], axis=1) / H["f_n"]
    for name, fitter in (("gp", HX.fit_gp), ("gp_direction", HX.fit_gp_direction)):
        P, F = cv(H, fitter)
        E = HX.errors(H, P, F)
        nd = normal_err(H, P)

        def st(m):
            row = HX.stats(E, m)
            x = nd[m & np.isfinite(nd)]
            row["normal_deg"] = float(np.median(x)) if len(x) else None
            row["normal_p90_deg"] = float(np.percentile(x, 90)) if len(x) else None
            return row
        rows = {"all": st(np.ones(len(H["id"]), bool))}
        for lo, hi in BINS:
            m = HX.in_bin(H, lo, hi)
            if m.any():
                rows[f"{lo:g}-{hi:g}"] = st(m)
        for r in REGIONS:
            m = H["region"] == r
            if m.any():
                rows[r] = st(m)
        out[name] = rows
    out["gain_uT_per_N"] = {f"{lo:g}-{hi:g}": float(np.median(gain[HX.in_bin(H, lo, hi)]))
                            for lo, hi in BINS if HX.in_bin(H, lo, hi).any()}
    out["gain_uT_per_N"].update({r: float(np.median(gain[(H["region"] == r) & HX.in_bin(H, 0.5, 3.5)]))
                                 for r in REGIONS if ((H["region"] == r) & HX.in_bin(H, 0.5, 3.5)).any()})
    out["noise_uT"] = float(np.nanmedian(sig))
    out["nef_N"] = {k: out["noise_uT"] / v for k, v in out["gain_uT_per_N"].items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("v3")
    ap.add_argument("--v3-session", required=True, help="v3 ID 1's session folder (its board.bin, for the noise)")
    ap.add_argument("--v2", default=str(V2), help="v2 ID 1's holds (default its sphere session; the rod's: "
                    "data/20261027-095303-rodv2id1)")
    ap.add_argument("--v2-session", default=str(V2_SESSION), help="v2 ID 1's session folder (its board.bin)")
    ap.add_argument("--v3-key", default="four_v3id1", help="the four-magnet tip's key in the output (four_v3id2 for v3 ID 2)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    k3 = a.v3_key
    v2, v2_session = Path(a.v2).resolve(), Path(a.v2_session)
    H3, H2 = holds(a.v3), holds(v2)
    ids = sorted(set(H3["id"].astype(str)) & set(H2["id"].astype(str)))
    H3 = K.sub(H3, np.isin(H3["id"].astype(str), ids))
    H2 = K.sub(H2, np.isin(H2["id"].astype(str), ids))
    res = dict(script="scripts/ur5_tip_calibration_four_vs_five.py", v3=str(a.v3), v2=str(v2.relative_to(ROOT)),
               actions=len(ids), holds=dict(v3=len(H3["id"]), v2=len(H2["id"])),
               regions={r: int(len(set(H3["id"][H3["region"] == r].astype(str)))) for r in REGIONS})
    res[k3], res["five_v2id1"] = score(H3, a.v3_session), score(H2, v2_session)
    for name in ("gp", "gp_direction"):
        for key in ["all"] + [f"{lo:g}-{hi:g}" for lo, hi in BINS] + list(REGIONS):
            r3, r2 = res[k3][name].get(key), res["five_v2id1"][name].get(key)
            if r3 and r2:
                print(f"{name:13s} {key:8s} n {r3['n']:4d}/{r2['n']:4d}  loc {r3['loc_mm']:.3f} / {r2['loc_mm']:.3f} mm  "
                      f"force {100 * r3['force_rel_err']:.1f} / {100 * r2['force_rel_err']:.1f} %  "
                      f"angle {r3['angle_deg']:.1f} / {r2['angle_deg']:.1f} deg  normal {r3['normal_deg']:.1f} / "
                      f"{r2['normal_deg']:.1f} deg   (four / five)")
    S3, S2 = holds(a.v3, "slide"), holds(v2, "slide")
    sids = sorted(set(S3["id"].astype(str)) & set(S2["id"].astype(str)))
    if len(sids) >= 5:
        S3, S2 = (K.sub(S, np.isin(S["id"].astype(str), sids)) for S in (S3, S2))
        res["slides"] = {k3: slide_scores(H3, S3), "five_v2id1": slide_scores(H2, S2)}
        for k in ("gp_presses", "gp_presses_and_other_slides", "gp_direction_presses", "gp_direction_presses_and_other_slides"):
            r3, r2 = res["slides"][k3][k], res["slides"]["five_v2id1"][k]
            print(f"slides {k:38s} shear error {r3['shear_err_N']:.3f} / {r2['shear_err_N']:.3f} N, angle "
                  f"{r3['angle_deg']:.1f} / {r2['angle_deg']:.1f} deg (four / five; {len(sids)} slides)")
    T3, T2 = holds(a.v3, "twist"), holds(v2, "twist")
    tids = sorted(set(T3["id"].astype(str)) & set(T2["id"].astype(str)))
    if len(tids) >= 5:
        T3, T2 = (K.sub(T, np.isin(T["id"].astype(str), tids) & np.isfinite(T["tn"])) for T in (T3, T2))
        res["twists"] = {"ids": tids, k3: twist_scores(H3, T3), "five_v2id1": twist_scores(H2, T2)}
        t3, t2 = res["twists"][k3], res["twists"]["five_v2id1"]
        for k in ("1N", "2.5N", "5N"):
            print(f"twists HEX-E torsion at {k:4s} {t3['hexe_' + k]['median_Nmm']} / {t2['hexe_' + k]['median_Nmm']} N mm "
                  f"({t3['hexe_' + k]['n_pairs']} / {t2['hexe_' + k]['n_pairs']} pairs; four / five)")
        for k in ("gp_twists", "gp_presses_and_twists"):
            print(f"twists {k:22s} r {t3[k]['r']:.2f} / {t2[k]['r']:.2f}, rms {t3[k]['rms_Nmm']:.2f} / {t2[k]['rms_Nmm']:.2f} "
                  f"N mm against the HEX-E's {t3[k]['hexe_rms_Nmm']:.2f} / {t2[k]['hexe_rms_Nmm']:.2f} (four / five; "
                  f"{len(tids)} twists)")
    for key, v3 in res[k3]["gain_uT_per_N"].items():
        v2 = res["five_v2id1"]["gain_uT_per_N"].get(key)
        n3, n2 = res[k3]["nef_N"][key], res["five_v2id1"]["nef_N"].get(key)
        print(f"gain {key:8s} {v3:.1f} / {v2:.1f} uT/N, noise-equivalent force {1000 * n3:.0f} / {1000 * n2:.0f} mN (four / five)"
              if v2 else f"gain {key:8s} {v3:.1f} uT/N (four)")
    print(f"noise {res[k3]['noise_uT']:.2f} / {res['five_v2id1']['noise_uT']:.2f} uT per record (four / five)")
    print(f"{len(ids)} actions; holds {len(H3['id'])} (four) and {len(H2['id'])} (five)")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print("->", a.out)


if __name__ == "__main__":
    main()
