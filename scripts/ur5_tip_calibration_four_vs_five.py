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
  angle_deg     the median angle between the estimated and the measured force;
  gain_uT_per_N the median of |d_board| / f_n over the holds of a bin: the board's signal per newton;
  noise_uT      the board's single-record noise at the holds: the standard deviation of each of the 15 components over
                the central 0.4 s of a hold after a linear trend is removed, the median over components and holds;
  nef_N         noise_uT / gain_uT_per_N: the force whose signal equals one record's noise.
Bins of f_n: (0, 1], (1, 2], (2, 3.5] and (3.5, 8] N; regions face, edge and flank.

    .venv/bin/python scripts/ur5_tip_calibration_four_vs_five.py V3_HOLDS_DIR --v3-session SESSION [--out FILE]
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


def presses(folder):
    H = K.load(Path(folder))
    m = ((H["kind"] == "press") & np.isfinite(H["d_board"]).all(1) & np.isfinite(H["p"]).all(1)
         & np.isfinite(H["f"]).all(1) & np.isfinite(H["f_n"]) & (H["f_n"] > 0.05))
    return K.sub(H, m)


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
        rows = {"all": HX.stats(E, np.ones(len(H["id"]), bool))}
        for lo, hi in BINS:
            m = HX.in_bin(H, lo, hi)
            if m.any():
                rows[f"{lo:g}-{hi:g}"] = HX.stats(E, m)
        for r in REGIONS:
            m = H["region"] == r
            if m.any():
                rows[r] = HX.stats(E, m)
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
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    H3, H2 = presses(a.v3), presses(V2)
    ids = sorted(set(H3["id"].astype(str)) & set(H2["id"].astype(str)))
    H3 = K.sub(H3, np.isin(H3["id"].astype(str), ids))
    H2 = K.sub(H2, np.isin(H2["id"].astype(str), ids))
    res = dict(script="scripts/ur5_tip_calibration_four_vs_five.py", v3=str(a.v3), v2=str(V2.relative_to(ROOT)),
               actions=len(ids), holds=dict(v3=len(H3["id"]), v2=len(H2["id"])),
               regions={r: int(len(set(H3["id"][H3["region"] == r].astype(str)))) for r in REGIONS})
    res["four_v3id1"], res["five_v2id1"] = score(H3, a.v3_session), score(H2, V2_SESSION)
    for name in ("gp", "gp_direction"):
        for key in ["all"] + [f"{lo:g}-{hi:g}" for lo, hi in BINS] + list(REGIONS):
            r3, r2 = res["four_v3id1"][name].get(key), res["five_v2id1"][name].get(key)
            if r3 and r2:
                print(f"{name:13s} {key:8s} n {r3['n']:4d}/{r2['n']:4d}  loc {r3['loc_mm']:.3f} / {r2['loc_mm']:.3f} mm  "
                      f"force {100 * r3['force_rel_err']:.1f} / {100 * r2['force_rel_err']:.1f} %  "
                      f"angle {r3['angle_deg']:.1f} / {r2['angle_deg']:.1f} deg   (four / five)")
    for key, v3 in res["four_v3id1"]["gain_uT_per_N"].items():
        v2 = res["five_v2id1"]["gain_uT_per_N"].get(key)
        n3, n2 = res["four_v3id1"]["nef_N"][key], res["five_v2id1"]["nef_N"].get(key)
        print(f"gain {key:8s} {v3:.1f} / {v2:.1f} uT/N, noise-equivalent force {1000 * n3:.0f} / {1000 * n2:.0f} mN (four / five)"
              if v2 else f"gain {key:8s} {v3:.1f} uT/N (four)")
    print(f"noise {res['four_v3id1']['noise_uT']:.2f} / {res['five_v2id1']['noise_uT']:.2f} uT per record (four / five)")
    print(f"{len(ids)} actions; holds {len(H3['id'])} (four) and {len(H2['id'])} (five)")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print("->", a.out)


if __name__ == "__main__":
    main()
