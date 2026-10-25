#!/usr/bin/env python3
"""The new tests of cb1r (R24 on v1 ID 1, 2026-10-25) against cb1t, the same tip on 10-23: left-right twins, slides, twists
at 5 N, and the rate set (owner, 10-25: "did we get good results from those new calibration tests?").

  twins   each twin press mirrors a grid target of the builder about y = 0. Location errors (along the surface, mm) of the
          physical estimate and of a Gaussian process from the board's 15 components to the contact point and force,
          trained on cb1t's presses (and, left out one twin at a time, on cb1t's presses and the other twins), by the side
          of the true contact (left y < -1.5 mm, right y > 1.5 mm) and force (under and over 1.5 N). Reference: the same
          GP on cb1t's presses, five folds grouped by press.
  slides  the tangential force (N) of the HEX-E against the physical estimate and the GP's, trained on cb1t's presses,
          and on those and the other slides (one slide left out at a time); the angle between the estimated and the
          measured 3D force (deg).
  twists  half the difference of the torsion about the normal (N mm) between the way out and the way back at each angle,
          by force; and the same from a GP from the board to the torsion, one twist left out at a time.
  rates   the rate set's presses at 0.045, 0.15 and 0.6 mm/s actual (the plan's v times the slider's 0.30): the board's
          change, the force at the hold and the GP's errors at each speed; its turns at 0.07, 0.28 and 0.69 deg/s.

    .venv/bin/python scripts/ur5_tip_calibration_cb1r_checks.py [--out data/20261025-114653-cb1r/checks.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration_predict as PR  # noqa: E402

EXP = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
A_DIR, B_DIR = EXP / "data/20261023-203938-cb1t", EXP / "data/20261025-114653-cb1r"
SLIDER = 0.30


def load(folder):
    z = np.load(folder / "holds.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def sub(H, m):
    return {k: v[m] for k, v in H.items()}


def cat(*Hs):
    return {k: np.concatenate([H[k] for H in Hs]) for k in Hs[0] if all(k in H for H in Hs)}


def gp(Htr):
    """A GP from the board's change to the contact point (mm) and the 3D force (N), in the pad's frame."""
    return PR.GP().fit(Htr["d_board"], np.c_[Htr["p"], Htr["f"]], PR.floors(0)[:15])


def loc_err(H, P):
    dp = P - H["p"]
    n = H["n"]
    return np.linalg.norm(dp - np.sum(dp * n, 1)[:, None] * n, axis=1)


def shear(f, n):
    return f - np.sum(f * n, 1)[:, None] * n


def angle(f1, f2):
    c = np.sum(f1 * f2, 1) / (np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def by_side(H, err):
    y, f = H["p"][:, 1], H["f_n"]
    out = {}
    for side, m in (("left", y < -1.5), ("mid", np.abs(y) <= 1.5), ("right", y > 1.5)):
        for band, b in (("under_1.5N", f < 1.5), ("over_1.5N", f >= 1.5)):
            k = m & b & np.isfinite(err)
            out[f"{side}/{band}"] = dict(n=int(k.sum()), median_mm=med(err[k]))
    return out


def twins(A, B):
    Ap = sub(A, A["kind"] == "press")
    T = sub(B, np.char.startswith(B["id"].astype(str), "twin"))
    res = dict(n_holds=int(len(T["id"])), n_twins=int(len(np.unique(T["id"]))))
    res["physical"] = by_side(T, loc_err(T, T["est_p"]))
    Y = gp(Ap).predict(T["d_board"])
    res["gp_cb1t"] = by_side(T, loc_err(T, Y[:, :3]))
    P = np.full((len(T["id"]), 3), np.nan)
    for tid in np.unique(T["id"]):
        m = T["id"] == tid
        P[m] = gp(cat(Ap, sub(T, ~m))).predict(T["d_board"][m])[:, :3]
    res["gp_cb1t_and_other_twins"] = by_side(T, loc_err(T, P))
    ids = np.unique(Ap["id"])
    fold = {i: k % 5 for k, i in enumerate(ids)}
    fk = np.array([fold[i] for i in Ap["id"]])
    P = np.full((len(Ap["id"]), 3), np.nan)
    for k in range(5):
        P[fk == k] = gp(sub(Ap, fk != k)).predict(Ap["d_board"][fk == k])[:, :3]
    res["gp_cb1t_5fold_on_cb1t"] = by_side(Ap, loc_err(Ap, P))
    res["physical_on_cb1t"] = by_side(Ap, loc_err(Ap, Ap["est_p"]))
    return res


def slides(A, B):
    Ap = sub(A, A["kind"] == "press")
    S = sub(B, B["kind"] == "slide")
    n = S["n"]
    ft = shear(S["f"], n)
    res = dict(n_holds=int(len(S["id"])), hexe_shear_N=dict(median=med(np.linalg.norm(ft, axis=1)),
                                                             max=float(np.linalg.norm(ft, axis=1).max())))
    def score(F):
        e = np.linalg.norm(shear(F, n) - ft, axis=1)
        return dict(shear_err_N=med(e), shear_est_N=med(np.linalg.norm(shear(F, n), axis=1)), angle_deg=med(angle(F, S["f"])),
                    r_components=float(np.corrcoef(shear(F, n).ravel(), ft.ravel())[0, 1]))
    res["physical"] = score(S["est_f"])
    res["gp_cb1t_presses"] = score(gp(Ap).predict(S["d_board"])[:, 3:])
    F = np.full((len(S["id"]), 3), np.nan)
    for sid in np.unique(S["id"]):
        m = S["id"] == sid
        F[m] = gp(cat(Ap, sub(S, ~m))).predict(S["d_board"][m])[:, 3:]
    res["gp_presses_and_other_slides"] = score(F)
    return res


def pairs(T):
    """(action id, angle, f_n, half the out-back difference of tn) for each angle reached twice in a twist."""
    out = []
    for aid in np.unique(T["id"]):
        r = np.nonzero(T["id"] == aid)[0]
        r = r[np.argsort(T["t"][r])]
        deg = np.round(T["turn_deg"][r], 1)
        for a in np.unique(deg):
            k = r[deg == a]
            if len(k) == 2:
                out.append((str(aid), float(a), float(T["f_n"][k].mean()), float((T["tn"][k[0]] - T["tn"][k[1]]) / 2), k))
    return out


def twists(A, B):
    T = cat(sub(A, A["kind"] == "twist"), sub(B, B["kind"] == "twist"))
    T["id"] = np.r_[np.char.add("cb1t:", A["id"][A["kind"] == "twist"].astype(str)),
                    np.char.add("cb1r:", B["id"][B["kind"] == "twist"].astype(str))]
    pr = pairs(T)
    res = {}
    for lo, hi, name in ((0, 1.5, "1N"), (1.5, 3.5, "2.5N"), (3.5, 9, "5N")):
        h = [p[3] for p in pr if lo <= p[2] < hi]
        res[name] = dict(n_pairs=len(h), median_Nmm=med(h), min_Nmm=float(min(h)) if h else None,
                         max_Nmm=float(max(h)) if h else None)
    pred = np.full(len(T["id"]), np.nan)
    for aid in np.unique(T["id"]):
        m = T["id"] == aid
        pred[m] = PR.GP().fit(T["d_board"][~m], T["tn"][~m, None], PR.floors(0)[:15]).predict(T["d_board"][m])[:, 0]
    x = np.array([p[3] for p in pr])
    y = np.array([(pred[p[4][0]] - pred[p[4][1]]) / 2 for p in pr])
    res["board_gp"] = dict(n_pairs=len(pr), r=float(np.corrcoef(x, y)[0, 1]), rms_Nmm=float(np.sqrt(np.mean((y - x) ** 2))),
                           hexe_rms_Nmm=float(np.sqrt(np.mean(x ** 2))))
    return res


def rates(A, B):
    Ap = sub(A, A["kind"] == "press")
    m = np.char.startswith(B["id"].astype(str), "rate")
    R = sub(B, m)
    tag = np.array([re.sub(r".*-(v\d+)$", r"\1", s) for s in R["id"].astype(str)])
    pt = np.array([re.sub(r"^rate-(\d+)-.*$", r"\1", s) for s in R["id"].astype(str)])
    Y = gp(Ap).predict(R["d_board"])
    loc, ferr = loc_err(R, Y[:, :3]), np.abs(Y[:, 3:] @ np.zeros(3) + np.linalg.norm(Y[:, 3:], axis=1) - np.linalg.norm(R["f"], axis=1))
    res = {}
    base = {}
    for k in range(len(R["id"])):
        if tag[k] == "v050":
            base[(pt[k], float(R["level_N"][k]))] = k
    for v, mm_s in (("v015", 0.045), ("v050", 0.15), ("v200", 0.6)):
        s = tag == v
        ratio = [np.linalg.norm(R["d_board"][k]) / np.linalg.norm(R["d_board"][base[(pt[k], float(R["level_N"][k]))]])
                 for k in np.nonzero(s)[0] if (pt[k], float(R["level_N"][k])) in base]
        res[v] = dict(actual_mm_s=mm_s, n=int(s.sum()), board_change_vs_0p15=med(ratio),
                      f_n_over_level=med(R["f_n"][s] - R["level_N"][s]), gp_loc_mm=med(loc[s]), gp_force_err_N=med(ferr[s]),
                      physical_loc_mm=med(loc_err(sub(R, s), R["est_p"][s])))
    T = sub(B, (B["kind"] == "twist") & np.char.startswith(B["id"].astype(str), "twist-0-2.5N"))
    tw = {}
    for p in pairs(T):
        v = {"twist-0-2.5N-tv004": 0.07, "twist-0-2.5N-tv016": 0.28, "twist-0-2.5N-tv040": 0.69}.get(p[0])
        if v:
            tw.setdefault(v, []).append(p[3])
    res["twist_2.5N_torsion_by_deg_s"] = {str(k): dict(n=len(v), median_Nmm=med(v)) for k, v in sorted(tw.items())}
    return res


def run(out):
    A, B = load(A_DIR), load(B_DIR)
    res = dict(twins=twins(A, B), slides=slides(A, B), twists=twists(A, B), rates=rates(A, B))
    Path(out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(B_DIR / "checks.json"))
    run(ap.parse_args().out)
