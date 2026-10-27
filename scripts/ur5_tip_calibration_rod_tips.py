#!/usr/bin/env python3
"""How well v2 ID 1 (five N52 magnets) senses on the rod indenter, against ID 3's rod (owner, 2026-10-27 12:2x, through
the lab session: "pass that analysis off to the main work pc session"). Sessions: v2 ID 1's rod passes 1 and 2
(20261027-095303-rodv2id1: 99 press actions on the face and edges to 8 N, the rate set, slides, rolls and twists) and
ID 3's rod pass 1 with the rate set (20261026-180938-rod3t: 32 press actions on the face).

Errors are medians: the location error (mm, along the surface), the force error |F_est - F| (N) and the angle between
the estimated and the measured force (deg), at 1.5 N and over, under 1.5 N and at 5 N and over.

  presses   v2 ID 1's rod presses, by region:
              physical    the viewer's estimator (the holds' est_p and est_f);
              rod         a GP trained on v2 ID 1's rod presses, five folds grouped by action;
              sphere      a GP trained on v2 ID 1's sphere presses (20261026-192158-v2id1) alone;
              sphere+rod  the sphere's presses with the rod's other folds;
              v1 scaled   a GP trained on the v1 tips' sphere presses and ID 3's rod presses, each tip's board change
                          scaled by the ratio of the strengths fitted at rest (as scripts/ur5_tip_calibration_scaled_transfer.py).
  shared    the actions both tips ran (the rate set at six face targets and three speeds, and the fast presses), for each
            tip: its physical estimate, its sphere GP, GPs of its rod presses left out by target (all of them, or only
            the set ID 3 ran) and a GP of the other tip's rod presses scaled to its strength.
  slides    v2 ID 1's slide holds: the shear error (N), the shear estimated (N) and the force angle (deg) of the physical
            estimate, a GP of the sphere's presses and slides, the rod presses' GP and the rod presses with the other
            slides (one slide out).
  rolls, twists  location and force errors of the same estimators.
  rates     each tip's rate set: the board change at 0.045 and 2.0 mm/s over the one at 0.6 mm/s (same target and level),
            the force at the hold over the level, and the rod GP's location error.

    .venv/bin/python scripts/ur5_tip_calibration_rod_tips.py [--out data/20261027-095303-rodv2id1/rod_tips.json]
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
import ur5_tip_calibration_cb1r_checks as K  # noqa: E402

D = K.EXP / "data"
ROD = {"v2 ID 1": "20261027-095303-rodv2id1", "ID 3": "20261026-180938-rod3t"}
SPHERE = {"v2 ID 1": "20261026-192158-v2id1", "ID 3": "20261026-134522-cb3t"}
V1_SPHERE = ("20261023-203938-cb1t", "20261025-114653-cb1r", "20261025-152920-cb2t", "20261026-134522-cb3t")
SPEEDS = {"v015": 0.045, "v200": 0.6, "v667": 2.0}      # the plan's v times the slider's 0.30, mm/s


def load(s, kinds=None):
    H = K.load(D / s)
    m = np.isfinite(H["d_board"]).all(1) & np.isfinite(H["p"]).all(1) & np.isfinite(H["f"]).all(1)
    if kinds:
        m &= np.isin(H["kind"], kinds)
    return K.sub(H, m), json.loads((D / s / "calib_rest.json").read_text())["k"]


def scaled(H, k_from, k_to):
    H = dict(H)
    H["d_board"] = H["d_board"] * (k_to / k_from)
    return H


def score(H, Y):
    e = K.loc_err(H, Y[:, :3])
    F = Y[:, 3:]
    ef = np.linalg.norm(F - H["f"], axis=1)
    ang = K.angle(F, H["f"])
    hi, top = H["f_n"] >= 1.5, H["f_n"] >= 5.0
    out = dict(n_over=int(hi.sum()), n_under=int((~hi).sum()), loc_over_1p5N=K.med(e[hi]), loc_under_1p5N=K.med(e[~hi]),
               force_err_over_1p5N=K.med(ef[hi]), force_err_under_1p5N=K.med(ef[~hi]), angle_over_1p5N=K.med(ang[hi]))
    if top.any():
        out.update(n_over_5N=int(top.sum()), loc_over_5N=K.med(e[top]), force_err_over_5N=K.med(ef[top]))
    return out


def by_region(H, Y):
    out = {"all": score(H, Y)}
    for r in np.unique(H["region"].astype(str)):
        m = H["region"].astype(str) == r
        out[r] = score(K.sub(H, m), Y[m])
    return out


def groups(ids, how):
    a = ids.astype(str)
    if how == "target":                  # rod-rate-017-v200 and rod-fast4-017 share target 017
        return np.array([(re.search(r"-(\d{3})", s) or re.search(r"(.*)", s)).group(1) for s in a])
    return a


def cv(H, extra=None, how="action", k=5):
    """Predictions for H: k folds over its actions (or its targets), each trained on extra plus H's other folds."""
    g = groups(H["id"], how)
    u = np.unique(g)
    fold = {x: i % k for i, x in enumerate(np.random.default_rng(0).permutation(u))}
    f = np.array([fold[x] for x in g])
    Y = np.full((len(g), 6), np.nan)
    for q in range(k):
        m = f == q
        if m.any():
            tr = K.sub(H, ~m)
            Y[m] = K.gp(K.cat(extra, tr) if extra is not None else tr).predict(H["d_board"][m])
    return Y


def presses(Rp, Sv, A1):
    Y = dict(physical=np.c_[Rp["est_p"], Rp["est_f"]], rod=cv(Rp), sphere=K.gp(Sv).predict(Rp["d_board"]),
             sphere_rod=cv(Rp, extra=Sv), v1_scaled=K.gp(A1).predict(Rp["d_board"]))
    return {name: by_region(Rp, y) for name, y in Y.items()}


def shared(R, S, k):
    """The actions both rod sessions ran: each tip's estimators on them. The rod GPs leave out a sixth of the targets at a
    time (a target's grid, rate and fast presses together): "own rod" is trained on all the tip's other rod presses,
    "own rod, ID 3's set" only on the shared actions, the pivots and the reference presses, which is all ID 3 ran."""
    ids = sorted(set(R["v2 ID 1"]["id"].astype(str)) & set(R["ID 3"]["id"].astype(str)))
    ids = [i for i in ids if i.startswith(("rod-rate", "rod-fast"))]
    out = dict(actions=ids)
    for t, o in (("v2 ID 1", "ID 3"), ("ID 3", "v2 ID 1")):
        H = R[t]
        a = H["id"].astype(str)
        m = np.isin(a, ids)
        T = K.sub(H, m)
        same = m | ~(np.char.startswith(a, "rod-grid") | np.char.startswith(a, "rod-more"))
        Hs = K.sub(H, same)
        other = scaled(R[o], k[o], k[t])
        out[t] = dict(n_holds=int(m.sum()), physical=score(T, np.c_[T["est_p"], T["est_f"]]),
                      sphere=score(T, K.gp(S[t]).predict(T["d_board"])),
                      own_rod=score(T, cv(H, how="target", k=6)[m]),
                      own_rod_id3_set=score(T, cv(Hs, how="target", k=6)[np.isin(Hs["id"].astype(str), ids)]),
                      other_tip_rod_scaled=score(T, K.gp(other).predict(T["d_board"])))
    return out


def slides(Rp, Rs, Sv):
    n = Rs["n"]
    ft = K.shear(Rs["f"], n)

    def sc(F):
        return dict(shear_err_N=K.med(np.linalg.norm(K.shear(F, n) - ft, axis=1)),
                    shear_est_N=K.med(np.linalg.norm(K.shear(F, n), axis=1)), angle_deg=K.med(K.angle(F, Rs["f"])),
                    loc_mm=None)

    out = dict(n_holds=int(len(Rs["id"])), n_actions=int(len(np.unique(Rs["id"]))),
               hexe_shear_N=K.med(np.linalg.norm(ft, axis=1)))
    Yp = np.c_[Rs["est_p"], Rs["est_f"]]
    Ys = K.gp(Sv).predict(Rs["d_board"])
    Yr = K.gp(Rp).predict(Rs["d_board"])
    Yo = np.full((len(Rs["id"]), 6), np.nan)
    for sid in np.unique(Rs["id"]):
        m = Rs["id"] == sid
        Yo[m] = K.gp(K.cat(Rp, K.sub(Rs, ~m))).predict(Rs["d_board"][m])
    for name, Y in (("physical", Yp), ("sphere", Ys), ("rod_presses", Yr), ("rod_presses_and_other_slides", Yo)):
        r = sc(Y[:, 3:])
        r["loc_mm"] = K.med(K.loc_err(Rs, Y[:, :3]))
        out[name] = r
    return out


def moving(Rp, Rs, H, Sv):
    """Rolls or twists: location and force errors."""
    Y = dict(physical=np.c_[H["est_p"], H["est_f"]], sphere=K.gp(Sv).predict(H["d_board"]),
             rod_presses=K.gp(Rp).predict(H["d_board"]), rod_presses_and_slides=K.gp(K.cat(Rp, Rs)).predict(H["d_board"]))
    out = dict(n_holds=int(len(H["id"])), n_actions=int(len(np.unique(H["id"]))))
    for name, y in Y.items():
        out[name] = dict(loc_mm=K.med(K.loc_err(H, y[:, :3])), force_err_N=K.med(np.linalg.norm(y[:, 3:] - H["f"], axis=1)),
                         angle_deg=K.med(K.angle(y[:, 3:], H["f"])))
    return out


def rates(H):
    ids = H["id"].astype(str)
    m = np.char.startswith(ids, "rod-rate")
    R = K.sub(H, m)
    a = R["id"].astype(str)
    tag = np.array([s.rsplit("-", 1)[1] for s in a])
    tgt = np.array([s.split("-")[2] for s in a])
    Y = cv(H, how="target", k=6)[m]
    loc = K.loc_err(R, Y[:, :3])
    base = {(tgt[i], float(R["level_N"][i])): i for i in range(len(a)) if tag[i] == "v200"}
    out = {}
    for v, mm_s in SPEEDS.items():
        s = tag == v
        ratio = [np.linalg.norm(R["d_board"][i]) / np.linalg.norm(R["d_board"][base[(tgt[i], float(R["level_N"][i]))]])
                 for i in np.nonzero(s)[0] if (tgt[i], float(R["level_N"][i])) in base]
        out[v] = dict(actual_mm_s=mm_s, n=int(s.sum()), board_change_vs_0p6=K.med(ratio),
                      f_n_over_level_N=K.med(R["f_n"][s] - R["level_N"][s]), rod_gp_loc_mm=K.med(loc[s]),
                      physical_loc_mm=K.med(K.loc_err(K.sub(R, s), R["est_p"][s])))
    return out


def run(a):
    R, k = {}, {}
    for t, s in ROD.items():
        R[t], k[t] = load(s, ["press"])
    S = {t: load(s, ["press"])[0] for t, s in SPHERE.items()}             # sphere presses: presses and shared
    Sa = load(SPHERE["v2 ID 1"], ["press", "slide"])[0]                   # and slides: slides, rolls, twists
    Rs, _ = load(ROD["v2 ID 1"], ["slide"])
    Rr, _ = load(ROD["v2 ID 1"], ["roll"])
    Rt, _ = load(ROD["v2 ID 1"], ["twist"])
    v1 = []
    for s in V1_SPHERE:
        H, kk = load(s, ["press"])
        v1.append(scaled(H, kk, k["v2 ID 1"]))
    A1 = K.cat(*v1, scaled(R["ID 3"], k["ID 3"], k["v2 ID 1"]))
    res = dict(k=k, n_press_holds={t: int(len(H["id"])) for t, H in R.items()},
               n_press_actions={t: int(len(np.unique(H["id"]))) for t, H in R.items()},
               sphere_holds={t: int(len(H["id"])) for t, H in S.items()})
    res["presses"] = presses(R["v2 ID 1"], S["v2 ID 1"], A1)
    print("presses", json.dumps({n: {r: {q: (round(v, 2) if isinstance(v, float) else v) for q, v in d.items()}
                                     for r, d in x.items()} for n, x in res["presses"].items()}, indent=0), flush=True)
    res["shared"] = shared(R, S, k)
    print("shared", json.dumps({t: res["shared"][t] for t in ROD}, indent=0, default=lambda v: round(float(v), 2)), flush=True)
    res["slides"] = slides(R["v2 ID 1"], Rs, Sa)
    res["rolls"] = moving(R["v2 ID 1"], Rs, Rr, Sa)
    res["twists"] = moving(R["v2 ID 1"], Rs, Rt, Sa)
    print("slides/rolls/twists", json.dumps({x: res[x] for x in ("slides", "rolls", "twists")}, indent=0,
                                            default=lambda v: round(float(v), 2)), flush=True)
    res["rates"] = {t: rates(H) for t, H in R.items()}
    print("rates", json.dumps(res["rates"], indent=0, default=lambda v: round(float(v), 3)), flush=True)
    out = Path(a.out)
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(D / ROD["v2 ID 1"] / "rod_tips.json"))
    run(ap.parse_args())
