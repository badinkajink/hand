#!/usr/bin/env python3
"""The tests of cb2t (R24 on v1 ID 2, mount v7, 2026-10-25, passes 1 to 6) with its own grid as the training set, in the
terms of scripts/ur5_tip_calibration_cb1r_checks.py, and two more:

  density   location error (mm) at pass 6's 37 new targets of a GP trained on the grid, the twins and the fast grid, and
            of the same GP with the other new targets added (five folds grouped by target); by contact speed.
  transfer  a GP trained on ID 1's presses (cb1t and cb1r) applied to ID 2's grid, against ID 2's own (five folds).

Groups of held-out holds (twins, slides, new targets) are taken five folds at a time, grouped by action.

    .venv/bin/python scripts/ur5_tip_calibration_cb2t_checks.py [--out data/20261025-152920-cb2t/checks.json]
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
B_DIR = D / "20261025-152920-cb2t"


def ids(H):
    return H["id"].astype(str)


def folds(H, m, k=5):
    """Fold index per hold of H[m], grouped by action id, in time order."""
    u = np.unique(ids(H)[m])
    f = {a: i % k for i, a in enumerate(u)}
    return np.array([f[a] for a in ids(H)[m]])


def cv(train, test, out):
    """Predictions for test's holds, five folds over test's actions, each fold trained on train plus test's other folds."""
    fk = folds(test, np.ones(len(test["id"]), bool))
    Y = np.full((len(test["id"]), 6), np.nan)
    for k in range(5):
        m = fk == k
        if m.any():
            Y[m] = K.gp(K.cat(train, K.sub(test, ~m))).predict(test["d_board"][m])
    return Y[:, :3] if out == "p" else Y[:, 3:]


def grid_of(B):
    s = ids(B)
    return (B["kind"] == "press") & (np.char.startswith(s, "grid-") | np.char.startswith(s, "ref-") | np.char.startswith(s, "final-ref"))


def twins(G, B):
    T = K.sub(B, np.char.startswith(ids(B), "twin"))
    return dict(n_twins=int(len(np.unique(ids(T)))), n_holds=int(len(T["id"])),
                physical=K.by_side(T, K.loc_err(T, T["est_p"])),
                gp_grid=K.by_side(T, K.loc_err(T, K.gp(G).predict(T["d_board"])[:, :3])),
                gp_grid_and_other_twins=K.by_side(T, K.loc_err(T, cv(G, T, "p"))),
                gp_grid_5fold_on_grid=K.by_side(G, K.loc_err(G, cv({k: v[:0] for k, v in G.items()}, G, "p"))))


def slides(P, B):
    S = K.sub(B, B["kind"] == "slide")
    n, ft = S["n"], K.shear(S["f"], S["n"])

    def score(F):
        e = np.linalg.norm(K.shear(F, n) - ft, axis=1)
        return dict(shear_err_N=K.med(e), shear_est_N=K.med(np.linalg.norm(K.shear(F, n), axis=1)),
                    angle_deg=K.med(K.angle(F, S["f"])), r_components=float(np.corrcoef(K.shear(F, n).ravel(), ft.ravel())[0, 1]))
    return dict(n_slides=int(len(np.unique(ids(S)))), n_holds=int(len(S["id"])),
                hexe_shear_N=dict(median=K.med(np.linalg.norm(ft, axis=1)), max=float(np.linalg.norm(ft, axis=1).max())),
                physical=score(S["est_f"]), gp_presses=score(K.gp(P).predict(S["d_board"])[:, 3:]),
                gp_presses_and_other_slides=score(cv(P, S, "f")))


def density(B):
    s = ids(B)
    base = K.sub(B, (B["kind"] == "press") & ~np.char.startswith(s, "dense") & ~np.char.startswith(s, "rate"))
    N = K.sub(B, (B["kind"] == "press") & np.char.startswith(s, "dense"))
    e0 = K.loc_err(N, K.gp(base).predict(N["d_board"])[:, :3])
    e1 = K.loc_err(N, cv(base, N, "p"))
    hi = N["f_n"] >= 1.5
    plan = json.loads((K.EXP / "lab/20261025-plan_sphere_lab_r24_pass6_cb2t.json").read_text())
    v = {a["id"]: a.get("v") for a in plan["actions"]}
    sp = np.array([v.get(a, np.nan) for a in ids(N)], float) * 0.30 * 1e3
    out = dict(n_targets=int(len(np.unique(ids(N)))), without=dict(over_1p5N=K.med(e0[hi]), under_1p5N=K.med(e0[~hi])),
               with_other_new=dict(over_1p5N=K.med(e1[hi]), under_1p5N=K.med(e1[~hi])), by_speed_mm_s={})
    for x in np.unique(sp[np.isfinite(sp)]):
        m = sp == x
        out["by_speed_mm_s"][f"{x:.3g}"] = dict(n=int(m.sum()), with_other_new_over_1p5N=K.med(e1[m & hi]),
                                                with_other_new_under_1p5N=K.med(e1[m & ~hi]))
    return out


def transfer(G):
    A1 = K.cat(*[K.sub(H, H["kind"] == "press") for H in (K.load(D / "20261023-203938-cb1t"), K.load(D / "20261025-114653-cb1r"))])
    e_t = K.loc_err(G, K.gp(A1).predict(G["d_board"])[:, :3])
    e_o = K.loc_err(G, cv({k: v[:0] for k, v in G.items()}, G, "p"))
    e_p = K.loc_err(G, G["est_p"])
    hi = G["f_n"] >= 1.5
    r = lambda e: dict(over_1p5N=K.med(e[hi]), under_1p5N=K.med(e[~hi]))   # noqa: E731
    return dict(gp_id1=r(e_t), gp_own_5fold=r(e_o), physical_own_rest=r(e_p))


def rates(G, B):
    s = ids(B)
    R = K.sub(B, np.char.startswith(s, "rate"))
    Y = K.gp(G).predict(R["d_board"])
    tag = np.array([re.sub(r".*-(v\d+)$", r"\1", x) for x in ids(R)])
    pt = np.array([re.sub(r"^rate-(\d+)-.*$", r"\1", x) for x in ids(R)])
    gb = {(re.sub(r"^grid-(\d+)$", r"\1", a), float(l)): k for k, (a, l) in enumerate(zip(ids(G), G["level_N"]))}
    out = {}
    for v, mm in (("v015", 0.045), ("v200", 0.6), ("v667", 2.0)):
        m = tag == v
        ratio = [np.linalg.norm(R["d_board"][k]) / np.linalg.norm(G["d_board"][gb[(pt[k], float(R["level_N"][k]))]])
                 for k in np.nonzero(m)[0] if (pt[k], float(R["level_N"][k])) in gb]
        out[f"{mm}"] = dict(n=int(m.sum()), board_change_vs_0p15=K.med(ratio), f_n_over_level=K.med(R["f_n"][m] - R["level_N"][m]),
                            gp_loc_mm=K.med(K.loc_err(K.sub(R, m), Y[m, :3])))
    return out


def run(out):
    B = K.load(B_DIR)
    G = K.sub(B, grid_of(B))
    P = K.sub(B, B["kind"] == "press")
    res = dict(n_holds=int(len(B["id"])), n_grid_holds=int(len(G["id"])), transfer=transfer(G), twins=twins(G, B),
               density=density(B), slides=slides(P, B), twists=K.twists(K.sub(B, np.zeros(len(B["id"]), bool)), B),
               rates=rates(G, B))
    Path(out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(B_DIR / "checks.json"))
    run(ap.parse_args().out)
