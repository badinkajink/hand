#!/usr/bin/env python3
"""Transfer of the fingertip's contact predictors across magnet strength and pole pattern, in simulation (owner,
2026-10-23: "how transferable do you think this result will be to the higher strength n52 magnets + the checkerboard
orientation ... transform our physical model to work with whatever detected orientation we have, and maybe even magnet
strength ... the GP/MLP can just be used to fine-tune").

The mechanics are the UR5 run sphere2's: each press hold's true contact point p and force f (pad frame, ground truth of
scripts/ur5_tip_calibration_analysis.py). The board's change is synthesized from the FE magnet motions and the dipole
field model (scripts/fingertip_wrench_model.py) for a magnet set: d = c_true S_cal(p + delta) f + noise. The true pad
differs from the FE model in ways that do not depend on the magnets: its compliance is c_true times the FE's and its
response is that of a point delta away (the 1.4 mm offset of the sphere2 analysis). Three magnet sets: A, today's
(k and poles of sphere2's rest fit); B, N52 at k = 1 with the same poles; C, N52 in a checkerboard (the magnet nearest the
pad's centre one way, the four around it the other); D, N52 in the checkerboard of the print procedure's advice
of 2026-10-22 (the centre and one diagonal one way, the other diagonal the other way), the pattern printed on 10-23.

Predictors, all trained on set A's training presses (every fourth grid press held out, as in
scripts/ur5_tip_calibration_predict.py) and scored on the held-out presses of each set:
  physical      the viewer's surface estimator built for each set's own k and poles (what a rest fit gives), untrained
  gp raw        a Gaussian process from the 15 board components to the contact point and the normal force
  physical+gp   the physical estimate corrected by a Gaussian process whose inputs are the physical estimate itself
                (contact point, normal force) and whose outputs are the corrections: inputs and outputs in the pad's
                units, so the correction learned on set A applies to any set whose physical model is rebuilt
and gp raw retrained on each set as the reference of a full recalibration.

    .venv/bin/python scripts/fingertip_magnet_transfer_sim.py <analysis output folder> [--out transfer.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import fingertip_wrench_model as W  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402


def checkerboard(cal):
    """Poles with the magnet nearest the pad's centre one way and the others the other way."""
    row = W.fe_row(cal.case)
    yz = np.array([[m[2], m[3]] for m in row["magnets"]], float)
    c = int(np.argmin(np.linalg.norm(yz - yz.mean(0), axis=1)))
    return tuple(1 if j == c else -1 for j in range(len(yz)))


def checkerboard_diagonal(cal):
    """The checkerboard of the print procedure's advice (2026-10-22): the centre magnet and one diagonal of the four
    around it one way, the other diagonal the other way (a net moment of one magnet)."""
    row = W.fe_row(cal.case)
    yz = np.array([[m[2], m[3]] for m in row["magnets"]], float)
    c = int(np.argmin(np.linalg.norm(yz - yz.mean(0), axis=1)))
    d = yz - yz[c]
    return tuple(1 if (j == c or d[j, 0] * d[j, 1] > 0) else -1 for j in range(len(yz)))


def synth(cal, P, F, rbf, rng, c_true, delta, noise_uT):
    Jm = W.Model(cal).motion_jacobian()
    M = rbf(P + delta[None]).reshape(len(P), 3, -1)
    d = c_true * np.einsum("ij,kfj,kf->ki", Jm, M, F)
    return d + rng.normal(0.0, noise_uT, d.shape)


def tangential(P, Pt, N):
    d = P - Pt
    return np.linalg.norm(d - (d * N).sum(1)[:, None] * N, axis=1)


def run(folder, out, c_true=1.15, delta=(-0.33, -1.21, 0.72), noise_uT=2.0, seed=0, want="ABCD"):
    H, _, cal0 = PR.load(folder)
    basis = W.load_basis(PR.BASIS)
    rbf = PR.Physical(cal0, basis).rbf
    train, test = PR.split(H)
    tr = np.nonzero(np.isin(H["id"], train))[0]
    te = np.nonzero(np.isin(H["id"], test))[0]
    P, F, N, fn = H["p"], H["f"], H["n"], H["f_n"]
    rng = np.random.default_rng(seed)
    sets = {"A": cal0, "B": replace(cal0, k=1.0), "C": replace(cal0, k=1.0, poles=checkerboard(cal0)),
            "D": replace(cal0, k=1.0, poles=checkerboard_diagonal(cal0)),
            "E": replace(cal0, poles=checkerboard_diagonal(cal0))}             # today's magnets in the printed checkerboard
    sets = {k: v for k, v in sets.items() if k in want}
    D, EST, PHYS = {}, {}, {}
    for k, cal in sets.items():
        cal = replace(cal, b_rest_uT=W.Model(cal).rest_uT().tolist())          # the rest a rest fit of this set gives
        sets[k] = cal
        D[k] = synth(cal, P, F, rbf, rng, c_true, np.asarray(delta), noise_uT)
        est = W.SurfaceEstimator(W.Model(cal), basis, floor_uT=noise_uT)
        EST[k] = est
        ph = np.full((len(P), 3), np.nan)
        pf = np.full(len(P), np.nan)
        for i in range(len(P)):
            e = est.estimate(D[k][i])
            if e.get("contact"):
                ph[i], pf[i] = e["p"], e["f_n"]
        PHYS[k] = (ph, pf)
        print(f"set {k}: k {cal.k:.3f}, poles {cal.poles}, |d| median {np.median(np.linalg.norm(D[k], axis=1)):.1f} uT", flush=True)

    def score(Pp, Fp, idx):
        loc = tangential(Pp, P[idx], N[idx])
        return dict(loc_mm=float(np.nanmedian(loc)), loc_p90_mm=float(np.nanpercentile(loc, 90)),
                    force_N=float(np.nanmedian(np.abs(Fp - fn[idx]))))

    snap = lambda Y: PR.snap(EST["A"], Y)                                         # noqa: E731
    res = dict(c_true=c_true, delta_mm=list(delta), noise_uT=noise_uT, sets={k: dict(k=c.k, poles=list(c.poles)) for k, c in sets.items()},
               n_train_holds=int(len(tr)), n_test_holds=int(len(te)), scores={})
    fl = PR.floors(0)
    gp_raw = PR.GP().fit(D["A"][tr], np.c_[P[tr], fn[tr]], fl)
    ok = np.all(np.isfinite(PHYS["A"][0][tr]), axis=1)
    Xa = np.c_[PHYS["A"][0][tr][ok], PHYS["A"][1][tr][ok]]
    Ra = np.c_[P[tr][ok] - PHYS["A"][0][tr][ok], fn[tr][ok] - PHYS["A"][1][tr][ok]]
    gp_res = PR.GP(center=True).fit(Xa, Ra, np.array([0.2, 0.2, 0.2, 0.05]))
    # magnet-motion inputs: the five magnets' translations (um) solved from the board change through each set's own field
    # model (translations only, 15 x 15), a description of the pad's deformation that does not depend on the magnets
    U = {}
    for k, cal in sets.items():
        Jm = W.Model(cal).motion_jacobian()
        Jt = Jm[:, [6 * j + q for j in range(5) for q in range(3)]]
        U[k] = D[k] @ np.linalg.pinv(Jt).T
    gp_u = PR.GP().fit(U["A"][tr], np.c_[P[tr], fn[tr]], np.full(15, 0.05))
    mlp_u = PR.MLP().fit(U["A"][tr], np.c_[P[tr], fn[tr]], np.full(15, 0.05))
    for k in sets:
        ph, pf = PHYS[k]
        s = {"physical": score(ph[te], pf[te], te)}
        Y = gp_raw.predict(D[k][te])
        s["gp raw (trained on A)"] = score(snap(Y[:, :3]), Y[:, 3], te)
        X = np.c_[ph[te], pf[te]]
        okt = np.all(np.isfinite(X), axis=1)
        Yc = np.full((len(te), 4), np.nan)
        Yc[okt] = gp_res.predict(X[okt])
        s["physical+gp (trained on A)"] = score(snap(np.where(np.isfinite(ph[te]), ph[te], 0.0) + Yc[:, :3]), pf[te] + Yc[:, 3], te)
        Yu = gp_u.predict(U[k][te])
        s["gp on magnet motions (trained on A)"] = score(snap(Yu[:, :3]), Yu[:, 3], te)
        Ym = mlp_u.predict(U[k][te])
        s["mlp on magnet motions (trained on A)"] = score(snap(Ym[:, :3]), Ym[:, 3], te)
        g_k = PR.GP().fit(D[k][tr], np.c_[P[tr], fn[tr]], fl)
        Yk = g_k.predict(D[k][te])
        s[f"gp raw (retrained on {k})"] = score(snap(Yk[:, :3]), Yk[:, 3], te)
        res["scores"][k] = s
        print(k, json.dumps({m: (round(v["loc_mm"], 2), round(v["force_N"], 3)) for m, v in s.items()}), flush=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print("->", out)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sets", default="ABCD", help="the sets to simulate (A always, as the training set); E is today's "
                    "magnets in the printed checkerboard")
    a = ap.parse_args()
    run(a.folder, a.out or str(Path(a.folder) / "predict/magnet_transfer_sim.json"), want="A" + a.sets.replace("A", ""))
