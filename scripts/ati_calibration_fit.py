#!/usr/bin/env python3
"""Calibration of an ATI transducer's raw gage values against reference wrenches (the handoff of 2026-10-23,
docs/experiments/20261023-ati_ft_calibration/): w = C (g - g0), C 6 x 6 from gage counts to N and N m at the ATI's origin.

References: a known mass hung at a known point of the tool, at many tool orientations (w = [m g_S, r x m g_S], g_S gravity
in the sensor frame from the robot's joints), which fixes the scale independently of any other sensor; and, as a
cross-check, the HEX-E's wrench carried to the ATI's origin through the stack's fixed transform.

Models (the handoff's section 5): "full", C by ridge regression on standardized gages; "rows6", C = diag(d) R with R the
row shapes of a factory matrix of the same model (R_31565 for a Mini45) and six output scales d; "rows12", C = diag(d) R
diag(k) with six gage gains k as well (their geometric mean fixed at 1). Scored per axis on held-out groups (a pose or a
press is a group).

    .venv/bin/python scripts/ati_calibration_fit.py pairs.npz [--out fit.json]    pairs: G (n, 6), W (n, 6), group (n,)
    .venv/bin/python scripts/ati_calibration_fit.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

R_31565 = np.array([[-0.026289, 0.000278, 0.056518, -0.702557, -0.032851, 0.708130],
                    [-0.095998, 0.795815, 0.015191, -0.420373, 0.026813, -0.424028],
                    [0.575115, 0.028712, 0.571098, 0.027078, 0.583951, 0.023091],
                    [-0.033322, 0.232680, -0.670809, -0.155471, 0.679074, -0.097216],
                    [0.774309, 0.038323, -0.420813, 0.191089, -0.366102, -0.226602],
                    [0.079040, -0.547565, 0.009397, -0.583523, 0.043265, -0.592846]])
R_21031 = np.array([[0.011364, 0.002690, 0.000075, 0.707878, 0.005062, -0.706220],
                    [-0.013851, -0.814428, 0.027480, 0.411090, -0.016918, 0.408017],
                    [0.572382, -0.023654, 0.580723, 0.023579, 0.577595, -0.020197],
                    [-0.008320, -0.175155, 0.687318, 0.116092, -0.686015, 0.112924],
                    [-0.798898, 0.033788, 0.398006, -0.137885, 0.404722, 0.139284],
                    [0.001289, -0.576372, -0.004072, -0.578715, 0.002758, -0.576939]])
AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def gravity_wrench(R_SB, mass_kg, r_S_m, g=9.81):
    """Wrench at the sensor origin (N, N m) of a point mass at r_S (sensor frame, m), the sensor turned by R_SB."""
    f = R_SB @ np.array([0.0, 0.0, -g * mass_kg])
    return np.r_[f, np.cross(r_S_m, f)]


def fit_full(G, W, lam=1e-6):
    s = G.std(0) + 1e-12
    Z = G / s
    Ct = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(6), Z.T @ W)
    return (Ct / s[:, None]).T


def fit_rows6(G, W, R):
    Q = G @ R.T
    d = (Q * W).sum(0) / ((Q * Q).sum(0) + 1e-30)
    return np.diag(d) @ R


def fit_rows12(G, W, R, iters=50):
    k = np.ones(6)
    C = fit_rows6(G, W, R)
    for _ in range(iters):
        d = np.diag(C @ np.linalg.pinv(R @ np.diag(k)))
        A = np.einsum("i,ij->ij", d, R)                                      # w_i = sum_j A_ij k_j g_j
        X = np.concatenate([A[i][None, :] * G for i in range(6)])            # (6n, 6) in k
        y = np.concatenate([W[:, i] for i in range(6)])
        k = np.linalg.lstsq(X, y, rcond=None)[0]
        k = k / np.exp(np.mean(np.log(np.abs(k)))) * np.sign(k)             # geometric mean 1
        Q = (G * k[None]) @ R.T
        d = (Q * W).sum(0) / ((Q * Q).sum(0) + 1e-30)
        C = np.diag(d) @ R @ np.diag(k)
    return C


def score(C, G, W):
    E = np.abs(G @ C.T - W)
    return dict(median=np.round(np.median(E, 0), 5).tolist(), p90=np.round(np.percentile(E, 90, 0), 5).tolist())


def crossval(G, W, group, R=R_31565, folds=4):
    ug = np.unique(group)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    out = {m: [] for m in ("full", "rows6", "rows12")}
    for k in range(folds):
        te = np.isin(group, ug[k::folds])
        for m, fn in (("full", lambda g, w: fit_full(g, w)), ("rows6", lambda g, w: fit_rows6(g, w, R)),
                      ("rows12", lambda g, w: fit_rows12(g, w, R))):
            C = fn(G[~te], W[~te])
            out[m].append(np.abs(G[te] @ C.T - W[te]))
    return {m: dict(median=np.round(np.median(np.concatenate(v), 0), 5).tolist(),
                    p90=np.round(np.percentile(np.concatenate(v), 90, 0), 5).tolist()) for m, v in out.items()}


def design(W):
    """Singular values of the reference wrenches, torques scaled by 0.1 m: how well each direction of the wrench is
    excited."""
    return np.round(np.linalg.svd(W / np.array([1, 1, 1, 0.1, 0.1, 0.1]), compute_uv=False), 3).tolist()


def selftest(seed=0):
    """A sensor of the FT31565 row shapes with its gages up to 10 % off and noise at the Mini45 SI-145-5 resolution,
    calibrated from 5 hook points x 8 orientations of a 0.5 kg mass; scored on a 0.3 kg mass at other points and
    orientations."""
    rng = np.random.default_rng(seed)
    C_true = (R_31565 * 3000.0) @ np.diag(1.0 + rng.uniform(-0.1, 0.1, 6))
    Ci = np.linalg.inv(C_true)
    hooks = np.array([(60, 0, 40), (-60, 0, 40), (0, 60, 40), (0, -60, 40), (0, 0, 100)], float) * 1e-3
    W, grp = [], []
    for h, r in enumerate(hooks):
        for o in range(8):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            a, b, c, d = q
            R = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                          [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                          [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])
            W.append(gravity_wrench(R, 0.5, r))
            grp.append(h * 8 + o)
    W = np.array(W)
    res_w = np.array([1 / 16, 1 / 16, 1 / 16, 1 / 752, 1 / 752, 1 / 752])   # Mini45 SI-145-5 resolution, N and N m
    noisy = lambda Wx: (Wx + rng.normal(0, 1.0, Wx.shape) * res_w[None]) @ Ci.T          # noqa: E731
    G = noisy(W)
    res = dict(design=design(W), fits={})
    Wt = np.array([gravity_wrench(np.linalg.qr(rng.normal(size=(3, 3)))[0], 0.3, rng.normal(0, 0.05, 3)) for _ in range(200)])
    Gt = noisy(Wt)
    for m, C in (("full", fit_full(G, W)), ("rows6", fit_rows6(G, W, R_31565)), ("rows12", fit_rows12(G, W, R_31565))):
        res["fits"][m] = score(C, Gt, Wt)
        print(f"{m:7s} held-out median |error|: F {np.round(res['fits'][m]['median'][:3], 4)} N, "
              f"T {np.round(np.array(res['fits'][m]['median'][3:]) * 1e3, 2)} N mm", flush=True)
    print("design singular values:", res["design"])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pairs", nargs="?")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rows", default="R_31565", choices=("R_31565", "R_21031"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        z = np.load(a.pairs)
        R = R_31565 if a.rows == "R_31565" else R_21031
        out = dict(design=design(z["W"]), crossval=crossval(z["G"], z["W"], z["group"], R),
                   C_full=fit_full(z["G"], z["W"]).tolist())
        print(json.dumps(out["crossval"], indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
