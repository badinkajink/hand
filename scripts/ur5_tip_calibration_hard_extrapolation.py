#!/usr/bin/env python3
"""Extrapolation of the fingertip's contact estimators from light to hard presses, on v2 ID 1's sphere session on the
UR5 (owner, 2026-10-27: "can we parameter tune the physical model to better match the resolution of the GP/MLP? ... is
there a way to leverage the physical model when the detected contacts are very hard? ... i guess then you have to
quantify model uncertainty, which is perhaps easier with the GP than the MLP").

Data: data/20261026-192158-v2id1/holds.npz (five N52 magnets in the checkerboard; R28's passes and R26's presses to
8 N, whose action ids start with "hard-"), press holds only, read with scripts/ur5_tip_calibration_cb1r_checks.py. f_n
is the HEX-E's normal force at the hold (N), p the registered contact point (mm, pad frame), f the 3D force on the pad
(N, pad frame), d_board the board's change (15 components, uT).

Splits. The training holds have f_n of at most 2 N (split "2N") or at most 3.5 N (split "3.5N"). The test holds are
the harder press holds, in bins of f_n: (2, 3.5], (3.5, 5] and (5, 8] N; their actions' lighter holds stay in
training. The in-range reference is the same estimators under five folds grouped by action over the training holds, in
bins (0, 1], (1, 2] and (2, 3.5] N. The strict test scores each fold's model on the harder holds of its held-out actions,
so that no test hold's action has a hold in training.

Estimators of the contact point p and the 3D force f:
  gp              K.gp: a Gaussian process from the board's 15 components (predict.py's GP: squared-exponential
                  kernel on standardized inputs, one length and one noise chosen by the marginal likelihood);
  physical        the viewer's estimator as the analysis ran it (est_p, est_f): the FE basis and the field model of the
                  rest fit, no press in training;
  physical_fit    predict.py's Physical: the magnets' strength k, the chips' offsets dx, dy, dz (priors 5 % and
                  0.2 mm around the rest fit) and a compliance scale c of the FE basis fitted to the training holds;
  physical_affine the physical estimate with an affine map of its point and a 3 x 3 map of its force, fitted by least
                  squares to the training holds over 1 N;
  hybrid          the physical estimate plus a zero-mean GP (GP(center=False)) on its residual (point and force), with
                  the board as input, trained on the training holds that the physical model places within 2 mm (the
                  rule of predict.py);
  hybrid_point    the same with the physical point as the only input, and the force's residual taken relative to the
                  physical force's magnitude, so that the correction does not change with force;
  gp_direction    a GP from the direction of the board's change d/|d| to the point and to f/|d|; the force is then |d|
                  times the second output, so the model is linear in force at each point, as the physical model is;
  mlp             predict.py's MLP ensemble (five members, two hidden layers of 64) from the board to p and f.

Uncertainty. The GP class returns a mean only. Here its latent posterior variance v at a hold is computed from the
kernel, length and noise that GP.fit selects: v = 1 - k*^T (K + s^2 I)^-1 k* on standardized outputs, and the point's
sd is sqrt(v + s^2) times the norm of the point outputs' scales (mm). The MLP's spread is the sd over its five members
of the predicted point (norm over x, y, z; mm). Each is compared with the location error by Spearman correlation (in
range, on the test holds and on both together), and by the area under the ROC curve (AUC) for picking the holds where
the GP or MLP errs more than the fallback. Gates switch from the GP (or MLP) to the physical or the hybrid estimate
where the sd exceeds a threshold chosen on the in-range folds alone: q95, the 95th percentile of the folds' sd; opt,
the threshold that minimizes the folds' mean location error of the gated estimate. blend is the contact viewer's rule
(wrench_viewer.BoardGP): w = 1 - v, the GP's point and force weighted by w and the fallback's by 1 - w. oracle takes the
better of the two at each hold: the floor of any switch.

Metrics: the location error along the surface (mm: the tangential part of p_est - p about the true normal, K.loc_err),
the force magnitude error ||f_est| - |f|| (N) and its ratio to |f|, the signed bias (|f_est| - |f|) / |f|, and the
angle between f_est and f (deg); medians by force bin and region (face, edge, flank). Also the board's change per newton
|d| / |f| against each action's own at (1, 2] N (whether the response stays linear), the physical force's ratio
|f_est| / |f| by bin, and each test hold's distance to the nearest training hold's point (mm).

    NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
        ~/.claude/bin/resguard.sh run --mem 3G --cpu 500 -- \\
        .venv/bin/python scripts/ur5_tip_calibration_hard_extrapolation.py [--no-mlp] [--no-fit] [--out FILE] \\
        > logs/20261027-hard_extrapolation.log 2>&1

Output: docs/experiments/20261022-ur5_tip_calibration/data/20261027-hard_extrapolation_v2id1.json, written after each
split. The GP fits form an n x n x 15 float64 array (n = 921 training holds: 100 MB).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import fingertip_wrench_model as W  # noqa: E402
import ur5_tip_calibration_cb1r_checks as K  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402

SESSION = "20261026-192158-v2id1"
DATA = K.EXP / "data" / SESSION
OUT = K.EXP / "data/20261027-hard_extrapolation_v2id1.json"
SPLITS = (("2N", 2.0), ("3.5N", 3.5))
TEST_BINS = ((2.0, 3.5), (3.5, 5.0), (5.0, 8.0))
CV_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.5))
LIN_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 5.0), (5.0, 8.0))
REGIONS = ("face", "edge", "flank")
FOLDS = 5
NEAR_MM = 2.0        # the hybrids learn from the training holds that the physical model places within this (predict.py)
FALLBACKS = ("physical", "hybrid")


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------------- data
def presses():
    H = K.load(DATA)
    m = ((H["kind"] == "press") & np.isfinite(H["d_board"]).all(1) & np.isfinite(H["p"]).all(1)
         & np.isfinite(H["f"]).all(1) & np.isfinite(H["f_n"]))
    return K.sub(H, m)


def in_bin(H, lo, hi):
    return (H["f_n"] > lo) & (H["f_n"] <= hi)


def folds(H, k=FOLDS):
    """Fold of each hold, grouped by action (the order of np.unique, as in ur5_tip_calibration_v1_tips.py)."""
    a = H["id"].astype(str)
    f = {x: i % k for i, x in enumerate(np.unique(a))}
    return np.array([f[x] for x in a])


# ------------------------------------------------------------------------------------------------------- GP helpers
def sqd(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2.0 * A @ B.T, 0.0)


def latent_var(g, X):
    """Latent posterior variance (standardized outputs, prior 1) of a fitted PR.GP at the inputs X."""
    Z = (X - g.mx) / g.sx
    L = np.linalg.cholesky(np.exp(-0.5 * sqd(g.Z, g.Z) / g.ell ** 2) + g.nz ** 2 * np.eye(len(g.Z)))
    V = solve_triangular(L, np.exp(-0.5 * sqd(Z, g.Z) / g.ell ** 2).T, lower=True)
    return np.clip(1.0 - (V ** 2).sum(0), 0.0, 1.0)


def mlp_members(model, X):
    import torch
    Z = torch.as_tensor((X - model.mx) / model.sx, dtype=torch.float32)
    with torch.no_grad():
        P = np.stack([net(Z).numpy() for net in model.nets])
    return P * model.sy + model.my                                        # (members, n, 6)


# ------------------------------------------------------------------------------------------------------- estimators
# Each fitter takes the training holds and returns (predict, info); predict(H) returns (P, F, extra), with extra holding
# the uncertainty where the estimator has one.
def fit_gp(Htr, ctx):
    g = K.gp(Htr)
    s_p = float(np.linalg.norm(g.sy[:3]))

    def predict(H):
        Y = g.predict(H["d_board"])
        v = latent_var(g, H["d_board"])
        return Y[:, :3], Y[:, 3:], dict(v=v, sd=np.sqrt(v + g.nz ** 2) * s_p)
    return predict, dict(length_per_sqrt_d=float(g.ell / math.sqrt(15)), noise=float(g.nz), point_scale_mm=s_p)


def fit_physical(Htr, ctx):
    return (lambda H: (H["est_p"], H["est_f"], {})), {}


def estimate(est, H):
    n = len(H["id"])
    P, F = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    for i in range(n):
        est.set_noise_scale(1.0 / math.sqrt(max(H["n_rec"][i], 1)))
        e = est.estimate(H["d_board"][i])
        if e.get("contact"):
            P[i], F[i] = e["p"], e["f"]
    return P, F


def fit_physical_fit(Htr, ctx):
    ph = PR.Physical(ctx["cal0"], ctx["basis"]).fit(Htr, np.arange(len(Htr["id"])))
    est = ph.estimator()
    th = ph.theta
    info = dict(k=float(th[0]), dx=float(th[1]), dy=float(th[2]), dz=float(th[3]), c=float(math.exp(th[4])),
                rms_uT=float(ph.rms_uT), k_rest=float(ctx["cal0"].k))
    return (lambda H: (*estimate(est, H), {})), info


def fit_physical_affine(Htr, ctx):
    m = (Htr["f_n"] > 1.0) & np.isfinite(Htr["est_p"]).all(1) & np.isfinite(Htr["est_f"]).all(1)
    A = np.linalg.lstsq(np.c_[Htr["est_p"][m], np.ones(m.sum())], Htr["p"][m], rcond=None)[0]
    M = np.linalg.lstsq(Htr["est_f"][m], Htr["f"][m], rcond=None)[0]

    def predict(H):
        return np.c_[H["est_p"], np.ones(len(H["id"]))] @ A, H["est_f"] @ M, {}
    sv = np.linalg.svd(A[:3])[1]
    return predict, dict(n_fit=int(m.sum()), point_map_singular_values=sv.tolist(),
                         force_map_singular_values=np.linalg.svd(M)[1].tolist())


def near(Htr):
    return (np.isfinite(Htr["est_p"]).all(1) & np.isfinite(Htr["est_f"]).all(1)
            & (np.linalg.norm(Htr["p"] - np.nan_to_num(Htr["est_p"]), axis=1) < NEAR_MM))


def fit_hybrid(Htr, ctx):
    ok = near(Htr)
    R = np.c_[Htr["p"] - Htr["est_p"], Htr["f"] - Htr["est_f"]]
    g = PR.GP(center=False).fit(Htr["d_board"][ok], R[ok], PR.floors(0))

    def predict(H):
        C = g.predict(H["d_board"])
        return H["est_p"] + C[:, :3], H["est_f"] + C[:, 3:], {}
    return predict, dict(n_fit=int(ok.sum()), length_per_sqrt_d=float(g.ell / math.sqrt(15)), noise=float(g.nz))


def fit_hybrid_point(Htr, ctx):
    ok = near(Htr)
    fm = np.linalg.norm(Htr["est_f"], axis=1)
    R = np.c_[Htr["p"] - Htr["est_p"], (Htr["f"] - Htr["est_f"]) / fm[:, None]]
    g = PR.GP(center=False).fit(Htr["est_p"][ok], R[ok], np.full(3, 0.2))

    def predict(H):
        C = g.predict(H["est_p"])
        return H["est_p"] + C[:, :3], H["est_f"] + np.linalg.norm(H["est_f"], axis=1)[:, None] * C[:, 3:], {}
    return predict, dict(n_fit=int(ok.sum()), length_per_sqrt_d=float(g.ell / math.sqrt(3)), noise=float(g.nz))


def fit_gp_direction(Htr, ctx):
    nd = np.linalg.norm(Htr["d_board"], axis=1)
    g = PR.GP().fit(Htr["d_board"] / nd[:, None], np.c_[Htr["p"], Htr["f"] / nd[:, None]], np.full(15, 0.01))

    def predict(H):
        n = np.linalg.norm(H["d_board"], axis=1)
        Y = g.predict(H["d_board"] / n[:, None])
        return Y[:, :3], Y[:, 3:] * n[:, None], {}
    return predict, dict(length_per_sqrt_d=float(g.ell / math.sqrt(15)), noise=float(g.nz))


def fit_mlp(Htr, ctx):
    t0 = time.time()
    mdl = PR.MLP().fit(Htr["d_board"], np.c_[Htr["p"], Htr["f"]], PR.floors(0))

    def predict(H):
        Pm = mlp_members(mdl, H["d_board"])
        Y = Pm.mean(0)
        return Y[:, :3], Y[:, 3:], dict(sd=np.linalg.norm(Pm[:, :, :3].std(0), axis=1))
    return predict, dict(steps=int(mdl.steps), fit_s=round(time.time() - t0, 1))


FITTERS = dict(gp=fit_gp, physical=fit_physical, physical_fit=fit_physical_fit, physical_affine=fit_physical_affine,
               hybrid=fit_hybrid, hybrid_point=fit_hybrid_point, gp_direction=fit_gp_direction, mlp=fit_mlp)


# ------------------------------------------------------------------------------------------------------- scores
def errors(H, P, F):
    fm, ft = np.linalg.norm(F, axis=1), np.linalg.norm(H["f"], axis=1)
    return dict(loc=K.loc_err(H, P), mag=np.abs(fm - ft), rel=np.abs(fm - ft) / ft, bias=(fm - ft) / ft,
                ang=K.angle(F, H["f"]))


def stats(E, m):
    def q(x, p):
        x = x[m & np.isfinite(x)]
        return float(np.percentile(x, p)) if len(x) else None
    return dict(n=int(m.sum()), loc_mm=q(E["loc"], 50), loc_p90_mm=q(E["loc"], 90), force_mag_err_N=q(E["mag"], 50),
                force_rel_err=q(E["rel"], 50), force_rel_bias=q(E["bias"], 50), angle_deg=q(E["ang"], 50),
                detected=float(np.mean(np.isfinite(E["loc"][m]))) if m.any() else None)


def binned(H, P, F, bins):
    E = errors(H, P, F)
    out = {"all": stats(E, np.ones(len(H["id"]), bool))}
    for lo, hi in bins:
        m = in_bin(H, lo, hi)
        if not m.any():
            continue
        row = stats(E, m)
        row["by_region"] = {r: stats(E, m & (H["region"] == r)) for r in REGIONS if (m & (H["region"] == r)).any()}
        out[f"{lo:g}-{hi:g}"] = row
    return out


def auc(score, label):
    """Area under the ROC curve of score for the binary label (Mann-Whitney)."""
    ok = np.isfinite(score)
    s, y = score[ok], label[ok]
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return None
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 3 else None


def gated(est, fb, switch):
    P, F = est[0].copy(), est[1].copy()
    s = switch & np.isfinite(fb[0]).all(1) & np.isfinite(fb[1]).all(1)
    P[s], F[s] = fb[0][s], fb[1][s]
    return P, F


def choose_tau(sd_cv, e_cv, e_fb_cv):
    """The threshold that minimizes the folds' mean location error of the gated estimate (ties: fewest switches)."""
    cands = np.r_[np.unique(np.quantile(sd_cv, np.linspace(0.0, 1.0, 201))), np.inf]
    best = None
    for t in cands:
        e = np.where((sd_cv > t) & np.isfinite(e_fb_cv), e_fb_cv, e_cv)
        c = float(np.nanmean(e))
        if best is None or c < best[0] - 1e-12 or (abs(c - best[0]) <= 1e-12 and t > best[1]):
            best = (c, float(t))
    return best[1]


def uncertainty(name, Hcv, cv, Hte, te, unc_cv, unc_te):
    """How the sd of estimator name (gp or mlp) follows its errors, and the gates and blends on it."""
    sd_cv, sd_te = unc_cv[name]["sd"], unc_te[name]["sd"]
    e_cv, e_te = errors(Hcv, *cv[name]), errors(Hte, *te[name])
    res = dict(spearman_loc=dict(in_range=rho(sd_cv, e_cv["loc"]), test=rho(sd_te, e_te["loc"]),
                                 both=rho(np.r_[sd_cv, sd_te], np.r_[e_cv["loc"], e_te["loc"]])),
               spearman_force_mag=dict(in_range=rho(sd_cv, e_cv["mag"]), test=rho(sd_te, e_te["mag"]),
                                       both=rho(np.r_[sd_cv, sd_te], np.r_[e_cv["mag"], e_te["mag"]])),
               sd_mm_by_bin={}, gates={})
    for lab, H, sd, E, bins in (("in_range", Hcv, sd_cv, e_cv, CV_BINS), ("test", Hte, sd_te, e_te, TEST_BINS)):
        for lo, hi in bins:
            m = in_bin(H, lo, hi)
            if m.any():
                res["sd_mm_by_bin"][f"{lab} {lo:g}-{hi:g}"] = dict(n=int(m.sum()), sd_mm=float(np.median(sd[m])),
                                                                    loc_mm=float(np.median(E["loc"][m])))
    if "v" in unc_te[name]:
        res["weight_w_by_bin"] = {f"{lo:g}-{hi:g}": float(np.median(1.0 - unc_te[name]["v"][in_bin(Hte, lo, hi)]))
                                  for lo, hi in TEST_BINS if in_bin(Hte, lo, hi).any()}
        res["weight_w_in_range_median"] = float(np.median(1.0 - unc_cv[name]["v"]))
    for fb in FALLBACKS:
        f_cv, f_te = errors(Hcv, *cv[fb]), errors(Hte, *te[fb])
        worse_te = e_te["loc"] > f_te["loc"]
        g = dict(auc_test=auc(sd_te, worse_te), auc_in_range=auc(sd_cv, e_cv["loc"] > f_cv["loc"]),
                 share_worse_than_fallback_test=float(np.mean(worse_te)))
        taus = dict(q95=float(np.quantile(sd_cv, 0.95)), opt=choose_tau(sd_cv, e_cv["loc"], f_cv["loc"]))
        for rule, t in taus.items():
            sw_te, sw_cv = sd_te > t, sd_cv > t
            g[rule] = dict(tau_sd_mm=t, test=binned(Hte, *gated(te[name], te[fb], sw_te), TEST_BINS),
                           in_range=binned(Hcv, *gated(cv[name], cv[fb], sw_cv), CV_BINS),
                           switched_test={f"{lo:g}-{hi:g}": float(np.mean(sw_te[in_bin(Hte, lo, hi)]))
                                          for lo, hi in TEST_BINS if in_bin(Hte, lo, hi).any()},
                           switched_in_range=float(np.mean(sw_cv)))
        if "v" in unc_te[name]:
            def blend(est, fbe, v):
                w = (1.0 - v)[:, None]
                ok = np.isfinite(fbe[0]).all(1)[:, None]
                return (np.where(ok, w * est[0] + (1 - w) * np.nan_to_num(fbe[0]), est[0]),
                        np.where(ok, w * est[1] + (1 - w) * np.nan_to_num(fbe[1]), est[1]))
            g["blend"] = dict(test=binned(Hte, *blend(te[name], te[fb], unc_te[name]["v"]), TEST_BINS),
                              in_range=binned(Hcv, *blend(cv[name], cv[fb], unc_cv[name]["v"]), CV_BINS))
        best = np.where(e_te["loc"] <= f_te["loc"], True, False)[:, None]
        g["oracle_loc"] = binned(Hte, np.where(best, te[name][0], te[fb][0]), np.where(best, te[name][1], te[fb][1]),
                                 TEST_BINS)
        res["gates"][fb] = g
    return res


# ------------------------------------------------------------------------------------------------------- side checks
def linearity(H):
    """The board's change per newton |d| / |f| over each action's own median at (1, 2] N, and the physical force's
    ratio |f_est| / |f|, by force bin; the reference point's ramp to 8 N hold by hold."""
    s = np.linalg.norm(H["d_board"], axis=1) / np.linalg.norm(H["f"], axis=1)
    ids = H["id"].astype(str)
    ref = {}
    for a in np.unique(ids):
        m = (ids == a) & in_bin(H, 1.0, 2.0)
        if m.any():
            ref[a] = float(np.median(s[m]))
    ratio = np.array([s[i] / ref[a] if a in ref else np.nan for i, a in enumerate(ids)])
    phys = np.linalg.norm(H["est_f"], axis=1) / np.linalg.norm(H["f"], axis=1)
    out = {}
    for lo, hi in LIN_BINS:
        m = in_bin(H, lo, hi)
        r = ratio[m & np.isfinite(ratio)]
        out[f"{lo:g}-{hi:g}"] = dict(n=int(m.sum()), n_with_ref=int(len(r)),
                                     sensitivity_over_own_1_2N=float(np.median(r)) if len(r) else None,
                                     sensitivity_uT_per_N=float(np.median(s[m])),
                                     physical_force_over_true=float(np.median(phys[m])))
    ramp = [dict(level_N=float(H["level_N"][i]), f_n=float(H["f_n"][i]), uT_per_N=float(s[i]),
                 physical_force_over_true=float(phys[i]), physical_loc_mm=float(K.loc_err(K.sub(H, [i]), H["est_p"][[i]])[0]))
            for i in np.nonzero(ids == "hard-ramp-020")[0]]
    return dict(by_bin=out, ramp_hard_ramp_020=ramp)


def nearest(Htr, Hte):
    d = cKDTree(Htr["p"]).query(Hte["p"])[0]
    return {f"{lo:g}-{hi:g}": dict(n=int(in_bin(Hte, lo, hi).sum()), median_mm=float(np.median(d[in_bin(Hte, lo, hi)])),
                                   p90_mm=float(np.percentile(d[in_bin(Hte, lo, hi)], 90)))
            for lo, hi in TEST_BINS if in_bin(Hte, lo, hi).any()}


def composition(H, bins):
    out = {}
    for lo, hi in bins:
        m = in_bin(H, lo, hi)
        if not m.any():
            continue
        lv, c = np.unique(H["level_N"][m], return_counts=True)
        rg, rc = np.unique(H["region"][m], return_counts=True)
        out[f"{lo:g}-{hi:g}"] = dict(n=int(m.sum()), n_actions=int(len(np.unique(H["id"][m]))),
                                     hard_actions=int(np.char.startswith(H["id"][m].astype(str), "hard").sum()),
                                     by_level={f"{a:g}": int(b) for a, b in zip(lv, c)},
                                     by_region={str(a): int(b) for a, b in zip(rg, rc)})
    return out


# ------------------------------------------------------------------------------------------------------- main
def check_physical(H, ctx):
    """predict.py's Physical with the rest fit and c = 1 reproduces the analysis's est_p (the hybrids' base)."""
    idx = np.linspace(0, len(H["id"]) - 1, 25).astype(int)
    P, _ = estimate(PR.Physical(ctx["cal0"], ctx["basis"]).estimator(), K.sub(H, idx))
    return float(np.nanmax(np.linalg.norm(P - H["est_p"][idx], axis=1)))


def run_split(H, thr, names, ctx):
    """Every estimator trained on the holds of f_n <= thr. The five fold models (trained on the other folds' training
    holds, folds grouped by action over all press holds) give the in-range reference on their fold's training holds
    and the strict test on their fold's harder holds, whose actions no training hold shares; the model trained on all
    training holds gives the test."""
    fall = folds(H)
    tr = H["f_n"] <= thr
    Htr, Hte = K.sub(H, tr), K.sub(H, ~tr)
    fk, fk_te = fall[tr], fall[~tr]
    log(f"split f_n <= {thr:g} N: {int(tr.sum())} training holds ({len(np.unique(Htr['id']))} actions), "
        f"{int((~tr).sum())} test holds ({len(np.unique(Hte['id']))} actions)")
    n, nt = len(Htr["id"]), len(Hte["id"])
    cv = {k: (np.full((n, 3), np.nan), np.full((n, 3), np.nan)) for k in names}
    strict = {k: (np.full((nt, 3), np.nan), np.full((nt, 3), np.nan)) for k in names}
    unc_cv = {k: dict(sd=np.full(n, np.nan), v=np.full(n, np.nan)) for k in ("gp", "mlp") if k in names}
    d_strict = np.full(nt, np.nan)
    info = {k: dict(folds=[]) for k in names}
    for k in range(FOLDS):
        a, b, c = fk != k, fk == k, fk_te == k
        Ha, Hb, Hc = K.sub(Htr, a), K.sub(Htr, b), K.sub(Hte, c)
        if c.any():
            d_strict[c] = cKDTree(Ha["p"]).query(Hc["p"])[0]
        for name in names:
            t0 = time.time()
            pred, inf = FITTERS[name](Ha, ctx)
            P, F, ex = pred(Hb)
            cv[name][0][b], cv[name][1][b] = P, F
            for q, v in ex.items():
                unc_cv[name][q][b] = v
            if c.any():
                P, F, _ = pred(Hc)
                strict[name][0][c], strict[name][1][c] = P, F
            info[name]["folds"].append(dict(inf, s=round(time.time() - t0, 1)))
        log(f"  fold {k}: {int(b.sum())} in-range and {int(c.sum())} strict test holds done")
    te, unc_te = {}, {}
    for name in names:
        t0 = time.time()
        pred, inf = FITTERS[name](Htr, ctx)
        P, F, ex = pred(Hte)
        te[name] = (P, F)
        if ex:
            unc_te[name] = ex
        info[name]["all_training"] = dict(inf, s=round(time.time() - t0, 1))
        log(f"  {name}: fitted on all training holds in {time.time() - t0:.1f} s")
    for k in list(unc_cv):
        if not np.isfinite(unc_cv[k]["v"]).any():
            del unc_cv[k]["v"]
    res = dict(threshold_N=thr, n_train=int(tr.sum()), n_train_actions=int(len(np.unique(Htr["id"]))),
               n_test=int((~tr).sum()), n_test_actions=int(len(np.unique(Hte["id"]))),
               test_composition=composition(Hte, TEST_BINS), in_range_composition=composition(Htr, CV_BINS),
               test_nearest_training_point=nearest(Htr, Hte),
               test_strict_nearest_training_point={
                   f"{lo:g}-{hi:g}": dict(median_mm=float(np.median(d_strict[in_bin(Hte, lo, hi)])),
                                          p90_mm=float(np.percentile(d_strict[in_bin(Hte, lo, hi)], 90)))
                   for lo, hi in TEST_BINS if in_bin(Hte, lo, hi).any()},
               fits=info, estimators={}, uncertainty={})
    for name in names:
        res["estimators"][name] = dict(test=binned(Hte, *te[name], TEST_BINS),
                                       test_strict=binned(Hte, *strict[name], TEST_BINS),
                                       in_range=binned(Htr, *cv[name], CV_BINS))
    for name in ("gp", "mlp"):
        if name in names:
            res["uncertainty"][name] = uncertainty(name, Htr, cv, Hte, te, unc_cv, unc_te)
    # the table in the log
    bins = [f"{lo:g}-{hi:g}" for lo, hi in TEST_BINS if f"{lo:g}-{hi:g}" in res["estimators"]["gp"]["test"]]
    cvb = [f"{lo:g}-{hi:g}" for lo, hi in CV_BINS if f"{lo:g}-{hi:g}" in res["estimators"]["gp"]["in_range"]]
    log("  location median (mm) | force magnitude error (N) | angle (deg), in range " + ", ".join(cvb) + "; test "
        + ", ".join(bins) + "; strict test " + ", ".join(bins))
    for name in names:
        r = res["estimators"][name]
        cells = [r["in_range"][b] for b in cvb] + [r["test"][b] for b in bins] + [r["test_strict"][b] for b in bins]
        log(f"  {name:16s} " + "  ".join(f"{c['loc_mm']:5.2f}|{c['force_mag_err_N']:5.2f}|{c['angle_deg']:4.1f}" for c in cells))
    for name, u in res["uncertainty"].items():
        log(f"  {name} sd: spearman with location error in range {u['spearman_loc']['in_range']:.2f}, test "
            f"{u['spearman_loc']['test']:.2f}, both {u['spearman_loc']['both']:.2f}; "
            + "; ".join(f"{k} sd {v['sd_mm']:.2f} mm err {v['loc_mm']:.2f}" for k, v in u["sd_mm_by_bin"].items()))
        for fb, g in u["gates"].items():
            log(f"    -> {fb}: AUC test {g['auc_test']}, " + "; ".join(
                f"{rule} (tau {g[rule]['tau_sd_mm']:.3g} mm) test " + ", ".join(
                    f"{b} {g[rule]['test'][b]['loc_mm']:.2f} mm {g[rule]['test'][b]['force_mag_err_N']:.2f} N "
                    f"sw {g[rule]['switched_test'][b] * 100:.0f}%" for b in bins) for rule in ("q95", "opt"))
                + ("; blend " + ", ".join(f"{b} {g['blend']['test'][b]['loc_mm']:.2f} mm {g['blend']['test'][b]['force_mag_err_N']:.2f} N"
                                          for b in bins) if "blend" in g else ""))
    return res


def write(res, out):
    tmp = Path(str(out) + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(res, indent=1, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-mlp", action="store_true", help="leave out the MLP ensemble (the slowest fit)")
    ap.add_argument("--no-fit", action="store_true", help="leave out physical_fit (predict.py's Physical)")
    a = ap.parse_args()
    t0 = time.time()
    H = presses()
    ctx = dict(cal0=W.Calib.load(DATA / "calib_rest.json"), basis=W.load_basis(PR.BASIS))
    names = [k for k in FITTERS if not (a.no_mlp and k == "mlp") and not (a.no_fit and k == "physical_fit")]
    res = dict(session=SESSION, n_press_holds=int(len(H["id"])), n_press_actions=int(len(np.unique(H["id"]))),
               estimators=names, folds=FOLDS, near_mm=NEAR_MM,
               physical_reproduces_est_p_max_mm=check_physical(H, ctx), linearity=linearity(H), splits={})
    log(f"{SESSION}: {res['n_press_holds']} press holds, {res['n_press_actions']} actions; Physical(c = 1) against est_p "
        f"{res['physical_reproduces_est_p_max_mm']:.3g} mm at most")
    for k, v in res["linearity"]["by_bin"].items():
        log(f"  {k} N: {v['n']} holds, |d|/|f| {v['sensitivity_uT_per_N']:.1f} uT/N, over the action's own at 1-2 N "
            f"{v['sensitivity_over_own_1_2N']}, physical |f| over true {v['physical_force_over_true']:.3f}")
    for name, thr in SPLITS:
        res["splits"][name] = run_split(H, thr, names, ctx)
        res["wall_s"] = round(time.time() - t0, 1)
        write(res, a.out)
        log(f"-> {a.out} ({res['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
