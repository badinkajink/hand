#!/usr/bin/env python3
"""Checks of the fingertip's physical estimator on a UR5 session (owner, 2026-10-25: "it would be nice to not have to rely
on the GP/MLP"; "since we have a good orientation estimator, doesnt that mean we can estimate wrenches, and not just 3d
forces? we have the data from the ur5 hex-e sensor").

1. Gate. The viewer's surface estimator (scripts/ur5_tip_calibration_predict.py's Physical, N = 0) reports a covariance
   of its contact point. Its position sigma, sqrt(var t1 + var t2) in mm, is compared with its location error, and the
   estimate is reported only below a threshold: the share of press holds reported and their error, by force level.
2. Distortion. A similarity and an affine map from the estimated to the true contact points, fitted on the training
   presses' holds at 2 N and over (every fourth grid press held out, as in predict.py) and scored on the held-out holds
   at 2 N and over: whether the error at high force is a global distortion of the model's geometry.
3. Shear. The estimator's tangential force against the HEX-E's (the analysis's f_t), on the press holds and on the turn
   holds at 1 N and over: the median error and the correlation of the components along the pad's tangents.
   The Gaussian process on the board (predict.py's GP) trained on the training presses' 3D force in the pad frame is
   scored the same way on the held-out presses, and trained on every press on the turn holds.
5. Torsion. A point torque tau about the contact normal at the estimated point p is modelled as the curl of the board's
   response to a tangential force, S(p) = J_m M(p)^T (the FE magnet motions per newton through the field model):
   s_tau = [S(p + h t1) t2 - S(p - h t1) t2 - S(p + h t2) t1 + S(p - h t2) t1] / (4h), in uT per N mm. The force f and
   tau are then fitted together to each hold's board change at p, with the uniform field and the rest's common change
   projected out. The HEX-E's torsion about the normal (the analysis's tn) is the reference. On the twists, both are taken
   as half the difference between the outbound and the return hold at the same angle, which removes their offsets.

    .venv/bin/python scripts/ur5_tip_calibration_estimator_checks.py <analysis output folder> [--out estimator_checks.json]
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
import fingertip_wrench_model as W  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402

GATES = (1.0, 1.5, 2.0, 3.0)                      # mm, the position sigma below which a contact point is reported
H_STEP = 0.5                                      # mm, the step of the curl


def tangential(P, Pt, N):
    d = P - Pt
    return np.linalg.norm(d - np.sum(d * N, axis=1)[:, None] * N, axis=1)


class Estimates:
    """The viewer's estimate at every hold of H, with its position sigma and its tangent frame."""
    def __init__(self, H, phys):
        est = phys.estimator()
        n = len(H["id"])
        self.P, self.F = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
        self.sig, self.T1, self.T2, self.Nn = np.full(n, np.nan), np.full((n, 3), np.nan), np.full((n, 3), np.nan), np.full((n, 3), np.nan)
        for i in range(n):
            est.set_noise_scale(1.0 / np.sqrt(max(H["n_rec"][i], 1)))
            e = est.estimate(H["d_board"][i])
            if not e.get("contact"):
                continue
            c = np.asarray(e["cov"], float)
            self.P[i], self.F[i] = e["p"], e["f"]
            self.sig[i] = float(np.sqrt(max(c[3, 3], 0.0) + max(c[4, 4], 0.0)))
            self.T1[i], self.T2[i], self.Nn[i] = e["t1"], e["t2"], e["n"]


def gate(H, E, idx):
    loc = tangential(E.P[idx], H["p"][idx], H["n"][idx])
    lv = np.round(H["level_N"][idx], 3)
    out = {}
    for g in GATES:
        rep = np.isfinite(loc) & (E.sig[idx] <= g)
        row = dict(reported=float(rep.mean()), loc_mm=float(np.median(loc[rep])) if rep.any() else None,
                   loc_p90_mm=float(np.percentile(loc[rep], 90)) if rep.any() else None, by_level={})
        for L in sorted(set(lv.tolist())):
            m = lv == L
            r = rep & m
            row["by_level"][f"{L:g}"] = dict(n=int(m.sum()), reported=float(r.sum() / m.sum()),
                                              loc_mm=float(np.median(loc[r])) if r.any() else None)
        out[f"{g:g}"] = row
    ok = np.isfinite(loc)
    from scipy.stats import spearmanr
    rho = float(spearmanr(loc[ok], E.sig[idx][ok])[0])
    by = {}
    for L in sorted(set(lv.tolist())):
        m = (lv == L) & ok
        by[f"{L:g}"] = dict(loc_mm=float(np.median(loc[m])), sigma_mm=float(np.median(E.sig[idx][m])))
    return dict(spearman=rho, by_level=by, gates=out)


def distortion(H, E, tr, te, snap):
    hi = lambda idx: idx[(H["level_N"][idx] >= 2.0) & np.all(np.isfinite(E.P[idx]), axis=1)]   # noqa: E731
    tr2, te2 = hi(tr), hi(te)
    X, Y = E.P[tr2], H["p"][tr2]
    # similarity (Umeyama): Y ~ s R X + t
    mx, my = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx) / len(X))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / np.mean(np.sum((X - mx) ** 2, axis=1)))
    sim = lambda P: s * (P - mx) @ R.T + my                                                    # noqa: E731
    A = np.linalg.lstsq(np.c_[X, np.ones(len(X))], Y, rcond=None)[0]
    aff = lambda P: np.c_[P, np.ones(len(P))] @ A                                              # noqa: E731
    out = dict(n_train=int(len(tr2)), n_test=int(len(te2)), similarity_scale=s,
               similarity_turn_deg=float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
               similarity_shift_mm=(my - s * mx @ R.T).tolist())
    for name, f in (("none", lambda P: P), ("similarity", sim), ("affine", aff)):
        out[name] = dict(test_mm=float(np.median(tangential(snap(f(E.P[te2])), H["p"][te2], H["n"][te2]))),
                         train_mm=float(np.median(tangential(snap(f(E.P[tr2])), H["p"][tr2], H["n"][tr2]))))
    return out


class Torsion:
    """The board's response to a point torque about the normal, and the joint fit of force and torque."""
    def __init__(self, phys, cal):
        self.phys, self.mod = phys, W.Model(cal)
        self.Jm = self.mod.motion_jacobian()
        Nu = np.c_[np.concatenate(list(self.mod.R), 0), np.asarray(cal.b_rest_uT, float).ravel()]
        self.Q, _ = np.linalg.qr(Nu)

    def S(self, p):
        M = self.phys.rbf(np.atleast_2d(p)).reshape(-1, 3, self.phys.nm * 6)
        return np.einsum("ij,kfj->kif", self.Jm, M)                  # (k, 15, 3): uT per N

    def s_tau(self, p, t1, t2, h=H_STEP):
        pts = np.array([p + h * t1, p - h * t1, p + h * t2, p - h * t2])
        Sp = self.S(pts)
        return (Sp[0] @ t2 - Sp[1] @ t2 - Sp[2] @ t1 + Sp[3] @ t1) / (4 * h)

    def fit(self, d, p, t1, t2):
        A = np.c_[self.S(p)[0], self.s_tau(p, t1, t2)]
        A = A - self.Q @ (self.Q.T @ A)
        b = d - self.Q @ (self.Q.T @ d)
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        return x[:3], float(x[3])


def torsion(H, T, EH, ET, tq):
    def taus(HH, E):
        out = np.full(len(HH["id"]), np.nan)
        for i in range(len(HH["id"])):
            if np.all(np.isfinite(E.P[i])):
                out[i] = tq.fit(HH["d_board"][i], E.P[i], E.T1[i], E.T2[i])[1]
        return out
    tp, tt = taus(H, EH), taus(T, ET)
    res = dict(press=dict(board_abs_median=float(np.nanmedian(np.abs(tp))), hexe_abs_median=float(np.nanmedian(np.abs(H["tn"]))),
                          board_median=float(np.nanmedian(tp)), hexe_median=float(np.nanmedian(H["tn"]))))
    pairs = []
    tw = np.nonzero(T["kind"] == "twist")[0]
    for a in sorted(set(T["id"][tw].tolist())):
        rows = tw[T["id"][tw] == a]
        rows = rows[np.argsort(T["t"][rows])]
        for ang in sorted(set(np.round(T["turn_deg"][rows], 1).tolist())):
            r = rows[np.round(T["turn_deg"][rows], 1) == ang]
            if len(r) == 2 and np.all(np.isfinite(tt[r])):
                pairs.append(dict(id=a, deg=ang, f_n=float(T["f_n"][r].mean()), hexe=float((T["tn"][r[0]] - T["tn"][r[1]]) / 2),
                                  board=float((tt[r[0]] - tt[r[1]]) / 2), hexe_mean=float(T["tn"][r].mean()),
                                  board_mean=float(tt[r].mean())))
    if pairs:
        x, y = np.array([q["hexe"] for q in pairs]), np.array([q["board"] for q in pairs])
        slope = float(np.sum(x * y) / np.sum(x * x))
        res["twist_pairs"] = dict(n=len(pairs), slope=slope, r=float(np.corrcoef(x, y)[0, 1]),
                                  rms_after_slope=float(np.sqrt(np.mean((y / slope - x) ** 2))) if slope else None,
                                  hexe_abs_median=float(np.median(np.abs(x))), pairs=pairs)
    return res


def shear(HH, E):
    """The estimator's tangential force against the HEX-E's, at the holds of 1 N and over."""
    m = (HH["level_N"] >= 1.0) & np.all(np.isfinite(E.F), axis=1)
    n = HH["n"][m]
    fe = E.F[m] - np.sum(E.F[m] * n, axis=1)[:, None] * n
    ft = HH["f_t"][m]
    err = np.linalg.norm(fe - ft, axis=1)
    a, b = np.concatenate([np.sum(fe * E.T1[m], 1), np.sum(fe * E.T2[m], 1)]), np.concatenate([np.sum(ft * E.T1[m], 1),
                                                                                              np.sum(ft * E.T2[m], 1)])
    return dict(n=int(m.sum()), err_N=float(np.median(err)), true_N=float(np.median(np.linalg.norm(ft, axis=1))),
                est_N=float(np.median(np.linalg.norm(fe, axis=1))), r_components=float(np.corrcoef(a, b)[0, 1]))


def run(folder, out):
    H, T, cal = PR.load(folder)
    basis = W.load_basis(PR.BASIS)
    phys = PR.Physical(cal, basis)
    snap = lambda P: PR.snap(phys.estimator(), P)                    # noqa: E731
    train, test = PR.split(H)
    te, tr = (np.nonzero(np.isin(H["id"], s))[0] for s in (test, train))
    EH, ET = Estimates(H, phys), Estimates(T, phys)
    res = dict(folder=str(folder), gate_all=gate(H, EH, np.arange(len(H["id"]))), gate_heldout=gate(H, EH, te),
               distortion=distortion(H, EH, tr, te, snap), shear=dict(press=shear(H, EH), turn=shear(T, ET)),
               torsion=torsion(H, T, EH, ET, Torsion(phys, cal)))
    for name, (rows_tr, HH, rows_te) in (("gp_press_heldout", (tr, H, te)), ("gp_turn", (np.arange(len(H["id"])), T, np.arange(len(T["id"]))))):
        g = PR.GP().fit(H["d_board"][rows_tr], H["f"][rows_tr], PR.floors(0))
        Fp = g.predict(HH["d_board"][rows_te])
        m = HH["level_N"][rows_te] >= 1.0
        n = HH["n"][rows_te][m]
        fe = Fp[m] - np.sum(Fp[m] * n, axis=1)[:, None] * n
        ft = HH["f_t"][rows_te][m]
        res["shear"][name] = dict(n=int(m.sum()), err_N=float(np.median(np.linalg.norm(fe - ft, axis=1))),
                                  true_N=float(np.median(np.linalg.norm(ft, axis=1))), est_N=float(np.median(np.linalg.norm(fe, axis=1))),
                                  fn_err_N=float(np.median(np.abs(-np.sum(Fp[m] * n, axis=1) - HH["f_n"][rows_te][m]))))
    for k, v in res["shear"].items():
        print(f"shear, {k} holds at 1 N and over ({v['n']}): HEX-E |f_t| {v['true_N']:.3f} N, estimate {v['est_N']:.3f} N, "
              f"error {v['err_N']:.3f} N" + (f", components r {v['r_components']:.2f}" if "r_components" in v else
                                             f", normal force error {v['fn_err_N']:.3f} N"), flush=True)
    g = res["gate_all"]
    print(f"gate: spearman {g['spearman']:.2f}; " + "; ".join(
        f"sigma <= {k} mm: {v['reported'] * 100:.0f} % reported, {v['loc_mm']:.2f} mm (p90 {v['loc_p90_mm']:.2f})"
        for k, v in g["gates"].items()), flush=True)
    for k, v in g["gates"].items():
        print(f"  sigma <= {k}: " + ", ".join(f"{L} N {q['reported'] * 100:.0f} % {q['loc_mm'] if q['loc_mm'] is None else round(q['loc_mm'], 2)}"
                                          for L, q in v["by_level"].items()), flush=True)
    d = res["distortion"]
    print(f"distortion at 2 N and over (train {d['n_train']}, test {d['n_test']}): none {d['none']['test_mm']:.2f}, "
          f"similarity {d['similarity']['test_mm']:.2f} (scale {d['similarity_scale']:.3f}, turn {d['similarity_turn_deg']:.1f} deg), "
          f"affine {d['affine']['test_mm']:.2f} mm (train {d['affine']['train_mm']:.2f})", flush=True)
    t = res["torsion"]
    print(f"torsion on presses: board |tau| median {t['press']['board_abs_median']:.2f}, HEX-E |tn| {t['press']['hexe_abs_median']:.2f} N mm",
          flush=True)
    if "twist_pairs" in t:
        q = t["twist_pairs"]
        print(f"twists, half the outbound-return difference ({q['n']} pairs): HEX-E |median| {q['hexe_abs_median']:.2f} N mm; "
              f"board against HEX-E slope {q['slope']:.2f}, r {q['r']:.2f}", flush=True)
        for p in q["pairs"]:
            print(f"   {p['id']:13s} {p['deg']:5.1f} deg f_n {p['f_n']:.2f}: HEX-E {p['hexe']:+6.2f}, board {p['board']:+6.2f}", flush=True)
    Path(out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.folder, a.out or str(Path(a.folder) / "estimator_checks.json"))
