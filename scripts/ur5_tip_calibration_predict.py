#!/usr/bin/env python3
"""Predictors of the contact point (mm, pad frame) and the normal force (N) from the board's change at each hold of a
calibration session on the UR5, trained on the session's own presses in time order (owner, 2026-10-22: "are we
training/building/calibrating a predictor on the side so we can see how each additional information improves the
prediction?").

The holds come from scripts/ur5_tip_calibration_analysis.py (holds.npz and calib_rest.json in its output folder): the
board's change at each hold of a press, zeroed by its two rests interpolated to the hold's time, with the ground truth
from the robot, the HEX-E and the registration. Every fourth grid press, in time order and leaving out the reference
presses, is held out and never trained on. The other presses train the predictors on the first N of them in time order:

  physical   the viewer's SurfaceEstimator with the magnets' strength k, the chips' offsets dx, dy, dz and a compliance
             scale c of the FE surface basis fitted to the training holds, whose contact point and force are known: the
             model of the change is c S(p) f, the uniform field and the magnets' common strength change projected out of
             each hold's residual. N = 0 is the viewer's estimator as it stands (magnets fitted to the first rest).
  gp         a Gaussian process (squared-exponential kernel on standardized inputs, length and noise by the marginal
             likelihood over a grid) from the inputs to the contact point and the normal force; ridge regression is the
             linear baseline. A point predicted off the surface is moved to the nearest surface point.
  physical+gp  the physical predictor plus a Gaussian process on its residual.

The inputs are added in turn: the board's 15 components; then the chips' temperature change since the press's rest; then
the time since that rest; then, apart, the tool's orientation (gravity in the pad frame and the Earth's field in the
flange frame), which in this protocol identifies the target and is reported as a leak. The errors are the
distance from the true contact point along the surface (the tangential part, mm) and the error of the normal force (N,
and relative above 0.4 N), as medians and 90th percentiles over the held-out holds, by force level.

    .venv/bin/python scripts/ur5_tip_calibration_predict.py <analysis output folder> [--out DIR] [--sizes 3,5,10,20,40]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_wrench_model as W  # noqa: E402
import ur5_tip_calibration as C  # noqa: E402

BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz"
# the lab's fit of the Earth's field to first2's 12 gravity stations (2026-10-22): w in the base frame, 44.3 uT, 59 deg dip
EARTH = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab/20261022-earth_field_fit.json"
# The inputs in the order they are added. Only the changes local to each press go in the ladder: its zero comes from its
# own rests, and the absolute temperature and time follow the grid's order of visits, hence the place pressed. The tool's
# orientation is tried last and apart: each target has a planned orientation in this protocol, so the orientation
# identifies the target (on a simulated session it took the GP's location error from 0.45 to 0.04 mm), while the Earth's
# field cancels from a press zeroed by its own rests at the same orientation.
STEPS = ("board", "+temperature change", "+time since rest", "+orientation (leaks the target)")


# ------------------------------------------------------------------------------------------------------- data
def load(folder):
    """The press holds, the turn holds (slides, rolls, twists) and the magnets fitted to the first rest."""
    z = np.load(Path(folder) / "holds.npz", allow_pickle=False)
    A = {k: z[k] for k in z.files}
    ok = np.isfinite(A["f_n"]) & np.all(np.isfinite(A["d_board"]), axis=1)
    H = {k: v[ok & (A["kind"] == "press")] for k, v in A.items()}
    T = {k: v[ok & np.isin(A["kind"], ["slide", "roll", "twist"])] for k, v in A.items()}
    cal = W.Calib.load(Path(folder) / "calib_rest.json")
    return H, T, cal


def features(H, step):
    """The inputs of a step (rows of holds)."""
    X = [H["d_board"]]
    if step >= 1:
        X.append(H["dT_chip"])
    if step >= 2:
        X.append(H["dt_zero"][:, None])
    if step >= 3:                                       # the orientation: gravity in the pad frame and the Earth's field in
        w = np.array(json.loads(EARTH.read_text())["w_base_uT"]) if EARTH.exists() else np.array([0.0, 1.0, 0.0])
        R = H["R_BE"].reshape(-1, 3, 3)                 # the flange's (direction only), which together fix the tool's turn
        X += [H["g_P"], np.einsum("kji,j->ki", R, w / np.linalg.norm(w))]
    return np.concatenate(X, axis=1)


def split(H):
    """Presses in time order; every fourth grid press (not a reference) held out."""
    ids, first = np.unique(H["id"], return_index=True)
    order = ids[np.argsort(H["t"][first])]
    grid = [i for i in order if H["stage"][H["id"] == i][0] == "grid" and H["role"][H["id"] == i][0] != "ref"]
    test = set(grid[3::4])
    train = [i for i in order if i not in test]
    return train, sorted(test)


# ------------------------------------------------------------------------------------------------------- regressors
def floors(step):
    """The smallest standard deviation an input is standardized by, at its resolution: 1 uT for the board, 0.05 C for a
    temperature change, 1 s for the time since the rest, 0.05 for a unit vector. Without it an input that barely varies
    over the training rows is divided by its noise, and a test row where it differs is multiplied by thousands (the ridge
    with the orientation inputs predicted 1781 N from three presses on one region)."""
    f = [np.full(15, 1.0)]
    if step >= 1:
        f.append(np.full(5, 0.05))
    if step >= 2:
        f.append(np.full(1, 1.0))
    if step >= 3:
        f.append(np.full(6, 0.05))
    return np.concatenate(f)


def scale(X, floor=None):
    sd = X.std(0)
    return np.maximum(sd, floor) if floor is not None else sd + 1e-9


class Ridge:
    def fit(self, X, Y, floor=None):
        self.mx, self.sx = X.mean(0), scale(X, floor)
        self.my = Y.mean(0)
        Z = (X - self.mx) / self.sx
        Zb = np.c_[Z, np.ones(len(Z))]
        best = None
        for lam in 10.0 ** np.arange(-3, 4):                     # leave-one-out error, closed form
            A = Zb.T @ Zb + lam * np.diag(np.r_[np.ones(Z.shape[1]), 0.0])
            Hm = Zb @ np.linalg.solve(A, Zb.T)
            R = (Y - self.my) - Hm @ (Y - self.my)
            loo = np.mean((R / np.clip(1 - np.diag(Hm), 1e-6, None)[:, None]) ** 2)
            if best is None or loo < best[0]:
                best = (loo, lam)
        lam = best[1]
        A = Zb.T @ Zb + lam * np.diag(np.r_[np.ones(Z.shape[1]), 0.0])
        self.W = np.linalg.solve(A, Zb.T @ (Y - self.my))
        return self

    def predict(self, X):
        Z = (X - self.mx) / self.sx
        return np.c_[Z, np.ones(len(Z))] @ self.W + self.my


class GP:
    """Squared-exponential Gaussian process on standardized inputs and outputs; one length and one noise for all outputs,
    chosen by the summed log marginal likelihood over a grid. center=False keeps a prior mean of zero (for residuals,
    whose mean a few gross misses would otherwise shift everywhere)."""

    def __init__(self, center=True):
        self.center = center

    def fit(self, X, Y, floor=None):
        self.mx, self.sx = X.mean(0), scale(X, floor)
        self.my = Y.mean(0) if self.center else np.zeros(Y.shape[1])
        self.sy = (Y.std(0) if self.center else np.sqrt(np.mean(Y ** 2, axis=0))) + 1e-9
        Z = (X - self.mx) / self.sx
        T = (Y - self.my) / self.sy
        D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        d = Z.shape[1]
        best = None
        for ell in np.sqrt(d) * np.array([0.25, 0.5, 1.0, 2.0, 4.0]):
            K0 = np.exp(-0.5 * D2 / ell ** 2)
            for nz in (0.03, 0.1, 0.3, 1.0):
                try:
                    L = np.linalg.cholesky(K0 + nz ** 2 * np.eye(len(Z)))
                except np.linalg.LinAlgError:
                    continue
                a = np.linalg.solve(L.T, np.linalg.solve(L, T))
                ll = -0.5 * np.sum(T * a) - T.shape[1] * np.sum(np.log(np.diag(L)))
                if best is None or ll > best[0]:
                    best = (ll, ell, nz, a)
        _, self.ell, self.nz, self.a = best
        self.Z = Z
        return self

    def predict(self, X):
        Z = (X - self.mx) / self.sx
        K = np.exp(-0.5 * ((Z[:, None, :] - self.Z[None, :, :]) ** 2).sum(-1) / self.ell ** 2)
        return K @ self.a * self.sy + self.my



class MLP:
    """An ensemble of small multilayer perceptrons (torch, CPU): standardized inputs and outputs, two hidden layers of 64
    SiLU units, AdamW with weight decay, full-batch steps. The number of steps is chosen on the last fifth of the
    training rows (the latest presses, rows being in time order), then each member is trained on all rows with it.
    The prediction is the members' mean (owner, 2026-10-23: "does training an MLP work any better than GP")."""

    def __init__(self, members=5, hidden=64, wd=1e-3, lr=3e-3, max_steps=3000, every=50):
        self.members, self.hidden, self.wd, self.lr, self.max_steps, self.every = members, hidden, wd, lr, max_steps, every

    def _net(self, d, k, seed):
        import torch
        torch.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Linear(d, self.hidden), torch.nn.SiLU(), torch.nn.Linear(self.hidden, self.hidden),
                                   torch.nn.SiLU(), torch.nn.Linear(self.hidden, k))

    def _train(self, Z, T, steps, seed, Zv=None, Tv=None):
        import torch
        net = self._net(Z.shape[1], T.shape[1], seed)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        x, y = torch.as_tensor(Z, dtype=torch.float32), torch.as_tensor(T, dtype=torch.float32)
        xv = torch.as_tensor(Zv, dtype=torch.float32) if Zv is not None else None
        yv = torch.as_tensor(Tv, dtype=torch.float32) if Tv is not None else None
        curve = []
        for k in range(1, steps + 1):
            opt.zero_grad()
            loss = torch.mean((net(x) - y) ** 2)
            loss.backward()
            opt.step()
            if xv is not None and k % self.every == 0:
                with torch.no_grad():
                    curve.append((k, float(torch.mean((net(xv) - yv) ** 2))))
        return net, curve

    def fit(self, X, Y, floor=None):
        import torch
        torch.set_num_threads(2)
        self.mx, self.sx = X.mean(0), scale(X, floor)
        self.my, self.sy = Y.mean(0), Y.std(0) + 1e-9
        Z, T = (X - self.mx) / self.sx, (Y - self.my) / self.sy
        n = len(Z)
        if n >= 10:
            cut = int(round(0.8 * n))
            _, curve = self._train(Z[:cut], T[:cut], self.max_steps, 0, Z[cut:], T[cut:])
            self.steps = min(curve, key=lambda c: c[1])[0]
        else:
            self.steps = 500
        self.nets = [self._train(Z, T, self.steps, m)[0] for m in range(self.members)]
        return self

    def predict(self, X):
        import torch
        Z = torch.as_tensor((X - self.mx) / self.sx, dtype=torch.float32)
        with torch.no_grad():
            P = np.mean([net(Z).numpy() for net in self.nets], axis=0)
        return P * self.sy + self.my

# ------------------------------------------------------------------------------------------------------- physical model
class Physical:
    """The viewer's estimator with k, dx, dy, dz and the compliance scale c fitted to training holds."""

    def __init__(self, cal0, basis):
        from scipy.interpolate import RBFInterpolator
        self.cal0, self.basis = cal0, basis
        self.nm = basis["motions"].shape[2]
        self.rbf = RBFInterpolator(basis["pts"], basis["motions"].reshape(len(basis["pts"]), -1),
                                   kernel="thin_plate_spline", degree=1)
        self.theta = np.array([cal0.k, cal0.dx, cal0.dy, cal0.dz, 0.0])     # k, dx, dy, dz, log c
        self.est = None

    def cal(self, th):
        d = dict(self.cal0.__dict__)
        d.update(k=float(th[0]), dx=float(th[1]), dy=float(th[2]), dz=float(th[3]))
        return W.Calib(**d)

    def fit(self, H, idx, iters=25, prior_sd=None):
        P, F, D = H["p"][idx], H["f"][idx], H["d_board"][idx]
        M = self.rbf(P).reshape(len(P), 3, self.nm * 6)                   # basis motions per newton at the true points
        sig = np.sqrt((np.tile(np.asarray(W.NOISE_UT, float), 5)[None] / np.sqrt(np.maximum(H["n_rec"][idx], 1))[:, None]) ** 2
                      + 2.0 ** 2)

        # the magnets' strength and the compliance both scale the change, and the first presses may all stand at one
        # point (the pivots): k and the offsets keep priors around the rest's fit (5 %, 0.2 mm), which the rest field
        # determines; the compliance scale is free
        th0 = np.array([self.cal0.k, self.cal0.dx, self.cal0.dy, self.cal0.dz])
        sd0 = np.array([0.05 * self.cal0.k, 0.2, 0.2, 0.2]) if prior_sd is None else np.asarray(prior_sd, float)

        def resid(th):
            mod = W.Model(self.cal(th))
            Jm = mod.motion_jacobian()
            pred = math.exp(th[4]) * np.einsum("ij,kfj,kf->ki", Jm, M, F)       # (k, 15)
            Nu = np.c_[np.concatenate(list(mod.R), 0), np.asarray(self.cal0.b_rest_uT, float).ravel()]
            Q, _ = np.linalg.qr(Nu)
            R = D - pred
            R = R - (R @ Q) @ Q.T                                          # the nuisances projected out
            return np.r_[(R / sig).ravel(), (th[:4] - th0) / sd0]

        th, r, cov = C.lm(resid, self.theta.copy(), iters=iters)
        self.theta, self.rms_uT = th, float(np.sqrt(np.mean((r[:sig.size] * sig.ravel()) ** 2)))
        self.est = None
        return self

    def estimator(self):
        if self.est is None:
            b = dict(self.basis)
            b["motions"] = self.basis["motions"] * math.exp(self.theta[4])
            self.est = W.SurfaceEstimator(W.Model(self.cal(self.theta)), b, floor_uT=2.0)
        return self.est

    def predict(self, H, idx):
        est = self.estimator()
        P, Fn = np.full((len(idx), 3), np.nan), np.full(len(idx), np.nan)
        for j, i in enumerate(idx):
            est.set_noise_scale(1.0 / math.sqrt(max(H["n_rec"][i], 1)))
            e = est.estimate(H["d_board"][i])
            if e.get("contact"):
                P[j], Fn[j] = e["p"], e["f_n"]
        return P, Fn


# ------------------------------------------------------------------------------------------------------- scores
def scores(H, idx, P, Fn, group="level_N"):
    n = H["n"][idx]
    dp = P - H["p"][idx]
    loc = np.linalg.norm(dp - np.sum(dp * n, axis=1)[:, None] * n, axis=1)
    fe = np.abs(Fn - H["f_n"][idx])
    ok = np.isfinite(loc)
    rel = fe / np.abs(H["f_n"][idx])
    big = H["f_n"][idx] > 0.4

    def st(x):
        x = x[np.isfinite(x)]
        return dict(n=int(len(x)), median=float(np.median(x)) if len(x) else None,
                    p90=float(np.percentile(x, 90)) if len(x) else None)

    out = dict(loc_mm=st(loc), force_N=st(fe), rel_force=st(rel[big]), detected=f"{int(ok.sum())}/{len(idx)}")
    if group == "kind":                                  # turn holds: by slide, roll and twist
        out["by_kind"] = {k: dict(loc_mm=st(loc[H["kind"][idx] == k]), force_N=st(fe[H["kind"][idx] == k]))
                          for k in sorted(set(H["kind"][idx].tolist()))}
        return out
    out["by_level"] = {}
    for L in sorted(set(np.round(H["level_N"][idx], 3)[np.isfinite(H["level_N"][idx])].tolist())):
        m = np.round(H["level_N"][idx], 3) == L
        out["by_level"][f"{L:g}"] = dict(loc_mm=st(loc[m]), force_N=st(fe[m]))
    return out


def snap(est, P):
    """Predicted points onto the nearest of the estimator's surface candidates."""
    from scipy.spatial import cKDTree
    tree = cKDTree(est.C)
    out = P.copy()
    ok = np.all(np.isfinite(P), axis=1)
    out[ok] = est.C[tree.query(P[ok])[1]]
    return out


# ------------------------------------------------------------------------------------------------------- main
def models(H, tr, Ht, te, cal0, basis, est_snap, group="level_N"):
    """Every predictor trained on the holds tr of H and scored on the holds te of Ht."""
    row = dict(holds=int(len(tr)))
    ph = Physical(cal0, basis).fit(H, tr)
    Pp, Fp = ph.predict(Ht, te)
    row["physical"] = scores(Ht, te, Pp, Fp, group)
    row["physical_params"] = dict(k=ph.theta[0], dx=ph.theta[1], dy=ph.theta[2], dz=ph.theta[3],
                                  c=math.exp(ph.theta[4]), rms_uT=ph.rms_uT)
    Ptr, Ftr = ph.predict(H, tr)
    for st, name in enumerate(STEPS):
        Xtr, Xte = features(H, st)[tr], features(Ht, st)[te]
        Ytr = np.c_[H["p"][tr], H["f_n"][tr]]
        fl = floors(st)
        for lab, model in (("gp", GP()), ("ridge", Ridge()), ("mlp", MLP())):
            Y = model.fit(Xtr, Ytr, fl).predict(Xte)
            if lab == "mlp":
                row.setdefault("mlp_steps", {})[name] = model.steps
            row[f"{lab}:{name}"] = scores(Ht, te, snap(est_snap, Y[:, :3]), Y[:, 3], group)
        # the physical predictor plus a GP on its residual, prior mean zero, trained on the holds it placed within 2 mm
        ok = np.all(np.isfinite(Ptr), axis=1)
        ok[ok] &= np.linalg.norm(H["p"][tr][ok] - Ptr[ok], axis=1) < 2.0
        if ok.sum() >= 3:
            Rtr = np.c_[H["p"][tr][ok] - Ptr[ok], H["f_n"][tr][ok] - Ftr[ok]]
            Y = GP(center=False).fit(Xtr[ok], Rtr, fl).predict(Xte)
            row[f"physical+gp:{name}"] = scores(Ht, te, snap(est_snap, Pp + Y[:, :3]), Fp + Y[:, 3], group)
        else:                                           # too few holds to learn a residual: the physical prediction
            row[f"physical+gp:{name}"] = row["physical"]
    return row, ph


def run(folder, sizes, out_dir):
    H, T, cal0 = load(folder)
    basis = W.load_basis(BASIS)
    train, test = split(H)
    te = np.nonzero(np.isin(H["id"], test))[0]
    print(f"{folder}: {len(np.unique(H['id']))} presses, {len(H['id'])} holds; held out {len(test)} grid presses "
          f"({len(te)} holds); training pool {len(train)} presses")
    phys0 = Physical(cal0, basis)
    res = dict(folder=str(folder), n_train_pool=len(train), n_test_presses=len(test), n_test_holds=int(len(te)),
               viewer=scores(H, te, *phys0.predict(H, te)), curve=[])
    print(f"  viewer's estimator (N = 0): location median {res['viewer']['loc_mm']['median']:.2f} mm, force "
          f"{res['viewer']['force_N']['median']:.3f} N")
    est_snap = phys0.estimator()
    for N in sorted(set([s for s in sizes if s < len(train)] + [len(train)])):
        tr = np.nonzero(np.isin(H["id"], train[:N]))[0]
        row, ph = models(H, tr, H, te, cal0, basis, est_snap)
        row = dict(N=N, **row)
        res["curve"].append(row)
        print(f"  N = {N:3d} presses ({len(tr)} holds): physical {row['physical']['loc_mm']['median']:.2f} mm, "
              f"{row['physical']['force_N']['median']:.3f} N (k {ph.theta[0]:.3f}, c {math.exp(ph.theta[4]):.2f}); gp board "
              f"{row['gp:board']['loc_mm']['median']:.2f} mm, {row['gp:board']['force_N']['median']:.3f} N; mlp board "
              f"{row['mlp:board']['loc_mm']['median']:.2f} mm, {row['mlp:board']['force_N']['median']:.3f} N; physical+gp board "
              f"{row.get('physical+gp:board', {}).get('loc_mm', {}).get('median', float('nan')):.2f} mm")
    # the turn holds (slides, rolls, twists), whose tool orientations no press has: every press trains
    if len(T["id"]) >= 3:
        tt = np.arange(len(T["id"]))
        row, ph = models(H, np.arange(len(H["id"])), T, tt, cal0, basis, est_snap, group="kind")
        res["turns"] = dict(n_actions=int(len(np.unique(T["id"]))), n_holds=int(len(tt)),
                            viewer=scores(T, tt, *phys0.predict(T, tt), group="kind"), **row)
        tv = res["turns"]
        print(f"  turn holds ({tv['n_holds']} from {tv['n_actions']} actions), trained on every press: viewer "
              f"{tv['viewer']['loc_mm']['median']:.2f} mm, {tv['viewer']['force_N']['median']:.3f} N; physical "
              f"{tv['physical']['loc_mm']['median']:.2f} mm; gp board {tv['gp:board']['loc_mm']['median']:.2f} mm, "
              f"{tv['gp:board']['force_N']['median']:.3f} N; mlp board {tv['mlp:board']['loc_mm']['median']:.2f} mm, "
              f"{tv['mlp:board']['force_N']['median']:.3f} N; gp with orientation "
              f"{tv['gp:+orientation (leaks the target)']['loc_mm']['median']:.2f} mm")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predict.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"-> {out_dir / 'predict.json'}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="the analysis's output folder of a session (holds.npz, calib_rest.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sizes", default="3,5,10,20,40")
    a = ap.parse_args()
    run(a.folder, [int(s) for s in a.sizes.split(",")], a.out or Path(a.folder) / "predict")


if __name__ == "__main__":
    main()
