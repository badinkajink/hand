#!/usr/bin/env python3
"""The pad's FE model against the calibration session's ground truth: for every press hold, the normal force that the
FE model (scripts/fingertip_wrench_fe.py --surface, docs/experiments/20261020-fingertip_contact_estimation/data/
20261020-b_v2_surface_basis.npz) gives with the sphere's centre where the robot and the HEX-E place it (the analysis's
c_P, the HEX-E's deflection removed), and the compliance scale s of the printed pad over the FE that fits the measured
forces best.

The contact is the simulator's normal solve (ur5_tip_calibration_sim.World.contact without friction and without the
HEX-E, whose deflection the ground truth has already removed): the surface points within 0.6 mm of the sphere and facing
it, their gaps closed by the FE's compliance matrix scaled by s, with non-negative normal forces (NNLS). The force at a
fixed sphere position is not proportional to 1/s, since the contact grows with depth, so s is fitted on a grid.

    .venv/bin/python scripts/ur5_tip_calibration_fe.py <holds.npz from ur5_tip_calibration_analysis.py> [--out fe.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz"
PLAN = ROOT / "docs/experiments/20261022-ur5_tip_calibration/plan/20261022-plan_sphere.json"


class FEPad:
    def __init__(self, basis=BASIS):
        B = np.load(basis)
        self.X, self.N = B["pts"].astype(float), B["normals"].astype(float)
        n = len(self.X)
        du = B["disp_u"].astype(np.float32)                       # [load i, dir a, node j, dir b] um / N
        Cm = np.transpose(du, (2, 3, 0, 1)).reshape(3 * n, 3 * n).astype(float) * 1e-3     # u = C F, mm / N
        self.C = 0.5 * (Cm + Cm.T)

    def force(self, c, s=1.0, R=5.0):
        """Normal contact force (N, pad frame) of the FE pad on a sphere of radius R centred at c (mm, pad frame), with
        the FE's compliance scaled by s; and the number of points in contact."""
        r = self.X - c
        dist = np.linalg.norm(r, axis=1)
        gap = dist - R
        nh = r / dist[:, None]
        K = np.nonzero((gap < 0.6) & (np.einsum("ij,ij->i", nh, self.N) < -0.2))[0]
        if len(K) == 0 or gap[K].min() >= 0:
            return np.zeros(3), 0
        idx = (3 * K[:, None] + np.arange(3)[None]).ravel()
        Ck = self.C[np.ix_(idx, idx)] * s
        Nk = np.zeros((3 * len(K), len(K)))
        for j, i in enumerate(K):
            Nk[3 * j:3 * j + 3, j] = nh[i]
        Mn = Nk.T @ Ck @ Nk
        Mn = 0.5 * (Mn + Mn.T) + 1e-9 * np.eye(len(K))
        Lc = np.linalg.cholesky(Mn)
        p, _ = nnls(Lc.T, -np.linalg.solve(Lc, gap[K]))
        F = (Nk @ p).reshape(-1, 3)
        return F.sum(axis=0), int((p > 1e-9).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("holds")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plan", default=str(PLAN), help="the session's plan.json, for the indenter's radius")
    ap.add_argument("--min-N", type=float, default=0.2, help="holds with a measured normal force under this are left out")
    a = ap.parse_args()
    H = np.load(a.holds, allow_pickle=False)
    m = (H["kind"] == "press") & np.isfinite(H["f_n"]) & (H["f_n"] > a.min_N)
    idx = np.nonzero(m)[0]
    R = float(json.loads(Path(a.plan).read_text())["geometry"]["indenter"]["radius"])
    pad = FEPad()
    grid = np.round(np.arange(0.6, 1.81, 0.05), 3)
    pred = np.zeros((len(grid), len(idx)))
    for k, s in enumerate(grid):
        for j, i in enumerate(idx):
            f, _ = pad.force(H["c_P"][i], s, R)
            pred[k, j] = float(-f @ H["n"][i])                    # the push along the contact normal, as f_n
    meas = H["f_n"][idx]
    rel = np.abs(pred / meas[None] - 1.0)
    cost = np.median(rel, axis=1)
    kb = int(np.argmin(cost))
    out = dict(n=len(idx), s_grid=grid.tolist(), median_rel_err=cost.tolist(), s_best=float(grid[kb]),
               median_rel_err_best=float(cost[kb]), median_rel_err_s1=float(cost[np.argmin(np.abs(grid - 1.0))]))
    by = {}
    reg = H["region"][idx]
    lv = H["level_N"][idx]
    for r in sorted(set(reg.tolist())):
        mm = reg == r
        by[r] = dict(n=int(mm.sum()), ratio_median_s1=float(np.median(meas[mm] / pred[np.argmin(np.abs(grid - 1.0)), mm])),
                     ratio_median_best=float(np.median(meas[mm] / pred[kb, mm])))
    out["by_region"] = by
    lvl = {}
    for L in sorted(set(lv[np.isfinite(lv)].tolist())):
        mm = lv == L
        lvl[f"{L:g}"] = dict(n=int(mm.sum()), ratio_median_s1=float(np.median(meas[mm] / pred[np.argmin(np.abs(grid - 1.0)), mm])))
    out["by_level"] = lvl
    print(f"{len(idx)} press holds over {a.min_N} N: compliance scale over the FE s = {grid[kb]:.2f} (median force error "
          f"{cost[kb] * 100:.1f} %; at s = 1 {out['median_rel_err_s1'] * 100:.1f} %)")
    for r, v in by.items():
        print(f"  {r:6s} n={v['n']:3d}: measured over FE at s = 1 {v['ratio_median_s1']:.3f}, at s = {grid[kb]:.2f} {v['ratio_median_best']:.3f}")
    for L, v in lvl.items():
        print(f"  level {L:>4s} N n={v['n']:3d}: measured over FE at s = 1 {v['ratio_median_s1']:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
