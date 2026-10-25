#!/usr/bin/env python3
"""One fingertip's predictors on another fingertip pressed on the UR5 (QUEUE.md R22, the first checkerboard tip; owner,
2026-10-23: "how transferable do you think this result will be to the higher strength n52 magnets + the checkerboard
orientation ... transform our physical model to work with whatever detected orientation we have, and maybe even magnet
strength ... the GP/MLP can just be used to fine-tune"). The real-data counterpart of
scripts/fingertip_magnet_transfer_sim.py.

Session A trains (sphere2's analysis output), session B is the other tip (cb1t's), each with its own rest fit
(calib_rest.json: poles, strength k, offsets). B's presses are split as in scripts/ur5_tip_calibration_predict.py (every
fourth grid press held out, in time order); every predictor is scored on B's held-out holds, and those trained on A
alone also on all of B's press holds.

  physical            the viewer's surface estimator built on B's rest fit, trained on nothing
  gp raw              a Gaussian process from the board's 15 components to the contact point and the normal force
  physical+gp         the physical estimate corrected by a Gaussian process whose inputs are the physical estimate itself
                      (contact point, normal force) and whose outputs are the corrections; trained on the holds that the
                      physical estimate placed within 2 mm, so a correction learned on A applies to any tip whose physical
                      model is rebuilt from its rest
  physical+offset     the physical estimate moved by one constant offset (mm) and its normal force times one gain: four
                      numbers, fitted by least squares on the same holds as physical+gp
  gp/mlp motions      a Gaussian process or an MLP ensemble from the five magnets' translations (um), solved from the
                      board's change through each session's own field model (translations only, 15 x 15)

Each is trained on A alone, on A and the first N of B's training presses in time order (fine-tuning), and on B's first N
alone; the physical model of each session is its own (N = 0, the viewer's).

    .venv/bin/python scripts/ur5_tip_calibration_tip_transfer.py <A folder> <B folder> [--out B/predict/transfer.json]
        [--sizes 3,5,10,20]
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

PRED = ("gp raw", "physical+gp", "physical+offset", "gp motions", "mlp motions")
FLOOR_EST = np.array([0.2, 0.2, 0.2, 0.05])          # mm, mm, mm, N: the physical estimate's resolution


def motions(H, cal):
    """The five magnets' translations (um) that explain each hold's board change through the session's field model."""
    Jm = W.Model(cal).motion_jacobian()
    Jt = Jm[:, [6 * j + q for j in range(5) for q in range(3)]]
    return H["d_board"] @ np.linalg.pinv(Jt).T


class Holds:
    """Holds with the viewer's estimate and the magnet motions of each."""
    def __init__(self, H, cal, phys):
        self.H, self.cal, self.phys = H, cal, phys
        self.P, self.F = phys.predict(H, np.arange(len(H["id"])))
        self.U = motions(H, cal)


class Session:
    def __init__(self, folder, basis):
        self.H, T, self.cal = PR.load(folder)
        self.T = T if len(T["id"]) else None                           # turn holds: rolls, twists, slides
        self.folder = str(folder)
        allr = np.arange(len(self.H["id"]))
        self.phys = PR.Physical(self.cal, basis)
        self.P, self.F = self.phys.predict(self.H, allr)              # the viewer's estimate at every hold
        self.U = motions(self.H, self.cal)
        self.Y = np.c_[self.H["p"], self.H["f_n"]]
        ok = np.all(np.isfinite(self.P), axis=1)
        ok[ok] &= np.linalg.norm(self.H["p"][ok] - self.P[ok], axis=1) < 2.0
        self.near = ok                                                # holds the physical estimate placed within 2 mm
        self.turns = Holds(self.T, self.cal, self.phys) if self.T is not None else None


def train(parts):
    """The four predictors trained on the rows of each (session, rows) in parts."""
    cat = lambda f: np.concatenate([f(s, r) for s, r in parts], axis=0)   # noqa: E731
    D, U, Y = cat(lambda s, r: s.H["d_board"][r]), cat(lambda s, r: s.U[r]), cat(lambda s, r: s.Y[r])
    near = [(s, r[s.near[r]]) for s, r in parts]
    X = np.concatenate([np.c_[s.P[r], s.F[r]] for s, r in near], axis=0)
    R = np.concatenate([np.c_[s.H["p"][r] - s.P[r], s.H["f_n"][r] - s.F[r]] for s, r in near], axis=0)
    m = {"gp raw": PR.GP().fit(D, Y, PR.floors(0)), "gp motions": PR.GP().fit(U, Y, np.full(15, 0.05)),
         "mlp motions": PR.MLP().fit(U, Y, np.full(15, 0.05))}
    m["physical+gp"] = PR.GP(center=True).fit(X, R, FLOOR_EST) if len(X) >= 3 else None
    m["physical+offset"] = ((R[:, :3].mean(0), float(np.sum((X[:, 3] + R[:, 3]) * X[:, 3]) / np.sum(X[:, 3] ** 2)))
                            if len(X) >= 1 else None)
    return m


def score(m, B, idx, snap):
    out = {}
    for name in PRED:
        if name == "physical+offset":
            b, g = m[name] if m[name] is not None else (np.zeros(3), 1.0)
            P, F = snap(B.P[idx] + b), g * B.F[idx]
        elif name == "physical+gp":
            if m[name] is None:
                P, F = B.P[idx], B.F[idx]
            else:
                X = np.c_[B.P[idx], B.F[idx]]
                ok = np.all(np.isfinite(X), axis=1)
                C = np.full((len(idx), 4), np.nan)
                C[ok] = m[name].predict(X[ok])
                P, F = snap(B.P[idx] + C[:, :3]), B.F[idx] + C[:, 3]
        else:
            Yp = m[name].predict((B.H["d_board"] if name == "gp raw" else B.U)[idx])
            P, F = snap(Yp[:, :3]), Yp[:, 3]
        out[name] = PR.scores(B.H, idx, P, F)
        out[name]["loc_mm_2N"] = loc_2n(B, idx, P)
    return out


def loc_2n(B, idx, P):
    """Median location error (mm) over the holds of idx at force levels of 2 N and over."""
    n, dp = B.H["n"][idx], P - B.H["p"][idx]
    loc = np.linalg.norm(dp - np.sum(dp * n, axis=1)[:, None] * n, axis=1)
    m = (B.H["level_N"][idx] >= 2.0) & np.isfinite(loc)
    return float(np.median(loc[m])) if m.any() else None


def short(s):
    return f"{s['loc_mm']['median']:.2f} mm, {s['force_N']['median']:.3f} N ({s['detected']})"


def run(fa, fb, out, sizes):
    basis = W.load_basis(PR.BASIS)
    A, B = Session(fa, basis), Session(fb, basis)
    trainB, testB = PR.split(B.H)
    te = np.nonzero(np.isin(B.H["id"], testB))[0]
    allB = np.arange(len(B.H["id"]))
    allA = np.arange(len(A.H["id"]))
    snap = lambda P: PR.snap(B.phys.estimator(), P)                  # noqa: E731
    res = dict(A=A.folder, B=B.folder, n_A_holds=int(len(allA)), n_B_holds=int(len(allB)), n_B_test_holds=int(len(te)),
               n_B_test_presses=len(testB), n_B_train_presses=len(trainB),
               cal_A=dict(k=A.cal.k, poles=list(A.cal.poles), rms_mT=A.cal.rms_mT),
               cal_B=dict(k=B.cal.k, poles=list(B.cal.poles), rms_mT=B.cal.rms_mT),
               physical=dict(heldout=PR.scores(B.H, te, B.P[te], B.F[te]), all=PR.scores(B.H, allB, B.P, B.F)))
    for part, idx in (("heldout", te), ("all", allB)):
        res["physical"][part]["loc_mm_2N"] = loc_2n(B, idx, B.P[idx])
    print(f"A {A.folder}: {len(allA)} holds, k {A.cal.k:.3f}, poles {A.cal.poles}; B {B.folder}: {len(allB)} holds, "
          f"k {B.cal.k:.3f}, poles {B.cal.poles}; B held out {len(testB)} presses ({len(te)} holds)", flush=True)
    print(f"  physical on B's rest fit: held out {short(res['physical']['heldout'])}; all {short(res['physical']['all'])}",
          flush=True)
    mA = train([(A, allA)])
    res["A_only"] = dict(heldout=score(mA, B, te, snap), all=score(mA, B, allB, snap))
    for name in PRED:
        print(f"  {name:12s} trained on A: held out {short(res['A_only']['heldout'][name])}; all "
              f"{short(res['A_only']['all'][name])}", flush=True)
    if B.turns is not None:                       # every press of B trains; scored on B's turn holds, at new orientations
        tt = np.arange(len(B.turns.H["id"]))
        res["turns"] = dict(n_holds=int(len(tt)), physical=PR.scores(B.turns.H, tt, B.turns.P, B.turns.F, group="kind"),
                            B_all=score(train([(B, allB)]), B.turns, tt, snap))
        res["turns"]["physical"]["loc_mm_2N"] = loc_2n(B.turns, tt, B.turns.P)
        print(f"  turn holds ({len(tt)}), trained on all of B's presses: physical "
              f"{res['turns']['physical']['loc_mm']['median']:.2f} / {res['turns']['physical']['loc_mm_2N']:.2f} mm (all / "
              f">= 2 N); " + "; ".join(f"{k} {v['loc_mm']['median']:.2f} / {v['loc_mm_2N']:.2f}"
                                      for k, v in res["turns"]["B_all"].items()), flush=True)
    res["curve"] = []
    for N in sorted(set([s for s in sizes if s < len(trainB)] + [len(trainB)])):
        rB = np.nonzero(np.isin(B.H["id"], trainB[:N]))[0]
        row = dict(N=N, holds=int(len(rB)), A_and_B=score(train([(A, allA), (B, rB)]), B, te, snap),
                   B_only=score(train([(B, rB)]), B, te, snap))
        res["curve"].append(row)
        print(f"  N = {N:3d} ({len(rB)} holds): " + "; ".join(
            f"{name} A+B {row['A_and_B'][name]['loc_mm']['median']:.2f} / B {row['B_only'][name]['loc_mm']['median']:.2f} mm"
            for name in PRED), flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("->", out)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("train_folder")
    ap.add_argument("test_folder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sizes", default="3,5,10,20")
    a = ap.parse_args()
    run(a.train_folder, a.test_folder, a.out or str(Path(a.test_folder) / "predict/transfer.json"),
        [int(s) for s in a.sizes.split(",")])
