#!/usr/bin/env python3
"""The Gaussian process from the board's change to the contact point and the 3D force, pooled over UR5 sessions, fitted
once and saved for the contact viewer (owner, 2026-10-26: ID 1 and ID 2 on the mux, "let's see if this additional
calibration has improved sensor quality"). The model is the calibration coordinator's (scripts/ur5_tip_calibration_
cb1r_checks.py gp: inputs the 15 board components, outputs the contact point in mm and the force in N, pad frame), here
on the presses and slides of ID 1 (cb1t, cb1r) and ID 2 (cb2t). The fit is ur5_tip_calibration_predict.GP's: inputs
standardized with the board's 1 uT floor, outputs standardized, one length and one noise picked on the same grid by the
summed log marginal likelihood. The squared distances are formed from dot products, so the fit needs O(N^2) memory
where GP.fit's broadcast needs O(N^2 d); --check compares the two on a subset.

The v3 tips' GP (2026-11-02) is pooled over the sphere and rod sessions of v3 ID 1 to ID 3, 9,185 presses and slides
analysed under logs/. The fit took 63 s and 3.4 GB on four cores, and the marginal likelihood was highest at length
1.94 and noise 0.3. On 300 of its holds the two fits agreed to 3e-13.

    .venv/bin/python scripts/ur5/ur5_tip_calibration_board_gp.py [--sessions 20261023-203938-cb1t 20261025-114653-cb1r
        20261025-152920-cb2t] [--kinds press slide] [--check] [--out data/20261026-board_gp_id1_id2.npz]
    ~/.claude/bin/resguard.sh run --mem 4G --cpu 500 -- env OPENBLAS_NUM_THREADS=4 taskset -c 0-3 .venv/bin/python \\
        scripts/ur5/ur5_tip_calibration_board_gp.py --check --sessions logs/20261028-spherev3id1_twists logs/20261028-rodv3id1 \\
        logs/20261029-spherev3id2 logs/20261029-rodv3id2 logs/20261030-spherev3id3 logs/20261030-rodv3id3 \\
        --out docs/experiments/20261022-ur5_tip_calibration/data/20261102-board_gp_v3_id1_id3.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import ur5_tip_calibration_predict as PR  # noqa: E402

DATA = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
SESSIONS = ("20261023-203938-cb1t", "20261025-114653-cb1r", "20261025-152920-cb2t")


def holds(sessions, kinds):
    """The holds of each session: a folder under DATA by name, or any analysed session folder by path (the v3 tips'
    are under logs/, 2026-11-02)."""
    X, Y, src = [], [], []
    for s in sessions:
        z = np.load((Path(s) if Path(s).is_dir() else DATA / s) / "holds.npz", allow_pickle=True)
        ok = (np.isin(z["kind"].astype(str), kinds) & np.isfinite(z["d_board"]).all(1) & np.isfinite(z["p"]).all(1)
              & np.isfinite(z["f"]).all(1))
        X.append(z["d_board"][ok])
        Y.append(np.c_[z["p"][ok], z["f"][ok]])
        src += [s] * int(ok.sum())
        print(f"{s}: {int(ok.sum())} holds ({', '.join(kinds)})", flush=True)
    return np.concatenate(X), np.concatenate(Y), np.array(src)


def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T, 0.0)


def fit(X, Y, floor):
    """GP.fit of ur5_tip_calibration_predict (center=True) with D2 from dot products."""
    mx, sx = X.mean(0), PR.scale(X, floor)
    my, sy = Y.mean(0), Y.std(0) + 1e-9
    Z, T = (X - mx) / sx, (Y - my) / sy
    D2 = sqdist(Z, Z)
    d = Z.shape[1]
    best = None
    for ell in np.sqrt(d) * np.array([0.25, 0.5, 1.0, 2.0, 4.0]):
        K0 = np.exp(-0.5 * D2 / ell ** 2)
        for nz in (0.03, 0.1, 0.3, 1.0):
            K0[np.diag_indices_from(K0)] = 1.0 + nz ** 2
            try:
                L = np.linalg.cholesky(K0)
            except np.linalg.LinAlgError:
                continue
            a = cho_solve((L, True), T)                 # the two triangular solves: no LU copies of the N x N factor
            ll = -0.5 * np.sum(T * a) - T.shape[1] * np.sum(np.log(np.diag(L)))
            if best is None or ll > best[0]:
                best = (ll, ell, nz, a)
        del K0
    ll, ell, nz, a = best
    return dict(mx=mx, sx=sx, my=my, sy=sy, Z=Z, a=a, ell=float(ell), nz=float(nz), loglik=float(ll))


def predict(m, X):
    Zq = (X - m["mx"]) / m["sx"]
    return np.exp(-0.5 * sqdist(Zq, m["Z"]) / m["ell"] ** 2) @ m["a"] * m["sy"] + m["my"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", nargs="+", default=list(SESSIONS))
    ap.add_argument("--kinds", nargs="+", default=["press", "slide"])
    ap.add_argument("--check", action="store_true", help="compare with ur5_tip_calibration_predict.GP on 300 holds")
    ap.add_argument("--out", default=str(DATA / "20261026-board_gp_id1_id2.npz"))
    a = ap.parse_args()
    X, Y, src = holds(a.sessions, a.kinds)
    floor = PR.floors(0)[:15]
    if a.check:
        rng = np.random.default_rng(0)
        i = rng.choice(len(X), 300, replace=False)
        ref = PR.GP().fit(X[i], Y[i], floor)
        mine = fit(X[i], Y[i], floor)
        q = rng.choice(len(X), 200, replace=False)
        diff = np.abs(ref.predict(X[q]) - predict(mine, X[q])).max()
        print(f"check on 300 holds: length {ref.ell:.3f} vs {mine['ell']:.3f}, noise {ref.nz} vs {mine['nz']}, largest "
              f"prediction difference {diff:.2e}", flush=True)
    t0 = time.time()
    m = fit(X, Y, floor)
    print(f"fitted on {len(X)} holds in {time.time() - t0:.1f} s: length {m['ell']:.3f}, noise {m['nz']}", flush=True)
    np.savez_compressed(a.out, **{k: np.asarray(v) for k, v in m.items()}, sessions=np.array(a.sessions),
                        kinds=np.array(a.kinds), n_holds=len(X),
                        n_by_session=np.array([int((src == s).sum()) for s in a.sessions]))
    print("->", a.out)


if __name__ == "__main__":
    main()
