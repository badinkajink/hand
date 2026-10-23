#!/usr/bin/env python3
"""Can the UR5 calibration plan calibrate an ATI Mini45 stacked on the HEX-E? A simulation on the UR5 run sphere2's
press wrenches (owner, 2026-10-23: "we can probably calibrate the ATI with just a 3dp mount to the HEX-E, with the current
indenter setup").

The stack: HEX-E, mount v3's adapter, a printed base, the ATI Mini45, a printed top, mount v3's fork and the finger. The
ATI's origin stands stack_mm above v3's adapter face plus the base, and every part above it moves up by the ATI and the
two discs. Each press hold of sphere2 gives a force at a contact point (pad frame), carried to M; its wrench at the ATI's
origin is the calibration's excitation. A true calibration matrix C (gage counts to N and N m) with the shape of ATI's
Mini45 FT31565 matrix (rows of the handoff's R_31565) and per-gage gains off by up to 10 % makes the gage readings g = C^-1 w
plus ATI noise; the reference is the HEX-E's wrench with its data-sheet noise and a gain error per axis. C is fitted by
regularized least squares on the first N presses in time order and scored on the held-out presses, per axis, with and
without gravity poses that carry a known mass on an arm (torsion about the tool axis).

    .venv/bin/python scripts/ati_calibration_sim.py <analysis output folder of sphere2> [--out ati_sim.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_cb3_cell as CELL  # noqa: E402

R_31565 = np.array([[-0.026289, 0.000278, 0.056518, -0.702557, -0.032851, 0.708130],
                    [-0.095998, 0.795815, 0.015191, -0.420373, 0.026813, -0.424028],
                    [0.575115, 0.028712, 0.571098, 0.027078, 0.583951, 0.023091],
                    [-0.033322, 0.232680, -0.670809, -0.155471, 0.679074, -0.097216],
                    [0.774309, 0.038323, -0.420813, 0.191089, -0.366102, -0.226602],
                    [0.079040, -0.547565, 0.009397, -0.583523, 0.043265, -0.592846]])
HEX_NOISE = np.array([0.035, 0.035, 0.15, 0.002, 0.002, 0.001])      # N, N m: HEX-E data sheet, 1 sigma
ATI_RES = np.array([1 / 16, 1 / 16, 1 / 16, 1 / 752, 1 / 752, 1 / 752])   # Mini45 SI-145-5 resolution, N and N m (1 sigma)
BASE_MM, ATI_MM, TOP_MM = 10.0, 15.7, 8.0                            # printed base, Mini45 height (with adapters), top


def wrenches(H, o_M):
    """Each hold's wrench (N, N m) at the ATI origin o_M (M frame), from its force and contact point (pad frame P)."""
    R_MP, t_MP = CELL.T_M_FROM_P[:3, :3], CELL.T_M_FROM_P[:3, 3]
    shift = np.array([0.0, 0.0, BASE_MM + ATI_MM + TOP_MM])             # the mount and finger move up by the stack
    p_M = H["p"] @ R_MP.T + t_MP + shift
    f_M = H["f"] @ R_MP.T
    tq = np.cross((p_M - o_M) * 1e-3, f_M)
    return np.c_[f_M, tq]


ARMS_MM = ((60.0, 0.0, 40.0), (-60.0, 0.0, 40.0), (0.0, 60.0, 40.0), (0.0, -60.0, 40.0), (0.0, 0.0, 100.0))


def gravity_poses(n, mass_kg, arm_mm, rng, arms=None):
    """Wrenches at the ATI origin of a known mass hung arm_mm off the tool axis (or at each of the points `arms`, mm), at
    n random tool orientations per point."""
    pts = [np.array([arm_mm, 0.0, 40.0])] if arms is None else [np.asarray(a, float) for a in arms]
    out = []
    for r in pts:
        for _ in range(n):
            g = rng.normal(size=3)
            g = g / np.linalg.norm(g) * 9.81 * mass_kg
            out.append(np.r_[g, np.cross(r * 1e-3, g)])
    return np.array(out)


def fit(G, W, lam=1e-6):
    """C (6 x 6) with W = G C^T, ridge on standardized gages."""
    s = G.std(0) + 1e-12
    Z = G / s
    A = Z.T @ Z + lam * len(Z) * np.eye(6)
    Ct = np.linalg.solve(A, Z.T @ W)
    return (Ct / s[:, None]).T


def run(folder, out, seeds=20, sizes=(20, 59)):
    import ur5_tip_calibration_predict as PR
    H, _, _ = PR.load(folder)
    train, test = PR.split(H)
    o_M = np.array([0.0, 0.0, 15.0 + BASE_MM])                         # the ATI's mounting face; origin assumed there
    W = wrenches(H, o_M)
    te = np.nonzero(np.isin(H["id"], test))[0]
    full = dict(range_N_Nm=np.round(np.abs(W).max(0), 4).tolist(), median_abs=np.round(np.median(np.abs(W), 0), 4).tolist())
    sc = np.array([1, 1, 1, 10, 10, 10.0])                              # N against N m over 0.1 m
    sv = np.linalg.svd(W * (1 / sc[None]) * sc[None] ** 0 / sc, compute_uv=False)
    res = dict(stack_mm=dict(base=BASE_MM, ati=ATI_MM, top=TOP_MM), origin_M=o_M.tolist(), excitation=full,
               singular_values_scaled=np.round(np.linalg.svd(W / np.array([1, 1, 1, 0.1, 0.1, 0.1]), compute_uv=False), 3).tolist(),
               curves={})
    cases = (("presses, HEX-E reference", 0, "hex"),
             ("presses (HEX-E) and 24 gravity poses (HEX-E)", 24, "hex"),
             ("presses (HEX-E) and 24 gravity poses (known mass)", 24, "mass"),
             ("24 gravity poses (known mass) alone", 24, "mass_only"),
             ("presses (HEX-E) and 5 hook points x 8 poses (known mass)", 8, "mass5"),
             ("5 hook points x 8 poses (known mass) alone", 8, "mass5_only"))
    for case, n_grav, ref in cases:
        rows = []
        for N in sizes:
            tr = np.nonzero(np.isin(H["id"], train[:N]))[0]
            e_press, e_gen = [], []
            for sd in range(seeds):
                rng = np.random.default_rng(sd)
                gains = 1.0 + rng.uniform(-0.1, 0.1, 6)
                C = (R_31565 * 3000.0) * gains[None, :]                     # counts -> N, N m, an arbitrary true scale
                Cinv = np.linalg.inv(C)
                hex_gain = 1.0 + rng.normal(0, 0.03, 6)                     # the reference's own gain error per axis
                noise = lambda n: (rng.normal(0, 1.0, (n, 6)) * ATI_RES[None]) @ Cinv.T              # noqa: E731  resolution, N and N m
                Wp = W[tr] if not ref.endswith("only") else np.zeros((0, 6))
                Rp = Wp * hex_gain[None] + rng.normal(0, 1.0, Wp.shape) * HEX_NOISE[None]
                Wg = (gravity_poses(n_grav, 0.5, 60.0, rng, ARMS_MM if ref.startswith("mass5") else None) if n_grav
                      else np.zeros((0, 6)))
                Rg = (Wg * hex_gain[None] + rng.normal(0, 1.0, Wg.shape) * HEX_NOISE[None]) if ref == "hex" else \
                    Wg * (1.0 + rng.normal(0, 0.005))                           # a weighed mass: 0.5 % on m g r
                Wtr, Wref = np.r_[Wp, Wg], np.r_[Rp, Rg]
                Ch = fit(Wtr @ Cinv.T + noise(len(Wtr)), Wref)
                e_press.append(np.abs((W[te] @ Cinv.T + noise(len(te))) @ Ch.T - W[te]))
                Wgen = gravity_poses(60, 0.3, 35.0, rng) + np.r_[[np.r_[rng.normal(0, 2, 3), rng.normal(0, 0.1, 3)]
                                                                  for _ in range(60)]]
                e_gen.append(np.abs((Wgen @ Cinv.T + noise(len(Wgen))) @ Ch.T - Wgen))
            Ep, Eg = np.concatenate(e_press), np.concatenate(e_gen)
            rows.append(dict(N=N, press_median_abs=np.round(np.median(Ep, 0), 4).tolist(),
                             general_median_abs=np.round(np.median(Eg, 0), 4).tolist()))
            print(f"{case:50s} N {N:3d}: presses F {np.round(np.median(Ep, 0)[:3], 3)} N, T {np.round(np.median(Ep, 0)[3:] * 1e3, 1)} "
                  f"N mm | general F {np.round(np.median(Eg, 0)[:3], 3)} N, T {np.round(np.median(Eg, 0)[3:] * 1e3, 1)} N mm", flush=True)
        res["curves"][case] = rows
    print("excitation range |w| max:", full["range_N_Nm"], "; singular values (N, N m / 0.1 m):", res["singular_values_scaled"])
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--out", default=str(ROOT / "docs/experiments/20261023-ati_ft_calibration/20261023-ati_calibration_sim.json"))
    a = ap.parse_args()
    run(a.folder, a.out)
