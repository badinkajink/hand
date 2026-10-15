#!/usr/bin/env python3
"""Quasi-static contact laws of a convex fingertip against the bed's tool cylinder (R 12.5 mm), the references of the
fingertip co-design study (docs/notes/20261014-tip_codesign_plan.md, step 3).

The tool's axis lies in the tip's y-z plane (y the PIP axis, z along the finger) at angle psi from y and is pressed a
depth delta into the palmar face (tip frame of fingertip_geometry, palmar normal +x). Four pressure fields:

  drake    Drake's compliant-convex field p = E (1 - g(x)), g the gauge of the faceted hull about its volume
           centroid c_V (pydrake 1.57 builds the volume mesh as a fan of tetrahedra from c_V)
  winkler  p = E delta_i / h with h 8.5 mm, delta_i the radial overlap of the tool at a surface point (the pad
           builder's law)
  local    the same with h_i = n_i . (x_i - c_V), the normal distance to the volume centroid (Drake's field to
           first order)
  pads     the builder's discrete spheres, K_i delta_i with the finger's median K (as reorient_backends sets it)
           or each pad's own K

For each field and (psi, N): the approach delta at pinch force N, the all-slip friction arm about the pinch axis
(friction torque over mu N for every contact point sliding under a spin about x), the patch area, rho90 (90th
percentile of the loaded points' distance from the pinch axis) and the number of loaded pads. The quasi-static brake
N(phi) = m g d cos(phi) / (2 mu arm(N, psi = phi)) is solved as a fixed point for N10, N45 and N80 (the bed's T5:
24.5 g tool, centre of mass 15 mm from the pinch).

    .venv/bin/python scripts/tip_codesign_law.py --tips tpu2.7 tpu6 tpu0 --out <rows.jsonl>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_geometry as G  # noqa: E402

E0, RC, MU = 1e7, 0.0125, 1.0
RS, S = 0.00075, 0.001
M_TOOL = 500.0 * math.pi * RC ** 2 * 0.1
D_CG, GRAV = 0.015, 9.81
PSI_DEG = (0, 15, 30, 45, 60, 75, 90)
N_LIST = (0.5, 1.0, 3.0, 5.0)
SCRIPT = "scripts/tip_codesign_law.py"


class TpuShape:
    """fingertip_geometry's block with every edge rounded at r (the 'tpu<r>' tips), palmar sign +1."""

    def __init__(self, r: float):
        self.r, self.key = r, f"tpu{r * 1e3:g}"
        self.reach = G.REACH

    def surface(self, s):
        P, Nn, A = G.surface_samples(self.r, s, 1.0)
        keep = G.pad_mask(Nn, 1.0) & (Nn[:, 0] > 0.05)
        return P[keep], Nn[keep], A[keep]

    def hull(self):
        return G.hull_mesh(self.r, 1.0)

    def centroid(self):
        return G.CENTRE.copy()

    def pads(self, E):
        C, Nn, A, K = G.pad_spheres(self.r, S, RS, 1.0, E)
        return C, K


def shape_of(key: str):
    if key.startswith("tpu"):
        return TpuShape(float(key[3:]) * 1e-3)
    import tip_codesign_geometry as TG               # step 4's shapes
    return TG.shape(key)


def _axis(reach, psi, delta):
    return np.array([reach + RC - delta, 0.0, 0.0]), np.array([0.0, math.cos(psi), math.sin(psi)])


def _arm(Xc, f, d):
    """Contact points Xc on the tool, normal force magnitudes f along unit d (pushing the tool); every point slides
    under a spin about x through (y, z) = 0. Returns N (x-component), the arm tau / (mu N), rho90 and n."""
    N = float(np.sum(f * d[:, 0]))
    r = Xc.copy()
    r[:, 0] = 0.0
    v = np.cross(np.array([1.0, 0, 0]), r)
    vt = v - np.sum(v * d, 1, keepdims=True) * d
    tdir = -vt / np.maximum(np.linalg.norm(vt, axis=1, keepdims=True), 1e-30)
    tau = -np.sum(np.cross(r, MU * f[:, None] * tdir)[:, 0])
    rho = np.linalg.norm(r, axis=1)
    return N, (tau / (MU * N) if N > 0 else 0.0), (float(np.percentile(rho, 90)) if len(rho) else 0.0), int(len(f))


class DrakeField:
    def __init__(self, shape, E=E0):
        V, F = shape.hull()
        tri = V[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        self.c, self.E, self.reach = shape.centroid(), E, shape.reach
        keep = n[:, 0] > -0.2
        self.n = n[keep]
        self.dk = np.sum(n[keep] * (tri[keep, 0] - self.c), 1)

    def p(self, X):
        out = np.empty(len(X))
        for s in range(0, len(X), 4000):
            Y = X[s:s + 4000] - self.c
            g = np.max(Y @ self.n.T / self.dk[None, :], axis=1)
            out[s:s + 4000] = self.E * np.clip(1 - g, 0, None)
        return out

    def forces(self, psi, delta, nu=240, nphi=48):
        a0, t = _axis(self.reach, psi, delta)
        e1 = np.array([-1.0, 0, 0])
        e2 = np.cross(t, e1)
        umax, phimax = 0.0125, 0.22
        u = (np.arange(nu) + 0.5) / nu * 2 * umax - umax
        ph = (np.arange(nphi) + 0.5) / nphi * 2 * phimax - phimax
        U, PH = np.meshgrid(u, ph, indexing="ij")
        nT = np.cos(PH)[..., None] * e1 + np.sin(PH)[..., None] * e2
        X = (a0 + U[..., None] * t + RC * nT).reshape(-1, 3)
        nT = nT.reshape(-1, 3)
        p = self.p(X)
        m = p > 0
        dA = (2 * umax / nu) * RC * (2 * phimax / nphi)
        out = _arm(X[m], p[m] * dA, -nT[m])
        return out + (float(m.sum() * dA),)


class Winkler:
    """p = E/h overlap on the analytic surface at 0.1 mm samples; h fixed (8.5 mm) or local (n . (x - c_V))."""

    def __init__(self, shape, E=E0, local=False):
        self.P, self.Nn, self.A = shape.surface(0.0001)
        self.reach = shape.reach
        h = (np.sum(self.Nn * (self.P - shape.centroid()), 1) if local else np.full(len(self.P), G.FOUNDATION))
        self.k = E / np.maximum(h, 1e-4) * self.A

    def forces(self, psi, delta):
        a0, t = _axis(self.reach, psi, delta)
        w = self.P - a0
        perp = w - np.outer(w @ t, t)
        dist = np.linalg.norm(perp, axis=1)
        ov = RC - dist
        m = ov > 0
        f = self.k[m] * ov[m]
        d = perp[m] / dist[m, None]
        Xc = a0 + np.outer(w[m] @ t, t) + d * RC
        return _arm(Xc, f, -d) + (float(self.A[m].sum()),)


class Pads:
    def __init__(self, shape, E=E0, each=False):
        C, K = shape.pads(E)
        self.C, self.K = C, (K if each else np.full(len(C), float(np.median(K))))
        self.reach = shape.reach

    def forces(self, psi, delta):
        a0, t = _axis(self.reach, psi, delta)
        w = self.C - a0
        perp = w - np.outer(w @ t, t)
        dist = np.linalg.norm(perp, axis=1)
        ov = RS - (dist - RC)
        m = ov > 0
        f = self.K[m] * ov[m]
        d = perp[m] / dist[m, None]
        Xc = a0 + np.outer(w[m] @ t, t) + d * RC
        return _arm(Xc, f, -d) + (float(m.sum()) * S * S,)


def solve(model, psi, N, tol=1e-4):
    """Approach delta with normal force N (secant on log F against log delta). Returns (delta, arm, rho90, n, area)."""
    d1, d2 = 30e-6, 90e-6
    f1, f2 = model.forces(psi, d1)[0], model.forces(psi, d2)[0]
    for _ in range(30):
        if f1 <= 0 or f2 <= 0 or abs(math.log(f2 / f1)) < 1e-12:
            d1, d2 = d1 * 1.6, d2 * 1.6
            f1, f2 = model.forces(psi, d1)[0], model.forces(psi, d2)[0]
            continue
        k = math.log(f2 / f1) / math.log(d2 / d1)
        d3 = d2 * (N / f2) ** (1 / k)
        d1, f1, d2 = d2, f2, d3
        f2 = model.forces(psi, d2)[0]
        if abs(f2 / N - 1) < tol:
            break
    _, arm, rho90, n, area = model.forces(psi, d2)
    return d2, arm, rho90, n, area


def brake(model, phi_deg, N0=0.5, it=8):
    """Quasi-static brake: the pinch force at which the gravity torque at swing angle phi equals 2 mu N arm."""
    N = N0
    for _ in range(it):
        _, arm, *_ = solve(model, math.radians(phi_deg), N)
        N = M_TOOL * GRAV * D_CG * math.cos(math.radians(phi_deg)) / (2 * MU * arm)
    return N, arm


def fields(shape, E=E0):
    return {"drake": DrakeField(shape, E), "winkler": Winkler(shape, E), "local": Winkler(shape, E, local=True),
            "pads": Pads(shape, E), "pads_each": Pads(shape, E, each=True)}


def law_rows(key, E=E0, psis=PSI_DEG, Ns=N_LIST, brake_phi=(10, 45, 80)):
    shape = shape_of(key)
    out = []
    for name, mdl in fields(shape, E).items():
        t0 = time.time()
        res = {}
        for N in Ns:
            for ps in psis:
                d, arm, rho90, n, area = solve(mdl, math.radians(ps), N)
                res[f"{N:g}|{ps}"] = {"delta_um": d * 1e6, "arm_mm": arm * 1e3, "rho90_mm": rho90 * 1e3, "n": n,
                                      "area_mm2": area * 1e6}
        spin = {}
        for N in Ns:                                  # the bed spin's window: the line at 23-69 deg
            spin[f"{N:g}"] = float(np.mean([solve(mdl, math.radians(ps), N)[1] for ps in (23, 46, 69)]) * 1e3)
        br = {f"N{p}": brake(mdl, p)[0] for p in brake_phi}
        out.append({"tip": key, "field": name, "E_Pa": E, "table": res, "spin_arm_mm": spin, "brake_N": br,
                    "script": SCRIPT, "wall_s": time.time() - t0, "when": time.strftime("%Y-%m-%d %H:%M")})
    return out


def onset_arm(key, N, field="drake", E=E0):
    """The law's twist-onset arm at psi 0 (m): the bed T2 ramp is scaled to it."""
    return solve(fields(shape_of(key), E)[field], 0.0, N)[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tips", nargs="+", required=True)
    ap.add_argument("--E", type=float, default=E0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="the scratch probes' values on tpu2.7")
    a = ap.parse_args()
    if a.check:
        F = fields(shape_of("tpu2.7"))
        print("drake arm psi 0, 1 N:", round(solve(F["drake"], 0.0, 1.0)[1] * 1e3, 3), "mm (2.554)")
        print("N45 drake/winkler/pads:", [round(brake(F[k], 45)[0], 3) for k in ("drake", "winkler", "pads")],
              "(0.373/0.364/0.393)")
        return
    have = set()
    if a.out.exists():
        have = {(r["tip"], r["field"], r["E_Pa"]) for r in map(json.loads, open(a.out))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for key in a.tips:
        rows = [r for r in law_rows(key, a.E) if (r["tip"], r["field"], r["E_Pa"]) not in have]
        with open(a.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
        for r in rows:
            t = r["table"]
            print(f"{key:8s} {r['field']:9s} delta(1 N) {t['1|0']['delta_um']:6.1f} um  arm psi0/45/90 at 1 N "
                  f"{t['1|0']['arm_mm']:.3f}/{t['1|45']['arm_mm']:.3f}/{t['1|90']['arm_mm']:.3f} mm  spin 1 N "
                  f"{r['spin_arm_mm']['1']:.3f} mm  N45 {r['brake_N']['N45']:.3f} N  ({r['wall_s']:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
