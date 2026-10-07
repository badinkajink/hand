#!/usr/bin/env python3
"""Contact references that need no simulator, for the bed's compliance tasks T8 (tangential load cycle) and T9 (normal
load sweep): the elastic solutions that the pressure law, the sphere pads and Drake hydroelastic leave out.

Geometry and material. The real_v1 fingertip sphere (R 10.55 mm) pressed on the screwdriver cylinder (r 12.5 mm), as on
the bed: principal relative radii R (along the tool axis) and R r / (R + r) = 5.72 mm (around it), represented by the
circular contact of the equivalent radius R_e = sqrt(R * 5.72 mm) = 7.77 mm. The tool is rigid. The tip is printed TPU,
whose modulus is not measured yet (owner item): E = 10 MPa, the value the pressure law and Drake use, and Poisson ratio
0.45 (TPU is nearly incompressible); G = E / (2 (1 + nu)), E* = E / (1 - nu^2). Friction mu = 1.

  Hertz       contact radius a = (3 N R_e / 4 E*)^(1/3), approach a^2 / R_e, pressure p0 sqrt(1 - r^2/a^2). Against the
              pressure law (hydroelastic, Winkler): approach ~ N^(1/2), radius ~ N^(1/4), and the load spread over the whole
              geometric overlap; Hertz's contact radius is 1/sqrt(2) of the overlap radius at its own approach.
  Mindlin     tangential force Q below slip on the Hertz contact (Cattaneo-Mindlin, Johnson 1985 Sec. 7.2): displacement
              d(Q) = C [1 - (1 - Q/mu N)^(2/3)], C = 3 mu N (2 - nu) / (16 G a); unloading and reloading by the
              Mindlin-Deresiewicz rule d_un(Q) = d(Q*) - 2 d((Q* - Q)/2); initial stiffness 8 G a / (2 - nu).
  Lubkin      torsion of the Hertz contact below spin (Lubkin 1951, Deresiewicz 1954), solved here on a grid of surface
              cells with Cerruti's half-space influence functions: stick cells take the twist of the rigid tool, cells
              whose traction would exceed mu p slip with |q| = mu p; initial stiffness 16 G a^3 / 3, full slip at
              3 pi mu N a / 16. The same solver run as a translation reproduces Mindlin's stiffness (self-check).

Bed task references (two pads in parallel, the tool force F = 2 Q): T8 cycles the axial force between +-F* = +-mu N, half
the slip force 2 mu N; T9 sweeps N.

    python3 scripts/contact_reference_laws.py            # tables at N = 0.5, 1, 3 N and the self-check
    python3 scripts/contact_reference_laws.py --json docs/experiments/<dir>/reference_laws.json
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

R_PAD, R_TOOL = 10.55e-3, 12.5e-3
E_TPU, NU_TPU, MU = 10e6, 0.45, 1.0
R_AROUND = R_PAD * R_TOOL / (R_PAD + R_TOOL)
R_E = math.sqrt(R_PAD * R_AROUND)


def mat(E=E_TPU, nu=NU_TPU):
    return dict(E=E, nu=nu, G=E / (2 * (1 + nu)), Es=E / (1 - nu ** 2))


def hertz(N, Re=R_E, E=E_TPU, nu=NU_TPU):
    m = mat(E, nu)
    a = (3 * N * Re / (4 * m["Es"])) ** (1 / 3)
    delta = a * a / Re
    p0 = 3 * N / (2 * math.pi * a * a)
    return dict(a=a, delta=delta, p0=p0, r_rms=math.sqrt(0.4) * a, overlap_radius=math.sqrt(2 * Re * delta))


# ------------------------------------------------------------------------------------------ Mindlin

def mindlin_mono(Q, N, a, G, nu, mu=MU):
    """Tangential displacement of the rigid tool on one pad under monotonic Q (Cattaneo-Mindlin)."""
    x = np.clip(np.asarray(Q, float) / (mu * N), -1.0, 1.0)
    C = 3 * mu * N * (2 - nu) / (16 * G * a)
    return np.sign(x) * C * (1 - (1 - np.abs(x)) ** (2 / 3))


def mindlin_cycle(N, frac=0.5, n=400, E=E_TPU, nu=NU_TPU, mu=MU):
    """T8 on the bed: tool force F cycled 0 -> F* -> -F* -> F* with F* = frac * 2 mu N (two pads, Q = F / 2 each).
    Returns the path (F, u) and the metrics."""
    m = mat(E, nu)
    a = hertz(N, E=E, nu=nu)["a"]
    Qs = frac * mu * N                                  # per pad: F* = frac * 2 mu N
    mono = lambda q: mindlin_mono(q, N, a, m["G"], nu, mu)  # noqa: E731
    q1 = np.linspace(0, Qs, n)
    u1 = mono(q1)
    q2 = np.linspace(Qs, -Qs, 2 * n)
    u2 = u1[-1] - 2 * mono((Qs - q2) / 2)              # unloading branch (Mindlin-Deresiewicz)
    q3 = np.linspace(-Qs, Qs, 2 * n)
    u3 = -u1[-1] + 2 * mono((Qs + q3) / 2)             # reloading branch
    F = np.concatenate([q1, q2, q3]) * 2
    u = np.concatenate([u1, u2, u3])
    loop = float(np.trapezoid(2 * q2, u2) + np.trapezoid(2 * q3, u3))   # \oint F du over one full cycle, J
    x = Qs / (mu * N)
    W_pad = 9 * mu ** 2 * N ** 2 * (2 - nu) / (10 * a * m["G"]) * (1 - (1 - x) ** (5 / 3) - 5 * x / 6 * (1 + (1 - x) ** (2 / 3)))
    u_peak = float(u1[-1])
    u_zero = float(u1[-1] - 2 * mono(Qs / 2))           # back at F = 0 on the unloading branch
    return dict(N=N, F_star=2 * Qs, a=a, k_t0_pad=8 * m["G"] * a / (2 - nu), u_presliding=u_peak, u_after_unload=u_zero,
                u_recovered=u_peak - u_zero, loop_area=abs(loop), loop_area_closed_form=2 * W_pad, F=F, u=u)


# ------------------------------------------------------------------------------------------ boundary-element solver

class HalfSpaceGrid:
    """Square cells of side h covering the Hertz contact disk of radius a; Cerruti influence of a uniform tangential
    traction on each cell on the tangential surface displacement at every cell centre (normal-tangential coupling
    neglected, Goodman's approximation)."""

    def __init__(self, a, G, nu, n=36):
        h = 2 * a / n
        c = (np.arange(n) + 0.5) * h - a
        X, Y = np.meshgrid(c, c, indexing="ij")
        keep = X ** 2 + Y ** 2 < a * a
        self.x, self.y, self.h, self.a = X[keep], Y[keep], h, a
        self.A = h * h
        dx = self.x[:, None] - self.x[None, :]
        dy = self.y[:, None] - self.y[None, :]
        r = np.hypot(dx, dy)
        np.fill_diagonal(r, 1.0)
        f = self.A / (2 * np.pi * G)
        Cxx = f * ((1 - nu) / r + nu * dx * dx / r ** 3)
        Cyy = f * ((1 - nu) / r + nu * dy * dy / r ** 3)
        Cxy = f * nu * dx * dy / r ** 3
        b = h / math.sqrt(math.pi)                       # self term: the cell as a disk of equal area
        s = b * (2 - nu) / (2 * G)
        np.fill_diagonal(Cxx, s)
        np.fill_diagonal(Cyy, s)
        np.fill_diagonal(Cxy, 0.0)
        n_ = len(self.x)
        self.C = np.block([[Cxx, Cxy], [Cxy, Cyy]])
        self.n = n_

    def solve(self, target, limit, iters=60):
        """Tractions q (2n) giving the displacement `target` (2n) on stick cells, with |q_i| <= limit_i; cells whose
        traction exceeds the limit slip with |q| = limit in the direction of their stick traction."""
        n = self.n
        slip = np.zeros(n, bool)
        qs = np.zeros(2 * n)
        for _ in range(iters):
            st = np.concatenate([~slip, ~slip])
            fixed = np.where(st, 0.0, qs)
            rhs = target[st] - self.C[np.ix_(st, ~st)] @ fixed[~st] if (~st).any() else target[st]
            q = fixed.copy()
            q[st] = np.linalg.solve(self.C[np.ix_(st, st)], rhs)
            mag = np.hypot(q[:n], q[n:])
            over = (~slip) & (mag > limit * (1 + 1e-9))
            if not over.any():
                return q, slip
            slip |= over
            dirn = np.stack([q[:n], q[n:]], 1) / np.maximum(mag, 1e-30)[:, None]
            qs[:n] = np.where(slip, limit * dirn[:, 0], 0.0)
            qs[n:] = np.where(slip, limit * dirn[:, 1], 0.0)
        return q, slip


def lubkin(N, n_beta=24, grid_n=36, E=E_TPU, nu=NU_TPU, mu=MU):
    """Torque against twist of the Hertz contact under a monotonic twist (Lubkin), by the cell solver."""
    m = mat(E, nu)
    hz = hertz(N, E=E, nu=nu)
    a = hz["a"]
    g = HalfSpaceGrid(a, m["G"], nu, grid_n)
    p = hz["p0"] * np.sqrt(np.clip(1 - (g.x ** 2 + g.y ** 2) / a ** 2, 0, None))
    k0 = 16 * m["G"] * a ** 3 / 3
    M_max = 3 * math.pi * mu * N * a / 16
    betas = M_max / k0 * np.geomspace(0.02, 6.0, n_beta)
    out = []
    for b in betas:
        tgt = np.concatenate([-b * g.y, b * g.x])
        q, slip = g.solve(tgt, mu * p)
        M = float(((g.x * q[g.n:] - g.y * q[:g.n]) * g.A).sum())
        out.append((float(b), M, float(slip.mean())))
    return dict(N=N, a=a, k_theta0=k0, M_full_slip=M_max, curve=out, cells=g.n)


def selfcheck(N=1.0, grid_n=36):
    """The cell solver as a no-slip translation (Mindlin's initial stiffness 8 G a / (2 - nu)) and twist (16 G a^3 / 3)."""
    m = mat()
    a = hertz(N)["a"]
    g = HalfSpaceGrid(a, m["G"], NU_TPU, grid_n)
    d = 1e-6
    q, _ = g.solve(np.concatenate([np.full(g.n, d), np.zeros(g.n)]), np.full(g.n, 1e9))
    kt = float(q[:g.n].sum() * g.A / d)
    b = 1e-4
    q, _ = g.solve(np.concatenate([-b * g.y, b * g.x]), np.full(g.n, 1e9))
    kth = float(((g.x * q[g.n:] - g.y * q[:g.n]) * g.A).sum() / b)
    return dict(k_t_cells=kt, k_t_closed=8 * m["G"] * a / (2 - NU_TPU), k_theta_cells=kth, k_theta_closed=16 * m["G"] * a ** 3 / 3,
                cells=g.n)


# ------------------------------------------------------------------------------------------ T9 law references

def t9_law(Ns=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Approach, force-weighted RMS patch radius and their exponents: the pressure law on the bed geometry
    (hom_contact_rig.winkler_sphere_cylinder) and Hertz on R_e."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hom_contact_rig as H
    rows = []
    for N in Ns:
        d, _, _ = H.winkler_law(N, 1e7)
        w = law_patch(d)
        hz = hertz(N)
        rows.append(dict(N=N, law_delta=d, law_r_rms=w["r_rms"], law_half_y=w["half_y"], law_half_z=w["half_z"],
                         hertz_delta=hz["delta"], hertz_a=hz["a"], hertz_r_rms=hz["r_rms"]))
    lN = np.log([r["N"] for r in rows])
    fit = lambda k: float(np.polyfit(lN, np.log([r[k] for r in rows]), 1)[0])  # noqa: E731
    return dict(rows=rows, law_delta_exp=fit("law_delta"), law_r_exp=fit("law_r_rms"), hertz_delta_exp=fit("hertz_delta"),
                hertz_r_exp=fit("hertz_r_rms"))


def law_patch(delta, E=1e7, n=401):
    """Pressure law on the sphere-cylinder patch at approach delta: force-weighted RMS distance of the patch from its
    centre (in the plane normal to the pinch axis) and the patch half-extents."""
    c = np.array([-(R_TOOL + R_PAD - delta), 0.0, 0.0])
    ha = min(math.pi / 2, 3.0 * math.sqrt(2 * delta * R_PAD) / R_TOOL)
    hy = 3.0 * math.sqrt(2 * delta * R_PAD)
    P, Y = np.meshgrid(np.linspace(math.pi - ha, math.pi + ha, n), np.linspace(-hy, hy, n), indexing="ij")
    X, Z = R_TOOL * np.cos(P), R_TOOL * np.sin(P)
    p = np.clip(R_PAD - np.sqrt((X - c[0]) ** 2 + Y ** 2 + Z ** 2), 0, None)
    w = p / max(p.sum(), 1e-30)
    m = p > 0
    return dict(r_rms=float(np.sqrt((w * (Y ** 2 + Z ** 2)).sum())), half_y=float(np.abs(Y[m]).max()) if m.any() else 0.0,
                half_z=float(np.abs(Z[m]).max()) if m.any() else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json")
    ap.add_argument("--grid", type=int, default=36)
    a = ap.parse_args()
    out = {"material": dict(E=E_TPU, nu=NU_TPU, mu=MU, R_pad=R_PAD, R_tool=R_TOOL, R_e=R_E), "selfcheck": selfcheck(grid_n=a.grid)}
    sc = out["selfcheck"]
    print(f"self-check ({sc['cells']} cells): translation {sc['k_t_cells']:.1f} vs {sc['k_t_closed']:.1f} N/m, "
          f"twist {sc['k_theta_cells'] * 1e3:.4f} vs {sc['k_theta_closed'] * 1e3:.4f} mN m/rad")
    out["hertz"], out["mindlin"], out["lubkin"] = {}, {}, {}
    for N in (0.5, 1.0, 3.0):
        hz = hertz(N)
        out["hertz"][str(N)] = hz
        mc = mindlin_cycle(N)
        out["mindlin"][str(N)] = {k: v for k, v in mc.items() if k not in ("F", "u")} | {"path_F": mc["F"][::20].tolist(),
                                                                                       "path_u": mc["u"][::20].tolist()}
        lb = lubkin(N, grid_n=a.grid)
        out["lubkin"][str(N)] = lb
        half = [c for c in lb["curve"] if c[1] <= 0.5 * lb["M_full_slip"]]
        print(f"N {N:3.1f} N: Hertz a {hz['a'] * 1e3:.3f} mm, approach {hz['delta'] * 1e6:.1f} um, p0 {hz['p0'] / 1e6:.3f} MPa | "
              f"T8 F* {mc['F_star']:.2f} N: presliding {mc['u_presliding'] * 1e6:.2f} um, recovered {mc['u_recovered'] * 1e6:.2f} um, "
              f"loop {mc['loop_area'] * 1e6:.3f} uJ (closed form {mc['loop_area_closed_form'] * 1e6:.3f}) | Lubkin k0 "
              f"{lb['k_theta0'] * 1e3:.4f} mN m/rad, full slip {lb['M_full_slip'] * 1e3:.3f} mN m, twist at half of it "
              f"{(half[-1][0] if half else float('nan')) * 180 / math.pi:.3f} deg")
    out["t9"] = t9_law()
    t = out["t9"]
    print(f"T9 exponents: approach law {t['law_delta_exp']:.3f} Hertz {t['hertz_delta_exp']:.3f}; RMS patch radius law "
          f"{t['law_r_exp']:.3f} Hertz {t['hertz_r_exp']:.3f}")
    if a.json:
        def conv(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(type(o))
        with open(a.json, "w") as fh:
            json.dump(out, fh, default=conv, indent=1)


if __name__ == "__main__":
    main()
