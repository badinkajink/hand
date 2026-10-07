"""Friction arm of the hydroelastic pressure law on the contact bed's geometry, integrated directly with no simulator:
compliant fingertip sphere R = 10.55 mm with Drake's field p = E (1 - rho/R) against a rigid tool, every surface point
slipping under a rotation about the pinch axis (world x, through the pad centre). Reference for the bed's twist task
(sliding arm) and static normal load.

Tools: the screwdriver cylinder (r_c = 12.5 mm, axis along y), and the square bar of the edge test (side 20 mm, axis
along y, an edge toward the pad: the two faces meeting at that edge are inclined 45 deg to the pinch axis). On each
face the friction traction opposes the tangential part of the slip velocity, so the torque about x on a face inclined
45 deg counts y / sqrt(2) of a point's lever along the edge.
Run: python3 scripts/hydroelastic_arm_integral.py
"""
import numpy as np
from scipy.optimize import brentq

E, R, RC, MU = 1e7, 10.55e-3, 12.5e-3, 1.0
BAR = 20e-3


def _forces(P, nrm, D, dA):
    """Normal force along +x on the tool, friction torque per mu about x, pressure-weighted distance from the pinch axis
    and patch extents, for surface points P (outward tool normals nrm) and the pad centre at x = -D."""
    c = np.array([-D, 0.0, 0.0])
    rho = np.linalg.norm(P - c, axis=-1)
    p = np.where(rho < R, E * (1 - rho / R), 0.0)
    N = -np.sum(p * nrm[..., 0]) * dA                  # the pad pushes along -n; its x component is the pinch force
    r = P.copy()
    r[..., 0] = 0.0                                    # lever about the pinch axis
    v = np.cross(np.array([1.0, 0, 0]), r)             # slip velocity of a rotation about the pinch axis
    vt = v - np.sum(v * nrm, -1, keepdims=True) * nrm
    t = -vt / np.maximum(np.linalg.norm(vt, axis=-1, keepdims=True), 1e-30)
    tau = -np.sum(np.cross(r, MU * p[..., None] * t * dA)[..., 0])
    w = p * dA
    ra = np.hypot(P[..., 1], P[..., 2])
    m = p > 0
    return dict(N=N, arm=tau / (MU * N) if N > 0 else 0.0, rbar=float((w * ra).sum() / max(w.sum(), 1e-30)),
                area=float(m.sum() * dA), half_y=float(np.abs(P[..., 1][m]).max()) if m.any() else 0.0,
                p_max=float(p.max()))


def patch(delta, n=1500):
    """Cylinder: (N, arm) at pad approach delta."""
    D = RC + R - delta
    th_max = np.arccos(np.clip((RC**2 + D**2 - R**2) / (2 * RC * D), -1, 1)) * 1.02 + 1e-6
    y_max = np.sqrt(max(R**2 - (D - RC)**2, 0)) * 1.02 + 1e-9
    th = np.linspace(-th_max, th_max, n)
    y = np.linspace(-y_max, y_max, n)
    TH, Y = np.meshgrid(th, y, indexing="ij")
    TH = TH + np.pi                                    # the -x side of the cylinder faces the pad
    P = np.stack([RC * np.cos(TH), Y, RC * np.sin(TH)], -1)
    nrm = np.stack([np.cos(TH), np.zeros_like(TH), np.sin(TH)], -1)
    f = _forces(P, nrm, D, RC * (th[1] - th[0]) * (y[1] - y[0]))
    return f["N"], f["arm"]


def bar_patch(delta, a=BAR, n=1200, full=False):
    """Square bar of side a with an edge toward the pad (vertex at x = -a/sqrt 2): the two faces meeting at that edge,
    integrated on an (s, y) grid, s the distance from the edge along the face."""
    h = a / np.sqrt(2)
    D = h + R - delta
    s_max = min(a, 1.05 * (-np.sqrt(2) * (R - delta) + np.sqrt(4 * R**2 - 2 * (R - delta)**2)) / 2 + 1e-9)
    y_max = np.sqrt(max(R**2 - (R - delta)**2, 0)) * 1.02 + 1e-9
    s = (np.arange(n) + 0.5) * s_max / n
    y = np.linspace(-y_max, y_max, n)
    S, Y = np.meshgrid(s, y, indexing="ij")
    k = np.sqrt(0.5)
    out = None
    for sz in (1.0, -1.0):
        P = np.stack([-h + S * k, Y, sz * S * k], -1)
        nrm = np.broadcast_to(np.array([-k, 0.0, sz * k]), P.shape)
        f = _forces(P, nrm, D, (s[1] - s[0]) * (y[1] - y[0]))
        if out is None:
            out = f
        else:
            tot = out["N"] + f["N"]
            out = dict(N=tot, arm=(out["arm"] * out["N"] + f["arm"] * f["N"]) / max(tot, 1e-30),
                       rbar=(out["rbar"] * out["N"] + f["rbar"] * f["N"]) / max(tot, 1e-30), area=out["area"] + f["area"],
                       half_y=max(out["half_y"], f["half_y"]), p_max=max(out["p_max"], f["p_max"]))
    return out if full else (out["N"], out["arm"])


def solve(N, fn=patch):
    """Pad approach delta at which the integral gives pinch force N, and the integral there."""
    d = brentq(lambda dl: fn(dl)[0] - N, 1e-7, 2e-3, xtol=1e-12)
    return d, fn(d)


def bar_law(N, a=BAR):
    d = brentq(lambda dl: bar_patch(dl, a)[0] - N, 1e-7, 2e-3, xtol=1e-12)
    return dict(delta=d, **bar_patch(d, a, full=True))


if __name__ == "__main__":
    for N in (0.5, 1.0, 3.0):
        d, (_, arm) = solve(N)
        print(f"N {N:3.1f} N: penetration {d * 1e6:6.1f} um, continuum arm {arm * 1e3:.3f} mm")
    # plate check against the closed form F = pi E delta^2, arm (8/15) a, a = sqrt(2 R delta)
    d = np.sqrt(1.0 / (np.pi * E))
    print(f"plate, 1 N closed form: arm {(8 / 15) * np.sqrt(2 * R * d) * 1e3:.3f} mm")
    for N in (0.5, 1.0, 3.0):
        b = bar_law(N)
        print(f"bar {BAR * 1e3:g} mm, edge, N {N:3.1f} N: penetration {b['delta'] * 1e6:6.1f} um, sliding arm "
              f"{b['arm'] * 1e3:.3f} mm, pressure-weighted distance from the axis {b['rbar'] * 1e3:.3f} mm, "
              f"patch half-length {b['half_y'] * 1e3:.2f} mm, area {b['area'] * 1e6:.2f} mm2, peak pressure {b['p_max'] / 1e6:.3f} MPa")
