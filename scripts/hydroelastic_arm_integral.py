"""Friction arm of the hydroelastic pressure law on the contact bed's geometry, integrated directly with no simulator:
compliant fingertip sphere R = 10.55 mm with Drake's field p = E (1 - rho/R), rigid screwdriver cylinder r_c = 12.5 mm,
every surface point slipping under a rotation about the pinch axis. Reference for the bed's twist task (sliding arm).
Run: python3 scripts/hydroelastic_arm_integral.py
"""
import numpy as np
from scipy.optimize import brentq

E, R, RC, MU = 1e7, 10.55e-3, 12.5e-3, 1.0


def patch(delta, n=1500):
    D = RC + R - delta
    th_max = np.arccos(np.clip((RC**2 + D**2 - R**2) / (2 * RC * D), -1, 1)) * 1.02 + 1e-6
    y_max = np.sqrt(max(R**2 - (D - RC)**2, 0)) * 1.02 + 1e-9
    th = np.linspace(-th_max, th_max, n)
    y = np.linspace(-y_max, y_max, n)
    TH, Y = np.meshgrid(th, y, indexing="ij")
    P = np.stack([RC * np.cos(TH), Y, RC * np.sin(TH)], -1)
    rho = np.linalg.norm(P - np.array([D, 0, 0]), axis=-1)
    p = np.where(rho < R, E * (1 - rho / R), 0.0)
    dA = RC * (th[1] - th[0]) * (y[1] - y[0])
    nrm = np.stack([np.cos(TH), np.zeros_like(TH), np.sin(TH)], -1)
    N = np.sum(p * nrm[..., 0]) * dA
    r = P - np.array([RC, 0, 0])                       # lever from the patch centre on the pinch axis
    v = np.cross(np.array([1.0, 0, 0]), r)             # slip velocity of a rotation about the pinch axis
    vt = v - np.sum(v * nrm, -1, keepdims=True) * nrm
    t = -vt / np.maximum(np.linalg.norm(vt, axis=-1, keepdims=True), 1e-30)
    f = MU * p[..., None] * t * dA
    tau = -np.sum(np.cross(r, f)[..., 0])
    return N, tau / (MU * N)


for N in (0.5, 1.0, 3.0):
    d = brentq(lambda dl: patch(dl)[0] - N, 1e-7, 2e-3, xtol=1e-12)
    print(f"N {N:3.1f} N: penetration {d * 1e6:6.1f} um, continuum arm {patch(d)[1] * 1e3:.3f} mm")
# plate check against the closed form F = pi E delta^2, arm (8/15) a, a = sqrt(2 R delta)
d = np.sqrt(1.0 / (np.pi * E))
print(f"plate, 1 N closed form: arm {(8 / 15) * np.sqrt(2 * R * d) * 1e3:.3f} mm")
