"""What a Mini45 still measures with gauge 5 (G4) dead, for presses at the fingertip pad.

FT11452 has no calibration here, so the shape of its sensitivities comes from another Mini45's Net F/T matrix, FT32816
(SI-290-10; section 2.3 of the hand-off). A press is a force f at a known point r, so five gauges give
f = (A' B(r))^+ (g' - g0'), with A' the five working rows of inv(C) and B(r) = [I; [r]x].
Run: /usr/bin/python3 docs/experiments/20261023-ati_ft_calibration/lab/five_gauge_check.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# rows Fx Fy Fz Tx Ty Tz, columns G0..G5; one scale for N and N m (counts per force = counts per torque)
C = np.array([
    [244.899880, 122.665891, 504.513087, -14623.326502, -454.065068, 14271.427992],
    [-428.357757, 16563.783909, 762.394477, -8472.677462, 148.145679, -8204.225926],
    [20644.543735, 435.232780, 20899.360903, 575.880343, 19996.727260, 611.411431],
    [-5.875063, 115.768759, -324.671182, -69.795067, 329.989907, -46.923261],
    [379.943333, 5.531701, -201.100684, 99.854951, -186.969075, -106.578726],
    [2.191109, -209.681036, 11.557499, -211.700516, 8.172437, -209.489931],
])
A = np.linalg.inv(C)
A5 = A[[0, 1, 2, 3, 5]]        # gauge 5 = G4 is dead


def B(r):
    """Wrench (F, T) per component of a force at r (m)."""
    x, y, z = r
    return np.vstack([np.eye(3), [[0, -z, y], [z, 0, -x], [-y, x, 0]]])


def noise(M):
    return np.sqrt(np.diag(np.linalg.inv(M.T @ M)))


factory = np.sqrt((C[:3] ** 2).sum(1))      # force noise of F = C g from six gauges, per unit noise on each gauge

# the load five gauges cannot see
n = np.linalg.svd(A5)[2][-1]
nf, nm = n[:3], n[3:]
p = np.cross(nf, nm) / (nf @ nf)
print(f"invisible load: a force along {np.round(nf / np.linalg.norm(nf), 3)} (pitch {nf @ nm / (nf @ nf) * 1e3:.3f} mm) "
      f"on a line {np.hypot(*p[:2]) * 1e3:.1f} mm off the axis at {np.degrees(np.arctan2(p[1], p[0])):.0f} deg")

# the pad with the finger on the ATI's top: sph3t's registration, the stack's added length and clocking
reg = json.loads((HERE.parents[1] / "20261022-ur5_tip_calibration/lab/20261023-124052-sph3t/registration.json").read_text())
stack = json.loads((HERE.parent / "20261023-ati_stack_parts.json").read_text())
T_MA = np.array(stack["T_MA"])
p_M = np.array(reg["T_MP"])[:3, 3] + [0, 0, stack["added_length_mm"]]
r0 = T_MA[:3, :3].T @ (p_M - T_MA[:3, 3]) / 1000
print(f"pad apex in the ATI frame: {np.round(r0 * 1e3, 2)} mm")
print("force noise, five gauges against six read the factory way (Fx Fy Fz):")
print("  at the apex", np.round(noise(A5 @ B(r0)) / factory, 2))
Bt = np.hstack([B(r0), [[0], [0], [0], [0], [0], [1.0]]])   # a twist's torsion about the tool axis as a fourth unknown
print("  at the apex, with a torsion unknown", np.round(noise(A5 @ Bt)[:3] / factory, 2))
worst = np.zeros(3)
for rad in np.arange(0, 8.01, 0.5):
    for th in np.radians(np.arange(0, 360, 10)):
        worst = np.maximum(worst, noise(A5 @ B(r0 + [rad * np.cos(th) / 1e3, rad * np.sin(th) / 1e3, 0])) / factory)
print("  worst within 8 mm of the apex", np.round(worst, 2))

# errors in a 5 N press with 0.6 N of shear
f = np.array([0.5, 0.3, -5.0])
g = A5 @ B(r0) @ f
for k, ax in enumerate("xyz"):
    d = np.zeros(3)
    d[k] = 0.5e-3
    err = np.linalg.lstsq(A5 @ B(r0 + d), g, rcond=None)[0] - f
    print(f"contact point 0.5 mm off along {ax}: force error {np.round(err, 3)} N")
w = B(r0) @ f + [0, 0, 0, 1e-3, 0, 0]
err = np.linalg.lstsq(A5 @ B(r0), A5 @ w, rcond=None)[0] - f
print(f"a contact moment of 1 N mm about x left out: force error {np.round(err, 3)} N")

# the box's own Fz with gauge 5 frozen, as a stuck gauge gives it
for label, w in (("an on-axis push", np.r_[0, 0, -5.0, 0, 0, 0]), ("this press at the pad", B(r0) @ f)):
    g6 = A @ w
    g6[4] = 0.0
    print(f"{label}: Fz {w[2]:.2f} N reads {(C @ g6)[2]:.2f} N with gauge 5 frozen")
