#!/usr/bin/env python3
"""Pendulum mass properties to log with each BAM (Better Actuator Models) recording on this rig.

Identification uses ONE multi-hole arm (arm.stl, holes at 45 / 60 / 90 mm), mounted once per servo,
so BAM's single zero offset (q_offset, bam/model.py) stays valid across all recordings. The arm runs
past the weight to 96 mm, so it does NOT fit stock BAM's pendulum model (bam/testbench.py: a point
mass at the tip of a uniform rod that ends at the mass):
    gravity torque = (mass + arm_mass/2) * g * length * sin(q)
    inertia        = (mass + arm_mass/3) * length**2
Instead each log records the REAL values, which the planned patch to BAM's Pendulum and its MuJoCo
pendulum (bam/testbench_mujoco.py) uses directly:
    first_moment = m*L + G                 G = arm mass * arm center-of-mass distance
    inertia      = m*L^2 + I_arm + I_stack  I_stack = washers' own spin inertia (annulus)

For stock BAM (no patch) it also prints an equivalent (mass, arm_mass) pair that reproduces both
exactly in the fitting formulas:
    A = 6 (G L - I_arm - I_stack) / L^2,   M = m + G / L - A / 2
A comes out negative for this arm: fine in bam/testbench.py, NOT usable in bam/testbench_mujoco.py.

    python3 bam_rig_params.py --stack-g 80.3 [--hole 60] [--arm-g 10.8] [--washer-od 30]
    python3 bam_rig_params.py --stack-g 80.3 --arm slim60      (link 2 of the double pendulum)

Arm values are from the STLs at 100% PLA (1.24 g/cm^3). Pass the weighed arm mass with --arm-g;
the first moment and inertia scale with it (uniform infill, so the shape is unchanged).
"""
from __future__ import annotations

import argparse

# arm -> (mass g, CoM from the axis mm, inertia about the axis g*mm^2, holes mm)
ARMS = {
    "id": (11.0, 37.3, 24710.0, (45, 60, 90)),      # arm.stl, identification (3 holes)
    "id5": (10.8, 37.0, 24088.0, (30, 45, 60, 75, 90)),   # earlier 5-hole arm (as printed 2026-10-29)
    "slim60": (6.1, 23.7, 6571.0, (60,)),           # arm_60.stl
    "slim90": (9.5, 40.3, 24690.0, (90,)),          # single-hole 90 mm arm (double pendulum link 2 as built)
}
G_ACC = 9.80665
WASHER_ID_MM = 4.3


def props(stack_g: float, hole: float, arm: str, arm_g: float | None, washer_od: float):
    m0, com, izz, _ = ARMS[arm]
    k = (arm_g or m0) / m0
    G = k * m0 * com                                                    # g*mm
    I_arm = k * izz                                                     # g*mm^2
    I_stack = stack_g * (washer_od**2 + WASHER_ID_MM**2) / 8            # annulus about its own axis
    L = hole
    first = stack_g * L + G
    inertia = stack_g * L**2 + I_arm + I_stack
    A = 6 * (G * L - I_arm - I_stack) / L**2
    M = stack_g + G / L - A / 2
    return first, inertia, I_stack, M, A


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack-g", type=float, required=True, help="weighed bolt + washers + nut, grams")
    ap.add_argument("--arm", choices=ARMS, default="id", help="id = arm.stl (default), slim60 = link 2")
    ap.add_argument("--hole", type=float, help="hole distance, mm (default: all holes of the arm)")
    ap.add_argument("--arm-g", type=float, help="weighed printed arm mass, grams")
    ap.add_argument("--washer-od", type=float, default=30.0, help="washer outer diameter, mm")
    a = ap.parse_args()
    m0, _, _, holes = ARMS[a.arm]
    if a.hole and a.hole not in holes:
        ap.error(f"arm '{a.arm}' has holes {holes}")
    if a.arm_g and abs(a.arm_g / m0 - 1) > 0.05:
        print(f"note: weighed arm {a.arm_g} g is >5% off the computed {m0} g -- check infill")
    print(f"arm '{a.arm}', stack {a.stack_g} g, washers {a.washer_od} mm OD")
    print(f"{'hole':>5} | log: {'first moment':>14} {'inertia':>14} {'(stack spin)':>13} | "
          f"stock-BAM equivalent: --mass --arm-mass --length")
    for L in ([a.hole] if a.hole else holes):
        first, inertia, I_s, M, A = props(a.stack_g, L, a.arm, a.arm_g, a.washer_od)
        print(f"{L:>4.0f}mm | {first * 1e-6:>11.6f} kg·m {inertia * 1e-9:>11.4e} kg·m² {I_s * 1e-9:>10.2e} | "
              f"{M / 1000:.4f} {A / 1000:.4f} {L / 1000:.3f}"
              f"   (peak gravity torque {first * 1e-6 * G_ACC * 1000:.0f} mN·m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
