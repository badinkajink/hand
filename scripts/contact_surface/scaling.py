"""Physical normal coefficients from MuJoCo's constant-impedance parameterization.

Lambda is the compiler's constraint-space diagonal approximation, not J M^-1 J^T.
These are equilibrium/interior-cone coefficients; acceleration and friction-cone
boundary terms remain in the joint constraint solve.
"""

import math


def physical_coefficients(solref, solimp, inverse_inertia, timestep):
    if not math.isclose(float(solimp[0]), float(solimp[1]), rel_tol=1e-12):
        raise ValueError("This diagnostic requires constant impedance")
    d0 = float(solimp[0])
    denominator = (1 - d0) * inverse_inertia
    if not 0 < d0 < 1 or inverse_inertia <= 0:
        raise ValueError("Impedance and inverse inertia must be positive")
    if solref[0] > 0 and solref[1] > 0:
        tc = max(float(solref[0]), 2 * timestep)  # default refsafe flag is enabled
        return (1 / (tc**2 * solref[1] ** 2 * denominator), 2 / (tc * denominator))
    if solref[0] < 0 and solref[1] <= 0:
        return -float(solref[0]) / denominator, -float(solref[1]) / denominator
    raise ValueError("Unsupported solref format")


def compensated_solref(stiffness, relaxation, impedance, inverse_inertia):
    """Set fixed physical k and c=k*relaxation using native direct-format solref.

    Geometry/material targets stay fixed. Only their analytic translation to the
    simulator changes with its known inverse-inertia approximation. No measured
    force, reference backend, load, or trajectory enters this function.
    """
    if stiffness <= 0 or relaxation <= 0 or inverse_inertia <= 0 or not 0 < impedance < 1:
        raise ValueError("Invalid physical coefficient or constraint scaling")
    scale = (1 - impedance) * inverse_inertia
    return [-stiffness * scale, -stiffness * relaxation * scale]
