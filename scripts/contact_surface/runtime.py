"""Dense audit adapter for current and pinned MuJoCo APIs.

Runtime translation changes native contact parameters before rebuilding the
constraint position/velocity stages. It never overwrites solved forces. The rig
uses one constant-impedance sample per contact. Articulated-hand benchmarks reuse
this adapter; the extra constraint rebuild is retained for auditability.
"""

import mujoco
import numpy as np
import time

from contact_surface.scaling import compensated_solref


def mass_matrix(model, data):
    result = np.zeros((model.nv, model.nv))
    if tuple(map(int, mujoco.__version__.split(".")[:2])) >= (3, 10):
        mujoco.mj_fullM(model, data, result)
    else:
        mujoco.mj_fullM(model, result, data.qM)
    return result


def solver_diagonal(data):
    return data.efc_diagA if hasattr(data, "efc_diagA") else data.efc_diagApprox


def translate_contacts(model, data, parameters, timings=None):
    """Use the diagonal actually selected by MuJoCo, then refresh all consumers."""
    if parameters.get("mapping_policy") != "runtime_selected_direct":
        return
    tick = time.perf_counter() if timings is not None else None
    diagonal = solver_diagonal(data).copy()
    stiffness_by_geom = parameters.get("sample_stiffness_by_geom")
    if stiffness_by_geom is not None:
        contact = data.contact
        stiffness = np.max(stiffness_by_geom[contact.geom[: data.ncon]], axis=1)
        address = contact.efc_address[: data.ncon]
        selected = np.flatnonzero((address >= 0) & (stiffness > 0))
        scale = (1 - contact.solimp[selected, 0]) * diagonal[address[selected]]
        normal = -stiffness[selected] * scale
        contact.solref[selected, 0] = normal
        contact.solref[selected, 1] = normal * parameters["relaxation"]
        if "tangent_damping_ratio" in parameters:
            contact.solreffriction[selected, 0] = 0.0
            contact.solreffriction[selected, 1] = (
                normal
                * parameters["relaxation"]
                * parameters["tangent_damping_ratio"]
                / model.opt.impratio
            )
    else:
        _translate_scalar(model, data, parameters, diagonal)
    if timings is not None:
        timings["scalar_translation"] = time.perf_counter() - tick
        tick = time.perf_counter()
    # Same order as mj_fwdPosition / mj_fwdVelocity, without rerunning collision
    # (which would overwrite the per-contact solref values).
    mujoco.mj_makeConstraint(model, data)
    mujoco.mj_island(model, data)
    mujoco.mj_projectConstraint(model, data)
    mujoco.mj_fwdVelocity(model, data)
    if timings is not None:
        timings["constraint_refresh"] = time.perf_counter() - tick


def _translate_scalar(model, data, parameters, diagonal):
    by_geom = parameters.get("sample_parameters_by_geom")
    samples = parameters.get("sample_parameters")
    for contact in data.contact[: data.ncon]:
        if contact.efc_address < 0:
            continue
        if by_geom is not None:
            local = next((by_geom[int(g)] for g in contact.geom if int(g) in by_geom), None)
            if local is None:
                continue
        else:
            name = next(
                (
                    model.geom(int(g)).name
                    for g in contact.geom
                    if model.geom(int(g)).name.startswith("sample_")
                ),
                None,
            )
            if name is None:
                continue
            sample = int(name.split("_")[1])
            local = samples[sample] if samples is not None else parameters
        contact.solref[:] = compensated_solref(
            local["stiffness"],
            parameters["relaxation"],
            contact.solimp[0],
            float(diagonal[contact.efc_address]),
        )
        if "tangent_damping_ratio" in parameters:
            # Elliptic isotropic contacts have R_t = R_n / impratio. Preserve
            # physical C_t while changing this solver regularizer ratio.
            contact.solreffriction[:] = [
                0.0,
                contact.solref[1] * parameters["tangent_damping_ratio"] / model.opt.impratio,
            ]


def step1(model, data, parameters):
    mujoco.mj_step1(model, data)
    translate_contacts(model, data, parameters)
