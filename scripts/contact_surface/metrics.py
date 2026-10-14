"""Forces on the object, world frame; moments about (0, 0, pad radius)."""

import numpy as np
import mujoco
from contact_surface.scaling import physical_coefficients
from contact_surface.runtime import mass_matrix, solver_diagonal


def contacts(model, data, parameters, time_value=0.0):
    rows = []
    jac = data.efc_J.reshape(data.nefc, model.nv)  # harness explicitly uses dense Jacobians
    mass = mass_matrix(model, data)
    inv_mass = np.linalg.inv(mass)
    for index in range(data.ncon):
        c = data.contact[index]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in c.geom]
        slot = next(
            (
                i
                for i, name in enumerate(names)
                if name
                and (
                    name.startswith("sample_")
                    or any(name.startswith(f"{f}_pad") for f in ("thumb", "index", "middle"))
                )
            ),
            None,
        )
        if slot is None or c.efc_address < 0:
            continue
        name = names[slot]
        if name.startswith("sample_"):
            sample, fingertip = int(name.split("_")[1]), "fixed_pad"
        else:
            if names[1 - slot] != "tool":
                continue
            fingertip, sample = name.split("_pad")
            sample = int(sample)
        local = (
            parameters["sample_parameters"][sample]
            if "sample_parameters" in parameters
            else parameters
        )
        adr = int(c.efc_address)
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        frame = np.array(c.frame).reshape(3, 3)
        # MuJoCo's force is on geom2. If the sample is geom2, reverse for the object.
        sign = 1 if slot == 0 else -1
        normal = sign * frame[0]
        fw = sign * frame.T @ force[:3]
        tw = sign * frame.T @ force[3:]
        tangential = fw - force[0] * normal
        velocity = data.efc_vel[adr : adr + 3]
        delta = max(0.0, -float(c.dist))
        expected = local["stiffness"] * delta
        actual_lambda = float(solver_diagonal(data)[adr])
        predicted_k, predicted_c = physical_coefficients(
            c.solref, c.solimp, actual_lambda, model.opt.timestep
        )
        row = dict(
            time=time_value,
            sample_id=sample,
            fingertip=fingertip,
            geom_sample=int(c.geom[slot]),
            geom_object=int(c.geom[1 - slot]),
            contact_id=index,
            constraint_address=adr,
            A_i=local["nominal_area"],
            A_corrected=local["corrected_area"],
            region=local.get("region", "spherical_cap"),
            k_i=local["stiffness"],
            signed_distance=float(c.dist),
            penetration=delta,
            f_n=float(force[0]),
            intended_force=expected,
            residual=float(force[0]) - expected,
            residual_relative=(float(force[0]) / expected - 1) if expected > 1e-9 else None,
            pressure_proxy=float(force[0]) / local["nominal_area"],
            Lambda_approx=actual_lambda,
            Lambda_solver=actual_lambda,
            Lambda_exact=float(jac[adr] @ inv_mass @ jac[adr]),
            stiffness_measured=float(force[0]) / delta if delta > 1e-12 else None,
            stiffness_predicted=predicted_k,
            damping_predicted=predicted_c,
            efc_R=float(data.efc_R[adr]),
            aref=float(data.efc_aref[adr]),
            acceleration=float(jac[adr] @ data.qacc),
            predicted_static_force=float(data.efc_aref[adr] / data.efc_R[adr]),
            slip_speed=float(np.linalg.norm(velocity[1:])),
            traction_utilization=float(
                np.linalg.norm(force[1:3]) / max(c.friction[0] * force[0], 1e-15)
            ),
            condim=int(c.dim),
        )
        for prefix, vector in [
            ("position", c.pos),
            ("normal", normal),
            ("force", fw),
            ("tangent", tangential),
            ("local_torque", tw),
            ("v_contact", velocity),
            ("solref", c.solref),
            ("solimp", c.solimp),
            ("friction", c.friction),
        ]:
            row.update({f"{prefix}_{i}": float(v) for i, v in enumerate(vector)})
        rows.append(row)
    return rows


def aggregate(rows, parameters):
    active = [r for r in rows if r["f_n"] > 1e-9]
    fn = sum(r["f_n"] for r in active)
    intended = sum(r["intended_force"] for r in rows)
    origin = np.array(parameters.get("wrench_origin", [0.0, 0.0, parameters["radius"]]))
    axis = np.array(parameters.get("contact_axis", [0.0, 0.0, 1.0]))
    wrench = np.zeros(6)
    tangential_force = np.zeros(3)
    cop = np.zeros(3)
    radial = 0.0
    for r in active:
        pos = np.array([r[f"position_{i}"] for i in range(3)])
        force = np.array([r[f"force_{i}"] for i in range(3)])
        torque = np.array([r[f"local_torque_{i}"] for i in range(3)])
        wrench[:3] += force
        tangential_force += np.array([r[f"tangent_{i}"] for i in range(3)])
        wrench[3:] += np.cross(pos - origin, force) + torque
        cop += r["f_n"] * pos
        dp = pos - origin
        radial += r["f_n"] * np.linalg.norm(dp - (dp @ axis) * axis)
    cop = cop / fn if fn > 1e-12 else np.full(3, np.nan)
    rel = [
        abs(r["residual_relative"])
        for r in rows
        if r["residual_relative"] is not None and r["intended_force"] > 1e-5
    ]
    weights = np.array([r["f_n"] for r in active])
    return dict(
        Fn_total=fn,
        Ft_total=float(np.linalg.norm(tangential_force)),
        intended_total=intended,
        aggregate_stiffness_error=fn / intended - 1 if intended > 1e-12 else None,
        contact_median_abs_error=float(np.median(rel)) if rel else None,
        contact_max_abs_error=max(rel) if rel else None,
        active_count=len(active),
        generated_count=len(rows),
        effective_contact_count=fn**2 / float(weights @ weights) if fn else 0.0,
        A_active_proxy=sum(r["A_i"] for r in active),
        pressure_weighted_arm=radial / fn if fn else None,
        r_eff_actual=abs(wrench[3:] @ axis) / fn if fn else None,
        cop_offset=float(np.linalg.norm(cop - origin - ((cop - origin) @ axis) * axis))
        if fn
        else None,
        **{f"CoP_{i}": float(v) if fn > 1e-12 else None for i, v in enumerate(cop)},
        **{f"wrench_{i}": float(v) for i, v in enumerate(wrench)},
    )
