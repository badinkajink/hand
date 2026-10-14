"""Force-controlled sliding and normal acceleration in the isolated contact fixture.

The force-ramp protocol has no prescribed object trajectory or measured-force
feedback: normal and tangential forces are fixed across policies. A guided-path
protocol also follows the nominal curved surface over 3 mm, testing local slip
before the object can leave a small-radius fixture. The cylinder axis is x.
"""

import math
import time

import mujoco
import numpy as np

from contact_surface.metrics import aggregate, contacts
from contact_surface.mujoco_model import build
from contact_surface.records import CsvLog, write_json
from contact_surface.runtime import mass_matrix, step1


def smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def command(t, scenario):
    if scenario == "guided_slip":
        return 1.0, 0.0, "settle" if t < 0.4 else "guided_slide" if t < 0.9 else "guided_hold"
    if scenario == "normal_pulse":
        pulse = smooth((t - 0.4) / 0.02) - smooth((t - 0.65) / 0.02)
        return 1 + 0.5 * pulse, 0.0, "normal_acceleration" if 0.4 <= t < 0.67 else "hold"
    if t < 0.4:
        return 1.0, 0.0, "settle"
    if t < 0.65:
        return 1.0, 1.2 * smooth((t - 0.4) / 0.25), "force_ramp"
    if t < 0.7:
        return 1.0, 1.2, "slip"
    if t < 0.9:
        return 1.0, 1.2 * (1 - smooth((t - 0.7) / 0.2)), "brake"
    return 1.0, 0.0, "restick"


def apply_wrench(data, parameters, config, t):
    """Shared physical fixture input for audited runs and state-only filming."""
    load, tangent, state = command(t, config["scenario"])
    slide = 0 if config.get("slide_axis", "x") == "x" else 1
    held = 1 - slide
    if config["scenario"] == "guided_slip":
        target = 0.003 * smooth((t - 0.4) / 0.5)
        tangent = 1000.0 * (target - data.qpos[slide])
    data.qfrc_applied[:] = 0
    data.qfrc_applied[slide] = tangent - config["tangent_fixture_damping"] * data.qvel[slide]
    data.qfrc_applied[held] = (
        -500 * data.qpos[held] - 2 * math.sqrt(500 * parameters["object_mass"]) * data.qvel[held]
    )
    data.qfrc_applied[2] = -load
    if config["scenario"] == "guided_slip":
        shape = config["geometry"]
        crosses_curvature = shape == "sphere" or (shape == "cylinder" and slide == 1)
        surface_drop = 0.0
        if crosses_curvature:
            radius = config["object_radius_mm"] * 1e-3
            distance = min(abs(data.qpos[slide]), radius)
            surface_drop = math.sqrt(max(0.0, radius * radius - distance * distance)) - radius
        height_target = parameters["zero_indentation_height"] + surface_drop
        data.qfrc_applied[2] += 1000.0 * (height_target - data.qpos[2]) - 5.0 * data.qvel[2]
    q = data.qpos[3:7].copy()
    if q[0] < 0:
        q *= -1
    kr = 0.5
    data.qfrc_applied[3:] = (
        -kr * 2 * q[1:] - 2 * np.sqrt(kr * np.array(parameters["object_inertia"])) * data.qvel[3:]
    )
    return load, tangent, state


def run(config, out):
    m, d, p, xml = build(config)
    (out / "model.xml").write_text(xml)
    write_json(out / "parameters.json", p)
    dt = m.opt.timestep
    slide = 0 if config.get("slide_axis", "x") == "x" else 1
    d.qpos[2] -= 1e-5
    history, latencies = [], []
    residuals, algebraic, cone_terms, acceleration_terms = [], [], [], []
    snapshot_times = [0.39, 0.61, 0.69, 0.85, 1.29]
    snapshots = []
    full_contact_logging = config.get("contact_log_stride", 1)
    with CsvLog(out / "contacts.csv.gz") as raw, CsvLog(out / "timeseries.csv") as timeline:
        for k in range(round(config["duration"] / dt)):
            t = k * dt
            load, tangent, state = apply_wrench(d, p, config, t)
            pre_pos, pre_vel = d.qpos.copy(), d.qvel.copy()
            started = time.perf_counter()
            step1(m, d, p)
            mujoco.mj_step2(m, d)
            latencies.append((time.perf_counter() - started) * 1000)
            rows = contacts(m, d, p, t)
            normal_expected = normal_acceleration = normal_cone = absolute_expected = 0.0
            max_translation_error = 0.0
            for row in rows:
                # Separation velocity is positive. Compression damping increases force.
                expected = (
                    row["k_i"] * row["penetration"]
                    - p["physical_damping_target"] * row["v_contact_0"]
                )
                target = max(0.0, expected)
                acceleration = -row["acceleration"] / row["efc_R"]
                interior = row["predicted_static_force"] + acceleration
                cone = row["f_n"] - interior
                row.update(
                    physical_dynamic_force=expected,
                    physical_unilateral_force=target,
                    dynamic_residual=row["f_n"] - target,
                    acceleration_force_correction=acceleration,
                    cone_force_correction=cone,
                    interior_force_prediction=interior,
                    coefficient_relative_error=row["stiffness_predicted"] / row["k_i"] - 1,
                )
                # Sum absolute per-point errors; cancellations must not look like fidelity.
                absolute_expected += target
                normal_expected += expected
                normal_acceleration += abs(acceleration)
                normal_cone += abs(cone)
                max_translation_error = max(
                    max_translation_error, abs(row["coefficient_relative_error"])
                )
                if k % full_contact_logging == 0:
                    raw.row(row)
                if t >= 0.35:
                    residuals.append((abs(row["f_n"] - target), target))
                    acceleration_terms.append((abs(acceleration), target))
                    cone_terms.append((abs(cone), target))
                    if row["traction_utilization"] < 0.98 and row["f_n"] > 1e-5:
                        algebraic.append(abs(row["f_n"] - interior))
            total = aggregate(rows, p)
            abs_error = sum(abs(r["dynamic_residual"]) for r in rows)
            slip = max((r["slip_speed"] for r in rows if r["f_n"] > 1e-5), default=0.0)
            record = dict(
                time=t,
                state=state,
                normal_command=load,
                tangent_command=tangent,
                tangent_applied=float(tangent - config["tangent_fixture_damping"] * pre_vel[slide]),
                x=float(pre_pos[slide]),
                vx=float(pre_vel[slide]),
                z=float(pre_pos[2]),
                vz=float(pre_vel[2]),
                ax=float(d.qacc[slide]),
                az=float(d.qacc[2]),
                tilt_radians=float(2 * np.arccos(np.clip(abs(pre_pos[3]), 0, 1))),
                physical_dynamic_force_sum=normal_expected,
                normal_force_relative_error=total["Fn_total"] / absolute_expected - 1
                if absolute_expected > 1e-9
                else None,
                local_force_L1_relative_error=abs_error / absolute_expected
                if absolute_expected > 1e-9
                else None,
                acceleration_correction_L1=normal_acceleration,
                cone_correction_L1=normal_cone,
                max_coefficient_relative_error=max_translation_error,
                slip_speed_max=slip,
                traction_utilization_max=max(
                    (r["traction_utilization"] for r in rows), default=0.0
                ),
                **total,
            )
            timeline.row(record)
            history.append(record)
            if snapshot_times and t >= snapshot_times[0] - dt / 2:
                snapshot_times.pop(0)
                jac = d.efc_J.reshape(d.nefc, m.nv)
                indices = [r["constraint_address"] for r in rows]
                Jn = jac[indices]
                A = Jn @ np.linalg.solve(mass_matrix(m, d), Jn.T)
                snapshots.append(
                    dict(
                        time=t,
                        sample_ids=[r["sample_id"] for r in rows],
                        A_normal=A.tolist(),
                        R_normal=[r["efc_R"] for r in rows],
                        Lambda_solver=[r["Lambda_solver"] for r in rows],
                        normal_rank=int(np.linalg.matrix_rank(Jn)) if indices else 0,
                    )
                )
            if (
                not np.isfinite(d.qpos).all()
                or np.linalg.norm(d.qpos[:2]) > 0.04
                or abs(d.qpos[2] - m.qpos0[2]) > 0.01
            ):
                raise RuntimeError("State diverged or left the declared contact fixture")
    write_json(out / "coupling_snapshots.json", snapshots)
    evaluated = [r for r in history if r["time"] >= 0.35]
    sliding = [r for r in evaluated if r["slip_speed_max"] > 0.001]
    sliding_errors = [
        r["local_force_L1_relative_error"]
        for r in sliding
        if r["local_force_L1_relative_error"] is not None
    ]
    onset = [r for r in evaluated if r["time"] >= 0.4 and abs(r["vx"]) > 0.001]
    cone_onset = [r for r in onset if r["traction_utilization_max"] > 0.99]

    def weighted_error(values):
        denominator = sum(v[1] for v in values)
        return float(sum(v[0] for v in values) / denominator) if denominator > 0 else None

    final = history[-1]
    return dict(
        status="complete",
        engine=mujoco.__version__,
        normal_law_L1_error=weighted_error(residuals),
        valid_contact_law_comparison=sum(v[1] for v in residuals) > 0,
        acceleration_term_L1=weighted_error(acceleration_terms),
        cone_term_L1=weighted_error(cone_terms),
        interior_equation_max_force_error=max(algebraic, default=None),
        coefficient_max_relative_error=max(r["max_coefficient_relative_error"] for r in history),
        slip_onset_time=onset[0]["time"] if onset else None,
        slip_onset_tangent_force=onset[0]["tangent_command"] if onset else None,
        cone_onset_time=cone_onset[0]["time"] if cone_onset else None,
        cone_onset_tangent_force=cone_onset[0]["tangent_applied"] if cone_onset else None,
        slip_distance_mm=(final["x"] - history[round(0.35 / dt)]["x"]) * 1000,
        peak_velocity=max(abs(r["vx"]) for r in evaluated),
        final_velocity=final["vx"],
        restuck=abs(final["vx"]) < 0.001,
        sliding_frames=len(sliding),
        sliding_normal_law_L1_median=float(np.median(sliding_errors)) if sliding_errors else None,
        frames_with_zero_dynamic_target=sum(
            r["local_force_L1_relative_error"] is None for r in evaluated
        ),
        final_normal_force=final["Fn_total"],
        peak_normal_acceleration=max(abs(r["az"]) for r in evaluated),
        peak_tilt_degrees=math.degrees(max(r["tilt_radians"] for r in evaluated)),
        max_active_contacts=max(r["active_count"] for r in history),
        sample_count=p["count"],
        physics_step_ms=float(np.median(latencies[50:])),
        warnings=[
            dict(number=int(w.number), lastinfo=int(w.lastinfo)) for w in d.warning if w.number
        ],
    )
