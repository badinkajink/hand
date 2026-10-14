"""Fixed pad, free rigid indenter. Parameters remain frozen across the mass sweep.

Displacement mode is a static inverse-dynamics fixture, checked by a forward solve
with the inferred supporting wrench. Load mode integrates native forward dynamics;
pose servos constrain the five non-normal coordinates, with no normal spring.
"""

import math
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from contact_surface.geometry import Surface
from contact_surface.scaling import compensated_solref
from contact_surface.runtime import step1

MASS = 500 * math.pi * 0.0125**2 * 0.1


def build(config):
    surface = Surface(spacing_mm=config["spacing_mm"], relaxation=config["relaxation"])
    p = surface.parameters()
    if "impedance_override" in config:
        d0 = float(config["impedance_override"])
        if not 0.0001 <= d0 <= 0.9999:
            raise ValueError("Impedance override outside MuJoCo's range")
        p["solimp"][:2] = [d0, d0]
    centers = surface.centers(config["phase_degrees"])
    normal, origin, quaternion = (
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, p["radius"]]),
        [1, 0, 0, 0],
    )
    if config.get("tip_shape") == "filleted_prism":
        from contact_surface.fillet import filleted_surface, support_point

        centers, p = filleted_surface(config["spacing_mm"], config["relaxation"])
        normal, origin, quaternion = support_point(
            p, config["tilt_degrees"], config["azimuth_degrees"]
        )
        if config["geometry"] != "plane" or config["mode"] != "displacement":
            raise ValueError("Fillet audit supports displacement against a tilted plane only")
    p.update(contact_axis=normal.tolist(), wrench_origin=origin.tolist())
    shape, radius = config["geometry"], config["object_radius_mm"] * 1e-3
    mass = MASS * config["mass_scale"]
    # Keep the same tool inertia reference when changing collision geometry: isolate curvature.
    # The cylinder's axis is local x (the collision geom rotates its native z axis).
    inertia = mass * np.array([0.0125**2 / 2] + [(3 * 0.0125**2 + 0.1**2) / 12] * 2)
    inertia *= config["inertia_scale"]
    if shape == "plane":
        geom, height = 'type="box" size=".1 .1 .005"', 0.005
    elif shape == "cylinder":
        geom, height = f'type="cylinder" size="{radius} .05" euler="0 90 0"', radius
    elif shape == "sphere":
        geom, height = f'type="sphere" size="{radius}"', radius
    else:
        raise ValueError(shape)

    def vstr(x):
        return " ".join(f"{v:.12g}" for v in x)

    def contact(i):
        local = p.get("sample_parameters", [p] * len(centers))[i]
        return (
            f'condim="3" friction="1 0 0" priority="1" '
            f'solref="{vstr(p["solref"])}" solimp="{vstr(local["solimp"])}"'
        )

    if config.get("redundant_count"):
        # Deliberately duplicate a single contact, for a rank-one solver diagnostic.
        # This is not a surface quadrature and is never used in physical pad sweeps.
        centers = np.tile([0, 0, p["radius"] - p["sphere_radius"]], (config["redundant_count"], 1))
    samples = "\n".join(
        f'<geom name="sample_{i}" type="sphere" pos="{vstr(pos)}" '
        f'size="{p["sphere_radius"]}" {contact(i)} contype="1" conaffinity="0"/>'
        for i, pos in enumerate(centers)
    )
    xml = f'''<mujoco model="sr2_foundation_audit">
    <option timestep="{config["timestep"]}" gravity="0 0 0" integrator="implicitfast"
      cone="elliptic" impratio="{config["impratio"]}" solver="Newton" jacobian="dense"
      iterations="{config["iterations"]}" ls_iterations="50" tolerance="{config["tolerance"]}"/>
    <worldbody>{samples}<body name="object" pos="{vstr(normal * (normal @ origin + height))}" quat="{vstr(quaternion)}">
    <freejoint/><inertial pos="0 0 0" mass="{mass}" diaginertia="{vstr(inertia)}"/>
    <geom name="object" {geom} contype="0" conaffinity="1" friction="0.3 0 0"
      solref=".006 1" solimp=".97 .995 .0005"/></body></worldbody></mujoco>'''
    if config.get("diagexact", False):
        tree = ET.fromstring(xml)
        ET.SubElement(tree.find("option"), "flag", diagexact="enable")
        xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    policy = config.get("mapping_policy", "frozen")
    if policy not in ("frozen", "lambda_compensated_direct", "runtime_selected_direct"):
        raise ValueError(f"Unknown mapping policy: {policy}")
    p["mapping_policy"] = policy
    p["diagexact"] = config.get("diagexact", False)
    if "tangent_damping_ratio" in config:
        p["tangent_damping_ratio"] = config["tangent_damping_ratio"]
    p["solref_nominal"] = p["solref"].copy()
    p["physical_damping_target"] = p["stiffness"] * p["relaxation"]
    if policy == "lambda_compensated_direct":
        # For this fixed-pad / free-body rig, each normal row uses the same
        # translational body_invweight0. This is not a general articulated adapter.
        inverse_inertia = float(model.body_invweight0[1, 0])
        tree = ET.fromstring(xml)
        for geom in tree.iter("geom"):
            if not geom.get("name", "").startswith("sample_"):
                continue
            i = int(geom.get("name").split("_")[1])
            local = p["sample_parameters"][i] if "sample_parameters" in p else p
            solref = compensated_solref(
                local["stiffness"], p["relaxation"], local["solimp"][0], inverse_inertia
            )
            geom.set("solref", vstr(solref))
            local["solref"] = solref
        xml = ET.tostring(tree, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    p.update(
        object_mass=mass,
        object_inertia=inertia.tolist(),
        actual_inverse_inertia=float(model.body_invweight0[1, 0]),
        contact_priority=1,
        zero_indentation_height=p["radius"] + height,
    )
    return model, data, p, xml


def runtime_static_fixture(model, data, p, indentation):
    """Native static inverse reaction with runtime contact coefficients."""
    data.qpos[:] = model.qpos0
    data.qpos[:3] -= indentation * np.array(p["contact_axis"])
    data.qvel[:] = 0
    data.qfrc_applied[:] = 0
    step1(model, data, p)
    data.qacc[:] = 0
    mujoco.mj_invConstraint(model, data)
    mujoco.mj_rne(model, data, 1, data.qfrc_inverse)
    data.qfrc_inverse[:] -= data.qfrc_passive + data.qfrc_constraint
    wrench = data.qfrc_inverse.copy()
    data.qfrc_applied[:] = wrench
    mujoco.mj_fwdActuation(model, data)
    mujoco.mj_fwdAcceleration(model, data)
    mujoco.mj_fwdConstraint(model, data)
    return wrench, float(np.max(np.abs(data.qacc)))


def static_fixture(model, data, p, indentation):
    """Solve static reaction and independently check forward force/acceleration residuals."""
    if p.get("mapping_policy") == "runtime_selected_direct":
        return runtime_static_fixture(model, data, p, indentation)
    data.qpos[:] = model.qpos0
    data.qpos[:3] -= indentation * np.array(p["contact_axis"])
    data.qvel[:] = 0
    data.qacc[:] = 0
    data.qfrc_applied[:] = 0
    mujoco.mj_inverse(model, data)
    supporting_wrench = data.qfrc_inverse.copy()
    data.qfrc_applied[:] = supporting_wrench
    data.qacc_warmstart[:] = 0
    mujoco.mj_forward(model, data)
    return supporting_wrench, float(np.max(np.abs(data.qacc)))


def load_step(model, data, p, load):
    # Ideal test fixture holds horizontal position and orientation. The normal force
    # is prescribed, and normal indentation remains a freely integrated coordinate.
    data.qfrc_applied[:] = 0
    mass = p["object_mass"]
    data.qfrc_applied[:2] = -500 * data.qpos[:2] - 2 * math.sqrt(500 * mass) * data.qvel[:2]
    data.qfrc_applied[2] = -load
    q = data.qpos[3:7].copy()
    if q[0] < 0:
        q *= -1
    inertia = np.array(p["object_inertia"])
    data.qfrc_applied[3:] = -0.02 * 2 * q[1:] - 2 * np.sqrt(0.02 * inertia) * data.qvel[3:]
    if p.get("mapping_policy") == "runtime_selected_direct":
        step1(model, data, p)
        mujoco.mj_step2(model, data)
    else:
        mujoco.mj_step(model, data)


def continuum(indentation, geometry, radius, p):
    """Independent continuum pressure quadrature on the object's contact surface.

    This is an analytic elastic-foundation reference, NOT a Drake solver result.
    Plane normal force also has exact F=pi E delta^2(1-2 delta/(3R)).
    """
    R, E = p["radius"], p["modulus"]
    if indentation <= 0:
        return dict(force=0.0, arm=0.0, area=0.0)
    span = 1.1 * math.sqrt(2 * R * indentation)
    grid = np.linspace(-span, span, 241)
    x, y = np.meshgrid(grid, grid)
    rho2 = x * x + y * y
    if geometry == "plane":
        z = np.full_like(x, R - indentation)
        jac = np.ones_like(x)
    else:
        r2 = y * y if geometry == "cylinder" else rho2
        h = np.sqrt(np.maximum(radius**2 - r2, 1e-15))
        z = R + radius - indentation - h
        jac = radius / h
    depth = np.maximum(R - np.sqrt(x * x + y * y + z * z), 0.0)
    pressure = E / R * depth
    area_cell = (grid[1] - grid[0]) ** 2
    force = float(pressure.sum() * area_cell)  # vertical projected force
    weights = pressure * jac
    arm = float((weights * np.sqrt(rho2)).sum() / max(weights.sum(), 1e-30))
    return dict(force=force, arm=arm, area=float(((depth > 0) * jac).sum() * area_cell))
