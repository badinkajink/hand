#!/usr/bin/env python3
"""Two-pad pinch rig in Drake and MuJoCo: which MuJoCo contact reproduces the hydroelastic pinch.

Wang, Oh and Pollard (arXiv 2609.25619) load a screwdriver by loosening a two-finger pinch so the
pinch acts as a hinge whose friction torque brakes the gravity swing, and they chose Drake's
hydroelastic contact because a point contact has no friction torque. This rig isolates that hinge.
Two pads (spheres of the real_v1 fingertip radius, 10.55 mm) slide on force-controlled rails along
the pinch axis (world x) and squeeze the real_v1 screwdriver (cylinder r 12.5 mm, 100 mm, 24.5 g)
whose axis starts along world y with its centre of mass `d` from the pinch line. Geometry, masses,
friction (mu 1), rail damping and the 1 ms step are identical in both simulators.

Contact models (spec strings):
  drake:hydro:E<Pa>:r<mm>   compliant hydroelastic pads (modulus E, tet resolution r), rigid
                            hydroelastic tool, SAP. The paper's model.
  drake:point               Drake point contact, SAP.
  mj:point3                 MuJoCo, one contact per pad, condim 3 (no friction torque).
  mj:point4:mt<m>           condim 4, constant torsional coefficient mu_t [m].
  mj:point4s                condim 4, mu_t rescheduled every step from the pad's normal force
                            with the hydroelastic law fitted on this rig (FIT below).
  mj:spheres:s<mm>:rs<mm>   each pad is a cap of small spheres (spacing s, radius rs) whose
                            envelope is the fingertip sphere; condim 3 per sphere; per-sphere
                            stiffness set from E so the cap is the same elastic foundation as
                            Drake's pressure field p = E * depth / R (no torsion fitted).
Options common to MuJoCo: timestep 1 ms, implicitfast, Newton, elliptic cone, impratio
(`:ir<k>`, default 10, the calibrated plant's value).

Experiments (each writes one JSON row; traces with --traces):
  torsion  gravity off, pinch N per pad, drive the tool's spin about the pinch axis at omega and
           read the steady friction torque.
  brake    gravity on, CG offset d, pinch held at N0, then lowered geometrically to N1 over T s;
           records the swing angle phi (0 horizontal, 90 hanging) against N.
  cop      kinematic: one pad at fixed penetration rotated about its own centre; centre of
           pressure against the analytic contact point (the GCF origin of the paper).

    PY=logs/20261001-hom_contact/venv/bin/python     # drake 1.57.0 + mujoco 3.6.0
    $PY scripts/hom_contact_rig.py torsion --spec drake:hydro:E1e7:r0.5 --N 1
    $PY scripts/hom_contact_rig.py brake --spec mj:spheres:s0.75:rs1.5 --d 15 --traces DIR
    $PY scripts/hom_contact_rig.py study --out docs/experiments/20261001-hom_contact_patch
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

R_PAD = 0.01055                      # real_v1 fingertip sphere (assets/mjcf/real_v1/real_hand.xml)
R_TOOL = 0.0125                      # real_v1 screwdriver (assets/objects/screwdriver_medium.xml)
HL_TOOL = 0.05
M_TOOL = 500.0 * math.pi * R_TOOL ** 2 * 2 * HL_TOOL      # density 500 as in the scene: 24.5 g
I_TOOL_T = M_TOOL * (3 * R_TOOL ** 2 + (2 * HL_TOOL) ** 2) / 12.0   # transverse, about the CG
I_TOOL_A = 0.5 * M_TOOL * R_TOOL ** 2
M_PAD = 0.02
RAIL_DAMP = 5.0                      # N s/m on each pad rail
MU = 1.0
G = 9.81
DT = 1e-3
X0 = R_TOOL + R_PAD                  # pad centre on the rail at first touch

# Fitted on this rig from Drake hydro E1e7 r0.5 torsion rows (see `study`): torsional arm
# rbar(N) = FIT_C * N ** FIT_P per pad, so tau = mu * N * rbar(N). Filled after the first fit;
# the analytic sphere-on-flat law is rbar = (8/15) sqrt(2 R) (N / pi E) ** 0.25.
FIT_C = None
FIT_P = 0.25


def parse_spec(spec: str) -> dict:
    parts = spec.split(":")
    out = {"spec": spec, "sim": parts[0], "model": parts[1], "ir": 10.0}
    for p in parts[2:]:
        if p.startswith("E"):
            out["E"] = float(p[1:])
        elif p.startswith("rs"):
            out["rs"] = float(p[2:]) * 1e-3
        elif p.startswith("rt"):
            out["rt"] = float(p[2:])
        elif p.startswith("tr"):
            out["tr"] = float(p[2:])
        elif p.startswith("r"):
            out["res"] = float(p[1:]) * 1e-3
        elif p.startswith("mt"):
            out["mu_t"] = float(p[2:])
        elif p.startswith("s"):
            out["s"] = float(p[1:]) * 1e-3
        elif p.startswith("ir"):
            out["ir"] = float(p[2:])
        elif p.startswith("cap"):
            out["cap_deg"] = float(p[3:])
        elif p.startswith("dt"):
            out["dt"] = float(p[2:]) * 1e-3
        elif p.startswith("fit"):
            c, pw = p[3:].split("x")
            out["fit_c"], out["fit_p"] = float(c), float(pw)
        else:
            raise ValueError(f"unknown spec field {p!r} in {spec}")
    out.setdefault("E", 1e7)
    return out


def rbar_sphere_flat(N, E, R=R_PAD):
    """Pressure-weighted torsion arm of a Winkler sphere on a flat: (8/15) a, a = sqrt(2 R delta)."""
    delta = math.sqrt(max(N, 0.0) / (math.pi * E))
    return 8.0 / 15.0 * math.sqrt(2 * R * delta)


def winkler_sphere_cylinder(delta, E, R=R_PAD, Rc=R_TOOL, n=241):
    """Elastic-foundation pad on the tool cylinder, by quadrature over the cylinder surface.

    The pad centre sits at x = -(Rc + R - delta) on the line through the cylinder axis; depth of a
    surface point into the pad sphere is R - |p - c|, pressure E * depth / R. Returns normal force,
    pressure-weighted torsion arm about the pinch axis, and the patch half-extents."""
    c = np.array([-(Rc + R - delta), 0.0, 0.0])
    half_ang = min(math.pi / 2, 3.0 * math.sqrt(2 * delta * R) / Rc)
    half_ax = 3.0 * math.sqrt(2 * delta * R)
    ps = np.linspace(math.pi - half_ang, math.pi + half_ang, n)
    ys = np.linspace(-half_ax, half_ax, n)
    P, Y = np.meshgrid(ps, ys, indexing="ij")
    X = Rc * np.cos(P)
    Z = Rc * np.sin(P)
    dist = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
    depth = np.clip(R - dist, 0.0, None)
    p = E * depth / R
    dA = Rc * (ps[1] - ps[0]) * (ys[1] - ys[0])
    nx = -np.cos(P)                               # inward normal of the cylinder seen by the pad
    Fn = float((p * nx).sum() * dA)
    rho = np.sqrt(Y ** 2 + Z ** 2)                # distance from the pinch axis (world x)
    arm = float((p * rho).sum() * dA / max((p).sum() * dA, 1e-30))
    m = depth > 0
    return {"Fn": Fn, "rbar": arm, "half_y": float(np.abs(Y[m]).max()) if m.any() else 0.0,
            "half_z": float(np.abs(Z[m]).max()) if m.any() else 0.0}


def winkler_law(N, E):
    """Invert winkler_sphere_cylinder for force N: returns (delta, rbar)."""
    lo, hi = 0.0, 5e-3
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if winkler_sphere_cylinder(mid, E, n=121)["Fn"] < N:
            lo = mid
        else:
            hi = mid
    w = winkler_sphere_cylinder(0.5 * (lo + hi), E)
    return 0.5 * (lo + hi), w["rbar"], w


def fib_cap(n, cap):
    """n near-uniform unit vectors on the cap of half-angle `cap` about +z (Fibonacci lattice)."""
    k = np.arange(n) + 0.5
    z = 1.0 - (1.0 - math.cos(cap)) * k / n
    r = np.sqrt(1.0 - z ** 2)
    phi = k * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


# ------------------------------------------------------------------------------------- MuJoCo

def _rot_z_to(v):
    """Rotation matrix taking +z to unit vector v."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    ax = np.cross(z, v)
    s = np.linalg.norm(ax)
    ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def mj_xml(sp: dict, d_cg: float, gravity: bool, kinematic: bool = False,
           pad_stiff: float | None = None, d0: float = 0.95, tau_d: float = 0.02) -> tuple[str, dict]:
    """MJCF for the rig. Returns (xml, info). `pad_stiff` is the solref stiffness for sphere pads."""
    model = sp["model"]
    g = "0 0 -9.81" if gravity else "0 0 0"
    info = {}
    if model in ("point3", "point4", "point4s"):
        condim = 3 if model == "point3" else 4
        mu_t = sp.get("mu_t", 0.0) if model == "point4" else (0.0 if model == "point3" else 1e-3)
        contact = (f'condim="{condim}" friction="{MU} {mu_t} 0" solref="0.006 1" '
                   f'solimp="0.97 0.995 0.0005"')
        tool_contact = (f'condim="{condim}" friction="{MU} 0 0" solref="0.006 1" '
                        f'solimp="0.97 0.995 0.0005"')
        pad_geoms = {s: f'<geom name="pad{s}" type="sphere" size="{R_PAD}" contype="1" conaffinity="0" '
                        f'{contact} rgba="0.85 0.55 0.35 1"/>' for s in "LR"}
    elif model == "spheres":
        s, rs = sp["s"], sp["rs"]
        cap = math.radians(sp.get("cap_deg", 35.0))
        area = 2 * math.pi * R_PAD ** 2 * (1 - math.cos(cap))
        n = max(1, int(round(area / s ** 2)))
        dirs = fib_cap(n, cap)
        A_s = area / n
        # N/m per sphere: the same elastic foundation as Drake's p = E * depth / R. The contact
        # points lie on the inner sphere R - rs, (R / (R - rs))^2 more densely than on the
        # envelope, so the per-sphere area is scaled down by that factor.
        K_s = sp["E"] / R_PAD * A_s * ((R_PAD - rs) / R_PAD) ** 2
        info.update(n_spheres=n, area_per_sphere_mm2=A_s * 1e6, K_sphere=K_s)
        # Critically damped reference dynamics: solref (tc, 1) with tc = 1/sqrt(K_s (1-d0) diag)
        # gives the per-contact steady stiffness K_s = 1 / (tc^2 (1-d0) diag). With hundreds of
        # contacts on one body the transient follows aref, so its damping ratio must be set
        # there (a direct-stiffness solref with damping ~ 0.02 s * stiffness rang at zeta 0.33).
        tc, d0 = pad_stiff if pad_stiff is not None else (0.02, d0)
        c = (f'condim="3" friction="{MU} 0 0" solref="{tc:.6g} 1" '
             f'solimp="{d0:.6g} {d0:.6g} 0.001 0.5 2"')
        tool_contact = c
        pad_geoms = {}
        for side, face in (("L", [1.0, 0, 0]), ("R", [-1.0, 0, 0])):
            Rm = _rot_z_to(face)
            lines = [f'<geom name="pad{side}_vis" type="sphere" size="{R_PAD}" contype="0" conaffinity="0" '
                     f'group="1" rgba="0.85 0.55 0.35 0.35"/>']
            for i, dv in enumerate(dirs @ Rm.T):
                p = (R_PAD - rs) * dv
                lines.append(f'<geom name="pad{side}_s{i}" type="sphere" size="{rs}" '
                             f'pos="{p[0]:.7f} {p[1]:.7f} {p[2]:.7f}" contype="1" conaffinity="0" '
                             f'{c} rgba="0.75 0.4 0.25 1"/>')
            pad_geoms[side] = "\n        ".join(lines)
    else:
        raise ValueError(model)
    pad_joint = {s: ('<freejoint/>' if kinematic else
                     f'<joint name="rail{s}" type="slide" axis="1 0 0" damping="{RAIL_DAMP}"/>') for s in "LR"}
    I_pad = 0.4 * M_PAD * R_PAD ** 2
    xml = f"""<mujoco model="hom_pinch_rig">
  <option timestep="{DT}" integrator="implicitfast" cone="elliptic" impratio="{sp['ir']}"
          solver="Newton" iterations="200" ls_iterations="50" tolerance="1e-12" gravity="{g}"/>
  <visual><global offwidth="960" offheight="720"/><quality shadowsize="2048"/></visual>
  <asset><texture name="grid" type="2d" builtin="checker" rgb1=".92 .92 .9" rgb2=".86 .86 .84" width="300" height="300"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".86 .89 .93" width="64" height="64"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0"/></asset>
  <worldbody>
    <light pos="0.3 -0.3 0.5" dir="-0.5 0.5 -1" directional="true" diffuse=".7 .7 .7" castshadow="false"/>
    <light pos="-0.2 -0.3 0.4" dir="0.4 0.6 -1" directional="true" diffuse=".3 .3 .3" castshadow="false"/>
    <geom name="floor" type="plane" pos="0 0 -0.09" size="0.4 0.4 0.01" material="grid" contype="0" conaffinity="0"/>
    <body name="padL" pos="{-X0} 0 0">
      {pad_joint['L']}
      <inertial pos="0 0 0" mass="{M_PAD}" diaginertia="{I_pad} {I_pad} {I_pad}"/>
      {pad_geoms['L']}
    </body>
    <body name="padR" pos="{X0} 0 0">
      {pad_joint['R']}
      <inertial pos="0 0 0" mass="{M_PAD}" diaginertia="{I_pad} {I_pad} {I_pad}"/>
      {pad_geoms['R']}
    </body>
    <body name="tool" pos="0 {d_cg} 0" quat="0.70710678 -0.70710678 0 0">
      <freejoint/>
      <inertial pos="0 0 0" mass="{M_TOOL}" diaginertia="{I_TOOL_T} {I_TOOL_T} {I_TOOL_A}"/>
      <geom name="tool" type="cylinder" size="{R_TOOL} {HL_TOOL}" contype="0" conaffinity="1"
            {tool_contact} rgba="0.55 0.6 0.68 1"/>
      <geom type="box" pos="0 0 {HL_TOOL}" size="0.0016 {R_TOOL * 0.8} 0.0008" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1"/>
    </body>
  </worldbody>
  {'' if kinematic else '''<actuator>
    <motor name="fL" joint="railL" ctrlrange="-50 50"/>
    <motor name="fR" joint="railR" ctrlrange="-50 50"/>
  </actuator>'''}
</mujoco>"""
    return xml, info


class MjRig:
    sim = "mujoco"

    def __init__(self, sp: dict, d_cg: float, gravity: bool, kinematic: bool = False):
        import mujoco
        self.mj, self.sp, self.d_cg = mujoco, sp, d_cg
        xml, self.info = mj_xml(sp, d_cg, gravity, kinematic)
        if sp["model"] == "spheres":
            # second pass: per-sphere stiffness from the compiled inverse weights, so that every
            # sphere contact is the spring K_s = E/R * A_s (steady state, solimp constant d0).
            m0 = mujoco.MjModel.from_xml_string(xml)
            diag = m0.body_invweight0[m0.body("padL").id, 0] + m0.body_invweight0[m0.body("tool").id, 0]
            K_s = self.info["K_sphere"]
            if "tr" in sp:
                # Per contact at steady state f = -K_s r - C_s v with K_s = 1 / (tc^2 (1 - d0) diag)
                # and C_s / K_s = 2 tc (solref (tc, 1), solimp d0 constant). The relaxation time
                # tr = C_s / K_s fixes tc = tr / 2, and d0 then sets K_s. d0 must stay positive,
                # so tr >= 2 / sqrt(K_s diag): about 15 ms for the 0.5 mm pad at E = 10 MPa.
                tc = sp["tr"] / 2.0
                d0 = 1.0 - 1.0 / (tc ** 2 * K_s * diag)
                if d0 < 0.05:
                    raise ValueError(f"relaxation {sp['tr']} s is below this pad's floor "
                                     f"{2.0 / math.sqrt(0.95 * K_s * diag):.4f} s")
            else:
                d0 = 0.95
                tc = 1.0 / math.sqrt(K_s * (1 - d0) * diag)
            xml, self.info = mj_xml(sp, d_cg, gravity, kinematic, pad_stiff=(tc, d0))
            self.info["solref_timeconst"] = tc
            self.info["solimp_d0"] = d0
            self.info["relaxation_s"] = 2 * tc
            self.info["diagApprox"] = float(diag)
        self.xml = xml
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.tool = self.m.body("tool").id
        self.pads = {s: self.m.body("pad" + s).id for s in "LR"}
        self.tool_geom = self.m.geom("tool").id
        self.geom_side = {}
        for gi in range(self.m.ngeom):
            b = self.m.geom_bodyid[gi]
            for s, bid in self.pads.items():
                if b == bid and self.m.geom_contype[gi]:
                    self.geom_side[gi] = s
        self.pad_geom = {s: self.m.geom("pad" + s).id for s in "LR"} if sp["model"].startswith("point") else {}
        self.lastN = {"L": 0.0, "R": 0.0}
        self.f6 = np.zeros(6)
        mujoco.mj_forward(self.m, self.d)
        self.theta_prev, self.theta_unwrap = None, 0.0

    @property
    def t(self):
        return float(self.d.time)

    def set_pad_force(self, N):
        self.d.ctrl[:] = [N, -N]

    def set_tool_wrench(self, f, tau):
        self.d.xfrc_applied[self.tool, :3] = f
        self.d.xfrc_applied[self.tool, 3:] = tau

    def _schedule(self):
        c, p = self.sp.get("fit_c", FIT_C), self.sp.get("fit_p", FIT_P)
        for s in "LR":
            self.m.geom_friction[self.pad_geom[s], 1] = MU * c * max(self.lastN[s], 1e-3) ** p

    def step(self, T):
        n = max(1, int(round(T / DT)))
        for _ in range(n):
            if self.sp["model"] == "point4s":
                self._schedule()
            self.mj.mj_step(self.m, self.d)
            if self.sp["model"] == "point4s":
                self.lastN = {s: v["N"] for s, v in self.contacts().items()}

    def contacts(self):
        out = {s: {"N": 0.0, "cop": np.zeros(3), "n": 0} for s in "LR"}
        acc = {s: np.zeros(3) for s in "LR"}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            s = self.geom_side.get(g1) or self.geom_side.get(g2)
            if s is None:
                continue
            self.mj.mj_contactForce(self.m, self.d, i, self.f6)
            fn = float(self.f6[0])
            out[s]["N"] += fn
            out[s]["n"] += 1 if fn > 1e-6 else 0
            acc[s] += fn * np.array(c.pos)
        for s in "LR":
            if out[s]["N"] > 1e-9:
                out[s]["cop"] = acc[s] / out[s]["N"]
        return out

    def tool_state(self):
        # From qpos/qvel, not xmat/cvel: after mj_step the derived quantities still describe the
        # state before the integration, and a feedback loop reading them runs one step late.
        qa = self.m.jnt_qposadr[self.m.body_jntadr[self.tool]]
        va = self.m.jnt_dofadr[self.m.body_jntadr[self.tool]]
        R9 = np.zeros(9)
        self.mj.mju_quat2Mat(R9, self.d.qpos[qa + 3:qa + 7])
        R = R9.reshape(3, 3)
        a = R[:, 2]
        theta = math.atan2(a[2], a[1])                # rotation about +x of the tool axis from +y
        if self.theta_prev is not None:
            dth = (theta - self.theta_prev + math.pi) % (2 * math.pi) - math.pi
            self.theta_unwrap += dth
        else:
            self.theta_unwrap = theta
        self.theta_prev = theta
        w = R @ self.d.qvel[va + 3:va + 6]            # free-joint angular velocity is body-local
        return {"theta": self.theta_unwrap, "omega_x": float(w[0]),
                "pos": self.d.qpos[qa:qa + 3].copy(), "axis": a.copy()}

    def pad_x(self):
        return {s: float(self.d.qpos[self.m.jnt_qposadr[self.m.body_jntadr[b]]]) + (-X0 if s == "L" else X0)
                for s, b in self.pads.items()}


# -------------------------------------------------------------------------------------- Drake

class DrakeRig:
    sim = "drake"

    def __init__(self, sp: dict, d_cg: float, gravity: bool, kinematic: bool = False):
        from pydrake.all import (
            AddCompliantHydroelasticProperties, AddContactMaterial, AddMultibodyPlant,
            AddRigidHydroelasticProperties, CoulombFriction, Cylinder, DiagramBuilder,
            FixedOffsetFrame, MultibodyPlantConfig, PrismaticJoint, ProximityProperties,
            RigidTransform, RotationMatrix, Simulator, SpatialInertia, Sphere,
        )
        self.sp, self.d_cg = sp, d_cg
        hydro = sp["model"] == "hydro"
        b = DiagramBuilder()
        cfg = MultibodyPlantConfig(time_step=sp.get("dt", DT), discrete_contact_approximation="sap",
                                   contact_model="hydroelastic_with_fallback" if hydro else "point")
        plant, sg = AddMultibodyPlant(cfg, b)
        if not gravity:
            plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, 0.0])
        self.pad_bodies, self.joints = {}, {}
        for side, sgn in (("L", -1.0), ("R", 1.0)):
            body = plant.AddRigidBody("pad" + side, SpatialInertia.SolidSphereWithMass(M_PAD, R_PAD))
            pp = ProximityProperties()
            if hydro:
                AddCompliantHydroelasticProperties(sp["res"], sp["E"], pp)
            AddContactMaterial(dissipation=sp.get("hc", 10.0), point_stiffness=sp.get("kp", 1e4),
                               friction=CoulombFriction(MU, MU), properties=pp)
            if "rt" in sp:
                pp.AddProperty("material", "relaxation_time", sp["rt"])
            plant.RegisterCollisionGeometry(body, RigidTransform(), Sphere(R_PAD), "pad" + side, pp)
            self.pad_bodies[side] = body
            if not kinematic:
                fr = plant.AddFrame(FixedOffsetFrame("slot" + side, plant.world_frame(),
                                                     RigidTransform([sgn * X0, 0.0, 0.0])))
                j = plant.AddJoint(PrismaticJoint("rail" + side, fr, body.body_frame(), [1.0, 0.0, 0.0],
                                                  damping=RAIL_DAMP))
                plant.AddJointActuator("f" + side, j)
                self.joints[side] = j
        tool = plant.AddRigidBody("tool", SpatialInertia.SolidCylinderWithMass(
            M_TOOL, R_TOOL, 2 * HL_TOOL, [0.0, 0.0, 1.0]))
        tp = ProximityProperties()
        if hydro:
            AddRigidHydroelasticProperties(sp.get("res_tool", 0.0005), tp)
        AddContactMaterial(dissipation=sp.get("hc", 10.0), point_stiffness=sp.get("kp", 1e4),
                           friction=CoulombFriction(MU, MU), properties=tp)
        if "rt" in sp:
            tp.AddProperty("material", "relaxation_time", sp["rt"])
        plant.RegisterCollisionGeometry(tool, RigidTransform(), Cylinder(R_TOOL, 2 * HL_TOOL), "tool", tp)
        plant.Finalize()
        self.plant, self.sg, self.toolb = plant, sg, tool
        self.diagram = b.Build()
        self.simulator = Simulator(self.diagram)
        self.ctx = self.simulator.get_mutable_context()
        self.pc = plant.GetMyMutableContextFromRoot(self.ctx)
        self.sgc = sg.GetMyMutableContextFromRoot(self.ctx)
        X_WT = RigidTransform(RotationMatrix.MakeXRotation(-math.pi / 2), [0.0, d_cg, 0.0])
        plant.SetFreeBodyPose(self.pc, tool, X_WT)
        self.geom_side = {}
        for s, body in self.pad_bodies.items():
            for gid in plant.GetCollisionGeometriesForBody(body):
                self.geom_side[gid] = s
        self.kinematic = kinematic
        if kinematic:
            for s, sgn in (("L", -1.0), ("R", 1.0)):
                plant.SetFreeBodyPose(self.pc, self.pad_bodies[s], RigidTransform([sgn * (X0 + 0.01), 0, 0]))
        else:
            self._fix_inputs(0.0, np.zeros(3), np.zeros(3))
            self.simulator.Initialize()
        self.N = 0.0
        self.f_tool, self.tau_tool = np.zeros(3), np.zeros(3)
        self.theta_prev, self.theta_unwrap = None, 0.0

    @property
    def t(self):
        return float(self.ctx.get_time())

    def _fix_inputs(self, N, f, tau):
        from pydrake.all import ExternallyAppliedSpatialForce, SpatialForce
        self.plant.get_actuation_input_port().FixValue(self.pc, np.array([N, -N]))
        e = ExternallyAppliedSpatialForce()
        e.body_index = self.toolb.index()
        e.p_BoBq_B = np.zeros(3)
        e.F_Bq_W = SpatialForce(np.asarray(tau, float), np.asarray(f, float))
        self.plant.get_applied_spatial_force_input_port().FixValue(self.pc, [e])

    def set_pad_force(self, N):
        self.N = N
        self._fix_inputs(self.N, self.f_tool, self.tau_tool)

    def set_tool_wrench(self, f, tau):
        self.f_tool, self.tau_tool = np.asarray(f, float), np.asarray(tau, float)
        self._fix_inputs(self.N, self.f_tool, self.tau_tool)

    def step(self, T):
        self.simulator.AdvanceTo(self.ctx.get_time() + T)

    def contacts(self):
        out = {s: {"N": 0.0, "cop": np.zeros(3), "n": 0} for s in "LR"}
        cr = self.plant.get_contact_results_output_port().Eval(self.pc)
        for i in range(cr.num_hydroelastic_contacts()):
            info = cr.hydroelastic_contact_info(i)
            srf = info.contact_surface()
            s = self.geom_side.get(srf.id_M()) or self.geom_side.get(srf.id_N())
            if s is None:
                continue
            Np, cop, nf = surface_cop(srf)
            # normal force from the solver's resultant, not the pressure integral: at a 1 ms step
            # SAP regularises these stiff faces and the two differ by up to ~20 % (see the page).
            F = np.array(info.F_Ac_W().translational())
            out[s].update(N=float(abs(F[0])), N_pressure=Np, cop=cop, n=nf)
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            pp = info.point_pair()
            s = self.geom_side.get(pp.id_A) or self.geom_side.get(pp.id_B)
            if s is None:
                continue
            f = np.array(info.contact_force())
            out[s].update(N=float(abs(f[0])), cop=np.array(info.contact_point()), n=1)
        return out

    def tool_state(self):
        X = self.plant.EvalBodyPoseInWorld(self.pc, self.toolb)
        a = X.rotation().matrix()[:, 2]
        theta = math.atan2(a[2], a[1])
        if self.theta_prev is not None:
            dth = (theta - self.theta_prev + math.pi) % (2 * math.pi) - math.pi
            self.theta_unwrap += dth
        else:
            self.theta_unwrap = theta
        self.theta_prev = theta
        V = self.plant.EvalBodySpatialVelocityInWorld(self.pc, self.toolb)
        return {"theta": self.theta_unwrap, "omega_x": float(V.rotational()[0]),
                "pos": np.array(X.translation()), "axis": a.copy()}

    def pad_x(self):
        return {s: float(self.plant.EvalBodyPoseInWorld(self.pc, b).translation()[0])
                for s, b in self.pad_bodies.items()}


def surface_cop(srf):
    """Normal force, centre of pressure and face count of a Drake hydroelastic contact surface."""
    tri = srf.is_triangle()
    mesh = srf.tri_mesh_W() if tri else srf.poly_mesh_W()
    field = srf.tri_e_MN() if tri else srf.poly_e_MN()
    nf = mesh.num_elements()
    if nf == 0:
        return 0.0, np.zeros(3), 0
    A = np.array([mesh.area(f) for f in range(nf)])
    C = np.array([mesh.element_centroid(f) for f in range(nf)])
    P = np.array([field.EvaluateCartesian(f, C[f]) for f in range(nf)])
    w = P * A
    N = float(w.sum())
    return N, (w @ C) / max(N, 1e-30), nf


def make_rig(spec: str, d_cg: float = 0.0, gravity: bool = True, kinematic: bool = False):
    sp = parse_spec(spec)
    return (DrakeRig if sp["sim"] == "drake" else MjRig)(sp, d_cg, gravity, kinematic)


# -------------------------------------------------------------------------------- experiments

def settle(rig, N, T=0.4, weight_comp=True):
    rig.set_pad_force(N)
    f = np.array([0.0, 0.0, M_TOOL * G]) if weight_comp else np.zeros(3)
    rig.set_tool_wrench(f, np.zeros(3))
    rig.step(T)


def exp_torsion(spec, N, omega=1.0, T_spin=1.2, traces=False):
    """Gravity off. Spin the pinched tool about the pinch axis at `omega` with a stiff PD torque on
    the tool; the steady torque the drive must supply is the two pads' friction torque."""
    t0w = time.time()
    rig = make_rig(spec, d_cg=0.0, gravity=False)
    settle(rig, N, T=1.0, weight_comp=False)
    st = rig.tool_state()
    th0, t0 = st["theta"], rig.t
    kp, kd = 0.5, 0.005
    rows, taus = [], []
    n = int(round(T_spin / DT))
    for k in range(n):
        st = rig.tool_state()
        tt = rig.t - t0
        ref = th0 + omega * tt
        tau = kp * (ref - st["theta"]) + kd * (omega - st["omega_x"])
        rig.set_tool_wrench(np.zeros(3), np.array([tau, 0.0, 0.0]))
        rig.step(DT)
        if tt > 0.4:
            taus.append(tau)
        if traces and k % 10 == 0:
            c = rig.contacts()
            rows.append([round(tt, 4), tau, st["theta"] - th0, st["omega_x"], c["L"]["N"], c["R"]["N"],
                         c["L"]["n"], c["R"]["n"]])
    c = rig.contacts()
    out = {"exp": "torsion", "spec": spec, "N": N, "omega": omega,
           "tau_fric": float(np.mean(taus)), "tau_sd": float(np.std(taus)),
           "rbar_per_pad_mm": float(np.mean(taus)) / (2 * MU * N) * 1e3,
           "N_L": c["L"]["N"], "N_R": c["R"]["N"], "n_L": c["L"]["n"], "n_R": c["R"]["n"],
           "pad_xL_mm": rig.pad_x()["L"] * 1e3, "wall_s": time.time() - t0w,
           "info": getattr(rig, "info", {})}
    if traces:
        out["trace_cols"] = ["t", "tau", "dtheta", "omega_x", "NL", "NR", "nL", "nR"]
        out["trace"] = rows
    return out


def pinch_station(st):
    """Coordinate along the tool axis (from the tool centre) of the point nearest the pinch axis
    (world x through the origin). Its change is the tool's slip through the pinch."""
    p, a = st["pos"], st["axis"]
    den = a[1] ** 2 + a[2] ** 2
    return float(-(p[1] * a[1] + p[2] * a[2]) / max(den, 1e-12))


def exp_brake(spec, d_cg, N0=6.0, N1=0.2, T_hold=0.5, T_ramp=4.0, T_end=1.0, traces=False, film=None,
              keep_frames=False, film_wh=(480, 360)):
    """Gravity on. Pinch at N0 with the tool horizontal (weight compensated while the pads settle),
    release the weight, hold, lower the pinch geometrically to N1 over T_ramp, hold."""
    t0w = time.time()
    rig = make_rig(spec, d_cg=d_cg, gravity=True)
    settle(rig, N0, T=0.4, weight_comp=True)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    st = rig.tool_state()
    th0, p0, t0 = st["theta"], st["pos"].copy(), rig.t
    s0 = pinch_station(st)
    rows, frames = [], []
    T = T_hold + T_ramp + T_end
    n = int(round(T / DT))
    renderer = None
    if film or keep_frames:
        renderer = FilmRig(spec, d_cg, *film_wh)
    for k in range(n):
        tt = rig.t - t0
        if tt < T_hold:
            N = N0
        elif tt < T_hold + T_ramp:
            N = N0 * (N1 / N0) ** ((tt - T_hold) / T_ramp)
        else:
            N = N1
        rig.set_pad_force(N)
        rig.step(DT)
        if k % 10 == 0:
            st = rig.tool_state()
            c = rig.contacts()
            phi = -math.degrees(st["theta"] - th0)
            slide = pinch_station(st) - s0
            rows.append([round(tt, 4), N, phi, -st["omega_x"], slide * 1e3,
                         c["L"]["N"], c["R"]["N"], c["L"]["n"], c["R"]["n"],
                         float(st["pos"][2] - p0[2]) * 1e3])
            if renderer is not None and k % 40 == 0:
                frames.append(renderer.frame(rig, f"{label_of(spec)}"))
                frames[-1] = annotate_bottom(frames[-1], f"N {N:4.2f} N   phi {phi:5.1f} deg   t {tt:4.2f} s")
    tr = np.array(rows)
    phi = tr[:, 2]
    res = {"exp": "brake", "spec": spec, "d_mm": d_cg * 1e3, "N0": N0, "N1": N1, "T_ramp": T_ramp,
           "phi_end": float(phi[-1]), "phi_max": float(phi.max()),
           "dropped": bool(tr[-1, 9] < -20.0),
           "slip_end_mm": float(tr[-1, 4]), "slip_max_mm": float(np.abs(tr[:, 4]).max()),
           "wall_s": time.time() - t0w}
    # the pinch force at which the tool first passes 10, 45, 80 deg
    for a in (10, 45, 80):
        idx = np.where(phi >= a)[0]
        res[f"N_at_{a}"] = float(tr[idx[0], 1]) if len(idx) else None
        res[f"t_at_{a}"] = float(tr[idx[0], 0]) if len(idx) else None
    res["peak_rate_dps"] = float(np.degrees(np.abs(tr[:, 3]).max()))
    if traces:
        res["trace_cols"] = ["t", "N", "phi_deg", "phidot", "slip_mm", "NL", "NR", "nL", "nR", "dz_mm"]
        res["trace"] = rows
    if renderer is not None and film:
        write_mp4(frames, film, fps=25)
        res["film"] = str(film)
    if keep_frames:
        res["_frames"] = frames
    return res


def exp_cop(spec, delta_mm=0.18, angles=np.linspace(0.0, 20.0, 81)):
    """Kinematic. The left pad sits at penetration delta (pad centre on the pinch axis) and is
    rotated about its own centre by `angle` about world z, then about world y. The centre of
    pressure should stay at the analytic contact point (on the pinch axis) for any rotation."""
    rig = make_rig(spec, d_cg=0.0, gravity=False, kinematic=True)
    delta = delta_mm * 1e-3
    c = np.array([-(X0 - delta), 0.0, 0.0])
    ideal = np.array([-R_TOOL, 0.0, 0.0])
    out = {"exp": "cop", "spec": spec, "delta_mm": delta_mm, "angle": [], "err_mm": [], "n": [], "cop": []}
    for ax in ("z", "y"):
        for ang in angles:
            a = math.radians(ang)
            if ax == "z":
                R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            else:
                R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            cop, n = kinematic_cop(rig, c, R)
            out["angle"].append(f"{ax}{ang:.2f}")
            # tangential error only: MuJoCo places a contact mid-overlap, Drake on the rigid surface
            out["err_mm"].append(float(np.linalg.norm((cop - ideal)[1:])) * 1e3 if n else None)
            out["n"].append(int(n))
            out["cop"].append([float(x) * 1e3 for x in cop])
    e = np.array([x for x in out["err_mm"] if x is not None])
    out["err_max_mm"] = float(e.max()) if len(e) else None
    out["err_rms_mm"] = float(np.sqrt((e ** 2).mean())) if len(e) else None
    steps = np.abs(np.diff(np.array([p for p in out["cop"]])[:, 1:], axis=0))
    out["max_step_mm"] = float(steps.max()) if len(steps) else None
    return out


def exp_plate(E=1e7, res_mm=0.5, F=1.0):
    """Drake only. A compliant hydroelastic pad (the fingertip sphere) loaded by a weight F onto a rigid
    plate: penetration against the elastic-foundation law F = pi E delta^2 and the pressure-weighted
    radius of the patch against (8/15) a, a = sqrt(2 R delta)."""
    from pydrake.all import (AddCompliantHydroelasticProperties, AddContactMaterial, AddMultibodyPlant,
                             AddRigidHydroelasticProperties, Box, CoulombFriction, DiagramBuilder,
                             MultibodyPlantConfig, PrismaticJoint, ProximityProperties, RigidTransform,
                             Simulator, SpatialInertia, Sphere)
    b = DiagramBuilder()
    plant, sg = AddMultibodyPlant(MultibodyPlantConfig(time_step=DT, discrete_contact_approximation="sap",
                                                       contact_model="hydroelastic_with_fallback"), b)
    pp = ProximityProperties()
    AddRigidHydroelasticProperties(0.002, pp)
    AddContactMaterial(dissipation=1.0, friction=CoulombFriction(MU, MU), properties=pp)
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform([0, 0, -0.005]), Box(0.1, 0.1, 0.01), "plate", pp)
    body = plant.AddRigidBody("tip", SpatialInertia.SolidSphereWithMass(F / G, R_PAD))
    sp = ProximityProperties()
    AddCompliantHydroelasticProperties(res_mm * 1e-3, E, sp)
    AddContactMaterial(dissipation=1.0, friction=CoulombFriction(MU, MU), properties=sp)
    sp.AddProperty("material", "relaxation_time", 0.01)
    plant.RegisterCollisionGeometry(body, RigidTransform(), Sphere(R_PAD), "tip", sp)
    j = plant.AddJoint(PrismaticJoint("z", plant.world_frame(), body.body_frame(), [0, 0, 1]))
    plant.Finalize()
    sim = Simulator(b.Build())
    ctx = sim.get_mutable_context()
    pc = plant.GetMyMutableContextFromRoot(ctx)
    j.set_translation(pc, R_PAD + 0.0002)
    for _ in range(40):
        sim.AdvanceTo(ctx.get_time() + 0.05)
        j.set_translation_rate(pc, 0.0)
    sim.AdvanceTo(ctx.get_time() + 0.3)
    delta = R_PAD - j.get_translation(pc)
    cr = plant.get_contact_results_output_port().Eval(pc)
    N, cop, nf = surface_cop(cr.hydroelastic_contact_info(0).contact_surface())
    srf = cr.hydroelastic_contact_info(0).contact_surface()
    mesh = srf.poly_mesh_W()
    field = srf.poly_e_MN()
    A = np.array([mesh.area(f) for f in range(mesh.num_elements())])
    C = np.array([mesh.element_centroid(f) for f in range(mesh.num_elements())])
    P = np.array([field.EvaluateCartesian(f, C[f]) for f in range(mesh.num_elements())])
    rho = np.linalg.norm(C[:, :2] - cop[:2], axis=1)
    a = math.sqrt(2 * R_PAD * delta)
    return {"exp": "plate", "spec": f"drake:hydro:E{E:g}:r{res_mm:g}", "E": E, "res_mm": res_mm, "N": F,
            "delta_mm": delta * 1e3, "F_winkler_at_delta": math.pi * E * delta ** 2,
            "N_pressure": N, "faces": nf, "patch_a_mm": a * 1e3,
            "rbar_mm": float((P * A * rho).sum() / (P * A).sum()) * 1e3, "rbar_over_a": float((P * A * rho).sum() / (P * A).sum()) / a}


def kinematic_cop(rig, center, R):
    """CoP of the left pad at a pose: Drake from the contact surface, MuJoCo from contact points
    weighted by penetration (the steady-state force of identical linear springs)."""
    if rig.sim == "drake":
        from pydrake.all import HydroelasticContactRepresentation, RigidTransform, RotationMatrix
        rig.plant.SetFreeBodyPose(rig.pc, rig.pad_bodies["L"], RigidTransform(RotationMatrix(R), center))
        q = rig.sg.get_query_output_port().Eval(rig.sgc)
        if rig.sp["model"] == "hydro":
            for srf in q.ComputeContactSurfaces(HydroelasticContactRepresentation.kPolygon):
                if rig.geom_side.get(srf.id_M()) == "L" or rig.geom_side.get(srf.id_N()) == "L":
                    N, cop, nf = surface_cop(srf)
                    return cop, nf
            return np.zeros(3), 0
        for pen in q.ComputePointPairPenetration():
            if rig.geom_side.get(pen.id_A) == "L" or rig.geom_side.get(pen.id_B) == "L":
                return 0.5 * (np.array(pen.p_WCa) + np.array(pen.p_WCb)), 1
        return np.zeros(3), 0
    mj = rig.mj
    jid = rig.m.body_jntadr[rig.pads["L"]]
    qa = rig.m.jnt_qposadr[jid]
    quat = np.zeros(4)
    mj.mju_mat2Quat(quat, R.reshape(-1))
    rig.d.qpos[qa:qa + 3] = center
    rig.d.qpos[qa + 3:qa + 7] = quat
    jr = rig.m.body_jntadr[rig.pads["R"]]
    qr = rig.m.jnt_qposadr[jr]
    rig.d.qpos[qr:qr + 3] = [X0 + 0.02, 0, 0]
    mj.mj_forward(rig.m, rig.d)
    w, acc, n = 0.0, np.zeros(3), 0
    for i in range(rig.d.ncon):
        cc = rig.d.contact[i]
        s = rig.geom_side.get(int(cc.geom[0])) or rig.geom_side.get(int(cc.geom[1]))
        if s != "L" or cc.dist >= 0:
            continue
        pen = -float(cc.dist)
        if rig.sp["model"] == "spheres":
            w += pen
            acc += pen * np.array(cc.pos)
        else:
            w, acc = 1.0, np.array(cc.pos)
        n += 1
    return (acc / w if w > 0 else np.zeros(3)), n


# ------------------------------------------------------------------------------------- films

def label_of(spec):
    sp = parse_spec(spec)
    if sp["sim"] == "drake":
        if sp["model"] != "hydro":
            return "Drake point contact"
        return f"Drake hydroelastic, relaxation {sp.get('rt', 0.1):g} s"
    if sp["model"] == "spheres":
        return f"MuJoCo {sp['s'] * 1e3:g} mm sphere pads, relaxation {sp.get('tr', 0):g} s"
    return {"point3": "MuJoCo point contact, condim 3",
            "point4": f"MuJoCo condim 4, mu_t {sp.get('mu_t', 0) * 1e3:.2f} mm",
            "point4s": "MuJoCo condim 4, mu_t rescheduled"}[sp["model"]]


class FilmRig:
    """Renders any rig through a MuJoCo copy of the same geometry (Drake runs are replayed)."""

    def __init__(self, spec, d_cg, w=640, h=480):
        import mujoco
        sp = parse_spec(spec)
        if sp["sim"] == "drake":
            sp = parse_spec("mj:point3")
        xml, _ = mj_xml(sp, d_cg, True)
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.0, 0.012, -0.025]
        self.cam.distance = 0.24
        self.cam.azimuth = 25.0
        self.cam.elevation = -10.0
        self.tool = self.m.body("tool").id

    def frame(self, rig, text):
        import mujoco
        st = rig.tool_state() if rig.sim == "drake" else None
        if rig.sim == "mujoco":
            self.d.qpos[:] = 0
            # copy tool and rail positions from the live rig
            for name in ("railL", "railR"):
                self.d.qpos[self.m.joint(name).qposadr[0]] = rig.d.qpos[rig.m.joint(name).qposadr[0]]
            ja = self.m.body_jntadr[self.tool]
            qa = self.m.jnt_qposadr[ja]
            jb = rig.m.body_jntadr[rig.tool]
            qb = rig.m.jnt_qposadr[jb]
            self.d.qpos[qa:qa + 7] = rig.d.qpos[qb:qb + 7]
        else:
            X = rig.plant.EvalBodyPoseInWorld(rig.pc, rig.toolb)
            ja = self.m.body_jntadr[self.tool]
            qa = self.m.jnt_qposadr[ja]
            self.d.qpos[qa:qa + 3] = X.translation()
            self.d.qpos[qa + 3:qa + 7] = X.rotation().ToQuaternion().wxyz()
            for s in "LR":
                xs = rig.pad_x()[s]
                x0 = -X0 if s == "L" else X0
                self.d.qpos[self.m.joint("rail" + s).qposadr[0]] = xs - x0
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, self.cam)
        img = self.r.render().copy()
        return annotate(img, text)


def annotate(img, text, size=15):
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            font = ImageFont.load_default(size=size)
        dr.rectangle([0, 0, im.width, size + 10], fill=(255, 255, 255))
        dr.text((8, 4), text, fill=(20, 20, 20), font=font)
        return np.asarray(im)
    except Exception:
        return img


def annotate_bottom(img, text, size=14):
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", size)
        except Exception:
            font = ImageFont.load_default(size=size)
        dr.rectangle([0, im.height - size - 10, im.width, im.height], fill=(255, 255, 255))
        dr.text((8, im.height - size - 6), text, fill=(20, 20, 20), font=font)
        return np.asarray(im)
    except Exception:
        return img


def write_mp4(frames, path, fps=50):
    import imageio.v2 as imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps, macro_block_size=1)


# --------------------------------------------------------------------------------------- CLI

def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=_json_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("torsion")
    a.add_argument("--spec", required=True)
    a.add_argument("--N", type=float, default=1.0)
    a.add_argument("--omega", type=float, default=1.0)
    a.add_argument("--traces", action="store_true")
    a.add_argument("--rows")
    b = sub.add_parser("brake")
    b.add_argument("--spec", required=True)
    b.add_argument("--d", type=float, default=15.0, help="CG offset from the pinch line, mm")
    b.add_argument("--N0", type=float, default=6.0)
    b.add_argument("--N1", type=float, default=0.2)
    b.add_argument("--T-ramp", type=float, default=4.0)
    b.add_argument("--traces", action="store_true")
    b.add_argument("--film")
    b.add_argument("--rows")
    c = sub.add_parser("cop")
    c.add_argument("--spec", required=True)
    c.add_argument("--delta", type=float, default=0.18)
    c.add_argument("--rows")
    w = sub.add_parser("winkler")
    w.add_argument("--E", type=float, default=1e7)
    args = ap.parse_args()
    if args.cmd == "torsion":
        row = exp_torsion(args.spec, args.N, args.omega, traces=args.traces)
    elif args.cmd == "brake":
        row = exp_brake(args.spec, args.d * 1e-3, args.N0, args.N1, T_ramp=args.T_ramp,
                        traces=args.traces, film=args.film)
    elif args.cmd == "cop":
        row = exp_cop(args.spec, args.delta)
    else:
        row = {"exp": "winkler", "E": args.E,
               "law": [dict(N=N, delta_mm=winkler_law(N, args.E)[0] * 1e3,
                            rbar_mm=winkler_law(N, args.E)[1] * 1e3,
                            rbar_flat_mm=rbar_sphere_flat(N, args.E) * 1e3)
                       for N in (0.1, 0.25, 0.5, 1, 2, 4, 8)]}
    if getattr(args, "rows", None):
        append_row(args.rows, row)
    slim = {k: v for k, v in row.items() if k not in ("trace", "cop", "angle", "err_mm", "n")}
    print(json.dumps(slim, default=_json_default))


if __name__ == "__main__":
    main()
