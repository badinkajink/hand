#!/usr/bin/env python3
"""Contact-model comparison bed, task 3: the pinched tool rolled between a fixed pad and a pad sliding along z.

The rig is the two-pad pinch of scripts/hom_contact_rig.py: two real_v1 fingertip spheres (r 10.55 mm,
20 g) on force-controlled rails along world x squeeze the real_v1 screwdriver (cylinder r 12.5 mm,
100 mm, 24.5 g, axis along world y), mu 1, gravity off. This script gives the +x pad a second slide,
along world z, with a prescribed motion: the slide carries an added inertia of M_LIFT (joint armature in
MuJoCo, a carriage body in Drake) and a PD servo with acceleration feed-forward drives it along s_ref(t)
(bandwidth 30 rad/s, critically damped; the contact load moves it by micrometres). Both pads stay
force-controlled at N along x and the -x pad has no z motion.

Per case (model, pinch N per pad, step dt, pad speed v): settle 0.4 s at N, move the +x pad up by
s = 10 mm (10 mm/s for 1.0 s, or 50 mm/s for 0.2 s; 20 ms raised-cosine speed ramps at both ends keep
the travel at v T), then hold 0.2 s. Each step records:

  rolling ratio  rho = theta r_tool / (s/2), theta the tool's rotation about world y (signed so that
                 rolling without slip is positive) and s the measured pad travel. Flat pads give
                 rho = 1 without slip. Between sphere pads the line of centres tilts by
                 alpha = asin(s / 2D) (D the pad-to-tool centre distance) and no-slip rolling gives
                 rho = alpha / sin(alpha) = 1.008 at s = 10 mm; rows carry it as rho_noslip_geom.
  slip           at each pad, the tangential relative velocity of tool and pad at the contact point,
                 integrated over the motion: path length, and the signed circumferential (positive
                 when the tool surface moves up relative to the pad) and axial components. The contact
                 point is the force-weighted centre of the contacts MuJoCo reports (mid-penetration
                 points), or the pressure-weighted centroid of Drake's contact surface (on the rigid
                 tool's surface). Its distance r_c from the tool axis is the radius the model rolls on:
                 no slip at that point gives rho = (alpha / sin alpha) R_tool / r_c (rho_noslip_contact).
                 Kinematics tie the slips to the tool: rho = rho_noslip_contact (1 + (slip_R - slip_L) / s)
                 and dz / (s/2) = 1 + (slip_R + slip_L) / s (circumferential slips; rows carry the first
                 as rho_from_slip, a check of the slip measurement). r_roll_mm = R_tool rho_noslip_geom /
                 rho is the radius on which the observed rotation is slip-free.
  drift          tool displacement along y, rotation of its axis about x and z, distance of the tool
                 centre from the no-slip path (x0, y0, z0 + s/2).
  forces         per pad, normal and tangential components of the total contact force on the tool.
                 Rolling without resistance has Fn = N / cos(alpha) (2.5 % above N at the end) and
                 Ft = 0; Ft / (mu Fn) near 1 means the contact is sliding.
  success        both pads keep Fn > 0.1 N during the motion and hold, and the tool centre stays within
                 5 mm of the no-slip path. `skid` flags a contact whose slip path exceeds 10 % of the
                 rolling distance s/2.

Physics cost is the wall time of the rig's step call alone. Films (N = 1 N, dt 1 ms, 10 mm/s): a MuJoCo
copy of the rig draws every model, pads and tool translucent, the reported contacts as spheres coloured
by pressure (sphere pads, hydroelastic faces) or force (point contacts), viewed 8 deg off the tool axis
from its -y end (a red dot and stripe on the end face and a red stripe along the top show the roll).

Another simulator plugs in by providing a rig with the methods of MjRollRig (set_pad_force,
set_tool_wrench, set_lift_force, lift_state, lift_mass, step, tool_kin, pad_kin, contact_patches) and
calling run_roll(rig, ...); roll_mj_xml gives the MJCF of the roll rig.

    PY=logs/20261001-hom_contact/venv/bin/python                  # MuJoCo 3.6 + Drake 1.57
    $PY scripts/contact_bed_roll.py run --models mj_pads1 --N 1 --dt 1 --v 10      # one case
    MUJOCO_GL=egl $PY scripts/contact_bed_roll.py run              # the grid of the task, films included
    MUJOCO_GL=egl $PY scripts/contact_bed_roll.py frame --model mj_pads1 --out /tmp/f.png
    MUJOCO_GL=egl $PY scripts/contact_bed_roll.py tile
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_contact_rig as H  # noqa: E402

MODELS = {   # bed model name -> rig spec (PROTOCOL.md)
    "mj_point3": "mj:point3:ir100",
    "mj_point4s": "mj:point4s:fit0.000996044x0.2498:ir100",
    "mj_pads1": "mj:spheres:s1:rs0.75:ir100:tr0.02",
    "mj_pads2": "mj:spheres:s2:rs0.75:ir100:tr0.02",
    "mj_pads05": "mj:spheres:s0.5:rs0.75:ir100:tr0.03",
    "drake_hydro": "drake:hydro:E1e7:r1:rt0.01",
}
MAIN_MODELS = ["mj_point3", "mj_point4s", "mj_pads1", "drake_hydro"]
SPEEDS = {10.0: 1.0, 50.0: 0.2}      # pad speed mm/s -> travel time s (s = 10 mm in both)
OUT_DIR = ROOT / "docs/experiments/20261005-contact_bed"
OUT = OUT_DIR / "roll.jsonl"
FRAMES_DIR = ROOT / "logs/20261005-contact_bed_roll"
SCRIPT = "scripts/contact_bed_roll.py"

M_LIFT = 100.0        # kg added to the +x pad's z slide
LIFT_WN = 30.0        # rad/s, servo bandwidth (kp = m wn^2, kd = 2 m wn)
T_SETTLE = 0.4
T_RAMP = 0.020        # raised-cosine speed ramp at each end of the motion
T_HOLD = 0.2
FN_LOST = 0.1         # contact counts as lost below this fraction of N
PATH_TOL = 0.005      # m, success: tool centre within this distance of the no-slip path
SKID_FRAC = 0.10      # skid: slip path above this fraction of s/2 at a contact
EJECT = 0.02          # m, tool displacement that ends a case

FILM_N, FILM_DT, FILM_V = 1.0, 1.0, 10.0
FILM_FPS, FILM_SPEED = 25, 0.25
FILM_WH = (480, 360)
FILM_PRE = 0.1        # s of the settled pinch shown before the motion
P_SCALE = 0.3e6       # Pa, top of the pressure colour scale
F_SCALE = 2.0         # N, top of the force colour scale (point contacts)


# ------------------------------------------------------------------------------- motion + servo

def lift_ref(t, V, T, tr=T_RAMP):
    """Pad z reference (s, v, a): raised-cosine ramp up over tr, cruise until T, ramp down; travel V T."""
    if t <= 0.0:
        return 0.0, 0.0, 0.0
    if t < tr:
        ph = math.pi * t / tr
        return (0.5 * V * (t - tr / math.pi * math.sin(ph)), 0.5 * V * (1 - math.cos(ph)),
                0.5 * V * math.pi / tr * math.sin(ph))
    if t < T:
        return V * tr / 2 + V * (t - tr), V, 0.0
    if t < T + tr:
        u = t - T
        ph = math.pi * u / tr
        return (V * (T - tr / 2) + 0.5 * V * (u + tr / math.pi * math.sin(ph)), 0.5 * V * (1 + math.cos(ph)),
                -0.5 * V * math.pi / tr * math.sin(ph))
    return V * T, 0.0, 0.0


class LiftServo:
    """PD with acceleration feed-forward on the heavy z slide; explicit, stable for wn dt < 1."""

    def __init__(self, m, wn=LIFT_WN):
        self.m, self.kp, self.kd = m, m * wn ** 2, 2.0 * m * wn

    def force(self, ref, s, v):
        s_ref, v_ref, a_ref = ref
        return self.m * a_ref + self.kp * (s_ref - s) + self.kd * (v_ref - v)


# ------------------------------------------------------------------------------------- MuJoCo rig

LIFT_JOINT = '<joint name="liftR" type="slide" axis="0 0 1" armature="{arm:g}"/>'


def add_lift(xml, armature=M_LIFT):
    """Insert the +x pad's z slide after its x rail, so the rail stays the body's first joint."""
    new, n = re.subn(r'(<joint name="railR"[^>]*/>)', lambda m: m.group(1) + "\n      " +
                     LIFT_JOINT.format(arm=armature), xml)
    if n != 1:
        raise RuntimeError("railR joint not found in the rig MJCF")
    return new


def roll_mj_xml(sp, d_cg=0.0, gravity=False, **kw):
    """MJCF of the roll rig: hom_contact_rig.mj_xml plus the +x pad's z slide (liftR)."""
    xml, info = H.mj_xml(sp, d_cg, gravity, **kw)
    return add_lift(xml), info


class MjRollRig(H.MjRig):
    """hom_contact_rig.MjRig with the z slide on the +x pad, driven through qfrc_applied."""

    def __init__(self, sp, d_cg=0.0, gravity=False):
        orig = H.mj_xml

        def wrapped(*a, **k):        # MjRig builds its MJCF (twice for sphere pads) through H.mj_xml
            xml, info = orig(*a, **k)
            return add_lift(xml), info
        H.mj_xml = wrapped
        try:
            super().__init__(sp, d_cg, gravity)
        finally:
            H.mj_xml = orig
        j = self.m.joint("liftR")
        self.lift_q, self.lift_v = int(j.qposadr[0]), int(j.dofadr[0])
        self.rail = {s: (int(self.m.joint("rail" + s).qposadr[0]), int(self.m.joint("rail" + s).dofadr[0]))
                     for s in "LR"}
        tj = self.m.body_jntadr[self.tool]
        self.tq, self.tv = int(self.m.jnt_qposadr[tj]), int(self.m.jnt_dofadr[tj])
        self.lift_mass = M_LIFT + H.M_PAD
        self.area_s = (self.info.get("area_per_sphere_mm2", 0.0) * 1e-6) if sp["model"] == "spheres" else None

    def set_lift_force(self, u):
        self.d.qfrc_applied[self.lift_v] = u

    def lift_state(self):
        return float(self.d.qpos[self.lift_q]), float(self.d.qvel[self.lift_v])

    def tool_kin(self):
        """Tool centre position, rotation matrix, linear and angular velocity (world), from qpos/qvel."""
        q = self.d.qpos[self.tq:self.tq + 7]
        R9 = np.zeros(9)
        self.mj.mju_quat2Mat(R9, q[3:7])
        R = R9.reshape(3, 3)
        return q[:3].copy(), R, self.d.qvel[self.tv:self.tv + 3].copy(), R @ self.d.qvel[self.tv + 3:self.tv + 6]

    def pad_kin(self):
        """Pad centre position and velocity (pads translate only)."""
        (qL, vL), (qR, vR) = self.rail["L"], self.rail["R"]
        q, v = self.d.qpos, self.d.qvel
        return {"L": (np.array([-H.X0 + q[qL], 0.0, 0.0]), np.array([v[vL], 0.0, 0.0])),
                "R": (np.array([H.X0 + q[qR], 0.0, q[self.lift_q]]), np.array([v[vR], 0.0, v[self.lift_v]]))}

    def contact_patches(self, detail=False):
        """Per pad: total contact force on the tool (world), force-weighted contact centre, contact count;
        with detail, every contact as (pos, normal, pressure or force, glyph radius, kind)."""
        out = {s: {"F": np.zeros(3), "cop": None, "n": 0, "pts": []} for s in "LR"}
        acc = {s: np.zeros(3) for s in "LR"}
        wsum = {s: 0.0 for s in "LR"}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            s = self.geom_side.get(g1) or self.geom_side.get(g2)
            if s is None:
                continue
            self.mj.mj_contactForce(self.m, self.d, i, self.f6)
            fr = np.asarray(c.frame).reshape(3, 3)          # rows: normal (geom[0] -> geom[1]), t1, t2
            f = fr.T @ self.f6[:3]                           # force on geom[1], world
            if g2 in self.geom_side:                         # the pad is geom[1]: the tool is geom[0]
                f = -f
            fn = float(self.f6[0])
            out[s]["F"] += f
            if fn > 1e-9:
                pos = np.array(c.pos)
                acc[s] += fn * pos
                wsum[s] += fn
                out[s]["n"] += 1
                if detail:
                    if self.area_s:
                        out[s]["pts"].append((pos, fr[0].copy(), fn / self.area_s,
                                              0.45 * self.sp["s"], "p"))
                    else:
                        out[s]["pts"].append((pos, fr[0].copy(), fn, 0.0012, "f"))
        for s in "LR":
            if wsum[s] > 0:
                out[s]["cop"] = acc[s] / wsum[s]
        return out


# -------------------------------------------------------------------------------------- Drake rig

class DrakeRollRig(H.DrakeRig):
    """hom_contact_rig.DrakeRig rebuilt with a carriage under the +x pad: the carriage (M_LIFT) slides
    along z on joint liftR, the pad slides along x on the carriage. Actuation vector [fL, fR, zR]."""

    def __init__(self, sp, d_cg=0.0, gravity=False):
        from pydrake.all import (
            AddCompliantHydroelasticProperties, AddContactMaterial, AddMultibodyPlant,
            AddRigidHydroelasticProperties, CoulombFriction, Cylinder, DiagramBuilder,
            FixedOffsetFrame, MultibodyPlantConfig, PrismaticJoint, ProximityProperties,
            RigidTransform, RotationMatrix, Simulator, SpatialInertia, Sphere,
        )
        self.sp, self.d_cg = sp, d_cg
        hydro = sp["model"] == "hydro"
        b = DiagramBuilder()
        cfg = MultibodyPlantConfig(time_step=sp.get("dt", H.DT), discrete_contact_approximation="sap",
                                   contact_model="hydroelastic_with_fallback" if hydro else "point")
        plant, sg = AddMultibodyPlant(cfg, b)
        if not gravity:
            plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, 0.0])

        def material(pp):
            AddContactMaterial(dissipation=sp.get("hc", 10.0), point_stiffness=sp.get("kp", 1e4),
                               friction=CoulombFriction(H.MU, H.MU), properties=pp)
            if "rt" in sp:
                pp.AddProperty("material", "relaxation_time", sp["rt"])
            return pp
        self.pad_bodies, self.joints = {}, {}
        for side, sgn in (("L", -1.0), ("R", 1.0)):
            body = plant.AddRigidBody("pad" + side, SpatialInertia.SolidSphereWithMass(H.M_PAD, H.R_PAD))
            pp = ProximityProperties()
            if hydro:
                AddCompliantHydroelasticProperties(sp["res"], sp["E"], pp)
            plant.RegisterCollisionGeometry(body, RigidTransform(), Sphere(H.R_PAD), "pad" + side, material(pp))
            self.pad_bodies[side] = body
            fr = plant.AddFrame(FixedOffsetFrame("slot" + side, plant.world_frame(),
                                                 RigidTransform([sgn * H.X0, 0.0, 0.0])))
            if side == "R":
                car = plant.AddRigidBody("carriageR", SpatialInertia.SolidSphereWithMass(M_LIFT, 0.01))
                self.lift = plant.AddJoint(PrismaticJoint("liftR", fr, car.body_frame(), [0.0, 0.0, 1.0]))
                fr = car.body_frame()
            j = plant.AddJoint(PrismaticJoint("rail" + side, fr, body.body_frame(), [1.0, 0.0, 0.0],
                                              damping=H.RAIL_DAMP))
            plant.AddJointActuator("f" + side, j)
            self.joints[side] = j
        plant.AddJointActuator("zR", self.lift)
        tool = plant.AddRigidBody("tool", SpatialInertia.SolidCylinderWithMass(
            H.M_TOOL, H.R_TOOL, 2 * H.HL_TOOL, [0.0, 0.0, 1.0]))
        tp = ProximityProperties()
        if hydro:
            AddRigidHydroelasticProperties(sp.get("res_tool", 0.0005), tp)
        plant.RegisterCollisionGeometry(tool, RigidTransform(), Cylinder(H.R_TOOL, 2 * H.HL_TOOL), "tool",
                                        material(tp))
        plant.Finalize()
        self.plant, self.sg, self.toolb = plant, sg, tool
        self.diagram = b.Build()
        self.simulator = Simulator(self.diagram)
        self.ctx = self.simulator.get_mutable_context()
        self.pc = plant.GetMyMutableContextFromRoot(self.ctx)
        self.sgc = sg.GetMyMutableContextFromRoot(self.ctx)
        plant.SetFreeBodyPose(self.pc, tool, RigidTransform(RotationMatrix.MakeXRotation(-math.pi / 2),
                                                            [0.0, d_cg, 0.0]))
        self.geom_side = {}
        for s, body in self.pad_bodies.items():
            for gid in plant.GetCollisionGeometriesForBody(body):
                self.geom_side[gid] = s
        self.kinematic = False
        self.N, self.u_lift = 0.0, 0.0
        self.f_tool, self.tau_tool = np.zeros(3), np.zeros(3)
        self._fix_inputs(0.0, np.zeros(3), np.zeros(3))
        self.simulator.Initialize()
        self.theta_prev, self.theta_unwrap = None, 0.0
        self.lift_mass = M_LIFT + H.M_PAD
        self.info = {}

    def _fix_inputs(self, N, f, tau):
        from pydrake.all import ExternallyAppliedSpatialForce, SpatialForce
        self.plant.get_actuation_input_port().FixValue(self.pc, np.array([N, -N, self.u_lift]))
        e = ExternallyAppliedSpatialForce()
        e.body_index = self.toolb.index()
        e.p_BoBq_B = np.zeros(3)
        e.F_Bq_W = SpatialForce(np.asarray(tau, float), np.asarray(f, float))
        self.plant.get_applied_spatial_force_input_port().FixValue(self.pc, [e])

    def set_lift_force(self, u):
        self.u_lift = float(u)
        self.plant.get_actuation_input_port().FixValue(self.pc, np.array([self.N, -self.N, self.u_lift]))

    def lift_state(self):
        return float(self.lift.get_translation(self.pc)), float(self.lift.get_translation_rate(self.pc))

    def tool_kin(self):
        X = self.plant.EvalBodyPoseInWorld(self.pc, self.toolb)
        V = self.plant.EvalBodySpatialVelocityInWorld(self.pc, self.toolb)
        return (np.array(X.translation()), np.array(X.rotation().matrix()), np.array(V.translational()),
                np.array(V.rotational()))

    def pad_kin(self):
        return {s: (np.array(self.plant.EvalBodyPoseInWorld(self.pc, b).translation()),
                    np.array(self.plant.EvalBodySpatialVelocityInWorld(self.pc, b).translational()))
                for s, b in self.pad_bodies.items()}

    def contact_patches(self, detail=False):
        """As MjRollRig.contact_patches; the contact point is the pressure-weighted centroid of the
        hydroelastic surface, the force the solver's resultant (body A of F_Ac_W is the pad, geometry M)."""
        out = {s: {"F": np.zeros(3), "cop": None, "n": 0, "pts": []} for s in "LR"}
        cr = self.plant.get_contact_results_output_port().Eval(self.pc)
        for i in range(cr.num_hydroelastic_contacts()):
            info = cr.hydroelastic_contact_info(i)
            srf = info.contact_surface()
            pad_is_M = srf.id_M() in self.geom_side
            s = self.geom_side.get(srf.id_M()) if pad_is_M else self.geom_side.get(srf.id_N())
            if s is None:
                continue
            F = np.array(info.F_Ac_W().translational())
            Np, cop, nf = H.surface_cop(srf)
            out[s].update(F=-F if pad_is_M else F, cop=cop if nf else None, n=nf, N_pressure=Np)
            if detail and nf:
                tri = srf.is_triangle()
                mesh = srf.tri_mesh_W() if tri else srf.poly_mesh_W()
                field = srf.tri_e_MN() if tri else srf.poly_e_MN()
                for f in range(nf):
                    C = np.array(mesh.element_centroid(f))
                    out[s]["pts"].append((C, np.array(mesh.face_normal(f)), float(field.EvaluateCartesian(f, C)),
                                          math.sqrt(mesh.area(f) / math.pi), "p"))
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            pp = info.point_pair()
            s = self.geom_side.get(pp.id_A) or self.geom_side.get(pp.id_B)
            if s is None:
                continue
            f = np.array(info.contact_force())                # on body B
            tool_is_B = info.bodyB_index() == self.toolb.index()
            out[s].update(F=f if tool_is_B else -f, cop=np.array(info.contact_point()), n=1)
            if detail:
                out[s]["pts"].append((np.array(info.contact_point()), np.array(pp.nhat_BA_W),
                                      float(np.linalg.norm(f)), 0.0012, "f"))
        return out


def make_roll_rig(spec, dt):
    H.DT = dt                    # the rigs read the module DT for the timestep and the step count
    sp = H.parse_spec(spec)
    return (DrakeRollRig if sp["sim"] == "drake" else MjRollRig)(sp, 0.0, False)


# ----------------------------------------------------------------------------------- measurement

def mat2quat(R):
    """Unit quaternion (w, x, y, z), w >= 0, of a rotation matrix."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        S = math.sqrt(t + 1.0) * 2
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.array(q)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def tool_angles(R, R0):
    """Roll (rotation about world y, signed so no-slip rolling under the rising +x pad is positive) and
    the tilt of the tool axis about world x and z, all relative to R0, in rad."""
    q = mat2quat(R @ R0.T)
    roll = -2.0 * math.atan2(q[2], q[0])
    a, a0 = R[:, 2], R0[:, 2]
    rot_x = math.atan2(a[2], a[1]) - math.atan2(a0[2], a0[1])
    rot_z = math.atan2(-a[0], a[1]) - math.atan2(-a0[0], a0[1])
    return roll, rot_x, rot_z


def contact_kinematics(p, R, v, w, pads, patches):
    """Per pad: contact point on the tool surface, outward tool normal there, tangential relative velocity
    of tool and pad at that point (circumferential and axial parts), normal and tangential force."""
    a = R[:, 2]
    res = {}
    for s in "LR":
        P = patches[s]
        if P["cop"] is None or P["n"] == 0:
            res[s] = None
            continue
        q = P["cop"] - p
        qa = float(q @ a)
        r = q - qa * a
        nr = float(np.linalg.norm(r))
        if nr < 1e-9:
            res[s] = None
            continue
        n = r / nr
        ps = P["cop"]                       # the reported contact point, not projected
        c_pad, v_pad = pads[s]
        v_rel = v + np.cross(w, ps - p) - v_pad
        vt = v_rel - float(v_rel @ n) * n
        e_c = np.cross(a, n)
        if e_c[2] < 0:
            e_c = -e_c
        F = P["F"]
        Fn = -float(F @ n)
        Ft = float(np.linalg.norm(F + Fn * n))
        res[s] = {"point": ps, "n": n, "vt": float(np.linalg.norm(vt)), "v_circ": float(vt @ e_c),
                  "v_ax": float(vt @ a), "Fn": Fn, "Ft": Ft, "count": P["n"], "r_c": nr}
    return res


def run_roll(rig, N, dt, V, T, film_every=None, on_frame=None):
    """Settle, roll, hold; returns (metrics dict, trace dict, step wall times). `on_frame(snapshot)` is
    called every `film_every` s of simulated time from FILM_PRE before the motion to the end."""
    servo = LiftServo(rig.lift_mass)
    rig.set_pad_force(N)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    n_settle = int(round(T_SETTLE / dt))
    n_film_pre = int(round(FILM_PRE / dt)) if film_every else 0
    stride_f = max(1, int(round(film_every / dt))) if film_every else 0
    t_mot = T + T_RAMP
    n_mot = int(round(t_mot / dt))
    n_hold = int(round(T_HOLD / dt))
    W = []

    def snapshot(t_rel, extra):
        p, R, v, w = rig.tool_kin()
        pk = rig.pad_kin()
        pat = rig.contact_patches(detail=True)
        return {"t": t_rel, "tool_p": p, "tool_q": mat2quat(R), "pads": {s: pk[s][0] for s in "LR"},
                "pts": pat["L"]["pts"] + pat["R"]["pts"], **extra}

    for k in range(n_settle):
        s_, v_ = rig.lift_state()
        rig.set_lift_force(servo.force((0.0, 0.0, 0.0), s_, v_))
        rig.step(dt)
        if on_frame and k >= n_settle - n_film_pre and (n_settle - 1 - k) % stride_f == 0:
            on_frame(snapshot((k + 1 - n_settle) * dt, {"s": 0.0, "roll": 0.0}))
    s0, _ = rig.lift_state()
    p0, R0, _, _ = rig.tool_kin()
    pads0 = rig.pad_kin()
    pat0 = rig.contact_patches()
    kin0 = contact_kinematics(p0, R0, np.zeros(3), np.zeros(3), pads0, pat0)
    D0 = float(np.linalg.norm(pads0["L"][0] - p0))
    settle = {f"Fn_{s}": (kin0[s]["Fn"] if kin0[s] else 0.0) for s in "LR"}
    settle.update({f"n_{s}": int(pat0[s]["n"]) for s in "LR"})
    settle["D_mm"] = D0 * 1e3
    settle["penetration_mm"] = (H.R_TOOL + H.R_PAD - D0) * 1e3

    cols = ["t", "s_mm", "dz_mm", "roll_deg", "dy_mm", "rotx_deg", "rotz_deg", "FnL", "FnR", "FtL", "FtR",
            "vslipL_mm_s", "vslipR_mm_s", "vcircL_mm_s", "vcircR_mm_s", "nL", "nR", "dev_mm"]
    rows = []
    acc = {s: {"path": 0.0, "circ": 0.0, "ax": 0.0, "path_hold": 0.0, "circ_hold": 0.0, "rc": 0.0, "nrc": 0}
           for s in "LR"}
    lost = {s: False for s in "LR"}
    status = "complete"
    end = {}
    for k in range(n_mot + n_hold):
        t_k = k * dt
        s_, v_ = rig.lift_state()
        ref = lift_ref(t_k, V, T)
        ref = (ref[0] + s0, ref[1], lift_ref(t_k + 0.5 * dt, V, T)[2])
        rig.set_lift_force(servo.force(ref, s_, v_))
        w0 = time.perf_counter()
        rig.step(dt)
        W.append(time.perf_counter() - w0)
        t = (k + 1) * dt
        s_, _ = rig.lift_state()
        s = s_ - s0
        p, R, v, w = rig.tool_kin()
        if not np.all(np.isfinite(p)) or np.linalg.norm(p - p0) > EJECT:
            status = "ejected"
            end.setdefault("t_eject", t)
            break
        pads = rig.pad_kin()
        pat = rig.contact_patches()
        kin = contact_kinematics(p, R, v, w, pads, pat)
        roll, rot_x, rot_z = tool_angles(R, R0)
        d = p - p0
        dev = float(np.linalg.norm(d - np.array([0.0, 0.0, 0.5 * s])))
        in_motion = k < n_mot
        for side in "LR":
            c = kin[side]
            if c is None or c["Fn"] < FN_LOST * N:
                lost[side] = True
            if c is None:
                continue
            if in_motion:
                acc[side]["rc"] += c["r_c"]
                acc[side]["nrc"] += 1
                acc[side]["path"] += c["vt"] * dt
                acc[side]["circ"] += c["v_circ"] * dt
                acc[side]["ax"] += c["v_ax"] * dt
            else:
                acc[side]["path_hold"] += c["vt"] * dt
                acc[side]["circ_hold"] += c["v_circ"] * dt

        def g(side, key, scale=1.0):
            return kin[side][key] * scale if kin[side] else float("nan")
        rows.append([t, s * 1e3, d[2] * 1e3, math.degrees(roll), d[1] * 1e3, math.degrees(rot_x),
                     math.degrees(rot_z), g("L", "Fn"), g("R", "Fn"), g("L", "Ft"), g("R", "Ft"),
                     g("L", "vt", 1e3), g("R", "vt", 1e3), g("L", "v_circ", 1e3), g("R", "v_circ", 1e3),
                     g("L", "count"), g("R", "count"), dev * 1e3])
        if k == n_mot - 1:
            end["motion"] = {"s": s, "roll": roll, "dz": d[2], "dy": d[1], "rot_x": rot_x, "rot_z": rot_z}
        if on_frame and (k + 1) % stride_f == 0:
            on_frame(snapshot(t, {"s": s, "roll": roll}))
    if status == "complete":
        end["hold"] = {"s": s, "roll": roll, "dz": d[2], "dy": d[1], "rot_x": rot_x, "rot_z": rot_z}
    tr = np.array(rows, float).reshape(-1, len(cols))
    met = {"status": status, "settle": settle, "acc": acc, "lost": lost, "end": end, "D0": D0}
    return met, (cols, tr), np.array(W)


def summarize(met, cols, tr, N, dt, V, T):
    """Row metrics from run_roll output."""
    ix = {c: i for i, c in enumerate(cols)}
    out = {"status": met["status"], "settle": met["settle"]}
    n_mot = int(round((T + T_RAMP) / dt))
    mot = tr[:n_mot]
    if met["status"] != "complete" or "motion" not in met["end"]:
        out.update(success=False, t_eject=met["end"].get("t_eject"))
        return out
    e, h = met["end"]["motion"], met["end"]["hold"]
    s, s_h = e["s"], h["s"]
    D0 = met["D0"]
    alpha = math.asin(min(0.5 * s / D0, 1.0))
    out.update(
        s_mm=s * 1e3,
        rho=e["roll"] * H.R_TOOL / (0.5 * s),
        rho_hold=h["roll"] * H.R_TOOL / (0.5 * s_h),
        rho_noslip_geom=alpha / math.sin(alpha) if alpha > 0 else 1.0,
        alpha_end_deg=math.degrees(alpha),
        z_ratio=e["dz"] / (0.5 * s),
        roll_end_deg=math.degrees(e["roll"]),
        roll_noslip_geom_deg=math.degrees(D0 * alpha / H.R_TOOL),
        dz_mm=e["dz"] * 1e3,
        dy_end_mm=e["dy"] * 1e3, dy_hold_mm=h["dy"] * 1e3,
        rot_x_end_deg=math.degrees(e["rot_x"]), rot_z_end_deg=math.degrees(e["rot_z"]),
        dy_max_mm=float(np.nanmax(np.abs(tr[:, ix["dy_mm"]]))),
        rot_x_max_deg=float(np.nanmax(np.abs(tr[:, ix["rotx_deg"]]))),
        rot_z_max_deg=float(np.nanmax(np.abs(tr[:, ix["rotz_deg"]]))),
        dev_max_mm=float(np.nanmax(tr[:, ix["dev_mm"]])),
        roll_hold_change_deg=math.degrees(h["roll"] - e["roll"]),
    )
    # rho over the cruise (pad travel 2 to 8 mm), free of the start transient
    sm = mot[:, ix["s_mm"]]
    m = (sm >= 2.0) & (sm <= 8.0)
    if m.sum() >= 3:
        i0, i1 = np.where(m)[0][[0, -1]]
        out["rho_cruise"] = (math.radians(mot[i1, ix["roll_deg"]] - mot[i0, ix["roll_deg"]]) * H.R_TOOL /
                             (0.5 * (sm[i1] - sm[i0]) * 1e-3))
    for side in "LR":
        a = met["acc"][side]
        out[f"slip_path_{side}_mm"] = a["path"] * 1e3
        out[f"slip_circ_{side}_mm"] = a["circ"] * 1e3
        out[f"slip_axial_{side}_mm"] = a["ax"] * 1e3
        out[f"slip_path_hold_{side}_mm"] = a["path_hold"] * 1e3
        fn = tr[:, ix["Fn" + side]]
        ft = tr[:, ix["Ft" + side]]
        out[f"Fn_{side}_min"] = float(np.nanmin(fn))
        out[f"Fn_{side}_max"] = float(np.nanmax(fn))
        out[f"Ft_{side}_max"] = float(np.nanmax(ft))
        out[f"util_{side}_max"] = float(np.nanmax(ft / (H.MU * np.maximum(fn, 1e-9))))
        out[f"vslip_{side}_max_mm_s"] = float(np.nanmax(tr[:, ix[f"vslip{side}_mm_s"]]))
        out[f"vslip_{side}_cruise_mm_s"] = float(np.nanmedian(mot[m, ix[f"vslip{side}_mm_s"]])) if m.any() else None
        out[f"n_{side}_min"] = float(np.nanmin(tr[:, ix["n" + side]]))
        out[f"n_{side}_max"] = float(np.nanmax(tr[:, ix["n" + side]]))
    out["Fn_ideal_end"] = N / math.cos(alpha)
    rc = [met["acc"][sd]["rc"] / met["acc"][sd]["nrc"] for sd in "LR" if met["acc"][sd]["nrc"]]
    for sd in "LR":
        if met["acc"][sd]["nrc"]:
            out[f"r_contact_{sd}_mm"] = met["acc"][sd]["rc"] / met["acc"][sd]["nrc"] * 1e3
    if rc:
        out["rho_noslip_contact"] = out["rho_noslip_geom"] * H.R_TOOL / float(np.mean(rc))
        out["rho_over_noslip"] = out["rho"] / out["rho_noslip_contact"]
    # kinematic identity: rho - 1 = (slip_R - slip_L) / s, z_ratio - 1 = (slip_R + slip_L) / s (flat limit)
    if rc:
        out["rho_from_slip"] = out["rho_noslip_contact"] * (1.0 + (met["acc"]["R"]["circ"] - met["acc"]["L"]["circ"]) / s)
    out["r_roll_mm"] = H.R_TOOL * out["rho_noslip_geom"] / out["rho"] * 1e3 if out["rho"] > 0 else None
    out["z_ratio_from_slip"] = 1.0 + (met["acc"]["R"]["circ"] + met["acc"]["L"]["circ"]) / s
    out["contact_lost"] = {s_: bool(v) for s_, v in met["lost"].items()}
    out["skid"] = {sd: bool(met["acc"][sd]["path"] > SKID_FRAC * 0.5 * s) for sd in "LR"}
    out["success"] = bool(not any(met["lost"].values()) and out["dev_max_mm"] <= PATH_TOL * 1e3)
    return out


# ------------------------------------------------------------------------------------------ films

def heat(u):
    u = float(np.clip(u, 0.0, 1.0))
    cold, warm, hot = np.array([0.25, 0.45, 0.85]), np.array([1.0, 0.78, 0.15]), np.array([0.85, 0.1, 0.08])
    c = cold * (1 - 2 * u) + warm * (2 * u) if u < 0.5 else warm * (2 - 2 * u) + hot * (2 * u - 1)
    return np.array([*c, 1.0], dtype=np.float32)


def film_xml():
    """Drawing model: the point3 roll rig, pads and tool translucent, roll markers on the tool."""
    xml, _ = roll_mj_xml(H.parse_spec("mj:point3"), 0.0, False)
    xml = re.sub(r'(name="pad[LR]"[^>]*rgba=")[^"]*"', r'\g<1>0.85 0.55 0.35 0.28"', xml)
    xml = re.sub(r'(<geom name="tool"[^>]*rgba=")[^"]*"', r'\g<1>0.55 0.6 0.68 0.42"', xml)
    R, HL = H.R_TOOL, H.HL_TOOL
    xml, n = re.subn(r'<geom type="box" pos="0 0 [0-9.e-]+" size="0.0016[^>]*/>', "", xml)   # rig's +y end mark
    if n != 1:
        raise RuntimeError("tool end marker not found in the rig MJCF")
    marks = []
    for zf in (-HL,):             # the end face the camera sees: a radius stripe with a red tip, a cross line
        marks.append(f'<geom type="box" pos="0 {-0.45 * R:.6f} {zf:.6f}" size="0.0011 {0.45 * R:.6f} 0.0004" '
                     f'contype="0" conaffinity="0" rgba="0.12 0.12 0.14 1"/>')
        marks.append(f'<geom type="cylinder" pos="0 {-0.85 * R:.6f} {zf:.6f}" size="0.0018 0.00045" '
                     f'contype="0" conaffinity="0" rgba="0.8 0.15 0.12 1"/>')
        marks.append(f'<geom type="box" pos="0 0 {zf:.6f}" size="{0.7 * R:.6f} 0.0005 0.0003" '
                     f'contype="0" conaffinity="0" rgba="0.12 0.12 0.14 0.6"/>')
    marks.append(f'<geom type="box" pos="0 {-R:.6f} 0" size="0.0009 0.0003 {HL:.6f}" contype="0" '
                 f'conaffinity="0" rgba="0.8 0.15 0.12 0.9"/>')
    xml = re.sub(r'(<geom name="tool"[^>]*/>)', lambda m: m.group(1) + "\n      " + "\n      ".join(marks), xml)
    return xml


def label(img, top, sub, bottom, size=13):
    """White bars: model label and case line on top, time line at the bottom."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
        mono = ImageFont.truetype("DejaVuSansMono.ttf", size - 1)
    except Exception:
        font = mono = ImageFont.load_default()
    dr.rectangle([0, 0, im.width, 2 * size + 12], fill=(255, 255, 255))
    dr.text((8, 3), top, fill=(20, 20, 20), font=font)
    dr.text((8, size + 7), sub, fill=(70, 70, 70), font=font)
    dr.rectangle([0, im.height - size - 10, im.width, im.height], fill=(255, 255, 255))
    dr.text((8, im.height - size - 6), bottom, fill=(20, 20, 20), font=mono)
    return np.asarray(im)


class RollFilm:
    """Draws any roll rig from snapshots (pad centres, tool pose, contacts) through a MuJoCo copy of the
    geometry. Near-orthographic view (fovy 12 deg) 8 deg off the tool axis, from the -y end."""

    def __init__(self, w=FILM_WH[0], h=FILM_WH[1], azimuth=98.0, elevation=-8.0, distance=0.32,
                 lookat=(0.0, 0.0, 0.004), fovy=12.0):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_string(film_xml())
        self.m.vis.global_.fovy = fovy
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w, max_geom=4000)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.qa = {n: int(self.m.joint(n).qposadr[0]) for n in ("railL", "railR", "liftR")}
        tj = self.m.body_jntadr[self.m.body("tool").id]
        self.tq = int(self.m.jnt_qposadr[tj])

    def frame(self, snap, top, sub, bottom):
        mj = self.mj
        self.d.qpos[:] = 0
        self.d.qpos[self.qa["railL"]] = snap["pads"]["L"][0] + H.X0
        self.d.qpos[self.qa["railR"]] = snap["pads"]["R"][0] - H.X0
        self.d.qpos[self.qa["liftR"]] = snap["pads"]["R"][2]
        self.d.qpos[self.tq:self.tq + 3] = snap["tool_p"]
        self.d.qpos[self.tq + 3:self.tq + 7] = snap["tool_q"]
        mj.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, self.cam)
        scn = self.r.scene
        for pos, nrm, val, rad, kind in snap["pts"]:
            if scn.ngeom >= scn.maxgeom:
                break
            u = val / (P_SCALE if kind == "p" else F_SCALE)
            rr = 0.0012 if kind == "f" else float(np.clip(rad, 0.0005, 0.0006))
            mj.mjv_initGeom(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_SPHERE, np.array([rr, rr, rr]),
                            np.asarray(pos, float), np.eye(3).ravel(), heat(u))
            scn.ngeom += 1
        img = self.r.render().copy()
        return label(img, top, sub, bottom)


def encode(path, frames, fps=FILM_FPS, crf=27):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-threads", "1",
           "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError(f"ffmpeg failed for {path}")


def film_label(model, N, dt_ms, v_mm_s):
    return H.label_of(MODELS[model]), f"N {N:g} N, pad {v_mm_s:g} mm/s, dt {dt_ms:g} ms; {scale_text(model)}"


def film_bottom(snap):
    s = snap["s"]
    rho_txt = f"rho {snap['roll'] * H.R_TOOL / (0.5 * s):5.3f}" if s > 1e-4 else "rho   -"
    return (f"t {snap['t']:5.2f} s  x{FILM_SPEED:g}  pad +{s * 1e3:4.1f} mm  roll {math.degrees(snap['roll']):5.1f} deg  "
            f"{rho_txt}")


def scale_text(model):
    return "contacts: force 0-2 N" if model in ("mj_point3", "mj_point4s") else "contacts: pressure 0-0.3 MPa"


# ----------------------------------------------------------------------------------------- cases

def git_rev():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def run_case(model, N, dt_ms, v_mm_s, film=False):
    spec = MODELS[model]
    dt, V, T = dt_ms * 1e-3, v_mm_s * 1e-3, SPEEDS[v_mm_s]
    t_wall = time.time()
    row = {"task": "roll", "model": model, "rig_spec": spec, "N": N, "dt_ms": dt_ms, "v_mm_s": v_mm_s,
           "T_s": T, "travel_mm": v_mm_s * T, "ramp_ms": T_RAMP * 1e3, "T_settle_s": T_SETTLE,
           "T_hold_s": T_HOLD, "mu": H.MU, "gravity": False, "lift_mass_kg": M_LIFT + H.M_PAD,
           "lift_wn_rad_s": LIFT_WN}
    snaps = []
    try:
        rig = make_roll_rig(spec, dt)
        row["info"] = getattr(rig, "info", {})
        met, (cols, tr), W = run_roll(rig, N, dt, V, T, film_every=(FILM_SPEED / FILM_FPS) if film else None,
                                      on_frame=snaps.append if film else None)
        row.update(summarize(met, cols, tr, N, dt, V, T))
        row["us_per_step_median"] = float(np.median(W) * 1e6) if len(W) else None
        row["us_per_step_mean"] = float(np.mean(W) * 1e6) if len(W) else None
        stride = max(1, int(round(0.010 / dt)))
        row["trace_cols"] = cols
        row["trace"] = [[round(float(x), 6) for x in r] for r in tr[stride - 1::stride]]
    except Exception as exc:  # noqa: BLE001
        import traceback
        row.update(status="failed", success=False, error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc()[-2000:])
    row["film"] = None
    if film and snaps:
        try:
            row["film"] = render_film(model, snaps, N, dt_ms, v_mm_s)
        except Exception as exc:  # noqa: BLE001
            row["film_error"] = f"{type(exc).__name__}: {exc}"
    row["wall_s"] = time.time() - t_wall
    row["script"] = SCRIPT
    row["git_rev"] = git_rev()
    row["when"] = time.strftime("%Y-%m-%d %H:%M")
    return row


def render_film(model, snaps, N, dt_ms, v_mm_s):
    fr = RollFilm()
    top, sub = film_label(model, N, dt_ms, v_mm_s)
    frames = [fr.frame(sn, top, sub, film_bottom(sn)) for sn in snaps]
    rel = f"media/roll_{model}.mp4"
    encode(OUT_DIR / rel, frames)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FRAMES_DIR / f"frames_{model}.npy", np.stack(frames))
    return rel


def done(path):
    have = set()
    if Path(path).exists():
        for line in open(path):
            try:
                r = json.loads(line)
                have.add((r["model"], float(r["N"]), float(r["dt_ms"]), float(r["v_mm_s"])))
            except Exception:
                pass
    return have


def grid(models, Ns, dts, vs, consistency=True):
    cases = [(m, N, dt, v) for m in models for dt in dts for N in Ns for v in vs]
    if consistency:
        cases += [("mj_pads1", 1.0, dt, v) for dt in (0.5, 2.0) for v in SPEEDS]
        cases += [(m, 1.0, 1.0, v) for m in ("mj_pads2", "mj_pads05") for v in SPEEDS]
    return cases


def print_row(r):
    keys = ("rho", "rho_noslip_contact", "rho_over_noslip", "rho_from_slip", "rho_cruise", "z_ratio",
            "r_contact_L_mm", "slip_circ_L_mm", "slip_circ_R_mm", "slip_path_L_mm", "slip_path_R_mm", "dy_end_mm", "rot_x_max_deg",
            "rot_z_max_deg", "Fn_L_min", "Fn_L_max", "util_R_max", "dev_max_mm", "us_per_step_median", "wall_s")
    print(r["model"], r["N"], r["dt_ms"], r["v_mm_s"], r.get("status"), r.get("success"),
          {k: round(r[k], 4) for k in keys if isinstance(r.get(k), (int, float))}, r.get("film") or "",
          r.get("error", ""), flush=True)


def cmd_run(a):
    have = done(a.out)
    cases = grid(a.models, a.N, a.dt, a.v, consistency=not a.no_consistency)
    for (m, N, dt, v) in cases:
        if (m, float(N), float(dt), float(v)) in have:
            continue
        film = (not a.no_films and N == FILM_N and dt == FILM_DT and v == FILM_V)
        r = run_case(m, N, dt, v, film=film)
        H.append_row(a.out, r)
        have.add((m, float(N), float(dt), float(v)))
        print_row(r)


def cmd_frame(a):
    """Camera check: one PNG with the settled pinch, mid-roll and end of roll side by side."""
    snaps = []
    H.DT = FILM_DT * 1e-3
    rig = make_roll_rig(MODELS[a.model], FILM_DT * 1e-3)
    run_roll(rig, FILM_N, FILM_DT * 1e-3, FILM_V * 1e-3, SPEEDS[FILM_V], film_every=FILM_SPEED / FILM_FPS,
             on_frame=snaps.append)
    fr = RollFilm()
    pick = [snaps[0], snaps[len(snaps) // 2], snaps[-1]]
    imgs = [fr.frame(sn, *film_label(a.model, FILM_N, FILM_DT, FILM_V), film_bottom(sn)) for sn in pick]
    from PIL import Image
    Image.fromarray(np.concatenate(imgs, axis=1)).save(a.out)
    print(a.out, len(snaps), "frames")


def cmd_films(a):
    """Re-run the film cases (no rows) and render media/roll_<model>.mp4."""
    for m in a.models:
        snaps = []
        rig = make_roll_rig(MODELS[m], FILM_DT * 1e-3)
        run_roll(rig, FILM_N, FILM_DT * 1e-3, FILM_V * 1e-3, SPEEDS[FILM_V], film_every=FILM_SPEED / FILM_FPS,
                 on_frame=snaps.append)
        print(m, render_film(m, snaps, FILM_N, FILM_DT, FILM_V), flush=True)


def cmd_tile(a):
    """2 x 2 tile of the per-model films (raw frames kept in FRAMES_DIR) and a JPEG poster."""
    from PIL import Image
    stacks = [np.load(FRAMES_DIR / f"frames_{m}.npy") for m in a.models]
    n = min(len(s) for s in stacks)
    cols = 2
    rows_ = int(math.ceil(len(stacks) / cols))
    h, w = stacks[0].shape[1:3]
    blank = np.full((h, w, 3), 255, np.uint8)
    frames = []
    for i in range(n):
        tiles = [s[i] for s in stacks] + [blank] * (rows_ * cols - len(stacks))
        frames.append(np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1)
                                      for r in range(rows_)], axis=0))
    encode(OUT_DIR / "media/roll_models.mp4", frames)
    i_poster = min(n - 1, int(round((FILM_PRE + 0.8 * (SPEEDS[FILM_V] + T_RAMP)) / (FILM_SPEED / FILM_FPS))))
    Image.fromarray(frames[i_poster]).save(OUT_DIR / "media/roll_models_poster.jpg", quality=88)
    print("media/roll_models.mp4", n, "frames; poster frame", i_poster)


def cmd_table(a):
    rows = [json.loads(line) for line in open(a.out)]
    hdr = ("model", "N", "dt", "v", "rho", "rho_cr", "z_rat", "slipL", "slipR", "dy", "rotx", "rotz", "FnL",
           "FnL_max", "utilR", "dev", "us", "ok")
    print(" ".join(f"{h:>8}" for h in hdr))
    for r in rows:
        def f(k, nd=3):
            v = r.get(k)
            return f"{v:8.{nd}f}" if isinstance(v, (int, float)) else f"{'-':>8}"
        print(f"{r['model']:>8.8} {r['N']:8g} {r['dt_ms']:8g} {r['v_mm_s']:8g} {f('rho', 4)} {f('rho_cruise', 4)} "
              f"{f('z_ratio', 4)} {f('slip_path_L_mm')} {f('slip_path_R_mm')} {f('dy_end_mm')} "
              f"{f('rot_x_max_deg')} {f('rot_z_max_deg')} {f('Fn_L_min')} {f('Fn_L_max')} {f('util_R_max')} "
              f"{f('dev_max_mm')} {f('us_per_step_median', 1)} {str(r.get('success')):>8}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", default=MAIN_MODELS, choices=list(MODELS))
    r.add_argument("--N", nargs="+", type=float, default=[0.5, 1.0, 3.0])
    r.add_argument("--dt", nargs="+", type=float, default=[1.0, 5.0], help="physics step, ms")
    r.add_argument("--v", nargs="+", type=float, default=list(SPEEDS), help="pad speed, mm/s (10 or 50)")
    r.add_argument("--no-consistency", action="store_true", help="skip the mj_pads1/2/05 consistency set")
    r.add_argument("--no-films", action="store_true")
    r.add_argument("--out", default=str(OUT))
    f = sub.add_parser("frame")
    f.add_argument("--model", default="mj_pads1", choices=list(MODELS))
    f.add_argument("--out", required=True)
    fl = sub.add_parser("films")
    fl.add_argument("--models", nargs="+", default=MAIN_MODELS, choices=list(MODELS))
    t = sub.add_parser("tile")
    t.add_argument("--models", nargs="+", default=MAIN_MODELS, choices=list(MODELS))
    tb = sub.add_parser("table")
    tb.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    {"run": cmd_run, "frame": cmd_frame, "films": cmd_films, "tile": cmd_tile, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
