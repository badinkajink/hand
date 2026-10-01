"""Drake side of scripts/hom_hand_brake.py: the same D8 hand and tool, Drake 1.57 SAP.

The MJCF that hom_hand_brake.mj_scene writes for MuJoCo is the geometry source, parsed the way
scripts/drake_sr2_hand.py (the 2026-10-01 Drake port) parses real_hand.xml: display elements, the
actuators and the solver options are stripped, every finger joint gets a PD-controlled actuator
with the calibrated servo's gains (kp 0.5, kd 0.02, effort 0.35 N m, rotor inertia = MJCF armature).
The fingertip geometry (tip sphere and distal capsule of each finger) is re-assigned compliant
hydroelastic properties (E, 1 mm resolution, relaxation time rt) and the tool rigid hydroelastic
(0.5 mm); everything else keeps point contact through hydroelastic_with_fallback.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

import hom_contact_rig as H
import hom_hand_brake as B


class DrakeHand:
    sim = "drake"

    def __init__(self, spec, offsets, d_cg, q_init):
        from pydrake.all import (
            AddCompliantHydroelasticProperties, AddContactMaterial, AddMultibodyPlant,
            AddRigidHydroelasticProperties, CoulombFriction, DiagramBuilder, MultibodyPlantConfig,
            Parser, PdControllerGains, ProximityProperties, RoleAssign, Simulator,
        )
        self.sp = B.parse_spec(spec)
        self.offsets, self.d_cg, self.q_init = offsets, d_cg, np.asarray(q_init, float)
        hydro = self.sp["model"] == "hydro"
        xml, _ = B.mj_scene("mj:point3", offsets, d_cg, q_init)
        root = ET.fromstring(xml)
        for tag in ("visual", "keyframe", "actuator", "option", "statistic", "size"):
            for el in root.findall(tag):
                root.remove(el)
        for parent in root.iter():
            for child in list(parent):
                if child.tag in {"site", "light", "texture", "position", "motor", "camera"}:
                    parent.remove(child)
                elif child.tag == "geom" and child.get("name") == "floor":
                    parent.remove(child)
            if parent.tag == "material":
                for attr in ("texture", "texuniform", "texrepeat", "reflectance"):
                    parent.attrib.pop(attr, None)
            if parent.tag in ("geom", "default") or parent.tag == "joint":
                for attr in ("solref", "solimp", "condim", "frictionloss", "contype", "conaffinity", "group"):
                    parent.attrib.pop(attr, None)
            if "friction" in parent.attrib:
                parent.set("friction", f"{H.MU} 0 0")
        b = DiagramBuilder()
        cfg = MultibodyPlantConfig(time_step=H.DT, discrete_contact_approximation="sap",
                                   contact_model="hydroelastic_with_fallback" if hydro else "point")
        plant, sg = AddMultibodyPlant(cfg, b)
        parser = Parser(plant, sg)
        self.instance, = parser.AddModelsFromString(ET.tostring(root, encoding="unicode"), "xml")
        # PD servos with the calibrated plant's gains
        ref_arm = 0.001
        for f in B.FINGERS:
            for n in B.JOINTS[f]:
                j = plant.GetJointByName(n)
                act = plant.AddJointActuator("a_" + n, j, B.PLANT["forcerange"])
                act.set_controller_gains(PdControllerGains(p=B.PLANT["kp"], d=B.PLANT["kv"]))
                act.set_default_rotor_inertia(ref_arm)
                act.set_default_gear_ratio(1.0)
        # contact properties
        sid = plant.get_source_id()
        inspector = sg.model_inspector()
        self.geom_finger, tool_gids = {}, []
        for f in B.FINGERS:
            for bn in (f"{f}_pip_frame", f"{f}_tip"):
                for gid in plant.GetCollisionGeometriesForBody(plant.GetBodyByName(bn)):
                    self.geom_finger[gid] = f
                    pp = ProximityProperties()
                    if hydro:
                        AddCompliantHydroelasticProperties(self.sp["res"], self.sp["E"], pp)
                    AddContactMaterial(dissipation=10.0, point_stiffness=1e4,
                                       friction=CoulombFriction(H.MU, H.MU), properties=pp)
                    if "rt" in self.sp:
                        pp.AddProperty("material", "relaxation_time", self.sp["rt"])
                    sg.AssignRole(sid, gid, pp, RoleAssign.kReplace)
        tool = plant.GetBodyByName("tool")
        for gid in plant.GetCollisionGeometriesForBody(tool):
            tool_gids.append(gid)
            pp = ProximityProperties()
            if hydro:
                AddRigidHydroelasticProperties(0.0005, pp)
            AddContactMaterial(dissipation=10.0, point_stiffness=1e4,
                               friction=CoulombFriction(H.MU, H.MU), properties=pp)
            if "rt" in self.sp:
                pp.AddProperty("material", "relaxation_time", self.sp["rt"])
            sg.AssignRole(sid, gid, pp, RoleAssign.kReplace)
        self.tool_gids = set(tool_gids)
        plant.Finalize()
        self.plant, self.sg, self.toolb = plant, sg, tool
        self.diagram = b.Build()
        self.simulator = Simulator(self.diagram)
        self.ctx = self.simulator.get_mutable_context()
        self.pc = plant.GetMyMutableContextFromRoot(self.ctx)
        self.joint_names = [n for f in B.FINGERS for n in B.JOINTS[f]]
        for n, q in zip(self.joint_names, self.q_init):
            plant.GetJointByName(n).set_angle(self.pc, float(q))
        self.act_order = [plant.get_joint_actuator(i).joint().name()
                          for i in plant.GetJointActuatorIndices(self.instance)]
        self.N = None
        self.f_tool, self.tau_tool = np.zeros(3), np.zeros(3)
        self.set_targets(self.q_init)
        self._fix_wrench()
        self.simulator.Initialize()

    @property
    def t(self):
        return float(self.ctx.get_time())

    def set_targets(self, q9):
        tgt = dict(zip(self.joint_names, np.asarray(q9, float)))
        pos = np.array([tgt[n] for n in self.act_order])
        self.plant.get_actuation_input_port().FixValue(self.pc, np.zeros(len(pos)))
        self.plant.get_desired_state_input_port(self.instance).FixValue(self.pc, np.r_[pos, np.zeros(len(pos))])

    def _fix_wrench(self):
        from pydrake.all import ExternallyAppliedSpatialForce, SpatialForce
        e = ExternallyAppliedSpatialForce()
        e.body_index = self.toolb.index()
        e.p_BoBq_B = np.zeros(3)
        e.F_Bq_W = SpatialForce(self.tau_tool, self.f_tool)
        self.plant.get_applied_spatial_force_input_port().FixValue(self.pc, [e])

    def set_tool_wrench(self, f, tau):
        self.f_tool, self.tau_tool = np.asarray(f, float), np.asarray(tau, float)
        self._fix_wrench()

    def step(self, T):
        self.simulator.AdvanceTo(self.ctx.get_time() + T)

    def contacts(self):
        out = {f: {"N": 0.0, "n": 0} for f in B.FINGERS}
        cr = self.plant.get_contact_results_output_port().Eval(self.pc)
        for i in range(cr.num_hydroelastic_contacts()):
            info = cr.hydroelastic_contact_info(i)
            srf = info.contact_surface()
            ids = (srf.id_M(), srf.id_N())
            if not (set(ids) & self.tool_gids):
                continue
            f = self.geom_finger.get(ids[0]) or self.geom_finger.get(ids[1])
            if f is None:
                continue
            F = np.array(info.F_Ac_W().translational())
            X = self.plant.EvalBodyPoseInWorld(self.pc, self.toolb)
            a = X.rotation().matrix()[:, 2]
            r = np.array(srf.centroid()) - X.translation()
            r -= (r @ a) * a                      # radial direction of the tool surface at the patch
            out[f]["N"] += float(abs(F @ r) / max(np.linalg.norm(r), 1e-9))
            out[f]["n"] += srf.num_faces()
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            pp = info.point_pair()
            if not ({pp.id_A, pp.id_B} & self.tool_gids):
                continue
            f = self.geom_finger.get(pp.id_A) or self.geom_finger.get(pp.id_B)
            if f is None:
                continue
            out[f]["N"] += float(abs(np.array(info.contact_force()) @ np.array(pp.nhat_BA_W)))
            out[f]["n"] += 1
        return out

    def tool_state(self):
        X = self.plant.EvalBodyPoseInWorld(self.pc, self.toolb)
        R = X.rotation().matrix()
        V = self.plant.EvalBodySpatialVelocityInWorld(self.pc, self.toolb)
        return {"R": R, "axis": R[:, 2].copy(), "pos": np.array(X.translation()), "w": np.array(V.rotational())}

    def copy_to_mujoco(self, m, d):
        for n in self.joint_names:
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = self.plant.GetJointByName(n).get_angle(self.pc)
        X = self.plant.EvalBodyPoseInWorld(self.pc, self.toolb)
        qa = m.jnt_qposadr[m.body_jntadr[m.body("tool").id]]
        d.qpos[qa:qa + 3] = X.translation()
        d.qpos[qa + 3:qa + 7] = X.rotation().ToQuaternion().wxyz()
