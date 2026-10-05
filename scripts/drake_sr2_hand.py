"""Exploratory Drake model of the fixed-palm real_v1 SR² hand.

This is a CPU planning/simulation sidecar, not a PhysicsBackend implementation.
Run in the isolated environment documented by drake_sr2_probe.py. The existing
MJCF is the geometry source; MuJoCo resolves its actuator defaults. Drake is an
optional dependency and no hardware or GPU stack is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ContactModel,
    DiagramBuilder,
    DiscreteContactApproximation,
    Parser,
    PdControllerGains,
)

ROOT = Path(__file__).resolve().parents[1]
HAND = ROOT / "assets/mjcf/real_v1/real_hand.xml"
ACTUATED = ROOT / "assets/mjcf/real_v1/real_hand_morphology_actuated.xml"
FINGERS = ("thumb", "index", "middle")
MORPH_NAMES = tuple(f"{f}_{a}" for f in FINGERS for a in ("x", "y"))
FINGER_NAMES = tuple(f"{f}_{a}" for f in FINGERS for a in ("yaw", "mcp", "pip"))


def checked(values, size, lower, upper, label):
    values = np.asarray(values, dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain {size} finite values")
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError(f"{label} exceeds the source MJCF joint limits")
    return values


@dataclass
class DrakeHand:
    diagram: object
    plant: object
    scene_graph: object
    model_instance: object
    mode: str
    morphology: np.ndarray
    reference: object

    def plant_context(self, root_context):
        return self.plant.GetMyMutableContextFromRoot(root_context)

    def new_context(self, morphology=None, fingers=None):
        """Reset configuration, then lock gantries if requested by the model mode.

        Changing morphology here represents a new trial, not physical motion.
        For motion, use mode='actuated' and command a time-varying desired state.
        """
        m = self.morphology if morphology is None else np.asarray(morphology, dtype=float)
        lo, hi = self.limits(MORPH_NAMES)
        m = checked(m, 6, lo, hi, "morphology offsets (m)")
        if self.mode == "frozen" and not np.array_equal(m, self.morphology):
            raise ValueError("A frozen hand requires a rebuild for a different morphology")
        q = np.tile([0.0, 0.55, 0.55], 3) if fingers is None else fingers
        lo, hi = self.limits(FINGER_NAMES)
        q = checked(q, 9, lo, hi, "finger angles (rad)")
        root = self.diagram.CreateDefaultContext()
        context = self.plant_context(root)
        for name, value in zip(FINGER_NAMES, q):
            self.plant.GetJointByName(name).set_angle(context, value)
        if self.mode != "frozen":
            for name, value in zip(MORPH_NAMES, m):
                joint = self.plant.GetJointByName(name)
                joint.set_translation(context, value)
                if self.mode == "locked":
                    joint.Lock(context)
        self.command(root, m, q)
        return root

    def limits(self, names):
        ranges = np.array([self.reference.jnt_range[self.reference.joint(n).id] for n in names])
        return ranges[:, 0], ranges[:, 1]

    def command(self, root_context, morphology, fingers):
        """Set PD position targets with zero desired velocity, in actuator order.

        The caller supplies rate-limited targets for an actual motion profile.
        Commands and efforts have separate bounds; never pass angle targets to
        Drake's feedforward force/torque port.
        """
        lo, hi = self.limits(MORPH_NAMES)
        m = checked(morphology, 6, lo, hi, "morphology offsets (m)")
        if self.mode == "frozen" and not np.array_equal(m, self.morphology):
            raise ValueError("A frozen hand requires a rebuild for a different morphology")
        lo, hi = self.limits(FINGER_NAMES)
        q = checked(fingers, 9, lo, hi, "finger angles (rad)")
        targets = dict(zip(MORPH_NAMES + FINGER_NAMES, np.r_[m, q]))
        positions = [targets[self.plant.get_joint_actuator(i).joint().name()]
                     for i in self.plant.GetJointActuatorIndices(self.model_instance)]
        context = self.plant_context(root_context)
        self.plant.get_actuation_input_port().FixValue(context, np.zeros(len(positions)))
        self.plant.get_desired_state_input_port(self.model_instance).FixValue(
            context, np.r_[positions, np.zeros(len(positions))])

    def tips(self, root_context):
        context = self.plant_context(root_context)
        return np.array([self.plant.EvalBodyPoseInWorld(
            context, self.plant.GetBodyByName(f"{f}_tip")).translation() for f in FINGERS])


def build_hand(mode="locked", morphology=None, time_step=0.002):
    """Build 9-DoF frozen or 15-DoF locked/actuated hand; palm is fixed to world.

    Gantry force servos retain the shipped MJCF gains and damping. These are
    numerical placeholders, not identified stepper dynamics. All contacts use
    Drake point contact; contact parameters are deliberately not called parity.
    """
    if mode not in {"frozen", "locked", "actuated"}:
        raise ValueError("mode must be frozen, locked, or actuated")
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("PD actuators require a positive discrete time step")
    reference = mujoco.MjModel.from_xml_path(str(HAND))
    actuator_reference = mujoco.MjModel.from_xml_path(str(ACTUATED))
    limits = np.array([reference.jnt_range[reference.joint(n).id] for n in MORPH_NAMES])
    m = checked(np.zeros(6) if morphology is None else morphology, 6,
                limits[:, 0], limits[:, 1], "morphology offsets (m)")
    root = ET.parse(HAND).getroot()
    # Restricted adapter for this primitive-only model. Keep body inertials,
    # shapes, transforms, material colors, joint limits and contact exclusions.
    # Remove display-only features, the noncolliding floor, and unsupported
    # solver settings explicitly rather than accepting unnoticed warnings.
    for tag in ("visual", "keyframe", "actuator"):
        root.remove(root.find(tag))
    for attr in ("timestep", "integrator"):
        root.find("option").attrib.pop(attr, None)
    for parent in root.iter():
        parent.attrib.pop("gravcomp", None)  # palm is welded; it cannot fall
        for child in list(parent):
            if child.tag in {"site", "light", "texture", "position", "motor"}:
                parent.remove(child)
            elif child.tag == "geom" and child.get("name") == "floor":
                parent.remove(child)
        if parent.tag == "material":
            for attr in ("texture", "texuniform", "texrepeat", "reflectance"):
                parent.attrib.pop(attr, None)
        if parent.tag == "geom" and "friction" in parent.attrib:
            parent.set("friction", parent.get("friction").split()[0] + " 0 0")
        if parent.tag == "body":
            for child in list(parent):
                if child.tag == "joint" and child.get("name", "").endswith("_len"):
                    # Zero-range legacy generator shim; no physical length DoF.
                    parent.remove(child)
    if mode == "frozen":
        for f, xy in zip(FINGERS, m.reshape(3, 2)):
            mount = root.find(f".//body[@name='{f}_mount']")
            position = np.fromstring(mount.get("pos"), sep=" ")
            position[:2] += xy
            mount.set("pos", " ".join(format(v, ".17g") for v in position))
            for joint in list(mount.findall("joint")):
                mount.remove(joint)
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    plant.set_contact_model(ContactModel.kPoint)
    parser = Parser(plant, scene_graph)
    parser.SetStrictParsing()
    instance, = parser.AddModelsFromString(ET.tostring(root, encoding="unicode"), "xml")
    names = FINGER_NAMES if mode == "frozen" else MORPH_NAMES + FINGER_NAMES
    for name in names:
        joint = plant.GetJointByName(name)
        source_joint = actuator_reference.joint(name).id
        source_actuator = actuator_reference.actuator("a_" + name).id
        force = actuator_reference.actuator_forcerange[source_actuator]
        if not np.isclose(force[0], -force[1]):
            raise ValueError("This prototype requires symmetric effort limits")
        actuator = plant.AddJointActuator("a_" + name, joint, float(force[1]))
        actuator.set_controller_gains(PdControllerGains(
            p=float(actuator_reference.actuator_gainprm[source_actuator, 0]),
            d=float(-actuator_reference.actuator_biasprm[source_actuator, 2])))
        actuator.set_default_rotor_inertia(
            float(actuator_reference.dof_armature[actuator_reference.jnt_dofadr[source_joint]]))
        actuator.set_default_gear_ratio(1.0)
    plant.Finalize()
    return DrakeHand(builder.Build(), plant, scene_graph, instance, mode, m.copy(), reference)
