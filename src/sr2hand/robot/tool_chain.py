"""Tools on the UR5's flange as data: a chain of named rigid transforms from the flange E to a contact point, and the
inverse kinematics of a contact-point target, which is ur_kinematics' of the target composed with the tool's inverse.

Units are millimetres and radians. A Link is one rigid transform, from the frame before it to the frame it names; a
ToolChain composes its links in order from E. Two kinds of tool:

  calibration_tool, ati_tool   the stacks of the UR5 cell's plan builders, with the values they use (2026-11-05): the
                               OptoForce HEX-E's tool face 35.0 mm out along z_E with the mount turned mount_turn_deg
                               about the tool axis (-92.15 measured by hand seatings, lab README of 2026-10-22; the
                               builders' --mount-turn), then the pad frame P on the mount (ur5_cb3_cell.T_M_FROM_P)
                               and a contact point in P, or the ATI's frame on the mount (its parts record's T_MA).
  SR2Hand                      the SR2 hand on the HEX-E: E -> H (the HEX-E, 35.0 mm and its -2.15 deg turn) -> A (the
                               adaptor at clocking k) -> P (the palm frame) from assets/cad_v3/frames.json; then for
                               one finger its gantry (x, y within the travel box, from the box's centre), yaw, MCP,
                               PIP, the tip body and a contact point, with the link lengths of sr2_hand.json beside
                               this file.

The finger's chain is the one of the cad_v3 MuJoCo scenes (assets/mjcf/experimental/20261104-cad_v3, built by
scripts/hardware/build_cad_v3_hand.py from cad_v2's finger): <f>_mount at the travel box's centre with the slides
<f>_x and <f>_y along x_P and y_P; <f>_yaw_frame turning about x by the yaw; <f>_mcp_frame yaw_to_mcp_mm below it,
turning about (0, flex, 0) by the MCP angle; <f>_pip_frame mcp_to_pip_mm + proximal_extension_mm below that, turning
about the same axis by the PIP angle; <f>_tip tip_body_mm below the PIP axis, on the pad's centre plane. The contact
point defaults to the pad's apex, reach_mm from the finger's axis on the palmar side (+x of the tip frame for the
thumb, -x for index and middle: face_sign). proximal_extension_mm lengthens the MCP link along its axis: the PIP axis
and everything past it move by it along the MCP link (the owner's proximal phalange extender of 2026-11-05; not built).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from sr2hand.geometry.rotations import rot_axis_angle, tf, tf_inv
from sr2hand.robot import ur_kinematics as URK

ROOT = Path(__file__).resolve().parents[3]
SR2_JSON = Path(__file__).resolve().with_name("sr2_hand.json")
FRAMES_JSON = ROOT / "assets/cad_v3/frames.json"
FINGERS = ("thumb", "index", "middle")
EZ = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True, eq=False)
class Link:
    """One rigid transform T (4 x 4, mm) from the frame before it to the frame `name`."""
    name: str
    T: np.ndarray
    note: str = ""

    def __post_init__(self):
        T = np.array(self.T, float).reshape(4, 4)
        T.setflags(write=False)
        object.__setattr__(self, "T", T)


@dataclass(frozen=True, eq=False)
class ToolChain:
    """Named rigid transforms from the flange E, in order; transform() is T_E_end, their product."""
    name: str
    links: tuple

    def transform(self, upto=None):
        """T_E_from the frame `upto` (the last link's by default): the product of the links in order."""
        out = np.eye(4)
        for ln in self.links:
            out = out @ ln.T
            if ln.name == upto:
                return out
        if upto is not None:
            raise KeyError(f"{self.name}: no link {upto!r} (links: {[ln.name for ln in self.links]})")
        return out

    def frames(self):
        """{link name: T_E_from that frame}."""
        out, T = {}, np.eye(4)
        for ln in self.links:
            T = T @ ln.T
            out[ln.name] = T
        return out

    def then(self, *links, name=None):
        return ToolChain(name or self.name, tuple(self.links) + tuple(links))

    def to_dict(self):
        links = [dict(name=ln.name, T=ln.T.tolist(), note=ln.note) for ln in self.links]
        return dict(name=self.name, units="mm", links=links)

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], tuple(Link(x["name"], x["T"], x.get("note", "")) for x in d["links"]))

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n")

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def flange_target(T_B_end, tool):
    """The flange pose T_BE that puts the tool's end on T_B_end (4 x 4 or (N, 4, 4))."""
    return np.asarray(T_B_end, float) @ tf_inv(tool.transform())


def tool_ik(T_B_end, tool, q_ref, P=URK.UR5_CB3, **kw):
    """UR joints (6,) that put the tool's end on T_B_end, nearest q_ref (ur_kinematics.ik_nearest), or None."""
    return URK.ik_nearest(flange_target(T_B_end, tool), q_ref, P, **kw)


# --------------------------------------------------------------------------------------------------- calibration stacks
def _hex_e_mount(mount_turn_deg, hex_height_mm):
    """The builders' T_EM, bit for bit: rotations.tf(rotations.rot_axis_angle(z, radians(turn)), (0, 0, height))."""
    return Link("M", tf(rot_axis_angle(EZ, math.radians(mount_turn_deg)), [0.0, 0.0, hex_height_mm]),
                f"the HEX-E's tool face {hex_height_mm} mm out along z_E (OptoForce HEX-E data sheet), the mount "
                f"turned {mount_turn_deg:+.2f} deg about the tool axis (hand seatings, lab README of 2026-10-22)")


def calibration_tool(mount_turn_deg, T_MP, contact_P=None, hex_height_mm=35.0, name="tip calibration"):
    """The tip calibration's stack as the plan builders build it (ur5_tip_calibration_plan.nominal): the HEX-E's tool
    face hex_height_mm out along z_E with the mount turned mount_turn_deg about the tool axis (T_EM), the pad frame P
    on the mount (T_MP, ur5_cb3_cell.T_M_FROM_P), and a contact point in P (mm) when given."""
    links = [_hex_e_mount(mount_turn_deg, hex_height_mm),
             Link("P", T_MP, "the pad's tip frame on the mount (ur5_cb3_cell.T_M_FROM_P: b v4 on mount v3)")]
    if contact_P is not None:
        links.append(Link("contact", tf(None, contact_P), "the contact point in P"))
    return ToolChain(name, tuple(links))


def ati_tool(mount_turn_deg, T_MA, hex_height_mm=35.0, name="ATI calibration"):
    """The ATI calibration's stack (ati_calibration_plan.geometry): the HEX-E and the mount as calibration_tool's,
    then the ATI's frame A on the mount (T_MA of the stack's parts record)."""
    return ToolChain(name, (_hex_e_mount(mount_turn_deg, hex_height_mm),
                            Link("A", T_MA, "the ATI's frame on the mount (scripts/ur5/ati_stack_mount.py parts)")))


# --------------------------------------------------------------------------------------------------- the SR2 hand
@dataclass(frozen=True)
class FingerGeometry:
    """One finger's chain below its gantry (mm): link lengths, the flexion axis' sign along y (flex) and the palmar
    side of the tip along x (face_sign). Defaults and sources: sr2_hand.json."""
    yaw_to_mcp_mm: float
    mcp_to_pip_mm: float
    tip_body_mm: float
    reach_mm: float
    flex: float
    face_sign: float
    proximal_extension_mm: float = 0.0


def _rot(axis, angle):
    return tf(rot_axis_angle(np.asarray(axis, float), float(angle)))


@dataclass(frozen=True, eq=False)
class SR2Hand:
    """The SR2 hand on the UR5's HEX-E at clocking k: the frames of assets/cad_v3/frames.json and each finger's
    FingerGeometry. SR2Hand.load() reads both files; with_geometry() changes link lengths, the extension among them."""
    frames: dict
    fingers: dict
    k: int = 1
    sources: dict = field(default_factory=dict)

    @classmethod
    def load(cls, k=None, frames_json=FRAMES_JSON, sr2_json=SR2_JSON, **geometry):
        """The hand as built (k from frames.json's k_as_built unless given); geometry overrides any FingerGeometry
        field for every finger (proximal_extension_mm=10.0, say)."""
        F = json.loads(Path(frames_json).read_text())
        S = json.loads(Path(sr2_json).read_text())
        g = {k_: v["value"] for k_, v in S["finger_geometry"].items()}
        fingers = {f: FingerGeometry(**{**g, **S["fingers"][f], **geometry}) for f in FINGERS}
        return cls(frames=F, fingers=fingers, k=int(F["flange"]["k_as_built"] if k is None else k),
                   sources=dict(frames=str(frames_json), sr2=str(sr2_json), notes=S["finger_geometry"]))

    def with_geometry(self, finger=None, **kw):
        """A copy with FingerGeometry fields changed for one finger (or all)."""
        fs = dict(self.fingers)
        for f in ([finger] if finger else FINGERS):
            fs[f] = replace(fs[f], **kw)
        return SR2Hand(self.frames, fs, self.k, self.sources)

    def palm_links(self):
        """The flange to the palm: E -> H -> A -> P."""
        F, k = self.frames, self.k
        T_PA = tf(np.array(F["P"]["R_P_from_A"], float), F["P"]["t_P_from_A_mm"])
        return (Link("H", np.array(F["flange"]["hex_e"]["T_E_from_H"], float).reshape(4, 4),
                     "the HEX-E's tool face: 35.0 mm out along z_E, turned -2.15 deg (frames.json T_E_from_H)"),
                Link("A", np.array(F["flange"]["T_H_from_A_by_k"][k], float).reshape(4, 4),
                     f"the adaptor at clocking k = {k} (frames.json flange.T_H_from_A_by_k)"),
                Link("P", tf_inv(T_PA), "the palm frame P (frames.json P.R_P_from_A, P.t_P_from_A_mm)"))

    def T_E_P(self):
        return ToolChain("palm", self.palm_links()).transform()

    def box(self, finger):
        """(centre in P (3,), half sizes along x_P and y_P (2,)) of the finger's travel box, mm."""
        f = self.frames["fingers"][finger]
        return np.array(f["box_centre_P_mm"], float), np.array(f["box_size_P_mm"], float) / 2.0

    def finger_links(self, finger, x_mm=0.0, y_mm=0.0, yaw=0.0, mcp=0.0, pip=0.0, contact_tip_mm=None):
        """The links from P to the contact point: the gantry (x_mm, y_mm from the box's centre, along x_P and y_P),
        the joint angles (rad, the simulator's <f>_yaw, <f>_mcp, <f>_pip), and the contact point in the tip frame
        (mm; the pad's apex by default)."""
        g = self.fingers[finger]
        c, _ = self.box(finger)
        ax = [0.0, g.flex, 0.0]
        if contact_tip_mm is None:
            contact_tip_mm = [g.face_sign * g.reach_mm, 0.0, 0.0]
        proximal = g.mcp_to_pip_mm + g.proximal_extension_mm
        return (Link(f"{finger}_mount", tf(None, c + [x_mm, y_mm, 0.0]), "the gantry: the box's centre + (x, y)"),
                Link(f"{finger}_yaw_frame", _rot([1.0, 0.0, 0.0], yaw), "yaw about x"),
                Link(f"{finger}_mcp_frame", tf(None, [0.0, 0.0, -g.yaw_to_mcp_mm]) @ _rot(ax, mcp),
                     f"yaw_to_mcp_mm below, MCP about (0, {g.flex:+.0f}, 0)"),
                Link(f"{finger}_pip_frame", tf(None, [0.0, 0.0, -proximal]) @ _rot(ax, pip),
                     "mcp_to_pip_mm + proximal_extension_mm below, PIP about the MCP's axis"),
                Link(f"{finger}_tip", tf(None, [0.0, 0.0, -g.tip_body_mm]), "tip_body_mm below: the pad's centre"),
                Link("contact", tf(None, contact_tip_mm), "the contact point in the tip frame (the pad's apex)"))

    def chain(self, finger, x_mm=0.0, y_mm=0.0, yaw=0.0, mcp=0.0, pip=0.0, contact_tip_mm=None):
        """The ToolChain from the flange to the finger's contact point."""
        return ToolChain(f"SR2 {finger}, k {self.k}", self.palm_links()
                         + self.finger_links(finger, x_mm, y_mm, yaw, mcp, pip, contact_tip_mm))

    def in_box(self, finger, x_mm, y_mm, tol=1e-9):
        _, h = self.box(finger)
        return abs(x_mm) <= h[0] + tol and abs(y_mm) <= h[1] + tol

    def T_P_contact(self, finger, x_mm=0.0, y_mm=0.0, yaw=0.0, mcp=0.0, pip=0.0, contact_tip_mm=None):
        return ToolChain("finger", self.finger_links(finger, x_mm, y_mm, yaw, mcp, pip, contact_tip_mm)).transform()

    def T_E_contact(self, finger, x_mm=0.0, y_mm=0.0, yaw=0.0, mcp=0.0, pip=0.0, contact_tip_mm=None):
        return self.chain(finger, x_mm, y_mm, yaw, mcp, pip, contact_tip_mm).transform()

    def ik(self, T_B_contact, finger, gantry_mm, joints, q_ref, contact_tip_mm=None, P=URK.UR5_CB3, check_box=True,
           **kw):
        """UR joints (6,) that put the finger's contact frame on T_B_contact (4 x 4, the base frame) with the finger's
        gantry at gantry_mm (x, y from the box's centre) and its joints (yaw, mcp, pip; rad), nearest q_ref
        (ur_kinematics.ik_nearest); None when the arm cannot. A gantry outside its travel box raises ValueError unless
        check_box is False."""
        x, y = (float(v) for v in gantry_mm)
        if check_box and not self.in_box(finger, x, y):
            raise ValueError(f"{finger}: gantry ({x}, {y}) mm outside its travel box (half sizes "
                             f"{self.box(finger)[1]})")
        tool = self.chain(finger, x, y, *(float(v) for v in joints), contact_tip_mm=contact_tip_mm)
        return tool_ik(T_B_contact, tool, q_ref, P, **kw)
