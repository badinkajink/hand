#!/usr/bin/env python3
"""The pinch brake of arXiv 2609.25619 on the real_v1 (SR2) hand, in MuJoCo and Drake.

D8's gantry (docs/experiments/20260829-real_v1_deploy/deploy/sv1_w0099_b100_plan.json) is baked into
assets/mjcf/real_v1/real_hand.xml and the palm is fixed in the air, palm down. Thumb and index pinch
the real_v1 screwdriver (cylinder r 12.5 mm, 100 mm, 24.5 g) at a station d from its centre of mass,
tool axis horizontal and perpendicular to the thumb-index line; the middle is parked out of the way.
The fingers are the calibrated position servos (`calfast` in real_v1_chain_hands.PLANT: kp 0.5 N m/rad,
kv 0.02, forcerange 0.35 N m, no joint damping; its frictionloss 0.0035 is left out because Drake
has none), so the pinch force is set by
a commanded pad force F: joint targets q_touch + J^T (F u) / kp with q_touch the IK posture whose tips
just touch the tool. The brake holds F0 with the tool's weight compensated, releases the weight, then
lowers F geometrically to F1 over T seconds.

Contact models (the two-pad rig's, scripts/hom_contact_rig.py, on the hand's fingertips):
  mj:point3                       MuJoCo, condim 3 on fingertips and tool
  mj:point4s[:fit<c>x<p>]         condim 4, mu_t = mu c N^p rescheduled per fingertip geom each step
  mj:spheres:s<mm>:rs<mm>:tr<s>   thumb and index tip spheres replaced by caps of small spheres
  drake:hydro[:rt<s>]             Drake 1.57 SAP, compliant hydroelastic fingertips (E 10 MPa,
                                  1 mm), rigid hydroelastic tool, PD servos with the same gains
  drake:point                     Drake point contact
MuJoCo runs at impratio 100 (`:ir<k>` overrides).

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/hom_hand_brake.py ik                      # pinch posture, tip errors, a still
    $PY scripts/hom_hand_brake.py brake --spec mj:point4s --d 15 --film out.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_contact_rig as H  # noqa: E402

HAND = ROOT / "assets/mjcf/real_v1/real_hand.xml"
PLAN = ROOT / "docs/experiments/20260829-real_v1_deploy/deploy/sv1_w0099_b100_plan.json"
BASE_MOUNTS = {"thumb": (-0.050, 0.000), "index": (0.050, 0.055), "middle": (0.050, -0.055)}
FINGERS = ("thumb", "index", "middle")
JOINTS = {f: (f"{f}_yaw", f"{f}_mcp", f"{f}_pip") for f in FINGERS}
PALM_Z = 0.30                      # palm_pose origin height; the floor is far below the swing
PINCH_DEPTH = 0.045                # pinch point below the mounts (D8 tips reach 47 mm past a 21 mm link)
# calfast without its frictionloss: Drake SAP has no joint friction loss, so both simulators run
# without it (it costs ~0.2 N of pad force at low commands on this hand, measured 2026-10-01)
PLANT = dict(kp=0.5, kv=0.02, forcerange=0.35, frictionloss=0.0, damping=0.0)
MIDDLE_PARK = (0.0, -0.26, 0.0)    # straight and tilted back, away from the tool
R_TIP = H.R_PAD
FIT = (0.996e-3, 0.25)             # Drake reference torsion law on the two-pad rig, E 10 MPa


def d8_offsets():
    mounts = json.load(open(PLAN))["mounts_palm_mm"]
    return {f: (mounts[f][0] * 1e-3 - BASE_MOUNTS[f][0], mounts[f][1] * 1e-3 - BASE_MOUNTS[f][1]) for f in FINGERS}


def parse_spec(spec):
    parts = spec.split(":")
    out = {"spec": spec, "sim": parts[0], "model": parts[1], "ir": 100.0, "E": 1e7, "res": 1e-3}
    for p in parts[2:]:
        if p.startswith("fit"):
            c, pw = p[3:].split("x")
            out["fit"] = (float(c), float(pw))
        elif p.startswith("hr"):                       # Drake compliant-hydroelastic resolution hint, mm
            out["res"] = float(p[2:]) * 1e-3
        elif p.startswith("rs"):
            out["rs"] = float(p[2:]) * 1e-3
        elif p.startswith("rt"):
            out["rt"] = float(p[2:])
        elif p.startswith("tr"):
            out["tr"] = float(p[2:])
        elif p.startswith("ir"):
            out["ir"] = float(p[2:])
        elif p.startswith("cap"):
            out["cap_deg"] = float(p[3:])
        elif p.startswith("s"):
            out["s"] = float(p[1:]) * 1e-3
        elif p.startswith("E"):
            out["E"] = float(p[1:])
        else:
            raise ValueError(f"unknown field {p!r} in {spec}")
    return out


# ------------------------------------------------------------------------------- the hand MJCF

def hand_tree(offsets):
    """real_hand.xml with D8's gantry baked into the mounts, the zero-range length joints and the
    morphology sliders removed, the palm raised to PALM_Z and the calibrated servo applied."""
    root = ET.parse(HAND).getroot()
    root.find("option").set("timestep", f"{H.DT}")
    for kf in root.findall("keyframe"):
        root.remove(kf)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "joint" and (child.get("class") == "morph"):
                parent.remove(child)
            if child.tag == "site" and child.get("name", "").startswith("workspace_"):
                parent.remove(child)
    for f in FINGERS:
        mount = root.find(f".//body[@name='{f}_mount']")
        p = np.fromstring(mount.get("pos"), sep=" ")
        p[:2] += offsets[f]
        mount.set("pos", " ".join(f"{v:.6f}" for v in p))
    palm = root.find(".//body[@name='palm_pose']")
    palm.set("pos", f"0 0 {PALM_Z}")
    palm.attrib.pop("gravcomp", None)
    for d in root.iter("default"):
        if d.get("class") == "ctrl":
            d.find("joint").set("damping", f"{PLANT['damping']:g}")
            if PLANT["frictionloss"] > 0:
                d.find("joint").set("frictionloss", f"{PLANT['frictionloss']:g}")
            pos = d.find("position")
            pos.set("kp", f"{PLANT['kp']:g}")
            pos.set("kv", f"{PLANT['kv']:g}")
            pos.set("forcerange", f"-{PLANT['forcerange']:g} {PLANT['forcerange']:g}")
    return root


def tip_geoms(root, f):
    """(distal capsule, tip sphere) elements of finger f."""
    pip = root.find(f".//body[@name='{f}_pip_frame']")
    tip = root.find(f".//body[@name='{f}_tip']")
    return pip.find("geom"), tip.find("geom")


def hand_model(offsets):
    import mujoco
    return mujoco.MjModel.from_xml_string(ET.tostring(hand_tree(offsets), encoding="unicode"))


# ----------------------------------------------------------------------------------------- IK

def solve_tip(m, d, f, target, q0, iters=200):
    """Damped least squares on finger f's three joints so the tip sphere centre reaches `target`."""
    import mujoco
    jids = [m.joint(n).id for n in JOINTS[f]]
    qa = [m.jnt_qposadr[j] for j in jids]
    va = [m.jnt_dofadr[j] for j in jids]
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])
    d.qpos[qa] = q0
    bid = m.body(f"{f}_tip").id
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        err = target - d.xpos[bid]
        if np.linalg.norm(err) < 1e-7:
            break
        mujoco.mj_jacBody(m, d, jacp, None, bid)
        J = jacp[:, va]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err)
        d.qpos[qa] = np.clip(d.qpos[qa] + dq, lo, hi)
    mujoco.mj_kinematics(m, d)
    return d.qpos[qa].copy(), float(np.linalg.norm(target - d.xpos[bid]))


def pinch_geometry(offsets, d_cg):
    """Pinch point, pinch axis u (thumb -> index, horizontal), tool axis a, tool centre."""
    th = np.array(BASE_MOUNTS["thumb"]) + offsets["thumb"]
    ix = np.array(BASE_MOUNTS["index"]) + offsets["index"]
    P = np.array([(th[0] + ix[0]) / 2, (th[1] + ix[1]) / 2, PALM_Z - PINCH_DEPTH])
    u = np.array([ix[0] - th[0], ix[1] - th[1], 0.0])
    u /= np.linalg.norm(u)
    a = np.cross([0.0, 0.0, 1.0], u)
    return P, u, a, P + d_cg * a


def pinch_posture(offsets, squeeze, d_cg=0.015, q_guess=None):
    """Joint targets (9) for thumb and index tips at R_tool + R_tip - squeeze from the tool axis."""
    import mujoco
    m = hand_model(offsets)
    d = mujoco.MjData(m)
    P, u, a, c = pinch_geometry(offsets, d_cg)
    gap = H.R_TOOL + R_TIP - squeeze
    g = {"thumb": np.array([0.3, 0.33, 0.98]), "index": np.array([-0.3, 0.33, 0.98])} if q_guess is None else \
        {f: np.asarray(q_guess).reshape(3, 3)[i] for i, f in enumerate(FINGERS)}
    q = {}
    errs = {}
    q["thumb"], errs["thumb"] = solve_tip(m, d, "thumb", P - gap * u, g["thumb"])
    q["index"], errs["index"] = solve_tip(m, d, "index", P + gap * u, g["index"])
    q["middle"] = np.array(MIDDLE_PARK)
    return np.concatenate([q[f] for f in FINGERS]), errs, (P, u, a, c)


def posture_path(offsets, squeezes, d_cg):
    """IK by continuation along a list of squeezes (keeps one branch); returns {s: q9}, max error."""
    out, worst, q = {}, 0.0, None
    for s in squeezes:
        q, errs, _ = pinch_posture(offsets, s, d_cg, q)
        out[s] = q
        worst = max(worst, max(errs.values()))
    return out, worst


# ------------------------------------------------------------------------------- MuJoCo scene

def mj_scene(spec, offsets, d_cg, q_init, geom_dirs=None, pad_tc=None):
    """Hand + tool MJCF for a contact spec. geom_dirs: per finger, the tip-frame direction of the
    contact (for sphere caps). Returns (xml, info)."""
    sp = parse_spec(spec)
    root = hand_tree(offsets)
    opt = root.find("option")
    opt.set("cone", "elliptic")
    opt.set("impratio", f"{sp['ir']:g}")
    opt.set("solver", "Newton")
    opt.set("iterations", "200")
    opt.set("tolerance", "1e-10")
    P, u, a, c = pinch_geometry(offsets, d_cg)
    info = {}
    model = sp["model"]
    fing_contact = {"point3": {"condim": "3", "friction": "1 0 0"},
                    "point4s": {"condim": "4", "friction": "1 0.001 0"},
                    "spheres": {"condim": "3", "friction": "1 0 0"}}[model]
    for f in FINGERS:
        cap_el, sph_el = tip_geoms(root, f)
        for el in (cap_el, sph_el):
            for k, v in fing_contact.items():
                el.set(k, v)
            el.set("solref", "0.006 1")
            el.set("solimp", "0.97 0.995 0.0005")
            el.set("name", f"{f}_{'tipsphere' if el is sph_el else 'distal'}")
    if model == "spheres" and geom_dirs is not None:
        s, rs = sp["s"], sp["rs"]
        cap = math.radians(sp.get("cap_deg", 45.0))
        area = 2 * math.pi * R_TIP ** 2 * (1 - math.cos(cap))
        n = max(1, int(round(area / s ** 2)))
        A_s = area / n
        K_s = sp["E"] / R_TIP * A_s * ((R_TIP - rs) / R_TIP) ** 2
        info.update(n_spheres=n, K_sphere=K_s)
        tc, d0 = pad_tc if pad_tc is not None else (0.01, 0.9)
        for f in ("thumb", "index"):
            tip = root.find(f".//body[@name='{f}_tip']")
            sph = tip.find("geom")
            sph.set("contype", "0")
            sph.set("conaffinity", "0")
            sph.set("group", "1")
            sph.set("rgba", "0.85 0.55 0.35 0.35")
            Rm = H._rot_z_to(geom_dirs[f])
            for i, dv in enumerate(H.fib_cap(n, cap) @ Rm.T):
                p = (R_TIP - rs) * dv
                ET.SubElement(tip, "geom", name=f"{f}_pad{i}", type="sphere", size=f"{rs}",
                              pos=f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f}", condim="3", friction="1 0 0",
                              solref=f"{tc:.6g} 1", solimp=f"{d0:.6g} {d0:.6g} 0.001 0.5 2",
                              rgba="0.75 0.4 0.25 1", mass="0")
    # tool: axis a, centre c; its contact params mirror the fingertips'
    R = np.column_stack([np.cross(a, [0, 0, 1.0]) / np.linalg.norm(np.cross(a, [0, 0, 1.0])), [0, 0, 1.0], a])
    R[:, 1] = np.cross(R[:, 2], R[:, 0])
    quat = np.zeros(4)
    import mujoco
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    wb = root.find("worldbody")
    tool = ET.SubElement(wb, "body", name="tool", pos=f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}",
                         quat=" ".join(f"{x:.8f}" for x in quat))
    ET.SubElement(tool, "freejoint")
    ET.SubElement(tool, "inertial", pos="0 0 0", mass=f"{H.M_TOOL}",
                  diaginertia=f"{H.I_TOOL_T} {H.I_TOOL_T} {H.I_TOOL_A}")
    tg = ET.SubElement(tool, "geom", name="tool", type="cylinder", size=f"{H.R_TOOL} {H.HL_TOOL}",
                       rgba="0.55 0.6 0.68 1", solref="0.006 1", solimp="0.97 0.995 0.0005")
    if model == "spheres":
        tg.set("condim", "3")
        tg.set("friction", "1 0 0")
        tg.set("solref", f"{(pad_tc or (0.01, 0.9))[0]:.6g} 1")
        tg.set("solimp", f"{(pad_tc or (0.01, 0.9))[1]:.6g} {(pad_tc or (0.01, 0.9))[1]:.6g} 0.001 0.5 2")
    else:
        tg.set("condim", fing_contact["condim"])
        tg.set("friction", "1 0 0")
    ET.SubElement(tool, "geom", type="box", pos=f"0 0 {H.HL_TOOL}", size=f"0.0016 {H.R_TOOL * 0.8} 0.0008",
                  contype="0", conaffinity="0", rgba="0.1 0.1 0.1 1")
    # visual: sky and a camera-friendly floor
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="1 1 1", rgb2=".86 .89 .93",
                  width="64", height="64")
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    g = vis.find("global")
    if g is None:
        g = ET.SubElement(vis, "global")
    g.set("offwidth", "960")
    g.set("offheight", "720")
    # initial controls = posture
    kf = ET.SubElement(root, "keyframe")
    return ET.tostring(root, encoding="unicode"), info


class MjHand:
    sim = "mujoco"

    def __init__(self, spec, offsets, d_cg, q_init, geom_dirs=None):
        import mujoco
        self.mj, self.sp = mujoco, parse_spec(spec)
        xml, self.info = mj_scene(spec, offsets, d_cg, q_init, geom_dirs)
        if self.sp["model"] == "spheres":
            m0 = mujoco.MjModel.from_xml_string(xml)
            diag = m0.body_invweight0[m0.body("thumb_tip").id, 0] + m0.body_invweight0[m0.body("tool").id, 0]
            tc = self.sp["tr"] / 2.0
            d0 = 1.0 - 1.0 / (tc ** 2 * self.info["K_sphere"] * diag)
            if d0 < 0.05:
                raise ValueError(f"relaxation {self.sp['tr']} s below this pad's floor")
            xml, self.info = mj_scene(spec, offsets, d_cg, q_init, geom_dirs, pad_tc=(tc, d0))
            self.info.update(solref_timeconst=tc, solimp_d0=d0, diagApprox=float(diag))
        self.xml = xml
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.tool = self.m.body("tool").id
        self.ja = [self.m.joint(n).id for f in FINGERS for n in JOINTS[f]]
        for j, q in zip(self.ja, q_init):
            self.d.qpos[self.m.jnt_qposadr[j]] = q
        self.d.ctrl[:] = q_init
        self.side = {}
        for gi in range(self.m.ngeom):
            b = self.m.body(self.m.geom_bodyid[gi]).name
            for f in FINGERS:
                if b.startswith(f + "_") and self.m.geom_contype[gi]:
                    self.side[gi] = f
        self.fing_geoms = {f: [g for g, s in self.side.items() if s == f] for f in FINGERS}
        self.lastN = {g: 0.0 for g in self.side}
        self.f6 = np.zeros(6)
        mujoco.mj_forward(self.m, self.d)
        self.theta_prev, self.theta_unwrap = None, 0.0

    @property
    def t(self):
        return float(self.d.time)

    def set_targets(self, q9):
        self.d.ctrl[:] = q9

    def set_tool_wrench(self, f, tau):
        self.d.xfrc_applied[self.tool, :3] = f
        self.d.xfrc_applied[self.tool, 3:] = tau

    def step(self, T):
        n = max(1, int(round(T / H.DT)))
        resched = self.sp["model"] == "point4s"
        c, pw = self.sp.get("fit", FIT)
        for _ in range(n):
            if resched:
                for g, N in self.lastN.items():
                    self.m.geom_friction[g, 1] = H.MU * c * max(N, 1e-3) ** pw
            self.mj.mj_step(self.m, self.d)
            if resched:
                self.lastN = self.per_geom_N()

    def per_geom_N(self):
        out = {g: 0.0 for g in self.side}
        for i in range(self.d.ncon):
            cc = self.d.contact[i]
            g = int(cc.geom[0]) if int(cc.geom[0]) in self.side else int(cc.geom[1])
            if g not in self.side:
                continue
            self.mj.mj_contactForce(self.m, self.d, i, self.f6)
            out[g] += float(self.f6[0])
        return out

    def contacts(self):
        out = {f: {"N": 0.0, "n": 0} for f in FINGERS}
        for i in range(self.d.ncon):
            cc = self.d.contact[i]
            g0, g1 = int(cc.geom[0]), int(cc.geom[1])
            if self.tool not in (self.m.geom_bodyid[g0], self.m.geom_bodyid[g1]):
                continue
            f = self.side.get(g0) or self.side.get(g1)
            if f is None:
                continue
            self.mj.mj_contactForce(self.m, self.d, i, self.f6)
            out[f]["N"] += float(self.f6[0])
            out[f]["n"] += 1
        return out

    def tool_state(self):
        qa = self.m.jnt_qposadr[self.m.body_jntadr[self.tool]]
        va = self.m.jnt_dofadr[self.m.body_jntadr[self.tool]]
        R9 = np.zeros(9)
        self.mj.mju_quat2Mat(R9, self.d.qpos[qa + 3:qa + 7])
        R = R9.reshape(3, 3)
        w = R @ self.d.qvel[va + 3:va + 6]
        return {"R": R, "axis": R[:, 2].copy(), "pos": self.d.qpos[qa:qa + 3].copy(), "w": w}

    def joint_q(self):
        return np.array([self.d.qpos[self.m.jnt_qposadr[j]] for j in self.ja])


# ---------------------------------------------------------------------------------- the brake

def swing_angle(axis, a0, u):
    """Rotation of the tool axis about the pinch axis u from its start a0, positive = centre of mass
    down; plus the out-of-plane tilt (deg) of the axis from the plane normal to u."""
    w = np.cross(u, a0)                       # in-plane direction that is 'up' or 'down'
    if w[2] > 0:
        w = -w                                # positive phi = toward -z
    x, y = float(axis @ a0), float(axis @ w)
    return math.degrees(math.atan2(y, x)), math.degrees(math.asin(np.clip(axis @ u, -1, 1)))


def tip_jacobians(off, q9):
    """3x3 position Jacobians of the thumb and index tip centres at q9 (columns yaw, mcp, pip)."""
    import mujoco
    m = hand_model(off)
    d = mujoco.MjData(m)
    for f, qq in zip(FINGERS, np.asarray(q9).reshape(3, 3)):
        for n, v in zip(JOINTS[f], qq):
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    out = {}
    jacp = np.zeros((3, m.nv))
    for f in ("thumb", "index"):
        mujoco.mj_jacBody(m, d, jacp, None, m.body(f"{f}_tip").id)
        out[f] = jacp[:, [m.jnt_dofadr[m.joint(n).id] for n in JOINTS[f]]].copy()
    return out


def exp_brake(spec, d_cg=0.015, F0=4.0, F1=0.5, T_hold=0.6, T_ramp=4.0, T_end=1.0, film=None,
              traces=True, settle=1.0):
    """The pinch force is commanded through the position servos as q = q_touch + J^T (F u) / kp, so
    each pad pushes along the pinch axis (the paper's impedance action) rather than along the
    servo's joint-space stiffness J (J J^T)^-1, which also pushes the tool sideways."""
    t0w = time.time()
    off = d8_offsets()
    q_start, errs, (P, u, a, c) = pinch_posture(off, 0.012, d_cg)
    sp = parse_spec(spec)
    path, ik_worst = posture_path(off, [float(x) for x in np.linspace(0.012, 0.0, 25)], d_cg)
    q_touch = path[0.0]
    q_open, e_open, _ = pinch_posture(off, -0.0015, d_cg, q_touch)
    ik_worst = max(ik_worst, max(e_open.values()))
    geom_dirs = contact_dirs(off, q_touch, P, u) if sp["model"] == "spheres" else None
    Jt = tip_jacobians(off, q_touch)

    def targets(F):
        q = q_touch.copy().reshape(3, 3)
        q[0] += Jt["thumb"].T @ (F * u) / PLANT["kp"]
        q[1] += Jt["index"].T @ (-F * u) / PLANT["kp"]
        return q.reshape(-1)
    if sp["sim"] == "mj":
        rig = MjHand(spec, off, d_cg, q_open, geom_dirs)
    else:
        import hom_hand_drake as HD
        rig = HD.DrakeHand(spec, off, d_cg, q_open)
    W = H.M_TOOL * H.G
    rig.set_targets(targets(F0))
    rig.set_tool_wrench(np.array([0.0, 0.0, W]), np.zeros(3))
    rig.step(settle)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    st0 = rig.tool_state()
    a0 = st0["axis"]
    p0 = st0["pos"].copy()
    t0 = rig.t
    rows, frames = [], []
    renderer = Renderer(rig) if film else None
    n = int(round((T_hold + T_ramp + T_end) / H.DT))
    for k in range(n):
        tt = rig.t - t0
        if tt < T_hold:
            F = F0
        elif tt < T_hold + T_ramp:
            F = F0 * (F1 / F0) ** ((tt - T_hold) / T_ramp)
        else:
            F = F1
        rig.set_targets(targets(F))
        rig.step(H.DT)
        if k % 10 == 0:
            st = rig.tool_state()
            phi, tilt = swing_angle(st["axis"], a0, u)
            cc = rig.contacts()
            slip = float((st["pos"] - P) @ st["axis"] - (p0 - P) @ a0)
            rows.append([round(tt, 4), F, phi, tilt, cc["thumb"]["N"], cc["index"]["N"], cc["middle"]["N"],
                         cc["thumb"]["n"], cc["index"]["n"], slip * 1e3, float(st["pos"][2] - p0[2]) * 1e3])
            if renderer is not None and k % 40 == 0:
                frames.append(renderer.frame(rig, f"{label(spec)}", f"pad {0.5 * (cc['thumb']['N'] + cc['index']['N']):4.2f} N   phi {phi:5.1f} deg   t {tt:4.2f} s"))
    tr = np.array(rows)
    res = {"exp": "hand_brake", "spec": spec, "d_mm": d_cg * 1e3, "F0": F0, "F1": F1,
           "T_ramp": T_ramp, "ik_err_mm": ik_worst * 1e3,
           "N_start": float(0.5 * (tr[0, 4] + tr[0, 5])), "phi_end": float(tr[-1, 2]),
           "phi_max": float(tr[:, 2].max()), "tilt_max": float(np.abs(tr[:, 3]).max()),
           "slip_end_mm": float(tr[-1, 9]), "dz_end_mm": float(tr[-1, 10]),
           "held_end": bool(tr[-1, 7] > 0 and tr[-1, 8] > 0), "wall_s": time.time() - t0w,
           "info": getattr(rig, "info", {})}
    for ang in (10, 45, 80):
        idx = np.where(tr[:, 2] >= ang)[0]
        res[f"F_at_{ang}"] = float(tr[idx[0], 1]) if len(idx) else None
        res[f"N_at_{ang}"] = float(0.5 * (tr[idx[0], 4] + tr[idx[0], 5])) if len(idx) else None
    res["peak_rate_dps"] = float(np.abs(np.diff(tr[:, 2])).max() / 0.01)
    if traces:
        res["trace_cols"] = ["t", "F_cmd", "phi", "tilt", "N_thumb", "N_index", "N_middle", "n_thumb",
                             "n_index", "slip_mm", "dz_mm"]
        res["trace"] = rows
    if renderer is not None:
        H.write_mp4(frames, film, fps=25)
        res["film"] = str(film)
    return res


HAND_SPECS = {
    "dpt": "drake:point", "d001": "drake:hydro:rt0.01", "d002": "drake:hydro:rt0.02", "ddef": "drake:hydro",
    "mp3": "mj:point3", "p4s": "mj:point4s", "s1": "mj:spheres:s1:rs0.75:tr0.02", "s1tr01": "mj:spheres:s1:rs0.75:tr0.1",
}
TILES = ["dpt", "d001", "d002", "ddef", "mp3", "s1", "p4s", "s1tr01"]


def study(out):
    """All hand-brake cells (resumable), one film per spec at d 15 mm, and the tiled film."""
    out = Path(out)
    (out / "media").mkdir(parents=True, exist_ok=True)
    rows_p = out / "hand_brake.jsonl"
    have = set()
    if rows_p.exists():
        have = {(r["spec"], r["d_mm"]) for r in map(json.loads, open(rows_p))}
    frames = {}
    for d_mm in (15.0, 30.0):
        for key, spec in HAND_SPECS.items():
            film = out / "media" / f"20261001-hand_brake_{key}_d{d_mm:g}.mp4" if d_mm == 15.0 else None
            if (spec, d_mm) in have and (film is None or film.exists()):
                continue
            t0 = time.time()
            r = exp_brake(spec, d_mm * 1e-3, film=str(film) if film else None)
            H.append_row(rows_p, r)
            print(f"  {spec:34s} d {d_mm:g}  45 deg at {r['N_at_45']}  80 deg at {r['N_at_80']}  end {r['phi_end']:.1f}"
                  f"  {time.time() - t0:.1f} s", flush=True)
    import imageio.v2 as imageio
    for key in TILES:
        frames[key] = [f for f in imageio.get_reader(str(out / "media" / f"20261001-hand_brake_{key}_d15.mp4"))]
    n = min(len(v) for v in frames.values())
    grid = [np.concatenate([np.concatenate([frames[k][i] for k in TILES[:4]], 1),
                            np.concatenate([frames[k][i] for k in TILES[4:]], 1)], 0) for i in range(n)]
    H.write_mp4(grid, out / "media" / "20261001-hand_brake_eight_models.mp4", fps=25)
    from PIL import Image
    Image.fromarray(grid[min(n - 1, 60)]).save(out / "media" / "20261001-hand_brake_eight_models_poster.png")
    print("done", flush=True)


def contact_dirs(off, q, P, u):
    """Tip-frame direction from each pinching tip's centre toward the tool axis (sphere cap axis)."""
    import mujoco
    m = hand_model(off)
    d = mujoco.MjData(m)
    for f, qq in zip(FINGERS, q.reshape(3, 3)):
        for n, v in zip(JOINTS[f], qq):
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    mujoco.mj_kinematics(m, d)
    out = {}
    for f, sgn in (("thumb", 1.0), ("index", -1.0)):
        b = m.body(f"{f}_tip").id
        R = d.xmat[b].reshape(3, 3)
        out[f] = R.T @ (sgn * u)
    return out


def label(spec):
    sp = parse_spec(spec)
    if sp["sim"] == "drake":
        return "Drake point contact" if sp["model"] == "point" else f"Drake hydroelastic, relaxation {sp.get('rt', 0.1):g} s"
    return {"point3": "MuJoCo point contact, condim 3", "point4s": "MuJoCo condim 4, mu_t rescheduled",
            "spheres": f"MuJoCo {sp.get('s', 0) * 1e3:g} mm sphere tips, relaxation {sp.get('tr', 0):g} s"}[sp["model"]]


class Renderer:
    def __init__(self, rig, w=400, h=300):
        import mujoco
        self.mj = mujoco
        if rig.sim == "mujoco":
            self.m, self.d = rig.m, mujoco.MjData(rig.m)
        else:
            xml, _ = mj_scene("mj:point3", rig.offsets, rig.d_cg, rig.q_init)
            self.m = mujoco.MjModel.from_xml_string(xml)
            self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        P, u, _, _ = pinch_geometry(rig.offsets if rig.sim == "drake" else d8_offsets(), 0.015)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [P[0], P[1], P[2] - 0.01]
        self.cam.distance = 0.30
        self.cam.azimuth = math.degrees(math.atan2(u[1], u[0]))     # along the pinch axis: the swing is in view
        self.cam.elevation = -8.0

    def frame(self, rig, title, line):
        if rig.sim == "mujoco":
            self.d.qpos[:] = rig.d.qpos
        else:
            rig.copy_to_mujoco(self.m, self.d)
        self.mj.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, self.cam)
        img = H.annotate(self.r.render().copy(), title)
        return H.annotate_bottom(img, line)


# ------------------------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("ik")
    k.add_argument("--squeeze", type=float, default=20.0)
    k.add_argument("--still")
    b = sub.add_parser("brake")
    b.add_argument("--spec", required=True)
    b.add_argument("--d", type=float, default=15.0)
    b.add_argument("--F0", type=float, default=4.0)
    b.add_argument("--F1", type=float, default=0.5)
    b.add_argument("--T-ramp", type=float, default=4.0)
    b.add_argument("--film")
    b.add_argument("--rows")
    st = sub.add_parser("study")
    st.add_argument("--out", default=str(ROOT / "docs/experiments/20261001-hom_hand_brake"))
    args = ap.parse_args()
    if args.cmd == "study":
        study(args.out)
        return
    if args.cmd == "ik":
        off = d8_offsets()
        q, errs, (P, u, a, c) = pinch_posture(off, args.squeeze * 1e-3)
        print(json.dumps({"offsets_mm": {f: [round(v * 1e3, 2) for v in off[f]] for f in FINGERS},
                          "P": P.round(4).tolist(), "u": u.round(4).tolist(), "a": a.round(4).tolist(),
                          "q": q.round(4).tolist(), "ik_err_mm": {f: round(e * 1e3, 4) for f, e in errs.items()}}))
        if args.still:
            import mujoco
            xml, _ = mj_scene("mj:point3", off, 0.015, q)
            m = mujoco.MjModel.from_xml_string(xml)
            d = mujoco.MjData(m)
            for j, v in zip([m.joint(n).id for f in FINGERS for n in JOINTS[f]], q):
                d.qpos[m.jnt_qposadr[j]] = v
            mujoco.mj_forward(m, d)
            r = mujoco.Renderer(m, 480, 640)
            imgs = []
            for az, el in ((112, -8), (22, -8), (112, -60)):
                cam = mujoco.MjvCamera()
                cam.lookat[:] = [0.0, 0.017, PALM_Z - PINCH_DEPTH]
                cam.distance, cam.azimuth, cam.elevation = 0.30, az, el
                r.update_scene(d, cam)
                imgs.append(r.render().copy())
            from PIL import Image
            Image.fromarray(np.concatenate(imgs, 1)).save(args.still)
            print("still", args.still, "ncon", d.ncon)
        return
    row = exp_brake(args.spec, args.d * 1e-3, args.F0, args.F1, T_ramp=args.T_ramp, film=args.film)
    if args.rows:
        H.append_row(args.rows, row)
    print(json.dumps({k: v for k, v in row.items() if k not in ("trace",)}, default=H._json_default))


if __name__ == "__main__":
    main()
