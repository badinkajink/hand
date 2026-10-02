#!/usr/bin/env python3
"""The screwdriver load-and-insert chain under relative contact-velocity control on the real_v1 (SR2)
hand, in MuJoCo and Drake.

D8 (gantry from its deploy plan) on a four-axis palm stage (x, y, z, yaw; the arm). The tool, the
real_v1 screwdriver cylinder (r 12.5 mm, 100 mm, 24.5 g), lies on two posts. Stages:

  approach  palm descends with thumb and index open 8 mm either side of the pinch station
  close     thumb and index close along their contact normals (v_sep reference, paper Eq. 2) until
            they touch, then press with F0 through the servos: q = q_touch + J^T(-F n)/kp
  lift      palm rises 90 mm, tool horizontal in the pinch
  brake     the tool swings about the pinch axis to hanging vertical; the pinch force is either
            reduced open-loop (the paper's load step) or set every 10 ms by a brake that tracks a
            swing-angle profile and backs off when the tool slides along its axis
  hold      pinch force raised to F_hold ("hold firm again")
  transport palm carries the tool's lower end over a chamfered peg hole (aimed with the measured pose)
  insert    palm descends 30 mm at 15 mm/s

The pinch line is the thumb-index mount line turned +45 deg about the palm normal, 50 mm below the
mounts: there both fingers keep 0.55 rad of joint range from 4 mm squeeze to 12 mm open (the brake
study's line through the mount midpoint at 45 mm sat on the mcp and pip limits and could not open).
The middle is curled out of the way. Controller rate 100 Hz (the servo bus runs 111 Hz).

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/hom_chain.py still --out /tmp/still.png
    $PY scripts/hom_chain.py run --spec mj:point4s --brake closed --seed 0 --film /tmp/chain.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_contact_rig as H  # noqa: E402
import hom_control as C  # noqa: E402
import hom_hand_brake as B  # noqa: E402

PINCH_DEPTH = 0.050
PINCH_ROT_DEG = 45.0
GAP_OPEN = 0.008
SQUEEZE_MAX = 0.004
MIDDLE_PARK = (0.0, 0.9, 0.9)          # curled under the palm, clear of the posts and the table
POST_H, POST_HALF, POST_AT = 0.015, 0.004, 0.042
BLOCK_H, CHAMFER, HOLE_AT, N_SEG = 0.030, 0.003, 0.110, 32
HOLE_GOAL = 0.020                      # insertion depth below the block top that counts as in
MU_ENV = 0.3                           # tool against table, posts and block
PALM_KP, PALM_KV, YAW_KP, YAW_KV = 4000.0, 100.0, 50.0, 2.0
RATE = 100.0
F0, F_HOLD = 4.0, 3.0
LIFT = 0.090
FIT = B.FIT


# ----------------------------------------------------------------------------------- geometry

def pinch_frame():
    """Pinch point P, pinch axis u (thumb -> index) and tool axis a, in the palm frame."""
    off = B.d8_offsets()
    th = np.array(B.BASE_MOUNTS["thumb"]) + off["thumb"]
    ix = np.array(B.BASE_MOUNTS["index"]) + off["index"]
    mid = 0.5 * (th + ix)
    u0 = (ix - th) / np.linalg.norm(ix - th)
    c, s = math.cos(math.radians(PINCH_ROT_DEG)), math.sin(math.radians(PINCH_ROT_DEG))
    u = np.array([c * u0[0] - s * u0[1], s * u0[0] + c * u0[1], 0.0])
    a = np.cross([0.0, 0.0, 1.0], u)
    return np.array([mid[0], mid[1], -PINCH_DEPTH]), u, a


def rotz(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def quat_from_R(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def tool_R(axis):
    """Rotation whose z column is `axis` (horizontal or not) and x column horizontal."""
    z = axis / np.linalg.norm(axis)
    x = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def make_trial(seed, d_cg=0.015, clearance=0.001, perturb=True):
    """Seed 0 is the nominal trial; other seeds draw tool placement, friction, mass and perception."""
    rng = np.random.default_rng(seed)
    tr = {"seed": int(seed), "d_cg": d_cg, "clearance": clearance, "dx": 0.0, "dy": 0.0, "dyaw": 0.0, "mu": 1.0,
          "mscale": 1.0, "perc": [0.0, 0.0, 0.0]}
    if seed != 0 and perturb:
        tr["dx"], tr["dy"] = (float(v) for v in rng.uniform(-0.003, 0.003, 2))
        tr["dyaw"] = float(rng.uniform(-math.radians(5), math.radians(5)))
        tr["mu"] = float(rng.uniform(0.7, 1.3))
        tr["mscale"] = float(rng.uniform(0.8, 1.2))
        tr["perc"] = [float(v) for v in rng.normal(0, [0.0005, 0.0005, math.radians(1.0)])]
    return tr


def world_layout(trial):
    """Nominal pick station and axes (palm yaw 0), the tool's true pose, posts, hole centre."""
    P, u, a = pinch_frame()
    s_nom = np.array([0.0, 0.0, POST_H + H.R_TOOL])
    c_nom = s_nom + trial["d_cg"] * a
    Rj = rotz(trial["dyaw"])
    c = c_nom + np.array([trial["dx"], trial["dy"], 0.0])
    axis = Rj @ a
    posts = [c_nom + POST_AT * a, c_nom - POST_AT * a]
    hole = s_nom - HOLE_AT * a
    return {"P": P, "u": u, "a": a, "s_nom": s_nom, "c": c, "axis": axis, "posts": posts, "hole": hole}


# -------------------------------------------------------------------------------------- scene

def chain_scene(spec, trial, geom_dirs=None, pad_tc=None):
    """MJCF of hand + palm stage + posts + tool + peg-hole block for a contact spec and a trial."""
    sp = B.parse_spec(spec)
    lay = world_layout(trial)
    off = B.d8_offsets()
    root = B.hand_tree(off)
    opt = root.find("option")
    opt.set("cone", "elliptic")
    opt.set("impratio", f"{sp['ir']:g}")
    opt.set("solver", "Newton")
    opt.set("iterations", "200")
    opt.set("tolerance", "1e-10")
    palm = root.find(".//body[@name='palm_pose']")
    palm.set("pos", "0 0 0")
    for i, (name, typ, axis) in enumerate((("palm_x", "slide", "1 0 0"), ("palm_y", "slide", "0 1 0"),
                                           ("palm_z", "slide", "0 0 1"), ("palm_yaw", "hinge", "0 0 1"))):
        el = ET.Element("joint", name=name, type=typ, axis=axis, damping="0", armature="0.01",
                        range="-1 1" if typ == "slide" else "-3.2 3.2")
        palm.insert(i, el)
    act = root.find("actuator")
    for name in ("palm_x", "palm_y", "palm_z"):
        ET.SubElement(act, "position", name=f"a_{name}", joint=name, kp=f"{PALM_KP:g}", kv=f"{PALM_KV:g}",
                      forcerange="-300 300", ctrlrange="-1 1")
    ET.SubElement(act, "position", name="a_palm_yaw", joint="palm_yaw", kp=f"{YAW_KP:g}", kv=f"{YAW_KV:g}",
                  forcerange="-50 50", ctrlrange="-3.2 3.2")
    mu = trial["mu"]
    model = sp["model"]
    fing = {"point3": {"condim": "3", "friction": f"{mu:g} 0 0"},
            "point4s": {"condim": "4", "friction": f"{mu:g} 0.001 0"},
            "spheres": {"condim": "3", "friction": f"{mu:g} 0 0"}}[model]
    for f in B.FINGERS:
        cap_el, sph_el = B.tip_geoms(root, f)
        for el in (cap_el, sph_el):
            for k, v in fing.items():
                el.set(k, v)
            el.set("priority", "1")
            el.set("solref", "0.006 1")
            el.set("solimp", "0.97 0.995 0.0005")
            el.set("name", f"{f}_{'tipsphere' if el is sph_el else 'distal'}")
    info = {}
    if model == "spheres" and geom_dirs is not None:
        s, rs = sp["s"], sp["rs"]
        cap = math.radians(sp.get("cap_deg", 45.0))
        area = 2 * math.pi * B.R_TIP ** 2 * (1 - math.cos(cap))
        n = max(1, int(round(area / s ** 2)))
        A_s = area / n
        K_s = sp["E"] / B.R_TIP * A_s * ((B.R_TIP - rs) / B.R_TIP) ** 2
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
                p = (B.R_TIP - rs) * dv
                ET.SubElement(tip, "geom", name=f"{f}_pad{i}", type="sphere", size=f"{rs}",
                              pos=f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f}", condim="3", friction=f"{mu:g} 0 0",
                              priority="1", solref=f"{tc:.6g} 1", solimp=f"{d0:.6g} {d0:.6g} 0.001 0.5 2",
                              rgba="0.75 0.4 0.25 1", mass="0")
    wb = root.find("worldbody")
    for el in list(wb):
        if el.tag == "geom" and el.get("name") == "floor":
            wb.remove(el)
    env = dict(condim="3", friction=f"{MU_ENV:g} 0 0", rgba="0.78 0.76 0.72 1")
    ET.SubElement(wb, "geom", name="table", type="box", pos="0.0 0.0 -0.01", size="0.30 0.30 0.01",
                  material="table_mat", condim="3", friction=f"{MU_ENV:g} 0 0")
    for i, p in enumerate(lay["posts"]):
        ET.SubElement(wb, "geom", name=f"post{i}", type="box", pos=f"{p[0]:.6f} {p[1]:.6f} {POST_H / 2:.6f}",
                      size=f"{POST_HALF} {POST_HALF} {POST_H / 2}", **env)
    _peg_block(wb, lay["hole"], H.R_TOOL + trial["clearance"], env)
    # tool: body z = its axis, centre c; mass scaled
    m_t = H.M_TOOL * trial["mscale"]
    tool = ET.SubElement(wb, "body", name="tool", pos=" ".join(f"{v:.7f}" for v in lay["c"]),
                         quat=" ".join(f"{v:.8f}" for v in quat_from_R(tool_R(lay["axis"]))))
    ET.SubElement(tool, "freejoint", name="tool_free")
    ET.SubElement(tool, "inertial", pos="0 0 0", mass=f"{m_t:.6g}",
                  diaginertia=f"{H.I_TOOL_T * trial['mscale']:.6g} {H.I_TOOL_T * trial['mscale']:.6g} "
                              f"{H.I_TOOL_A * trial['mscale']:.6g}")
    ET.SubElement(tool, "geom", name="tool", type="cylinder", size=f"{H.R_TOOL} {H.HL_TOOL}", condim="3",
                  friction=f"{MU_ENV:g} 0 0", rgba="0.55 0.6 0.68 1", solref="0.006 1", solimp="0.97 0.995 0.0005")
    ET.SubElement(tool, "geom", type="cylinder", pos=f"0 0 {H.HL_TOOL - 0.002}", size=f"{H.R_TOOL * 1.002} 0.002",
                  contype="0", conaffinity="0", rgba="0.85 0.3 0.2 1")          # lower end marked red
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="1 1 1", rgb2=".86 .89 .93",
                  width="64", height="64")
    g = root.find("visual").find("global")
    g.set("offwidth", "1280")
    g.set("offheight", "960")
    return ET.tostring(root, encoding="unicode"), info, lay


def _peg_block(wb, hole, r_h, env):
    """Chamfered round hole of radius r_h through a 30 mm block, as convex boxes (32 sectors)."""
    hx, hy = hole[0], hole[1]
    half_t = math.tan(math.pi / N_SEG)
    for k in range(N_SEG):
        th = 2 * math.pi * k / N_SEG
        e = np.array([math.cos(th), math.sin(th), 0.0])
        t = np.array([-math.sin(th), math.cos(th), 0.0])
        q = quat_from_R(np.column_stack([e, t, [0, 0, 1.0]]))
        qs = " ".join(f"{v:.8f}" for v in q)
        T = 0.012
        h_w = BLOCK_H - CHAMFER
        rc = r_h + T / 2
        ET.SubElement(wb, "geom", type="box", pos=f"{hx + rc * e[0]:.6f} {hy + rc * e[1]:.6f} {h_w / 2:.6f}",
                      quat=qs, size=f"{T / 2:.6f} {(r_h + T) * half_t * 1.04:.6f} {h_w / 2:.6f}", **env)
        rt = r_h + CHAMFER + T / 2
        ET.SubElement(wb, "geom", type="box", pos=f"{hx + rt * e[0]:.6f} {hy + rt * e[1]:.6f} {BLOCK_H - CHAMFER / 2:.6f}",
                      quat=qs, size=f"{T / 2:.6f} {(r_h + CHAMFER + T) * half_t * 1.04:.6f} {CHAMFER / 2:.6f}", **env)
        # chamfer: inner face on the cone from (r_h, BLOCK_H - CHAMFER) to (r_h + CHAMFER, BLOCK_H)
        n_c = (-e + np.array([0, 0, 1.0])) / math.sqrt(2)
        s_d = (e + np.array([0, 0, 1.0])) / math.sqrt(2)
        mid = np.array([hx, hy, 0]) + (r_h + CHAMFER / 2) * e + np.array([0, 0, BLOCK_H - CHAMFER / 2])
        tau = 0.004
        cpos = mid - n_c * tau / 2
        qc = " ".join(f"{v:.8f}" for v in quat_from_R(np.column_stack([t, n_c, s_d])))
        ET.SubElement(wb, "geom", type="box", pos=f"{cpos[0]:.6f} {cpos[1]:.6f} {cpos[2]:.6f}", quat=qc,
                      size=f"{(r_h + CHAMFER) * half_t * 1.04:.6f} {tau / 2:.6f} {CHAMFER * math.sqrt(2) / 2 * 1.02:.6f}",
                      **env)


# -------------------------------------------------------------------------------------- postures

def ik_model():
    """Hand at palm pose zero (palm frame = world) for posture IK."""
    import mujoco
    xml, _, _ = chain_scene("mj:point3", make_trial(0))
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)


def postures(gaps):
    """Thumb/index joint postures at each pinch gap (positive = open), by continuation from the first."""
    m, d = ik_model()
    P, u, a = pinch_frame()
    out, q, worst = {}, {"thumb": np.array([0.3, 0.33, 0.98]), "index": np.array([-0.3, 0.33, 0.98])}, 0.0
    for g in gaps:
        r = H.R_TOOL + B.R_TIP + g
        qq = {}
        for f, sgn in (("thumb", -1.0), ("index", 1.0)):
            qq[f], e = B.solve_tip(m, d, f, P + sgn * r * u, q[f])
            worst = max(worst, e)
        q = qq
        out[g] = np.concatenate([qq["thumb"], qq["index"], np.array(MIDDLE_PARK)])
    return out, worst


def contact_dirs(q9):
    """Tip-frame direction from each pinching tip's centre toward the tool (sphere-cap axis)."""
    import mujoco
    m, d = ik_model()
    _, u, _ = pinch_frame()
    for f, qq in zip(B.FINGERS, np.asarray(q9).reshape(3, 3)):
        for n, v in zip(B.JOINTS[f], qq):
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    mujoco.mj_kinematics(m, d)
    out = {}
    for f, sgn in (("thumb", 1.0), ("index", -1.0)):
        R = d.xmat[m.body(f"{f}_tip").id].reshape(3, 3)
        out[f] = R.T @ (sgn * u)
    return out


# ---------------------------------------------------------------------------------------- plants

class MjChainPlant:
    sim = "mujoco"

    def __init__(self, spec, trial, q_init, palm_init):
        import mujoco
        self.mj, self.sp, self.trial = mujoco, B.parse_spec(spec), trial
        geom_dirs = contact_dirs(q_init) if self.sp["model"] == "spheres" else None
        xml, self.info, self.lay = chain_scene(spec, trial, geom_dirs)
        if self.sp["model"] == "spheres":
            m0 = mujoco.MjModel.from_xml_string(xml)
            diag = m0.body_invweight0[m0.body("thumb_tip").id, 0] + m0.body_invweight0[m0.body("tool").id, 0]
            tc = self.sp["tr"] / 2.0
            d0 = 1.0 - 1.0 / (tc ** 2 * self.info["K_sphere"] * diag)
            if d0 < 0.05:
                raise ValueError(f"relaxation {self.sp['tr']} s below this pad's floor")
            xml, self.info, self.lay = chain_scene(spec, trial, geom_dirs, pad_tc=(tc, d0))
            self.info.update(solref_timeconst=tc, solimp_d0=d0, diagApprox=float(diag))
        self.xml = xml
        m = self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(m)
        self.fq = np.array([m.jnt_qposadr[m.joint(n).id] for f in B.FINGERS for n in B.JOINTS[f]])
        self.fv = np.array([m.jnt_dofadr[m.joint(n).id] for f in B.FINGERS for n in B.JOINTS[f]])
        self.pq = np.array([m.jnt_qposadr[m.joint(n).id] for n in ("palm_x", "palm_y", "palm_z", "palm_yaw")])
        self.pv = np.array([m.jnt_dofadr[m.joint(n).id] for n in ("palm_x", "palm_y", "palm_z", "palm_yaw")])
        self.tool = m.body("tool").id
        ja = m.body_jntadr[self.tool]
        self.tqa, self.tva = m.jnt_qposadr[ja], m.jnt_dofadr[ja]
        self.d.qpos[self.fq] = q_init
        self.d.qpos[self.pq] = palm_init
        self.d.ctrl[:9] = q_init
        self.d.ctrl[9:13] = palm_init
        self.side = {}
        for gi in range(m.ngeom):
            b = m.body(m.geom_bodyid[gi]).name
            for f in B.FINGERS:
                if b.startswith(f + "_") and m.geom_contype[gi]:
                    self.side[gi] = f
        self.support = {gi for gi in range(m.ngeom) if m.geom_bodyid[gi] == 0}
        self.lastN = {g: 0.0 for g in self.side}
        self.f6 = np.zeros(6)
        self.d.qfrc_applied[self.pv[2]] = m.body_subtreemass[m.body("palm_pose").id] * H.G
        mujoco.mj_forward(m, self.d)
        self.resched = self.sp["model"] == "point4s"
        self.fit = self.sp.get("fit", FIT)

    @property
    def t(self):
        return float(self.d.time)

    def set_targets(self, q9, palm4):
        self.d.ctrl[:9] = q9
        self.d.ctrl[9:13] = palm4

    def step(self, T):
        n = max(1, int(round(T / H.DT)))
        c, pw = self.fit
        mu = self.trial["mu"]
        for _ in range(n):
            if self.resched:
                for g, N in self.lastN.items():
                    self.m.geom_friction[g, 1] = mu * c * max(N, 1e-3) ** pw
            self.mj.mj_step(self.m, self.d)
            if self.resched:
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

    def state(self):
        d = self.d
        R9 = np.zeros(9)
        q = d.qpos[self.tqa + 3:self.tqa + 7].copy()
        self.mj.mju_quat2Mat(R9, q)
        R = R9.reshape(3, 3)
        return {"q": d.qpos[self.fq].copy(), "v": d.qvel[self.fv].copy(), "palm_q": d.qpos[self.pq].copy(),
                "palm_v": d.qvel[self.pv].copy(), "tool_pos": d.qpos[self.tqa:self.tqa + 3].copy(), "tool_quat": q,
                "tool_R": R, "tool_v": d.qvel[self.tva:self.tva + 3].copy(),
                "tool_w": R @ d.qvel[self.tva + 3:self.tva + 6]}

    def contacts(self):
        out = {f: {"N": 0.0, "n": 0} for f in B.FINGERS}
        out["support"] = {"N": 0.0, "n": 0}
        for i in range(self.d.ncon):
            cc = self.d.contact[i]
            g0, g1 = int(cc.geom[0]), int(cc.geom[1])
            b0, b1 = self.m.geom_bodyid[g0], self.m.geom_bodyid[g1]
            if self.tool not in (b0, b1):
                continue
            self.mj.mj_contactForce(self.m, self.d, i, self.f6)
            f = self.side.get(g0) or self.side.get(g1)
            key = f if f is not None else ("support" if (g0 in self.support or g1 in self.support) else None)
            if key is None:
                continue
            out[key]["N"] += float(self.f6[0])
            out[key]["n"] += 1
        return out


class DrakeChainPlant:
    """The same scene in Drake 1.57 SAP: compliant hydroelastic fingertips (or point contact), rigid tool
    and environment, PD servos with the calibrated gains, stiff PD palm stage with weight feedforward."""
    sim = "drake"

    def __init__(self, spec, trial, q_init, palm_init):
        from pydrake.all import (
            AddCompliantHydroelasticProperties, AddContactMaterial, AddMultibodyPlant,
            AddRigidHydroelasticProperties, CoulombFriction, DiagramBuilder, MultibodyPlantConfig,
            Parser, PdControllerGains, ProximityProperties, RoleAssign, Simulator,
        )
        self.sp, self.trial = B.parse_spec(spec), trial
        hydro = self.sp["model"] == "hydro"
        xml, self.info, self.lay = chain_scene("mj:point3", trial)
        self.xml_mirror = xml
        root = ET.fromstring(xml)
        for tag in ("visual", "keyframe", "actuator", "option", "statistic", "size", "contact"):
            for el in root.findall(tag):
                root.remove(el)
        for parent in root.iter():
            for child in list(parent):
                if child.tag in {"site", "light", "texture", "position", "motor", "camera"}:
                    parent.remove(child)
            if parent.tag == "material":
                for attr in ("texture", "texuniform", "texrepeat", "reflectance"):
                    parent.attrib.pop(attr, None)
            if parent.tag in ("geom", "default", "joint"):
                for attr in ("solref", "solimp", "condim", "frictionloss", "contype", "conaffinity", "group",
                             "priority"):
                    parent.attrib.pop(attr, None)
        # drop visual-only geoms (they carried contype 0 in MuJoCo)
        mj_root = ET.fromstring(xml)
        visual_only = {id(e) for e in mj_root.iter("geom") if e.get("contype") == "0"}
        del visual_only
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "geom" and child.get("rgba") in ("0.85 0.3 0.2 1",):
                    parent.remove(child)
        b = DiagramBuilder()
        cfg = MultibodyPlantConfig(time_step=H.DT, discrete_contact_approximation="sap",
                                   contact_model="hydroelastic_with_fallback" if hydro else "point")
        plant, sg = AddMultibodyPlant(cfg, b)
        parser = Parser(plant, sg)
        self.instance, = parser.AddModelsFromString(ET.tostring(root, encoding="unicode"), "xml")
        self.finger_joints = [n for f in B.FINGERS for n in B.JOINTS[f]]
        self.palm_joints = ["palm_x", "palm_y", "palm_z", "palm_yaw"]
        for n in self.finger_joints:
            j = plant.GetJointByName(n)
            act = plant.AddJointActuator("a_" + n, j, B.PLANT["forcerange"])
            act.set_controller_gains(PdControllerGains(p=B.PLANT["kp"], d=B.PLANT["kv"]))
            act.set_default_rotor_inertia(0.001)
            act.set_default_gear_ratio(1.0)
        for n in self.palm_joints:
            j = plant.GetJointByName(n)
            kp, kd, eff = (YAW_KP, YAW_KV, 50.0) if n == "palm_yaw" else (PALM_KP, PALM_KV, 300.0)
            act = plant.AddJointActuator("a_" + n, j, eff)
            act.set_controller_gains(PdControllerGains(p=kp, d=kd))
        sid = plant.get_source_id()
        mu = trial["mu"]
        # MuJoCo uses the pad's mu for pad-tool and MU_ENV for tool-environment; Drake combines two
        # geometries' coefficients as 2 m1 m2 / (m1 + m2), so the environment gets m_e with
        # 2 mu m_e / (mu + m_e) = MU_ENV
        mu_e = MU_ENV * mu / (2 * mu - MU_ENV)
        self.geom_finger, self.tool_gids = {}, set()

        def assign(gid, kind, fr):
            pp = ProximityProperties()
            if hydro and kind == "compliant":
                AddCompliantHydroelasticProperties(self.sp["res"], self.sp["E"], pp)
            elif hydro and kind == "rigid":
                AddRigidHydroelasticProperties(0.0005, pp)
            AddContactMaterial(dissipation=10.0, point_stiffness=1e4, friction=CoulombFriction(fr, fr), properties=pp)
            if "rt" in self.sp:
                pp.AddProperty("material", "relaxation_time", self.sp["rt"])
            sg.AssignRole(sid, gid, pp, RoleAssign.kReplace)
        for f in B.FINGERS:
            for bn in (f"{f}_pip_frame", f"{f}_tip"):
                for gid in plant.GetCollisionGeometriesForBody(plant.GetBodyByName(bn)):
                    self.geom_finger[gid] = f
                    assign(gid, "compliant", mu)
        tool = plant.GetBodyByName("tool")
        for gid in plant.GetCollisionGeometriesForBody(tool):
            self.tool_gids.add(gid)
            assign(gid, "rigid", mu)
        self.env_gids = set(plant.GetCollisionGeometriesForBody(plant.world_body()))
        for gid in self.env_gids:
            assign(gid, "rigid", mu_e)
        plant.Finalize()
        self.plant, self.sg, self.toolb = plant, sg, tool
        self.diagram = b.Build()
        self.simulator = Simulator(self.diagram)
        self.ctx = self.simulator.get_mutable_context()
        self.pc = plant.GetMyMutableContextFromRoot(self.ctx)
        for n, q in zip(self.finger_joints, q_init):
            plant.GetJointByName(n).set_angle(self.pc, float(q))
        for n, q in zip(self.palm_joints, palm_init):
            j = plant.GetJointByName(n)
            (j.set_angle if n == "palm_yaw" else j.set_translation)(self.pc, float(q))
        self.act_order = [plant.get_joint_actuator(i).joint().name() for i in plant.GetJointActuatorIndices(self.instance)]
        self.m_hand = sum(plant.GetBodyByName(n).default_mass() for n in self._hand_bodies(plant))
        self.set_targets(q_init, palm_init)
        self._fix_wrench()
        self.simulator.Initialize()

    def _hand_bodies(self, plant):
        out = []
        for i in plant.GetBodyIndices(self.instance):
            bd = plant.get_body(i)
            if bd.name() not in ("tool",) and bd.name() != "world":
                out.append(bd.name())
        return out

    @property
    def t(self):
        return float(self.ctx.get_time())

    def set_targets(self, q9, palm4):
        tgt = dict(zip(self.finger_joints, np.asarray(q9, float)))
        tgt.update(zip(self.palm_joints, np.asarray(palm4, float)))
        pos = np.array([tgt[n] for n in self.act_order])
        ff = np.array([self.m_hand * H.G if n == "palm_z" else 0.0 for n in self.act_order]) if hasattr(self, "m_hand") \
            else np.zeros(len(pos))
        self.plant.get_actuation_input_port().FixValue(self.pc, ff)
        self.plant.get_desired_state_input_port(self.instance).FixValue(self.pc, np.r_[pos, np.zeros(len(pos))])

    def _fix_wrench(self):
        self.plant.get_applied_spatial_force_input_port().FixValue(self.pc, [])

    def step(self, T):
        self.simulator.AdvanceTo(self.ctx.get_time() + T)

    def state(self):
        p = self.plant
        q = np.array([p.GetJointByName(n).get_angle(self.pc) for n in self.finger_joints])
        v = np.array([p.GetJointByName(n).get_angular_rate(self.pc) for n in self.finger_joints])
        pq, pv = [], []
        for n in self.palm_joints:
            j = p.GetJointByName(n)
            if n == "palm_yaw":
                pq.append(j.get_angle(self.pc))
                pv.append(j.get_angular_rate(self.pc))
            else:
                pq.append(j.get_translation(self.pc))
                pv.append(j.get_translation_rate(self.pc))
        X = p.EvalBodyPoseInWorld(self.pc, self.toolb)
        V = p.EvalBodySpatialVelocityInWorld(self.pc, self.toolb)
        R = X.rotation().matrix()
        return {"q": q, "v": v, "palm_q": np.array(pq), "palm_v": np.array(pv), "tool_pos": np.array(X.translation()),
                "tool_quat": X.rotation().ToQuaternion().wxyz(), "tool_R": R, "tool_v": np.array(V.translational()),
                "tool_w": np.array(V.rotational())}

    def contacts(self):
        out = {f: {"N": 0.0, "n": 0} for f in B.FINGERS}
        out["support"] = {"N": 0.0, "n": 0}
        cr = self.plant.get_contact_results_output_port().Eval(self.pc)
        X = self.plant.EvalBodyPoseInWorld(self.pc, self.toolb)
        a = X.rotation().matrix()[:, 2]
        for i in range(cr.num_hydroelastic_contacts()):
            info = cr.hydroelastic_contact_info(i)
            srf = info.contact_surface()
            ids = (srf.id_M(), srf.id_N())
            if not (set(ids) & self.tool_gids):
                continue
            f = self.geom_finger.get(ids[0]) or self.geom_finger.get(ids[1])
            F = np.array(info.F_Ac_W().translational())
            if f is None:
                out["support"]["N"] += float(np.linalg.norm(F))
                out["support"]["n"] += 1
                continue
            r = np.array(srf.centroid()) - X.translation()
            r -= (r @ a) * a
            out[f]["N"] += float(abs(F @ r) / max(np.linalg.norm(r), 1e-9))
            out[f]["n"] += srf.num_faces()
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            pp = info.point_pair()
            if not ({pp.id_A, pp.id_B} & self.tool_gids):
                continue
            f = self.geom_finger.get(pp.id_A) or self.geom_finger.get(pp.id_B)
            Nn = float(abs(np.array(info.contact_force()) @ np.array(pp.nhat_BA_W)))
            key = f if f is not None else ("support" if ({pp.id_A, pp.id_B} & self.env_gids) else None)
            if key is None:
                continue
            out[key]["N"] += Nn
            out[key]["n"] += 1
        return out

    def point_contacts(self):
        """[(finger, pos, normal into the pad, normal force)] for point-pair contacts between fingertips and tool."""
        out = []
        cr = self.plant.get_contact_results_output_port().Eval(self.pc)
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            pp = info.point_pair()
            if not ({pp.id_A, pp.id_B} & self.tool_gids):
                continue
            f = self.geom_finger.get(pp.id_A) or self.geom_finger.get(pp.id_B)
            if f is None:
                continue
            n = np.array(pp.nhat_BA_W)
            if pp.id_B in self.geom_finger:
                n = -n
            fN = float(abs(np.array(info.contact_force()) @ np.array(pp.nhat_BA_W)))
            out.append((f, np.array(info.contact_point()), n, fN))
        return out

    def surfaces(self):
        """Hydroelastic contact surfaces between fingertips and tool: list of (finger, centroids, normals,
        areas, pressures) for rendering."""
        out = []
        cr = self.plant.get_contact_results_output_port().Eval(self.pc)
        for i in range(cr.num_hydroelastic_contacts()):
            srf = cr.hydroelastic_contact_info(i).contact_surface()
            ids = (srf.id_M(), srf.id_N())
            if not (set(ids) & self.tool_gids):
                continue
            f = self.geom_finger.get(ids[0]) or self.geom_finger.get(ids[1])
            if f is None:
                continue
            if srf.is_triangle():
                mesh, field = srf.tri_mesh_W(), srf.tri_e_MN()
            else:
                mesh, field = srf.poly_mesh_W(), srf.poly_e_MN()
            cen, nor, area, pres = [], [], [], []
            for k in range(mesh.num_elements()):
                c = np.array(mesh.element_centroid(k))
                cen.append(c)
                nor.append(np.array(mesh.face_normal(k)))
                area.append(mesh.area(k))
                pres.append(field.EvaluateCartesian(k, c))
            out.append((f, np.array(cen), np.array(nor), np.array(area), np.array(pres)))
        return out


def make_plant(spec, trial, q_init, palm_init):
    sp = B.parse_spec(spec)
    return (MjChainPlant if sp["sim"] == "mj" else DrakeChainPlant)(spec, trial, q_init, palm_init)


# -------------------------------------------------------------------------------------- the brake

def brake_torque_capacity(N, mu, W):
    """Pinch friction torque about the pinch axis, both pads, with the tool's weight W carried
    tangentially (elliptic cone): 2 mu N rbar(N) sqrt(1 - (W / 2 mu N)^2)."""
    c, p = FIT
    if N <= W / (2 * mu):
        return 0.0
    return 2 * mu * N * c * N ** p * math.sqrt(1 - (W / (2 * mu * N)) ** 2)


def brake_force_for(tau, mu, W, N_lo, N_hi):
    """Smallest N in [N_lo, N_hi] whose capacity reaches tau (bisection)."""
    if tau <= brake_torque_capacity(N_lo, mu, W):
        return N_lo
    if tau >= brake_torque_capacity(N_hi, mu, W):
        return N_hi
    lo, hi = N_lo, N_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if brake_torque_capacity(mid, mu, W) < tau:
            lo = mid
        else:
            hi = mid
    return hi


def minjerk(t, T):
    s = min(max(t / T, 0.0), 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5, (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T, \
        (60 * s - 180 * s ** 2 + 120 * s ** 3) / T ** 2


class ClosedBrake:
    """Tracks phi_ref (min-jerk, 0 -> 90 deg over T_sw) with the pinch force: the friction torque the
    swing needs, I phi'' = m g d cos(phi) - tau, is inverted through the nominal brake law. The force
    floor starts at twice the nominal slip limit W / 2 mu, falls while the tool is stalled short of
    vertical, and is raised 50 % whenever the tool slides along its axis."""

    def __init__(self, d_cg, T_sw=1.8, kp=64.0, kd=24.0, floor_k=2.0, lam=1.5, slip_v=0.003, N_max=6.0):
        self.d, self.T, self.kp, self.kd = d_cg, T_sw, kp, kd
        self.m, self.mu = H.M_TOOL, 1.0                        # nominal model; the trial's are unknown
        self.W = self.m * H.G
        self.I = H.I_TOOL_T + self.m * d_cg ** 2
        self.floor = floor_k * self.W / (2 * self.mu)
        self.lam, self.slip_v, self.N_max = lam, slip_v, N_max
        self.last_raise = -1.0
        self.n_raise = 0

    def __call__(self, t, phi, phidot, v_ax, dt):
        s, sd, sdd = minjerk(t, self.T)
        ph_r, phd_r, phdd_r = math.radians(90) * s, math.radians(90) * sd, math.radians(90) * sdd
        a_des = phdd_r + self.kd * (phd_r - phidot) + self.kp * (ph_r - phi)
        tau = self.m * H.G * self.d * math.cos(phi) - self.I * a_des
        if abs(v_ax) > self.slip_v and t - self.last_raise > 0.2:
            self.floor *= 1.5
            self.last_raise = t
            self.n_raise += 1
        elif t > self.T and abs(phi - math.radians(90.0)) > math.radians(1.0) and abs(phidot) < math.radians(2.0):
            self.floor *= math.exp(-self.lam * dt)          # stalled off vertical (either side): loosen
        self.floor = max(self.floor, 0.05)
        return brake_force_for(tau, self.mu, self.W, self.floor, self.N_max), math.degrees(ph_r)


# ---------------------------------------------------------------------------------- the chain

def run_chain(spec, seed=0, brake="closed", d_cg=0.015, clearance=0.001, film=None, perturb=True,
              F1=0.5, T_ramp=3.0, insert_force=None, verbose=False, film_every=4, closeup=None):
    """One chain rollout. Returns a result row (metrics per stage, a decimated trace)."""
    t0w = time.time()
    trial = make_trial(seed, d_cg, clearance, perturb)
    lay = world_layout(trial)
    P, u, a = lay["P"], lay["u"], lay["a"]
    path, ik_err = postures([GAP_OPEN] + [float(g) for g in np.linspace(GAP_OPEN - 0.001, -SQUEEZE_MAX, 12)])
    q_open = path[GAP_OPEN]
    q_touch_nom = path[min(path, key=lambda g: abs(g))]
    # perceived tool pose -> palm pick pose
    perc = trial["perc"]
    ax_hat = rotz(perc[2]) @ lay["axis"]
    c_hat = lay["c"] + np.array([perc[0], perc[1], 0.0])
    psi = math.atan2(ax_hat[1], ax_hat[0]) - math.atan2(a[1], a[0])
    psi = (psi + math.pi) % (2 * math.pi) - math.pi
    s_hat = c_hat - d_cg * ax_hat
    pick = np.r_[s_hat - rotz(psi) @ P, psi]
    pick[3] = psi
    start = pick.copy()
    start[2] += 0.060
    plant = make_plant(spec, trial, q_open, start)
    mirror = C.Mirror(plant.xml if plant.sim == "mujoco" else plant.xml_mirror)
    if plant.sim == "mujoco" and B.parse_spec(spec)["model"] == "spheres":
        mirror = C.Mirror(chain_scene("mj:point3", trial)[0])
    kp = B.PLANT["kp"]
    dt = 1.0 / RATE
    n_sub = int(round(dt / H.DT))
    q_ref = q_open.copy()
    palm_tgt = start.copy()
    phase, t_phase = "approach", 0.0
    mode = {"thumb": "approach", "index": "approach"}
    F = {"thumb": 0.0, "index": 0.0}
    t_contact = {}
    rows, frames, events = [], [], {}
    renderer = ChainRenderer(plant, lay, spec, closeup=closeup if closeup is not None else True) if film else None
    brake_ctl = ClosedBrake(d_cg) if brake == "closed" else None
    phi_prev, sax_prev = None, None
    a0 = u_w = None
    res = {"exp": "chain", "spec": spec, "brake": brake, "trial": trial, "ik_err_mm": ik_err * 1e3}
    stable_t = 0.0
    lift_from = palm_from = None
    hole = lay["hole"]
    T_MAX = 22.0

    def pinch_targets(q_ref, Fd):
        st_q = q_ref.copy()
        for i, f in enumerate(("thumb", "index")):
            if mode[f] != "force":
                continue
            fr = mirror.frame(f)
            mirror.mj.mj_jac(mirror.m, mirror.d, mirror.jacp, None, fr["p_fing"], mirror.tip[f])
            J = mirror.jacp[:, mirror.fv[f]]
            st_q[3 * i:3 * i + 3] = q_ref[3 * i:3 * i + 3] + J.T @ (-Fd[f] * fr["n"]) / kp
        return st_q

    def palm_move(frm, to, t, T):
        s, _, _ = minjerk(t, T)
        return frm + s * (to - frm)

    perf = {"phys": {}, "sim": {}, "ctrl": 0.0, "log": 0.0, "render": 0.0}
    while plant.t < T_MAX and phase != "end":
        tc0 = time.perf_counter()
        st = plant.state()
        mirror.set_state(st["q"], st["v"], st["tool_pos"], st["tool_quat"], st["tool_v"], st["tool_w"],
                         st["palm_q"], st["palm_v"])
        t = plant.t
        tp = t - t_phase
        R = st["tool_R"]
        axis = R[:, 2]
        frs = {f: mirror.frame(f) for f in ("thumb", "index")}
        pinch_mid = 0.5 * (mirror.d.xpos[mirror.tip["thumb"]] + mirror.d.xpos[mirror.tip["index"]])
        sax = float((st["tool_pos"] - pinch_mid) @ axis)
        v_ax = 0.0 if sax_prev is None else (sax - sax_prev) / dt
        sax_prev = sax
        phi = phidot = None
        if a0 is not None:
            phi_d, tilt = B.swing_angle(axis, a0, u_w)
            phi = math.radians(phi_d)
            phidot = 0.0 if phi_prev is None else (phi - phi_prev) / dt
            phi_prev = phi
        Fcmd = None
        # ------------------------------------------------------------------ phase machine
        if phase == "approach":
            palm_tgt = palm_move(start, pick, tp, 1.0)
            if tp >= 1.1:
                phase, t_phase = "close", t
        elif phase == "close":
            for i, f in enumerate(("thumb", "index")):
                if mode[f] == "approach":
                    A, bvec, fr = mirror.twist(f, frs[f])
                    vref = np.zeros(6)
                    vref[3] = -0.015 / C.ELL
                    w = np.array([0, 0, 0, 1.0, 1.0, 1.0])
                    qm = st["q"][3 * i:3 * i + 3]
                    uu = C.solve_u(A, bvec, vref, w, qm, mirror.lo[f], mirror.hi[f], dt,
                                   q_touch_nom[3 * i:3 * i + 3])
                    q_ref[3 * i:3 * i + 3] = np.clip(q_ref[3 * i:3 * i + 3] + uu * dt, qm - C.LEAD, qm + C.LEAD)
                    if fr["dist"] <= 0.0003:
                        # touching posture: the measured one moved the remaining gap along the normal
                        mode[f] = "force"
                        mirror.mj.mj_jac(mirror.m, mirror.d, mirror.jacp, None, fr["p_fing"], mirror.tip[f])
                        Jf = mirror.jacp[:, mirror.fv[f]]
                        q_ref[3 * i:3 * i + 3] = qm + np.linalg.pinv(Jf) @ (-fr["dist"] * fr["n"])
                        t_contact[f] = t
                if mode[f] == "force":
                    F[f] = F0 * min(1.0, (t - t_contact[f]) / 0.25)
            if all(m_ == "force" for m_ in mode.values()) and t - max(t_contact.values()) > 0.5:
                events["closed_t"] = t
                phase, t_phase, lift_from = "lift", t, palm_tgt.copy()
            elif tp > 2.5:
                events["close_timeout"] = True
                phase, t_phase, lift_from = "lift", t, palm_tgt.copy()
        elif phase == "lift":
            palm_tgt = palm_move(lift_from, lift_from + np.array([0, 0, LIFT, 0]), tp, 1.2)
            if tp >= 1.5:
                a0 = axis.copy()
                u_w = rotz(st["palm_q"][3]) @ u
                phi_prev = None
                events["lift_axis_tilt_deg"] = float(math.degrees(math.asin(abs(a0[2]))))
                phase, t_phase = "brake", t
        elif phase == "brake":
            if brake == "open":
                if tp < T_ramp:
                    Fcmd = F0 * (F1 / F0) ** (tp / T_ramp)
                else:
                    Fcmd = F1
                done = tp >= T_ramp + 1.0
            else:
                Fcmd, phi_ref = brake_ctl(tp, phi, phidot, v_ax, dt)
                ok = abs(math.degrees(phi) - 90) < 2.5 and abs(math.degrees(phidot)) < 5
                stable_t = stable_t + dt if ok else 0.0
                done = (tp > brake_ctl.T and stable_t >= 0.2) or tp > brake_ctl.T + 2.5
            F["thumb"] = F["index"] = Fcmd
            if done:
                events["brake_end_t"] = tp
                events["phi_brake_end"] = math.degrees(phi)
                phase, t_phase = "hold", t
                F_from = Fcmd
        elif phase == "hold":
            F["thumb"] = F["index"] = F_from + (F_HOLD - F_from) * min(1.0, tp / 0.3)
            if tp >= 0.7:
                events["phi_hold"] = math.degrees(phi)
                end = st["tool_pos"] + H.HL_TOOL * axis
                rel = end - np.r_[st["palm_q"][:3]]
                palm_from = palm_tgt.copy()
                palm_to = palm_from.copy()
                palm_to[:3] = np.array([hole[0], hole[1], BLOCK_H + 0.008]) - rel
                phase, t_phase = "transport", t
        elif phase == "transport":
            palm_tgt = palm_move(palm_from, palm_to, tp, 1.2)
            if tp >= 1.4:
                end = st["tool_pos"] + H.HL_TOOL * axis
                corr = np.array([hole[0] - end[0], hole[1] - end[1]])
                events["aim_err_mm"] = float(np.linalg.norm(corr) * 1e3)
                palm_from = palm_tgt.copy()
                palm_to = palm_from.copy()
                palm_to[:2] += corr
                phase, t_phase = "aim", t
        elif phase == "aim":
            palm_tgt = palm_move(palm_from, palm_to, tp, 0.3)
            if tp >= 0.4:
                palm_from = palm_tgt.copy()
                phase, t_phase = "insert", t
        elif phase == "insert":
            palm_tgt = palm_from - np.array([0, 0, min(tp, 2.0) * 0.015, 0])
            if tp >= 2.3:
                phase, t_phase = "end", t
        if phase in ("lift", "brake", "hold", "transport", "aim", "insert") and Fcmd is None and phase != "brake":
            pass
        q_cmd = pinch_targets(q_ref, F) + mirror.gravity() / kp
        plant.set_targets(q_cmd, palm_tgt)
        tc1 = time.perf_counter()
        perf["ctrl"] += tc1 - tc0
        # ------------------------------------------------------------------ log
        k = int(round(t / dt))
        if k % 2 == 0 or phase == "end":
            cc = plant.contacts()
            end = st["tool_pos"] + H.HL_TOOL * axis
            rows.append([round(t, 3), phase, F["thumb"], F["index"],
                         None if phi is None else round(math.degrees(phi), 3),
                         None if phidot is None else round(math.degrees(phidot), 2),
                         cc["thumb"]["N"], cc["index"]["N"], cc["middle"]["N"], cc["support"]["N"],
                         cc["thumb"]["n"], cc["index"]["n"], round(sax * 1e3, 3), round(float(end[2]) * 1e3, 2),
                         round(float(np.linalg.norm(end[:2] - hole[:2])) * 1e3, 2),
                         round(float(math.degrees(math.acos(min(1.0, abs(axis[2]))))), 2),
                         None if brake_ctl is None else round(brake_ctl.floor, 4),
                         None if u_w is None else round(math.degrees(math.asin(max(-1.0, min(1.0, float(axis @ u_w))))), 3),
                         round(math.degrees(math.asin(max(-1.0, min(1.0, float(
                             (mirror.d.xpos[mirror.tip["index"]] - mirror.d.xpos[mirror.tip["thumb"]])[2]
                             / np.linalg.norm(mirror.d.xpos[mirror.tip["index"]] - mirror.d.xpos[mirror.tip["thumb"]])))))), 3)])
        tc2 = time.perf_counter()
        perf["log"] += tc2 - tc1
        if renderer is not None and k % film_every == 0:
            frames.append(renderer.frame(plant, st, label(spec, brake), phase, phi, F["thumb"], frs["thumb"]))
        tc3 = time.perf_counter()
        perf["render"] += tc3 - tc2
        plant.step(dt)
        perf["phys"][phase] = perf["phys"].get(phase, 0.0) + time.perf_counter() - tc3
        perf["sim"][phase] = perf["sim"].get(phase, 0.0) + dt
    tr = rows
    cols = ["t", "phase", "F_thumb", "F_index", "phi", "phidot", "N_thumb", "N_index", "N_middle", "N_support",
            "n_thumb", "n_index", "sax_mm", "end_z_mm", "end_off_mm", "tilt_deg", "floor", "oop_deg", "pinch_tilt_deg"]
    res.update(events)
    res.update(_score(tr, cols, events, lay))
    res["wall_s"] = time.time() - t0w
    res["sim_s"] = plant.t
    res["perf"] = perf
    res["info"] = getattr(plant, "info", {})
    res["trace_cols"] = cols
    res["trace"] = tr
    if renderer is not None:
        H.write_mp4(frames, film, fps=25)
        res["film"] = str(film)
    return res


def _score(tr, cols, ev, lay):
    ix = {c: i for i, c in enumerate(cols)}

    def last(phase):
        r = [row for row in tr if row[ix["phase"]] == phase]
        return r[-1] if r else None
    out = {}
    lift = last("lift")
    out["pick_ok"] = bool(lift is not None and lift[ix["n_thumb"]] > 0 and lift[ix["n_index"]] > 0
                          and lift[ix["N_support"]] < 0.05)
    b = last("brake")
    out["phi_end"] = None if b is None or b[ix["phi"]] is None else b[ix["phi"]]
    brows = [row for row in tr if row[ix["phase"]] == "brake" and row[ix["phi"]] is not None]
    out["phi_max"] = max((row[ix["phi"]] for row in brows), default=None)
    out["rate_max_dps"] = max((abs(row[ix["phidot"]]) for row in brows), default=None)
    held = lambda row: row is not None and row[ix["n_thumb"]] > 0 and row[ix["n_index"]] > 0  # noqa: E731
    out["brake_held"] = held(b)
    out["brake_ok"] = bool(out["brake_held"] and out["phi_end"] is not None and abs(out["phi_end"] - 90) <= 5)
    h = last("hold")
    out["phi_hold_end"] = None if h is None else h[ix["phi"]]
    out["hold_ok"] = bool(held(h) and h[ix["phi"]] is not None and abs(h[ix["phi"]] - 90) <= 5)
    e = last("end") or last("insert")
    out["end_depth_mm"] = None if e is None else BLOCK_H * 1e3 - e[ix["end_z_mm"]]
    out["end_tilt_deg"] = None if e is None else e[ix["tilt_deg"]]
    out["end_held"] = held(e)
    out["insert_ok"] = bool(e is not None and held(e) and out["end_depth_mm"] >= HOLE_GOAL * 1e3 - 5
                            and e[ix["end_off_mm"]] < H.R_TOOL * 1e3)
    sl = [row[ix["sax_mm"]] for row in tr if row[ix["phase"]] in ("brake", "hold", "transport", "aim", "insert", "end")]
    out["slip_mm"] = (max(sl) - min(sl)) if sl else None
    out["chain_ok"] = bool(out["pick_ok"] and out["brake_ok"] and out["hold_ok"] and out["insert_ok"])
    return out


def label(spec, brake=None):
    sp = B.parse_spec(spec)
    if sp["sim"] == "drake":
        s = "Drake point" if sp["model"] == "point" else f"Drake hydroelastic {sp.get('rt', 0.1):g} s"
        if sp["model"] == "hydro" and abs(sp["res"] - 1e-3) > 1e-9:
            s += f", {sp['res'] * 1e3:g} mm mesh"
    else:
        s = {"point3": "MuJoCo point, condim 3", "point4s": "MuJoCo condim 4, mu_t rescheduled",
             "spheres": f"MuJoCo {sp.get('s', 0) * 1e3:g} mm spheres"}[sp["model"]]
    if brake:
        s += ", " + {"open": "open-loop brake", "closed": "closed-loop brake"}[brake]
    return s


# -------------------------------------------------------------------------------------- films

def heat(x):
    """Dark violet -> red -> orange -> pale yellow for x in [0, 1]."""
    x = min(max(float(x), 0.0), 1.0)
    stops = [(0.0, (0.22, 0.06, 0.38)), (0.35, (0.70, 0.13, 0.32)), (0.7, (0.96, 0.47, 0.10)), (1.0, (1.0, 0.94, 0.45))]
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if x <= x1:
            f = (x - x0) / (x1 - x0)
            return [c0[i] + f * (c1[i] - c0[i]) for i in range(3)] + [1.0]
    return list(stops[-1][1]) + [1.0]


SCALE_POINT_N, SCALE_PRESSURE = 4.0, 3e5      # colour full scale in the close-up: point force; pressure, or a
                                               # sphere's force over its share of pad area


class ChainRenderer:
    """Wide shot (480 x 360, camera along the pinch axis) beside a close-up of the thumb pad (360 x 360), tool and
    index see-through. Sphere-packed tips are drawn as their collision spheres (the visual tip sphere is a ghost in
    the wide shot and hidden in the close-up), each sphere coloured by its normal force; Drake hydroelastic contact
    is drawn as its contact-surface faces coloured by pressure; point contacts as a dot coloured by force with a
    normal-force arrow. The black dot is the centre of pressure."""

    def __init__(self, plant, lay, spec, w=480, h=360, cw=360, closeup=True):
        import mujoco
        self.mj = mujoco
        self.sp = B.parse_spec(spec)
        self.spheres = self.sp["model"] == "spheres"
        if plant.sim == "mujoco":
            self.m = plant.m
        else:
            self.m = mujoco.MjModel.from_xml_string(plant.xml_mirror)
        m = self.m
        self.d = mujoco.MjData(m)
        self.r = mujoco.Renderer(m, h, w)
        self.rc = mujoco.Renderer(m, h, cw, max_geom=20000) if closeup else None
        self.cam = mujoco.MjvCamera()
        mid = 0.5 * (lay["s_nom"] + lay["hole"])
        self.cam.lookat[:] = [mid[0], mid[1], 0.075]
        self.cam.distance = 0.42
        self.cam.azimuth = math.degrees(math.atan2(lay["u"][1], lay["u"][0]))
        self.cam.elevation = -12.0
        self.camc = mujoco.MjvCamera()
        self.camc.distance, self.camc.elevation = 0.036, -20.0
        self.opt = mujoco.MjvOption()
        self.optc = mujoco.MjvOption()
        self.rgba0 = m.geom_rgba.copy()
        self.tool_g = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "tool"]
        self.tip_vis = [m.geom(f"{f}_tipsphere").id for f in ("thumb", "index")]
        self.index_g = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("index_")]
        self.pads = {f: [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(f"{f}_pad")] for f in ("thumb", "index")}
        n_sph = max(1, len(self.pads["thumb"]))
        cap = math.radians(self.sp.get("cap_deg", 45.0))
        self.A_s = 2 * math.pi * B.R_TIP ** 2 * (1 - math.cos(cap)) / n_sph      # each sphere's share of pad area
        self.closeup = closeup

    def sync(self, plant, st):
        m, d = self.m, self.d
        if plant.sim == "mujoco":
            d.qpos[:] = plant.d.qpos
            d.qvel[:] = plant.d.qvel
        else:
            for n, q in zip(plant.finger_joints, st["q"]):
                d.qpos[m.jnt_qposadr[m.joint(n).id]] = q
            for n, q in zip(plant.palm_joints, st["palm_q"]):
                d.qpos[m.jnt_qposadr[m.joint(n).id]] = q
            ja = m.body_jntadr[m.body("tool").id]
            qa = m.jnt_qposadr[ja]
            d.qpos[qa:qa + 3] = st["tool_pos"]
            d.qpos[qa + 3:qa + 7] = st["tool_quat"]
        self.mj.mj_forward(m, d)

    def thumb_contacts(self, plant):
        """[(pos, normal, force_N, geom)] for thumb-tool contacts in a MuJoCo plant (from the plant's own data)."""
        out = []
        f6 = np.zeros(6)
        pm, pd = plant.m, plant.d
        for i in range(pd.ncon):
            cc = pd.contact[i]
            g0, g1 = int(cc.geom[0]), int(cc.geom[1])
            if plant.tool not in (pm.geom_bodyid[g0], pm.geom_bodyid[g1]):
                continue
            g = g0 if plant.side.get(g0) == "thumb" else (g1 if plant.side.get(g1) == "thumb" else None)
            if g is None:
                continue
            self.mj.mj_contactForce(pm, pd, i, f6)
            n = np.array(cc.frame[:3])
            if pm.geom_bodyid[g0] == plant.tool:
                n = -n                                   # normal pointing from the tool into the pad
            out.append((np.array(cc.pos), n, float(f6[0]), g))
        return out

    def frame(self, plant, st, title, phase, phi, F, fr_thumb=None):
        self.sync(plant, st)
        m = self.m
        mujoco = self.mj
        ph = "  -  " if phi is None else f"{math.degrees(phi):5.1f}"
        sphere_force = {}
        cons = []
        if plant.sim == "mujoco":
            cons = self.thumb_contacts(plant)
            for pos, n, fN, g in cons:
                sphere_force[g] = sphere_force.get(g, 0.0) + fN
        # ---------------------------------------------------------------- wide
        m.geom_rgba[:] = self.rgba0
        if self.spheres:
            for g in self.tip_vis:
                m.geom_rgba[g, 3] = 0.12
            for f in ("thumb", "index"):
                for g in self.pads[f]:
                    m.geom_rgba[g] = heat(sphere_force[g] / self.A_s / SCALE_PRESSURE) if g in sphere_force else [0.95, 0.55, 0.15, 1]
        self.r.update_scene(self.d, self.cam, self.opt)
        wide = H.annotate_bottom(H.annotate(self.r.render().copy(), title, size=14),
                                 f"{phase:9s} phi {ph} deg  F {F:4.2f} N  t {plant.t:5.2f} s", size=13)
        if not self.closeup or fr_thumb is None:
            m.geom_rgba[:] = self.rgba0
            return wide
        # ---------------------------------------------------------------- close-up of the thumb pad
        for g in self.tool_g:
            m.geom_rgba[g, 3] = 0.22
        for g in self.index_g:
            m.geom_rgba[g, 3] = 0.12
        if self.spheres:
            for g in self.tip_vis:
                m.geom_rgba[g, 3] = 0.0
            for g in self.pads["thumb"]:
                m.geom_rgba[g] = heat(sphere_force[g] / self.A_s / SCALE_PRESSURE) if g in sphere_force else [0.80, 0.80, 0.82, 1]
            for g in self.pads["index"]:
                m.geom_rgba[g, 3] = 0.0
        else:
            for g in self.tip_vis[:1]:
                m.geom_rgba[g, 3] = 0.35
        n = fr_thumb["n"]
        self.camc.lookat[:] = fr_thumb["p_fing"]
        self.camc.azimuth = math.degrees(math.atan2(n[1], n[0])) + 35.0
        self.rc.update_scene(self.d, self.camc, self.optc)
        scn = self.rc.scene
        line = ""
        pts, wts = [], []
        if plant.sim == "mujoco":
            Ntot = sum(fN for _, _, fN, _ in cons)
            if self.spheres:
                line = f"thumb pad {Ntot:4.2f} N on {sum(1 for c in cons if c[2] > 1e-4)} of {len(self.pads['thumb'])} spheres"
            else:
                line = f"thumb pad {Ntot:4.2f} N, {len(cons)} contact point{'s' if len(cons) != 1 else ''}"
                for pos, nn, fN, g in cons:
                    C.add_sphere(scn, pos, 0.0007, heat(fN / SCALE_POINT_N))
                    C.add_connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, pos, pos + nn * 0.004 * fN, 0.0005, (0.1, 0.1, 0.1, 1))
            for pos, nn, fN, g in cons:
                pts.append(pos)
                wts.append(max(fN, 0.0))
        elif plant.sim == "drake":
            if self.sp["model"] == "hydro":
                area_tot, nf, Ntot = 0.0, 0, 0.0
                for f, cen, nor, area, pres in plant.surfaces():
                    if f != "thumb":
                        continue
                    for c, nv, a_, p_ in zip(cen, nor, area, pres):
                        r_ = math.sqrt(max(a_, 1e-12) / math.pi)
                        if scn.ngeom < scn.maxgeom:
                            gg = scn.geoms[scn.ngeom]
                            Rm = H._rot_z_to(nv)
                            mujoco.mjv_initGeom(gg, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([r_ * 1.15, r_ * 1.15, 0.00006]),
                                                np.asarray(c, float), Rm.reshape(-1), np.asarray(heat(p_ / SCALE_PRESSURE), np.float32))
                            scn.ngeom += 1
                        pts.append(c)
                        wts.append(max(p_, 0.0) * a_)
                        area_tot += a_
                        nf += 1
                Ntot = plant.contacts()["thumb"]["N"]
                line = f"thumb patch {Ntot:4.2f} N, {area_tot * 1e6:4.1f} mm^2, {nf} faces"
            else:
                Ntot, k = 0.0, 0
                for f, pos, nn, fN in plant.point_contacts():
                    if f != "thumb":
                        continue
                    C.add_sphere(scn, pos, 0.0007, heat(fN / SCALE_POINT_N))
                    C.add_connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, pos, pos + nn * 0.004 * fN, 0.0005, (0.1, 0.1, 0.1, 1))
                    pts.append(pos)
                    wts.append(fN)
                    Ntot += fN
                    k += 1
                line = f"thumb pad {Ntot:4.2f} N, {k} contact point{'s' if k != 1 else ''}"
        if pts and sum(wts) > 0:
            cop = np.average(np.array(pts), axis=0, weights=np.array(wts))
            C.add_sphere(scn, cop, 0.00045, (0.02, 0.02, 0.02, 1))
        close = self.rc.render().copy()
        m.geom_rgba[:] = self.rgba0
        close = H.annotate(close, "thumb pad, tool and index see-through", size=13)
        close = H.annotate_bottom(close, line or "no contact", size=12)
        return np.concatenate([wide, close], 1)


def still(out):
    import mujoco
    from PIL import Image
    trial = make_trial(0)
    path, err = postures([GAP_OPEN, 0.0, -SQUEEZE_MAX])
    xml, _, lay = chain_scene("mj:point3", trial)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    P, u, a = pinch_frame()
    s_hat = lay["c"] - trial["d_cg"] * lay["axis"]
    palm = np.r_[s_hat - P, 0.0]
    imgs = []
    for g in (GAP_OPEN, 0.0):
        d.qpos[[m.jnt_qposadr[m.joint(n).id] for f in B.FINGERS for n in B.JOINTS[f]]] = path[g]
        d.qpos[[m.jnt_qposadr[m.joint(n).id] for n in ("palm_x", "palm_y", "palm_z", "palm_yaw")]] = palm
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 360, 480)
        for az, el, dist in ((math.degrees(math.atan2(u[1], u[0])), -12, 0.42), (math.degrees(math.atan2(a[1], a[0])), -20, 0.25)):
            cam = mujoco.MjvCamera()
            mid = 0.5 * (lay["s_nom"] + lay["hole"]) if dist > 0.3 else lay["s_nom"]
            cam.lookat[:] = [mid[0], mid[1], 0.04]
            cam.distance, cam.azimuth, cam.elevation = dist, az, el
            r.update_scene(d, cam)
            imgs.append(r.render().copy())
        print(f"gap {g * 1e3:+.0f} mm: ncon {d.ncon}", [(m.geom(d.contact[i].geom[0]).name, m.geom(d.contact[i].geom[1]).name)
                                                   for i in range(d.ncon)][:6])
    Image.fromarray(np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:], 1)], 0)).save(out)
    print("ik err mm", err * 1e3, "postures", {k: np.round(v, 3).tolist() for k, v in path.items()})


# ------------------------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("still")
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--spec", required=True)
    r.add_argument("--brake", default="closed", choices=("open", "closed"))
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--d", type=float, default=15.0)
    r.add_argument("--clearance", type=float, default=1.0)
    r.add_argument("--film")
    r.add_argument("--rows")
    args = ap.parse_args()
    if args.cmd == "still":
        still(args.out)
        return
    row = run_chain(args.spec, args.seed, args.brake, args.d * 1e-3, args.clearance * 1e-3, film=args.film)
    if args.rows:
        H.append_row(args.rows, row)
    print(json.dumps({k: v for k, v in row.items() if k not in ("trace",)}, default=H._json_default))


if __name__ == "__main__":
    main()
