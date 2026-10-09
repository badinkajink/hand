#!/usr/bin/env python3
"""Grasp robustness and in-hand reorientation over the (layout, object) cells of the hand-object scale study, in
CPU MuJoCo on the bench-calibrated servo plant with the printed TPU fingertip as 1 mm sphere pads.

Scene per cell. The real_v1 base scene with its object replaced (a 100 mm cylinder along palm y, or a sphere; 25 g;
the body z axis along -y is the marked axis the turn stands up), the layout baked in by
morphohand.tools.morphology_xml.create_rigid_hand_and_scene_xmls (the generator every real_v1 design goes through),
the measured plant (scripts/apply_measured_plant.py at kp 4 N m/rad, kv 0, forcerange 1 N m, joint damping 0.08:
the 2026-10-06 servo refit), finger joint ranges cut to the servos' calibrated ranges, 1 ms steps with the
elliptic cone and impratio 100, friction 1.0 on every geom, the palm welded at the cell's dexterous equator grasp
(hand_object_scale_kin.py reach), the object resting on a 3 mm post, and each tip replaced by the TPU block with
2.7 mm fillets as 1 mm-spaced sphere pads (reorient_backends.replace_tips; model 'pt' = the block as one convex
mesh with MuJoCo point contact, the check).

Grasp. The cell's dexterous equator grasp (the feasible palm height and straddle whose least-conditioned finger has
the largest smallest Jacobian singular value); each touching pose moved along the contact normal until the TPU block
mesh just touches the object; the grip command adds the servo-spring offset J^T (4 N n) / kp that presses each pad
with 4 N at rest. The open pose sits 8 mm off the surface.

  hold   close over 0.5 s, settle 0.5 s, drop the post, hold 1.2 s; held = the object within 5 mm of its start and
         at least two fingers each pressing with at least its weight. From the held state, twelve ramps of an
         external force (+-x +-y +-z, 4 N/s to 20 N) and torque (+-x +-y +-z, the same ramp as a couple at the
         object radius) on the object until it moves 3 mm or turns 5 deg: the threshold of each, or the cap.
  turn   the relative contact-velocity controller of hom_turn3.py (Wang, Oh and Pollard, arXiv 2609.25619) at its
         default parameters, about the pinch axis toward 90 deg; held turn = the angle at the end of the hold with
         at least two fingers each pressing with at least the object's weight.

  .venv/bin/python scripts/hand_object_scale_sim.py run --cells coarse --what hold,turn --seeds 0 1 2
Rows: docs/experiments/20261008-hand_object_scale/{hold,turn}.jsonl, one fsynced line per rollout; resumable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import hand_object_scale_kin as K  # noqa: E402
import hom_turn3 as H  # noqa: E402
import reorient_backends as RB  # noqa: E402

OUT_DIR = K.OUT_DIR
GEN = ROOT / "assets/mjcf/experimental/20261008-hand_object_scale"
BASE_HAND = ROOT / "assets/mjcf/real_v1/real_hand.xml"
OBJ = RB.OBJ                     # the object body keeps the screwdriver's name so the shared tooling finds it
PLANT = "kp4_kv0_fr1_fl0_dp0.08"
MU, IMPRATIO = 1.0, 100.0
MASS = 0.025
Z_OBJ = 0.120                    # object centre above the table, m
F_GRIP, OPEN_MM = 4.0, 8.0          # N per pad at rest; mm the open pose stands off
POST_R = 0.003
FINGERS, JOINTS = RB.FINGERS, RB.JOINTS
NAMES = [f"{f}_{j}" for f in FINGERS for j in JOINTS]
F_RATE, F_CAP = 4.0, 20.0        # N/s, N
SLIP_MM, SLIP_DEG = 3.0, 5.0
T_CLOSE, T_SETTLE, T_HOLD = 0.5, 0.5, 1.2


def obj_tag(shape: str, d_mm: float) -> str:
    return f"{shape[:3]}{d_mm:05.1f}"


# ------------------------------------------------------------------------------------------------ scenes

def base_scene(shape: str, d_mm: float) -> Path:
    """The real_v1 base scene (morph joints present) with the object replaced."""
    import build_real_v1_scenes as B
    out = GEN / "base" / f"scene_{obj_tag(shape, d_mm)}.xml"
    if out.exists():
        return out
    r = d_mm / 2000.0
    if shape == "cylinder":
        geom = (f'<geom type="cylinder" size="{r:.6f} 0.05" mass="{MASS}" material="object_mat" '
                f'friction="2.4 0.2 0.02" />')
    else:
        geom = f'<geom type="sphere" size="{r:.6f}" mass="{MASS}" material="object_mat" friction="2.4 0.2 0.02" />'
    saved = B.OBJECTS[OBJ]
    B.OBJECTS[OBJ] = dict(geom=geom, pos=f"0.0 0.0 {r:.6f}", quat="0.70711 0.70711 0 0",
                          key_qpos=[0, 0, r, 0.70711, 0.70711, 0, 0], palm_z=B.PALM_Z)
    try:
        xml = B.build_scene(actuated=False, obj=OBJ)
    finally:
        B.OBJECTS[OBJ] = saved
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    return out


def design_scene(x_sep_mm: float, y_sep_mm: float, shape: str, d_mm: float) -> Path:
    """The layout baked into the object's base scene by the shared generator (morph joints removed)."""
    from morphohand.tools.morphology_xml import MorphologyValues, build_morphology_suffix, \
        create_rigid_hand_and_scene_xmls
    vec = K.morph_vector(x_sep_mm / 1000, y_sep_mm / 1000)
    mv = MorphologyValues(*vec)
    out_dir = GEN / "designs" / obj_tag(shape, d_mm)
    path = out_dir / f"scene_{build_morphology_suffix(mv)}.xml"
    if not path.exists():
        _, path = create_rigid_hand_and_scene_xmls(BASE_HAND, base_scene(shape, d_mm), mv, out_dir)
    return path


def plant_scene(design: Path) -> Path:
    out = design.with_name(design.stem + f"__{PLANT}.xml")
    if not out.exists():
        P = RB.plant_args(PLANT)
        args = ["--kp", f"{P['kp']:g}", "--kv", f"{P['kv']:g}", "--forcerange", f"{P['forcerange']:g}",
                "--frictionloss", f"{P['frictionloss']:g}", "--damping", f"{P['damping']:g}"]
        tmp = out.with_name(f"{out.stem}.{os.getpid()}.xml")
        subprocess.run([sys.executable, str(ROOT / "scripts/apply_measured_plant.py"), "--scene", str(design),
                        "--out", str(tmp)] + args, check=True, capture_output=True)
        os.replace(tmp, out)
    return out


SPREADS = (0.0425, 0.0325, 0.0225)     # straddles tried on the 100 mm cylinder, m
N_H = 3                                # palm heights tried per straddle (spheres: 5)
SCENES = ROOT / "logs/20261008-hand_object_scale/scenes"    # cell scenes are 0.7 MB each: kept out of git


def candidates(cell: dict, shape: str, d_mm: float) -> list[dict]:
    """The grasps a cell is evaluated at: per straddle, N_H palm heights spread evenly over its feasible band
    (5 for a sphere), each with its kinematic variables (hand_object_scale_kin.grasp_scan)."""
    xs, ys = cell["x_sep_mm"] / 1000, cell["y_sep_mm"] / 1000
    lim = K.limits(False)
    sps = SPREADS if shape == "cylinder" else (None,)
    rows = K.grasp_scan(xs, ys, shape, d_mm / 1000, lim, sps)
    out = []
    for row in rows:
        idx = np.flatnonzero(row["ok"])
        if not idx.size:
            continue
        n = N_H if shape == "cylinder" else 5
        pick = sorted({int(idx[int(round(i))]) for i in np.linspace(0, idx.size - 1, min(n, idx.size))})
        for k in pick:
            out.append({
                "h": float(row["h"][k]), "spread": float(row["spread"] or 0.0),
                "q_deg": {f: [round(float(v), 2) for v in np.degrees(row["q"][f][k])] for f in FINGERS},
                "min_margin_deg": round(float(np.degrees(row["mmin"][k])), 2),
                "ext_mm": {f: round(float(row["ext"][f][k]) * 1000, 2) for f in FINGERS},
                "sigma_min_mm": {f: round(float(row["smin"][f][k]) * 1000, 2) for f in FINGERS},
                "cond": {f: round(float(row["cond"][f][k]), 2) for f in FINGERS},
                "contact_angle_deg": {f: round(float(row["angle"][f][k]), 1) for f in FINGERS},
                "ff_clear_mm": round(float(row["ff_clear"][k]) * 1000, 2),
                "plate_clear_mm": round(float(row["plate"][k]) * 1000, 2)})
    return out


def grasp_geometry(cell: dict, shape: str, d_mm: float, h: float, sp: float) -> dict:
    """Pad-centre targets (touching), contact normals and the palm pose of one candidate grasp."""
    xs, ys = cell["x_sep_mm"] / 1000, cell["y_sep_mm"] / 1000
    rel, c = K.contact_targets(shape, xs, ys, d_mm / 2000 - K.GAP, sp)         # gap 0
    centre = np.array([c[0], c[1], -h])
    n_out = {}
    for f in FINGERS:
        v = rel[f].copy()
        if shape == "cylinder":
            v[1] = 0.0
        n_out[f] = v / np.linalg.norm(v)
    palm = np.array([-centre[0], -centre[1], Z_OBJ - centre[2]])          # object centre at (0, 0, Z_OBJ)
    return {"centre_palm": centre, "rel": rel, "n_out": n_out, "mounts": K.mounts(xs, ys),
            "lim": K.limits(False), "palm": palm}


def _ik_near(f, T, lim, q_prev=None):
    """Analytic IK inside the limits; the branch nearest q_prev when both fit; else the first branch clipped."""
    qq, ok = K.ik(f, T[None])
    qq, ok = qq[0], bool(ok[0])
    inside = [k for k in range(2) if ok and np.all(qq[k] >= lim[:, 0]) and np.all(qq[k] <= lim[:, 1])]
    if not inside:
        return np.clip(qq[0], lim[:, 0], lim[:, 1]), False
    if q_prev is not None and len(inside) == 2:
        return qq[min(inside, key=lambda k: np.linalg.norm(qq[k] - q_prev))], True
    return qq[inside[0]], True


def set_palm(m, palm) -> None:
    m.body_pos[m.body("palm_pose").id] = palm


def grip_poses(m_pt, gg: dict, kp: float, f_grip: float = F_GRIP, iters: int = 6) -> dict:
    """Touching, open and grip commands per finger. The touching pose is solved for the sphere pad and then moved
    along the contact normal until the TPU block itself (one convex mesh, mj_geomDistance) just touches the object;
    the open pose sits OPEN_MM further out; the grip command adds J^T (f_grip n_in) / kp to the touching pose, the
    servo-spring offset that presses the pad with f_grip newtons at rest (no Jacobian inverse, so a straight finger
    is no singularity). `m_pt` is the cell's 'pt' model."""
    import mujoco
    m = m_pt
    set_palm(m, gg["palm"])
    d = mujoco.MjData(m)
    tb = m.body(OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    og = [g for g in range(m.ngeom) if m.geom_bodyid[g] == tb][0]
    d.qpos[qa:qa + 7] = [0.0, 0.0, Z_OBJ, 0.70711, 0.70711, 0.0, 0.0]
    out = {"q_touch": {}, "q_open": {}, "q_grip": {}, "touch_gap_mm": {}, "touch_ok": {}}
    for f in FINGERS:
        lim = gg["lim"][f]
        base = gg["centre_palm"] + gg["rel"][f] - gg["mounts"][f]
        n = gg["n_out"][f]
        delta, q, dist, ok = 0.0, None, 0.0, True
        for _ in range(iters):
            q, ok = _ik_near(f, base + delta * n, lim, q)
            for k, j in enumerate(JOINTS):
                d.qpos[m.jnt_qposadr[m.joint(f"{f}_{j}").id]] = q[k]
            mujoco.mj_forward(m, d)
            dist = mujoco.mj_geomDistance(m, d, m.geom(f"{f}_tipgeom").id, og, 0.05, None)
            if abs(dist) < 5e-5:
                break
            delta -= dist
        out["q_touch"][f], out["touch_gap_mm"][f], out["touch_ok"][f] = q, round(1e3 * dist, 3), ok
        out["q_open"][f], _ = _ik_near(f, base + (delta + OPEN_MM / 1000) * n, lim, q)
        tau = K.jac_pad(f, q).T @ (-n * f_grip)
        out["q_grip"][f] = np.clip(q + tau / kp, lim[:, 0], lim[:, 1])
    return out


def cell_scenes(cell: dict, shape: str, d_mm: float, models=("pads", "pt")) -> dict:
    """The cell's welded-palm scenes, one per tip model; the palm pose is set per candidate on the model."""
    xs, ys = cell["x_sep_mm"], cell["y_sep_mm"]
    base = plant_scene(design_scene(xs, ys, shape, d_mm))
    lim = K.limits(False)
    r = d_mm / 2000
    tool7 = [0.0, 0.0, Z_OBJ, 0.70711, 0.70711, 0.0, 0.0]
    out = {}
    for mdl in models:
        out_dir = SCENES / obj_tag(shape, d_mm)
        path = out_dir / f"{cell['tag']}_{mdl}.xml"
        meta_path = path.with_suffix(".json")
        if path.exists() and meta_path.exists():
            out[mdl] = (path, json.loads(meta_path.read_text()))
            continue
        root = ET.parse(base).getroot()
        for kf in root.findall("keyframe"):
            root.remove(kf)
        opt = root.find("option")
        for k, v in RB.NUMERICS["bed"].items():
            opt.set(k, v)
        opt.set("impratio", f"{IMPRATIO:g}")
        palm = root.find(".//body[@name='palm_pose']")
        for jn in list(palm.findall("joint")):
            palm.remove(jn)
        palm.attrib.pop("gravcomp", None)
        palm.set("pos", f"0 0 {Z_OBJ + 0.05:.6f}")
        palm.set("quat", "1 0 0 0")
        act = root.find("actuator")
        for a in list(act):
            if a.get("joint", "").startswith("palm_"):
                act.remove(a)
        # finger joint and command ranges: the servos' calibrated ranges (the kinematics' limits)
        for f in FINGERS:
            for k, j in enumerate(JOINTS):
                lo, hi = lim[f][k]
                root.find(f".//joint[@name='{f}_{j}']").set("range", f"{lo:.6f} {hi:.6f}")
                an = root.find(f".//actuator/position[@joint='{f}_{j}']")
                if an is not None:
                    an.set("ctrlrange", f"{lo:.6f} {hi:.6f}")
        for g in root.iter("geom"):
            g.set("friction", f"{MU:g} 0.005 0.0001")
        for dg in root.find("default").iter("geom"):
            dg.set("friction", f"{MU:g} 0.005 0.0001")
        tb = root.find(f".//body[@name='{OBJ}']")
        tb.set("pos", " ".join(f"{v:.7f}" for v in tool7[:3]))
        tb.set("quat", " ".join(f"{v:.8f}" for v in tool7[3:]))
        hh = (Z_OBJ - r) / 2.0
        post = ET.SubElement(root.find("worldbody"), "body", name="post", pos=f"0 0 {hh:.6f}")
        ET.SubElement(post, "geom", name="post", type="cylinder", size=f"{min(POST_R, 0.4 * r):.6f} {hh:.6f}",
                      rgba="0.4 0.4 0.45 1", friction=f"{MU:g} 0.005 0.0001")
        _, pads_meta = RB.replace_tips(root, "tpu2.7", mdl, MU, out_dir, tool=OBJ, s=RB.PAD_S)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        ET.ElementTree(root).write(tmp)
        os.replace(tmp, path)
        meta = {"hand": cell["tag"], "tag": cell["tag"], "shape": shape, "d_mm": d_mm, "model": mdl, "plant": PLANT,
                "mu": MU, "impratio": IMPRATIO, "tool7": tool7, "mass_kg": MASS, "f_grip_N": F_GRIP,
                "design_scene": str(design_scene(xs, ys, shape, d_mm).relative_to(ROOT)),
                **{k: v for k, v in pads_meta.items() if k != "meshes"}}
        meta_path.write_text(json.dumps(meta))
        out[mdl] = (path, meta)
    return out


def candidate_meta(meta: dict, gg: dict, gp: dict) -> dict:
    """The cell's metadata with one candidate's palm pose and finger commands (hom_turn3.MjPlant reads q0)."""
    deg = lambda q: {f: [float(v) for v in q[f]] for f in FINGERS}  # noqa: E731
    return {**meta, "palm_pos": gg["palm"].tolist(),
            "q0": {f"{f}_{j}": float(gp["q_open"][f][k]) for f in FINGERS for k, j in enumerate(JOINTS)},
            "grip": deg(gp["q_grip"]), "touch": deg(gp["q_touch"])}


# ------------------------------------------------------------------------------------------------ hold test

def _contacts(m, d, tool_geoms, finger_of):
    import mujoco
    F = {f: 0.0 for f in FINGERS}
    n = {f: 0 for f in FINGERS}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        if (g1 in tool_geoms) == (g2 in tool_geoms):
            continue
        f = finger_of.get(g2 if g1 in tool_geoms else g1)
        if f is None:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        if f6[0] > 1e-6:
            F[f] += float(f6[0])
            n[f] += 1
    return F, n


def _angle(Ra, Rb) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def hold_rollout(path: Path, meta: dict, seed: int, disturb: bool = True) -> dict:
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(path))
    set_palm(m, np.asarray(meta["palm_pos"]))
    d = mujoco.MjData(m)
    tb = m.body(OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    act = np.array([m.actuator(f"a_{n}").id for n in NAMES])
    qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in NAMES])
    tool_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == tb}
    finger_of = {}
    for i in range(m.ngeom):
        bn = m.body(m.geom_bodyid[i]).name
        for f in FINGERS:
            if bn.startswith(f + "_"):
                finger_of[i] = f
    post = m.geom("post").id
    q_open = np.array([meta["q0"][n] for n in NAMES])
    q_grip = np.array([meta["grip"][f][k] for f in FINGERS for k in range(3)])
    tool7 = np.asarray(meta["tool7"], float)
    dx, dy, dyaw = RB.jitter(seed)
    d.qpos[qa:qa + 3] = tool7[:3] + [dx, dy, 0.0]
    d.qpos[qa + 3:qa + 7] = RB._yaw_quat(tool7[3:], dyaw)
    d.qpos[qadr] = q_open
    d.ctrl[act] = q_open
    mujoco.mj_forward(m, d)
    w0 = time.perf_counter()
    weight = meta["mass_kg"] * 9.81
    while d.time < T_CLOSE - 1e-9:
        u = min(1.0, d.time / T_CLOSE)
        d.ctrl[act] = q_open + (q_grip - q_open) * (3 * u * u - 2 * u ** 3)
        mujoco.mj_step(m, d)
    d.ctrl[act] = q_grip
    while d.time < T_CLOSE + T_SETTLE - 1e-9:
        mujoco.mj_step(m, d)
    F_grip, n_grip = _contacts(m, d, tool_geoms, finger_of)
    p0 = d.xpos[tb].copy()
    m.geom_contype[post] = 0
    m.geom_conaffinity[post] = 0
    while d.time < T_CLOSE + T_SETTLE + T_HOLD - 1e-9:
        mujoco.mj_step(m, d)
        if d.xpos[tb][2] < p0[2] - 0.03:
            break
    F_hold, n_hold = _contacts(m, d, tool_geoms, finger_of)
    p1, R1 = d.xpos[tb].copy(), d.xmat[tb].reshape(3, 3).copy()
    loaded = sum(1 for f in FINGERS if F_hold[f] >= weight)
    held = bool(p1[2] > p0[2] - 0.005 and loaded >= 2)
    row = {"F_grip": {f: round(v, 3) for f, v in F_grip.items()}, "n_grip": n_grip,
           "F_hold": {f: round(v, 3) for f, v in F_hold.items()}, "n_hold": n_hold,
           "drop_mm": round(1e3 * float(p0[2] - p1[2]), 2), "slip_mm": round(1e3 * float(np.linalg.norm(p1 - p0)), 2),
           "fingers_loaded": loaded, "held": held}
    if held and disturb:
        state = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        r_eff = max(meta["d_mm"] / 2000.0, 0.010)
        res = {}
        for kind in ("F", "T"):
            for ax, an in enumerate("xyz"):
                for sgn in (1, -1):
                    mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
                    mujoco.mj_forward(m, d)
                    t0 = d.time
                    cap = F_CAP if kind == "F" else F_CAP * r_eff
                    rate = F_RATE if kind == "F" else F_RATE * r_eff
                    val = None
                    while True:
                        mag = rate * (d.time - t0)
                        if mag > cap:
                            break
                        d.xfrc_applied[tb, :] = 0.0
                        d.xfrc_applied[tb, (0 if kind == "F" else 3) + ax] = sgn * mag
                        mujoco.mj_step(m, d)
                        if (np.linalg.norm(d.xpos[tb] - p1) > SLIP_MM / 1000
                                or _angle(R1, d.xmat[tb].reshape(3, 3)) > SLIP_DEG):
                            val = mag
                            break
                    d.xfrc_applied[tb, :] = 0.0
                    res[f"{kind}{'+' if sgn > 0 else '-'}{an}"] = None if val is None else round(val, 4)
        row["disturb"] = res
        row["r_eff_m"] = r_eff
    row["wall_s"] = round(time.perf_counter() - w0, 2)
    return row


# ------------------------------------------------------------------------------------------------ HOM turn

class Mirror(H.Mirror):
    """hom_turn3.Mirror for a cylinder or a sphere: the contact normal of a sphere points from its centre."""

    def __init__(self, path):
        import mujoco
        self.mj = mujoco
        self.m = m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(m)
        self.tb = m.body(OBJ).id
        self.qa = m.jnt_qposadr[m.body_jntadr[self.tb]]
        self.va = m.jnt_dofadr[m.body_jntadr[self.tb]]
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in NAMES])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in NAMES])
        self.lo = np.array([m.jnt_range[m.joint(n).id][0] for n in NAMES])
        self.hi = np.array([m.jnt_range[m.joint(n).id][1] for n in NAMES])
        g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.tb][0]
        self.cyl = g
        self.sphere = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        self.r_cyl = float(m.geom_size[g][0])
        self.hl = 0.0 if self.sphere else float(m.geom_size[g][1])
        self.tipg = {f: m.geom(f"{f}_tipgeom").id for f in FINGERS}
        self.tipb = {f: m.body(f"{f}_tip").id for f in FINGERS}
        self.flex = {f: m.joint(f"{f}_pip").id for f in FINGERS}
        self.jacp, self.jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def gcf(self, f):
        if not self.sphere:
            return super().gcf(f)
        m, d = self.m, self.d
        ft = np.zeros(6)
        dist = self.mj.mj_geomDistance(m, d, self.tipg[f], self.cyl, 0.05, ft)
        p_f, p_o = ft[:3].copy(), ft[3:].copy()
        rad = p_o - d.geom_xpos[self.cyl]
        n = rad / max(np.linalg.norm(rad), 1e-9)
        y = d.xaxis[self.flex[f]] - (d.xaxis[self.flex[f]] @ n) * n
        if np.linalg.norm(y) < 1e-6:
            y = np.cross(n, [0.0, 0.0, 1.0])
        y /= np.linalg.norm(y)
        z = np.cross(n, y)
        return {"dist": float(dist), "p_f": p_f, "p_o": p_o, "n": n, "E": np.column_stack([n, y, z])}


def turn_rollout(plant, mirror, meta, seed, prm) -> dict:
    """hom_turn3.rollout with the cell's grip in place of a deploy plan's, and the load test of this study."""
    grip = np.array([meta["grip"][f][k] for f in FINGERS for k in range(3)])
    plant.reset(seed)
    plant.set_targets(grip)
    plant.advance(prm["t_grip"])
    s0 = plant.state()
    ctl = H.Turn3(mirror, s0, prm, grip)
    goal = math.radians(prm["goal_deg"]) - ctl.tilt0
    rate = math.radians(prm["rate_deg"])
    t_sq_end = prm["t_grip"] + prm["t_squeeze"]
    t_end = t_sq_end + max(0.0, goal) / rate + prm["t_hold"]
    t = prm["t_grip"]
    th_max = th3_max = th2_max = -10.0
    w0 = time.perf_counter()
    z0 = s0["p"][2]
    frozen, n_lim, first_loss, lim_joint = None, 0, None, None
    th_cmd = 0.0
    weight = meta["mass_kg"] * 9.81
    trace = []
    while t < t_end - 1e-9:
        s = plant.state()
        nf = sum(1 for f in FINGERS if s["n"][f] > 0 and s["F"][f] >= weight)
        th_now = ctl.theta(s["R"])
        if t >= t_sq_end and nf < 3 and first_loss is None:
            first_loss = round(math.degrees(th_now), 2)
        if nf >= 2 and s["p"][2] > z0 - 0.020:
            th2_max = max(th2_max, th_now)
            if nf == 3:
                th3_max = max(th3_max, th_now)
        if t < t_sq_end:
            th_ref, dth = 0.0, 0.0
        elif frozen is not None:
            th_ref, dth = frozen, 0.0
        else:
            th_cmd = min(goal, th_cmd + rate * prm["dt_c"])
            th_ref, dth = th_cmd, (rate if th_cmd < goal else 0.0)
        q_cmd, th, info = ctl.step(s, th_ref, dth, prm["f_des"])
        if t >= t_sq_end and frozen is None and prm["governor"]:
            weak = min(s["F"].values()) < prm["f_frac"] * prm["f_des"] or th_ref - th > math.radians(prm["err_deg"])
            n_lim = n_lim + 1 if weak else 0
            if n_lim >= 3:
                frozen, lim_joint = th, (ctl.limited[0] if ctl.limited else None)
                t_end = min(t_end, t + prm["t_hold"])
        th_max = max(th_max, th)
        if not trace or t - trace[-1]["t"] >= 0.05 - 1e-9:
            trace.append({"t": round(t, 3), "th": round(math.degrees(th), 2), "F": {f: round(s["F"][f], 2) for f in FINGERS}})
        plant.set_targets(q_cmd)
        t += prm["dt_c"]
        plant.advance(t)
        if s["p"][2] < z0 - 0.03 or not np.all(np.isfinite(s["p"])):
            break
    se = plant.state()
    loaded = sum(1 for f in FINGERS if se["n"][f] > 0 and se["F"][f] >= weight)
    dropped = bool(se["p"][2] < z0 - 0.020)
    th_end = ctl.theta(se["R"])
    held = (not dropped) and loaded >= 2
    return {"turn_end_deg": round(math.degrees(th_end), 2), "held": held, "held_turn_deg":
            round(math.degrees(th_end), 2) if held else 0.0, "turn_max_deg": round(math.degrees(th_max), 2),
            "turn_held2_max_deg": round(math.degrees(th2_max), 2), "turn_held3_max_deg": round(math.degrees(th3_max), 2),
            "first_loss_deg": first_loss, "frozen_deg": None if frozen is None else round(math.degrees(frozen), 2),
            "limit_joint": lim_joint, "fingers_loaded_end": loaded, "dropped": dropped,
            "F_end": {f: round(se["F"][f], 3) for f in FINGERS}, "z_drop_mm": round(1e3 * float(z0 - se["p"][2]), 2),
            "sat_frac": [round(float(x) / max(1, ctl.ticks), 3) for x in ctl.sat],
            "wall_s": round(time.perf_counter() - w0, 2), "trace": trace}


# ------------------------------------------------------------------------------------------------ cells

COARSE_T = (-1.0, -0.5, 0.0, 0.5, 1.0)
COARSE_D = {"cylinder": (6, 16, 25, 40, 63), "sphere": (16, 40, 63, 100, 160)}


def cell_list(which: str) -> list[tuple[dict, str, float]]:
    R_ = json.loads((OUT_DIR / "reach.json").read_text())
    cells = {(c["tag"], c["shape"], c["d_mm"]): c for c in R_["cells"]}
    out = []
    if which == "coarse":
        lays = [l for l in R_["families"]["diag"] if l["t"] in COARSE_T]
        objs = [(s, float(dd)) for s, ds in COARSE_D.items() for dd in ds]
    elif which == "diag":
        lays = R_["families"]["diag"]
        objs = [(s, float(dd)) for s, ds in K.OBJECT_SET.items() for dd in ds]
    elif which in ("xsep", "ysep"):
        lays = R_["families"][which]
        objs = [(s, float(dd)) for s, ds in K.OBJECT_SET.items() for dd in ds]
    else:
        raise ValueError(which)
    for lay in lays:
        for shape, dd in objs:
            c = cells.get((lay["tag"], shape, dd))
            if c is not None:
                out.append((dict(c, t=lay.get("t")), shape, dd))
    return out


def _done(path: Path) -> set:
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") in ("ok", "infeasible"):
                out.add((r["tag"], r["shape"], r["d_mm"], r["model"], r["cand"], r["seed"]))
    return out


def _append(path: Path, row: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def cmd_run(a) -> int:
    """Every candidate grasp of every cell: the hold test (seed 0, with the disturbance ramps) and the HOM turn
    (each seed). Rows carry the candidate's kinematic variables; resumable per (cell, candidate, seed)."""
    import mujoco
    whats = a.what.split(",")
    outs = {w: OUT_DIR / f"{w}.jsonl" for w in whats}
    done = {w: _done(p) for w, p in outs.items()}
    todo = cell_list(a.cells)
    k0, n0 = (int(v) for v in a.shard.split("/"))
    todo = [c for i, c in enumerate(todo) if i % n0 == k0]
    prm = dict(H.DEFAULTS)
    for kv in a.set:
        key, v = kv.split("=")
        prm[key] = float(v)
    kp = RB.plant_args(PLANT)["kp"]
    print(f"{len(todo)} cells in shard {a.shard}", flush=True)
    for cell, shape, dd in todo:
        base = {"tag": cell["tag"], "t": cell.get("t"), "x_sep_mm": cell["x_sep_mm"], "y_sep_mm": cell["y_sep_mm"],
                "palm_radius_mm": cell["palm_radius_mm"], "shape": shape, "d_mm": dd, "model": a.model}
        if not cell["feasible"]:
            for w in whats:
                if (cell["tag"], shape, dd, a.model, -1, 0) not in done[w]:
                    _append(outs[w], {**base, "cand": -1, "seed": 0, "status": "infeasible", "fail": cell.get("fail")})
            continue
        try:
            cands = candidates(cell, shape, dd)
            sc = None
            for ci, cand in enumerate(cands):
                need = {"hold": [0] if (cell["tag"], shape, dd, a.model, ci, 0) not in done.get("hold", {(0,)}) else [],
                        "turn": [s_ for s_ in a.seeds if (cell["tag"], shape, dd, a.model, ci, s_) not in done.get("turn", set())]}
                need = {w: v for w, v in need.items() if w in whats and v}
                if not need:
                    continue
                if sc is None:
                    sc = cell_scenes(cell, shape, dd, models=tuple(dict.fromkeys((a.model, "pt"))))
                    m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
                gg = grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
                gp = grip_poses(m_pt, gg, kp)
                path, meta0 = sc[a.model]
                meta = candidate_meta(meta0, gg, gp)
                geo = {"cand": ci, "n_cand": len(cands), "h_mm": round(cand["h"] * 1000, 2),
                       "spread_mm": round(cand["spread"] * 1000, 2), "kin": cand,
                       "touch_gap_mm": gp["touch_gap_mm"], "touch_ok": gp["touch_ok"]}
                if "hold" in need:
                    row = {**base, **geo, "seed": 0, "when": time.strftime("%Y-%m-%d %H:%M")}
                    try:
                        row.update(hold_rollout(path, meta, 0, disturb=True), status="ok")
                    except Exception as e:
                        row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1200:])
                    _append(outs["hold"], row)
                    print(f"hold {cell['tag']} {shape} {dd:5.1f} c{ci} h{geo['h_mm']} sp{geo['spread_mm']}: " + (
                        f"held {row['held']} F {row['F_hold']} dist {row.get('disturb')} {row['wall_s']} s"
                        if row["status"] == "ok" else row["error"]), flush=True)
                if "turn" in need:
                    plant = H.MjPlant(path, meta)
                    set_palm(plant.m, gg["palm"])
                    mirror = Mirror(Path(sc["pt"][0]))
                    set_palm(mirror.m, gg["palm"])
                    for s_ in need["turn"]:
                        row = {**base, **geo, "seed": s_, "when": time.strftime("%Y-%m-%d %H:%M"),
                               **{f"prm_{k2}": v for k2, v in prm.items()}}
                        try:
                            res = turn_rollout(plant, mirror, meta, s_, prm)
                            if not a.traces:
                                res.pop("trace", None)
                            row.update(res, status="ok")
                        except Exception as e:
                            row.update(status="error", error=f"{type(e).__name__}: {e}",
                                       tb=traceback.format_exc()[-1200:])
                        _append(outs["turn"], row)
                        print(f"turn {cell['tag']} {shape} {dd:5.1f} c{ci} s{s_}: " + (
                            f"held {row['held']} turn {row['turn_end_deg']} frozen {row['frozen_deg']} "
                            f"{row['limit_joint']} F {row['F_end']} {row['wall_s']} s"
                            if row["status"] == "ok" else row["error"]), flush=True)
        except Exception as e:
            for w in whats:
                _append(outs[w], {**base, "cand": -2, "seed": -1, "status": "error",
                                  "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]})
            print(f"ERROR {cell['tag']} {shape} {dd}: {e}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cells", default="coarse", help="coarse | diag | xsep | ysep")
    r.add_argument("--what", default="hold,turn")
    r.add_argument("--model", default="pads", help="pads (1 mm sphere pads) | pt (the TPU block, point contact)")
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="HOM turn seeds (object jitter)")
    r.add_argument("--set", nargs="*", default=[], help="HOM controller parameters k=v (hom_turn3.DEFAULTS)")
    r.add_argument("--shard", default="0/1")
    r.add_argument("--traces", action="store_true")
    a = ap.parse_args()
    return {"run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
