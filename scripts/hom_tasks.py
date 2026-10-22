#!/usr/bin/env python3
"""Four manipulation tasks under the HOM controller (src/morphohand/control/hom.py) on the CAD-model SR2 hand, across
object sizes, in CPU MuJoCo on the bench servo plant (kp 4 N m/rad, 1 N m, damping 0.08) with the printed TPU block as
1 mm sphere pads (builder v2), 1 ms physics, 100 Hz control.

Tasks. Each maps one sub-assembly of the HOM paper (arXiv 2609.25619v2) onto the three-finger SR2 hand, whose fingers
hang from a palm plate above the object.
  pinpush  Cylinder (70 mm long) along palm x. Index and middle pinch it across its middle (palm y); the pinch is a
           prismatic joint (1.5 N per pad: enough torsional friction to hold the gravity torque as the centre of mass
           leaves the pinch axis, little enough for the thumb to slide it). The thumb is a pusher on the -x end cap:
           its reference object twist is a translation along the cylinder axis at 10 mm/s for 1.0 s (Eq. 3 with a
           translation in place of the rotation s), while the pinch fingers hold their reference (V_PB0 = 0, as the
           paper's pinching fingers in its Exp. C). Then the pinch locks at 3 N. Score: axial displacement against the
           commanded 10 mm, axial-velocity RMSE, axis tilt.
  press    The pinpush grasp with a wall 4 mm beyond the +x end cap. The thumb pushes the cylinder to the wall
           (8 mm/s), then holds its reference and commands its contact force through the servo spring (the paper's
           impedance) in steps of 0.5, 1.0, 2.0 and 0.5 N, 1 s each, with the pinch relaxed to 0.6 N: the pusher
           presses the tool on an environmental contact. Score: wall force against the command, per step.
  roll     Sphere at the circumcentre of the three mounts, equator tripod (the scale study's grasp). All three fingers
           follow the object's reference body twist with free rolling (w_roll = 0), as the paper's Exp. B: steps of
           +-40 deg/s about the two horizontal axes, 0.5 s each, with 2.5 N per pad (at 1.5 N the 40 mm sphere left the
           grasp on the first y-axis step). Score: angular-velocity RMSE / step (the paper's
           metric), achieved angle per step, drift of the centre.
  twist    The roll grasp; the same steps about the vertical axis, then a ramp at 30 deg/s to the largest held twist.
Every task starts with the paper's pick: fingers 8 mm off the object approach at v_sep = 20 mm/s until the anti-windup
stops them 0.5 mm away, the commanded forces turn on, and the support post is removed.

    .venv/bin/python scripts/hom_tasks.py design                                  # grasp per task and size
    .venv/bin/python scripts/hom_tasks.py run --tasks pinpush press --d 10 16 25 40 --seeds 0 1 2
    .venv/bin/python scripts/hom_tasks.py run --tasks roll --d 40 --seeds 0 --film
Rows: docs/experiments/20261021-hom_framework/rows/<task>.jsonl, one fsynced line per rollout, resumable.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import fingertip_geometry as G  # noqa: E402
import hand_object_scale_kin as K  # noqa: E402
import hand_object_scale_sim as S  # noqa: E402
from morphohand.control import hom as H  # noqa: E402
from morphohand.control.surfaces import Cylinder, Posed, RoundedBox, Sphere, unit  # noqa: E402

OUT = ROOT / "docs/experiments/20261021-hom_framework"
ROWS = OUT / "rows"
MEDIA = OUT / "media"
SCN = ROOT / "logs/20261021-hom_tasks/scenes"
FINGERS, JOINTS = S.FINGERS, S.JOINTS
TASKS = ("pinpush", "press", "roll", "twist")
SIZES = {"pinpush": (10.0, 16.0, 20.0, 25.0), "press": (10.0, 16.0, 20.0, 25.0),
         "roll": (30.0, 40.0, 50.0, 63.0), "twist": (30.0, 40.0, 50.0, 63.0)}
SHAPE = {"pinpush": "cylinder", "press": "cylinder", "roll": "sphere", "twist": "sphere"}
CYL_LEN = 0.070                  # m: shorter than the scale study's 100 mm so the thumb reaches the end cap
KP = 4.0
DT_C = 0.01
FILLET = 0.0027
AXIS_X = (0.70710678, 0.0, 0.70710678, 0.0)        # body z (the cylinder axis) along world x
SPHERE_Q = (0.70710678, 0.70710678, 0.0, 0.0)       # the scale study's object orientation
WALL_GAP = 0.004


# ------------------------------------------------------------------------------------------------ grasp design

def _reach():
    return json.loads((S.OUT_DIR / "reach.json").read_text())


def _pinch_x_targets(r, L, x_p, h):
    """Pad-centre targets (palm frame), object-outward normals: index and middle across y at x_p, thumb on the -x
    end cap. The object centre is (x_p, 0, -h)."""
    rr = r + K.R + K.GAP
    tg = {"index": np.array([x_p, rr, -h]), "middle": np.array([x_p, -rr, -h]),
          "thumb": np.array([x_p - L / 2 - K.R - K.GAP, 0.0, -h])}
    n_out = {"index": np.array([0.0, 1.0, 0.0]), "middle": np.array([0.0, -1.0, 0.0]),
             "thumb": np.array([-1.0, 0.0, 0.0])}
    return tg, n_out


def design_pinch_x(d_mm: float, L: float = CYL_LEN):
    """Best pinpush/press grasp for a cylinder of diameter d: over the tripod layouts of the scale study, the pinch
    position x_p and the palm height h, the reachable grasp (servo limits with 3 deg margin, the object 15 mm below the
    plate) with the largest smallest pad-Jacobian singular value. The pad centre is the scale study's: a sphere of the
    phalange radius about the distal link's end; open_pose then moves each finger until the TPU block touches."""
    r = d_mm / 2000.0
    lim = K.limits(False)
    best = None
    ranked = []
    for lay in _reach()["families"]["diag"]:
        xs, ys = lay["x_sep"] / 1000, lay["y_sep"] / 1000
        mt = K.mounts(xs, ys)
        for x_p in np.arange(-0.03, 0.0601, 0.0025):
            for h in np.arange(r + 0.015, 0.1101, 0.0025):
                tg, n_out = _pinch_x_targets(r, L, x_p, h)
                rec = {}
                for f in FINGERS:
                    q, ok = K.ik(f, (tg[f] - mt[f])[None])
                    qq, good, mmin, _ = K.pick_branch(q[0], np.asarray(ok[0]), lim[f], math.radians(3.0))
                    if not good:
                        break
                    palm = K.fk(f, qq)["palmar"]            # recorded only: these grasps touch with the block's end
                    _, smin = K.jac_cond(f, qq)
                    rec[f] = (qq, float(mmin), float(smin), float(palm @ (-n_out[f])))
                if len(rec) < 3:
                    continue
                score = min(v[2] for v in rec.values())
                ranked.append({"score": score, "tag": lay["tag"], "t": lay["t"], "x_sep_mm": lay["x_sep"],
                               "y_sep_mm": lay["y_sep"], "x_p": float(x_p), "h": float(h), "L": L, "d_mm": d_mm,
                               "q_deg": {f: np.degrees(v[0]).round(2).tolist() for f, v in rec.items()},
                               "margin_deg": {f: round(math.degrees(v[1]), 2) for f, v in rec.items()},
                               "sigma_min_mm": {f: round(v[2] * 1000, 2) for f, v in rec.items()},
                               "palmar_cos": {f: round(v[3], 3) for f, v in rec.items()}})
    ranked.sort(key=lambda c: -c["score"])
    return ranked


def link_clearance(task, des, d_mm):
    """Smallest distance (m) from any finger geom other than the tip's to the object at the touch and the open pose,
    on the TPU-block mesh model (pt) of the design's layout; and the finger joints at both poses."""
    import mujoco
    tmp = dict(json.loads((OUT / "design.json").read_text())) if (OUT / "design.json").exists() else {}
    key = f"{SHAPE[task]}{d_mm:g}"
    tmp[key] = des
    _write_design(tmp)
    _p_pads, p_pt, setup = build(task, d_mm, models=("pt",))
    m = mujoco.MjModel.from_xml_path(str(p_pt))
    q_open, q_touch = open_pose(setup, m)
    S.set_palm(m, setup["palm"])
    dd = mujoco.MjData(m)
    tb = m.body(S.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    og = int(np.flatnonzero(m.geom_bodyid == tb)[0])
    out = {}
    for name, q0 in (("touch", q_touch), ("open", q_open)):
        dd.qpos[qa:qa + 7] = [0.0, 0.0, S.Z_OBJ, *setup["quat"]]
        for f in FINGERS:
            for k, j in enumerate(JOINTS):
                dd.qpos[m.jnt_qposadr[m.joint(f"{f}_{j}").id]] = q0[f][k]
        mujoco.mj_forward(m, dd)
        dmin = 1.0
        fg = {f: [] for f in FINGERS}
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            fing = [f for f in FINGERS if bn.startswith(f + "_")]
            if not fing or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            fg[fing[0]].append(g)
            if not bn.endswith("_tip"):
                dmin = min(dmin, mujoco.mj_geomDistance(m, dd, g, og, 0.05, None))
        out[name] = dmin
        ff = 1.0                                    # finger against finger (the pinch fingers meet over thin objects)
        for a, b in (("index", "middle"), ("thumb", "index"), ("thumb", "middle")):
            for g1 in fg[a]:
                for g2 in fg[b]:
                    ff = min(ff, mujoco.mj_geomDistance(m, dd, g1, g2, 0.02, None))
        out[name + "_ff"] = ff
    return out, q_open, q_touch


def _write_design(dct):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"design.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(dct, indent=1))
    os.replace(tmp, OUT / "design.json")


def design_sphere(d_mm: float):
    """Best roll/twist grasp for a sphere: the scale study's equator tripod (hand_object_scale_sim.candidates) on the
    layout and palm height whose least-conditioned finger has the largest smallest singular value."""
    R_ = _reach()
    cells = {(c["tag"], c["shape"], c["d_mm"]): c for c in R_["cells"]}
    best = None
    for lay in R_["families"]["diag"]:
        c = cells.get((lay["tag"], "sphere", float(d_mm)))
        if c is None:
            continue
        for k, cand in enumerate(S.candidates(dict(c), "sphere", float(d_mm))):
            score = min(cand["sigma_min_mm"].values())
            if cand["min_margin_deg"] < 3.0:
                continue
            if best is None or score > best["score"]:
                best = {"score": score, "tag": lay["tag"], "t": lay["t"], "x_sep_mm": lay["x_sep"],
                        "y_sep_mm": lay["y_sep"], "cand": k, "h": cand["h"], "spread": cand["spread"], "d_mm": d_mm,
                        "q_deg": cand["q_deg"], "min_margin_deg": cand["min_margin_deg"],
                        "sigma_min_mm": cand["sigma_min_mm"]}
    return best


def cmd_design(a) -> int:
    out = {}
    for task in a.tasks:
        for d in (a.d or SIZES[task]):
            key = f"{SHAPE[task]}{d:g}"
            if key in out:
                continue
            t0 = time.time()
            if SHAPE[task] == "cylinder":
                out[key] = None
                passing = []
                ranked = design_pinch_x(d)
                for des in ranked[:a.max_try]:
                    clr, _qo, _qt = link_clearance(task, des, d)
                    if (clr["touch"] >= 0.0005 and clr["open"] >= 0.0015     # links 0.5 mm off when the block touches
                            and clr["touch_ff"] >= 0.001 and clr["open_ff"] >= 0.001):
                        des["link_clearance_mm"] = {k: round(1e3 * v, 2) for k, v in clr.items()}
                        passing.append(des)
                if passing:
                    out[key] = max(passing, key=lambda c: c["score"])
                    out[key]["n_ranked"], out[key]["n_passing"] = len(ranked), len(passing)
            else:
                out[key] = design_sphere(d)
            print(key, None if out[key] is None else {k: out[key].get(k) for k in ("tag", "x_p", "h", "score", "n_passing",
                                                                                    "link_clearance_mm")},
                  f"{time.time() - t0:.1f} s", flush=True)
            path = OUT / "design.json"
            old = json.loads(path.read_text()) if path.exists() else {}
            old[key] = out[key]
            _write_design(old)
    return 0


# ------------------------------------------------------------------------------------------------ scenes

def build(task: str, d_mm: float, models=("pads", "pt")):
    """(pads scene path, pt scene path, setup dict) for one task and size."""
    shape = SHAPE[task]
    des = json.loads((OUT / "design.json").read_text())[f"{shape}{d_mm:g}"]
    if des is None:
        raise RuntimeError(f"no feasible grasp for {shape} {d_mm}")
    R_ = _reach()
    cells = {(c["tag"], c["shape"], c["d_mm"]): c for c in R_["cells"]}
    cell = dict(cells[(des["tag"], shape, float(d_mm))])
    sc = S.cell_scenes(cell, shape, float(d_mm), models=models)
    r = d_mm / 2000.0
    xs, ys = des["x_sep_mm"] / 1000, des["y_sep_mm"] / 1000
    if shape == "cylinder":
        tg, n_out = _pinch_x_targets(r, CYL_LEN, des["x_p"], des["h"])
        centre = np.array([des["x_p"], 0.0, -des["h"]])
        quat = AXIS_X
        hl = CYL_LEN / 2
    else:
        gg = S.grasp_geometry(cell, shape, float(d_mm), des["h"], des["spread"])
        centre = gg["centre_palm"]
        tg = {f: centre + gg["rel"][f] for f in FINGERS}
        n_out = gg["n_out"]
        quat = SPHERE_Q
        hl = 0.0
    palm = np.array([-centre[0], -centre[1], S.Z_OBJ - centre[2]])
    paths = {"pads": None}
    for mdl in models:
        src = Path(sc[mdl][0])
        dst = src.with_name(f"{src.stem}__hom_{task}{'_wall' if task == 'press' else ''}.xml")
        root = ET.parse(src).getroot()
        tb = root.find(f".//body[@name='{S.OBJ}']")
        tb.set("quat", " ".join(f"{v:.8f}" for v in quat))
        og = tb.find("geom")
        if shape == "cylinder":
            og.set("size", f"{r:.6f} {hl:.6f}")
        if task == "press":
            # a 6 mm plate standing on the floor, its top 2 mm above the cylinder, in collision class 2 so that only
            # the object touches it (the finger links pass above it)
            x_wall = hl + WALL_GAP
            hz = (S.Z_OBJ + r + 0.002) / 2.0
            wb = ET.SubElement(root.find("worldbody"), "body", name="wall", pos=f"{x_wall + 0.003:.6f} 0 {hz:.6f}")
            ET.SubElement(wb, "geom", name="wall", type="box", size=f"0.003 0.015 {hz:.6f}", rgba="0.55 0.55 0.6 1",
                          friction=f"{S.MU:g} 0.005 0.0001", contype="2", conaffinity="2")
            og.set("contype", str(int(og.get("contype", "1")) | 2))
            og.set("conaffinity", str(int(og.get("conaffinity", "1")) | 2))
        tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")
        ET.ElementTree(root).write(tmp)
        os.replace(tmp, dst)
        paths[mdl] = dst
    mounts = K.mounts(xs, ys)
    setup = {"task": task, "shape": shape, "d_mm": d_mm, "r": r, "hl": hl, "palm": palm, "quat": np.array(quat),
             "design": des, "targets": tg, "n_out": n_out, "mounts": mounts, "meta": sc[models[0]][1]}
    return paths["pads"], paths["pt"], setup


def sr2_specs(setup):
    fingers = []
    for f in FINGERS:
        sg = G.FACE_SIGN[f]
        ctr = np.array([sg * G.CENTRE[0], G.CENTRE[1], G.CENTRE[2]])
        fingers.append(H.FingerSpec(f, [f"{f}_{j}" for j in JOINTS], f"{f}_tip", f"{f}_tipgeom", f"{f}_pip",
                                    Posed(RoundedBox(G.HALF, FILLET, centre=ctr))))
    surf = Cylinder(setup["r"], setup["hl"]) if setup["shape"] == "cylinder" else Sphere(setup["r"])
    return fingers, H.ObjectSpec(S.OBJ, None, Posed(surf))


def open_pose(setup, m_pt, open_mm=8.0):
    """Finger joint angles 8 mm off the object along each contact normal (pad-centre IK, then the TPU block mesh
    moved until it touches, as hand_object_scale_sim.grip_poses)."""
    import mujoco
    S.set_palm(m_pt, setup["palm"])
    d = mujoco.MjData(m_pt)
    tb = m_pt.body(S.OBJ).id
    qa = m_pt.jnt_qposadr[m_pt.body_jntadr[tb]]
    d.qpos[qa:qa + 7] = [0.0, 0.0, S.Z_OBJ, *setup["quat"]]
    og = int(np.flatnonzero(m_pt.geom_bodyid == tb)[0])
    lim = K.limits(False)
    q_open, q_touch = {}, {}
    for f in FINGERS:
        base = setup["targets"][f] - setup["mounts"][f]
        n = setup["n_out"][f]
        delta, q = 0.0, None
        for _ in range(6):
            q, _ok = S._ik_near(f, base + delta * n, lim[f], q)
            for k, j in enumerate(JOINTS):
                d.qpos[m_pt.jnt_qposadr[m_pt.joint(f"{f}_{j}").id]] = q[k]
            mujoco.mj_forward(m_pt, d)
            dist = mujoco.mj_geomDistance(m_pt, d, m_pt.geom(f"{f}_tipgeom").id, og, 0.05, None)
            if abs(dist) < 5e-5:
                break
            delta -= dist
        q_touch[f] = q
        q_open[f], _ = S._ik_near(f, base + (delta + open_mm / 1000) * n, lim[f], q)
    return q_open, q_touch


# ------------------------------------------------------------------------------------------------ plant

class Plant:
    """The pads scene in MuJoCo: servo targets in; joint angles, object pose and per-finger centre of pressure out."""

    def __init__(self, path: Path, setup: dict):
        import mujoco
        self.mj = mujoco
        self.m = m = mujoco.MjModel.from_xml_path(str(path))
        S.set_palm(m, setup["palm"])
        self.d = mujoco.MjData(m)
        self.setup = setup
        self.tb = m.body(S.OBJ).id
        self.qa = m.jnt_qposadr[m.body_jntadr[self.tb]]
        self.va = m.jnt_dofadr[m.body_jntadr[self.tb]]
        self.qadr = {f: np.array([m.jnt_qposadr[m.joint(f"{f}_{j}").id] for j in JOINTS]) for f in FINGERS}
        self.act = {f: np.array([m.actuator(f"a_{f}_{j}").id for j in JOINTS]) for f in FINGERS}
        self.obj_mask = np.zeros(m.ngeom, bool)
        self.obj_mask[m.geom_bodyid == self.tb] = True
        self.fidx = -np.ones(m.ngeom, int)
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            for k, f in enumerate(FINGERS):
                if bn.startswith(f + "_"):
                    self.fidx[g] = k
        self.post = m.geom("post").id
        self.post0 = (int(m.geom_contype[self.post]), int(m.geom_conaffinity[self.post]))
        self.wall = m.geom("wall").id if task_has_wall(setup) else -1
        self.v6 = np.zeros(6)

    def reset(self, q0: dict, seed: int):
        m, d = self.m, self.d
        self.mj.mj_resetData(m, d)
        m.geom_contype[self.post], m.geom_conaffinity[self.post] = self.post0
        rng = np.random.default_rng(seed)
        jit = np.zeros(3) if seed == 0 else np.array([rng.uniform(-1, 1) * 1e-3, rng.uniform(-1, 1) * 1e-3, 0.0])
        yaw = 0.0 if seed == 0 else math.radians(rng.uniform(-2, 2))
        d.qpos[self.qa:self.qa + 3] = np.array([0.0, 0.0, S.Z_OBJ]) + jit
        qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        q = np.zeros(4)
        self.mj.mju_mulQuat(q, qz, np.asarray(self.setup["quat"], float))
        d.qpos[self.qa + 3:self.qa + 7] = q
        for f in FINGERS:
            d.qpos[self.qadr[f]] = q0[f]
            d.ctrl[self.act[f]] = q0[f]
        self.mj.mj_forward(m, d)

    def post_off(self):
        self.m.geom_contype[self.post] = self.m.geom_conaffinity[self.post] = 0

    def set_targets(self, q_cmd: dict):
        for f in FINGERS:
            self.d.ctrl[self.act[f]] = q_cmd[f]

    def advance(self, t1: float):
        while self.d.time < t1 - 1e-9:
            self.mj.mj_step(self.m, self.d)

    def contacts(self):
        """Per finger (centre of pressure, normal force) on the object, and the wall's normal force."""
        d = self.d
        n = d.ncon
        cop = {}
        wallF = 0.0
        if n:
            g = d.contact.geom[:n]
            adr = d.contact.efc_address[:n]
            ok = adr >= 0
            fn = np.zeros(n)
            fn[ok] = d.efc_force[adr[ok]]
            fn = np.maximum(fn, 0.0)
            t1, t2 = self.obj_mask[g[:, 0]], self.obj_mask[g[:, 1]]
            other = np.where(t1, g[:, 1], g[:, 0])
            pair = t1 != t2
            fing = np.where(pair, self.fidx[other], -1)
            pos = d.contact.pos[:n]
            for k, f in enumerate(FINGERS):
                sel = (fing == k) & (fn > 0)
                F = float(fn[sel].sum())
                if F > 0:
                    cop[f] = ((fn[sel, None] * pos[sel]).sum(0) / F, F)
            if self.wall >= 0:
                wallF = float(fn[pair & (other == self.wall)].sum())
        return cop, wallF

    def state(self):
        d = self.d
        q = {f: d.qpos[self.qadr[f]].copy() for f in FINGERS}
        pos = d.qpos[self.qa:self.qa + 3].copy()
        quat = d.qpos[self.qa + 3:self.qa + 7].copy()
        self.mj.mj_objectVelocity(self.m, d, self.mj.mjtObj.mjOBJ_BODY, self.tb, self.v6, 0)
        cop, wallF = self.contacts()
        return {"t": float(d.time), "q": q, "pos": pos, "quat": quat, "w": self.v6[:3].copy(),
                "v": self.v6[3:].copy(), "cop": cop, "wallF": wallF}


def task_has_wall(setup):
    return setup["task"] == "press"


# ------------------------------------------------------------------------------------------------ task schedules

T_OPEN, T_CLOSE, T_SQUEEZE, T_LIFT = 0.1, 0.8, 1.2, 1.6
V_APPROACH = 0.020                                    # m/s: the paper's linear step
W_STEP = math.radians(40.0)                           # rad/s: the paper's angular step (PRM w_step_dps)


def _approach(cmd_f, t):
    """Shared pick: approach at v_sep = 20 mm/s (anti-windup stops it), then the commanded force."""
    if T_OPEN <= t < T_CLOSE:
        cmd_f.v_ref[H.SEP] = V_APPROACH
    return cmd_f


class Schedule:
    """Commands of one task as a function of time and the measured state (the paper's teleoperation sliders)."""

    def __init__(self, task, setup, prm):
        self.task, self.setup, self.prm = task, setup, prm
        self.t_end = {"pinpush": T_LIFT + 0.4 + prm["t_push"] + 1.0, "press": T_LIFT + 0.4 + 1.0 + 4.0 + 0.3,
                      "roll": T_LIFT + 0.4 + 2 * 1.3 + 0.3, "twist": T_LIFT + 0.4 + 1.3 + 4.0 + 0.5}[task]
        self.t_wall = None
        self.ref = {"ax_disp_mm": 0.0}

    def phase(self, t):
        if t < T_OPEN:
            return "open"
        if t < T_CLOSE:
            return "close"
        if t < T_LIFT:
            return "squeeze"
        if t < T_LIFT + 0.4:
            return "lift"
        return "task"

    def __call__(self, t, s):
        task = self.task
        ph = self.phase(t)
        fc = {f: H.FingerCommand() for f in FINGERS}
        cmd = H.HandCommand(fingers=fc)
        info = {"phase": ph}
        tt = t - (T_LIFT + 0.4)
        if task in ("pinpush", "press"):
            for f in ("index", "middle"):
                c = fc[f]
                c.mode, c.follow_object = "pinch", False
                c.force = 0.0 if t < T_CLOSE else self.prm["f_pinch"]
                _approach(c, t)
            th = fc["thumb"]
            th.mode = "pusher"
            th.force = 0.0 if t < T_CLOSE else self.prm["f_thumb"]
            _approach(th, t)
            cmd.pinch = ("index", "middle")
            R = H.quat_to_mat(s["quat"])
            if ph == "task":
                if task == "pinpush":
                    tp = self.prm["t_push"]
                    v_ax = self.prm["v_push"] if tt < tp else 0.0
                    if tt >= tp:
                        for f in ("index", "middle"):
                            fc[f].force = self.prm["f_lock"]
                    info["phase"] = "push" if tt < tp else "lock"
                    self.ref["ax_disp_mm"] = 1e3 * self.prm["v_push"] * min(tt, tp)
                    info["v_ax_ref"] = v_ax
                    cmd.pusher_twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, v_ax])     # along the body z axis
                else:
                    for f in ("index", "middle"):
                        fc[f].force = self.prm["f_press_pinch"]       # light prismatic pinch: the wall takes the load
                    if self.t_wall is None and (s["wallF"] > 0.05 or tt > 1.0):
                        self.t_wall = tt
                        info["reanchor"] = ["thumb"]                   # motion -> force: drop the wound-up reference
                    if self.t_wall is None:
                        info["phase"] = "to wall"
                        cmd.pusher_twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.prm["v_wall"]])
                    else:
                        for f in ("index", "middle"):
                            fc[f].force = self.prm["f_press_pinch"]   # the wall takes the axial load
                        k = int((tt - self.t_wall) // 1.0)
                        steps = self.prm["press_steps"]
                        F = steps[min(k, len(steps) - 1)]
                        th.force = F
                        info["phase"] = f"press {F:g} N"
                        info["F_ref"] = F
                        cmd.pusher_twist = np.zeros(6)
            info["R"] = R
        else:
            w_ref = np.zeros(3)
            for f in FINGERS:
                c = fc[f]
                c.mode = "pinch"
                c.weights = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])            # free rolling (paper Exp. B)
                c.force = 0.0 if t < T_CLOSE else self.prm["f_tripod"]
                _approach(c, t)
            if ph == "task":
                if task == "roll":
                    seq = [(0, +1), (0, -1), (None, 0), (1, +1), (1, -1), (None, 0)]
                    durs = [0.5, 0.5, 0.3, 0.5, 0.5, 0.3]
                else:
                    seq = [(2, +1), (2, -1), (None, 0)]
                    durs = [0.5, 0.5, 0.3]
                edges = np.cumsum([0.0] + durs)
                i = int(np.searchsorted(edges, tt, side="right")) - 1
                if i < len(seq):
                    ax, sg = seq[i]
                    if ax is not None:
                        w_ref[ax] = sg * math.radians(self.prm["w_step_dps"])
                    info["phase"] = "rest" if ax is None else f"w{'xyz'[ax]} {'+' if sg > 0 else '-'}"
                elif task == "twist":
                    tr = tt - edges[-1]                                          # ramp to the largest twist
                    if tr < 3.0:
                        w_ref[2] = math.radians(self.prm["ramp_dps"])
                        info["phase"] = "twist ramp"
                    else:
                        info["phase"] = "hold"
                else:
                    info["phase"] = "hold"
            R = H.quat_to_mat(s["quat"])
            cmd.object_twist = np.concatenate([R.T @ w_ref, np.zeros(3)])       # rotation about the centre
            info["w_ref"] = w_ref
            info["R"] = R
        return cmd, info


PRM = {"f_pinch": 1.5, "f_press_pinch": 0.6, "f_lock": 3.0, "f_thumb": 0.3, "v_push": 0.010, "t_push": 1.0,
       "v_wall": 0.008, "press_steps": (0.5, 1.0, 2.0, 0.5), "f_tripod": 2.5, "ramp_dps": 30.0, "w_step_dps": 40.0}


# ------------------------------------------------------------------------------------------------ rollout

def rollout(task, setup, plant, mirror, q_open, seed, prm, hom_prm, film=None, noise=None):
    """One task from the open pose. `noise` (deployment checks): dict(pos_mm, rot_deg, cop_mm, f_N, delay_s)."""
    sch = Schedule(task, setup, prm)
    plant.reset(q_open, seed)
    ctl = H.HomController(mirror, hom_prm, healthy={f: np.asarray(q_open[f]) for f in FINGERS})
    rng = np.random.default_rng(1000 + seed)
    dt = hom_prm.dt
    t = 0.0
    s0 = None
    rows = []
    frames = []
    lifted = False
    hist = []
    w0 = time.perf_counter()
    n_sat = 0
    while t < sch.t_end - 1e-9:
        s = plant.state()
        hist.append(s)
        if noise and noise.get("delay_s", 0) > 0:
            k = max(0, len(hist) - 1 - int(round(noise["delay_s"] / dt)))
            sm = dict(hist[k])
        else:
            sm = s
        meas = H.Measurement(q=sm["q"], obj_pos=sm["pos"], obj_quat=sm["quat"], cop=dict(sm["cop"]))
        if noise:
            meas.obj_pos = sm["pos"] + rng.normal(size=3) * noise.get("pos_mm", 0.0) * 1e-3
            ax = unit(rng.normal(size=3))
            ang = math.radians(noise.get("rot_deg", 0.0)) * rng.normal()
            dq = np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * ax])
            qn = np.zeros(4)
            plant.mj.mju_mulQuat(qn, dq, sm["quat"])
            meas.obj_quat = qn
            meas.cop = {f: (p + rng.normal(size=3) * noise.get("cop_mm", 0.0) * 1e-3,
                            max(0.0, F + rng.normal() * noise.get("f_N", 0.0))) for f, (p, F) in sm["cop"].items()}
        if t >= T_LIFT and not lifted:
            plant.post_off()
            lifted = True
        if s0 is None and t >= T_LIFT + 0.4 - 1e-9:
            s0 = s
        cmd, info = sch(t, sm)
        for f in info.get("reanchor", ()):
            ctl.reanchor(f, sm["q"][f])
        q_cmd, ci = ctl.step(meas, cmd)
        n_sat += sum(int(np.any(np.abs(ci[f]["u"]) >= hom_prm.u_max - 1e-6)) for f in FINGERS)
        R = H.quat_to_mat(s["quat"])
        row = {"t": round(t, 3), "phase": info["phase"], "pos_mm": (1e3 * s["pos"]).round(3).tolist(),
               "w_dps": np.degrees(s["w"]).round(2).tolist(), "v_mms": (1e3 * s["v"]).round(2).tolist(),
               "axis": R[:, 2].round(5).tolist(),
               "F": {f: round(s["cop"].get(f, (None, 0.0))[1], 3) for f in FINGERS},
               "dist_mm": {f: round(1e3 * ci[f]["dist"], 3) for f in FINGERS},
               "src": {f: ci[f]["source"] for f in FINGERS}}
        if "w_ref" in info:
            row["w_ref_dps"] = np.degrees(info["w_ref"]).round(2).tolist()
        if "v_ax_ref" in info:
            row["v_ax_ref_mms"] = round(1e3 * info["v_ax_ref"], 3)
        if task == "press":
            row["wallF"] = round(s["wallF"], 4)
            row["F_ref"] = info.get("F_ref")
        rows.append(row)
        plant.set_targets(q_cmd)
        t += dt
        plant.advance(t)
        if film is not None and (len(rows) - 1) % 4 == 0:
            frames.append(film.frame(plant, f"{task}, {setup['shape']} {setup['d_mm']:g} mm, seed {seed}",
                                     f"{info['phase']:<16s} t {t:4.2f} s  " + film_line(task, s, info)))
    wall = time.perf_counter() - w0
    res = metrics(task, setup, rows, s0, plant.state())
    res.update(wall_s=round(wall, 2), sat_ticks=n_sat, n_ticks=len(rows))
    return res, rows, frames


def film_line(task, s, info):
    F = " ".join(f"{f[0].upper()}{s['cop'].get(f, (None, 0.0))[1]:4.1f}" for f in FINGERS)
    if task == "press":
        return f"wall {s['wallF']:4.2f} N  pads {F} N"
    if task in ("roll", "twist"):
        return f"w {np.degrees(s['w']).round(0)} deg/s  pads {F} N"
    return f"x {1e3 * s['pos'][0]:6.2f} mm  pads {F} N"


def metrics(task, setup, rows, s0, se):
    """Scores of one rollout (see the module docstring)."""
    out = {}
    z0 = rows[0]["pos_mm"][2]
    loaded = sum(1 for f in FINGERS if se["cop"].get(f, (None, 0.0))[1] >= 0.025 * 9.81 / 2)
    out["dropped"] = bool(se["pos"][2] * 1e3 < z0 - 15.0)
    out["fingers_end"] = loaded
    out["held"] = (not out["dropped"]) and loaded >= 2
    task_rows = [r for r in rows if r["t"] >= T_LIFT + 0.4 - 1e-9]
    if not task_rows or s0 is None:
        return out
    p0 = np.array(task_rows[0]["pos_mm"])
    a0 = np.array(task_rows[0]["axis"])
    if task in ("pinpush", "press"):
        disp = np.array([(np.array(r["pos_mm"]) - p0) @ a0 for r in task_rows])
        tilt = np.array([math.degrees(math.acos(max(-1.0, min(1.0, np.array(r["axis"]) @ a0)))) for r in task_rows])
        out["tilt_max_deg"] = round(float(tilt.max()), 2)
        out["lateral_max_mm"] = round(float(max(np.linalg.norm((np.array(r["pos_mm"]) - p0) - ((np.array(r["pos_mm"]) - p0) @ a0) * a0)
                                                for r in task_rows)), 2)
        if task == "pinpush":
            push = [i for i, r in enumerate(task_rows) if r["phase"] == "push"]
            v = np.gradient(disp, 0.01)
            vr = np.array([task_rows[i].get("v_ax_ref_mms", 0.0) for i in range(len(task_rows))])
            out["disp_end_mm"] = round(float(disp[-1]), 2)
            out["disp_cmd_mm"] = round(1e3 * PRM["v_push"] * PRM["t_push"], 2)
            out["disp_err_mm"] = round(float(disp[-1] - out["disp_cmd_mm"]), 2)
            out["v_rmse_mms"] = round(float(np.sqrt(np.mean((v[push] - vr[push]) ** 2))), 2) if push else None
            out["v_rmse_ratio"] = round(out["v_rmse_mms"] / (1e3 * PRM["v_push"]), 3) if push else None
            out["thumb_contact_frac"] = round(float(np.mean([r["F"]["thumb"] > 0.02 for i, r in enumerate(task_rows)
                                                             if r["phase"] == "push"])), 3) if push else None
        else:
            st = {}
            for r in task_rows:
                if r.get("F_ref") is not None:
                    st.setdefault(r["phase"], []).append((r["t"], r["F_ref"], r["wallF"], r["F"]["thumb"]))
            steps = []
            for ph, v in st.items():
                v = np.array(v)
                tail = v[v[:, 0] >= v[-1, 0] - 0.5]
                steps.append({"F_ref": float(tail[0, 1]), "wall_mean": round(float(tail[:, 2].mean()), 3),
                              "wall_sd": round(float(tail[:, 2].std()), 3),
                              "thumb_mean": round(float(tail[:, 3].mean()), 3)})
            out["press_steps"] = steps
            out["wall_err_rms"] = round(float(np.sqrt(np.mean([(s_["wall_mean"] - s_["F_ref"]) ** 2 for s_ in steps]))), 3) \
                if steps else None
            out["reached_wall"] = bool(steps)
    else:
        W = np.array([r["w_dps"] for r in task_rows])
        Wr = np.array([r.get("w_ref_dps", [0, 0, 0]) for r in task_rows])
        act = np.abs(Wr).sum(1) > 0
        err = W - Wr
        out["w_rmse_dps"] = np.sqrt((err[act] ** 2).mean(0)).round(2).tolist() if act.any() else None
        out["w_rmse_ratio"] = round(float(np.sqrt((err[act] ** 2).sum(1).mean()) / PRM["w_step_dps"]), 3) \
            if act.any() else None
        steps = []
        for ph in dict.fromkeys(r["phase"] for r in task_rows):
            if not ph.startswith("w"):
                continue
            ax = "xyz".index(ph[1])
            idx = [i for i, r in enumerate(task_rows) if r["phase"] == ph]
            ang = float(np.sum(W[idx, ax]) * 0.01)
            ref = float(np.sum(Wr[idx, ax]) * 0.01)
            steps.append({"step": ph, "angle_deg": round(ang, 2), "ref_deg": round(ref, 2),
                          "gain": round(ang / ref, 3) if ref else None})
        out["steps"] = steps
        P = np.array([r["pos_mm"] for r in task_rows])
        out["drift_max_mm"] = round(float(np.linalg.norm(P - P[0], axis=1).max()), 2)
        if task == "twist":
            ramp = [i for i, r in enumerate(task_rows) if r["phase"] in ("twist ramp", "hold")]
            if ramp:
                yaw = np.cumsum(W[ramp, 2]) * 0.01
                ok = [i for i in ramp if sum(1 for f in FINGERS if task_rows[i]["F"][f] > 0.05) == 3]
                out["ramp_max_deg"] = round(float(yaw.max()), 2)
                out["ramp_three_pads_max_deg"] = round(float(np.cumsum(W[ramp, 2])[[ramp.index(i) for i in ok]].max()
                                                             * 0.01), 2) if ok else 0.0
                out["ramp_end_deg"] = round(float(yaw[-1]), 2)
    return out


# ------------------------------------------------------------------------------------------------ films

class Film:
    def __init__(self, plant, setup, w=640, h=480):
        import mujoco
        self.mj = mujoco
        self.r = mujoco.Renderer(plant.m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.0, 0.0, S.Z_OBJ + 0.01]
        self.cam.distance = 0.20 if setup["shape"] == "cylinder" else 0.22
        self.cam.azimuth = 90.0                      # from -y: the cylinder side-on, the thumb on the left end cap
        self.cam.elevation = 10.0 if setup["shape"] == "cylinder" else 5.0

    def frame(self, plant, title, line):
        import hom_contact_rig as HR
        self.r.update_scene(plant.d, self.cam)
        return HR.annotate_bottom(HR.annotate(self.r.render().copy(), title), line)


# ------------------------------------------------------------------------------------------------ CLI

def _done(path: Path) -> set:
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok":
                out.add((r["task"], r["d_mm"], r["seed"], r.get("variant", "base")))
    return out


def _append(path: Path, row: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


VARIANTS = {
    "base": ({}, None),
    "nofilter": ({"use_filter": False}, None),
    "rate50": ({"dt": 0.02}, None),
    "sense": ({"dt": 0.02}, {"pos_mm": 0.5, "rot_deg": 1.0, "cop_mm": 1.0, "f_N": 0.03, "delay_s": 0.04}),
}


def cmd_run(a) -> int:
    import mujoco
    ROWS.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        path = ROWS / f"{task}.jsonl"
        done = _done(path)
        for d_mm in (a.d or SIZES[task]):
            for var in a.variants:
                vname = var + (f"_{a.tag}" if a.tag else "")
                todo = [sd for sd in a.seeds if (task, float(d_mm), sd, vname) not in done]
                if not todo and not a.film:
                    continue
                try:
                    p_pads, p_pt, setup = build(task, float(d_mm))
                except Exception as e:
                    print(task, d_mm, "build failed:", e, flush=True)
                    continue
                m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
                S.set_palm(m_pt, setup["palm"])
                fs, os_ = sr2_specs(setup)
                mirror = H.Mirror(m_pt, fs, os_)
                q_open, _ = open_pose(setup, mujoco.MjModel.from_xml_path(str(p_pt)))
                plant = Plant(p_pads, setup)
                over, noise = VARIANTS[var]
                hprm = replace(H.HomParams(dt=DT_C), **over)
                for seed in (a.seeds if a.film else todo):
                    film = Film(plant, setup) if (a.film and seed == a.seeds[0]) else None
                    row = {"task": task, "shape": setup["shape"], "d_mm": float(d_mm), "seed": seed,
                           "variant": var + (f"_{a.tag}" if a.tag else ""),
                           "design": {k: setup["design"][k] for k in ("tag", "h", "score")},
                           "prm": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PRM.items()},
                           "hom": {k: v for k, v in vars(hprm).items() if k != "filter"},
                           "filter": vars(hprm.filter), "when": time.strftime("%Y-%m-%d %H:%M")}
                    try:
                        res, trace, frames = rollout(task, setup, plant, mirror, q_open, seed, PRM, hprm, film, noise)
                        row.update(res, status="ok")
                        if a.traces or film is not None:
                            row["trace"] = trace[::2]
                        if film is not None and frames:
                            import hom_contact_rig as HR
                            fp = MEDIA / f"20261021-hom_{task}_{setup['shape']}{d_mm:g}_{var}.mp4"
                            HR.write_mp4(frames, fp, fps=25)
                            row["film"] = str(fp.relative_to(ROOT))
                    except Exception as e:
                        row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
                    if (task, float(d_mm), seed, vname) not in done or film is not None:
                        _append(path, row)
                    brief = {k: row.get(k) for k in ("held", "disp_end_mm", "v_rmse_ratio", "wall_err_rms",
                                                     "w_rmse_ratio", "ramp_three_pads_max_deg", "tilt_max_deg",
                                                     "drift_max_mm", "wall_s", "error")
                             if row.get(k) is not None}
                    print(task, d_mm, var, seed, brief, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("design")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--d", nargs="+", type=float)
    p.add_argument("--max-try", type=int, default=400)
    p = sub.add_parser("run")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--d", nargs="+", type=float)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--variants", nargs="+", default=["base"], choices=sorted(VARIANTS))
    p.add_argument("--film", action="store_true")
    p.add_argument("--traces", action="store_true")
    p.add_argument("--set", nargs="*", default=[], help="task parameters k=v (PRM), recorded in each row's variant")
    p.add_argument("--tag", default="", help="suffix of the variant name for rows run with --set")
    a = ap.parse_args()
    if a.cmd == "run":
        for kv in a.set:
            k, v = kv.split("=")
            PRM[k] = type(PRM[k])(float(v)) if not isinstance(PRM[k], tuple) else tuple(float(x) for x in v.split(","))
    return cmd_design(a) if a.cmd == "design" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
