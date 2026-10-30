#!/usr/bin/env python3
"""Four manipulation tasks under the HOM controller (src/morphohand/control/hom.py) on the CAD-model SR2 hand, across
object sizes, in CPU MuJoCo on the bench servo plant (kp 4 N m/rad, 1 N m, damping 0.08) with the printed TPU block as
1 mm sphere pads (builder v2), 1 ms physics, 100 Hz control.

Tasks. Each maps a sub-assembly of the HOM paper (arXiv 2609.25619v2) onto the three-finger SR2 hand, whose fingers
hang from a palm plate above the object. The demonstrations are the four clips in docs/experiments/20261021-hom_framework
(cylinder_push.mov, cylinder_press.mov, roll_sphere.mov, twist_sphere.mov).
  pinpush  Pin-pusher with a friction drive. A 100 mm cylinder lies along palm y. Thumb and index pinch it across palm x
           at y_p: the pinch is a revolute joint about the line through the two contacts (the paper's Exp. C, both
           pinching fingers with V_PB0 = 0 and w_spin = 0). The middle finger's pad touches the cylinder beside the
           index at y_m, LEVER_MIN or more from the pinch, with its normal parallel to the pinch axis. It is the
           pusher: its reference object twist is Eq. (3)'s rotation s about the pinch axis, and with its pad held
           sticking (w_slide) and pressed with f_drive, its tangential motion drags the cylinder round the axis by
           friction, in both directions. (The demonstration pushes with the back of a finger, where SR2 has no
           tactile sensing.) Command: +-15 deg at 15 deg/s. Score: angle about the pinch axis against the command
           (RMSE, gain per stroke), sliding of the middle's pad on the cylinder, rotation off the pinch axis.
  press    The paper's pusher on the same pinch: the middle's pad touches the cylinder 30 deg below its equator, so its
           normal force has a component across the pinch axis and lifts the middle's side; it presses lightly (f_press)
           and its reference motion pushes. Command: up 20 deg at 20 deg/s, hold 1 s, down again. Scores as pinpush.
           Both cylinder tasks use the weights of the paper's Exp. C: every finger rolls freely (w_roll = 0), the
           pinching pads also spin (w_spin = 0), and the pusher does not slide.
  roll     Sphere at the circumcentre of the three mounts, equator tripod (the scale study's grasp). The sphere rolls
           along the thumb's pad, as in the demonstration: its reference body twist is a rotation about palm y through
           the thumb's contact (the centre moves along the thumb at r w), steps of +-40 deg/s for 0.5 s; index and
           middle drive it, every finger with free rolling (w_roll = 0, the paper's Exp. B), 2.5 N per pad. Score:
           angular-velocity RMSE / step (the paper's metric), achieved angle per step, error of the centre's path.
  twist    The roll grasp; steps of +-40 deg/s about the vertical through the centre, then a ramp at 30 deg/s to the
           largest twist held on three pads.
Every task starts with the paper's pick: fingers 8 mm off the object approach at v_sep = 20 mm/s until the anti-windup
stops them 0.5 mm away, the commanded forces turn on, and the support post is removed.
Commits before 2026-10-22 read the pin-push as a prismatic pinch slid by the thumb on the end cap, the press as the thumb
pushing the cylinder against a wall, and the roll as a rotation in place about the horizontal axes.

    .venv/bin/python scripts/hom_tasks.py design                                  # grasp per task and size
    .venv/bin/python scripts/hom_tasks.py design --tasks pinpush --lever 30       # middle 30-35 mm from the pinch
    .venv/bin/python scripts/hom_tasks.py run --tasks pinpush --variants l30 l60  # ... and the runs on those grasps
    .venv/bin/python scripts/hom_tasks.py run --tasks pinpush press --d 10 25 --seeds 0 1 2
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
from morphohand.control.surfaces import Cylinder, HalfEllipsoid, Posed, RoundedBox, Sphere, unit  # noqa: E402
from hom_rows import rescale_metrics  # noqa: E402,F401  (stored rows scored with the fixed 0.01 s step)

OUT = ROOT / "docs/experiments/20261021-hom_framework"
ROWS = OUT / "rows"
MEDIA = OUT / "media"
SCN = ROOT / "logs/20261021-hom_tasks/scenes"
FINGERS, JOINTS = S.FINGERS, S.JOINTS
TASKS = ("pinpush", "press", "roll", "twist", "expa")
SIZES = {"pinpush": (10.0, 16.0, 25.0, 40.0), "press": (10.0, 16.0, 25.0, 40.0),
         "roll": (30.0, 40.0, 50.0, 63.0), "twist": (30.0, 40.0, 50.0, 63.0), "expa": (25.0,)}
SHAPE = {"pinpush": "cylinder", "press": "cylinder", "roll": "sphere", "twist": "sphere", "expa": "cylinder"}
GRASP = {"pinpush": "pinch_y", "press": "pinch_y", "roll": "equator", "twist": "equator", "expa": "equator"}
CYL_LEN = 0.100                  # m: the scale study's tool length
EXPA_MOVER = "thumb"             # Exp. A: the finger whose palmar side faces the cylinder in the equator grasp
PINCH, PUSHER = ("thumb", "index"), "middle"
LEVER_MIN = 0.040                # m: the middle's contact at least this far from the pinch along the cylinder
LEVER_WIN = 0.005                # m: with design --lever, the lever lies in [lever, lever + LEVER_WIN]
PHI_P = (0.0, 15.0)              # deg below the equator: the pinching pads' contact latitudes searched (at 30 deg the
                                 # pinch squeezes the cylinder out upward)
PHI_M = {"pinpush": (0.0, 15.0), "press": (30.0, 45.0)}   # the middle's: at or near the equator (friction), below it
                                                         # (normal force)


def dkey(task, d_mm, lever=None):
    """design.json key: thumb-index pinch grasps per task (pinpush_cylinder25, the middle's latitude differs), with the
    lever window's lower end in mm when one was set (pinpush_l30_cylinder25); equator grasps of a sphere plain
    (sphere40), of a cylinder with eq_ (eq_cylinder25)."""
    shape = SHAPE[task]
    if GRASP[task] == "pinch_y":
        return f"{task}_{'' if lever is None else f'l{lever:g}_'}{shape}{d_mm:g}"
    return f"{'eq_' if shape == 'cylinder' else ''}{shape}{d_mm:g}"
KP = 4.0
DT_C = 0.01
FILLET = 0.0027
SPHERE_Q = (0.70710678, 0.70710678, 0.0, 0.0)       # the scale study's object orientation: body z (a cylinder's axis)
                                                    # along palm -y


# ------------------------------------------------------------------------------------------------ grasp design

def _reach():
    return json.loads((S.OUT_DIR / "reach.json").read_text())


def _pinch_y_targets(r, des):
    """Pad-centre targets (palm frame) and object-outward contact normals of the thumb-index pinch: the cylinder's axis
    along palm y through (x_c, ., -h); thumb and index across x at y_p, phi_p below the equator; the middle on the
    index's side at y_m, phi_m below the equator (phi_m = 0: its normal parallel to the pinch axis)."""
    pp, pm = math.radians(des["phi_p"]), math.radians(des["phi_m"])
    n_out = {"thumb": np.array([-math.cos(pp), 0.0, -math.sin(pp)]),
             "index": np.array([math.cos(pp), 0.0, -math.sin(pp)]),
             "middle": np.array([math.cos(pm), 0.0, -math.sin(pm)])}
    yy = {"thumb": des["y_p"], "index": des["y_p"], "middle": des["y_m"]}
    rr = r + K.R + K.GAP
    tg = {f: np.array([des["x_c"], yy[f], -des["h"]]) + rr * n_out[f] for f in FINGERS}
    return tg, n_out


def design_pinch_y(d_mm: float, task: str, L: float = CYL_LEN, keep: int = 400, lever=None):
    """Ranked grasps of the thumb-index pinch for a cylinder of diameter d along palm y: over the tripod layouts of the
    scale study, the axis position x_c and depth h, the pinch's y_p and latitude phi_p (PHI_P) and the middle's y_m and
    latitude (PHI_M[task]), the reachable grasps (servo limits with 3 deg margin) whose distal links face the object
    (palmar normal within 25 deg of the contact normal), with the middle LEVER_MIN to L - 20 mm from the pinch (with
    `lever` in mm: lever to lever + LEVER_WIN), ranked by the smallest pad-Jacobian singular value of the three fingers.
    The pad centre is the scale study's (a sphere of the phalange radius about the distal link's end); open_pose then
    moves each finger until the TPU block touches."""
    lo, hi = (LEVER_MIN, L - 0.020) if lever is None else (lever / 1000.0, lever / 1000.0 + LEVER_WIN)
    r = d_mm / 2000.0
    lim = K.limits(False)
    rr = r + K.R + K.GAP
    cos_ok = math.cos(math.radians(25.0))
    xc = np.arange(-0.03, 0.0301, 0.0025)
    ranked = []
    for lay in _reach()["families"]["diag"]:
        xs, ys = lay["x_sep"] / 1000, lay["y_sep"] / 1000
        mt = K.mounts(xs, ys)
        b = ys / 2
        hh = np.arange(r + 0.02, 0.1001, 0.0025)
        yv = np.arange(-b, b + 1e-9, 0.0025)

        def solve(f, phis):
            X, Hh, Y, PH = np.meshgrid(xc, hh, yv, np.radians(phis), indexing="ij")
            sx = -1.0 if f == "thumb" else 1.0
            n = np.stack([sx * np.cos(PH), np.zeros_like(PH), -np.sin(PH)], -1)
            T = np.stack([X, Y, -Hh], -1) + rr * n
            q, ok = K.ik(f, T - mt[f])
            qq, good, _, _ = K.pick_branch(q, ok, lim[f], math.radians(3.0))
            pc = (K.fk(f, qq)["palmar"] * (-n)).sum(-1)
            _, smin = K.jac_cond(f, qq)
            return np.where(good & (pc >= cos_ok), smin, -np.inf), qq, pc

        st, qt, pct = solve("thumb", PHI_P)
        si, qi, pci = solve("index", PHI_P)
        sm, qm, pcm = solve("middle", PHI_M[task])
        sti = np.minimum(st, si)                                 # (x_c, h, y_p, phi_p)
        kp_ = sti.argmax(-1)
        bti = np.take_along_axis(sti, kp_[..., None], -1)[..., 0]
        km = sm.argmax(-1)
        bm = np.take_along_axis(sm, km[..., None], -1)[..., 0]  # (x_c, h, y_m)
        lever = yv[:, None] - yv[None, :]                        # y_p - y_m
        comb = np.minimum(bti[:, :, :, None], bm[:, :, None, :])
        comb = np.where((lever >= lo - 1e-9) & (lever <= hi + 1e-9), comb, -np.inf)
        flat = np.argsort(comb, axis=None)[::-1][:keep]
        for i, j, k, l in zip(*np.unravel_index(flat, comb.shape)):
            if not np.isfinite(comb[i, j, k, l]):
                break
            a, c = kp_[i, j, k], km[i, j, l]
            q = {"thumb": qt[i, j, k, a], "index": qi[i, j, k, a], "middle": qm[i, j, l, c]}
            ranked.append({"score": float(comb[i, j, k, l]), "tag": lay["tag"], "t": lay["t"],
                           "x_sep_mm": lay["x_sep"], "y_sep_mm": lay["y_sep"], "x_c": float(xc[i]), "h": float(hh[j]),
                           "y_p": float(yv[k]), "y_m": float(yv[l]), "phi_p": float(PHI_P[a]),
                           "phi_m": float(PHI_M[task][c]), "L": L, "d_mm": d_mm,
                           "lever_mm": round(1e3 * float(yv[k] - yv[l]), 2),
                           "q_deg": {f: np.degrees(v).round(2).tolist() for f, v in q.items()},
                           "palmar_cos": {"thumb": round(float(pct[i, j, k, a]), 3),
                                          "index": round(float(pci[i, j, k, a]), 3),
                                          "middle": round(float(pcm[i, j, l, c]), 3)}})
    ranked.sort(key=lambda c: -c["score"])
    return ranked


def link_clearance(task, des, d_mm, lever=None):
    """Smallest distance (m) from any finger geom other than the tip's to the object at the touch and the open pose,
    on the TPU-block mesh model (pt) of the design's layout; and the finger joints at both poses."""
    import mujoco
    tmp = dict(json.loads((OUT / "design.json").read_text())) if (OUT / "design.json").exists() else {}
    key = dkey(task, d_mm, lever)
    tmp[key] = des
    _write_design(tmp)
    _p_pads, p_pt, setup = build(task, d_mm, models=("pt",), lever=lever)
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


def design_sphere(d_mm: float, shape: str = "sphere"):
    """Best equator grasp (roll, twist, Exp. A): the scale study's equator tripod (hand_object_scale_sim.candidates) on
    the layout and palm height whose least-conditioned finger has the largest smallest singular value."""
    R_ = _reach()
    cells = {(c["tag"], c["shape"], c["d_mm"]): c for c in R_["cells"]}
    best = None
    for lay in R_["families"]["diag"]:
        c = cells.get((lay["tag"], shape, float(d_mm)))
        if c is None:
            continue
        for k, cand in enumerate(S.candidates(dict(c), shape, float(d_mm))):
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
            lever = a.lever if GRASP[task] == "pinch_y" else None
            key = dkey(task, d, lever)
            if key in out:
                continue
            t0 = time.time()
            if GRASP[task] == "pinch_y":
                out[key] = None
                passing = []
                ranked = design_pinch_y(d, task, lever=lever)
                for des in ranked[:a.max_try]:
                    clr, _qo, _qt = link_clearance(task, des, d, lever)
                    if (clr["touch"] >= 0.0005 and clr["open"] >= 0.0015     # links 0.5 mm off when the block touches
                            and clr["touch_ff"] >= 0.001 and clr["open_ff"] >= 0.001):
                        des["link_clearance_mm"] = {k: round(1e3 * v, 2) for k, v in clr.items()}
                        passing.append(des)
                if passing:
                    out[key] = max(passing, key=lambda c: c["score"])
                    out[key]["n_ranked"], out[key]["n_passing"] = len(ranked), len(passing)
            else:
                out[key] = design_sphere(d, SHAPE[task])
            print(key, None if out[key] is None else {k: out[key].get(k) for k in ("tag", "x_c", "h", "y_p", "y_m", "phi_p",
                                                                                    "phi_m", "score", "n_passing",
                                                                                    "link_clearance_mm")},
                  f"{time.time() - t0:.1f} s", flush=True)
            path = OUT / "design.json"
            old = json.loads(path.read_text()) if path.exists() else {}
            old[key] = out[key]
            _write_design(old)
    return 0


# ------------------------------------------------------------------------------------------------ scenes

def build(task: str, d_mm: float, models=("pads", "pt"), lever=None):
    """(pads scene path, pt scene path, setup dict) for one task and size (`lever`: the grasp of design --lever)."""
    shape = SHAPE[task]
    des = json.loads((OUT / "design.json").read_text())[dkey(task, d_mm, lever)]
    if des is None:
        raise RuntimeError(f"no feasible grasp for {shape} {d_mm}")
    R_ = _reach()
    cells = {(c["tag"], c["shape"], c["d_mm"]): c for c in R_["cells"]}
    cell = dict(cells[(des["tag"], shape, float(d_mm))])
    sc = S.cell_scenes(cell, shape, float(d_mm), models=models)
    r = d_mm / 2000.0
    xs, ys = des["x_sep_mm"] / 1000, des["y_sep_mm"] / 1000
    if GRASP[task] == "pinch_y":
        tg, n_out = _pinch_y_targets(r, des)
        centre = np.array([des["x_c"], 0.5 * (des["y_p"] + des["y_m"]), -des["h"]])   # mass between pinch and middle
    else:
        gg = S.grasp_geometry(cell, shape, float(d_mm), des["h"], des["spread"])
        centre = gg["centre_palm"]
        tg = {f: centre + gg["rel"][f] for f in FINGERS}
        n_out = gg["n_out"]
    quat = SPHERE_Q                                      # the scale study's: a cylinder lies along palm y
    hl = CYL_LEN / 2 if shape == "cylinder" else 0.0
    palm = np.array([-centre[0], -centre[1], S.Z_OBJ - centre[2]])
    paths = {"pads": None}
    for mdl in models:
        src = Path(sc[mdl][0])
        dst = src.with_name(f"{src.stem}__hom_{task}.xml")
        root = ET.parse(src).getroot()
        tb = root.find(f".//body[@name='{S.OBJ}']")
        tb.set("quat", " ".join(f"{v:.8f}" for v in quat))
        if shape == "cylinder":
            tb.find("geom").set("size", f"{r:.6f} {hl:.6f}")
        tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")
        ET.ElementTree(root).write(tmp)
        os.replace(tmp, dst)
        paths[mdl] = dst
    mounts = K.mounts(xs, ys)
    setup = {"task": task, "shape": shape, "d_mm": d_mm, "r": r, "hl": hl, "palm": palm, "quat": np.array(quat),
             "design": des, "targets": tg, "n_out": n_out, "mounts": mounts, "meta": sc[models[0]][1]}
    return paths["pads"], paths["pt"], setup


ELL_PAD = (0.0085, 0.0074, 0.011, 0.022)    # cad_v2's 11/22 pad: depth, across, distal and proximal semi-axes (m)


def tip_surface(f):
    """The analytic surface of finger f's tip in its tip frame: the TPU block (cad_v1, real_v1), or under
    SCALE_HAND=cad_v2 the half-ellipsoid pad whose apex lies K.R from the finger's axis on the palmar side."""
    sg = G.FACE_SIGN[f]
    if K.HAND == "cad_v2":
        return Posed(HalfEllipsoid(*ELL_PAD), pos=(sg * (K.R - ELL_PAD[0]), 0.0, 0.0))
    return Posed(RoundedBox(G.HALF, FILLET, centre=np.array([sg * G.CENTRE[0], G.CENTRE[1], G.CENTRE[2]])))


def sr2_specs(setup):
    fingers = []
    for f in FINGERS:
        fingers.append(H.FingerSpec(f, [f"{f}_{j}" for j in JOINTS], f"{f}_tip", f"{f}_tipgeom", f"{f}_pip",
                                    tip_surface(f)))
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
        self.vadr = {f: np.array([m.jnt_dofadr[m.joint(f"{f}_{j}").id] for j in JOINTS]) for f in FINGERS}
        self.act = {f: np.array([m.actuator(f"a_{f}_{j}").id for j in JOINTS]) for f in FINGERS}
        self.obj_mask = np.zeros(m.ngeom, bool)
        self.obj_mask[m.geom_bodyid == self.tb] = True
        self.fidx = -np.ones(m.ngeom, int)                 # any body of finger k
        self.tidx = -np.ones(m.ngeom, int)                 # the tip body of finger k (its pads)
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            for k, f in enumerate(FINGERS):
                if bn.startswith(f + "_"):
                    self.fidx[g] = k
                    if bn == f"{f}_tip":
                        self.tidx[g] = k
        self.post = m.geom("post").id
        self.post0 = (int(m.geom_contype[self.post]), int(m.geom_conaffinity[self.post]))
        self.v6 = np.zeros(6)
        self.hold_s = 0.0                                  # the object held at its start pose up to this time (s)

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
        self.obj0 = d.qpos[self.qa:self.qa + 7].copy()
        self.mj.mj_forward(m, d)

    def post_off(self):
        self.m.geom_contype[self.post] = self.m.geom_conaffinity[self.post] = 0

    def set_targets(self, q_cmd: dict):
        for f in FINGERS:
            self.d.ctrl[self.act[f]] = q_cmd[f]

    def advance(self, t1: float):
        while self.d.time < t1 - 1e-9:
            self.mj.mj_step(self.m, self.d)
        if self.hold_s and self.d.time <= self.hold_s + 1e-9:
            # variant hold: the object back at its start pose, at rest (a cradle in place of the post, on which the
            # cylinder rolls off within the approach: 2026-10-30, scripts/hom_reanchor_check.py seeds)
            self.d.qpos[self.qa:self.qa + 7] = self.obj0
            self.d.qvel[self.va:self.va + 6] = 0.0
            self.mj.mj_forward(self.m, self.d)

    def contacts(self):
        """Per finger the centre of pressure and normal force of its tip's pads on the object (what a fingertip sensor
        measures, and the paper's contact of the distal link), and the force of all its bodies on the object (links
        included)."""
        d = self.d
        n = d.ncon
        cop, Ftot = {}, {f: 0.0 for f in FINGERS}
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
            tip = np.where(pair, self.tidx[other], -1)
            pos = d.contact.pos[:n]
            for k, f in enumerate(FINGERS):
                Ftot[f] = float(fn[(fing == k) & (fn > 0)].sum())
                sel = (tip == k) & (fn > 0)
                F = float(fn[sel].sum())
                if F > 0:
                    cop[f] = ((fn[sel, None] * pos[sel]).sum(0) / F, F)
        return cop, Ftot

    def state(self):
        d = self.d
        q = {f: d.qpos[self.qadr[f]].copy() for f in FINGERS}
        qd = {f: d.qvel[self.vadr[f]].copy() for f in FINGERS}
        pos = d.qpos[self.qa:self.qa + 3].copy()
        quat = d.qpos[self.qa + 3:self.qa + 7].copy()
        self.mj.mj_objectVelocity(self.m, d, self.mj.mjtObj.mjOBJ_BODY, self.tb, self.v6, 0)
        cop, Ftot = self.contacts()
        return {"t": float(d.time), "q": q, "qd": qd, "pos": pos, "quat": quat, "w": self.v6[:3].copy(),
                "v": self.v6[3:].copy(), "cop": cop, "Ftot": Ftot}


# ------------------------------------------------------------------------------------------------ task schedules

T_OPEN, T_CLOSE, T_SQUEEZE, T_LIFT = 0.1, 0.8, 1.2, 1.6
EXPA_T0, EXPA_STEP, EXPA_REST = 0.3, 0.3, 0.3          # s; a 0.3 s step of 20 mm/s keeps the 8 mm gap open
EXPA_ORDER = (0, 1, 2, 3, 4, 5)                      # spin, roll_y, roll_z, sep, slide_y, slide_z
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
        self.prof = None
        if task == "pinpush":                            # +A, -A, back to 0, hold: both directions by friction
            A, w = prm["rot_deg"], prm["rot_dps"]
            self.prof = [(A / w, w, "up"), (2 * A / w, -w, "down"), (A / w, w, "back"), (0.5, 0.0, "hold")]
        elif task == "press":                            # up by the normal force, hold, down by the weight
            A, w = prm["press_deg"], prm["press_dps"]
            self.prof = [(A / w, w, "press"), (1.0, 0.0, "hold up"), (A / w, -w, "release"), (0.5, 0.0, "hold")]
        t_task = sum(p_[0] for p_ in self.prof) if self.prof else {"roll": 2 * 1.3, "twist": 1.3 + 4.0}.get(task, 0.0)
        self.t_end = (EXPA_T0 + 6 * (2 * EXPA_STEP + EXPA_REST) + 0.2 if task == "expa"
                      else T_LIFT + 0.4 + t_task + (0.3 if task == "roll" else 0.5 if task == "twist" else 0.0))

    def angle(self, tt):
        """Commanded rotation about the pinch axis at task time tt: angle (deg, + lifts the middle's side), rate
        (deg/s) and the segment's name."""
        th = 0.0
        for dur, w, name in self.prof:
            if tt < dur:
                return th + w * tt, w, name
            th += w * dur
            tt -= dur
        return th, 0.0, self.prof[-1][2]

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

    def expa(self, t):
        """The paper's Exp. A: one finger (EXPA_MOVER, the thumb, whose palmar side faces the cylinder in the equator
        grasp) moves round the cylinder without contact; each element of its reference relative contact velocity is
        stepped + then - (40 deg/s or 20 mm/s), the others held at zero; the other fingers hold their open pose. The
        100 mm cylinder of the scale study rests on its post along palm y."""
        fc = {f: H.FingerCommand(follow_object=False) for f in FINGERS}
        cmd = H.HandCommand(fingers=fc)
        info = {"phase": "rest", "v_ref": np.zeros(6)}
        tt = t - EXPA_T0
        if tt >= 0:
            k, r = divmod(tt, 2 * EXPA_STEP + EXPA_REST)
            k = int(k)
            if k < 6:
                mag = math.radians(40.0) if k < 3 else 0.020
                elem = EXPA_ORDER[k]
                sg = 1.0 if r < EXPA_STEP else (-1.0 if r < 2 * EXPA_STEP else 0.0)
                v = np.zeros(6)
                v[elem] = sg * mag
                fc[EXPA_MOVER].v_ref = v
                info = {"phase": f"{H.COMPONENTS[elem]} {'+' if sg > 0 else '-' if sg < 0 else ''}".strip(), "v_ref": v}
        return cmd, info

    def __call__(self, t, s):
        task = self.task
        if task == "expa":
            return self.expa(t)
        ph = self.phase(t)
        fc = {f: H.FingerCommand() for f in FINGERS}
        cmd = H.HandCommand(fingers=fc)
        info = {"phase": ph}
        tt = t - (T_LIFT + 0.4)
        if task in ("pinpush", "press"):
            for f in PINCH:                              # the revolute pinch of Exp. C: held, free to spin and roll
                c = fc[f]
                c.mode, c.follow_object = "pinch", False
                c.weights = np.array([0.0, self.prm["w_roll_pinch"], self.prm["w_roll_pinch"], 1.0, 1.0, 1.0])
                c.force = 0.0 if t < T_CLOSE else self.prm["f_pinch"]
                _approach(c, t)
            pu = fc[PUSHER]
            pu.mode = "pusher"
            pu.force = 0.0 if t < T_CLOSE else self.prm["f_drive" if task == "pinpush" else "f_press"]
            pu.weights = np.array([1.0, self.prm["w_roll_push"], self.prm["w_roll_push"], 1.0, 1.0, 1.0])
            # the thumb alone opposes the index and the middle along the pinch axis: it presses with both forces'
            # components along that axis, so the commanded forces balance and the cylinder stays between the pads
            n_t, n_m = self.setup["n_out"][PINCH[0]], self.setup["n_out"][PUSHER]
            fc[PINCH[0]].force += pu.force * max(0.0, -(n_m @ n_t))
            _approach(pu, t)
            cmd.pinch = PINCH
            if ph == "task":
                th, w, name = self.angle(tt)
                cmd.s = -math.radians(w)                 # {M}'s x runs thumb -> index (+x): + s lowers the middle's side
                info["phase"] = name
                info["theta_ref"] = th
            info["R"] = H.quat_to_mat(s["quat"])
        else:
            w_ref = np.zeros(3)
            for f in FINGERS:
                c = fc[f]
                c.mode = "pinch"
                c.weights = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])            # free rolling (paper Exp. B)
                c.force = 0.0 if t < T_CLOSE else self.prm["f_tripod"]
                _approach(c, t)
            if ph == "task":
                if task == "roll":                       # about palm y through the thumb's contact: down, up, up, down
                    seq = [(1, +1), (1, -1), (None, 0), (1, -1), (1, +1), (None, 0)]
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
                    rep = " 2" if seq[:i].count(seq[i]) else ""          # the second step of the same sign
                    info["phase"] = "rest" if ax is None else f"w{'xyz'[ax]} {'+' if sg > 0 else '-'}{rep}"
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
            v_ref = np.zeros(3)
            if task == "roll":
                # rolling on a still thumb: the instantaneous axis passes through the thumb's contact, r along the
                # thumb's inward normal from the centre, so the centre moves at w x (r n_in)
                v_ref = np.cross(w_ref, -self.setup["r"] * self.setup["n_out"]["thumb"])
            cmd.object_twist = np.concatenate([R.T @ w_ref, R.T @ v_ref])
            info["w_ref"] = w_ref
            info["v_ref"] = v_ref
            info["R"] = R
        return cmd, info


PRM = {"f_pinch": 3.0, "f_drive": 1.5, "f_press": 0.5, "rot_deg": 15.0, "rot_dps": 15.0, "press_deg": 20.0,
       "press_dps": 20.0, "w_roll_pinch": 0.0, "w_roll_push": 1.0, "f_tripod": 2.5, "ramp_dps": 30.0,
       "w_step_dps": 40.0}


# ------------------------------------------------------------------------------------------------ rollout

def rollout(task, setup, plant, mirror, q_open, seed, prm, hom_prm, film=None, noise=None, hold_s=0.0):
    """One task from the open pose. `noise` (deployment checks): dict(pos_mm, rot_deg, cop_mm, f_N, delay_s).
    hold_s: the object is held at its start pose up to this time (PLANT_VARIANTS)."""
    sch = Schedule(task, setup, prm)
    plant.reset(q_open, seed)
    plant.hold_s = hold_s
    ctl = H.HomController(mirror, hom_prm, healthy={f: np.asarray(q_open[f]) for f in FINGERS})
    rng = np.random.default_rng(1000 + seed)
    dt = hom_prm.dt
    t = 0.0
    s0 = None
    rows = []
    frames = []
    lifted = False
    hist = []
    tip = cal = None
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
        if noise and cal is None:
            # calibration errors, constant over a rollout: each servo reports and executes about an angle offset from
            # the model's joint angle, and the tag tracker's object frame is offset from the object
            if noise.get("q_bias_deg") or noise.get("pos_bias_mm"):
                cal = {"q": {f: np.radians(rng.normal(size=3) * noise.get("q_bias_deg", 0.0)) for f in FINGERS},
                       "pos": rng.normal(size=3) * noise.get("pos_bias_mm", 0.0) * 1e-3}
            else:                                # no draws: the other variants keep their random streams
                cal = {"q": {f: np.zeros(3) for f in FINGERS}, "pos": np.zeros(3)}
        if noise:
            meas.obj_pos = sm["pos"] + cal["pos"] + rng.normal(size=3) * noise.get("pos_mm", 0.0) * 1e-3
            meas.q = {f: np.asarray(v) + cal["q"][f] for f, v in sm["q"].items()}
            ax = unit(rng.normal(size=3))
            ang = math.radians(noise.get("rot_deg", 0.0)) * rng.normal()
            dq = np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * ax])
            qn = np.zeros(4)
            plant.mj.mju_mulQuat(qn, dq, sm["quat"])
            meas.obj_quat = qn
            if noise.get("no_tip"):
                meas.cop = {}                    # no fingertip sensing: every contact frame from the witness points
            elif noise.get("tip_model"):
                if tip is None:
                    tip = tip_errors(noise, rng)
                meas.cop = tip_reading(sm["cop"], noise, rng, tip)
            else:
                meas.cop = {f: (p + rng.normal(size=3) * noise.get("cop_mm", 0.0) * 1e-3,
                                max(0.0, F + rng.normal() * noise.get("f_N", 0.0))) for f, (p, F) in sm["cop"].items()}
        if t >= T_LIFT and not lifted and task != "expa":
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
               "F_link": {f: round(max(0.0, s["Ftot"][f] - s["cop"].get(f, (None, 0.0))[1]), 3) for f in FINGERS},
               "dist_mm": {f: round(1e3 * ci[f]["dist"], 3) for f in FINGERS},
               "src": {f: ci[f]["source"] for f in FINGERS}}
        if task == "expa":
            g = ctl.gcf(EXPA_MOVER, meas)                                # Eq. (1) at the measured joint rates
            A, b = ctl.eq1_rows(EXPA_MOVER, g, np.zeros(6))
            row["v_ref"] = info["v_ref"].round(5).tolist()
            row["v_ach"] = (A @ s["qd"][EXPA_MOVER] + b).round(5).tolist()
            row["gap_mm"] = round(1e3 * g.dist, 3)
        row["cop_obj_mm"] = {f: (1e3 * (R.T @ (p_ - s["pos"]))).round(3).tolist() for f, (p_, _F) in s["cop"].items()}
        if "w_ref" in info:
            row["w_ref_dps"] = np.degrees(info["w_ref"]).round(2).tolist()
            row["v_ref_mms"] = (1e3 * info["v_ref"]).round(3).tolist()
        if "theta_ref" in info:
            row["theta_ref_deg"] = round(info["theta_ref"], 3)
        rows.append(row)
        plant.set_targets(q_cmd if cal is None else {f: q_cmd[f] - cal["q"][f] for f in q_cmd})
        t += dt
        plant.advance(t)
        if film is not None and (len(rows) - 1) % 4 == 0:
            frames.append(film.frame(plant, f"{task}, {setup['shape']} {setup['d_mm']:g} mm, seed {seed}",
                                     f"{info['phase']:<10s} {t:4.2f} s  " + film_line(task, s, info)))
    wall = time.perf_counter() - w0
    res = metrics(task, setup, rows, s0, plant.state())
    res.update(wall_s=round(wall, 2), sat_ticks=n_sat, n_ticks=len(rows))
    return res, rows, frames


# The magnetic fingertips as measured on the UR5 with the HEX-E (docs/experiments/20261026-fingertip_calibration,
# 2026-10-22 to 10-29; four-magnet v3 tips, each with a model trained on its own presses): contact location 0.25-0.53 mm
# at 1.5 N and over, 0.59-0.75 mm under 1 N, 1.10-1.60 mm at 0.5 N and under; normal force within 5-6 %; noise-equivalent
# force 62-83 mN; contacts are shown above 0.25 N. Latency was not measured; the variants delay every measurement 40 ms.
TIP_SIGMA_MM = ((1.5, 0.4), (0.5, 0.7), (0.0, 1.3))     # (force band floor N, location error sigma mm)
TIP_GAIN, TIP_NOISE_N, TIP_BIAS_MM, TIP_F_MIN = 0.05, 0.07, 0.4, 0.25


def tip_errors(noise, rng):
    """Per rollout and finger: a force gain error and a location offset (the trained model's error at a place)."""
    k = noise["tip_model"]
    out = {}
    for f in FINGERS:
        d = rng.normal(size=3)
        out[f] = {"gain": rng.normal() * TIP_GAIN * k,
                  "offset": d / np.linalg.norm(d) * noise.get("cop_bias_mm", TIP_BIAS_MM * k) * 1e-3}
    return out


def tip_reading(cop, noise, rng, tip):
    """What a fingertip reports for each tip contact (centre of pressure, normal force): the force with its gain error,
    a bias and noise, dropped below the detection floor; the point with its offset and noise that grows at light
    force. Errors scale with noise['tip_model']."""
    k = noise["tip_model"]
    out = {}
    for f, (p, F) in cop.items():
        Fm = F * (1.0 + tip[f]["gain"]) + noise.get("f_bias", 0.0) + rng.normal() * TIP_NOISE_N * k
        if Fm < noise.get("f_min", TIP_F_MIN):
            continue
        sig = next(sg for lo, sg in TIP_SIGMA_MM if F >= lo) * k * 1e-3
        out[f] = (p + tip[f]["offset"] + rng.normal(size=3) * sig, Fm)
    return out


def pinch_angle(axis) -> float:
    """Angle (deg) of the cylinder's axis about palm x (the pinch axis), + when the middle's side (the axis' +z end,
    which points to palm -y) is up."""
    return math.degrees(math.atan2(axis[2], -axis[1]))


def film_line(task, s, info):
    F = " ".join(f"{f[0].upper()} {s['cop'].get(f, (None, 0.0))[1]:.1f}" for f in FINGERS)
    if task in ("pinpush", "press"):
        th = pinch_angle(H.quat_to_mat(s["quat"])[:, 2])
        ref = info.get("theta_ref")
        return f"{th:+5.1f} deg" + (f" (cmd {ref:+5.1f})" if ref is not None else "") + f"  {F} N"
    return f"w {np.degrees(s['w']).round(0)} deg/s  {F} N"


def metrics(task, setup, rows, s0, se):
    """Scores of one rollout (see the module docstring)."""
    out = {}
    z0 = rows[0]["pos_mm"][2]
    loaded = sum(1 for f in FINGERS if se["Ftot"][f] >= 0.025 * 9.81 / 2)
    out["dropped"] = bool(se["pos"][2] * 1e3 < z0 - 15.0)
    out["fingers_end"] = loaded
    out["held"] = (not out["dropped"]) and loaded >= 2
    out["link_force_max_N"] = round(max(max(r["F_link"].values()) for r in rows), 3) if rows and "F_link" in rows[0] else None
    if task == "expa":
        V = np.array([r["v_ref"] for r in rows])
        Ach = np.array([r["v_ach"] for r in rows])
        scale = np.array([180 / math.pi] * 3 + [1e3] * 3)               # deg/s and mm/s
        err = (Ach - V) * scale
        per = {}
        for k, elem in enumerate(EXPA_ORDER):
            on = np.abs(V[:, elem]) > 0
            if not on.any():
                continue
            mag = 40.0 if elem < 3 else 20.0
            per[H.COMPONENTS[elem]] = {
                "rmse_ratio": round(float(np.sqrt(np.mean(err[on, elem] ** 2)) / mag), 3),
                "gain": round(float(np.median(Ach[on, elem] / V[on, elem])), 3),
                "crosstalk_ang_dps": round(float(np.sqrt(np.mean(np.delete(err[on, :3], elem, 1) ** 2))) if elem < 3
                                           else float(np.sqrt(np.mean(err[on, :3] ** 2))), 2),
                "crosstalk_lin_mms": round(float(np.sqrt(np.mean(np.delete(err[on, 3:], elem - 3, 1) ** 2))) if elem >= 3
                                           else float(np.sqrt(np.mean(err[on, 3:] ** 2))), 2)}
        out["elements"] = per
        out["min_gap_mm"] = round(float(min(r["gap_mm"] for r in rows)), 2)
        out["held"] = True
        return out
    task_rows = [r for r in rows if r["t"] >= T_LIFT + 0.4 - 1e-9]
    if not task_rows or s0 is None:
        return out
    # the rows' own period: angles and the centre's reference path are integrated with it (before 2026-10-29 evening a
    # fixed 0.01 s, which halved them at 50 Hz; rescale_metrics corrects rows scored then)
    h = round(task_rows[1]["t"] - task_rows[0]["t"], 6) if len(task_rows) > 1 else DT_C
    out["metrics_dt"] = h
    p0 = np.array(task_rows[0]["pos_mm"])
    a0 = np.array(task_rows[0]["axis"])
    if task in ("pinpush", "press"):
        A = np.array([r["axis"] for r in task_rows])
        th = np.array([pinch_angle(a_) for a_ in A])
        th -= th[0]
        ref = np.array([r.get("theta_ref_deg", 0.0) for r in task_rows])
        out["theta_rmse_deg"] = round(float(np.sqrt(np.mean((th - ref) ** 2))), 2)
        out["theta_max_deg"], out["theta_min_deg"] = round(float(th.max()), 2), round(float(th.min()), 2)
        strokes = []
        names = [r["phase"] for r in task_rows]
        for name in dict.fromkeys(names):
            idx = [i for i, n in enumerate(names) if n == name]
            dref = ref[idx[-1]] - (ref[idx[0] - 1] if idx[0] > 0 else 0.0)
            if abs(dref) < 1e-6:
                continue
            dth = th[idx[-1]] - (th[idx[0] - 1] if idx[0] > 0 else 0.0)
            strokes.append({"stroke": name, "ref_deg": round(float(dref), 2), "angle_deg": round(float(dth), 2),
                            "gain": round(float(dth / dref), 3)})
        out["strokes"] = strokes
        out["gain_min"] = round(min(s_["gain"] for s_ in strokes), 3) if strokes else None
        # rotation off the pinch axis: the axis leaving the palm y-z plane
        out["off_axis_max_deg"] = round(float(np.degrees(np.abs(np.arcsin(np.clip(A[:, 0], -1, 1)) -
                                                               math.asin(max(-1.0, min(1.0, A[0, 0]))))).max()), 2)
        def path(f):                                     # displacement of finger f's centre of pressure on the object
            P = [np.array(r["cop_obj_mm"][f]) for r in task_rows if f in r["cop_obj_mm"]]
            return float(max(np.linalg.norm(q_ - P[0]) for q_ in P)) if P else None
        sl = path(PUSHER)
        out["pusher_slip_max_mm"] = round(sl, 2) if sl is not None else None
        pinch = [0.5 * (np.array(r["cop_obj_mm"][PINCH[0]]) + np.array(r["cop_obj_mm"][PINCH[1]]))[2]
                 for r in task_rows if all(f in r["cop_obj_mm"] for f in PINCH)]
        out["pinch_slide_max_mm"] = round(float(np.abs(np.array(pinch) - pinch[0]).max()), 2) if pinch else None
        out["pusher_contact_frac"] = round(float(np.mean([r["F"][PUSHER] > 0.02 for r in task_rows])), 3)
        out["F_mean"] = {f: round(float(np.mean([r["F"][f] for r in task_rows])), 3) for f in FINGERS}
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
            ang = float(np.sum(W[idx, ax]) * h)
            ref = float(np.sum(Wr[idx, ax]) * h)
            steps.append({"step": ph, "angle_deg": round(ang, 2), "ref_deg": round(ref, 2),
                          "gain": round(ang / ref, 3) if ref else None})
        out["steps"] = steps
        P = np.array([r["pos_mm"] for r in task_rows])
        out["drift_max_mm"] = round(float(np.linalg.norm(P - P[0], axis=1).max()), 2)
        Vr = np.array([r.get("v_ref_mms", [0.0, 0.0, 0.0]) for r in task_rows])
        Pr = P[0] + np.vstack([np.zeros(3), np.cumsum(Vr[:-1], 0) * h])
        out["centre_err_max_mm"] = round(float(np.linalg.norm(P - Pr, axis=1).max()), 2)
        out["centre_travel_ref_mm"] = round(float(np.linalg.norm(Pr - P[0], axis=1).max()), 2)
        if task == "twist":
            ramp = [i for i, r in enumerate(task_rows) if r["phase"] in ("twist ramp", "hold")]
            if ramp:
                yaw = np.cumsum(W[ramp, 2]) * h
                ok = [i for i in ramp if sum(1 for f in FINGERS if task_rows[i]["F"][f] > 0.05) == 3]
                out["ramp_max_deg"] = round(float(yaw.max()), 2)
                out["ramp_three_pads_max_deg"] = round(float(np.cumsum(W[ramp, 2])[[ramp.index(i) for i in ok]].max()
                                                             * h), 2) if ok else 0.0
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
        if GRASP[setup["task"]] == "pinch_y":         # from the thumb's side, 20 deg off the pinch axis: the tilt
            self.cam.azimuth, self.cam.elevation = 340.0, 10.0
        else:
            self.cam.azimuth, self.cam.elevation = 90.0, 5.0

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
    "si": ({"ell": 1.0}, None),                       # Eq. (2) in SI units with equal weights (rad/s against m/s)
    "gapq": ({"impedance": "gap"}, None),             # the impedance gap at the measured angles, while apart (before 10-22)
    "surface": ({"impedance": "surface"}, None),      # ... and also in contact
    "floop": ({"force_loop_gain": 0.01}, None),       # v_sep += 10 mm/s per N of force error (needs fingertip force)
    "reanchor": ({"reanchor": True}, None),           # a touching finger's reference re-anchored beyond the gap cut
    "hold": ({}, None),                               # the object held at its start until the forces come on
    # 2026-10-29, the fingertip sensing review (docs/experiments/20261029-hom_control_review): 50 Hz, the tag tracker
    # as in "sense" (0.5 mm, 1 deg, every measurement 40 ms late), and the fingertips absent or as measured
    "notip0": ({"dt": 0.02}, {"no_tip": True}),                            # no fingertips, exact object pose
    "notip": ({"dt": 0.02}, {"pos_mm": 0.5, "rot_deg": 1.0, "delay_s": 0.04, "no_tip": True}),
    "tip": ({"dt": 0.02, "contact_force_min": TIP_F_MIN},
            {"pos_mm": 0.5, "rot_deg": 1.0, "delay_s": 0.04, "tip_model": 1.0}),
    "tip3": ({"dt": 0.02, "contact_force_min": TIP_F_MIN},
             {"pos_mm": 0.5, "rot_deg": 1.0, "delay_s": 0.04, "tip_model": 3.0}),
    "tipbias": ({"dt": 0.02, "contact_force_min": TIP_F_MIN},            # the press-step residual and an untrained model
                {"pos_mm": 0.5, "rot_deg": 1.0, "delay_s": 0.04, "tip_model": 1.0, "f_bias": 0.3, "cop_bias_mm": 1.5}),
    "tipfl": ({"dt": 0.02, "contact_force_min": TIP_F_MIN, "force_loop_gain": 0.01},
              {"pos_mm": 0.5, "rot_deg": 1.0, "delay_s": 0.04, "tip_model": 1.0}),
    "tipflb": ({"dt": 0.02, "contact_force_min": TIP_F_MIN, "force_loop_gain": 0.01},
               {"pos_mm": 0.5, "rot_deg": 1.0, "delay_s": 0.04, "tip_model": 1.0, "f_bias": 0.3}),
    # ... and with calibration errors: each joint off by sigma 1 deg (reported and executed), the object frame off by
    # sigma 1 mm per axis; without fingertips, with them, and with them closing a force loop
    "cal_notip": ({"dt": 0.02}, {"pos_mm": 0.5, "rot_deg": 1.0, "delay_s": 0.04, "no_tip": True, "q_bias_deg": 1.0,
                                 "pos_bias_mm": 1.0}),
    "cal_tip": ({"dt": 0.02, "contact_force_min": TIP_F_MIN},
                {"pos_mm": 0.5, "rot_deg": 1.0, "delay_s": 0.04, "tip_model": 1.0, "q_bias_deg": 1.0, "pos_bias_mm": 1.0}),
    "cal_tipfl": ({"dt": 0.02, "contact_force_min": TIP_F_MIN, "force_loop_gain": 0.01},
                  {"pos_mm": 0.5, "rot_deg": 1.0, "delay_s": 0.04, "tip_model": 1.0, "q_bias_deg": 1.0,
                   "pos_bias_mm": 1.0}),
    "ell5": ({"ell": 0.005}, None),
    "ell100": ({"ell": 0.1}, None),
    "l30": ({}, None),                                # pin-push on the grasps of design --lever 30 / 60
    "l60": ({}, None),
}
LEVER_VARIANTS = {"l30": 30.0, "l60": 60.0}           # variant -> design --lever (mm)
PLANT_VARIANTS = {"hold": {"hold_s": T_CLOSE}}        # variant -> plant settings (Plant.hold_s)


def cmd_run(a) -> int:
    import mujoco
    ROWS.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        path = (OUT / f"{a.rows_file}.jsonl") if a.rows_file else ROWS / f"{task}.jsonl"
        done = _done(path)
        for d_mm in (a.d or SIZES[task]):
            for var in a.variants:
                vname = var + (f"_{a.tag}" if a.tag else "")
                todo = [sd for sd in a.seeds if (task, float(d_mm), sd, vname) not in done]
                if not todo and not a.film:
                    continue
                try:
                    p_pads, p_pt, setup = build(task, float(d_mm), lever=LEVER_VARIANTS.get(var))
                except Exception as e:
                    print(task, d_mm, var, "build failed:", e, flush=True)
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
                           "design": {k: setup["design"][k] for k in ("tag", "h", "score", "lever_mm")
                                      if k in setup["design"]},
                           "prm": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PRM.items()},
                           "hom": {k: v for k, v in vars(hprm).items() if k != "filter"},
                           "filter": vars(hprm.filter), "when": time.strftime("%Y-%m-%d %H:%M")}
                    try:
                        res, trace, frames = rollout(task, setup, plant, mirror, q_open, seed, PRM, hprm, film, noise,
                                                     PLANT_VARIANTS.get(var, {}).get("hold_s", 0.0))
                        row.update(res, status="ok")
                        if a.traces or film is not None:
                            row["trace"] = trace[::2]
                        if film is not None and frames:
                            import hom_contact_rig as HR
                            fp = MEDIA / f"20261021-hom_{task}_{setup['shape']}{d_mm:g}_{vname}.mp4"
                            HR.write_mp4(frames, fp, fps=25)
                            row["film"] = str(fp.relative_to(ROOT))
                    except Exception as e:
                        row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
                    if (task, float(d_mm), seed, vname) not in done or film is not None:
                        _append(path, row)
                    brief = {k: row.get(k) for k in ("held", "theta_rmse_deg", "gain_min", "pusher_slip_max_mm",
                                                     "w_rmse_ratio", "ramp_three_pads_max_deg", "centre_err_max_mm",
                                                     "drift_max_mm", "wall_s", "error")
                             if row.get(k) is not None}
                    print(task, d_mm, var, seed, brief, flush=True)
    return 0


def cmd_filter_sweep(a) -> int:
    """The estimator's settings on the sphere tasks: the filter's source of the reference geodesic velocity (the
    paper's reference relative velocities, zero when a task commands the object twist, against the twist Eq. (2)
    plans), its largest gain k_max, and the controller period. Rows: OUT/filter_sweep.jsonl (fsynced, resumable)."""
    import mujoco
    path = OUT / "filter_sweep.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task"], r["d_mm"], r["source"], r["k_max"], r["dt"], r["seed"]))
    for task in a.tasks:
        for d_mm in (a.d or SIZES[task]):
            p_pads, p_pt, setup = build(task, float(d_mm))
            m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
            S.set_palm(m_pt, setup["palm"])
            fs, os_ = sr2_specs(setup)
            mirror = H.Mirror(m_pt, fs, os_)
            q_open, _ = open_pose(setup, mujoco.MjModel.from_xml_path(str(p_pt)))
            plant = Plant(p_pads, setup)
            for src in ("reference", "planned"):
                for kmax in (100.0, 25.0):
                    for dt in (0.01, 0.02):
                        for seed in a.seeds:
                            key = (task, float(d_mm), src, kmax, dt, seed)
                            if key in done:
                                continue
                            hp = H.HomParams(dt=dt, filter=H.FilterParams(k_max=kmax, source=src))
                            row = {"task": task, "d_mm": float(d_mm), "source": src, "k_max": kmax, "dt": dt, "seed": seed}
                            try:
                                res, _, _ = rollout(task, setup, plant, mirror, q_open, seed, PRM, hp)
                                row.update({k: res.get(k) for k in ("held", "w_rmse_ratio", "drift_max_mm", "steps",
                                                                     "ramp_three_pads_max_deg")}, status="ok")
                            except Exception as e:
                                row.update(status="error", error=f"{type(e).__name__}: {e}")
                            _append(path, row)
                            print(key, row.get("held"), row.get("w_rmse_ratio"), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("filter-sweep")
    p.add_argument("--tasks", nargs="+", default=["roll", "twist"])
    p.add_argument("--d", nargs="+", type=float)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p = sub.add_parser("design")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--d", nargs="+", type=float)
    p.add_argument("--max-try", type=int, default=400)
    p.add_argument("--lever", type=float, default=None,
                   help="pinch grasps with the middle lever to lever + 5 mm from the pinch (mm); keys <task>_l<lever>_...")
    p = sub.add_parser("run")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--d", nargs="+", type=float)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--variants", nargs="+", default=["base"], choices=sorted(VARIANTS))
    p.add_argument("--film", action="store_true")
    p.add_argument("--traces", action="store_true")
    p.add_argument("--set", nargs="*", default=[], help="task parameters k=v (PRM), recorded in each row's variant")
    p.add_argument("--tag", default="", help="suffix of the variant name for rows run with --set")
    p.add_argument("--rows-file", default="", help="write rows to OUT/<name>.jsonl instead of rows/<task>.jsonl")
    a = ap.parse_args()
    if a.cmd == "run":
        for kv in a.set:
            k, v = kv.split("=")
            PRM[k] = type(PRM[k])(float(v)) if not isinstance(PRM[k], tuple) else tuple(float(x) for x in v.split(","))
    if a.cmd == "filter-sweep":
        return cmd_filter_sweep(a)
    return cmd_design(a) if a.cmd == "design" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
