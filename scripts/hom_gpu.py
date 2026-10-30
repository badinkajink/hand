#!/usr/bin/env python3
"""HOM control of the four tasks of scripts/hom_tasks.py (pin-push, press, roll, twist) on MuJoCo-Warp: many hand
layouts, seeds and sensing variants as parallel GPU worlds, with the controller of src/morphohand/control/hom.py batched
in torch on the GPU (owner, 2026-10-29: "can we leverage GPU acceleration for higher throughput evaluation, since our
mjwarp sphere packed model is ok enough?"; the same evening: port the pin-push, the press and the fingertip-sensing
variants).

The scale study's layouts compile to models that differ only in the positions of the three mount bodies (checked
field by field, 2026-10-29), so one MuJoCo-Warp model serves every layout of one object: body_pos gets one row per world
with that world's mounts, and the palm and the post are mocap bodies (MuJoCo-Warp computes the poses of geoms welded to
the world once, in make_data). One batch runs one task on one object size at one control period; each world is one
(layout and grasp, seed, sensing variant). The grasps (--layouts):
  design   the task study's grasp for the task and size (docs/experiments/20261021-hom_framework/design.json);
  2d       the symmetric tripods of the toy's 2-D case with the task study's equator grasp (roll and twist);
  2dfp, rrt6d, sobol8k   the toy's layouts (scripts/fp_sr2_toy.py; rows rrt2d, rrt6d, sobol8k) from their
           foundational poses: the tripod pose for the roll and the twist (40 mm sphere), the pinch pose for the
           pin-push and the press pose for the press (25 mm cylinder);
  sobol8k_size   every Sobol layout that reaches the task's pose on the object of --d (fp_sr2_toy.py poses);
  grid2d_size    the 2-D grid of symmetric tripods with every grasp of each layout's grasp set (fp_sr2_toy.py grid2d).

The controller is hom.HomController.step for every world and finger, in torch:
  - the measurement: the plant's joint angles, object pose and per-finger centre of pressure, delayed, offset and
    noised as hom_tasks.rollout does for the variant (hom_tasks.VARIANTS). The draws made once per rollout (calibration
    offsets, each fingertip's force gain and location offset) are the CPU's: numpy's generator 1000 + seed, in the CPU's
    order. The draws of every control period come from a torch generator;
  - a second MuJoCo-Warp data, the mirror, holds the measured state; its kinematics give the tip poses, the joint axes
    and anchors, and the gravity torques, as hom.Mirror does on the CPU;
  - the GCF from the measured centre of pressure at contact_force_min or more, else from the witness points of the
    rounded TPU block and the object: analytic for the sphere, alternating projections between the block's core box and
    the cylinder (the CPU uses mj_geomDistance on the block's convex mesh);
  - the pinch frame {M} from the pinching fingers' contact points on the object when the pusher starts, and Eq. (3)'s
    twist for the pusher;
  - Eq. (1) from the joint axes and anchors; Eq. (2) solved exactly by enumerating the 27 active-bound patterns of the
    three joint rates (the CPU uses SciPy's BVLS on the same problem); the force loop on v_sep and the anti-windup;
  - the complementary filter at k_min: every commanded roll and slide of these tasks is zero, so Eqs. (4)-(6) reduce to
    a low-pass of the contact point on the block's surface, stepped along the tangent and projected back;
  - the servo impedance in the 'reference' mode with gravity; the plant's servos execute the targets less their
    calibration offsets.
The physics differs from the CPU runner in float32, a solver tolerance of 1e-6 with 20 iterations, and pads that
collide only with the object. Scores come from hom_tasks.metrics on the same per-period rows, so they compare directly.

  .venv/bin/python scripts/hom_gpu.py run --task twist --layouts 2d --seeds 0 1 2 --out hom2d_gpu_twist
  .venv/bin/python scripts/hom_gpu.py run --task pinpush press --layouts design --d 10 16 25 40 \\
        --variants rate50 notip0 tip cal_tip --seeds 0 1 2 --out sensing_gpu --rows-dir <dir>
  .venv/bin/python scripts/hom_gpu.py run --task press --layouts sobol8k --seeds 0 1 2 --out hom6d_gpu_press
  .venv/bin/python scripts/hom_gpu.py compare --cpu <cpu rows> --gpu <gpu rows>
Rows: docs/experiments/20261029-foundational_pose_review/rows/<out>.jsonl unless --rows-dir is given.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")     # the v4b fingertip (2026-10-30); cad_v1: the TPU block

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402  (module level: kernel annotations resolve against these globals)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import fp_hom_check as C  # noqa: E402
import hom_tasks as T  # noqa: E402

S, H, K, G = T.S, T.H, T.K, T.G
FINGERS, JOINTS = T.FINGERS, T.JOINTS
assert tuple(FINGERS) == ("thumb", "index", "middle") and T.PINCH == ("thumb", "index") and T.PUSHER == "middle"
TASKS = ("pinpush", "press", "roll", "twist")
D_FP = {"pinpush": 25.0, "press": 25.0, "roll": 40.0, "twist": 40.0}   # the toy's objects
REF = (92.5, 102.5)                    # the layout whose scenes are compiled (the task study's 40 mm sphere grasp)
NCONMAX = int(os.environ.get("HOM_GPU_NCONMAX", 1024))  # contact and constraint rows per world; the press with
NJMAX = int(os.environ.get("HOM_GPU_NJMAX", 2048))      # the v4b pads overflows 2,048 rows in a few steps (10-30)
ROWS = C.ROWS
N_AP = 24                              # alternating projections for the block-cylinder witness points
N_AP_WARM = 4                          # ... per control period for the v4b pad, from the previous period's point
DMAX = 4                               # longest measurement delay, control periods
OVER_OK = {"dt", "contact_force_min", "force_loop_gain", "reanchor"}    # the HomParams a variant may change here
FP_ROWS = {"2dfp": "rrt2d", "rrt6d": "rrt6d", "sobol8k": "sobol8k", "sobol6d": "sobol6d"}


@wp.kernel
def finger_forces(nacon: wp.array(dtype=int), worldid: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
                  pos: wp.array(dtype=wp.vec3), efc_addr: wp.array2d(dtype=int), efc_force: wp.array2d(dtype=float),
                  objmask: wp.array(dtype=int), fidx: wp.array(dtype=int), tidx: wp.array(dtype=int),
                  ftip: wp.array2d(dtype=float), ftot: wp.array2d(dtype=float), cops: wp.array3d(dtype=float)):
    """Per world and finger: the normal force of all the finger's contacts with the object, and of its tip pads' with
    their force-weighted contact positions (hom_tasks.Plant.contacts)."""
    i = wp.tid()
    if i >= nacon[0]:
        return
    g = geom[i]
    o1 = objmask[g[0]]
    o2 = objmask[g[1]]
    if o1 == o2:
        return
    other = g[0]
    if o1 == 0:
        other = g[0]
    else:
        other = g[1]
    k = fidx[other]
    if k < 0:
        return
    adr = efc_addr[i, 0]
    if adr < 0:
        return
    w = worldid[i]
    f = efc_force[w, adr]
    if f <= 0.0:
        return
    wp.atomic_add(ftot, w, k, f)
    if tidx[other] >= 0:
        p = pos[i]
        wp.atomic_add(ftip, w, k, f)
        wp.atomic_add(cops, w, k, 0, f * p[0])
        wp.atomic_add(cops, w, k, 1, f * p[1])
        wp.atomic_add(cops, w, k, 2, f * p[2])


# ------------------------------------------------------------------------------------------------ layouts and grasps

def mount_xy(d_mm):
    """Palm-frame (x, y) mount positions (m) of a six-coordinate layout (mm)."""
    d = np.asarray(d_mm, float) / 1000.0
    return {"thumb": d[0:2], "index": d[2:4], "middle": d[4:6]}


def sym6(xs, ys):
    """The six coordinates (mm) of a symmetric tripod (hand_object_scale_kin.mounts)."""
    return [-xs / 2, 0.0, xs / 2, ys / 2, xs / 2, -ys / 2]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def tripod_setup(task, mounts_xy, grasp, d_mm=40.0):
    """hom_tasks' setup dict for an equator-type sphere grasp: grasp = (centre in the palm frame, {f: outward normal})."""
    centre, n_out = grasp
    r = d_mm / 2000.0
    rr = r + K.R                                   # the task study's touching grasp (open_pose adds 8 mm)
    tg = {f: centre + rr * n_out[f] for f in FINGERS}
    palm = np.array([-centre[0], -centre[1], S.Z_OBJ - centre[2]])
    mounts = {f: np.array([mounts_xy[f][0], mounts_xy[f][1], 0.0]) for f in FINGERS}
    return {"task": task, "shape": "sphere", "d_mm": d_mm, "r": r, "hl": 0.0, "palm": palm,
            "quat": np.array(T.SPHERE_Q), "design": {}, "targets": tg, "n_out": n_out, "mounts": mounts, "psi": 0.0}


def pinch_setup(task, v, d6):
    """hom_tasks' setup dict for the toy's pinch or press pose v of a 25 mm cylinder on layout d6 (mm). The cylinder's
    axis is palm y yawed by psi = v[3]; its body z points to the middle's side, as in the task study's grasp."""
    import fp_sr2_toy as FP
    M = FP.mounts6(d6)
    obj, tg, n, _pc, _ax = FP.geometry("pinch", np.asarray(v, float), M)
    c, psi = np.asarray(obj["centre"], float), float(v[3])
    palm = np.array([-c[0], -c[1], S.Z_OBJ - c[2]])
    quat = quat_mul([math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)], T.SPHERE_Q)
    return {"task": task, "shape": "cylinder", "d_mm": FP.CYL_D * 1000, "r": FP.CYL_D / 2, "hl": FP.CYL_L / 2,
            "palm": palm, "quat": quat, "design": {}, "targets": tg, "n_out": {f: np.asarray(n[f], float) for f in FINGERS},
            "mounts": M, "psi": psi}


def grasp_2d(xs, ys):
    """The task study's equator grasp on a symmetric tripod (hom_tasks.design_sphere's rule)."""
    cell = C.cell_of(xs, ys)
    des = C.design(cell)
    if des is None:
        return None, None
    gg = S.grasp_geometry(cell, "sphere", C.D_MM, des["h"], des["spread"])
    n_out = {f: np.asarray(gg["n_out"][f], float) for f in FINGERS}
    return (np.asarray(gg["centre_palm"], float), n_out), des


def grasp_fp(row):
    """The tripod foundational pose that fp_sr2_toy.py solved for a layout (sphere centre and contact normals)."""
    import fp_sr2_toy as FP
    rec = row["fp"].get("tripod", {})
    if not rec.get("ok"):
        return None
    M = FP.mounts6(row["d"])
    obj, _tg, n, _pc, _ax = FP.geometry("tripod", np.asarray(rec["v"], float), M)
    return np.asarray(obj["centre"], float), {f: np.asarray(n[f], float) for f in FINGERS}


POSE = {"roll": "tripod", "twist": "tripod", "pinpush": "pinch", "press": "press"}


def size_layouts(task, src, nmax, d_mm):
    """The layouts of rows/<src>.jsonl that reach the task's pose on its object of diameter d_mm, from the toy's rows of
    that size (fp_sr2_toy.size_name: `poses` for the Sobol layouts, `grid2d` for the 2-D grid). A pose with a grasp set
    (fp_sr2_toy.grasp_set) gives one entry per grasp, `grasp` its index (0, the foundational pose)."""
    import fp_sr2_toy as FP
    shape = T.SHAPE[task]
    FP.set_sizes(**({"sph_d_mm": d_mm} if shape == "sphere" else {"cyl_d_mm": d_mm}))
    path = ROWS / f"{FP.size_name(src, shape, d_mm)}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path.name}: run fp_sr2_toy.py poses or grid2d --shape {shape} --d {d_mm:g} first")
    out = []
    for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip()):
        rec = r["fp"].get(POSE[task]) or {}
        if not rec.get("ok"):
            continue
        mxy = mount_xy(r["d"])
        for j, g in enumerate(rec.get("grasps") or [rec]):
            key = {"d": r["d"], "call": r["call"], "kin": {k: g[k] for k in ("tmr_deg", "mft_mNm", "smin_mm")}}
            if "grasps" in rec:
                key.update(grasp=j, gsv=g.get("gsv"))
            if shape == "sphere":
                obj, _tg, n, _pc, _ax = FP.geometry("tripod", np.asarray(g["v"], float), FP.mounts6(r["d"]))
                setup = tripod_setup(task, mxy, (np.asarray(obj["centre"], float),
                                                 {f: np.asarray(n[f], float) for f in FINGERS}), d_mm)
            else:
                setup = pinch_setup(task, g["v"], r["d"])
            out.append((key, mxy, setup))
        if nmax and len({k["call"] for k, _, _ in out}) >= nmax:
            break
    return out


def layout_list(task, kind, nmax, d_mm=None):
    """[(key, mounts xy, setup)] for the requested layouts and task. A toy set with the suffix _size (sobol8k_size)
    takes the poses solved for the object of diameter d_mm; the plain toy sets hold the poses of D_FP only."""
    if kind.endswith("_size"):
        return size_layouts(task, kind[:-len("_size")], nmax, d_mm)
    if K.HAND == "cad_v2":
        raise ValueError(f"--layouts {kind} holds the TPU block's grasps; with the v4b tip use sobol8k_size or grid2d_size")
    if kind in FP_ROWS and d_mm is not None and float(d_mm) != D_FP[task]:
        raise ValueError(f"--layouts {kind} holds poses of the {D_FP[task]:g} mm object; use {kind}_size for {d_mm:g} mm")
    out = []
    if kind == "design":
        _pp, _pt, setup = T.build(task, float(d_mm))
        des = setup["design"]
        xs, ys = des["x_sep_mm"], des["y_sep_mm"]
        setup["psi"] = 0.0
        return [({"design": T.dkey(task, float(d_mm)), "x_sep_mm": xs, "y_sep_mm": ys}, mount_xy(sym6(xs, ys)), setup)]
    if kind == "2d":
        if T.SHAPE[task] != "sphere":
            raise ValueError("the 2d layouts carry the equator grasp only; use 2dfp for the pin-push and the press")
        for xs, ys, kin in C.pick_layouts(0, everyone=True):
            grasp, _des = grasp_2d(xs, ys)
            if grasp is None:
                continue
            mxy = mount_xy(sym6(xs, ys))
            out.append(({"x_sep_mm": xs, "y_sep_mm": ys, "kin": {k: kin[k] for k in ("tmr_deg", "mft_mNm", "smin_mm")}},
                        mxy, tripod_setup(task, mxy, grasp)))
            if nmax and len(out) >= nmax:
                break
        return out
    name = FP_ROWS[kind]
    rows = [json.loads(x) for x in (ROWS / f"{name}.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("ok")]
    press = None
    if task == "press":
        pr = [json.loads(x) for x in (ROWS / f"{name}_press.jsonl").read_text().splitlines() if x.strip()]
        press = {tuple(r["d"]): r["fp"]["press"] for r in pr}
    for r in rows:
        kin = {k: r["fp"]["tripod"][k] for k in ("tmr_deg", "mft_mNm", "smin_mm")}
        kin_p = {k: r["fp"]["pinch"][k] for k in ("tmr_deg", "mft_mNm", "smin_mm")}
        key = {"d": r["d"], "kin": kin, "kin_pinch": kin_p}
        mxy = mount_xy(r["d"])
        if task in ("roll", "twist"):
            g = grasp_fp(r)
            if g is None:
                continue
            setup = tripod_setup(task, mxy, g)
        elif task == "pinpush":
            setup = pinch_setup(task, r["fp"]["pinch"]["v"], r["d"])
        else:
            rec = press.get(tuple(r["d"]))
            if not rec or not rec.get("ok"):
                continue
            key["kin_press"] = {k: rec[k] for k in ("tmr_deg", "mft_mNm", "smin_mm")}
            setup = pinch_setup(task, rec["v"], r["d"])
        out.append((key, mxy, setup))
        if nmax and len(out) >= nmax:
            break
    return out


class Scene:
    """The reference layout's scenes for one object (shape, size): the pads plant, and the pt model whose mount bodies
    are moved to each layout to find the open pose (hom_tasks.open_pose)."""

    def __init__(self, shape, d_mm):
        import mujoco
        self.shape, self.d_mm = shape, float(d_mm)
        self.r = self.d_mm / 2000.0
        self.hl = T.CYL_LEN / 2 if shape == "cylinder" else 0.0
        sc = S.cell_scenes(dict(C.cell_of(*REF)), shape, self.d_mm, models=("pads", "pt"))
        self.pads_path, self.pt_path = Path(sc["pads"][0]), Path(sc["pt"][0])
        self.m_pt = mujoco.MjModel.from_xml_path(str(self.pt_path))
        ob = self.m_pt.body(S.OBJ).id
        og = int(np.flatnonzero(self.m_pt.geom_bodyid == ob)[0])
        want = [self.r] if shape == "sphere" else [self.r, self.hl]
        if not np.allclose(self.m_pt.geom_size[og][:len(want)], want, atol=1e-6):
            raise RuntimeError(f"object size {self.m_pt.geom_size[og]} is not {want}")
        self.mount_bid = {f: self.m_pt.body(f"{f}_mount").id for f in FINGERS}
        ref_xy = mount_xy(sym6(*REF))
        self.base = {f: self.m_pt.body_pos[self.mount_bid[f]].copy() - np.array([*ref_xy[f], 0.0]) for f in FINGERS}

    def body_mounts(self, mxy):
        return {f: self.base[f] + np.array([mxy[f][0], mxy[f][1], 0.0]) for f in FINGERS}

    def prepare(self, mxy, setup):
        bm = self.body_mounts(mxy)
        for f in FINGERS:
            self.m_pt.body_pos[self.mount_bid[f]] = bm[f]
        q_open, _q_touch = T.open_pose(setup, self.m_pt)
        return bm, q_open


def check_mounts():
    """The mount positions built from offsets against a scene compiled for another layout."""
    import mujoco
    L = Scene("sphere", 40.0)
    for xs, ys in ((70.0, 80.0), (115.0, 140.0)):
        sc = S.cell_scenes(dict(C.cell_of(xs, ys)), "sphere", 40.0, models=("pt",))
        m = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
        bm = L.body_mounts(mount_xy(sym6(xs, ys)))
        err = max(float(np.abs(m.body_pos[L.mount_bid[f]] - bm[f]).max()) for f in FINGERS)
        print(f"layout {xs} x {ys}: mount body_pos error {err:.2e} m")


# ------------------------------------------------------------------------------------------------ variants

def variant_world(var, seed, dt_default=T.DT_C):
    """Per-world sensing parameters of hom_tasks.VARIANTS[var], with the draws hom_tasks.rollout makes once per rollout
    (numpy generator 1000 + seed, in its order: calibration offsets, the first period's pose noise, the fingertips'
    gains and offsets)."""
    over, noise = T.VARIANTS[var]
    bad = set(over) - OVER_OK
    if bad:
        raise ValueError(f"variant {var}: {sorted(bad)} not supported on the GPU")
    dt = over.get("dt", dt_default)
    hp = H.HomParams()
    nz = noise or {}
    if noise is None:
        mode = 0                                           # exact measurement
    elif nz.get("no_tip"):
        mode = 1                                           # no fingertips: witness points only
    elif nz.get("tip_model"):
        mode = 2                                           # the measured fingertips
    else:
        mode = 3                                           # Gaussian noise on the centre of pressure and force
    p = {"var": var, "dt": dt, "cfmin": over.get("contact_force_min", hp.contact_force_min),
         "flg": over.get("force_loop_gain", hp.force_loop_gain), "reanchor": float(over.get("reanchor", hp.reanchor)),
         "hold": float(T.PLANT_VARIANTS.get(var, {}).get("hold_s", 0.0)),
         "mode": mode,
         "pos_mm": nz.get("pos_mm", 0.0), "rot_deg": nz.get("rot_deg", 0.0), "cop_mm": nz.get("cop_mm", 0.0),
         "f_N": nz.get("f_N", 0.0), "delay": int(round(nz.get("delay_s", 0.0) / dt)), "tip_k": float(nz.get("tip_model", 0.0)),
         "f_bias": nz.get("f_bias", 0.0), "f_min": nz.get("f_min", T.TIP_F_MIN)}
    if p["delay"] > DMAX:
        raise ValueError(f"variant {var}: delay of {p['delay']} periods > {DMAX}")
    cal_q, cal_pos, gain, off = np.zeros((3, 3)), np.zeros(3), np.zeros(3), np.zeros((3, 3))
    if noise is not None:
        rng = np.random.default_rng(1000 + seed)
        if nz.get("q_bias_deg") or nz.get("pos_bias_mm"):
            for k in range(3):
                cal_q[k] = np.radians(rng.normal(size=3) * nz.get("q_bias_deg", 0.0))
            cal_pos = rng.normal(size=3) * nz.get("pos_bias_mm", 0.0) * 1e-3
        rng.normal(size=3)                                  # the first period's position noise,
        rng.normal(size=3)                                  # rotation axis
        rng.normal()                                        # and angle
        if mode == 2:
            k_ = nz["tip_model"]
            for k in range(3):
                d = rng.normal(size=3)
                gain[k] = rng.normal() * T.TIP_GAIN * k_
                off[k] = d / np.linalg.norm(d) * nz.get("cop_bias_mm", T.TIP_BIAS_MM * k_) * 1e-3
    p.update(cal_q=cal_q, cal_pos=cal_pos, tip_gain=gain, tip_off=off)
    return p


# ------------------------------------------------------------------------------------------------ the batch

def mocap_model(path):
    """The scene compiled with the palm and the post as mocap bodies (both are welded to the world and have no joints,
    so the physics is unchanged; MuJoCo-Warp then recomputes their geoms' poses every step)."""
    import mujoco
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    for name in ("palm_pose", "post"):
        b = root.find(f".//worldbody/body[@name='{name}']")
        if b is None:
            raise RuntimeError(f"{name} is not a child of the world body")
        b.set("mocap", "true")
    tmp = Path(path).with_name(f"{Path(path).stem}.mocap.{os.getpid()}.xml")
    ET.ElementTree(root).write(tmp)
    try:
        return mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()


def finger_commands(task, n_out):
    """Per finger (thumb, index, middle): Eq. (2)'s weights, whether the object's reference twist enters Eq. (1),
    whether Eq. (3)'s pusher twist does, and the commanded force after the close (hom_tasks.Schedule)."""
    P = T.PRM
    if task in ("pinpush", "press"):
        wpin = [0.0, P["w_roll_pinch"], P["w_roll_pinch"], 1.0, 1.0, 1.0]
        wpu = [1.0, P["w_roll_push"], P["w_roll_push"], 1.0, 1.0, 1.0]
        fpu = P["f_drive" if task == "pinpush" else "f_press"]
        f_thumb = P["f_pinch"] + fpu * max(0.0, -float(n_out["middle"] @ n_out["thumb"]))
        return (np.array([wpin, wpin, wpu]), np.array([False, False, True]), np.array([False, False, True]),
                np.array([f_thumb, P["f_pinch"], fpu]))
    return (np.tile([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], (3, 1)), np.ones(3, bool), np.zeros(3, bool),
            np.full(3, P["f_tripod"]))


def solve3(Mm, rhs):
    """Batched 3 x 3 linear solves by the adjugate: no host synchronisation, so a control period can be captured."""
    import torch as th
    a, b, c = Mm[..., 0, 0], Mm[..., 0, 1], Mm[..., 0, 2]
    d, e, f = Mm[..., 1, 0], Mm[..., 1, 1], Mm[..., 1, 2]
    g, h, i = Mm[..., 2, 0], Mm[..., 2, 1], Mm[..., 2, 2]
    A_, B_, C_ = e * i - f * h, f * g - d * i, d * h - e * g
    inv = th.stack([th.stack([A_, c * h - b * i, b * f - c * e], -1),
                    th.stack([B_, a * i - c * g, c * d - a * f], -1),
                    th.stack([C_, b * g - a * h, a * e - b * d], -1)], -2) / (a * A_ + b * B_ + c * C_)[..., None, None]
    return (inv @ rhs[..., None])[..., 0]


class Batch:
    """nworld worlds of one task on one object at one control period, with the batched HOM controller. The controller
    has no host synchronisation, so the task phase's period runs as one captured CUDA graph (run, capture_control)."""

    NEWTON_IT = 6                          # Newton steps of the pad's closest point (surfaces.ellipsoid_closest)
    CAPTURE = True                         # run the task phase's control period as a captured CUDA graph

    def __init__(self, scene: Scene, task, worlds, dt, gen_seed=0, phys_dt=None):
        """phys_dt: the physics step (s), the scene's 1 ms when None; a control period takes dt / phys_dt steps."""
        import mujoco
        import mujoco_warp as mjw
        import torch
        self.mj, self.mjw, self.torch = mujoco, mjw, torch
        self.scene, self.task, self.worlds = scene, task, worlds
        self.pinch_task = task in ("pinpush", "press")
        self.prm = H.HomParams(dt=dt)
        W = self.W = len(worlds)
        m = mocap_model(scene.pads_path)
        if phys_dt is not None:
            m.opt.timestep = phys_dt
        self.m = m
        self.ob = m.body(S.OBJ).id
        # pads, and the v4b tip's small rigid spheres (cassette, cap, the DIP link's fine ones, r < 2 mm), meet the object
        # only: every other pair of them multiplies the broadphase (447 rigid spheres per finger made a 2,048-world batch
        # about ten times slower); the link spheres of 2 mm and more keep the fingers apart
        fine = re.compile(r"_(dip_link|cassette|cap)_s\d+$")
        pads = np.array([g for g in range(m.ngeom) if "_pad" in (m.geom(g).name or "")
                         or (fine.search(m.geom(g).name or "") and m.geom_size[g][0] < 0.002)])
        tool = np.flatnonzero(m.geom_bodyid == self.ob)
        m.geom_contype[pads], m.geom_conaffinity[pads] = 2, 0         # pads meet the object only
        m.geom_conaffinity[tool] = m.geom_conaffinity[tool] | 2
        m.opt.tolerance, m.opt.iterations, m.opt.ls_iterations = 1e-6, 20, 10
        self.wm = mjw.put_model(m)
        self.wm.opt.contact_sensor_maxmatch = max(int(self.wm.opt.contact_sensor_maxmatch), 256)
        bp = np.repeat(m.body_pos[None].astype(np.float32), W, 0)
        mount_bid = {f: m.body(f"{f}_mount").id for f in FINGERS}
        for w, wd_ in enumerate(worlds):
            for f in FINGERS:
                bp[w, mount_bid[f]] = wd_["body_mounts"][f]
        self.wm.body_pos = wp.array(bp, dtype=wp.vec3)
        self.wd = mjw.make_data(m, nworld=W, nconmax=NCONMAX, njmax=NJMAX)
        self.wdm = mjw.make_data(m, nworld=W, nconmax=4, njmax=8)            # the mirror: the measured state
        self.mo_palm = int(m.body_mocapid[m.body("palm_pose").id])
        self.mo_post = int(m.body_mocapid[m.body("post").id])
        self.qa = int(m.jnt_qposadr[m.body_jntadr[self.ob]])
        self.va = int(m.jnt_dofadr[m.body_jntadr[self.ob]])
        jid = np.array([[m.joint(f"{f}_{j}").id for j in JOINTS] for f in FINGERS])
        self.jid = jid
        self.qadr = np.array([[m.jnt_qposadr[i] for i in r] for r in jid])
        self.act = np.array([[m.actuator(f"a_{f}_{j}").id for j in JOINTS] for f in FINGERS])
        self.tip = np.array([m.body(f"{f}_tip").id for f in FINGERS])
        self.lo = m.jnt_range[jid, 0]
        self.hi = m.jnt_range[jid, 1]
        self.kp = np.array([[m.actuator_gainprm[m.actuator(f"a_{f}_{j}").id][0] for j in JOINTS] for f in FINGERS])
        # gravity on each finger joint from the bodies below it (hom.Mirror.gravity: mj_rne at zero velocity)
        sub = np.zeros((3, 3, m.nbody))
        for k in range(3):
            for j in range(3):
                b0 = m.jnt_bodyid[jid[k, j]]
                for b in range(m.nbody):
                    a = b
                    while a > 0 and a != b0:
                        a = m.body_parentid[a]
                    sub[k, j, b] = float(a == b0)
        self.grav_sub = sub
        self.mg = m.body_mass[:, None] * m.opt.gravity[None, :]
        objmask = np.zeros(m.ngeom, np.int32)
        objmask[tool] = 1
        fidx = -np.ones(m.ngeom, np.int32)
        tidx = -np.ones(m.ngeom, np.int32)
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            for k, f in enumerate(FINGERS):
                if bn.startswith(f + "_"):
                    fidx[g] = k
                    if bn == f"{f}_tip":
                        tidx[g] = k
        self.objmask, self.fidx, self.tidx = (wp.array(objmask, dtype=int), wp.array(fidx, dtype=int),
                                              wp.array(tidx, dtype=int))
        self.ftip = wp.zeros((W, 3), dtype=float)
        self.ftot = wp.zeros((W, 3), dtype=float)
        self.cops = wp.zeros((W, 3, 3), dtype=float)
        specs, _ = T.sr2_specs(worlds[0]["setup"])
        surf = [sp.surface for sp in specs]                             # Posed, in each finger's tip frame
        self.ell = isinstance(surf[0].s, T.HalfEllipsoid)              # cad_v2's pad, else the TPU block
        if self.ell:
            e = surf[0].s
            self.box_c = np.array([sf.pos for sf in surf])              # the ellipsoid's centre per finger
            self.box_a = np.array([e.ax, e.ay, e.az, e.azp])
            self.box_r = 0.0
        else:
            boxes = [sf.s for sf in surf]
            self.box_c = np.array([b.c + sf.pos for b, sf in zip(boxes, surf)])
            self.box_a = np.array([b.a for b in boxes])
            self.box_r = float(boxes[0].r)
        self.gen_seed = gen_seed
        self._init_state()
        self._capture()

    # ---------------------------------------------------------------- state
    def _init_state(self):
        m, W = self.m, self.W
        qpos = np.repeat(m.qpos0[None], W, 0).astype(np.float64)
        ctrl = np.zeros((W, m.nu))
        for w, wd_ in enumerate(self.worlds):
            seed = wd_["seed"]
            rng = np.random.default_rng(seed)
            jit = np.zeros(3) if seed == 0 else np.array([rng.uniform(-1, 1) * 1e-3, rng.uniform(-1, 1) * 1e-3, 0.0])
            yaw = 0.0 if seed == 0 else math.radians(rng.uniform(-2, 2))
            qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            q = np.zeros(4)
            self.mj.mju_mulQuat(q, qz, np.asarray(wd_["setup"]["quat"], float))
            qpos[w, self.qa:self.qa + 3] = np.array([0.0, 0.0, S.Z_OBJ]) + jit
            qpos[w, self.qa + 3:self.qa + 7] = q
            for k, f in enumerate(FINGERS):
                qpos[w, self.qadr[k]] = wd_["q_open"][f]
                ctrl[w, self.act[k]] = wd_["q_open"][f]
        mp = np.zeros((W, m.nmocap, 3))
        for b in range(m.nbody):
            if m.body_mocapid[b] >= 0:
                mp[:, m.body_mocapid[b]] = m.body_pos[b]
        for w, wd_ in enumerate(self.worlds):
            mp[w, self.mo_palm] = wd_["setup"]["palm"]
        for dd in (self.wd, self.wdm):
            dd.mocap_pos.assign(mp.astype(np.float32))
            dd.qpos.assign(qpos.astype(np.float32))
            dd.qvel.zero_()
        self.wd.ctrl.assign(ctrl.astype(np.float32))
        self.obj0 = qpos[:, self.qa:self.qa + 7].astype(np.float32)            # the start poses, for the variant hold
        self.mjw.forward(self.wm, self.wd)
        self.mjw.kinematics(self.wm, self.wdm)
        wp.synchronize()

    def _launch_forces(self):
        self.ftip.zero_()
        self.ftot.zero_()
        self.cops.zero_()
        c = self.wd.contact
        wp.launch(finger_forces, dim=c.pos.shape[0],
                  inputs=[self.wd.nacon, c.worldid, c.geom, c.pos, c.efc_address, self.wd.efc.force, self.objmask,
                          self.fidx, self.tidx, self.ftip, self.ftot, self.cops])

    def _capture(self):
        mjw = self.mjw
        self.nsub = int(round(self.prm.dt / self.m.opt.timestep))

        def block():
            for _ in range(self.nsub):
                mjw.step(self.wm, self.wd)
            mjw.kinematics(self.wm, self.wd)
            self._launch_forces()

        def mirror():
            mjw.kinematics(self.wm, self.wdm)

        block()                                                   # compile every kernel before the captures
        mirror()
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            block()
        self.graph = cap.graph
        with wp.ScopedCapture() as cap:
            mirror()
        self.graph_m = cap.graph
        self._init_state()                                         # the warm-up block moved the state: start again
        self._launch_forces()
        wp.synchronize()

    # ---------------------------------------------------------------- geometry helpers (torch)
    def _tip_closest(self, x):
        """Closest point of each finger's tip surface (tip frame) to points x (W, 3, 3), its outward normal and the
        signed distance: the half-ellipsoid pad (cad_v2) or the rounded TPU block."""
        return self._ell_closest(x) if self.ell else self._rb_closest(x)

    def _ell_newton(self, p, a, k):
        """surfaces.ellipsoid_closest for points p (..., 3) and semi-axes a (3,), a_k the smallest: the root t of
        s(t) = sum (a_i p_i / (a_i^2 + t))^2 = 1 lies between t0 = -a_k^2 + a_k |p_k| and t1 = -a_k^2 + |a p|; Newton's
        method runs NEWTON_IT steps on s^(-1/2) - 1 from t0, a step leaving the bracket being a bisection."""
        th = self.torch
        a2 = a * a
        pp = th.where(p.abs() < 1e-12, th.full_like(p, 1e-12), p)
        c = (a * pp) ** 2
        lo = -a2[k] + a[k] * pp[..., k].abs()
        hi = -a2[k] + (a * pp).norm(dim=-1)
        t = lo.clone()
        for _ in range(self.NEWTON_IT):
            d = a2 + t[..., None]
            s_ = (c / d ** 2).sum(-1)
            phi = s_.rsqrt() - 1.0
            lo = th.where(phi < 0, t, lo)
            hi = th.where(phi < 0, hi, t)
            tn = t - phi / (s_ ** -1.5 * (c / d ** 3).sum(-1)).clamp_min(1e-300)
            t = th.where((tn > lo) & (tn < hi), tn, 0.5 * (lo + hi))
        x = a2 * pp / (a2 + t[..., None])
        n = th.nn.functional.normalize(x / a2, dim=-1)
        dist = (pp - x).norm(dim=-1)
        return x, n, th.where(((pp / a) ** 2).sum(-1) < 1.0, -dist, dist)

    def _ell_closest(self, x):
        """surfaces.HalfEllipsoid.closest for points x (W, 3, 3) in the tip frames: both halves queried, the point on
        its own half kept (the nearer if both are, the distal half's if neither)."""
        th = self.torch
        p = x - self.c_t
        qd, nd, sd_d = self._ell_newton(p, self.a_halves[0], self.k_small[0])
        qp, np_, sd_p = self._ell_newton(p, self.a_halves[1], self.k_small[1])
        ok_d, ok_p = qd[..., 2] <= 1e-12, qp[..., 2] >= -1e-12
        use_p = ok_p & (~ok_d | (sd_p.abs() < sd_d.abs()))
        q = th.where(use_p[..., None], qp, qd)
        n = th.where(use_p[..., None], np_, nd)
        return q + self.c_t, n, th.where(use_p, sd_p, sd_d)

    def _ell_support(self, u):
        """The pad's point farthest along unit directions u (W, 3, 3) in the tip frames."""
        th = self.torch
        ax, ay, az, azp = self.a_t.unbind(-1)
        a = th.stack([ax.expand_as(u[..., 0]), ay.expand_as(u[..., 0]), th.where(u[..., 2] < 0, az, azp)], -1)
        return self.c_t + a * a * u / (a * u).norm(dim=-1, keepdim=True).clamp_min(1e-15)

    def _rb_closest(self, x):
        """Closest point on each finger's rounded box (tip frame) to points x (W, 3, 3), its outward normal and the
        signed distance (surfaces.RoundedBox.closest)."""
        th = self.torch
        c, a = self.c_t, self.a_t
        xx = x - c
        core = th.maximum(th.minimum(xx, a), -a)
        d = xx - core
        dist = th.linalg.norm(d, dim=-1, keepdim=True)
        out = dist > 1e-12
        n_out = d / dist.clamp_min(1e-12)
        q = xx.abs() - a
        kk = q.argmax(-1, keepdim=True)
        sg = th.where(th.gather(xx, -1, kk) >= 0, 1.0, -1.0).to(x.dtype)
        n_in = th.zeros_like(x).scatter(-1, kk, sg)
        s_in = xx.clone().scatter(-1, kk, sg * (th.gather(a.expand_as(xx), -1, kk) + self.box_r))
        n = th.where(out, n_out, n_in)
        s = th.where(out, core + self.box_r * n_out, s_in) + c
        sd = th.where(out, dist - self.box_r, th.gather(q, -1, kk) - self.box_r)[..., 0]
        return s, n, sd

    def _cyl_local(self, p):
        """surfaces.Cylinder.closest for points p (..., 3) in the cylinder's frame: surface point, outward normal,
        signed distance."""
        th = self.torch
        r, hl = self.scene.r, self.scene.hl
        rho = th.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2)
        big = rho > 1e-15
        ex = th.where(big, p[..., 0] / rho.clamp_min(1e-15), th.ones_like(rho))
        ey = th.where(big, p[..., 1] / rho.clamp_min(1e-15), th.zeros_like(rho))
        e = th.stack([ex, ey, th.zeros_like(rho)], -1)
        z = p[..., 2]
        inside = (rho <= r) & (z.abs() <= hl)
        side = inside & ((r - rho) <= (hl - z.abs()))
        sz = th.where(z >= 0, 1.0, -1.0).to(p.dtype)
        q_side = th.stack([r * ex, r * ey, z], -1)
        q_cap = th.stack([p[..., 0], p[..., 1], sz * hl], -1)
        n_cap = th.stack([th.zeros_like(z), th.zeros_like(z), sz], -1)
        rc = th.minimum(rho, th.full_like(rho, r))
        q_out = th.stack([rc * ex, rc * ey, z.clamp(-hl, hl)], -1)
        dd = p - q_out
        dist = th.linalg.norm(dd, dim=-1)
        n_out = th.where((dist > 1e-15)[..., None], dd / dist.clamp_min(1e-15)[..., None], e)
        q = th.where(side[..., None], q_side, th.where(inside[..., None], q_cap, q_out))
        n = th.where(side[..., None], e, th.where(inside[..., None], n_cap, n_out))
        sd = th.where(side, -(r - rho), th.where(inside, -(hl - z.abs()), dist))
        return q, n, sd

    def _obj_closest(self, x, o, Ro):
        """Closest surface point of the object to world points x (W, 3, 3) and its outward normal (world)."""
        th = self.torch
        oo = o[:, None, :]
        if self.scene.shape == "sphere":
            n = th.nn.functional.normalize(x - oo, dim=-1)
            return oo + self.scene.r * n, n
        loc = th.einsum("wji,wfj->wfi", Ro, x - oo)
        q, n, _ = self._cyl_local(loc)
        return oo + th.einsum("wij,wfj->wfi", Ro, q), th.einsum("wij,wfj->wfi", Ro, n)

    def _witness(self, pb, Rb, o, Ro, to_tip, to_world):
        """GCF from the closest points of the rounded block and the object: p1 (block), p0 (object), x1 (block to
        object) and the signed distance."""
        th = self.torch
        oo = o[:, None, :].expand(-1, 3, -1)
        rf = self.box_r
        if self.ell:
            return self._ell_witness(pb, Rb, o, Ro, to_tip, to_world)
        if self.scene.shape == "sphere":
            s1, _n, _sd = self._rb_closest(to_tip(oo))
            p1 = to_world(s1)
            u = th.nn.functional.normalize(p1 - oo, dim=-1)
            return p1, oo + self.scene.r * u, -u, th.linalg.norm(p1 - oo, dim=-1) - self.scene.r
        r, hl = self.scene.r, self.scene.hl
        c, a = self.c_t, self.a_t

        def cyl_proj(x):                                   # closest point of the solid cylinder
            loc = th.einsum("wji,wfj->wfi", Ro, x - oo)
            rho = th.sqrt(loc[..., 0] ** 2 + loc[..., 1] ** 2).clamp_min(1e-15)
            sc = th.clamp(r / rho, max=1.0)
            q = th.stack([loc[..., 0] * sc, loc[..., 1] * sc, loc[..., 2].clamp(-hl, hl)], -1)
            return oo + th.einsum("wij,wfj->wfi", Ro, q)

        def box_proj(y):                                   # closest point of the block's solid core box
            loc = to_tip(y)
            return to_world(th.maximum(th.minimum(loc - c, a), -a) + c)

        x = to_world(c.expand(self.W, -1, -1))
        for _ in range(N_AP):
            x = box_proj(cyl_proj(x))
        y = cyl_proj(x)
        d = y - x
        dc = th.linalg.norm(d, dim=-1)
        sep = dc > 1e-9
        # the core inside the cylinder (the block sunk more than its fillet): the cylinder's normal at the core point
        q_in, n_in = self._obj_closest(x, o, Ro)
        x1 = th.where(sep[..., None], d / dc.clamp_min(1e-12)[..., None], -n_in)
        dist = th.where(sep, dc, -th.linalg.norm(q_in - x, dim=-1)) - rf
        p1 = x + rf * x1
        p0 = th.where(sep[..., None], y, q_in)
        return p1, p0, x1, dist

    def _ell_witness(self, pb, Rb, o, Ro, to_tip, to_world):
        """GCF from the closest points of the half-ellipsoid pad and the object (hom.Mirror.gcf_witness). Sphere: the
        pad's point nearest the centre. Cylinder: alternating projections between the solid pad and the solid cylinder
        while they are apart; overlapping, the pad's support point along the cylinder's inward normal at the overlap and
        that point's foot on the cylinder."""
        th = self.torch
        oo = o[:, None, :].expand(-1, 3, -1)
        RbT = Rb.transpose(-1, -2)
        if self.scene.shape == "sphere":
            p1 = to_world(self._ell_closest(to_tip(oo))[0])
            u = th.nn.functional.normalize(p1 - oo, dim=-1)
            return p1, oo + self.scene.r * u, -u, th.linalg.norm(p1 - oo, dim=-1) - self.scene.r
        r, hl = self.scene.r, self.scene.hl

        def cyl_proj(x):                                   # closest point of the solid cylinder
            loc = th.einsum("wji,wfj->wfi", Ro, x - oo)
            rho = th.sqrt(loc[..., 0] ** 2 + loc[..., 1] ** 2).clamp_min(1e-15)
            sc = th.clamp(r / rho, max=1.0)
            q = th.stack([loc[..., 0] * sc, loc[..., 1] * sc, loc[..., 2].clamp(-hl, hl)], -1)
            return oo + th.einsum("wij,wfj->wfi", Ro, q)

        def ell_proj(y):                                   # closest point of the solid pad
            q, _n, sd = self._ell_closest(to_tip(y))
            return th.where((sd > 0)[..., None], to_world(q), y)

        x = to_world(self.c_t.expand(self.W, -1, -1)) if self.ap_x is None else self.ap_x
        for _ in range(N_AP if self.ap_x is None else N_AP_WARM):
            x = ell_proj(cyl_proj(x))
        if self.ap_x is None:                               # the next period starts here
            self.ap_x = x.clone()
        else:
            self.ap_x.copy_(x)
        y = cyl_proj(x)
        d = y - x
        dc = th.linalg.norm(d, dim=-1)
        sep = dc > 1e-9
        _q, n_in = self._obj_closest(x, o, Ro)              # the cylinder's outward normal near the overlap
        s_sup = to_world(self._ell_support(th.einsum("wfij,wfj->wfi", RbT, -n_in)))
        q2, n2 = self._obj_closest(s_sup, o, Ro)
        p1 = th.where(sep[..., None], x, s_sup)
        p0 = th.where(sep[..., None], y, q2)
        x1 = th.where(sep[..., None], d / dc.clamp_min(1e-12)[..., None], -n2)
        dist = th.where(sep, dc, -th.linalg.norm(q2 - s_sup, dim=-1))
        return p1, p0, x1, dist

    @staticmethod
    def _quat_mat(q):
        import torch as th
        w, x, y, z = q.unbind(-1)
        return th.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                         2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)

    @staticmethod
    def _quat_mul(a, b):
        import torch as th
        w1, x1, y1, z1 = a.unbind(-1)
        w2, x2, y2, z2 = b.unbind(-1)
        return th.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)

    # ---------------------------------------------------------------- the controller
    def reset_controller(self):
        th = self.torch
        dev, f64 = "cuda", th.float64
        self.dev, self.f64 = dev, f64

        def T_(x, dtype=f64):
            return th.tensor(np.asarray(x), dtype=dtype, device=dev)

        self.c_t, self.a_t = T_(self.box_c), T_(self.box_a)
        self.lo_t, self.hi_t, self.kp_t = T_(self.lo), T_(self.hi), T_(self.kp)
        self.q_open = T_([[wd_["q_open"][f] for f in FINGERS] for wd_ in self.worlds])
        self.healthy = self.q_open.clone()
        self.theta_ref = None
        self.c1 = None
        self.ap_x = None                                   # the ellipsoid-cylinder witness' previous point
        self.R_bm = self.p_bm = None
        pr = self.prm
        self.scale = T_([1.0, 1.0, 1.0] + [1.0 / pr.ell] * 3)
        pats = np.array(np.meshgrid([0, 1, 2], [0, 1, 2], [0, 1, 2], indexing="ij")).reshape(3, -1).T
        self.pats = th.tensor(pats, device=dev)
        self.tip_t = th.tensor(self.tip, device=dev)
        if self.ell:
            ax, ay, az, azp = self.a_t.unbind(-1)
            self.a_halves = (th.stack([ax, ay, az]), th.stack([ax, ay, azp]))
            self.k_small = tuple(int(h_.argmin()) for h_ in self.a_halves)
        self.graph_c = None
        self.n_out = T_([[wd_["setup"]["n_out"][f] for f in FINGERS] for wd_ in self.worlds])
        cmds = [finger_commands(self.task, wd_["setup"]["n_out"]) for wd_ in self.worlds]
        self.weights = T_(cmds[0][0])                                          # (3, 6), the same in every world
        self.follow = T_(cmds[0][1], th.bool)
        self.pusher = T_(cmds[0][2], th.bool)
        self.follow_f, self.pusher_f = self.follow.to(f64), self.pusher.to(f64)
        self.s_t = th.zeros((), dtype=f64, device=dev)          # Eq. (3)'s speed, the captured period's input
        self.force_on = T_([c[3] for c in cmds])                               # (W, 3)
        V = [wd_["var"] for wd_ in self.worlds]
        self.cfmin = T_([v["cfmin"] for v in V])
        self.flg = T_([v["flg"] for v in V])
        self.ra_w = T_([v["reanchor"] for v in V])
        self.hold_t = T_([v["hold"] for v in V])
        self.hold_max = float(max(v["hold"] for v in V))
        self.obj0_t = th.tensor(self.obj0, device=dev)
        self.ra_any = bool((self.ra_w > 0).any())
        self.n_ra = th.zeros((self.W, 3), dtype=f64, device=dev)            # re-anchored periods per finger
        self.mode = T_([v["mode"] for v in V], th.long)
        self.delay = T_([v["delay"] for v in V], th.long)
        for k in ("pos_mm", "rot_deg", "cop_mm", "f_N", "tip_k", "f_bias", "f_min"):
            setattr(self, k, T_([v[k] for v in V]))
        self.cal_q = T_([v["cal_q"] for v in V])
        self.cal_pos = T_([v["cal_pos"] for v in V])
        self.tip_gain = T_([v["tip_gain"] for v in V])
        self.tip_off = T_([v["tip_off"] for v in V])
        self.L = DMAX + 1
        W = self.W
        self.buf = {"q": th.zeros((self.L, W, 3, 3), dtype=f64, device=dev),
                    "pos": th.zeros((self.L, W, 3), dtype=f64, device=dev),
                    "quat": th.zeros((self.L, W, 4), dtype=f64, device=dev),
                    "cop": th.zeros((self.L, W, 3, 3), dtype=f64, device=dev),
                    "F": th.zeros((self.L, W, 3), dtype=f64, device=dev)}
        self.gen = th.Generator(device=dev)
        self.gen.manual_seed(int(self.gen_seed))
        self.qadr_t = th.tensor(self.qadr, device=dev)
        self.jid_t = th.tensor(self.jid, device=dev)
        self.act_t = th.tensor(self.act.reshape(-1), device=dev)
        self.sub_t = T_(self.grav_sub)
        self.mg_t = T_(self.mg)

    def views(self, d):
        to = wp.to_torch
        return {"qpos": to(d.qpos), "qvel": to(d.qvel), "xpos": to(d.xpos), "xmat": to(d.xmat), "xipos": to(d.xipos),
                "xanchor": to(d.xanchor), "xaxis": to(d.xaxis), "ctrl": to(d.ctrl), "mocap_pos": to(d.mocap_pos)}

    def _randn(self, *shape):
        return self.torch.randn(*shape, generator=self.gen, device=self.dev, dtype=self.f64)

    def measure(self, k, true):
        """What the controller receives in period k: hom_tasks.rollout's delay, calibration offsets, tracker noise and
        fingertip model applied to the plant's state."""
        th = self.torch
        W = self.W
        slot = k % self.L
        for key in self.buf:
            self.buf[key][slot] = true[key]
        idx = (k - self.delay).clamp(min=0) % self.L
        ar = th.arange(W, device=self.dev)
        q, pos, quat, cop, F = (self.buf[key][idx, ar] for key in ("q", "pos", "quat", "cop", "F"))
        has = F > 0
        pos_m = pos + self.cal_pos + self._randn(W, 3) * (self.pos_mm * 1e-3)[:, None]
        ax = th.nn.functional.normalize(self._randn(W, 3), dim=-1)
        ang = th.deg2rad(self.rot_deg) * self._randn(W)
        dq = th.cat([th.cos(ang / 2)[:, None], th.sin(ang / 2)[:, None] * ax], -1)
        quat_m = self._quat_mul(dq, quat)
        q_m = q + self.cal_q
        # the fingertip model (hom_tasks.tip_reading) and the Gaussian contact noise of the 'sense' variant
        F_tip = F * (1.0 + self.tip_gain) + self.f_bias[:, None] + self._randn(W, 3) * (T.TIP_NOISE_N * self.tip_k)[:, None]
        sig = th.where(F >= T.TIP_SIGMA_MM[0][0], T.TIP_SIGMA_MM[0][1],
                       th.where(F >= T.TIP_SIGMA_MM[1][0], T.TIP_SIGMA_MM[1][1], T.TIP_SIGMA_MM[2][1]))
        cop_tip = cop + self.tip_off + self._randn(W, 3, 3) * (sig * self.tip_k[:, None] * 1e-3)[..., None]
        cop_sn = cop + self._randn(W, 3, 3) * (self.cop_mm * 1e-3)[:, None, None]
        F_sn = (F + self._randn(W, 3) * self.f_N[:, None]).clamp_min(0.0)
        md = self.mode[:, None]
        valid = th.where(md == 1, th.zeros_like(has), th.where(md == 2, has & (F_tip >= self.f_min[:, None]), has))
        Fsel = th.where(md == 2, F_tip, th.where(md == 3, F_sn, F))
        csel = th.where((md == 2)[..., None], cop_tip, th.where((md == 3)[..., None], cop_sn, cop))
        Fm = th.where(valid, Fsel, th.zeros_like(Fsel))
        return {"q": q_m, "pos": pos_m, "quat": quat_m, "cop": csel, "F": Fm, "valid": valid, "quat_d": quat}

    def mirror(self, meas):
        """The mirror's kinematics at the measured state; returns its views."""
        mv = self.mv
        W = self.W
        mv["qpos"][:, self.qadr_t.reshape(-1)] = meas["q"].reshape(W, 9).float()
        mv["qpos"][:, self.qa:self.qa + 3] = meas["pos"].float()
        mv["qpos"][:, self.qa + 3:self.qa + 7] = meas["quat"].float()
        wp.capture_launch(self.graph_m)
        return mv

    def gravity(self, xipos, anc, axs):
        """Joint torques that hold each finger against its weight at the mirror's posture (mj_rne at zero velocity)."""
        th = self.torch
        rel = xipos[:, None, None, :, :] - anc[:, :, :, None, :]                 # (W, 3f, 3j, nbody, 3)
        tq = th.cross(rel, self.mg_t[None, None, None].expand_as(rel), dim=-1)    # torque of each body's weight
        tq = (tq * self.sub_t[None, :, :, :, None]).sum(-2)                      # (W, 3f, 3j, 3)
        return -(tq * axs).sum(-1)

    def control(self, t, meas, mv, V_obj, s_push):
        """One control period of hom.HomController.step for every world and finger. V_obj: the object's reference body
        twist (W, 6); s_push: Eq. (3)'s speed (rad/s). Returns the servo targets (W, 3, 3)."""
        th, f64, pr = self.torch, self.f64, self.prm
        dt = pr.dt
        W = self.W
        theta = meas["q"]
        if self.theta_ref is None:
            self.theta_ref = theta.clone()
        xpos, xmat = mv["xpos"].to(f64), mv["xmat"].to(f64)
        pb, Rb = xpos[:, self.tip_t], xmat[:, self.tip_t]
        o, Ro = xpos[:, self.ob], xmat[:, self.ob]
        anc = mv["xanchor"].to(f64)[:, self.jid_t]                                       # (W, 3f, 3j, 3)
        axs = mv["xaxis"].to(f64)[:, self.jid_t]
        Fm, valid, cop = meas["F"], meas["valid"], meas["cop"]
        RbT = Rb.transpose(-1, -2)

        def to_tip(p):
            return th.einsum("wfij,wfj->wfi", RbT, p - pb)

        def to_world(p):
            return pb + th.einsum("wfij,wfj->wfi", Rb, p)

        oo = o[:, None, :].expand(-1, 3, -1)
        # GCF from the centre of pressure (hom.Mirror.gcf_from_cop)
        s1c, n1c, _ = self._tip_closest(to_tip(cop))
        p1c = to_world(s1c)
        n1w = th.einsum("wfij,wfj->wfi", Rb, n1c)
        p0c, n0c = self._obj_closest(cop, o, Ro)
        x1c = th.nn.functional.normalize(n1w - n0c, dim=-1)
        distc = ((p0c - p1c) * x1c).sum(-1)
        # GCF from the witness points (hom.Mirror.gcf_witness)
        p1w, p0w, x1w, distw = self._witness(pb, Rb, o, Ro, to_tip, to_world)
        use_cop = (valid & (Fm >= self.cfmin[:, None]))[..., None]
        p1 = th.where(use_cop, p1c, p1w)
        p0 = th.where(use_cop, p0c, p0w)
        x1 = th.where(use_cop, x1c, x1w)
        dist = th.where(use_cop[..., 0], distc, distw)
        ay = axs[:, :, 2]                                                                 # the PIP's axis
        y = ay - (ay * x1).sum(-1, keepdim=True) * x1
        y = th.nn.functional.normalize(y, dim=-1)
        z = th.cross(x1, y, dim=-1)
        R1T = th.stack([x1, y, z], -1).transpose(-1, -2)
        # the pinch frame {M}, fixed to the object when the pusher starts (hom.pinch_frame)
        V = V_obj[:, None, :] * self.follow_f[None, :, None]
        if self.pinch_task and self.R_bm is None and float(s_push) != 0.0:
            pa, pb_ = p0[:, 0], p0[:, 1]
            xm = th.nn.functional.normalize(pb_ - pa, dim=-1)
            up = th.zeros_like(xm)
            up[:, 2] = 1.0
            ym = up - (up * xm).sum(-1, keepdim=True) * xm
            ym = th.nn.functional.normalize(ym, dim=-1)
            Rm = th.stack([xm, ym, th.cross(xm, ym, dim=-1)], -1)
            self.R_bm = Ro.transpose(-1, -2) @ Rm
            self.p_bm = th.einsum("wji,wj->wi", Ro, 0.5 * (pa + pb_) - o)
        if self.pinch_task and self.R_bm is not None:
            xb = self.R_bm[:, :, 0]
            Vp = s_push * th.cat([xb, th.cross(self.p_bm, xb, dim=-1)], -1)               # Eq. (3)
            V = V + self.pusher_f[None, :, None] * Vp[:, None, :]
        # Eq. (1)
        w_o = th.einsum("wij,wfj->wfi", Ro, V[..., :3])
        v_p1 = th.einsum("wij,wfj->wfi", Ro, V[..., 3:]) + th.cross(w_o, p1 - oo, dim=-1)
        Jr = axs.transpose(-1, -2)                                                        # (W, 3, 3xyz, 3j)
        Jp = th.cross(axs, p1[:, :, None, :] - anc, dim=-1).transpose(-1, -2)
        A = th.cat([R1T @ Jr, R1T @ Jp], -2)                                               # (W, 3, 6, 3)
        b = -th.cat([th.einsum("wfij,wfj->wfi", R1T, w_o), th.einsum("wfij,wfj->wfi", R1T, v_p1)], -1)
        # the finger's reference relative twist: the approach, the force loop, the anti-windup
        v_ref = th.zeros_like(b)
        if T.T_OPEN <= t < T.T_CLOSE:
            v_ref[..., 3] = T.V_APPROACH
        force = self.force_on if t >= T.T_CLOSE else th.zeros_like(self.force_on)
        touching = (Fm >= self.cfmin[:, None]) | (dist <= 0.0)
        floop = (self.flg > 0)[:, None] & touching
        v_ref[..., 3] = v_ref[..., 3] + th.where(floop & (force > 0), self.flg[:, None] * (force - Fm), th.zeros_like(Fm))
        aw = (dist < pr.aw_dist) & (v_ref[..., 3] > 0) & ~floop
        v_ref[..., 3] = th.where(aw, th.zeros_like(dist), v_ref[..., 3])
        # Eq. (2): min |sw (A u + b - v_ref)|^2 + w_damp |u|^2 + w_healthy |k (theta_h - theta) - u|^2, box bounds
        sw = th.sqrt(self.weights) * self.scale                                          # (3, 6)
        As = sw[None, :, :, None] * A
        rs = sw[None] * (v_ref - b)
        I3 = th.eye(3, dtype=f64, device=self.dev)
        Hm = As.transpose(-1, -2) @ As + (pr.w_damp + pr.w_healthy) * I3
        g = th.einsum("wfij,wfi->wfj", As, rs) + pr.w_healthy * pr.k_healthy * (self.healthy - theta)
        lb = th.maximum(th.full_like(theta, -pr.u_max), (self.lo_t - self.theta_ref) / dt)
        ub = th.minimum(th.full_like(theta, pr.u_max), (self.hi_t - self.theta_ref) / dt)
        lb, ub = th.minimum(lb, ub - 1e-12), th.maximum(ub, lb + 1e-12)
        P = self.pats
        fixed = (P > 0)[:, None, None, :]
        bound = th.where(P[:, None, None, :] == 1, lb[None], ub[None])
        Mm = th.where(fixed[..., None], I3.expand(27, 1, 1, 3, 3) * 1.0, Hm[None].expand(27, -1, -1, -1, -1))
        cc = th.where(fixed, bound, g[None].expand(27, -1, -1, -1))
        uu = solve3(Mm, cc)
        feas = ((uu >= lb[None] - 1e-9) & (uu <= ub[None] + 1e-9)).all(-1)
        obj = 0.5 * th.einsum("pwfi,wfij,pwfj->pwf", uu, Hm, uu) - (uu * g[None]).sum(-1)
        obj = th.where(feas, obj, th.full_like(obj, float("inf")))
        best = obj.argmin(0)
        u_sol = th.gather(uu, 0, best[None, ..., None].expand(1, -1, -1, 3))[0]
        u_sol = th.minimum(th.maximum(u_sol, lb), ub)
        tr = self.theta_ref + dt * u_sol
        tr = th.minimum(th.maximum(tr, theta - pr.lead), theta + pr.lead)
        self.theta_ref.copy_(th.minimum(th.maximum(tr, self.lo_t), self.hi_t))
        # the filter at k_min: the contact point low-passed on the block's surface
        l1 = to_tip(p1)
        if self.c1 is None:
            self.c1 = self._tip_closest(l1)[0].clone()
        q1, n1, _ = self._tip_closest(self.c1)
        e = l1 - q1
        e_t = e - (e * n1).sum(-1, keepdim=True) * n1
        self.c1.copy_(self._tip_closest(q1 + pr.filter.k_min * dt * e_t)[0])
        s_act, n_act_l, _ = self._tip_closest(self.c1)
        p_act = to_world(s_act)
        n_act = th.einsum("wfij,wfj->wfi", Rb, n_act_l)
        # the impedance of the 'reference' mode, and gravity
        Ja = th.cross(axs, p_act[:, :, None, :] - anc, dim=-1).transpose(-1, -2)        # (W, 3, 3xyz, 3j)
        gap = dist - (n_act * th.einsum("wfij,wfj->wfi", Ja, self.theta_ref - theta)).sum(-1)
        if self.ra_any:                                    # hom.HomParams.reanchor
            ra = (self.ra_w[:, None] > 0) & touching & (force > 0) & (gap > pr.gap_max)
            self.theta_ref.copy_(th.where(ra[..., None], theta, self.theta_ref))
            gap = th.where(ra, dist, gap)
            self.n_ra.add_(ra.to(f64))
        gap = gap.clamp(0.0, pr.gap_max)
        JJ = Ja @ Ja.transpose(-1, -2) + pr.dls ** 2 * I3
        step = solve3(JJ, gap[..., None] * n_act)
        imp = th.einsum("wfij,wfi->wfj", Ja, step) + th.einsum("wfij,wfi->wfj", Ja, force[..., None] * n_act) / self.kp_t
        offset = th.where((force > 0)[..., None], imp, th.zeros_like(imp))
        if pr.gravity_ff:
            offset = offset + self.gravity(mv["xipos"].to(f64), anc, axs) / self.kp_t
        return th.minimum(th.maximum(self.theta_ref + offset, self.lo_t), self.hi_t)

    def capture_control(self, t, meas, mv, V_obj):
        """One control period of the task phase (approach over, forces on, {M} set) captured as a CUDA graph on static
        inputs: the measurement, the object's reference twist and Eq. (3)'s speed (self.s_t). The controller's state is
        left as it was. Returns (graph, measurement inputs, twist input, servo-target output)."""
        th = self.torch
        sm = {k: v.clone() for k, v in meas.items()}
        sV = V_obj.clone()
        state = (self.theta_ref, self.c1, self.ap_x, self.n_ra)
        keep = [x.clone() if x is not None else None for x in state]
        st = th.cuda.Stream()
        st.wait_stream(th.cuda.current_stream())
        with th.cuda.stream(st):
            for _ in range(3):
                self.control(t, sm, mv, sV, self.s_t)
        th.cuda.current_stream().wait_stream(st)
        g = th.cuda.CUDAGraph()
        with th.cuda.graph(g):
            out = self.control(t, sm, mv, sV, self.s_t)
        for x, k in zip(state, keep):
            if x is not None:
                x.copy_(k)
        return g, sm, sV, out

    def control_period(self, t, meas, mv, V, s_push):
        """Batch.control, through the captured graph from the second period of the task phase on."""
        if not self.CAPTURE or t < T.T_LIFT + 0.4 + self.prm.dt - 1e-9:
            return self.control(t, meas, mv, V, s_push)
        self.s_t.fill_(s_push)
        if self.graph_c is None:
            self.graph_c = self.capture_control(t, meas, mv, V)
        g, sm, sV, out = self.graph_c
        for k, v in meas.items():
            sm[k].copy_(v)
        sV.copy_(V)
        g.replay()
        return out

    # ---------------------------------------------------------------- the schedules (hom_tasks.Schedule)
    def schedule(self, t):
        """The object's reference angular velocity (world, rad/s), Eq. (3)'s speed s (rad/s), the commanded angle about
        the pinch axis (deg, None outside the task) and the phase name at time t."""
        w_ref = np.zeros(3)
        ph = Tsched_phase(t)
        if ph != "task":
            return w_ref, 0.0, None, ph
        tt = t - (T.T_LIFT + 0.4)
        if self.pinch_task:
            th_, w, name = self.sch.angle(tt)
            return w_ref, -math.radians(w), th_, name
        if self.task == "roll":
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
                w_ref[ax] = sg * math.radians(T.PRM["w_step_dps"])
            rep = " 2" if seq[:i].count(seq[i]) else ""
            return w_ref, 0.0, None, ("rest" if ax is None else f"w{'xyz'[ax]} {'+' if sg > 0 else '-'}{rep}")
        if self.task == "twist":
            tr = tt - edges[-1]
            if tr < 3.0:
                w_ref[2] = math.radians(T.PRM["ramp_dps"])
                return w_ref, 0.0, None, "twist ramp"
        return w_ref, 0.0, None, "hold"

    # ---------------------------------------------------------------- one batch
    def run(self):
        th = self.torch
        self.reset_controller()
        self.sch = T.Schedule(self.task, self.worlds[0]["setup"], T.PRM)
        vw, self.mv = self.views(self.wd), self.views(self.wdm)
        ftip, ftot, cops = wp.to_torch(self.ftip), wp.to_torch(self.ftot), wp.to_torch(self.cops)
        dtc = self.prm.dt
        clock, t = [], 0.0                       # hom_tasks.rollout's accumulated clock: its phases change on the same ticks
        while t < self.sch.t_end - 1e-9:
            clock.append(t)
            t += dtc
        n = len(clock)
        W = self.W
        rec = {k: th.zeros((n, W, 3), device=self.dev) for k in ("pos", "w", "ft", "fo", "vr", "ax")}
        rec["cop"] = th.zeros((n, W, 3, 3), device=self.dev)
        phases, wrefs, times, thref = [], [], [], []
        post_off = False
        t0 = time.perf_counter()
        for k, t in enumerate(clock):
            qpos = vw["qpos"].to(self.f64)
            R = vw["xmat"][:, self.ob].to(self.f64)
            pos = qpos[:, self.qa:self.qa + 3]
            F = ftip.to(self.f64)
            cop = cops.to(self.f64) / F.clamp_min(1e-12)[..., None]
            cop = th.where((F > 0)[..., None], cop, th.zeros_like(cop))
            true = {"q": qpos[:, self.qadr_t], "pos": pos, "quat": qpos[:, self.qa + 3:self.qa + 7], "cop": cop, "F": F}
            wl = vw["qvel"][:, self.va + 3:self.va + 6].to(self.f64)
            rec["pos"][k] = pos.float()
            rec["w"][k] = th.einsum("wij,wj->wi", R, wl).float()
            rec["ft"][k] = ftip
            rec["fo"][k] = ftot
            rec["ax"][k] = R[:, :, 2].float()
            rec["cop"][k] = th.einsum("wji,wfj->wfi", R, cop - pos[:, None, :]).float()
            if t >= T.T_LIFT and not post_off:
                vw["mocap_pos"][:, self.mo_post, 2] -= 1.0                              # the post goes away
                post_off = True
            meas = self.measure(k, true)
            mv = self.mirror(meas)
            w_ref, s_push, th_cmd, ph = self.schedule(t)
            w_ref_t = th.tensor(w_ref, dtype=self.f64, device=self.dev)
            v_ref = th.zeros((W, 3), dtype=self.f64, device=self.dev)
            if self.task == "roll":
                v_ref = th.cross(w_ref_t.expand(W, 3), -self.scene.r * self.n_out[:, 0], dim=-1)
            rec["vr"][k] = v_ref.float()
            Rd = self._quat_mat(meas["quat_d"])                    # the schedule's frame: the delayed object pose
            RdT = Rd.transpose(-1, -2)
            V = th.cat([th.einsum("wij,j->wi", RdT, w_ref_t), th.einsum("wij,wj->wi", RdT, v_ref)], -1)
            phases.append(ph)
            wrefs.append(w_ref)
            times.append(t)
            thref.append(th_cmd)
            q_cmd = self.control_period(t, meas, mv, V, s_push)
            vw["ctrl"][:, self.act_t] = (q_cmd - self.cal_q).reshape(W, 9).float()
            wp.capture_launch(self.graph)
            if t + dtc <= self.hold_max + 1e-9:          # variant hold: the object back at its start pose, at rest
                hm = (self.hold_t >= t + dtc - 1e-9)[:, None]
                q_o = vw["qpos"][:, self.qa:self.qa + 7]
                q_o.copy_(th.where(hm, self.obj0_t, q_o))
                v_o = vw["qvel"][:, self.va:self.va + 6]
                v_o.copy_(th.where(hm, th.zeros_like(v_o), v_o))
        wp.synchronize()
        wall = time.perf_counter() - t0
        out = {k: v.cpu().numpy() for k, v in rec.items()}
        final = {"pos": vw["qpos"][:, self.qa:self.qa + 3].cpu().numpy(), "ftot": ftot.cpu().numpy(),
                 "n_ra": self.n_ra.cpu().numpy()}
        return out, final, phases, np.array(wrefs), times, thref, wall

    def score(self, rec, final, phases, wrefs, times, thref):
        out = []
        for w, wd_ in enumerate(self.worlds):
            psi = float(wd_["setup"].get("psi", 0.0))
            Rz = np.array([[math.cos(psi), math.sin(psi), 0.0], [-math.sin(psi), math.cos(psi), 0.0], [0, 0, 1.0]])
            ax = rec["ax"][:, w] @ Rz.T                          # the axis in the frame where the pinch runs along x
            rows = []
            for k, t in enumerate(times):
                ft = rec["ft"][k, w]
                row = {"t": round(t, 3), "phase": phases[k], "pos_mm": (1e3 * rec["pos"][k, w]).tolist(), "axis": ax[k].tolist(),
                       "F": {f: float(ft[i]) for i, f in enumerate(FINGERS)},
                       "F_link": {f: float(max(0.0, rec["fo"][k, w, i] - ft[i])) for i, f in enumerate(FINGERS)}}
                if self.pinch_task:
                    row["cop_obj_mm"] = {f: (1e3 * rec["cop"][k, w, i]).tolist() for i, f in enumerate(FINGERS)
                                         if ft[i] > 0}
                    if thref[k] is not None:
                        row["theta_ref_deg"] = thref[k]
                else:
                    row.update(w_dps=np.degrees(rec["w"][k, w]).tolist(), w_ref_dps=np.degrees(wrefs[k]).tolist(),
                               v_ref_mms=(1e3 * rec["vr"][k, w]).tolist())
                rows.append(row)
            se = {"pos": final["pos"][w], "Ftot": {f: float(final["ftot"][w, i]) for i, f in enumerate(FINGERS)}}
            res = T.metrics(self.task, wd_["setup"], rows, {"ok": True}, se)
            if self.ra_any:
                res["reanchors"] = {f: int(final["n_ra"][w, i]) for i, f in enumerate(FINGERS)}
            out.append(res)
        return out


def Tsched_phase(t):
    if t < T.T_OPEN:
        return "open"
    if t < T.T_CLOSE:
        return "close"
    if t < T.T_LIFT:
        return "squeeze"
    if t < T.T_LIFT + 0.4:
        return "lift"
    return "task"


# ------------------------------------------------------------------------------------------------ commands

def batch_child(spec):
    """One batch in its own process (spawned by cmd_run): build, run, score, and append its rows and timing line,
    fsynced. The process's exit returns all of its GPU memory; within one process, a finished batch's captured graphs
    kept enough of it that the next 2,048-world cylinder batch could not allocate (2026-10-29)."""
    import torch
    scene = Scene(spec["shape"], spec["d_mm"])
    chunk, task = spec["worlds"], spec["task"]
    t1 = time.time()
    B = Batch(scene, task, chunk, spec["dt"], gen_seed=spec["gen_seed"])
    t_build = time.time() - t1
    rec, final, phases, wrefs, times, thref, wall = B.run()
    t2 = time.time()
    res = B.score(rec, final, phases, wrefs, times, thref)
    t_score = time.time() - t2
    with open(spec["path"], "a") as fh:
        for wd_, r in zip(chunk, res):
            row = dict(wd_["key"], task=task, d_mm=float(spec["d_mm"]), seed=wd_["seed"], variant=wd_["var"]["var"],
                       status="ok", engine="mjwarp", hand=K.HAND, **r)
            fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    held = sum(bool(r.get("held")) for r in res)
    with open(spec["tpath"], "a") as fh:
        fh.write(json.dumps({"task": task, "d_mm": float(spec["d_mm"]), "dt": spec["dt"], "batch": spec["batch"],
                             "worlds": len(chunk), "build_s": round(t_build, 2), "run_s": round(wall, 2),
                             "score_s": round(t_score, 2), "layouts": spec["layouts"], "njmax": NJMAX,
                             "gpu": torch.cuda.get_device_name(0),
                             "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                             "when": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    print(f"{task} {spec['d_mm']:g} mm dt {spec['dt']}: batch {spec['batch']}, {len(chunk)} worlds, build {t_build:.1f} s, "
          f"run {wall:.1f} s ({wall / len(chunk) * 1000:.0f} ms per rollout), score {t_score:.1f} s, held {held}",
          flush=True)


def cmd_run(a):
    import multiprocessing as mp
    rows_dir = Path(a.rows_dir) if a.rows_dir else ROWS
    rows_dir.mkdir(parents=True, exist_ok=True)
    path = rows_dir / f"{a.out}.jsonl"
    tpath = rows_dir / f"{a.out}_timing.jsonl"
    if not a.append:
        for q in (path, tpath):
            if q.exists():
                q.unlink()
    ctx = mp.get_context("spawn")
    for task in a.task:
        sizes = a.d or ([D_FP[task]] if a.layouts != "design" else list(T.SIZES[task]))
        for d_mm in sizes:
            t0 = time.time()
            scene = Scene(T.SHAPE[task], d_mm)
            lays = layout_list(task, a.layouts, a.max, d_mm)
            worlds = []
            for key, mxy, setup in lays:
                bm, q_open = scene.prepare(mxy, setup)
                for var in a.variants:
                    for sd in a.seeds:
                        worlds.append({"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open,
                                       "var": variant_world(var, sd)})
            print(f"{task} {d_mm:g} mm: {len(lays)} layouts, {len(worlds)} worlds prepared in {time.time() - t0:.1f} s",
                  flush=True)
            by_dt = {}
            for wd_ in worlds:
                by_dt.setdefault(wd_["var"]["dt"], []).append(wd_)
            for dt, ws in by_dt.items():
                for b0 in range(0, len(ws), a.batch):
                    spec = {"shape": T.SHAPE[task], "d_mm": float(d_mm), "task": task, "worlds": ws[b0:b0 + a.batch],
                            "dt": dt, "gen_seed": a.gen_seed + b0, "batch": b0 // a.batch, "path": str(path),
                            "tpath": str(tpath), "layouts": a.layouts}
                    pr = ctx.Process(target=batch_child, args=(spec,))
                    pr.start()
                    pr.join()
                    if pr.exitcode != 0:
                        print(f"{task} {d_mm:g} mm dt {dt}: batch {b0 // a.batch} failed (exit {pr.exitcode})", flush=True)
                        return 1
    return 0


def cpu_rollout(scene, task, mxy, setup, q_open, seed, var="base"):
    """The CPU runner (hom_tasks.rollout) on the same layout, grasp and open pose as a GPU world: the reference scenes
    with the mount bodies moved, in the plant and in the mirror."""
    import mujoco
    from dataclasses import replace
    bm = scene.body_mounts(mxy)
    plant = T.Plant(scene.pads_path, setup)
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for m in (plant.m, m_pt):
        for f in FINGERS:
            m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = H.Mirror(m_pt, fs, os_)
    over, noise = T.VARIANTS[var]
    res, _trace, _ = T.rollout(task, setup, plant, mirror, q_open, seed, T.PRM, replace(H.HomParams(dt=T.DT_C), **over),
                               None, noise, T.PLANT_VARIANTS.get(var, {}).get("hold_s", 0.0))
    return res


def cmd_cpu_check(a):
    """The CPU runner on every k-th layout of a population (the GPU rows' layouts and grasps), for the comparison of
    GPU and CPU on the foundational poses. Rows: rows/<out>.jsonl, fsynced, one per (layout, seed)."""
    scene = Scene(T.SHAPE[a.task], a.d or D_FP[a.task])
    lays = layout_list(a.task, a.layouts, 0, a.d or D_FP[a.task])
    pick = lays[a.offset::max(1, len(lays) // a.n)][:a.n]
    path = (Path(a.rows_dir) if a.rows_dir else ROWS) / f"{a.out}.jsonl"
    for key, mxy, setup in pick:
        _bm, q_open = scene.prepare(mxy, setup)
        for sd in a.seeds:
            t0 = time.time()
            res = cpu_rollout(scene, a.task, mxy, setup, q_open, sd, a.variant)
            row = dict(key, task=a.task, d_mm=float(a.d or D_FP[a.task]), seed=sd, variant=a.variant, status="ok",
                       engine="mujoco_cpu", **res)
            T._append(path, row)
            print(a.task, key.get("d", key.get("x_sep_mm")), sd, res.get("held"), res.get("theta_rmse_deg"),
                  res.get("w_rmse_ratio"), f"{time.time() - t0:.1f} s", flush=True)
    return 0


MAIN = {"pinpush": "theta_rmse_deg", "press": "theta_rmse_deg", "roll": "w_rmse_ratio", "twist": "w_rmse_ratio"}


def load_rows(p):
    """Rows by (task, size, variant, seed), scored rows only, rescaled when scored before the integration fix."""
    import hom_rows
    out = {}
    for line in Path(p).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "ok":
                out[(r["task"], float(r["d_mm"]), r.get("variant", "base"), r["seed"])] = hom_rows.rescale_metrics(r)
    return out


def cmd_compare(a):
    """GPU rows against CPU rows of the same (task, size, variant, seed) on the task study's grasps: the share with the
    same held outcome, the medians of the task's score over held runs, and the rank correlation of the score run by
    run (both held)."""
    from scipy.stats import spearmanr
    cpu, gpu = load_rows(a.cpu), load_rows(a.gpu)
    common = sorted(set(cpu) & set(gpu))
    print(f"{len(common)} rollouts in both")
    by = {}
    for k in common:
        by.setdefault((k[0], k[2]), []).append(k)
    for (task, var), ks in sorted(by.items()):
        m = MAIN[task]
        same = np.mean([bool(cpu[k].get("held")) == bool(gpu[k].get("held")) for k in ks])
        hc = [cpu[k][m] for k in ks if cpu[k].get("held") and cpu[k].get(m) is not None]
        hg = [gpu[k][m] for k in ks if gpu[k].get("held") and gpu[k].get(m) is not None]
        both = [(cpu[k][m], gpu[k][m]) for k in ks if cpu[k].get("held") and gpu[k].get("held")
                and cpu[k].get(m) is not None and gpu[k].get(m) is not None]
        rho = spearmanr(*np.array(both).T)[0] if len(both) >= 4 else float("nan")
        line = (f"{task:8s} {var:10s} n {len(ks):3d} held cpu {sum(bool(cpu[k].get('held')) for k in ks):3d} "
                f"gpu {sum(bool(gpu[k].get('held')) for k in ks):3d} same {same:.2f} | {m} median cpu "
                f"{np.median(hc) if hc else float('nan'):.2f} gpu {np.median(hg) if hg else float('nan'):.2f} rho {rho:.2f}")
        if task == "twist":
            rc = [cpu[k].get("ramp_three_pads_max_deg") for k in ks if cpu[k].get("held")]
            rg = [gpu[k].get("ramp_three_pads_max_deg") for k in ks if gpu[k].get("held")]
            line += f" | ramp cpu {np.median([x for x in rc if x is not None]):.0f} gpu {np.median([x for x in rg if x is not None]):.0f}"
        print(line)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--task", nargs="+", choices=TASKS, default=["twist"])
    p.add_argument("--layouts", default="2d", help="design, 2d, 2dfp, rrt6d, sobol8k, or sobol8k_size or grid2d_size with --d")
    p.add_argument("--d", type=float, nargs="+", help="object sizes (mm); default the toy's (layouts) or the study's (design)")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--variants", nargs="+", default=["base"], choices=sorted(T.VARIANTS))
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--gen-seed", type=int, default=0, help="seed of the torch generator of the per-period noise")
    p.add_argument("--out", default="hom_gpu")
    p.add_argument("--rows-dir", default="")
    p.add_argument("--append", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("--cpu", required=True)
    p.add_argument("--gpu", required=True)
    p = sub.add_parser("cpu-check")
    p.add_argument("--task", choices=TASKS, default="pinpush")
    p.add_argument("--layouts", default="2dfp")
    p.add_argument("--d", type=float, default=None)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--variant", default="base")
    p.add_argument("--out", default="cpu_check")
    p.add_argument("--rows-dir", default="")
    sub.add_parser("check-mounts")
    a = ap.parse_args()
    if a.cmd == "check-mounts":
        check_mounts()
        return 0
    if a.cmd == "compare":
        return cmd_compare(a)
    if a.cmd == "cpu-check":
        return cmd_cpu_check(a)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
