#!/usr/bin/env python3
"""Step 2 of the object seating and tracking plan (docs/notes/20261105-object_seating_and_tracking/), the physics: the
HOM tasks of the five layouts of the task-conflict study (generalist (70,110); specialists twist (100,70), roll
(85,170), pin-push (75,55), press (70,150)), each with its S3 pick (grasp, forces and contact modes,
docs/experiments/20261103-hom_task_conflict/rows_refine/sv25), on the SCS0009 plant with pads at 2.5 MPa, CPU MuJoCo,
seeds 20-24, in two configurations:

  base   the study's scene and protocol: the 3 mm post under the object, the cylinder held at its start pose until
         0.8 s (variant hold), the post removed at 1.6 s;
  built  the object as built (scripts/hardware/object_tracking_mjcf.py): stem and tag block, 25 g with no ballast, its
         block resting in the fixed seat's pocket with no hold; at 1.6 s the arm lifts the hand by --lift mm in 0.4 s
         on a minimum-jerk profile (owner, 2026-11-05: "just move the ur5 up more"), simulated in the hand's frame as
         the seat moving down while gravity grows by the hand's upward acceleration.

Per rollout: the task score (hom_landscape.rollout_score), the smallest distance between the stem or block and any
finger geom (contacts within 10 mm reported through the added geoms' margin and gap, inactive until they touch), the
height of the block's lowest corner above the seat's pocket after 2.0 s, and the trajectory (qpos and the seat's pose
at every control period) for the camera checks of scripts/hardware/object_tracking_sim.py.

    HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 CUDA_VISIBLE_DEVICES= \\
      ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- .venv/bin/python scripts/hardware/object_tracking_rollouts.py stem
    ... object_tracking_rollouts.py run [--configs base built] [--seeds 20 21 22 23 24] [--part 0 --parts 4]
Rows: docs/experiments/20261105-object_tracking_sim/rows/rollouts.jsonl (fsynced, resumable); stem lengths
rows/stem.json; trajectories logs/20261105-object_tracking_sim/traj/.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("MUJOCO_GL", "egl")

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths  # noqa: E402,F401
import fp_sr2_toy as FP  # noqa: E402
import hom_gpu as HG  # noqa: E402
import hom_grasp_refine as R  # noqa: E402
import hom_landscape as L  # noqa: E402
import object_tracking_mjcf as OM  # noqa: E402
from sr2hand.geometry.rotations import quat_apply, quat_conj, quat_mul  # noqa: E402

T, H, S = HG.T, HG.H, HG.S
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/20261105-object_tracking_sim"
ROWS = OUT / "rows"
LOGS = ROOT / "logs/20261105-object_tracking_sim"
TRAJ = LOGS / "traj"
R.ROWS_ROOT = ROOT / "docs/experiments/20261103-hom_task_conflict/rows_refine"
LAYOUTS = {(70.0, 110.0): "generalist", (100.0, 70.0): "twist", (85.0, 170.0): "roll", (75.0, 55.0): "pinpush",
           (70.0, 150.0): "press"}
TASKS = ("pinpush", "press", "roll", "twist")
SHAPE = {t: ("sphere" if R.POSE[t] == "tripod" else "cylinder") for t in TASKS}
SEEDS = (20, 21, 22, 23, 24)
# The measured SCS0009 (post-reset step 1, 2026-11-10: kp 0.55 N m/rad and the 0.59 deg dead band) reorders the layouts;
# its grids (press, roll, twist; no pin-push) give the second set the stems must clear (owner, 2026-11-12: "Both sets").
GRID_SV55DB = ROOT / "docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b_sv55db"
GRID_TOP = 5                    # layouts per task by tracking score
# The larger objects (owner, 2026-11-17; object_tracking_mjcf.OBJECT_SETS["large"]): --set large runs the ten best
# layouts per task of the HOM grids at the size-ranking test's large sizes on the same SCS0009 plant, 67 mm spheres and
# 45 mm cylinders, with their best grasps, into rows/stem_large.json, rows/rollouts_large.jsonl and traj/large/.
GRID_SV25R = ROOT / "docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b_sv25r"
LARGE_D = {"twist": 67.0, "roll": 67.0, "pinpush": 45.0, "press": 45.0}
LARGE_TOP = 10
SET = "20261112"


def use_set(name):
    """Select an object set: its sizes in hom_grasp_refine.TASKS (read by prepare and grid_picks), its objects and block
    ids in object_tracking_mjcf, and the rows' and trajectories' names."""
    global SET, TRAJ
    SET = name
    OM.select(name)
    if name == "large":
        R.TASKS.update(LARGE_D)
        TRAJ = LOGS / "traj" / "large"


def suffix():
    return "" if SET == "20261112" else f"_{SET}"


def set_picks():
    """The picks of the selected set: the plan's five layouts, or the large grids' best layouts."""
    return picks() if SET == "20261112" else grid_picks(GRID_SV25R, LARGE_TOP)
BLOCK_BELOW_TIP = 15.0          # mm: the block's top below the lowest finger point at the grasp (plan)
CLEAR_RANGE = 0.010             # m: the added geoms' margin and gap


def picks():
    """[(task, layout xy, role, call, d6, v, prm)] of the five layouts' S3 picks."""
    out = []
    for task in TASKS:
        lays = {tuple(L_["xy"]): L_ for L_ in R.layouts(task, "sv25")}
        p3 = R.picks(task, "sv25", "full")
        for xy, role in LAYOUTS.items():
            L_ = lays[xy]
            p = p3[L_["call"]]
            out.append((task, xy, role, L_["call"], L_["d"], p["v"], p["prm"]))
    return out


def grid_picks(rows_dir=GRID_SV55DB, top=GRID_TOP):
    """[(task, None, role, call, d6, v, None)] of a grid's best layouts: per task with grid rows, the `top` layouts by
    mean tracking score over seeds (hom_landscape.track_score), each with its best grasp, whose vector comes from the
    toy's pose rows of that object size (fp_sr2_toy.size_name, the grid's layouts `grid2dff`)."""
    out = []
    for task in TASKS:
        p = rows_dir / f"{task}_d{R.TASKS[task]:g}.jsonl"
        if not p.exists():
            continue
        acc = {}
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                acc.setdefault((int(r["call"]), int(r.get("grasp", 0))), []).append(L.track_score(task, r))
        best = {}
        for (call, g), sc in acc.items():
            if call not in best or np.mean(sc) > best[call][0]:
                best[call] = (float(np.mean(sc)), g)
        toy_path = HG.ROWS / f"{FP.size_name('grid2dff', SHAPE[task], R.TASKS[task])}.jsonl"
        toy = {int(r["call"]): r for r in (json.loads(x) for x in toy_path.read_text().splitlines() if x.strip())}
        for k, (call, (score, g)) in enumerate(sorted(best.items(), key=lambda kv: -kv[1][0])[:top]):
            rec = toy[call]["fp"][HG.POSE[task]]
            v = (rec.get("grasps") or [rec])[g]["v"]
            out.append((task, tuple(toy[call]["xy"]), f"{rows_dir.name} #{k + 1}, tracking {score:.2f}", call,
                        toy[call]["d"], v, None))
    return out


def prepare(task, d6, v):
    import mujoco
    FP.set_sizes(**({"sph_d_mm": R.TASKS[task]} if R.POSE[task] == "tripod" else {"cyl_d_mm": R.TASKS[task]}))
    setup = R.setup_of(HG, task, d6, np.asarray(v, float))
    mxy = HG.mount_xy(d6)
    scene = HG.Scene(SHAPE[task], R.TASKS[task])
    bm, q_open = scene.prepare(mxy, setup)
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for f in HG.FINGERS:
        m_pt.body_pos[m_pt.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    return setup, scene, bm, q_open, m_pt


def finger_geoms(m):
    return np.array([any(m.body(m.geom_bodyid[g]).name.startswith(f + "_") for f in HG.FINGERS) for g in range(m.ngeom)])


def lowest_points(m, d, sel):
    """Lowest world z (m) of each selected geom: mesh vertices, sphere, capsule and cylinder ends, box corners."""
    import mujoco
    out = np.full(m.ngeom, np.inf)
    for g in np.flatnonzero(sel):
        R_, p, sz, ty = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g], m.geom_type[g]
        if ty == mujoco.mjtGeom.mjGEOM_MESH:
            k = m.geom_dataid[g]
            V = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            out[g] = (V @ R_.T + p)[:, 2].min()
        elif ty == mujoco.mjtGeom.mjGEOM_SPHERE:
            out[g] = p[2] - sz[0]
        elif ty in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            ax = R_[:, 2] * sz[1]
            out[g] = min(p[2] + ax[2], p[2] - ax[2]) - sz[0] * (1.0 if ty == mujoco.mjtGeom.mjGEOM_CAPSULE
                                                                 else math.sqrt(max(0.0, 1 - R_[2, 2] ** 2)))
        elif ty == mujoco.mjtGeom.mjGEOM_BOX:
            c = np.array([[sx, sy, sz_] for sx in (-1, 1) for sy in (-1, 1) for sz_ in (-1, 1)]) * sz[:3]
            out[g] = (c @ R_.T + p)[:, 2].min()
        elif ty == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            out[g] = p[2] - np.linalg.norm(R_[2, :] * sz[:3])
    return out


def cmd_stem(a):
    """The lowest finger point below the object's lowest point at each pick's open and touch poses; the stem of each
    shape puts the block's top BLOCK_BELOW_TIP below the lowest over the five layouts and the shape's two tasks."""
    import mujoco
    res = {"rows": [], "stem_mm": {}, "sets": ["the plan's five layouts (sv25 S3 picks)"]}
    todo = set_picks()
    if SET != "20261112":
        res["sets"] = [f"{GRID_SV25R.name} at {LARGE_D}: top {LARGE_TOP} per task by tracking score"]
    elif a.with_sv55db:
        todo += grid_picks()
        res["sets"].append(f"{GRID_SV55DB.name}: top {GRID_TOP} per task by tracking score")
    for task, xy, role, call, d6, v, prm in todo:
        setup, scene, bm, q_open, m_pt = prepare(task, d6, v)
        q_open2, q_touch = T.open_pose(setup, m_pt)
        d = mujoco.MjData(m_pt)
        tb = m_pt.body(S.OBJ).id
        qa = m_pt.jnt_qposadr[m_pt.body_jntadr[tb]]
        sel = finger_geoms(m_pt) & ((m_pt.geom_contype != 0) | (m_pt.geom_conaffinity != 0) | (m_pt.geom_group <= 2))
        row = {"task": task, "xy": xy, "role": role, "call": call, "d6": list(map(float, d6))}
        for name, q in (("open", q_open), ("touch", q_touch)):
            d.qpos[qa:qa + 7] = [0.0, 0.0, S.Z_OBJ, *setup["quat"]]
            for f in HG.FINGERS:
                for k, j in enumerate(HG.JOINTS):
                    d.qpos[m_pt.jnt_qposadr[m_pt.joint(f"{f}_{j}").id]] = q[f][k]
            mujoco.mj_forward(m_pt, d)
            low = lowest_points(m_pt, d, sel).min()
            row[f"below_bottom_{name}_mm"] = round(1e3 * ((S.Z_OBJ - setup["r"]) - low), 2)
        res["rows"].append(row)
        print(row, flush=True)
    for shape in ("cylinder", "sphere"):
        worst = max(max(r["below_bottom_open_mm"], r["below_bottom_touch_mm"]) for r in res["rows"]
                    if SHAPE[r["task"]] == shape)
        res["stem_mm"][shape] = round(max(0.0, worst) + BLOCK_BELOW_TIP, 1)
    ROWS.mkdir(parents=True, exist_ok=True)
    (ROWS / a.out).write_text(json.dumps(res, indent=1))
    print("stem", res["stem_mm"], "->", ROWS / a.out)
    return 0


def built_scene(scene, shape, r_mm, stem_mm):
    """The reference layout's pads scene with the stem, block and seat, written beside it (relative mesh paths)."""
    src = Path(scene.pads_path)
    dst = src.with_name(f"{src.stem}__otrack_s{stem_mm:g}.xml")
    if dst.exists():
        return dst
    root = ET.parse(src).getroot()
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    gl = vis.find("global")
    if gl is None:
        gl = ET.SubElement(vis, "global")
    gl.set("offwidth", "3840")
    gl.set("offheight", "2160")
    names = OM.add_assembly(root, S.OBJ, shape, r_mm, stem_mm)
    for g in root.iter("geom"):
        if g.get("name") in names:
            g.set("margin", f"{CLEAR_RANGE:g}")
            g.set("gap", f"{CLEAR_RANGE:g}")
    OM.add_seat(root, shape, (0.0, 0.0, 0.0))
    tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")
    ET.ElementTree(root).write(tmp)
    os.replace(tmp, dst)
    return dst


def minjerk(tau):
    """Position share and its second derivative in tau of a minimum-jerk move, tau in [0, 1]."""
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5, 60 * tau - 180 * tau ** 2 + 120 * tau ** 3


class Recorder:
    """qpos and the seat's pose after every control period; the stem's and block's clearance to the fingers and the
    block's height over the seat's pocket."""

    def __init__(self, plant, built, r_mm=None, stem_mm=None):
        m = plant.m
        self.plant, self.built = plant, built
        self.qpos, self.t, self.seat = [], [], []
        self.fing = finger_geoms(m)
        self.clear = []
        if built:
            self.add = np.zeros(m.ngeom, bool)
            for nm in ("otrack_stem", "otrack_block"):
                self.add[m.geom(nm).id] = True
            self.verts = OM.BLOCK.vertices()[:6] / 1000.0 @ OM.R_OB.T + OM.block_centre_O(r_mm, stem_mm) / 1000.0
            self.fname = {g: next((f for f in HG.FINGERS if m.body(m.geom_bodyid[g]).name.startswith(f + "_")), None)
                          for g in np.flatnonzero(self.fing)}

    def __call__(self):
        p, d = self.plant, self.plant.d
        self.t.append(float(d.time))
        self.qpos.append(d.qpos.copy())
        if not self.built:
            return
        self.seat.append(np.r_[d.mocap_pos[p.seat_mid], d.mocap_quat[p.seat_mid]])
        n = d.ncon
        cl = (CLEAR_RANGE, None)
        if n:
            g = d.contact.geom[:n]
            dist = d.contact.dist[:n]
            hit = (self.add[g[:, 0]] & self.fing[g[:, 1]]) | (self.add[g[:, 1]] & self.fing[g[:, 0]])
            if hit.any():
                k = int(np.argmin(np.where(hit, dist, np.inf)))
                other = g[k, 1] if self.add[g[k, 0]] else g[k, 0]
                cl = (float(dist[k]), self.fname.get(int(other)))
        pos, quat = d.qpos[p.qa:p.qa + 3], d.qpos[p.qa + 3:p.qa + 7]
        low = float((quat_apply(quat, self.verts) + pos)[:, 2].min())
        top = float(d.mocap_pos[p.seat_mid][2]) + OM.POCKET_DEPTH / 1000.0
        self.clear.append((cl[0], cl[1], low - top))


LIFT_T0 = 1.6                   # s: the arm's lift starts when the simulation removes its post (hom_tasks.T_LIFT)
# config -> seat in place, the arm's lift, the object held at its start pose up to this time (s), the lift's duration
# (s), its acceleration felt as gravity, and the delay of the task's start after the study's 2.0 s (s)
CONFIGS = {
    "built": dict(seat=True, lift=True, hold_s=0.0, lift_s=0.4, accel=True, shift=0.0),
    "built_slow": dict(seat=True, lift=True, hold_s=0.0, lift_s=1.0, accel=True, shift=0.6),
    "built_hold": dict(seat=True, lift=True, hold_s=0.8, lift_s=0.4, accel=True, shift=0.0),
    "built_hold_noacc": dict(seat=True, lift=True, hold_s=0.8, lift_s=0.4, accel=False, shift=0.0),
    "stem": dict(seat=False, lift=False, hold_s=1.6, lift_s=0.4, accel=False, shift=0.0),
    "stem_acc": dict(seat=False, lift=True, hold_s=1.6, lift_s=0.4, accel=True, shift=0.0),
    "flat": dict(seat="flat", lift=True, hold_s=0.0, lift_s=0.4, accel=True, shift=0.0),
    "flat_hold": dict(seat="flat", lift=True, hold_s=0.8, lift_s=0.4, accel=True, shift=0.0),
}


def make_built_plant(path, setup, shape, r_mm, stem_mm, lift_mm, seat=True, lift=True, hold_s=0.0, lift_s=0.4,
                     accel=True, shift=0.0):
    class BuiltPlant(T.Plant):
        def __init__(self):
            super().__init__(path, setup)
            m = self.m
            self.post0 = (0, 0)
            for g in range(m.ngeom):          # seat False: no seat; "flat": its floor alone, without the pocket
                if m.geom_bodyid[g] == m.body("otrack_seat").id and (
                        not seat or (seat == "flat" and m.geom(g).name.startswith("otrack_wall"))):
                    m.geom_contype[g] = m.geom_conaffinity[g] = 0
            self.seat_mid = int(m.body_mocapid[m.body("otrack_seat").id])
            self.g0 = m.opt.gravity.copy()
            self.bottom_O = OM.block_bottom_O(r_mm, stem_mm) / 1000.0
            self.lift = lift_mm / 1000.0 if lift else 0.0
            self.rec = None
            self.hold_until = hold_s

        def reset(self, q0, seed):
            super().reset(q0, seed)
            d = self.d
            pos, quat = self.obj0[:3], self.obj0[3:]
            yq = quat_mul(quat, quat_conj(np.asarray(setup["quat"], float)))     # the start's yaw jitter
            self.seat0 = pos + quat_apply(quat, self.bottom_O) - np.array([0.0, 0.0, 5e-5])
            d.mocap_pos[self.seat_mid] = self.seat0
            d.mocap_quat[self.seat_mid] = yq
            self.m.opt.gravity[:] = self.g0
            self.mj.mj_forward(self.m, d)

        def advance(self, t1):
            m, d = self.m, self.d
            while d.time < t1 - 1e-9:
                tau = (d.time - LIFT_T0) / lift_s
                if 0.0 <= tau <= 1.0:
                    s, acc = minjerk(tau)
                    d.mocap_pos[self.seat_mid, 2] = self.seat0[2] - self.lift * s
                    if accel:
                        m.opt.gravity[2] = self.g0[2] - self.lift * acc / lift_s ** 2
                elif tau > 1.0:
                    d.mocap_pos[self.seat_mid, 2] = self.seat0[2] - self.lift
                    m.opt.gravity[:] = self.g0
                self.mj.mj_step(m, d)
            if self.hold_until and d.time <= self.hold_until + 1e-9:
                d.qpos[self.qa:self.qa + 7] = self.obj0
                d.qvel[self.va:self.va + 6] = 0.0
                self.mj.mj_forward(m, d)
            if self.rec is not None:
                self.rec()
    return BuiltPlant()


def run_one(task, xy, role, call, d6, v, prm, config, seed, stem_mm, lift_mm):
    import mujoco
    setup, scene, bm, q_open, m_pt = prepare(task, d6, v)
    shape = SHAPE[task]
    r_mm = 1000.0 * setup["r"]
    if config == "base":
        plant = T.Plant(scene.pads_path, setup)
        var = "hold" if shape == "cylinder" else "base"
    else:
        path = built_scene(scene, shape, r_mm, stem_mm)
        plant = make_built_plant(path, setup, shape, r_mm, stem_mm, lift_mm, **CONFIGS[config])
        var = "base"
    for f in HG.FINGERS:
        plant.m.body_pos[plant.m.body(f"{f}_mount").id] = bm[f]
    fs, os_ = T.sr2_specs(setup)
    mirror = H.Mirror(m_pt, fs, os_)
    rec = Recorder(plant, config != "base", r_mm, stem_mm)
    if config == "base":
        adv = plant.advance

        def advance(t1):
            adv(t1)
            rec()
        plant.advance = advance
    else:
        plant.rec = rec
    over, noise = T.VARIANTS[var]
    t0 = time.time()
    shift = CONFIGS.get(config, {}).get("shift", 0.0)
    T.T_LIFT = LIFT_T0 + shift                     # the task's start (T_LIFT + 0.4) moves with a slower lift
    try:
        res, _tr, _ = T.rollout(task, setup, plant, mirror, q_open, seed, {**T.PRM, **(prm or {})},
                                replace(H.HomParams(dt=T.DT_C), **over), None, noise,
                                T.PLANT_VARIANTS.get(var, {}).get("hold_s", 0.0))
    finally:
        T.T_LIFT = LIFT_T0
    row = {"task": task, "xy": list(xy), "role": role, "call": call, "pick": "S3" if prm is not None else "grid best",
           "config": config, "seed": seed, "d_mm": R.TASKS[task],
           "score": round(float(L.rollout_score(task, res)), 4), "wall_s": round(time.time() - t0, 1),
           **{k: res.get(k) for k in ("held", "dropped", "fingers_end", "ramp_three_pads_max_deg", "w_rmse_ratio",
                                      "theta_rmse_deg", "drift_max_mm", "link_force_max_N")}}
    TRAJ.mkdir(parents=True, exist_ok=True)
    tp = TRAJ / f"{config}_{task}_{call}_s{seed}.npz"
    extra = {}
    if config != "base":
        C = np.array([(c, -1 if f is None else HG.FINGERS.index(f), s_) for c, f, s_ in rec.clear])
        tt = np.array(rec.t)
        k = int(np.argmin(C[:, 0]))
        row.update(stem_mm=stem_mm, lift_mm=lift_mm, **{f"cfg_{k}": v for k, v in CONFIGS[config].items()},
                   clear_min_mm=round(1e3 * C[k, 0], 2),
                   clear_finger=None if C[k, 1] < 0 else HG.FINGERS[int(C[k, 1])], clear_t=round(tt[k], 2))
        task_ = tt >= LIFT_T0 + 0.4 + shift - 1e-9
        j = int(np.argmin(np.where(task_, C[:, 2], np.inf)))
        row.update(seat_clear_min_mm=round(1e3 * C[j, 2], 2), seat_clear_t=round(tt[j], 2))
        pre = tt < LIFT_T0
        P = np.array(rec.qpos)[:, plant.qa:plant.qa + 3]
        row["seated_drift_mm"] = round(1e3 * float(np.linalg.norm(P[pre] - P[0], axis=1).max()), 3)
        extra = dict(clear=C, seat=np.array(rec.seat), scene=str(path.relative_to(ROOT)), r_mm=r_mm, stem_mm=stem_mm)
    else:
        extra = dict(scene=str(Path(scene.pads_path).relative_to(ROOT)), r_mm=r_mm)
    np.savez_compressed(tp, t=np.array(rec.t), qpos=np.array(rec.qpos), mounts=np.array([bm[f] for f in HG.FINGERS]),
                        palm=np.asarray(setup["palm"]), obj_quat0=np.asarray(setup["quat"]), **extra)
    row["traj"] = str(tp.relative_to(ROOT))
    return row


def done_keys(path):
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok":
                    out.add((r["task"], r["call"], r["config"], r["seed"]))
    return out


def cmd_run(a):
    ROWS.mkdir(parents=True, exist_ok=True)
    path = ROWS / f"rollouts{suffix()}.jsonl"
    done = done_keys(path)
    stem = json.loads((ROWS / f"stem{suffix()}.json").read_text())["stem_mm"]
    jobs = [(p, c, s) for p in set_picks() for c in a.configs for s in a.seeds if not a.tasks or p[0] in a.tasks]
    jobs = [j for j in jobs[a.part::a.parts] if (j[0][0], j[0][3], j[1], j[2]) not in done]   # parts before done
    print(f"{len(jobs)} rollouts in part {a.part} of {a.parts}", flush=True)
    for (task, xy, role, call, d6, v, prm), config, seed in jobs:
        try:
            row = run_one(task, xy, role, call, d6, v, prm, config, seed, stem[SHAPE[task]], a.lift)
            row["status"] = "ok"
        except Exception as e:
            row = {"task": task, "call": call, "config": config, "seed": seed, "status": "error",
                   "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}
        row["when"] = time.strftime("%Y-%m-%d %H:%M")
        T._append(path, row)
        print({k: row.get(k) for k in ("task", "role", "config", "seed", "score", "held", "clear_min_mm",
                                       "clear_finger", "seat_clear_min_mm", "seated_drift_mm", "wall_s", "error")},
              flush=True)
    return 0


BLOCKS_V2 = ROOT / "assets/tracking/20261116-tag_blocks/20261116-tag_blocks.json"


def cmd_rootclear(a):
    """Finger clearance of the tag blocks' second-version stem root (owner, 2026-11-16) over the recorded rollouts. The
    root is drawn by two cylinders that enclose it: ROOT_D across from the block's top up to ROOT_H, and the fillet's
    foot across up to FILLET_R. They are added to each rollout's built scene, every recorded period is set and
    collided, and contacts within CLEAR_RANGE give the distance to the nearest finger geom, as in the rollouts."""
    import mujoco
    root_cfg = json.loads(BLOCKS_V2.read_text())["root"]
    rows = [json.loads(x) for x in (ROWS / "rollouts.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("status") == "ok" and r["config"] in a.configs and (ROOT / r["traj"]).exists()]
    models, out = {}, []
    for r in rows:
        z = np.load(ROOT / r["traj"])
        r_mm, stem_mm, scene = float(z["r_mm"]), float(z["stem_mm"]), ROOT / str(z["scene"])
        key = (scene, r_mm, stem_mm)
        if key not in models:
            xml = ET.parse(scene).getroot()
            ob = xml.find(f".//body[@name='{S.OBJ}']")
            stem = xml.find(".//geom[@name='otrack_stem']")
            top = -(r_mm + stem_mm)                         # the block's top on y_O (mm); the stem runs along y_O
            for name, rad, h in (("otrack_root", root_cfg["root_d_mm"] / 2, root_cfg["root_h_mm"]),
                                 ("otrack_foot", root_cfg["foot_d_mm"] / 2, root_cfg["fillet_r_mm"])):
                ET.SubElement(ob, "geom", name=name, type="cylinder", size=f"{rad / 1000.0:g}",
                              fromto=f"0 {top / 1000.0:g} 0 0 {(top + h) / 1000.0:g} 0", mass="0", group="3",
                              contype=stem.get("contype", "1"), conaffinity=stem.get("conaffinity", "1"),
                              margin=f"{CLEAR_RANGE:g}", gap=f"{CLEAR_RANGE:g}")
            tmp = scene.with_name(f"{scene.stem}.root{os.getpid()}.xml")
            ET.ElementTree(xml).write(tmp)
            try:
                m = mujoco.MjModel.from_xml_path(str(tmp))
            finally:
                tmp.unlink()
            fing = finger_geoms(m)
            fname = {g: next(f for f in HG.FINGERS if m.body(m.geom_bodyid[g]).name.startswith(f + "_"))
                     for g in np.flatnonzero(fing)}
            ids = {nm: m.geom(nm).id for nm in ("otrack_root", "otrack_foot", "otrack_stem")}
            models[key] = (m, mujoco.MjData(m), fing, fname, ids)
        m, d, fing, fname, ids = models[key]
        S.set_palm(m, z["palm"])
        for f, p in zip(HG.FINGERS, z["mounts"]):
            m.body_pos[m.body(f"{f}_mount").id] = p
        best = {"root": (CLEAR_RANGE, None, None), "stem": (CLEAR_RANGE, None, None)}
        for t, q in zip(z["t"], z["qpos"]):
            d.qpos[:] = q
            mujoco.mj_fwdPosition(m, d)
            n = d.ncon
            if not n:
                continue
            g, dist = d.contact.geom[:n], d.contact.dist[:n]
            for part, sel in (("root", (ids["otrack_root"], ids["otrack_foot"])), ("stem", (ids["otrack_stem"],))):
                hit = (np.isin(g[:, 0], sel) & fing[g[:, 1]]) | (np.isin(g[:, 1], sel) & fing[g[:, 0]])
                if hit.any():
                    k = int(np.argmin(np.where(hit, dist, np.inf)))
                    if dist[k] < best[part][0]:
                        other = g[k, 1] if g[k, 0] in sel else g[k, 0]
                        best[part] = (float(dist[k]), fname[int(other)], float(t))
        row = {k: r[k] for k in ("task", "xy", "role", "config", "seed", "stem_mm", "clear_min_mm")}
        for part, (dist, f, t) in best.items():
            row.update({f"{part}_min_mm": round(1e3 * dist, 2), f"{part}_finger": f,
                        f"{part}_t": None if t is None else round(t, 2)})
        out.append(row)
        print({k: row[k] for k in ("task", "xy", "config", "seed", "root_min_mm", "stem_min_mm", "clear_min_mm")},
              flush=True)
    res = {"what": "finger clearance of the second-version stem root over the recorded rollouts (mm; 10 = none "
                   "within CLEAR_RANGE)", "root": root_cfg, "configs": a.configs, "rows": out}
    dst = BLOCKS_V2.with_name("20261116-root_clearance.json")
    dst.write_text(json.dumps(res, indent=1))
    print("root min", min(x["root_min_mm"] for x in out), "mm over", len(out), "rollouts ->", dst.relative_to(ROOT))
    return 0


TOUCH_MM = 0.5                  # mm: a finger geom within this of the object's surface touches it
NEAR_MM = 2.0                   # mm: a touch this close to an opening counts as near it


def opening_boxes(shape):
    """{opening: (lo, hi)} of the cotter joint's slot on each side of the socket in the joint frame J (w, v, u), open
    outward along w (tracking_objects_print.socket)."""
    import tracking_objects_print as TO
    hs, (h_in, h_far) = TO.half_diag(TO.SQ_H), TO.slot_heights()
    return {"entry": (np.array([hs, -TO.SLOT_W / 2, TO.FLOOR_U]), np.array([np.inf, TO.SLOT_W / 2, TO.FLOOR_U + h_in])),
            "far": (np.array([-np.inf, -TO.SLOT_W / 2, TO.FLOOR_U]), np.array([-hs, TO.SLOT_W / 2, TO.FLOOR_U + h_far]))}


def surface_angles(shape, P):
    """Angles (deg) of points in O on the object: the cylinder's turn from its bottom line toward +x_O and z_O (mm); the
    sphere's angle from its bottom pole and its azimuth atan2(z_O, x_O)."""
    if shape == "cylinder":
        return np.degrees(np.arctan2(P[:, 0], -P[:, 1])), P[:, 2]
    r = np.linalg.norm(P, axis=1)
    return np.degrees(np.arccos(np.clip(-P[:, 1] / r, -1, 1))), np.degrees(np.arctan2(P[:, 2], P[:, 0]))


def cmd_touchmap(a):
    """Where the fingers touch the objects over the recorded rollouts, against the cotter joint's openings (owner,
    2026-11-16: the blocks attached without glue). Every second recorded period of the built and flat rollouts is set and
    collided with the object's own geom given a margin; finger contacts within TOUCH_MM of its surface are touches. Per
    opening of the joint's slot: the rollouts with a touch within NEAR_MM of it and the nearest touch. The maps count the
    rollouts with a touch in each bin of the surface; the sphere's are those of its two layouts in the series."""
    import mujoco
    import tracking_objects_print as TO
    rows = [json.loads(x) for x in (ROWS / f"rollouts{suffix()}.jsonl").read_text().splitlines() if x.strip()]
    rows = list({(r["task"], r["call"], r["config"], r["seed"]): r for r in rows   # the last row of a rollout run twice
                 if r.get("status") == "ok" and r["config"] in a.configs and (ROOT / r["traj"]).exists()}.values())
    shape_of = {OM.OBJECTS[k]["r_mm"]: k for k in ("cylinder", "sphere")}
    bins = {"cylinder": (np.arange(-180.0, 181.0, 5.0), np.arange(-50.0, 51.0, 2.0)),
            "sphere": (np.arange(0.0, 91.0, 3.0), np.arange(-180.0, 181.0, 5.0))}
    maps = {k: np.zeros((len(b[0]) - 1, len(b[1]) - 1), int) for k, b in bins.items()}
    boxes = {k: opening_boxes(k) for k in ("cylinder", "sphere")}
    models, out, n = {}, [], {"cylinder": 0, "sphere": 0}
    for r in rows:
        z = np.load(ROOT / r["traj"])
        scene, shape = ROOT / str(z["scene"]), shape_of[float(z["r_mm"])]
        if SET == "20261112" and shape == "sphere" and tuple(r["xy"]) not in TO.SPHERE_LAYOUTS:
            continue
        if scene not in models:
            m = mujoco.MjModel.from_xml_path(str(scene))
            b = m.body(S.OBJ).id
            g0 = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == b and not m.geom(g).name)
            m.geom_margin[g0] = m.geom_gap[g0] = TOUCH_MM / 1000.0
            models[scene] = (m, mujoco.MjData(m), b, g0, finger_geoms(m))
        m, d, b, g0, fing = models[scene]
        S.set_palm(m, z["palm"])
        for f, p in zip(HG.FINGERS, z["mounts"]):
            m.body_pos[m.body(f"{f}_mount").id] = p
        pts = []
        for q in z["qpos"][::2]:
            d.qpos[:] = q
            mujoco.mj_fwdPosition(m, d)
            g = d.contact.geom[:d.ncon]
            hit = ((g[:, 0] == g0) & fing[g[:, 1]]) | ((g[:, 1] == g0) & fing[g[:, 0]])
            if hit.any():
                pts.append((d.contact.pos[:d.ncon][hit] - d.xpos[b]) @ d.xmat[b].reshape(3, 3) * 1e3)   # mm in O
        P = np.unique(np.round(np.concatenate(pts), 1), axis=0) if pts else np.zeros((0, 3))
        n[shape] += 1
        A, B_ = surface_angles(shape, P)
        maps[shape] += np.histogram2d(A, B_, bins=bins[shape])[0] > 0
        JT = TO.joint_T(shape)
        PJ = (P - JT[:, 3]) @ JT[:, :3]                      # O to J
        row = {k: r[k] for k in ("task", "xy", "config", "seed")}
        row["shape"], row["touch_points"] = shape, int(len(P))
        for name, (lo, hi) in boxes[shape].items():
            dist = np.linalg.norm(np.maximum(0.0, np.maximum(lo - PJ, PJ - hi)), axis=1)
            row[f"{name}_min_mm"] = round(float(dist.min()), 2) if len(dist) else None
        out.append(row)
        print({k: row[k] for k in ("task", "xy", "config", "seed", "touch_points", "entry_min_mm", "far_min_mm")},
              flush=True)
    openings = []
    for shape in ("cylinder", "sphere"):
        for name, (lo, hi) in boxes[shape].items():
            d_ = [x[f"{name}_min_mm"] for x in out if x["shape"] == shape and x[f"{name}_min_mm"] is not None]
            # the opening's outline on the printed surface: the slot's section where it meets the surface
            JT = TO.joint_T(shape)
            rp = TO.seat(shape)[0]
            corners = []
            for v_ in (lo[1], hi[1]):
                for u_ in (lo[2], hi[2]):
                    s = 1.0 if name == "entry" else -1.0
                    for w_ in np.linspace(0.0, 30.0, 3001):
                        p_ = JT[:, :3] @ np.array([s * w_, v_, u_]) + JT[:, 3]
                        if np.linalg.norm(p_ if shape == "sphere" else p_[:2]) >= rp:
                            corners.append(p_)
                            break
            A, B_ = surface_angles(shape, np.array(corners))
            openings.append({"shape": shape, "opening": name, "rollouts": len(d_),
                             "near": int(sum(x <= NEAR_MM for x in d_)), "min_mm": round(min(d_), 2),
                             "outline_deg": [[round(float(x), 2), round(float(y), 2)] for x, y in zip(A, B_)]})
            print(openings[-1], flush=True)
    res = {"what": "finger touches on the tracked objects over the recorded rollouts, against the cotter joint's "
                   "openings (object_tracking_rollouts.py touchmap)", "configs": a.configs, "touch_mm": TOUCH_MM,
           "near_mm": NEAR_MM, "n_rollouts": n, "openings": openings,
           "maps": {k: {"bins": [b.tolist() for b in bins[k]], "rollouts_with_touch": maps[k].tolist()} for k in maps},
           "map_axes": {"cylinder": ["turn from the bottom line toward +x_O (deg)", "z_O (mm)"],
                        "sphere": ["angle from the bottom pole (deg)", "azimuth atan2(z_O, x_O) (deg)"]},
           "rows": out}
    dst = TO.J_OUT / f"{TO.J_DATE}-opening_touches.json"
    dst.write_text(json.dumps(res, indent=1))
    print("wrote", dst.relative_to(ROOT))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument("--set", choices=sorted(OM.OBJECT_SETS), default="20261112",
                    help="the objects: those printed on 2026-11-12, or the larger ones of 2026-11-17")
    p = sub.add_parser("stem")
    p.add_argument("--with-sv55db", action="store_true", help="also the measured servo's best layouts (GRID_SV55DB)")
    p.add_argument("--out", default=None, help="default stem.json, or stem_<set>.json")
    p = sub.add_parser("run")
    p.add_argument("--configs", nargs="+", default=["base", "built"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--tasks", nargs="*", default=[])
    p.add_argument("--lift", type=float, default=40.0, help="the arm's lift at 1.6 s (mm)")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p = sub.add_parser("rootclear", help="finger clearance of the tag blocks' second-version root (2026-11-16)")
    p.add_argument("--configs", nargs="+", default=["built", "flat"])
    p = sub.add_parser("touchmap", help="finger touches on the objects against the cotter joint's openings (2026-11-16)")
    p.add_argument("--configs", nargs="+", default=["built", "flat"])
    a = ap.parse_args()
    use_set(a.set)
    if a.cmd == "stem" and a.out is None:
        a.out = f"stem{suffix()}.json"
    if a.cmd == "touchmap" and a.set != "20261112":
        import tracking_objects_print as TO
        TO.use_set(a.set)
    if a.cmd == "rootclear":
        return cmd_rootclear(a)
    if a.cmd == "touchmap":
        return cmd_touchmap(a)
    if S.PLANT != S.SERVO_PLANT:
        raise SystemExit("run with HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25")
    return cmd_stem(a) if a.cmd == "stem" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
