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
            out.append((task, None, f"{rows_dir.name} #{k + 1}, tracking {score:.2f}", call, toy[call]["d"], v, None))
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
    todo = picks()
    if a.with_sv55db:
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
        res, _tr, _ = T.rollout(task, setup, plant, mirror, q_open, seed, {**T.PRM, **prm},
                                replace(H.HomParams(dt=T.DT_C), **over), None, noise,
                                T.PLANT_VARIANTS.get(var, {}).get("hold_s", 0.0))
    finally:
        T.T_LIFT = LIFT_T0
    row = {"task": task, "xy": list(xy), "role": role, "call": call, "pick": "S3", "config": config, "seed": seed,
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
    path = ROWS / "rollouts.jsonl"
    done = done_keys(path)
    stem = json.loads((ROWS / "stem.json").read_text())["stem_mm"]
    jobs = [(p, c, s) for p in picks() for c in a.configs for s in a.seeds
            if (p[0], p[3], c, s) not in done and (not a.tasks or p[0] in a.tasks)]
    jobs = jobs[a.part::a.parts]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stem")
    p.add_argument("--with-sv55db", action="store_true", help="also the measured servo's best layouts (GRID_SV55DB)")
    p.add_argument("--out", default="stem.json")
    p = sub.add_parser("run")
    p.add_argument("--configs", nargs="+", default=["base", "built"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--tasks", nargs="*", default=[])
    p.add_argument("--lift", type=float, default=40.0, help="the arm's lift at 1.6 s (mm)")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    a = ap.parse_args()
    if S.PLANT != S.SERVO_PLANT:
        raise SystemExit("run with HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25")
    return cmd_stem(a) if a.cmd == "stem" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
