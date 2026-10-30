#!/usr/bin/env python3
"""Finger-to-finger clearance of every grasp of the grid's grasp sets at its touch pose, with the v4b fingertip's pads and
rigid sphere packs (2026-10-30).

The pose toy (scripts/fp_sr2_toy.py) keeps the fingers apart by the closed-form finger skeleton (K.finger_clearance) and
the v4b parts clear of the object (rigid_clearance), not the v4b parts of two fingers clear of each other. The GPU
evaluator lets the pads and the rigid spheres under 2 mm meet the object only, so a grasp whose index and middle caps
overlap scores there as if they did not (the twist's top layouts: scripts/hom_top_cpu_check.py). For each grasp this
script solves the touch pose in the model's joints (hom_tasks.open_pose), places the plant model (pads scene) there with
the object removed, and takes the smallest surface distance between the collidable spheres of each pair of fingers.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/hom_grasp_overlap.py twist 40 [--part 0 --parts 4]
    .venv/bin/python scripts/hom_grasp_overlap.py export          # the spheres of fp_sr2_toy.ff_clearance_v4b
    .venv/bin/python scripts/hom_grasp_overlap.py check twist 40 200   # that check against this script's, 200 grasps
Rows: docs/experiments/20261029-foundational_pose_review/rows/grid2d_v4b/overlap/<task>_d<d>[_p<part>].jsonl.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_gpu as HG  # noqa: E402
import hom_landscape as L  # noqa: E402

T, S = HG.T, HG.S
FINGERS = HG.FINGERS
PAIRS = (("thumb", "index"), ("thumb", "middle"), ("index", "middle"))
OUT = L.GRID_ROWS / "overlap"


def finger_spheres(m):
    """Per finger: the collidable sphere geoms of its bodies (pads, rigid packs, link spheres)."""
    import mujoco
    out = {f: [] for f in FINGERS}
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        bn = m.body(m.geom_bodyid[g]).name
        for f in FINGERS:
            if bn.startswith(f + "_"):
                out[f].append(g)
    return {f: np.array(v) for f, v in out.items()}


def gap(ca, ra, cb, rb, k=8):
    """Smallest surface distance between two sets of spheres (m): k nearest centres of b to each centre of a."""
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(cb).query(ca, k=min(k, len(cb)))
    return float(np.min(dist - ra[:, None] - rb[idx]))


def touch_gaps(scene, m, d, setup, mxy, sph=None):
    """Smallest surface distance (mm) between each pair of fingers' collidable spheres at the grasp's touch pose, with the
    object out of the way; m, d: the pads model of the scene and its data."""
    import mujoco
    sph = sph if sph is not None else finger_spheres(m)
    rad = {f: m.geom_size[sph[f], 0] for f in FINGERS}
    bm = scene.body_mounts(mxy)
    for f in FINGERS:
        scene.m_pt.body_pos[scene.mount_bid[f]] = bm[f]
        m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    _q_open, q_touch = T.open_pose(setup, scene.m_pt)
    S.set_palm(m, setup["palm"])
    tb = m.body(S.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    d.qpos[:] = m.qpos0
    d.qpos[qa:qa + 3] = [0.0, 0.0, 10.0]                         # the object out of the way
    for f in FINGERS:
        d.qpos[[m.jnt_qposadr[m.joint(f"{f}_{j}").id] for j in T.JOINTS]] = q_touch[f]
    mujoco.mj_kinematics(m, d)
    c = {f: d.geom_xpos[sph[f]] for f in FINGERS}
    return {f"{p}-{q}": round(1e3 * min(gap(c[p], rad[p], c[q], rad[q]), gap(c[q], rad[q], c[p], rad[p])), 3)
            for p, q in PAIRS}


def export_spheres():
    """Writes fp_sr2_toy.FF_SPHERES, each finger's collidable spheres per body of the pads model (centres in the body
    frame, radii), and checks fp_sr2_toy.finger_frames against MuJoCo's body poses at 50 random sets of joint angles."""
    import mujoco
    import fp_sr2_toy as FP
    scene = HG.Scene("sphere", 40.0)
    m = mujoco.MjModel.from_xml_path(str(scene.pads_path))
    out = {}
    for f in FINGERS:
        out[f] = {}
        for b in FP.FF_BODIES:
            gs = [g for g in np.flatnonzero(m.geom_bodyid == m.body(f"{f}_{b}").id)
                  if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and (m.geom_contype[g] or m.geom_conaffinity[g])]
            out[f][b] = {"c": m.geom_pos[gs].round(7).tolist(), "r": m.geom_size[gs, 0].round(7).tolist()}
    FP.FF_SPHERES.write_text(json.dumps({"source": str(scene.pads_path.relative_to(ROOT)), "fingers": out}))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    palm = m.body("palm_pose").id
    err = 0.0
    for _ in range(50):
        qs = {f: rng.uniform(FP.LIM[f][:, 0], FP.LIM[f][:, 1]) for f in FINGERS}
        for f in FINGERS:
            d.qpos[[m.jnt_qposadr[m.joint(f"{f}_{j}").id] for j in T.JOINTS]] = qs[f]
        mujoco.mj_kinematics(m, d)
        Rp = d.xmat[palm].reshape(3, 3)
        for f in FINGERS:
            for b, (R, p) in FP.finger_frames(f, qs[f], m.body_pos[m.body(f"{f}_mount").id]).items():
                i = m.body(f"{f}_{b}").id
                err = max(err, float(np.abs(Rp.T @ d.xmat[i].reshape(3, 3) - R).max()),
                          float(np.abs(Rp.T @ (d.xpos[i] - d.xpos[palm]) - p).max()))
    n = {f: sum(len(v["r"]) for v in out[f].values()) for f in FINGERS}
    print(f"{FP.FF_SPHERES.relative_to(ROOT)}: spheres per finger {n}; finger_frames against MuJoCo, largest error {err:.2e}",
          flush=True)


def check_toy(task, size, n):
    """The toy's touch-pose clearance (fp_sr2_toy.ff_clearance_v4b) against touch_gaps on the plant for n grasps of the
    grid's grasp sets (every k-th): the largest difference and the counts on each side of FF_CLEAR."""
    import mujoco
    import fp_sr2_toy as FP
    scene = HG.Scene(T.SHAPE[task], size)
    m = mujoco.MjModel.from_xml_path(str(scene.pads_path))
    d = mujoco.MjData(m)
    sph = finger_spheres(m)
    lays = HG.layout_list(task, "grid2d_size", 0, size)
    poses = L.grid_poses(task, size)
    fp = L.POSE_KEY[task]
    out = []
    for key, mxy, setup in lays[::max(1, len(lays) // n)]:
        g = poses[int(key["call"])][2]["grasps"][int(key.get("grasp", 0))]
        M = FP.mounts6(key["d"])
        _obj, tg, nn, pc, _ax = FP.geometry(fp, np.asarray(g["v"], float), M)
        qt = {f: FP.touch_q(f, pc[f], nn[f], M[f], g["branch"][f],
                            FP.finger_state(f, tg[f], M[f], g["branch"][f])[0]) for f in FINGERS}
        out.append((1e3 * FP.ff_clearance_v4b(qt, M), min(touch_gaps(scene, m, d, setup, mxy, sph).values())))
    a = np.array(out)
    near = a[:, 1] < 1e3 * FP.FF_CAP
    c = 1e3 * FP.FF_CLEAR
    res = {"grasps": len(a), "under_cap": int(near.sum()), "median_diff_mm": float(np.median(a[near, 0] - a[near, 1])),
           "max_abs_diff_mm": float(np.abs(a[near, 0] - a[near, 1]).max()),
           "both_under": int(((a[:, 0] < c) & (a[:, 1] < c)).sum()), "toy_only": int(((a[:, 0] < c) & (a[:, 1] >= c)).sum()),
           "plant_only": int(((a[:, 0] >= c) & (a[:, 1] < c)).sum()), "clear_mm": c}
    jp = HG.ROWS / "grid2dff_v4b" / "toy_check.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    allr = json.loads(jp.read_text()) if jp.exists() else {}
    allr[f"{task}_d{size:g}"] = res
    jp.write_text(json.dumps(allr, indent=1))
    print(f"{task} {size:g} mm, {len(a)} grasps: toy minus plant, median {np.median(a[near, 0] - a[near, 1]):+.3f} mm, "
          f"largest |.| {np.abs(a[near, 0] - a[near, 1]).max():.3f} mm over {near.sum()} grasps under the cap; "
          f"both under {c:g} mm {int(((a[:, 0] < c) & (a[:, 1] < c)).sum())}, only the toy "
          f"{int(((a[:, 0] < c) & (a[:, 1] >= c)).sum())}, only the plant {int(((a[:, 0] >= c) & (a[:, 1] < c)).sum())}",
          flush=True)


def main():
    import mujoco
    if sys.argv[1:2] == ["export"]:
        return export_spheres()
    if sys.argv[1:2] == ["check"]:
        return check_toy(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 200)
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("d", type=float)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--layouts", default="grid2d_size", help="a layout set of hom_gpu.layout_list (sobol8k_size: the "
                    "Sobol layouts' foundational poses)")
    a = ap.parse_args()
    scene = HG.Scene(T.SHAPE[a.task], a.d)
    m = mujoco.MjModel.from_xml_path(str(scene.pads_path))
    d = mujoco.MjData(m)
    sph = finger_spheres(m)
    lays = HG.layout_list(a.task, a.layouts, 0, a.d)[a.part::a.parts]
    poses = L.grid_poses(a.task, a.d) if a.layouts == "grid2d_size" else {}
    OUT.mkdir(parents=True, exist_ok=True)
    pre = "" if a.layouts == "grid2d_size" else a.layouts.replace("_size", "") + "_"
    path = OUT / (f"{pre}{a.task}_d{a.d:g}" + (f"_p{a.part}" if a.parts > 1 else "") + ".jsonl")
    path.unlink(missing_ok=True)
    t0 = time.time()
    with open(path, "a") as fh:
        for i, (key, mxy, setup) in enumerate(lays):
            cl = touch_gaps(scene, m, d, setup, mxy, sph)
            xs, ys, _rec = poses.get(int(key["call"]), (None, None, None))
            row = {"task": a.task, "d_mm": a.d, "call": int(key["call"]), "grasp": int(key.get("grasp", 0)), "x_sep": xs,
                   "y_sep": ys, "d": key.get("d"), "clear_mm": cl, "min_mm": min(cl.values())}
            fh.write(json.dumps(row) + "\n")
            if i % 200 == 0:
                fh.flush()
                os.fsync(fh.fileno())
                print(f"{a.task} {a.d:g} mm part {a.part}: {i + 1}/{len(lays)} in {time.time() - t0:.0f} s", flush=True)
        fh.flush()
        os.fsync(fh.fileno())
    print(f"{a.task} {a.d:g} mm part {a.part}: {len(lays)} grasps in {time.time() - t0:.0f} s -> {path.relative_to(ROOT)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
