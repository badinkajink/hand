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


def main():
    import mujoco
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
