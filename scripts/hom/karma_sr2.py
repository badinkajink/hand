#!/usr/bin/env python3
"""KaRMA (Peticco and Agrawal, arXiv 2605.15548, 2026) on the SR2 hand's layout lattice, the kinematic inner-loop
evaluator of the inner-loop benchmark (owner, 2026-11-12: "evaluate some of the other inner loop controls we tried?
Karma, uhas?"; "show these other methods ... are both worse and slower than ... gpu parallelized HOM").

KaRMA scores a hand from its joint tree alone: a thumb-index pinch on a sphere, and the sphere-centre positions
(KaRMA-T, a reachable volume normalized by the hand's size) and orientations (KaRMA-R, the share of 228 orientation bins)
that the pinch reaches by rolling without letting go. It runs on one CPU core per hand. The published code
(github.com/mfpeticco/karma-hand-metric, commit c8d3698, in external/karma) runs unmodified in the conda environment
karma-hand-metric; this script writes its inputs and reads its outputs.

Export. The SR2 hand of the HOM grids (SCALE_HAND=cad_v2, the v4b tip as sphere pads, hom_gpu.Scene's pads model) with
its mount bodies moved to a lattice layout (hom_gpu.sym6), at zero joint angles, in the frame of the palm body
(palm_pose). Each finger becomes three revolute joints (yaw, mcp, pip) whose origins are the joint anchors and whose axes
are the joint axes in that frame, plus a fixed fingertip link at the pad point farthest from the pip anchor (the
template's method A for tip_length_m). The pair is the thumb and the index; the layouts are mirror-symmetric about the
thumb's line, so the thumb and the middle give the same score.

Variants. 'scaled' is KaRMA as published: every length of karma_config.yaml times L_ref / 200 mm, a test sphere of
20 mm at the nominal size. 'sph40' and 'sph25' set the test sphere to the tasks' objects (the 40 mm sphere of the twist
and the roll; a 25 mm sphere for the 25 mm cylinder of the pin-push and the press) and the capsule radius to the pad's
half-width, with the scaling switched off (l_ref_nominal_m = the hand's own L_ref), as karma_batch.py's 'absolute'
variant did on the real_v1 hands (scripts/archive/karma_batch.py).

    .venv/bin/python scripts/hom/karma_sr2.py one --xy 70 110 --variant sph40      # export and run one layout
    .venv/bin/python scripts/hom/karma_sr2.py batch --variant sph40 --jobs 8         # every reached layout, resumable
    .venv/bin/python scripts/hom/karma_sr2.py batch --variant sph40 --subset lattice10   # the 10 mm sub-lattice
Rows: docs/experiments/20261112-inner_loop_benchmark/rows/karma_<variant>.jsonl, one fsynced line per layout.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

KARMA = ROOT / "external/karma"
KPY = Path.home() / "miniconda3/envs/karma-hand-metric/bin/python"
OUT = ROOT / "docs/experiments/20261112-inner_loop_benchmark"
WORK = ROOT / "logs/20261112-karma_sr2"
FINGERS, JOINTS = ("thumb", "index", "middle"), ("yaw", "mcp", "pip")
PAIR = ("thumb", "index")
VARIANTS = {"scaled": None, "sph40": 0.020, "sph25": 0.0125}       # test-sphere radius (m); None = as published
SHAPE = {"sph40": ("sphere", 40.0), "sph25": ("cylinder", 25.0), "scaled": ("sphere", 40.0)}
# KaRMA's internal seed-selection pool sizes itself from the core count; one process per run (karma_batch.py's wrapper)
SERIAL_WRAPPER = '''#!/usr/bin/env python3
"""run_metric.py with KaRMA's internal seed-selection pool pinned to one process."""
import runpy

import karma.seed_selection as _ss

_ss._physical_core_count = lambda: 1
runpy.run_path("run_metric.py", run_name="__main__")
'''


def hand_frames(xs, ys, shape="sphere", d_mm=40.0):
    """The SR2 hand at layout (x_sep, y_sep) mm, zero joint angles, in the palm frame: per finger the three joint anchors
    and axes, their ranges, the fingertip point (the farthest pad point from the pip anchor) and the pad's half-width."""
    import mujoco
    import hom_gpu as HG
    sc = HG.Scene(shape, d_mm)
    m = mujoco.MjModel.from_xml_path(str(sc.pads_path))
    bm = sc.body_mounts(HG.mount_xy(HG.sym6(xs, ys)))
    for f in FINGERS:
        m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    pb = m.body("palm_pose").id
    Rp, pp = d.xmat[pb].reshape(3, 3), d.xpos[pb]
    to_palm = (lambda x: Rp.T @ (np.asarray(x) - pp))
    out = {}
    for f in FINGERS:
        jid = [m.joint(f"{f}_{j}").id for j in JOINTS]
        anc = [to_palm(d.xanchor[j]) for j in jid]
        ax = [Rp.T @ d.xaxis[j] for j in jid]
        tip = m.body(f"{f}_tip").id
        pads = [g for g in range(m.ngeom) if m.geom_bodyid[g] == tip and "_pad" in (m.geom(g).name or "")]
        if not pads:
            raise RuntimeError(f"{f}: no pad geoms on {f}_tip in {sc.pads_path}")
        P = np.array([to_palm(d.geom_xpos[g]) for g in pads])
        rad = float(np.max(m.geom_size[pads, 0]))
        dist = np.linalg.norm(P - anc[2], axis=1)
        far = P[int(np.argmax(dist))]
        u = (far - anc[2]) / np.linalg.norm(far - anc[2])
        tip_pt = far + rad * u
        lat = P - anc[2] - np.outer((P - anc[2]) @ u, u)        # pad points off the distal axis
        out[f] = {"anchors": anc, "axes": ax, "ranges": m.jnt_range[jid].tolist(), "tip": tip_pt,
                  "half_width": float(np.max(np.linalg.norm(lat, axis=1)) + rad)}
    return out


def write_urdf(H, name, path):
    inert = ('    <inertial><origin xyz="0 0 0"/><mass value="0.02"/>'
             '<inertia ixx="4e-6" ixy="0" ixz="0" iyy="4e-6" iyz="0" izz="4e-6"/></inertial>\n')
    out = [f'<?xml version="1.0"?>\n<robot name="{name}">\n', f'  <link name="palm">\n{inert}  </link>\n']
    for f in FINGERS:
        h, parent, prev = H[f], "palm", np.zeros(3)
        for i, j in enumerate(JOINTS):
            child = f"{f}_link{i + 1}"
            o, a = h["anchors"][i] - prev, h["axes"][i]
            lo, hi = h["ranges"][i]
            out.append(f'  <link name="{child}">\n{inert}  </link>\n')
            out.append(f'  <joint name="{f}_{j}" type="revolute">\n    <parent link="{parent}"/>\n'
                       f'    <child link="{child}"/>\n    <origin xyz="{o[0]:.7f} {o[1]:.7f} {o[2]:.7f}" rpy="0 0 0"/>\n'
                       f'    <axis xyz="{a[0]:.7f} {a[1]:.7f} {a[2]:.7f}"/>\n'
                       f'    <limit lower="{lo:.6f}" upper="{hi:.6f}" effort="1" velocity="6"/>\n  </joint>\n')
            parent, prev = child, h["anchors"][i]
        o = h["tip"] - prev
        out.append(f'  <link name="{f}_fingertip">\n{inert}  </link>\n')
        out.append(f'  <joint name="{f}_tip_joint" type="fixed">\n    <parent link="{parent}"/>\n'
                   f'    <child link="{f}_fingertip"/>\n    <origin xyz="{o[0]:.7f} {o[1]:.7f} {o[2]:.7f}" rpy="0 0 0"/>\n'
                   f'  </joint>\n')
    out.append("</robot>\n")
    path.write_text("".join(out))


def write_robot_yaml(H, name, urdf_rel, path):
    import yaml
    cfg = {"robot_name": name, "urdf_path": urdf_rel, "base_link": "palm",
           "fingers": {role: {"active_joints": [f"{f}_{j}" for j in JOINTS],
                              "contact_links": [f"{f}_link3", f"{f}_link2", f"{f}_link1"]}
                       for role, f in zip(("thumb", "index"), PAIR)},
           "seed": {"thumb_link": f"{PAIR[0]}_link3", "index_link": f"{PAIR[1]}_link3"},
           "links": {f"{f}_link3": {"tip_length_m": round(float(np.linalg.norm(H[f]["tip"] - H[f]["anchors"][2])), 6)}
                     for f in FINGERS}}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def lref(robot_cfg):
    """KaRMA's L_ref of an exported hand, from its own code in its own environment."""
    code = ("import sys; sys.path.insert(0, %r); from karma.lref import compute_lref; print(compute_lref(%r))"
            % (str(KARMA), str(robot_cfg)))
    r = subprocess.run([str(KPY), "-c", code], capture_output=True, text=True, cwd=str(KARMA), timeout=300)
    if r.returncode:
        raise RuntimeError(r.stderr[-400:])
    return float(r.stdout.strip().splitlines()[-1])


def run_one(xs, ys, variant, work=WORK):
    """Export layout (xs, ys) and run KaRMA once; returns the row."""
    import yaml
    name = f"sr2_x{xs:g}_y{ys:g}"
    rdir = work / variant / name
    (rdir / "urdfs").mkdir(parents=True, exist_ok=True)
    shape, d_mm = SHAPE[variant]
    H = hand_frames(xs, ys, shape, d_mm)
    write_urdf(H, name, rdir / "urdfs" / f"{name}.urdf")
    rcfg = rdir / f"robot_{name}.yaml"
    write_robot_yaml(H, name, f"urdfs/{name}.urdf", rcfg)
    metric = yaml.safe_load((KARMA / "karma_config.yaml").read_text())
    metric.update(out_dir=str(rdir.resolve()), rotation_workers=1)
    if VARIANTS[variant] is not None:
        metric.update(l_ref_nominal_m=lref(rcfg), sphere_radius_m=VARIANTS[variant],
                      link_radius_m=round(max(H[f]["half_width"] for f in PAIR), 6))
    mpath = rdir / "metric.yaml"
    mpath.write_text(yaml.safe_dump(metric))
    wrapper = KARMA / "run_metric_serial.py"
    if not wrapper.exists() or wrapper.read_text() != SERIAL_WRAPPER:
        wrapper.write_text(SERIAL_WRAPPER)
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    p = subprocess.run([str(KPY), "run_metric_serial.py", "--config", str(rcfg.resolve()), "--metric-config",
                        str(mpath.resolve()), "--rotation-workers", "1"], cwd=str(KARMA), env=env, capture_output=True,
                       text=True, timeout=3600)
    row = {"x_sep": xs, "y_sep": ys, "variant": variant, "secs": round(time.time() - t0, 1),
           "tip_length_mm": round(1e3 * float(np.linalg.norm(H["thumb"]["tip"] - H["thumb"]["anchors"][2])), 2),
           "half_width_mm": round(1e3 * max(H[f]["half_width"] for f in PAIR), 2)}
    res = rdir / "current.yaml"
    if p.returncode or not res.exists():
        row["error"] = (p.stderr or p.stdout or "no output")[-600:]
        return row
    y = yaml.safe_load(res.read_text()) or {}
    sm, ss = y.get("summary") or {}, y.get("seed_sensitivity") or {}
    row.update(karma_t=sm.get("translational_score"), karma_r=sm.get("global_rotational_score"),
               karma_s=ss.get("karma_s"), n_voxels=sm.get("n_voxels_reached"), n_states=sm.get("n_states_reached"),
               l_ref_mm=round(1e3 * float(sm.get("l_ref_m") or 0.0), 2))
    for big in ("current.pkl",):
        (rdir / big).unlink(missing_ok=True)                  # the reachable set; the summary is kept
    return row


TASKS_OF = {"sph40": ("twist", "roll"), "scaled": ("twist", "roll"), "sph25": ("pinpush", "press")}


def lattice_layouts(variant, subset="all"):
    """The lattice layouts that reach the poses of the variant's tasks on the HOM grids (layouts with rollouts on the J
    grids). subset 'lattice10': the layouts on a 10 mm sub-lattice (x_sep 40, 50, ..., y_sep 50, 60, ...) and the ten best
    by J of each task."""
    import hom_search_qd as Q
    keys, top = set(), set()
    for task in TASKS_OF[variant]:
        look = Q.lookup(task, 40.0 if task in ("twist", "roll") else 25.0)[0]
        run = {k: f for k, (f, n) in look.items() if n > 0}
        keys |= set(run)
        top |= set(sorted(run, key=run.get, reverse=True)[:10])
    if subset == "lattice10":
        keys = {k for k in keys if (k[0] - 40.0) % 10.0 == 0 and (k[1] - 50.0) % 10.0 == 0} | top
    return sorted(keys)


def cmd_one(a):
    row = run_one(float(a.xy[0]), float(a.xy[1]), a.variant)
    print(json.dumps(row))


def cmd_batch(a):
    rows_path = OUT / "rows" / f"karma_{a.variant}.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if rows_path.exists():
        for x in rows_path.read_text().splitlines():
            r = json.loads(x)
            if "error" not in r:
                done.add((r["x_sep"], r["y_sep"]))
    todo = [k for k in lattice_layouts(a.variant, a.subset) if k not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.variant}: {len(done)} layouts done, {len(todo)} to run on {a.jobs} workers", flush=True)
    t0 = time.time()
    with open(rows_path, "a") as fh, ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(run_one, xs, ys, a.variant): (xs, ys) for xs, ys in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                row = fu.result()
            except Exception as e:  # a crashed worker must not stop the batch
                xs, ys = futs[fu]
                row = {"x_sep": xs, "y_sep": ys, "variant": a.variant, "error": f"{type(e).__name__}: {e}"[:400]}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            el = time.time() - t0
            print(f"{i}/{len(todo)} ({row['x_sep']:g}, {row['y_sep']:g}): KaRMA-T {row.get('karma_t')}, KaRMA-R "
                  f"{row.get('karma_r')}, {row.get('secs')} s; {el / 60:.1f} min, ~{el / i * (len(todo) - i) / 60:.0f} "
                  f"min left" + (f"; ERROR {row['error'][-120:]}" if "error" in row else ""), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("one")
    o.add_argument("--xy", nargs=2, type=float, required=True)
    o.add_argument("--variant", default="sph40", choices=list(VARIANTS))
    b = sub.add_parser("batch")
    b.add_argument("--variant", default="sph40", choices=list(VARIANTS))
    b.add_argument("--jobs", type=int, default=8)
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--subset", default="all", choices=("all", "lattice10"))
    a = ap.parse_args()
    {"one": cmd_one, "batch": cmd_batch}[a.cmd](a)


if __name__ == "__main__":
    main()
