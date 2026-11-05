#!/usr/bin/env python3
"""The CPU runner with a table under the object until the lift, the full version of the ground check (owner,
2026-10-30: "doing these tasks with an initial grasp, raise, and then manipulate"; scripts/hom/hom_ground.py is the quick
version). A box's top lies at the object's lowest point. It collides with the object and with every finger body until the
squeeze ends, when it goes with the post, as if the hand raised the object off the table. Each run is the best grasp
(by the GPU's seeds 0-2) of one of a grid's top layouts, seed 0, with and without the table, for the grids with the
plant's finger clearance (grid2dff) and for those whose grasps clear the ground (grid2dgr). The cylinder tasks hold the
object at its start until the grasp closes, as on the GPU.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/hom/hom_table_check.py twist 40 --n 20 \\
        [--src grid2dff grid2dgr] [--part 0 --parts 2]
    .venv/bin/python scripts/hom/hom_table_check.py film      # the best layouts of both grids with the table, filmed
Rows: docs/experiments/20261029-foundational_pose_review/rows/cpu_best/table_<task>_d<d>.jsonl (fsynced, one per run).
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_films as F  # noqa: E402
import hom_landscape as L  # noqa: E402

HG, T, S, H = F.HG, F.T, F.S, F.H
OUT = F.ROWS / "cpu_best"
HOLD = {"twist": 0.0, "roll": 0.0, "pinpush": T.T_CLOSE, "press": T.T_CLOSE}


def table_scene(path, z_top):
    """The scene with a 300 x 300 x 10 mm box whose top is at z_top (m), written beside it (its relative paths hold)."""
    root = ET.parse(path).getroot()
    post = root.find(".//geom[@name='post']")
    body = ET.SubElement(root.find("worldbody"), "body", name="table", pos=f"0 0 {z_top - 0.005:.6f}")
    ET.SubElement(body, "geom", name="table", type="box", size="0.15 0.15 0.005", rgba="0.55 0.45 0.35 0.3",   # the hand
                  # shows through it in the films, whose camera looks up from below the table's top
                  friction=post.get("friction", "1 0.005 0.0001"))
    out = Path(path).with_name(f"{Path(path).stem}.table.{os.getpid()}.xml")
    ET.ElementTree(root).write(out)
    return out


def run(task, d, src, call, grasp, seed, table, label=None):
    """One CPU rollout: its result, with the table the control periods in which a finger touched it before the lift and
    the largest normal force between the table and the fingers (N), and with a label its film's frames ({cpu}: the
    score)."""
    import mujoco
    _k, mxy, setup = F.find(task, f"{src}_size", d, call, grasp)
    scene = HG.Scene(T.SHAPE[task], d)
    bm, q_open = scene.prepare(mxy, setup)
    path = table_scene(scene.pads_path, S.Z_OBJ - scene.r) if table else scene.pads_path
    try:
        plant = T.Plant(path, setup)
    finally:
        if table:
            path.unlink()
    m = plant.m
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for mm in (m, m_pt):
        for f in F.FINGERS:
            mm.body_pos[mm.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = H.Mirror(m_pt, fs, os_)
    touch = {"periods": 0, "force_max_N": 0.0}
    if table:
        tg = m.geom("table").id
        finger_geom = np.array([bool(next((f for f in F.FINGERS if m.body(m.geom_bodyid[g]).name.startswith(f + "_")),
                                          None)) for g in range(m.ngeom)])
        off, adv = plant.post_off, plant.advance

        def post_off():
            off()
            m.geom_contype[tg] = m.geom_conaffinity[tg] = 0
            m.geom_rgba[tg, 3] = 0.12

        def advance(t1):
            adv(t1)
            dd, fsum = plant.d, 0.0
            for i in range(dd.ncon):
                c = dd.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if tg in (g1, g2) and finger_geom[g2 if g1 == tg else g1]:
                    f6 = np.zeros(6)
                    mujoco.mj_contactForce(m, dd, i, f6)
                    fsum += abs(f6[0])
            if fsum > 0:
                touch["periods"] += 1
                touch["force_max_N"] = max(touch["force_max_N"], fsum)
        plant.post_off, plant.advance = post_off, advance
    film = F.Film(plant, setup) if label else None
    try:
        res, _tr, frames = T.rollout(task, setup, plant, mirror, q_open, seed, T.PRM, replace(H.HomParams(dt=T.DT_C)),
                                     film, None, HOLD[task])
    finally:
        if film:
            film.close()
    if label:
        import hom_contact_rig as HR
        frames = [HR.annotate(f, label.replace("{cpu}", f"{L.rollout_score(task, res):.2f}"), size=22) for f in frames]
    return res, touch, frames


def trace(task, d, src, k=0, table=True):
    """The k-th best layout of a grid, seed 0, traced with the table (or on the post alone): per finger body the first
    control step with table contact, its deepest penetration (mm) and its table force (N) at the phase ends, and per pad
    when it first meets the object and how deep it sinks into it before the lift (mm). Writes
    rows/cpu_best/table_trace_<task>_d<d>_<src>[_post].json."""
    import json
    import mujoco
    r = top_layouts(task, d, src, k + 1)[k]
    _k, mxy, setup = F.find(task, f"{src}_size", d, r["call"], r["best_grasp"])
    scene = HG.Scene(T.SHAPE[task], d)
    bm, q_open = scene.prepare(mxy, setup)
    path = table_scene(scene.pads_path, S.Z_OBJ - scene.r) if table else scene.pads_path
    try:
        plant = T.Plant(path, setup)
    finally:
        if table:
            path.unlink()
    m = plant.m
    for f in F.FINGERS:
        m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    tg, ob = (m.geom("table").id if table else -1), m.body(S.OBJ).id
    log, adv, off = [], plant.advance, plant.post_off
    lift = {}

    def post_off():                                    # the table goes with the post at the lift, as in run()
        off()
        lift["t"], lift["z"] = float(plant.d.time), float(plant.d.xpos[ob][2])
        if table:
            m.geom_contype[tg] = m.geom_conaffinity[tg] = 0

    def advance(t1):
        adv(t1)
        dd = plant.d
        if "t" in lift and "drop_s" not in lift and float(dd.xpos[ob][2]) < lift["z"] - 0.010:
            lift["drop_s"] = round(float(dd.time), 3)          # the object 10 mm below its height at the lift
        for i in range(dd.ncon):
            c = dd.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = (m.body(m.geom_bodyid[g]).name for g in (g1, g2))
            if tg in (g1, g2):
                g, b = (g2, b2) if g1 == tg else (g1, b1)
                kind = "table"
            elif ob in (int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])):
                g, b = (g2, b2) if int(m.geom_bodyid[g1]) == ob else (g1, b1)
                kind = "object"
            else:
                continue
            if not b.startswith(tuple(f"{f}_" for f in F.FINGERS)):
                continue
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, dd, i, f6)
            log.append((round(float(dd.time), 3), kind, b, float(c.dist) * 1000, float(f6[0])))
    plant.advance, plant.post_off = advance, post_off
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for f in F.FINGERS:
        m_pt.body_pos[m_pt.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    res, _tr, _fr = T.rollout(task, setup, plant, H.Mirror(m_pt, fs, os_), q_open, 0, T.PRM,
                              replace(H.HomParams(dt=T.DT_C)), None, None, HOLD[task])
    tab = [x for x in log if x[1] == "table"]
    out = {"task": task, "d_mm": d, "src": src, "rank": k, "call": r["call"], "layout": [r["x_sep"], r["y_sep"]],
           "grasp": r["best_grasp"], "table": table, "score": round(L.rollout_score(task, res), 3),
           "held": res.get("held"), "table_first_s": {}, "table_deepest_mm": {}, "table_force_N": {},
           "object_first_s": {}, "object_deepest_before_lift_mm": {}, "lift_s": lift.get("t"),
           "object_10mm_down_s": lift.get("drop_s")}
    for t, _kd, b, dist, fn in tab:
        out["table_first_s"].setdefault(b, t)
        out["table_deepest_mm"][b] = round(min(out["table_deepest_mm"].get(b, 0.0), dist), 2)
    for ph, t_end in (("open", T.T_OPEN), ("close", T.T_CLOSE), ("squeeze", T.T_LIFT)):
        rows = [x for x in tab if abs(x[0] - t_end) < 0.006]
        out["table_force_N"][ph] = {b: round(sum(x[4] for x in rows if x[2] == b), 1) for b in sorted({x[2] for x in rows})}
    for t, kd, b, dist, _fn in log:
        if kd == "object" and b.endswith("_tip"):
            out["object_first_s"].setdefault(b, t)
            if t < T.T_LIFT:
                out["object_deepest_before_lift_mm"][b] = round(min(out["object_deepest_before_lift_mm"].get(b, 0.0),
                                                                    dist), 2)
    path = OUT / f"table_trace_{task}_d{d:g}_{src}{'' if table else '_post'}.json"
    path.write_text(json.dumps(out, indent=1))
    return out


def top_layouts(task, d, src, n):
    """The n best layouts of a grid by the GPU's best-of-16 score on seeds 0-2, with their best grasps."""
    keep = L.GRID_SRC, L.GRID_ROWS
    L.GRID_SRC, L.GRID_ROWS = src, L.ROWS / f"{src}_v4b"
    try:
        tab = [r for r in L.grid_table(task, d) if r.get("n_grasps")]
    finally:
        L.GRID_SRC, L.GRID_ROWS = keep
    return sorted(tab, key=lambda r: -r["best"])[:n]


NAMES = {"grid2dff": "unconstrained", "grid2dgr": "clear, 1 mm allowed", "grid2dgn": "clear, no penetration",
         "grid2dgs": "clear, squeeze"}                       # the films' labels, short enough for a third of the width


def film(task, d, srcs=("grid2dff", "grid2dgr"), stem=None):
    """The best layout of each grid with its best grasp, seed 0, the table under the object until the lift, side by side:
    by default the grid with the plant's finger clearance (left) and the grid whose grasps clear the ground (right).
    Writes media/<stem or 20261030-film_table_<task>>.mp4 and its entry in rows/films_v4b.json."""
    import json
    clips, info = [], {}
    for src, name in ((s_, NAMES[s_]) for s_ in srcs):
        r = top_layouts(task, d, src, 1)[0]
        res, touch, frames = run(task, d, src, r["call"], r["best_grasp"], 0, True,
                                 f"{name}: {{cpu}} (GPU {r['best']:.2f})")
        clips.append(frames)
        info[src] = {"layout": [r["x_sep"], r["y_sep"]], "call": r["call"], "grasp": r["best_grasp"], "gpu": r["best"],
                     "score": round(L.rollout_score(task, res), 3), "held": res.get("held"), **touch}
    stem = stem or f"20261030-film_table_{task}"
    F.compose(clips, F.MEDIA / f"{stem}.mp4")
    allf = json.loads(F.INFO.read_text()) if F.INFO.exists() else {}
    allf[stem.split("-film_", 1)[1]] = dict(info, when=time.strftime("%Y-%m-%d %H:%M"))
    F.INFO.write_text(json.dumps(allf, indent=1, default=float))
    return info


def main():
    if sys.argv[1:2] == ["trace"]:                      # trace task d src [rank] [post]
        print(trace(sys.argv[2], float(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 0,
                    sys.argv[6:7] != ["post"]))
        return 0
    if sys.argv[1:2] == ["film"]:                       # film [task d stem src ...]
        if len(sys.argv) > 2:
            task, d, stem, srcs = sys.argv[2], float(sys.argv[3]), sys.argv[4], tuple(sys.argv[5:])
            print(task, film(task, d, srcs, stem), flush=True)
            return 0
        for task, d in (("twist", 40.0), ("pinpush", 25.0)):
            print(task, film(task, d), flush=True)
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task")
    ap.add_argument("d", type=float)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--src", nargs="+", default=["grid2dff", "grid2dgr"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"table_{a.task}_d{a.d:g}.jsonl"
    jobs = [(src, r) for src in a.src for r in top_layouts(a.task, a.d, src, a.n)][a.part::a.parts]
    for src, r in jobs:
        for table in (False, True):
            t0 = time.time()
            res, touch, _f = run(a.task, a.d, src, r["call"], r["best_grasp"], 0, table)
            row = {"task": a.task, "d_mm": a.d, "src": src, "call": r["call"], "x_sep": r["x_sep"], "y_sep": r["y_sep"],
                   "grasp": r["best_grasp"], "gpu_best": r["best"], "seed": 0, "table": table,
                   "table_touch_periods": touch["periods"], "table_force_max_N": round(touch["force_max_N"], 3),
                   "score": L.rollout_score(a.task, res),
                   **{k: res.get(k) for k in ("held", "ramp_three_pads_max_deg", "w_rmse_ratio", "theta_rmse_deg",
                                              "link_force_max_N")}}
            T._append(path, row)
            print(src, r["call"], (r["x_sep"], r["y_sep"]), "table" if table else "post only",
                  f"score {row['score']:.2f} (GPU {r['best']:.2f})", touch, f"{time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
