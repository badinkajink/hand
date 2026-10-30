#!/usr/bin/env python3
"""The CPU runner with a table under the object until the lift, the full version of the ground check (owner,
2026-10-30: "doing these tasks with an initial grasp, raise, and then manipulate"; scripts/hom_ground.py is the quick
version). A box's top lies at the object's lowest point. It collides with the object and with every finger body until the
squeeze ends, when it goes with the post, as if the hand raised the object off the table. Each run is the best grasp
(by the GPU's seeds 0-2) of one of a grid's top layouts, seed 0, with and without the table, for the grids with the
plant's finger clearance (grid2dff) and for those whose grasps clear the ground (grid2dgr). The cylinder tasks hold the
object at its start until the grasp closes, as on the GPU.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/hom_table_check.py twist 40 --n 20 \\
        [--src grid2dff grid2dgr] [--part 0 --parts 2]
    .venv/bin/python scripts/hom_table_check.py film      # the best layouts of both grids with the table, filmed
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
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
    ET.SubElement(body, "geom", name="table", type="box", size="0.15 0.15 0.005", rgba="0.55 0.45 0.35 1",
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
        frames = [HR.annotate(f, label.replace("{cpu}", f"{L.rollout_score(task, res):.2f}"), size=14) for f in frames]
    return res, touch, frames


def top_layouts(task, d, src, n):
    """The n best layouts of a grid by the GPU's best-of-16 score on seeds 0-2, with their best grasps."""
    keep = L.GRID_SRC, L.GRID_ROWS
    L.GRID_SRC, L.GRID_ROWS = src, L.ROWS / f"{src}_v4b"
    try:
        tab = [r for r in L.grid_table(task, d) if r.get("n_grasps")]
    finally:
        L.GRID_SRC, L.GRID_ROWS = keep
    return sorted(tab, key=lambda r: -r["best"])[:n]


def film(task, d):
    """The best layout of each grid with its best grasp, seed 0, the table under the object until the lift: the grid with
    the plant's finger clearance (left) and the grid whose grasps clear the ground (right). Writes
    media/20261030-film_table_<task>.mp4 and its entry in rows/films_v4b.json."""
    import json
    clips, info = [], {}
    for src, name in (("grid2dff", "best grasp"), ("grid2dgr", "best grasp clear of the table")):
        r = top_layouts(task, d, src, 1)[0]
        res, touch, frames = run(task, d, src, r["call"], r["best_grasp"], 0, True,
                                 f"{name} ({r['x_sep']:.0f}, {r['y_sep']:.0f}) mm: score {{cpu}}, {r['best']:.2f} on the GPU")
        clips.append(frames)
        info[src] = {"layout": [r["x_sep"], r["y_sep"]], "call": r["call"], "grasp": r["best_grasp"], "gpu": r["best"],
                     "score": round(L.rollout_score(task, res), 3), "held": res.get("held"), **touch}
    F.compose(clips, F.MEDIA / f"20261030-film_table_{task}.mp4")
    allf = json.loads(F.INFO.read_text()) if F.INFO.exists() else {}
    allf[f"table_{task}"] = dict(info, when=time.strftime("%Y-%m-%d %H:%M"))
    F.INFO.write_text(json.dumps(allf, indent=1, default=float))
    return info


def main():
    if sys.argv[1:2] == ["film"]:
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
