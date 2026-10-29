#!/usr/bin/env python3
"""Does a layout's kinematic foundational-pose score predict its HOM-controlled outcome? For symmetric tripod layouts
from the 2-D case of scripts/fp_sr2_toy.py (rows/rrt2d.jsonl), runs the HOM twist of a 40 mm sphere
(scripts/hom_tasks.py, task 'twist': steps of 40 deg/s, then a 30 deg/s ramp) on the CAD-model SR2 hand, servo plant,
1 mm sphere pads, and compares the twist's ramp on three pads with the tripod pose's fixed-contact rotation range.

Each layout gets the HOM task study's own equator grasp (hom_tasks.design_sphere's rule: of the scale study's candidate
palm heights, the one whose least-conditioned finger has the largest smallest singular value, joint margin 3 deg or
more), so the dynamic run does not reuse the toy's pose.

  .venv/bin/python scripts/fp_hom_check.py [--n 24] [--seeds 0 1 2]
Rows: docs/experiments/20261029-foundational_pose_review/rows/hom_twist_check.jsonl (one fsynced line per rollout).
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v1")

import argparse  # noqa: E402
import json  # noqa: E402
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
import hom_tasks as T  # noqa: E402

S, H = T.S, T.H
D = ROOT / "docs/experiments/20261029-foundational_pose_review"
ROWS = D / "rows"
D_MM = 40.0


def cell_of(xs, ys):
    return {"tag": f"x{xs:05.1f}_y{ys:05.1f}", "x_sep_mm": float(xs), "y_sep_mm": float(ys)}


def design(cell):
    best = None
    for k, cand in enumerate(S.candidates(dict(cell), "sphere", D_MM)):
        score = min(cand["sigma_min_mm"].values())
        if cand["min_margin_deg"] < 3.0:
            continue
        if best is None or score > best["score"]:
            best = {"score": score, "tag": cell["tag"], "x_sep_mm": cell["x_sep_mm"], "y_sep_mm": cell["y_sep_mm"],
                    "cand": k, "h": cand["h"], "spread": cand["spread"], "d_mm": D_MM, "q_deg": cand["q_deg"],
                    "min_margin_deg": cand["min_margin_deg"], "sigma_min_mm": cand["sigma_min_mm"]}
    return best


def build(cell, des):
    """hom_tasks.build for an equator grasp of a sphere on any symmetric tripod."""
    sc = S.cell_scenes(dict(cell), "sphere", D_MM, models=("pads", "pt"))
    gg = S.grasp_geometry(cell, "sphere", D_MM, des["h"], des["spread"])
    centre = gg["centre_palm"]
    tg = {f: centre + gg["rel"][f] for f in T.FINGERS}
    palm = np.array([-centre[0], -centre[1], S.Z_OBJ - centre[2]])
    paths = {}
    for mdl in ("pads", "pt"):
        src = Path(sc[mdl][0])
        dst = src.with_name(f"{src.stem}__hom_twist.xml")
        root = ET.parse(src).getroot()
        root.find(f".//body[@name='{S.OBJ}']").set("quat", " ".join(f"{v:.8f}" for v in T.SPHERE_Q))
        tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")
        ET.ElementTree(root).write(tmp)
        os.replace(tmp, dst)
        paths[mdl] = dst
    xs, ys = cell["x_sep_mm"] / 1000, cell["y_sep_mm"] / 1000
    setup = {"task": "twist", "shape": "sphere", "d_mm": D_MM, "r": D_MM / 2000, "hl": 0.0, "palm": palm,
             "quat": np.array(T.SPHERE_Q), "design": des, "targets": tg, "n_out": gg["n_out"],
             "mounts": T.K.mounts(xs, ys), "meta": sc["pads"][1]}
    return paths["pads"], paths["pt"], setup


def pick_layouts(n):
    rows = [json.loads(x) for x in (ROWS / "rrt2d.jsonl").read_text().splitlines() if x.strip()]
    cand = [r for r in rows if r["ok"]]
    tmr = np.array([r["fp"]["tripod"]["tmr_deg"] for r in cand])
    order = np.argsort(tmr)
    idx = order[np.unique(np.round(np.linspace(0, len(order) - 1, n)).astype(int))]
    out = []
    for i in idx:
        d = cand[i]["d"]
        out.append((round(d[2] - d[0], 1), round(d[3] - d[5], 1), cand[i]["fp"]["tripod"]))
    return out


def main():
    import mujoco
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    ROWS.mkdir(parents=True, exist_ok=True)
    path = ROWS / "hom_twist_check.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["x_sep_mm"], r["y_sep_mm"], r["seed"]))
            except (json.JSONDecodeError, KeyError):
                pass
    for xs, ys, kin in pick_layouts(a.n):
        todo = [s for s in a.seeds if (xs, ys, s) not in done]
        if not todo:
            continue
        cell = cell_of(xs, ys)
        t0 = time.time()
        try:
            des = design(cell)
            if des is None:
                T._append(path, {"x_sep_mm": xs, "y_sep_mm": ys, "seed": -1, "status": "no_grasp",
                                 "kin": {k: kin[k] for k in ("tmr_deg", "mft_mNm", "smin_mm")}})
                print(cell["tag"], "no equator grasp", flush=True)
                continue
            p_pads, p_pt, setup = build(cell, des)
            m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
            S.set_palm(m_pt, setup["palm"])
            fs, os_ = T.sr2_specs(setup)
            mirror = H.Mirror(m_pt, fs, os_)
            q_open, _ = T.open_pose(setup, mujoco.MjModel.from_xml_path(str(p_pt)))
            plant = T.Plant(p_pads, setup)
        except Exception as e:
            T._append(path, {"x_sep_mm": xs, "y_sep_mm": ys, "seed": -1, "status": "error",
                             "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]})
            print(cell["tag"], "build failed", e, flush=True)
            continue
        for seed in todo:
            row = {"x_sep_mm": xs, "y_sep_mm": ys, "seed": seed, "task": "twist", "d_mm": D_MM,
                   "kin": {k: kin[k] for k in ("tmr_deg", "tmr_pm", "mft_mNm", "smin_mm")},
                   "design": {k: des[k] for k in ("h", "score", "min_margin_deg")}}
            try:
                res, _trace, _ = T.rollout("twist", setup, plant, mirror, q_open, seed, T.PRM,
                                           replace(H.HomParams(dt=T.DT_C)))
                row.update(res, status="ok")
            except Exception as e:
                row.update(status="error", error=f"{type(e).__name__}: {e}")
            T._append(path, row)
            print(cell["tag"], seed, row.get("held"), row.get("ramp_three_pads_max_deg"), row.get("w_rmse_ratio"),
                  f"kin tmr {kin['tmr_deg']}", f"{time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
