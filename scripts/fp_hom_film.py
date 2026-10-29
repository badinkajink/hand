#!/usr/bin/env python3
"""Films for the foundational-pose review page: the HOM twist of a 40 mm sphere on two symmetric tripods of the 2-D
case, side by side. The layout whose twist went furthest on three pads in every seed (rows/hom2d_gpu_twist.jsonl), and
the layout with the largest kinematic rotation range among those whose twist dropped the sphere. CPU MuJoCo with the
task study's runner (scripts/hom_tasks.py), seed 0.

  MUJOCO_GL=egl .venv/bin/python scripts/fp_hom_film.py
Writes docs/experiments/20261029-foundational_pose_review/media/20261029-hom_twist_two_layouts.mp4 (+ .jpg poster).
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import fp_hom_check as C  # noqa: E402
import fp_review_figures as FF  # noqa: E402
import hom_tasks as T  # noqa: E402

MEDIA = C.D / "media"


def pick():
    lay = FF.load_hom("hom2d_gpu_twist")
    held = [x for x in lay if x["held"] >= 1.0]
    best = max(held, key=lambda x: x["ramp"])
    dropped = [x for x in lay if x["held"] < 0.5]
    worst = max(dropped, key=lambda x: x["kin"]["tmr_deg"])
    return best, worst


def film_one(xs, ys, label, out):
    import mujoco
    import hom_contact_rig as HR
    cell = C.cell_of(xs, ys)
    des = C.design(cell)
    p_pads, p_pt, setup = C.build(cell, des, "twist")
    m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
    T.S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = T.H.Mirror(m_pt, fs, os_)
    q_open, _ = T.open_pose(setup, mujoco.MjModel.from_xml_path(str(p_pt)))
    plant = T.Plant(p_pads, setup)
    film = T.Film(plant, setup)
    res, _trace, frames = T.rollout("twist", setup, plant, mirror, q_open, 0, T.PRM, replace(T.H.HomParams(dt=T.DT_C)),
                                    film)
    frames = [HR.annotate(f, label) for f in frames]
    HR.write_mp4(frames, out, fps=25)
    for q in (p_pads, p_pt):
        Path(q).unlink(missing_ok=True)
    return res


def main():
    best, worst = pick()
    MEDIA.mkdir(parents=True, exist_ok=True)
    tmp = []
    info = {}
    for name, x in (("best", best), ("dropped", worst)):
        xs, ys = x["key"]
        label = (f"x_sep {xs:.0f} mm, y_sep {ys:.0f} mm: kinematic range {x['kin']['tmr_deg']:.0f} deg, "
                 f"GPU ramp {x['ramp']:.0f} deg")
        out = MEDIA / f"_tmp_{name}.mp4"
        res = film_one(xs, ys, label, out)
        tmp.append(out)
        info[name] = {"x_sep_mm": xs, "y_sep_mm": ys, "kin_tmr_deg": x["kin"]["tmr_deg"], "gpu_ramp_deg": x["ramp"],
                      "cpu_seed0": {k: res.get(k) for k in ("held", "ramp_three_pads_max_deg", "w_rmse_ratio")}}
    final = MEDIA / "20261029-hom_twist_two_layouts.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp[0]), "-i", str(tmp[1]), "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                    str(final)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(final)], capture_output=True, text=True).stdout)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{0.62 * dur:.2f}", "-i", str(final), "-frames:v", "1",
                    "-q:v", "3", str(final.with_suffix(".jpg"))], check=True)
    for q in tmp:
        q.unlink(missing_ok=True)
    (C.ROWS / "film_two_layouts.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
