#!/usr/bin/env python3
"""Films for the foundational-pose review page: the HOM twist of a 40 mm sphere on two symmetric tripods of the 2-D
case, side by side. The layout whose twist went furthest on three pads in every seed (rows/hom2d_gpu_twist.jsonl), and
the layout with the largest kinematic rotation range among those whose twist dropped the sphere. CPU MuJoCo with the
task study's runner (scripts/hom_tasks.py), seed 0.

  MUJOCO_GL=egl .venv/bin/python scripts/fp_hom_film.py
Writes docs/experiments/20261029-foundational_pose_review/media/20261029-hom_twist_two_layouts.mp4 (+ .jpg poster).

  MUJOCO_GL=egl .venv/bin/python scripts/fp_hom_film.py pinpush
The pin-push of the 25 mm cylinder from the toy's pinch pose on two 2-D layouts (rows/hom2dfp_gpu_pinpush.jsonl): the
layout with the smallest pinch-angle RMSE among those held in every seed with the thumb within 60 mm of the pair, and
the layout with the largest kinematic pinch range among those with the thumb more than 80 mm away that dropped the
cylinder in every GPU seed and in the CPU check's seed 0 (rows/cpu_check_2dfp_pinpush.jsonl). Writes media/20261029-hom_pinpush_two_layouts.mp4 (+ .jpg) and rows/film_pinpush_two_layouts.json.
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


def film_pinch(row, label, out, task="pinpush"):
    """One pin-push from the toy's pinch pose of a 2-D layout (row of rows/rrt2d.jsonl), CPU runner, seed 0."""
    import mujoco
    import hom_contact_rig as HR
    import hom_gpu as HG
    d6 = row["d"]
    xs, ys = round(d6[2] - d6[0], 1), round(d6[3] - d6[5], 1)
    setup = HG.pinch_setup(task, row["fp"]["pinch"]["v"], d6)
    sc = T.S.cell_scenes(dict(C.cell_of(xs, ys)), "cylinder", 25.0, models=("pads", "pt"))
    p_pads, p_pt = sc["pads"][0], sc["pt"][0]
    m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
    T.S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = T.H.Mirror(m_pt, fs, os_)
    q_open, _ = T.open_pose(setup, mujoco.MjModel.from_xml_path(str(p_pt)))
    plant = T.Plant(p_pads, setup)
    film = T.Film(plant, setup)
    res, _trace, frames = T.rollout(task, setup, plant, mirror, q_open, 0, T.PRM, replace(T.H.HomParams(dt=T.DT_C)), film)
    HR.write_mp4([HR.annotate(f, label) for f in frames], out, fps=25)
    return res


def main_pinch():
    lay = {x["key"]: x for x in FF.load_hom("hom2dfp_gpu_pinpush")}
    rows = {tuple(r["d"]): r for r in FF.load("rrt2d") if r.get("ok")}
    sep = {k: FF.sep2(k) for k in lay}
    near = [x for k, x in lay.items() if sep[k][0] <= 60 and x["held"] >= 1.0 and np.isfinite(x["theta"])]
    # the dropped example must drop on the CPU runner too, in seed 0 as filmed (rows/cpu_check_2dfp_pinpush.jsonl)
    cpu0 = {tuple(r["d"]): r for r in FF.load("cpu_check_2dfp_pinpush") if r["seed"] == 0}
    far = [x for k, x in lay.items() if sep[k][0] > 80 and x["held"] == 0.0 and k in cpu0 and not cpu0[k]["held"]]
    best = min(near, key=lambda x: x["theta"])
    worst = max(far, key=lambda x: x["kin_pinch"]["tmr_deg"])
    MEDIA.mkdir(parents=True, exist_ok=True)
    tmp, info = [], {}
    for name, x in (("held", best), ("dropped", worst)):
        xs, ys = sep[x["key"]]
        label = (f"x_sep {xs:.0f} mm, y_sep {ys:.0f} mm: pinch range {x['kin_pinch']['tmr_deg']:.0f} deg, "
                 f"GPU held {round(3 * x['held'])} of 3")
        out = MEDIA / f"_tmp_pin_{name}.mp4"
        res = film_pinch(rows[x["key"]], label, out)
        tmp.append(out)
        info[name] = {"x_sep_mm": xs, "y_sep_mm": ys, "kin_pinch_tmr_deg": x["kin_pinch"]["tmr_deg"],
                      "gpu_held": x["held"], "gpu_theta_rmse_deg": x["theta"],
                      "cpu_seed0": {k: res.get(k) for k in ("held", "theta_rmse_deg", "gain_min", "pusher_slip_max_mm")}}
    final = MEDIA / "20261029-hom_pinpush_two_layouts.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp[0]), "-i", str(tmp[1]), "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                    str(final)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(final)], capture_output=True, text=True).stdout)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{0.55 * dur:.2f}", "-i", str(final), "-frames:v", "1",
                    "-q:v", "3", str(final.with_suffix(".jpg"))], check=True)
    for q in tmp:
        q.unlink(missing_ok=True)
    (C.ROWS / "film_pinpush_two_layouts.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


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
    main_pinch() if sys.argv[1:] == ["pinpush"] else main()
