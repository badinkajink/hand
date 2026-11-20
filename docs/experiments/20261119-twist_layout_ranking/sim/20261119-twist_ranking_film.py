#!/usr/bin/env python3
"""Film of the simulated twist at three proposed layouts side by side: the best, a middle and the lowest-ranked of the
deployable layouts proposed for the hand (hand-4a, 2026-11-19), so that the twist each plan is predicted to make can be
seen before the trials.

Each panel is the run of sim/20261119-twist_ranking_search.py's model with the sphere pads at seed 0 (the plant, the
controller and the start as in the ranking), drawn at 30 frames per simulated second by MuJoCo's renderer from one fixed
camera. The sphere, its stem and its block are bodies of revolution about the twist's axis, so two segments 35 mm long
are drawn from the sphere's centre along its body x axis (red) and z axis (blue); they turn with the twist. The label bar
gives the layout, the grasp and the predicted tracking (both models, seeds 0-2); the bottom line the time and the
schedule's phase.

    MUJOCO_GL=egl HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 OMP_NUM_THREADS=1 .venv/bin/python \\
        docs/experiments/20261119-twist_layout_ranking/sim/20261119-twist_ranking_film.py [--calls C1 C2 C3 --grasps K1 K2 K3]
Writes media/20261119-simulated_twist_three_layouts.mp4 and a still at 5 s beside it.
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
spec = importlib.util.spec_from_file_location("twist_search", HERE / "20261119-twist_ranking_search.py")
SR = importlib.util.module_from_spec(spec)
sys.modules["twist_search"] = SR
spec.loader.exec_module(SR)

import numpy as np  # noqa: E402

FPS, W, H = 30, 640, 480
CAM = [0.20, 60.0, 5.0]                           # distance (m), azimuth and elevation (deg) of the free camera
OUT = STUDY / "media/20261119-simulated_twist_three_layouts.mp4"


def run_filmed(job):
    """The ranking's run of one job with frames: [(t, RGB array)] and the run's row."""
    import mujoco
    R = SR._R["mod"]
    GP = R.RG._G["GP"]
    frames, made = [], []

    def plant(scene, st, mxy, params, *a, **k):
        p = GP(scene, st, mxy, params, *a, **k)
        made.append(p)
        rend = mujoco.Renderer(p.m, H, W)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = CAM
        state = {"next": 0.0, "lookat": None}
        orig = p.advance

        def draw():
            s = p.state()
            c = np.asarray(s["pos"], float)
            if state["lookat"] is None:
                state["lookat"] = c + np.array([0.0, 0.0, 0.02])
            cam.lookat[:] = state["lookat"]
            rend.update_scene(p.d, cam)
            from sr2hand.geometry.rotations import to_scipy
            Rm = to_scipy(np.asarray(s["quat"], float)).as_matrix()
            for axis, rgba in ((0, (0.85, 0.1, 0.1, 1.0)), (2, (0.1, 0.25, 0.9, 1.0))):
                scn = rend.scene
                if scn.ngeom >= scn.maxgeom:
                    break
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                    np.asarray(rgba, np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025, c, c + 0.035 * Rm[:, axis])
                scn.ngeom += 1
            frames.append((float(p.d.time), rend.render().copy()))

        def adv(t1):
            while True:
                while state["next"] <= p.d.time + 1e-9:
                    draw()
                    state["next"] += 1.0 / FPS
                if p.d.time >= t1 - 1e-9:
                    break
                orig(min(t1, state["next"]))
        p.advance = adv
        return p
    R.RG._G["GP"] = plant
    try:
        row = SR._run(job)
    finally:
        R.RG._G["GP"] = GP
    return frames, row


def phase_at(t):
    for a, b, name in ((0, 0.6, "open, close"), (0.6, 1.6, "squeeze"), (1.6, 2.0, "lift"), (2.0, 3.0, "wz"),
                       (3.0, 3.4, "rest"), (3.4, 6.4, "twist ramp"), (6.4, 99, "hold")):
        if a <= t < b:
            return name
    return ""


def main():
    from PIL import Image, ImageDraw, ImageFont
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, nargs="*", default=None)
    ap.add_argument("--grasps", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cam", type=float, nargs=3, default=None, help="distance (m), azimuth, elevation (deg)")
    a = ap.parse_args()
    if a.cam:
        CAM[:] = a.cam
    R = SR.load_replan()
    R.RG.gear_class = R.RC2.clutch_class
    R.RC2.STIFF, R.RC2.BREAKAWAY = True, 0.0
    R.RG.RF.RL.rows_file()
    SR._R["mod"] = R
    SR._init()
    prop = json.loads((STUDY / "data/20261119-trial_proposal.json").read_text())
    lays = {c: xy for c, xy, n in SR.layouts()}
    if a.calls:
        picks = [{"call": c, "xy": list(lays[c]), "k": k, "score": None, "rank_among_deployable": None}
                 for c, k in zip(a.calls, a.grasps)]
    else:
        pk = prop["pick"]
        picks = [pk[0], pk[len(pk) // 2], pk[-1]]
    panels, rows = [], []
    for p in picks:
        job = {**SR.base_job(int(p["call"]), tuple(p["xy"]), "pads", contact="pads"), "k": int(p["k"]), "seed": 0}
        fr, row = run_filmed(job)
        panels.append(fr)
        rows.append(row)
        print(f"{p['xy']} g{p['k']}: {len(fr)} frames, track {row.get('track')}, twist {row.get('twist_end')}", flush=True)
    n = min(len(f) for f in panels)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = small = ImageFont.load_default()
    tmp = Path(tempfile.mkdtemp())
    bar, foot = 64, 36
    for i in range(n):
        im = Image.new("RGB", (W * len(panels), H + bar + foot), "white")
        dr = ImageDraw.Draw(im)
        for j, (p, fr) in enumerate(zip(picks, panels)):
            im.paste(Image.fromarray(fr[i][1]), (j * W, bar))
            rank = f"rank {p['rank_among_deployable']}, " if p.get("rank_among_deployable") else ""
            sc = f"predicted {p['score']:.2f}" if p.get("score") is not None else ""
            dr.text((j * W + 10, 6), f"({p['xy'][0]:g}, {p['xy'][1]:g}) mm, grasp {p['k']}", fill="black", font=font)
            dr.text((j * W + 10, 34), f"{rank}{sc}", fill="black", font=small)
        t = panels[0][i][0]
        dr.text((10, H + bar + 6), f"t = {t:4.2f} s   {phase_at(t)}   (simulation, pads, seed 0; red and blue: the "
                                   f"sphere's body axes)", fill="black", font=small)
        im.save(tmp / f"f{i:05d}.png")
        if abs(t - 5.0) < 0.5 / FPS:
            im.save(Path(a.out).with_suffix(".png"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "f%05d.png"), "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    str(a.out)], check=True)
    shutil.rmtree(tmp)
    print("wrote", a.out, f"{Path(a.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
