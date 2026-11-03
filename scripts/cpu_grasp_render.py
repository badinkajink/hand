#!/usr/bin/env python3
"""Render plan-alone vs plan + residual policy for the same cylinder, stacked, to mp4 + a still sheet.

    .venv-cpu/bin/python scripts/cpu_grasp_render.py --policy runs/cpu_grasp/A_palm_s0_full/policy_best.pt \
        --out manta/sim_media/20261102-policyA --mu 2.0 --cells "25.8:28.5,20:60"
Top row: plan alone (zero residual). Bottom row: policy (--policy none: the plan alone only). Left: side view along the cylinder axis
(like the RealSense); right: isometric. Simulation states are recorded and rendered afterwards on the
main thread (macOS GL).
"""
import argparse
import os
import subprocess
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from morphohand.cpu_grasp import env as E  # noqa: E402
from cpu_grasp_eval_policy import load_policy  # noqa: E402

W, H = 640, 480


def cams(obj):
    side = mujoco.MjvCamera(); side.lookat[:] = obj + [0, 0, 0.07]; side.distance = 0.42
    side.azimuth, side.elevation = 90, -8
    iso = mujoco.MjvCamera(); iso.lookat[:] = obj + [0, 0, 0.07]; iso.distance = 0.45
    iso.azimuth, iso.elevation = 135, -25
    return side, iso


def rollout(dm, mg, mu, act):
    """Returns (qpos states, model, result string)."""
    env = E.GraspEnv(E.EnvCfg(n_envs=1, threads=1, palm_action=True))
    o, _ = env.reset(cells=[(dm / 1000, mg / 1000, (0, 0, 0), mu)])
    e = env.envs[0]
    obj0 = e.d.qpos[e.ix["obj_qpos"]:e.ix["obj_qpos"] + 3].copy()
    states = [e.d.qpos.copy()]
    real_run = E.judge.run

    class R:  # judge calls renderer.update_scene(d, cam) then .render() each frame
        def update_scene(self, dd, cam): pass
        def render(self):
            states.append(e.d.qpos.copy()); return None
    E.judge.run = lambda m_, d_, ix_, K_: real_run(m_, d_, ix_, K_, frames=[], renderer=R(), cam=None)
    info = {}
    try:
        for k in range(E.N_STEPS):
            a = np.zeros((1, 10), np.float32) if act is None else act(o)
            o, _, _, done, inf = env.step(a)
            if inf[0]:
                info = inf[0]
            if (not done[0] and k % 2 == 1) or k == E.N_STEPS - 1:
                states.append(e.d.qpos.copy())
            if done[0]:
                break
    finally:
        E.judge.run = real_run
    v = e.verdict
    res = ("DROPPED in lift" if info.get("dropped") else
           f"{'HELD' if v.held else 'lost it'} through the turn, tilt {v.tilt_deg:.0f} deg"
           + ("  -> success" if info.get("success") else ""))
    return states, e.m, obj0, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mu", type=float, default=2.0)
    ap.add_argument("--cells", default="25.8:28.5")
    ap.add_argument("--label", default="plan + policy A")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    act = None if a.policy == "none" else load_policy(a.policy)[0]
    runs = (("plan alone", None),) if act is None else (("plan alone", None), (a.label, act))
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for cell in a.cells.split(","):
        dm, mg = (float(x) for x in cell.split(":"))
        rows = []
        for label, pol in runs:
            states, m, obj0, res = rollout(dm, mg, a.mu, pol)
            rows.append((label, states, m, obj0, res))
        n = max(len(r[1]) for r in rows)
        frames = []
        rend = mujoco.Renderer(rows[0][2], H, W)
        for label, states, m, obj0, res in rows:
            rd = mujoco.MjData(m)
            side, iso = cams(obj0)
            rf = []
            for qp in states + [states[-1]] * (n - len(states)):
                rd.qpos[:] = qp; mujoco.mj_forward(m, rd)
                rend.update_scene(rd, side); s = rend.render().copy()
                rend.update_scene(rd, iso); b = rend.render().copy()
                im = Image.fromarray(np.hstack([s, b])); dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, 2 * W, 30], fill=(0, 0, 0))
                dr.text((10, 5), f"{label}  |  {res}", fill=(255, 255, 255), font=font)
                rf.append(np.asarray(im))
            frames.append(rf)
        cap = f"{dm:g} mm  {mg:g} g  friction {a.mu:g}"
        out = []
        for i in range(n):
            im = Image.fromarray(np.vstack([fr[i] for fr in frames])); dr = ImageDraw.Draw(im)
            hh = len(frames) * H
            dr.rectangle([2 * W - 330, hh - 30, 2 * W, hh], fill=(0, 0, 0))
            dr.text((2 * W - 320, hh - 25), f"{cap}   t {i * 0.04:.2f} s", fill=(255, 255, 255), font=font)
            out.append(np.asarray(im))
        tag = f"d{dm:g}mm_m{mg:g}g_mu{a.mu:g}"
        mp4 = os.path.join(a.out, tag + ".mp4")
        ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                               f"{2*W}x{len(frames)*H}", "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4],
                              stdin=subprocess.PIPE)
        for f in out:
            ff.stdin.write(f.tobytes())
        ff.stdin.close(); ff.wait()
        # stills: grip (~0.9 s), lifted (~1.8 s), end; side views only, plan above policy
        idx = [min(23, n - 1), min(45, n - 1), n - 1]
        sheet = np.hstack([np.vstack([fr[i][:, :W] for fr in frames]) for i in idx])
        Image.fromarray(sheet).save(os.path.join(a.out, tag + "_stills.jpg"), quality=88)
        print(f"{tag}: " + " | ".join(f"{r[0]}: {r[4]}" for r in rows) + f" -> {mp4}", flush=True)


if __name__ == "__main__":
    main()
