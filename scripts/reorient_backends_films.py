#!/usr/bin/env python3
"""Films of the bench reorientation in each contact model: one hand, one seed, every condition side by side.

MuJoCo conditions render from their own model. A Drake rollout records the nine finger angles and the tool pose
at 25 fps and is drawn in the MuJoCo scene of the same fingertip (the single-geom variant), so every panel uses
one camera and one renderer. Each panel is labelled with the condition, simulated time and the tool's signed
cosine; finger pads touching the tool are drawn red in the MuJoCo panels.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/reorient_backends_films.py --hand D7 --seed 0
Output: docs/experiments/20261006-fingertip_backends/media/<hand>_s<seed>_<cond>.mp4 and <hand>_s<seed>_models.mp4/.jpg
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import contact_bed_common as CB  # noqa: E402
import reorient_backends as RB  # noqa: E402

MEDIA = ROOT / "docs/experiments/20261006-fingertip_backends/media"
W, H = 480, 360
CONDS = [   # label, sim, tip, model, numerics, ir
    ("MuJoCo, legacy box, 2 ms pyramidal", "mujoco", "legacy", "pt", "scene", 1.0),
    ("MuJoCo, legacy box, 1 ms elliptic", "mujoco", "legacy", "pt", "bed", 100.0),
    ("MuJoCo, 10.55 mm sphere", "mujoco", "sphere", "pt", "bed", 100.0),
    ("MuJoCo, TPU r 6 mm, 1 mm pads", "mujoco", "tpu6", "pads", "bed", 100.0),
    ("MuJoCo, TPU r 2.7 mm, 1 mm pads", "mujoco", "tpu2.7", "pads", "bed", 100.0),
    ("Drake hydro, legacy box", "drake", "legacy", "hydro", "bed", 0.0),
    ("Drake hydro, TPU r 6 mm", "drake", "tpu6", "hydro", "bed", 0.0),
    ("Drake hydro, TPU r 2.7 mm", "drake", "tpu2.7", "hydro", "bed", 0.0),
    ("Drake hydro, 10.55 mm sphere", "drake", "sphere", "hydro", "bed", 0.0),
]


def label(frame, lines):
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    y = 6
    for ln in lines:
        dr.rectangle([4, y - 1, 8 + 7 * len(ln), y + 13], fill=(255, 255, 255))
        dr.text((6, y), ln, fill=(20, 20, 20))
        y += 15
    return np.asarray(im)


def renderer_for(m, lookat):
    """One fixed camera on the grasp: looking at the tool's start position, so a dropped tool leaves the frame and
    the hand stays in it."""
    import mujoco
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 0.30, 150.0, -18.0
    cam.lookat[:] = np.asarray(lookat, float) + [0.0, 0.0, 0.012]
    return r, cam


def film_mujoco(hand, seed, tip, model, numerics, ir, title, plant="fast", mu="scene"):
    import mujoco
    plan, traj = RB.load_plan(hand)
    scene, meta = RB.build_scene(hand, tip, model, plant, numerics, ir if ir > 0 else 100.0, mu)
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    tb = m.body(RB.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    tool7 = np.asarray(meta["tool7"], float)
    dx, dy, dyaw = RB.jitter(seed)
    d.qpos[qa:qa + 3] = tool7[:3] + [dx, dy, 0.0]
    d.qpos[qa + 3:qa + 7] = RB._yaw_quat(tool7[3:], dyaw)
    act = {}
    for f in RB.FINGERS:
        for j in RB.JOINTS:
            n = f"{f}_{j}"
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = meta["q0"][n]
            act[n] = m.actuator(f"a_{n}").id
    mujoco.mj_forward(m, d)
    rend, cam = renderer_for(m, tool7[:3])
    pad_rgba = {i: m.geom_rgba[i].copy() for i in range(m.ngeom) if "_pad" in (m.geom(i).name or "")}
    tool_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == tb}
    segs = RB.schedule(plan, traj)
    k_film = max(1, int(round(1 / 25 / m.opt.timestep)))
    frames, step = [], 0
    for t_end, tgt, _ in segs:
        for n_, a in act.items():
            d.ctrl[a] = tgt[n_]
        while d.time < t_end - 1e-9:
            mujoco.mj_step(m, d)
            step += 1
            if step % k_film == 0:
                for i, c in pad_rgba.items():
                    m.geom_rgba[i] = c
                for i in range(d.ncon):
                    g1, g2 = int(d.contact[i].geom[0]), int(d.contact[i].geom[1])
                    for g, o in ((g1, g2), (g2, g1)):
                        if g in pad_rgba and o in tool_geoms:
                            m.geom_rgba[g] = (0.9, 0.1, 0.1, 1.0)
                rend.update_scene(d, cam)
                R9 = np.zeros(9)
                mujoco.mju_quat2Mat(R9, d.qpos[qa + 3:qa + 7])
                frames.append(label(rend.render().copy(), [title, f"{hand} seed {seed}  t {d.time:4.2f} s  cos {R9[8]:+.2f}"]))
    rend.close()
    return frames


def film_drake(hand, seed, tip, title, plant="fast", mu="scene"):
    """Drake rollout drawn in the MuJoCo scene of the same tip, one frame every 40 ms of simulated time."""
    import mujoco
    plan, traj = RB.load_plan(hand)
    scene, meta = RB.build_scene(hand, tip, "pt", plant, "bed", 100.0, mu)
    db = RB.DrakeBench(scene, meta)
    db.reset(meta["q0"], meta["tool7"], seed)
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    tb = m.body(RB.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    jadr = {n: m.jnt_qposadr[m.joint(n).id] for n in db.joints}
    rend, cam = renderer_for(m, np.asarray(meta["tool7"], float)[:3])
    segs = RB.schedule(plan, traj)
    frames, t, dtf = [], 0.0, 1 / 25
    t_next = dtf
    p = db.plant
    for t_end, tgt, _ in segs:
        db.set_targets(tgt)
        while t < t_end - 1e-9:
            t = min(t_end, t_next)
            db.sim.AdvanceTo(t)
            if t < t_next - 1e-9:
                continue
            t_next += dtf
            for n in db.joints:
                d.qpos[jadr[n]] = p.GetJointByName(n).get_angle(db.pc)
            X = p.EvalBodyPoseInWorld(db.pc, db.tool)
            d.qpos[qa:qa + 3] = X.translation()
            d.qpos[qa + 3:qa + 7] = X.rotation().ToQuaternion().wxyz()
            mujoco.mj_forward(m, d)
            rend.update_scene(d, cam)
            c = float(X.rotation().matrix()[2, 2])
            frames.append(label(rend.render().copy(), [title, f"{hand} seed {seed}  t {t:4.2f} s  cos {c:+.2f}"]))
    rend.close()
    return frames


def tile_stream(clips, out_mp4, out_jpg, cols=3, poster_at=0.62, fps=25):
    """Tile equal-size clips into a grid one frame at a time (nine 480x360 clips held in memory need ~1.7 GB);
    shorter clips hold their last frame."""
    import imageio.v2 as imageio
    from PIL import Image
    readers = [imageio.get_reader(str(c)) for c in clips]
    counts = [r.count_frames() for r in readers]
    n = max(counts)
    rows = int(math.ceil(len(clips) / cols))
    w = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=8,
                           ffmpeg_params=["-crf", "27", "-preset", "medium"], ffmpeg_log_level="error")
    last = [None] * len(clips)
    its = [iter(r) for r in readers]
    k_poster = int(round(poster_at * (n - 1)))
    for k in range(n):
        canvas = np.full((rows * H, cols * W, 3), 255, np.uint8)
        for i, it in enumerate(its):
            if k < counts[i]:
                last[i] = np.asarray(next(it))[:, :, :3]
            r_, c_ = divmod(i, cols)
            canvas[r_ * H:(r_ + 1) * H, c_ * W:(c_ + 1) * W] = last[i]
        w.append_data(canvas)
        if k == k_poster:
            Image.fromarray(canvas).save(out_jpg, quality=88)
    w.close()
    for r in readers:
        r.close()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", default="D7")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", type=int, help="indices into CONDS")
    ap.add_argument("--tile-only", action="store_true", help="tile the clips already in media/")
    a = ap.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, (title, sim, tip, model, numerics, ir) in enumerate(CONDS):
        if a.only and i not in a.only:
            continue
        slug = f"{a.hand}_s{a.seed}_{sim}_{tip}_{model}_{numerics}".replace(".", "p")
        out = MEDIA / f"{slug}.mp4"
        if a.tile_only and not out.exists():
            continue
        if not out.exists():
            fr = film_drake(a.hand, a.seed, tip, title) if sim == "drake" else \
                film_mujoco(a.hand, a.seed, tip, model, numerics, ir, title)
            CB.write_h264(fr, out)
        clips.append(out)
        print("wrote", out.relative_to(ROOT), flush=True)
    if len(clips) > 1:
        n = tile_stream(clips, MEDIA / f"{a.hand}_s{a.seed}_models.mp4", MEDIA / f"{a.hand}_s{a.seed}_models.jpg")
        print("tile", n, "frames", flush=True)


if __name__ == "__main__":
    main()
