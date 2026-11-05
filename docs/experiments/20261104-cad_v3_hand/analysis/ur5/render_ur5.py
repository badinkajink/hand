#!/usr/bin/env python3
"""Renders of the draft UR5 (CB3) MJCF: the arm at the lab's home and at hand_down, the collision primitives over the
visual meshes, and the flange face in tool0 axes (tool0's frame drawn: x red, y green, z blue). One tiled PNG.

    MUJOCO_GL=egl ~/.claude/bin/resguard.sh run --mem 3G -- .venv/bin/python logs/20261104-cad_v3/ur5/render_ur5.py
    MUJOCO_GL=egl .venv/bin/python logs/20261104-cad_v3/ur5/render_ur5.py --no-meshes     primitives only, no mesh loaded
"""
import math
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 800, 600


def label_bar(text, width, h=40):
    im = Image.new("RGB", (width, h), "white")
    dr = ImageDraw.Draw(im)
    dr.text((10, (h - 22) // 2), text, fill="black", font=ImageFont.truetype("DejaVuSerif.ttf", 19))
    dr.line([(0, h - 1), (width, h - 1)], fill="black", width=1)
    return im


def cam(lookat, dist, az, el):
    c = mujoco.MjvCamera()
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


def load(no_meshes):
    if not no_meshes:
        return mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    arm = (HERE / "ur5_cb3.xml").read_text()
    arm = re.sub(r"\s*<mesh [^>]*/>", "", arm)
    arm = re.sub(r'\s*<geom class="visual"[^>]*/>', "", arm)
    return mujoco.MjModel.from_xml_string((HERE / "scene.xml").read_text(), {"ur5_cb3.xml": arm.encode()})


def main():
    no_meshes = "--no-meshes" in sys.argv
    m = load(no_meshes)
    m.vis.scale.framelength = 0.045 / m.stat.meansize      # tool0's axes 45 mm long, 3 mm thick
    m.vis.scale.framewidth = 0.003 / m.stat.meansize
    for nm in ("base", "base_link"):                       # only tool0's frame is drawn
        m.site_group[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, nm)] = 5
    d = mujoco.MjData(m)
    what = "primitives" if no_meshes else "meshes"
    keys = {m.key(i).name: i for i in range(m.nkey)}
    tool = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tool0")

    def pose(name, q=None):
        if q is None:
            mujoco.mj_resetDataKeyframe(m, d, keys[name])
        else:
            d.qpos[:] = q
        mujoco.mj_forward(m, d)
        return d.site_xpos[tool].copy(), d.site_xmat[tool].reshape(3, 3).copy()

    def opt(collision=False, floor=True, frames=False):
        o = mujoco.MjvOption()
        o.geomgroup[:] = 0
        o.geomgroup[2] = 1
        o.geomgroup[0] = int(floor)
        o.geomgroup[3] = int(collision or no_meshes)
        o.sitegroup[:] = 0
        if frames:
            o.sitegroup[4] = 1
            o.frame = mujoco.mjtFrame.mjFRAME_SITE
        return o

    panels = []
    with mujoco.Renderer(m, H, W) as ren:
        def shot(c, o, text):
            ren.update_scene(d, camera=c, scene_option=o)
            panels.append((text, Image.fromarray(ren.render())))

        p, Rt = pose("lab_home")
        shot(cam([-0.12, -0.25, 0.3], 1.25, 145.0, -15.0), opt(frames=True),
             f"lab_home (recorded 2026-10-22), {what}: tool axis along base -y")
        p, Rt = pose("hand_down")
        shot(cam([-0.15, -0.3, 0.3], 1.25, 145.0, -15.0), opt(frames=True),
             f"hand_down (computed), {what}: tool axis straight down")
        shot(cam([-0.15, -0.3, 0.3], 1.25, 145.0, -15.0), opt(collision=True),
             "hand_down: collision primitives (orange)" + ("" if no_meshes else " over the meshes"))
        p, Rt = pose("zero")
        shot(cam([-0.42, -0.1, 0.05], 1.1, 90.0, -8.0), opt(collision=True),
             "Zero pose, side view" + ("" if no_meshes else ": primitives over the meshes"))
        # the flange face: at q = 0, z_tool0 = -y_B and y_tool0 = +z_B, so a camera looking along +y_B sees x right, y up
        z_t = Rt[:, 2]
        f = -z_t
        az, el = math.degrees(math.atan2(f[1], f[0])), math.degrees(math.asin(f[2]))
        shot(cam(p, 0.22, az, el), opt(floor=False, frames=True),
             f"Flange face ({what}) from the tool: x_tool0 right, y up")
        shot(cam(p - 0.02 * Rt[:, 1], 0.26, az - 50.0, el + 30.0), opt(floor=False, frames=True),
             f"Flange ({what}) from below: the connector on -y_tool0")
    tiles = []
    for text, im in panels:
        bar = label_bar(text, W)
        t = Image.new("RGB", (W, H + bar.height), "white")
        t.paste(bar, (0, 0))
        t.paste(im, (0, bar.height))
        tiles.append(t)
    cols = 3
    rows = math.ceil(len(tiles) / cols)
    th = tiles[0].height
    sheet = Image.new("RGB", (cols * W, rows * th), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * W, (k // cols) * th))
    dr = ImageDraw.Draw(sheet)
    for c in range(1, cols):
        dr.line([(c * W, 0), (c * W, rows * th)], fill="black", width=1)
    for r in range(1, rows):
        dr.line([(0, r * th), (cols * W, r * th)], fill="black", width=1)
    out = HERE / ("ur5_cb3_render_primitives.png" if no_meshes else "ur5_cb3_render.png")
    sheet.save(out)
    print("wrote", out, sheet.size)


if __name__ == "__main__":
    main()
