#!/usr/bin/env python3
"""Scratch: render the solids of the hand file (frame A, mm -> m) in MuJoCo with chosen parts coloured, to identify the
camera and tag mounts. Usage: render_hand.py OUT.png 'name:rgba,...' view1;view2... (view = lookat_x,y,z,dist,az,el in A mm)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO = Path("/home/humanoid/Programs/hand")
EX = REPO / "logs/20261104-cad_v3/extract"


def build(highlight, hide=(), files=("hand",)):
    parts = [p for p in json.load(open(EX / "parts.json"))["parts"] if p["file"] in files]
    assets, geoms = [], []
    for p in parts:
        key = f"{p['file']}_{p['name']}"
        if p["name"] in hide:
            continue
        rgba = highlight.get(p["name"], "0.82 0.82 0.80 1")
        assets.append(f'<mesh name="{key}" file="{REPO / p["stl"]}" scale="0.001 0.001 0.001" inertia="shell"/>')
        geoms.append(f'<geom type="mesh" mesh="{key}" rgba="{rgba}"/>')
    xml = f"""<mujoco><compiler angle="radian"/>
<visual><global offwidth="1600" offheight="1200"/><quality shadowsize="4096"/>
<headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/></visual>
<asset>{''.join(assets)}</asset>
<worldbody><light pos="0 0 1" dir="0 0 -1" directional="true" diffuse="0.4 0.4 0.4"/>
<light pos="0.5 -0.5 0.3" dir="-1 1 -0.5" directional="true" diffuse="0.25 0.25 0.25"/>
{''.join(geoms)}</worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def render(m, views, w=1200, h=900):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, h, w)
    out = []
    for v in views:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.array(v[:3]) / 1000.0
        cam.distance, cam.azimuth, cam.elevation = v[3] / 1000.0, v[4], v[5]
        r.update_scene(d, camera=cam)
        out.append(r.render().copy())
    r.close()
    return out


if __name__ == "__main__":
    outp = sys.argv[1]
    hl = {}
    for tok in sys.argv[2].split(","):
        if tok:
            n, c = tok.split(":")
            hl[n] = c.replace("_", " ")
    views = [[float(x) for x in s.split(",")] for s in sys.argv[3].split(";")]
    hide = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else ()
    m = build(hl, hide)
    imgs = render(m, views)
    n = len(imgs)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    axs = np.atleast_1d(axs).ravel()
    for a, im, v in zip(axs, imgs, views):
        a.imshow(im)
        a.set_title("lookat %s  dist %.0f  az %.0f  el %.0f" % (v[:3], v[3], v[4], v[5]), fontsize=13)
        a.axis("off")
    for a in axs[n:]:
        a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=[float(x) for x in c.split()[:3]]) for c in hl.values()]
    fig.legend(handles, list(hl.keys()), loc="lower center", ncol=min(8, len(hl)), fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(outp, dpi=80)
    print("wrote", outp)
