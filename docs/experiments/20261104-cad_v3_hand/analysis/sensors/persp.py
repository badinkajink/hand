#!/usr/bin/env python3
"""Scratch: perspective MuJoCo renders of chosen hand solids (frame A, mm), camera given by eye and target in A (mm).
Usage: persp.py OUT.png 'name:r_g_b_a,...' 'eye_x,eye_y,eye_z,tgt_x,tgt_y,tgt_z,fovy;...' [only=1: draw only the named]
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


def look_quat(eye, tgt, up=(0, 0, 1)):
    """MuJoCo camera quaternion (wxyz) looking from eye to tgt: camera -z along the view, +y up."""
    f = np.asarray(tgt, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    upv = np.asarray(up, float)
    if abs(f @ upv) > 0.99:
        upv = np.array([0, 1.0, 0])
    x = np.cross(f, upv)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    R = np.stack([x, y, -f], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def scene_xml(colors, only=False, extra_assets="", extra_geoms="", cams=()):
    parts = [p for p in json.load(open(EX / "parts.json"))["parts"] if p["file"] == "hand"]
    assets, geoms = [], []
    for p in parts:
        if only and p["name"] not in colors:
            continue
        key = f"h{p['name']}"
        rgba = colors.get(p["name"], "0.82 0.82 0.80 1")
        assets.append(f'<mesh name="{key}" file="{REPO / p["stl"]}" scale="0.001 0.001 0.001" inertia="shell"/>')
        geoms.append(f'<geom type="mesh" mesh="{key}" rgba="{rgba}"/>')
    camxml = "".join(cams)
    return f"""<mujoco><compiler angle="radian"/>
<visual><global offwidth="1600" offheight="1200"/><quality shadowsize="4096"/><map znear="0.001" zfar="20"/>
<headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/></visual>
<asset>{''.join(assets)}{extra_assets}</asset>
<worldbody><light pos="0 0 1" dir="0 0 -1" directional="true" diffuse="0.4 0.4 0.4"/>
<light pos="0.5 -0.5 0.3" dir="-1 1 -0.5" directional="true" diffuse="0.25 0.25 0.25"/>
{''.join(geoms)}{extra_geoms}{camxml}</worldbody></mujoco>"""


def cam_xml(name, eye, tgt, fovy, up=(0, 0, 1)):
    q = look_quat(eye, tgt, up)
    e = np.asarray(eye, float) / 1000.0
    return (f'<camera name="{name}" pos="{e[0]:.6f} {e[1]:.6f} {e[2]:.6f}" '
            f'quat="{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}" fovy="{fovy}"/>')


if __name__ == "__main__":
    outp = sys.argv[1]
    colors = {}
    for tok in sys.argv[2].split(","):
        if tok:
            n, c = tok.split(":", 1)
            colors[n] = c.replace("_", " ")
    views = [[float(x) for x in s.split(",")] for s in sys.argv[3].split(";")]
    only = len(sys.argv) > 4 and sys.argv[4] == "1"
    cams = [cam_xml(f"c{i}", v[:3], v[3:6], v[6]) for i, v in enumerate(views)]
    m = mujoco.MjModel.from_xml_string(scene_xml(colors, only, cams=cams))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 900, 1200)
    imgs = []
    for i in range(len(views)):
        r.update_scene(d, camera=f"c{i}")
        imgs.append(r.render().copy())
    r.close()
    cols = 2 if len(imgs) > 1 else 1
    rows = (len(imgs) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    axs = np.atleast_1d(axs).ravel()
    for a, im, v in zip(axs, imgs, views):
        a.imshow(im)
        a.set_title("eye (%.0f, %.0f, %.0f) -> (%.0f, %.0f, %.0f) mm in A" % tuple(v[:6]), fontsize=13)
        a.axis("off")
    for a in axs[len(imgs):]:
        a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=[float(x) for x in c.split()[:3]]) for c in colors.values()]
    fig.legend(handles, list(colors), loc="lower center", ncol=min(8, len(colors)), fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(outp, dpi=80)
    print("wrote", outp)
