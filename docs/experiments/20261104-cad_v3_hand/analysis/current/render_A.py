"""Render the hand assembly (file 'hand', frame A, mm) in MuJoCo from four views, parts coloured by sub-assembly, with
the current cad_v2 palm_plate drawn where it sits in P if P's origin is the yaw-centre plane (z_A -75.9) and the axes
x_P = -y_A, y_P = x_A (inferred). Visual only. Writes render_A.png beside this file."""
import json
import os
from pathlib import Path

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path("/home/humanoid/Programs/hand")
EX = ROOT / "logs/20261104-cad_v3/extract"
OUT = ROOT / "logs/20261104-cad_v3/current"
P = json.load(open(EX / "parts.json"))["parts"]
COL = {"hand v3 (anthro config)": "0.6 0.6 0.62 1", "gantry_v3 <1>": "0.25 0.45 0.85 1", "gantry_v3 <2>": "0.2 0.7 0.5 1",
       "gantry_v3_thumb <1>": "0.85 0.35 0.3 1", "Finger v3 <1>": "0.9 0.75 0.3 1", "Finger v3 <2>": "0.9 0.75 0.3 1",
       "Finger v3 <3>": "0.9 0.6 0.2 1"}
assets, geoms = [], []
for p in P:
    if p["file"] != "hand":
        continue
    n = p["name"]
    assets.append(f'<mesh name="m{n}" file="{ROOT / p["stl"]}" scale="0.001 0.001 0.001"/>')
    a = 0.35 if n in ("001", "002") else 1.0
    rgba = COL[p["chain"][0]].rsplit(" ", 1)[0] + f" {a}"
    geoms.append(f'<geom type="mesh" mesh="m{n}" rgba="{rgba}" contype="0" conaffinity="0"/>')
# the cad_v2 palm_plate (P: centre z 25 mm, 170 x 180 x 3 mm) in A, with P's origin at (-111, 120.2, -75.9) (provisional)
o = np.array([-111.0, 120.2, -75.9])
plate_c = o + np.array([0, 0, 25.0])
geoms.append(f'<geom type="box" pos="{" ".join(f"{v/1000:.5f}" for v in plate_c)}" size="0.090 0.085 0.0015" '
             f'rgba="1 0 1 0.35" contype="0" conaffinity="0"/>')        # x_A half 90 (= y_P), y_A half 85 (= x_P)
geoms.append(f'<geom type="sphere" pos="{" ".join(f"{v/1000:.5f}" for v in o)}" size="0.004" rgba="1 0 1 1" contype="0" conaffinity="0"/>')
xml = f"""<mujoco><compiler inertiafromgeom="false"/><visual><global offwidth="1600" offheight="1200"/>
<headlight ambient=".45 .45 .45" diffuse=".6 .6 .6"/></visual><asset>{''.join(assets)}</asset>
<worldbody><light pos="0 0 1" dir="0 0 -1" directional="true"/><light pos="0 0 -1" dir="0 0 1" directional="true" diffuse=".4 .4 .4"/>
<body name="hand" pos="0.111 -0.12 0">{''.join(geoms)}</body></worldbody></mujoco>"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
W, H = 800, 600
r = mujoco.Renderer(m, H, W)
views = [("side, looking along +y_A (x_A right, z_A up)", 90.0, 0.0), ("side, looking along -x_A (y_A right)", 180.0, 0.0),
         ("from below (z_A toward the viewer)", 90.0, 89.0), ("iso", 135.0, -25.0)]
tiles = []
for title, az, el in views:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, -0.06]
    cam.distance = 0.75
    cam.azimuth, cam.elevation = az, el
    r.update_scene(d, cam)
    tiles.append(r.render().copy())
r.close()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
for a, t, (title, *_r) in zip(ax.ravel(), tiles, views):
    a.imshow(t); a.set_title(title, fontsize=14); a.axis("off")
fig.tight_layout()
fig.savefig(OUT / "render_A.png", dpi=80)
print("wrote", OUT / "render_A.png")
