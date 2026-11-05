"""Render groups of extracted solids (logs/20261104-cad_v3/extract) with MuJoCo (EGL) and label them with matplotlib.

Each render is a static MJCF of the STLs in their file's frame (mm, scaled to m), one colour per part, viewed by MuJoCo's
free camera; labels are drawn at the projection of each part's centre of mass (sr2hand.geometry.vectors)."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from sr2hand.geometry.vectors import project_free_camera

R = Path("/home/humanoid/Programs/hand")
PARTS = json.load(open(R / "logs/20261104-cad_v3/extract/parts.json"))["parts"]
BY_KEY = {f"{p['file']}/{p['name']}": p for p in PARTS}


def build_model(items, W=1600, H=1200, floor=False):
    """items: list of dict(key='hand/012', rgba=(r,g,b,a)). Returns (model, data)."""
    assets, geoms = [], []
    for i, it in enumerate(items):
        p = BY_KEY[it["key"]]
        assets.append(f'<mesh name="m{i}" file="{R / p["stl"]}" scale="0.001 0.001 0.001"/>')
        r, g, b, a = it["rgba"]
        geoms.append(f'<geom type="mesh" mesh="m{i}" rgba="{r:.3f} {g:.3f} {b:.3f} {a:.3f}" contype="0" conaffinity="0"/>')
    xml = f"""<mujoco>
  <compiler inertiafromgeom="false"/>
  <visual><global offwidth="{W}" offheight="{H}" fovy="30"/><rgba haze="1 1 1 1" fog="1 1 1 1"/><quality shadowsize="4096"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/></visual>
  <asset><texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>{''.join(assets)}</asset>
  <worldbody>
    <light pos="0.3 -0.3 0.6" dir="-0.4 0.4 -1" diffuse="0.35 0.35 0.35"/>
    <light pos="-0.3 0.3 -0.6" dir="0.4 -0.4 1" diffuse="0.25 0.25 0.25"/>
    {''.join(geoms)}
  </worldbody>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def render_views(items, views, out_png, title="", labels=None, W=1200, H=900, ncols=2, legend=None, label_size=8):
    """views: list of dict(lookat (m), distance (m), azimuth, elevation, title). labels: list of (text, point_mm) or
    None for the part names at their centres of mass. legend: list of (text, rgba)."""
    m, d = build_model(items, W, H)
    fovy = float(m.vis.global_.fovy)
    if labels is None:
        labels = [(BY_KEY[it["key"]]["name"], BY_KEY[it["key"]]["com"]) for it in items if it.get("label", True)]
    n = len(views)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * W / 150, nrows * H / 150 + (0.6 if title else 0)), dpi=150)
    axs = np.atleast_1d(axs).ravel()
    with mujoco.Renderer(m, H, W) as ren:
        for ax, v in zip(axs, views):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = v["lookat"]
            cam.distance = v["distance"]
            cam.azimuth = v["azimuth"]
            cam.elevation = v["elevation"]
            ren.update_scene(d, camera=cam)
            img = ren.render()
            ax.imshow(img)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.axis("off")
            ax.set_title(v.get("title", ""), fontsize=11)
            if labels:
                pts = np.array([np.asarray(pt, float) / 1000.0 for _, pt in labels])
                uv = project_free_camera(pts, v["lookat"], v["distance"], v["azimuth"], v["elevation"], fovy, W, H)
                c = np.array([W / 2, H / 2])
                for (txt, _), (u, vv) in zip(labels, uv):
                    if not (0 <= u <= W and 0 <= vv <= H):
                        continue
                    dirn = np.array([u, vv]) - c
                    nn = np.linalg.norm(dirn)
                    off = dirn / nn * 18 if nn > 1 else np.array([12, -12])
                    ax.annotate(txt, (u, vv), (u + off[0], vv + off[1]), fontsize=label_size, ha="center", va="center",
                                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.3", lw=0.4, alpha=0.85),
                                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.2"))
    for ax in axs[n:]:
        ax.axis("off")
    if legend:
        handles = [plt.Line2D([], [], ls="", marker="s", ms=9, mfc=tuple(c[:3]), mec="0.3") for _, c in legend]
        fig.legend(handles, [t for t, _ in legend], loc="lower center", ncol=min(6, len(legend)), fontsize=10,
                   frameon=False)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.05 if legend else 0, 1, 0.97 if title else 1))
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def centre_mm(keys):
    b = np.array([BY_KEY[k]["bbox"] for k in keys])
    lo, hi = b[:, :3].min(0), b[:, 3:].max(0)
    return (lo + hi) / 2, hi - lo
