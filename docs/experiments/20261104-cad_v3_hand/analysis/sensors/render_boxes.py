#!/usr/bin/env python3
"""Scratch: the D435's assumed view drawn with MuJoCo primitives only (no mesh is loaded), for when the resguard gate
is closed: every hand solid as its inertia-equivalent box (principal axes and moments from parts.json), the d435_plate,
the static tag mount and the base plate as exact boxes from their faces, the 40 mm tag36h11 id 6 as 5 mm cells on a
45 x 50 mm white patch (the plate face; the 2026-08-30 probe shows the paper trimmed to about this), x axis up.
Usage: render_boxes.py [variant ...]"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render_views import (AN, FOVY_COLOR, FOVY_DEPTH, H_COLOR, H_DEPTH, TABLE_Z, W_COLOR, W_DEPTH, look_cam, m,  # noqa
                          project, quat_of)

REPO = HERE.parents[2]
PARTS = {p["name"]: p for p in json.load(open(REPO / "logs/20261104-cad_v3/extract/parts.json"))["parts"]
         if p["file"] == "hand"}
GREY = "0.78 0.78 0.76 1"


def box(lo, hi, rgba):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return f'<geom type="box" pos="{m(0.5 * (lo + hi))}" size="{m(0.5 * (hi - lo))}" rgba="{rgba}"/>'


def inertia_box(p, rgba=GREY):
    I = np.array(p["inertia_mm5"])
    V = p["volume"]
    w, E = np.linalg.eigh(I)
    if np.linalg.det(E) < 0:
        E[:, 2] *= -1
    d2 = 6.0 * np.array([w[1] + w[2] - w[0], w[0] + w[2] - w[1], w[0] + w[1] - w[2]]) / V
    half = 0.5 * np.sqrt(np.clip(d2, 1.0, None))
    return (f'<geom type="box" pos="{m(p["com"])}" quat="{quat_of(E)}" size="{m(half)}" rgba="{rgba}"/>')


def tag_cells(face_y, cx, cz, normal=+1):
    """5 mm cells of the 40 mm tag (8 x 8 cells of the OpenCV image's inner 8 x 8), white patch 45 x 50 mm behind."""
    img = np.array(Image.open(HERE / "tag36h11_id6_quiet2.png").convert("L"))
    cell = img.shape[0] // 12
    right = np.array([-1.0, 0, 0]) if normal > 0 else np.array([1.0, 0, 0])
    up = np.array([0, 0, 1.0])
    y0 = face_y + normal * 0.25
    out = [box([cx - 22.5, y0 - 0.2, cz - 25.0], [cx + 22.5, y0 + 0.2, cz + 25.0], "1 1 1 1")]
    for r in range(2, 10):                     # image rows (top to bottom) inside the quiet zone
        for c in range(2, 10):                 # image columns (left to right)
            if img[r * cell + cell // 2, c * cell + cell // 2] > 128:
                continue
            # image top -> the viewer's right, image left -> up (apriltag x up, y to the viewer's right)
            ctr = np.array([cx, y0 + normal * 0.3, cz]) + (5.5 - r) * 5.0 * right + (5.5 - c) * 5.0 * up
            lo = ctr - np.abs(2.5 * right) - np.array([0, 0.15, 0]) - 2.5 * up
            hi = ctr + np.abs(2.5 * right) + np.array([0, 0.15, 0]) + 2.5 * up
            out.append(box(lo, hi, "0.02 0.02 0.02 1"))
    return out


def scene(variant, extra):
    v = AN["variants"][variant]
    Rb = np.column_stack([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])
    bc = np.array(v["body_centre_A"])
    post = v["post_centre_xy_A"]
    zc = bc[2]
    g = []
    for n, p in PARTS.items():
        if n in ("000", "001", "002", "004", "005"):
            continue
        g.append(inertia_box(p))
    # robot mount and adaptor as bounding boxes, base plate as its outline's two boxes
    g.append(box(PARTS["001"]["bbox"][:3], PARTS["001"]["bbox"][3:], GREY))
    g.append(box(PARTS["004"]["bbox"][:3], PARTS["004"]["bbox"][3:], "0.95 0.5 0.1 1"))
    g.append(box([-279.3, 0.0, 0.0], [11.0, 242.9, 3.0], "0.6 0.6 0.66 1"))
    g.append(box([11.0, 220.4, 0.0], [25.8, 242.9, 3.0], "0.6 0.6 0.66 1"))
    red, green = "0.85 0.25 0.2 1", "0.2 0.65 0.25 1"
    # d435_plate (solid 000) from its faces
    g += [box([3.0, 111.0, -4.0], [11.0, 123.0, 0.0], red), box([-14.2, 123.0, -4.0], [11.0, 267.1, 0.0], red),
          box([11.0, 220.4, -4.0], [25.8, 295.5, 0.0], red), box([6.2, 267.1, -4.0], [11.0, 295.5, 0.0], red),
          box([-14.2, 267.1, -10.0], [6.2, 271.1, 0.0], red), box([-14.2, 291.5, -10.0], [6.2, 295.5, 0.0], red)]
    # static tag mount (solid 005)
    g += [box([0.1, 103.3, -57.0], [45.1, 105.3, -7.0], green), box([0.1, 103.3, -7.0], [14.2, 120.0, -4.0], green),
          box([11.2, 103.3, -4.0], [14.2, 120.0, 18.0], green)]
    tp = AN["tag_plate"]
    g += tag_cells(tp["y_pos"], 0.5 * (tp["x"][0] + tp["x"][1]), -32.0)
    # assumed camera, bracket and post
    g.append(f'<geom type="box" pos="{m(bc)}" quat="{quat_of(Rb)}" size="{m([45.0, 12.5, 12.525])}" '
             f'rgba="0.18 0.18 0.2 1" group="3"/>')
    for xb, r, rgba in ((-32.5, 4.5, "0.5 0.05 0.08 1"), (-17.5, 4.0, "0.02 0.02 0.02 1"),
                        (7.5, 4.0, "0.3 0.66 0.33 1"), (32.5, 4.0, "0.02 0.02 0.02 1")):
        p = bc + Rb @ np.array([xb, 0.0, 12.525 + 0.4])
        g.append(f'<geom type="cylinder" pos="{m(p)}" quat="{quat_of(Rb)}" size="{r / 1000:.5f} 0.0005" rgba="{rgba}" group="3"/>')
    back_y = bc[1] + 12.525
    g.append(box([post[0] - 30, back_y, zc - 12.5], [post[0] + 30, back_y + 4, zc + 12.5], "0.95 0.55 0.15 1"))
    g.append(box([post[0] - 10.8, back_y, zc - 23.3], [post[0] + 10.8, back_y + 4, zc + 23.3], "0.95 0.55 0.15 1"))
    zb = zc - 23.3 - 5.0
    g.append(box([post[0] - 10, post[1] - 10, zb], [post[0] + 10, post[1] + 10, 0.0], "0.6 0.62 0.68 1"))
    tc = AN["fingertip_centroid_A"]
    g.append(f'<geom type="cylinder" pos="{m(tc)}" quat="0.70710678 0 0.70710678 0" size="0.0125 0.05" '
             f'rgba="0.95 0.85 0.3 1"/>')
    g.append(f'<geom type="plane" pos="{m([-110, 120, TABLE_Z])}" size="0.6 0.6 0.01" rgba="0.55 0.5 0.45 1"/>')
    cams = []
    for k, (Tf, fovy) in {"depth": (v["T_A_from_depth_optical"], FOVY_DEPTH),
                          "color": (v["T_A_from_color_optical"], FOVY_COLOR)}.items():
        Tm = np.array(Tf).reshape(4, 4)
        ax = " ".join(f"{a:.6f}" for a in list(Tm[:3, 0]) + list(-Tm[:3, 1]))
        cams.append(f'<camera name="{k}" pos="{m(Tm[:3, 3])}" xyaxes="{ax}" fovy="{fovy}"/>')
    xml = f"""<mujoco><compiler angle="radian"/>
<visual><global offwidth="1600" offheight="1200"/><map znear="0.002" zfar="20"/>
<headlight ambient="0.5 0.5 0.5" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/></visual>
<worldbody><light pos="0 0 1" dir="0 0 -1" directional="true" diffuse="0.35 0.35 0.35"/>
{''.join(g)}{''.join(cams)}{''.join(extra)}</worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def main():
    variants = sys.argv[1:] or ["highest", "tips_level"]
    over = [look_cam("ov1", [180, 420, -40], [-60, 150, -80], 55), look_cam("ov2", [260, -60, -120], [-30, 170, -60], 55)]
    for variant in variants:
        mdl = scene(variant, over)
        d = mujoco.MjData(mdl)
        mujoco.mj_forward(mdl, d)
        v = AN["variants"][variant]
        pts = dict(AN["fingertips_A"], **{"fingertip centroid": AN["fingertip_centroid_A"],
                                          "tag centre": AN["tag_centre_pos_face_A"]})
        for cam, (w, h, fovy, key) in {"depth": (W_DEPTH, H_DEPTH, FOVY_DEPTH, "T_A_from_depth_optical"),
                                       "color": (W_COLOR, H_COLOR, FOVY_COLOR, "T_A_from_color_optical")}.items():
            r = mujoco.Renderer(mdl, h, w)
            opt = mujoco.MjvOption()
            opt.geomgroup[3] = 0
            r.update_scene(d, camera=cam, scene_option=opt)
            img = r.render().copy()
            r.close()
            plt.imsave(HERE / f"boxes_{variant}_{cam}_raw.png", img)
            uv, f = project(v[key], pts, w, h, fovy)
            fig, ax = plt.subplots(figsize=(13, 13 * h / w + 0.9))
            ax.imshow(img)
            for k, (u, vv, z) in uv.items():
                inside = z > 0 and 0 <= u < w and 0 <= vv < h
                ax.plot(np.clip(u, 0, w - 1), np.clip(vv, 0, h - 1), "o" if inside else "x", ms=11, mew=2.5,
                        color="cyan" if inside else "magenta")
                ax.annotate(k.split(" (")[0], (np.clip(u, 0, w - 1), np.clip(vv, 0, h - 1)), color="white", fontsize=13,
                            xytext=(6, -8), textcoords="offset points", bbox=dict(fc="black", alpha=0.55, lw=0))
            hf = 2 * math.degrees(math.atan(w / 2 / f))
            ax.set_title(f"D435 {cam} view (solids as inertia boxes), assumed mount '{variant}', camera centre z_A = "
                         f"{v['body_centre_A'][2]:.1f} mm, {hf:.0f} x {fovy:.0f} deg; x = outside the image", fontsize=12)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(HERE / f"boxes_{variant}_{cam}.png", dpi=80)
            plt.close(fig)
        imgs = []
        for c in ("ov1", "ov2"):
            r = mujoco.Renderer(mdl, 900, 1200)
            opt = mujoco.MjvOption()
            opt.geomgroup[3] = 1
            r.update_scene(d, camera=c, scene_option=opt)
            imgs.append(r.render().copy())
            r.close()
        fig, axs = plt.subplots(1, 2, figsize=(16, 6.4))
        for a, im, t in zip(axs, imgs, ["from the thumb end, +x side", "from the index side (-y_A), +x"]):
            a.imshow(im)
            a.set_title(t, fontsize=14)
            a.axis("off")
        fig.suptitle(f"Assumed D435 mount '{variant}' (solids as inertia boxes; camera dark, bracket orange, post grey)",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(HERE / f"boxes_{variant}_overview.png", dpi=80)
        plt.close(fig)
        print("rendered", variant)


if __name__ == "__main__":
    main()
