#!/usr/bin/env python3
"""Scratch: MuJoCo renders of the D435's view on the cad_v3 hand for the assumed mounting of sensors_analysis.py
(analysis.json), plus overview renders. Hand solids from logs/20261104-cad_v3/extract (frame A, mm -> m). The camera
body, the 2020 post and the bracket are drawn as boxes because none of them is in the CAD export; the tag is the
tag36h11 id 6 texture on the static tag mount's +y face, with its x axis up (as measured in the 2026-08-30 probe).
Writes PNGs beside this file. Run under resguard (it loads 121 meshes)."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EX = REPO / "logs/20261104-cad_v3/extract"
AN = json.load(open(HERE / "analysis.json"))
TABLE_Z = -220.0                      # assumed table height in A (mm), 82 mm under the fingertip centroid
W_DEPTH, H_DEPTH, FOVY_DEPTH = 1280, 748, 58.0      # 87 x 58 deg
W_COLOR, H_COLOR, FOVY_COLOR = 1280, 716, 42.0      # 69 x 42 deg


def m(v):
    return " ".join(f"{x / 1000.0:.6f}" for x in v)


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return " ".join(f"{x:.8f}" for x in q)


def tag_quad(path, face_y, cx, cz, half=30.0, normal=+1):
    """A two-sided quad carrying the 60 mm patch (40 mm tag + 2-cell quiet zone) centred at (cx, face_y, cz) in A.
    Texture orientation: the OpenCV image's left edge -> world up (+z_A), its top edge -> the viewer's right, so the
    apriltag x axis points up and its y axis to the viewer's right (checked by detecting the rendered tag)."""
    y = face_y + 0.3 * normal
    right = np.array([-1.0, 0, 0]) if normal > 0 else np.array([1.0, 0, 0])   # viewer's right
    up = np.array([0, 0, 1.0])
    c = np.array([cx, y, cz])
    # image corners (u right, v up in OBJ convention): image-left -> up, image-top -> right
    def P(u, v):                       # u, v in [0, 1]: u along the image's width, v from its bottom to its top
        return c + ((v - 0.5) * 2 * half) * right + ((0.5 - u) * 2 * half) * up
    vs = [P(0, 0), P(1, 0), P(1, 1), P(0, 1)]
    with open(path, "w") as f:
        for p in vs:
            f.write("v %.6f %.6f %.6f\n" % tuple(p / 1000.0))
        for uv in ((0, 0), (1, 0), (1, 1), (0, 1)):
            f.write("vt %.3f %.3f\n" % uv)
        f.write("f 1/1 2/2 3/3\nf 1/1 3/3 4/4\nf 1/1 3/3 2/2\nf 1/1 4/4 3/3\n")


def scene(variant, extra_cams):
    v = AN["variants"][variant]
    Rb = np.column_stack([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])        # D435 body axes in A
    bc = np.array(v["body_centre_A"])
    post = v["post_centre_xy_A"]
    zc = bc[2]
    assets, geoms = [], []
    hl = {"000": "0.85 0.25 0.2 1", "005": "0.2 0.65 0.25 1"}
    for p in json.load(open(EX / "parts.json"))["parts"]:
        if p["file"] != "hand":
            continue
        key = f"h{p['name']}"
        assets.append(f'<mesh name="{key}" file="{REPO / p["stl"]}" scale="0.001 0.001 0.001" inertia="shell"/>')
        geoms.append(f'<geom type="mesh" mesh="{key}" rgba="{hl.get(p["name"], "0.78 0.78 0.76 1")}"/>')
    tp = AN["tag_plate"]
    tag_quad(HERE / "tag_quad.obj", tp["y_pos"], 0.5 * (tp["x"][0] + tp["x"][1]), -32.0)
    assets.append(f'<texture name="tag6" type="2d" file="{HERE / "tag36h11_id6_quiet2.png"}"/>'
                  '<material name="tag6" texture="tag6" emission="0.35" specular="0" shininess="0"/>'
                  f'<mesh name="tagq" file="{HERE / "tag_quad.obj"}" inertia="shell"/>')
    geoms.append('<geom type="mesh" mesh="tagq" material="tag6" contype="0" conaffinity="0"/>')
    # D435 body (box) and lenses (assumed placement)
    geoms.append(f'<geom type="box" pos="{m(bc)}" quat="{quat_of(Rb)}" size="{m([45.0, 12.5, 12.525])}" '
                 f'rgba="0.18 0.18 0.2 1" group="3"/>')
    zf = 12.525
    for name, xb, r, rgba in (("color", -32.5, 4.5, "0.5 0.05 0.08 1"), ("depth", -17.5, 4.0, "0.02 0.02 0.02 1"),
                              ("proj", 7.5, 4.0, "0.3 0.66 0.33 1"), ("rir", 32.5, 4.0, "0.02 0.02 0.02 1")):
        p = bc + Rb @ np.array([xb, 0.0, zf + 0.4])
        geoms.append(f'<geom type="cylinder" pos="{m(p)}" quat="{quat_of(Rb)}" size="{r / 1000:.5f} 0.0005" '
                     f'rgba="{rgba}" group="3"/>')
    # bracket (d435_mount_80-20, simplified) and the 2020 post (both assumptions)
    back_y = bc[1] + 12.525
    geoms.append(f'<geom type="box" pos="{m([post[0], back_y + 2.0, zc])}" size="{m([30.0, 2.0, 12.5])}" '
                 f'rgba="0.95 0.55 0.15 0.9"/>')
    geoms.append(f'<geom type="box" pos="{m([post[0], back_y + 2.0, zc])}" size="{m([10.8, 2.0, 23.3])}" '
                 f'rgba="0.95 0.55 0.15 0.9"/>')
    zb = zc - 23.3 - 5.0
    geoms.append(f'<geom type="box" pos="{m([post[0], post[1], 0.5 * (0.0 + zb)])}" '
                 f'size="{m([10.0, 10.0, 0.5 * (0.0 - zb)])}" rgba="0.6 0.62 0.68 0.85"/>')
    # object between the fingertips and an assumed table
    tc = AN["fingertip_centroid_A"]
    geoms.append(f'<geom type="cylinder" pos="{m(tc)}" quat="0.70710678 0 0.70710678 0" size="0.0125 0.05" '
                 f'rgba="0.95 0.85 0.3 1"/>')
    geoms.append(f'<geom type="plane" pos="{m([-110, 120, TABLE_Z])}" size="0.6 0.6 0.01" rgba="0.55 0.5 0.45 1"/>')
    cams = []
    for k, (Tf, fovy) in {"depth": (v["T_A_from_depth_optical"], FOVY_DEPTH),
                          "color": (v["T_A_from_color_optical"], FOVY_COLOR)}.items():
        Tm = np.array(Tf).reshape(4, 4)
        xo, yo = Tm[:3, 0], Tm[:3, 1]
        ax = " ".join(f"{a:.6f}" for a in list(xo) + list(-yo))              # MuJoCo camera: x right, y up
        cams.append(f'<camera name="{k}" pos="{m(Tm[:3, 3])}" xyaxes="{ax}" fovy="{fovy}"/>')
    cams += extra_cams
    xml = f"""<mujoco><compiler angle="radian"/>
<visual><global offwidth="1600" offheight="1200"/><quality shadowsize="4096"/><map znear="0.002" zfar="20"/>
<headlight ambient="0.5 0.5 0.5" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/></visual>
<asset>{''.join(assets)}</asset>
<worldbody><light pos="0 0 1" dir="0 0 -1" directional="true" diffuse="0.35 0.35 0.35"/>
<light pos="0.3 0.6 0.2" dir="-0.4 -1 -0.3" directional="true" diffuse="0.25 0.25 0.25"/>
{''.join(geoms)}{''.join(cams)}</worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def look_cam(name, eye, tgt, fovy=45.0):
    f = np.subtract(tgt, eye).astype(float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    ax = " ".join(f"{a:.6f}" for a in list(x) + list(y))
    return f'<camera name="{name}" pos="{m(eye)}" xyaxes="{ax}" fovy="{fovy}"/>'


def project(Tf, pts, w, h, fovy):
    Tm = np.array(Tf).reshape(4, 4)
    f = (h / 2.0) / math.tan(math.radians(fovy / 2.0))
    out = {}
    for k, p in pts.items():
        c = Tm[:3, :3].T @ (np.asarray(p) - Tm[:3, 3])
        out[k] = (f * c[0] / c[2] + w / 2.0, f * c[1] / c[2] + h / 2.0, c[2])
    return out, f


def main():
    over = [look_cam("ov1", [180, 420, -40], [-60, 150, -80], 55), look_cam("ov2", [260, -60, -120], [-30, 170, -60], 55),
            look_cam("ov3", [-120, 470, -260], [-60, 170, -90], 50), look_cam("ov4", [90, 330, -60], [5, 240, -40], 50)]
    stats = {}
    for variant in ("highest", "tips_level"):
        mdl = scene(variant, over if variant == "highest" else [])
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
            plt.imsave(HERE / f"view_{variant}_{cam}_raw.png", img)
            uv, f = project(v[key], pts, w, h, fovy)
            stats[f"{variant}_{cam}"] = dict(f_px=f, w=w, h=h, uv={k: [round(a, 1) for a in x] for k, x in uv.items()})
            fig, ax = plt.subplots(figsize=(13, 13 * h / w + 0.9))
            ax.imshow(img)
            for k, (u, vv, z) in uv.items():
                inside = z > 0 and 0 <= u < w and 0 <= vv < h
                ax.plot(np.clip(u, 0, w - 1), np.clip(vv, 0, h - 1), "o" if inside else "x", ms=11, mew=2.5,
                        color="cyan" if inside else "magenta")
                ax.annotate(k.split(" (")[0], (np.clip(u, 0, w - 1), np.clip(vv, 0, h - 1)), color="white",
                            fontsize=13, xytext=(6, -8), textcoords="offset points",
                            bbox=dict(fc="black", alpha=0.55, lw=0))
            hf = 2 * math.degrees(math.atan(w / 2 / f))
            ax.set_title(f"D435 {cam} view, assumed mount '{variant}' (camera centre z_A = {v['body_centre_A'][2]:.1f} mm),"
                         f" {hf:.0f} x {fovy:.0f} deg; x = outside the image", fontsize=13)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(HERE / f"view_{variant}_{cam}.png", dpi=80)
            plt.close(fig)
        if variant == "highest":
            imgs = []
            for c in ("ov1", "ov2", "ov3", "ov4"):
                r = mujoco.Renderer(mdl, 900, 1200)
                opt = mujoco.MjvOption()
                opt.geomgroup[3] = 1
                r.update_scene(d, camera=c, scene_option=opt)
                imgs.append(r.render().copy())
                r.close()
            fig, axs = plt.subplots(2, 2, figsize=(16, 12.4))
            titles = ["from the thumb end, +x side", "from the index side (-y_A), +x", "from below, thumb end",
                      "close-up: D435 on the fork"]
            for a, im, t in zip(axs.ravel(), imgs, titles):
                a.imshow(im)
                a.set_title(t, fontsize=14)
                a.axis("off")
            fig.suptitle("Assumed D435 mounting (dark box on an orange bracket and a grey 2020 post, none of them in the "
                         "CAD), d435_plate red, static tag mount green, object yellow, table assumed at z_A = -220 mm",
                         fontsize=13)
            fig.tight_layout()
            fig.savefig(HERE / "overview_assumed_mount.png", dpi=80)
            plt.close(fig)
    (HERE / "render_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1)[:3000])


if __name__ == "__main__":
    main()
