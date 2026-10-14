#!/usr/bin/env python3
"""Figures and films of the cad_v1 finger for its result page (2026-10-13).

Writes into docs/experiments/20261013-cad_v1_finger/media/:
  parts.png         every CAD part with the bores the assembly is read from (red: 10.2 mm bearing bore,
                    blue: 5.2 mm slotted bore, green: servo spline)
  assembly.png      the assembled finger from both sides beside the owner's CAD views (assets/cad_v1/diagram.png)
  compare.png       the CAD finger with real_v1's capsule finger drawn translucent on the same mount, straight
                    and at the real_v1 'open' angles
  spheres.png       collision spheres and pads over the meshes, per body
  sweep.mp4/.jpg    one joint sweep shown three ways: CAD meshes, collision spheres and pads, real_v1 capsules
  hand.png          the three-finger hand at the 'open' key

    MUJOCO_GL=egl .venv/bin/python scripts/cad_v1_finger_media.py [--only parts,assembly,...]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from morphohand.geometry import cad_features as cf  # noqa: E402
import build_cad_v1_finger as B  # noqa: E402

MODEL = B.OUT
MEDIA = REPO / "docs/experiments/20261013-cad_v1_finger/media"
WHITE = '<texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>'


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


COLUMN_PX = 828  # width of the page's text column (retro_style), where every figure is shown


def label_px(total_width: int) -> int:
    """Font size in image pixels that displays at 14 px once an image total_width wide is fitted to the column."""
    return int(np.ceil(14 * total_width / COLUMN_PX))


def label_bar(img: Image.Image, text: str, size=20, h=None) -> Image.Image:
    h = h or int(1.6 * size)
    out = Image.new("RGB", (img.width, img.height + h), "white")
    out.paste(img, (0, h))
    ImageDraw.Draw(out).text((10, (h - size) // 2), text, fill=(0, 0, 0), font=font(size))
    return out


def tile(images, cols):
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(images):
        out.paste(im, ((i % cols) * w, (i // cols) * h))
    return out


def renderer(m, w, h):
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    m.vis.global_.offheight = max(m.vis.global_.offheight, h)
    return mujoco.Renderer(m, height=h, width=w)


def cam(lookat, dist, az, el):
    c = mujoco.MjvCamera()
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


def opt_groups(groups):
    o = mujoco.MjvOption()
    for g in range(6):
        o.geomgroup[g] = 1 if g in groups else 0
    return o


# ------------------------------------------------------------------------------------------------ parts

def parts_figure():
    A = B.derive()
    cyl_col = {"bearing": "0.9 0.1 0.1 1", "slot": "0.1 0.3 0.95 1", "spline": "0.1 0.7 0.2 1"}
    panels = []
    for name, title in [("carriage", "Carriage"), ("yaw_link", "Yaw link"), ("mcp_link", "MCP link"), ("dip_link", "DIP link"),
                        ("tip", "TPU tip"), ("servo", "SCS0009 servo")]:
        m = A["meshes"][name]
        feats = []
        cyls = cf.cylinders(m) if name != "tip" else []
        marks = (("spline", B.SPLINE_D, "boss", 300),) if name == "servo" else (("bearing", B.BEARING_D, "hole", 300), ("slot", B.SLOT_D, "hole", 300))
        for kind, d, k, cover in marks:
            for c in cf.find(cyls, d, k, tol=0.06, min_cover=cover):
                feats.append((kind, c))
        span = float(np.linalg.norm(m.extents))
        tmp = MEDIA / f"_part_{name}.stl"
        m.export(tmp)
        geoms = [f'<geom type="mesh" mesh="m" rgba="0.62 0.55 0.85 1"/>']
        for kind, c in feats:
            a, p = c.axis, c.point
            p0, p1 = (p - a * 0.75 * span) / 1000, (p + a * 0.75 * span) / 1000
            geoms.append(f'<geom type="cylinder" fromto="{B.fmt(p0)} {B.fmt(p1)}" size="0.0006" rgba="{cyl_col[kind]}" contype="0" conaffinity="0"/>')
        xml = (f'<mujoco><asset>{WHITE}<mesh name="m" file="{tmp}" scale="0.001 0.001 0.001"/></asset>'
               '<visual><headlight ambient=".5 .5 .5" diffuse=".55 .55 .55"/></visual>'
               f'<worldbody><light pos="0.3 -0.4 0.6" dir="-0.4 0.5 -0.7" directional="true" diffuse=".4 .4 .4"/>{"".join(geoms)}</worldbody></mujoco>')
        mm = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(mm)
        mujoco.mj_forward(mm, d)
        r = renderer(mm, 520, 520)
        c = cam(m.bounds.mean(0) / 1000, 2.2 * span / 1000, 135, -28)
        r.update_scene(d, c)
        panels.append(label_bar(Image.fromarray(r.render()), title, size=label_px(3 * 520)))
        r.close()
        tmp.unlink()
    tile(panels, 3).save(MEDIA / "parts.png")


# ---------------------------------------------------------------------------------------------- assembly

def assembly_figure():
    m = mujoco.MjModel.from_xml_path(str(MODEL / "finger.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    m.vis.global_.fovy = 8
    r = renderer(m, 560, 900)
    out = []
    diag = Image.open(REPO / "assets/cad_v1/diagram.png").convert("RGB")
    mid = diag.crop((835, 10, 1215, 1087))    # without the diagram's black border rows
    right = diag.crop((1215, 0, 1812, 1046))
    mid, right = (im.resize((int(im.width * 900 / im.height), 900)) for im in (mid, right))
    fs = label_px(mid.width + right.width + 2 * 560)
    for az, title in [(270, "Model, +y"), (90, "Model, \u2212y")]:
        c = cam([0.012, 0, 0.108], 1.42, az, 0)
        r.update_scene(d, c, opt_groups({2}))
        out.append(label_bar(Image.fromarray(r.render()), title, size=fs))
    r.close()
    mid = label_bar(mid, "CAD, +y", size=fs)
    right = label_bar(right, "CAD, \u2212y, flexed", size=fs)
    tile_w = [mid, out[0], right, out[1]]
    W = Image.new("RGB", (sum(i.width for i in tile_w), max(i.height for i in tile_w)), "white")
    x = 0
    for im in tile_w:
        W.paste(im, (x, 0))
        x += im.width
    W.save(MEDIA / "assembly.png")


# -------------------------------------------------------------------------------- comparison model (CAD + real_v1)

def combined_xml() -> str:
    """finger.xml plus real_v1's index finger (capsules, group 5, translucent) on the same mount, joints rv1_*."""
    cad = (MODEL / "finger.xml").read_text()
    rv1 = (REPO / "assets/mjcf/real_v1/real_hand.xml").read_text()
    s = rv1[rv1.index('<body name="index_mount"'):]
    depth, i = 0, 0
    for mt in re.finditer(r"<(/?)body\b[^>]*?(/?)>", s):
        if mt.group(1):
            depth -= 1
        elif not mt.group(2):
            depth += 1
        if depth == 0:
            i = mt.end()
            break
    body = s[:i].replace("index_", "rv1_")
    body = body.replace('<body name="rv1_mount" pos="0.050000 0.055000 0">', '<body name="rv1_mount" pos="0 0 0.12">')
    body = re.sub(r'<joint class="morph" name="rv1_[xy]"[^>]*/>', "", body)
    body = re.sub(r'<geom ([^>]*?)material="finger_mat"', r'<geom \1rgba="0.1 0.65 0.6 0.33" group="5" contype="0" conaffinity="0"', body)
    cad = cad.replace("  </worldbody>", body + "\n  </worldbody>")
    cad = cad.replace('<compiler angle="radian" autolimits="true" meshdir="meshes"/>',
                      f'<compiler angle="radian" autolimits="true" meshdir="{MODEL / "meshes"}"/>')
    cad = cad.replace("<asset>", "<asset>" + WHITE, 1)
    return re.sub(r"<keyframe>.*?</keyframe>", "", cad, flags=re.S)


def set_q(m, d, yaw, mcp, pip, rv1=True):
    for pre in ("index", "rv1") if rv1 else ("index",):
        d.qpos[m.joint(f"{pre}_yaw").qposadr[0]] = yaw
        d.qpos[m.joint(f"{pre}_mcp").qposadr[0]] = mcp
        d.qpos[m.joint(f"{pre}_pip").qposadr[0]] = pip
    mujoco.mj_forward(m, d)


def compare_figure():
    m = mujoco.MjModel.from_xml_string(combined_xml())
    d = mujoco.MjData(m)
    m.vis.global_.fovy = 10
    r = renderer(m, 620, 820)
    out = []
    for q, title in [((0, 0, 0), "Straight"), ((0, 0.55, 0.55), "MCP and PIP 32\u00b0"), ((0, 1.4, 1.4), "MCP and PIP 80\u00b0")]:
        set_q(m, d, *q)
        c = cam([-0.006, 0, 0.098], 1.02, 270, -4)
        r.update_scene(d, c, opt_groups({2, 5}))
        out.append(label_bar(Image.fromarray(r.render()), title, size=label_px(3 * 620)))
    r.close()
    tile(out, 3).save(MEDIA / "compare.png")


def spheres_figure():
    m = mujoco.MjModel.from_xml_string(combined_xml())
    d = mujoco.MjData(m)
    set_q(m, d, 0, 0.55, 0.55)
    m.vis.global_.fovy = 10
    r = renderer(m, 620, 820)
    shots = []
    for groups, az, title in [({2}, 225, "CAD meshes"), ({3, 4}, 225, "Spheres and tip pads"),
                              ({2, 3, 4}, 225, "Both"), ({3, 4}, 45, "Spheres and pads, dorsal side")]:
        c = cam([-0.006, 0, 0.098], 1.02, az, -15)
        r.update_scene(d, c, opt_groups(groups))
        shots.append((r.render().copy(), title))
    r.close()
    rows = np.flatnonzero(np.any([(im < 245).any(axis=(1, 2)) for im, _ in shots], axis=0))  # rows with content
    r0, r1 = max(0, rows[0] - 16), rows[-1] + 17
    out = [label_bar(Image.fromarray(im[r0:r1]), title, size=label_px(2 * 620)) for im, title in shots]
    tile(out, 2).save(MEDIA / "spheres.png")


def sweep_film(fps=25):
    import imageio.v2 as imageio
    m = mujoco.MjModel.from_xml_string(combined_xml())
    d = mujoco.MjData(m)
    m.vis.global_.fovy = 10
    W, H = 480, 640
    r = renderer(m, W, H)
    lo = {j: B.ROM[j][0] for j in B.ROM}
    hi = {j: B.ROM[j][1] for j in B.ROM}

    def seg(a, b, n):
        s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        return [tuple(np.array(a) + (np.array(b) - np.array(a)) * x) for x in s]

    z = (0, 0, 0)
    curl = (0, hi["mcp"], hi["pip"])
    ext = (0, lo["mcp"], lo["pip"])
    path = (seg(z, curl, 60) + seg(curl, ext, 70) + seg(ext, z, 30) + seg(z, (lo["yaw"], 0, 0), 40)
            + seg((lo["yaw"], 0, 0), (hi["yaw"], 0.55, 0.55), 70) + seg((hi["yaw"], 0.55, 0.55), z, 40))
    views = [({2}, "CAD meshes"), ({3, 4}, "Collision spheres and pads"), ({5}, "real_v1 capsules")]
    out_mp4 = MEDIA / "sweep.mp4"
    w = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=8,
                           ffmpeg_params=["-crf", "24", "-preset", "medium"], ffmpeg_log_level="error")
    poster_k = int(0.2 * len(path))
    fs = label_px(3 * W)
    f = font(fs)
    for k, q in enumerate(path):
        set_q(m, d, *q)
        panels = []
        for groups, title in views:
            c = cam([-0.006, 0, 0.095], 1.10, 235, -12)
            r.update_scene(d, c, opt_groups(groups))
            panels.append(np.asarray(r.render()))
        frame = Image.fromarray(np.concatenate(panels, axis=1))
        frame = label_bar(frame, "", size=fs, h=40)  # 640 + 40 rows, a multiple of 8: the poster has the film's size
        dr = ImageDraw.Draw(frame)
        for i, (_, title) in enumerate(views):
            dr.text((i * W + 10, (40 - fs) // 2), title, fill=(0, 0, 0), font=f)
        dr.text((10, frame.height - fs - 8), f"Yaw {np.degrees(q[0]):+4.0f}\u00b0   MCP {np.degrees(q[1]):+4.0f}\u00b0   PIP {np.degrees(q[2]):+4.0f}\u00b0",
                fill=(0, 0, 0), font=f)
        w.append_data(np.asarray(frame))
        if k == poster_k:
            frame.save(MEDIA / "sweep.jpg", quality=88)
    w.close()
    r.close()


def hand_figure():
    m = mujoco.MjModel.from_xml_path(str(MODEL / "hand.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    r = renderer(m, 760, 560)
    out = []
    for az, el, groups, title in [(150, -28, {0, 2}, "CAD meshes"), (150, -28, {0, 3, 4}, "Spheres and pads")]:
        c = cam([0, 0, 0.065], 0.40, az, el)
        r.update_scene(d, c, opt_groups(groups))
        out.append(label_bar(Image.fromarray(r.render()), title, size=label_px(2 * 760)))
    r.close()
    tile(out, 2).save(MEDIA / "hand.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="parts,assembly,compare,spheres,sweep,hand")
    args = ap.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)
    jobs = {"parts": parts_figure, "assembly": assembly_figure, "compare": compare_figure, "spheres": spheres_figure,
            "sweep": sweep_film, "hand": hand_figure}
    for k in args.only.split(","):
        jobs[k]()
        print("wrote", k)


if __name__ == "__main__":
    main()
