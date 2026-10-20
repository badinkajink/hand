#!/usr/bin/env python3
"""Figures comparing DIP link concept b v2 (printed 2026-10-19) with v3 (2026-10-20): the neck under the pad from the
palmar side, the cover with its side walls, and the cable's leg toward the back from above; and a turntable film of v3 with
its cover and pad. pyvista off screen (femtip), into docs/experiments/20261018-dip_link_board_mount/v3/b/media/.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python scripts/dip_b_v3_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DIP = REPO / "docs/experiments/20261018-dip_link_board_mount"
MEDIA = DIP / "v3/b/media"
COL = dict(link="#d4d4cc", cover="#4a78c8", pad="#e2c48e", cable="#2f48d8", plug="#f2f2ea")


def load(rel):
    import trimesh
    m = trimesh.load(DIP / rel, force="mesh")
    return m


def pvm(m):
    import pyvista as pv
    f = np.hstack([np.full((len(m.faces), 1), 3), m.faces]).ravel()
    return pv.PolyData(np.asarray(m.vertices, float), f)


def shot(meshes, dst, cam, size=(900, 900), zoom=1.3, labels=None):
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
    p.set_background("white")
    for m, c, op in meshes:
        p.add_mesh(pvm(m), color=c, opacity=op, smooth_shading=True, split_sharp_edges=True, specular=0.15)
    pos, focal, up = cam
    p.camera.position, p.camera.focal_point, p.camera.up = pos, focal, up
    p.camera.zoom(zoom)
    p.screenshot(str(dst))
    p.close()


def panels(files, titles, dst, suptitle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    ims = [Image.open(f) for f in files]
    fig, axs = plt.subplots(1, len(ims), figsize=(6.0 * len(ims), 6.6), dpi=110)
    for ax, im, t in zip(np.atleast_1d(axs), ims, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=17, family="serif")
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=19, family="serif")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(dst)
    plt.close(fig)
    for f in files:
        Path(f).unlink()


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    tmp = MEDIA / "_tmp"
    tmp.mkdir(exist_ok=True)
    v = {k: dict(link=load(f"{k}/b/cad/{pre}_dip_link.stl"), cover=load(f"{k}/b/cad/{pre}_cover.stl"),
                 cable=load(f"{k}/b/cad/{pre}_cable.stl"))
         for k, pre in (("v2", "20261018-dip_link_v2b"), ("v3", "20261020-dip_link_v3b"))}
    pad = load("v2/b/cad/20261018-dip_link_v2b_pad_ell11_5.stl")
    c = np.array([-2.0, 0.0, -78.0])
    # 1. palmar side, finger up: the neck under the pad
    f1 = []
    for k in ("v2", "v3"):
        f = tmp / f"palmar_{k}.png"
        shot([(v[k]["link"], COL["link"], 1.0), (v[k]["cover"], COL["cover"], 1.0), (pad, COL["pad"], 1.0)], f,
             (tuple(c + [-95.0, -28.0, 6.0]), tuple(c), (0, 0, -1)), zoom=1.45)
        f1.append(f)
    panels(f1, ["v2 as printed on 2026-10-19", "v3: column under the pad, walls on the cover"],
           MEDIA / "20261020-dip_link_b_v2_v3_palmar.png", "DIP link b from the palmar side, finger up")
    # 2. the covers, from the palmar side and above their distal end
    f2 = []
    cc = np.array([2.0, 0.0, -88.0])
    for k in ("v2", "v3"):
        f = tmp / f"cover_{k}.png"
        shot([(v[k]["cover"], COL["cover"], 1.0)], f, (tuple(cc + [-48.0, -30.0, -30.0]), tuple(cc), (0, 0, -1)), zoom=1.35)
        f2.append(f)
    f = tmp / "cover_v3_pad.png"
    shot([(v["v3"]["cover"], COL["cover"], 1.0), (pad, COL["pad"], 0.55)], f, (tuple(cc + [-48.0, -30.0, -30.0]), tuple(cc),
                                                                          (0, 0, -1)), zoom=1.35)
    f2.append(f)
    panels(f2, ["v2 cover", "v3 cover: side walls, screw counterbores", "v3 cover with the pad"],
           MEDIA / "20261020-dip_link_b_v2_v3_cover.png", "The cover with the pad's rails")
    # 3. from above (the bridge's top), dorsal to the right: the cable's leg toward the back
    f3 = []
    ct = np.array([4.0, 2.0, -64.0])
    for k in ("v2", "v3"):
        f = tmp / f"top_{k}.png"
        shot([(v[k]["link"], COL["link"], 1.0), (v[k]["cable"], COL["cable"], 1.0)], f,
             (tuple(ct + [6.0, -10.0, 60.0]), tuple(ct), (1, 0, 0)), zoom=2.0)
        f3.append(f)
    panels(f3, ["v2: the leg toward the back roofed for x > 6 mm", "v3: the leg open to the top out through the back"],
           MEDIA / "20261020-dip_link_b_v2_v3_cable_leg.png", "The Qwiic cable's first turn and leg, from the PIP side")
    tmp.rmdir()
    # 4. turntable of v3 with its cover, pad and cable (mp4, H.264 by the system's ffmpeg)
    import subprocess
    import pyvista as pv
    from PIL import Image
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(720, 720), lighting="three lights")
    p.set_background("white")
    for m, col_ in ((v["v3"]["link"], COL["link"]), (v["v3"]["cover"], COL["cover"]), (pad, COL["pad"]), (v["v3"]["cable"], COL["cable"])):
        p.add_mesh(pvm(m), color=col_, smooth_shading=True, split_sharp_edges=True, specular=0.15)
    ctr = np.array([0.0, -3.0, -72.0])
    fr = MEDIA / "_frames"
    fr.mkdir(exist_ok=True)
    for k, a in enumerate(np.linspace(0, 2 * np.pi, 120, endpoint=False)):
        p.camera.position = tuple(ctr + 105.0 * np.array([np.cos(a), np.sin(a), 0.35]))
        p.camera.focal_point = tuple(ctr)
        p.camera.up = (0, 0, -1)
        Image.fromarray(p.screenshot(return_img=True)).save(fr / f"f{k:04d}.png")
    p.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(fr / "f%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", str(MEDIA / "20261020-dip_link_v3b_turntable.mp4")], check=True)
    for f in fr.glob("*.png"):
        f.unlink()
    fr.rmdir()
    print("figures ->", MEDIA.relative_to(REPO))


if __name__ == "__main__":
    sys.exit(main())
