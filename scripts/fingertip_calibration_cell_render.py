#!/usr/bin/env python3
"""Figure 1 of the fingertip calibration page: the UR5 cell rendered in MuJoCo, with the pad at two touch-offs on the low
sphere (owner, 2026-10-27: the coordinator's in-MuJoCo render of the calibration scene, "you can maybe use that in the
corresponding docpage"). The scene is scripts/ur5_cell_render.py's, the cell as it stood on 2026-10-27. The robot stands
at the contact onsets of the touch-offs pad-face-00 and pad-flank-05 of v2 ID 1's sphere session: the flange poses in
lab/20261026-192158-v2id1/touches.json, the joints by the controller's IK from the session's q_home, the mount placed by
the session's T_EM. Writes docs/experiments/20261026-fingertip_calibration/media/20261027-cell.png: (a) the cell at the
face touch-off with the fixtures named, (b) and (c) the pad on the sphere at the two touch-offs.

    MUJOCO_GL=egl NUMPY_MADVISE_HUGEPAGE=0 ~/.claude/bin/resguard.sh run --mem 4G --cpu 200 -- \\
        .venv/bin/python scripts/fingertip_calibration_cell_render.py [--preview DIR]

--preview DIR renders the close-up cameras' candidates (sides and azimuths) at 550 x 410 into DIR and stops.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ.setdefault("UR5_MOUNT", "v7")            # the mount on the robot since R24 (cb2t, 2026-10-25; lab NOTES)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_cell_render as R  # noqa: E402

CELL = R.CELL
SESSION = R.LAB / "20261026-192158-v2id1"
OUT = ROOT / "docs/experiments/20261026-fingertip_calibration/media/20261027-cell.png"
TOUCH = dict(face="pad-face-00", flank="pad-flank-05")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
W = 1100                    # the page embeds figures 1100 px wide (ur5_tip_calibration_page.fig)
SS = 2                      # render at SS times the panel size, then downsample
OVERVIEW = (W, 640)         # panel (a)
CLOSE = ((W - 4) // 2, 410)  # panels (b) and (c), 4 px apart
BAR = 34                    # title bar height (px)
# Close-up cameras: which side of the finger (+1 or -1 along tool axis x up), azimuth toward the tip (deg), elevation
# (deg), distance (mm), vertical field of view (deg), aim above the sphere's centre (mm)
CLOSE_CAM = dict(face=dict(side=1, az=30.0, el=16.0, dist=210.0, fovy=28.0, lift=14.0),
                 flank=dict(side=1, az=30.0, el=16.0, dist=210.0, fovy=28.0, lift=14.0))
# Panel (a): ur5_cell_render's overview camera moved 15 % toward its target and raised, so that the fixtures are larger
OVERVIEW_CAM = ((336.0, -1626.0, 1032.0), (-215.0, -420.0, 170.0), 34.0)
# Labels in panel (a): fixture -> (text, offset of the label from the fixture's top in px of the panel)
LABELS = dict(sphere=("low sphere", (-150, -95)), rod=("rod", (60, -110)), tall=("tall indenter", (95, -40)),
              holder=("fingertip holder", (-210, 60)))


def session():
    reg = json.loads((SESSION / "registration.json").read_text())
    touches = {t["id"]: t for t in json.loads((SESSION / "touches.json").read_text())}
    q_home = np.array(reg["q_home"], float)
    poses = {}
    for k, tid in TOUCH.items():
        T = np.array(touches[tid]["T_BE"], float)
        q = CELL.SIM.ik(T, q_home)
        if q is None:
            raise RuntimeError(f"no IK for {tid}")
        err = np.abs(CELL.SIM.fk(q)[:3, 3] - T[:3, 3]).max()
        poses[k] = dict(q=q, T_BE=T, id=tid, target=touches[tid]["target"], fk_err_mm=float(err))
    return np.array(reg["T_EM"], float), poses


def close_camera(fx, T_BE, T_EM, side, az, el, dist, fovy, lift):
    """Camera on the low sphere: horizontal direction across the finger (side), turned az toward the tip, raised el."""
    c = (fx["sphere"] @ np.array([0.0, 0.0, 41.0, 1.0]))[:3]      # the sphere's centre: (0, 0, 41) mm in I
    a = (T_BE @ T_EM)[:3, 2]                                        # the tool axis (z_M) in B
    a_h = np.array([a[0], a[1], 0.0])
    a_h /= np.linalg.norm(a_h)
    across = np.cross(a_h, [0.0, 0.0, 1.0])
    d = side * across * math.cos(math.radians(az)) + a_h * math.sin(math.radians(az))
    e = math.radians(el)
    target = c + np.array([0.0, 0.0, lift])
    pos = target + dist * (math.cos(e) * d + math.sin(e) * np.array([0.0, 0.0, 1.0]))
    return tuple(pos), tuple(target), fovy


def project(cam, X, w, h):
    """Pixel of a point X (mm, B) in a look_at camera's image of w x h."""
    pos, target, fovy = np.asarray(cam[0], float), np.asarray(cam[1], float), cam[2]
    f = (target - pos) / np.linalg.norm(target - pos)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    v = np.asarray(X, float) - pos
    fp = (h / 2) / math.tan(math.radians(fovy) / 2)
    return w / 2 + fp * (v @ right) / (v @ f), h / 2 - fp * (v @ up) / (v @ f)


def fixture_tops(fx):
    """The top centre (mm, B) of each fixture's mesh at its registered pose."""
    import trimesh
    out = {}
    for key, path in (("sphere", CELL.INDENTER_STL["sphere"]), ("rod", CELL.INDENTER_STL["rod"]), ("tall", R.TALL_STL),
                      ("holder", R.HOLDER_STL)):
        V = R.transformed(trimesh.load(path, force="mesh"), fx[key]).vertices
        lo, hi = V.min(0), V.max(0)
        out[key] = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])
    return out


def render(m, d, q, cam_name, w, h):
    import mujoco
    R.set_pose(m, d, q)
    ren = mujoco.Renderer(m, h, w)
    try:
        ren.update_scene(d, camera=cam_name)
        return ren.render().copy()
    finally:
        ren.close()


def compose(a, b, c, tops, cam_a):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(FONT, 20)
    small = ImageFont.truetype(FONT, 19)
    wa, ha = OVERVIEW
    wc, hc = CLOSE
    H = BAR + ha + BAR + hc
    fig = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(fig)
    pa = Image.fromarray(a).resize((wa, ha), Image.LANCZOS)
    fig.paste(pa, (0, BAR))
    for k, im in enumerate((b, c)):
        fig.paste(Image.fromarray(im).resize((wc, hc), Image.LANCZOS), (k * (wc + 4), 2 * BAR + ha))
    titles = [((8, 6), "(a) The cell on 2026-10-27, the pad at a face touch-off on the low sphere"),
              ((8, BAR + ha + 6), "(b) Face touch-off, pad-face-00"),
              ((wc + 4 + 8, BAR + ha + 6), "(c) Flank touch-off, pad-flank-05")]
    for xy, t in titles:
        dr.text(xy, t, font=font, fill="black")
    dr.line([(0, BAR - 1), (W, BAR - 1)], fill="black", width=1)
    dr.line([(0, 2 * BAR + ha - 1), (W, 2 * BAR + ha - 1)], fill="black", width=1)
    dr.line([(0, BAR + ha), (W, BAR + ha)], fill="black", width=1)
    dr.rectangle([wc, 2 * BAR + ha, wc + 3, H], fill="white")
    for key, (text, (dx, dy)) in LABELS.items():
        u, v = project(cam_a, tops[key], wa, ha)
        v += BAR
        x, y = u + dx, v + dy
        tw = dr.textlength(text, font=small)
        box = [x - 5, y - 3, x + tw + 5, y + 24]
        # leader from the label box's nearest edge to the fixture's top
        ex = min(max(u, box[0]), box[2])
        ey = box[3] if v > box[3] else (box[1] if v < box[1] else (box[1] + box[3]) / 2)
        dr.line([(ex, ey), (u, v)], fill="black", width=2)
        dr.rectangle(box, fill="white", outline="black", width=1)
        dr.text((x, y), text, font=small, fill="black")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", default=None, help="render the close-up cameras' candidates into this folder and stop")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    import mujoco
    from PIL import Image
    t0 = time.time()
    fx = R.fixtures()
    T_EM, poses = session()
    fx["T_EM"] = T_EM                                  # the tool as registered in the sphere session
    for k, p in poses.items():
        print(f"{k}: {p['id']} ({p['target']['region']}), joints {np.round(np.degrees(p['q']), 2).tolist()} deg, "
              f"FK error {p['fk_err_mm']:.4f} mm")
    cams = {"overview": OVERVIEW_CAM}
    if a.preview:
        for k in TOUCH:
            for side in (1, -1):
                for az in (-30.0, 0.0, 30.0):
                    cc = dict(CLOSE_CAM[k], side=side, az=az)
                    cams[f"{k}_s{side:+d}_a{az:+.0f}"] = close_camera(fx, poses[k]["T_BE"], T_EM, **cc)
    else:
        for k in TOUCH:
            cams[k] = close_camera(fx, poses[k]["T_BE"], T_EM, **CLOSE_CAM[k])
    R.CAMERAS.clear()
    R.CAMERAS.update(cams)
    with tempfile.TemporaryDirectory(prefix="calib_cell_render_") as tmp:
        work = Path(tmp)
        wmax, hmax = SS * OVERVIEW[0], SS * OVERVIEW[1]
        xml, log, _ = R.build(work, fx, wmax, hmax)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        print(f"model: {m.nmesh} meshes, {m.ngeom} geoms ({time.time() - t0:.1f} s)")
        if a.preview:
            out = Path(a.preview)
            out.mkdir(parents=True, exist_ok=True)
            for name in cams:
                if name == "overview":
                    continue
                k = name.split("_")[0]
                Image.fromarray(render(m, d, poses[k]["q"], name, *CLOSE)).save(out / f"{name}.png")
                print("wrote", out / f"{name}.png")
            img = render(m, d, poses["face"]["q"], "overview", *OVERVIEW)
            Image.fromarray(img).save(out / "overview.png")
            print("wrote", out / "overview.png", f"({time.time() - t0:.1f} s)")
            return
        ia = render(m, d, poses["face"]["q"], "overview", SS * OVERVIEW[0], SS * OVERVIEW[1])
        ib = render(m, d, poses["face"]["q"], "face", SS * CLOSE[0], SS * CLOSE[1])
        ic = render(m, d, poses["flank"]["q"], "flank", SS * CLOSE[0], SS * CLOSE[1])
    fig = compose(ia, ib, ic, fixture_tops(fx), cams["overview"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.save(a.out)
    print(f"wrote {Path(a.out).relative_to(ROOT)} ({fig.width} x {fig.height}); {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
