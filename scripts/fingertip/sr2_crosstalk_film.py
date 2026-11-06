#!/usr/bin/env python3
"""Film of the SR2 hand through the crosstalk sweeps (docs/experiments/20261104-sr2_hand_crosstalk): the cad_v3 full
model at each pose's measured joints and its layout's mounts, and beside it each board's change in force equivalent,
uncompensated and after the own-angle GP, with the poses where the contact viewer would show a contact with nothing
touching. Data: scripts/fingertip/sr2_crosstalk_hand.py's <STEM>.json and <STEM>_rows.npz.

The mounts are placed as the driver commands them (handgeo.ORIGIN plus the mount), which puts the cad_v3 fingers about
6 mm inside their CAD travel boxes (the cad_v3 page, section "Finger travel"); the film shows the motion, the analysis
uses cad_v2's kinematics.

    MUJOCO_GL=egl ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- .venv/bin/python scripts/fingertip/sr2_crosstalk_film.py [STEM] [--per-layout 16]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401
import cad_v3_film as CF  # noqa: E402

EXP = ROOT / "docs/experiments/20261104-sr2_hand_crosstalk"
sys.path.insert(0, str(EXP / "lab"))
from handgeo import ORIGIN  # noqa: E402

DATA, MEDIA = EXP / "data", EXP / "media"
F = ("thumb", "index", "middle")
J = ("yaw", "mcp", "pip")
COLOR = {"thumb": (176, 58, 46), "index": (31, 78, 156), "middle": (91, 140, 58)}
LABEL = {"home": "Home", "hom_roll_85_170": "HOM roll", "hom_press_70_150": "HOM press",
         "hom_generalist_70_110": "HOM generalist", "hom_twist_100_70": "HOM twist", "hom_cover_50_75": "HOM cover"}
W3, H, WP = 840, 720, 440                      # the 3D view, the height, the panel
NMAX = 6.0                                       # N at the panel's right edge (log axis from 0.03 N)


def xbar(v):
    lo, hi = np.log10(0.03), np.log10(NMAX)
    return int(30 + (WP - 60) * (np.clip(np.log10(max(v, 0.03)), lo, hi) - lo) / (hi - lo))


def panel(vals, title, font, small):
    """vals: {finger: (raw N, own N, shown)}"""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (WP, H), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.line([0, 0, 0, H], fill=(0, 0, 0), width=1)
    dr.text((20, 18), title, fill=(0, 0, 0), font=font)
    dr.text((20, 52), "Change of each board, force equivalent (N)", fill=(0, 0, 0), font=small)
    y0 = 110
    for k, f in enumerate(F):
        y = y0 + k * 170
        dr.text((20, y), f"{f.capitalize()} board", fill=COLOR[f], font=font)
        if f not in vals:
            dr.text((20, y + 40), "no reading", fill=(90, 90, 90), font=small)
            continue
        raw, own, shown = vals[f]
        dr.rectangle([30, y + 42, xbar(raw), y + 70], fill=(150, 150, 150))
        dr.text((xbar(raw) + 8, y + 44), f"{raw:.2f}", fill=(0, 0, 0), font=small)
        dr.rectangle([30, y + 80, xbar(own), y + 108], fill=(0, 0, 0))
        dr.text((xbar(own) + 8, y + 82), f"{own:.2f}", fill=(0, 0, 0), font=small)
        if shown:
            dr.text((30, y + 114), "false contact shown, uncompensated", fill=(176, 58, 46), font=small)
    xt = xbar(0.25)
    dr.line([xt, y0 + 30, xt, y0 + 3 * 170 - 30], fill=(90, 90, 90), width=1)
    dr.text((xt - 30, y0 + 3 * 170 - 26), "0.25 N", fill=(90, 90, 90), font=small)
    yl = H - 70
    dr.rectangle([30, yl, 54, yl + 18], fill=(150, 150, 150))
    dr.text((62, yl - 2), "uncompensated", fill=(0, 0, 0), font=small)
    dr.rectangle([30, yl + 30, 54, yl + 48], fill=(0, 0, 0))
    dr.text((62, yl + 28), "after the own-angle GP", fill=(0, 0, 0), font=small)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", nargs="?", default="20261105-sr2_crosstalk_hand_20261104_20261105")
    ap.add_argument("--per-layout", type=int, default=16)
    ap.add_argument("--out", default=str(MEDIA / "20261105-film_cad_sweep.mp4"))
    a = ap.parse_args()
    import mujoco
    from PIL import Image, ImageFont
    R = json.loads((DATA / f"{a.stem}.json").read_text())
    Z = np.load(DATA / f"{a.stem}_rows.npz")
    upn = R["uT_per_N"]
    fam, lay, q_all, mounts = Z["family"].astype(str), Z["layout"].astype(str), Z["q"], Z["mounts"]
    per = {f: {int(i): n for n, i in enumerate(Z[f"{f}_rows"])} for f in F if f"{f}_rows" in Z}
    steps = [i for i in range(len(fam)) if fam[i].startswith("random") and any(i in per[f] for f in per)]
    chosen, count = [], {}
    for i in steps:
        if count.get(lay[i], 0) < a.per_layout:
            chosen.append(i)
            count[lay[i]] = count.get(lay[i], 0) + 1
    m = CF.load(CF.SCENES / "scene_screwdriver_medium_full.xml", W3, H, floor=False)
    d = mujoco.MjData(m)
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 3] = (0, 0, -5)
    adr = {n: m.jnt_qposadr[m.joint(n).id] for f in F for n in (f"{f}_x", f"{f}_y", f"{f}_bridge", *(f"{f}_{j}" for j in J))}
    font = ImageFont.truetype(CF.FONT, 30)
    small = ImageFont.truetype(CF.FONT, 22)
    r = mujoco.Renderer(m, H, W3, max_geom=12000)
    opt = CF.scene_option()
    out = Path(a.out)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W3 + WP}x{H}",
                            "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow",
                            "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def pose(q, mt):
        for k, f in enumerate(F):
            x, y = mt[2 * k] - ORIGIN[f][0], mt[2 * k + 1] - ORIGIN[f][1]
            d.qpos[adr[f"{f}_x"]], d.qpos[adr[f"{f}_y"]], d.qpos[adr[f"{f}_bridge"]] = x / 1000, y / 1000, y / 1000
            for jj, jn in enumerate(J):
                d.qpos[adr[f"{f}_{jn}"]] = np.radians(q[3 * k + jj])
        mujoco.mj_forward(m, d)

    pose(np.zeros(9), mounts[chosen[0]])
    look = np.mean([d.body(f"{f}_tip").xpos for f in F], 0) + [0, 0, 0.085]
    cam = CF.free_cam(look, 0.40, 140, 4)
    n, poster = 0, None
    prev = np.zeros(9)
    try:
        for c, i in enumerate(chosen):
            vals = {}
            for f in per:
                if i in per[f]:
                    rn = per[f][i]
                    raw = float(np.linalg.norm(Z[f"{f}_Y"][rn])) / upn[f]
                    own = float(np.linalg.norm(Z[f"{f}_Y"][rn] - Z[f"{f}_P_own_angles"][rn])) / upn[f]
                    sh = bool(np.nan_to_num(Z[f"{f}_shown_raw"][rn]) > 0) if f"{f}_shown_raw" in Z else False
                    vals[f] = (raw, own, sh)
            title = f"{LABEL.get(lay[i], lay[i])}, pose {count_pos(chosen, lay, c)}"
            p_img = panel(vals, title, font, small)
            new_layout = c == 0 or lay[chosen[c - 1]] != lay[i]
            start = np.zeros(9) if new_layout else prev
            for k in range(12):
                s = min(1.0, (k + 1) / 8)
                pose(start + (q_all[i] - start) * (0.5 - 0.5 * np.cos(np.pi * s)), mounts[i])
                r.update_scene(d, cam, scene_option=opt)
                frame = np.concatenate([r.render(), p_img], 1)
                enc.stdin.write(frame.tobytes())
                n += 1
                if poster is None and k == 11 and any(v[2] for v in vals.values()):
                    poster = frame.copy()
            prev = q_all[i]
    finally:
        r.close()
        enc.stdin.close()
        enc.wait()
    if poster is not None:
        Image.fromarray(poster).convert("RGB").save(out.with_suffix(".jpg"), quality=88)
    print(f"wrote {out}: {n} frames, {n / 25:.0f} s, {out.stat().st_size / 1e6:.1f} MB; poses {len(chosen)} "
          f"over {len(count)} layouts", flush=True)


def count_pos(chosen, lay, c):
    return sum(1 for k in range(c + 1) if lay[chosen[k]] == lay[chosen[c]])


if __name__ == "__main__":
    main()
