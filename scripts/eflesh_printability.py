#!/usr/bin/env python3
"""Printability of eFlesh lattices and of slicer gyroid on the owner's PrusaSlicer FLEX profile (2026-10-13).

Owner, 2026-10-13: "these micro-variations in generated structures never actually realized themselves in the real print.
the spring coefficient variable does something, but is very hard to actually map to a real outcome. for a small finger,
tight spacing is important to get a lot of layers of the microcstructure, but i found that delamination was a big issue
with the tighter spacing."

For each G-code file (scripts/gcode_audit.py) every layer's beads are drawn at 0.025 mm per pixel with their slicer
widths. Island: a connected region of one layer's beads. Support of a region: the share of its area that lies on the
beads of the layer below; a bead that lies on nothing bonds only at its edges. Rows go to
docs/experiments/20261013-eflesh_printability/audit.jsonl; `film` writes the layer-by-layer film of two files side by side.

    ~/miniconda3/envs/femtip/bin/python scripts/eflesh_printability.py audit
    ~/miniconda3/envs/femtip/bin/python scripts/eflesh_printability.py film
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gcode_audit as GA  # noqa: E402

SL = ROOT / "logs/20261013-eflesh_slice"
CL = ROOT / "logs/20261013-eflesh_cells"
OUT = ROOT / "docs/experiments/20261013-eflesh_printability"
RES = 0.025                                     # mm per pixel
FILES = [  # name, G-code, what it is, layers to skip at the bottom and top (solid layers of the lattice part)
    ("v17_project", SL / "v17.gcode", "eFlesh v17 project as printed: E 0.0035, cell 8 mm, six pouches, solid back", (0, 0)),
    ("v15_E0.002", SL / "v15_E0.002_c8.gcode", "eFlesh lattice alone, E 0.002, cell 8 mm", (3, 5)),
    ("v17_E0.0035", SL / "v17_E0.0035_c8.gcode", "eFlesh lattice alone with pouches, E 0.0035, cell 8 mm", (3, 5)),
    ("v14_E0.01", SL / "v14_E0.01_c8.gcode", "eFlesh lattice alone, E 0.01, cell 8 mm", (3, 5)),
    ("cube24_c8", CL / "cube24_c8.gcode", "eFlesh lattice in a 24 mm cube, E 0.0035, cell 8 mm", (3, 5)),
    ("cube24_c6", CL / "cube24_c6.gcode", "eFlesh lattice in a 24 mm cube, E 0.0035, cell 6 mm", (3, 5)),
    ("cube24_c4", CL / "cube24_c4.gcode", "eFlesh lattice in a 24 mm cube, E 0.0035, cell 4 mm", (3, 5)),
    ("cube24_c6_n025", CL / "cube24_c6_n025.gcode", "Same cube, cell 6 mm, 0.25 mm nozzle, 0.1 mm layers", (5, 8)),
    ("cube24_c4_n025", CL / "cube24_c4_n025.gcode", "Same cube, cell 4 mm, 0.25 mm nozzle, 0.1 mm layers", (5, 8)),
    ("tip_gyroid20", SL / "tip_gyroid20_d5.gcode", "Whole SR2 tip, gyroid 20 %, magnet pocket at 5 mm", (0, 0)),
    ("tip_gyroid10", SL / "tip_gyroid10_d5.gcode", "Whole SR2 tip, gyroid 10 %, magnet pocket at 5 mm", (0, 0)),
]


def raster(seg, lo, shape):
    img = Image.new("L", (int(shape[0]), int(shape[1])), 0)
    dr = ImageDraw.Draw(img)
    for x0, y0, x1, y1, w, _, _ in seg:
        a = ((x0 - lo[0]) / RES, (y0 - lo[1]) / RES)
        b = ((x1 - lo[0]) / RES, (y1 - lo[1]) / RES)
        r = 0.5 * w / RES
        dr.line([a, b], fill=255, width=max(1, int(round(w / RES))))
        dr.ellipse((a[0] - r, a[1] - r, a[0] + r, a[1] + r), fill=255)
        dr.ellipse((b[0] - r, b[1] - r, b[0] + r, b[1] + r), fill=255)
    return np.asarray(img) > 0


def bounds(layers):
    s = np.vstack([L["seg"] for L in layers])
    lo = np.array([min(s[:, 0].min(), s[:, 2].min()), min(s[:, 1].min(), s[:, 3].min())]) - 1.0
    hi = np.array([max(s[:, 0].max(), s[:, 2].max()), max(s[:, 1].max(), s[:, 3].max())]) + 1.0
    return lo, np.ceil((hi - lo) / RES).astype(int)


def audit(name, path, what, skip):
    L, T = GA.parse(path, 0)
    S = GA.summary(L, T)
    lo, shape = bounds(L)
    prev = None
    per = []
    isl = []
    for i, Lr in enumerate(L):
        A = raster(Lr["seg"], lo, shape)
        if prev is not None and skip[0] <= i < len(L) - skip[1]:
            lab, n = ndimage.label(A)
            area = ndimage.sum(A, lab, np.arange(1, n + 1)) * RES ** 2
            sup = ndimage.sum(A & prev, lab, np.arange(1, n + 1)) * RES ** 2
            frac = np.where(area > 0, sup / np.maximum(area, 1e-12), 0)
            per.append(dict(z=Lr["z"], islands=int(n), area_mm2=float(area.sum()), supported=float(sup.sum() / max(area.sum(), 1e-12)),
                            paths=S[i]["paths"], retracts=S[i]["retracts"]))
            isl += [(float(a), float(f)) for a, f in zip(area, frac)]
        prev = A
    isl = np.array(isl)
    a, f = isl[:, 0], isl[:, 1]
    row = dict(name=name, what=what, gcode=str(Path(path).relative_to(ROOT)), layers=len(per),
               islands_median=float(np.median([p["islands"] for p in per])), islands_max=int(max(p["islands"] for p in per)),
               paths_median=float(np.median([p["paths"] for p in per])), retracts_per_layer=float(np.mean([p["retracts"] for p in per])),
               island_area_median_mm2=float(np.median(a)),
               supported_area=float(sum(p["supported"] * p["area_mm2"] for p in per) / sum(p["area_mm2"] for p in per)),
               islands_under_half_supported=float(np.mean(f < 0.5)), island_support_median=float(np.median(f)),
               area_in_islands_under_half=float(a[f < 0.5].sum() / a.sum()),
               time=next((l.split("=")[1].strip() for l in open(path) if "estimated printing time (normal mode)" in l), None),
               per_layer=per)
    return row


def run_audit(only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    old = [json.loads(l) for l in open(OUT / "audit.jsonl")] if (only and (OUT / "audit.jsonl").exists()) else []
    keep = [r for r in old if r["name"] not in only] if only else []
    with open(OUT / "audit.jsonl", "w") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
        for name, path, what, skip in FILES:
            if only and name not in only:
                continue
            r = audit(name, path, what, skip)
            fh.write(json.dumps(r) + "\n"); fh.flush(); os.fsync(fh.fileno())
            rows.append(r)
            print(f"{name:14s} layers {r['layers']:3d} islands med {r['islands_median']:6.0f} max {r['islands_max']:4d} "
                  f"paths {r['paths_median']:5.0f} retr/layer {r['retracts_per_layer']:5.1f} island area {r['island_area_median_mm2']:.3f} mm2 "
                  f"supported {r['supported_area']:.3f} islands<50% {r['islands_under_half_supported']:.3f} "
                  f"(area {r['area_in_islands_under_half']:.3f}) time {r['time']}")
    return rows


FONT = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
COL_ON, COL_OFF, COL_BELOW = (42, 120, 214), (200, 20, 40), (205, 205, 205)


def frames(name_a, name_b, out_dir, labels, px_per_mm=14):
    """One PNG per print height: the layer below in grey, the layer's beads blue where they lie on it and red where they
    lie on nothing; the two prints side by side at one scale, with titles, the height, a legend and a 5 mm bar."""
    from PIL import ImageFont
    font, small = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 18)
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = []
    for nm in (name_a, name_b):
        path = dict((n, p) for n, p, _, _ in FILES)[nm]
        L, _ = GA.parse(path, 0)
        lo, shape = bounds(L)
        sets.append((L, lo, shape))
    zs = sorted(set(round(L_["z"], 3) for L, _, _ in sets for L_ in L))
    scale = RES * px_per_mm
    sizes = [(int(sh[0] * scale), int(sh[1] * scale)) for _, _, sh in sets]
    top, bot, gapw = 84, 64, 40                 # titles and legend above the panels; the player's controls cover the bottom
    Wt = sizes[0][0] + gapw + sizes[1][0] + 40
    Ht = top + max(h for _, h in sizes) + bot
    Wt += Wt % 2; Ht += Ht % 2
    for k, z in enumerate(zs):
        img = Image.new("RGB", (Wt, Ht), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        x = 20
        for (L, lo, shape), (W, H), lab in zip(sets, sizes, labels):
            idx = [i for i, L_ in enumerate(L) if abs(L_["z"] - z) < 1e-6]
            pan = np.full((H, W, 3), 255, np.uint8)
            if idx:
                i = idx[0]
                A = raster(L[i]["seg"], lo, shape)
                P = raster(L[i - 1]["seg"], lo, shape) if i > 0 else np.zeros_like(A)
                small_ = lambda M: np.asarray(Image.fromarray(M.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)) > 127
                aa, pp = small_(A), small_(P)
                pan[pp] = COL_BELOW
                pan[aa & pp] = COL_ON
                pan[aa & ~pp] = COL_OFF
            img.paste(Image.fromarray(np.flipud(pan)), (x, top))
            dr.text((x, 10), lab, fill=(0, 0, 0), font=font)
            x += W + gapw
        y = 46
        dr.text((20, y), f"Print height {z:.2f} mm", fill=(0, 0, 0), font=small)
        lx = 230
        for col, txt in ((COL_ON, "On the layer below"), (COL_OFF, "Over nothing"), (COL_BELOW, "Layer below")):
            dr.rectangle((lx, y + 3, lx + 16, y + 17), fill=col)
            dr.text((lx + 22, y), txt, fill=(0, 0, 0), font=small)
            lx += 30 + int(dr.textlength(txt, font=small)) + 14
        bx = Wt - 20 - int(5 * px_per_mm)
        dr.line((bx, y + 10, bx + int(5 * px_per_mm), y + 10), fill=(0, 0, 0), width=3)
        dr.text((bx, y + 16), "5 mm", fill=(0, 0, 0), font=small)
        img.save(out_dir / f"f{k:04d}.png")
    return len(zs)


def film(a, b, name, labels, fps=8):
    """Frames of two prints side by side, encoded to docs/experiments/20261013-eflesh_printability/media/<name>.mp4."""
    d = ROOT / "logs/20261013-eflesh_slice" / f"frames_{name}"
    for f in d.glob("f*.png") if d.exists() else []:
        f.unlink()
    n = frames(a, b, d, labels)
    media = OUT / "media"
    media.mkdir(parents=True, exist_ok=True)
    out = media / f"20261013-{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(d / "f%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", str(out)], check=True)
    Image.open(d / f"f{n // 2:04d}.png").save(media / f"20261013-{name}_poster.png")
    print(out, n, "frames", round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["audit", "film"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.cmd == "audit":
        run_audit(a.only)
    else:
        film("cube24_c8", "cube24_c4", "eflesh_cells_8_4mm", ("eFlesh lattice, 8 mm cells", "eFlesh lattice, 4 mm cells"))
        film("v17_project", "tip_gyroid20", "eflesh_v17_vs_gyroid_tip", ("eFlesh v17 as printed", "SR2 tip, gyroid 20 %"))
