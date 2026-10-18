#!/usr/bin/env python3
"""Figures of the 85A fingertip print procedure (docs/handoff/20261018-print_procedure.md, step 2): each print geometry on
the Prusa XL bed and, for each pause, the part as printed up to that pause with its magnets seated.

The geometries are the STLs of the 2026-10-16 Siraya 85A projects (docs/experiments/20261016-sim2real_fingertip_readiness/
print/, one sheet per project). Every project places its STL at the bed centre (180, 180) without rotation, so the STL frame
is the bed frame: +X to the printer's right, +Z up. Two stages, because manifold3d and MuJoCo live in different
environments:

  cut      (femtip env) the part trimmed to z <= the top of the last layer before each pause, as STL in logs/
  walls    (femtip env) the least TPU around a pocket and between pockets of every tip geometry at seats of 2.7, 2.8
           and 2.9 mm (metric_magnet_prints.build at the 85A depths), against the builders' 1.8 mm rule:
           docs/experiments/20261018-fingertip_print_procedure/seat_walls.json
  render   (.venv) MuJoCo renders: the whole part on a bed of 10 mm squares, and each cut part with its magnets (north half
           red, south half grey) and a blue arrow toward the palmar side; the seat coupon's top view with its seat sizes

    ~/.claude/bin/resguard.sh run --mem 1G -- taskset -c 2,3 ~/miniconda3/envs/femtip/bin/python \\
        scripts/fingertip_print_procedure_figures.py cut
    MUJOCO_GL=egl ~/.claude/bin/resguard.sh run --mem 1500M -- taskset -c 2,3 .venv/bin/python \\
        scripts/fingertip_print_procedure_figures.py render
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRINT = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness/print"
OUT = ROOT / "docs/experiments/20261018-fingertip_print_procedure/media"
WORK = ROOT / "logs/20261018-fingertip_print_procedure/cut"
MAGNET_D, MAGNET_T = 3.0, 2.0                  # mm, the sheets' magnet
TPU_RGBA = "0.93 0.80 0.62 1"
W, H = 1600, 1000


def geometries():
    """{STL stem: a representative sheet}: one geometry per STL (the 30 % sheet with two walls where there is one)."""
    out = {}
    for f in sorted(PRINT.glob("20261016-siraya85a_*.json")):
        if f.name == "template_changes.json":
            continue
        r = json.loads(f.read_text())
        stem = Path(r["stl"]).stem
        if stem not in out or (r.get("infill_percent", 30) == 30 and r.get("pocket_walls") == 2):
            out[stem] = r
    return out


def cut_heights(r):
    """Just under the top of the last layer printed before each pause (mm): 0.04 mm below it, so that the cut does not
    coincide with the pockets' top faces, which lie on that layer's top; a coupon has none."""
    layer = r["recipe"]["layer_height_mm"]
    return [round(p["print_z"] - layer - 0.04, 3) for p in r.get("pauses", [])]


# ------------------------------------------------------------------------------------------------- cut (femtip)

def cmd_cut(a):
    import manifold3d as M
    import trimesh
    WORK.mkdir(parents=True, exist_ok=True)
    for stem, r in geometries().items():
        mesh = trimesh.load(ROOT / r["stl"], force="mesh")
        man = M.Manifold(M.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                                tri_verts=np.asarray(mesh.faces, np.uint32)))
        for z in cut_heights(r):
            part = man.trim_by_plane((0.0, 0.0, -1.0), -z)          # keeps z <= cut height
            m = part.to_mesh()
            t = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
            dst = WORK / f"{stem}_z{z:g}.stl"
            t.export(dst)
            print(f"{dst.name}: {len(t.faces)} faces, z {t.bounds[0][2]:.2f}-{t.bounds[1][2]:.2f} mm, "
                  f"watertight {t.is_watertight}", flush=True)


def cmd_walls(a):
    import time
    sys.path.insert(0, str(ROOT / "scripts"))
    import metric_magnet_prints as MP
    import sim2real_fingertip_prints as SP
    rows = []
    for tip, lay, walls in SP.PRODUCTS:
        if walls != 2:                                    # the one-wall block has the four-magnet geometry
            continue
        for seat in (2.7, 2.8, 2.9):
            _, _, pockets, gaps, _ = MP.build(tip, MP.layout(lay), seat, SP.DEPTHS[tip])
            rows.append(dict(tip=tip, layout=lay, seat_mm=seat, depth_mm=SP.DEPTHS[tip],
                             wall_min_mm=round(min(p["wall_min_mm"] for p in pockets), 3),
                             gap_min_mm=round(min(g["gap_mm"] for g in gaps), 3) if gaps else None,
                             walls_ok=bool(MP.walls_pass(pockets, gaps))))
            print(rows[-1], flush=True)
    dst = OUT.parent / "seat_walls.json"
    dst.write_text(json.dumps(dict(rule_mm=MP.WALL_MIN, rows=rows, script="scripts/fingertip_print_procedure_figures.py walls",
                                   when=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    print(dst.relative_to(ROOT), flush=True)


# ------------------------------------------------------------------------------------------------- render (.venv)

def magnets(r, upto_pause):
    """(centre mm, axis, palmar sign) of each magnet seated by the end of pause `upto_pause` (1-based)."""
    seated = [i for p in r["pauses"][:upto_pause] for i in p["magnets"]]
    out = []
    for i in seated:
        p = r["pockets"][i]
        if p["kind"] == "pocket":                                   # block: flat on the floor, north up
            x, y = p["centre_print_mm"]
            out.append(((x, y, p["floor_mm"] + MAGNET_T / 2), "z"))
        else:                                                       # upright slot: on its edge, north toward +x
            yp, zp = p["centre_print_mm"]
            x_lo, x_hi = p["slot_x_mm"]
            zc = zp + (MAGNET_D / 2 - p["seat_diameter_mm"] / 2)
            out.append((((x_lo + x_hi) / 2, yp, zc), "x"))
    return out


def cone_mesh(r=1.6, h=3.6, n=24):
    v = [(0.0, 0.0, h)] + [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), 0.0) for k in range(n)]
    v.append((0.0, 0.0, 0.0))
    f = [(0, 1 + k, 1 + (k + 1) % n) for k in range(n)] + [(n + 1, 1 + (k + 1) % n, 1 + k) for k in range(n)]
    return " ".join(f"{c * 1e-3:.6f}" for p in v for c in p), " ".join(str(i) for t in f for i in t)


def scene_xml(stl, r, mags, arrow):
    """MJCF of one part (mm scaled to m), its magnets and the palmar arrow (start mm, direction)."""
    cv, cf = cone_mesh()
    body = []
    for k, ((x, y, z), ax) in enumerate(mags):
        # two halves of 1 mm: north half on the palmar side (+z for the block, +x for the upright tips)
        for half, rgba in ((+1, "0.85 0.10 0.10 1"), (-1, "0.62 0.64 0.67 1")):
            off = half * MAGNET_T / 4
            pos = (x, y, z + off) if ax == "z" else (x + off, y, z)
            za = "0 0 1" if ax == "z" else "1 0 0"
            body.append(f'<geom type="cylinder" size="{MAGNET_D / 2e3:.6f} {MAGNET_T / 4e3:.6f}" '
                        f'pos="{pos[0] * 1e-3:.6f} {pos[1] * 1e-3:.6f} {pos[2] * 1e-3:.6f}" zaxis="{za}" rgba="{rgba}"/>')
    for (sx, sy, sz), d in ([arrow] if arrow else []):
        body += arrow_geoms(sx, sy, sz, d)
    return _xml(stl, cv, cf, body)


def arrow_geoms(sx, sy, sz, d):
    out = []
    L = 7.0
    ex, ey, ez = sx + d[0] * L, sy + d[1] * L, sz + d[2] * L
    out.append(f'<geom type="capsule" fromto="{sx * 1e-3:.6f} {sy * 1e-3:.6f} {sz * 1e-3:.6f} {ex * 1e-3:.6f} '
               f'{ey * 1e-3:.6f} {ez * 1e-3:.6f}" size="0.0006" rgba="0.10 0.30 0.85 1"/>')
    za = " ".join(str(c) for c in d)
    out.append(f'<geom type="mesh" mesh="cone" pos="{ex * 1e-3:.6f} {ey * 1e-3:.6f} {ez * 1e-3:.6f}" zaxis="{za}" '
               f'rgba="0.10 0.30 0.85 1"/>')
    return out


def _xml(stl, cv, cf, body):
    return f"""<mujoco>
<visual><global offwidth="{W}" offheight="{H}" fovy="30"/><quality shadowsize="4096" offsamples="8"/>
<headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55" specular="0.05 0.05 0.05"/></visual>
<asset>
<texture name="bed" type="2d" builtin="checker" rgb1="0.24 0.24 0.26" rgb2="0.30 0.30 0.32" width="200" height="200"/>
<material name="bed" texture="bed" texrepeat="50 50" texuniform="true" reflectance="0.05"/>
<mesh name="part" file="{stl}" scale="0.001 0.001 0.001"/>
<mesh name="cone" vertex="{cv}" face="{cf}"/>
</asset>
<worldbody>
<light pos="0.03 -0.05 0.12" dir="-0.2 0.4 -1" diffuse="0.55 0.55 0.55" castshadow="true"/>
<geom type="plane" size="0.08 0.08 0.001" material="bed"/>
<geom type="mesh" mesh="part" rgba="{TPU_RGBA}"/>
{chr(10).join(body)}
</worldbody>
</mujoco>"""


def render(xml, lookat, distance, azimuth, elevation, dst):
    import mujoco
    from PIL import Image
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = np.asarray(lookat) * 1e-3
    cam.distance, cam.azimuth, cam.elevation = distance * 1e-3, azimuth, elevation
    with mujoco.Renderer(m, H, W) as ren:
        ren.update_scene(d, camera=cam)
        img = ren.render()
    Image.fromarray(img).save(dst, optimize=True)
    print(f"{dst.relative_to(ROOT)}", flush=True)


def coupon_top(r, dst):
    """Top view of the seat coupon from its sheet: the slab, its chamfered corner, round pockets and slots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle
    Lx, Ly, _ = r["size_mm"]
    c = 1.5                                                   # chamfer leg, metric_magnet_prints.COUPON
    fig, ax = plt.subplots(figsize=(8.0, 7.6), dpi=200)
    ax.add_patch(Polygon([(c, 0), (Lx, 0), (Lx, Ly), (0, Ly), (0, c)], closed=True, fc="#eecc9e", ec="k", lw=2))
    for p in r["pockets"]:
        s = p["seat_diameter_mm"]
        if p["kind"] == "pocket":
            x, y = p["centre_print_mm"]
            ax.add_patch(Circle((x, y), s / 2, fc="white", ec="k", lw=1.6))
            ax.text(x, y + 2.3, f"{s:g}", ha="center", va="bottom", fontsize=30)
        else:
            x0, x1 = p["slot_x_mm"]
            yc = r["pockets"][0]["centre_print_mm"][1] - 7.0      # the slot row (COUPON y_slot)
            yc = 4.5 if abs(yc - 4.5) > 1e-6 else yc
            ax.add_patch(Rectangle((x0, yc - s / 2), x1 - x0, s, fc="white", ec="k", lw=1.6))
            ax.text((x0 + x1) / 2, yc - s / 2 - 0.6, f"{s:g}", ha="center", va="top", fontsize=30)
    ax.text(Lx / 2, Ly + 0.6, "Round pockets: block seat", ha="center", va="bottom", fontsize=26)
    ax.text(Lx / 2, -0.5, "Slots: curved-tip seat", ha="center", va="top", fontsize=26)
    ax.annotate("Chamfer", xy=(0.6, 0.6), xytext=(-5.5, -3.2), fontsize=26, ha="left",
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_xlim(-6, Lx + 1.5)
    ax.set_ylim(-4.6, Ly + 3.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(dst, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"{dst.relative_to(ROOT)}", flush=True)


def cmd_render(a):
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, r in geometries().items():
        lo, hi = np.array(json.loads(json.dumps(r.get("mesh", {}).get("bounds_mm") or [[0, 0, 0], r["size_mm"]])))
        ctr = (lo + hi) / 2
        upright = "tip" in r and r["tip"] != "block"
        span = float(np.max(hi - lo))
        if "tip" not in r:                                    # seat coupon: on the bed, then its top view
            xml = scene_xml(ROOT / r["stl"], r, [], None)
            render(xml, ctr, 4.2 * span, 120, -38, OUT / f"{stem}_bed.png")
            coupon_top(r, OUT / f"{stem}_top.png")
            continue
        if upright:                                           # palmar side +x
            arrow = ((hi[0] + 1.5, 0.0, ctr[2]), (1, 0, 0))
            az = 150
        else:                                                 # palmar face up
            arrow = ((ctr[0], ctr[1], hi[2] + 1.5), (0, 0, 1))
            az = 120
        xml = scene_xml(ROOT / r["stl"], r, [], arrow)
        render(xml, (ctr[0] + (3 if upright else 0), ctr[1], ctr[2] + 1.5), 4.4 * span, az, -28,
               OUT / f"{stem}_bed.png")
        for k, z in enumerate(cut_heights(r), start=1):
            cut = WORK / f"{stem}_z{z:g}.stl"
            if not cut.exists():
                raise SystemExit(f"{cut} missing: run the cut stage first")
            # the arrow beside the pockets: past the palmar side (upright) or beyond the block's distal end (X = 0)
            ar = ((hi[0] + 1.5, 0.0, z - 1.5), (1, 0, 0)) if upright else ((lo[0] - 2.5, ctr[1], 1.0), (0, 0, 1))
            xml = scene_xml(cut, r, magnets(r, k), ar)
            look = (ctr[0] + (4.5 if upright else -1.0), ctr[1], z - 1.0)
            render(xml, look, 2.5 * span, az, -55, OUT / f"{stem}_pause{k}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cut")
    sub.add_parser("walls")
    sub.add_parser("render")
    a = p.parse_args()
    {"cut": cmd_cut, "walls": cmd_walls, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    main()
