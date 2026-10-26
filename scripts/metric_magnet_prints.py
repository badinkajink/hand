#!/usr/bin/env python3
"""Fingertip prints for the owner's 3 x 2 mm magnets (docs/handoff/20261016-metric_magnet_prints.md).

Owner, 2026-10-16: "we already have 3x2 magnets. can you add to the queue generation jobs for fingertips to print for that
profile? we can add more magnets too, maybe 5 (matching the 5 magnetometers?). if we can reduce the number of walls around
the press-fit pocket we can try that too ... send some prints off as soon as possible."

Magnets: AplysiaTech N52 discs 3 x 2 mm, axially magnetized (amazon.com/dp/B0G52HW8SY), all with the north pole toward the
palmar face. Layout "four": the grid of scripts/tpu_tip_magnet_layouts.py, 6 mm across and 7 mm along the finger, centred
on the contact point; each magnet's top face 5 mm under the palmar reach, its axis along the palmar normal. Seat: 2.7 mm,
90 % of the diameter (tpu_tip_design.FIT_D, the owner's press fit), unless --seat says otherwise.

Tips (outer shapes and the recipe 95A TPU, 20 % gyroid, two 0.45 mm perimeters, four top and four bottom layers unchanged):
  block                   the block of the 2026-10-13 prints (scripts/tpu_tip_design.py), printed palmar face up at 0.2 mm;
                          each pocket a closed cylinder, floor and top on the layer grid with air above the magnet; one
                          pause before the first layer that bridges the pockets
  tcd_ell7.4x8.5x16x32,   the 2026-10-15 curved finalists (scripts/curved_tips_print.py), printed upright on the mount
  tcd_cap8.5,             face at 0.1 mm after a 0.2 mm first layer; each magnet stands on its edge in a slot as wide as
  tcd_ell7.4x8.5x11x22    the seat, closed below by the seat's half circle and open upward to its pause; the two
                          proximal magnets go in at a first pause, the two distal ones at a second

Checks: the least TPU between each complete pocket (slot included) and the outer surface, and between pockets (k-d trees on
surface samples, as curved_tips_print.build_part); every pocket sample inside the design; the TPU over each magnet along the
palmar normal. Slicing (PrusaSlicer 2.9.2 through flatpak, the owner's v17 template, FLEX on extruder 2) and the G-code check
of every pocket: the layers voided over the pocket, the floor under it, the pause, the bridge over it, islands, supported area.

    ~/.claude/bin/resguard.sh run --mem 2G -- taskset -c 19 ~/miniconda3/envs/femtip/bin/python \\
        scripts/metric_magnet_prints.py build --tips block tcd_cap8.5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_design as TD  # noqa: E402

DATE = "20261016"
D = ROOT / "docs/experiments/20261016-metric_magnet_prints"
OUT = D / "print"
WORK = ROOT / "logs/20261016-metric_magnet_prints/print"
MAGNET = (3.0, 2.0)                 # diameter, thickness, mm
MAGNET_SRC = "AplysiaTech N52 disc 3 x 2 mm, axially magnetized, nickel coated, +-0.1 mm (amazon.com/dp/B0G52HW8SY)"
DEPTH = 5.0                         # palmar reach to the magnet's top face, mm
GAP = 0.10                          # air above the seated magnet (upright: and across its thickness), mm
WALL_MIN = 1.8                      # four 0.45 mm perimeters: two of the outer wall and two of the pocket's, mm
SNAP = 0.0                          # upright: the seat's circle runs this far above its centre before the walls rise, mm
CURVED = ["tcd_ell7.4x8.5x16x32", "tcd_cap8.5", "tcd_ell7.4x8.5x11x22"]
TIPS = ["block"] + CURVED
LABEL = {"block": "block", "tcd_ell7.4x8.5x16x32": "half-ellipsoid 7.4 x 8.5 x 16/32 mm",
         "tcd_cap8.5": "capsule r 8.5 mm", "tcd_ell7.4x8.5x11x22": "half-ellipsoid 7.4 x 8.5 x 11/22 mm"}
RECIPE = dict(material="TPU 95A", slicer_filament="Generic FLEX @XL (template v17)", perimeters=2, line_width_mm=0.45,
              fill_pattern="gyroid", fill_density="20%", top_solid_layers=4, bottom_solid_layers=4, supports="none")
BED_XY = (180.0, 180.0)             # where the project places the part's origin on the XL's 360 mm bed
RES = 0.025                         # G-code raster, mm per pixel


def layout(name):
    """Magnet centres (y, z) in tpu_tip_design's frame (y across the finger, z from the mount face; contact point at
    (0, Z_C)), proximal first."""
    four = sorted(TD.grid4(), key=lambda c: (c[1], c[0]))
    if name == "four":
        return four
    if name == "five":                                   # the four corners and the contact point
        return sorted(four + [(0.0, TD.Z_C)], key=lambda c: (c[1], c[0]))
    if name.startswith("five_z"):                         # the corners pitch_z apart along the finger, and the centre
        pz = float(name[6:].replace("p", "."))
        return sorted([(sy * 3.0, TD.Z_C + sz * pz / 2) for sy in (-1, 1) for sz in (-1, 1)] + [(0.0, TD.Z_C)],
                      key=lambda c: (c[1], c[0]))
    raise ValueError(name)


# ------------------------------------------------------------------------------------------------------ geometry

def block_body():
    """The block as a trimesh in its print frame (mm): X = 21.41 - z (the distal end at X = 0), Y = y, Z = x + 8.5 (back
    face on the bed, palmar face up at Z = 17), the frame of eflesh_build.shape('sr2tip')."""
    m = TD.mesh(None, None)
    R = np.eye(4)
    R[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    m.apply_transform(R)
    m.apply_translation([TD.Z_END, 0.0, -TD.X_BACK])
    return m


def block_xy(y, z):
    return TD.Z_END - z, y


def block_pockets(centres, seat, depth=DEPTH, layer=0.2):
    """Closed cylindrical pockets, the magnet lying flat on the floor; floor and top on the layer grid (TD.recess)."""
    import manifold3d as M
    Dm, t = MAGNET
    r = seat / 2
    bottom = TD.X_FACE - depth - t - TD.X_BACK               # magnet's bottom face as a print height
    fl, top = TD.recess(bottom, t, layer, layer, GAP)
    n_top = int(round(top / layer)) + 1                       # first layer above the pocket (layer k spans (k-1) h .. k h)
    out = []
    for y, z in centres:
        X, Y = block_xy(y, z)
        cyl = M.Manifold.cylinder(top - fl, r, r, 128).translate([X, Y, fl])
        out.append(dict(solid=cyl, y_mm=y, z_mm=z, centre_print_mm=[round(X, 3), round(Y, 3)], kind="pocket",
                        seat_diameter_mm=seat, floor_mm=round(fl, 3), top_mm=round(top, 3),
                        magnet_top_print_mm=round(fl + t, 3), air_above_magnet_mm=round(top - fl - t, 3),
                        magnet_top_depth_mm=round(TD.X_FACE - TD.X_BACK - (fl + t), 3),
                        pause_print_z=round(n_top * layer, 3), pause_layer=n_top))
    return out


def upright_xy(y, z):
    """Print frame of curved_tips_print: x along the palmar normal (reach at 8.5), y_print = -y, z_print = z."""
    return -y, z


def upright_slots(centres, seat, depth=DEPTH):
    """Slots of curved_tips_print.slot at each centre: the magnet on its edge, a seat of the slot's width closed by its
    lower half circle, walls running up to the slot's own pause, the magnet's thickness plus GAP across x. With SNAP > 0 the
    circle runs SNAP mm above its centre and the walls rise from there, a throat narrower than the magnet (a snap fit)."""
    import manifold3d as M
    import curved_tips_print as CP
    Dm, t = MAGNET
    r = seat / 2
    x_hi = TD.X_FACE - depth
    x_lo = x_hi - t - GAP
    out = []
    for y, z in centres:
        yp, zp = upright_xy(y, z)
        top = zp + Dm / 2 + (Dm / 2 - r) + GAP                # the magnet sits (D/2 - r) above the seat's centre
        n_pause, z_bot, z_print = CP.pause_layer(top)
        disc = M.Manifold.cylinder(x_hi - x_lo, r, r, 128).rotate([0.0, 90.0, 0.0]).translate([x_lo, yp, zp])
        hw = math.sqrt(r * r - SNAP * SNAP)                  # the walls' half spacing: r without the snap fit
        box = M.Manifold.cube([x_hi - x_lo, 2 * hw, z_bot - zp - SNAP]).translate([x_lo, yp - hw, zp + SNAP])
        out.append(dict(solid=disc + box, y_mm=y, z_mm=z, centre_print_mm=[round(yp, 3), round(zp, 3)], kind="slot",
                        seat_diameter_mm=seat, snap_mm=SNAP, throat_mm=round(2 * hw, 3), slot_x_mm=[round(x_lo, 3), round(x_hi, 3)], seat_bottom_mm=round(zp - r, 3),
                        slot_top_needed_mm=round(top, 3), slot_top_mm=z_bot, magnet_top_depth_mm=depth,
                        magnet_top_print_mm=round(zp + Dm - r, 3), air_above_magnet_mm=round(z_bot - (zp + Dm - r), 3),
                        pause_print_z=z_print, pause_layer=n_pause))
    return out


def surface_x_tip(tip, y, z):
    """x of the palmar surface over (y, z) of tpu_tip_design's frame, in that frame (palmar reach 8.5 mm)."""
    if tip == "block":
        return float(TD.surface_x(y, z))
    import tip_codesign_geometry as TG
    return float(TG.shape(tip).fe_surface_x(np.array([y]), np.array([z]))[0])


def inside_tip(tip, P):
    """Points P (n, 3) of the print frame (mm) inside the design surface."""
    if tip == "block":
        x = P[:, 2] + TD.X_BACK
        z = TD.Z_END - P[:, 0]
        return TD.inside_body(x, P[:, 1], z)
    import tip_codesign_geometry as TG
    q = np.column_stack([P[:, 0] * 1e-3 + (TG.REACH - 0.0085), -P[:, 1] * 1e-3, TG.Z_MOUNT - P[:, 2] * 1e-3])
    return TG.shape(tip).inside_design(q)


def build(tip, centres, seat, depth=DEPTH):
    """Body, part (body minus every pocket) and the pocket records with their wall checks."""
    import eflesh_build as EB
    import trimesh
    from scipy.spatial import cKDTree
    if tip == "block":
        body = block_body()
        pockets = block_pockets(centres, seat, depth)
    else:
        import curved_tips_print as CP
        body = CP.upright_body(tip)
        pockets = upright_slots(centres, seat, depth)
    cut = pockets[0]["solid"]
    for p in pockets[1:]:
        cut = cut + p["solid"]
    part = EB.to_trimesh(EB.to_manifold(body) - cut)
    outer = trimesh.sample.sample_surface(body, 400000, seed=1)[0]
    tree = cKDTree(outer)
    samples = []
    for k, p in enumerate(pockets):
        sm = EB.to_trimesh(p.pop("solid"))
        pts = trimesh.sample.sample_surface(sm, 20000, seed=2 + k)[0]
        d, j = tree.query(pts)
        k0 = int(np.argmin(d))
        p["wall_min_mm"] = round(float(d.min()), 3)
        p["wall_min_at_print_mm"] = dict(pocket=np.round(pts[k0], 2).tolist(), surface=np.round(outer[j[k0]], 2).tolist())
        p["wall_le_2mm_area_frac"] = round(float(np.mean(d < 2.0)), 4)
        p["inside_body"] = bool(inside_tip(tip, pts).all())
        sx = surface_x_tip(tip, p["y_mm"], p["z_mm"])
        p["tpu_over_magnet_mm"] = round(sx - (TD.X_FACE - depth), 3)
        p["pocket_volume_mm3"] = round(float(sm.volume), 2)
        samples.append(pts)
    gaps = []
    for i in range(len(pockets)):
        ti = cKDTree(samples[i])
        for j in range(i + 1, len(pockets)):
            d, _ = ti.query(samples[j])
            gaps.append(dict(pair=[i, j], gap_mm=round(float(d.min()), 3)))
    shells = part.split(only_watertight=False)
    vols = sorted((float(s.volume) for s in shells), reverse=True)
    mesh = dict(watertight=bool(part.is_watertight), shells=len(shells), outer_shells=sum(v > 0 for v in vols),
                void_shells=sum(v < 0 for v in vols), void_volumes_mm3=[round(-v, 2) for v in vols if v < 0],
                body_volume_mm3=round(float(body.volume), 1), part_volume_mm3=round(float(part.volume), 1),
                bounds_mm=np.round(part.bounds, 3).tolist(), faces=int(len(part.faces)))
    return body, part, pockets, gaps, mesh


# ------------------------------------------------------------------------------------------------------ project

def pauses(pockets, tip):
    """One pause per distinct pause height, with the magnets it takes."""
    out = {}
    for i, p in enumerate(pockets):
        out.setdefault((p["pause_print_z"], p["pause_layer"]), []).append(i)
    rows = []
    for (z, n), idx in sorted(out.items()):
        if tip == "block":
            where = "into their pockets"
            pole = "north pole up (toward the palmar face)"
        else:
            zs = {round(pockets[i]["z_mm"], 2) for i in idx}
            where = "down their slots" if len(idx) > 1 else "down its slot"
            pos = "proximal" if max(zs) < TD.Z_C else ("distal" if min(zs) > TD.Z_C else "middle")
            where = f"({pos}) " + where
            pole = "north pole toward the curved palmar side"
        cnt = {1: "the magnet", 2: "the two magnets", 3: "the three magnets", 4: "the four magnets", 5: "the five magnets"}
        msg = f"Press {cnt[len(idx)]} {where} to the floor, {pole}, and resume"
        rows.append(dict(print_z=z, layer=n, magnets=idx, message=msg))
    return rows


def write_project(path, part, name, settings, layer_h, pause_rows, modifiers=()):
    """PrusaSlicer project on the v17 template (eflesh_build.write_3mf) with the object's layer height, every pause in
    pause_rows, and optional modifier volumes [(trimesh, settings dict)] appended to the object."""
    import eflesh_build as EB
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    first = pause_rows[0] if pause_rows else None
    EB.write_3mf(tmp, part, name, EB.TEMPLATE, settings, 2, first["print_z"] if first else None,
                 first["message"] if first else "", BED_XY)
    zin = zipfile.ZipFile(tmp)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            data = zin.read(it.filename)
            if it.filename == "Metadata/Slic3r_PE_model.config":
                s = data.decode()
                anchor = '<metadata type="object" key="extruder" value="2"/>'
                assert anchor in s
                s = s.replace(anchor, anchor + f'\n  <metadata type="object" key="layer_height" value="{layer_h:g}"/>')
                if modifiers:
                    s = _add_modifier_config(s, len(part.faces), modifiers)
                data = s.encode()
            elif it.filename == "3D/3dmodel.model" and modifiers:
                data = _add_modifier_mesh(data.decode(), part, modifiers).encode()
            elif it.filename == "Metadata/Prusa_Slicer_custom_gcode_per_print_z.xml":
                codes = "".join(f'<code print_z="{p["print_z"]:g}" type="1" extruder="2" color="" extra="{p["message"]}" '
                                f'gcode="M601"/>\n' for p in pause_rows)
                data = (f'<?xml version="1.0" encoding="utf-8"?>\n<custom_gcodes_per_print_z bed_idx="0">\n{codes}'
                        f'<mode value="MultiAsSingle"/>\n</custom_gcodes_per_print_z>\n').encode()
            zout.writestr(it, data)
    zin.close()
    tmp.unlink()


def _add_modifier_mesh(model, part, modifiers):
    """Append the modifiers' vertices and triangles to the object's mesh (one object, volumes by triangle ranges)."""
    nv = len(part.vertices)
    verts, tris = [], []
    for m, _ in modifiers:
        verts += [f'     <vertex x="{a:.5f}" y="{b:.5f}" z="{c:.5f}"/>' for a, b, c in np.asarray(m.vertices)]
        tris += [f'     <triangle v1="{a + nv}" v2="{b + nv}" v3="{c + nv}"/>' for a, b, c in np.asarray(m.faces)]
        nv += len(m.vertices)
    model = model.replace("\n    </vertices>", "\n" + "\n".join(verts) + "\n    </vertices>", 1)
    return model.replace("\n    </triangles>", "\n" + "\n".join(tris) + "\n    </triangles>", 1)


def _add_modifier_config(cfg, n_part_faces, modifiers):
    first = n_part_faces
    vols = []
    for k, (m, st) in enumerate(modifiers):
        last = first + len(m.faces) - 1
        keys = "".join(f'   <metadata type="volume" key="{a}" value="{b}"/>\n' for a, b in st.items())
        vols.append(f'  <volume firstid="{first}" lastid="{last}">\n   <metadata type="volume" key="name" value="modifier_{k}"/>\n'
                    f'   <metadata type="volume" key="volume_type" value="ParameterModifier"/>\n'
                    f'   <metadata type="volume" key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n{keys}'
                    f'   <mesh edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n'
                    f'  </volume>\n')
        first = last + 1
    return cfg.replace("  </volume>\n </object>", "  </volume>\n" + "".join(vols) + " </object>", 1)


def slice_project(project, work):
    """PrusaSlicer 2.9.2 on the project with the template's configuration (eflesh_build.slice_check), G-code kept in work."""
    import eflesh_build as EB
    _, _, kv = EB.template_config(EB.TEMPLATE)
    work.mkdir(parents=True, exist_ok=True)
    ini = work / "template.ini"
    ini.write_text("\n".join(f"{k} = {v}" for k, v in kv.items() if k != "binary_gcode") + "\nbinary_gcode = 0\n")
    gc = work / (project.stem + ".gcode")
    r = subprocess.run(EB.PRUSA + ["--load", str(ini), "--export-gcode", "-o", str(gc), str(project)], capture_output=True,
                       text=True)
    if r.returncode != 0 or not gc.exists():
        raise SystemExit(f"PrusaSlicer failed: {(r.stdout + r.stderr)[-800:]}")
    return gc


# ------------------------------------------------------------------------------------------------------ G-code check

def parse_arcs(path, obj=0, skip_types=("Custom", "Wipe tower", "Skirt/Brim", "Skirt"), chord=0.05):
    """gcode_audit.parse with the arcs of PrusaSlicer's arc fitting (the template's arc_fitting = emit_center writes G2/G3
    moves with I/J centre offsets): each arc is split into chords of at most `chord` mm. gcode_audit.parse reads G0/G1 only,
    so on this template it joins the end of every arc's predecessor to the start of its successor by a straight bead."""
    import re
    rx = re.compile(r"([XYZEFIJ])(-?\d*\.?\d+)")
    layers, types = [], {}
    cur = None
    typ, width, label = "", 0.45, None
    x = y = 0.0
    path_id, in_path = -1, False
    for line in open(path):
        if line.startswith(";"):
            if line.startswith(";LAYER_CHANGE"):
                cur = dict(z=None, h=None, seg=[], vol=0.0, travels=0, retracts=0)
                layers.append(cur)
                in_path = False
            elif line.startswith(";Z:") and cur is not None:
                cur["z"] = float(line[3:])
            elif line.startswith(";HEIGHT:") and cur is not None:
                cur["h"] = float(line[8:])
            elif line.startswith(";TYPE:"):
                typ = line[6:].strip()
                in_path = False
            elif line.startswith(";WIDTH:"):
                width = float(line[7:])
            continue
        if line.startswith("M486 S"):
            v = int(line.split()[1][1:])
            label = None if v < 0 else v
            in_path = False
            continue
        g = line[:3]
        if g not in ("G1 ", "G0 ", "G2 ", "G3 "):
            continue
        f = dict((k, float(v)) for k, v in rx.findall(line.split(";", 1)[0]))
        nx, ny = f.get("X", x), f.get("Y", y)
        e = f.get("E", 0.0)
        moved = ("X" in f) or ("Y" in f)
        mine = cur is not None and label == obj and typ not in skip_types
        if g in ("G2 ", "G3 ") and moved:
            cx, cy = x + f.get("I", 0.0), y + f.get("J", 0.0)
            r = math.hypot(x - cx, y - cy)
            a0, a1 = math.atan2(y - cy, x - cx), math.atan2(ny - cy, nx - cx)
            if g == "G2 ":
                while a1 >= a0 - 1e-12:
                    a1 -= 2 * math.pi
            else:
                while a1 <= a0 + 1e-12:
                    a1 += 2 * math.pi
            k = max(1, int(math.ceil(abs(a1 - a0) * r / chord)))
            th = np.linspace(a0, a1, k + 1)
            pts = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])
            pts[0], pts[-1] = (x, y), (nx, ny)
        else:
            pts = np.array([[x, y], [nx, ny]])
        if moved and e > 0 and mine:
            if not in_path:
                path_id += 1
                in_path = True
            tid = types.setdefault(typ, len(types))
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                cur["seg"].append((x0, y0, x1, y1, width, tid, path_id))
            cur["vol"] += e * math.pi * (1.75 / 2) ** 2
        elif moved and mine:
            if in_path:
                cur["travels"] += 1
            in_path = False
        elif e < 0 and mine:
            cur["retracts"] += 1
            in_path = False
        x, y = nx, ny
    for Lr in layers:
        Lr["seg"] = np.array(Lr["seg"], float).reshape(-1, 7)
    return [Lr for Lr in layers if len(Lr["seg"])], {v: k for k, v in types.items()}


def audit(project, gc):
    """eflesh_printability.audit (islands per layer, supported area, print time) on the arc-aware parse."""
    import eflesh_printability as EP
    real = EP.GA.parse
    EP.GA.parse = parse_arcs
    try:
        row = EP.audit(project.stem, gc, "scripts/metric_magnet_prints.py", (0, 0))
    finally:
        EP.GA.parse = real
    per = row.pop("per_layer", [])
    row["islands_by_layer"] = {f"{p['z']:.2f}": p["islands"] for p in per if p["islands"] > 1}
    return row


def gcode_layers(gc):
    """Layers (print_z, height, segments) of object 0, the pauses' print_z, the print time and the filament mass."""
    L, T = parse_arcs(gc, 0)
    pz, z = [], None
    t_print, grams = None, None
    left, at = None, []
    for line in open(gc):
        if line.startswith(";Z:"):
            z = float(line[3:])
        elif line.startswith("M73 P") and " R" in line:
            left = float(line.split(" R")[1].split()[0])          # minutes left in normal mode
        elif line.startswith("M601"):
            pz.append(z)
            at.append(left)
        elif "estimated printing time (normal mode)" in line:
            t_print = line.split("=")[1].strip()
        elif line.startswith("; total filament used [g]"):
            grams = float(line.split("=")[1])
    total = _minutes(t_print) if t_print else None
    t_pause = [round(total - a, 1) if (total is not None and a is not None) else None for a in at]
    return L, T, pz, t_print, grams, t_pause


def _minutes(txt):
    """PrusaSlicer's '1h 2m 3s' as minutes."""
    m = 0.0
    for part in txt.split():
        v, u = float(part[:-1]), part[-1]
        m += v * {"d": 1440, "h": 60, "m": 1, "s": 1 / 60}[u]
    return m


def _window_raster(seg, lo, n):
    from PIL import Image, ImageDraw
    img = Image.new("L", (n, n), 0)
    dr = ImageDraw.Draw(img)
    for x0, y0, x1, y1, w, _, _ in seg:
        a = ((x0 - lo[0]) / RES, (y0 - lo[1]) / RES)
        b = ((x1 - lo[0]) / RES, (y1 - lo[1]) / RES)
        r = 0.5 * w / RES
        dr.line([a, b], fill=255, width=max(1, int(round(w / RES))))
        dr.ellipse((a[0] - r, a[1] - r, a[0] + r, a[1] + r), fill=255)
        dr.ellipse((b[0] - r, b[1] - r, b[0] + r, b[1] + r), fill=255)
    return np.asarray(img) > 0


def section_mask(p, z_slice, X, Y, shrink=0.10):
    """Pocket section at the slice plane z_slice on the pixel centres (X, Y) of the bed, shrunk by `shrink` mm."""
    if p["kind"] == "pocket":
        cx, cy = BED_XY[0] + p["centre_print_mm"][0], BED_XY[1] + p["centre_print_mm"][1]
        if not (p["floor_mm"] < z_slice < p["top_mm"]):
            return None
        return (X - cx) ** 2 + (Y - cy) ** 2 <= (p["seat_diameter_mm"] / 2 - shrink) ** 2
    yp, zp = p["centre_print_mm"]
    r = p["seat_diameter_mm"] / 2
    if not (p["seat_bottom_mm"] < z_slice < p["slot_top_mm"]):
        return None
    h = p.get("snap_mm", 0.0)                             # snap fit: the circle runs h above the centre, then the throat
    w = math.sqrt(max(r * r - (z_slice - zp) ** 2, 0.0)) if z_slice < zp + h else math.sqrt(r * r - h * h)
    if w <= shrink:
        return None
    x0, x1 = p["slot_x_mm"]
    cy = BED_XY[1] + yp
    return ((X >= BED_XY[0] + x0 + shrink) & (X <= BED_XY[0] + x1 - shrink) & (np.abs(Y - cy) <= w - shrink))


def footprint_mask(p, X, Y, shrink=0.10):
    """The pocket's widest section, for the floor under it and the bridge over it."""
    if p["kind"] == "pocket":
        cx, cy = BED_XY[0] + p["centre_print_mm"][0], BED_XY[1] + p["centre_print_mm"][1]
        return (X - cx) ** 2 + (Y - cy) ** 2 <= (p["seat_diameter_mm"] / 2 - shrink) ** 2
    yp, _ = p["centre_print_mm"]
    x0, x1 = p["slot_x_mm"]
    return ((X >= BED_XY[0] + x0 + shrink) & (X <= BED_XY[0] + x1 - shrink)
            & (np.abs(Y - BED_XY[1] - yp) <= p["seat_diameter_mm"] / 2 - shrink))


def check_gcode(gc, pockets):
    """Per pocket: beads inside its section on the layers it voids, beads over its footprint on the layer under it and on the
    first layer above it (and that layer's feature types), and whether the pause precedes that layer."""
    L, T, pz, t_print, grams, t_pause = gcode_layers(gc)
    zs = [Lr["z"] for Lr in L]
    hs = [Lr["h"] if Lr["h"] else (zs[i] - zs[i - 1] if i else zs[0]) for i, Lr in enumerate(L)]
    out = []
    half = 3.0                                       # window half-size around a pocket, mm
    n = int(round(2 * half / RES))
    for p in pockets:
        if p["kind"] == "pocket":
            c = (BED_XY[0] + p["centre_print_mm"][0], BED_XY[1] + p["centre_print_mm"][1])
        else:
            c = (BED_XY[0] + 0.5 * sum(p["slot_x_mm"]), BED_XY[1] + p["centre_print_mm"][0])
        lo = (c[0] - half, c[1] - half)
        xs = lo[0] + RES * (np.arange(n) + 0.5)
        ys = lo[1] + RES * (np.arange(n) + 0.5)
        X, Y = np.meshgrid(xs, ys)                   # image rows are y, columns x
        fp = footprint_mask(p, X, Y)

        def beads(i, typ=None):
            s = L[i]["seg"]
            k = ((np.maximum(s[:, 0], s[:, 2]) > lo[0] - 1) & (np.minimum(s[:, 0], s[:, 2]) < lo[0] + 2 * half + 1)
                 & (np.maximum(s[:, 1], s[:, 3]) > lo[1] - 1) & (np.minimum(s[:, 1], s[:, 3]) < lo[1] + 2 * half + 1))
            if typ is not None:
                k &= s[:, 5] == typ
            return _window_raster(s[k], lo, n)

        masks = [section_mask(p, zs[i] - hs[i] / 2, X, Y) for i in range(len(L))]
        void = [i for i, m in enumerate(masks) if m is not None and m.any()]
        if not void:
            out.append(dict(void_layers=0))
            continue
        i0, i1 = void[0], void[-1]
        intrusion = max(float((beads(i) & masks[i]).sum() / masks[i].sum()) for i in void)
        floor = dict(z=zs[i0 - 1], covered=round(float((beads(i0 - 1) & fp).sum() / fp.sum()), 3)) if i0 > 0 else None
        bridge = None
        if i1 + 1 < len(L):
            j = i1 + 1
            types = sorted(T[int(t)] for t in np.unique(L[j]["seg"][:, 5]) if (beads(j, t) & fp).any())
            bridge = dict(z=zs[j], layer=j + 1, covered=round(float((beads(j) & fp).sum() / fp.sum()), 3), types=types)
        out.append(dict(void_layers=len(void), contiguous=bool(i1 - i0 + 1 == len(void)), void_from_z=round(zs[i0] - hs[i0], 3),
                        void_to_z=round(zs[i1], 3), beads_in_void_max=round(intrusion, 4), floor=floor, bridge=bridge,
                        pause_before_bridge=bool(bridge is not None and any(abs(z - bridge["z"]) < 1e-6 for z in pz))))
    return dict(pauses_found_z=pz, pause_at_min=t_pause, print_time=t_print, filament_g=grams, layers=len(L),
                layer_heights_mm=sorted({round(h, 3) for h in hs}), pockets=out)


# ------------------------------------------------------------------------------------------------------ commands

def name_of(tip, lay, seat, depth=DEPTH, walls=None):
    s = f"{DATE}-{tip}_{lay}_d3x2_depth{depth:g}_seat{seat:g}".replace(".", "p")
    return s + (f"_pocketwalls{walls}" if walls else "")


def _rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def walls_pass(pockets, gaps):
    return bool(min(p["wall_min_mm"] for p in pockets) >= WALL_MIN and min(g["gap_mm"] for g in gaps) >= WALL_MIN
                and all(p["inside_body"] for p in pockets))


DEPTH_STEPS = (5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)


def chip_field(centres, depth, gap=None):
    """Static field of the magnets (point dipoles, N52 Br 1.45 T, north toward the palmar face) at the five chips of the
    ReSkin board, its sensing points `gap` mm behind the back face (0.42: the chips against it), and the highest GAIN_SEL
    whose range covers the largest component (4.8 mT guaranteed at GAIN_SEL 7, scaled by the LSB at RES 0)."""
    import mlx90393_model as MLX
    import tpu_tip_magnet_sensing as MS
    gap = MLX.GAPS["touch"] if gap is None else gap
    row = dict(magnets=[(MAGNET[0], MAGNET[1], y, z) for y, z in centres], depth_mm=depth)
    B = MS.B0(row, MS.chips(row, "board", gap)) * 1e3                  # mT, (5 chips, tip x y z)
    bmax = float(np.abs(B).max())
    gain = next((g for g in range(7, -1, -1) if MS.B_RANGE_MT * MLX.SENS[g][0][0] / MLX.SENS[7][0][0] >= bmax), None)
    return dict(gap_mm=gap, B_chips_mT=np.round(B, 3).tolist(), B_max_component_mT=round(bmax, 3),
                B_U1_mT=round(float(np.linalg.norm(B[0])), 3), gain_sel=gain)


def cmd_build(a):
    out_dir = Path(a.out) if a.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for tip in a.tips:
        t0 = time.time()
        centres = layout(a.layout)
        search = []
        for depth in ([a.depth] if a.depth else DEPTH_STEPS):
            body, part, pockets, gaps, mesh = build(tip, centres, a.seat, depth)
            search.append(dict(depth_mm=depth, wall_min_mm=min(p["wall_min_mm"] for p in pockets),
                               gap_min_mm=min(g["gap_mm"] for g in gaps), passes=walls_pass(pockets, gaps)))
            print(f"{tip} depth {depth:g} mm: wall {search[-1]['wall_min_mm']} mm, gap {search[-1]['gap_min_mm']} mm", flush=True)
            if search[-1]["passes"]:
                break
        name = name_of(tip, a.layout, a.seat, depth, a.pocket_walls if a.pocket_walls != 2 else None)
        stl = out_dir / f"{name_of(tip, a.layout, a.seat, depth)}.stl"            # one part for both wall counts
        part.export(stl)
        upright = tip != "block"
        layer = 0.1 if upright else 0.2
        prow = pauses(pockets, tip)
        settings = dict(perimeters=RECIPE["perimeters"], fill_density=RECIPE["fill_density"], fill_pattern="gyroid",
                        top_solid_layers=RECIPE["top_solid_layers"], bottom_solid_layers=RECIPE["bottom_solid_layers"])
        proj = out_dir / f"{name}_w95_20.3mf"
        mods, mod_info = [], None
        if a.pocket_walls == 1:
            mod, mod_layers = onewall_modifier(body, pockets, upright)
            mods = [(mod, dict(perimeters=1))]
            mod_stl = out_dir / f"{name}_modifier.stl"
            mod.export(mod_stl)
            mod_info = dict(stl=_rel(mod_stl), perimeters=1, inset_mm=round(inset(upright), 4), layers=len(mod_layers),
                            z_mm=[mod_layers[0][0], mod_layers[-1][1]], volume_mm3=round(float(mod.volume), 1),
                            what=("ParameterModifier: on every layer that crosses a pocket, the body's section offset inward by the "
                                  "perimeter spacing; the band outside it keeps two perimeters and has room for one bead"))
        write_project(proj, part, f"{name}_w95_20", settings, layer, prow, mods)
        rep = dict(name=name, tip=tip, tip_label=LABEL[tip], layout=a.layout, stl=_rel(stl), project=_rel(proj),
                   recipe=dict(RECIPE, layer_height_mm=layer, first_layer_mm=0.2),
                   orientation=("upright, mount face on the bed" if upright else "back face on the bed, palmar face up"),
                   template=str(Path("~/wxie_workspace/fingers/merged_finger_v17.3mf")), extruder=2,
                   magnet=dict(source=MAGNET_SRC, diameter_mm=MAGNET[0], thickness_mm=MAGNET[1], count=len(pockets),
                               orientation="north pole toward the palmar face", top_depth_mm=depth),
                   pockets=pockets, pocket_gaps=gaps, wall_min_mm=min(p["wall_min_mm"] for p in pockets),
                   wall_rule_mm=WALL_MIN, walls_ok=walls_pass(pockets, gaps), depth_search=search,
                   chip_field=chip_field(centres, depth), pauses=prow, mesh=mesh, part_settings=settings,
                   pocket_walls=(a.pocket_walls if a.pocket_walls != 2 else None), modifier=mod_info,
                   script="scripts/metric_magnet_prints.py", when=time.strftime("%Y-%m-%d %H:%M"))
        if not a.no_slice:
            work = WORK / name
            gc = slice_project(proj, work)
            rep["slice"] = dict(audit(proj, gc), check=check_gcode(gc, pockets))
            zl = sorted({round(0.5 * (p["floor_mm"] + p["top_mm"]) if p["kind"] == "pocket" else p["centre_print_mm"][1] + 0.5, 1)
                         for p in pockets})
            rep["slice"]["loops"] = count_loops(gc, pockets, zl)
            rep["gcode"] = _rel(gc)
        rep["seconds"] = round(time.time() - t0, 1)
        (out_dir / f"{name}_w95_20.json").write_text(json.dumps(rep, indent=1, default=float))
        sl = rep.get("slice", {})
        ck = sl.get("check", {})
        print(f"{tip}: depth {depth:g} mm, watertight {mesh['watertight']}, shells {mesh['outer_shells']}+{mesh['void_shells']} "
              f"voids {mesh['void_volumes_mm3']} mm3; walls {[p['wall_min_mm'] for p in pockets]} mm, gaps "
              f"{[g['gap_mm'] for g in gaps]} mm, inside {[p['inside_body'] for p in pockets]}, over magnet "
              f"{[p['tpu_over_magnet_mm'] for p in pockets]} mm; field {rep['chip_field']['B_max_component_mT']} mT "
              f"(GAIN_SEL {rep['chip_field']['gain_sel']}); pauses {[(r['print_z'], r['layer']) for r in prow]}; "
              f"slice: pauses {ck.get('pauses_found_z')}, time {ck.get('print_time')}, islands max {sl.get('islands_max')}, "
              f"supported {sl.get('supported_area')}; pockets "
              + "; ".join(f"void {q['void_layers']} ({q.get('void_from_z')}-{q.get('void_to_z')}) in {q.get('beads_in_void_max')}, "
                          f"floor {(q.get('floor') or {}).get('covered')}, bridge {(q.get('bridge') or {}).get('z')} "
                          f"{(q.get('bridge') or {}).get('covered')} {(q.get('bridge') or {}).get('types')}, "
                          f"pause {q.get('pause_before_bridge')}" for q in ck.get("pockets", []))
              + f" ({rep['seconds']} s)", flush=True)


COUPON = dict(length=22.0, width=16.0, height=5.0, seats=(2.7, 2.8, 2.9), x=(4.5, 11.0, 17.5), y_pocket=11.5, y_slot=4.5,
              pocket_depth=2.2, slot_centre_below_top=1.8, chamfer=1.5)


def coupon_part():
    """A 22 x 16 x 5 mm slab printed flat with three open round pockets (the block's seat: the magnet lying flat, 2.2 mm deep)
    and three open slots (the curved tips' seat: the magnet on its edge, the seat's half circle 1.8 mm under the top), 2.7,
    2.8 and 2.9 mm wide from the chamfered end."""
    import eflesh_build as EB
    import manifold3d as M
    Dm, t = MAGNET
    c = COUPON
    H = c["height"]
    body = M.Manifold.cube([c["length"], c["width"], H])
    s = c["chamfer"] * math.sqrt(2)
    body = body - M.Manifold.cube([s, s, H + 2], center=True).rotate([0, 0, 45]).translate([0, 0, H / 2])
    slab = body
    pockets = []
    for x, seat in zip(c["x"], c["seats"]):
        r = seat / 2
        fl = H - c["pocket_depth"]
        body = body - M.Manifold.cylinder(H + 1 - fl, r, r, 128).translate([x, c["y_pocket"], fl])
        pockets.append(dict(kind="pocket", seat_diameter_mm=seat, centre_print_mm=[x, c["y_pocket"]], floor_mm=fl,
                            top_mm=H + 1, magnet_top_below_surface_mm=round(H - (fl + t), 3)))
        zc = H - c["slot_centre_below_top"]
        w = t + GAP
        disc = M.Manifold.cylinder(w, r, r, 128).rotate([0.0, 90.0, 0.0]).translate([x - w / 2, c["y_slot"], zc])
        box = M.Manifold.cube([w, seat, H + 1 - zc]).translate([x - w / 2, c["y_slot"] - r, zc])
        body = body - (disc + box)
        pockets.append(dict(kind="slot", seat_diameter_mm=seat, centre_print_mm=[c["y_slot"], zc],
                            slot_x_mm=[round(x - w / 2, 3), round(x + w / 2, 3)], seat_bottom_mm=round(zc - r, 3), slot_top_mm=H + 1,
                            magnet_top_below_surface_mm=round(H - (zc + Dm - r), 3)))
    return EB.to_trimesh(body), pockets, EB.to_trimesh(slab)


def cmd_coupon(a):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    part, pockets, slab = coupon_part()
    base = f"{DATE}-pocket_fit_coupon_d3x2_seats2p7_2p8_2p9"
    name = base + ("_pocketwalls1" if a.pocket_walls == 1 else "")
    stl = OUT / f"{base}.stl"
    part.export(stl)
    settings = dict(perimeters=RECIPE["perimeters"], fill_density=RECIPE["fill_density"], fill_pattern="gyroid",
                    top_solid_layers=RECIPE["top_solid_layers"], bottom_solid_layers=RECIPE["bottom_solid_layers"])
    proj = OUT / f"{name}_w95_20.3mf"
    mods, mod_info = [], None
    if a.pocket_walls == 1:
        mod, mod_layers = onewall_modifier(slab, pockets, False)
        mods = [(mod, dict(perimeters=1))]
        mod_stl = OUT / f"{name}_modifier.stl"
        mod.export(mod_stl)
        mod_info = dict(stl=_rel(mod_stl), perimeters=1, inset_mm=round(inset(False), 4), layers=len(mod_layers),
                        z_mm=[mod_layers[0][0], mod_layers[-1][1]], volume_mm3=round(float(mod.volume), 1))
    write_project(proj, part, f"{name}_w95_20", settings, 0.2, [], mods)
    rep = dict(name=name, stl=_rel(stl), project=_rel(proj), recipe=dict(RECIPE, layer_height_mm=0.2, first_layer_mm=0.2),
               orientation="flat, 22 x 16 mm on the bed", size_mm=[COUPON["length"], COUPON["width"], COUPON["height"]],
               magnet=dict(source=MAGNET_SRC, diameter_mm=MAGNET[0], thickness_mm=MAGNET[1]), pockets=pockets,
               mesh=dict(watertight=bool(part.is_watertight), volume_mm3=round(float(part.volume), 1)),
               pocket_walls=(1 if a.pocket_walls == 1 else None), modifier=mod_info,
               use=("Press a magnet into each seat with a flat, non-magnetic tool. The seat to use is the narrowest that takes the "
                    "magnet to its floor without splitting the wall; it should hold the magnet when the coupon is turned over and "
                    "tapped. Round pockets: the block's seat. Slots: the curved tips' seat, the magnet on its edge."),
               script="scripts/metric_magnet_prints.py", when=time.strftime("%Y-%m-%d %H:%M"))
    if not a.no_slice:
        gc = slice_project(proj, WORK / name)
        rep["slice"] = dict(audit(proj, gc), check=check_gcode(gc, pockets),
                            loops=count_loops(gc, pockets, [COUPON["height"] - 1.0]))
        rep["gcode"] = _rel(gc)
    rep["seconds"] = round(time.time() - t0, 1)
    (OUT / f"{name}_w95_20.json").write_text(json.dumps(rep, indent=1, default=float))
    ck = rep.get("slice", {}).get("check", {})
    print("loops", rep.get("slice", {}).get("loops"))
    print(f"coupon: watertight {rep['mesh']['watertight']}, time {ck.get('print_time')}, pockets "
          + "; ".join(f"{p['kind']} {p['seat_diameter_mm']}: void {q.get('void_layers')} in {q.get('beads_in_void_max')}, floor "
                      f"{(q.get('floor') or {}).get('covered')}" for p, q in zip(pockets, ck.get("pockets", []))), flush=True)


def sheets(layout_name=None, walls=None):
    out = []
    for f in sorted(OUT.glob(f"{DATE}-*_w95_20.json")):
        r = json.loads(f.read_text())
        if "tip" not in r or (layout_name and r["layout"] != layout_name) or r.get("pocket_walls") != walls:
            continue
        out.append(r)
    return sorted(out, key=lambda r: TIPS.index(r["tip"]))


def _pause_txt(r):
    ck = r.get("slice", {}).get("check", {})
    at = ck.get("pause_at_min") or [None] * len(r["pauses"])
    parts = []
    for p, m in zip(r["pauses"], at):
        n = len(p["magnets"])
        parts.append(f"{p['print_z']:g} mm (layer {p['layer']}{f', {m:.0f} min in' if m is not None else ''}), {n} magnets")
    return "; ".join(parts)


def _files(r):
    return f"{Path(r['project']).name}, {Path(r['stl']).name}, {Path(r['project']).stem}.json"


def readme_four(rows):
    lines = ["| Print | Orientation and layers | Magnet top under the apex (mm) | Least TPU around a pocket (mm) | Pauses | "
             "Print time | Files |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        ori = ("back face on the bed, 0.2 mm" if r["tip"] == "block" else "upright on the mount face, 0.1 mm")
        lines.append(f"| {r['tip_label'].capitalize()} | {ori} | {r['magnet']['top_depth_mm']:g} | {r['wall_min_mm']:.2f} | "
                     f"{_pause_txt(r)} | {r['slice']['check']['print_time']} | {_files(r)} |")
    return lines


def cmd_readme(a):
    four = sheets("four")
    cp = json.loads((OUT / f"{DATE}-pocket_fit_coupon_d3x2_seats2p7_2p8_2p9_w95_20.json").read_text())
    blk = next(r for r in four if r["tip"] == "block")
    cur = [r for r in four if r["tip"] != "block"]
    by = {r["tip"]: r for r in four}
    sup = [r["slice"]["supported_area"] for r in four]
    br = [q["bridge"]["covered"] for r in four for q in r["slice"]["check"]["pockets"]]
    air_c = sorted({p["air_above_magnet_mm"] for r in cur for p in r["pockets"]})
    e16, e11 = by["tcd_ell7.4x8.5x16x32"], by["tcd_ell7.4x8.5x11x22"]
    s11 = [s for s in e11["depth_search"] if not s["passes"]]
    order = [t for t in TIPS if t in by]
    short = {"block": "block", "tcd_ell7.4x8.5x16x32": "16/32", "tcd_cap8.5": "capsule", "tcd_ell7.4x8.5x11x22": "11/22"}
    L = [f"# Fingertips for 3 x 2 mm magnets ({DATE[:4]}-{DATE[4:6]}-{DATE[6:]})", "",
         "Four fingertips with four magnets each are ready to print: the block of the 2026-10-13 prints and the three curved "
         "finalists of 2026-10-15, each with four press-fit pockets for the owner's 3 x 2 mm magnets. A coupon of "
         f"{_minutes(cp['slice']['check']['print_time']):.0f} min tests the 2.7 mm seat against 2.8 and 2.9 mm before the tips. "
         "Every print has a PrusaSlicer project (3MF on the owner's v17 template, TPU on tool 2, pauses included), its STL and a "
         "sheet (JSON) with the geometry and G-code checks, written by `scripts/metric_magnet_prints.py build` and `coupon`.", "",
         "Recipe, as in the earlier prints: TPU 95A (the template's Generic FLEX @XL at 230 C), gyroid infill 20 %, two "
         "perimeters of 0.45 mm, four top and four bottom solid layers, no supports. The block prints on its back face in 0.2 mm "
         "layers. The curved tips print upright on their mount face in 0.1 mm layers after a 0.2 mm first layer, as on "
         "2026-10-15; PrusaSlicer raises their solid layers to the template's minimum shells, 0.7 mm on top and 0.5 mm at the "
         "bottom.", "",
         f"Magnets: {MAGNET_SRC}. Four per tip on the grid of the readout model, 6 mm across and 7 mm along the finger and "
         "centred on the contact point, each axis along the palmar normal, every north pole toward the palmar face. Seat: "
         f"{blk['pockets'][0]['seat_diameter_mm']:g} mm, 90 % of the diameter, the owner's press fit. Magnet depth: from the "
         "highest point of the palmar surface, over the contact point, to the magnets' top faces.", ""]
    L += readme_four(four)
    L += ["", "## Pauses", "",
          "The printer stops (M601) before the first layer over a pocket. Press each magnet to the pocket's floor with a flat, "
          "non-magnetic tool, north pole toward the palmar face, and resume. On the block all four magnets go in at one pause, "
          "lying flat, north pole up. On the curved tips the magnets stand on their edge in slots: the two proximal magnets go "
          "in at the first pause and the two distal ones at the second, north pole toward the curved palmar side. The layer "
          f"after each pause bridges {blk['pockets'][0]['air_above_magnet_mm']:.2f} mm (block) or "
          f"{', '.join(f'{v:.2f}' for v in air_c)} mm (curved tips) of air over the magnets, so the nozzle lays no plastic on "
          "them.", "",
          "## Magnet depth and TPU around the pockets", "",
          "Least TPU around a pocket: the smallest distance from the complete pocket (for a slot, its part above the seated "
          "magnet included) to the outer surface or to another pocket, from 400,000 samples of the outer surface and 20,000 of "
          "each pocket. Each tip uses the shallowest magnet depth, in 0.5 mm steps from 5 mm, that leaves "
          f"{WALL_MIN:g} mm, four 0.45 mm perimeters, around every pocket. The block and the capsule keep 5 mm. On the "
          "half-ellipsoids the top corners of the distal slots approach the distal surface: at 5 mm the 16/32 tip leaves "
          f"{e16['depth_search'][0]['wall_min_mm']:.2f} mm, and the 11/22 tip leaves "
          f"{', '.join(f'{x['wall_min_mm']:.2f}' for x in s11)} mm at {', '.join(f'{x['depth_mm']:g}' for x in s11)} mm, so "
          f"they use {e16['magnet']['top_depth_mm']:g} and {e11['magnet']['top_depth_mm']:g} mm.", "",
          "A deeper magnet lies closer to the magnetometer. With the five-chip board against the back face, the four magnets "
          "put at most " + ", ".join(f"{by[t]['chip_field']['B_max_component_mT']:.1f}" for t in order)
          + " mT on one axis of a chip (" + ", ".join(short[t] for t in order) + "). The MLX90393 reads that at gain setting "
          "GAIN_SEL " + ", ".join(str(by[t]["chip_field"]["gain_sel"]) for t in order)
          + "; its datasheet guarantees 4.8 mT at GAIN_SEL 7, and the range grows with the step size at lower settings. "
          "Field: point dipoles, N52 remanence 1.45 T.", "",
          "## Checks of the sliced G-code", "",
          "Every part is watertight with one outer shell and four void shells. PrusaSlicer 2.9.2 sliced each project with its "
          "own settings (G-code in logs/20261016-metric_magnet_prints/print/, untracked). In the G-code each pocket is empty on "
          "every layer it spans (no bead inside its section shrunk by 0.1 mm), the layer under it covers its footprint, and the "
          f"first layer over it is bridge infill over {min(br) * 100:.0f}-{max(br) * 100:.0f} % of the footprint, printed right "
          f"after the pause. Each layer is one island, and {min(sup) * 100:.0f}-{max(sup) * 100:.0f} % of the beads rest on the "
          "layer below. The template fits arcs (G2/G3); `scripts/gcode_audit.py` reads only straight moves, so the builder "
          "reads the arcs itself.", "",
          "## Seat coupon", "",
          f"{Path(cp['project']).name} ({Path(cp['stl']).name}, sheet {Path(cp['project']).stem}.json): "
          f"{cp['size_mm'][0]:g} x {cp['size_mm'][1]:g} x {cp['size_mm'][2]:g} mm, printed flat with the same recipe in 0.2 mm "
          f"layers, no pause, {_minutes(cp['slice']['check']['print_time']):.0f} min. One row has round pockets 2.2 mm deep, the "
          "block's seat with the magnet lying flat; the other has slots 2.1 mm across, the curved tips' seat with the magnet on "
          "its edge. Seats are 2.7, 2.8 and 2.9 mm wide from the chamfered end. Press a magnet into each seat with a flat, "
          "non-magnetic tool. The seat to use is the narrowest that takes the magnet to its floor without splitting the wall and "
          "keeps it in when the coupon is turned over and tapped. If a wider seat does better, "
          "`scripts/metric_magnet_prints.py build --seat 2.8` (or 2.9) writes the tips with it; the seat is in the file names.", "",
          "## Print risks", "",
          "The curved tips need someone at the printer for two pauses, about 17 and 29 minutes into the print. Magnets 6 mm apart "
          "pull toward each other: seat each one before picking up the next, and check each pole with a marked magnet, because a "
          "flipped magnet reverses its field at the chips. The slots are 2.1 mm across the magnet's thickness, 0.1 mm over the "
          "nominal 2 mm, and the magnets' tolerance is 0.1 mm: a thick magnet may stick in a slot printed undersize, which the "
          "coupon's slots show before a tip is printed.", "",
          "The 2026-10-13 four-pocket block for 4.76 x 1.59 mm magnets and the one-magnet prints are in "
          "docs/experiments/20261009-tpu_tip_fem/print/; the curved single-magnet prints are in "
          "docs/experiments/20261015-curved_tips/print/."]
    L += readme_extra()
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    print((OUT / "README.md").read_text())


def _count_word(v):
    words = {0: "no loop", 1: "one loop", 2: "two loops", 3: "three loops"}
    return words.get(v[0], f"{v[0]} loops") if len(set(v)) == 1 else ", ".join(str(x) for x in v) + " loops"


def readme_extra():
    """Sections for the five-magnet and one-wall variants, when their sheets exist."""
    L = []
    sense = {r["layout"]: r for r in json.loads(SENSE.read_text())["rows"]} if SENSE.exists() else {}
    four, five, w1 = sense.get("four_d3x2_d5"), sense.get("five_d3x2_d5"), sense.get("four_d3x2_d5_w1")
    f5 = sheets("five")
    if f5:
        r = f5[0]
        gaps = sorted({g["gap_mm"] for g in r["pocket_gaps"]})
        xy = "; ".join(f"({p['y_mm']:g}, {p['z_mm']:.2f})" for p in r["pockets"]).replace("(-0,", "(0,")
        L += ["", "## Five magnets", "",
              f"{Path(r['project']).name} ({Path(r['stl']).name}, sheet {Path(r['project']).stem}.json) is the block with a fifth "
              "magnet at the contact point. Magnet centres (y across the finger, z from the mount face, mm): "
              f"{xy}, every top {r['magnet']['top_depth_mm']:g} mm under the palmar face. The board's chips sit at U1 = "
              "(0, 10.86) and 7 mm from it along both axes; magnets 7 mm across would reach past the 7.4 mm half-width, so only "
              f"the centre magnet sits over a chip. The pockets keep {gaps[0]:.2f} mm between the centre and the corner pockets "
              f"(the 1.8 mm rule) and {r['wall_min_mm']:.2f} mm to the surface; all five go in at the one pause at "
              f"{r['pauses'][0]['print_z']:g} mm, and the print takes {r['slice']['check']['print_time']}. The five magnets put "
              f"{r['chip_field']['B_max_component_mT']:.1f} mT on a chip, which needs GAIN_SEL {r['chip_field']['gain_sel']}."]
        if four and five:
            L += ["", "Finite elements of the block (`scripts/metric_magnet_prints.py fe`, the model and grid of the 2026-10-13 "
                  "layout study, whose four-magnet row they reproduce: 16.70 MPa) and the board's Cramer-Rao bound "
                  "(`sense`; five chips against the back face, one 141 Hz sample at the noise of GAIN_SEL 7, contact position "
                  "and the Earth's field unknown) compare the two:", "",
                  "| Magnets | Pad modulus E* (MPa) | Face compliance lost at the contact point (%) | Force noise at the contact "
                  "point (N) | Force noise over 15 points, median and range (N) | Field on a chip (mT) |",
                  "|---|---|---|---|---|---|"]
            for k, r_ in (("Four", four), ("Five", five)):
                L.append(f"| {k} | {r_['E_star_MPa']:.1f} | {100 * r_['face_compliance_lost_centre']:.0f} | "
                         f"{r_['centre_both_unknown_N']:.3f} | {r_['median_both_unknown_N']:.2f} "
                         f"({r_['range_both_unknown_N'][0]:.2f}-{r_['range_both_unknown_N'][1]:.2f}) | {r_['B_max_component_mT']:.1f} |")
            L += ["", "E*: the sphere pads' modulus that reproduces the tip's approach under the screwdriver at 0.5, 1 and 3 N; "
                  "face compliance lost: the drop in the palmar face's deflection under a 1 N patch (Gaussian, sigma 0.6 mm) "
                  "against the tip without magnets. The bound at the contact point is "
                  f"{four['centre_both_unknown_N'] / five['centre_both_unknown_N']:.1f} times lower with five magnets: the four "
                  "corner magnets lie 4.6 mm from the contact point, the fifth under it. Below GAIN_SEL 7 the datasheet gives no "
                  "noise, so the five-magnet bound, taken at GAIN_SEL 7's noise for a field that needs GAIN_SEL 5, is a lower "
                  "bound.",
                  ]
            ck = sorted((D / "checks").glob(f"{DATE}-tcd_*_five_*.json"))
            if ck:
                c = json.loads(ck[0].read_text())
                n = {2: "two", 3: "three", 4: "four"}.get(len(c["pauses"]), str(len(c["pauses"])))
                L += ["", f"On the upright curved tips the same layout fails the rule: on the {LABEL[c['tip']]} each slot, open "
                      f"upward to its own pause, comes within {c['gap_min_mm']:.2f} mm of the next seat up, and the five magnets "
                      f"need {n} pauses (checks/{ck[0].name}). No curved five-magnet print is written."]
    w = sheets("four", walls=1)
    if w:
        r = w[0]
        lp = r["slice"]["loops"][0]
        L += ["", "## One wall at the pockets", "",
              f"{Path(r['project']).name} (sheet {Path(r['project']).stem}.json; part {Path(r['stl']).name}, modifier "
              f"{Path(r['modifier']['stl']).name}) is the four-magnet block with one perimeter around each pocket and the two of "
              "the recipe at the outer surface. PrusaSlicer has no setting for the walls of a hole alone, so the project adds a "
              "modifier volume with one perimeter: on each of the "
              f"{r['modifier']['layers']} layers that cross the pockets ({r['modifier']['z_mm'][0]:g}-{r['modifier']['z_mm'][1]:g} "
              f"mm), the block's section inset by {r['modifier']['inset_mm']:.3f} mm, the centre-to-centre spacing of the "
              "recipe's 0.45 mm perimeters at 0.2 mm layers. "
              "The band outside the modifier keeps the recipe's two perimeters and has room for one bead; the modifier adds its "
              "own outer loop beside it. In the G-code at "
              f"{lp['z']:g} mm every pocket has {_count_word(lp['loops_around_pockets'])} and the outer surface has "
              f"{_count_word([lp['loops_at_outer_surface']])}, their centrelines 0.20 and 0.63 mm in from the surface against the "
              "two-wall print's 0.23 and 0.63 mm. The pause, the bridges and the floors check as in the two-wall print; the "
              f"one-wall print takes {r['slice']['check']['print_time']} against 22m 35s."]
        if four and w1:
            L += ["", f"In the finite elements one 0.45 mm wall at the pocket lowers the pad modulus from {four['E_star_MPa']:.2f} "
                  f"to {w1['E_star_MPa']:.2f} MPa ({100 * (w1['E_star_MPa'] / four['E_star_MPa'] - 1):+.1f} %), the face compliance "
                  f"lost at the contact point from {100 * four['face_compliance_lost_centre']:.1f} to "
                  f"{100 * w1['face_compliance_lost_centre']:.1f} %, and the force noise at the contact point from "
                  f"{four['centre_both_unknown_N']:.3f} to {w1['centre_both_unknown_N']:.3f} N, as each magnet moves "
                  f"{100 * (np.mean(w1['magnet_travel_1N_um']) / np.mean(four['magnet_travel_1N_um']) - 1):.0f} % farther under "
                  "1 N. The press fit then rests on one bead backed by 20 % gyroid instead of two; the bridge 0.2 mm over the "
                  "magnet closes the pocket after the pause, so the fit has to keep the magnet in place only until that layer."]
        cw = OUT / f"{DATE}-pocket_fit_coupon_d3x2_seats2p7_2p8_2p9_pocketwalls1_w95_20.json"
        if cw.exists():
            c = json.loads(cw.read_text())
            lp = c["slice"]["loops"][0]
            L += ["", f"{Path(c['project']).name} (sheet {cw.name}, modifier {Path(c['modifier']['stl']).name}) is the seat "
                  f"coupon with the same modifier, {_minutes(c['slice']['check']['print_time']):.0f} min: at {lp['z']:g} mm every "
                  f"seat has {_count_word(lp['loops_around_pockets'])} and the outer surface "
                  f"{_count_word([lp['loops_at_outer_surface']])}. "
                  "Pressing magnets into it next to the two-wall coupon shows whether one wall keeps them in."]
    return L


def cmd_check(a):
    """Geometry only (no files to print): walls, pocket gaps and pauses of a layout on a tip, into checks/<name>.json."""
    out = D / "checks"
    out.mkdir(parents=True, exist_ok=True)
    for tip in a.tips:
        centres = layout(a.layout)
        _, _, pockets, gaps, mesh = build(tip, centres, a.seat, a.depth)
        rep = dict(tip=tip, layout=a.layout, depth_mm=a.depth, seat_mm=a.seat, pockets=pockets, pocket_gaps=gaps,
                   wall_min_mm=min(p["wall_min_mm"] for p in pockets), gap_min_mm=min(g["gap_mm"] for g in gaps),
                   walls_ok=walls_pass(pockets, gaps), pauses=pauses(pockets, tip), mesh=mesh,
                   script="scripts/metric_magnet_prints.py check", when=time.strftime("%Y-%m-%d %H:%M"))
        f = out / f"{name_of(tip, a.layout, a.seat, a.depth)}.json"
        f.write_text(json.dumps(rep, indent=1, default=float))
        print(f"{tip} {a.layout}: wall {rep['wall_min_mm']} mm, gap {rep['gap_min_mm']} mm, pauses "
              f"{[p['print_z'] for p in rep['pauses']]} -> {_rel(f)}", flush=True)


# ------------------------------------------------------------------------------------------------------ one wall at the pockets

def inset(upright):
    """Perimeter spacing of 0.45 mm beads at the layer height (PrusaSlicer: width - h (1 - pi/4)), the distance between the
    centrelines of the reference's two outer perimeters: the band at the outer surface that keeps the object's perimeters."""
    return RECIPE["line_width_mm"] - (0.1 if upright else 0.2) * (1 - math.pi / 4)


def layer_grid(upright, top):
    """Layer boundaries (bottom, top) up to `top` mm: 0.2 mm first layer, then 0.1 (upright) or 0.2 mm."""
    h = 0.1 if upright else 0.2
    out, z = [(0.0, 0.2)], 0.2
    while z < top - 1e-9:
        out.append((round(z, 4), round(z + h, 4)))
        z += h
    return out


def onewall_modifier(body, pockets, upright):
    """A ParameterModifier volume (perimeters 1) on every layer whose slice plane crosses a pocket: that layer's section of the
    body without pockets, offset inward by the perimeter spacing (inset()), extruded over the layer. PrusaSlicer perimeters each region on its own
    slices, so in those layers the band at the outer surface (perimeters 2, room for one bead) and the modifier's
    own outer loop give the outer surface its two perimeters, and each pocket gets the modifier's single loop."""
    import eflesh_build as EB
    import manifold3d as M
    B = EB.to_manifold(body)
    spans = []
    for p in pockets:
        lo = p["floor_mm"] if p["kind"] == "pocket" else p["seat_bottom_mm"]
        hi = p["top_mm"] if p["kind"] == "pocket" else p["slot_top_mm"]
        spans.append((lo, hi))
    prisms = []
    layers = []
    for zb, zt in layer_grid(upright, max(h for _, h in spans) + 0.2):
        zm = 0.5 * (zb + zt)
        if not any(lo < zm < hi for lo, hi in spans):
            continue
        cs = B.slice(zm).offset(-inset(upright), M.JoinType.Round)
        if cs.is_empty():
            continue
        prisms.append(cs.extrude(zt - zb).translate([0.0, 0.0, zb]))
        layers.append((zb, zt))
    mod = M.Manifold.batch_boolean(prisms, M.OpType.Add)
    return EB.to_trimesh(mod), layers


def count_loops(gc, pockets, z_list):
    """Closed perimeter loops on the given layers: around each pocket (every point of the path within 1.3 mm outside the
    pocket's edge, spanning at least 300 degrees around it) and at the outer surface (the path's extent at least 90 % of the
    layer's in x and y)."""
    L, T = parse_arcs(gc, 0)
    per_types = {k for k, v in T.items() if v in ("External perimeter", "Perimeter", "Overhang perimeter")}
    out = []
    for z in z_list:
        Lr = min(L, key=lambda l: abs(l["z"] - z))
        s = Lr["seg"]
        ext = np.array([s[:, [0, 2]].max() - s[:, [0, 2]].min(), s[:, [1, 3]].max() - s[:, [1, 3]].min()])
        paths = {}
        for row in s:
            if int(row[5]) in per_types:
                paths.setdefault(int(row[6]), []).append(row)
        outer, around = 0, []
        for p in pockets:
            if p["kind"] == "pocket":
                c = np.array([BED_XY[0] + p["centre_print_mm"][0], BED_XY[1] + p["centre_print_mm"][1]])
            else:
                c = np.array([BED_XY[0] + 0.5 * sum(p["slot_x_mm"]), BED_XY[1] + p["centre_print_mm"][0]])
            n = 0
            for rows in paths.values():
                q = np.array(rows)
                k = np.maximum(1, np.ceil(np.hypot(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1]) / 0.1)).astype(int)
                P = np.vstack([q[i, 0:2] + np.linspace(0, 1, k[i] + 1)[:, None] * (q[i, 2:4] - q[i, 0:2]) for i in range(len(q))])
                if p["kind"] == "pocket":
                    d = np.linalg.norm(P - c, axis=1) - p["seat_diameter_mm"] / 2
                else:
                    hx, hy = 0.5 * (p["slot_x_mm"][1] - p["slot_x_mm"][0]), p["seat_diameter_mm"] / 2
                    d = np.maximum(np.abs(P[:, 0] - c[0]) - hx, np.abs(P[:, 1] - c[1]) - hy)
                if d.min() < -0.05 or d.max() > 1.3:
                    continue
                ang = np.sort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
                gap = np.max(np.diff(np.r_[ang, ang[0] + 2 * np.pi]))
                if 2 * np.pi - gap >= np.radians(300):
                    n += 1
            around.append(n)
        for rows in paths.values():
            q = np.array(rows)
            e = np.array([q[:, [0, 2]].max() - q[:, [0, 2]].min(), q[:, [1, 3]].max() - q[:, [1, 3]].min()])
            if np.all(e >= 0.9 * ext):
                outer += 1
        out.append(dict(z=Lr["z"], loops_around_pockets=around, loops_at_outer_surface=outer,
                        types={T[int(t)]: int((s[:, 5] == t).sum()) for t in np.unique(s[:, 5])}))
    return out


# ------------------------------------------------------------------------------------------------------ finite elements

FE_ROWS = D / "fem.jsonl"


def fe_layout(name):
    """tpu_tip_magnet_layouts' layout record (magnets as diameter, thickness, y, z; tops `depth` under the palmar face)."""
    lay, depth = name.split("_d3x2_d")
    depth = float(depth.split("_")[0])
    return dict(depth=depth, mags=[(MAGNET[0], MAGNET[1], y, z) for y, z in layout(lay)])


def classes_walls(lay, pocket_walls=2):
    """tpu_tip_magnet.classes for the magnets of `lay`, with `pocket_walls` perimeters of 0.45 mm along each magnet's outline
    and tpu_tip_fem_tip.WALL (two perimeters) along the outer outline. pocket_walls = 2 is tpu_tip_magnet.classes."""
    from scipy import ndimage
    import tpu_tip_fem_tip as T
    import tpu_tip_magnet as TM
    import tpu_tip_magnet_layouts as TL
    H = TM.H_AUX
    occ, o = TL.occupancy(H, lay)
    body = TL.occupancy(H, dict(depth=None, mags=[]))[0] == 1
    tpu = occ == 1
    d_out = np.zeros(tpu.shape, np.float32)
    d_mag = np.full(tpu.shape, np.inf, np.float32)
    for k in range(tpu.shape[0]):                       # build direction x: one slice per layer plane
        if body[k].any():
            d_out[k] = ndimage.distance_transform_edt(np.pad(body[k], 1))[1:-1, 1:-1] * H
        mg = occ[k] == 2
        if mg.any():
            d_mag[k] = ndimage.distance_transform_edt(~mg) * H
    wall = tpu & ((d_out <= T.WALL + 1e-9) | (d_mag <= pocket_walls * T.WALL / 2 + 1e-9))
    n = tpu.shape[0]
    up = np.zeros(tpu.shape, np.float32)
    down = np.zeros(tpu.shape, np.float32)
    last = np.full(tpu.shape[1:], -1, int)
    for k in range(n):
        last = np.where(~tpu[k], k, last)
        down[k] = np.where(tpu[k], (k - last) * H, 0)
    nxt = np.full(tpu.shape[1:], n, int)
    for k in range(n - 1, -1, -1):
        nxt = np.where(~tpu[k], k, nxt)
        up[k] = np.where(tpu[k], (nxt - k) * H, 0)
    skin = tpu & ((up <= T.N_SKIN * T.LAYER + 1e-9) | (down <= T.N_SKIN * T.LAYER + 1e-9)) & ~wall
    cls = np.zeros(tpu.shape, np.uint8)
    cls[tpu] = 3
    cls[skin] = 2
    cls[wall] = 1
    cls[occ == 2] = 4
    return cls, o


def cmd_fe(a):
    """One finite-element row per layout name (tpu_tip_magnet_layouts.run on the block, recipe 95A 20 % gyroid): e.g.
    four_d3x2_d5, five_d3x2_d5, four_d3x2_d5_w1 (one perimeter along each pocket)."""
    import tpu_tip_magnet_layouts as TL
    done = {json.loads(l)["layout"] for l in open(FE_ROWS)} if FE_ROWS.exists() else set()
    for name in a.layouts:
        if name in done:
            print(name, "done")
            continue
        walls = int(name.split("_w")[1]) if "_w" in name else 2
        TL.LAYOUTS[name] = fe_layout(name)
        real = TL.classes
        TL.classes = lambda lay, w=walls: classes_walls(lay, w)
        try:
            TL.run(name, out=FE_ROWS)
        finally:
            TL.classes = real


SENSE = D / "sensing.json"
REF_ROWS = ROOT / "docs/experiments/20261013-magnet_layouts/fem.jsonl"     # the layout study's rows, same grid and model


def sense_row(row, none=None):
    """Cramer-Rao bounds of tpu_tip_magnet_sensing at the five chips against the back face (141 Hz setting), the pad modulus,
    and the face compliance each layout keeps relative to the tip without magnets."""
    import mlx90393_model as MLX
    import tpu_tip_magnet_sensing as MS
    a = MS.analyse(row, "board", MLX.GAPS["touch"])
    pts = a["points"]
    both = [p["sigF_n_unk_bg"] for p in pts]
    out = dict(layout=row["layout"], magnets=len(row["magnets"]), E_star_MPa=round(row["E_best_MPa"], 2),
               approach_1N_um=round(row["per_force"]["1"]["delta_um"], 1), B_max_component_mT=round(a["B0_max_mT"], 2),
               centre_known_N=round(a["centre"]["sigF_n"], 3), centre_both_unknown_N=round(a["centre"]["sigF_n_unk_bg"], 3),
               centre_sig_z_mm=round(a["centre"]["sig_z_mm_bg"], 2), centre_sig_y_mm=round(a["centre"]["sig_y_mm_bg"], 2),
               median_both_unknown_N=round(float(np.median(both)), 3), range_both_unknown_N=[round(min(both), 3), round(max(both), 3)],
               bias_range=[round(min(p["bias"] for p in pts), 2), round(max(p["bias"] for p in pts), 2)],
               magnet_travel_1N_um=[round(-m[0][0], 2) for m in row["per_force"]["1"].get("magnets", [])])
    if none:
        ref = {(p["y"], round(p["z"], 6)): p["normal_face_um_per_N"] for p in none["patches"]}
        lost = [1 - p["normal_face_um_per_N"] / ref[(p["y"], round(p["z"], 6))] for p in row["patches"]]
        cen = [1 - p["normal_face_um_per_N"] / ref[(p["y"], round(p["z"], 6))] for p in row["patches"]
               if p["y"] == 0.0 and abs(p["z"] - TD.Z_C) < 1e-6]
        out.update(face_compliance_lost_centre=round(cen[0], 3), face_compliance_lost_range=[round(min(lost), 3), round(max(lost), 3)])
    return out


def cmd_sense(a):
    rows = [json.loads(l) for l in open(FE_ROWS)]
    ref = {r["layout"]: r for r in (json.loads(l) for l in open(REF_ROWS))}
    none = ref.get("none")
    out = [sense_row(r, none) for r in rows]
    if "four_d3x2_d5" in ref:
        out.append(dict(sense_row(ref["four_d3x2_d5"], none), layout="four_d3x2_d5 (2026-10-13 row)"))
    SENSE.write_text(json.dumps(dict(rows=out, none_E_star_MPa=round(none["E_best_MPa"], 2) if none else None,
                                     setting="GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5 noise (1.15/1.75 uT, 141 Hz); chips against the back face",
                                     script="scripts/metric_magnet_prints.py sense", when=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    for r in out:
        print(json.dumps(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("readme")
    sub.add_parser("sense")
    k = sub.add_parser("check")
    k.add_argument("--tips", nargs="+", default=["tcd_cap8.5"], choices=TIPS)
    k.add_argument("--layout", default="five")
    k.add_argument("--seat", type=float, default=round(TD.FIT_D * MAGNET[0], 3))
    k.add_argument("--depth", type=float, default=DEPTH)
    f = sub.add_parser("fe")
    f.add_argument("--layouts", nargs="+", required=True)
    c = sub.add_parser("coupon")
    c.add_argument("--no-slice", action="store_true")
    c.add_argument("--pocket-walls", type=int, default=2, choices=[1, 2])
    b = sub.add_parser("build")
    b.add_argument("--tips", nargs="+", default=TIPS, choices=TIPS)
    b.add_argument("--layout", default="four")
    b.add_argument("--seat", type=float, default=round(TD.FIT_D * MAGNET[0], 3), help="seat diameter, mm (press fit)")
    b.add_argument("--depth", type=float, default=None,
                   help="palmar reach to the magnets' top faces, mm; default: the shallowest of 5, 5.5, ... 8 mm at which every "
                        "complete pocket keeps WALL_MIN of TPU to the surface and to the next pocket")
    b.add_argument("--pocket-walls", type=int, default=2, choices=[1, 2],
                   help="perimeters along each pocket (1: a modifier volume; the outer surface keeps two)")
    b.add_argument("--no-slice", action="store_true")
    b.add_argument("--out", default=None, help="directory for the files (default the print directory)")
    a = ap.parse_args()
    {"build": cmd_build, "coupon": cmd_coupon, "readme": cmd_readme, "fe": cmd_fe, "sense": cmd_sense, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
