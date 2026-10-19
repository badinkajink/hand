#!/usr/bin/env python3
"""Print plates of the DIP-link fit tests: every test piece of a print run on one plate per printer, each piece with a
printed ID (owner, 2026-10-19: "is there a more organized manner you can present the prints (and documentation) in? i
would also like to maximize printer plate utilization, although i guess it'll be hard to discern coupons").

PLA pieces go on the Bambu Lab P1S as one Bambu Studio project, placed by `layout` and sliced by Bambu Studio's command
line (the P1S, "0.12mm Fine @BBL X1C" and "Bambu PLA Basic" system presets with the overrides in P1S_PROCESS); 85A pieces
go on tool 2 of the Prusa XL 5T as one PrusaSlicer project on the v17 template (dip_link_print_project.build_project). The
run folder RUN contains both projects, each piece's STL (stl/, named by ID), the plate maps, pieces.json, records.csv keyed
by ID, and the print sheet that scripts/dip_fit_print_plates_page.py builds from them.

IDs: K = PLA key at a detent height, F = 85A channel block at a flank interference, C = board carrier at a chip-window
clearance (the digits are hundredths of a millimetre), CP = clamp plate, P = pin coupon, H = self-forming hole coupon,
W = cable-groove coupon. Concept b's run (--run 20261019b, folder 20261019-dip_link_fit_prints_b): R = the cover's rail
strip, G = 85A groove block at a flank interference, S = tongue with the seat pocket at a depth, N = neck carrier with
J1's slot and the cable groove, V = cover. Keys and carriers have the ID on a 1.4 mm tag at the bed, the channel blocks engraved in the
face that lies inside the tip, the coupons engraved beside each variant.

    unset VIRTUAL_ENV
    ~/.claude/bin/resguard.sh run --mem 5G --cpu 300 -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python scripts/dip_fit_print_plates.py [--run 20261019b] pieces bambu xl

Concept a's geometry comes from scripts/dip_link_v2.py (setup_a, coupon_carrier, _notches_y) at
docs/experiments/20261018-dip_link_board_mount/params/20261018-v2_concept_a.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

DATE = "20261019"
RUN = REPO / f"docs/experiments/{DATE}-dip_link_fit_prints"
STL = RUN / "stl"
WORK = REPO / f"logs/{DATE}-fit_prints"
PFX = f"{DATE}-dip_link_fit"
A_DIR = REPO / "docs/experiments/20261018-dip_link_board_mount/v2/a"
BAMBU = ["flatpak", "run", "--command=bambu-studio", "com.bambulab.BambuStudio"]
BBL = Path.home() / ".var/app/com.bambulab.BambuStudio/config/BambuStudio/system/BBL"
P1S = dict(machine="Bambu Lab P1S 0.4 nozzle", process="0.12mm Fine @BBL X1C",
           filament="Bambu PLA Basic @BBL P1S 0.4 nozzle")
P1S_PROCESS = dict(layer_height="0.1", wall_loops="3", sparse_infill_density="100%", sparse_infill_pattern="zig-zag",
                   seam_position="back", brim_type="auto_brim", enable_support="0",
                   curr_bed_type="Textured PEI Plate")      # the owner's plate in Bambu Studio (bed type 4): PLA at 55 C
P1S_BED, XL_BED, A1M_BED = (256.0, 256.0), (360.0, 360.0), (180.0, 180.0)
# 85A on the Bambu Lab A1 mini (owner, 2026-10-19 16:20: "the prusa is in use. ive loaded the sirayatech 85a tpu onto the
# external spool of the bambu a1 mini (using tube 3 for tool 3 on the ams)"): the TPU is the project's only filament, mapped
# to the AMS lite's slot 3 in Bambu Studio's send dialog. (A command-line project with the TPU as filament 3 of four wrote
# the first filament into its slice info and left the weight empty.)
A1M = dict(machine="Bambu Lab A1 mini 0.4 nozzle", process="0.20mm Standard @BBL A1M", filament="Generic TPU @BBL A1M")
A1M_SLOT = 3
A1M_PROCESS = dict(layer_height="0.2", wall_loops="2", sparse_infill_density="30%", sparse_infill_pattern="gyroid",
                   top_shell_layers="4", bottom_shell_layers="4", elefant_foot_compensation="0.2",
                   initial_layer_line_width="0.5", seam_position="back", brim_type="no_brim", enable_support="0",
                   enable_prime_tower="0", curr_bed_type="Textured PEI Plate",
                   minimum_sparse_infill_area="0", top_shell_thickness="0.7")
# minimum_sparse_infill_area: Bambu's 15 mm2 turned every infill island of the 4 mm blocks into solid zig-zag (the A1 mini
# plates of 16:55 had no gyroid); the XL projects keep gyroid in small islands (solid_infill_below_area 0). top_shell_thickness:
# Bambu's 1.0 mm adds a fifth top layer; 0.7 is the XL projects' top_solid_min_thickness, so four 0.2 mm skins as on the XL.
# Siraya Tech's XL profile as the 85A prints of 2026-10-16 to 10-18 ran it on tool 2 (print/template_changes.json of the
# readiness study: 240 C, 35 C bed, flow 0.96, 0.6 mm retraction, fan off for 4 layers), over Bambu's Generic TPU @BBL A1M;
# the volumetric cap is 2 mm3/s (about 24 mm/s at 0.2 mm layers) because the A1 mini's process speeds of 200-300 mm/s
# would otherwise run up to it, where the XL's 40-60 mm/s stayed under Siraya's 12 mm3/s.
SIRAYA_85A = dict(filament_vendor=["Siraya Tech"], nozzle_temperature=["240"], nozzle_temperature_initial_layer=["240"],
                  textured_plate_temp=["35"], textured_plate_temp_initial_layer=["35"], hot_plate_temp=["35"],
                  hot_plate_temp_initial_layer=["35"], cool_plate_temp=["35"], cool_plate_temp_initial_layer=["35"],
                  filament_flow_ratio=["0.96"], filament_retraction_length=["0.6"], filament_density=["1.26"],
                  filament_max_volumetric_speed=["2"], close_fan_the_first_x_layers=["4"], filament_colour=["#FFFFFF"])
SIRAYA_NAME = "Siraya Tech TPU 85A @BBL A1M"
# Siraya Tech's own A1 mini profile for the 85A (owner, 2026-10-19 16:50; their project of 2025-07-24, kept in the repo):
# its filament values replace the XL-derived SIRAYA_85A except the volumetric cap, 2 mm3/s against Siraya's 8, because the
# 85A reaches the extruder through AMS lite tube 3. Walls, gyroid and skins stay the finite elements' recipe on every
# printer (owner: "we shouldn't play with the gyroid infill % or the num walls"); Siraya's process (3 walls, 25 %) is unused.
SIRAYA_A1M_3MF = REPO / "docs/experiments/20261019-dip_link_fit_prints/siraya/siraya_tpu85a_hf_a1mini_0p4_20250724.3mf"
SIRAYA_KEEP = dict(filament_max_volumetric_speed=["2"])

DETENTS = (0.15, 0.25, 0.35, 0.45)                    # PLA key detent height, mm (v2/coupons: 0.15-0.35)
FLANKS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)    # 85A channel interference per flank, mm (v2/coupons: 0-0.20)
CHIP_CLEAR = (0.15, 0.20, 0.30)                       # chip-window clearance of the board carrier, mm
PIN_D = (2.25, 2.30, 2.35, 2.40)                      # pins of dip_link_board_mount.pin_coupon, from its chamfered end
M2_HOLES = (1.7, 1.8, 1.9)                            # self-forming M2 holes, mm (owner, 2026-10-18: 1.8)
M3_HOLES = (2.65, 2.75, 2.85)                         # self-forming M3 holes, mm (owner, 2026-10-18: 2.75)
GROOVES = (2.2, 2.4, 2.6)                             # cable-groove width, mm (v2 a: 2.4 for a 2.0 mm bundle)
GROOVE_DEPTH, GROOVE_R = 2.4, 2.0                     # groove depth; centre-line radius of its turn (v2 a cable_r)
# run b's grooves (owner, 2026-10-19 17:55: the Qwiic cable is a flat ribbon about 4 x 1 mm that had to be squished into
# W's 2.2-2.6 mm grooves; it is massaged into a bundle, modelled as 2.6 mm in a 3.0 mm groove)
GROOVES_B, GROOVE_DEPTH_B = (2.8, 3.0, 3.4), 3.0
TAG_T, ENGRAVE, ENGRAVE_TPU = 1.4, 0.6, 0.8           # tag thickness, engraving depth in PLA and in 85A, mm
FLANKS_B = (0.0, 0.05, 0.10, 0.15, 0.20)              # 85A pad-groove interference per flank on b's rails, mm (step 9)
SEAT_DEPTHS = (0.35, 0.45, 0.55)                      # depth of b's board seat pocket in the tongue, mm (design 0.45)

# Runs: each a folder of its own with one plate per printer. 20261019: concept a's fit tests (built 2026-10-19 14:58);
# 20261019b: concept b's (autopilot step 9, 2026-10-19). --run picks one; the default is the first.
RUNS = {"20261019": dict(folder="20261019-dip_link_fit_prints", pfx="20261019-dip_link_fit", work="20261019-fit_prints",
                         concept="a", params="docs/experiments/20261018-dip_link_board_mount/params/20261018-v2_concept_a.json"),
        "20261019b": dict(folder="20261019-dip_link_fit_prints_b", pfx="20261019-dip_link_fit_b", work="20261019-fit_prints_b",
                          concept="b", params="docs/experiments/20261018-dip_link_board_mount/params/20261018-v2_concept_b.json")}
CONCEPT = "a"


def set_run(rid):
    """Point RUN, STL, WORK, PFX and CONCEPT at run rid of RUNS."""
    global RUN, STL, WORK, PFX, CONCEPT
    r = RUNS[rid]
    RUN = REPO / f"docs/experiments/{r['folder']}"
    STL, WORK, PFX, CONCEPT = RUN / "stl", REPO / f"logs/{r['work']}", r["pfx"], r["concept"]


def hund(v):
    return f"{int(round(v * 100)):02d}"


def rel(p):
    return str(Path(p).resolve().relative_to(REPO))


# ------------------------------------------------------------------------------------------------ geometry
def M():
    import manifold3d
    return manifold3d


def to_man(tm):
    """trimesh -> manifold3d.Manifold (vertices merged first)."""
    import trimesh
    import trimesh.util
    t = trimesh.Trimesh(np.asarray(tm.vertices, float), np.asarray(tm.faces), process=True)
    t.merge_vertices()
    m = M()
    for attempt in range(2):
        mesh = m.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32))
        mesh.merge()
        man = m.Manifold(mesh)
        if man.status() == m.Error.NoError:
            return man
        # zero-volume slivers left by a boolean: keep the bodies over 1 mm3
        t = trimesh.util.concatenate([b for b in t.split(only_watertight=False) if abs(b.volume) > 1.0])
        t.merge_vertices()
    raise RuntimeError(f"not a manifold: {man.status()}")


def to_tm(man):
    import trimesh
    me = man.to_mesh()
    return trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3].astype(float), np.asarray(me.tri_verts), process=True)


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return M().Manifold.cube(list(hi - lo)).translate(list(lo))


def cyl_z(x, y, d, z0, z1, seg=64):
    return M().Manifold.cylinder(z1 - z0, d / 2, d / 2, seg).translate([x, y, z0])


def cyl_y(x, z, d, y0, y1, seg=64):
    return M().Manifold.cylinder(y1 - y0, d / 2, d / 2, seg).rotate([-90.0, 0.0, 0.0]).translate([x, y0, z])   # +z -> +y


def union(ms):
    m = M()
    return m.Manifold.batch_boolean(list(ms), m.OpType.Add)


def text_cs(s, cap, cx, cy):
    """The string s as a CrossSection in DejaVu Sans Bold, capital height `cap` mm, centred on (cx, cy), read along +x."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    h = TextPath((0, 0), "H", size=10.0, prop=fp).get_extents().height
    polys = [np.asarray(p, float) for p in TextPath((0, 0), s, size=10.0, prop=fp).to_polygons() if len(p) >= 3]
    allp = np.vstack(polys)
    c = (allp.min(axis=0) + allp.max(axis=0)) / 2
    k = cap / h
    m = M()
    return m.CrossSection([(p - c) * k + (cx, cy) for p in polys], m.FillRule.EvenOdd)


def engrave(man, s, cap, cx, cy, z_top, depth):
    return man - M().Manifold.extrude(text_cs(s, cap, cx, cy), depth + 0.05).translate([0.0, 0.0, z_top - depth])


def tag(x0, x1, y0, y1, s, cap):
    """A flat tag on the bed, TAG_T thick, with s engraved ENGRAVE deep in its centre."""
    return engrave(box((x0, y0, 0.0), (x1, y1, TAG_T)), s, cap, (x0 + x1) / 2, (y0 + y1) / 2, TAG_T, ENGRAVE)


def single(tm, name):
    """The mesh checked: watertight, one body, positive volume."""
    if not tm.is_watertight or len(tm.split(only_watertight=False)) != 1 or tm.volume <= 0:
        raise RuntimeError(f"{name}: watertight {tm.is_watertight}, bodies {len(tm.split(only_watertight=False))}, "
                           f"volume {tm.volume:.1f}")
    return tm


# ------------------------------------------------------------------------------------------------ pieces
def concept_a():
    import dip_link_v2 as V2
    A, DLF, *_ = V2.setup_a()
    return V2, A, A.MeshK()


def piece_key(V2, A, K, h, n):
    """Concept a's key with its detent at height h on the backing plate and stop flange of dip_link_v2.coupon_keys, n
    notches on the flange, printed as the DIP link prints; the tag behind the backing plate (the channel block slides on
    from the key's side and stops on the flange 2 mm above the bed)."""
    import dip_link_print_project as PP
    P = dict(A.P)
    yc, Zm, XB = P["key_yc"], A.Z_MOUNT, A.X_BACK
    key = A.keys(K, dict(P, detent_h=h))[1]
    unit = K.union([key, K.box((XB - 0.01, yc - 4.5, Zm - 6.0), (XB + 2.0, yc + 4.5, Zm + 0.01)),
                    K.box((XB - 3.0, yc - 4.5, Zm), (XB + 2.0, yc + 4.5, Zm + 2.0))])
    unit = K.sub(unit, V2._notches_y(K, XB - 3.1, XB - 2.4, yc - 3.6, Zm + 1.3, Zm + 2.1, n))
    tm, _ = PP.to_print(A.to_trimesh(unit), PP.R_LINK)
    lo, hi = tm.bounds
    return to_tm(to_man(tm) + tag(lo[0] - 10.5, lo[0] + 0.6, lo[1], hi[1], f"K{hund(h)}", 3.0))


def base_fill(man, step=0.05, look=1.0, zmax=5.0):
    """The lowest height from which a piece's cross-section grows outward by less than 0.5 mm2 over the next `look` mm,
    and the solid that fills everything under the section there down to the bed."""
    m = M()
    z = step
    while z < zmax:
        s0 = man.slice(z)
        grow = max(m.CrossSection.batch_boolean([man.slice(z + d), s0], m.OpType.Subtract).area()
                   for d in np.arange(0.1, look + 1e-9, 0.1))
        if grow < 0.5:
            break
        z += step
    return z, m.Manifold.extrude(man.slice(z + step), z + step) - man


def piece_carrier(V2, A, K, c):
    """dip_link_v2.coupon_carrier at chip-window clearance c (the DIP link's board region with the J1 slot, the cable
    groove, keys and shoulders), printed as the DIP link prints; the tag on the bed beyond the cut face at -y_F.
    The cut at the bridge's flat (z_F -63.10) leaves the carrier standing on two 0.8 mm strips under the bridge's curved
    face, which reaches its full section 1.85 mm above the bed; that flare is filled down to the bed from the full section
    (199.5 mm3; the J1 plug, the cable and the board lie outside it)."""
    import dip_link_print_project as PP
    tm, _ = PP.to_print(V2.coupon_carrier(A, K, c), PP.R_LINK)
    lo, hi = tm.bounds
    man = to_man(tm)
    z, fill = base_fill(man)
    print(f"   C{hund(c)}: base filled to {z:.2f} mm ({fill.volume():.1f} mm3)", flush=True)
    return to_tm(man + fill + tag(-7.0, 7.0, lo[1] - 7.0, lo[1] + 0.6, f"C{hund(c)}", 3.2))


def piece_clamp():
    import dip_link_print_project as PP
    import trimesh
    tm, _ = PP.to_print(trimesh.load(A_DIR / "cad/20261018-dip_link_v2a_clamp_plate.stl", force="mesh"), PP.R_CLAMP)
    return to_tm(to_man(tm))


def piece_pins():
    """dip_link_board_mount.pin_coupon as printed for v2 a, with a tag along its -y side naming each pin's diameter in
    hundredths over 2 mm (25 = 2.25 mm)."""
    import dip_link_print_project as PP
    import trimesh
    tm, _ = PP.to_print(trimesh.load(A_DIR / "print/20261018-dip_link_v2a_pin_fit_coupon.stl", force="mesh"))
    lo, hi = tm.bounds
    t = box((lo[0], lo[1] - 5.6, 0.0), (hi[0], lo[1] + 0.6, TAG_T))
    for k, d in enumerate(PIN_D):
        t = engrave(t, hund(d - 2.0), 2.5, lo[0] + 6.0 * (k + 0.5), lo[1] - 2.5, TAG_T, ENGRAVE)
    return to_tm(to_man(tm) + t)


def piece_holes():
    """Self-forming screw holes: M3 through holes (2.65, 2.75, 2.85 mm) and blind M2 holes 6 mm deep (1.7, 1.8, 1.9 mm)
    from the top, and the same M2 holes 7 mm deep into the front face, the sizes engraved on the top."""
    blk = box((-23.0, -13.0, 0.0), (16.0, 13.0, 8.0))
    xs = (-10.0, 0.0, 10.0)
    cuts = [cyl_z(x, 7.0, d, -0.1, 8.1) for x, d in zip(xs, M3_HOLES)]
    cuts += [cyl_z(x, -4.0, d, 2.0, 8.1) for x, d in zip(xs, M2_HOLES)]
    cuts += [cyl_y(x, 3.5, d, -13.1, -6.0) for x, d in zip(xs, M2_HOLES)]
    man = blk - union(cuts)
    for x, d in zip(xs, M3_HOLES):
        man = engrave(man, f"{d:.2f}", 2.4, x, 2.0, 8.0, ENGRAVE)
    for x, d in zip(xs, M2_HOLES):
        man = engrave(man, f"{d:.1f}", 2.4, x, -9.4, 8.0, ENGRAVE)
    man = engrave(man, "M3", 3.0, -18.0, 7.0, 8.0, ENGRAVE)
    man = engrave(man, "M2", 3.0, -18.0, -4.0, 8.0, ENGRAVE)
    return to_tm(man)


def groove_cs(x0, w):
    """An open groove of width w: 9 mm along +y, a quarter turn of centre-line radius GROOVE_R to +x, 5 mm along +x."""
    m = M()
    R, hw = GROOVE_R, w / 2
    cx, cy = x0 + R, -2.0
    th = np.radians(np.linspace(90.0, 180.0, 33))
    sector = np.vstack([np.column_stack([cx + (R + hw) * np.cos(th), cy + (R + hw) * np.sin(th)]),
                        np.column_stack([cx + (R - hw) * np.cos(th[::-1]), cy + (R - hw) * np.sin(th[::-1])])])
    rect = lambda a, b: np.array([[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]])   # noqa: E731
    t = np.linspace(0, 2 * np.pi, 49)[:-1]
    end = np.column_stack([x0 + R + 5.0 + hw * np.cos(t), cy + R + hw * np.sin(t)])
    parts = [m.CrossSection([p], m.FillRule.NonZero) for p in
             (rect((x0 - hw, -11.2), (x0 + hw, cy + 0.01)), sector, rect((cx - 0.01, cy + R - hw), (x0 + R + 5.0, cy + R + hw)),
              end)]
    return m.CrossSection.batch_boolean(parts, m.OpType.Add)


def piece_grooves(widths=GROOVES, depth=GROOVE_DEPTH):
    """Three open cable grooves `depth` deep and `widths` wide (run a: 2.2, 2.4 and 2.6 mm, 2.4 deep), each turning 90
    degrees about a GROOVE_R centre line as the routed cable turns over the plug and at the slotted arm; the widths
    engraved above."""
    m = M()
    blk = box((-20.0, -11.0, 0.0), (21.0, 11.0, 4.4))
    man = blk
    for x0, w in zip((-15.0, -2.0, 11.0), widths):
        man = man - m.Manifold.extrude(groove_cs(x0, w), depth + 0.1).translate([0.0, 0.0, 4.4 - depth])
        man = engrave(man, f"{w:.1f}", 3.0, x0 + 4.5, 6.5, 4.4, ENGRAVE)
    return to_tm(man)


def piece_channel(A, K, fl):
    """An 85A channel block of the tip's back region (dip_link_v2.coupon_channels) at flank interference fl, its ID
    engraved ENGRAVE_TPU deep in the face that lies inside the tip (the top as printed). Returns the block in its print
    frame and in frame F (dip_link_print_project.build_project turns channel blocks from F)."""
    import dip_link_print_project as PP
    import trimesh
    P = dict(A.P)
    yc, Zm, XB = P["key_yc"], A.Z_MOUNT, A.X_BACK
    blk = K.box((XB - 4.0, yc - 4.5, Zm - 7.0), (XB, yc + 4.5, Zm))
    tm, (R, t) = PP.to_print(A.to_trimesh(K.sub(blk, A.tip_cutter(K, dict(P, fit_flank=fl)))), PP.R_CLAMP)
    pr = to_tm(engrave(to_man(tm), hund(fl), 3.4, 0.0, 0.0, float(tm.bounds[1][2]), ENGRAVE_TPU))
    return pr, trimesh.Trimesh((np.asarray(pr.vertices) - t) @ R, np.asarray(pr.faces), process=False)


def concept_b():
    import dip_link_v2 as V2
    B = V2.b_module()
    return V2, B, B.MeshB()


def piece_rails(B, K):
    """A 10 mm length of concept b's cover with its two dovetail rails (as v2/b's rail strip), printed as the cover prints,
    on its back face with the rails up; the tag R along its +y side."""
    import dip_link_print_project as PP
    z0, z1 = B.Z_CONTACT - 5.0, B.Z_CONTACT + 5.0
    strip = K.box((B.X_SLOT, -B.Y_RIM, z0), (B.X_COMP, B.Y_RIM, z1))
    rails = [K.prism_z([(s_ * y, (B.X_SLOT + 0.1) if x > B.X_SLOT else x) for y, x in B.rail_section()], z0, z1)
             for s_ in (-1, 1)]
    tm, _ = PP.to_print(B.to_trimesh(K.fuse(strip, rails)), PP.R_CLAMP)
    lo, hi = tm.bounds
    return to_tm(to_man(tm) + tag(lo[0], hi[0], hi[1] - 0.6, hi[1] + 6.0, "R", 3.0))


def piece_groove_block(B, K, fl):
    """An 85A block of b's pad back (4 mm deep, the pad's width, 9 mm long) with the two rail grooves at flank
    interference fl, printed groove mouths down as the pad's back; its ID engraved ENGRAVE_TPU deep in the top face, which
    lies inside the pad. Returns the block in its print frame and in frame F (dip_link_print_project.build_project turns
    channel blocks from F)."""
    import dip_link_print_project as PP
    import trimesh
    z0, z1 = B.Z_CONTACT - 5.0, B.Z_CONTACT + 5.0
    blk = K.box((B.X_SLOT - 4.0, -B.TIP["y_half"], z0 + 0.5), (B.X_SLOT, B.TIP["y_half"], z1 - 0.5))
    cut = [B.prism([(s_ * y, B.X_SLOT + 0.5 if x > B.X_SLOT else x) for y, x in B.rail_section(-fl, B.GROOVE_FLOOR)], z0, z1)
           for s_ in (-1, 1)]
    tm, (R, t) = PP.to_print(B.to_trimesh(K.cut(blk, cut)), PP.R_CLAMP)
    pr = to_tm(engrave(to_man(tm), hund(fl), 3.4, 0.0, 0.0, float(tm.bounds[1][2]), ENGRAVE_TPU))
    return pr, trimesh.Trimesh((np.asarray(pr.vertices) - t) @ R, np.asarray(pr.faces), process=False)


def piece_seat(V2, B, K, depth):
    """The tongue of b's DIP link with the board's seat pocket at `depth` (rim_h), from its distal end to 2 mm past the
    proximal datum wall, with the tab slot and the four 1.8 mm screw holes, printed on its back face with the pocket up;
    the tag along its +y side."""
    import dip_link_print_project as PP
    vals, des = B.load_params()
    B.apply(dict(vals, rim_h=depth), des)
    try:
        body, _ = B.dip_link_parts(K)
        box_ = K.box((B.X_SEAT - depth - 0.01, -B.Y_RIM - 0.5, B.TIP["z_end"] - 0.1),
                     (B.X_TONGUE_BACK + 0.01, B.Y_RIM + 0.5, B.Z_BOARD[1] + 2.0))
        tm, _ = PP.to_print(B.to_trimesh(body ^ box_), PP.R_CLAMP)
    finally:
        B.apply(vals, des)
    lo, hi = tm.bounds
    return to_tm(to_man(tm) + tag(lo[0], lo[0] + 12.0, hi[1] - 0.6, hi[1] + 6.0, f"S{hund(depth)}", 3.0))


def piece_neck_carrier(V2, B, K):
    """b's DIP link around the board: the tongue with the seat pocket, the neck with the J1 slot and the bridge with the
    cable's groove (open to the top over the plug, roofed along the run), cut to x -1.5..9.6, |y| <= 11 and z up to -61.5;
    printed on its back face (the spine's back on the bed). The tongue's back stands 1.45 mm above that plane, so the
    tongue is thickened dorsally to it, the four screw holes kept through; the tag N on the bed beyond the tongue's distal
    end."""
    import dip_link_print_project as PP
    body, _ = B.dip_link_parts(K)
    box_ = K.box((-1.5, -11.0, B.TIP["z_end"] - 0.5), (9.6, 11.0, -61.5))
    back = K.box((B.X_TONGUE_BACK - 0.01, -B.Y_RIM, B.TIP["z_end"]), (B.X_SPINE_BACK, B.Y_RIM, B.TIP["z_mount"] + 0.01))
    holes = [K.cylx(y, z, B.P["screw_thread_d"] / 2, B.X_TONGUE_BACK - 0.2, B.X_SPINE_BACK + 0.2) for y, z in B.screw_yz()]
    carrier = K.cut(K.fuse(body ^ box_, [back]), holes)
    tm, _ = PP.to_print(B.to_trimesh(carrier), PP.R_CLAMP)
    print(f"   N: tongue thickened by {B.X_SPINE_BACK - B.X_TONGUE_BACK:.2f} mm to the spine's back", flush=True)
    lo, hi = tm.bounds
    return drop_slivers(to_tm(to_man(tm) + tag(lo[0] - 7.0, lo[0] + 0.6, -7.0, 7.0, "N", 3.2)))


def drop_slivers(tm):
    """The mesh without the zero-volume bodies a boolean leaves where two faces touched."""
    import trimesh
    bodies = [b for b in tm.split(only_watertight=False) if abs(b.volume) > 0.5]
    out = trimesh.util.concatenate(bodies) if len(bodies) > 1 else bodies[0]
    out.merge_vertices()
    return out


def piece_cover(B):
    import dip_link_print_project as PP
    import trimesh
    tm, _ = PP.to_print(trimesh.load(B.CAD_DIR / f"{B.PREFIX}_cover.stl", force="mesh"), PP.R_CLAMP)
    return to_tm(to_man(tm))


# ------------------------------------------------------------------------------------------------ piece list
def piece_specs():
    return piece_specs_a() if CONCEPT == "a" else piece_specs_b()


def piece_specs_b():
    """Every piece of concept b's run: ID, test, variant, printer and what the owner checks on it."""
    S = [dict(id="R", stl="R_rail_strip.stl", printer="p1s", material="PLA", test="Snap fit", piece="Rail strip",
              variant="the cover's two rails, 10 mm long", value=None, mark="tag R", copies=1, short="rails")]
    for fl in FLANKS_B:
        S.append(dict(id=f"G{hund(fl)}", stl=f"G{hund(fl)}_groove_flank{hund(fl)}_85a.stl", printer="xl", material="85A",
                      test="Snap fit", piece="85A groove block", variant=f"{fl:.2f} mm per flank", value=fl,
                      mark=f"{hund(fl)} engraved on the face inside the pad", copies=1, short=f"{fl:.2f}/flank"))
    for d in SEAT_DEPTHS:
        S.append(dict(id=f"S{hund(d)}", stl=f"S{hund(d)}_seat_pocket{hund(d)}.stl", printer="p1s", material="PLA",
                      test="Board seat", piece="Tongue with the seat pocket", variant=f"pocket {d:.2f} mm deep", value=d,
                      mark=f"tag S{hund(d)}", copies=1, short=f"pocket {d:.2f}"))
    S.append(dict(id="N", stl="N_neck_carrier.stl", printer="p1s", material="PLA", test="Board seat and cable",
                  piece="Neck carrier", variant="J1 slot, seat pocket 0.45 mm, cable groove", value=None, mark="tag N",
                  copies=1, short="neck, J1, cable",
                  note="the tongue's back stands 1.45 mm above the spine's back; that gap is filled down to the bed"))
    S.append(dict(id="V", stl="V_cover.stl", printer="p1s", material="PLA", test="Board seat", piece="Cover",
                  variant="v2 b", value=None, mark="its shape (windows and two rails; two copies)", copies=2, short="cover"))
    S.append(dict(id="W", stl="W_cable_grooves_28_30_34.stl", printer="p1s", material="PLA", test="Cable groove",
                  piece="Groove coupon", variant=f"widths {', '.join(f'{w:.1f}' for w in GROOVES_B[:-1])} and "
                  f"{GROOVES_B[-1]:.1f} mm, {GROOVE_DEPTH_B:.1f} mm deep", value=None,
                  mark="widths engraved above the grooves", copies=1, short="grooves 2.8-3.4"))
    return S


def piece_specs_a():
    """Every piece of the run: ID, test, variant, printer and what the owner checks on it."""
    S = []
    for k, h in enumerate(DETENTS):
        S.append(dict(id=f"K{hund(h)}", stl=f"K{hund(h)}_key_detent{hund(h)}.stl", printer="p1s", material="PLA",
                      test="Snap fit", piece="PLA key", variant=f"detent {h:.2f} mm", value=h, mark=f"tag K{hund(h)}, "
                      f"{k + 1} notch{'es' if k else ''} on the flange", copies=1, short=f"detent {h:.2f}"))
    for fl in FLANKS:
        S.append(dict(id=f"F{hund(fl)}", stl=f"F{hund(fl)}_channel_flank{hund(fl)}_85a.stl", printer="a1m", material="85A",
                      test="Snap fit", piece="85A channel block", variant=f"{fl:.2f} mm per flank", value=fl,
                      mark=f"{hund(fl)} engraved on the face inside the tip", copies=1, short=f"{fl:.2f}/flank"))
    for c in CHIP_CLEAR:
        S.append(dict(id=f"C{hund(c)}", stl=f"C{hund(c)}_board_carrier_chip{hund(c)}.stl", printer="p1s", material="PLA",
                      test="Board fit", piece="Board carrier", variant=f"chip windows {c:.2f} mm", value=c,
                      mark=f"tag C{hund(c)}", copies=1, short=f"windows {c:.2f}",
                      note="the bridge's curved face under the first 1.85 mm is filled down to the bed"))
    S.append(dict(id="CP", stl="CP_clamp_plate.stl", printer="p1s", material="PLA", test="Board fit", piece="Clamp plate",
                  variant="v2 a", value=None, mark="its shape (two copies)", copies=2, short="clamp plate"))
    S.append(dict(id="P", stl="P_pin_coupon_225_240.stl", printer="p1s", material="PLA", test="Board fit",
                  piece="Pin coupon", variant="pins 2.25, 2.30, 2.35, 2.40 mm", value=None,
                  mark="25, 30, 35, 40 on the tag under each pin", copies=1, short="pins 2.25-2.40"))
    S.append(dict(id="H", stl="H_screw_holes_m2_m3.stl", printer="p1s", material="PLA", test="Screw holes",
                  piece="Hole coupon", variant="M2 1.7, 1.8, 1.9 mm; M3 2.65, 2.75, 2.85 mm", value=None,
                  mark="sizes engraved beside the holes", copies=1, short="M2 / M3 holes"))
    S.append(dict(id="W", stl="W_cable_grooves_22_24_26.stl", printer="p1s", material="PLA", test="Cable groove",
                  piece="Groove coupon", variant="widths 2.2, 2.4, 2.6 mm", value=None,
                  mark="widths engraved above the grooves", copies=1, short="grooves 2.2-2.6"))
    return S


def cmd_pieces(a):
    import trimesh  # noqa: F401
    STL.mkdir(parents=True, exist_ok=True)
    if CONCEPT == "b":
        V2, B, K = concept_b()
    else:
        V2, A, K = concept_a()
    specs = piece_specs()
    built = {}
    for s in specs:
        t0 = time.time()
        i = s["id"]
        if CONCEPT == "b":
            if i == "R":
                tm = piece_rails(B, K)
            elif i.startswith("G"):
                tm, tmF = piece_groove_block(B, K, s["value"])
                (WORK / "xl").mkdir(parents=True, exist_ok=True)
                single(tmF, i).export(WORK / "xl" / s["stl"].replace(".stl", "_F.stl"))
            elif i.startswith("S"):
                tm = piece_seat(V2, B, K, s["value"])
            elif i == "N":
                tm = piece_neck_carrier(V2, B, K)
            elif i == "W":
                tm = piece_grooves(GROOVES_B, GROOVE_DEPTH_B)
            else:
                tm = piece_cover(B)
        elif i.startswith("K"):
            tm = piece_key(V2, A, K, s["value"], DETENTS.index(s["value"]) + 1)
        elif i.startswith("F"):
            tm, tmF = piece_channel(A, K, s["value"])
            (WORK / "xl").mkdir(parents=True, exist_ok=True)
            single(tmF, i).export(WORK / "xl" / s["stl"].replace(".stl", "_F.stl"))
        elif i.startswith("C") and i != "CP":
            tm = piece_carrier(V2, A, K, s["value"])
        elif i == "CP":
            tm = piece_clamp()
        elif i == "P":
            tm = piece_pins()
        elif i == "H":
            tm = piece_holes()
        else:
            tm = piece_grooves()
        single(tm, i).export(STL / s["stl"])
        s.update(size_mm=np.round(tm.extents, 2).tolist(), volume_mm3=round(float(tm.volume), 1))
        built[i] = s
        print(f"{i:4s} {s['piece']:18s} {s['variant']:42s} size {s['size_mm']} vol {s['volume_mm3']} "
              f"[{time.time() - t0:.1f} s]", flush=True)
    rid = next(k for k, v in RUNS.items() if v["pfx"] == PFX)
    rep = dict(run=rid, concept=CONCEPT, source=f"scripts/dip_fit_print_plates.py --run {rid}", params=RUNS[rid]["params"],
               owner="2026-10-19 14:25: 'is there a more organized manner you can present the prints (and documentation) "
                     "in? i would also like to maximize printer plate utilization, although i guess it'll be hard to "
                     "discern coupons'", pieces=[dict(s, stl=f"stl/{s['stl']}") for s in specs],
               when=time.strftime("%Y-%m-%d %H:%M"))
    old = RUN / "pieces.json"
    if old.exists():
        rep["plates"] = json.loads(old.read_text()).get("plates", {})
    old.write_text(json.dumps(rep, indent=1) + "\n")
    write_records(specs)
    print("pieces ->", rel(RUN), flush=True)


def write_records(specs):
    """records.csv: one row per check, keyed by piece ID (a pairing as K..+F.. or R+G..), result and notes for the owner."""
    if CONCEPT == "b":
        return write_records_rows(records_b())
    rows = []
    for fl in FLANKS:
        for chk in ("slides on by hand", "detent clicks", "stays when pulled along the channel", "stays when twisted"):
            rows.append((f"K25+F{hund(fl)}", "Snap fit, stage 1", chk))
    for h in DETENTS:
        if h != 0.25:
            for chk in ("slides on by hand", "detent clicks", "stays when pulled along the channel", "stays when twisted"):
                rows.append((f"K{hund(h)}+F*", "Snap fit, stage 2 (F* = the block chosen in stage 1)", chk))
    for c in CHIP_CLEAR:
        rows.append((f"C{hund(c)}", "Board fit", "all five chips drop into their windows, board flat without rocking"))
    for chk in ("J1 with the Qwiic plug mated sits in its slot without touching the walls",
                "the cable lies in its groove around the cut-off arm's corner",
                "the clamp plate's pins pass the board's holes into the window plate",
                "M2 x 4 countersunk screws form their thread in the clamp plate's 1.8 mm holes",
                "M2 x 5 screws through the ears hold in the shoulders' 1.8 mm holes"):
        rows.append(("C*+CP", "Board fit (C* = the carrier chosen above)", chk))
    for d in PIN_D:
        rows.append((f"P{hund(d - 2.0)}", "Board fit", f"the {d:.2f} mm pin enters the board's 2.40 mm hole without play"))
    for d in M2_HOLES:
        rows.append((f"H M2 {d:.1f} top", "Screw holes", "an M2 x 5 forms its thread without cracking and does not pull out"))
        rows.append((f"H M2 {d:.1f} front", "Screw holes", "an M2 x 5 forms its thread without cracking and does not pull out"))
    for d in M3_HOLES:
        rows.append((f"H M3 {d:.2f}", "Screw holes", "an M3 x 12 forms its thread without cracking and does not pull out"))
    for w in GROOVES:
        rows.append((f"W {w:.1f}", "Cable groove", "the Qwiic cable lies below the surface around the turn without forcing"))
    write_records_rows(rows)


def records_b():
    rows = []
    for fl in FLANKS_B:
        for chk in ("presses onto the rail strip by hand", "both rails snap into their grooves",
                    "stays under a 3 N pull away from the strip", "stays when pushed along the rails by hand"):
            rows.append((f"R+G{hund(fl)}", "Snap fit", chk))
    for d in SEAT_DEPTHS:
        for chk in ("the board slides into the pocket from its open end until it meets the proximal wall",
                    "the board lies flat in the pocket without play across it",
                    "a cover (V) and four M2 x 5 screws clamp the board without play"):
            rows.append((f"S{hund(d)}+V", "Board seat", chk))
    for chk in ("J1 with the Qwiic plug mated slides through the slot from the neck's underside without touching its walls",
                "the board reaches the proximal wall with J1 and the plug in the slot",
                "M2 x 5 screws through the cover and the board form their thread in the tongue's 1.8 mm holes",
                "the cable turns toward the back in the open groove over the plug and lies in the groove along the run",
                "a strip of Kapton tape over the open groove keeps the cable's first turn down"):
        rows.append(("N+V", "Board seat and cable", chk))
    for w in GROOVES_B:
        rows.append((f"W {w:.1f}", "Cable groove", "the Qwiic ribbon, massaged into a bundle, lies below the surface around "
                                                   "the turn without forcing"))
    return rows


def write_records_rows(rows):
    """records.csv with a row per check; the result and notes already entered for an (id, check) are kept."""
    path = RUN / "records.csv"
    kept = {}
    if path.exists():
        with open(path, newline="") as f:
            kept = {(r["id"], r["check"]): (r["result"], r["notes"]) for r in csv.DictReader(f)}
    with open(path, "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["id", "test", "check", "result", "notes"])
        for r in rows:
            w_.writerow(list(r) + list(kept.get((r[0], r[2]), ("", ""))))
    n = sum(1 for r in rows if any(kept.get((r[0], r[2]), ("", ""))))
    print(f"records.csv: {len(rows)} rows ({n} with results kept)", flush=True)


# ------------------------------------------------------------------------------------------------ plates
def layout(meshes, bed, gap=8.0, width=200.0):
    """Shelf packing in the given order: rows of at most `width` mm from the front of the bed, `gap` mm apart, the whole
    block centred on the bed. Returns each mesh's (dx, dy) from its print frame to the bed."""
    rows, row, x, h = [], [], 0.0, 0.0
    for i, tm in enumerate(meshes):
        lo, hi = tm.bounds[0][:2], tm.bounds[1][:2]
        w, d = hi - lo
        if row and x + w > width:
            rows.append((row, h))
            row, x, h = [], 0.0, 0.0
        row.append((i, x, lo, w, d))
        x += w + gap
        h = max(h, d)
    rows.append((row, h))
    pos, y = [None] * len(meshes), 0.0
    for row, h in rows:
        for i, x0, lo, w, d in row:
            pos[i] = np.array([x0 - lo[0], y - lo[1]])
        y += h + gap
    allb = np.vstack([np.vstack([meshes[i].bounds[0][:2] + pos[i], meshes[i].bounds[1][:2] + pos[i]])
                      for i in range(len(meshes))])
    shift = np.asarray(bed) / 2 - (allb.min(axis=0) + allb.max(axis=0)) / 2
    return [p + shift for p in pos]


def flat_presets(out):
    """The P1S system presets through bbl_preset (inherits chains and the printer's G-code templates), P1S_PROCESS applied
    to the process."""
    paths = {}
    for kind in ("machine", "process", "filament"):
        if kind == "process":
            d0 = bbl_preset(kind, P1S[kind])
            missing = [k for k in P1S_PROCESS if k not in d0 and k not in ("brim_type", "curr_bed_type")]
            if missing:
                raise SystemExit(f"process preset lacks {missing}")
        d = bbl_preset(kind, P1S[kind], P1S_PROCESS if kind == "process" else None)
        paths[kind] = out / f"{kind}.json"
        paths[kind].write_text(json.dumps(d, indent=1))
    return paths


def bbl_preset(kind, name, over=None, rename=None):
    """A Bambu Studio system preset with its `inherits` chain resolved, the G-code of every preset in the chain from its
    '<name> template <key>.json' files, and `over` applied (a renamed one is a user preset). Bambu keeps a printer's start,
    end, filament-change, layer-change and timelapse G-code in those files; without them the command line writes
    fdm_machine_common's generic start (no AMS load, the nozzle left at its 205 C purge temperature)."""
    import glob

    def load(n):
        d = json.loads((BBL / kind / f"{n}.json").read_text())
        chain = [n]
        if d.get("inherits"):
            base, bchain = load(d["inherits"])
            base.update(d)
            d, chain = base, bchain + chain
        return d, chain
    d, chain = load(name)
    for n in chain:
        for f in sorted((BBL / kind).glob(f"{glob.escape(n)} template *.json")):
            t = json.loads(f.read_text())
            d.update({k: v for k, v in t.items() if k not in ("name", "instantiation", "inherits", "from", "setting_id", "type")})
    d.update(over or {})
    d.update(inherits="", name=rename or name)
    if rename:
        d.update({"from": "User", "filament_settings_id": [rename]} if kind == "filament" else {"from": "User"})
    return d


def check_bambu_gcode(gcode, machine_tag, nozzle_c, bed_c):
    """The problems of a Bambu G-code: the printer's own start sequence (its ';===== machine: <tag>' header), a filament
    load from the AMS (M620 S<n>A and T<n>), the nozzle and the bed at the filament's temperatures."""
    import re
    out = []
    if not re.search(rf"^;===== machine: {re.escape(machine_tag)}", gcode, re.M):     # a line, not the settings dump
        out.append(f"not the {machine_tag} start G-code")
    if not re.search(r"^\s*M620 S\d+A", gcode, re.M) or not re.search(r"^\s*T\d\b", gcode, re.M):
        out.append("no filament load from the AMS")
    if not re.search(rf"^\s*M10[49] S{nozzle_c}\b", gcode, re.M):
        out.append(f"nozzle never set to {nozzle_c} C")
    if not re.search(rf"^\s*M1[49]0 S{bed_c}\b", gcode, re.M):
        out.append(f"bed never set to {bed_c} C")
    return out


def infill_by_object(gcode, objs, pad=1.0):
    """Filament (mm) of each feature type in each object of a sliced Bambu G-code, an extrusion going to the object whose
    bed bounds (objs[i]["bounds_mm"], grown by pad) contain its end point."""
    import re
    out = {o["id"]: {} for o in objs}
    x = y = None
    feat = None
    for ln in gcode.splitlines():
        if ln.startswith("; FEATURE:"):
            feat = ln.split(":", 1)[1].strip()
            continue
        if not ln.startswith(("G0", "G1", "G2", "G3")):
            continue
        mx, my, me = (re.search(rf"{c}(-?[\d.]+)", ln) for c in "XYE")
        x = float(mx.group(1)) if mx else x
        y = float(my.group(1)) if my else y
        if not me or feat is None or x is None or y is None or float(me.group(1)) <= 0:
            continue
        for o in objs:
            (x0, y0, _), (x1, y1, _) = o["bounds_mm"]
            if x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad:
                out[o["id"]][feat] = out[o["id"]].get(feat, 0.0) + float(me.group(1))
                break
    return out


def siraya_a1m_filament():
    """Bambu's Generic TPU @BBL A1M with every filament key that Siraya Tech's A1 mini project sets otherwise, then
    SIRAYA_KEEP. Returns the preset and the keys taken from Siraya."""
    base = bbl_preset("filament", A1M["filament"])
    with zipfile.ZipFile(SIRAYA_A1M_3MF) as z:
        sir = json.loads(z.read("Metadata/project_settings.config"))
    skip = {"name", "inherits", "from", "setting_id", "instantiation", "filament_settings_id", "filament_id", "version",
            "compatible_printers", "compatible_printers_condition", "filament_colour", "filament_ids"}
    taken = {k: sir[k] for k, v in base.items()
             if k not in skip and k in sir and sir[k] != v and type(sir[k]) is type(v)}
    d = dict(base)
    d.update(taken)
    d.update(SIRAYA_KEEP)
    d.update(inherits="", name=SIRAYA_NAME, filament_settings_id=[SIRAYA_NAME], filament_colour=["#FFFFFF"])
    d["from"] = "User"
    return d, taken


def cmd_a1m(a):
    """The 85A plate on the Bambu Lab A1 mini: the channel blocks placed by `layout`, on filament A1M_SLOT (Siraya Tech TPU
    85A), sliced by Bambu Studio's command line into one project with its G-code; checks that only that filament prints, at
    240 C on a 35 C textured PEI plate, without a prime tower."""
    import re
    import trimesh
    rep = json.loads((RUN / "pieces.json").read_text())
    work = WORK / "a1m"
    placed = work / "placed"
    shutil.rmtree(work, ignore_errors=True)
    placed.mkdir(parents=True)
    paths = {}
    for kind, d in (("machine", bbl_preset("machine", A1M["machine"])),
                    ("process", bbl_preset("process", A1M["process"], A1M_PROCESS)),
                    ("filament", siraya_a1m_filament()[0])):
        paths[kind] = work / f"{kind}.json"
        paths[kind].write_text(json.dumps(d, indent=1))
    items = [(s, trimesh.load(RUN / s["stl"], force="mesh")) for s in rep["pieces"] if s["material"] == "85A"]
    pos = layout([t for _, t in items], A1M_BED, gap=12.0, width=150.0)
    files, objs = [], []
    for (s, tm), p_ in zip(items, pos):
        tm.apply_translation((p_[0], p_[1], 0.0))
        f = placed / Path(s["stl"]).name
        tm.export(f)
        files.append(str(f))
        objs.append(dict(id=s["id"], name=f.stem, bounds_mm=np.round(tm.bounds, 2).tolist()))
    proj = f"{PFX}_a1mini_85a.3mf"
    cmd = BAMBU + ["--debug", "1", "--arrange", "0", "--orient", "0",
                   "--load-settings", f"{paths['machine']};{paths['process']}", "--load-filaments", str(paths["filament"]),
                   "--slice", "0", "--export-3mf", proj, "--outputdir", str(work)] + files
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    (work / "cli.log").write_text(r.stdout + r.stderr)
    res = json.loads((work / "result.json").read_text())
    if res.get("return_code") != 0:
        raise SystemExit(f"Bambu Studio: {res.get('error_string')} (log {rel(work / 'cli.log')})")
    shutil.copy(work / proj, RUN / proj)
    with zipfile.ZipFile(RUN / proj) as z:
        info = z.read("Metadata/slice_info.config").decode()
        gcode = z.read("Metadata/plate_1.gcode").decode(errors="replace")
        msets = z.read("Metadata/model_settings.config").decode()
    pred = float(re.search(r'key="prediction" value="([\d.]+)"', info).group(1))
    weight = float(re.search(r'key="weight" value="([\d.]+)"', info).group(1))
    used = re.findall(r'<filament id="(\d+)"[^>]*type="([^"]+)"[^>]*used_g="([\d.]+)"', info)
    layers = int(re.search(r"; total layer number: (\d+)", gcode).group(1))
    bed_c = [int(float(v)) for v in re.findall(r"^M190 S([\d.]+)", gcode, re.M)]
    noz_c = [int(float(v)) for v in re.findall(r"^M10[49] S([\d.]+)", gcode, re.M)]
    bed_type = re.search(r"; curr_bed_type = (.+)", gcode).group(1).strip()
    names = sorted(re.findall(r'<object id="\d+">\s*<metadata key="name" value="([^"]+)"', msets))
    ext = sorted(set(re.findall(r'<metadata key="extruder" value="(\d+)"', msets)))
    problems = []
    if names != sorted(f"{o['name']}.stl" for o in objs):
        problems.append(f"objects {names}")
    if [u[0] for u in used] != ["1"] or used[0][1] != "TPU" or ext != ["1"]:
        problems.append(f"filaments used {used}, object extruders {ext}")
    if bed_type != A1M_PROCESS["curr_bed_type"]:
        problems.append(f"plate {bed_type}")
    fil = json.loads(paths["filament"].read_text())
    problems += check_bambu_gcode(gcode, "A1 mini", int(fil["nozzle_temperature"][0]), int(fil["textured_plate_temp"][0]))
    if re.search(r"; (WIPE_TOWER|PRIME_TOWER)_START", gcode) or "; FEATURE: Prime tower" in gcode:
        problems.append("prime tower in the G-code")
    feats = infill_by_object(gcode, objs)
    for o in objs:                                          # the FE recipe's gyroid in every block, as on the XL
        f = feats[o["id"]]
        sp, so = f.get("Sparse infill", 0.0), f.get("Internal solid infill", 0.0)
        o["sparse_infill_mm"], o["internal_solid_infill_mm"] = round(sp, 1), round(so, 1)
        if sp <= 0.0:
            problems.append(f"{o['id']} has no sparse infill ({so:.0f} mm of internal solid infill)")
    if problems:
        raise SystemExit("A1 mini project: " + "; ".join(problems))
    warn = res["sliced_plates"][0].get("warning_message")
    if warn:
        print("Bambu Studio warning:", warn, flush=True)
    fig = RUN / f"{PFX}_a1mini_85a_map.png"
    plate_map([(next(x for x in rep["pieces"] if x["id"] == o["id"]), trimesh.load(placed / f"{o['name']}.stl"))
               for o in objs], A1M_BED, f"Bambu Lab A1 mini, Siraya Tech 85A (AMS lite slot {A1M_SLOT}): {len(objs)} pieces, "
               f"{fmt_time(pred)}, {weight:.1f} g", fig)
    rep.setdefault("plates", {})["a1m"] = dict(
        project=rel(RUN / proj), map=rel(fig), printer="Bambu Lab A1 mini 0.4 nozzle", filament=SIRAYA_NAME,
        slot=A1M_SLOT, presets=A1M, overrides=A1M_PROCESS, time_s=pred,
        filament_source=rel(SIRAYA_A1M_3MF), filament_from_siraya=sorted(siraya_a1m_filament()[1]),
        filament_kept=SIRAYA_KEEP, filament_settings={k: fil[k] for k in (
            "nozzle_temperature", "textured_plate_temp", "filament_flow_ratio", "filament_retraction_length",
            "filament_retraction_speed", "close_fan_the_first_x_layers", "filament_max_volumetric_speed")},
        time=fmt_time(pred), filament_g=weight, layers=layers, bed_type=bed_type,
        bed_c=int(fil["textured_plate_temp"][0]), nozzle_c=int(fil["nozzle_temperature"][0]),
        objects=objs, slicer="Bambu Studio 2.5.0.66 command line, arrange off (positions from layout)",
        when=time.strftime("%Y-%m-%d %H:%M"))
    (RUN / "pieces.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(f"A1 mini: {len(objs)} objects, TPU for slot {A1M_SLOT}, {fmt_time(pred)}, {weight:.2f} g, {layers} layers, bed "
          f"{bed_type} {bed_c[:1]} C, nozzle {sorted(set(noz_c))} [{time.time() - t0:.0f} s] -> {rel(RUN / proj)}", flush=True)


def cmd_bambu(a):
    """The P1S plate: the PLA pieces placed by `layout`, sliced by Bambu Studio's command line into one project with its
    G-code; the plate map and the slicer's time and filament into pieces.json."""
    import trimesh
    rep = json.loads((RUN / "pieces.json").read_text())
    work = WORK / "bambu"
    placed = work / "placed"
    shutil.rmtree(work, ignore_errors=True)
    placed.mkdir(parents=True)
    pr = flat_presets(work)
    items = []
    for s in rep["pieces"]:
        if s["printer"] != "p1s":
            continue
        tm = trimesh.load(RUN / s["stl"], force="mesh")
        for c in range(s.get("copies", 1)):
            name = Path(s["stl"]).stem + (f"_{c + 1}" if s.get("copies", 1) > 1 else "")
            items.append((s, name, tm.copy()))
    pos = layout([t for _, _, t in items], P1S_BED, gap=12.0)
    files, objs = [], []
    for (s, name, tm), p in zip(items, pos):
        tm.apply_translation((p[0], p[1], 0.0))
        f = placed / f"{name}.stl"
        tm.export(f)
        files.append(str(f))
        objs.append(dict(id=s["id"], name=name, bounds_mm=np.round(tm.bounds, 2).tolist()))
    proj = f"{PFX}_p1s_pla.3mf"
    cmd = BAMBU + ["--debug", "1", "--arrange", "0", "--orient", "0",
                   "--load-settings", f"{pr['machine']};{pr['process']}", "--load-filaments", str(pr["filament"]),
                   "--slice", "0", "--export-3mf", proj, "--outputdir", str(work)] + files
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    (work / "cli.log").write_text(r.stdout + r.stderr)
    res = json.loads((work / "result.json").read_text())
    if res.get("return_code") != 0:
        raise SystemExit(f"Bambu Studio: {res.get('error_string')} (log {rel(work / 'cli.log')})")
    shutil.copy(work / proj, RUN / proj)
    with zipfile.ZipFile(RUN / proj) as z:
        info = z.read("Metadata/slice_info.config").decode()
        gcode = z.read("Metadata/plate_1.gcode").decode(errors="replace")
        msets = z.read("Metadata/model_settings.config").decode()
    import re
    pred = float(re.search(r'key="prediction" value="([\d.]+)"', info).group(1))
    weight = float(re.search(r'key="weight" value="([\d.]+)"', info).group(1))
    layers = int(re.search(r"; total layer number: (\d+)", gcode).group(1))
    bed_c = [int(float(v)) for v in re.findall(r"^M190 S([\d.]+)", gcode, re.M)]
    bed_type = re.search(r"; curr_bed_type = (.+)", gcode).group(1).strip()
    if bed_type != P1S_PROCESS["curr_bed_type"] or not bed_c or bed_c[0] < 50:
        raise SystemExit(f"plate {bed_type!r}, bed {bed_c[:2]} C: the project is not set to the owner's textured PEI plate")
    bad = check_bambu_gcode(gcode, "P1S", 220, 55)
    if bad:
        raise SystemExit("P1S project: " + "; ".join(bad))
    warn = res["sliced_plates"][0].get("warning_message")
    if warn:
        print("Bambu Studio warning:", warn, flush=True)
    names = sorted(re.findall(r'<object id="\d+">\s*<metadata key="name" value="([^"]+)"', msets))
    if names != sorted(f"{o['name']}.stl" for o in objs):
        raise SystemExit(f"objects in the project {names} differ from the pieces")
    fig = RUN / f"{PFX}_p1s_pla_map.png"
    plate_map([(next(s for s in rep["pieces"] if s["id"] == o["id"]), trimesh.load(placed / f"{o['name']}.stl"))
               for o in objs], P1S_BED, f"Bambu Lab P1S, PLA: {len(objs)} pieces, {fmt_time(pred)}, {weight:.1f} g", fig)
    rep.setdefault("plates", {})["p1s"] = dict(
        project=rel(RUN / proj), map=rel(fig), printer="Bambu Lab P1S 0.4 nozzle", filament="Bambu PLA Basic",
        presets=P1S, overrides=P1S_PROCESS, time_s=pred, time=fmt_time(pred), filament_g=weight, layers=layers,
        bed_type=bed_type, bed_c=bed_c[0],
        objects=objs, slicer="Bambu Studio 2.5.0.66 command line, arrange off (positions from layout)",
        when=time.strftime("%Y-%m-%d %H:%M"))
    (RUN / "pieces.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(f"P1S: {len(objs)} objects, {fmt_time(pred)}, {weight:.2f} g, {layers} layers [{time.time() - t0:.0f} s] -> "
          f"{rel(RUN / proj)}", flush=True)


def cmd_xl(a):
    """The XL plate: the 85A channel blocks on tool 2 (v17 template, 0.2 mm layers, 30 % gyroid), placed, sliced and
    checked by dip_link_print_project.build_project."""
    import dip_link_print_project as PP
    import trimesh
    rep = json.loads((RUN / "pieces.json").read_text())
    tpu = [s for s in rep["pieces"] if s["material"] == "85A"]
    PP.PFX, PP.WORK = PFX, WORK / "xl" / "slice"
    PP.COUPON_FLANKS = FLANKS if CONCEPT == "a" else FLANKS_B
    PP.bed_figure = lambda *args, **kw: None            # the plate map below replaces its notch-labelled bed figure
    parts = {}
    for s in tpu:
        k = f"channel_block_{s['value']:.2f}".replace(".", "p")    # build_project turns a channel block from F
        parts[k] = str(WORK / "xl" / Path(s["stl"]).name.replace(".stl", "_F.stl"))
        PP.SETTINGS[k] = dict(PP.SETTINGS["tip"])
        PP.LABEL[k] = f"{s['id']}: 85A {'channel' if CONCEPT == 'a' else 'groove'} block, {s['value']:.2f} mm per flank"
    name = f"{PFX}_xl_85a_t2"
    r = PP.build_project(parts, RUN, name, source="scripts/dip_fit_print_plates.py xl")
    items = []
    for s, o in zip(tpu, r["objects"]):
        tm = trimesh.load(RUN / s["stl"], force="mesh")
        t = np.zeros(3)
        t[:len(o["bed_translation_mm"])] = o["bed_translation_mm"]
        tm.apply_translation(t)
        items.append((s, tm, o))
    # build_project's translation applies to its own print-frame mesh (footprint centred, lowest point on the bed), as
    # ours is; check that the footprints agree before drawing
    for s, tm, o in items:
        if not np.allclose(np.asarray(o["size_mm"]), np.round(tm.extents, 2), atol=0.02):
            raise SystemExit(f"{s['id']}: size {o['size_mm']} in the project, {tm.extents} here")
    sl = r.get("slice", {})
    fig = RUN / f"{name}_map.png"
    plate_map([(s, tm) for s, tm, _ in items], XL_BED,
              f"Prusa XL 5T, tool 2, Siraya Tech 85A: {len(items)} pieces, {sl.get('print_time')}, "
              f"{sl.get('filament_g')} g", fig)
    jr = json.loads((RUN / f"{name}.json").read_text())
    jr["figure"] = rel(fig)
    (RUN / f"{name}.json").write_text(json.dumps(jr, indent=1, default=float))
    rep.setdefault("plates", {})["xl"] = dict(
        project=r["project"], report=rel(RUN / f"{name}.json"), map=rel(fig), printer="Prusa XL 5T, tool 2",
        filament="Siraya Tech TPU 85A", template=r["template"], settings=PP.SETTINGS["tip"], time=sl.get("print_time"),
        filament_g=sl.get("filament_g"), checks_pass=sl.get("checks_pass"),
        objects=[dict(id=s["id"], name=o["key"], bed_translation_mm=o["bed_translation_mm"]) for s, _, o in items],
        when=time.strftime("%Y-%m-%d %H:%M"))
    (RUN / "pieces.json").write_text(json.dumps(rep, indent=1) + "\n")


def fmt_time(s):
    s = int(round(s))
    return f"{s // 3600} h {s % 3600 // 60} min" if s >= 3600 else f"{s // 60} min"


def plate_map(items, bed, title, out):
    """Top view of the plate: each piece's faces shaded by height, its ID above it and its variant below; the view
    zoomed to the pieces with the bed's outline and coordinates in mm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    allb = np.vstack([tm.bounds[:, :2] for _, tm in items])
    lo, hi = allb.min(axis=0) - (8, 14), allb.max(axis=0) + (8, 12)
    w, h = hi - lo
    fig, ax = plt.subplots(figsize=(11.0, 11.0 * h / w + 0.9))
    zmax = max(float(tm.bounds[1][2]) for _, tm in items)
    for s, tm in items:
        V, F = np.asarray(tm.vertices), np.asarray(tm.faces)
        up = tm.face_normals[:, 2] > 0.05
        F = F[up]
        zc = V[F].mean(axis=1)[:, 2]
        order = np.argsort(zc)
        col = plt.cm.Blues(0.25 + 0.6 * zc[order] / zmax) if s["material"] == "PLA" else plt.cm.Oranges(
            0.25 + 0.6 * zc[order] / zmax)
        ax.add_collection(PolyCollection(V[F[order]][:, :, :2], facecolors=col, edgecolors="none"))
        (x0, y0), (x1, y1) = tm.bounds[0][:2], tm.bounds[1][:2]
        ax.text((x0 + x1) / 2, y1 + 1.5, s["id"], ha="center", va="bottom", fontsize=15, fontweight="bold")
        ax.text((x0 + x1) / 2, y0 - 1.5, s["short"], ha="center", va="top", fontsize=10.5)
    ax.plot([0, bed[0], bed[0], 0, 0], [0, 0, bed[1], bed[1], 0], color="k", lw=1)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("Bed x (mm), front edge at y = 0", fontsize=12)
    ax.set_ylabel("Bed y (mm)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("map ->", rel(out), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", choices=["pieces", "bambu", "a1m", "xl"])
    ap.add_argument("--run", default="20261019", choices=list(RUNS), help="the run: 20261019 concept a, 20261019b concept b")
    a = ap.parse_args()
    set_run(a.run)
    RUN.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    for st in a.stages:
        t0 = time.time()
        print(f"== {st}", flush=True)
        dict(pieces=cmd_pieces, bambu=cmd_bambu, a1m=cmd_a1m, xl=cmd_xl)[st](a)
        print(f"== {st} [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
