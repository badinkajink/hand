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
W = cable-groove coupon. Keys and carriers have the ID on a 1.4 mm tag at the bed, the channel blocks engraved in the
face that lies inside the tip, the coupons engraved beside each variant.

    unset VIRTUAL_ENV
    ~/.claude/bin/resguard.sh run --mem 5G --cpu 300 -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python scripts/dip_fit_print_plates.py pieces bambu xl

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
P1S_BED, XL_BED = (256.0, 256.0), (360.0, 360.0)

DETENTS = (0.15, 0.25, 0.35, 0.45)                    # PLA key detent height, mm (v2/coupons: 0.15-0.35)
FLANKS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)    # 85A channel interference per flank, mm (v2/coupons: 0-0.20)
CHIP_CLEAR = (0.15, 0.20, 0.30)                       # chip-window clearance of the board carrier, mm
PIN_D = (2.25, 2.30, 2.35, 2.40)                      # pins of dip_link_board_mount.pin_coupon, from its chamfered end
M2_HOLES = (1.7, 1.8, 1.9)                            # self-forming M2 holes, mm (owner, 2026-10-18: 1.8)
M3_HOLES = (2.65, 2.75, 2.85)                         # self-forming M3 holes, mm (owner, 2026-10-18: 2.75)
GROOVES = (2.2, 2.4, 2.6)                             # cable-groove width, mm (v2 a: 2.4 for a 2.0 mm bundle)
GROOVE_DEPTH, GROOVE_R = 2.4, 2.0                     # groove depth; centre-line radius of its turn (v2 a cable_r)
TAG_T, ENGRAVE, ENGRAVE_TPU = 1.4, 0.6, 0.8           # tag thickness, engraving depth in PLA and in 85A, mm


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


def piece_grooves():
    """Three open cable grooves, GROOVE_DEPTH deep, widths 2.2, 2.4 and 2.6 mm, each turning 90 degrees about a
    GROOVE_R centre line as the routed cable turns over the plug and at the slotted arm; the widths engraved above."""
    m = M()
    blk = box((-20.0, -11.0, 0.0), (21.0, 11.0, 4.4))
    man = blk
    for x0, w in zip((-15.0, -2.0, 11.0), GROOVES):
        man = man - m.Manifold.extrude(groove_cs(x0, w), GROOVE_DEPTH + 0.1).translate([0.0, 0.0, 4.4 - GROOVE_DEPTH])
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


# ------------------------------------------------------------------------------------------------ piece list
def piece_specs():
    """Every piece of the run: ID, test, variant, printer and what the owner checks on it."""
    S = []
    for k, h in enumerate(DETENTS):
        S.append(dict(id=f"K{hund(h)}", stl=f"K{hund(h)}_key_detent{hund(h)}.stl", printer="p1s", material="PLA",
                      test="Snap fit", piece="PLA key", variant=f"detent {h:.2f} mm", value=h, mark=f"tag K{hund(h)}, "
                      f"{k + 1} notch{'es' if k else ''} on the flange", copies=1, short=f"detent {h:.2f}"))
    for fl in FLANKS:
        S.append(dict(id=f"F{hund(fl)}", stl=f"F{hund(fl)}_channel_flank{hund(fl)}_85a.stl", printer="xl", material="85A",
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
    V2, A, K = concept_a()
    specs = piece_specs()
    built = {}
    for s in specs:
        t0 = time.time()
        i = s["id"]
        if i.startswith("K"):
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
    rep = dict(run=DATE, source="scripts/dip_fit_print_plates.py", params=
               "docs/experiments/20261018-dip_link_board_mount/params/20261018-v2_concept_a.json",
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
    """records.csv: one row per check, keyed by piece ID (a pairing as K..+F..), result and notes for the owner."""
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
    with open(RUN / "records.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["id", "test", "check", "result", "notes"])
        for r in rows:
            w_.writerow(list(r) + ["", ""])
    print(f"records.csv: {len(rows)} rows", flush=True)


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
    """The P1S system presets with their `inherits` chains resolved, P1S_PROCESS applied to the process."""
    def load(kind, name):
        d = json.loads((BBL / kind / f"{name}.json").read_text())
        if d.get("inherits"):
            base = load(kind, d["inherits"])
            base.update(d)
            d = base
        return d
    paths = {}
    for kind in ("machine", "process", "filament"):
        d = load(kind, P1S[kind])
        if kind == "process":
            missing = [k for k in P1S_PROCESS if k not in d and k not in ("brim_type", "curr_bed_type")]
            if missing:
                raise SystemExit(f"process preset lacks {missing}")
            d.update(P1S_PROCESS)
        d.update(inherits="", name=P1S[kind])
        paths[kind] = out / f"{kind}.json"
        paths[kind].write_text(json.dumps(d, indent=1))
    return paths


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
    tpu = [s for s in rep["pieces"] if s["printer"] == "xl"]
    PP.PFX, PP.WORK = PFX, WORK / "xl" / "slice"
    PP.COUPON_FLANKS = FLANKS
    PP.bed_figure = lambda *args, **kw: None            # the plate map below replaces its notch-labelled bed figure
    parts = {}
    for s in tpu:
        k = f"channel_block_{s['value']:.2f}".replace(".", "p")
        parts[k] = str(WORK / "xl" / Path(s["stl"]).name.replace(".stl", "_F.stl"))
        PP.SETTINGS[k] = dict(PP.SETTINGS["tip"])
        PP.LABEL[k] = f"{s['id']}: 85A channel block, {s['value']:.2f} mm per flank"
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
    ap.add_argument("stages", nargs="+", choices=["pieces", "bambu", "xl"])
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    for st in a.stages:
        t0 = time.time()
        print(f"== {st}", flush=True)
        dict(pieces=cmd_pieces, bambu=cmd_bambu, xl=cmd_xl)[st](a)
        print(f"== {st} [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
