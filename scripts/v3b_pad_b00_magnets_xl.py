#!/usr/bin/env python3
"""Concept b's 11/22 pad at groove fit B00 (0.00 mm per flank) with its 3 x 2 mm magnets and their pauses, 85A, on the Prusa
XL's tool 2: the five magnets of the v2 tip (--magnets five) or the four of the 10-18 print (--magnets four).

Owner, 2026-10-20 16:20: "can you also prepare a print for the b00 fingertip with the pauses back in? i want to try to get
sensing today"; 18:10: "can you also re-prepare the 4 magnets configuration?". B00 of the fit-size plate
(scripts/v2_tip_fit_sizes_xl.py, printed without pauses) was the groove fit the owner chose on the rails ("the concept B
rails slot great with the B00 sizing"). The pad is make_pad of the v3 parameters at GROOVE_FIT 0.00: five magnets from the
v2 tip (pad set v2_five, key ell11_5, five_y2p5_z9p8_s0p5) or four from the 2.9 mm-seat print of 2026-10-18 (pad set
seat2p9, key ell11, four_d3x2: y +-3 mm, 3.5 mm either side of the contact height), both 7.5 mm under the apex. Its ID is
engraved 0.4 mm deep in the mount face as on the fit-size plate (B00 for the five, B4 for the four), in the tip's print frame,
so that the tip's sheet's pockets and pauses apply unchanged (metric_magnet_prints.pauses). Object settings are those of the
pipeline's pads (2 perimeters, 30 % gyroid, four top and bottom layers, 0.1 mm layers, tool 2); the global settings are those
of the owner's good print of 2026-10-18, verbatim. R4 (dip_link_concept_b's rule: 2.0 mm from the rail grooves and the
screw-head notches to a magnet pocket) is measured on the pad. Sliced with PrusaSlicer and checked: one M601 per pause height,
each before the bridge layer over its pockets, no bead inside a pocket, tool 2 only at 240 C on a 35 C bed.
--pocket-walls 1 (owner, 2026-10-20 18:35: "is it possible to run just one wall around the magnets instead of 2?") adds
metric_magnet_prints.onewall_modifier, a modifier of one perimeter over the pocket layers that leaves two loops at the outer
surface and one around each pocket, counted in the G-code (IDs 5W1 and 4W1, files *_pocketwalls1_*).
--pin-hole (owner, 2026-10-20 20:36, choosing the caps over the tip end) cuts the 2.0 mm hole, 3.0 mm deep, that takes the
caps' pin (scripts/dip_b_tip_cap.py: x_F 1.40 on the centre line, from the tip end, the print's top), IDs B00P and B4P, files
*_pin_*.
Output: docs/experiments/20261018-dip_link_board_mount/v3/b/print/20261020-dip_link_v3b_print_pad_ell11_{5,4}_B00_85a_t2_g30.3mf.
--groove-fit, --hole-d, --pid, --copies, --out, --date (owner, 2026-10-22: "for the new fingers with the hole in the top, we
should really undersize this hole too ... undersize the slot and oversize the rail just a little bit for an even tighter fit
... prepare a tpu print on the prusa for three new fingertips (with pauses)"): the grooves that much narrower than the nominal
rail per flank, the pin hole that wide, the ID, and the pad as that many instances 30 mm apart along x on one plate. The one-pad
project is sliced and checked pocket by pocket as before (in the work folder); the plate of copies takes the same pauses, and
its G-code is checked for the pauses, one label per copy, tool 2 only and the filament against the one-pad slice
(n times it, less up to 0.3 g a copy for the purge line and skirt they share).
--magnets four_compact (owner, 2026-10-26: "the same 3 fingertips, but with only 4 magnet pouches"; "Keep the four compact and
symmetric about the centre chip") cuts the pad from the tip of scripts/dip_b_four_compact_tip.py: four magnets in a 5.0 mm
square centred on U1, at the five's depth and seat (pad set v4_four). --poles checkerboard writes each pause's poles slot by
slot: +1 (north toward the palmar face) where y_F < 0 in a proximal row and y_F > 0 in a distal row, -1 elsewhere, +1 on the
centre line; the corners of the five-magnet checkerboard (fingertip_contact_estimation_study.POLE_PATTERNS) follow the same
rule. In the print the slots of a row lie along the bed's y axis, y_F < 0 toward the back of the plate.

    unset VIRTUAL_ENV
    ~/.claude/bin/resguard.sh run --mem 4G --cpu 200 -- taskset -c 14,15 env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 \\
        ~/miniconda3/envs/femtip/bin/python -u scripts/v3b_pad_b00_magnets_xl.py --magnets four
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import scipy.spatial  # noqa: F401  (the environment's libstdc++ before manifold3d)
import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ["DLV2_PARAMS_B"] = "docs/experiments/20261018-dip_link_board_mount/params/20261020-v3_concept_b.json"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import dip_link_print_project as PP  # noqa: E402
import dip_link_v2 as V2  # noqa: E402
import metric_magnet_prints as MP  # noqa: E402
import sim2real_fingertip_prints as SP  # noqa: E402
import v2_tip_fit_sizes_xl as FS  # noqa: E402

FIT = 0.00
VARIANTS = {   # --magnets: pad set, key in it, engraved ID, file key, the owner's words
    "five": ("v2_five", "ell11_5", "B00", "ell11_5",
             "2026-10-20 16:20: 'prepare a print for the b00 fingertip with the pauses back in? i want to try to get sensing today'"),
    "four": ("seat2p9", "ell11", "B4", "ell11_4",
             "2026-10-20 18:10: 'can you also re-prepare the 4 magnets configuration?'"),
    "four_compact": ("v4_four", "ell11_4c", "B4C", "ell11_4c",
                     "2026-10-26: 'can you prepare a 3mf for the prusa xl 5t with the same 3 fingertips, but with only 4 magnet "
                     "pouches instead of [5]?'; 'Keep the four compact and symmetric about the centre chip.'"),
    "five_snap": ("v4_snap", "ell11_5s", "05S", "ell11_5s",
                  "2026-10-26: 'i want to try having the pockets have a little more bite to them ... have the pocket come up "
                  "past the diameter of the disc'"),
    "four_snap": ("v4_snap", "ell11_4s", "05S4", "ell11_4s",
                  "2026-10-26: 'i want to try having the pockets have a little more bite to them ... have the pocket come up "
                  "past the diameter of the disc'"),
}
OUT = V2.B_DIR / "print"
SETTINGS = dict(perimeters=2, fill_density="30%", fill_pattern="gyroid", top_solid_layers=4, bottom_solid_layers=4)


def r4(B, tm, fit=FIT):
    """dip_link_concept_b's R4 on a pad in F: least distance from its magnet pockets (the small void bodies) to the rail
    grooves at `fit` and to the screw-head notches, mm."""
    voids = [b for b in tm.split(only_watertight=False) if abs(b.volume) < 100]
    Vv = np.vstack([b.vertices for b in voids])
    lo = np.array([B.X_SLOT - B.RAIL_H - B.GROOVE_FLOOR, B.RAIL_Y - B.RAIL_TOP / 2 + fit, B.RAIL_Z[0] - 0.2])
    hi = np.array([B.X_SLOT, B.RAIL_Y + B.RAIL_TOP / 2 - fit, B.RAIL_Z[1] + 0.2])
    dist = lambda P: np.linalg.norm(np.maximum(np.maximum(lo - P, P - hi), 0.0), axis=1)   # noqa: E731
    dg = float(min(dist(Vv).min(), dist(Vv * [1, -1, 1]).min()))
    dn = min(float(np.min(np.maximum(np.hypot(Vv[:, 1] - y, Vv[:, 2] - z) - B.HEAD_NOTCH_D / 2, 0.0))) for y, z in B.screw_yz())
    return dict(groove_to_pocket_mm=round(dg, 2), notch_to_pocket_mm=round(dn, 2), need_mm=2.0, ok=bool(min(dg, dn) >= 2.0),
                pocket_dorsal_x_F=round(float(Vv[:, 0].max()), 2), tpu_behind_pockets_mm=round(B.X_SLOT - float(Vv[:, 0].max()), 2))


def copies_project(src, dst, n, cfg, work, summ1, prow, dx=30.0):
    """The one-pad project src as n instances of its object dx apart along x (PrusaSlicer slices an object once and prints
    every instance with the same paths), sliced; checked against the one-pad slice summ1: the same pauses, one label per
    instance, tool 2 only, n times the filament."""
    import re
    with zipfile.ZipFile(src) as z:
        files = {k: z.read(k) for k in z.namelist()}
    model = files["3D/3dmodel.model"].decode()
    item = re.search(r'<item objectid="1" transform="([^"]+)"([^/]*)/>', model)
    T = item.group(1).split()
    x0 = float(T[9])
    items = []
    for k in range(n):
        Tk = T[:]
        Tk[9] = f"{x0 + (k - (n - 1) / 2) * dx:.4f}"
        items.append(f'<item objectid="1" transform="{" ".join(Tk)}"{item.group(2)}/>')
    files["3D/3dmodel.model"] = model.replace(item.group(0), "\n  ".join(items), 1).encode()
    mc = files["Metadata/Slic3r_PE_model.config"].decode()
    assert 'instances_count="1"' in mc
    files["Metadata/Slic3r_PE_model.config"] = mc.replace('instances_count="1"', f'instances_count="{n}"', 1).encode()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for k, b in files.items():
            z.writestr(k, b)
    gc, _ = PP.slice_project(dst, cfg, work)
    s = PP.gcode_summary(gc)
    pz = sorted({round(float(m), 2) for m in re.findall(r"^;Z:([\d.]+)\s*\n(?:.*\n){0,40}?M601", Path(gc).read_text(), re.M)})
    problems = []
    if s["m601"] != len(prow) or pz != [p["print_z"] for p in prow]:
        problems.append(f"copies: pauses at {pz} ({s['m601']} M601), asked {[p['print_z'] for p in prow]}")
    if len(s["labels"]) != n or s.get("tool_changes"):
        problems.append(f"copies: objects {list(s['labels'])}, tool changes {s.get('tool_changes')}")
    f1, fn = summ1.get("filament_g", 0.0), s.get("filament_g", 0.0)
    if not (n * f1 - (n - 1) * 0.3 <= fn <= n * f1 + 0.2):     # one purge line and one skirt for all: up to 0.3 g less a copy
        problems.append(f"copies: filament {fn} g against {n} x {f1} g")
    return dict(project=PP.rel(dst), n=n, dx_mm=dx, gcode=PP.rel(gc), labels=list(s["labels"]), pauses_found_z=pz,
                summary={k: v for k, v in s.items() if k != "labels"}, problems=problems)


def main():
    import trimesh
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--magnets", choices=list(VARIANTS), default="five")
    ap.add_argument("--pocket-walls", type=int, choices=(1, 2), default=2)
    ap.add_argument("--pin-hole", action="store_true", help="the hole for the tip caps' pin (scripts/dip_b_tip_cap.py)")
    ap.add_argument("--raised-seats", action="store_true",
                    help="for the solid-wall covers I05-I15: proximal notches seat_raise deeper, the pin hole's mouth chamfered "
                         "for TP2/TW2's fillet (pad B00R; needs --pin-hole)")
    ap.add_argument("--groove-fit", type=float, default=FIT, help="grooves this much narrower than the nominal rail per flank, mm")
    ap.add_argument("--hole-d", type=float, default=None, help="the pin hole's diameter, mm (default dip_b_tip_cap's CAP)")
    ap.add_argument("--pid", default=None, help="the engraved ID (default: the variant's)")
    ap.add_argument("--copies", type=int, default=1, help="instances of the pad on the plate, 30 mm apart along x")
    ap.add_argument("--out", default=None, help="output folder (default: v3/b/print)")
    ap.add_argument("--date", default=None, help="date prefix of the files (default: the variant's)")
    ap.add_argument("--id-x", type=float, default=None,
                    help="the engraved ID's centre x in the centred print frame, mm (default v2_tip_fit_sizes_xl.X_TEXT, 2.5, where "
                         "the first character of a four-letter ID runs off the curved end of the face; 0.5 keeps five on it)")
    ap.add_argument("--id-cap", type=float, default=None,
                    help="the engraved ID's cap height, mm (default v2_tip_fit_sizes_xl.CAP, 2.5; 2.2 leaves a five-letter ID "
                         "0.9 mm from both ends of the face)")
    ap.add_argument("--poles", choices=("north", "checkerboard"), default="north",
                    help="the poles each pause asks for: all north toward the palmar face, or the checkerboard slot by slot")
    a = ap.parse_args()
    new = a.groove_fit != FIT or a.hole_d is not None or a.copies > 1 or a.pid is not None
    if a.raised_seats and not a.pin_hole:
        raise SystemExit("--raised-seats builds pad B00R, which has the pin hole: add --pin-hole")
    pset, key, pid, fkey, owner = VARIANTS[a.magnets]
    w1 = a.pocket_walls == 1
    if w1:
        pid = {"five": "5W1", "four": "4W1"}.get(a.magnets, pid + "W1")
        owner += "; 18:35: 'is it possible to run just one wall around the magnets instead of 2?'"
    if a.pin_hole:
        pid = {"five": "B00P", "four": "B4P"}.get(a.magnets, pid + "P") if not w1 else pid + "P"
        owner += "; 20:36: 'sure we can go with your recommendation' (the caps over the tip end, with a pin into the pad)"
    if a.raised_seats:
        pid = {"five": "B00R", "four": "B4R"}.get(a.magnets, pid[:-1] + "R") if not w1 else pid[:-1] + "R"
        owner += ("; 2026-10-21: 'boost the bottom screws up a little bit' (cover I05-I15: the proximal heads 3 mm up, so the "
                  "pad's proximal notches are 3 mm deeper)")
    tag = ("_pocketwalls1" if w1 else "") + ("_pin" if a.pin_hole else "") + ("_raised" if a.raised_seats else "")
    date = a.date or ("20261021" if a.raised_seats else "20261020")
    name = f"{date}-dip_link_v3b_print_pad_{fkey}_B00{tag}_85a_t2_g30"
    out_dir = Path(a.out) if a.out else OUT
    if new:
        pid = a.pid or pid
        owner += ("; 2026-10-22: 'for the new fingers with the hole in the top, we should really undersize this hole too ... "
                  "undersize the slot and oversize the rail just a little bit for an even tighter fit ... prepare a tpu print on "
                  "the prusa for three new fingertips (with pauses)'")
        name = f"{date}-dip_link_print_pad_{fkey}_{pid}" + (f"_x{a.copies}" if a.copies > 1 else "") + "_85a_t2_g30"
    work = REPO / f"logs/20261020-dip_link/v3/b/print_b00_{a.magnets}{'_w1' if w1 else ''}{'_pin' if a.pin_hole else ''}" \
                  f"{'_raised' if a.raised_seats else ''}"
    if new:
        work = REPO / f"logs/{date}-dip_link/print_pad_{pid}"
    t0 = time.time()
    work.mkdir(parents=True, exist_ok=True)
    B = V2.b_module()
    tips0, fit0 = B.TIPS, B.GROOVE_FIT
    B.TIPS, B.GROOVE_FIT = B.PAD_SETS[pset], a.groove_fit
    walls0 = B.DESIGN.get("cover_walls")
    if a.raised_seats:
        B.DESIGN["cover_walls"] = "solid"                       # pad_cutters: the proximal notches seat_raise deeper
    try:
        pad, _ = B.make_pad(key)
        tip_name, orient, _ = B.TIPS[key]
    finally:
        B.TIPS, B.GROOVE_FIT = tips0, fit0
        B.DESIGN["cover_walls"] = walls0
    hole = None
    if a.pin_hole:
        import dip_b_tip_cap as TC
        if a.hole_d is not None:
            TC.CAP["hole_d"] = a.hole_d
        pad = pad - TC.pad_hole(TC.CAP2["hole_chamfer"] if a.raised_seats else 0.0)
        hole = dict(TC.CAP, x_F=TC.CAP["pin_x"], from_tip_end_z_F=TC.ZE,
                    mouth_chamfer=TC.CAP2["hole_chamfer"] if a.raised_seats else 0.0)
    tm = B.to_trimesh(pad)
    tm.update_faces(tm.nondegenerate_faces())
    tm.remove_unreferenced_vertices()
    rule = r4(B, tm, a.groove_fit)
    if hole is not None:                                        # the hole's least distance to a magnet pocket
        voids = [b for b in tm.split(only_watertight=False) if abs(b.volume) < 100]
        Vv = np.vstack([b.vertices for b in voids])
        dr = np.maximum(np.hypot(Vv[:, 0] - hole["x_F"], Vv[:, 1]) - hole["hole_d"] / 2, 0.0)
        dz = np.maximum(np.maximum(hole["from_tip_end_z_F"] - Vv[:, 2], Vv[:, 2] - (hole["from_tip_end_z_F"] + hole["hole_l"])), 0.0)
        hole = dict(d_mm=hole["hole_d"], depth_mm=hole["hole_l"], x_F=hole["x_F"], y_F=0.0, tip_end_z_F=hole["from_tip_end_z_F"],
                    mouth_chamfer_mm=hole["mouth_chamfer"], to_pocket_mm=round(float(np.min(np.hypot(dr, dz))), 2))
        if hole["to_pocket_mm"] < 2.0:
            raise SystemExit(f"pin hole {hole['to_pocket_mm']} mm from a pocket")
    tp = B.pad_print_frame(tm, orient)                          # the tip's sheet frame: its pockets apply as they are
    tc, (_, t) = PP.to_print(tp)                                # centred as on the fit-size plate, engraved, moved back
    if a.id_x is not None:
        FS.X_TEXT = a.id_x
    if a.id_cap is not None:
        FS.CAP = a.id_cap
    me, removed = FS.engrave_bed_face(tc, pid)
    part = trimesh.Trimesh(np.asarray(me.vertices) - t, np.asarray(me.faces), process=False)
    sheet_path = B.pad_source(f"{tip_name}_g30.json")
    sheet = json.loads(sheet_path.read_text())
    pockets = json.loads(json.dumps(sheet["pockets"]))
    bodies = sorted(float(b.volume) for b in part.split(only_watertight=False))
    if not part.is_watertight or len(bodies) != len(pockets) + 1 or bodies[-1] <= 0:   # one outer skin and the closed pockets
        raise SystemExit(f"{pid}: watertight {part.is_watertight}, bodies {bodies}")
    stl = OUT / f"{date}-dip_link_v3b_pad_{fkey}_B00{'_pin' if a.pin_hole else ''}{'_raised' if a.raised_seats else ''}_print.stl"
    if new:
        out_dir.mkdir(parents=True, exist_ok=True)
        stl = out_dir / f"{date}-dip_link_pad_{fkey}_{pid}_print.stl"
    if not w1:
        part.export(stl)
    mods, mod_info = [], None
    if w1:                                                      # the pad without its pockets, sliced per pocket layer
        outer = max(part.split(only_watertight=False), key=lambda b: float(b.volume))
        mod, mod_layers = MP.onewall_modifier(outer, pockets, True)
        mod_stl = out_dir / f"{name}_modifier.stl"
        mod.export(mod_stl)
        mods = [(mod, dict(perimeters=1))]
        mod_info = dict(stl=PP.rel(mod_stl), perimeters=1, inset_mm=round(MP.inset(True), 4), layers=len(mod_layers),
                        z_mm=[mod_layers[0][0], mod_layers[-1][1]], volume_mm3=round(float(mod.volume), 1))

    prow = MP.pauses(pockets, sheet["tip"])
    poles = None
    if a.poles == "checkerboard":                               # +1: north pole toward the palmar face
        zc = 0.5 * (min(q["z_mm"] for q in pockets) + max(q["z_mm"] for q in pockets))
        sign = lambda q: 1 if abs(q["y_mm"]) < 1e-6 else (-1 if q["y_mm"] > 0 else 1) * (1 if q["z_mm"] < zc else -1)  # noqa: E731
        poles = [dict(y_F=round(q["y_mm"], 3), z_tip_mm=round(q["z_mm"], 3), pole=sign(q),
                      bed=("centre" if abs(q["y_mm"]) < 1e-6 else "back" if q["y_mm"] < 0 else "front")) for q in pockets]
        word = {1: "north", -1: "south"}
        for r in prow:
            mags = [poles[i] for i in r["magnets"]]
            row = "proximal" if all(pockets[i]["z_mm"] < zc for i in r["magnets"]) else \
                "distal" if all(pockets[i]["z_mm"] > zc for i in r["magnets"]) else "centre"
            if len(mags) == 1:
                r["message"] = (f"Press the magnet ({row}) down its slot to the floor, {word[mags[0]['pole']]} pole toward the "
                                "curved palmar side (checkerboard), and resume")
            else:
                slots = sorted(mags, key=lambda m: m["bed"] != "back")
                r["message"] = (f"Press the {({2: 'two', 3: 'three'}).get(len(mags), len(mags))} magnets ({row}) down their slots "
                                "to the floor in a checkerboard: "
                                + ", ".join(f"in the slot toward the {m['bed']} of the plate the {word[m['pole']]} pole"
                                            for m in slots) + " toward the curved palmar side; then resume")
    if a.copies > 1:
        each = {2: "both pads", 3: "each of the three pads", 4: "each of the four pads"}.get(a.copies, f"each of the {a.copies} pads")
        for q in prow:
            q["message"] = f"In {each}: " + q["message"][0].lower() + q["message"][1:]
    SP.use_template()
    proj = (work if a.copies > 1 else out_dir) / f"{name}.3mf"
    MP.write_project(proj, part, name, SETTINGS, 0.1, prow, mods)
    with zipfile.ZipFile(FS.GOOD) as z:
        good_cfg = z.read("Metadata/Slic3r_PE.config").decode()
    with zipfile.ZipFile(proj) as z:
        files = {n: z.read(n) for n in z.namelist()}
    files["Metadata/Slic3r_PE.config"] = good_cfg.encode()      # the owner's good print's global settings, verbatim
    with zipfile.ZipFile(proj, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)

    gc, _ = PP.slice_project(proj, good_cfg, work)
    summ = PP.gcode_summary(gc)
    ck = MP.check_gcode(gc, pockets)
    mt = SP.check_material(gc)
    zl = sorted({round(q["centre_print_mm"][1] + 0.5, 1) for q in pockets})
    loops = MP.count_loops(gc, pockets, zl)
    problems = [] if rule["ok"] else [f"R4: {rule}"]
    for lp in loops:
        if (set(lp["loops_around_pockets"]) - {0} != {a.pocket_walls} or lp["loops_at_outer_surface"] != 2):
            problems.append(f"loops at z {lp['z']}: {lp['loops_around_pockets']} around the pockets, "
                            f"{lp['loops_at_outer_surface']} at the outer surface")
    if summ["m601"] != len(prow) or [round(z, 2) for z in ck["pauses_found_z"]] != [p["print_z"] for p in prow]:
        problems.append(f"pauses at {ck['pauses_found_z']} ({summ['m601']} M601), asked {[p['print_z'] for p in prow]}")
    if len(summ["labels"]) != 1 or summ.get("tool_changes"):
        problems.append(f"objects {list(summ['labels'])}, tool changes {summ.get('tool_changes')}")
    fpt = summ.get("filament_g_per_tool", [])
    if not fpt or fpt[PP.TPU_TOOL - 1] <= 0 or sum(fpt) - fpt[PP.TPU_TOOL - 1] > 1e-6:
        problems.append(f"filament per tool {fpt}")
    for i, q in enumerate(ck["pockets"]):
        if not (q.get("void_layers", 0) > 0 and q["beads_in_void_max"] < 0.01 and q["pause_before_bridge"]):
            problems.append(f"pocket {i}: {q}")
    if mt.get("part_nozzle_setpoints_C") != [240.0] or 35.0 not in mt.get("bed_setpoints_C", []):
        problems.append(f"nozzle {mt.get('part_nozzle_setpoints_C')} C, bed {mt.get('bed_setpoints_C')} C")
    copies = None
    if a.copies > 1:                                            # the same object as a.copies instances on one plate
        projn = out_dir / f"{name}.3mf"
        copies = copies_project(proj, projn, a.copies, good_cfg, work / "copies", summ, prow)
        problems += copies["problems"]
    rec = dict(owner=owner, magnets=a.magnets, layout=sheet.get("layout"), project=PP.rel(proj), stl=PP.rel(stl),
               params=os.environ["DLV2_PARAMS_B"], groove_fit_per_flank_mm=a.groove_fit, id=pid,
               engraving_removed_mm3=round(removed, 2), id_centre_x_mm=FS.X_TEXT, id_cap_mm=FS.CAP, copies=copies,
               volume_mm3=round(float(part.volume), 1), size_mm=np.round(part.extents, 2).tolist(), r4=rule,
               good_project=PP.rel(FS.GOOD), object_settings=dict(SETTINGS, layer_height="0.1"), sheet=PP.rel(sheet_path),
               magnet=sheet["magnet"], poles=poles, pocket_wall_min_mm=sheet.get("wall_min_mm"),
               chip_field_sheet=sheet.get("chip_field"),
               pocket_walls=a.pocket_walls, modifier=mod_info, loops=loops, pin_hole=hole,
               proximal_notch_depth_mm=round(B.HEAD_NOTCH_H + (B.rp("seat_raise") if a.raised_seats else 0.0), 2),
               pauses=prow, gcode=PP.rel(gc), summary={k: v for k, v in summ.items() if k != "labels"}, check=ck, material=mt,
               problems=problems, when=time.strftime("%Y-%m-%d %H:%M"), script="scripts/v3b_pad_b00_magnets_xl.py")
    (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(dict(volume_mm3=rec["volume_mm3"], engraving=rec["engraving_removed_mm3"], r4=rule, pin_hole=hole,
                          pauses=[p["print_z"] for p in prow], summary=rec["summary"],
                          pockets=[(q.get("void_layers"), q.get("beads_in_void_max"), (q.get("bridge") or {}).get("covered"),
                                    q.get("pause_before_bridge")) for q in ck["pockets"]],
                          loops=[(lp["z"], lp["loops_around_pockets"], lp["loops_at_outer_surface"]) for lp in loops],
                          nozzle=mt.get("part_nozzle_setpoints_C"), bed=mt.get("bed_setpoints_C"), problems=problems),
                     indent=1, default=float), flush=True)
    print(f"-> {PP.rel(out_dir / (name + '.3mf')) if copies else PP.rel(proj)} [{time.time() - t0:.0f} s]", flush=True)
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
