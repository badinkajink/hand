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
Output: docs/experiments/20261018-dip_link_board_mount/v3/b/print/20261020-dip_link_v3b_print_pad_ell11_{5,4}_B00_85a_t2_g30.3mf.

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
}
OUT = V2.B_DIR / "print"
SETTINGS = dict(perimeters=2, fill_density="30%", fill_pattern="gyroid", top_solid_layers=4, bottom_solid_layers=4)


def r4(B, tm):
    """dip_link_concept_b's R4 on a pad in F: least distance from its magnet pockets (the small void bodies) to the rail
    grooves at GROOVE_FIT and to the screw-head notches, mm."""
    voids = [b for b in tm.split(only_watertight=False) if abs(b.volume) < 100]
    Vv = np.vstack([b.vertices for b in voids])
    lo = np.array([B.X_SLOT - B.RAIL_H - B.GROOVE_FLOOR, B.RAIL_Y - B.RAIL_TOP / 2 + FIT, B.RAIL_Z[0] - 0.2])
    hi = np.array([B.X_SLOT, B.RAIL_Y + B.RAIL_TOP / 2 - FIT, B.RAIL_Z[1] + 0.2])
    dist = lambda P: np.linalg.norm(np.maximum(np.maximum(lo - P, P - hi), 0.0), axis=1)   # noqa: E731
    dg = float(min(dist(Vv).min(), dist(Vv * [1, -1, 1]).min()))
    dn = min(float(np.min(np.maximum(np.hypot(Vv[:, 1] - y, Vv[:, 2] - z) - B.HEAD_NOTCH_D / 2, 0.0))) for y, z in B.screw_yz())
    return dict(groove_to_pocket_mm=round(dg, 2), notch_to_pocket_mm=round(dn, 2), need_mm=2.0, ok=bool(min(dg, dn) >= 2.0),
                pocket_dorsal_x_F=round(float(Vv[:, 0].max()), 2), tpu_behind_pockets_mm=round(B.X_SLOT - float(Vv[:, 0].max()), 2))


def main():
    import trimesh
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--magnets", choices=list(VARIANTS), default="five")
    ap.add_argument("--pocket-walls", type=int, choices=(1, 2), default=2)
    a = ap.parse_args()
    pset, key, pid, fkey, owner = VARIANTS[a.magnets]
    w1 = a.pocket_walls == 1
    if w1:
        pid = {"five": "5W1", "four": "4W1"}[a.magnets]
        owner += "; 18:35: 'is it possible to run just one wall around the magnets instead of 2?'"
    name = f"20261020-dip_link_v3b_print_pad_{fkey}_B00{'_pocketwalls1' if w1 else ''}_85a_t2_g30"
    work = REPO / f"logs/20261020-dip_link/v3/b/print_b00_{a.magnets}{'_w1' if w1 else ''}"
    t0 = time.time()
    work.mkdir(parents=True, exist_ok=True)
    B = V2.b_module()
    tips0, fit0 = B.TIPS, B.GROOVE_FIT
    B.TIPS, B.GROOVE_FIT = B.PAD_SETS[pset], FIT
    try:
        pad, _ = B.make_pad(key)
        tip_name, orient, _ = B.TIPS[key]
    finally:
        B.TIPS, B.GROOVE_FIT = tips0, fit0
    tm = B.to_trimesh(pad)
    tm.update_faces(tm.nondegenerate_faces())
    tm.remove_unreferenced_vertices()
    rule = r4(B, tm)
    tp = B.pad_print_frame(tm, orient)                          # the tip's sheet frame: its pockets apply as they are
    tc, (_, t) = PP.to_print(tp)                                # centred as on the fit-size plate, engraved, moved back
    me, removed = FS.engrave_bed_face(tc, pid)
    part = trimesh.Trimesh(np.asarray(me.vertices) - t, np.asarray(me.faces), process=False)
    sheet_path = B.pad_source(f"{tip_name}_g30.json")
    sheet = json.loads(sheet_path.read_text())
    pockets = json.loads(json.dumps(sheet["pockets"]))
    bodies = sorted(float(b.volume) for b in part.split(only_watertight=False))
    if not part.is_watertight or len(bodies) != len(pockets) + 1 or bodies[-1] <= 0:   # one outer skin and the closed pockets
        raise SystemExit(f"{pid}: watertight {part.is_watertight}, bodies {bodies}")
    stl = OUT / f"20261020-dip_link_v3b_pad_{fkey}_B00_print.stl"
    if not w1:
        part.export(stl)
    mods, mod_info = [], None
    if w1:                                                      # the pad without its pockets, sliced per pocket layer
        outer = max(part.split(only_watertight=False), key=lambda b: float(b.volume))
        mod, mod_layers = MP.onewall_modifier(outer, pockets, True)
        mod_stl = OUT / f"{name}_modifier.stl"
        mod.export(mod_stl)
        mods = [(mod, dict(perimeters=1))]
        mod_info = dict(stl=PP.rel(mod_stl), perimeters=1, inset_mm=round(MP.inset(True), 4), layers=len(mod_layers),
                        z_mm=[mod_layers[0][0], mod_layers[-1][1]], volume_mm3=round(float(mod.volume), 1))

    prow = MP.pauses(pockets, sheet["tip"])
    SP.use_template()
    proj = OUT / f"{name}.3mf"
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
    rec = dict(owner=owner, magnets=a.magnets, layout=sheet.get("layout"), project=PP.rel(proj), stl=PP.rel(stl),
               params=os.environ["DLV2_PARAMS_B"], groove_fit_per_flank_mm=FIT, id=pid, engraving_removed_mm3=round(removed, 2),
               volume_mm3=round(float(part.volume), 1), size_mm=np.round(part.extents, 2).tolist(), r4=rule,
               good_project=PP.rel(FS.GOOD), object_settings=dict(SETTINGS, layer_height="0.1"), sheet=PP.rel(sheet_path),
               magnet=sheet["magnet"], pocket_wall_min_mm=sheet.get("wall_min_mm"), chip_field_sheet=sheet.get("chip_field"),
               pocket_walls=a.pocket_walls, modifier=mod_info, loops=loops,
               pauses=prow, gcode=PP.rel(gc), summary={k: v for k, v in summ.items() if k != "labels"}, check=ck, material=mt,
               problems=problems, when=time.strftime("%Y-%m-%d %H:%M"), script="scripts/v3b_pad_b00_magnets_xl.py")
    proj.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(dict(volume_mm3=rec["volume_mm3"], engraving=rec["engraving_removed_mm3"], r4=rule,
                          pauses=[p["print_z"] for p in prow], summary=rec["summary"],
                          pockets=[(q.get("void_layers"), q.get("beads_in_void_max"), (q.get("bridge") or {}).get("covered"),
                                    q.get("pause_before_bridge")) for q in ck["pockets"]],
                          loops=[(lp["z"], lp["loops_around_pockets"], lp["loops_at_outer_surface"]) for lp in loops],
                          nozzle=mt.get("part_nozzle_setpoints_C"), bed=mt.get("bed_setpoints_C"), problems=problems),
                     indent=1, default=float), flush=True)
    print(f"-> {PP.rel(proj)} [{time.time() - t0:.0f} s]", flush=True)
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
