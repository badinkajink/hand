#!/usr/bin/env python3
"""Concept b's 11/22 pad at groove fit B00 (0.00 mm per flank) with its five 3 x 2 mm magnets and their three pauses, 85A,
on the Prusa XL's tool 2.

Owner, 2026-10-20 16:20: "can you also prepare a print for the b00 fingertip with the pauses back in? i want to try to get
sensing today". B00 of the fit-size plate (scripts/v2_tip_fit_sizes_xl.py, printed without pauses) was the groove fit the
owner chose on the rails ("the concept B rails slot great with the B00 sizing"). The pad is make_pad("ell11_5") of the v3
parameters at GROOVE_FIT 0.00, its ID engraved 0.4 mm deep in the mount face as on the fit-size plate, in the pad's print
frame, so that the 11/22 sheet's pockets and pauses apply unchanged (7.3, 12.2, 17.1 mm, metric_magnet_prints.pauses). Object
settings are those of the pipeline's pads (2 perimeters, 30 % gyroid, four top and bottom layers, 0.1 mm layers, tool 2); the
global settings are those of the owner's good print of 2026-10-18, verbatim. Sliced with PrusaSlicer and checked: three M601
pauses, each before the bridge layer over its pockets, no bead inside a pocket, tool 2 only at 240 C on a 35 C bed.
Output: docs/experiments/20261018-dip_link_board_mount/v3/b/print/20261020-dip_link_v3b_print_pad_ell11_5_B00_85a_t2_g30.3mf.

    unset VIRTUAL_ENV
    ~/.claude/bin/resguard.sh run --mem 4G --cpu 200 -- taskset -c 14,15 env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 \\
        ~/miniconda3/envs/femtip/bin/python -u scripts/v3b_pad_b00_magnets_xl.py
"""
from __future__ import annotations

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

KEY, FIT, PID = "ell11_5", 0.00, "B00"
NAME = f"20261020-dip_link_v3b_print_pad_{KEY}_{PID}_85a_t2_g30"
OUT = V2.B_DIR / "print"
WORK = REPO / "logs/20261020-dip_link/v3/b/print_b00"
SETTINGS = dict(perimeters=2, fill_density="30%", fill_pattern="gyroid", top_solid_layers=4, bottom_solid_layers=4)


def main():
    import trimesh
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    B = V2.b_module()
    fit0, B.GROOVE_FIT = B.GROOVE_FIT, FIT
    pad, _ = B.make_pad(KEY)
    B.GROOVE_FIT = fit0
    tm = B.to_trimesh(pad)
    tm.update_faces(tm.nondegenerate_faces())
    tm.remove_unreferenced_vertices()
    tp = B.pad_print_frame(tm, B.TIPS[KEY][1])                  # the 11/22 sheet's print frame: its pockets apply as they are
    tc, (_, t) = PP.to_print(tp)                                # centred as on the fit-size plate, engraved, moved back
    me, removed = FS.engrave_bed_face(tc, PID)
    part = trimesh.Trimesh(np.asarray(me.vertices) - t, np.asarray(me.faces), process=False)
    bodies = sorted(float(b.volume) for b in part.split(only_watertight=False))
    if not part.is_watertight or len(bodies) != 6 or bodies[-1] <= 0:  # one outer skin and the five closed magnet pockets
        raise SystemExit(f"{PID}: watertight {part.is_watertight}, bodies {bodies}")
    stl = OUT / f"20261020-dip_link_v3b_pad_{KEY}_{PID}_print.stl"
    part.export(stl)

    sheet = json.loads((REPO / B.tip_records()[KEY]["sheet"]).read_text())
    pockets = json.loads(json.dumps(sheet["pockets"]))
    prow = MP.pauses(pockets, sheet["tip"])
    SP.use_template()
    proj = OUT / f"{NAME}.3mf"
    MP.write_project(proj, part, NAME, SETTINGS, 0.1, prow)
    with zipfile.ZipFile(FS.GOOD) as z:
        good_cfg = z.read("Metadata/Slic3r_PE.config").decode()
    with zipfile.ZipFile(proj) as z:
        files = {n: z.read(n) for n in z.namelist()}
    files["Metadata/Slic3r_PE.config"] = good_cfg.encode()      # the owner's good print's global settings, verbatim
    with zipfile.ZipFile(proj, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)

    gc, _ = PP.slice_project(proj, good_cfg, WORK)
    summ = PP.gcode_summary(gc)
    ck = MP.check_gcode(gc, pockets)
    mt = SP.check_material(gc)
    problems = []
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
    rec = dict(owner="2026-10-20 16:20: 'prepare a print for the b00 fingertip with the pauses back in? i want to try to get "
                     "sensing today'",
               project=PP.rel(proj), stl=PP.rel(stl), params=os.environ["DLV2_PARAMS_B"], groove_fit_per_flank_mm=FIT,
               id=PID, engraving_removed_mm3=round(removed, 2), volume_mm3=round(float(part.volume), 1),
               size_mm=np.round(part.extents, 2).tolist(), good_project=PP.rel(FS.GOOD), object_settings=dict(SETTINGS, layer_height="0.1"),
               sheet=B.tip_records()[KEY]["sheet"], magnet=sheet["magnet"], pauses=prow, gcode=PP.rel(gc),
               summary={k: v for k, v in summ.items() if k != "labels"}, check=ck, material=mt, problems=problems,
               when=time.strftime("%Y-%m-%d %H:%M"), script="scripts/v3b_pad_b00_magnets_xl.py")
    proj.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(dict(volume_mm3=rec["volume_mm3"], engraving=rec["engraving_removed_mm3"], pauses=[p["print_z"] for p in prow],
                          summary=rec["summary"], pockets=[(q.get("void_layers"), q.get("beads_in_void_max"),
                                                            (q.get("bridge") or {}).get("covered"), q.get("pause_before_bridge"))
                                                           for q in ck["pockets"]],
                          nozzle=mt.get("part_nozzle_setpoints_C"), bed=mt.get("bed_setpoints_C"), problems=problems),
                     indent=1, default=float), flush=True)
    print(f"-> {PP.rel(proj)} [{time.time() - t0:.0f} s]", flush=True)
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
