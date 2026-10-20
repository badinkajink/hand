#!/usr/bin/env python3
"""Concept a's 11/22 tip at four key-channel interferences and concept b's 11/22 pad at four rail-groove fits, 85A, on one
Prusa XL plate (tool 2), without magnets, for the fit on the printed v2 links and cover.

Owner, 2026-10-19 19:15: "dont we wanna try printing out multiple size grooves for the finger tips and for the concept b
cassette rails? i already sent out the pla print so too late for the cassette rail sizes but we can still print out more tips".
a: the five-magnet 11/22 tip (dip_link_concept_a.tip_F) minus tip_cutter at fit_flank 0.00, 0.05, 0.10, 0.20 mm per flank
(design 0.05); b: make_pad("ell11_5") at GROOVE_FIT 0.00, 0.05, 0.10, 0.15 mm per flank (design 0.05). Each part's ID (A00 ..
B15: the concept and the interference in hundredths of a millimetre) is engraved 0.4 mm deep in its mount face, the face on
the bed, on the half away from the channels. The object settings are those of the v2 tip project (2 perimeters, 30 % gyroid,
four top and bottom layers, 0.1 mm layers, tool 2), the global settings those of the owner's good print of 2026-10-18, and the
magnet pauses are left out so that the plate prints unattended; the pockets stay empty. Sliced with PrusaSlicer and checked.
Output: docs/experiments/20261018-dip_link_board_mount/v2/print/20261019-dip_link_v2_fit_sizes_tips_a_pads_b_85a_t2_g30.3mf.

    unset VIRTUAL_ENV
    ~/.claude/bin/resguard.sh run --mem 4G --cpu 200 -- taskset -c 14,15 env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 \\
        ~/miniconda3/envs/femtip/bin/python -u scripts/v2_tip_fit_sizes_xl.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import zipfile
from pathlib import Path

import scipy.spatial  # noqa: F401  (the environment's libstdc++ before manifold3d)
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import dip_fit_print_plates as D  # noqa: E402
import dip_link_print_project as PP  # noqa: E402
import dip_link_v2 as V2  # noqa: E402

FLANKS_A = (0.00, 0.05, 0.10, 0.20)
FITS_B = (0.00, 0.05, 0.10, 0.15)
KEY = "ell11_5"
GOOD = REPO / ("docs/experiments/20261016-sim2real_fingertip_readiness/print/"
               "20261018-siraya85a_plate_block_cap8p5_ell16x32_ell11x22_seat2p9_g30.3mf")
TIP_PROJ = V2.V2 / "print/20261019-dip_link_v2_tip_a_pad_b_ell11_5_85a_t2_g30.3mf"
OUT = V2.V2 / "print/20261019-dip_link_v2_fit_sizes_tips_a_pads_b_85a_t2_g30.3mf"
WORK = REPO / "logs/20261019-v2_tip_fit_sizes"
CAP, DEPTH, X_TEXT = 2.5, 0.4, 2.5     # text cap height and depth (mm); its centre's x in the print frame, palmar of centre


def object_settings():
    """The v2 tip's object and volume settings (object 1 of the a-and-b plate), without names, matrices and types."""
    with zipfile.ZipFile(TIP_PROJ) as z:
        cfg = z.read("Metadata/Slic3r_PE_model.config").decode()
    blk = re.search(r'<object id="1".*?</object>', cfg, re.S).group(0)
    st = {k: v for k, v in re.findall(r'key="(\w+)" value="([^"]*)"', blk)
          if k not in ("name", "matrix", "volume_type", "source_file", "source_object_id", "source_volume_id",
                       "source_offset_x", "source_offset_y", "source_offset_z")}
    st.pop("extruder", None)
    return st


def engrave_bed_face(tm, text):
    """The ID engraved DEPTH deep in the face on the bed (z = 0), reading from below: the part turned 180 degrees about y,
    engraved from its top, turned back. Returns the mesh and the volume removed."""
    import trimesh
    R = np.diag([-1.0, 1.0, -1.0])
    top = float(tm.bounds[1][2])
    flip = trimesh.Trimesh(np.asarray(tm.vertices) @ R.T + [0.0, 0.0, top], np.asarray(tm.faces), process=False)
    if flip.volume < 0:
        flip.invert()
    man = D.engrave(D.to_man(flip), text, CAP, -X_TEXT, 0.0, top, DEPTH)
    out = D.to_tm(man)
    back = trimesh.Trimesh((np.asarray(out.vertices) - [0.0, 0.0, top]) @ R.T, np.asarray(out.faces), process=False)
    if back.volume < 0:
        back.invert()
    return back, float(tm.volume - back.volume)


def main():
    import trimesh
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    parts = []
    A, DLF, *_ = V2.setup_a()
    sp = A.spec()
    K = A.MeshK()
    tm, inv = A.tip_F(KEY, sp)
    mt = A.from_trimesh(tm)
    for fl in FLANKS_A:
        tF = A.to_trimesh(mt - A.tip_cutter(K, dict(A.P, fit_flank=fl)))
        tP = trimesh.Trimesh(inv(np.asarray(tF.vertices)), np.asarray(tF.faces), process=False)
        if tP.volume < 0:
            tP.invert()
        tP, _ = PP.to_print(tP)
        parts.append((f"A{D.hund(fl)}", "a", fl, tP))
    B = V2.b_module()
    fit0 = B.GROOVE_FIT
    for fit in FITS_B:
        B.GROOVE_FIT = fit
        pad, _ = B.make_pad(KEY)
        tm_ = B.to_trimesh(pad)
        tm_.update_faces(tm_.nondegenerate_faces())
        tm_.remove_unreferenced_vertices()
        tp = B.pad_print_frame(tm_, B.TIPS[KEY][1])
        tp, _ = PP.to_print(tp)
        parts.append((f"B{D.hund(fit)}", "b", fit, tp))
    B.GROOVE_FIT = fit0
    st = object_settings()
    objs, rows = [], []
    for pid, concept, v, m in parts:
        me, removed = engrave_bed_face(m, pid)
        if not me.is_watertight or me.volume <= 0:              # one outer skin and the five closed magnet pockets
            raise SystemExit(f"{pid}: watertight {me.is_watertight}, volume {me.volume:.1f}")
        name = (f"{pid}_v2a_tip_{KEY}_flank{D.hund(v)}" if concept == "a" else f"{pid}_v2b_pad_{KEY}_groove{D.hund(v)}")
        me.export(WORK / f"{name}.stl")
        objs.append(dict(key=name, name=name, label=name, short=pid, mesh=me, Rt=None, extruder=PP.TPU_TOOL, settings=st,
                         volumes=[]))
        rows.append(dict(id=pid, concept=concept, interference_per_flank_mm=v, name=name,
                         size_mm=np.round(me.extents, 2).tolist(), volume_mm3=round(float(me.volume), 1),
                         engraving_removed_mm3=round(removed, 2)))
        print(f"{pid}: {v:.2f} mm per flank, {me.volume:.1f} mm3, engraving {removed:.2f} mm3", flush=True)
    for o, t in zip(objs, PP.layout([o["mesh"] for o in objs])):
        o["translation"] = t
    PP.write_project(OUT, objs)
    with zipfile.ZipFile(GOOD) as z:
        good_cfg = z.read("Metadata/Slic3r_PE.config").decode()
    with zipfile.ZipFile(OUT) as z:
        files = {n: z.read(n) for n in z.namelist()}
    files["Metadata/Slic3r_PE.config"] = good_cfg.encode()          # the owner's good print's global settings, verbatim
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    gc, log = PP.slice_project(OUT, good_cfg, WORK)
    summ = PP.gcode_summary(gc)
    text = Path(gc).read_text()
    temps = sorted({int(v) for v in re.findall(r"^M10[49] (?:T\d )?S(\d+)", text, re.M) if int(v) > 0})
    beds = sorted({int(v) for v in re.findall(r"^M1[49]0 S(\d+)", text, re.M)})
    problems = []
    if summ["m601"]:
        problems.append(f"{summ['m601']} pauses")
    if len(summ["labels"]) != len(objs):
        problems.append(f"{len(summ['labels'])} objects in the G-code, {len(objs)} on the plate")
    if summ.get("tool_changes"):
        problems.append(f"{summ['tool_changes']} tool changes")
    fpt = summ.get("filament_g_per_tool", [])
    if not fpt or fpt[PP.TPU_TOOL - 1] <= 0 or sum(fpt) - fpt[PP.TPU_TOOL - 1] > 1e-6:
        problems.append(f"filament per tool {fpt}")
    for r in rows:
        if abs(r["engraving_removed_mm3"]) < 0.5:
            problems.append(f"{r['id']}: engraving removed {r['engraving_removed_mm3']} mm3")
    rec = dict(owner="2026-10-19 19:15: 'we can still print out more tips'", project=PP.rel(OUT), good_project=PP.rel(GOOD),
               object_settings=st, parts=rows, gcode=PP.rel(gc), summary={k: v for k, v in summ.items() if k != "labels"},
               nozzle_c=temps, bed_c=beds, problems=problems, when=time.strftime("%Y-%m-%d %H:%M"),
               note="no magnet pauses: the pockets stay empty; the interference that fits prints again with magnets "
                    "(v2/print/20261019-dip_link_v2_tip_a_pad_b_ell11_5_85a_t2_g30.3mf at that size)")
    OUT.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("summary", "nozzle_c", "bed_c", "problems")}, indent=1), flush=True)
    print(f"-> {PP.rel(OUT)} [{time.time() - t0:.0f} s]", flush=True)
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
