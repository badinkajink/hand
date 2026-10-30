#!/usr/bin/env python3
"""Bambu Lab P1S projects for the inverted calibration rig (owner, 2026-10-29: "yea, we can build the inverted rod ...
let's shore up the plan and get it ready to print tomorrow").

Two PLA plates from the STLs that scripts/ur5_inverted_rig.py build writes (docs/experiments/20261029-inverted_calibration_rig/
cad/): plate a holds the probe base and the four tools, plate b the indenter plate v2e alone (232 x 236 mm of the bed's
256 x 256). Each part lies in its print orientation: the probe base on its foot, each tool on its flange (the sphere, rod or
flat face up), the plate on its base. The owner slices again in Bambu Studio, so each part's settings are written into the
project as object settings, which the GUI keeps (scripts/v2_ur5_pla_plate.py inject). The command line builds each plate,
writes the settings in and slices it; the G-code is checked part by part (layer step, top, supports, fill ratio, the P1S's
own start G-code). Output: docs/experiments/20261029-inverted_calibration_rig/print/.

    unset VIRTUAL_ENV
    ~/.claude/bin/resguard.sh run --mem 5G --cpu 400 -- taskset -c 14-18 env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 \\
        ~/miniconda3/envs/femtip/bin/python -u scripts/inverted_rig_print.py
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import dip_fit_print_plates as D  # noqa: E402
import dip_link_print_project as PP  # noqa: E402
import v2_ur5_pla_plate as VP  # noqa: E402

RIG = REPO / "docs/experiments/20261029-inverted_calibration_rig"
CAD, OUT = RIG / "cad", RIG / "print"
WORK = REPO / "logs/20261029-inverted_rig_print"
STEM = "20261029-inverted_rig"

# object settings, Bambu Studio keys
SOLID = dict(layer_height="0.1", wall_loops="4", top_shell_layers="6", bottom_shell_layers="6", sparse_infill_density="100%",
             sparse_infill_pattern="zig-zag", enable_support="0")          # the tools: the force passes through them
BASE = dict(SOLID, layer_height="0.15")                                     # the probe base, solid as v7's standoff-fork
PLATE = dict(layer_height="0.16", wall_loops="5", top_shell_layers="6", bottom_shell_layers="5", sparse_infill_density="25%",
             sparse_infill_pattern="gyroid", enable_support="0")          # walls 2 mm: the seats' supports bend as shells
PLATES = {
    "a": dict(title="probe base and tools", parts=[
        ("PB", "probe base", "probe_base", BASE),
        ("S5", "tool, 5 mm sphere", "tool_sphere5", SOLID),
        ("S2", "tool, 2 mm sphere", "tool_sphere2", SOLID),
        ("RD", "tool, half rod R 12.5", "tool_rod", SOLID),
        ("FL", "tool, flat 14 mm", "tool_flat", SOLID),
        ("F30", "tool, flat 14 mm at 30 deg", "tool_flat30", dict(SOLID, layer_height="0.08"))],   # its face slopes 30 deg:
#                                                                     the P1S's finest layer keeps the steps 0.08 mm
        glob=dict(layer_height="0.1", brim_type="auto_brim")),
    "b": dict(title="indenter plate v2e", parts=[("V2E", "indenter plate v2e", "plate_v2e", PLATE)],
              glob=dict(layer_height="0.16", brim_type="no_brim")),   # 232 mm of 256: a brim would reach the wiper's corner
}
EXCLUDE = (18.0, 28.0)                                                  # the P1S's bed_exclude_area, from the bed's corner
OWNER = "2026-10-29 15:2x: 'yea, we can build the inverted rod ... let's shore up the plan and get it ready to print tomorrow'"


def build(key, a):
    import trimesh
    spec = PLATES[key]
    work = WORK / key
    shutil.rmtree(work, ignore_errors=True)
    (work / "placed").mkdir(parents=True)
    glob = dict(D.P1S_PROCESS, sparse_infill_pattern="gyroid", sparse_infill_density="25%", enable_support="0",
                **spec["glob"])
    paths = {}
    for kind, d in (("machine", D.bbl_preset("machine", D.P1S["machine"])),
                    ("process", D.bbl_preset("process", D.P1S["process"], glob)),
                    ("filament", D.bbl_preset("filament", D.P1S["filament"]))):
        paths[kind] = work / f"{kind}.json"
        paths[kind].write_text(json.dumps(d, indent=1))
    meshes = []
    for pid, label, name, st in spec["parts"]:
        tm = trimesh.load(CAD / f"{STEM}_{name}.stl", force="mesh")
        tm, _ = PP.to_print(tm, np.eye(3))                 # every part prints in its CAD frame's orientation, z up
        tm.merge_vertices()
        tm = D.single(tm, pid)
        tm.export(OUT / f"{STEM}_{name}_print.stl")        # the print-oriented STL, centred, on z = 0
        meshes.append(tm)
    pos = D.layout(meshes, D.P1S_BED, gap=12.0, width=200.0)
    lo = min(tm.bounds[0][0] + p_[0] for tm, p_ in zip(meshes, pos))
    hi = max(tm.bounds[1][0] + p_[0] for tm, p_ in zip(meshes, pos))
    dx = max(0.0, EXCLUDE[0] + 1.0 - lo)               # clear of the P1S's wiper corner (bed_exclude_area 18 x 28 mm)
    if hi + dx > D.P1S_BED[0] - 1.0:
        raise SystemExit(f"plate {key} cannot clear the excluded corner: x {lo:.1f}..{hi:.1f} mm")
    pos = [p_ + (dx, 0.0) for p_ in pos]
    files, objs = [], []
    for (pid, label, name, st), tm, p_ in zip(spec["parts"], meshes, pos):
        tm = tm.copy()
        tm.apply_translation((p_[0], p_[1], 0.0))
        f = work / "placed" / f"{pid}_{name}.stl"
        tm.export(f)
        files.append(str(f))
        objs.append(dict(id=pid, label=label, name=f.name, stl=D.rel(OUT / f"{STEM}_{name}_print.stl"), settings=st,
                         bounds_mm=np.round(tm.bounds, 2).tolist(), volume_mm3=round(float(tm.volume), 1)))
    for (_, _, _, _), o in zip(spec["parts"], objs):
        (x0, y0, _), (x1, y1, _) = o["bounds_mm"]
        if x0 < 1.0 or y0 < 1.0 or x1 > D.P1S_BED[0] - 1.0 or y1 > D.P1S_BED[1] - 1.0:
            raise SystemExit(f"{o['id']} leaves the bed: {o['bounds_mm']}")
    base = [*D.BAMBU, "--debug", "1", "--load-settings", f"{paths['machine']};{paths['process']}",
            "--load-filaments", str(paths["filament"])]
    r = subprocess.run(base + ["--arrange", "0", "--orient", "0", "--export-3mf", "built.3mf", "--outputdir", str(work)]
                       + files, capture_output=True, text=True)
    (work / "cli_build.log").write_text(r.stdout + r.stderr)
    if not (work / "built.3mf").exists():
        raise SystemExit(f"Bambu Studio did not write plate {key} (log {D.rel(work / 'cli_build.log')})")
    VP.inject(work / "built.3mf", work / "with_settings.3mf", {o["name"]: o["settings"] for o in objs})
    proj = f"{STEM}_p1s_{key}.3mf"
    r = subprocess.run(base + ["--slice", "0", "--export-3mf", proj, "--outputdir", str(work), str(work / "with_settings.3mf")],
                       capture_output=True, text=True)
    (work / "cli_slice.log").write_text(r.stdout + r.stderr)
    res = json.loads((work / "result.json").read_text())
    if res.get("return_code") != 0:
        raise SystemExit(f"Bambu Studio: {res.get('error_string')} (log {D.rel(work / 'cli_slice.log')})")
    shutil.copy(work / proj, OUT / proj)
    with zipfile.ZipFile(OUT / proj) as z:
        info = z.read("Metadata/slice_info.config").decode()
        gcode = z.read("Metadata/plate_1.gcode").decode(errors="replace")
        msets = z.read("Metadata/model_settings.config").decode()
    pred = float(re.search(r'key="prediction" value="([\d.]+)"', info).group(1))
    weight = float(re.search(r'key="weight" value="([\d.]+)"', info).group(1))
    layers = int(re.search(r"; total layer number: (\d+)", gcode).group(1))
    problems = D.check_bambu_gcode(gcode, "P1S", 220, 55)
    if msets.count('<metadata key="layer_height"') < len(objs):
        problems.append("object settings missing from the sliced project")
    po = VP.per_object(gcode, objs)
    for o in objs:
        r_ = po[o["id"]]
        zs = sorted(r_["z"])
        steps = np.round(np.diff(zs), 3) if len(zs) > 1 else np.array([])
        lh = float(o["settings"]["layer_height"])
        o["layers"], o["top_z"] = len(zs), (zs[-1] if zs else None)
        o["layer_step_mm"] = float(np.median(steps)) if len(steps) else None
        o["support_mm"] = round(sum(v for k, v in r_["feats"].items() if k.startswith("Support")), 1)
        part_e = sum(v for k, v in r_["feats"].items() if not k.startswith(("Support", "Brim", "Skirt", "Custom", "Ironing")))
        o["fill_ratio"] = round(part_e * VP.FILAMENT_AREA / o["volume_mm3"], 3)
        o["filament_g"] = round(part_e * VP.FILAMENT_AREA / 1e3 * 1.24, 1)
        if o["layer_step_mm"] is None or abs(o["layer_step_mm"] - lh) > 1e-3:
            problems.append(f"{o['id']} layer step {o['layer_step_mm']} mm, asked {lh}")
        if o["top_z"] is None or abs(o["top_z"] - o["bounds_mm"][1][2]) > lh + 0.05:
            problems.append(f"{o['id']} top {o['top_z']} mm, part {o['bounds_mm'][1][2]} mm")
        if o["support_mm"] > 0:
            problems.append(f"{o['id']} support {o['support_mm']} mm, none asked")
        if o["settings"]["sparse_infill_density"] == "100%" and o["fill_ratio"] < 0.85:
            problems.append(f"{o['id']} fill ratio {o['fill_ratio']} though asked solid")
    rec = dict(owner=OWNER, plate=key, title=spec["title"], project=D.rel(OUT / proj), printer="Bambu Lab P1S 0.4 nozzle",
               filament=D.P1S["filament"], bed="Textured PEI Plate, 55 C", process_base=D.P1S["process"],
               global_settings=glob, time_s=pred, time=D.fmt_time(pred), filament_g=weight, layers=layers, objects=objs,
               problems=problems, slicer="Bambu Studio command line: plate built, object settings written into "
                                         "Metadata/model_settings.config, sliced again", when=time.strftime("%Y-%m-%d %H:%M"))
    fig = OUT / f"{STEM}_p1s_{key}_map.png"
    D.plate_map([(dict(id=o["id"], material="PLA", short=o["label"]), trimesh.load(work / "placed" / o["name"])) for o in objs],
                D.P1S_BED, f"Bambu Lab P1S, PLA: {spec['title']}, {D.fmt_time(pred)}, {weight:.0f} g", fig)
    rec["map"] = D.rel(fig)
    (OUT / f"{STEM}_p1s_{key}.json").write_text(json.dumps(rec, indent=1) + "\n")
    for o in objs:
        print(f"{o['id']:4s} {o['label']:24s} layers {o['layers']:4d} step {o['layer_step_mm']} top {o['top_z']} "
              f"(part {o['bounds_mm'][1][2]}) support {o['support_mm']} fill {o['fill_ratio']} {o['filament_g']} g", flush=True)
    print(f"P1S plate {key}: {len(objs)} parts, {D.fmt_time(pred)}, {weight:.1f} g, {layers} layers -> {D.rel(OUT / proj)}"
          + (f"; PROBLEMS: {'; '.join(problems)}" if problems else ""), flush=True)
    return rec


def main():
    global RIG, CAD, OUT, WORK, STEM, OWNER
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plates", default="a,b")
    ap.add_argument("--block", action="store_true", help="the seat block's folder (2026-10-30): plate a with the tools' "
                    "taper-seated sockets and the probe base's captive nuts, plate b the seat block")
    a = ap.parse_args()
    if a.block:
        RIG = REPO / "docs/experiments/20261030-inverted_rig_seat_block"
        CAD, OUT, WORK, STEM = RIG / "cad", RIG / "print", REPO / "logs/20261030-seat_block_print", "20261030-seat_block"
        OWNER = ("2026-10-30 11:2x: 'a smaller ... print out block that screws into the same 2x4 ... (can you space it out "
                 "farther from the fingertip holders so a drill can more easily drive in?)'; 11:4x: 'can we actually create a "
                 "fastening point on the homing block for different size indenters?'")
        PLATES["b"] = dict(title="seat block", parts=[("SB", "seat block", "seat_block", PLATE)],
                           glob=dict(layer_height="0.16", brim_type="no_brim"))
    OUT.mkdir(parents=True, exist_ok=True)
    for key in a.plates.split(","):
        build(key, a)


if __name__ == "__main__":
    main()
