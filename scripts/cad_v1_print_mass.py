#!/usr/bin/env python3
"""Printed masses of the cad_v1 finger's PLA parts, from PrusaSlicer (2026-10-13).

Owner, 2026-10-13: "the fingers are printed at 15% infill pla and the servos are as you measured, 13g". Each PLA part
STL in assets/cad_v1/ is sliced in its own orientation with the owner's active PrusaSlicer presets (PrusaSlicer.ini of
the flatpak: printer "Original Prusa XL - 5T Input Shaper 0.4 nozzle", print "0.20mm STRUCTURAL @XLIS 0.4", filament
"Generic PLA @XLIS") at 15 % infill, without skirt or brim; the preset prints no supports. The filament mass the slicer
reports is the part's printed mass. scripts/build_cad_v1_finger.py spreads it uniformly over the part (density = mass /
CAD volume). The TPU tip and the servos are not sliced.

Writes print_mass.json beside the model; the resolved slicer settings go to logs/20261013-cad_v1_spheres_work/print/.

    .venv/bin/python scripts/cad_v1_print_mass.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import trimesh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_cad_v1_finger import CAD, FILES, OUT  # noqa: E402

PRUSA = ["flatpak", "run", "--command=prusa-slicer", "com.prusa3d.PrusaSlicer"]
PRESETS = {"printer": "Original Prusa XL - 5T Input Shaper 0.4 nozzle", "print": "0.20mm STRUCTURAL @XLIS 0.4",
           "material": "Generic PLA @XLIS"}
FILL = "15%"  # owner, 2026-10-13
PLA_PARTS = ("carriage", "yaw_link", "mcp_link", "dip_link")
WORK = REPO / "logs/20261013-cad_v1_spheres_work/print"
KEYS = ("perimeters", "top_solid_layers", "bottom_solid_layers", "layer_height", "fill_density", "fill_pattern",
        "perimeter_extrusion_width", "support_material", "filament_density")


def slicer_args():
    return [f"--printer-profile={PRESETS['printer']}", f"--print-profile={PRESETS['print']}",
            f"--material-profile={PRESETS['material']}", f"--fill-density={FILL}", "--skirts=0", "--brim-width=0"]


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    rows = {}
    for p in PLA_PARTS:
        stl = WORK / f"{p}.stl"
        shutil.copy(CAD / FILES[p], stl)  # the flatpak reads paths without spaces more reliably
        gc = WORK / f"{p}.gcode"
        r = subprocess.run(PRUSA + slicer_args() + ["--export-gcode", "-o", str(gc), str(stl)], capture_output=True, text=True)
        if r.returncode or not gc.exists():
            raise SystemExit(f"{p}: PrusaSlicer failed\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        head = gc.read_bytes()[:400_000].decode("latin-1")  # binary G-code keeps its metadata as plain key=value text
        grams = sum(float(x) for x in re.search(r"filament used \[g\]\s*=\s*([0-9., ]+)", head).group(1).split(","))
        vol_mm3 = float(trimesh.load(stl).volume)
        rows[p] = dict(grams=round(grams, 3), volume_mm3=round(vol_mm3, 1), density_kg_m3=round(grams / vol_mm3 * 1e6, 1),
                       solid_pla_grams=round(1.24e-3 * vol_mm3, 3))
        print(p, rows[p], flush=True)
    cfg = WORK / "resolved.ini"
    subprocess.run(PRUSA + slicer_args() + ["--save", str(cfg), str(WORK / f"{PLA_PARTS[0]}.stl")], capture_output=True, text=True)
    settings = {}
    for line in cfg.read_text().splitlines():
        k, _, v = line.partition(" = ")
        if k in KEYS:
            settings[k] = v.split(",")[0]
    version = subprocess.run(PRUSA + ["--help"], capture_output=True, text=True).stdout.splitlines()[0].strip()
    out = dict(slicer=version, presets=PRESETS, settings=settings, parts=rows,
               note="owner 2026-10-13: printed at 15 % infill PLA; servos 13 g; tip TPU not sliced")
    (OUT / "print_mass.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "print_mass.json")


if __name__ == "__main__":
    main()
