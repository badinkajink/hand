#!/usr/bin/env python3
"""STEP envelopes of the two cad_v4 finger layouts for detailing in Onshape or Inventor.

Reads docs/experiments/20261118-sr2_actuator_layout/data/20261118-prims.json (scripts/cad/cad_v4_layouts.py prims) and
writes one STEP per layout with a named solid per part (servo cases, housings, links, idler, rod, coupler, yoke), in the
index finger's mount frame at the zero pose, mm. The STEP files stay on the workstation (assets/cad_v4/concepts/step/ is in
.gitignore; CAD files move by rsync).

    env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/bin:/bin CADV4_REPO=$PWD freecadcmd scripts/cad/cad_v4_step.py
"""
import json
import os
from pathlib import Path

import FreeCAD
import Import
import Part

ROOT = Path(os.environ.get("CADV4_REPO", "."))
SRC = ROOT / "docs/experiments/20261118-sr2_actuator_layout/data/20261118-prims.json"
OUT = ROOT / "assets/cad_v4/concepts/step"


def placement(pos, rot):
    m = FreeCAD.Matrix(rot[0][0], rot[0][1], rot[0][2], pos[0], rot[1][0], rot[1][1], rot[1][2], pos[1],
                       rot[2][0], rot[2][1], rot[2][2], pos[2], 0, 0, 0, 1)
    return FreeCAD.Placement(m)


def solid(p):
    s = p["size_mm"]
    if p["type"] == "box":
        shape = Part.makeBox(2 * s[0], 2 * s[1], 2 * s[2], FreeCAD.Vector(-s[0], -s[1], -s[2]))
    else:  # capsule along local z: cylinder plus two spheres
        r, h = s[0], s[1]
        shape = Part.makeCylinder(r, 2 * h, FreeCAD.Vector(0, 0, -h))
        shape = shape.fuse(Part.makeSphere(r, FreeCAD.Vector(0, 0, h))).fuse(Part.makeSphere(r, FreeCAD.Vector(0, 0, -h)))
        shape = shape.removeSplitter()
    shape.Placement = placement(p["pos_mm"], p["rot"])
    return shape


def main():
    data = json.loads(SRC.read_text())["result"]
    OUT.mkdir(parents=True, exist_ok=True)
    for v, rec in data.items():
        doc = FreeCAD.newDocument(f"cad_v4_{v}")
        objs = []
        for p in rec["parts"]:
            o = doc.addObject("Part::Feature", p["name"])
            o.Shape = solid(p)
            objs.append(o)
        doc.recompute()
        path = OUT / f"20261118-cad_v4_{v}_index_envelopes.step"
        Import.export(objs, str(path))
        print("wrote", path, len(objs), "solids")
        FreeCAD.closeDocument(doc.Name)


main()
