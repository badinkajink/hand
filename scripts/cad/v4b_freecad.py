#!/usr/bin/env python3
"""Solids of the v4b build's printed and laser-cut parts through FreeCAD (OpenCASCADE): reads the spec that
scripts/cad/v4b_build.py printed writes, fuses each part's pieces (extruded outlines with holes, boxes, cylinders),
cuts its holes, and writes one STEP and one STL per part in its body's frame (mm).

    env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/bin:/bin V4B_SPEC=<spec.json> freecadcmd scripts/cad/v4b_freecad.py

The spec names the output folders; the summary (volume, bounding box, closed shell) goes to the spec's "summary" path.
"""
import json
import os

import FreeCAD
import Import
import Mesh
import MeshPart
import Part

V = FreeCAD.Vector


def flat_solid(p):
    U, Vv, N, o = (V(*p[k]) for k in ("U", "V", "N", "o"))

    def pt(u, v, w=0.0):
        return o + U * u + Vv * v + N * (p["w0"] + w)
    pts = [pt(u, v) for u, v in p["outline"]]
    face = Part.Face(Part.makePolygon(pts + [pts[0]]))
    s = face.extrude(N * p["t"])
    for u, v, d in p["holes"]:
        c = Part.makeCylinder(d / 2, p["t"] + 2, pt(u, v, -1.0), N)
        s = s.cut(c)
    for poly in p["polys"]:
        q = [pt(u, v, -1.0) for u, v in poly]
        s = s.cut(Part.Face(Part.makePolygon(q + [q[0]])).extrude(N * (p["t"] + 2)))
    return s


def box_solid(p):
    lo, hi = p["lo"], p["hi"]
    return Part.makeBox(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], V(*lo))


def cyl_solid(p):
    a, b = V(*p["a"]), V(*p["b"])
    d = b - a
    return Part.makeCylinder(p["d"] / 2, d.Length, a, d.normalize())


MAKE = {"flat": flat_solid, "box": box_solid, "cyl": cyl_solid}


def main():
    spec = json.load(open(os.environ["V4B_SPEC"]))
    os.makedirs(spec["step_dir"], exist_ok=True)
    os.makedirs(spec["stl_dir"], exist_ok=True)
    summary = {}
    for name, part in spec["parts"].items():
        shape = None
        for p in part["pieces"]:
            s = MAKE[p["type"]](p)
            shape = s if shape is None else shape.fuse(s)
        for c in part["cuts"]:
            shape = shape.cut(MAKE[c["type"]](c))
        shape = shape.removeSplitter()
        doc = FreeCAD.newDocument("v4b")
        obj = doc.addObject("Part::Feature", part["file"])
        obj.Shape = shape
        doc.recompute()
        Import.export([obj], os.path.join(spec["step_dir"], part["file"] + ".step"))
        mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=part.get("deflection", 0.02), AngularDeflection=0.2)
        mesh.write(os.path.join(spec["stl_dir"], part["file"] + ".stl"))
        bb = shape.BoundBox
        summary[name] = {"file": part["file"], "volume_mm3": round(shape.Volume, 1), "closed": bool(shape.isClosed()),
                         "solids": len(shape.Solids), "bbox_mm": [round(v, 2) for v in (bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax)],
                         "facets": mesh.CountFacets}
        FreeCAD.closeDocument(doc.Name)
        print(name, summary[name], flush=True)
    json.dump(summary, open(spec["summary"], "w"), indent=1)


main()
