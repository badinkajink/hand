#!/usr/bin/env python3
"""assets/cad_v3: the whole SR2 hand as built for the UR5, read from the owner's Onshape export of 2026-11-04
(assets/cad_v2/full_cad: the assembly "hand v3 (anthro config)", the gantries gantry_v3 and gantry_v3_thumb, and the
single parts base_plate, robot_mount and "UR5 Adaptor").

Owner, 2026-11-04: "it's time to go sim2real for real now on the sr2 hand, meaning we are putting the hardware hand on the
ur5 ... we do need to fully understand it (and probably have a visual mesh version to look pretty for renders which we do
not actually use for control)"; "1) generate a full visual fidelity model of this setup in mujoco 2) generate a simplified
geometry version of this model that is more or less just an update to the current model we are using with the corrected
geometry/offsets, with, crucially, the right transforms known. the ur5 mount adaptor will probably be mounted in one of
four possible orientations (i did not design it with the little clocking nub), and i will correct that when time comes.
this can be cad v3".

The STEP files name their parts "Part 1", "Part 2" and so on, so every solid is identified here by its geometry. FreeCAD
0.19 reads the assembly structure (Import.insert); each solid is written as a mesh in the frame of its file, with its mass
properties and the axes of its cylindrical faces (holes and bosses), which the hole matching and the gantry kinematics use.

Commands, from the repository root:

  extract  inside freecadcmd: every solid of the six STEP files -> logs/20261104-cad_v3/extract/<file>/<nnn>.stl (mm, the
           file's frame) and logs/20261104-cad_v3/extract/parts.json (per solid: label, sub-assembly, volume, centre of
           mass, inertia about the centre of mass, bounding box, face census, cylindrical faces)
             ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/bin:/bin \\
                 CADV3_CMD=extract CADV3_REPO=$PWD freecadcmd scripts/cad/cad_v3_assets.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

try:
    REPO = Path(__file__).resolve().parents[2]
except NameError:                                   # freecadcmd executes the file without __file__
    REPO = Path(os.environ.get("CADV3_REPO", os.getcwd()))

SRC = REPO / "assets/cad_v2/full_cad"
SOURCES = {                                          # key: STEP file of the owner's export of 2026-11-04
    "hand": SRC / "assemblies/hand v3 (anthro config).step",
    "gantry": SRC / "assemblies/gantry_v3.step",
    "gantry_thumb": SRC / "assemblies/gantry_v3_thumb.step",
    "base_plate": SRC / "base_plate.step",
    "robot_mount": SRC / "robot_mount.step",
    "adaptor": SRC / "UR5 Adaptor.step",
}
WORK = REPO / "logs/20261104-cad_v3"
EXTRACT = WORK / "extract"
LIN_DEFL_MM, ANG_DEFL_RAD = 0.05, 0.25               # tessellation of the visual meshes


def _vec(v):
    return [float(v.x), float(v.y), float(v.z)]


def _global_placement(o):
    """The placement of a Part::Feature in its file's frame: its own placement under every App::Part above it."""
    pl = o.Placement
    q = o
    while True:
        ps = [x for x in q.InList if x.TypeId == "App::Part"]
        if not ps:
            return pl
        q = ps[0]
        pl = q.Placement.multiply(pl)


def _chain(o):
    out, q = [], o
    while True:
        ps = [x for x in q.InList if x.TypeId == "App::Part"]
        if not ps:
            return out
        q = ps[0]
        out.append(q.Label)


def _cylinders(shape):
    """Cylindrical faces: radius, axis, a point on the axis, the axial extent and the angular span of the face, and
    whether the face is concave (a hole: its outward normal points toward the axis)."""
    out = []
    for i, f in enumerate(shape.Faces):
        s = f.Surface
        if type(s).__name__ != "Cylinder":
            continue
        ax = s.Axis
        ax.normalize()
        c = s.Center
        u0, u1, v0, v1 = f.ParameterRange
        um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        p = f.valueAt(um, vm)
        n = f.normalAt(um, vm)
        d = p.sub(c)
        radial = d.sub(_scaled(ax, d.dot(ax)))       # FreeCAD's Vector.multiply works in place
        ts = [(v.Point.sub(c)).dot(ax) for v in f.Vertexes] or [v0, v1]
        bb = f.BoundBox
        corners = [(bb.XMin, bb.YMin, bb.ZMin), (bb.XMax, bb.YMax, bb.ZMax)]
        out.append(dict(face=i, r=float(s.Radius), axis=_vec(ax), point=_vec(c), t=[float(min(ts)), float(max(ts))],
                        span_deg=float(math.degrees(abs(u1 - u0))), concave=bool(n.dot(radial) < 0),
                        area=float(f.Area), bbox=[list(corners[0]), list(corners[1])]))
    return out


def _scaled(v, k):
    import FreeCAD as App
    return App.Vector(v.x * k, v.y * k, v.z * k)


def _planes(shape, min_area=20.0):
    out = []
    for i, f in enumerate(shape.Faces):
        s = f.Surface
        if type(s).__name__ != "Plane" or f.Area < min_area:
            continue
        u0, u1, v0, v1 = f.ParameterRange
        n = f.normalAt(0.5 * (u0 + u1), 0.5 * (v0 + v1))
        c = f.CenterOfMass
        out.append(dict(face=i, normal=_vec(n), centre=_vec(c), area=float(f.Area)))
    return out


def extract():
    import FreeCAD as App
    import Import
    import MeshPart
    prm = App.ParamGet("User parameter:BaseApp/Preferences/Mod/Import")
    prm.SetBool("UseLegacyImporter", False)
    prm.SetBool("UseLinkGroup", False)
    prm.SetBool("UseBaseName", True)
    prm.SetInt("ImportMode", 0)
    prm.SetBool("ReduceObjects", False)
    prm.SetBool("ExpandCompound", False)
    EXTRACT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, path in SOURCES.items():
        doc = App.newDocument(key)
        Import.insert(str(path), doc.Name)
        (EXTRACT / key).mkdir(exist_ok=True)
        idx = 0
        for o in doc.Objects:
            if o.TypeId != "Part::Feature" or not hasattr(o, "Shape") or o.Shape.isNull():
                continue
            gpl = _global_placement(o)
            s = o.Shape.copy()
            s.Placement = gpl
            solids = s.Solids
            if not solids:
                continue
            for k, so in enumerate(solids):
                name = f"{idx:03d}"
                bb = so.BoundBox
                census = {}
                for f in so.Faces:
                    t = type(f.Surface).__name__
                    census[t] = census.get(t, 0) + 1
                com = so.CenterOfMass
                I = so.MatrixOfInertia                      # about the centre of mass, axes of the file's frame
                inertia = [[I.A11, I.A12, I.A13], [I.A21, I.A22, I.A23], [I.A31, I.A32, I.A33]]
                mesh = MeshPart.meshFromShape(Shape=so, LinearDeflection=LIN_DEFL_MM, AngularDeflection=ANG_DEFL_RAD,
                                              Relative=False)
                stl = EXTRACT / key / f"{name}.stl"
                mesh.write(str(stl))
                rows.append(dict(
                    file=key, idx=idx, name=name, object=o.Name, label=o.Label, solid=k, chain=_chain(o),
                    placement=dict(pos=_vec(gpl.Base), quat_xyzw=[float(x) for x in gpl.Rotation.Q]),
                    volume=float(so.Volume), area=float(so.Area), com=_vec(com),
                    inertia_mm5=[[float(x) for x in r] for r in inertia],
                    bbox=[bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax], nfaces=len(so.Faces), census=census,
                    cylinders=_cylinders(so), planes=_planes(so), stl=str(stl.relative_to(REPO)),
                    mesh_faces=int(mesh.CountFacets)))
                idx += 1
        App.closeDocument(doc.Name)
        print(key, idx, "solids", flush=True)
    out = dict(when=time.strftime("%Y-%m-%d %H:%M"), units="mm", sources={k: str(p.relative_to(REPO)) for k, p in
                                                                         SOURCES.items()},
               tessellation=dict(linear_mm=LIN_DEFL_MM, angular_rad=ANG_DEFL_RAD), parts=rows)
    (EXTRACT / "parts.json").write_text(json.dumps(out, indent=1))
    print("wrote", EXTRACT / "parts.json", len(rows), "solids", flush=True)


CMD = os.environ.get("CADV3_CMD") or (sys.argv[1] if len(sys.argv) > 1 else "")
if CMD == "extract":
    extract()
elif __name__ == "__main__" and CMD:
    raise SystemExit(f"unknown command {CMD!r}")
