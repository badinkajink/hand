#!/usr/bin/env python3
"""assets/cad_v2: the DIP link with its fingertip parts, and the bench fixtures for the UR5e and the ATI Mini40 and Mini45,
gathered from their experiment folders, with exact STEP files where those folders have only meshes or faceted STEP.

Owner, 2026-10-26: "can we create a cad_v2 folder in assets/ with our updated files? it should just be the dip link. include
subfolders for the ur5 mount and the atimini40/45 fixtures ... include a readme ... that links the new files to the
corresponding docs/experiments, and describe how parts have changed from the original human cad"; "as long as the step is a
simple geometry it should be fine mating stuff".

The DIP link's own STEP (v4/b/cad/20261021-dip_link_v4b_dip_link.step.zip) has 6866 faces, 6608 of them triangles: its two
clevis arms are the unchanged arms of assets/cad_v1/fingertip_mount_DIP_link.stl, a mesh. Both arms are prisms along the PIP
axis between a few steps, so each band between two steps is extruded here from its mid-plane section, fitted with lines, arcs
and circles, and fused with the exact new body by the concept b pipeline itself.

Commands, from the repository root:

  arms   sections of the two arms (as the pipeline cut them for v4: logs/20261021-dip_link/v4/b/arm_*.stl) at the middle of
         each band, fitted -> logs/20261026-cad_v2/arm_profiles.json
             ~/miniconda3/envs/femtip/bin/python scripts/cad_v2_assets.py arms
  step   inside freecadcmd: the DIP link v4b with the fitted arms (dip_link_concept_b.dip_link_parts on the v4 parameters),
         the cassette 3T and the top cover TW15H3 from their STEP files, all moved into frame D; the ATI fixtures from
         ati_stack_mount's builders on FreeCAD Part -> assets/cad_v2/**/*.step, a fine STL of each new solid in its source
         frame -> logs/20261026-cad_v2/stl/, log -> logs/20261026-cad_v2/step.json
             env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/bin:/bin CADV2_CMD=step CADV2_REPO=$PWD freecadcmd scripts/cad_v2_assets.py
         The v7 fork's STEP comes from the mount script's own step command:
             env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/bin:/bin UR5M_REV=v7 UR5M_ONLY=standoff_fork UR5M_CMD=step \\
                 UR5M_REPO=$PWD freecadcmd scripts/ur5_calibration_mount.py
  check  each new solid against the mesh it stands for: volumes and surface distances both ways -> assets/cad_v2/checks.json
  copy   the print STLs, the other STEP files and the meshes from their experiment folders, then each file's source, sha256
         and STEP face census -> assets/cad_v2/manifest.json; copy --check lists the copies whose source has changed

Frames: the DIP-link parts are in frame D, the coordinates of assets/cad_v1/fingertip_mount_DIP_link.stl (the PIP axis is the
y axis), so they replace that file in an assembly; the pipelines' frame F maps to it by x_D = -x_F, y_D = y_F + 21.25,
z_D = -z_F - 51.55. The mount and ATI parts are in the mount frame M (origin at the tool flange's centre, z along the tool
axis), the indenters in the table frame, the mass holder in its own frame.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:                                   # freecadcmd executes the file without __file__
    REPO = Path(os.environ.get("CADV2_REPO", os.getcwd()))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

OUT = REPO / "assets/cad_v2"
WORK = REPO / "logs/20261026-cad_v2"
PROFILES = WORK / "arm_profiles.json"
DIP = REPO / "docs/experiments/20261018-dip_link_board_mount"
V4 = DIP / "v4/b"
ARMS = REPO / "logs/20261021-dip_link/v4/b"         # the arms as the concept b pipeline cut them for v4
PARAMS4 = DIP / "params/20261021-v4_concept_b.json"
V3M = REPO / "docs/experiments/20261021-ur5_calibration_mount_v3"
V7M = REPO / "docs/experiments/20261025-ur5_calibration_mount_v7"
ATI = REPO / "docs/experiments/20261023-ati_ft_calibration"
T_D_FROM_F = [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 21.25], [0.0, 0.0, -1.0, -51.55], [0.0, 0.0, 0.0, 1.0]]
TOL_ARC = 2e-3      # mm: the largest distance of a section vertex from its fitted arc or circle
R_MAX = 500.0       # mm: a fitted radius above this is a line
SAG_MAX = 0.03      # mm: the largest gap between a fitted arc and one of its facets (a few long flat facets also lie on a
                    # wide circle, which bows up to 0.25 mm off them between the vertices)

# (file under assets/cad_v2, source) for the files copied unchanged
COPIES = [
    ("dip_link_v4b_print.stl", V4 / "print/20261021-dip_link_v4b_dip_link_pla_bambu.stl"),
    ("cassette_3T_print.stl", V4 / "print/20261022-dip_link_v4b_cover_I15H3_3T_pla_bambu.stl"),
    ("top_cover_TW15H3_print.stl", V4 / "print/20261021-dip_link_v4b_tip_cap_TW15H3_pla_bambu.stl"),
    ("pad_B05R_print.stl", V4 / "print/20261022-dip_link_pad_ell11_5_B05R_print.stl"),
    ("ur5_mount/standoff_fork_v7.step", V7M / "cad/20261025-ur5_calibration_mount_v7_standoff_fork.step"),
    ("ur5_mount/standoff_fork_v7.stl", V7M / "cad/20261025-ur5_calibration_mount_v7_standoff_fork.stl"),
] + [(f"ur5_mount/{p}.{e}", V3M / f"cad/20261021-ur5_calibration_mount_v3_{p}.{e}")    # v7's are byte-identical to v3's
     for p in ("adapter", "clamp_plate", "indenter_sphere", "indenter_rod") for e in ("step", "stl")] + [
    ("ati_mini40_45/mini45_base.stl", ATI / "cad/20261023-ati_stack_ati_base.stl"),
    ("ati_mini40_45/mini45_top.stl", ATI / "cad/20261023-ati_stack_ati_top.stl"),
    ("ati_mini40_45/mini40_base.stl", ATI / "cad/20261023-ati_stack_mini40_ati_base.stl"),
    ("ati_mini40_45/mini40_top.stl", ATI / "cad/20261023-ati_stack_mini40_ati_top.stl"),
    ("ati_mini40_45/hook_cross.stl", ATI / "cad/20261023-ati_stack_hook_cross.stl"),
    ("ati_mini40_45/mass_cup.stl", ATI / "cad/20261023-ati_mass_cup.stl"),
    ("ati_mini40_45/mass_cap.stl", ATI / "cad/20261023-ati_mass_cap.stl"),
]
# (new solid's fine STL in logs/20261026-cad_v2/stl, the mesh it stands for), for check
PAIRS = [
    ("dip_link_v4b_F.stl", V4 / "cad/20261021-dip_link_v4b_dip_link.stl"),
    ("mini45_base.stl", ATI / "cad/20261023-ati_stack_ati_base.stl"),
    ("mini45_top.stl", ATI / "cad/20261023-ati_stack_ati_top.stl"),
    ("mini40_base.stl", ATI / "cad/20261023-ati_stack_mini40_ati_base.stl"),
    ("mini40_top.stl", ATI / "cad/20261023-ati_stack_mini40_ati_top.stl"),
    ("hook_cross.stl", ATI / "cad/20261023-ati_stack_hook_cross.stl"),
    ("mass_cup.stl", ATI / "cad/20261023-ati_mass_cup.stl"),
    ("mass_cap.stl", ATI / "cad/20261023-ati_mass_cap.stl"),
]


def rel(p):
    return str(Path(p).resolve().relative_to(REPO))


# ---------------------------------------------------------------------------------------------------------------- arms
def section(mesh, y):
    """Closed loops of the mesh's section at y, as (x, z) vertices without repeats or collinear points."""
    s = mesh.section(plane_origin=[0.0, y, 0.0], plane_normal=[0.0, 1.0, 0.0])
    out = []
    for loop in s.discrete:
        P = np.asarray(loop, float)[:, [0, 2]]
        if np.linalg.norm(P[0] - P[-1]) < 1e-7:
            P = P[:-1]
        P = P[np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-7]]
        while True:                     # a point on the chord of its neighbours is where the plane cut a facet's diagonal
            a, c = np.roll(P, 1, 0), np.roll(P, -1, 0)
            ac, ab = c - a, P - a
            d = np.abs(ac[:, 0] * ab[:, 1] - ac[:, 1] * ab[:, 0]) / np.maximum(np.hypot(ac[:, 0], ac[:, 1]), 1e-12)
            drop = d < 1e-6
            drop &= ~np.roll(drop, 1)
            if not drop.any():
                break
            P = P[~drop]
        out.append(P)
    return out


def circle(P):
    """Least-squares circle through the points: centre, radius, largest distance of a point from it."""
    A = np.c_[2.0 * P, np.ones(len(P))]
    (cx, cz, k), *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    r = float(np.sqrt(k + cx * cx + cz * cz))
    return [float(cx), float(cz)], r, float(np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cz) - r).max())


def one_sense(Q):
    """The polyline turns one way, by less than 30 deg at each vertex (the facets of one arc)."""
    d = np.diff(Q, axis=0)
    cr = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    dot = (d[:-1] * d[1:]).sum(1)
    ang = np.arctan2(cr, dot)
    return bool((np.all(ang > -1e-9) or np.all(ang < 1e-9)) and np.abs(ang).max() < np.radians(30.0))


def segments(P):
    """A closed loop as lines, arcs (three points on each) and full circles."""
    c, r, res = circle(P)
    if res < TOL_ARC and len(P) >= 8:
        return [dict(t="circle", c=c, r=r, res=res)]
    n = len(P)
    seg = np.roll(P, -1, 0) - P
    P = np.roll(P, -int(np.argmax(np.hypot(seg[:, 0], seg[:, 1]))), 0)     # start at the longest facet
    Q = np.vstack([P, P[:1]])
    out, i = [], 0
    while i < n:
        best, j = None, i + 3
        while j <= n:
            c, r, res = circle(Q[i:j + 1])
            if res > TOL_ARC or r > R_MAX or not one_sense(Q[i:j + 1]):
                break
            half = 0.5 * np.hypot(*np.diff(Q[i:j + 1], axis=0).T)
            sag = float((r - np.sqrt(np.maximum(r * r - half * half, 0.0))).max())
            if sag > SAG_MAX:
                break
            best, j = (j, c, r, res, sag), j + 1
        if best:
            j, c, r, res, sag = best
            out.append(dict(t="arc", p=[Q[i].tolist(), Q[(i + j) // 2].tolist(), Q[j].tolist()], c=c, r=r, res=res, sag=sag))
            i = j
        else:
            out.append(dict(t="line", p=[Q[i].tolist(), Q[i + 1].tolist()]))
            i += 1
    return out


def cmd_arms():
    import trimesh
    rec = dict(when=time.strftime("%Y-%m-%d %H:%M"), tol_arc_mm=TOL_ARC, arms=[])
    for name in ("arm_horn", "arm_slot"):
        m = trimesh.load(ARMS / f"{name}.stl", force="mesh")
        ys, cnt = np.unique(np.round(m.vertices[:, 1], 3), return_counts=True)
        levels = [float(y) for y, c in zip(ys, cnt) if c >= 100]          # the arm's faces normal to the PIP axis
        nr = m.face_normals
        odd = (np.abs(nr[:, 1]) > 1e-3) & (np.abs(nr[:, 1]) < 0.999)     # faces neither along nor across the axis
        arm = dict(name=name, source=rel(ARMS / f"{name}.stl"), volume_mm3=round(float(m.volume), 3), levels_y_F=levels,
                   faces_not_prismatic=dict(n=int(odd.sum()), area_mm2=round(float(m.area_faces[odd].sum()), 2),
                                            y_F=np.round([m.triangles_center[odd, 1].min(), m.triangles_center[odd, 1].max()],
                                                         3).tolist() if odd.any() else None), bands=[])
        for y0, y1 in zip(levels[:-1], levels[1:]):
            loops = [segments(P) for P in section(m, 0.5 * (y0 + y1))]
            kinds = [it["t"] for lp in loops for it in lp]
            arm["bands"].append(dict(y=[y0, y1], loops=loops))
            print(f"{name} y_F {y0:.3f}..{y1:.3f}: {len(loops)} loops, {kinds.count('line')} lines, {kinds.count('arc')} arcs, "
                  f"{kinds.count('circle')} circles", flush=True)
        rec["arms"].append(arm)
    WORK.mkdir(parents=True, exist_ok=True)
    PROFILES.write_text(json.dumps(rec, indent=1))
    print("->", rel(PROFILES))


# ------------------------------------------------------------------------------------------------------ step (freecadcmd)
def fc_step():
    import FreeCAD
    import Import
    import MeshPart
    import Part

    def V(x, y, z):
        return FreeCAD.Vector(float(x), float(y), float(z))

    def matrix(T):
        return FreeCAD.Matrix(*[float(v) for row in T for v in row])

    log = dict(when=time.strftime("%Y-%m-%d %H:%M"), files={})
    (WORK / "stl").mkdir(parents=True, exist_ok=True)

    def census(shape):
        c = {}
        for f in shape.Faces:
            k = type(f.Surface).__name__
            c[k] = c.get(k, 0) + 1
        return dict(sorted(c.items(), key=lambda kv: -kv[1]))

    def export(shape, dest, label, src, check_stl=None, T=None):
        """STEP through a document object (the mount script's note: Shape.exportStep once wrote a rotated part wrongly),
        read back; with check_stl a fine STL of the solid before T."""
        if check_stl:
            MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.005, AngularDeflection=0.1, Relative=False).write(
                str(WORK / "stl" / check_stl))
        if T is not None:                           # copy=True moves the geometry; a bare location is lost on export
            shape = shape.copy()
            shape.transformShape(matrix(T), True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        doc = FreeCAD.newDocument("cadv2")
        o = doc.addObject("Part::Feature", label)
        o.Label = label
        o.Shape = shape
        Import.export([o], str(dest))
        FreeCAD.closeDocument("cadv2")
        back = Part.read(str(dest))
        bb = back.BoundBox
        log["files"][rel(dest)] = dict(label=label, source=src, valid=bool(back.isValid()), solids=len(back.Solids),
                                       faces=len(back.Faces), surfaces=census(back), volume_mm3=round(back.Volume, 2),
                                       volume_before_mm3=round(shape.Volume, 2),
                                       bounds=[round(v, 3) for v in (bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax)])
        print(rel(dest), json.dumps(log["files"][rel(dest)]), flush=True)

    # the arms from their fitted sections
    def wire(items, y):
        edges = []
        for it in items:
            if it["t"] == "circle":
                edges.append(Part.Circle(V(it["c"][0], y, it["c"][1]), V(0, 1, 0), it["r"]).toShape())
                continue
            p = [V(q[0], y, q[1]) for q in it["p"]]
            edges.append((Part.Arc(*p) if it["t"] == "arc" else Part.LineSegment(*p)).toShape())
        return Part.Wire(Part.__sortEdges__(edges))

    def roundtrip_valid(s):
        """Valid in memory and after a STEP write and read-back (sliver faces pass the first and fail the second)."""
        if not s.isValid():
            return False
        tmp = WORK / "roundtrip.step"
        doc = FreeCAD.newDocument("rt")
        o = doc.addObject("Part::Feature", "rt")
        o.Shape = s
        Import.export([o], str(tmp))
        FreeCAD.closeDocument("rt")
        ok = Part.read(str(tmp)).isValid()
        tmp.unlink()
        return bool(ok)

    def first_valid(*cands):
        """The first shape of (name, thunk) candidates that is one solid, valid through a STEP round trip, with the names
        tried."""
        tried = []
        for nm, fn in cands:
            try:
                s = fn()
            except Exception as e:                  # noqa: BLE001  (an OCC failure is one more candidate gone)
                tried.append(f"{nm}: {type(e).__name__}")
                continue
            if len(s.Solids) != 1:
                tried.append(f"{nm}: {len(s.Solids)} solids")
                continue
            ok = roundtrip_valid(s)
            tried.append(f"{nm}: {'valid' if ok else 'invalid'}")
            if ok:
                return s.Solids[0], tried
        raise RuntimeError(f"no valid shape: {tried}")

    # Neighbouring bands of the horn arm share stretches of outline whose fitted arcs differ by microns; a plain fuse leaves
    # sliver faces there that fail after a STEP round trip, a fuzzy fuse (2 um) merges them
    prof = json.loads(PROFILES.read_text())
    arms, log["arms"] = [], {}
    for a in prof["arms"]:
        bands = []
        for b in a["bands"]:
            y0, y1 = b["y"]
            face = Part.makeFace([wire(lp, y0) for lp in b["loops"]], "Part::FaceMakerBullseye")
            bands.append(face.extrude(V(0, y1 - y0, 0)))
        if len(bands) == 1:
            s, tried = bands[0], ["one band"]
        else:
            s, tried = first_valid(("fuzzy fuse 2 um, refine", lambda: bands[0].fuse(bands[1:], 2e-3).removeSplitter()),
                                   ("fuzzy fuse 10 um, refine", lambda: bands[0].fuse(bands[1:], 1e-2).removeSplitter()),
                                   ("fuse, refine", lambda: bands[0].fuse(bands[1:]).removeSplitter()))
        s = Part.Solid(s.Shells[0]) if s.ShapeType != "Solid" and len(s.Solids) == 1 else s
        log["arms"][a["name"]] = dict(volume_mm3=round(s.Volume, 3), mesh_volume_mm3=a["volume_mm3"], faces=len(s.Faces),
                                      surfaces=census(s), valid=bool(s.isValid()), tried=tried)
        print(a["name"], json.dumps(log["arms"][a["name"]]), flush=True)
        arms.append(s)

    # the DIP link: the concept b pipeline's own body and cuts, with these arms
    os.environ["DIPB_PARAMS"] = str(PARAMS4)
    os.environ.pop("DIPB_FREECAD", None)
    import dip_link_concept_b as B
    B.BrepB.arms = lambda self: [a.copy() for a in arms]
    plain_fuse = B.BrepB.fuse

    def build_dip(fuzzy):
        """The pipeline's DIP link; with fuzzy > 0 its fuses run fuzzy (the arms' edges carry the band fuse's tolerance)."""
        B.BrepB.fuse = (lambda self, a, others: a.fuse(list(others), fuzzy)) if fuzzy else plain_fuse
        t0 = time.time()
        dip = B.dip_link_parts(B.BrepB())[1]
        print(f"DIP link fused (fuzzy {fuzzy * 1e3:g} um) in {time.time() - t0:.1f} s: {len(dip.Solids)} solids", flush=True)
        return dip

    dip, log["dip_link_tried"] = first_valid(("fuse, refine", lambda: build_dip(0.0).removeSplitter()),
                                             ("fuzzy fuse 2 um, refine", lambda: build_dip(2e-3).removeSplitter()),
                                             ("fuzzy fuse 10 um, refine", lambda: build_dip(1e-2).removeSplitter()))
    print("DIP link:", log["dip_link_tried"], flush=True)
    export(dip, OUT / "dip_link_v4b.step", "dip_link_v4b", "built here: fitted arms + concept b body (v4 parameters)",
           check_stl="dip_link_v4b_F.stl", T=T_D_FROM_F)
    for dest, src, label in (("cassette_3T.step", V4 / "cad/20261022-dip_link_v4b_cover_I15H3_3T.step", "cassette_3T"),
                             ("top_cover_TW15H3.step", V4 / "cad/20261021-dip_link_v4b_tip_cap_TW15H3.step", "top_cover_TW15H3")):
        export(Part.read(str(src)), OUT / dest, label, rel(src) + " moved into frame D", T=T_D_FROM_F)

    # the ATI fixtures (v3's patterns: ati_stack_mount sets UR5M_REV=v3 unless it is set)
    import ati_stack_mount as A
    UM = A.UM

    def prism_z(K, poly, z0, z1):           # ati_stack_mount's _prism_z calls manifold3d itself
        pts = [V(x, y, z0) for x, y in poly]
        return Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(V(0, 0, z1 - z0))

    A._prism_z = prism_z
    K = UM.BrepK()
    for sensor, rec in (("mini45", ATI / "20261023-ati_stack_parts.json"), ("mini40", ATI / "20261023-ati_stack_mini40_parts.json")):
        A.S = A.SENSORS[sensor]
        clock = float(json.loads(rec.read_text())["clock_deg"])
        parts = [("base", lambda: A.base(K, clock)), ("top", lambda: A.top(K, clock))]
        if sensor == "mini45":
            parts.append(("hook_cross", lambda: A.cross(K)))
        for nm, fn in parts:
            stem = nm if nm == "hook_cross" else f"{sensor}_{nm}"
            export(fn(), OUT / f"ati_mini40_45/{stem}.step", stem, f"built here: ati_stack_mount {nm} ({sensor}, clock {clock:g} deg)",
                   check_stl=f"{stem}.stl")
    for nm, fn in (("mass_cup", A.holder_cup), ("mass_cap", A.holder_cap)):
        export(fn(K), OUT / f"ati_mini40_45/{nm}.step", nm, f"built here: ati_stack_mount holder_{nm[5:]}", check_stl=f"{nm}.stl")
    (WORK / "step.json").write_text(json.dumps(log, indent=1))
    print("done")


# --------------------------------------------------------------------------------------------------------------- check
def cmd_check():
    import trimesh
    rng = np.random.default_rng(0)

    def dist(a, b, n=20000):
        """Distances from n points spread over a's surface to b's surface (chunked: proximity queries grow with the batch)."""
        pts, _ = trimesh.sample.sample_surface(a, n, seed=int(rng.integers(1 << 30)))
        d = np.concatenate([trimesh.proximity.closest_point(b, pts[i:i + 2048])[1] for i in range(0, len(pts), 2048)])
        return d, pts

    rec = dict(when=time.strftime("%Y-%m-%d %H:%M"), note="distances in mm between the new solid's fine STL and the mesh it "
               "stands for, from 20000 points on each surface; volumes in mm3", parts={})
    for new, ref in PAIRS:
        a = trimesh.load(WORK / "stl" / new, force="mesh")
        b = trimesh.load(ref, force="mesh")
        dab, pa = dist(a, b)
        dba, pb = dist(b, a)
        far = pa[dab > 0.05]
        r = dict(reference=rel(ref), volume_new=round(float(a.volume), 2), volume_reference=round(float(b.volume), 2),
                 new_to_reference=dict(max=round(float(dab.max()), 4), p99=round(float(np.percentile(dab, 99)), 4)),
                 reference_to_new=dict(max=round(float(dba.max()), 4), p99=round(float(np.percentile(dba, 99)), 4)),
                 share_over_0p05mm=round(float(np.mean(np.r_[dab, dba] > 0.05)), 5))
        if len(far):
            r["over_0p05mm_bounds_new_frame"] = np.round([far.min(0), far.max(0)], 2).tolist()
        far_b = pb[dba > 0.05]
        if len(far_b):
            r["reference_over_0p05mm_bounds"] = np.round([far_b.min(0), far_b.max(0)], 2).tolist()
        rec["parts"][new] = r
        print(new, json.dumps(r), flush=True)
    (OUT / "checks.json").write_text(json.dumps(rec, indent=1) + "\n")


# ---------------------------------------------------------------------------------------------------------------- copy
SURF = {"plane": r"\bPLANE\(", "cylinder": r"\bCYLINDRICAL_SURFACE\(", "cone": r"\bCONICAL_SURFACE\(",
        "sphere": r"\bSPHERICAL_SURFACE\(", "torus": r"\bTOROIDAL_SURFACE\(", "b-spline": r"\bB_SPLINE_SURFACE_WITH_KNOTS\(",
        "extrusion": r"\bSURFACE_OF_LINEAR_EXTRUSION\(", "revolution": r"\bSURFACE_OF_REVOLUTION\("}


def step_census(path):
    t = Path(path).read_text(errors="replace")
    c = {k: len(re.findall(v, t)) for k, v in SURF.items()}
    return dict(faces=len(re.findall(r"\bADVANCED_FACE\(", t)), **{k: v for k, v in c.items() if v})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_copy(check=False):
    if check:
        man = json.loads((OUT / "manifest.json").read_text())
        stale = [f for f, e in man["files"].items() if e.get("source_sha256") and (REPO / e["source"]).exists()
                 and sha(REPO / e["source"]) != e["source_sha256"]]
        print("copies whose source changed:", stale or "none")
        return
    for dest, src in COPIES:
        (OUT / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, OUT / dest)
    built = json.loads((WORK / "step.json").read_text())["files"]
    files = {}
    for f in sorted(p for p in OUT.rglob("*") if p.is_file() and p.suffix in (".step", ".stl")):
        k = str(f.relative_to(OUT))
        src = dict(COPIES).get(k)
        e = dict(bytes=f.stat().st_size, sha256=sha(f))
        if src is not None:
            e.update(source=rel(src), source_sha256=sha(src))
        elif rel(f) in built:
            e.update(source=built[rel(f)]["source"])
        if f.suffix == ".step":
            e["step"] = step_census(f)
        files[k] = e
    (OUT / "manifest.json").write_text(json.dumps(dict(when=time.strftime("%Y-%m-%d %H:%M"), built_by="scripts/cad_v2_assets.py",
                                                       files=files), indent=1) + "\n")
    for k, e in files.items():
        print(f"{k:42s} {e['bytes'] / 1e3:8.0f} kB  {json.dumps(e.get('step', ''))}")


if os.environ.get("CADV2_CMD") == "step":           # freecadcmd runs the file as a module
    fc_step()
elif __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"arms": cmd_arms, "check": cmd_check, "copy": lambda: cmd_copy("--check" in sys.argv)}[cmd]()
