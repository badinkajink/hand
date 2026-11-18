#!/usr/bin/env python3
"""The v4b fingertip as exact STEP for Onshape: the DIP link b v4, the cassette 3T, the top cover TW15H3 trimmed as on the
hand, the pad 05R4 rebuilt as an exact solid, its four magnets and the ReSkin 5X board with J1 and the Qwiic plug, all in
frame D, the coordinates of the original DIP link's Part Studio.

Owner, 2026-11-18: "are the claude/simulation designed fingertip parts (new dip links, casettes, fingertips themselves)
ready for export to onshape? i want to update the assembly and cad document"; earlier, 2026-10-26: "try to check again if
there are any easy ways to import the new parts to onshape natively ... as long as the step is a simple geometry it should
be fine mating stuff".

assets/cad_v2 had exact STEP of the link, the cassette and the cover since 2026-10-26, and the pad only as print STL and
3MF. The pad here is built from its design: the 11/22 half-ellipsoid of tip_codesign_geometry (semi-axes 8.5 mm deep,
7.4 mm across, 11 mm distal and 22 mm proximal of the apex, the section running straight back behind the equator), cut at
the cassette's face x_F = X_SLOT, less dip_link_concept_b's rail grooves at 0.05 mm per flank and screw-head notches
(raised seats), metric_magnet_prints' four upright slots of four_y2p5_z5 and dip_b_tip_cap's 1.6 mm pin hole with its
0.5 mm mouth chamfer. The engraved ID (05R4, 0.4 mm deep in the mount face) is left out. The cover is
assets/cad_v2/top_cover_TW15H3.step less everything palmar of x_F = -0.10, the trim of the caps on the hand
(ur5_cb3_cell.CAP_TRIM_X; owner, 2026-10-27: all tips trimmed).

Commands, from the repository root:

  params  the design numbers from the modules, on the v4 parameters -> logs/20261118-onshape_fingertip/params.json
              unset VIRTUAL_ENV; ~/miniconda3/envs/femtip/bin/python scripts/cad/fingertip_v4b_onshape_export.py params
  step    inside freecadcmd: every solid in frame D -> OUT/20261118-fingertip_v4b_05R4.step (one file, one named part per
          solid), a fine STL of each -> logs/20261118-onshape_fingertip/stl/, solids and overlaps -> .../step.json
              ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/bin:/bin \\
                  OVF_CMD=step OVF_REPO=$PWD freecadcmd scripts/cad/fingertip_v4b_onshape_export.py
  check   the pad against its print STL (both ways), the parts' frame against the three DIP links of the owner's Onshape
          export of 2026-11-04 (assets/cad_v3, through the cad_v3 extract), and the record OUT/20261118-fingertip_v4b_05R4.json
              ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- env -u VIRTUAL_ENV ~/miniconda3/envs/femtip/bin/python \
                  scripts/cad/fingertip_v4b_onshape_export.py check

Frames: F is the pipelines' finger frame (x palmar-, y along the PIP axis, z along the finger, the tip at z_F < 0); D is
x_D = -x_F, y_D = y_F + 21.25, z_D = -z_F - 51.55 (cad_v2_assets.T_D_FROM_F), the PIP axis along y_D through the origin.
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
    REPO = Path(os.environ.get("OVF_REPO", os.getcwd()))

DATE = "20261118"
OUT = REPO / "assets/cad_v2/20261118-onshape_fingertip"
STEM = f"{DATE}-fingertip_v4b_05R4"
WORK = REPO / "logs/20261118-onshape_fingertip"
PARAMS4 = REPO / "docs/experiments/20261018-dip_link_board_mount/params/20261021-v4_concept_b.json"
CAD2 = REPO / "assets/cad_v2"
PAD_STL = CAD2 / "fingertip_prints/pad_05R4_print.stl"
T_D_FROM_F = [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 21.25], [0.0, 0.0, -1.0, -51.55], [0.0, 0.0, 0.0, 1.0]]
GROOVE_FIT = 0.05                                   # pad 05R4: grooves 0.05 mm inside the nominal rail per flank
CAP_TRIM_X_F = -0.10                                # the caps on the hand keep x_F >= -0.10
TIP_KEY = "tcd_ell7.4x8.5x11x22"
LAYOUT = "four_y2p5_z5"
# the parts in the order of the STEP file: (name in the file and in Onshape, key)
PARTS = [("DIP link b v4", "dip_link"), ("Cassette 3T", "cassette"), ("Top cover TW15H3 trimmed", "cover"),
         ("Pad 05R4 (TPU 85A)", "pad"), ("Magnet", "magnet0"), ("Magnet", "magnet1"), ("Magnet", "magnet2"),
         ("Magnet", "magnet3"), ("ReSkin 5X board", "board"),
         ("J1 header and Qwiic plug", "j1")]


def rel(p):
    return str(Path(p).resolve().relative_to(REPO))


# ------------------------------------------------------------------------------------------------------------- params
def cmd_params():
    """The numbers the solids are built from, read from the design modules on the v4 parameters."""
    os.environ["DIPB_PARAMS"] = str(PARAMS4)
    sys.path.insert(0, str(REPO / "scripts"))
    import _paths  # noqa: F401  every folder of scripts/ and src/ on sys.path
    import dip_link_concept_b as B
    import dip_b_tip_cap as TC
    import metric_magnet_prints as MP
    import curved_tips_print as CP
    import dip_b_four_compact_tip as FC
    import tip_codesign_geometry as TG
    import tpu_tip_design as TD
    B.apply(*B.load_params(PARAMS4))              # the v4 parameters again, after any module above applied its own
    assert B.solid_walls() and B.DESIGN.get("qwiic") == "j1_kept"
    sh = TG.shape(TIP_KEY)
    ellipse = dict(x_apex_F=-TG.REACH * 1e3, x_centre_F=-sh.xc * 1e3, a_x=sh.ax * 1e3, a_y=sh.ay * 1e3, a_z_distal=sh.az * 1e3,
                   a_z_proximal=sh.azp * 1e3, z_apex_F=B.TIP["z_mount"] - TG.Z_MOUNT * 1e3)
    # the tip frame of the print sheet maps to F by tip_to_F (upright): x_F = -x_p - 2.05, z_F = TIP z_mount - z_p
    assert abs(ellipse["z_apex_F"] - B.Z_CONTACT) < 1e-9 and abs(-TD.X_FACE - 2.05 - ellipse["x_apex_F"]) < 1e-9
    Dm, t = MP.MAGNET
    r = 2.9 / 2                                      # the 2.9 mm seat
    x_hi_p = TD.X_FACE - 7.5                         # the magnet's palmar face, 7.5 mm under the apex
    x_lo_p = x_hi_p - t - MP.GAP
    slots, magnets = [], []
    for y, z in FC.four_layout(LAYOUT):
        yp, zp = MP.upright_xy(y, z)
        top = zp + Dm / 2 + (Dm / 2 - r) + MP.GAP
        _, z_bot, _ = CP.pause_layer(top)
        slots.append(dict(y_F=yp, z_seat_F=B.TIP["z_mount"] - zp, z_top_F=B.TIP["z_mount"] - z_bot, r=r,
                          x_F=sorted([-x_lo_p - 2.05, -x_hi_p - 2.05])))
        magnets.append(dict(y_F=yp, z_F=B.TIP["z_mount"] - (zp + Dm / 2 - r), r=Dm / 2,
                            x_F=sorted([-x_hi_p - 2.05, -x_hi_p - 2.05 + t])))
    sec = [(y, B.X_SLOT + 0.5 if x > B.X_SLOT else x) for y, x in B.rail_section(-GROOVE_FIT, B.GROOVE_FLOOR)]
    notches = []
    for y, z in B.screw_yz():
        hn = B.HEAD_NOTCH_H + (B.rp("seat_raise") if z > B.Z_CONTACT else 0.0)
        notches.append(dict(y_F=y, z_F=z, r=B.HEAD_NOTCH_D / 2, x_F=[B.X_SLOT - hn, B.X_SLOT + 1.0]))
    uvz = B.uv_z
    board = dict(x_F=[B.X_COMP, B.X_SEAT],
                 boxes=[[[-10.0, uvz(10.05)], [10.0, uvz(-10.10)]], [[-5.0, uvz(-10.10) - 0.01], [5.0, uvz(B.TAB_CUT_V)]]],
                 holes=[dict(y_F=y, z_F=z, d=B.HOLE_D) for y, z in B.screw_yz()],
                 # the MLX90393s at the design's CHIP_H (0.9 mm, tops on the cassette's face), not the catalogue's 1.0 mm
                 parts={k: [[B.X_COMP - (B.CHIP_H if k.startswith("U") else h), u0, uvz(v1)], [B.X_COMP + 0.001, u1, uvz(v0)]]
                        for k, (u0, u1, v0, v1, h) in B.board_parts().items()})
    j1 = [dict(name=n, lo=list(lo), hi=list(hi)) for n, lo, hi in B.j1_boxes_b()]
    rec = dict(when=time.strftime("%Y-%m-%d %H:%M"), params=rel(PARAMS4), tip=TIP_KEY, layout=LAYOUT, groove_fit=GROOVE_FIT,
               X_SLOT=B.X_SLOT, TIP=B.TIP, Z_CONTACT=B.Z_CONTACT, ellipse=ellipse,
               box=dict(x=[ellipse["x_apex_F"], B.X_SLOT], y=[-B.TIP["y_half"], B.TIP["y_half"]],
                        z=[B.TIP["z_end"], B.TIP["z_mount"]]),
               grooves=dict(section_yx=sec, z=[B.RAIL_Z[0] - 0.2, B.RAIL_Z[1] + 0.2]), notches=notches,
               slots=slots, magnets=magnets,
               pin_hole=dict(x_F=TC.CAP["pin_x"], y_F=0.0, d=1.6, depth=3.0, chamfer=TC.CAP2["hole_chamfer"],
                             z_end_F=B.TIP["z_end"]),
               board=board, j1=j1, cap_trim_x_F=CAP_TRIM_X_F, T_D_from_F=T_D_FROM_F)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "params.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("ellipse", "box", "slots", "magnets", "pin_hole")}, indent=1))


# --------------------------------------------------------------------------------------------------------------- step
def cmd_step():
    import FreeCAD
    import Import
    import Mesh
    import MeshPart
    import Part
    V = FreeCAD.Vector
    p = json.loads((WORK / "params.json").read_text())
    M = FreeCAD.Matrix(*[float(v) for row in T_D_FROM_F for v in row])
    Minv = M.inverse()

    def box(lo, hi):
        return Part.makeBox(*(float(h - l) for l, h in zip(lo, hi)), V(*(float(c) for c in lo)))

    def cyl_x(y, z, r, x0, x1):
        return Part.makeCylinder(r, x1 - x0, V(x0, y, z), V(1, 0, 0))

    def scaled(shape, sx, sy, sz, at):
        """shape (made about the origin) scaled by (sx, sy, sz) and moved to at: B-spline faces for an ellipsoid."""
        s = shape.transformGeometry(FreeCAD.Matrix(sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, 0, 0, 0, 1))
        s.translate(V(*at))
        return s

    def pad():
        e, bx = p["ellipse"], p["box"]
        xc, zc = e["x_centre_F"], e["z_apex_F"]
        big = 50.0
        halves = []
        for az, zlo, zhi in ((e["a_z_distal"], zc - big, zc), (e["a_z_proximal"], zc, zc + big)):
            # the palmar half: an ellipsoid about (xc, 0, zc), x deep toward -x_F; behind the equator the section runs
            # straight back: an elliptic cylinder along x
            ell = scaled(Part.makeSphere(1.0), e["a_x"], e["a_y"], az, (xc, 0.0, zc))
            cyl = scaled(Part.makeCylinder(1.0, 1.0, V(0, 0, 0), V(1, 0, 0)), bx["x"][1] - xc + 1.0, e["a_y"], az, (xc, 0.0, zc))
            halves.append(ell.fuse(cyl).common(box((-big, -big, zlo), (big, big, zhi))))
        # the envelope's width (+-7.4 mm) equals a_y and would touch the body along its equator: the box is wider, a_y bounds y
        body = halves[0].fuse(halves[1]).common(box((bx["x"][0] - 1.0, bx["y"][0] - 1.0, bx["z"][0]),
                                                    (bx["x"][1], bx["y"][1] + 1.0, bx["z"][1])))
        body = body.removeSplitter()
        cuts = []
        g = p["grooves"]
        for s in (-1, 1):
            pts = [V(float(x), float(s * y), float(g["z"][0])) for y, x in g["section_yx"]]
            face = Part.Face(Part.makePolygon(pts + [pts[0]]))
            cuts.append(face.extrude(V(0, 0, float(g["z"][1] - g["z"][0]))))
        for n in p["notches"]:
            cuts.append(cyl_x(n["y_F"], n["z_F"], n["r"], n["x_F"][0], n["x_F"][1]))
        for s in p["slots"]:
            x0, x1 = s["x_F"]
            cuts.append(cyl_x(s["y_F"], s["z_seat_F"], s["r"], x0, x1))
            cuts.append(box((x0, s["y_F"] - s["r"], s["z_top_F"]), (x1, s["y_F"] + s["r"], s["z_seat_F"])))
        h = p["pin_hole"]
        r, c, ze = h["d"] / 2, h["chamfer"], h["z_end_F"]
        cuts.append(Part.makeCone(r + c + 1.0, r, c + 1.0, V(h["x_F"], h["y_F"], ze - 1.0), V(0, 0, 1)))
        cuts.append(Part.makeCylinder(r, h["depth"] - c + 0.01, V(h["x_F"], h["y_F"], ze + c - 0.01), V(0, 0, 1)))
        out = body.cut(cuts).removeSplitter()
        return (out.Solids[0] if len(out.Solids) == 1 else out), body      # one solid: the outer shell and four slot voids

    def board():
        b = p["board"]
        x0, x1 = b["x_F"]
        fr4 = box((x0, b["boxes"][0][0][0], b["boxes"][0][0][1]), (x1, b["boxes"][0][1][0], b["boxes"][0][1][1]))
        fr4 = fr4.fuse(box((x0, b["boxes"][1][0][0], b["boxes"][1][0][1]), (x1, b["boxes"][1][1][0], b["boxes"][1][1][1])))
        fr4 = fr4.cut([cyl_x(hh["y_F"], hh["z_F"], hh["d"] / 2, x0 - 0.1, x1 + 0.1) for hh in b["holes"]])
        comps = [box(lo, hi) for lo, hi in b["parts"].values()]
        return fr4.fuse(comps).removeSplitter()

    def j1():
        boxes = [box(q["lo"], q["hi"]) for q in p["j1"]]
        return boxes[0].fuse(boxes[1:]).removeSplitter()

    def to_D(s, T=M):
        s = s.copy()
        s.transformShape(T, True)                 # True: the geometry moves (a placement alone is dropped by Import.export)
        return s

    t0 = time.time()
    solids = {}
    for key, f in (("dip_link", "dip_link_v4b.step"), ("cassette", "cassette_3T.step"), ("cover", "top_cover_TW15H3.step")):
        s = Part.read(str(CAD2 / f))
        solids[key] = s if len(s.Solids) != 1 else s.Solids[0]
    cover_F = to_D(solids["cover"], Minv)
    trim_F = cover_F.common(box((p["cap_trim_x_F"], -50, -200), (50, 50, 0)))
    removed = solids["cover"].Volume - trim_F.Volume
    solids["cover"] = to_D(trim_F.removeSplitter())
    pad_F, body_F = pad()
    solids["pad"] = to_D(pad_F)
    for k, m in enumerate(p["magnets"]):
        solids[f"magnet{k}"] = to_D(cyl_x(m["y_F"], m["z_F"], m["r"], m["x_F"][0], m["x_F"][1]))
    solids["board"] = to_D(board())
    solids["j1"] = to_D(j1())
    print(f"solids built [{time.time() - t0:.0f} s]", flush=True)

    doc = FreeCAD.newDocument("fingertip_v4b_05R4")          # the STEP's root assembly
    objs, rows = [], []
    (WORK / "stl").mkdir(parents=True, exist_ok=True)
    for name, key in PARTS:
        s = solids[key]
        if key.startswith("magnet"):
            # 05R4's checkerboard: north toward the palmar face at y_F < 0 in the proximal row and y_F > 0 in the distal row
            m = p["magnets"][int(key[-1])]
            prox = m["z_F"] > p["Z_CONTACT"]
            north = (m["y_F"] < 0) == prox
            name = (f"Magnet N52 3x2 {'proximal' if prox else 'distal'} {'+' if m['y_F'] > 0 else '-'}y, "
                    f"{'north' if north else 'south'} pole to the pad face")
        o = doc.addObject("Part::Feature", key)
        o.Label = name
        o.Shape = s
        objs.append(o)
        census = {}
        for fc in s.Faces:
            census[fc.Surface.__class__.__name__] = census.get(fc.Surface.__class__.__name__, 0) + 1
        bb = s.BoundBox
        rows.append(dict(name=name, key=key, valid=bool(s.isValid()), solids=len(s.Solids), faces=len(s.Faces), census=census,
                         volume_mm3=round(s.Volume, 2), bbox_D=[[round(bb.XMin, 3), round(bb.YMin, 3), round(bb.ZMin, 3)],
                                                               [round(bb.XMax, 3), round(bb.YMax, 3), round(bb.ZMax, 3)]]))
        mesh = MeshPart.meshFromShape(Shape=s, LinearDeflection=0.01, AngularDeflection=0.1)
        mesh.write(str(WORK / "stl" / f"{key}.stl"))
        print(f"  {name}: valid {s.isValid()}, {len(s.Solids)} solid(s), {len(s.Faces)} faces {census}, {s.Volume:.2f} mm3",
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{STEM}.step"
    Import.export(objs, str(dst))
    back = Part.read(str(dst))
    # overlaps between parts: the press fits (pad on the rails, cover's blocks on the pad) and anything unintended
    keys = [k for _, k in PARTS]
    over = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if not solids[a].BoundBox.intersect(solids[b].BoundBox):
                continue
            v = solids[a].common(solids[b]).Volume
            if v > 1e-4:
                over.append(dict(a=a, b=b, volume_mm3=round(v, 4)))
    rec = dict(when=time.strftime("%Y-%m-%d %H:%M"), step=rel(dst), bytes=dst.stat().st_size,
               read_back=dict(solids=len(back.Solids), valid=bool(back.isValid())), parts=rows, overlaps=over,
               cover_trim=dict(x_F=p["cap_trim_x_F"], removed_mm3=round(removed, 2)),
               pad_body_mm3=round(body_F.Volume, 2))
    (WORK / "step.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(dict(read_back=rec["read_back"], overlaps=over, cover_trim=rec["cover_trim"]), indent=1), flush=True)
    print(f"{rel(dst)}: {dst.stat().st_size / 1e6:.2f} MB [{time.time() - t0:.0f} s]", flush=True)


# -------------------------------------------------------------------------------------------------------------- check
def cmd_check():
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation

    def to_surface(points, mesh):
        """Distance of each point to the mesh's surface, in chunks of 4,096 points (CLAUDE.md: unchunked queries on fine
        meshes have taken 24 GB)."""
        q = trimesh.proximity.ProximityQuery(mesh)
        return np.concatenate([q.on_surface(points[i:i + 4096])[1] for i in range(0, len(points), 4096)])
    p = json.loads((WORK / "params.json").read_text())
    st = json.loads((WORK / "step.json").read_text())
    T = np.array(T_D_FROM_F)
    # the pad: its print STL (print frame) -> F by tip_to_F (upright) -> D, against the exact pad's fine tessellation
    pr = trimesh.load(PAD_STL, force="mesh")
    Vp = np.asarray(pr.vertices, float)
    VF = np.column_stack([-Vp[:, 0] - 2.05, Vp[:, 1], p["TIP"]["z_mount"] - Vp[:, 2]])
    VD = VF @ T[:3, :3].T + T[:3, 3]
    printed = trimesh.Trimesh(VD, pr.faces, process=True)
    exact = trimesh.load(WORK / "stl/pad.stl", force="mesh")
    a = trimesh.sample.sample_surface(exact, 40000, seed=1)[0]
    b = trimesh.sample.sample_surface(printed, 40000, seed=2)[0]
    da = to_surface(a, printed)                   # exact -> print
    db = to_surface(b, exact)                     # print -> exact
    # the engraving: the print's ID in the mount face (z_D near -z_mount - 51.55, 0.4 mm deep) has no counterpart
    z_mount_D = -p["TIP"]["z_mount"] - 51.55
    eng = np.abs(b[:, 2] - z_mount_D) < 0.45
    pad = dict(print_stl=rel(PAD_STL), volume_print_mm3=round(float(printed.volume), 2),
               volume_exact_brep_mm3=next(r["volume_mm3"] for r in st["parts"] if r["key"] == "pad"),
               exact_to_print_mm=dict(p50=round(float(np.percentile(da, 50)), 4), p99=round(float(np.percentile(da, 99)), 4),
                                      max=round(float(da.max()), 4)),
               print_to_exact_mm=dict(p50=round(float(np.percentile(db, 50)), 4), p99=round(float(np.percentile(db, 99)), 4),
                                      max=round(float(db.max()), 4),
                                      max_outside_engraving=round(float(db[~eng].max()), 4)),
               engraving_note="the print's ID 05R4 is engraved 0.4 mm deep in the mount face; the exact pad has none")
    # frame D against the owner's Onshape export: each DIP link's local shape (its global mesh moved back by its placement)
    ext = json.loads((REPO / "logs/20261104-cad_v3/extract/parts.json").read_text())["parts"]
    inv = json.loads((REPO / "assets/cad_v3/inventory.json").read_text())
    ref = trimesh.load(REPO / "assets/cad_v1/fingertip_mount_DIP_link.stl", force="mesh")
    rs = trimesh.sample.sample_surface(ref, 20000, seed=3)[0]
    fingers = []
    for r in ext:
        if r["file"] != "hand" or not any(q["name"] == r["name"] and q["kind"] == "dip_link" for q in inv):
            continue
        m = trimesh.load(REPO / r["stl"], force="mesh")
        R = Rotation.from_quat(r["placement"]["quat_xyzw"]).as_matrix()
        t = np.array(r["placement"]["pos"])
        local = (np.asarray(m.vertices) - t) @ R
        lm = trimesh.Trimesh(local, m.faces, process=False)
        ls = trimesh.sample.sample_surface(lm, 20000, seed=4)[0]
        d1, d2 = to_surface(ls, ref), to_surface(rs, lm)
        T_A = np.eye(4)
        T_A[:3, :3], T_A[:3, 3] = R, t
        finger = next(q["finger"] for q in inv if q["name"] == r["name"] and q["file"] == "hand")
        fingers.append(dict(finger=finger, solid=r["name"], label=r["label"], T_A_from_D=np.round(T_A, 6).tolist(),
                            local_vs_cad_v1_mm=dict(p99=round(float(np.percentile(np.r_[d1, d2], 99)), 4),
                                                    max=round(float(np.r_[d1, d2].max()), 4))))
    rec = dict(when=time.strftime("%Y-%m-%d %H:%M"), generator="scripts/cad/fingertip_v4b_onshape_export.py",
               owner="2026-11-18: 'are the claude/simulation designed fingertip parts (new dip links, casettes, fingertips "
                     "themselves) ready for export to onshape? i want to update the assembly and cad document'",
               frame="D: the original DIP link's Part Studio frame (assets/cad_v1/fingertip_mount_DIP_link.stl), mm; the PIP "
                     "axis is the y axis; x_D = -x_F, y_D = y_F + 21.25, z_D = -z_F - 51.55",
               step=st["step"], bytes=st["bytes"], read_back=st["read_back"], parts=st["parts"], overlaps=st["overlaps"],
               cover_trim=st["cover_trim"], pad_check=pad, onshape_export_dip_links=fingers,
               sources=dict(dip_link="assets/cad_v2/dip_link_v4b.step", cassette="assets/cad_v2/cassette_3T.step",
                            cover="assets/cad_v2/top_cover_TW15H3.step, trimmed at x_F -0.10",
                            pad=f"{TIP_KEY} with {LAYOUT}, groove fit {GROOVE_FIT} mm, raised seats, pin hole 1.6 mm "
                                "(the design of pad 05R4, assets/cad_v2/fingertip_prints/pad_05R4_print.stl)",
                            board="dip_link_concept_b.board_parts and j1_boxes_b on the v4 parameters", params=p["params"]))
    (OUT / f"{STEM}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(dict(pad_check=pad, fingers=[(f["finger"], f["local_vs_cad_v1_mm"]) for f in fingers]), indent=1))


# ------------------------------------------------------------------------------------------------------------- render
COLOURS = dict(dip_link=(0.78, 0.78, 0.80), cassette=(0.93, 0.93, 0.90), cover=(0.55, 0.62, 0.75), pad=(0.20, 0.20, 0.22),
               board=(0.15, 0.45, 0.20), j1=(0.85, 0.85, 0.85), magnet=(0.75, 0.55, 0.20))
# the exploded view: each part moved palmar (+x_D) and the cover distal (+z_D), mm
EXPLODE = dict(board=(9.0, 0.0, 0.0), j1=(9.0, 0.0, 0.0), cassette=(18.0, 0.0, 0.0), pad=(30.0, 0.0, 0.0),
               magnet=(42.0, 0.0, 0.0), cover=(30.0, 0.0, 12.0))


def cmd_render():
    """OUT/<stem>.png: the parts assembled, the original DIP link and TPU block of the owner's export of 2026-11-04 (moved
    into frame D by that DIP link's placement), and the parts exploded palmar and distal (matplotlib: one collection per
    panel, so that its faces are drawn back to front, Lambert shading)."""
    import numpy as np
    import trimesh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial.transform import Rotation
    light = np.array([0.6, -0.35, 0.72])
    light /= np.linalg.norm(light)

    def load(key):
        return trimesh.load(WORK / "stl" / f"{key}.stl", force="mesh")

    def panel(ax, items, title, elev, azim, box=None):
        tris, cols = [], []
        for m, colour, shift in items:
            tris.append(m.triangles + np.asarray(shift, float))
            shade = 0.3 + 0.7 * np.abs(m.face_normals @ light)
            cols.append(np.clip(np.outer(shade, colour), 0, 1))
        tri, col = np.vstack(tris), np.vstack(cols)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolors="none"))
        pts = tri.reshape(-1, 3) if box is None else np.vstack([np.vstack([m.vertices for m, _, _ in b]) for b in box])
        lo, hi = pts.min(0), pts.max(0)
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title, fontsize=15)

    keys = [k for _, k in PARTS]
    colour = lambda k: COLOURS["magnet" if k.startswith("magnet") else k]   # noqa: E731
    shift = lambda k: EXPLODE.get("magnet" if k.startswith("magnet") else k, (0, 0, 0))   # noqa: E731
    # the owner's export: thumb finger's DIP link (solid 009) and TPU block (012), back into the DIP link's frame D
    ext = {r["name"]: r for r in json.loads((REPO / "logs/20261104-cad_v3/extract/parts.json").read_text())["parts"]
           if r["file"] == "hand"}
    pl = ext["009"]["placement"]
    R, t = Rotation.from_quat(pl["quat_xyzw"]).as_matrix(), np.array(pl["pos"])
    old = []
    for name, c in (("009", COLOURS["dip_link"]), ("012", COLOURS["pad"])):
        m = trimesh.load(REPO / ext[name]["stl"], force="mesh")
        old.append((trimesh.Trimesh((np.asarray(m.vertices) - t) @ R, m.faces, process=False), c, (0, 0, 0)))
    new = [(load(k), colour(k), (0, 0, 0)) for k in keys if not k.startswith("magnet")]
    fig = plt.figure(figsize=(15, 6.4), dpi=110)
    # the first two panels at one scale, so that the new link's 9.75 mm longer neck shows
    panel(fig.add_subplot(1, 3, 1, projection="3d"), old, "Original DIP link and TPU block", 20, 28, box=(old, new))
    panel(fig.add_subplot(1, 3, 2, projection="3d"), new, "New DIP link, cassette, pad, cover", 20, 28, box=(old, new))
    panel(fig.add_subplot(1, 3, 3, projection="3d"), [(load(k), colour(k), shift(k)) for k in keys], "Exploded view", 14, 62)
    fig.text(0.5, 0.025, "Frame D, the original DIP link's Part Studio frame: PIP axis along y_D, finger along +z_D, palmar "
             "face toward +x_D.", ha="center", fontsize=13)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.07, wspace=0.0)
    dst = OUT / f"{STEM}.png"
    fig.savefig(dst)
    plt.close(fig)
    print(rel(dst))


if __name__ == "__main__" or os.environ.get("OVF_CMD"):
    cmd = os.environ.get("OVF_CMD") or (sys.argv[1] if len(sys.argv) > 1 else "")
    {"params": cmd_params, "step": cmd_step, "check": cmd_check, "render": cmd_render}[cmd]()
