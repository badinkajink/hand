#!/usr/bin/env python3
"""STEP files for designing the DIP link in Onshape (owner, 2026-10-18 16:36: "i think concept A and B are both
worthwhile, but need the qwiic connector correction. im also more than happy to design this in myself, but not sure how to
import these parts to onshape natively").

Onshape translates STEP into solid parts that its features can edit; STL arrives as a mesh. Every part here is a named
solid in the coordinates of assets/cad_v1/fingertip_mount_DIP_link.stl (frame D: the PIP axis is the y axis, the finger
runs along +z, the palmar face toward +x), so the parts line up with the owner's DIP link when that STL came from its
Part Studio. Frame F of spec.json maps to D by x_D = -x_F, y_D = y_F + 21.25, z_D = -z_F - 51.55.

    onshape/20261018-dip_link_onshape_parts.step     exact solids: the ReSkin 5X board as supplied (tab, J1 and the mated
                                                     Qwiic plug) at the placements of concepts a and b, concept a's final
                                                     parts, concept b's new body and cover
    onshape/20261018-dip_link_onshape_tips.step      faceted solids (coplanar facets merged): concept a's block tip with
                                                     its key channels, concept b's block pad
    onshape/20261018-dip_link_onshape_context_stl.zip  meshes for reference: the original DIP link, and the PIP servo and
                                                     the MCP link at PIP -18, 0 and 92 deg

As faceted STEP the context parts take 141 MB (the DIP link's fillets alone keep 13,500 faces), so they stay meshes.
STEP files over 1.5 MB and the context meshes are zipped so that git LFS stores them (docs/**/*.zip).

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- taskset -c 4-5"
    $RG env DLO_REPO=$PWD freecadcmd scripts/dip_link_onshape_export.py
"""
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(os.environ["DLO_REPO"])                  # freecadcmd executes the file without __file__
os.environ.setdefault("DIPA_REPO", str(REPO))
os.environ.setdefault("DLF_REPO", str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

import FreeCAD  # noqa: E402
import Import  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402

import dip_link_concept_a as A  # noqa: E402
import dip_link_board_mount as DLF  # noqa: E402  (sets A.clamp_plate to the final's, with the Qwiic pocket)

STUDY = REPO / "docs/experiments/20261018-dip_link_board_mount"
OUT = STUDY / "onshape"
PFX = "20261018-dip_link_onshape"
CAD_A = STUDY / "final/cad"
CAD_B = STUDY / "concept_b/cad"
X_COMP_B = 1.35                                     # concept b: the FR-4's component face (dip_link_concept_b.X_COMP)
PIP_C = (0.0, 0.0, -51.55)                          # a point on the PIP axis, F
POSES = (-18.0, 0.0, 92.0)                          # PIP range ends and zero (build_cad_v1_finger.ROM)
V = FreeCAD.Vector


def T_D_from_F():
    """Inverse of spec.json's transforms_F_from_stl['dip_link'] (x_F = -x_D, y_F = y_D - 21.25, z_F = -z_D - 51.55), with
    the rotation's 8e-6 entries set to zero so that the transform stays a rigid placement of the exact solids."""
    T = A._T_F_from_stl("dip_link")
    R = np.round(T[:3, :3])
    t = -R.T @ T[:3, 3]
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R.T, t
    return M


def to_D(shape, M):
    s = shape.copy()
    s.transformShape(FreeCAD.Matrix(*[float(v) for v in M.ravel()]))
    return s


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Part.makeBox(*(float(d) for d in hi - lo), V(*(float(c) for c in lo)))


def board_box_F(lo, hi, x_comp):
    """A box given in the board frame (u, v, w; w the height above the component face, toward the tip) in F."""
    P = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])], float)
    Q = np.column_stack([x_comp - P[:, 2], P[:, 0], A.Z_C - P[:, 1]])
    return box(Q.min(0), Q.max(0))


def board_as_supplied(x_comp):
    """The ReSkin 5X as delivered: FR-4 with its 10 x 9 mm tab, the four 2.40 mm holes and the populated parts (one
    solid), J1 (JST SH side entry, mating face toward the tab's end), and the mated SHR-04V-S with 3 mm of cable."""
    it = A.board_items()
    outline = [(-10, -10.1), (-5, -10.1), (-5, -19.1), (5, -19.1), (5, -10.1), (10, -10.1), (10, 10.051), (-10, 10.051)]
    pts = [V(x_comp, u, A.Z_C - v) for u, v in outline]
    fr4 = Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(V(A.P["fr4"], 0, 0))
    for y, z, d in it["holes"]:
        fr4 = fr4.cut(Part.makeCylinder(d / 2, A.P["fr4"] + 0.2, V(x_comp - 0.1, y, z), V(1, 0, 0)))
    comps = [box((x_comp - v["h"], v["y"] - v["sy"] / 2, v["z"] - v["sz"] / 2), (x_comp + 0.01, v["y"] + v["sy"] / 2,
                                                                                 v["z"] + v["sz"] / 2))
             for v in it["parts"].values()]
    board = fr4.fuse(comps).removeSplitter()
    j1 = {name: board_box_F(lo, hi, x_comp) for name, lo, hi in DLF.j1_parts_board() if name != "tab_fr4"}
    plug = j1["plug"].fuse([s for k, s in j1.items() if k.startswith("wire")]).removeSplitter()
    return board, j1["j1_header"], plug


def exact_parts():
    """(name, solid in F) of every exact part."""
    K = A.BrepK()
    out = []
    for label, x_comp in (("concept a", A.levels()["x_cf"]), ("concept b", X_COMP_B)):
        b, j1, plug = board_as_supplied(x_comp)
        out += [(f"Board ReSkin 5X as supplied ({label} placement)", b),
                (f"J1 Qwiic header on the board ({label} placement)", j1),
                (f"Qwiic plug and cable on J1 ({label} placement)", plug)]
    for name, f in (("a DIP link new features", "dip_link_new_features"), ("a clamp plate with Qwiic pocket", "clamp_plate"),
                    ("a tip channel cutter", "tip_cutter")):
        out.append((name, Part.read(str(CAD_A / f"20261018-dip_link_final_{f}.step"))))
    board_a, _ = A.board_model(K)        # the 0.30 mm wires from J1's pads (A.wires_model) fuse to an invalid solid
    out.append(("a board with tab cut to 4.9 mm and J1 removed", board_a))
    q = {name: box(lo, hi) for name, lo, hi in DLF.qwiic_header_final()[0]}
    out.append(("a Qwiic header SM04B-SRSS-TB on clamp plate", q["q_header"].fuse(q["q_leads"]).removeSplitter()))
    out.append(("a Qwiic plug and cable", q["q_plug"].fuse([s for k, s in q.items() if k.startswith("q_wire")])))
    out.append(("a pigtail on clamp plate back", q["q_pigtail"].fuse(q["q_pigtail_turn"]).removeSplitter()))
    count = {}
    names = {"cs_screw": "a M1.6x4 countersunk screw", "nut": "a M1.6 nut", "m2_screw": "a M2x5 screw"}
    for kind, s in A.fasteners(K):
        count[kind] = count.get(kind, 0) + 1
        out.append((f"{names[kind]} {count[kind]}", s))
    for name, f in (("b DIP link new body", "dip_link_new_body"), ("b cover", "cover")):
        out.append((name, Part.read(str(CAD_B / f"20261018-concept_b_{f}.step"))))
    return out


def faceted(path, M=None):
    """A solid from a closed STL: the largest closed component is the skin, the others (magnet pockets) are voids;
    coplanar facets are merged."""
    m = Mesh.Mesh(str(path))
    if M is not None:
        m.transform(FreeCAD.Matrix(*[float(v) for v in np.asarray(M, float).ravel()]))
    solids = []
    for comp in m.getSeparateComponents():
        if comp.CountFacets < 20:
            continue
        sh = Part.Shape()
        sh.makeShapeFromMesh(comp.Topology, 0.01)
        so = Part.makeSolid(Part.Shell(sh.Faces))
        if so.Volume < 0:
            so.reverse()
        solids.append(so)
    solids.sort(key=lambda s: -s.Volume)
    so = solids[0]
    for cav in solids[1:]:
        so = so.cut(cav)
    return so.removeSplitter()


def pip_T(q_deg):
    """4 x 4 form of dip_link_geometry.rot_pip: a part fixed to the MCP link as seen from the DIP link at PIP angle q, a
    turn by -q about the PIP axis."""
    phi = math.radians(-q_deg)
    c, s = math.cos(phi), math.sin(phi)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.array(PIP_C) - R @ np.array(PIP_C)
    return T


def T_F_from_stl(part, src):
    G = json.loads((REPO / "assets/mjcf/experimental/20261013-cad_v1/assembly.json").read_text())
    return np.array(G["T_F_from_part_design_mm"][part]) @ np.array(G["T_part_design_from_stl"][src])


def tip_parts():
    return [("a tip block4 with key channels", faceted(CAD_A / "20261018-dip_link_final_tip_block4_F.stl")),
            ("b pad block4", faceted(CAD_B / "20261018-concept_b_pad_block4.stl"))]


def context_stl(M, rec):
    """The original DIP link, and the PIP servo and the MCP link at PIP -18, 0 and 92 deg, as binary STL in D, zipped."""
    cad = REPO / "assets/cad_v1"
    items = [("dip_link_original", cad / "fingertip_mount_DIP_link.stl", T_F_from_stl("dip_link", "dip_link"))]
    for q in POSES:
        items.append((f"pip_servo_pip{q:.0f}", cad / "feetech scs0009.stl", pip_T(q) @ T_F_from_stl("servo_pip", "servo")))
        items.append((f"mcp_link_pip{q:.0f}", cad / "pip_mount_MCP_link.stl", pip_T(q) @ T_F_from_stl("mcp_link", "mcp_link")))
    z = OUT / f"{PFX}_context_stl.zip"
    rows = []
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for stem, src, T in items:
            m = Mesh.Mesh(str(src))
            m.transform(FreeCAD.Matrix(*[float(v) for v in (M @ T).ravel()]))
            dst = OUT / f"{PFX}_{stem}.stl"
            m.write(str(dst))
            zf.write(dst, dst.name)
            bb = m.BoundBox
            rows.append(dict(file=dst.name, source=str(src.relative_to(REPO)), facets=m.CountFacets,
                             bbox_D=[[round(bb.XMin, 2), round(bb.YMin, 2), round(bb.ZMin, 2)],
                                     [round(bb.XMax, 2), round(bb.YMax, 2), round(bb.ZMax, 2)]]))
            dst.unlink()
    rec["context_stl"] = dict(file=str(z.relative_to(REPO)), zip_bytes=z.stat().st_size, meshes=rows)
    print(f"{z.relative_to(REPO)}: {len(rows)} meshes, {z.stat().st_size / 1e6:.2f} MB", flush=True)


def write(parts, stem, M, rec):
    doc = FreeCAD.newDocument("dip_link_" + stem.split("_")[-1])          # the STEP root: dip_link_parts, dip_link_tips
    objs = []
    rows = []
    for name, s in parts:
        sD = to_D(s, M)
        o = doc.addObject("Part::Feature", f"p{len(objs)}")
        o.Label = name
        o.Shape = sD
        objs.append(o)
        bb = sD.BoundBox
        rows.append(dict(name=name, valid=bool(sD.isValid()), solids=len(sD.Solids), faces=len(sD.Faces),
                         volume_mm3=round(sD.Volume, 2),
                         bbox_D=[[round(bb.XMin, 2), round(bb.YMin, 2), round(bb.ZMin, 2)],
                                 [round(bb.XMax, 2), round(bb.YMax, 2), round(bb.ZMax, 2)]]))
        print(f"  {name}: valid {sD.isValid()}, {len(sD.Faces)} faces, {sD.Volume:.1f} mm3", flush=True)
    dst = OUT / f"{stem}.step"
    Import.export(objs, str(dst))
    entry = dict(parts=rows, bytes=dst.stat().st_size)
    if dst.stat().st_size > 1.5e6:
        z = dst.with_suffix(".step.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.write(dst, dst.name)
        dst.unlink()
        entry.update(file=str(z.relative_to(REPO)), zip_bytes=z.stat().st_size)
    else:
        entry["file"] = str(dst.relative_to(REPO))
    rec[stem] = entry
    print(f"{entry['file']}: {entry['bytes'] / 1e6:.2f} MB STEP", flush=True)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    M = T_D_from_F()
    rec = dict(when=time.strftime("%Y-%m-%d %H:%M"), generator="scripts/dip_link_onshape_export.py",
               frame="D: assets/cad_v1/fingertip_mount_DIP_link.stl's coordinates, mm (PIP axis = y axis, finger along +z, "
                     "palmar face toward +x); x_D = -x_F, y_D = y_F + 21.25, z_D = -z_F - 51.55",
               T_D_from_F=np.round(M, 6).tolist())
    print("exact parts", flush=True)
    write(exact_parts(), f"{PFX}_parts", M, rec)
    print(f"tips [{time.time() - t0:.0f} s]", flush=True)
    write(tip_parts(), f"{PFX}_tips", M, rec)
    context_stl(M, rec)
    (OUT / f"{PFX}.json").write_text(json.dumps(rec, indent=1))
    print(f"done [{time.time() - t0:.0f} s]", flush=True)


main()
