#!/usr/bin/env python3
"""Concept c of the DIP-link redesign (2026-10-18): a tip cartridge printed in one job on the Prusa XL, PLA carrier and
85A tip, with the ReSkin 5X board placed at an M601 pause and captured by the PLA printed over it.

Owner, 2026-10-18: "we are going to redesign the DIP linkage entirely so that the board can be as close to the printed
tip as possible, while trying to keep the contact geometry the same ... it's important that the board does not wiggle at
all--youll want something to fasten/sandwich it in place". Specification: docs/experiments/20261018-dip_link_board_mount/
spec.json (scripts/dip_link_spec.py); geometry and checks: scripts/dip_link_geometry.py.

The DIP link is split in two parts.
  body       the clevis arms of assets/cad_v1/fingertip_mount_DIP_link.stl, unchanged, joined by a new bridge whose distal
             face supports the tip's mount face, two shoulders beside a notch for the board's tab, a cable post and a lip.
  cartridge  one print. Tool 1 (PLA) prints the carrier: a 3.0 mm base, rims around a pocket for the board, four pins in
             the board's holes (crush ribs on the two proximal ones), the wings that sit on the shoulders. At pause P1 the
             board, its four-wire pigtail soldered to J1's signal pads, goes onto the pins chips up and a printed spacer
             (as thick as the tallest part) onto the board. Tool 1 then prints a 0.2 mm cap over spacer, rims and pin tops,
             with windows over U1-U5; tool 2 prints the 85A tip on the cap; the magnets go in at pause P2 as before.
The cartridge slides onto the body along +z: its tab region enters the notch, a tower on its -y wing runs under the
body's lip, the tip's mount face meets the bridge, and one brass M2 screw in a slot of the +y wing clamps it.

Frame F (scripts/dip_link_geometry.py): mm, x toward the dorsal side, y along the PIP axis, z up; the straight finger
hangs along -z; the tip's back face at x 6.45, its mount face at z -67.30, the contact point (-10.55, 0, -78.16).
Print frame of the cartridge: z_p = X_BED - x_F (the carrier's dorsal face on the bed), x_p = z_F + 88.71, y_p = y_F.

Subcommands (heavy steps through ~/.claude/bin/resguard.sh on cores 12-15):
  design   parameters, the keep-out ceiling for the bridge -> concept_c/design.json        (femtip env)
  cad      BRep solids, STEP and STL of the new parts (freecadcmd on this file)          (FreeCAD 0.19)
  mesh     body = arms (STL) + bridge; TPU tips; board model; assembly; print frame       (femtip env, manifold3d)
  step     faceted STEP of the meshed parts (body, 85A tips), zipped (freecadcmd on this file)
  check    clearance over the PIP range, gaps between parts, widths, walls, mass          (femtip env)
  sense    gap, field at the chips, force resolution, false force per micrometre          (femtip env)
  render   iso, section through the chips, exploded, print at the board pause            (femtip env, pyvista)
  eval     concept_c/evaluation.json from the above
  all      every step in order

    ~/.claude/bin/resguard.sh run --mem 2500M --cpu 200 -- taskset -c 12-15 env NUMPY_MADVISE_HUGEPAGE=0 \\
        OMP_NUM_THREADS=1 ~/miniconda3/envs/femtip/bin/python scripts/dip_link_concept_c.py all
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "docs/experiments/20261018-dip_link_board_mount"
OUT = STUDY / "concept_c"
CAD = OUT / "cad"
PRINT = OUT / "print"
MEDIA = OUT / "media"
DESIGN = OUT / "design.json"
EVAL = OUT / "evaluation.json"
WORK = REPO / "logs/20261018-dip_link/concept_c"          # intermediate meshes and check results (untracked)
DATE = "20261018"
PFX = f"{DATE}-concept_c"

# ------------------------------------------------------------------------------------------------- design parameters
# Lengths in mm, frame F. Model values; FR-4 thickness, part heights, the board's outline and flatness are pending
# measurements on the owner's board (spec.json "pending"); the spacer thickness follows the tallest part measured.
P = dict(
    x_back=6.45,            # tip back plane = top of the cap
    x_palmar=-10.55, y_tip=7.40, z_mount=-67.30, z_end=-88.71, z_contact=-78.16,
    t_cap=0.20,             # PLA cap printed after pause P1 (two 0.10 mm layers)
    t_sp=1.00,              # spacer = tallest part on the component side (MLX90393 0.80-1.00 mm; datasheet maximum)
    t_fr4=0.60,             # FR-4
    t_base=3.00,            # carrier base under the board
    h_chip=0.90,            # nominal MLX90393 package height
    spot=0.42,              # sensing point under the package top (datasheet Fig. 26)
    y_out=10.50,            # cartridge half-width over the tip's length (R5: rim 0.5 mm past the board's 10.0 mm)
    c_xy=0.10,              # pocket clearance per side around the board's outline
    board_u=10.0, board_v=(-10.10, 10.05), tab_u=5.0, tab_len=4.9,
    holes_uv=[(-8.0, -8.1), (8.0, -8.1), (8.0, 7.9), (-8.0, 7.9)], hole_d=2.40,
    pin_d=2.00,             # pin core in the 2.40 mm holes
    rib_d=2.55, rib_w=0.60, rib_n=3,   # crush ribs on the two proximal pins, FR-4 zone only
    sp_hole_d=2.30,         # spacer holes over the pin cores
    win_clear=0.20,         # spacer windows around each part
    cap_win=3.40,           # cap windows over U1-U5
    x_head=9.25,            # dorsal face of the body's shoulders = palmar face of the cartridge's wings
    z_head_top=-62.30,      # proximal end of the cartridge's wings
    y_wing_pos=13.05, y_wing_neg=-16.25,
    y_tab_half=5.95,        # cartridge tab region (board tab 5.0 + pocket 0.1 + wall 0.85)
    y_notch_half=6.15,      # body notch (0.20 mm clearance per side)
    x_bridge_dorsal=6.25,   # bridge's dorsal face in the notch (0.20 mm below the cap)
    y_arm_neg=-17.05, y_arm_pos=13.40,
    y_post=(-17.05, -12.60), z_post_top=-59.0,
    lip=dict(x=(11.35, 12.55), y=(-17.05, -14.85), z=(-62.0, -59.5)),
    web=dict(x=(9.25, 12.55), y=(-17.05, -16.45), z=(-62.0, -59.5)),
    tower=dict(y=(-16.25, -14.85), z=(-62.30, -59.50)),
    # one brass screw in a slot of the +y wing; the cartridge slides on along +z, so the slot opens toward +z and its
    # distal end, meeting the shank, is the cartridge's z stop
    screw=dict(y=9.6, z=-65.2, d_clear=2.20, pilot=1.60, depth=3.0, head_d=3.8, head_h=1.3, length=5.0, d=2.0,
               item="M2 x 5 brass cheese head, DIN 84, into a tapped M2 hole (1.60 mm pilot)"),
    tpu_preload=0.10,       # the tip's mount face printed 0.10 mm proud; the bridge compresses it on assembly
    # four 32 AWG PTFE wires (0.40 mm): a flat ribbon 0.45 x 2.0 mm on the tab under the spacer, a 1.0 mm twisted
    # bundle in the body's grooves (1.5 mm wide, floor at x 7.0)
    cable=dict(d=1.00, w=1.50, ribbon_w=2.00, ribbon_t=0.45, z=-64.46, x_floor=7.0, y_turn=-13.90, z_exit=-59.0),
    notch=dict(x=(6.65, 7.75), y=(-6.35, -5.05)),
    keep_margin=0.45,       # below the 0.5 mm keep-out level, for its cell accuracy (0.38 mm seen at the MCP link's end)
    r_mcp_end=12.70,        # the MCP link's end: a cylinder about the PIP axis (spec: current bridge 12.75)
)


def derived(p=P):
    d = dict(p)
    d["x_sp_top"] = p["x_back"] + p["t_cap"]
    d["x_comp"] = d["x_sp_top"] + p["t_sp"]
    d["x_floor"] = d["x_comp"] + p["t_fr4"]
    d["x_bed"] = d["x_floor"] + p["t_base"]
    d["gap"] = p["t_cap"] + (p["t_sp"] - p["h_chip"]) + p["spot"]
    d["gap_chip_spacer"] = p["t_cap"] + p["spot"]      # spacer cut to the measured chip height
    d["gap_thin_cap"] = 0.10 + p["spot"]               # and a single 0.10 mm cap layer
    zc = p["z_contact"]
    d["z_board"] = (-p["board_v"][1] + zc, -p["board_v"][0] + zc)          # main square, z_F
    d["z_tab_end"] = d["z_board"][1] + p["tab_len"]
    d["pocket"] = dict(y=p["board_u"] + p["c_xy"], z=(d["z_board"][0] - p["c_xy"], d["z_board"][1] + p["c_xy"]),
                       tab_y=p["tab_u"] + p["c_xy"])
    d["holes_F"] = [(u, -v + zc) for u, v in p["holes_uv"]]
    # print frame
    d["z_p"] = dict(base_top=p["t_base"], fr4_top=p["t_base"] + p["t_fr4"], pause_board=p["t_base"] + p["t_fr4"] + p["t_sp"],
                    cap_top=p["t_base"] + p["t_fr4"] + p["t_sp"] + p["t_cap"])
    return d


def uv_to_yz(u, v):
    return u, -v + P["z_contact"]


# ---------------------------------------------------------------------------------------------------- helpers (numpy)

def _np():
    import numpy as np
    return np


def board_parts():
    """Board parts from spec.json (ReSkin 5X fabrication files): name -> dict(kind, centre (y, z), size (dy, dz), h)."""
    spec = json.loads((STUDY / "spec.json").read_text())
    out = {}
    for k, r in spec["board"]["parts"].items():
        if not r["populated"] or k == "J1":
            continue
        y, z = uv_to_yz(*r["centre_uv"])
        out[k] = dict(kind=r["kind"], yz=(y, z), size=tuple(r["size_uv"]), h=r["height_mm"])
    j1 = spec["board"]["parts"]["J1"]
    out["J1_pads"] = dict(kind="pads", pads=[uv_to_yz(*pp["centre_uv"]) + tuple(pp["size_mm"]) for pp in j1["pads"]
                                             if pp["centre_uv"][1] > -15.0])
    return out


def ceiling_profiles(d):
    """z_top(x) of the bridge, shoulders and post: the lowest z where the swept MCP link and PIP servo come within 0.5 mm,
    minimum over each part's y range, less keep_margin. From keepout_pip.npz (0.25 mm cells)."""
    np = _np()
    sys.path.insert(0, str(REPO / "scripts"))
    import dip_link_geometry as G
    K = G.keepout()
    sd = K.signed
    gx = K.lo[0] + K.pitch * np.arange(sd.shape[0])
    gy = K.lo[1] + K.pitch * np.arange(sd.shape[1])
    gz = K.lo[2] + K.pitch * np.arange(sd.shape[2])
    bad = sd < 0.5
    first = np.argmax(bad, axis=2)
    ceil = np.where(bad.any(axis=2), gz[first] - K.pitch / 2, np.inf)

    def prof(y0, y1, x0, x1, step=0.25):
        n = max(2, int(math.ceil((x1 - x0) / step - 1e-9)) + 1)
        xs = np.round(np.linspace(x0, x1, n), 4)          # both ends included, so adjoining parts share a face
        js = np.flatnonzero((gy >= y0 - K.pitch) & (gy <= y1 + K.pitch))
        out = []
        for x in xs:
            i = np.flatnonzero(np.abs(gx - x) <= K.pitch * 1.01)     # the cells around x
            out.append(float(ceil[np.ix_(i, js)].min()) - d["keep_margin"])
        return [[float(x), round(z, 3)] for x, z in zip(xs, out)]

    y0, y1 = d["y_arm_neg"], d["y_arm_pos"]
    front = prof(y0, y1, d["x_palmar"], d["x_bridge_dorsal"])
    r = d["r_mcp_end"] + 0.5 + 0.05                    # exact: the MCP link's end turns about its own axis
    front = [[x, round(min(z, -51.55 - math.sqrt(max(r * r - x * x, 0.0))), 3) if abs(x) < r else z] for x, z in front]
    return dict(
        front=front,
        sh_pos=prof(d["y_notch_half"], y1, d["x_bridge_dorsal"], d["x_head"]),
        sh_neg=prof(y0, -d["y_notch_half"], d["x_bridge_dorsal"], d["x_head"]),
        sh_neg_inner=prof(d["y_post"][1], -d["y_notch_half"], d["x_bridge_dorsal"], d["x_head"]),
        post=prof(d["y_post"][0], d["y_post"][1], -4.0, d["x_head"]),
        head=prof(d["y_wing_neg"], d["y_wing_pos"], d["x_head"], d["x_bed"]),
    )


def cmd_design(a):
    d = derived()
    d["profiles"] = ceiling_profiles(d)
    d["parts"] = board_parts()
    OUT.mkdir(parents=True, exist_ok=True)
    DESIGN.write_text(json.dumps(d, indent=1))
    pr = d["profiles"]
    print(f"design: gap {d['gap']:.2f} mm (spacer at chip height {d['gap_chip_spacer']:.2f}); x: cap {d['x_back']}..{d['x_sp_top']:.2f}, "
          f"spacer ..{d['x_comp']:.2f}, FR-4 ..{d['x_floor']:.2f}, base ..{d['x_bed']:.2f}; pause P1 at z_p "
          f"{d['z_p']['pause_board']:.2f}", flush=True)
    for k, v in pr.items():
        zs = [z for _, z in v]
        print(f"  profile {k}: x {v[0][0]}..{v[-1][0]}, z_top {min(zs):.2f}..{max(zs):.2f}")


# ================================================================================================= FreeCAD (cad, step)

def _fc():
    import FreeCAD
    import Part
    return FreeCAD, Part


def fc_box(x0, x1, y0, y1, z0, z1):
    FreeCAD, Part = _fc()
    return Part.makeBox(x1 - x0, y1 - y0, z1 - z0, FreeCAD.Vector(x0, y0, z0))


def fc_cyl_x(y, z, r, x0, x1):
    FreeCAD, Part = _fc()
    return Part.makeCylinder(r, x1 - x0, FreeCAD.Vector(x0, y, z), FreeCAD.Vector(1, 0, 0))


def fc_prism_xz(pts, y0, y1):
    """Polygon in the x-z plane (list of (x, z)) extruded from y0 to y1."""
    FreeCAD, Part = _fc()
    V = [FreeCAD.Vector(x, y0, z) for x, z in pts]
    w = Part.makePolygon(V + [V[0]])
    return Part.Face(w).extrude(FreeCAD.Vector(0, y1 - y0, 0))


def fc_prism_yz(pts, x0, x1):
    FreeCAD, Part = _fc()
    V = [FreeCAD.Vector(x0, y, z) for y, z in pts]
    w = Part.makePolygon(V + [V[0]])
    return Part.Face(w).extrude(FreeCAD.Vector(x1 - x0, 0, 0))


def profile_poly(prof, z_bot, x_lo=None, x_hi=None, cap=None):
    """Closed polygon (x, z) under a ceiling profile: bottom edge at z_bot, top following prof (lower envelope over each
    step so the solid stays below the profile at every x)."""
    pts = [(x, z) for x, z in prof if (x_lo is None or x >= x_lo - 1e-9) and (x_hi is None or x <= x_hi + 1e-9)]
    top = []
    for i, (x, z) in enumerate(pts):
        zl = min(z, pts[i - 1][1] if i > 0 else z, pts[i + 1][1] if i + 1 < len(pts) else z)
        if cap is not None:
            zl = min(zl, cap)
        top.append((x, zl))
    # merge collinear runs of equal z
    simp = [top[0]]
    for i in range(1, len(top) - 1):
        if not (abs(top[i][1] - simp[-1][1]) < 1e-6 and abs(top[i + 1][1] - top[i][1]) < 1e-6):
            simp.append(top[i])
    simp.append(top[-1])
    # step polygon: horizontal runs between sample points use the lower of the two neighbours (already enveloped)
    poly = [(simp[0][0], z_bot)] + simp + [(simp[-1][0], z_bot)]
    return poly


def fc_body_new(D):
    """The new middle of the DIP link between the arms: bridge, shoulders, post, lip and web; tapped hole and cable
    grooves cut."""
    pr = D["profiles"]
    zb = D["z_mount"]
    ov = 0.05                    # into the arm spans; the mesh step keeps only what lies inside the original part there
    y0, y1 = D["y_arm_neg"] - ov, D["y_arm_pos"] + ov
    bridge = fc_prism_xz(profile_poly(pr["front"], zb), y0, y1)
    sh_pos = fc_prism_xz(profile_poly(pr["sh_pos"], zb), D["y_notch_half"], y1)
    sh_neg = fc_prism_xz(profile_poly(pr["sh_neg_inner"], zb), D["y_post"][1], -D["y_notch_half"])
    post = fc_prism_xz(profile_poly(pr["post"], zb, x_lo=D["x_bridge_dorsal"], cap=D["z_post_top"]), D["y_post"][0] - ov,
                       D["y_post"][1])
    lip, web = D["lip"], D["web"]
    s = bridge.fuse([sh_pos, sh_neg, post, fc_box(*lip["x"], *lip["y"], *lip["z"]), fc_box(*web["x"], *web["y"], *web["z"])])
    sc, cb = D["screw"], D["cable"]
    s = s.cut(fc_cyl_x(sc["y"], sc["z"], sc["pilot"] / 2, D["x_head"] - sc["depth"], D["x_head"] + 0.5))
    zc0, zc1 = cb["z"] - cb["w"] / 2, cb["z"] + cb["w"] / 2
    yt0, yt1 = cb["y_turn"] - cb["w"] / 2, cb["y_turn"] + cb["w"] / 2
    cuts = [fc_box(cb["x_floor"], D["x_head"] + 0.5, yt0, -D["y_notch_half"] + 0.01, zc0, zc1),        # along -y
            fc_box(cb["x_floor"], D["x_head"] + 0.5, yt0, yt1, zc0, D["z_post_top"] + 0.5),          # up the post
            fc_box(cb["x_floor"], D["x_head"] + 0.5, -7.6, -D["y_notch_half"] + 0.01, zb - 0.5, zc1)]  # slot for slide-in
    for c in cuts:
        s = s.cut(c)
    return s.removeSplitter()


def fc_carrier(D):
    """The cartridge's PLA volume: base, wings, tower, tab region, pocket zone with rims, pins with crush ribs, cap with
    windows; pocket, cable notch and screw slot cut."""
    FreeCAD, Part = _fc()
    xb, xf, xc, xs, xk = D["x_bed"], D["x_floor"], D["x_comp"], D["x_sp_top"], D["x_back"]
    yo, zm, ze = D["y_out"], D["z_mount"], D["z_end"]
    yt = D["y_tab_half"]
    zt_end = D["z_tab_end"]
    pk = D["pocket"]
    base = fc_box(xf, xb, -yo, yo, ze, zm)
    head = fc_box(D["x_head"], xb, D["y_wing_neg"], D["y_wing_pos"], zm, D["z_head_top"])
    tower = fc_box(D["x_head"], xb, *D["tower"]["y"], *D["tower"]["z"])
    tab = fc_box(xs, D["x_head"], -yt, yt, pk["z"][1] - 0.0, zt_end)
    tab_low = fc_box(xf, D["x_head"], -yt, yt, pk["z"][1], zt_end)
    zone = fc_box(xs, xf, -yo, yo, ze, zm)
    s = base.fuse([head, tower, tab, tab_low, zone])
    # pocket for board and spacer (FR-4 and spacer zones), main square and tab, open at the tab's end
    s = s.cut(fc_box(xs, xf, -pk["y"], pk["y"], pk["z"][0], pk["z"][1]))
    s = s.cut(fc_box(xs, xf, -pk["tab_y"], pk["tab_y"], pk["z"][1] - 0.01, zt_end + 0.5))
    # pins, ribs on the proximal two (FR-4 zone)
    pins = []
    for (y, z) in D["holes_F"]:
        pins.append(fc_cyl_x(y, z, D["pin_d"] / 2, xs, xf))
        if z > D["z_contact"]:
            for k in range(D["rib_n"]):
                a = math.radians(90 + 360 * k / D["rib_n"])
                r0, r1 = D["pin_d"] / 2 - 0.2, D["rib_d"] / 2
                rib = fc_box(xc, xf, -D["rib_w"] / 2, D["rib_w"] / 2, r0, r1)
                rib.rotate(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(1, 0, 0), math.degrees(a) - 90)
                rib.translate(FreeCAD.Vector(0, y, z))
                pins.append(rib)
    s = s.fuse(pins)
    # cap over the board region and the tab region, windows over U1-U5
    cap = fc_box(xk, xs, -yo, yo, ze, zm).fuse(fc_box(xk, xs, -yt, yt, zm - 0.01, zt_end))
    w = D["cap_win"] / 2
    for k, prt in D["parts"].items():
        if prt.get("kind") == "MLX90393":
            y, z = prt["yz"]
            cap = cap.cut(fc_box(xk - 0.5, xs + 0.01, y - w, y + w, z - w, z + w))
    s = s.fuse(cap)
    # cable notch through the tab region's -y wall
    n = D["notch"]
    cb = D["cable"]
    s = s.cut(fc_box(*n["x"], *n["y"], cb["z"] - cb["ribbon_w"] / 2, cb["z"] + cb["ribbon_w"] / 2))
    # screw slot in the +y wing, open toward +z (the shank enters it as the cartridge slides proximally)
    sc = D["screw"]
    r = sc["d_clear"] / 2
    slot = fc_box(D["x_head"] - 0.05, xb + 0.05, sc["y"] - r, sc["y"] + r, sc["z"], D["z_head_top"] + 0.5).fuse(
        fc_cyl_x(sc["y"], sc["z"], r, D["x_head"] - 0.05, xb + 0.05))
    s = s.cut(slot)
    return s.removeSplitter()


def fc_spacer(D):
    """Spacer placed on the board at pause P1: the board's outline, windows around the parts, J1's solder pads and the
    wire channel; holes over the pin cores."""
    xs, xc = D["x_sp_top"], D["x_comp"]
    zb0, zb1 = D["z_board"]
    s = fc_box(xs, xc, -D["board_u"], D["board_u"], zb0, zb1).fuse(
        fc_box(xs, xc, -D["tab_u"], D["tab_u"], zb1 - 0.01, D["z_tab_end"]))
    c = D["win_clear"]
    for k, prt in D["parts"].items():
        if prt.get("kind") in ("MLX90393", "CAP0402", "RES0402"):
            y, z = prt["yz"]
            du, dv = prt["size"]
            s = s.cut(fc_box(xs - 0.1, xc + 0.1, y - du / 2 - c, y + du / 2 + c, z - dv / 2 - c, z + dv / 2 + c))
    pads = D["parts"]["J1_pads"]["pads"]
    ys = [p[0] for p in pads]
    zs = [p[1] for p in pads]
    s = s.cut(fc_box(xs - 0.1, xc + 0.1, min(ys) - 0.75, max(ys) + 0.75, min(zs) - 0.9, max(zs) + 0.9))
    cb = D["cable"]
    s = s.cut(fc_box(xs - 0.1, xc + 0.1, -D["tab_u"] - 0.5, min(ys), cb["z"] - cb["ribbon_w"] / 2,
                     cb["z"] + cb["ribbon_w"] / 2))
    for (y, z) in D["holes_F"]:
        s = s.cut(fc_cyl_x(y, z, D["sp_hole_d"] / 2, xs - 0.1, xc + 0.1))
    return s.removeSplitter()


def fc_export(shape, stem, folder):
    import MeshPart
    folder.mkdir(parents=True, exist_ok=True)
    shape.exportStep(str(folder / f"{stem}.step"))
    m = MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.01, AngularDeflection=0.08, Relative=False)
    m.write(str(folder / f"{stem}.stl"))
    return dict(volume_mm3=round(shape.Volume, 2), valid=bool(shape.isValid()), solids=len(shape.Solids),
                bounds=[round(v, 3) for v in (shape.BoundBox.XMin, shape.BoundBox.YMin, shape.BoundBox.ZMin,
                                              shape.BoundBox.XMax, shape.BoundBox.YMax, shape.BoundBox.ZMax)])


def fc_main_cad():
    D = json.loads(DESIGN.read_text())
    t0 = time.time()
    info = {}
    bn = fc_body_new(D)
    info["body_new"] = fc_export(bn, f"{PFX}-body_new", WORK)
    mid = bn.common(fc_box(-50, 50, D["y_arm_neg"], D["y_arm_pos"], -120, 0)).removeSplitter()
    info["body_middle"] = fc_export(mid, f"{PFX}-body_middle", CAD)     # the new middle as solids, for the owner's CAD
    info["carrier"] = fc_export(fc_carrier(D), f"{PFX}-carrier", CAD)
    info["spacer"] = fc_export(fc_spacer(D), f"{PFX}-spacer", CAD)
    info["seconds"] = round(time.time() - t0, 1)
    (WORK / "cad_info.json").write_text(json.dumps(info, indent=1))
    print("cad:", json.dumps(info), flush=True)


def fc_main_step():
    """Faceted STEP of the meshed parts (the body with the STL's arms, the 85A tips), zipped (docs/**/*.zip is stored in
    LFS); coplanar facets merged on the mostly planar parts."""
    import zipfile
    import Mesh
    import Part
    jobs = json.loads((WORK / "step_jobs.json").read_text())
    out = {}
    only = [x for x in os.environ.get("DLC_STEP_ONLY", "").split(",") if x]
    prev = json.loads((WORK / "step_info.json").read_text()) if only and (WORK / "step_info.json").exists() else {}
    out.update(prev)
    for stl, step, refine in jobs:
        if only and not any(o in Path(step).name for o in only):
            continue
        t0 = time.time()
        m = Mesh.Mesh(stl)
        sh = Part.Shape()
        sh.makeShapeFromMesh(m.Topology, 1e-4)        # sewing tolerance below the meshes' smallest edges
        so = Part.makeSolid(sh)
        if refine:
            try:
                so = so.removeSplitter()
            except Exception:
                pass
        tmp = WORK / Path(step).name
        so.exportStep(str(tmp))
        with zipfile.ZipFile(step + ".zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(tmp, arcname=Path(step).name)
        tmp.unlink()
        out[Path(step).name + ".zip"] = dict(faces=len(so.Faces), valid=bool(so.isValid()), volume_mm3=round(so.Volume, 2),
                                             seconds=round(time.time() - t0, 1))
        print("step:", step, out[Path(step).name + ".zip"], flush=True)
    (WORK / "step_info.json").write_text(json.dumps(out, indent=1))


# ======================================================================================================= meshes (femtip)

def _m3():
    import manifold3d as m3
    return m3


def to_manifold(mesh):
    np = _np()
    m3 = _m3()
    return m3.Manifold(m3.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                               tri_verts=np.asarray(mesh.faces, np.uint32)))


def to_trimesh(man):
    np = _np()
    import trimesh
    mm = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=True)


def m_box(lo, hi):
    np = _np()
    m3 = _m3()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return m3.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


def m_cyl_x(y, z, r, x0, x1, n=48):
    m3 = _m3()
    c = m3.Manifold.cylinder(x1 - x0, r, r, n)          # along +z from 0
    return c.rotate((0, 90, 0)).translate((x0, y, z))   # +z -> +x


def tip_to_F(V, orient):
    """Readiness print-frame STL vertices to F (scripts/dip_link_spec.py tip_outlines)."""
    np = _np()
    V = np.asarray(V, float)
    if orient.startswith("back face on the bed"):
        return np.column_stack([6.45 - V[:, 2], V[:, 1], V[:, 0] - 88.71])
    return np.column_stack([-V[:, 0] - 2.05, V[:, 1], -67.30 - V[:, 2]])


TIP_KEYS = ("block4", "block5", "cap8.5", "ell16", "ell11")
CURVED = ("cap8.5", "ell16", "ell11")       # their meshes are 2-3 MB each: committed zipped (docs/**/*.zip is in LFS)


def tip_stl(key):
    """The 85A tip's STL in F: in cad/ for the blocks, in the work folder (and a zip in cad/) for the curved tips."""
    return (WORK if key in CURVED else CAD) / f"{PFX}-tpu_{key}.stl"


def tpu_tip(key, D, spec):
    """The 85A tip as printed in the cartridge, in F: the readiness tip's outer surface with its mount face 0.10 mm
    proud, and magnet pockets opening toward the palmar side (back-face-down print) at the readiness magnet positions:
    seat diameter 2.7 mm, from the magnet's back face to 0.20 mm beyond its front face."""
    np = _np()
    import trimesh
    t = spec["tips"][key]
    m = trimesh.load(REPO / t["stl"], force="mesh")
    parts = m.split(only_watertight=False)
    outer = max(parts, key=lambda q: q.bounding_box.volume)        # drop the readiness voids (pockets or slots)
    V = tip_to_F(outer.vertices, t["print_orientation"])
    on_mount = np.abs(V[:, 2] - D["z_mount"]) < 1e-3          # the mount face's own vertices: the face moves 0.10 mm
    V[on_mount, 2] = D["z_mount"] + D["tpu_preload"]
    tm = trimesh.Trimesh(V, outer.faces, process=True)
    if tm.volume < 0:
        tm.invert()
    man = to_manifold(tm)
    pockets = []
    for it in t["magnets"]["items"]:
        x0, x1 = it["x_F"]
        y, z = it["centre_F"][1], it["centre_F"][2]
        pockets.append(m_cyl_x(y, z, 2.7 / 2, x0 - 0.20, x1, n=64))
    for pk in pockets:
        man = man - pk
    return man, t


def board_model(D):
    """FR-4 with the tab cut at 4.9 mm and the four holes, the populated parts as boxes, the four wires as one bundle."""
    np = _np()
    xc, xf = D["x_comp"], D["x_floor"]
    zb0, zb1 = D["z_board"]
    fr4 = m_box((xc, -D["board_u"], zb0), (xf, D["board_u"], zb1)) + m_box((xc, -D["tab_u"], zb1 - 0.01),
                                                                           (xf, D["tab_u"], D["z_tab_end"]))
    for (y, z) in D["holes_F"]:
        fr4 = fr4 - m_cyl_x(y, z, D["hole_d"] / 2, xc - 0.1, xf + 0.1)
    parts = None
    for k, prt in D["parts"].items():
        if prt.get("kind") in ("MLX90393", "CAP0402", "RES0402"):
            y, z = prt["yz"]
            du, dv = prt["size"]
            h = D["h_chip"] if prt["kind"] == "MLX90393" else float(np.mean(prt["h"]))
            b = m_box((xc - h, y - du / 2, z - dv / 2), (xc, y + du / 2, z + dv / 2))
            parts = b if parts is None else parts + b
    return fr4, parts


def cable_path(D):
    """Centreline of the twisted bundle from the tab's notch to the exit at the post's top, then 6 mm free."""
    cb = D["cable"]
    x_gr = cb["x_floor"] + cb["d"] / 2
    x_rb = D["x_comp"] - cb["ribbon_t"] / 2
    return [(x_rb, -5.8, cb["z"]), (x_gr, -6.9, cb["z"]), (x_gr, cb["y_turn"], cb["z"]),
            (x_gr, cb["y_turn"], cb["z_exit"]), (x_gr, cb["y_turn"], cb["z_exit"] + 6.0)]


def cable_model(D):
    """The ribbon on the tab (from J1's pads to the notch) and the bundle beyond it."""
    cb = D["cable"]
    xc = D["x_comp"]
    pads = D["parts"]["J1_pads"]["pads"]
    y1 = max(p[0] for p in pads) + 0.3
    rib = m_box((xc - cb["ribbon_t"], -5.8, cb["z"] - cb["ribbon_w"] / 2), (xc, y1, cb["z"] + cb["ribbon_w"] / 2))
    return rib + tube(cable_path(D), cb["d"] / 2)


def tube(path, r, n=16):
    m3 = _m3()
    np = _np()
    out = None
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        c = m3.Manifold.cylinder(L, r, r, n)
        d = (b - a) / L
        # rotate +z onto d
        yaw = math.degrees(math.atan2(d[1], d[0]))
        pitch = math.degrees(math.acos(max(-1.0, min(1.0, d[2]))))
        c = c.rotate((0, pitch, 0)).rotate((0, 0, yaw)).translate(tuple(a))
        s = m3.Manifold.sphere(r, n).translate(tuple(a))
        out = (c + s) if out is None else (out + c + s)
    return out


def screw_model(D):
    sc = D["screw"]
    x0 = D["x_bed"]
    shank = m_cyl_x(sc["y"], sc["z"], sc["d"] / 2, x0 - sc["length"], x0)
    head = m_cyl_x(sc["y"], sc["z"], sc["head_d"] / 2, x0, x0 + sc["head_h"])
    return shank + head


def magnets(key, spec):
    out = None
    for it in spec["tips"][key]["magnets"]["items"]:
        x0, x1 = it["x_F"]
        c = m_cyl_x(it["centre_F"][1], it["centre_F"][2], 1.5, x0, x1, n=48)
        out = c if out is None else out + c
    return out


def to_print(mesh, D):
    """F -> the cartridge's print frame (dorsal face on the bed): x_p = z_F + 88.71, y_p = y_F, z_p = X_BED - x_F."""
    np = _np()
    import trimesh
    V = np.asarray(mesh.vertices, float)
    W = np.column_stack([V[:, 2] - D["z_end"], V[:, 1], D["x_bed"] - V[:, 0]])     # a rotation (det +1)
    m = trimesh.Trimesh(W, mesh.faces, process=True)
    if m.volume < 0:
        m.invert()
    return m


def cmd_mesh(a):
    np = _np()
    import trimesh
    sys.path.insert(0, str(REPO / "scripts"))
    import dip_link_geometry as G
    D = json.loads(DESIGN.read_text())
    spec = json.loads((STUDY / "spec.json").read_text())
    CAD.mkdir(parents=True, exist_ok=True)
    PRINT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    info = {}
    # body: the original part's arms (|y| outside the bridge span) + the new middle
    old = G.load_part("dip_link")
    om = to_manifold(old)
    big = 200.0
    eps = 0.002                  # the arms' cut sits 2 um outside their inner faces, so no cut face is coplanar with them
    arms = (om ^ m_box((-big, -big, -big), (big, D["y_arm_neg"] - eps, big))) + \
        (om ^ m_box((-big, D["y_arm_pos"] + eps, -big), (big, big, big)))
    new_ext = to_manifold(trimesh.load(WORK / f"{PFX}-body_new.stl", force="mesh"))
    middle = m_box((-big, D["y_arm_neg"], -big), (big, D["y_arm_pos"], big))
    new = new_ext ^ middle
    body = arms + (new_ext ^ (om + middle))      # the 0.05 mm overlap into the arms lies inside the original part
    bt = to_trimesh(body)
    comps = bt.split(only_watertight=False)
    slivers = [c for c in comps if abs(c.volume) < 1e-3]
    body_t = max(comps, key=lambda c: abs(c.volume))
    new.to_mesh()
    to_trimesh(new).export(WORK / f"{PFX}-body_new_middle.stl")
    body_t.export(CAD / f"{PFX}-body.stl")
    info["body"] = dict(volume_mm3=round(float(body.volume()), 1), arms_mm3=round(float(arms.volume()), 1),
                        new_mm3=round(float(new.volume()), 1), genus=int(body.genus()), components=len(comps),
                        slivers_dropped=len(slivers), watertight=bool(body_t.is_watertight))
    # R6: the arms of the new body against the original, within the arm spans
    for name, lo, hi in (("horn_side", -big, D["y_arm_neg"] - 0.005), ("slotted", D["y_arm_pos"] + 0.005, big)):
        clip = m_box((-big, lo, -big), (big, hi, big))
        v_old, v_new = float((om ^ clip).volume()), float((body ^ clip).volume())
        diff = float(((body ^ clip) - (om ^ clip)).volume() + ((om ^ clip) - (body ^ clip)).volume())
        info[f"arm_{name}"] = dict(old_mm3=round(v_old, 3), new_mm3=round(v_new, 3), sym_diff_mm3=round(diff, 4))
    # TPU tips
    for key in TIP_KEYS:
        man, t = tpu_tip(key, D, spec)
        tm = to_trimesh(man)
        tm.export(tip_stl(key))
        info[f"tpu_{key}"] = dict(volume_mm3=round(float(man.volume()), 1), watertight=bool(tm.is_watertight),
                                  magnets=t["magnets"]["count"])
    # board, cable, screw, magnets
    fr4, prt = board_model(D)
    to_trimesh(fr4).export(WORK / "board_fr4.stl")
    to_trimesh(prt).export(WORK / "board_parts.stl")
    cab = cable_model(D)
    to_trimesh(cab).export(WORK / "cable.stl")
    to_trimesh(screw_model(D)).export(WORK / "screw.stl")
    for key in ("block4", "block5"):
        to_trimesh(magnets(key, spec)).export(WORK / f"magnets_{key}.stl")
    # assembly at PIP 0 with the block of four magnets
    pieces = {"body": CAD / f"{PFX}-body.stl", "carrier": CAD / f"{PFX}-carrier.stl", "spacer": CAD / f"{PFX}-spacer.stl",
              "tpu": CAD / f"{PFX}-tpu_block4.stl", "fr4": WORK / "board_fr4.stl", "parts": WORK / "board_parts.stl",
              "cable": WORK / "cable.stl", "screw": WORK / "screw.stl", "magnets": WORK / "magnets_block4.stl"}
    ms = [trimesh.load(p, force="mesh") for p in pieces.values()]
    ms += [G.load_part("servo_pip"), G.load_part("mcp_link")]
    asm = trimesh.util.concatenate(ms)
    asm.export(CAD / f"{PFX}-assembly_pip0.stl")
    info["assembly"] = dict(parts=list(pieces) + ["servo_pip", "mcp_link"], faces=int(len(asm.faces)))
    # print frame: carrier (tool 1, PLA) and tip (tool 2, 85A), one object in the slicer, shared origin
    car = trimesh.load(CAD / f"{PFX}-carrier.stl", force="mesh")
    to_print(car, D).export(PRINT / f"{PFX}-print_carrier_pla_tool1.stl")
    (WORK / "print").mkdir(exist_ok=True)
    for key in TIP_KEYS:
        tm = trimesh.load(tip_stl(key), force="mesh")
        name = f"{PFX}-print_tip_{key.replace('.', 'p')}_85a_tool2.stl"
        to_print(tm, D).export((WORK / "print" if key in CURVED else PRINT) / name)
    import zipfile
    for zpath, files in ((CAD / f"{PFX}-tpu_curved_stl.zip", [tip_stl(k) for k in CURVED]),
                         (PRINT / f"{PFX}-print_tips_curved_85a_tool2_stl.zip",
                          [WORK / "print" / f"{PFX}-print_tip_{k.replace('.', 'p')}_85a_tool2.stl" for k in CURVED])):
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for f in files:
                z.write(f, arcname=f.name)
    sp = trimesh.load(CAD / f"{PFX}-spacer.stl", force="mesh")
    Vs = np.asarray(sp.vertices, float)
    sp_print = trimesh.Trimesh(np.column_stack([Vs[:, 2] - D["z_end"], Vs[:, 1], D["x_comp"] - Vs[:, 0]]), sp.faces)
    if sp_print.volume < 0:
        sp_print.invert()
    sp_print.export(PRINT / f"{PFX}-print_spacer_pla.stl")
    (WORK / "mesh_info.json").write_text(json.dumps(info, indent=1))
    # the body stays as triangles: merging its coplanar facets gave an invalid solid (FreeCAD 0.19, 2026-10-18)
    jobs = [[str(CAD / f"{PFX}-body.stl"), str(CAD / f"{PFX}-body.step"), False]] + \
           [[str(tip_stl(k)), str(CAD / f"{PFX}-tpu_{k}.step"), k.startswith("block")] for k in TIP_KEYS]
    (WORK / "step_jobs.json").write_text(json.dumps(jobs))
    print("mesh:", json.dumps(info), flush=True)


# =========================================================================================================== checks

def surface_and_volume_points(mesh, spacing=0.25, vox=0.5):
    np = _np()
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import dip_link_geometry as G
    S = G.surface_points(mesh, spacing)
    try:
        from morphohand.geometry import cad_features as cf
        Vc = cf.voxelize_solid(mesh, vox).centres()
        S = np.vstack([S, Vc])
    except Exception:
        pass
    return S


def cmd_check(a):
    np = _np()
    import trimesh
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import dip_link_geometry as G
    D = json.loads(DESIGN.read_text())
    res = {}
    t0 = time.time()
    # 1. clearance over the PIP range (exact method of the specification)
    cands = {
        "body_new": (trimesh.load(WORK / f"{PFX}-body_new_middle.stl", force="mesh"), 0.5),
        "carrier": (trimesh.load(CAD / f"{PFX}-carrier.stl", force="mesh"), 0.5),
        "board_fr4": (trimesh.load(WORK / "board_fr4.stl", force="mesh"), 0.5),
        "board_parts": (trimesh.load(WORK / "board_parts.stl", force="mesh"), 0.5),
        "spacer": (trimesh.load(CAD / f"{PFX}-spacer.stl", force="mesh"), 0.5),
        "screw": (trimesh.load(WORK / "screw.stl", force="mesh"), 0.5),
        "tpu_block4": (trimesh.load(CAD / f"{PFX}-tpu_block4.stl", force="mesh"), 0.5),
        "cable_fixed": (trimesh.load(WORK / "cable.stl", force="mesh"), 1.0),
        # R10: the board (FR-4 and parts) against the PIP servo alone; the plate placement keeps 6.3 mm
        "board_to_servo": (trimesh.util.concatenate([trimesh.load(WORK / "board_fr4.stl", force="mesh"),
                                                     trimesh.load(WORK / "board_parts.stl", force="mesh")]), 0.5),
        "chips_to_servo": (trimesh.util.concatenate([to_trimesh(m_box(
            (D["x_comp"] - D["h_chip"], prt["yz"][0] - 1.5, prt["yz"][1] - 1.5), (D["x_comp"], prt["yz"][0] + 1.5,
                                                                                prt["yz"][1] + 1.5)))
            for prt in D["parts"].values() if prt.get("kind") == "MLX90393"]), 0.5),
        # the same chips at the current plate placement (component face x_F 10.35, chips toward the plate)
        "chips_to_servo_plate": (trimesh.util.concatenate([to_trimesh(m_box(
            (10.35 - D["h_chip"], prt["yz"][0] - 1.5, prt["yz"][1] - 1.5), (10.35, prt["yz"][0] + 1.5, prt["yz"][1] + 1.5)))
            for prt in D["parts"].values() if prt.get("kind") == "MLX90393"]), 0.0),
    }
    clear = {}
    only = [x for x in os.environ.get("DLC_PARTS", "").split(",") if x]
    if only:                                   # re-check some parts, keep the others' results
        prev = json.loads((WORK / "check.json").read_text())["clearance"] if (WORK / "check.json").exists() else {}
        clear = {k: v for k, v in prev.items() if k not in only}
        cands = {k: v for k, v in cands.items() if k in only}
    else:
        (WORK / "check_progress.txt").write_text("")
    for name, (m, need) in cands.items():
        t1 = time.time()
        P_ = G.surface_points(m, 0.25)      # a part's least clearance lies on its surface (no parent part fits inside one)
        if name == "cable_fixed":     # the segment fixed to the DIP link ends at the post's top
            P_ = P_[P_[:, 2] <= D["cable"]["z_exit"] + 0.01]
        if name in ("board_to_servo", "chips_to_servo", "chips_to_servo_plate"):
            v, i, q, part = G.points_clearance_exact(P_, parts=("servo_pip",), reach=10.0)
        else:
            v, i, q, part = G.points_clearance_exact(P_, reach=2.0)
        if i is None:
            i = 0
        clear[name] = dict(min_mm=round(v, 3), at_F=[round(float(c), 2) for c in P_[i]], pip_deg=q, part=part,
                           need_mm=need, ok=bool(v >= need), n_points=int(len(P_)), seconds=round(time.time() - t1, 1))
        with open(WORK / "check_progress.txt", "a") as fh:            # one fsynced line per part
            fh.write(json.dumps({name: clear[name]}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"clearance {name}: {v:.3f} mm (need {need}) at {np.round(P_[i], 2)} PIP {q} {part}", flush=True)
    res["clearance"] = clear
    # 2. gaps between the parts as assembled (k-d tree on surface samples; interference volume by manifold3d)
    asm = {k: trimesh.load(p, force="mesh") for k, p in {
        "body": CAD / f"{PFX}-body.stl", "carrier": CAD / f"{PFX}-carrier.stl", "spacer": CAD / f"{PFX}-spacer.stl",
        "tpu": CAD / f"{PFX}-tpu_block4.stl", "fr4": WORK / "board_fr4.stl", "parts": WORK / "board_parts.stl",
        "screw": WORK / "screw.stl", "magnets": WORK / "magnets_block4.stl", "cable": WORK / "cable.stl"}.items()}
    pairs = [("body", "carrier"), ("body", "tpu"), ("body", "screw"), ("carrier", "spacer"), ("carrier", "fr4"),
             ("carrier", "parts"), ("spacer", "parts"), ("spacer", "fr4"), ("carrier", "tpu"), ("tpu", "parts"),
             ("body", "cable"), ("carrier", "cable"), ("tpu", "magnets")]
    gaps = {}
    samp = {k: G.surface_points(m, 0.2) for k, m in asm.items()}
    for a_, b_ in pairs:
        A, B = asm[a_], asm[b_]
        inter = float((to_manifold(A) ^ to_manifold(B)).volume())
        d1, _ = cKDTree(samp[b_]).query(samp[a_], k=1)
        gaps[f"{a_}|{b_}"] = dict(min_surface_distance_mm=round(float(d1.min()), 3), interference_mm3=round(inter, 4))
        print(f"gap {a_}|{b_}: {d1.min():.3f} mm, overlap {inter:.4f} mm3", flush=True)
    res["gaps"] = gaps
    # 3. widths over the tip's length and anything palmar of the back plane beyond the tip's outline (R5)
    zr = (D["z_end"] - 1e-6, D["z_mount"] - 0.01)
    width = {}
    asm["body_new"] = cands["body_new"][0] if "body_new" in cands else trimesh.load(WORK / f"{PFX}-body_new_middle.stl",
                                                                                     force="mesh")
    for k in ("body_new", "carrier", "spacer", "fr4", "screw"):      # new parts only: the arms are kept as they are
        V = asm[k].vertices
        sel = (V[:, 2] >= zr[0]) & (V[:, 2] <= zr[1])
        width[k] = None if not sel.any() else round(float(np.abs(V[sel, 1]).max()), 3)
        pal = sel & (V[:, 0] < D["x_back"] - 0.01) & (np.abs(V[:, 1]) > D["y_tip"] + 0.01)
        width[k + "_palmar_beyond_tip"] = int(pal.sum())
    res["widths_half_mm"] = width
    # 3b. fasteners near the chips (R10): least distance from a chip's sensing point to the screw
    sc_pts = G.surface_points(asm["screw"], 0.1)
    chips = [(k, prt["yz"]) for k, prt in D["parts"].items() if prt.get("kind") == "MLX90393"]
    xs_ = D["x_comp"] - D["h_chip"] + D["spot"]
    dch = {k: round(float(cKDTree(sc_pts).query([[xs_, y, z]])[0][0]), 2) for k, (y, z) in chips}
    res["screw_to_chips_mm"] = dch
    # 4. board protrusion past the tip and the rim
    res["protrusion"] = dict(board_past_tip_mm=round(D["board_u"] - D["y_tip"], 3),
                             rim_past_tip_mm=round(D["y_out"] - D["y_tip"], 3),
                             x_range_of_overhang=[D["x_back"], round(D["x_bed"], 3)])
    # 5. thin features (designed values)
    res["walls_designed_mm"] = dict(
        side_rims=round(D["y_out"] - D["pocket"]["y"], 3), distal_rim=round(D["pocket"]["z"][0] - D["z_end"], 3),
        proximal_rims=round(D["z_mount"] - D["pocket"]["z"][1], 3), tab_walls=round(D["y_tab_half"] - D["pocket"]["tab_y"], 3),
        cap=D["t_cap"], wings=round(D["x_bed"] - D["x_head"], 3), base=D["t_base"],
        spacer_ligament_pin_to_edge=round(D["board_u"] - (8.0 + D["sp_hole_d"] / 2), 3),
        insert_or_tap_walls=None)
    # 6. mass
    rho_link, rho_solid = 0.9482, 1.24
    vol = {k: float(trimesh.load(p, force="mesh").volume) for k, p in {
        "body": CAD / f"{PFX}-body.stl", "carrier": CAD / f"{PFX}-carrier.stl", "spacer": CAD / f"{PFX}-spacer.stl"}.items()}
    sc = D["screw"]
    screw_g = 8.5e-3 * (math.pi * (sc["d"] / 2) ** 2 * sc["length"] + math.pi * (sc["head_d"] / 2) ** 2 * sc["head_h"])
    mass = dict(body_g=round(vol["body"] * rho_link * 1e-3, 2), carrier_g=round(vol["carrier"] * rho_solid * 1e-3, 2),
                spacer_g=round(vol["spacer"] * rho_solid * 1e-3, 2), board_g=0.64, screw_g=round(screw_g, 2),
                wires_g=0.10)
    mass["total_without_tip_g"] = round(sum(mass.values()), 2)
    mass["limit_g"] = 7.6
    mass["basis"] = ("body at the links' sliced density 948.2 kg/m^3 (PLA, 15 % infill, assembly.json); carrier and spacer "
                     "solid PLA 1240 kg/m^3; board 0.64 g (spec); brass screw 8.5 g/cm^3; wires 0.10 g (estimate)")
    mass["volumes_mm3"] = {k: round(v, 1) for k, v in vol.items()}
    res["mass"] = mass
    res["seconds"] = round(time.time() - t0, 1)
    (WORK / "check.json").write_text(json.dumps(res, indent=1))
    print("check:", json.dumps(dict(mass=mass, widths=width)), flush=True)


# ========================================================================================================== sensing

def cmd_sense(a):
    np = _np()
    sys.path.insert(0, str(REPO / "scripts"))
    import mlx90393_model as MLX
    import sim2real_fingertip_sensing as SS
    import tpu_tip_design as TD
    import tpu_tip_magnet_sensing as MS
    D = json.loads(DESIGN.read_text())
    spec = json.loads((STUDY / "spec.json").read_text())
    rows = [json.loads(l) for l in open(REPO / "docs/experiments/20261016-sim2real_fingertip_readiness/fe/fem.jsonl")]
    gaps = dict(concept_c=D["gap"], chip_spacer=D["gap_chip_spacer"], thin_cap=D["gap_thin_cap"], touch=0.42, plate=3.42)
    out = []
    for r in rows:
        if r["layout"].endswith("_w1"):
            continue
        ys, zs, Gp = MS.patch_grid(r)
        zc = min(zs, key=lambda z: abs(z - TD.Z_C))
        for name, gap in gaps.items():
            a_ = MS.analyse(r, "board", gap)
            ch = MS.chips(r, "board", gap)
            B = MS.B0(r, ch) * 1e3
            bmax = float(np.abs(B).max())
            g, rng = SS.gain_for(bmax)
            both = [p["sigF_n_unk_bg"] for p in a_["points"]]
            i0, j0 = ys.index(0.0), zs.index(zc)
            Sg = {k: np.array([[MS.signal(r, Gp[(y, z)][k], ch) for z in zs] for y in ys]) for k in ("normal", "shear_y", "shear_z")}
            J = np.concatenate([np.stack([Sg["normal"][i0, j0], Sg["shear_y"][i0, j0], Sg["shear_z"][i0, j0],
                                          MS.deriv(Sg["normal"][:, j0], ys, i0), MS.deriv(Sg["normal"][i0, :], zs, j0)], 1),
                                np.tile(np.eye(3), (len(ch), 1))], 1)
            sig = np.tile(MS.NOISE, len(ch))
            feq = []
            for k in range(3):
                e = np.zeros(3)
                e[k] = 1e-6
                dB = (MS.B0(r, [p + e for p in ch]) - MS.B0(r, [p - e for p in ch])).ravel() / 2 * 1e6
                est = np.linalg.lstsq(J / sig[:, None], dB / sig, rcond=None)[0]
                feq.append([round(abs(float(est[0])), 4), round(float(np.hypot(est[1], est[2])), 4)])
            # chip to nearest magnet: sensing point to the magnet's surface (disc of 3 mm, 2 mm thick) and to its centre
            Mc = [(TD.X_FACE - r["depth_mm"] - t / 2, y, z, dd / 2, t / 2) for dd, t, y, z in r["magnets"]]
            dsurf, dcen = [], []
            for c in ch:
                c = np.asarray(c) * 1e3
                best_s, best_c = 1e9, 1e9
                for xm, ym, zm, rr, hh in Mc:
                    radial = math.hypot(c[1] - ym, c[2] - zm)
                    dx = max(abs(c[0] - xm) - hh, 0.0)
                    dr = max(radial - rr, 0.0)
                    best_s = min(best_s, math.hypot(dx, dr))
                    best_c = min(best_c, math.dist(c, (xm, ym, zm)))
                dsurf.append(best_s)
                dcen.append(best_c)
            out.append(dict(layout=r["layout"], recipe=r["recipe"], magnets=len(r["magnets"]), case=name, gap_mm=round(gap, 3),
                            centre_force_N=round(a_["centre"]["sigF_n_unk_bg"], 4),
                            median_force_N=round(float(np.median(both)), 4),
                            range_force_N=[round(min(both), 4), round(max(both), 4)],
                            B_chips_mT=np.round(B, 3).tolist(), B_U1_mT=round(float(np.linalg.norm(B[0])), 2),
                            B_max_mT=round(bmax, 2), gain_sel=g, headroom=None if rng is None else round(rng / bmax, 2),
                            false_force_N_per_um={ax: dict(normal=v[0], shear=v[1]) for ax, v in zip("xyz", feq)},
                            chip_to_magnet_surface_mm=[round(v, 3) for v in dsurf],
                            chip_to_magnet_centre_mm=[round(v, 3) for v in dcen]))
            print(f"{r['layout']:14s} {r['recipe']} {name:18s} gap {gap:.2f}: {a_['centre']['sigF_n_unk_bg']:.3f} N, "
                  f"B U1 {np.linalg.norm(B[0]):.2f} mT, GAIN_SEL {g}, chip-magnet {min(dsurf):.2f} mm", flush=True)
    # static field at U1 for the curved tips (no finite elements of them in this frame): point dipoles at the magnets
    tips = {}
    for key in TIP_KEYS:
        t = spec["tips"][key]
        mags = t["magnets"]["items"]
        BR = MS.BR
        Bc = []
        for gap in (D["gap"], D["gap_chip_spacer"], D["gap_thin_cap"]):
            pts = [np.array([D["x_back"] + gap, u, D["z_contact"] - v]) for u, v in MLX.BOARD_CHIPS.values()]
            B = []
            for p in pts:
                s = np.zeros(3)
                for it in mags:
                    xm = float(np.mean(it["x_F"]))
                    V = math.pi * (1.5e-3) ** 2 * 2e-3
                    m = np.array([-BR * V / (4e-7 * math.pi), 0, 0])      # north toward the palmar face (-x_F)
                    s += MS.dipole(p * 1e-3, np.array([xm, it["centre_F"][1], it["centre_F"][2]]) * 1e-3, m)
                B.append(s * 1e3)
            B = np.array(B)
            Bc.append(dict(gap_mm=round(gap, 3), B_U1_mT=round(float(np.linalg.norm(B[0])), 2),
                           B_max_mT=round(float(np.abs(B).max()), 2), gain_sel=SS.gain_for(float(np.abs(B).max()))[0]))
        tips[key] = Bc
    (WORK / "sense.json").write_text(json.dumps(dict(rows=out, tips_static=tips, gaps=gaps), indent=1))


# =========================================================================================================== render

COL = dict(body="#c9b38a", carrier="#7a9cc6", spacer="#e3e3e3", tpu="#d98c5f", fr4="#2f7d3b", parts="#222222",
           cable="#b03030", screw="#c8a040", magnets="#7f7f7f", servo_pip="#5b5b5b", mcp_link="#d9d9d9")


def _pv_setup():
    lib = str(Path.home() / "miniconda3/envs/femtip/lib")
    if lib not in os.environ.get("LD_LIBRARY_PATH", ""):        # VTK needs the env's libstdc++; the loader reads this at start
        os.environ["LD_LIBRARY_PATH"] = lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    import pyvista as pv
    pv.OFF_SCREEN = True
    return pv


def _load_pv(pv, path):
    import trimesh
    m = trimesh.load(path, force="mesh")
    return pv.wrap(m)


def cmd_render(a):
    pv = _pv_setup()
    sys.path.insert(0, str(REPO / "scripts"))
    import dip_link_geometry as G
    import trimesh
    D = json.loads(DESIGN.read_text())
    MEDIA.mkdir(parents=True, exist_ok=True)
    files = {"body": CAD / f"{PFX}-body.stl", "carrier": CAD / f"{PFX}-carrier.stl", "spacer": CAD / f"{PFX}-spacer.stl",
             "tpu": CAD / f"{PFX}-tpu_block4.stl", "fr4": WORK / "board_fr4.stl", "parts": WORK / "board_parts.stl",
             "cable": WORK / "cable.stl", "screw": WORK / "screw.stl", "magnets": WORK / "magnets_block4.stl"}
    T = {k: trimesh.load(p, force="mesh") for k, p in files.items()}
    T["servo_pip"] = G.load_part("servo_pip")
    T["mcp_link"] = G.load_part("mcp_link")
    big = 300.0
    # the carrier split at the board pause: what tool 1 prints before pause P1 and the cap it prints after
    car_m = to_manifold(T["carrier"])
    T["cap"] = to_trimesh(car_m ^ m_box((-big, -big, -big), (D["x_sp_top"], big, big)))
    T["carrier_pre"] = to_trimesh(car_m ^ m_box((D["x_sp_top"], -big, -big), (big, big, big)))
    col = dict(COL, cap="#2b4f7e", carrier_pre=COL["carrier"])
    FS = 20

    def plotter(size=(1600, 1200), shape=(1, 1)):
        p = pv.Plotter(off_screen=True, window_size=size, shape=shape, border=False)
        p.set_background("white")
        p.enable_anti_aliasing("ssaa")
        return p

    def add(p, keys, opacity=None, offsets=None):
        for k in keys:
            m = pv.wrap(T[k]).copy()
            if offsets and k in offsets:
                m.translate(offsets[k], inplace=True)
            p.add_mesh(m, color=col[k], opacity=(opacity or {}).get(k, 1.0), smooth_shading=False, specular=0.15)

    asm_keys = ["body", "carrier", "tpu", "screw", "cable", "servo_pip", "mcp_link"]
    # 1. iso views of the assembly at PIP 0
    p = plotter((1800, 900), (1, 2))
    p.subplot(0, 0)
    add(p, asm_keys, opacity={"mcp_link": 0.22, "servo_pip": 0.9})
    p.camera_position = [(-75, -85, -40), (0, -2, -68), (0, 0, 1)]
    p.add_text("Palmar side", font_size=FS, color="black")
    p.subplot(0, 1)
    add(p, asm_keys, opacity={"mcp_link": 0.22, "servo_pip": 0.9})
    p.camera_position = [(95, -75, -25), (2, -2, -68), (0, 0, 1)]
    p.add_text("Dorsal side", font_size=FS, color="black")
    p.screenshot(MEDIA / f"{PFX}-iso.png")
    p.close()

    # 2. sections: keep y >= yc, camera on the -y side, so the cut faces look at the camera
    def cut(k, yc):
        m = T[k]
        h = to_manifold(m) ^ m_box((-big, yc, -big), (big, big, big))
        return to_trimesh(h) if h.volume() > 1e-6 else None

    for yc, tag, title in ((0.0, "section_chips_y0", "Section y = 0 (U4, U1, U2) at PIP 0"),
                           (7.0, "section_chips_y7", "Section y = 7 (U3, board holes) at PIP 0")):
        p = plotter((1500, 1150))
        for k in ("body", "carrier", "spacer", "tpu", "fr4", "parts", "cable", "screw", "magnets", "servo_pip", "mcp_link"):
            c = cut(k, yc)
            if c is not None:
                p.add_mesh(pv.wrap(c), color=col[k], smooth_shading=False, specular=0.05)
        p.camera_position = [(-1.0, -150, -70.5), (-1.0, 0, -70.5), (0, 0, 1)]
        p.camera.parallel_projection = True
        p.camera.parallel_scale = 21.0
        p.add_text(title, font_size=FS, color="black")
        p.screenshot(MEDIA / f"{PFX}-{tag}.png")
        p.close()
    # close-up of the stack at U1 with dimension labels
    p = plotter((1700, 1100))
    for k in ("body", "carrier", "spacer", "tpu", "fr4", "parts", "cable", "magnets"):
        c = cut(k, 0.0)
        if c is not None:
            p.add_mesh(pv.wrap(c), color=col[k], smooth_shading=False)
    lab = [((D["x_back"] - 2.0, -1, -77.0), "85A tip"), ((D["x_back"] - 1.3, -1, -76.0), "cap 0.20 (blue)"),
           ((D["x_sp_top"] + 0.5, -1, -80.9), "spacer 1.00"), ((D["x_comp"] - 0.45, -1, -78.16), "U1"),
           ((D["x_comp"] + 0.3, -1, -79.6), "FR-4 0.60"), ((D["x_floor"] + 1.5, -1, -76.4), "base 3.00")]
    p.add_point_labels([q for q, _ in lab], [t for _, t in lab], font_size=26, point_size=1, shape_opacity=0.75,
                       text_color="black", always_visible=True)
    p.camera_position = [(7.0, -60, -78.0), (7.0, 0, -78.0), (0, 0, 1)]
    p.camera.parallel_projection = True
    p.camera.parallel_scale = 4.2
    p.add_text(f"Stack at U1, section y = 0: sensing plane {D['gap']:.2f} mm behind the tip's back face", font_size=FS,
               color="black")
    p.screenshot(MEDIA / f"{PFX}-section_stack_u1.png")
    p.close()

    # 3. exploded view along x
    off = dict(magnets=(-26, 0, 0), tpu=(-16, 0, 0), cap=(-7, 0, 0), spacer=(-2, 0, 0), fr4=(3, 0, 0), parts=(3, 0, 0),
               cable=(3, 0, 0), carrier_pre=(9, 0, 0), screw=(22, 0, 0), body=(0, 0, 26))
    off["body"] = (0, 0, 30)
    p = plotter((1700, 1150))
    add(p, ["magnets", "tpu", "cap", "spacer", "fr4", "parts", "cable", "carrier_pre", "screw", "body"], offsets=off)
    p.camera_position = [(-60, -125, -16), (0, 0, -64), (0, 0, 1)]
    p.add_text("Exploded along x", font_size=FS, color="black")
    p.screenshot(MEDIA / f"{PFX}-exploded.png")
    p.close()

    # 4. the print in its own frame: at pause P1, and finished
    zP = D["z_p"]["pause_board"]
    carP = to_print(T["carrier"], D)
    pre = to_trimesh(to_manifold(carP) ^ m_box((-50, -50, -1), (80, 50, zP - 0.01)))   # below the cap's underside
    fr4 = to_print(T["fr4"], D)
    prt = to_print(T["parts"], D)
    spc = to_print(T["spacer"], D)
    cab = to_print(T["cable"], D)
    tip = to_print(T["tpu"], D)
    p = plotter((1800, 1500), (2, 2))
    cam = [(36, -46, 58), (13, 0, 4), (0, 0, 1)]
    steps = [("1. Tool 1 prints to z 4.60: pocket and four pins", []),
             ("2. Pause P1: board onto the pins", [(fr4, COL["fr4"]), (prt, COL["parts"]), (cab, COL["cable"])]),
             ("3. Spacer onto the board; resume", [(fr4, COL["fr4"]), (prt, COL["parts"]), (cab, COL["cable"]),
                                                    (spc, COL["spacer"])])]
    for n, (title, extra) in enumerate(steps):
        p.subplot(n // 2, n % 2)
        p.add_mesh(pv.wrap(pre), color=COL["carrier"], smooth_shading=False)
        for m, c in extra:
            p.add_mesh(pv.wrap(m), color=c, smooth_shading=False)
        p.camera_position = cam
        p.add_text(title, font_size=FS, color="black")
    p.subplot(1, 1)
    p.add_mesh(pv.wrap(carP), color=COL["carrier"], smooth_shading=False)
    p.add_mesh(pv.wrap(tip), color=COL["tpu"], smooth_shading=False)
    p.camera_position = [(62, -58, 46), (13, 0, 9), (0, 0, 1)]
    p.add_text("4. Cap (PLA), tip (85A), magnets at P2", font_size=FS, color="black")
    p.screenshot(MEDIA / f"{PFX}-print_frame.png")
    p.close()
    print("render: done", flush=True)


# ============================================================================================================== eval

def narrative(D, chk):
    """Retention, print plan, printability, assembly, replacement, parts and pending items, with the design's numbers."""
    spec = json.loads((STUDY / "spec.json").read_text())
    zp = D["z_p"]
    pauses = {}
    for key in TIP_KEYS:
        xs = [x for it in spec["tips"][key]["magnets"]["items"] for x in it["x_F"]]
        top = D["x_bed"] - (min(xs) - 0.20)                 # pocket top in the print, 0.20 mm of air over the magnet
        lh = 0.20 if key.startswith("block") else 0.10
        pauses[key] = dict(magnet_pocket_top_z=round(top, 2), layer_mm=lh, pause_before_layer_ending_at=round(top + lh, 2),
                           magnets=spec["tips"][key]["magnets"]["count"])
    sc = D["screw"]
    rib_int = (D["rib_d"] - D["hole_d"]) / 2
    # walls of the tapped M2 hole in the +y shoulder (axis x from x_head - depth to x_head)
    x0, x1 = D["x_head"] - sc["depth"], D["x_head"]
    tops = [z for x, z in D["profiles"]["sh_pos"] + D["profiles"]["front"] if x0 - 1e-9 <= x <= x1 + 1e-9]
    tap = dict(top_mm=round(min(tops) - (sc["z"] + sc["pilot"] / 2), 2), bottom_mm=round(sc["z"] - sc["pilot"] / 2 - D["z_mount"], 2),
               side_mm=round(min(sc["y"] - sc["pilot"] / 2 - D["y_notch_half"], D["y_arm_pos"] - sc["y"] - sc["pilot"] / 2), 2))
    walls = dict(chk["walls_designed_mm"], insert_or_tap_walls=tap)
    cl = chk["clearance"]
    rigid = min(v["min_mm"] for k, v in cl.items() if v["need_mm"] == 0.5 and k not in ("board_to_servo", "chips_to_servo"))
    return dict(
        retention=dict(
            board_x="normal to the board: the FR-4 lies on the pocket floor (a top surface of the carrier, ironed), the "
                    "spacer on the FR-4, and the cap is printed onto the spacer, the 0.40 mm rims and the four pin tops; "
                    "zero clearance when the cap solidifies, no screw preload, contact pressure from the cap's first-layer "
                    "squish only (creep of the stack under load pending: zero-load field before and after 50 presses at 3 N)",
            board_yz=f"in the board's plane: two pins with {D['rib_n']} crush ribs in the proximal holes (core "
                     f"{D['pin_d']:.2f} mm, ribs to {D['rib_d']:.2f} mm, {rib_int:.3f} mm radial interference in the "
                     f"{D['hole_d']:.2f} mm plated holes, PLA tolerance 0.1-0.2 mm, so 0 to 0.2 mm of rib crush) remove the "
                     f"placement clearance; the distal pins ({D['pin_d']:.2f} mm, 0.20 mm radial clearance) and the "
                     f"{D['c_xy']:.2f} mm pocket clearance guide only; after the cap is printed all four pins are fused to it",
            tip=f"printed onto the cap: TPU on PLA over the whole back face except five {D['cap_win']:.1f} mm windows over "
                f"U1-U5, which the first TPU layer bridges 0.30 mm above the chip tops (0.20 mm with the spacer at chip "
                f"height); mount face printed {D['tpu_preload']:.2f} mm "
                f"proud and compressed against the bridge when the cartridge is installed; no tongue, lip or screw in the TPU",
            cartridge=f"y: the tab region in the body's notch (0.20 mm per side) and the screw shank in the "
                      f"{sc['d_clear']:.2f} mm slot; -x: wings and tower on the shoulders and post (x_F {D['x_head']}); "
                      f"+x: screw preload on the +y wing and the body's lip over the tower on -y (0.10 mm gap); +z: slot end "
                      f"on the shank, with the tip's {D['tpu_preload']:.2f} mm preload on the bridge; -z: screw friction",
            play=("Nothing between the chips and the magnets is assembled: board and tip are fixed to each other by the "
                  "print, so the board's position relative to the magnets does not change at remounting. The 0.10-0.20 mm "
                  "clearances of the cartridge on the body move the whole fingertip and are closed by the screw preload."),
            fasteners=f"one {sc['item']}, {min(chk['screw_to_chips_mm'].values()):.1f} mm from the nearest chip's sensing "
                      f"point; brass, which R10 allows within 15 mm of a chip"),
        print_plan=dict(
            printer="Prusa XL 5T, 0.4 mm nozzles; tool 1 PLA (carrier and cap), tool 2 Siraya Tech TPU 85A (tip)",
            orientation="cartridge on its dorsal face (x_F 11.25 on the bed); spacer printed flat on its board side, "
                        "separately or beside the cartridge in the same job",
            layers=f"PLA 0.20 mm to z {zp['base_top']:.2f}, 0.10 mm from {zp['base_top']:.2f} to {zp['cap_top']:.2f} (pocket "
                   f"and cap); tip 0.20 mm (block) or 0.10 mm (curved tips)",
            pause_board=dict(z=zp["pause_board"], what="M601 after the layer ending at z 4.60: press the board (chips up, "
                             "pigtail soldered) onto the four pins until it sits on the floor, lay the wire ribbon in the "
                             "spacer's channel and the tab's notch, drape the bundle over the -y wing and tape it to the bed, "
                             "put the spacer on the board, resume"),
            first_layer_after_pause=f"PLA cap at z {zp['pause_board']:.2f}-{zp['cap_top']:.2f} (two 0.10 mm layers), the "
                                    "first one printed onto spacer, rims and pin tops; nothing protrudes above z 4.60 "
                                    "(spacer as tall as the tallest part), so no Z-hop or travel rule is needed",
            tool_change=f"one, PLA to TPU at z {zp['cap_top']:.2f}",
            pause_magnets=pauses,
            bed="60 C (PLA) for the whole job",
            spacer_thickness="1.00 mm until the board's parts are measured; then the tallest measured part, rounded up to 0.05 mm"),
        printability=dict(
            supports="none: every face of the carrier is a vertical wall or a top surface; the cap's first layer bridges "
                     "the five 3.4 mm windows and the 4.5 x 1.8 mm solder window; the 85A tips are a block or domes",
            walls=walls,
            thin_features="side and distal rims 0.40 mm (one Arachne line) because R5 limits the cartridge to 21.0 mm and the "
                          "board is 20.0 mm wide; they cover the FR-4's edge and take no load; spacer ligaments 0.85 mm; "
                          "all other walls 0.85 mm or more",
            body="palmar face down (x_F -10.55 on the bed): the bridge's palmar face is the largest flat face; the arms' "
                 "holes then lie horizontal (ream the 5.2 mm bore); the lip over the tower is a 1.6 mm overhang; the 1.60 mm "
                 "pilot for the M2 screw is vertical and tapped after printing"),
        assembly=[
            "Cut the board's tab at v = -15.0 mm (4.9 mm long), desolder J1, solder four 32 AWG PTFE wires (VDD, GND, SDA, "
            "SCL) to J1's signal pads and dress them as a flat ribbon toward -y; crimp the inline connector at the far end.",
            "Measure the FR-4 thickness and the tallest part; set t_fr4 and t_sp in the generator, regenerate, slice.",
            "Print the spacer flat; check its thickness with a micrometer.",
            "Print the cartridge: PLA to the board pause (z 4.60), place board and spacer, resume; PLA cap; tool change; "
            "85A tip; magnets at the second pause, north pole toward the palmar side.",
            "Print the DIP link body; tap the 1.60 mm pilot M2.",
            "Lay the wire bundle in the -y shoulder's groove and up the post's groove; slide the cartridge onto the body "
            "along +z under the loosened screw until the shank meets the slot's end; tighten the M2 brass screw.",
            "Route the pigtail across the PIP joint with a service loop (pending: the cable's path on the MCP link), plug "
            "the inline connector, read the zero-load field."],
        replacement=dict(
            cartridge=("Loosen the M2 screw half a turn, slide the cartridge 5 mm distally (the tower leaves the lip, the "
                       "slot leaves the screw), lift the bundle out of the grooves, unplug the inline connector; reverse "
                       "with a spare cartridge. About 2 minutes; the body, the horn and the PIP joint are not touched. "
                       "Re-read the zero-load field after every swap."),
            tip="a worn or torn tip is replaced as a whole cartridge: the board is inside the print",
            board_recovery=("From a failed cartridge: cut the 85A tip off at the cap, peel the TPU plugs off the chip tops, "
                            "cut the 0.2 mm cap along the rims and around the four pin tops, lift the spacer and pull the "
                            "board off the pins (the crush ribs give); the pigtail stays soldered. Risk to the board: peel "
                            "and pry loads on U1-U5's solder joints."),
            spares="each cartridge contains a board, so a spare cartridge costs a board (two spare boards per hand)"),
        parts_list=[
            dict(item="DIP link body", material="PLA, 15 % infill as the links", qty_per_finger=1,
                 file=f"cad/{PFX}-body.stl; cad/{PFX}-body.step.zip (faceted); cad/{PFX}-body_middle.step (new middle, solids)"),
            dict(item="cartridge carrier with cap (tool 1)", material="PLA, solid", qty_per_finger=1,
                 file=f"cad/{PFX}-carrier.step"),
            dict(item="tip (tool 2)", material="Siraya Tech TPU 85A, 30 % gyroid", qty_per_finger=1,
                 file=f"cad/{PFX}-tpu_block4.stl and .step.zip (or block5, cap8.5, ell16, ell11)"),
            dict(item="spacer", material="PLA, solid", qty_per_finger=1, file=f"cad/{PFX}-spacer.step"),
            dict(item="ReSkin 5X board, tab cut to 4.9 mm, J1 removed", qty_per_finger=1),
            dict(item="N52 disc magnet 3 x 2 mm", qty_per_finger="4 (5 for block5)"),
            dict(item=sc["item"], qty_per_finger=1),
            dict(item="32 AWG PTFE wire, four conductors, 120 mm, and an inline 4-pin connector", qty_per_finger=1),
            dict(item="M2 x 0.4 hand tap", qty="1 per workshop")],
        contact_geometry=("Block: unchanged outer surface (the readiness print, 17.0 x 14.8 x 21.41 mm); its back face is "
                          "printed onto the PLA cap at x_F 6.45 instead of resting on the 3.0 mm plate; its mount face is "
                          "printed 0.10 mm proud and compressed to z_F -67.30 on assembly. Behind the back plane the "
                          "cartridge is 21.0 mm wide (0.5 mm rim past the 20.0 mm board, 3.1 mm past each side of the tip) "
                          "from x_F 6.45 to 11.25; the current plate placement overhangs 2.6 mm at x_F 9.45-11.95. Curved "
                          "tips keep their outer surfaces but print back face down with pocket seats instead of upright with "
                          "slots, so their contact surfaces are stacked from 0.10 mm layers across the finger."),
        requirements=dict(
            R1=dict(status="not met by default", value=f"sensing plane {D['gap']:.2f} mm behind the back face with the 1.00 mm "
                    f"spacer over 0.90 mm chips; {D['gap_chip_spacer']:.2f} mm with the spacer at the measured chip height; "
                    f"{D['gap_thin_cap']:.2f} mm (the threshold) with that spacer and a single 0.10 mm cap layer"),
            R2=dict(status="met", value="back face printed onto the PLA cap; five 3.4 mm windows over U1-U5 (58 mm2, 18 % of "
                    "317 mm2) bridged by the tip; mount face pressed 0.10 mm onto the bridge"),
            R3=dict(status="partly met", value="clamped by printing instead of screw preload: zero clearance when the cap "
                    "solidifies, no defined preload; located in its plane by two crush-rib pins (interference instead of "
                    "0.05 mm clearance); independent of the tip and of TPU friction"),
            R4=dict(status="partly met", value="tip printed onto the rigid cap, no feature or fastener near the magnet pockets "
                    "or in the palmar half; the tip comes off only together with the board (cartridge exchange)"),
            R5=dict(status="met", value=f"outer surface unchanged; cartridge {2 * chk['widths_half_mm']['carrier']:.1f} mm wide "
                    "over the tip's length; FR-4 edge covered by 0.40 mm rims; nothing palmar of the back plane beyond the tip"),
            R6=dict(status="met", value="arms cut from the STL 2 um outside their inner faces; symmetric difference 0.0 mm3 "
                    "beyond 5 um of those faces"),
            R7=dict(status="met", value=f"rigid parts at least {rigid:.3f} mm (body {cl['body_new']['min_mm']:.3f}, carrier "
                    f"{cl['carrier']['min_mm']:.3f}); cable {cl['cable_fixed']['min_mm']:.3f} mm against 1.0"),
            R8=dict(status="met up to the PIP crossing", value="tab cut to 4.9 mm, four wires soldered to J1's signal pads; the "
                    "ribbon is captured under the cap at the notch and the bundle runs in the body's grooves to the post's "
                    "top; its crossing of the PIP joint and the connector on the MCP link side are pending"),
            R9=dict(status="partly met", value="tip and board are exchanged together as a cartridge with one screw in about "
                    "2 minutes, the DIP link body staying on the finger; the board alone comes out only by cutting the "
                    "cartridge open"),
            R10=dict(status="met", value=f"one brass M2 screw, {min(chk['screw_to_chips_mm'].values()):.1f} mm from the nearest "
                     f"chip; chips {cl['chips_to_servo']['min_mm']:.2f} mm from the PIP servo over the range "
                     f"({cl['chips_to_servo_plate']['min_mm']:.2f} mm at the current plate); the tab's end "
                     f"{cl['board_to_servo']['min_mm']:.2f} mm"),
            R11=dict(status="partly met", value="PLA; board seat a top surface of the carrier; side and distal rims 0.40 mm, "
                     "under the 0.9 mm wall rule because R5 limits the width to 21.0 mm"),
            R12=dict(status="met", value=f"{chk['mass']['total_without_tip_g']:.2f} g against 7.6 g")),
        risks=[
            "No defined preload on the board's stack: where the cap's first layer does not bear on the spacer, the board can "
            "move by the residual gap, and 1 um toward the magnets reads as 0.160 N of normal force.",
            "A worn tip costs a board: the cartridge is the replaceable unit; recovery means cutting the cap and peeling the "
            "TPU off the chips.",
            "The crush-rib fit in plated 2.40 mm holes is unmeasured; undersized ribs leave about 0.1 mm of in-plane play.",
            "Nozzle strike at the first layer after pause P1 if the spacer is thinner than the tallest part.",
            "The cap's first layer goes onto PLA that cooled during the pause; weak fusion lets the stack lift.",
            "Curved tips change print orientation (back face down, pocket seats), so their printed stiffness and surface "
            "differ from the upright prints.",
            "The cartridge takes +x loads in the screw's preload and the -y lip; the M2 thread is tapped in PLA and wears over "
            "many exchanges.",
            "The cable's crossing of the PIP joint is not designed."],
        pending=["FR-4 thickness, part heights and outline of the owner's board (sets t_fr4, t_sp and the pocket)",
                 "crush-rib fit of a printed PLA pin in a plated 2.40 mm hole (print a coupon: pins of 2.45, 2.55, 2.65 mm)",
                 "TPU 85A peel strength from PLA and from the MLX90393 package after printing",
                 "zero-load field before and after 50 presses at 3 N and after 1 h at 5 N (creep of the board's stack in the print)",
                 "the cable's route across the PIP joint and on the MCP link",
                 "the PIP servo's motor field at the chips over the PIP range"])


def cmd_eval(a):
    D = json.loads(DESIGN.read_text())
    chk = json.loads((WORK / "check.json").read_text())
    sen = json.loads((WORK / "sense.json").read_text())
    mesh = json.loads((WORK / "mesh_info.json").read_text())
    cad = json.loads((WORK / "cad_info.json").read_text())
    step = json.loads((WORK / "step_info.json").read_text()) if (WORK / "step_info.json").exists() else {}

    def row(layout, recipe, case):
        for r in sen["rows"]:
            if r["layout"] == layout and r["recipe"] == recipe and r["case"] == case:
                return r

    fr = {}
    for case in ("concept_c", "chip_spacer", "thin_cap", "touch", "plate"):
        fr[case] = {f"{lay.split('_')[0]}_{rec[-2:]}": row(lay, rec, case)["centre_force_N"]
                    for lay in ("four_d3x2_d5", "five_d3x2_d5") for rec in ("w85_30", "w85_50")}
    r4 = row("four_d3x2_d5", "w85_30", "concept_c")
    r5 = row("five_d3x2_d5", "w85_30", "concept_c")
    clear = chk["clearance"]
    rigid = {k: v for k, v in clear.items() if v["need_mm"] == 0.5}
    worst = min(rigid.items(), key=lambda kv: kv[1]["min_mm"])
    files = sorted(str(p.relative_to(REPO)) for p in OUT.rglob("*") if p.is_file())
    ev = dict(
        title="DIP-link redesign, concept c: co-printed tip cartridge with the board captured at a pause",
        date="2026-10-18", status="model; nothing printed; board dimensions, part heights and flatness pending",
        generator="scripts/dip_link_concept_c.py", spec="docs/experiments/20261018-dip_link_board_mount/spec.json",
        terms=dict(
            gap="distance from the tip's back plane (x_F 6.45, top of the cap) to the chips' sensing plane, 0.42 mm under "
                "the package top, mm; the current DIP link gives 3.42",
            chip_to_magnet="shortest distance from a chip's sensing point to the surface of a magnet (3 x 2 mm disc), mm",
            force_resolution="Cramer-Rao bound (one standard deviation, N) of the normal force at the contact point from "
                             "one 141 Hz sample of the five chips, contact position and a uniform background field "
                             "unknown (scripts/tpu_tip_magnet_sensing.py), 85A block of the readiness finite elements",
            false_force="normal force or shear (N) that the estimator reads per micrometre of board motion relative to "
                        "the magnets",
            clearance="least distance (mm) from a part fixed to the DIP link to the MCP link and the PIP servo over PIP "
                      "-18..92 deg (dip_link_geometry.points_clearance_exact, 0.25 mm samples, 0.5 deg steps)"),
        design=dict(
            stack_x_F=dict(tip=[D["x_palmar"], D["x_back"]], cap=[D["x_back"], D["x_sp_top"]],
                           spacer_and_chips=[D["x_sp_top"], D["x_comp"]], fr4=[D["x_comp"], D["x_floor"]],
                           base=[D["x_floor"], D["x_bed"]]),
            parameters={k: D[k] for k in ("t_cap", "t_sp", "t_fr4", "t_base", "c_xy", "pin_d", "rib_d", "sp_hole_d",
                                          "cap_win", "y_out", "x_head", "tpu_preload")},
            screw=D["screw"], cable=D["cable"]),
        sensing=dict(
            gap_mm=round(D["gap"], 3), gap_spacer_at_chip_height_mm=round(D["gap_chip_spacer"], 3),
            gap_thin_cap_mm=round(D["gap_thin_cap"], 3), gap_current_mm=3.42,
            gap_note="default spacer 1.00 mm (datasheet maximum part height) over 0.90 mm chips; a spacer cut to the "
                     "measured chip height gives 0.62 mm, and a single 0.10 mm cap layer on top of that 0.52 mm (R1)",
            chip_to_magnet_mm=dict(four=min(r4["chip_to_magnet_surface_mm"]), five=min(r5["chip_to_magnet_surface_mm"]),
                                   four_U1=r4["chip_to_magnet_surface_mm"][0], five_U1=r5["chip_to_magnet_surface_mm"][0],
                                   centre_U1_four=r4["chip_to_magnet_centre_mm"][0]),
            force_resolution_N=fr, field_four_30_mT=dict(B_chips=r4["B_chips_mT"], B_U1=r4["B_U1_mT"], gain_sel=r4["gain_sel"]),
            field_five_30_mT=dict(B_chips=r5["B_chips_mT"], B_U1=r5["B_U1_mT"], gain_sel=r5["gain_sel"]),
            false_force_four_30=r4["false_force_N_per_um"], curved_tips_static=sen["tips_static"],
            inputs_changed="only the sensing plane: gaps 0.72, 0.62 and 0.52 mm in place of the specification's "
                           "placements; magnet motions from the readiness finite elements of the 85A block, which fixed "
                           "the back face and the mount face; here the back face is printed onto the PLA cap and the mount "
                           "face is pressed 0.10 mm against the bridge"),
        clearance=clear, worst_rigid=dict(new_part=worst[0], **worst[1]), gaps=chk["gaps"],
        widths_half_mm=chk["widths_half_mm"], screw_to_chips_mm=chk["screw_to_chips_mm"],
        protrusion=chk["protrusion"], walls_designed_mm=chk["walls_designed_mm"], mass=chk["mass"],
        arms=dict(horn_side=mesh["arm_horn_side"], slotted=mesh["arm_slotted"]),
        **narrative(D, chk), cad=cad, step=step, mesh=mesh, files=files)
    ev["risks"].append(f"R1 is missed by {D['gap'] - 0.52:.2f} mm with the default spacer: {fr['concept_c']['four_30']:.3f} N "
                       f"against {fr['touch']['four_30']:.3f} N at the touch placement (four magnets, 30 % gyroid); the "
                       f"spacer at chip height and a 0.10 mm cap give {fr['thin_cap']['four_30']:.3f} N.")
    EVAL.write_text(json.dumps(ev, indent=1))
    print("eval:", EVAL.relative_to(REPO), flush=True)


# ============================================================================================================== main

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["design", "cad", "mesh", "step", "check", "sense", "render", "eval", "all"])
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    me = str(Path(__file__).resolve())
    if a.cmd == "cad":
        subprocess.run(["freecadcmd", me], env=dict(os.environ, DLC_STEP="cad"), stdin=subprocess.DEVNULL, check=True)
    elif a.cmd == "step":
        subprocess.run(["freecadcmd", me], env=dict(os.environ, DLC_STEP="step"), stdin=subprocess.DEVNULL, check=True)
    elif a.cmd == "all":
        for c in ("design", "cad", "mesh", "step", "check", "sense", "render", "eval"):
            subprocess.run([sys.executable, me, c], check=True)
    else:
        dict(design=cmd_design, mesh=cmd_mesh, check=cmd_check, sense=cmd_sense, render=cmd_render, eval=cmd_eval)[a.cmd](a)


if os.environ.get("DLC_STEP") == "cad":          # freecadcmd runs this file with DLC_STEP set (see main)
    fc_main_cad()
elif os.environ.get("DLC_STEP") == "step":
    fc_main_step()
elif __name__ == "__main__":
    main()
