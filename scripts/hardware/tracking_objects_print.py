#!/usr/bin/env python3
"""Print files of the tracked objects of the object seating and tracking plan (docs/notes/20261105-object_seating_and_
tracking/): the 25 mm x 100 mm cylinder and the 40 mm sphere of the HOM tasks, each with its hexagonal tag block on a
stem below it (owner, 2026-11-12: stems for both layout sets, grip tape of about 0.8 mm on the cylinder, the sphere bare,
the sphere at the plan's two clear layouts).

Each object is printed in parts that are glued: the block and its stem in one piece, upright like the test block of
2026-11-07, so the stem lies on the block's axis; the stem ends in a 45 deg cone, a flat seat SEAT_D across and a
rectangular key, which sit in a matching flat and slot on the object and fix its turn about the stem. The cylinder is
printed standing, so it comes out round, at the taped diameter less twice the tape; its end faces bear tag36h11 ids 22
and 23 at 13 mm, so a bundle adjustment of the assembled object measures the cylinder against its block and the glue
joint's error drops out. The sphere is printed as two hemispheres, flat side down, joined by three dowels. The geometry
is object_tracking_mjcf.OBJECTS and object_map, the tracker's: the block's top lies stem_mm below the object's taped or
bare surface (rows/stem_sv25_sv55db.json: 15 mm below the lowest finger point of the plan's five layouts and the measured
servo's five best per task).

Files lie in print orientation on z = 0, each two-colour part as a white body and its black cells in one frame. `build`
(femtip, manifold3d) writes them; `check` (camera stack) draws every tag face of the black bodies and detects it against
the map; `bambu` writes one Bambu Studio project of all parts (black on filament 1, white on 2). `build` also writes two
alternatives of the cylinder, with end tag 23 only and without end tags (owner, 2026-11-13: the end tags "cause a lot of
pla usage"), and `bambu --cylinder NAME` slices the plate with one of them in a project of its own.

`blocks build|check|bambu` writes the blocks' second version into assets/tracking/20261116-tag_blocks/: the stem's root
widened to ROOT_D and filleted to the block's top (owner, 2026-11-16), the rest as in the first version.

    ~/.claude/bin/resguard.sh run --mem 3G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python scripts/hardware/tracking_objects_print.py build
    ~/.claude/bin/resguard.sh run --mem 2G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/bin/python scripts/hardware/tracking_objects_print.py check [--preview PNG]
    ~/.claude/bin/resguard.sh run --mem 3G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/bin/python scripts/hardware/tracking_objects_print.py bambu
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths  # noqa: E402,F401
import object_tracking_mjcf as OM  # noqa: E402
import object_tracking_print as OP  # noqa: E402
from sr2hand.tracking import tags as TG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATE = "20261112"
OUT = ROOT / f"assets/tracking/{DATE}-objects"
REPORT = OUT / f"{DATE}-objects.json"
PROJECT = OUT / f"{DATE}-objects_p1s.3mf"
SEAT_D, CONE_H = 10.0, 3.0      # mm: the flat seat between stem and object, and the 45 deg cone up to it
KEY = (2.4, 5.0, 4.0)           # mm: the key on the seat along x_B, y_B and up
KEY_CLEAR, SLOT_EXTRA = 0.1, 0.5   # mm: the slot's clearance per side and its depth beyond the key
END_DEPTH = 0.6                 # mm: the black of the cylinder's end tags, three 0.2 mm layers
PIN_D, PIN_HOLE_D, PIN_R, PIN_DEPTH = 2.9, 3.1, 10.0, 4.0   # mm: the sphere's dowels, their holes, radius, depth per half
PIN_ANGLES = (90.0, 210.0, 330.0)
SEG = 160                       # segments of the round surfaces
PLA, MASS_G = 1.24e-3, 25.0     # g/mm^3; the HOM objects' mass with stem, block and tape
# the blocks' second version (owner, 2026-11-16: "we probably shouldve filleted the mounting point between the rod and
# the base of the apriltag hex block"; "no breaks, just anticipating problems"): the stem's root widens to ROOT_D at the
# block's top and tapers to the stem over ROOT_H, joined to the top face by a concave fillet of FILLET_R. It stays 5 mm
# below the lowest finger point, so the clearances, the tag faces and the maps are those of the first version.
ROOT_D, ROOT_H, FILLET_R = 6.0, 10.0, 3.0      # mm
FINGER_GAP = 15.0               # mm from the block's top up to the lowest finger point (the stems' design, above)
BLOCKS_DATE = "20261116"
BLOCKS_OUT = ROOT / f"assets/tracking/{BLOCKS_DATE}-tag_blocks"
BLOCKS_REPORT = BLOCKS_OUT / f"{BLOCKS_DATE}-tag_blocks.json"
BLOCKS_PROJECT = BLOCKS_OUT / f"{BLOCKS_DATE}-tag_blocks_p1s.3mf"
BLOCKS = ("cylinder_block", "sphere_block")
SPHERE_LAYOUTS = ((70.0, 110.0), (75.0, 55.0))   # the sphere's layouts in the series (plan, decision 2)
# The cotter joint (owner, 2026-11-16: "can we think of ways to attach the hex block to the cylinder/sphere without
# glue? ... some kind of joinery perpendicular to the block shaft ... tight enough to prevent slop but fit enough to
# insert (and remove in the case of a break)"). The stem ends in a square taper whose faces lie in the planes of a
# matching socket in the object, and above it in a square post with a slot across the stem. A cotter, a pin with a 1:5
# taper on its top face, is pushed through the object's slot and the post's slot, under the post's ceiling, which slopes
# as the cotter does: it draws the taper into the socket, which then fixes the block in all six degrees of freedom, and
# friction holds it there. It comes out by a push on its tip through the slot's far opening. Joint frame J, coordinates
# (w, v, u): u along the stem into the object, 0 at the seat plane; w along the cotter toward its entry; v = u x w. The
# square's corners lie on the w and v axes, so in the standing cylinder the socket's upper faces print at 45 deg.
J_DATE = "20261116"
J_OUT = ROOT / f"assets/tracking/{J_DATE}-cotter_joint"
J_REPORT = J_OUT / f"{J_DATE}-cotter_joint.json"
TAPER_DEG = 20.0                    # the taper's faces to the stem
SQ0, SQ_H = 6.5, 2.0                # mm: the socket's side at the seat plane and the taper's depth; straight beyond
MALE_U = (-0.7, 1.2)                # mm: the stem's taper along u, its faces in the socket's planes
CONE_R = 4.98                       # mm: the stem's cone up to the taper, inside the flat's 5.01 mm
POST_SIDE, POST_HALF_W, POST_TOP = 4.7, 2.2, 7.8   # mm: the post's side, its half length along w, its top
SOCKET_TOP = 8.3                    # mm: the socket's end along u
KEY_V, KEY_CUT = 3.47, 0.15         # mm: the socket's -v corner filled beyond v = -KEY_V, the stem's cut KEY_CUT inside
FLOOR_U, SLOT_W = 3.0, 2.6          # mm: the object's slot floor, where the cotter rests, and the slots' width along v
POST_SLOT_U = 2.5                   # mm: the post's slot floor, below FLOOR_U so that the cotter rests on the object
COT_W, COT_T, COT_SLOPE, COT_ZONE = 2.4, 2.5, 0.2, 3.7   # mm: width, thickness at w = 0, 1:5 taper over |w| <= ZONE
COT_ENDS = (-6.85, 6.85)            # mm: the cotter's tip and thick end along w at its nominal place
COT_SIZES = {"thin": -0.25, "mid": 0.0, "thick": 0.25}   # mm added to the thickness; 1, 2 and 3 dimples
TRAVEL = 1.5                        # mm of the cotter along w each way: +-0.3 mm of fit along u at 1:5
SLOT_CLEAR = 0.3                    # mm above the thickest cotter in the object's slots
J_B = {"cylinder": np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]]),    # columns w, v, u in B: the cylinder's
       "sphere": np.array([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 1.0]])}       # cotter enters at -x_O, the sphere's at +z_O
# O to print: the cylinder stands on its -z_O end; the hemispheres lie on the equator, the lower one pole up
R_PRINT = {"cylinder": np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
           "sphere_lower": np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),
           "sphere_upper": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])}


def seat(shape):
    """(printed radius, depth of the flat seat below the printed surface, the seat plane's y_O) of an object."""
    o = OM.OBJECTS[shape]
    r = o["r_mm"] - o["tape_mm"]
    depth = r - math.sqrt(r * r - (SEAT_D / 2) ** 2)
    return r, depth, -(r - depth)


def m4(R, t=(0.0, 0.0, 0.0)):
    return np.hstack([np.asarray(R, float), np.asarray(t, float)[:, None]])


def stem_root():
    """(the stem's root of the second version on z = 0, the block's top; its foot's radius on the top face, mm): a cone
    from ROOT_D at z = 0 to the stem at ROOT_H, joined to the top face by a concave fillet of FILLET_R, revolved from
    its (r, z) profile. It reaches 0.5 mm below the top face, as the stem does."""
    import manifold3d as M
    r0, rs, R = ROOT_D / 2, OM.STEM_D / 2, FILLET_R
    n = np.array([ROOT_H, r0 - rs]) / math.hypot(ROOT_H, r0 - rs)      # the cone's outward normal in (r, z)
    c = np.array([r0 + R * (1.0 - n[1]) / n[0], R])                     # the fillet's centre: R from the face and the cone
    arc = [c + R * np.array([math.cos(t), math.sin(t)])
           for t in np.linspace(-math.pi / 2, math.atan2(-n[1], -n[0]), 48)]
    pts = [(0.0, -0.5), (float(c[0]), -0.5)] + [(float(p[0]), float(p[1])) for p in arc] + [(rs, ROOT_H), (0.0, ROOT_H)]
    return M.Manifold.revolve(M.CrossSection([pts]), SEG), float(c[0])


def block_stem(shape, root=False, joint=False):
    """(white, black, the seat plane's height) of the block and its stem in the block frame B moved up by half the block
    (print frame); `root` adds the second version's root (stem_root), `joint` ends the stem in the cotter joint's taper
    and post (male) in place of the seat's flat and key."""
    import manifold3d as M
    B, o = OM.BLOCK, OM.OBJECTS[shape]
    _r, _d, seat_y = seat(shape)
    z_seat = B.height / 2 + seat_y + (o["r_mm"] + o["stem_mm"])      # the seat plane in B: the block's top is r + stem below O
    rs = OM.STEM_D / 2
    if joint:                                                         # the 45 deg cone up to just outside the flat
        z_cone, r_cone, h_cone = z_seat + MALE_U[0], CONE_R, CONE_R - rs
        end = male().transform(m4(J_B[shape], (0.0, 0.0, z_seat)))
    else:
        z_cone, r_cone, h_cone = z_seat, SEAT_D / 2, CONE_H
        end = M.Manifold.cube([KEY[0], KEY[1], KEY[2]]).translate([-KEY[0] / 2, -KEY[1] / 2, z_seat])
    prism = M.Manifold.extrude(M.CrossSection([B.vertices()[:6, :2].tolist()]), B.height).translate(
        [0.0, 0.0, -B.height / 2])
    stem = M.Manifold.cylinder(z_cone - h_cone - B.height / 2 + 0.5, rs, rs, SEG).translate([0.0, 0.0, B.height / 2 - 0.5])
    cone = M.Manifold.cylinder(h_cone, rs, r_cone, SEG).translate([0.0, 0.0, z_cone - h_cone])
    parts = [prism, stem, cone, end] + ([stem_root()[0].translate([0.0, 0.0, B.height / 2])] if root else [])
    S = M.Manifold.batch_boolean(parts, M.OpType.Add)
    cells = OP.black_cells(OM.block_map(shape), OP.BLOCK_DEPTH)
    lift = np.eye(3), (0.0, 0.0, B.height / 2)
    return (S - cells).transform(m4(*lift)), (S ^ cells).transform(m4(*lift)), z_seat + B.height / 2


def slot(seat_y):
    """The key's slot in the object, from the seat plane into +y_O, its long side along z_O (= -y_B)."""
    import manifold3d as M
    w, l, d = KEY[0] + 2 * KEY_CLEAR, KEY[1] + 2 * KEY_CLEAR, KEY[2] + SLOT_EXTRA
    return M.Manifold.cube([w, d + 0.5, l]).translate([-w / 2, seat_y - 0.5, -l / 2])


def flat(r, seat_y):
    """What the flat seat removes below the plane y_O = seat_y: a cylinder SEAT_D across along y_O."""
    import manifold3d as M
    h = seat_y + r + 1.0
    return M.Manifold.cylinder(h, SEAT_D / 2 + 0.01, SEAT_D / 2 + 0.01, SEG).transform(
        m4(np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]), (0.0, -r - 1.0, 0.0)))  # its axis along +y_O, up to seat_y


def half_diag(u):
    """Half-diagonal of the taper's square at u (mm)."""
    return (SQ0 - 2.0 * u * math.tan(math.radians(TAPER_DEG))) / math.sqrt(2.0)


def diamond(h0, u0, u1, h1=None):
    """A square prism (or frustum) in J from u0 to u1, its corners on the w and v axes, half-diagonal h0 at u0 (h1 at u1)."""
    import manifold3d as M
    sq = M.CrossSection([[(h0, 0.0), (0.0, h0), (-h0, 0.0), (0.0, -h0)]])
    return M.Manifold.extrude(sq, u1 - u0, 0, 0.0, (h1 / h0,) * 2 if h1 else (1.0, 1.0)).translate([0.0, 0.0, u0])


def wu_prism(poly, v0, v1):
    """A prism in J of the polygon [(w, u)] from v0 to v1."""
    import manifold3d as M
    a = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]))
    p = M.Manifold.extrude(M.CrossSection([poly if a > 0 else poly[::-1]]), v1 - v0)     # (x, y, z) = (w, u, v1 - v)
    return p.transform(m4(np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]), (0.0, v1, 0.0)))


def box(lo, hi):
    import manifold3d as M
    return M.Manifold.cube((np.asarray(hi) - np.asarray(lo)).tolist()).translate(list(lo))


def cotter_t(w, size=0.0):
    """The cotter's thickness (mm) at w along it, at its nominal place."""
    return COT_T + size + COT_SLOPE * max(-COT_ZONE, min(COT_ZONE, w))


def slot_heights():
    """(entry, far) heights of the object's slot above FLOOR_U: the thickest cotter at its thickest point on each side of
    the socket over its travel, and SLOT_CLEAR."""
    big, hs = max(COT_SIZES.values()), half_diag(SQ_H)
    return cotter_t(COT_ZONE, big) + SLOT_CLEAR, cotter_t(-hs + TRAVEL, big) + SLOT_CLEAR


def male(ceiling=0.0):
    """The stem's end in J: the taper from MALE_U[0] to MALE_U[1], the post up to POST_TOP with the slot whose ceiling
    slopes as the cotter's top (raised by `ceiling` mm, for the fit checks), and the key's cut at the -v corner."""
    post = diamond(POST_SIDE / math.sqrt(2.0), MALE_U[1] - 0.01, POST_TOP) ^ box(
        (-POST_HALF_W, -10.0, 0.0), (POST_HALF_W, 10.0, 20.0))
    ceil = [(10.0, FLOOR_U + ceiling + COT_T + COT_SLOPE * 10.0), (-10.0, FLOOR_U + ceiling + COT_T - COT_SLOPE * 10.0)]
    slot_ = wu_prism([(-10.0, POST_SLOT_U), (10.0, POST_SLOT_U)] + ceil, -SLOT_W / 2, SLOT_W / 2)
    key = box((-10.0, -20.0, -5.0), (10.0, -(KEY_V - KEY_CUT), 20.0))
    return (diamond(half_diag(MALE_U[0]), MALE_U[0], MALE_U[1], half_diag(MALE_U[1])) + post) - slot_ - key


def socket():
    """What the joint removes from the object, in J: the taper's socket and its straight part up to SOCKET_TOP, its -v
    corner filled beyond -KEY_V, and the slot along w, taller on the entry side (+w) than on the far side; both run out
    through the object's surface."""
    hs = half_diag(SQ_H)
    cav = diamond(half_diag(-0.01), -0.01, SQ_H, hs) + diamond(hs, SQ_H - 0.01, SOCKET_TOP)
    cav = cav - box((-15.0, -30.0, -10.0), (15.0, -KEY_V, 30.0))
    h_in, h_far = slot_heights()
    return (cav + box((0.0, -SLOT_W / 2, FLOOR_U), (40.0, SLOT_W / 2, FLOOR_U + h_in))
            + box((-40.0, -SLOT_W / 2, FLOOR_U), (0.0, SLOT_W / 2, FLOOR_U + h_far)))


def cotter(size=0.0, dimples=0, at=0.0):
    """The cotter in J at its nominal place moved by `at` mm along w: its bottom on FLOOR_U, its top at FLOOR_U +
    cotter_t, a 0.5 mm chamfer at the tip's top edge, and `dimples` marks 0.7 mm across on its +v face by the thick end."""
    import manifold3d as M
    a, b = COT_ENDS

    def top(w):
        return FLOOR_U + cotter_t(w, size)
    poly = [(a, FLOOR_U), (b, FLOOR_U), (b, top(b)), (COT_ZONE, top(COT_ZONE)), (-COT_ZONE, top(-COT_ZONE)),
            (a + 0.5, top(a)), (a, top(a) - 0.5)]
    c = wu_prism(poly, -COT_W / 2, COT_W / 2)
    for k in range(dimples):
        d = M.Manifold.cylinder(0.5, 0.35, 0.35, 24).transform(
            m4(np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]), (b - 0.9 - 1.0 * k, COT_W / 2 - 0.4, (FLOOR_U + top(b)) / 2)))
        c = c - d
    return c.translate([at, 0.0, 0.0])


def joint_T(shape):
    """J to O (3 x 4): the stem's axis, the seat plane at u = 0."""
    return m4(OM.R_OB @ J_B[shape], (0.0, seat(shape)[2], 0.0))


def cylinder_body(ids=OM.END_IDS, joint=False):
    """(white, black) of the cylinder in O turned to print, standing on its -z_O end, with the end tags `ids`; black is
    None when `ids` is empty. `joint` cuts the cotter joint's socket and slots (socket) in place of the key's slot."""
    import manifold3d as M
    r, _d, seat_y = seat("cylinder")
    L = OM.OBJECTS["cylinder"]["length_mm"]
    cut = socket().transform(joint_T("cylinder")) if joint else slot(seat_y)
    S = M.Manifold.cylinder(L, r, r, SEG).translate([0.0, 0.0, -L / 2]) - flat(r, seat_y) - cut
    T = m4(R_PRINT["cylinder"], (0.0, 0.0, L / 2))
    if not ids:
        return S.transform(T), None
    ends = TG.TagMap(OM.BLOCK.family, OM.BLOCK.tag_size)
    full = OM.object_map("cylinder")
    for i in ids:
        ends.frames[i], ends.corners[i] = full.frames[i], full.corners[i]
    cells = OP.black_cells(ends, END_DEPTH)
    return (S - cells).transform(T), (S ^ cells).transform(T)


def sphere_halves(joint=False):
    """(lower, upper) hemispheres of the sphere, each on its equator face (print frames), and one dowel pin; `joint` cuts
    the cotter joint's socket and slots in the lower half in place of the key's slot."""
    import manifold3d as M
    r, _d, seat_y = seat("sphere")
    ball = M.Manifold.sphere(r, SEG)
    big = 2 * r + 4
    low = ball ^ M.Manifold.cube([big, big / 2, big]).translate([-big / 2, -big / 2, -big / 2])
    up = ball ^ M.Manifold.cube([big, big / 2, big]).translate([-big / 2, 0.0, -big / 2])
    holes = []
    for a in PIN_ANGLES:
        x, z = PIN_R * math.cos(math.radians(a)), PIN_R * math.sin(math.radians(a))
        holes.append(M.Manifold.cylinder(2 * PIN_DEPTH, PIN_HOLE_D / 2, PIN_HOLE_D / 2, 48).transform(
            m4(np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]), (x, -PIN_DEPTH, z))))  # along y_O, PIN_DEPTH each side
    H = M.Manifold.batch_boolean(holes, M.OpType.Add)
    low = low - H - flat(r, seat_y) - (socket().transform(joint_T("sphere")) if joint else slot(seat_y))
    up = up - H
    pin = M.Manifold.cylinder(2 * PIN_DEPTH - 0.4, PIN_D / 2, PIN_D / 2, 48)
    return low.transform(m4(R_PRINT["sphere_lower"])), up.transform(m4(R_PRINT["sphere_upper"])), pin


def stl(name, colour="white"):
    return OUT / f"{DATE}-{name}_{colour}.stl"


PARTS = {"cylinder_block": "cylinder block and stem, ids 10-15", "sphere_block": "sphere block and stem, ids 16-21",
         "cylinder": "cylinder body, end tags 22 and 23", "sphere_lower": "sphere, lower hemisphere with the seat",
         "sphere_upper": "sphere, upper hemisphere", "sphere_pin": "dowel pin of the sphere (print three)"}
# the cylinder with fewer end tags (owner, 2026-11-13: "do we need the apriltag on the end of the cylinder? it causes a
# lot of pla usage"). Tag 22 lies on the printed cylinder's top, so with it the prime tower rises to 100 mm; tag 23
# prints in the first three layers. Each alternative replaces the cylinder on the plate (bambu --cylinder).
CYLINDERS = {"cylinder": OM.END_IDS, "cylinder_tag23": (23,), "cylinder_plain": ()}
ALTERNATIVES = {"cylinder_tag23": "cylinder body, end tag 23 only", "cylinder_plain": "cylinder body without end tags"}
PRINTED = {"cylinder": "cylinder_plain"}   # owner, 2026-11-13: "i sent the print with no cylinder end tag"


def cmd_build(a):
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"what": "the tracked cylinder and sphere with their tag blocks, as print files",
           "script": "scripts/hardware/tracking_objects_print.py", "objects": OM.OBJECTS,
           "seat": {s: dict(zip(("r_print_mm", "flat_depth_mm", "seat_y_O_mm"), map(float, seat(s))))
                    for s in ("cylinder", "sphere")},
           "parts": {}}
    if REPORT.exists():                   # the projects' slices stay until bambu slices again
        old = json.loads(REPORT.read_text())
        rep.update({k: old[k] for k in ("bambu", "bambu_variants") if k in old})
    bodies = {}
    for shape in ("cylinder", "sphere"):
        w, b, top = block_stem(shape)
        bodies[f"{shape}_block"] = (w, b)
        rep["parts"][f"{shape}_block"] = {"height_mm": round(top + KEY[2], 3)}
    bodies.update({name: cylinder_body(ids) for name, ids in CYLINDERS.items()})
    low, up, pin = sphere_halves()
    # the assembly: each block and stem moved back into O against its body; the seat only touches the flat and the key
    # clears its slot, so the shared volume is zero
    rep["assembly_overlap_mm3"] = {}
    for shape, body, R in (("cylinder", bodies["cylinder"][0] + bodies["cylinder"][1], R_PRINT["cylinder"]),
                           ("sphere", low, R_PRINT["sphere_lower"])):
        o = OM.OBJECTS[shape]
        w, b = bodies[f"{shape}_block"]
        blk = (w + b).transform(m4(OM.R_OB, OM.block_centre_O(o["r_mm"], o["stem_mm"])
                                   - OM.R_OB @ np.array([0.0, 0.0, OM.BLOCK.height / 2])))
        shift = np.array([0.0, 0.0, -o["length_mm"] / 2 if shape == "cylinder" else 0.0])
        rep["assembly_overlap_mm3"][shape] = abs(round((blk ^ body.transform(m4(R.T, shift))).volume(), 6))
        hexagon = 3 * OM.BLOCK.face_width * OM.BLOCK.across_flats / 2 * OM.BLOCK.height
        rep["parts"][f"{shape}_block"]["stem_solid_g"] = round(((w + b).volume() - hexagon) * PLA, 2)
    print("assembly overlap (mm3)", rep["assembly_overlap_mm3"])
    bodies.update(sphere_lower=(low, None), sphere_upper=(up, None), sphere_pin=(pin, None))
    ok = True
    for name, (w, b) in bodies.items():
        info = rep["parts"].setdefault(name, {})
        info["what"] = {**PARTS, **ALTERNATIVES}[name]
        ok &= export_part(info, w, b, lambda c, n=name: stl(n, c))
    rep["ok"] = bool(ok)
    REPORT.write_text(json.dumps(rep, indent=1))
    print("wrote", REPORT.relative_to(ROOT))
    if not ok:
        raise SystemExit(1)


def export_part(info, w, b, path):
    """Write a part's white and black bodies to path(colour) and its volumes, mass and meshes into `info`; True when
    the colours do not overlap and every mesh is a closed, consistently wound solid."""
    ok = True
    vol = {"white": w.volume(), "black": b.volume() if b is not None else 0.0}
    if b is not None:
        vol["overlap"] = (w ^ b).volume()
        ok &= vol["overlap"] < 1e-6
    info["volume_mm3"] = {k: round(v, 3) for k, v in vol.items()}
    info["solid_pla_g"] = round((vol["white"] + vol["black"]) * PLA, 2)
    for colour, m in (("white", w), ("black", b)):
        if m is None:
            continue
        t = OP.to_trimesh(m)
        good = bool(t.is_watertight and t.is_winding_consistent and t.volume > 0)
        ok &= good
        t.export(path(colour))
        info[colour] = {"file": str(path(colour).relative_to(ROOT)), "triangles": int(len(t.faces)),
                        "watertight": good, "shells": len(t.split(only_watertight=False)),
                        "bounds_mm": np.round(t.bounds, 3).tolist()}
    print(info["what"], "ok" if ok else "FAILED", info["volume_mm3"], info.get("white", {}).get("bounds_mm"), flush=True)
    return ok


def print_maps():
    """{part: [TagMap in its print frame]} of the two-colour parts."""
    out = {}
    for shape in ("cylinder", "sphere"):
        out[f"{shape}_block"] = [OP.shifted(OM.block_map(shape), OM.BLOCK.height / 2)]
    full = OM.object_map("cylinder").transformed(np.r_[m4(R_PRINT["cylinder"], (0.0, 0.0, 50.0)), [[0, 0, 0, 1.0]]])
    for name, ids in CYLINDERS.items():
        if ids:
            ends = TG.TagMap(full.family, full.size)
            for i in ids:
                ends.frames[i], ends.corners[i] = full.frames[i], full.corners[i]
            out[name] = [ends]
    return out


def check_tags(path, tms, part, tiles):
    """(rows, ok) of one black body's tag faces, each drawn from the mesh and detected against its print-frame map."""
    import trimesh
    from sr2hand.tracking import pose as TP
    mesh = trimesh.load(path, force="mesh")
    rows, ok = [], True
    for tm in tms:
        det = TP.detector(tm.family)
        for tid, (c, r, d, n) in tm.frames.items():
            cell = OP.edge(tm, tid) / TG.n_cells(tm.family)
            half = OP.edge(tm, tid) / 2 + 2 * cell
            img = OP.draw(OP.in_plane(mesh, c, n), c, r, d, half)
            found = TP.detect(det, img)
            want = OP.to_px(tm.corners[tid], c, r, d, half)
            err = float(np.sqrt(((found[tid] - want) ** 2).sum(1).mean())) / OP.PPM if tid in found else None
            rows.append({"id": tid, "found": sorted(found), "corner_rms_mm": None if err is None else round(err, 4)})
            tiles.append((f"{part} {tid}", img, found.get(tid), want))
            ok &= sorted(found) == [tid] and err is not None and err < 0.05
    print(part, rows, flush=True)
    return rows, ok


def cmd_check(a):
    rep = json.loads(REPORT.read_text())
    tiles, ok = [], True
    for part, tms in print_maps().items():
        rows, good = check_tags(stl(part, "black"), tms, part, tiles)
        rep["parts"][part]["check"] = rows
        ok &= good
    rep["check_ok"] = bool(ok)
    REPORT.write_text(json.dumps(rep, indent=1))
    if a.preview:
        OP.sheet(tiles, a.preview)
    write_readme(rep)
    print("check", "ok" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


# on the P1S bed (mm): each part's white body centred here; the prime tower clear of all
PLACE = {"cylinder_block": (45.0, 60.0), "sphere_block": (45.0, 110.0), "cylinder": (45.0, 165.0),
         "sphere_lower": (110.0, 60.0), "sphere_upper": (110.0, 120.0), "sphere_pin": (150.0, 60.0)}
TOWER = (190.0, 180.0)
# the projects' filaments (owner, 2026-11-13: "please have the blocks use the black pla on a1"): black on filament 1 for
# the black spool in AMS slot A1, white on filament 2 for the slot that holds the white; no third filament
FILAMENTS, COLOURS = {"black": 1, "white": 2}, ["#000000", "#FFFFFF"]


def cmd_bambu(a):
    items = []
    for name in PARTS:
        src = a.cylinder if name == "cylinder" else name
        it = {"name": name, "white": stl(src), "black": stl(src, "black") if stl(src, "black").exists() else None,
              "place": PLACE[name]}
        items.append(it)
    pins = [dict(items[-1], name=f"sphere_pin_{k}", place=(PLACE["sphere_pin"][0] + 12.0 * k, PLACE["sphere_pin"][1]))
            for k in range(3)]
    items = items[:-1] + pins
    variant = a.cylinder != "cylinder"
    project = OUT / f"{DATE}-objects_p1s_{a.cylinder}.3mf" if variant else PROJECT
    work = ROOT / f"logs/{DATE}-objects/bambu{'_' + a.cylinder if variant else ''}"
    q = OP.bambu_project(items, project, work, "tracked objects", tower=TOWER, colours=COLOURS, filament=FILAMENTS)
    rep = json.loads(REPORT.read_text())
    if variant:
        rep.setdefault("bambu_variants", {})[a.cylinder] = q
    else:
        rep["bambu"] = q
    REPORT.write_text(json.dumps(rep, indent=1))
    write_readme(rep)
    print(f"{project.relative_to(ROOT)}: {q['prediction_s'] / 3600:.1f} h, {q['layers']} layers, "
          f"{q['filament_loads']} filament loads, used {q['used_g']} g")
    print("problems:", q["problems"] or "none")
    if q["problems"]:
        raise SystemExit(1)


def blocks_stl(name, colour="white"):
    return BLOCKS_OUT / f"{BLOCKS_DATE}-{name}_{colour}.stl"


def bending(shape):
    """Nominal bending stress in the stem per newton of side force at the object's centre (MPa/N), M / (pi d^3 / 32):
    at the first version's root, and at the second version's root and at the top of its cone, where its largest
    nominal stress lies. The first version's square corner raises its value by a factor of about 2."""
    o = OM.OBJECTS[shape]
    arm = o["r_mm"] + o["stem_mm"]                  # mm from the block's top to the object's centre

    def s(d, a):
        return round(32.0 * a / (math.pi * d ** 3), 2)
    return {"arm_mm": arm, "v1_root": s(OM.STEM_D, arm), "v2_root": s(ROOT_D, arm),
            "v2_cone_top": s(OM.STEM_D, arm - ROOT_H)}


def cmd_blocks_build(a):
    BLOCKS_OUT.mkdir(parents=True, exist_ok=True)
    root, foot = stem_root()
    rep = {"what": "the tag blocks of the tracked cylinder and sphere, second version: the stem's root widened and "
                   "filleted (owner, 2026-11-16)",
           "script": "scripts/hardware/tracking_objects_print.py blocks", "first_version": str(OUT.relative_to(ROOT)),
           "root": {"root_d_mm": ROOT_D, "root_h_mm": ROOT_H, "fillet_r_mm": FILLET_R, "foot_d_mm": round(2 * foot, 3),
                    "below_lowest_finger_mm": round(FINGER_GAP - ROOT_H, 1)},
           "bending_mpa_per_n": {s: bending(s) for s in ("cylinder", "sphere")}, "parts": {}}
    if BLOCKS_REPORT.exists():
        old = json.loads(BLOCKS_REPORT.read_text())
        rep.update({k: old[k] for k in ("bambu",) if k in old})
    bodies = {"cylinder": cylinder_body(())[0], "sphere": sphere_halves()[0]}
    ok, hexagon = True, 3 * OM.BLOCK.face_width * OM.BLOCK.across_flats / 2 * OM.BLOCK.height
    rep["assembly_overlap_mm3"] = {}
    for shape in ("cylinder", "sphere"):
        o, name = OM.OBJECTS[shape], f"{shape}_block"
        w, b, top = block_stem(shape, root=True)
        w1, b1, _ = block_stem(shape)
        info = rep["parts"][name] = {"what": PARTS[name] + ", root widened and filleted", "height_mm": round(top + KEY[2], 3)}
        info["stem_solid_g"] = round(((w + b).volume() - hexagon) * PLA, 2)
        info["added_g"] = round(((w + b).volume() - (w1 + b1).volume()) * PLA, 3)
        info["black_unchanged_mm3"] = round(abs(b.volume() - b1.volume()) + abs((b ^ b1).volume() - b1.volume()), 6)
        ok &= info["black_unchanged_mm3"] < 1e-6
        R = R_PRINT["cylinder" if shape == "cylinder" else "sphere_lower"]
        shift = np.array([0.0, 0.0, -o["length_mm"] / 2 if shape == "cylinder" else 0.0])
        blk = (w + b).transform(m4(OM.R_OB, OM.block_centre_O(o["r_mm"], o["stem_mm"])
                                   - OM.R_OB @ np.array([0.0, 0.0, OM.BLOCK.height / 2])))
        rep["assembly_overlap_mm3"][shape] = abs(round((blk ^ bodies[shape].transform(m4(R.T, shift))).volume(), 6))
        ok &= rep["assembly_overlap_mm3"][shape] < 1e-6
        ok &= export_part(info, w, b, lambda c, n=name: blocks_stl(n, c))
    rep["ok"] = bool(ok)
    BLOCKS_REPORT.write_text(json.dumps(rep, indent=1))
    write_blocks_readme(rep)
    print("wrote", BLOCKS_REPORT.relative_to(ROOT), "ok" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


def cmd_blocks_check(a):
    rep = json.loads(BLOCKS_REPORT.read_text())
    tiles, ok, maps = [], True, print_maps()
    for part in BLOCKS:
        rows, good = check_tags(blocks_stl(part, "black"), maps[part], part, tiles)
        rep["parts"][part]["check"] = rows
        ok &= good
    rep["check_ok"] = bool(ok)
    BLOCKS_REPORT.write_text(json.dumps(rep, indent=1))
    if a.preview:
        OP.sheet(tiles, a.preview)
    write_blocks_readme(rep)
    print("check", "ok" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


def reslice(project, work):
    """The saved project sliced again by the command line, as the window opens it (owner, 2026-11-13: a project that
    sliced right before its filaments were written on the objects opened wrong in the window): hours, filament loads
    and grams per filament (1-based)."""
    work.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["flatpak", "run", "--command=bambu-studio", "com.bambulab.BambuStudio", "--slice", "0",
                        "--outputdir", str(work), "--export-3mf", "resliced.3mf", str(project)],
                       capture_output=True, text=True)
    (work / "cli.log").write_text(r.stdout + r.stderr)
    with zipfile.ZipFile(work / "resliced.3mf") as z:
        gcode = z.read("Metadata/plate_1.gcode").decode(errors="replace")
        info = z.read("Metadata/slice_info.config").decode()
    loads = {}
    for v in re.findall(r"^\s*M620 S(\d+)A", gcode, re.M):
        loads[str(int(v) + 1)] = loads.get(str(int(v) + 1), 0) + 1
    pred = re.search(r'key="prediction" value="(\d+)"', info)
    return {"hours": round(int(pred.group(1)) / 3600, 2), "loads": dict(sorted(loads.items())),
            "used_g": {k: float(v) for k, v in re.findall(r'<filament id="(\d+)"[^>]*used_g="([^"]+)"', info)}}


def cmd_blocks_bambu(a):
    items = [{"name": n, "white": blocks_stl(n), "black": blocks_stl(n, "black"), "place": PLACE[n]} for n in BLOCKS]
    work = ROOT / f"logs/{BLOCKS_DATE}-tag_blocks"
    q = OP.bambu_project(items, BLOCKS_PROJECT, work / "bambu", "tag blocks", tower=TOWER, colours=COLOURS,
                         filament=FILAMENTS)
    q["reslice"] = reslice(BLOCKS_PROJECT, work / "reslice")
    if sorted(q["reslice"]["loads"]) != sorted(str(f) for f in FILAMENTS.values()):
        q["problems"].append(f"re-slice loads {q['reslice']['loads']}")
    rep = json.loads(BLOCKS_REPORT.read_text())
    rep["bambu"] = q
    BLOCKS_REPORT.write_text(json.dumps(rep, indent=1))
    write_blocks_readme(rep)
    print(f"{BLOCKS_PROJECT.relative_to(ROOT)}: {q['prediction_s'] / 3600:.1f} h, {q['layers']} layers, "
          f"{q['filament_loads']} filament loads, used {q['used_g']} g")
    print("problems:", q["problems"] or "none")
    if q["problems"]:
        raise SystemExit(1)


def write_blocks_readme(rep):
    P, rt, bd = rep["parts"], rep["root"], rep["bending_mpa_per_n"]
    first = json.loads(REPORT.read_text())["parts"] if REPORT.exists() else {}
    L = [f"# Tag blocks of the tracked cylinder and sphere, second version, {BLOCKS_DATE}",
         "",
         "The two block-and-stem parts of `assets/tracking/20261112-objects/` with the stem's root widened and filleted "
         "(owner, 2026-11-16: \"we probably shouldve filleted the mounting point between the rod and the base of the "
         "apriltag hex block\"; no stem had broken). Written by `scripts/hardware/tracking_objects_print.py blocks`. "
         "They replace the first version's blocks; the cylinder body, the sphere's halves and the dowels stay those of "
         "2026-11-12.",
         "",
         "| Part | Files | Print |",
         "|---|---|---|"]
    for name in BLOCKS:
        p = P[name]
        files = ", ".join(f"`{Path(p[c]['file']).name}`" for c in ("white", "black") if c in p)
        L.append(f"| {p['what']} | {files} | upright, block on the bed |")
    L += ["",
          f"Geometry. In the first version the {OM.STEM_D:g} mm stem met the block's top face at a square corner and "
          f"reached 0.5 mm into it. Here its root is {rt['root_d_mm']:g} mm across at the top face and tapers to the "
          f"stem over {rt['root_h_mm']:g} mm. A concave fillet of {rt['fillet_r_mm']:g} mm joins the taper to the face, "
          f"so the root's foot is {rt['foot_d_mm']:.1f} mm across. The block, its tags, the stem's length, the cone, "
          f"the seat and the key are unchanged, and so are the tag maps. The root ends {rt['root_h_mm']:g} mm above "
          f"the block's top, {rt['below_lowest_finger_mm']:g} mm below the lowest finger point of the plan's five "
          "layouts and the measured servo's five best per task, at the grasps' open and touch poses. The simulation's "
          "stem stays a plain cylinder.",
          "",
          "Bending. A side force on the object bends the stem about its root. The lever arm from the block's top to "
          f"the object's centre is {bd['cylinder']['arm_mm']:.1f} mm on the cylinder and {bd['sphere']['arm_mm']:.1f} "
          "mm on the sphere. Per newton of side force, the nominal stress at the first version's root was "
          f"{bd['cylinder']['v1_root']:.1f} MPa on the cylinder and {bd['sphere']['v1_root']:.1f} MPa on the sphere, "
          "and the square corner raised it by a factor of about 2. In this version the largest nominal stress lies "
          f"where the taper meets the stem: {bd['cylinder']['v2_cone_top']:.1f} and {bd['sphere']['v2_cone_top']:.1f} "
          f"MPa per newton, without a corner. At the root it is {bd['cylinder']['v2_root']:.1f} and "
          f"{bd['sphere']['v2_root']:.1f} MPa per newton.",
          "",
          f"Mass. The root adds {P['cylinder_block']['added_g']:.2f} g (cylinder) and "
          f"{P['sphere_block']['added_g']:.2f} g (sphere) of solid PLA. The stem with its root, cone and key is "
          f"{P['cylinder_block']['stem_solid_g']} g and {P['sphere_block']['stem_solid_g']} g"
          + (f", against {first['cylinder_block']['stem_solid_g']} g and {first['sphere_block']['stem_solid_g']} g "
             "in the first version" if first else "") + ". Weigh each assembled object; the simulation takes the "
          "measured mass.",
          "",
          "Checks. The black cells are those of the first version (shared volume equal to their volume). In the CAD "
          f"assembly the parts share {rep['assembly_overlap_mm3']['cylinder']:g} mm3 with the cylinder and "
          f"{rep['assembly_overlap_mm3']['sphere']:g} mm3 with the sphere."]
    if "check_ok" in rep:
        rms = [r["corner_rms_mm"] for n in BLOCKS for r in P[n].get("check", [])]
        span = f"{min(rms):.3f}" if f"{min(rms):.3f}" == f"{max(rms):.3f}" else f"{min(rms):.3f} to {max(rms):.3f}"
        L[-1] += (f" Every tag face of the black bodies, drawn at {OP.PPM:.0f} px/mm from the mesh, decodes as its own id "
                  f"with corners {span} mm RMS from the map.")
    rc_path = BLOCKS_OUT / f"{BLOCKS_DATE}-root_clearance.json"
    if rc_path.exists():
        rows = json.loads(rc_path.read_text())["rows"]
        sph = [x for x in rows if x["task"] in ("roll", "twist")]
        run = [x for x in sph if tuple(x["xy"]) in SPHERE_LAYOUTS]
        out = [x for x in sph if tuple(x["xy"]) not in SPHERE_LAYOUTS]
        cyl = [x for x in rows if x["task"] in ("pinpush", "press")]
        L += ["",
              "Clearance in the rollouts. The rollouts of 2026-11-05 with the seat and the arm's lift (configurations "
              f"built and flat, {len(rows)} rollouts; `object_tracking_rollouts.py rootclear`) were replayed with the "
              f"root drawn by two cylinders that enclose it: {rt['root_d_mm']:g} mm across up to {rt['root_h_mm']:g} mm "
              f"and {rt['foot_d_mm']:.1f} mm across up to {rt['fillet_r_mm']:g} mm. On the cylinder's tasks at all five "
              f"layouts no finger came within {min(x['root_min_mm'] for x in cyl):.1f} mm of them ({len(cyl)} "
              f"rollouts). On the sphere at the generalist (70, 110) and pin-push (75, 55) layouts, where the sphere runs, "
              f"none came within {min(x['root_min_mm'] for x in run):.1f} mm ({len(run)} rollouts); the trial series "
              "runs the pin-push layout at (75, 56.8). At the other three layouts the 4 mm "
              f"stem already overlapped a finger by up to {-min(x['stem_min_mm'] for x in out):.2f} mm, and the enclosing "
              f"cylinders by up to {-min(x['root_min_mm'] for x in out):.2f} mm. Those rollouts used the sphere's earlier "
              "25.2 mm stem, which put the root 0.5 mm nearer the fingers than the printed 25.7 mm."]
    if J_OUT.exists():
        L += ["", f"Third version (owner, 2026-11-16), the blocks held on their objects without glue by a cotter joint: "
                  f"`{J_OUT.relative_to(ROOT)}/`."]
    if "bambu" in rep:
        q, fil = rep["bambu"], rep["bambu"].get("filament", FILAMENTS)
        rs = q.get("reslice") or {"hours": q["prediction_s"] / 3600, "used_g": q["used_g"],
                                  "loads": {"all": q["filament_loads"]}}
        u, w, b = rs["used_g"], str(fil["white"]), str(fil["black"])
        L += ["", f"Bambu Studio project: `{Path(q['project']).name}`, both blocks on one plate, the black on filament "
                  f"{b} and the white on filament {w}, written on each object. Map them to the black and white spools "
                  "when sending. The saved project, sliced again by the command line: "
                  f"{rs['hours']:.1f} h, {u.get(w, 0):.1f} g white and {u.get(b, 0):.1f} g black, "
                  f"{sum(rs['loads'].values())} filament loads ({rs['loads'].get(b, 0)} black, "
                  f"{rs['loads'].get(w, 0)} white). Not in git; `tracking_objects_print.py blocks bambu` rebuilds it."]
    (BLOCKS_OUT / "README.md").write_text("\n".join(L) + "\n")


def joint_stl(name, colour="white"):
    return J_OUT / f"{J_DATE}-{name}_{colour}.stl"


J_PARTS = {"cylinder_block": "cylinder block and stem, ids 10-15, the stem ending in the joint's taper and post",
           "sphere_block": "sphere block and stem, ids 16-21, the stem ending in the joint's taper and post",
           "cylinder": "cylinder body without end tags, with the joint's socket and slot",
           "sphere_lower": "sphere, lower hemisphere with the joint's socket and slot",
           "cotter_thin": "cotter 0.25 mm thinner, one dimple",
           "cotter_mid": "cotter, two dimples",
           "cotter_thick": "cotter 0.25 mm thicker, three dimples",
           "coupon_stem": "fit coupon: the stem's end on a 6 mm stub and a 12 mm base",
           "coupon_cylinder": "fit coupon: the cylinder's middle 16 mm below its axis, with the socket and slot",
           "coupon_sphere": "fit coupon: the sphere's lower cap, with the socket and slot"}
J_PRINT = {"cylinder_block": "upright, block on the bed", "sphere_block": "upright, block on the bed",
           "cylinder": "standing on its -z end", "sphere_lower": "flat side down, the socket on top",
           "cotter_thin": "on its side, dimples up", "cotter_mid": "on its side, dimples up",
           "cotter_thick": "on its side, dimples up", "coupon_stem": "upright, base on the bed",
           "coupon_cylinder": "standing, as the cylinder", "coupon_sphere": "cut face down, as the hemisphere"}
DIMPLES = {"thin": 1, "mid": 2, "thick": 3}


def centred(m):
    """(m moved so that its bounding box is centred on x = y = 0 with its lowest point on z = 0, the move)."""
    lo, hi = np.array(m.bounding_box()[:3]), np.array(m.bounding_box()[3:])
    t = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return m.translate(t.tolist()), t


def inverse(R, t):
    """3 x 4 of the inverse of p -> R p + t."""
    return m4(np.asarray(R).T, -np.asarray(R).T @ np.asarray(t))


def joint_parts():
    """{name: (white, black or None, frame, T)} of the joint's parts in their print frames; T (3 x 4) takes the print
    frame to `frame`: O of the cylinder or the sphere, or the joint frame J, the same on both."""
    import manifold3d as M
    out = {}
    B, L = OM.BLOCK, OM.OBJECTS["cylinder"]["length_mm"]
    for shape in ("cylinder", "sphere"):
        w, b, _top = block_stem(shape, root=True, joint=True)
        o = OM.OBJECTS[shape]
        out[f"{shape}_block"] = (w, b, shape, m4(OM.R_OB, OM.block_centre_O(o["r_mm"], o["stem_mm"])
                                                  - OM.R_OB @ np.array([0.0, 0.0, B.height / 2])))
    Rc, Rs = R_PRINT["cylinder"], R_PRINT["sphere_lower"]
    cyl = cylinder_body((), joint=True)[0]
    low = sphere_halves(joint=True)[0]
    out["cylinder"] = (cyl, None, "cylinder", inverse(Rc, (0.0, 0.0, L / 2)))
    out["sphere_lower"] = (low, None, "sphere", inverse(Rs, (0.0, 0.0, 0.0)))
    R_tp = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])         # x = w, y = -u, z = v: lying on -v
    for name, size in COT_SIZES.items():
        m, t = centred(cotter(size, DIMPLES[name]).transform(m4(R_tp)))
        out[f"cotter_{name}"] = (m, None, "J", inverse(R_tp, t))
    u_cone = MALE_U[0] - (CONE_R - OM.STEM_D / 2)
    stub = M.Manifold.batch_boolean([
        male(), M.Manifold.cylinder(CONE_R - OM.STEM_D / 2, OM.STEM_D / 2, CONE_R, SEG).translate([0.0, 0.0, u_cone]),
        M.Manifold.cylinder(6.5, OM.STEM_D / 2, OM.STEM_D / 2, SEG).translate([0.0, 0.0, u_cone - 6.0]),
        M.Manifold.cylinder(2.0, 6.0, 6.0, SEG).translate([0.0, 0.0, u_cone - 8.0])], M.OpType.Add)
    m, t = centred(stub)                                                # J is right-handed in (w, v, u): u up
    out["coupon_stem"] = (m, None, "J", inverse(np.eye(3), t))
    m, t = centred(cyl ^ box((-13.0, -13.0, L / 2 - 8.0), (13.0, -0.5, L / 2 + 8.0)))
    out["coupon_cylinder"] = (m, None, "cylinder", inverse(Rc, np.array([0.0, 0.0, L / 2]) + t))
    m, t = centred(low ^ box((-12.0, -14.0, 9.0), (12.0, 14.0, 21.0)))
    out["coupon_sphere"] = (m, None, "sphere", inverse(Rs, t))
    return out


def body_O(shape):
    """The object's printed body with the joint in O, and its envelope (the body before any cut)."""
    import manifold3d as M
    r, _d, _y = seat(shape)
    if shape == "cylinder":
        L = OM.OBJECTS["cylinder"]["length_mm"]
        return (cylinder_body((), joint=True)[0].translate([0.0, 0.0, -L / 2]),
                M.Manifold.cylinder(L, r, r, SEG).translate([0.0, 0.0, -L / 2]))
    return sphere_halves(joint=True)[0].transform(m4(R_PRINT["sphere_lower"].T)), M.Manifold.sphere(r, SEG)


def joint_checks(shape):
    """Overlaps (mm3) of the assembled joint at the cotter's nominal place and at both ends of its travel for each size,
    the cotter beyond the object's envelope, the stem turned 180 deg, and the cotter's ends against the surface."""
    body, env = body_O(shape)
    JT = joint_T(shape)
    rows = []
    for name, size in COT_SIZES.items():
        for ceiling in (size - COT_SLOPE * TRAVEL, size, size + COT_SLOPE * TRAVEL):
            at = (size - ceiling) / COT_SLOPE
            m, c = male(ceiling).transform(JT), cotter(size, at=at).transform(JT)
            rows.append({"cotter": name, "ceiling_mm": round(ceiling, 3), "cotter_at_mm": round(at, 3),
                         "body_male": abs(round((body ^ m).volume(), 6)), "body_cotter": abs(round((body ^ c).volume(), 6)),
                         "male_cotter": abs(round((m ^ c).volume(), 6)), "cotter_out": abs(round((c - env).volume(), 6))})
    turned, stop = [], None
    for k in range(0, 21):
        du = -0.1 * k
        v = (body ^ male().transform(m4(np.diag([-1.0, -1.0, 1.0]), (0.0, 0.0, du))).transform(JT)).volume()
        turned.append([round(-du, 2), round(v, 4)])
        if v < 1e-6 and stop is None:
            stop = round(-du, 2)
    r, _d, y_seat = seat(shape)
    surf = math.sqrt(r * r - (y_seat + FLOOR_U) ** 2)       # the slot floor meets the surface at |w| = surf
    hs = half_diag(SQ_H)
    a, b = COT_ENDS
    return {"rows": rows, "ok": all(max(x[k] for k in ("body_male", "body_cotter", "male_cotter", "cotter_out")) < 1e-6
                                    for x in rows),
            "turned_overlap_mm3": turned, "turned_stops_out_mm": stop,
            # the turned post's slot at its entry edge (+w), its ceiling sloping the other way, against the cotter's tip
            "turned_slot_below_tip_mm": None if stop is None else round(
                cotter_t(-COT_ZONE) - (COT_T - COT_SLOPE * POST_HALF_W - stop), 3),
            "surface_at_floor_mm": round(surf, 3), "socket_half_diag_mm": round(hs, 3),
            "end_recess_mm": [round(surf - (b + TRAVEL), 3), round(surf - b, 3), round(surf - (b - TRAVEL), 3)],
            "tip_recess_mm": [round(surf + (a - TRAVEL), 3), round(surf + a, 3), round(surf + (a + TRAVEL), 3)],
            "support_min_mm": round(min(b - TRAVEL - hs, -(a + TRAVEL) - hs), 3)}


def cmd_joint_build(a):
    J_OUT.mkdir(parents=True, exist_ok=True)
    h_in, h_far = slot_heights()
    hp = POST_SIDE / math.sqrt(2.0)
    rep = {"what": "the cotter joint between the tag blocks and the tracked cylinder and sphere (owner, 2026-11-16)",
           "script": "scripts/hardware/tracking_objects_print.py joint",
           "joint": {"taper_deg": TAPER_DEG, "socket_side_mm": SQ0, "taper_depth_mm": SQ_H, "male_u_mm": MALE_U,
                     "cone_r_mm": CONE_R, "post_side_mm": POST_SIDE, "post_half_w_mm": POST_HALF_W,
                     "post_top_mm": POST_TOP, "socket_top_mm": SOCKET_TOP, "key_v_mm": KEY_V, "key_cut_mm": KEY_CUT,
                     "floor_u_mm": FLOOR_U, "slot_w_mm": SLOT_W, "slot_h_entry_mm": round(h_in, 3),
                     "slot_h_far_mm": round(h_far, 3), "cotter_w_mm": COT_W, "cotter_t_mm": COT_T,
                     "cotter_slope": COT_SLOPE, "cotter_zone_mm": COT_ZONE, "cotter_ends_mm": COT_ENDS,
                     "cotter_sizes_mm": COT_SIZES, "travel_mm": TRAVEL, "slot_clear_mm": SLOT_CLEAR,
                     "post_wall_mm": round(hp - SLOT_W / 2, 3),
                     "post_bridge_mm": round(POST_TOP - (FLOOR_U + COT_T + COT_SLOPE * POST_HALF_W), 3),
                     "cotter_tip_t_mm": round(cotter_t(-COT_ZONE, min(COT_SIZES.values())), 3),
                     "entry": {"cylinder": "-x_O", "sphere": "+z_O"}},
           "parts": {}, "checks": {}}
    if J_REPORT.exists():
        old = json.loads(J_REPORT.read_text())
        rep.update({k: old[k] for k in ("bambu", "bambu_coupons") if k in old})
    ok = True
    parts = joint_parts()
    hexagon = 3 * OM.BLOCK.face_width * OM.BLOCK.across_flats / 2 * OM.BLOCK.height
    for name, (w, b, frame, T) in parts.items():
        info = rep["parts"][name] = {"what": J_PARTS[name], "print": J_PRINT[name],
                                     "print_to": {"frame": frame, "T": np.round(T, 6).tolist()}}
        if name.endswith("_block"):
            shape = name.split("_")[0]
            w2, b2, top = block_stem(shape, root=True)
            info["height_mm"] = round(top + POST_TOP, 3)
            info["stem_solid_g"] = round(((w + b).volume() - hexagon) * PLA, 2)
            info["v2_stem_solid_g"] = round(((w2 + b2).volume() - hexagon) * PLA, 2)
            info["black_unchanged_mm3"] = round(abs(b.volume() - b2.volume()) + abs((b ^ b2).volume() - b2.volume()), 6)
            ok &= info["black_unchanged_mm3"] < 1e-6
        ok &= export_part(info, w, b, lambda c, n=name: joint_stl(n, c))
    for shape in ("cylinder", "sphere"):
        rep["checks"][shape] = joint_checks(shape)
        ok &= rep["checks"][shape]["ok"] and rep["checks"][shape]["turned_stops_out_mm"] is not None
        print(shape, {k: v for k, v in rep["checks"][shape].items() if k not in ("rows", "turned_overlap_mm3")}, flush=True)
    rep["ok"] = bool(ok)
    J_REPORT.write_text(json.dumps(rep, indent=1))
    write_joint_readme(rep)
    print("wrote", J_REPORT.relative_to(ROOT), "ok" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


def cmd_joint_check(a):
    rep = json.loads(J_REPORT.read_text())
    tiles, ok, maps = [], True, print_maps()
    for part in BLOCKS:
        rows, good = check_tags(joint_stl(part, "black"), maps[part], part, tiles)
        rep["parts"][part]["check"] = rows
        ok &= good
    rep["check_ok"] = bool(ok)
    J_REPORT.write_text(json.dumps(rep, indent=1))
    if a.preview:
        OP.sheet(tiles, a.preview)
    write_joint_readme(rep)
    print("check", "ok" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


# on the P1S bed (mm): the joint's parts, and the fit coupons on a plate of their own
J_PLACE = {"cylinder_block": (45.0, 60.0), "sphere_block": (45.0, 110.0), "cylinder": (45.0, 165.0),
           "sphere_lower": (115.0, 60.0)}
C_PLACE = {"coupon_stem": (60.0, 70.0), "coupon_cylinder": (110.0, 70.0), "coupon_sphere": (165.0, 70.0)}
J_COTTERS = [(110.0 + 22.0 * i, 115.0 + 12.0 * j) for j in range(2) for i in range(3)]


def cmd_joint_bambu(a):
    work = ROOT / f"logs/{J_DATE}-cotter_joint"
    rep = json.loads(J_REPORT.read_text())
    cot = [f"cotter_{n}" for n in COT_SIZES]
    plates = {
        "bambu_coupons": (J_OUT / f"{J_DATE}-joint_coupons_p1s.3mf", "joint fit coupons",
                          [(n, C_PLACE[n]) for n in C_PLACE] + [(n, J_COTTERS[i]) for i, n in enumerate(cot)]),
        "bambu": (J_OUT / f"{J_DATE}-cotter_joint_p1s.3mf", "cotter joint parts",
                  [(n, J_PLACE[n]) for n in J_PLACE] + [(n, J_COTTERS[i + 3 * j]) for j in range(2)
                                                        for i, n in enumerate(cot)])}
    bad = False
    for key, (project, title, placed) in plates.items():
        if a.only and key != a.only:
            continue
        items = []
        for k, (n, xy) in enumerate(placed):
            black = joint_stl(n, "black")
            items.append({"name": n if n not in [x for x, _ in placed[:k]] else f"{n}_2", "white": joint_stl(n),
                          "black": black if black.exists() else None, "place": xy})
        q = OP.bambu_project(items, project, work / key, title, tower=TOWER, colours=COLOURS, filament=FILAMENTS)
        white_only = all(it["black"] is None for it in items)
        if white_only:          # one filament: no loads of the black
            q["problems"] = [p for p in q["problems"] if not p.startswith("filaments loaded")]
        q["reslice"] = reslice(project, work / f"{key}_reslice")
        want = [str(FILAMENTS["white"])] if white_only else sorted(str(f) for f in FILAMENTS.values())
        if sorted(q["reslice"]["loads"]) != want and not (white_only and not q["reslice"]["loads"]):
            q["problems"].append(f"re-slice loads {q['reslice']['loads']}")
        rep[key] = q
        print(f"{project.relative_to(ROOT)}: {q['reslice']['hours']:.1f} h, {q['layers']} layers, loads "
              f"{q['reslice']['loads']}, used {q['reslice']['used_g']} g; problems: {q['problems'] or 'none'}")
        bad |= bool(q["problems"])
    J_REPORT.write_text(json.dumps(rep, indent=1))
    write_joint_readme(rep)
    if bad:
        raise SystemExit(1)


def far_touch_sentence(t):
    """The cylinder's far opening against the touches of the rollouts (opening_touches.json), in a sentence."""
    rows = [x for x in t["rows"] if x["shape"] == "cylinder"]
    on = [x for x in rows if x["far_min_mm"] is not None and x["far_min_mm"] <= 0.0]
    near = [x for x in rows if x["far_min_mm"] is not None and 0.0 < x["far_min_mm"] <= t["near_mm"]]

    def where(x):
        return f"{x['task'].replace('pinpush', 'pin-push')} at ({x['xy'][0]:.0f}, {x['xy'][1]:.0f}), seed {x['seed']}"
    def count(k, noun="rollout"):
        return f"{['no', 'one', 'two', 'three', 'four', 'five'][k] if k < 6 else k} {noun}{'s' if k != 1 else ''}"
    out = (f"A finger touched the tape over the far opening in {count(len(on))} "
           f"({'; '.join(where(x) for x in on)})." if on else "No finger touched the tape over the far opening.")
    if near:
        out += (f" In {count(len(near), 'more rollout')} a finger came within {max(x['far_min_mm'] for x in near):.1f} "
                f"mm of it ({'; '.join(where(x) for x in near)}).")
    return out


def write_joint_readme(rep):
    P, j, C = rep["parts"], rep["joint"], rep["checks"]
    cc, cs = C["cylinder"], C["sphere"]
    L = [f"# Cotter joint between the tag blocks and the tracked objects, {J_DATE}",
         "",
         "The tag blocks of `assets/tracking/20261116-tag_blocks/` attached to the cylinder and the sphere without glue "
         "(owner, 2026-11-16: \"can we think of ways to attach the hex block to the cylinder/sphere without glue? ... "
         "some kind of joinery perpendicular to the block shaft ... tight enough to prevent slop but fit enough to "
         "insert (and remove in the case of a break)\"). Written by `scripts/hardware/tracking_objects_print.py "
         "joint`. The blocks, the cylinder body and the sphere's lower half replace those of 2026-11-12 and 2026-11-16; "
         "the sphere's upper half and its dowels stay those of 2026-11-12.",
         "",
         "| Part | Files | Print |",
         "|---|---|---|"]
    for name in J_PARTS:
        p = P[name]
        files = ", ".join(f"`{Path(p[c]['file']).name}`" for c in ("white", "black") if c in p)
        L.append(f"| {p['what']} | {files} | {p['print']} |")
    L += ["",
          f"Joint. The stem ends in a square taper, its faces at {j['taper_deg']:g} deg to the stem, {j['socket_side_mm']:g} "
          f"mm across at the object's flat seat. It sits in a socket of the same taper in the object, which is "
          f"{j['taper_depth_mm']:g} mm deep and continues straight beyond. Above the taper a square post "
          f"{j['post_side_mm']:g} mm across has a slot across the stem. The cotter is a pin {j['cotter_w_mm']:g} mm "
          f"wide whose top face rises 1 mm in {1 / j['cotter_slope']:g}; it is {COT_ENDS[1] - COT_ENDS[0]:g} mm long and "
          f"{j['cotter_t_mm']:g} mm thick at the stem's axis. It is pushed through the object's slot and the post's slot, "
          "under the post's ceiling, which slopes as the cotter does. The cotter rests on the object's slot floor "
          f"{j['floor_u_mm']:g} mm above the seat, so pushing it in lifts the post and draws the taper into its socket. "
          "The four taper faces then fix the block's position, tilt and turn on the object, and friction keeps the "
          f"cotter in place. A wedge stays in place while its angle is below the sum of the friction angles on its two "
          f"faces. The cotter's angle is {math.degrees(math.atan(j['cotter_slope'])):.1f} deg, so it stays in place for "
          f"friction coefficients above {math.tan(math.atan(j['cotter_slope']) / 2):.2f}.",
          "",
          "Fit. Each cotter takes up the printing error of the joint over its travel: "
          f"{j['travel_mm']:g} mm along the slot each way from its nominal place is "
          f"{j['travel_mm'] * j['cotter_slope']:g} mm along the stem. The three cotters differ by 0.25 mm in thickness, so "
          f"together they cover {-0.25 - j['travel_mm'] * j['cotter_slope']:g} to "
          f"+{0.25 + j['travel_mm'] * j['cotter_slope']:g} mm. Use the cotter whose thick end stops "
          "inside the slot's mouth when the block is firm. The stem's -v corner and the socket's are cut flat, so the "
          "stem goes in one way only. Turned by 180 deg, it stops "
          f"{cc['turned_stops_out_mm']:g} mm short of its seat. Its slot then lies "
          f"{cc['turned_slot_below_tip_mm']:.1f} mm below the cotter's tip, and the cotter does not enter it.",
          "",
          "Openings. The cotter enters on the cylinder's -x side and on the sphere's +z side. The slot's far end is the "
          "opening through which a pin of 2 mm or less pushes the cotter out. The entry is "
          f"{j['slot_w_mm']:g} x {j['slot_h_entry_mm']:.1f} mm and the far opening {j['slot_w_mm']:g} x "
          f"{j['slot_h_far_mm']:.1f} mm in section. On the cylinder both lie at the middle of its length, under the grip "
          "tape, which can be cut away over them or lifted to take the block off."
          + (" They span {} deg (entry) and {} deg (far) from the bottom line; on the sphere {} and {} deg from the "
             "bottom pole.".format(*(f"{min(abs(a) for a, _b in o_['outline_deg']):.0f} to "
                                     f"{max(abs(a) for a, _b in o_['outline_deg']):.0f}"
                                     for o_ in json.loads((J_OUT / f"{J_DATE}-opening_touches.json").read_text())["openings"]))
             if (J_OUT / f"{J_DATE}-opening_touches.json").exists() else ""),
          "",
          "Assembly. Push the block's taper into the socket, the stem's cut corner toward the socket's cut corner. "
          "Slide the middle cotter (two dimples) in from the entry, chamfered tip first, its sloped face toward the "
          "object's centre and its dimples away from the cut corner. Press its thick end with a small screwdriver "
          "until the block no longer rocks. If the block is firm while the cotter's thick end still stands out of the "
          "entry, take the thin cotter (one dimple). If the cotter's tip reaches the far opening's mouth while the block "
          "is still loose, take the thick one (three dimples). To take the block off, push the cotter's tip back through "
          "the far opening with a pin and pull the block straight out.",
          "",
          "Checks. In the CAD assembly the block, the object and the cotter share 0 mm3, on both objects. The shared "
          "volume is 0 at the cotter's nominal place and at both ends of its travel, for each of the three cotters. Over its travel the "
          f"cotter's ends stay {min(cc['end_recess_mm']):.2f} to {max(cc['end_recess_mm']):.2f} mm inside the cylinder's "
          f"surface and {min(cs['end_recess_mm']):.2f} to {max(cs['end_recess_mm']):.2f} mm inside the sphere's. It rests at least "
          f"{cc['support_min_mm']:.2f} mm on the slot floor on each side of the socket. The post's walls beside its slot "
          f"are {j['post_wall_mm']:.2f} mm at the axis and the post's top {j['post_bridge_mm']:.2f} mm above the slot; "
          f"the thin cotter is {j['cotter_tip_t_mm']:.2f} mm thick at its tip. The black cells of the blocks are those of "
          "the second version (shared volume equal to their volume)."]
    if "check_ok" in rep:
        rms = [r["corner_rms_mm"] for n in BLOCKS for r in P[n].get("check", [])]
        span = f"{min(rms):.3f}" if f"{min(rms):.3f}" == f"{max(rms):.3f}" else f"{min(rms):.3f} to {max(rms):.3f}"
        L[-1] += (f" Every tag face of the black bodies, drawn at {OP.PPM:.0f} px/mm from the mesh, decodes as its own id "
                  f"with corners {span} mm RMS from the map.")
    L += ["",
          f"Mass. The stem with its root, cone, taper and post is {P['cylinder_block']['stem_solid_g']} g (cylinder) and "
          f"{P['sphere_block']['stem_solid_g']} g (sphere) of solid PLA, against {P['cylinder_block']['v2_stem_solid_g']} g "
          f"and {P['sphere_block']['v2_stem_solid_g']} g in the second version; a cotter is "
          f"{P['cotter_mid']['solid_pla_g']} g. Weigh each assembled object; the simulation takes the measured mass."]
    tm = J_OUT / f"{J_DATE}-opening_touches.json"
    if tm.exists():
        t = json.loads(tm.read_text())
        o = {(x["shape"], x["opening"]): x for x in t["openings"]}
        L += ["",
              f"Finger contacts near the openings. The rollouts of 2026-11-05 with the seat and the arm's lift "
              f"(configurations built and flat, {t['n_rollouts']['cylinder']} on the cylinder, "
              f"{t['n_rollouts']['sphere']} on the sphere at its two layouts; `object_tracking_rollouts.py touchmap`) "
              f"were replayed and every finger contact within {t['touch_mm']:g} mm of the object's surface recorded. "
              f"On the cylinder no contact came within {o[('cylinder', 'entry')]['min_mm']:.1f} mm of the entry. "
              + far_touch_sentence(t) + " On the sphere the nearest contacts were "
              f"{o[('sphere', 'entry')]['min_mm']:.1f} mm from the entry and {o[('sphere', 'far')]['min_mm']:.1f} mm "
              "from the far opening."]
    for key, what in (("bambu_coupons", "the fit coupons and one cotter of each size, in white"),
                      ("bambu", "the two blocks, the cylinder body, the sphere's lower half and two cotters of each size")):
        if key not in rep:
            continue
        q, fil = rep[key], rep[key].get("filament", FILAMENTS)
        rs, w, b = q["reslice"], str(fil["white"]), str(fil["black"])
        u = rs["used_g"]
        col = (f"{u.get(w, 0):.1f} g white on filament {w}" if not u.get(b) else
               f"{u.get(w, 0):.1f} g white on filament {w} and {u.get(b, 0):.1f} g black on filament {b}, "
               f"{sum(rs['loads'].values())} filament loads")
        L += ["", f"Bambu Studio project `{Path(q['project']).name}`: {what}. The saved project, sliced again by the "
                  f"command line: {rs['hours']:.1f} h, {col}. Map the filaments to the spools when sending. Not in git; "
                  "`tracking_objects_print.py joint bambu` rebuilds it."]
    (J_OUT / "README.md").write_text("\n".join(L) + "\n")


# The larger objects (owner, 2026-11-17: "prepare these but with a larger diameter sphere, cylinder, since our current
# ones aren't broken yet"; "a sphere and cylinder print job ready to go, using just a single pla filament (i will glue on
# a paper apriltag onto the cylinder ends if it turns out being crucial)"): object_tracking_mjcf.OBJECT_SETS["large"],
# the 45 mm cylinder without end tags and the 67 mm sphere, with the same cotter joint. The cotters are longer, so that
# their ends lie as far inside the large cylinder's surface as on the 25 mm one. The bodies, the upper hemisphere, the
# dowels and the cotters print in white on one plate; the blocks and stems wait for the modular block (same message).
L_DATE = "20261117"
L_OUT = ROOT / f"assets/tracking/{L_DATE}-large_objects"
L_COT_ENDS = (-9.9, 9.9)            # mm: 2.05 mm inside the 45 mm cylinder's surface at the slot floor, as on the 25 mm one
L_PARTS = {"cylinder": "cylinder body 45 mm (43.4 mm printed for the tape), no end tags, with the joint's socket and slot",
           "sphere_lower": "sphere 67 mm, lower hemisphere with the joint's socket and slot",
           "sphere_upper": "sphere 67 mm, upper hemisphere",
           "sphere_pin": "dowel pin of the sphere (print three)",
           "cotter_thin": "cotter 0.25 mm thinner, one dimple",
           "cotter_mid": "cotter, two dimples",
           "cotter_thick": "cotter 0.25 mm thicker, three dimples"}
L_PRINT = {"cylinder": "standing on its -z end", "sphere_lower": "flat side down, the socket on top",
           "sphere_upper": "flat side down", "sphere_pin": "standing; print three",
           "cotter_thin": "on its side, dimples up", "cotter_mid": "on its side, dimples up",
           "cotter_thick": "on its side, dimples up"}
L_PLACE = {"cylinder": (55.0, 60.0), "sphere_lower": (140.0, 60.0), "sphere_upper": (140.0, 150.0),
           "sphere_pin": [(40.0, 120.0), (55.0, 120.0), (70.0, 120.0)]}
L_COTTERS = [(40.0 + 28.0 * i, 160.0 + 12.0 * j) for j in range(2) for i in range(3)]
L_FILAMENT, L_COLOURS = {"white": 1, "black": 2}, ["#FFFFFF", "#000000"]   # one spool, filament 1
# the 67 mm sphere's joint turned 180 deg about the stem: its cotter enters at -z_O, where no finger touched the replayed
# rollouts within 10 mm of the opening; at +z_O, the 40 mm sphere's entry, fingers touched it (20261117-opening_touches)
L_J_B = {"sphere": np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])}
L_INFILL = (("gyroid", 0), ("gyroid", 5), ("gyroid", 10), ("lightning", 10), ("lightning", 20))   # sliced per body


J_B0, J0 = {k: v.copy() for k, v in J_B.items()}, (J_DATE, J_OUT, J_REPORT, COT_ENDS)
SET = "20261112"


def use_set(name):
    """Select an object set: its objects and block ids (object_tracking_mjcf.select) and, for the large set, the joint's
    folder, report, cotter length and the sphere's turned joint."""
    global J_DATE, J_OUT, J_REPORT, COT_ENDS, SET
    OM.select(name)
    SET = name
    J_B.update({k: v.copy() for k, v in J_B0.items()})
    J_DATE, J_OUT, J_REPORT, COT_ENDS = J0
    if name == "large":
        J_DATE, J_OUT, COT_ENDS = L_DATE, L_OUT, L_COT_ENDS
        J_REPORT = J_OUT / f"{J_DATE}-large_objects.json"
        J_B.update(L_J_B)


def large_parts():
    """{name: (white, None, frame, T)} of the large set's parts in their print frames, as joint_parts."""
    out = {}
    L = OM.OBJECTS["cylinder"]["length_mm"]
    Rc, Rs, Ru = R_PRINT["cylinder"], R_PRINT["sphere_lower"], R_PRINT["sphere_upper"]
    low, up, pin = sphere_halves(joint=True)
    out["cylinder"] = (cylinder_body((), joint=True)[0], None, "cylinder", inverse(Rc, (0.0, 0.0, L / 2)))
    out["sphere_lower"] = (low, None, "sphere", inverse(Rs, (0.0, 0.0, 0.0)))
    out["sphere_upper"] = (up, None, "sphere", inverse(Ru, (0.0, 0.0, 0.0)))
    out["sphere_pin"] = (pin, None, "pin", m4(np.eye(3)))
    R_tp = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])         # x = w, y = -u, z = v: lying on -v
    for name, size in COT_SIZES.items():
        m, t = centred(cotter(size, DIMPLES[name]).transform(m4(R_tp)))
        out[f"cotter_{name}"] = (m, None, "J", inverse(R_tp, t))
    return out


def cmd_large_build(a):
    J_OUT.mkdir(parents=True, exist_ok=True)
    stem = json.loads((ROOT / "docs/experiments/20261105-object_tracking_sim/rows/stem_large.json").read_text())
    rep = {"what": "the larger tracked cylinder and sphere with the cotter joint (owner, 2026-11-17)",
           "script": "scripts/hardware/tracking_objects_print.py --set large large", "objects": OM.OBJECTS,
           "entry": {s: "".join(f"{'+-'[int(c < 0)]}{'xyz'[i]}_O" for i, c in enumerate(OM.R_OB @ J_B[s][:, 0]) if c)
                     for s in ("cylinder", "sphere")},
           "ids": OM.IDS, "stem_rows": "docs/experiments/20261105-object_tracking_sim/rows/stem_large.json",
           "stem_sets": stem["sets"], "stem_mm": stem["stem_mm"], "cotter_ends_mm": COT_ENDS,
           "seat": {s: dict(zip(("printed_r_mm", "flat_depth_mm", "seat_y_mm"), map(lambda x: round(x, 3), seat(s))))
                    for s in ("cylinder", "sphere")},
           "parts": {}, "checks": {}}
    if J_REPORT.exists():
        old = json.loads(J_REPORT.read_text())
        rep.update({k: old[k] for k in ("bambu", "mass_slices", "check_ok") if k in old})
    ok = True
    for name, (w, b, frame, T) in large_parts().items():
        info = rep["parts"][name] = {"what": L_PARTS[name], "print": L_PRINT[name],
                                     "print_to": {"frame": frame, "T": np.round(T, 6).tolist()}}
        ok &= export_part(info, w, b, lambda c, n=name: joint_stl(n, c))
    for shape in ("cylinder", "sphere"):
        rep["checks"][shape] = joint_checks(shape)
        ok &= rep["checks"][shape]["ok"] and rep["checks"][shape]["turned_stops_out_mm"] is not None
        print(shape, {k: v for k, v in rep["checks"][shape].items() if k not in ("rows", "turned_overlap_mm3")}, flush=True)
    # the joint's stem (taper and post) is that of 2026-11-16: the same male() in the same frame J
    rep["ok"] = bool(ok)
    J_REPORT.write_text(json.dumps(rep, indent=1))
    write_large_readme(rep)
    print("wrote", J_REPORT.relative_to(ROOT), "ok" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


def large_items(names):
    items = []
    for n in names:
        if n == "sphere_pin":
            items += [{"name": f"sphere_pin_{k + 1}", "white": joint_stl(n), "black": None, "place": xy}
                      for k, xy in enumerate(L_PLACE[n])]
        elif n.startswith("cotter_"):
            k = list(COT_SIZES).index(n[len("cotter_"):])
            items += [{"name": f"{n}_{j + 1}", "white": joint_stl(n), "black": None, "place": L_COTTERS[k + 3 * j]}
                      for j in range(2)]
        else:
            items.append({"name": n, "white": joint_stl(n), "black": None, "place": L_PLACE[n]})
    return items


def cmd_large_bambu(a):
    """The plate in white on filament 1, and each body sliced alone at the infills L_INFILL (pattern, %) for its mass."""
    work = ROOT / f"logs/{J_DATE}-large_objects"
    rep = json.loads(J_REPORT.read_text())
    if not a.only or a.only == "mass":
        sl = {}
        for body, names in (("cylinder", ["cylinder"]), ("sphere", ["sphere_lower", "sphere_upper", "sphere_pin"])):
            for pat, dens in L_INFILL:
                key = f"{pat}_{dens}"
                q = OP.bambu_project(large_items(names), work / f"mass_{body}_{key}.3mf", work / f"mass_{body}_{key}",
                                     f"{body} at {dens} % {pat}", tower=TOWER, colours=L_COLOURS, filament=L_FILAMENT,
                                     extra={"sparse_infill_pattern": pat, "sparse_infill_density": f"{dens}%"})
                sl.setdefault(body, {})[key] = {"used_g": q["used_g"].get(str(L_FILAMENT["white"])),
                                                "hours": round(q["prediction_s"] / 3600, 2)}
                print(body, key, sl[body][key], flush=True)
        rep["mass_slices"] = sl
    if not a.only or a.only == "bambu":
        pat, dens = a.infill.split(":")
        project = J_OUT / f"{J_DATE}-large_objects_p1s.3mf"
        q = OP.bambu_project(large_items(list(L_PARTS)), project, work / "bambu", "large objects", tower=TOWER,
                             colours=L_COLOURS, filament=L_FILAMENT,
                             extra={"sparse_infill_pattern": pat, "sparse_infill_density": f"{dens}%"})
        q["problems"] = [x for x in q["problems"] if not x.startswith("filaments loaded")]
        q["reslice"] = reslice(project, work / "bambu_reslice")
        if [k for k in q["reslice"]["loads"] if k != str(L_FILAMENT["white"])]:
            q["problems"].append(f"re-slice loads {q['reslice']['loads']}")
        q["infill"] = {"pattern": pat, "density_pct": float(dens)}
        rep["bambu"] = q
        print(f"{project.relative_to(ROOT)}: {q['reslice']['hours']:.1f} h, {q['layers']} layers, loads "
              f"{q['reslice']['loads']}, used {q['reslice']['used_g']} g; problems: {q['problems'] or 'none'}")
    J_REPORT.write_text(json.dumps(rep, indent=1))
    write_large_readme(rep)
    if rep.get("bambu", {}).get("problems"):
        raise SystemExit(1)


L_ROLLOUTS = ROOT / "docs/experiments/20261105-object_tracking_sim/rows/rollouts_large.jsonl"
TASK_NAME = {"twist": "twist", "roll": "roll", "pinpush": "pin-push", "press": "press"}


def large_rollouts():
    """Per task and configuration over the large set's rollouts (object_tracking_rollouts.py --set large run): rollouts,
    layouts, mean score and held share in the study's scene (base), on the flat seat (flat) and in the seat's pocket
    (built), and on the seats the smallest finger distance to the stem and block and the block's lowest height over the
    seat after the lift; None before the run."""
    if not L_ROLLOUTS.exists():
        return None
    rows = {}
    for x in L_ROLLOUTS.read_text().splitlines():         # the last row of a rollout run twice
        r = json.loads(x) if x.strip() else {}
        if r.get("status") == "ok":
            rows[(r["task"], r["call"], r["config"], r["seed"])] = r
    rows = list(rows.values())
    out = {}
    for task in ("twist", "roll", "pinpush", "press"):
        t = {}
        for cfg in ("base", "flat", "built"):
            rr = [r for r in rows if r["task"] == task and r["config"] == cfg]
            if rr:
                t[cfg] = {"n": len(rr), "layouts": len({r["call"] for r in rr}),
                          "score": round(float(np.mean([r["score"] for r in rr])), 3),
                          "held": round(float(np.mean([bool(r["held"]) for r in rr])), 3)}
        for cfg in ("flat", "built"):
            bb = [r for r in rows if r["task"] == task and r["config"] == cfg]
            if bb:
                t[cfg].update(clear_min_mm=round(min(r["clear_min_mm"] for r in bb), 2),
                              seat_clear_min_mm=round(min(r["seat_clear_min_mm"] for r in bb), 2))
        out[task] = t
    return out


def write_large_readme(rep):
    P, C, o, sm = rep["parts"], rep["checks"], rep["objects"], rep["stem_mm"]
    cc, cs = C["cylinder"], C["sphere"]
    L = [f"# Larger tracked cylinder and sphere with the cotter joint, {J_DATE}",
         "",
         "The tracked objects at the size-ranking test's large sizes (`scripts/hom/hom_size_rank.py`), a 45 mm x 100 mm "
         "cylinder and a 67 mm sphere, each held on its tag block by the cotter joint of "
         "`assets/tracking/20261116-cotter_joint/` (owner, 2026-11-17: \"prepare these but with a larger diameter "
         "sphere, cylinder, since our current ones aren't broken yet\"). The owner asked for one print job in a single "
         "PLA filament: \"i will glue on a paper apriltag onto the cylinder ends if it turns out being crucial\". "
         "Written by `scripts/hardware/tracking_objects_print.py --set large large build|bambu`. Their tag blocks and "
         "stems are the modular ones of `assets/tracking/20261117-modular_blocks/`, codes C45 and S67.",
         "",
         "| Part | File | Print |",
         "|---|---|---|"]
    for name in L_PARTS:
        L.append(f"| {P[name]['what']} | `{Path(P[name]['white']['file']).name}` | {P[name]['print']} |")
    L += ["",
          f"Geometry. The cylinder prints at {2 * rep['seat']['cylinder']['printed_r_mm']:.1f} mm, "
          f"{o['cylinder']['length_mm']:g} mm long, so {o['cylinder']['tape_mm']:g} mm grip tape brings it to "
          f"{2 * o['cylinder']['r_mm']:g} mm. Its end faces are bare and flat, "
          f"{2 * rep['seat']['cylinder']['printed_r_mm']:.1f} mm across, for paper tags. The sphere is "
          f"{2 * o['sphere']['r_mm']:g} mm and bare, printed as two hemispheres joined by three dowels "
          f"{PIN_R:g} mm from its axis. The flat seat is {SEAT_D:g} mm across and lies "
          f"{rep['seat']['cylinder']['flat_depth_mm']:.2f} mm (cylinder) and {rep['seat']['sphere']['flat_depth_mm']:.2f} "
          "mm (sphere) below the printed surface.",
          "",
          "Joint. The socket, its slot and the stem's taper and post are those of 2026-11-16. The cotter enters on "
          "the cylinder's -x side, as on the 25 mm cylinder. On the sphere the joint is turned 180 deg about the stem, "
          "so its cotter enters on the -z side: in the replayed rollouts fingers touched the 67 mm sphere at the +z "
          "opening (see below). The slot reaches the surface further from the stem on the larger objects, so the cotters are "
          f"{COT_ENDS[1] - COT_ENDS[0]:g} mm long in place of 13.7 mm. At its nominal place each end of a cotter lies "
          f"{cc['end_recess_mm'][1]:.2f} mm inside the cylinder's surface and {cs['end_recess_mm'][1]:.2f} mm inside the "
          "sphere's. Their thickness along the taper is unchanged, so the fit coupons of 2026-11-16 test this joint "
          "too, and the cotter that fits a coupon fits these objects.",
          "",
          f"Stems and blocks. The block's top is to lie {sm['cylinder']:g} mm (cylinder) and {sm['sphere']:g} mm "
          "(sphere) below the object's taped or bare surface, against 35.6 and 25.7 mm on the smaller objects. These "
          "lengths put the block 15 mm below the lowest finger point at the open and touch poses of the ten best layouts "
          "per task in the HOM grids at these sizes (`grid2dff_v4b_sv25r`: the SCS0009 plant, 2.5 MPa pads; "
          "`object_tracking_rollouts.py --set large stem`). The blocks bear ids "
          f"{rep['ids']['cylinder'][0]}-{rep['ids']['cylinder'][-1]} (cylinder) and "
          f"{rep['ids']['sphere'][0]}-{rep['ids']['sphere'][-1]} (sphere), so the tracker tells them from the smaller "
          "objects' blocks (`object_tracking_mjcf.OBJECT_SETS`).",
          "",
          "Checks. In the CAD assembly the stem's end, the object and the cotter share 0 mm3 on both objects. The shared "
          "volume is 0 at the cotter's nominal place and at both ends of its travel, for each of the three cotters. Over "
          f"its travel each end stays {min(cc['end_recess_mm']):.2f} to {max(cc['end_recess_mm']):.2f} mm inside the "
          f"cylinder's surface and {min(cs['end_recess_mm']):.2f} to {max(cs['end_recess_mm']):.2f} mm inside the "
          f"sphere's. A stem turned by 180 deg stops {cc['turned_stops_out_mm']:g} mm short of its seat."]
    ro = large_rollouts()
    if ro:
        n = sum(t[c]["n"] for t in ro.values() for c in t)
        lay = sum(t["base"]["layouts"] for t in ro.values() if "base" in t)
        L += ["",
              f"Rollouts. Each of the {lay} layouts above ran its task with its best grasp at seeds 20-24 in three "
              "scenes: the study's (a 3 mm post under the object until 1.6 s), and the object on its stem and block "
              "with the block on a flat seat or in the seat's 4 mm pocket, the arm lifting the hand 40 mm at 1.6 s "
              f"(`object_tracking_rollouts.py --set large run`, {n} rollouts on the SCS0009 plant, CPU MuJoCo). Scores "
              "are means; the finger distance to the stem or block is capped at 10 mm, the range of the contact query.",
              "",
              "| Task | Object | Layouts | Score, study's scene | Score, flat seat | Score, pocket seat | Held, pocket seat | Nearest finger, flat seat (mm) | Nearest finger, pocket seat (mm) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for task, t in ro.items():
            if not {"base", "flat", "built"} <= set(t):
                continue
            obj = (f"{2 * o['sphere']['r_mm']:g} mm sphere" if task in ("twist", "roll")
                   else f"{2 * o['cylinder']['r_mm']:g} mm cylinder")
            L.append(f"| {TASK_NAME[task]} | {obj} | {t['base']['layouts']} | {t['base']['score']:.2f} | "
                     f"{t['flat']['score']:.2f} | {t['built']['score']:.2f} | {100 * t['built']['held']:.0f} % | "
                     f"{t['flat']['clear_min_mm']:.1f} | {t['built']['clear_min_mm']:.1f} |")
        L += ["",
              "On the flat seat the scores stay within 0.02 of the study's scene, except the pin-push (0.08 lower). In "
              "the pocket the press loses 0.27, as on the 25 mm cylinder, where the series uses the flat seat. The "
              "twist's nearest finger came at its fourth layout, (150, 70): late in the twist the middle finger came "
              "within 1.1 mm of the stem or block on the flat seat and 0.3 mm in the pocket, without touching. At the "
              "other layouts no finger came within 9 mm. The stem's length can change with the modular stem without "
              "reprinting the sphere."]
    tm = J_OUT / f"{J_DATE}-opening_touches.json"
    if tm.exists():
        t = json.loads(tm.read_text())
        op = {(x["shape"], x["opening"]): x for x in t["openings"]}
        L += ["",
              f"Finger contacts near the openings. The rollouts on the two seats were replayed "
              f"({t['n_rollouts']['cylinder']} on the cylinder, {t['n_rollouts']['sphere']} on the sphere; "
              f"`object_tracking_rollouts.py --set large touchmap`) and every finger contact within {t['touch_mm']:g} mm "
              "of the object's surface recorded. The "
              f"nearest contacts were {op[('cylinder', 'entry')]['min_mm']:.1f} mm (entry) and "
              f"{op[('cylinder', 'far')]['min_mm']:.1f} mm (far opening) on the cylinder, and "
              f"{op[('sphere', 'entry')]['min_mm']:.1f} and {op[('sphere', 'far')]['min_mm']:.1f} mm on the sphere. "
              f"Rollouts with a contact within {t['near_mm']:g} mm of an opening: "
              f"{op[('cylinder', 'entry')]['near']} and {op[('cylinder', 'far')]['near']} of "
              f"{t['n_rollouts']['cylinder']} on the cylinder, {op[('sphere', 'entry')]['near']} and "
              f"{op[('sphere', 'far')]['near']} of {t['n_rollouts']['sphere']} on the sphere."]
    if "mass_slices" in rep:
        ms = rep["mass_slices"]
        keys = list(ms["cylinder"])
        L += ["",
              "Mass. The simulations take each object with its block, stem and tape as 25 g. Each body was sliced alone "
              "in the plate's process (0.20 mm Standard, two walls) at five infills, the sphere with its dowels:",
              "",
              "| Infill | Cylinder body (g) | Sphere, halves and dowels (g) |",
              "|---|---|---|"]
        for k in keys:
            pat, d = k.split("_")
            L.append(f"| {d} % {pat} | {ms['cylinder'][k]['used_g']:.1f} | {ms['sphere'][k]['used_g']:.1f} |")
        L += ["",
              "Lightning infill supports only the top faces, so it adds 1.5 g (cylinder) and 1.9 g (sphere) to the empty "
              "bodies; gyroid adds 1.8 to 1.9 g per percent. With the block (3 g in the plan), a stem of about 1 g and a "
              f"cotter of {P['cotter_mid']['solid_pla_g']} g, the cylinder at 20 % lightning comes to about "
              f"{ms['cylinder']['lightning_20']['used_g'] + 4.2:.0f} g before its tape and the sphere to about "
              f"{ms['sphere']['lightning_20']['used_g'] + 4.2:.0f} g. The sphere is above 25 g even with no infill. "
              "Weigh each assembled object; the simulation takes the measured mass."]
    if "bambu" in rep:
        q = rep["bambu"]
        rs = q["reslice"]
        L += ["",
              f"Bambu Studio project `{Path(q['project']).name}`: the cylinder body, both hemispheres, three dowels and "
              f"two cotters of each size, in white on filament 1, {q['infill']['density_pct']:g} % "
              f"{q['infill']['pattern']} infill. The saved project, sliced again by the command line: "
              f"{rs['hours']:.1f} h, {rs['used_g'][str(L_FILAMENT['white'])]:.1f} g, one filament and no prime tower. "
              "Map filament 1 to the white spool when sending. Not in git; `tracking_objects_print.py --set large large "
              "bambu` rebuilds it."]
    (J_OUT / "README.md").write_text("\n".join(L) + "\n")


# The modular block and stem (owner, 2026-11-17: "design a modular attachment for the stem base with the apriltag hex
# block. it's quite costly printing these ... im trying to anticipate breaks, which would make these blocks useless. i am
# not sure how to design joinery here since we cant interfere with the side faces with the blocks on them ... if we are
# going to have unique ids per block, and specific stem heights per object, then some kind of identification for these
# modular components would also be helpful"). The block keeps its six tag faces and takes a square socket in its top
# face. The stem is a part of its own: its foot is a square taper that one M3 self-forming screw, up through the block
# from a counterbore in its bottom face, draws into the socket; its other end is the cotter joint's taper and post. A
# broken stem is reprinted in white and the block kept. In B the socket's faces are normal to x_B and y_B and its
# (-x_B, -y_B) corner is filled as a key, so the stem goes in one way only; the depth u is measured down from the top face.
# Each object has a code: text on the block's bottom face, and as many dimples on the stem's root and the block's bottom
# as the code's number.
M_DATE = "20261117"
M_OUT = ROOT / f"assets/tracking/{M_DATE}-modular_blocks"
M_REPORT = M_OUT / f"{M_DATE}-modular_blocks.json"
FOOT_S0, FOOT_H, FOOT_DEPTH = 8.0, 3.0, 4.0     # mm: the socket's side at the top face, its taper's depth, its floor's u
FOOT_U = (-0.5, 2.7)                # mm: the foot's taper from 0.5 mm above the top face to 2.7 mm into the socket
FOOT_KEY, FOOT_KEY_CUT = 3.6, 0.15  # mm: the socket's corner filled beyond FOOT_KEY on the diagonal; the foot cut inside
M_ROOT_R, M_ROOT_H = 4.0, 10.0      # mm: the root cone from radius 4 on the foot down to the stem's 2 mm over 10 mm
SCREW = {"name": "M3 x 16", "length_mm": 16.0, "clear_d_mm": 3.4, "head_d_mm": 6.0, "head_depth_mm": 3.4,
         "pilot_d_mm": 2.75, "pilot_depth_mm": 8.0}     # the owner's self-forming M3 pilot is 2.75 mm
LABEL_H, LABEL_DEPTH, LABEL_Y = 4.0, 0.4, 7.0   # mm: the label's text height, depth and lines at y_B = +-LABEL_Y
DOT_R, DOT_PITCH = 0.45, 1.8        # mm: the identification dimples' radius and spacing
CODES = {("20261112", "cylinder"): ("C25", 1), ("20261112", "sphere"): ("S40", 2),
         ("large", "cylinder"): ("C45", 3), ("large", "sphere"): ("S67", 4)}
M_PARTS = {"block": "tag block, socket in its top face, counterbore in its bottom face",
           "stem": "stem, square-taper foot to the cotter joint's taper and post"}
M_PRINT = {"block": "upright, bottom face on the bed", "stem": "upright, foot on the bed, 5 mm brim",
           "test_block": "upright, bottom face on the bed", "test_stem": "upright, foot on the bed, 5 mm brim"}


def side_at(u):
    """The socket's square side (mm) at depth u below the top face."""
    return FOOT_S0 - 2.0 * u * math.tan(math.radians(TAPER_DEG))


def square_frustum(u0, u1, s0=None, s1=None):
    """A square frustum on z_B in B, faces normal to x_B and y_B, from depth u0 (higher) to u1 (lower); sides side_at
    unless given."""
    import manifold3d as M
    zt = OM.BLOCK.height / 2
    s0 = side_at(u0) if s0 is None else s0
    s1 = side_at(u1) if s1 is None else s1
    sq = M.CrossSection.square([s1, s1], True)
    return M.Manifold.extrude(sq, u1 - u0, 0, 0.0, (s0 / s1, s0 / s1)).translate([0.0, 0.0, zt - u1])


def key_region(d):
    """The half-space beyond d from the axis along the (-x_B, -y_B) diagonal, as a large box (B)."""
    a = 40.0
    R = np.array([[math.cos(math.radians(225)), -math.sin(math.radians(225)), 0.0],
                  [math.sin(math.radians(225)), math.cos(math.radians(225)), 0.0], [0.0, 0.0, 1.0]])
    return box((d, -a, -a), (d + a, a, a)).transform(m4(R))


def label_cs(text, height):
    """A CrossSection of text (DejaVu Sans Bold) about the origin, its cap height about `height` mm."""
    import manifold3d as M
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    tp = TextPath((0.0, 0.0), text, size=height * 1.37, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    cs = M.CrossSection([p.tolist() for p in tp.to_polygons()], M.FillRule.EvenOdd)
    lo, hi = np.array(cs.bounds()[:2]), np.array(cs.bounds()[2:])
    return cs.translate((-(lo + hi) / 2).tolist())


def dots(n, centre, normal_z=True):
    """n spheres of DOT_R in a row along x about `centre` (B), for dimples."""
    import manifold3d as M
    xs = (np.arange(n) - (n - 1) / 2) * DOT_PITCH
    return M.Manifold.batch_boolean([M.Manifold.sphere(DOT_R, 24).translate([centre[0] + x, centre[1], centre[2]])
                                     for x in xs], M.OpType.Add)


def socket_B():
    """What the socket, the screw's hole and its counterbore remove from the block (B)."""
    import manifold3d as M
    zt, zb = OM.BLOCK.height / 2, -OM.BLOCK.height / 2
    cav = square_frustum(-0.01, FOOT_H) + square_frustum(FOOT_H - 0.01, FOOT_DEPTH, side_at(FOOT_H), side_at(FOOT_H))
    cav = cav - key_region(FOOT_KEY)
    hole = M.Manifold.cylinder(zt - FOOT_DEPTH - zb + 1.01, SCREW["clear_d_mm"] / 2, SCREW["clear_d_mm"] / 2,
                               64).translate([0.0, 0.0, zb - 1.0])
    head = M.Manifold.cylinder(SCREW["head_depth_mm"] + 1.0, SCREW["head_d_mm"] / 2, SCREW["head_d_mm"] / 2,
                               64).translate([0.0, 0.0, zb - 1.0])
    return cav + hole + head


def block_label(shape):
    """The label cut into the bottom face (B): the object's code and the block's ids, mirrored so that they read from
    below, and the code's number of dimples."""
    import manifold3d as M
    code, n = CODES[(SET, shape)]
    ids = OM.IDS[shape]
    zb = -OM.BLOCK.height / 2
    out = []
    for text, y in ((code, LABEL_Y), (f"{ids[0]}-{ids[-1]}", -LABEL_Y)):
        cs = label_cs(text, LABEL_H).mirror([1.0, 0.0]).translate([0.0, y])
        out.append(M.Manifold.extrude(cs, LABEL_DEPTH + 1.0).translate([0.0, 0.0, zb - 1.0]))
    out.append(dots(n, (0.0, -LABEL_Y - 4.6, zb)))
    return M.Manifold.batch_boolean(out, M.OpType.Add)


def modular_block(shape):
    """(white, black) of the block in its print frame (B moved up by half the block): the prism less its tag cells, the
    socket, the screw's hole and counterbore, and the label."""
    import manifold3d as M
    B = OM.BLOCK
    prism = M.Manifold.extrude(M.CrossSection([B.vertices()[:6, :2].tolist()]), B.height).translate(
        [0.0, 0.0, -B.height / 2])
    cells = OP.black_cells(OM.block_map(shape), OP.BLOCK_DEPTH)
    body = prism - socket_B() - block_label(shape)
    lift = m4(np.eye(3), (0.0, 0.0, B.height / 2))
    return (body - cells).transform(lift), (body ^ cells).transform(lift)


def foot_B(cut=True):
    """The stem's foot in B: the square taper over FOOT_U with the key's corner cut, less the screw's pilot."""
    import manifold3d as M
    f = square_frustum(*FOOT_U)
    if not cut:
        return f
    zt = OM.BLOCK.height / 2
    pilot = M.Manifold.cylinder(SCREW["pilot_depth_mm"] + 1.0, SCREW["pilot_d_mm"] / 2, SCREW["pilot_d_mm"] / 2,
                                48).translate([0.0, 0.0, zt - FOOT_U[1] - 1.0])
    return f - key_region(FOOT_KEY - FOOT_KEY_CUT) - pilot


def modular_stem(shape):
    """The stem in B, its foot seated in the socket: the foot, the root cone, the 4 mm stem, the 45 deg cone and the
    cotter joint's taper and post at the object's seat plane, less the pilot and the code's dimples on the root."""
    import manifold3d as M
    o = OM.OBJECTS[shape]
    zt, rs = OM.BLOCK.height / 2, OM.STEM_D / 2
    z_seat = zt + seat(shape)[2] + (o["r_mm"] + o["stem_mm"])
    z_cone, h_cone = z_seat + MALE_U[0], CONE_R - rs
    z_root = zt - FOOT_U[0]                                     # the foot's top, 0.5 mm above the top face
    root = M.Manifold.cylinder(M_ROOT_H, M_ROOT_R, rs, SEG).translate([0.0, 0.0, z_root - 0.01])
    stem = M.Manifold.cylinder(z_cone - h_cone - (z_root + M_ROOT_H) + 1.0, rs, rs, SEG).translate(
        [0.0, 0.0, z_root + M_ROOT_H - 0.5])
    cone = M.Manifold.cylinder(h_cone, rs, CONE_R, SEG).translate([0.0, 0.0, z_cone - h_cone])
    end = male().transform(m4(J_B[shape], (0.0, 0.0, z_seat)))
    S = M.Manifold.batch_boolean([square_frustum(*FOOT_U) - key_region(FOOT_KEY - FOOT_KEY_CUT), root, stem, cone, end],
                                 M.OpType.Add)
    zt_ = zt
    pilot = M.Manifold.cylinder(SCREW["pilot_depth_mm"] + 1.0, SCREW["pilot_d_mm"] / 2, SCREW["pilot_d_mm"] / 2,
                                48).translate([0.0, 0.0, zt_ - FOOT_U[1] - 1.0])
    z_dot = z_root + 2.0                                        # dimples on the root cone, 2 mm above the foot
    r_dot = M_ROOT_R - (M_ROOT_R - rs) * 2.0 / M_ROOT_H
    n = CODES[(SET, shape)][1]
    angs = np.radians(45.0 + (np.arange(n) - (n - 1) / 2) * 28.0)        # on the side of the key's opposite corner
    dd = M.Manifold.batch_boolean([M.Manifold.sphere(DOT_R, 24).translate([r_dot * math.cos(a), r_dot * math.sin(a), z_dot])
                                   for a in angs], M.OpType.Add)
    return S - pilot - dd, z_seat


def modular_checks(shape, stem_B):
    """Shared volumes (mm3) of the seated foot with the block, the turned stems' stop above their seat, the screw's
    reach into the foot and the root's clearance to the block's top face."""
    import manifold3d as M
    B = OM.BLOCK
    w, b = modular_block(shape)
    blk = (w + b).transform(m4(np.eye(3), (0.0, 0.0, -B.height / 2)))      # back to B
    seated = abs((blk ^ stem_B).volume())
    turned = {}
    for k in (1, 2, 3):
        R = np.array([[math.cos(k * math.pi / 2), -math.sin(k * math.pi / 2), 0.0],
                      [math.sin(k * math.pi / 2), math.cos(k * math.pi / 2), 0.0], [0.0, 0.0, 1.0]])
        foot = foot_B(cut=False) - key_region(FOOT_KEY - FOOT_KEY_CUT)
        stop = None
        for i in range(0, 41):                                 # 0.1 mm steps up to 4 mm above the seat
            if (blk ^ foot.transform(m4(R, (0.0, 0.0, 0.1 * i)))).volume() < 1e-6:
                stop = round(0.1 * i, 2)
                break
        turned[f"{90 * k}"] = stop
    zt, zb = B.height / 2, -B.height / 2
    head_top = zb + SCREW["head_depth_mm"]
    tip = head_top + SCREW["length_mm"]
    foot_end = zt - FOOT_U[1]
    return {"seated_overlap_mm3": round(seated, 6), "turned_stops_out_mm": turned,
            "screw_engagement_mm": round(tip - foot_end, 2),
            "pilot_beyond_tip_mm": round(foot_end + SCREW["pilot_depth_mm"] - tip, 2),
            "foot_wall_at_end_mm": round((side_at(FOOT_U[1]) - SCREW["pilot_d_mm"]) / 2, 3),
            "foot_gap_to_floor_mm": round(FOOT_DEPTH - FOOT_U[1], 2),
            "root_gap_to_top_face_mm": round(-FOOT_U[0], 2)}


def modular_stl(code, name, colour="white"):
    return M_OUT / f"{M_DATE}-{name}_{code}_{colour}.stl"


def cmd_modular_build(a):
    """Blocks and stems of both object sets, and the white test block and stem of the fit test."""
    M_OUT.mkdir(parents=True, exist_ok=True)
    rep = {"what": "the modular tag block and stem (owner, 2026-11-17)",
           "script": "scripts/hardware/tracking_objects_print.py modular",
           "foot": {"side_at_top_face_mm": FOOT_S0, "taper_deg": TAPER_DEG, "taper_depth_mm": FOOT_H,
                    "socket_depth_mm": FOOT_DEPTH, "foot_u_mm": FOOT_U, "key_mm": FOOT_KEY, "key_cut_mm": FOOT_KEY_CUT,
                    "root_r_mm": M_ROOT_R, "root_h_mm": M_ROOT_H},
           "screw": SCREW, "label": {"height_mm": LABEL_H, "depth_mm": LABEL_DEPTH, "dot_r_mm": DOT_R},
           "root": {"root_d_mm": 2 * M_ROOT_R, "root_h_mm": M_ROOT_H - FOOT_U[0],
                    "foot_d_mm": round(side_at(FOOT_U[0]) * math.sqrt(2.0), 3), "fillet_r_mm": -FOOT_U[0]},
           "objects": {}, "ok": True}
    if M_REPORT.exists():
        old = json.loads(M_REPORT.read_text())
        rep.update({k: old[k] for k in ("bambu", "bambu_test", "check_ok") if k in old})
        for code, ob in old.get("objects", {}).items():
            if "check" in ob:
                rep.setdefault("_checks", {})[code] = ob["check"]
    ok = True
    hexagon = 3 * OM.BLOCK.face_width * OM.BLOCK.across_flats / 2 * OM.BLOCK.height
    for set_ in ("20261112", "large"):
        use_set(set_)
        for shape in ("cylinder", "sphere"):
            code, n = CODES[(set_, shape)]
            stem_B, z_seat = modular_stem(shape)
            w, b = modular_block(shape)
            ob = rep["objects"][code] = {"set": set_, "shape": shape, "dots": n, "ids": list(OM.IDS[shape]),
                                         "stem_mm": OM.OBJECTS[shape]["stem_mm"], "d_mm": 2 * OM.OBJECTS[shape]["r_mm"],
                                         "entry": "".join(f"{'+-'[int(c < 0)]}{'xyz'[i]}_O"
                                                          for i, c in enumerate(OM.R_OB @ J_B[shape][:, 0]) if c)}
            info = ob["block"] = {"what": f"{M_PARTS['block']}, ids {OM.IDS[shape][0]}-{OM.IDS[shape][-1]}, label {code}",
                                  "print": M_PRINT["block"]}
            ok &= export_part(info, w, b, lambda c, k=code: modular_stl(k, "block", c))
            info["solid_less_hexagon_mm3"] = round((w + b).volume() - hexagon, 2)
            drop = OM.BLOCK.height / 2 - FOOT_U[1]                       # B to the stem's print frame (foot's end at 0)
            st = stem_B.translate([0.0, 0.0, -drop])
            info = ob["stem"] = {"what": f"{M_PARTS['stem']}, {code}, {n} dimple{'s' if n > 1 else ''}",
                                 "print": M_PRINT["stem"],
                                 "print_to_B": m4(np.eye(3), (0.0, 0.0, drop)).tolist(),
                                 "length_mm": round(z_seat + POST_TOP - drop, 2)}
            ok &= export_part(info, st, None, lambda c, k=code: modular_stl(k, "stem", c))
            ob["checks"] = modular_checks(shape, stem_B)
            ok &= ob["checks"]["seated_overlap_mm3"] < 1e-6 and all(v is not None and v >= 0.5
                                                                     for v in ob["checks"]["turned_stops_out_mm"].values())
            if code in rep.get("_checks", {}):
                ob["check"] = rep["_checks"][code]
            print(code, ob["checks"], flush=True)
    rep.pop("_checks", None)
    # the fit test: a block in white only (no tags) and a short stem, its foot and root and 3 mm of stem
    use_set("large")
    import manifold3d as M
    B = OM.BLOCK
    prism = M.Manifold.extrude(M.CrossSection([B.vertices()[:6, :2].tolist()]), B.height).translate(
        [0.0, 0.0, -B.height / 2])
    tb = (prism - socket_B() - block_label("cylinder")).transform(m4(np.eye(3), (0.0, 0.0, B.height / 2)))
    info = rep["test_block"] = {"what": "fit test: the C45 block in white without tags", "print": M_PRINT["test_block"]}
    ok &= export_part(info, tb, None, lambda c: M_OUT / f"{M_DATE}-test_block_{c}.stl")
    zt, rs = B.height / 2, OM.STEM_D / 2
    z_root = zt - FOOT_U[0]
    ts = M.Manifold.batch_boolean([foot_B(), M.Manifold.cylinder(M_ROOT_H, M_ROOT_R, rs, SEG).translate(
        [0.0, 0.0, z_root - 0.01])], M.OpType.Add) - M.Manifold.cylinder(
        SCREW["pilot_depth_mm"] + 1.0, SCREW["pilot_d_mm"] / 2, SCREW["pilot_d_mm"] / 2, 48).translate(
        [0.0, 0.0, zt - FOOT_U[1] - 1.0])
    info = rep["test_stem"] = {"what": "fit test: the stem's foot and root", "print": M_PRINT["test_stem"]}
    ok &= export_part(info, ts.translate([0.0, 0.0, -(zt - FOOT_U[1])]), None, lambda c: M_OUT / f"{M_DATE}-test_stem_{c}.stl")
    rep["ok"] = bool(ok)
    M_REPORT.write_text(json.dumps(rep, indent=1))
    write_modular_readme(rep)
    print("wrote", M_REPORT.relative_to(ROOT), "ok" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


def cmd_modular_check(a):
    """Every tag face of the four blocks' black bodies, drawn from the mesh and detected against the tracker's map."""
    rep = json.loads(M_REPORT.read_text())
    tiles, ok = [], True
    for code, ob in rep["objects"].items():
        use_set(ob["set"])
        tm = OP.shifted(OM.block_map(ob["shape"]), OM.BLOCK.height / 2)
        rows, good = check_tags(modular_stl(code, "block", "black"), [tm], f"block {code}", tiles)
        ob["check"] = rows
        ok &= good
    rep["check_ok"] = bool(ok)
    M_REPORT.write_text(json.dumps(rep, indent=1))
    write_modular_readme(rep)
    if a.preview:
        OP.sheet(tiles, a.preview)
    print("check", "ok" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


M_OBJECT = {"C25": "25 mm cylinder", "S40": "40 mm sphere", "C45": "45 mm cylinder", "S67": "67 mm sphere"}


def write_modular_readme(rep):
    f, sc, O = rep["foot"], rep["screw"], rep["objects"]
    ck = O["C45"]["checks"]
    L = [f"# Modular tag blocks and stems, {M_DATE}",
         "",
         "The tag block and its stem as two parts (owner, 2026-11-17: \"it's quite costly printing these, in terms of "
         "filament and time, and im trying to anticipate breaks, which would make these blocks useless. i am not sure "
         "how to design joinery here since we cant interfere with the side faces with the blocks on them\"; \"some kind "
         "of identification for these modular components would also be helpful\"). A broken stem is reprinted in white "
         "and the block kept. Written by `scripts/hardware/tracking_objects_print.py modular build|check|bambu`.",
         "",
         "| Part | Files | Print |",
         "|---|---|---|"]
    for code, ob in O.items():
        for kind in ("block", "stem"):
            info = ob[kind]
            files = ", ".join(f"`{Path(info[c]['file']).name}`" for c in ("white", "black") if c in info)
            L.append(f"| {M_OBJECT[code]}: {info['what']} | {files} | {info['print']} |")
    for k in ("test_block", "test_stem"):
        L.append(f"| {rep[k]['what']} | `{Path(rep[k]['white']['file']).name}` | {rep[k]['print']} |")
    L += ["",
          f"Joint. The block's six tag faces are those of the earlier blocks. Its top face has a square socket "
          f"{f['side_at_top_face_mm']:g} mm across at the face. The socket's faces lie at {f['taper_deg']:g} deg to the "
          f"axis for {f['taper_depth_mm']:g} mm and run straight to {f['socket_depth_mm']:g} mm. The stem's foot is the "
          f"matching square taper, from {-f['foot_u_mm'][0]:g} mm above the top face to {f['foot_u_mm'][1]:g} mm into "
          f"the socket, so it seats on the four taper faces with {ck['foot_gap_to_floor_mm']:g} mm left below it. One "
          f"{sc['name']} screw goes up through a {sc['clear_d_mm']:g} mm hole from a counterbore "
          f"{sc['head_d_mm']:g} mm across and {sc['head_depth_mm']:g} mm deep in the bottom face. It threads "
          f"{ck['screw_engagement_mm']:g} mm into a {sc['pilot_d_mm']:g} mm pilot in the foot, the owner's size for "
          "self-forming M3 threads, and draws the taper into the socket. The head sits below the bottom face, which "
          "stays flat on the seat. The screw threads into the stem, so a stripped thread costs a stem, not a block. "
          "The socket's (-x_B, -y_B) corner is filled and the foot's cut, so the stem goes in one way only: turned by "
          f"90, 180 or 270 deg it stands {ck['turned_stops_out_mm']['90']:g} mm proud of its seat.",
          "",
          f"Stem. Above the foot a cone {2 * f['root_r_mm']:g} mm across narrows to the 4 mm stem over "
          f"{f['root_h_mm']:g} mm. The stem's other end is the cotter joint's taper and post of 2026-11-16 at the "
          "object's flat seat. Each object's stem has its own length, and its taper and post are turned as that "
          "object's socket requires.",
          "",
          "Identification. Each object has a code. It is cut into the block's bottom face with the block's ids, and "
          "both the block's bottom face and the stem's root bear as many dimples as the code's number. The labels read "
          "from below.",
          "",
          "| Code | Object | Block ids | Stem: block's top to the object's surface (mm) | Dimples | Stem as printed (mm) |",
          "|---|---|---|---|---|---|"]
    for code, ob in O.items():
        L.append(f"| {code} | {M_OBJECT[code]} | {ob['ids'][0]}-{ob['ids'][-1]} | {ob['stem_mm']:.1f} | {ob['dots']} | "
                 f"{ob['stem']['length_mm']:.1f} |")
    L += ["",
          "C45 and S67 are the objects of `assets/tracking/20261117-large_objects/`. C25 and S40 fit the cotter-joint "
          "objects of `assets/tracking/20261116-cotter_joint/`; the 25 mm cylinder and 40 mm sphere printed on "
          "2026-11-13 have their blocks glued on.",
          "",
          f"Assembly. Push the stem's foot into the socket, its cut corner to the socket's filled corner. Turn the "
          f"{sc['name']} screw in from below until the stem no longer rocks, and no further: self-forming threads in PLA "
          "strip after repeated screwing (owner, 2026-11-04). Then fit the stem's other end to the object with the "
          "cotter. To replace a broken stem, take the screw out and push the foot out of the socket with the "
          "screwdriver through the hole."]
    if rep.get("check_ok"):
        L += ["",
              "Checks. In the CAD the seated foot and the block share 0 mm3 for all four objects. Every tag face of the "
              "four blocks' black bodies, drawn at 20 px/mm from the mesh, decodes as its own id, 0.022 mm RMS from the "
              "tracker's map (ids 32-43 included, `src/sr2hand/tracking/tag_bits.json` now has tag36h11 ids 0-63)."]
    for st, what in (("large", "the larger objects' 400 rollouts on the two seats"),
                     ("20261112", "the series' layouts of the rollouts of 2026-11-05")):
        pth = M_OUT / f"{M_DATE}-root_clearance_{st}.json"
        if pth.exists():
            rows = json.loads(pth.read_text())["rows"]
            if st == "20261112":
                rows = [x for x in rows if x["task"] in ("pinpush", "press")
                        or tuple(x["xy"]) in SPHERE_LAYOUTS]
            L[-1] += (f" Over {what} no finger came within {min(x['root_min_mm'] for x in rows):.1f} mm of the root"
                      + (" (6.8 mm for the 6 mm root of 2026-11-16)." if st == "20261112" else "."))
    g = {c: O[c]["stem"]["solid_pla_g"] for c in O}
    L += ["",
          f"Mass. The stems are {min(g.values()):.2f} to {max(g.values()):.2f} g of solid PLA, and an {sc['name']} steel "
          "screw is about 1.4 g. With the screw the 67 mm sphere comes to about 31 g with its block, stem and cotter, "
          "against the simulations' 25 g. Weigh each assembled object; the simulation takes the measured mass."]
    if "bambu_test" in rep:
        q = rep["bambu_test"]["reslice"]
        L += ["",
              f"Fit test. Print `{M_DATE}-modular_test_p1s.3mf` first: the C45 block in white without tags and a foot "
              f"with its root ({q['hours']:.2f} h, {q['used_g']['1']:.1f} g white on filament 1). Screw them together. "
              "The joint works if the foot seats without rocking when the screw is firm, and comes out by hand once the "
              "screw is out. If it rocks with the screw firm, the foot bottoms out: report it, and FOOT_U or FOOT_DEPTH "
              "in `tracking_objects_print.py` changes."]
    for key, what in (("bambu_blocks", "the C45 and S67 blocks"), ("bambu_stems", "the C45 and S67 stems, two each")):
        if key in rep:
            q, fil = rep[key], rep[key]["filament"]
            rs = q["reslice"]
            u = rs["used_g"]
            col = (f"{u.get(str(fil['white']), 0):.1f} g white on filament {fil['white']} and "
                   f"{u.get(str(fil['black']), 0):.1f} g black on filament {fil['black']}, "
                   f"{sum(rs['loads'].values())} filament loads" if u.get(str(fil.get('black'))) else
                   f"{u.get(str(fil['white']), 0):.1f} g white on filament {fil['white']}, 5 mm brim")
            L += ["", f"Bambu Studio project `{Path(q['project']).name}`: {what}. Sliced again by the command line: "
                      f"{rs['hours']:.2f} h, {col}. Not in git; `tracking_objects_print.py modular bambu` rebuilds it."]
    (M_OUT / "README.md").write_text("\n".join(L) + "\n")


M_PLATES = {"bambu_test": ("test", "modular fit test", [("test_block", (90.0, 110.0)), ("test_stem", (140.0, 110.0))]),
            "bambu_blocks": ("blocks", "modular blocks C45 and S67", [("block_C45", (60.0, 80.0)),
                                                                      ("block_S67", (60.0, 140.0))]),
            "bambu_stems": ("stems", "modular stems C45 and S67, two each",
                            [("stem_C45", (80.0, 100.0)), ("stem_C45", (110.0, 100.0)), ("stem_S67", (140.0, 100.0)),
                             ("stem_S67", (170.0, 100.0))])}
M_BRIM = {"brim_type": "outer_only", "brim_width": "5"}       # the stems stand on a 6 mm foot


def cmd_modular_bambu(a):
    """Three projects: the fit test (white), the larger objects' two blocks (black on filament 1, white on 2) and their
    stems, two each (white)."""
    work = ROOT / f"logs/{M_DATE}-modular_blocks"
    rep = json.loads(M_REPORT.read_text())
    bad = False
    for key, (tag, title, placed) in M_PLATES.items():
        if a.only and (a.only == "test") != (key == "bambu_test"):
            continue
        items, seen = [], {}
        for n, xy in placed:
            kind, _, code = n.partition("_")
            white = M_OUT / f"{M_DATE}-{n}_white.stl" if kind == "test" else modular_stl(code, kind)
            black = modular_stl(code, kind, "black") if kind == "block" else None
            seen[n] = seen.get(n, 0) + 1
            items.append({"name": n if seen[n] == 1 else f"{n}_{seen[n]}", "white": white, "black": black, "place": xy})
        project = M_OUT / f"{M_DATE}-modular_{tag}_p1s.3mf"
        two = any(it["black"] is not None for it in items)
        fil, col = (FILAMENTS, COLOURS) if two else (L_FILAMENT, L_COLOURS)
        extra = None if two else M_BRIM
        q = OP.bambu_project(items, project, work / key, title, tower=TOWER, colours=col, filament=fil, extra=extra)
        if not two:
            q["problems"] = [x for x in q["problems"] if not x.startswith("filaments loaded")]
        q["reslice"] = reslice(project, work / f"{key}_reslice")
        want = sorted(str(f) for f in fil.values()) if two else [str(fil["white"])]
        if sorted(q["reslice"]["loads"]) != want and two:
            q["problems"].append(f"re-slice loads {q['reslice']['loads']}")
        q["filament"] = fil
        rep[key] = q
        print(f"{project.relative_to(ROOT)}: {q['reslice']['hours']:.2f} h, {q['layers']} layers, loads "
              f"{q['reslice']['loads']}, used {q['reslice']['used_g']} g; problems: {q['problems'] or 'none'}", flush=True)
        bad |= bool(q["problems"])
    M_REPORT.write_text(json.dumps(rep, indent=1))
    write_modular_readme(rep)
    if bad:
        raise SystemExit(1)


def write_readme(rep):
    P, sc, o = rep["parts"], rep["seat"], OM.OBJECTS
    blk = {s: OM.BLOCK_G + P[f"{s}_block"]["stem_solid_g"] for s in ("cylinder", "sphere")}
    L = [f"# Tracked cylinder and sphere, {DATE}",
         "",
         "Print files of the objects of `docs/notes/20261105-object_seating_and_tracking/`, written by "
         "`scripts/hardware/tracking_objects_print.py`; the tracker's maps of the assembled objects are "
         "`object_tracking_mjcf.object_map(shape)`. Two-colour parts are a white body and its black cells in one "
         "frame: load both files at once as one object with two parts and give the black part the black filament.",
         "",
         "| Part | Files | Print |",
         "|---|---|---|"]
    names = list(PARTS)
    k = names.index("cylinder") + 1
    names[k:k] = [n for n in ALTERNATIVES if n in P]
    for name in names:
        what, p = {**PARTS, **ALTERNATIVES}[name], P[name]
        files = ", ".join(f"`{Path(p[c]['file']).name}`" for c in ("white", "black") if c in p)
        how = {"cylinder_block": "upright, block on the bed; 65 layers in both colours",
               "sphere_block": "upright, block on the bed; 65 layers in both colours",
               "cylinder": "standing on its -z end (tag 23 on the bed, tag 22 on top)",
               "cylinder_tag23": "in place of the cylinder above, standing on its -z end (tag 23 on the bed)",
               "cylinder_plain": "in place of the cylinder above, standing; white only",
               "sphere_lower": "flat side down, the seat on top", "sphere_upper": "flat side down",
               "sphere_pin": "standing; print three"}[name]
        L.append(f"| {what} | {files} | {how} |")
    L += ["",
          f"Geometry. The cylinder prints at {2 * sc['cylinder']['r_print_mm']:.1f} mm, {o['cylinder']['length_mm']:g} mm "
          f"long, so {o['cylinder']['tape_mm']} mm grip tape brings it to {2 * o['cylinder']['r_mm']:g} mm; the end faces "
          f"stay bare for their tags. The sphere is {2 * o['sphere']['r_mm']:g} mm and bare. The block's top lies "
          f"{o['cylinder']['stem_mm']} mm (cylinder) and {o['sphere']['stem_mm']} mm (sphere) below the object's taped or "
          "bare surface, 15 mm below the lowest finger point of the plan's five layouts and the measured servo's five "
          f"best per task. The flat seat is {SEAT_D:g} mm across and lies {sc['cylinder']['flat_depth_mm']:.2f} mm "
          f"(cylinder) and {sc['sphere']['flat_depth_mm']:.2f} mm (sphere) below the printed surface; the key is "
          f"{KEY[0]} x {KEY[1]} x {KEY[2]} mm with {KEY_CLEAR} mm clearance per side in its slot, its long side along "
          "the cylinder's axis.",
          "",
          "Assembly. Glue the key into its slot with the seat flat on the object's flat (thin CA glue or epoxy), the "
          "block's tag faces then upright. Glue the hemispheres with the three dowels. Wrap the cylinder's curved "
          f"surface in the tape, leaving a {SEAT_D:g} mm hole at the seat. Weigh: each object with its block and tape is "
          f"{MASS_G:g} g in the simulations. The plan expects {OM.BLOCK_G:g} g for a block with 0.8 mm walls, and the stem "
          f"with its cone and key is {P['cylinder_block']['stem_solid_g']} g (cylinder) and "
          f"{P['sphere_block']['stem_solid_g']} g (sphere) of solid PLA, so the cylinder body with its tape has about "
          f"{MASS_G - blk['cylinder']:.1f} g and the sphere about {MASS_G - blk['sphere']:.1f} g. Weigh the printed blocks "
          "and the tape first, then set the bodies' infill to make up the rest.",
          "",
          "Calibration. A bundle adjustment of the assembled cylinder over its eight tags (`pose.bundle_adjust` with "
          "the block's three across-flats as lengths) gives its map, and with it where the cylinder sits on its block. "
          "The sphere's map is its block's, placed by the seat and key. In the CAD the assembled parts share "
          f"{rep['assembly_overlap_mm3']['cylinder']:g} mm3 (cylinder) and {rep['assembly_overlap_mm3']['sphere']:g} mm3 "
          "(sphere): the seats touch the flats and the keys clear their slots. With end tag 23 alone the adjustment has "
          "seven tags, and the cylinder's turn about the stem is read from where tag 23 sits, "
          f"{o['cylinder']['length_mm'] / 2:g} mm from the stem. Without end tags the cylinder's map is the CAD's; the "
          f"key's clearance then leaves its turn about the stem within +-{math.degrees(2 * KEY_CLEAR / KEY[1]):.1f} deg.",
          "",
          f"Printed: the owner printed the {ALTERNATIVES[PRINTED['cylinder']].replace('body, ', '')} on 2026-11-13. "
          "Its map is then its block's, placed by the seat and key, as for the sphere. A flat plate pressed on the face "
          f"of tag {OM.IDS['cylinder'][0]} or {OM.IDS['cylinder'][3]} lies parallel to the cylinder's axis, "
          f"{OM.BLOCK.across_flats / 2 - sc['cylinder']['r_print_mm']:.1f} mm outside its printed side. Equal gaps "
          "at both ends of the cylinder set its turn on the block before the glue sets (epoxy leaves time to "
          "adjust). A gap difference d over a spacing L along the axis is a turn of d / L rad."]
    if "check_ok" in rep:
        rms = [r["corner_rms_mm"] for p in P.values() for r in p.get("check", [])]
        span = f"{min(rms):.3f}" if f"{min(rms):.3f}" == f"{max(rms):.3f}" else f"{min(rms):.3f} to {max(rms):.3f}"
        L += ["", f"Check: every tag face of the black bodies, drawn at {OP.PPM:.0f} px/mm from the mesh, decodes as its "
                  f"own id with corners {span} mm RMS from the map."]
    def sliced(q):
        f, u = q.get("filament", {"white": 2, "black": 3}), q["used_g"]
        return (f"{q['prediction_s'] / 3600:.1f} h, {u.get(str(f['white']), 0):.1f} g white and "
                f"{u.get(str(f['black']), 0):.1f} g black, {q['filament_loads']} filament loads")

    if "bambu" in rep:
        q = rep["bambu"]
        fil = q.get("filament", {"white": 2, "black": 3})
        s = (f"Bambu Studio project: `{Path(q['project']).name}`, all parts on one plate, the black on filament "
             f"{fil['black']} and the white on filament {fil['white']}. Map filament {fil['black']} to the black spool "
             f"(AMS slot A1) and filament {fil['white']} to the white one when sending (owner, 2026-11-13). Each "
             "object's filament is written on the object, where Bambu Studio's window reads it; before 2026-11-13 the "
             "window put the one-part objects on filament 1. Sliced by the command line: "
             f"{sliced(q)}.")
        phrase = {"cylinder_tag23": "the cylinder bearing only end tag 23", "cylinder_plain": "the cylinder without end tags"}
        if rep.get("bambu_variants"):
            s += (" Tag 22 lies on the printed cylinder's top, so with it the prime tower rises to "
                  f"{o['cylinder']['length_mm']:g} mm.")
        for v, qv in rep.get("bambu_variants", {}).items():
            s += f" With {phrase[v]} in its place, `{Path(qv['project']).name}`: {sliced(qv)}."
        L += ["", s + " Not in git; `tracking_objects_print.py bambu [--cylinder NAME]` rebuilds them."]
    if BLOCKS_REPORT.exists():
        L += ["", f"Second version of the two blocks (owner, 2026-11-16), with the stem's root widened to {ROOT_D:g} mm "
                  f"and filleted to the block's top: `{BLOCKS_OUT.relative_to(ROOT)}/`."]
    if J_OUT.exists():
        L += ["", "Third version (owner, 2026-11-16), the blocks held without glue by a cotter joint, with a new cylinder "
                  f"body and lower hemisphere: `{J_OUT.relative_to(ROOT)}/`."]
    if L_OUT.exists():
        L += ["", "Larger objects (owner, 2026-11-17), a 45 mm cylinder and a 67 mm sphere with the cotter joint, on one "
                  f"plate in white: `{L_OUT.relative_to(ROOT)}/`."]
    (OUT / "README.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=sorted(OM.OBJECT_SETS), default="20261112",
                    help="the objects: those printed on 2026-11-12, or the larger ones of 2026-11-17")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("check")
    p.add_argument("--preview")
    p = sub.add_parser("bambu")
    p.add_argument("--cylinder", choices=list(CYLINDERS), default="cylinder",
                   help="the cylinder on the plate; an alternative writes its own project beside the default one")
    p = sub.add_parser("blocks", help=f"the blocks' second version ({BLOCKS_OUT.relative_to(ROOT)})")
    p.add_argument("step", choices=("build", "check", "bambu"))
    p.add_argument("--preview")
    p = sub.add_parser("joint", help=f"the cotter joint's parts and fit coupons ({J_OUT.relative_to(ROOT)})")
    p.add_argument("step", choices=("build", "check", "bambu"))
    p.add_argument("--preview")
    p.add_argument("--only", choices=("bambu", "bambu_coupons"), help="bambu: one of the two projects")
    p = sub.add_parser("modular", help=f"the modular tag blocks and stems ({M_OUT.relative_to(ROOT)})")
    p.add_argument("step", choices=("build", "check", "bambu"))
    p.add_argument("--preview")
    p.add_argument("--only", choices=("test", "parts"), help="bambu: the fit test's plate or the parts' plates")
    p = sub.add_parser("large", help=f"the larger objects with the cotter joint ({L_OUT.relative_to(ROOT)}); needs --set large")
    p.add_argument("step", choices=("build", "bambu"))
    p.add_argument("--only", choices=("mass", "bambu"), help="bambu: the bodies' mass slices or the plate")
    p.add_argument("--infill", default="lightning:10", help="bambu: the plate's infill, PATTERN:PERCENT")
    a = ap.parse_args()
    use_set(a.set)
    if a.cmd == "modular":
        {"build": cmd_modular_build, "check": cmd_modular_check, "bambu": cmd_modular_bambu}[a.step](a)
        return
    if a.cmd == "large":
        if a.set != "large":
            raise SystemExit("large: run with --set large")
        {"build": cmd_large_build, "bambu": cmd_large_bambu}[a.step](a)
        return
    if a.cmd == "joint":
        {"build": cmd_joint_build, "check": cmd_joint_check, "bambu": cmd_joint_bambu}[a.step](a)
        return
    if a.cmd == "blocks":
        {"build": cmd_blocks_build, "check": cmd_blocks_check, "bambu": cmd_blocks_bambu}[a.step](a)
        return
    {"build": cmd_build, "check": cmd_check, "bambu": cmd_bambu}[a.cmd](a)


if __name__ == "__main__":
    main()
