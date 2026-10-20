#!/usr/bin/env python3
"""Concept b of the DIP-link redesign (2026-10-18): the ReSkin 5X board in a rigid cassette that takes the place of the
printed tip's dorsal part, so the five chips sit inside the tip's former volume, closer to the magnets.

Owner, 2026-10-18: "we are going to redesign the DIP linkage entirely so that the board can be as close to the printed
tip as possible, while trying to keep the contact geometry the same. although that might change as well ... it's
important that the board does not wiggle at all--youll want something to fasten/sandwich it in place". Specification:
docs/experiments/20261018-dip_link_board_mount/spec.json (scripts/dip_link_spec.py); checks: scripts/dip_link_geometry.py.

The board (20.0 mm wide) is wider than the tip (14.8 mm), so a slot inside the TPU would cut the tip through both sides.
The cassette therefore replaces the TPU dorsal of the plane x_F = X_SLOT (0.45 mm, 4.0 mm behind the block's magnets):
a tongue of the DIP link behind the board, the board, and a cover over the board's component face whose front face is
flush with the chip tops. The TPU becomes a pad 11.0 mm deep with the printed tip's palmar and side surfaces; its back face
rests on the cover and on the five chips. Four brass M1.6 screws clamp cover, board and tongue through the board's four
holes into brass nuts on the tongue's back; the board is located by two datum walls of the tongue's seat pocket. Two
dovetail rails on the cover snap into grooves of the pad's back face and press it onto the cover; four notches of the pad
fit around the screw heads; its mount face rests on the bridge.

Frame F (scripts/dip_link_geometry.py): mm, x toward the dorsal side, y along the PIP axis, z up; the straight finger
hangs along -z. Tip frame of the finite elements (scripts/tpu_tip_design.py): palmar face x_t = +8.5, x_t = -x_F - 2.05.

Subcommands, in order (heavy steps through ~/.claude/bin/resguard.sh; femtip = ~/miniconda3/envs/femtip/bin/python):
  fe         finite elements of the pad against its depth on a coarse grid shared by the whole block and the pads
             (femtip, 1.5-2.6 GB) -> concept_b/fe/fem_coarse.jsonl
  prep       clevis arms cut from assets/cad_v1/fingertip_mount_DIP_link.stl, the bridge's profile from the keep-out,
             the pads cut from the readiness print STLs (femtip, manifold3d) -> logs/20261018-dip_link/concept_b/,
             concept_b/cad/*_pad_*.stl
  cad        B-rep solids, STEP and STL of the DIP link and the cover, faceted STEP of the pads (runs freecadcmd on this
             file); STEP files over 1.5 MB are zipped, which git LFS stores
  clearance  exact clearance of every new part to the MCP link and PIP servo over PIP -18..92 deg (femtip, ~1 GB)
  sensing    field at the chips and force resolution per FE depth (.venv)
  render     iso view, sections through the chips, exploded view, assembly STL (femtip, pyvista)
  eval       concept_b/evaluation.json

    ~/.claude/bin/resguard.sh run --mem 2500M --cpu 100 -- taskset -c 12 env NUMPY_MADVISE_HUGEPAGE=0 \\
        OMP_NUM_THREADS=1 ~/miniconda3/envs/femtip/bin/python scripts/dip_link_concept_b.py fe
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import scipy.spatial  # noqa: F401  (loads the environment's libstdc++ before manifold3d; matplotlib and VTK need it)
except ImportError:                                 # freecadcmd's Python
    pass

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "docs/experiments/20261018-dip_link_board_mount"
OUT = STUDY / "concept_b"
FE_ROWS = OUT / "fe/fem_coarse.jsonl"
DATE = "20261018"
# The parameter file ($DIPB_PARAMS; default the committed concept b, params/20261018-concept_b.json) names its design's
# folders under "outputs" (concept_b/ for the committed cassette, concept_b/final/ for b's final). A saved parameter set has
# none: its files go to logs/20261018-dip_link/params/<file stem>/, or to $DIPB_OUT (CAD files; print files beside it) and
# $DIPB_WORK (intermediate meshes) when scripts/dip_link_params.py sets them.
PARAMS = Path(os.environ.get("DIPB_PARAMS") or STUDY / "params/20261018-concept_b.json")
_OUTPUTS = json.loads(PARAMS.read_text()).get("outputs", {})
_SCRATCH = REPO / "logs/20261018-dip_link/params" / PARAMS.stem
CAD_DIR = Path(os.environ["DIPB_OUT"]) if os.environ.get("DIPB_OUT") else REPO / _OUTPUTS.get("cad", _SCRATCH / "cad")
PRINT_OUT = CAD_DIR.parent / "print" if os.environ.get("DIPB_OUT") else REPO / _OUTPUTS.get("print", _SCRATCH / "print")
MEDIA = REPO / _OUTPUTS.get("media", _SCRATCH / "media")
EVAL = REPO / _OUTPUTS.get("evaluation", _SCRATCH / "evaluation.json")
WORK = Path(os.environ["DIPB_WORK"]) if os.environ.get("DIPB_WORK") else REPO / _OUTPUTS.get("work", _SCRATCH / "work")
PREFIX = os.environ.get("DIPB_PREFIX") or _OUTPUTS.get("prefix", f"{DATE}-concept_b")   # $DIPB_PREFIX: an export's name

# ------------------------------------------------------------------------------------------------- design parameters
# Every length in mm, frame F. Fixed by the specification (the printed tip, the board, the clevis arms):
TIP0 = dict(x_palmar=-10.55, x_back=6.45, y_half=7.4, z_mount=-67.30, z_end=-88.71)   # at neck_len 0 (apply() moves z)
TIP = dict(TIP0)
Z_CONTACT0 = Z_CONTACT = -78.16
MAG_BACK_BLOCK = -3.55          # x_F of the block's and capsule's magnets' back faces (3 x 2 mm, tops 5 mm deep)
SPOT = 0.42                     # sensing spot under the package top (MLX90393 datasheet Fig. 26)
Y_BRIDGE = (-17.05, 13.40)      # between the clevis arms (spec R6)
Z_BOARD0 = Z_BOARD = (-88.21, -68.06)      # the board's main square along z (v 10.05 .. -10.10)
HOLES_UV = [(-8.0, -8.10), (8.0, -8.10), (8.0, 7.90), (-8.0, 7.90)]
HOLE_D = 2.40
PRINT_DIR = REPO / "docs/experiments/20261016-sim2real_fingertip_readiness/print"
# The design's sizes, clearances and spacings form one dict P, read from the parameter file
# docs/experiments/20261018-dip_link_board_mount/params/20261018-concept_b.json (or the file named by $DIPB_PARAMS), which
# gives each key's value, unit, what it sets, allowed range and the checks it affects; DESIGN holds the discrete choices.
# apply() sets both and derives the module-level names below from them; every function reads these names when it runs,
# so apply(values) changes the design in a running process (scripts/dip_link_viewer.py). Model values: the chip height,
# the FR-4 thickness and the board's flatness are pending measurements on the owner's board (spec.json "pending").
P: dict = {}
DESIGN: dict = {}
PAD_SETS = {   # printed 85A tips cut into pads: print STL, orientation, magnets' back face x_F
    "seat2p7": {"block4": ("20261016-siraya85a_block_four_d3x2_depth5_seat2p7", "back", -3.55),
                "block5": ("20261016-siraya85a_block_five_d3x2_depth5_seat2p7", "back", -3.55),
                "cap8.5": ("20261016-siraya85a_tcd_cap8p5_four_d3x2_depth5_seat2p7", "upright", -3.55),
                "ell16": ("20261016-siraya85a_tcd_ell7p4x8p5x16x32_four_d3x2_depth5p5_seat2p7", "upright", -3.05),
                "ell11": ("20261016-siraya85a_tcd_ell7p4x8p5x11x22_four_d3x2_depth7_seat2p7", "upright", -1.55)},
    "seat2p9": {"block4": ("20261016-siraya85a_block_four_d3x2_depth5_seat2p9", "back", -3.55),
                "block5": ("20261016-siraya85a_block_five_z7p4_d3x2_depth5_seat2p9", "back", -3.55),
                "cap8.5": ("20261016-siraya85a_tcd_cap8p5_four_d3x2_depth5_seat2p9", "upright", -3.55),
                "ell16": ("20261016-siraya85a_tcd_ell7p4x8p5x16x32_four_d3x2_depth5p5_seat2p9", "upright", -3.05),
                "ell11": ("20261016-siraya85a_tcd_ell7p4x8p5x11x22_four_d3x2_depth7p5_seat2p9", "upright", -1.05)},
    "v2_five": {"block5": ("docs/experiments/20261018-dip_link_board_mount/v2/print/"
                           "20261018-siraya85a_block_five_z7p4_d3x2_depth5_seat2p9", "back", -3.55),
                "ell11_5": ("docs/experiments/20261018-dip_link_board_mount/v2/print/"
                            "20261018-siraya85a_tcd_ell7p4x8p5x11x22_five_y2p5_z9p8_s0p5_d3x2_depth7p5_seat2p9", "upright", -1.05)},
}


def pad_source(name):
    """A pad set's print file stem: a name in the readiness print folder, or a path from the repository's root."""
    return REPO / name if "/" in name else PRINT_DIR / name


def load_params(path=None):
    """({key: value}, {design key: value}) of a parameter file (default PARAMS: $DIPB_PARAMS or the committed concept b)."""
    doc = json.loads(Path(path or PARAMS).read_text())
    return ({q["key"]: float(q["value"]) for q in doc["parameters"]},
            {d["key"]: d["value"] for d in doc.get("design", [])})


def apply(values=None, design=None):
    """Set P and DESIGN from a parameter set and derive the geometry's names from them (mm, frame F)."""
    global T_BEHIND, X_SLOT, CHIP_H, FR4, X_COMP, X_SEAT, RIM_H, T_TONGUE, X_TONGUE_BACK, Y_RIM, TAB_CUT_V, SCREW_D, \
        SCREW_CLEAR, HEAD_D, HEAD_H, NUT_AF, NUT_H, NUT_POCKET_AF, NUT_POCKET_H, SCREW_L, HEAD_NOTCH_D, HEAD_NOTCH_H, \
        RAIL_Y, RAIL_BASE, RAIL_TOP, RAIL_H, RAIL_Z, GROOVE_FIT, GROOVE_FLOOR, WINDOW_MARGIN, COVER_TAB_Y, COVER_TAB_Z, \
        WIRE_SLOT, TIE, KEEP_MARGIN, X_SPINE_BACK, Z_WEB_LOW, TIPS, TIP, Z_CONTACT, Z_BOARD, NECK
    P.update(values or {})
    DESIGN.update(design or {})
    NECK = P.get("neck_len", 0.0)       # the DIP link lengthened between its bridge and the pad: the tip side moves by -NECK
    TIP = dict(TIP0, z_mount=TIP0["z_mount"] - NECK, z_end=TIP0["z_end"] - NECK)
    Z_CONTACT = Z_CONTACT0 - NECK
    Z_BOARD = (Z_BOARD0[0] - NECK, Z_BOARD0[1] - NECK)
    T_BEHIND = P["t_behind"]             # TPU behind the block's magnets (the fe rows: 10, 6, 4, 3 mm; 10.0 = whole tip)
    X_SLOT = MAG_BACK_BLOCK + T_BEHIND   # the pad's back face = the cover's front face = the chip tops (0.45)
    CHIP_H = P["chip_h"]                 # MLX90393 package height (0.80-1.00 mm; the cover is printed to the measured value)
    FR4 = P["fr4"]
    X_COMP = X_SLOT + CHIP_H             # the FR-4's component face (1.35)
    X_SEAT = X_COMP + FR4                # the FR-4's bare face = the tongue's seat (1.95)
    RIM_H = P["rim_h"]                   # the seat pocket's depth: the rim stops 0.15 mm short of the cover's back face
    T_TONGUE = P["t_tongue"]             # tongue behind the seat
    X_TONGUE_BACK = X_SEAT + T_TONGUE    # 5.55
    Y_RIM = P["y_rim"]                   # half-width of cover and tongue: the board's 10.0 and a 0.5 mm rim (spec R5)
    TAB_CUT_V = P["tab_cut_v"]           # the tab cut 1.60 mm from the main square, past R1 and R2 (v -11.40)
    # fasteners (brass, spec R10): M1.6 x 5 cheese head DIN 84 (head 3.0 x 1.0) through cover, board and tongue into an
    # M1.6 hex nut ISO 4032 (3.2 across flats, 1.3 thick) in a pocket on the tongue's back
    SCREW_D, SCREW_CLEAR = P["screw_d"], P["screw_clear"]
    HEAD_D, HEAD_H = P["head_d"], P["head_h"]
    NUT_AF, NUT_H = P["nut_af"], P["nut_h"]
    NUT_POCKET_AF, NUT_POCKET_H = P["nut_pocket_af"], P["nut_pocket_h"]
    SCREW_L = P["screw_l"]
    HEAD_NOTCH_D = P["head_notch_d"]     # the pad's corner notch around a head (0.2 mm all round)
    HEAD_NOTCH_H = P["head_notch_h"]
    # dovetail rails on the cover's front face, and the pad's grooves (snap fit)
    RAIL_Y = P["rail_y"]                 # rail centre |y|, between U1/U2/U4 (|y| <= 1.7), C1/C2 (<= 3.55) and U3/U5 (>= 5.3)
    RAIL_BASE, RAIL_TOP, RAIL_H = P["rail_base"], P["rail_top"], P["rail_h"]   # width at the cover, at the top, height
    RAIL_Z = (P["rail_z0"] - NECK, P["rail_z1"] - NECK)
    GROOVE_FIT = P["groove_fit"]         # the groove is narrower than the rail by this much per flank (the pad's preload)
    GROOVE_FLOOR = P["groove_floor"]     # clearance between the rail's top and the groove's floor
    WINDOW_MARGIN = P["window_margin"]   # windows clear every part by this much (spec R2)
    COVER_TAB_Y = P["cover_tab_y"]       # the cover stops at z -68.30 over the tab (|y| <= 5.6), at the mount plane elsewhere
    COVER_TAB_Z = P["cover_tab_z"] - NECK
    WIRE_SLOT = dict(y=(P["wire_slot_y0"], P["wire_slot_y1"]), z=(P["wire_slot_z0"] - NECK, P["wire_slot_z1"] - NECK))
    TIE = dict(y=(P["tie_y0"], P["tie_y1"]), z=(P["tie_z0"] - NECK, P["tie_z1"] - NECK), x_top=P["tie_x_top"],  # 2.5 mm tie
               tunnel_x=(P["tie_tunnel_x0"], P["tie_tunnel_x1"]),
               tunnel_z=(P["tie_tunnel_z0"] - NECK, P["tie_tunnel_z1"] - NECK))
    KEEP_MARGIN = P["keep_margin"]       # the bridge stays this far from the swept MCP link and PIP servo (0.5 + 0.3)
    X_SPINE_BACK = P["x_spine_back"]     # a dorsal spine behind the tongue joins both arms (the old plate's back face)
    Z_WEB_LOW = P["z_web_low"]           # webs beside the tongue (|y| >= 10.5, dorsal of the cover) reach the horn arm,
    #                                      which does not move with the neck
    TIPS = PAD_SETS[DESIGN.get("pads", "seat2p7")]


apply(*load_params())


def uv_to_yz(u, v):
    return u, -v + Z_CONTACT


def screw_yz():
    return [uv_to_yz(u, v) for u, v in HOLES_UV]


# ================================================================================================= finite elements

FE_GRID = dict(hx_back=0.45, hx_mag=0.4, hx_face=0.3, hy=0.5, hz=0.5)
FE_CASES = [("four_d3x2_d5", 10.0), ("four_d3x2_d5", 6.0), ("four_d3x2_d5", 4.0), ("four_d3x2_d5", 3.0),
            ("five_d3x2_d5", 10.0), ("five_d3x2_d5", 4.0)]


def fe_grid(x_back_t, g=FE_GRID):
    """Zones of the coarse grid (tip frame) for a pad whose back face lies at x_t = x_back_t; the fine zones around the
    magnets (x_t 1.5-3.5) and under the face are the same for every depth."""
    hb = g["hx_back"] if x_back_t > -2.0 else 0.8
    return dict(x=[(x_back_t, 1.0, hb), (1.0, 7.0, g["hx_mag"]), (7.0, 8.5, g["hx_face"])],
                y=[(-7.4, 7.4, g["hy"])],
                z=[(0.0, 0.9, 0.45), (0.9, 5.0, 0.8), (5.0, 17.0, g["hz"]), (17.0, 18.0, 0.5), (18.0, 21.41, g["hz"])])


def fe_done():
    if not FE_ROWS.exists():
        return set()
    return {(r["layout"], r["t_behind_mm"]) for r in (json.loads(l) for l in open(FE_ROWS))}


def cmd_fe(a):
    """The readiness study's model (scripts/sim2real_fingertip_fe.py: tpu_tip_magnet_layouts.run with two 0.45 mm walls
    at every outline, 85A at 30 % gyroid, bonded magnets, back and mount faces fixed) on a coarser grid, for the full
    block and for pads cut at T_BEHIND mm behind the magnets. The coarse grid enters both, so the ratio thinned / full
    is what the evaluation applies to the readiness rows."""
    sys.path.insert(0, str(REPO / "scripts"))
    import metric_magnet_prints as MP
    import tpu_tip_design as TD
    import tpu_tip_fem_tip as T
    import tpu_tip_magnet as TM
    import tpu_tip_magnet_layouts as TL
    TM.H_AUX = a.h_aux
    orig_comp = T.TipModel.compliance
    T.TipModel.compliance = lambda self, nodes, comps=(0,), block=40: orig_comp(self, nodes, comps, block)
    have = fe_done()
    FE_ROWS.parent.mkdir(parents=True, exist_ok=True)
    cases = FE_CASES if not a.only else [(l, float(t)) for l, t in (s.split("@") for s in a.only)]
    for layout, tb in cases:
        if (layout, tb) in have:
            print(layout, tb, "done", flush=True)
            continue
        x_slot = MAG_BACK_BLOCK + tb
        x_back_t = -x_slot - 2.05
        saved = (TD.X_BACK, T.X_BACK, TL.GRID, TL.RECIPE)
        TD.X_BACK = T.X_BACK = x_back_t
        TL.GRID = fe_grid(x_back_t)
        TL.RECIPE = TM.RECIPES["w85_30"]
        TL.LAYOUTS[layout] = MP.fe_layout(layout)
        real = TL.classes
        TL.classes = lambda l: MP.classes_walls(l, 2)
        tmpd = Path(tempfile.mkdtemp(dir=WORK if WORK.exists() else None))
        tmp = tmpd / "row.jsonl"
        try:
            t0 = time.time()
            TL.run(layout, out=tmp)
        finally:
            TL.classes = real
            TD.X_BACK, T.X_BACK, TL.GRID, TL.RECIPE = saved
        row = json.loads(tmp.read_text().splitlines()[-1])
        tmp.unlink()
        tmpd.rmdir()
        row.update(recipe="w85_30", recipe_params=TM.RECIPES["w85_30"], pocket_walls=2, t_behind_mm=tb,
                   x_slot_F=round(x_slot, 3), x_back_t=round(x_back_t, 3), pad_depth_mm=round(x_slot + 10.55, 3),
                   h_aux_mm=a.h_aux, grid=fe_grid(x_back_t), t_wall_s=round(time.time() - t0, 1),
                   script="scripts/dip_link_concept_b.py fe (tpu_tip_magnet_layouts.run on a coarse grid)")
        with open(FE_ROWS, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        tr = [math.hypot(*m[0]) for m in row["per_force"]["1"]["magnets"]]
        print(f"{layout} t_behind {tb:4.1f}: E* {row['E_best_MPa']:.2f} MPa, approach at 1 N "
              f"{row['per_force']['1']['delta_um']:.1f} um, magnet travel {sum(tr) / len(tr):.2f} um/N, "
              f"{row['n_dof']} dof, {row['t_total_s']:.0f} s", flush=True)


# ================================================================================================= mesh preparation

def _m3():
    import manifold3d as m3
    return m3


def to_manifold(mesh):
    import numpy as np
    m3 = _m3()
    return m3.Manifold(m3.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                               tri_verts=np.asarray(mesh.faces, np.uint32)))


def to_trimesh(man):
    import numpy as np
    import trimesh
    m = man.to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)


def box(lo, hi):
    m3 = _m3()
    lo = [float(v) for v in lo]
    return m3.Manifold.cube([float(h) - l for l, h in zip(lo, hi)]).translate(lo)


def prism(points_yx, z0, z1):
    """A convex prism along z over a polygon given in (y, x)."""
    import numpy as np
    m3 = _m3()
    P = [(x, y, z) for y, x in points_yx for z in (z0, z1)]
    return m3.Manifold.hull_points(np.array(P, float))


def cyl_x(y, z, r, x0, x1, n=48):
    """Cylinder along x."""
    import numpy as np
    m3 = _m3()
    c = m3.Manifold.cylinder(float(x1 - x0), float(r), float(r), n)          # along +z from 0
    return c.rotate([0.0, 90.0, 0.0]).translate([float(x0), float(y), float(z)])  # +z -> +x


def hex_x(y, z, af, x0, x1):
    """Hexagonal prism along x, flats parallel to z."""
    import numpy as np
    rc = af / math.sqrt(3.0)
    pts = [(y + rc * math.cos(math.radians(a)), z + rc * math.sin(math.radians(a))) for a in range(0, 360, 60)]
    m3 = _m3()
    P = [(x, py, pz) for py, pz in pts for x in (x0, x1)]
    return m3.Manifold.hull_points(np.array(P, float))


def rail_section(widen=0.0, top_extra=0.0):
    """Dovetail (y, x) outline of one rail at y > 0: RAIL_BASE wide on the cover's face, RAIL_TOP wide RAIL_H palmar of
    it. widen > 0 grows it per flank; top_extra moves its top palmar."""
    xb, xt = X_SLOT + 0.001, X_SLOT - RAIL_H - top_extra
    b, t = RAIL_BASE / 2 + widen, RAIL_TOP / 2 + widen
    return [(RAIL_Y - b, xb), (RAIL_Y + b, xb), (RAIL_Y + t, xt), (RAIL_Y - t, xt)]


def tip_to_F(V, orient):
    import numpy as np
    V = np.asarray(V, float)
    if orient == "back":          # printed on its back face: x_p = 21.41 - z_t, z_p = x_t + 8.5
        return np.column_stack([-(V[:, 2] - 8.5) - 2.05, V[:, 1], TIP["z_mount"] - (21.41 - V[:, 0])])
    return np.column_stack([-V[:, 0] - 2.05, V[:, 1], TIP["z_mount"] - V[:, 2]])


def F_to_tip(V, orient):
    """Inverse of tip_to_F (the pads are written back in their print frames)."""
    import numpy as np
    V = np.asarray(V, float)
    x_t, z_t = -V[:, 0] - 2.05, TIP["z_mount"] - V[:, 2]
    if orient == "back":
        return np.column_stack([21.41 - z_t, V[:, 1], x_t + 8.5])
    return np.column_stack([x_t, V[:, 1], z_t])


def pad_cutters():
    """Grooves for the rails and notches for the screw heads, in F."""
    m3 = _m3()
    z0, z1 = RAIL_Z[0] - 0.2, RAIL_Z[1] + 0.2
    parts = []
    for s in (-1, 1):
        sec = [(s * y, x) for y, x in rail_section(-GROOVE_FIT, GROOVE_FLOOR)]
        sec = [(y, X_SLOT + 0.5 if x > X_SLOT else x) for y, x in sec]       # through the back face
        parts.append(prism(sec, z0, z1))
    for y, z in screw_yz():
        parts.append(cyl_x(y, z, HEAD_NOTCH_D / 2, X_SLOT - HEAD_NOTCH_H, X_SLOT + 1.0))
    return m3.Manifold.batch_boolean(parts, m3.OpType.Add)


def pad_print_frame(tm, orient):
    """A pad (trimesh in F) in its print frame: the tip's sheet frame (F_to_tip), the block pads lowered so that their cut
    back face (x_F = X_SLOT) lies on the bed; the curved pads stand on their mount face as the whole tips do."""
    import numpy as np
    import trimesh
    V = F_to_tip(tm.vertices, orient)
    if orient == "back":
        V[:, 2] -= TIP["x_back"] - X_SLOT
    out = trimesh.Trimesh(V, np.asarray(tm.faces), process=False)
    if out.volume < 0:
        out.invert()
    return out


def pad_pauses(key):
    """The pad's pause heights (mm above the bed) from the tip's 2.9 or 2.7 mm print sheet: the block pads print on their
    cut back face, 6.0 mm (TIP x_back - X_SLOT) lower than the whole tip; the upright pads keep the sheet's heights."""
    name, orient, _ = TIPS[key]
    sheet = json.loads(pad_source(f"{name}_g30.json").read_text())
    dz = (TIP["x_back"] - X_SLOT) if orient == "back" else 0.0
    return [dict(print_z_mm=round(q["print_z"] - dz, 2), magnets=q["magnets"], message=q["message"]) for q in sheet["pauses"]]


def make_pad(key):
    """The printed tip cut at the slot plane, with the rail grooves and the head notches, in F."""
    import trimesh
    name, orient, _ = TIPS[key]
    t = trimesh.load(pad_source(f"{name}.stl"), force="mesh")
    t = trimesh.Trimesh(tip_to_F(t.vertices, orient), t.faces, process=True)    # both maps are proper rotations
    if t.volume < 0:
        t.invert()
    M = to_manifold(t)
    keep = box((-20, -20, TIP["z_end"] - 5.0), (X_SLOT, 20, TIP["z_mount"] + 7.3))
    return (M ^ keep) - pad_cutters(), t


def bridge_profile(margin=None):
    """Top boundary z_top(x) of the bridge between the arms: from z -67.30 upward, the highest z whose column stays
    `margin` mm (default KEEP_MARGIN) from the keep-out swept by the MCP link and the PIP servo over PIP -18..92 deg,
    projected over the bridge's y range widened by the margin. Returns [(x, z_top)] at the keep-out's 0.25 mm pitch."""
    import numpy as np
    margin = KEEP_MARGIN if margin is None else margin
    from scipy.ndimage import distance_transform_edt
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import dip_link_geometry as G
    K = G.keepout()
    j0 = int(math.floor((Y_BRIDGE[0] - margin - K.lo[1]) / K.pitch))
    j1 = int(math.ceil((Y_BRIDGE[1] + margin - K.lo[1]) / K.pitch))
    occ2 = K.occ[:, j0:j1 + 1, :].any(axis=1)                      # (x, z)
    d2 = distance_transform_edt(~occ2) * K.pitch
    xs = K.lo[0] + K.pitch * np.arange(occ2.shape[0])
    zs = K.lo[2] + K.pitch * np.arange(occ2.shape[1])
    k0 = int(np.searchsorted(zs, TIP["z_mount"] - 1e-6))
    out = []
    for i, x in enumerate(xs):
        if x < TIP["x_palmar"] - 0.25 or x > X_SPINE_BACK + 0.25:
            continue
        k = k0
        while k + 1 < len(zs) and d2[i, k + 1] >= margin:
            k += 1
        out.append((float(x), float(zs[k])))
    # the lower envelope of each point and its neighbours, so the polyline stays under the staircase
    zt = [min(out[max(i - 1, 0)][1], out[i][1], out[min(i + 1, len(out) - 1)][1]) for i in range(len(out))]
    pts = [(round(x, 3), round(z, 3)) for (x, _), z in zip(out, zt) if TIP["x_palmar"] - 1e-6 <= x <= X_SPINE_BACK + 1e-6]
    zl = min(z for x, z in zip([o[0] for o in out], zt) if x <= TIP["x_palmar"] + 0.25)
    zr = min(z for x, z in zip([o[0] for o in out], zt) if x >= X_SPINE_BACK - 0.25)
    pts = [(TIP["x_palmar"], zl)] + pts + [(X_SPINE_BACK, zr)]
    keep = [pts[0]] + [p for i, p in enumerate(pts[1:-1], 1)
                       if not (abs(p[1] - pts[i - 1][1]) < 1e-9 and abs(p[1] - pts[i + 1][1]) < 1e-9)] + [pts[-1]]
    return keep


def decimate_for_step(tm, target=6000):
    """A coarser copy of a pad's mesh for its faceted STEP: the outer shell decimated (VTK quadric) to about `target`
    triangles, the magnet pockets kept. Returns (mesh, largest distance of its vertices from the original surface)."""
    import numpy as np
    import pyvista as pv
    import trimesh
    bodies = sorted(tm.split(only_watertight=False), key=lambda b: -abs(b.volume))
    outer = bodies[0]
    if len(outer.faces) > target:
        pm = pv.PolyData(np.asarray(outer.vertices, float), np.hstack([np.full((len(outer.faces), 1), 3), outer.faces]).ravel())
        d = pm.decimate(1.0 - target / len(outer.faces), volume_preservation=True)
        f = np.asarray(d.faces).reshape(-1, 4)[:, 1:]
        outer2 = trimesh.Trimesh(np.asarray(d.points), f, process=True)
        if outer2.volume < 0:
            outer2.invert()
    else:
        outer2 = outer
    _, dev, _ = trimesh.proximity.closest_point(outer, outer2.vertices) if len(outer2.vertices) < 20000 else (None, [0.0], None)
    return trimesh.util.concatenate([outer2] + bodies[1:]), float(np.max(dev))


def cmd_prep(a):
    """Arms from the STL, the bridge's profile from the keep-out, the pads from the print STLs -> WORK/."""
    import numpy as np
    import trimesh
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import dip_link_geometry as G
    dip = G.load_part("dip_link")
    M = to_manifold(dip)
    arms = {"arm_horn": M ^ box((-40, -24.0, -110), (40, Y_BRIDGE[0] - 0.01, -30)),
            "arm_slot": M ^ box((-40, Y_BRIDGE[1] + 0.01, -110), (40, 18.0, -30))}
    info = {}
    for k, m in arms.items():
        tm = to_trimesh(m)
        tm.export(WORK / f"{k}.stl")
        info[k] = dict(volume_mm3=round(float(tm.volume), 1), bounds=np.round(tm.bounds, 3).tolist(),
                       faces=int(len(tm.faces)), watertight=bool(tm.is_watertight))
        print(k, info[k], flush=True)
    prof = bridge_profile()
    (WORK / "bridge_profile.json").write_text(json.dumps(dict(margin=KEEP_MARGIN, x_spine_back=X_SPINE_BACK, profile=prof)))
    print("bridge profile", len(prof), "points; z_top at x -10.55, -5, 0, 5:",
          [z for x, z in prof if any(abs(x - v) < 0.13 for v in (-10.5, -5.0, 0.0, 5.0))], flush=True)
    CAD_DIR.mkdir(parents=True, exist_ok=True)
    for key in TIPS:
        pad, whole = make_pad(key)
        tm = to_trimesh(pad)
        tm.update_faces(tm.nondegenerate_faces())          # zero-area slivers of the boolean (the 11/22's cut leaves two)
        tm.remove_unreferenced_vertices()
        tm.export(WORK / f"pad_{key}.stl")
        tm.export(CAD_DIR / f"{PREFIX}_pad_{key}.stl")              # the pad in F (the print STL cut, unchanged)
        tp = pad_print_frame(tm, TIPS[key][1])
        PRINT_OUT.mkdir(parents=True, exist_ok=True)
        tp.export(PRINT_OUT / f"{PREFIX}_pad_{key.replace('.', 'p')}_print.stl")
        dec, dev = decimate_for_step(tm)
        dec.export(WORK / f"pad_{key}_step.stl")
        info[f"pad_{key}_step"] = dict(faces=int(len(dec.faces)), max_deviation_mm=round(dev, 4))
        info[f"pad_{key}"] = dict(volume_mm3=round(float(tm.volume), 1), whole_volume_mm3=round(float(whole.volume), 1),
                                 bounds=np.round(tm.bounds, 3).tolist(), watertight=bool(tm.is_watertight))
        print(key, info[f"pad_{key}"], flush=True)
    if cable_on():                              # the Qwiic cable along its route (a model, for the assembly and checks)
        cm = cable_mesh()
        cm.export(CAD_DIR / f"{PREFIX}_cable.stl")
        info["cable"] = dict(file=str((CAD_DIR / f"{PREFIX}_cable.stl").relative_to(REPO)) if REPO in CAD_DIR.parents else
                             str(CAD_DIR / f"{PREFIX}_cable.stl"), length_mm=round(cable_length(), 2),
                             bounds=np.round(cm.bounds, 3).tolist(), watertight=bool(cm.is_watertight))
        print("cable", info["cable"], flush=True)
    (WORK / "prep.json").write_text(json.dumps(info, indent=1))


# ================================================================================================= the board

def spec():
    return json.loads((STUDY / "spec.json").read_text())


def board_parts(populated_only=True):
    """The board's parts from spec.json: name -> (u0, u1, v0, v1, height) in the board frame, mm."""
    out = {}
    for k, p in spec()["board"]["parts"].items():
        if populated_only and not p["populated"]:
            continue
        if p["kind"] == "J1":
            continue                              # removed with the tab (cut at TAB_CUT_V)
        (cu, cv), (su, sv) = p["centre_uv"], p["size_uv"]
        out[k] = (cu - su / 2, cu + su / 2, cv - sv / 2, cv + sv / 2, max(p["height_mm"]))
    return out


def uv_z(v):
    return -v + Z_CONTACT


def cover_windows():
    """Windows of the cover's web, F (y0, y1, z0, z1): every populated part on the main square grown by WINDOW_MARGIN."""
    out = {}
    for k, (u0, u1, v0, v1, h) in board_parts().items():
        if v1 < -10.10:                           # R1, R2 on the tab, beyond the cover
            continue
        m = WINDOW_MARGIN
        out[k] = (u0 - m, u1 + m, uv_z(v1 + m), uv_z(v0 - m))
    return out


# ================================================================================================= solids, two kernels
# dip_link_parts() and cover_part() build the DIP link and the cover from P with either kernel: BrepB (FreeCAD, inside
# freecadcmd) gives the exact solids of the STEP files and the committed STL files; MeshB (manifold3d) gives the same
# solids as meshes in well under a second, for scripts/dip_link_viewer.py.

class BrepB:
    """FreeCAD Part inside freecadcmd: exact solids, by the operations of the committed concept b."""
    name = "brep"

    def __init__(self):
        import FreeCAD as App
        import Part
        self.App, self.Part = App, Part

    def box(self, lo, hi):
        return self.Part.makeBox(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], self.App.Vector(*lo))

    def cylx(self, y, z, r, x0, x1):
        return self.Part.makeCylinder(r, x1 - x0, self.App.Vector(x0, y, z), self.App.Vector(1, 0, 0))

    def hexx(self, y, z, af, x0, x1):
        V = self.App.Vector
        rc = af / math.sqrt(3.0)
        pts = [V(x0, y + rc * math.cos(math.radians(a)), z + rc * math.sin(math.radians(a))) for a in range(0, 360, 60)]
        return self.Part.Face(self.Part.makePolygon(pts + [pts[0]])).extrude(V(x1 - x0, 0, 0))

    def prism_y(self, points_xz, y0, y1):
        V = self.App.Vector
        pts = [V(x, y0, z) for x, z in points_xz]
        return self.Part.Face(self.Part.makePolygon(pts + [pts[0]])).extrude(V(0, y1 - y0, 0))

    def prism_z(self, points_yx, z0, z1):
        V = self.App.Vector
        pts = [V(x, y, z0) for y, x in points_yx]
        return self.Part.Face(self.Part.makePolygon(pts + [pts[0]])).extrude(V(0, 0, z1 - z0))

    def prism_x(self, points_yz, x0, x1):
        V = self.App.Vector
        pts = [V(x0, y, z) for y, z in points_yz]
        return self.Part.Face(self.Part.makePolygon(pts + [pts[0]])).extrude(V(x1 - x0, 0, 0))

    def cyly(self, x, z, r, y0, y1):
        return self.Part.makeCylinder(r, y1 - y0, self.App.Vector(x, y0, z), self.App.Vector(0, 1, 0))

    def fuse(self, a, others):
        return a.fuse(others)

    def cut(self, a, others):
        return a.cut(others)

    def finish(self, s):
        s = s.removeSplitter()
        return s.Solids[0] if len(s.Solids) == 1 else s

    def mesh_solid(self, path):
        """A faceted solid from a closed mesh; further closed components (the magnet pockets) become voids."""
        import Mesh
        solids = []
        for comp in Mesh.Mesh(str(path)).getSeparateComponents():
            sh = self.Part.Shape()
            sh.makeShapeFromMesh(comp.Topology, 0.005)
            so = self.Part.Solid(self.Part.Shell(sh.Faces))
            if so.Volume < 0:
                so.reverse()
            solids.append(so)
        solids.sort(key=lambda s: -s.Volume)
        out = solids[0].cut(solids[1:]) if len(solids) > 1 else solids[0]
        return out.removeSplitter()

    def arms(self):
        return [self.mesh_solid(WORK / "arm_horn.stl"), self.mesh_solid(WORK / "arm_slot.stl")]


class MeshB:
    """manifold3d: the same solids as meshes (cylinders of 48 segments)."""
    name = "mesh"
    _arms = None

    def box(self, lo, hi):
        return box(lo, hi)

    def cylx(self, y, z, r, x0, x1):
        return cyl_x(y, z, r, x0, x1)

    def hexx(self, y, z, af, x0, x1):
        return hex_x(y, z, af, x0, x1)

    @staticmethod
    def _extrude(poly_uv, w0, w1, T):
        """A polygon in (u, v) extruded along w from w0 to w1, then mapped to F by the 3 x 4 matrix T (an even permutation
        of the axes)."""
        import numpy as np
        m3 = _m3()
        Q = np.asarray(poly_uv, float)
        if np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1]) < 0:
            Q = Q[::-1]
        s = m3.Manifold.extrude(m3.CrossSection([Q]), float(w1 - w0)).translate([0.0, 0.0, float(w0)])
        return s.transform(np.asarray(T, float))

    def prism_y(self, points_xz, y0, y1):
        return self._extrude([(z, x) for x, z in points_xz], y0, y1, [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])

    def prism_z(self, points_yx, z0, z1):
        return self._extrude([(x, y) for y, x in points_yx], z0, z1, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    def prism_x(self, points_yz, x0, x1):
        return self._extrude(list(points_yz), x0, x1, [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]])

    def cyly(self, x, z, r, y0, y1, n=48):
        """Cylinder along y."""
        m3 = _m3()
        return m3.Manifold.cylinder(float(y1 - y0), float(r), float(r), n).rotate([-90.0, 0.0, 0.0]).translate(
            [float(x), float(y0), float(z)])                                       # +z -> +y

    def fuse(self, a, others):
        m3 = _m3()
        return m3.Manifold.batch_boolean([a] + list(others), m3.OpType.Add)

    def cut(self, a, others):
        m3 = _m3()
        return a - m3.Manifold.batch_boolean(list(others), m3.OpType.Add)

    def finish(self, s):
        return s

    def arms(self):
        """The clevis arms of fingertip_mount_DIP_link.stl, cut as cmd_prep cuts them (computed once per process)."""
        if MeshB._arms is None:
            sys.path.insert(0, str(REPO / "scripts"))
            sys.path.insert(0, str(REPO / "src"))
            import dip_link_geometry as G
            M = to_manifold(G.load_part("dip_link"))
            MeshB._arms = [M ^ box((-40, -24.0, -110), (40, Y_BRIDGE[0] - 0.01, -30)),
                           M ^ box((-40, Y_BRIDGE[1] + 0.01, -110), (40, 18.0, -30))]
        return MeshB._arms


_PROFILES = {}


def profile():
    """The bridge's top profile for the current KEEP_MARGIN and X_SPINE_BACK: prep's file when it matches them (the B-rep
    build reads only that), otherwise bridge_profile() (cached per margin and spine)."""
    key = (KEEP_MARGIN, X_SPINE_BACK)
    if key not in _PROFILES:
        f = WORK / "bridge_profile.json"
        d = json.loads(f.read_text()) if f.exists() else {}
        if d.get("margin") == KEEP_MARGIN and d.get("x_spine_back", 9.45) == X_SPINE_BACK:
            _PROFILES[key] = d["profile"]
        else:
            _PROFILES[key] = bridge_profile()
    return _PROFILES[key]


def j1_boxes_b():
    """J1, its plug and the cable's first 3 mm kept on the board (dip_link_concept_a.j1_parts_board, tab proximal) at b's
    placement, the component face at X_COMP and the parts toward the pad: (name, lo, hi) boxes in F. With the routed
    cable (cable_on) the cable's straight 3 mm is left out: cable_segments replaces it."""
    sys.path.insert(0, str(REPO / "scripts"))             # freecadcmd starts without it
    import dip_link_concept_a as A
    out = []
    for name, lo, hi in A.j1_parts_board():
        if name == "tab_fr4" or (name.startswith("wire") and cable_on()):
            continue
        out.append((name, (X_COMP - hi[2], lo[0], Z_CONTACT - hi[1]), (X_COMP - lo[2], hi[0], Z_CONTACT - lo[1])))
    return out


# The Qwiic cable's route (design cable = idler_side; owner, 2026-10-19: "how are you planning to route the cable out
# though?", "i would like to minimize range of motion loss, if any, on the back extension of the DIP joint"). b's plug
# leaves its wires at x_F 2.40, 3.50 mm palmar of concept a's 5.90, and so 0.7 mm closer to the PIP axis at the height of
# the first turn, where the MCP link's end sweeps a cylinder of radius 12.70 about the axis: a's turn to +y right over the
# plug would pass the MCP link at about 0.3 mm at neck_len 9.75. The cable therefore turns dorsally first, out of the
# plug toward the back, then to +y at x cable_run_x, and from there takes concept a's path: through the slotted arm's lower
# dorsal corner, up its outer face (y 17.40, where no part of the MCP link or the PIP servo reaches at any PIP angle) to the
# anchor below the PIP axis. The groove is open to the back, so the cable is laid in with its far plug on.
ARM_OUT = 17.40                 # outer face of the slotted arm (fingertip_mount_DIP_link.stl)
CABLE = dict(cable_d=2.0, cable_r=2.0, cable_lead=0.0, cable_clear=0.2, cable_run_x=6.40, cable_face_y=17.40,
             cable_anchor_x=4.0, cable_anchor_z=-56.5)          # values for a parameter file without the cable keys
X_OPEN = 11.5                   # the groove runs out past the DIP link's back and the slotted arm's dorsal edge
ROOF_MIN = 0.8                  # R11: the thinnest PLA wall, two 0.4 mm lines


def cable_on():
    return DESIGN.get("cable") == "idler_side" and DESIGN.get("qwiic") == "j1_kept"


def cp(key):
    return float(P.get(key, CABLE[key]))


def r_first():
    """Bend radius of the first turn, toward the back out of the plug: cable_r1, or cable_r when the file has none."""
    return float(P["cable_r1"]) if "cable_r1" in P else cp("cable_r")


def cable_segments():
    """Centre line of the cable's bundle from the plug's rear face to the anchor on the slotted arm's outer face, F:
    straight out of the plug toward the PIP joint (cable_lead), a turn toward the back (cable_r1, in the x-z plane), a run
    toward the back if cable_run_x leaves room, a turn to +y (x-y plane), a run along +y at x cable_run_x to the slotted
    arm, a turn to +z into the arm's outer face (y-z plane) and a line up the face to the anchor. Segments ('line', a, b)
    and ('arc', a, b, centre, radius, plane, t0_deg, t1_deg), as dip_link_concept_a.cable_segments."""
    import numpy as np
    R, lead = cp("cable_r"), cp("cable_lead")
    x0 = X_COMP - 1.4                           # the plug's wires leave at mid-height of the housing (concept a's 1.4 mm)
    y0 = 0.05                                   # J1's centre line (spec.json: body u -2.95 .. 3.05)
    z0 = Z_CONTACT + 18.09 + QW["plug_out"]     # the plug's rear face (board v -20.09)
    xr, yf = cp("cable_run_x"), cp("cable_face_y")
    R1 = r_first()
    a1 = np.array([x0, y0, z0 + lead])
    c1 = np.array([x0 + R1, y0, z0 + lead])
    b1 = np.array([x0 + R1, y0, z0 + lead + R1])          # heading +x
    z1 = b1[2]
    b2 = np.array([xr - R, y0, z1])
    c2 = np.array([xr - R, y0 + R, z1])
    b3 = np.array([xr, y0 + R, z1])                       # heading +y
    b4 = np.array([xr, yf - R, z1])
    c4 = np.array([xr, yf - R, z1 + R])
    b5 = np.array([xr, yf, z1 + R])                       # heading +z, on the arm's outer face
    anchor = np.array([cp("cable_anchor_x"), yf, cp("cable_anchor_z")])
    segs = [("line", np.array([x0, y0, z0]), a1), ("arc", a1, b1, c1, R1, "xz", 180.0, 90.0)]
    if b2[0] - b1[0] > 1e-6:
        segs.append(("line", b1, b2))
    return segs + [("arc", b2, b3, c2, R, "xy", -90.0, 0.0), ("line", b3, b4), ("arc", b4, b5, c4, R, "yz", -90.0, 0.0),
                   ("line", b5, anchor)]


def cable_polyline(step=0.2):
    sys.path.insert(0, str(REPO / "scripts"))
    import dip_link_concept_a as A
    return A.polyline(cable_segments(), step)


def cable_mesh(sections=24):
    """The bundle as a closed tube (trimesh) along the centre line."""
    sys.path.insert(0, str(REPO / "scripts"))
    import dip_link_concept_a as A
    return A.tube_mesh(cable_polyline(0.25), cp("cable_d") / 2, sections)


def cable_length():
    import numpy as np
    return float(np.linalg.norm(np.diff(cable_polyline(), axis=0), axis=1).sum())


def cable_open_top_x():
    """x up to which the groove over the plug and along the leg toward the back is open to the top. The bridge's top keeps
    0.8 mm (KEEP_MARGIN) from the volume the MCP link sweeps and the cable about 1.16 mm, so near the plug the bridge's top
    stands less than ROOF_MIN above the groove and a roof there would be a sliver; the profile is the same at every y between
    the arms. A strip of Kapton tape over the opening keeps the cable's first turn in place."""
    segs = cable_segments()
    rc = cp("cable_d") / 2 + cp("cable_clear")
    x0, top = segs[0][1][0], segs[1][2][2] + rc
    thin = [x for x, z in profile() if x >= x0 - rc and z - top < ROOF_MIN]
    return (max(thin) + 0.25) if thin else x0 - rc


def cable_cuts(K):
    """The groove that takes the bundle (clearance cable_clear all round), open toward the back from the bundle's palmar
    side: over the plug and out through the neck's back, across to the slotted arm (the inner wall of the turn to +y
    stays), through the arm's lower dorsal corner, then half the bundle deep up the arm's outer face to the anchor, with
    a round end. A list of solids of kernel K."""
    import numpy as np
    rc = cp("cable_d") / 2 + cp("cable_clear")
    segs = cable_segments()
    _, a0, a1 = segs[0]
    _, _, b1, c1, R, *_ = segs[1]
    _, b2, b3, c2, R2, *_ = segs[-4]
    _, _, b4 = segs[-3]
    _, _, b5, c4, R4, *_ = segs[-2]
    _, _, anchor = segs[-1]
    y0, z1 = a0[1], b1[2]
    # out of the plug and the turn toward the back: the band rc either side of the centre line in (x, z), across y0 +- rc,
    # its leg toward the back run out to X_OPEN
    t = np.radians(np.linspace(180.0, 90.0, 33))
    outer = [(c1[0] + (R + rc) * np.cos(a), c1[2] + (R + rc) * np.sin(a)) for a in t]
    ri = max(R - rc, 0.05)
    inner = [(c1[0] + ri * np.cos(a), c1[2] + ri * np.sin(a)) for a in t]          # from the plug's side to the leg
    band = ([(a0[0] - rc, a0[2] - 0.05)] + outer + [(X_OPEN, z1 + rc), (X_OPEN, z1 - rc)] + inner[::-1]
            + [(a0[0] + rc, a0[2] - 0.05)])
    turn1 = K.prism_y(band, y0 - rc, y0 + rc)
    # open to the top over the band where a roof would be thinner than ROOF_MIN (over the plug and the leg toward the back)
    xt = max(cable_open_top_x(), c1[0] + 0.01)
    top = K.prism_y([(a0[0] - rc, a0[2] - 0.05), (a0[0] + rc, a0[2] - 0.05)] + inner + [(xt, z1 - rc), (xt, -50.0),
                                                                                         (a0[0] - rc, -50.0)], y0 - rc, y0 + rc)
    # the turn to +y and the run to the arm: a section in (x, y) across z1 +- rc; the turn's inner wall stays
    ti = np.radians(np.linspace(0.0, -90.0, 33))
    inner = [(c2[0] + (R2 - rc) * np.cos(a), c2[1] + (R2 - rc) * np.sin(a)) for a in ti]
    run = [(c2[0], y0 - rc), (X_OPEN, y0 - rc), (X_OPEN, b4[1] + 0.01), (b3[0] - rc, b4[1] + 0.01)] + inner
    run = K.prism_z([(y, x) for x, y in run], z1 - rc, z1 + rc)
    # the turn into the arm's outer face: a sector in (y, z) from the bundle's palmar side out to X_OPEN
    t = np.radians(np.linspace(-90.0, 0.0, 33))
    sec = ([(c4[1] + (R4 + rc) * np.cos(a), c4[2] + (R4 + rc) * np.sin(a)) for a in t] +
           [(c4[1] + max(R4 - rc, 0.05) * np.cos(a), c4[2] + max(R4 - rc, 0.05) * np.sin(a)) for a in t[::-1]])
    turn3 = K.prism_x(sec, b4[0] - rc, X_OPEN)
    # up the arm's outer face, half the bundle deep, and the groove's round end at the anchor
    d = np.array([anchor[0] - b5[0], anchor[2] - b5[2]])
    n_ = np.array([-d[1], d[0]]) / np.linalg.norm(d) * rc
    strip = [(b5[0] + n_[0], b5[2] - 0.05 + n_[1]), (anchor[0] + n_[0], anchor[2] + n_[1]),
             (anchor[0] - n_[0], anchor[2] - n_[1]), (b5[0] - n_[0], b5[2] - 0.05 - n_[1])]
    face = K.prism_y(strip, cp("cable_face_y") - rc, ARM_OUT + 1.0)
    end = K.cyly(anchor[0], anchor[2], rc, cp("cable_face_y") - rc, ARM_OUT + 1.0)
    joint = K.cyly(b5[0], b5[2], rc, cp("cable_face_y") - rc, ARM_OUT + 1.0)       # the kink from the turn into the face line
    return [turn1, top, run, turn3, face, end, joint]


def dip_link_parts(K):
    """The DIP link's new body (bridge, tongue, webs and tie block, with the board's seat pocket, the tab's room, the wire
    exit, the tie tunnel, the screw holes and the nut pockets) and the whole DIP link (the body fused with the clevis arms).
    With neck_len > 0 the bridge keeps its underside at the arms' mount plane (z -67.30) and a neck as wide as the cassette
    (|y| <= Y_RIM) joins it to the pad's mount face, neck_len lower. With J1 kept, the board's wires are not cut free, so
    the tongue has no wire exit and no tie block."""
    prof = profile()
    zb = TIP0["z_mount"]
    poly = [(TIP["x_palmar"], zb)] + [tuple(p) for p in prof] + [(X_SPINE_BACK, zb)]
    bridge = K.prism_y(poly, Y_BRIDGE[0] - 0.02, Y_BRIDGE[1] + 0.02)
    tongue = K.box((X_SEAT - RIM_H, -Y_RIM, TIP["z_end"]), (X_TONGUE_BACK, Y_RIM, TIP["z_mount"] + 0.5))
    webs = [K.box((X_SEAT - RIM_H, Y_BRIDGE[0] - 0.02, Z_WEB_LOW), (X_SPINE_BACK, -Y_RIM + 0.01, zb + 0.5)),
            K.box((X_SEAT - RIM_H, Y_RIM - 0.01, Z_WEB_LOW), (X_SPINE_BACK, Y_BRIDGE[1] + 0.02, zb + 0.5))]
    j1 = DESIGN.get("qwiic") == "j1_kept"
    extra = [] if j1 else [K.box((X_TONGUE_BACK - 0.1, TIE["y"][0], TIE["z"][0]), (TIE["x_top"], TIE["y"][1], TIE["z"][1]))]
    if NECK > 0:
        extra.append(K.box((TIP["x_palmar"], -Y_RIM, TIP["z_mount"]), (X_SPINE_BACK, Y_RIM, zb + 0.5)))
    body = K.fuse(bridge, [tongue] + extra + webs)
    cuts = [
        # the board's seat pocket: main square (datum walls at y = -10.00 and at the proximal edge) and the tab
        K.box((X_SEAT - RIM_H - 0.1, -10.00, Z_BOARD[0] - 0.05), (X_SEAT, 10.05, Z_BOARD[1])),
        K.box((X_SEAT - RIM_H - 0.1, -5.05, Z_BOARD[1] - 0.01), (X_SEAT, 5.05, uv_z(TAB_CUT_V) + 0.10)),
        # room for the tab's component side (R1, R2 0.4 mm tall) inside the bridge
        K.box((X_COMP - 0.5, -5.2, COVER_TAB_Z - 0.1), (X_SEAT, 5.2, uv_z(TAB_CUT_V) + 0.2)),
    ]
    if not j1:      # the wire exit under the tab's root, and the tie tunnel
        cuts += [K.box((X_SEAT - 0.01, WIRE_SLOT["y"][0], WIRE_SLOT["z"][0]),
                       (X_TONGUE_BACK + 0.1, WIRE_SLOT["y"][1], WIRE_SLOT["z"][1])),
                 K.box((TIE["tunnel_x"][0] - 0.2, TIE["y"][0] - 0.1, TIE["tunnel_z"][0]),
                       (TIE["tunnel_x"][1], TIE["y"][1] + 0.1, TIE["tunnel_z"][1]))]
    for y, z in screw_yz():
        if DESIGN.get("board_screws") == "self_forming":    # the screw forms its thread in the tongue (owner: M2 into 1.8 mm)
            cuts.append(K.cylx(y, z, P["screw_thread_d"] / 2, X_SEAT - RIM_H - 0.2, X_SPINE_BACK + 0.2))
            continue
        cuts.append(K.cylx(y, z, SCREW_CLEAR / 2, X_SEAT - RIM_H - 0.2, X_SPINE_BACK + 0.2))
        cuts.append(K.hexx(y, z, NUT_POCKET_AF, X_TONGUE_BACK - NUT_POCKET_H, X_SPINE_BACK + 0.2))   # nut in from the back
    if DESIGN.get("qwiic") == "header_on_tongue":           # the Qwiic header's pocket and the pigtail's groove
        cuts += [K.box(lo, hi) for _, lo, hi in qwiic_header_b()[1]]
    if j1:
        # J1, its plug and the cable in a slot from the neck's underside through the bridge's top, and the seat pocket open
        # at the tongue's distal end: the board slides in along +z, J1 first, until the main square meets the proximal wall
        sys.path.insert(0, str(REPO / "scripts"))         # freecadcmd starts without it
        import dip_link_concept_a as A
        B = [list(lo) + list(hi) for _, lo, hi in j1_boxes_b()]
        lo = [min(b[k] for b in B) - A.J1_SLOT_CLEAR[k] for k in range(3)]
        hi = [max(b[k + 3] for b in B) + A.J1_SLOT_CLEAR[k] for k in range(3)]
        cuts.append(K.box((lo[0], lo[1], TIP["z_mount"] - 0.1), (X_COMP + 0.05, hi[1], -46.55)))
        cuts.append(K.box((X_SEAT - RIM_H - 0.1, -10.00, TIP["z_end"] - 0.1), (X_SEAT, 10.05, Z_BOARD[0])))
    groove = cable_cuts(K) if cable_on() else []
    body = K.cut(body, cuts + groove)
    dip = K.fuse(body, K.arms())
    if groove:                                  # the groove crosses the slotted arm's lower dorsal corner and outer face
        dip = K.cut(dip, groove)
    return body, K.finish(dip)


def cover_part(K):
    """The cover: a web as thick as the chips over the board's component face, with windows around every part, a notch
    over the tab, the four screw holes, and the two dovetail rails on its front face."""
    web = K.box((X_SLOT, -Y_RIM, TIP["z_end"]), (X_COMP, Y_RIM, TIP["z_mount"]))
    notch = K.box((X_SLOT - 0.1, -COVER_TAB_Y, COVER_TAB_Z), (X_COMP + 0.1, COVER_TAB_Y, TIP["z_mount"] + 0.1))
    win = [K.box((X_SLOT - 0.1, y0, z0), (X_COMP + 0.1, y1, z1)) for y0, y1, z0, z1 in cover_windows().values()]
    holes = [K.cylx(y, z, SCREW_CLEAR / 2, X_SLOT - 0.2, X_COMP + 0.2) for y, z in screw_yz()]
    cover = K.cut(web, [notch] + win + holes)
    rails = []
    for s in (-1, 1):
        sec = [(s * y, (X_SLOT + 0.1) if x > X_SLOT else x) for y, x in rail_section()]
        rails.append(K.prism_z(sec, RAIL_Z[0], RAIL_Z[1]))
    return K.finish(K.fuse(cover, rails))


def fc_build():
    """Run by freecadcmd (DIPB_FREECAD=1): B-rep solids of the DIP link and the cover, faceted solids of the pads and
    the arms; STEP and STL of each part into CAD_DIR."""
    import MeshPart
    K = BrepB()
    CAD_DIR.mkdir(parents=True, exist_ok=True)
    log = {}

    def export(shape, name, lin=0.02, ang=0.15):
        step = CAD_DIR / f"{PREFIX}_{name}.step"
        stl = CAD_DIR / f"{PREFIX}_{name}.stl"
        shape.exportStep(str(step))
        msh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=lin, AngularDeflection=ang, Relative=False)
        msh.write(str(stl))
        log[name] = dict(step=str(step.relative_to(REPO)), stl=str(stl.relative_to(REPO)), volume_mm3=round(shape.Volume, 2),
                         valid=bool(shape.isValid()), solids=len(shape.Solids), faces=len(shape.Faces),
                         bounds=[round(v, 3) for v in (shape.BoundBox.XMin, shape.BoundBox.YMin, shape.BoundBox.ZMin,
                                                        shape.BoundBox.XMax, shape.BoundBox.YMax, shape.BoundBox.ZMax)])
        print(name, log[name], flush=True)

    t0 = time.time()
    body, dip = dip_link_parts(K)
    print("DIP fused", round(time.time() - t0, 1), "s", flush=True)
    export(dip, "dip_link")
    export(body, "dip_link_new_body")          # the redesigned body alone (B-rep), for editing
    export(cover_part(K), "cover")
    if cable_on():                              # the cable's groove as one solid, to edit or re-cut in CAD
        g = cable_cuts(K)
        export(K.finish(K.fuse(g[0], g[1:])), "cable_groove_cutter")
    for key in TIPS:                            # the pads, faceted, from their meshes
        p = K.mesh_solid(WORK / f"pad_{key}_step.stl")
        step = CAD_DIR / f"{PREFIX}_pad_{key}.step"
        p.exportStep(str(step))
        log[f"pad_{key}"] = dict(step=str(step.relative_to(REPO)), volume_mm3=round(p.Volume, 2), valid=bool(p.isValid()),
                                 faces=len(p.Faces))
        print(f"pad_{key}", log[f"pad_{key}"], round(time.time() - t0, 1), "s", flush=True)
    (WORK / "fc_build.json").write_text(json.dumps(log, indent=1))
    print("done", round(time.time() - t0, 1), "s", flush=True)


def cmd_cad(a):
    env = dict(os.environ, DIPB_FREECAD="1")
    r = subprocess.run(["freecadcmd", str(Path(__file__).resolve())], env=env, cwd=str(REPO), capture_output=True,
                       text=True)
    for line in r.stdout.splitlines():
        if line.strip() and "%" not in line and "saving" not in line:
            print(line[:400])
    if r.returncode != 0 or "done" not in r.stdout:
        print(r.stderr[-4000:])
        raise SystemExit("freecadcmd failed")
    import zipfile
    for f in sorted(CAD_DIR.glob("*.step")):
        if f.stat().st_size > 1.5e6:            # faceted STEP: zipped, so git LFS stores it (docs/**/*.zip)
            z = f.with_suffix(".step.zip")
            with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
                zf.write(f, f.name)
            print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB -> {z.name} {z.stat().st_size / 1e6:.1f} MB")
            f.unlink()


# ================================================================================================= assembly

RHO = dict(pla=1.24, pla_15=0.948, brass=8.5, tpu=1.20)   # g/cm^3; pla_15: the current DIP link's sliced density
MAG_MASS_G = 0.106                                         # one 3 x 2 mm N52 disc (7.5 g/cm^3)


def tip_records():
    """The printed tips of the pad set (spec.json "tips" for the 2.7 mm seats; the 2.9 mm seats' records of the final,
    final/20261018-dip_link_final_tips.json, with the as-printed magnets)."""
    pads = DESIGN.get("pads", "seat2p7")
    if pads == "seat2p7":
        tips = spec()["tips"]
    elif pads == "v2_five":
        tips = {k: v["record"] for k, v in json.loads((STUDY / "v2/20261018-v2_tips.json").read_text())["tips"].items()}
    else:
        tips = json.loads((STUDY / "final/20261018-dip_link_final_tips.json").read_text())
    for t in tips.values():
        for m in t["magnets"]["items"]:
            m["centre_F"] = [m["centre_F"][0], m["centre_F"][1], m["centre_F"][2] - NECK]
    return tips


def magnets(key="block4"):
    """Magnet cylinders of a printed tip of the pad set: [(centre_F, x0, x1)], 3 x 2 mm, axes along x."""
    out = []
    for m in tip_records()[key]["magnets"]["items"]:
        out.append((m["centre_F"], m["x_F"][0], m["x_F"][1]))
    return out


def hardware():
    """Board, magnets and fasteners as manifolds in F: {name: Manifold}."""
    m3 = _m3()
    fr4 = box((X_COMP, -10.0, uv_z(10.05)), (X_SEAT, 10.0, uv_z(-10.10)))
    fr4 = fr4 + box((X_COMP, -5.0, uv_z(-10.10) - 0.01), (X_SEAT, 5.0, uv_z(TAB_CUT_V)))
    fr4 = fr4 - m3.Manifold.batch_boolean([cyl_x(y, z, HOLE_D / 2, X_COMP - 1, X_SEAT + 1) for y, z in screw_yz()],
                                          m3.OpType.Add)
    parts = [box((X_COMP - h, u0, uv_z(v1)), (X_COMP + 0.001, u1, uv_z(v0)))
             for u0, u1, v0, v1, h in board_parts().values()]
    chips = [box((X_COMP - h, u0, uv_z(v1)), (X_COMP + 0.001, u1, uv_z(v0)))
             for k, (u0, u1, v0, v1, h) in board_parts().items() if k.startswith("U")]
    screws = [cyl_x(y, z, HEAD_D / 2, X_SLOT - HEAD_H, X_SLOT) + cyl_x(y, z, SCREW_D / 2, X_SLOT - 0.01, X_SLOT + SCREW_L)
              for y, z in screw_yz()]
    x_nut = X_TONGUE_BACK - NUT_POCKET_H
    nuts = [hex_x(y, z, NUT_AF, x_nut, x_nut + NUT_H) - cyl_x(y, z, SCREW_D / 2, x_nut - 1, x_nut + NUT_H + 1)
            for y, z in screw_yz()]
    add = lambda L: m3.Manifold.batch_boolean(L, m3.OpType.Add)  # noqa: E731
    out = dict(fr4=fr4, board_parts=add(parts), chips=add(chips), screws=add(screws))
    if DESIGN.get("board_screws") != "self_forming":
        out["nuts"] = add(nuts)
    if DESIGN.get("qwiic") == "j1_kept":
        out["j1"] = add([box(lo, hi) for _, lo, hi in j1_boxes_b()])
    for key in tip_records():
        if key in ("block4", "block5", "ell11_5"):
            out[f"magnets_{key}"] = add([cyl_x(c[1], c[2], 1.5, x0, x1) for c, x0, x1 in magnets(key)])
    return out


QW = dict(hdr_w=6.0, hdr_d=4.25, lead=0.7, hdr_h=2.9, plug_w=7.0, plug_out=2.0, plug_h=2.96, wire_d=0.8, wire_run=3.0)
# JST SH catalogue sizes (scripts/dip_link_board_mount.py QW): side-entry header SM04B-SRSS-TB 6.0 wide, 4.25 deep plus
# 0.7 mm of leads, 2.9 tall; housing SHR-04V-S-B (the Qwiic cable's) 7.0 wide at its protrusions, 2.96 tall mated, 2.0 out
# of the header (5.0 wide, SHR-04V-S, until 2026-10-19 17:45); the cable's first 3 mm


def qwiic_header_b():
    """b's Qwiic connection where J1 fits nowhere (concept_b/final/*_qwiic.json): J1 is cut off with the tab; four wires
    from the vias at the tab's root run through the wire exit and down a groove in the tongue's back to a side-entry
    SM04B-SRSS-TB header seated in a pocket of the tongue's back, mating face proximal, so a Qwiic cable plugs in there and
    leaves toward the PIP joint over the tie block, where a cable tie holds it. Returns the connector parts and the cuts in
    the DIP link, as (name, lo, hi) boxes in F."""
    xb = X_TONGUE_BACK                              # the tongue's back face
    x0 = xb - P["qwiic_pocket_depth"]               # pocket floor = the header's base
    y0, z_back, c, g = P["qwiic_y"], P["qwiic_z_back"] - NECK, P["qwiic_pocket_clear"], P["qwiic_groove_depth"]
    gy0, gy1 = P["qwiic_groove_y0"], P["qwiic_groove_y1"]
    z_mate = z_back + QW["hdr_d"]
    z_exit = 0.5 * (WIRE_SLOT["z"][0] + WIRE_SLOT["z"][1])
    parts = [("q_header", (x0, y0 - QW["hdr_w"] / 2, z_back), (x0 + QW["hdr_h"], y0 + QW["hdr_w"] / 2, z_mate)),
             ("q_leads", (x0, y0 - 1.8, z_back - QW["lead"]), (x0 + 0.3, y0 + 1.8, z_back)),
             ("q_plug", (x0, y0 - QW["plug_w"] / 2, z_mate), (x0 + QW["plug_h"], y0 + QW["plug_w"] / 2, z_mate + QW["plug_out"]))]
    for k in range(4):
        y = y0 + (k - 1.5) * 1.0
        zc = z_mate + QW["plug_out"]
        parts.append((f"q_wire{k}", (x0 + 1.4 - 0.4, y - 0.4, zc), (x0 + 1.4 + 0.4, y + 0.4, zc + QW["wire_run"])))
    parts.append(("q_pigtail", (xb - g, gy0 + 0.2, z_back - QW["lead"] - 0.5), (xb, gy1 - 0.2, z_exit)))
    parts.append(("q_pigtail_turn", (xb - g, y0 - 1.8, z_back - QW["lead"] - 0.5), (xb, gy1 - 0.2, z_back - QW["lead"])))
    cuts = [("pocket_body", (x0, y0 - QW["hdr_w"] / 2 - c, z_back), (xb + 0.1, y0 + QW["hdr_w"] / 2 + c, z_mate + c)),
            ("pocket_leads", (x0, y0 - 1.8 - c, z_back - QW["lead"] - 0.1), (xb + 0.1, y0 + 1.8 + c, z_back + 0.01)),
            ("groove_run", (xb - g, gy0, z_back - QW["lead"] - g), (xb + 0.1, gy1, WIRE_SLOT["z"][1])),
            ("groove_turn", (xb - g, y0 - 1.8 - c, z_back - QW["lead"] - g), (xb + 0.1, gy1, z_back - QW["lead"] + 0.01))]
    return parts, cuts


def qwiic_mesh():
    """The header, its leads, the mated plug, the cable's first 3 mm and the pigtail as one trimesh (F)."""
    import trimesh
    return trimesh.util.concatenate([to_trimesh(box(lo, hi)) for _, lo, hi in qwiic_header_b()[0]])


def load_cad(name):
    import trimesh
    return trimesh.load(CAD_DIR / f"{PREFIX}_{name}.stl", force="mesh")


def assembly_meshes(tip="block4"):
    """{name: trimesh} of the assembly at PIP 0, F."""
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import dip_link_geometry as G
    hw = hardware()
    out = {"dip_link": load_cad("dip_link"), "cover": load_cad("cover"), f"pad_{tip}": load_cad(f"pad_{tip}")}
    for k in ("fr4", "board_parts", "screws", "nuts"):
        if k in hw:
            out[k] = to_trimesh(hw[k])
    out["magnets"] = to_trimesh(hw[f"magnets_{tip if f'magnets_{tip}' in hw else 'block4'}"])
    if DESIGN.get("qwiic") == "header_on_tongue":
        out["qwiic"] = qwiic_mesh()
    if DESIGN.get("qwiic") == "j1_kept":
        out["qwiic"] = to_trimesh(hw["j1"])
    if cable_on():
        out["cable"] = cable_mesh()
    out["mcp_link"] = G.load_part("mcp_link")
    out["servo_pip"] = G.load_part("servo_pip")
    return out


def sample_part(mesh, spacing=0.35, pitch=0.5):
    """Surface samples and interior voxel centres of a part (the inputs of points_clearance_exact). The surface is
    subdivided until no edge exceeds `spacing`, so a long thin CAD triangle gets as many samples as its area needs; its
    vertices and edge midpoints lie within spacing / 2 of every surface point."""
    import numpy as np
    import trimesh
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    from morphohand.geometry import cad_features as cf
    V, F = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=spacing, max_iter=40)
    E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    E = np.unique(E, axis=0)
    P = np.vstack([V, 0.5 * (V[E[:, 0]] + V[E[:, 1]])])
    try:
        V = cf.voxelize_solid(mesh, pitch).centres()
        P = np.vstack([P, V])
    except Exception as e:                                       # noqa: BLE001
        print("  voxel centres skipped:", e, flush=True)
    return P


def clearance_exact(points, reach=1.6, step=0.5, workers=1, parts=("mcp_link", "servo_pip")):
    """dip_link_geometry.points_clearance_exact with the k-d tree's search cut at `reach` (scipy distance_upper_bound),
    which leaves the result unchanged when it lies under reach and makes the sweep tractable for parts with 1e5-1e6 surface
    samples. Least signed distance (mm) of points fixed to the DIP link to the MCP link and the PIP servo over PIP -18..92
    deg; returns (value, point index, q_deg, part); (lower bound, index, None, None) when every point stays beyond reach."""
    import numpy as np
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import dip_link_geometry as G
    parts = tuple(parts)
    P, idx = G.parent_samples(parts)
    X0 = np.asarray(points, float)
    ry = lambda A: np.column_stack([np.hypot(A[:, 0] - G.PIP_C[0], A[:, 2] - G.PIP_C[2]), A[:, 1]])  # noqa: E731
    d_lb, _ = cKDTree(ry(P)).query(ry(X0), k=1, workers=workers)
    if parts == ("mcp_link", "servo_pip"):
        K = G.keepout()
        d_k = K.query(X0)
        d_lb = np.maximum(d_lb, np.where(np.isfinite(d_k), d_k - 0.3, d_lb))
    near = np.flatnonzero(d_lb < reach)
    if not len(near):
        return (float(d_lb.min()), int(np.argmin(d_lb)), None, None)
    X = X0[near]
    tree = cKDTree(P)
    sol = G.parent_solids(parts)
    best = (np.inf, None, None, None)
    for q in G.q_grid(G.PIP_RANGE_DEG, step):
        Xp = G.rot_pip(X, -q)
        ins = np.zeros(len(Xp), bool)
        for s_ in sol:
            ins |= s_.contains(Xp)
        d, j = tree.query(Xp, k=1, distance_upper_bound=reach, workers=workers)
        if ins.any():                                   # inside a part: the depth needs the unbounded distance
            d[ins], j[ins] = tree.query(Xp[ins], k=1, workers=workers)
        ok = np.isfinite(d)
        if not ok.any():
            continue
        sd = np.where(ins, -d, d)
        sd[~ok] = np.inf
        i = int(np.argmin(sd))
        if sd[i] < best[0]:
            best = (float(sd[i]), int(near[i]), float(q), parts[int(idx[j[i]])])
    if not np.isfinite(best[0]):
        return (float(reach), None, None, None)
    return best


def cmd_clearance(a):
    """Least clearance of each new part fixed to the DIP link to the MCP link and the PIP servo over PIP -18..92 deg
    (dip_link_geometry.points_clearance_exact: their surfaces at 0.25 mm, PIP every 0.5 deg, a k-d tree) ->
    WORK/clearance.json."""
    import numpy as np
    import trimesh
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import dip_link_geometry as G
    hw = hardware()
    wires = box((X_SEAT, WIRE_SLOT["y"][0] + 0.2, WIRE_SLOT["z"][0] + 0.1),
                (X_TONGUE_BACK + 0.6, WIRE_SLOT["y"][1] - 0.2, WIRE_SLOT["z"][1] - 0.1))   # the four wires in the exit
    pad0 = next(iter(TIPS))
    parts = {"dip_link_new_body": load_cad("dip_link_new_body"), "cover": load_cad("cover"),
             "board_with_tab": to_trimesh(hw["fr4"] + hw["board_parts"]), "screws": to_trimesh(hw["screws"]),
             "nuts": to_trimesh(hw["nuts"]) if "nuts" in hw else None, "wires_at_exit": to_trimesh(wires),
             f"pad_{pad0}": load_cad(f"pad_{pad0}"),
             "arms_kept": trimesh.util.concatenate([trimesh.load(WORK / "arm_horn.stl"), trimesh.load(WORK / "arm_slot.stl")])}
    if DESIGN.get("qwiic") == "header_on_tongue":
        parts["qwiic_connector"] = qwiic_mesh()
    if DESIGN.get("qwiic") == "j1_kept":
        parts["qwiic_connector"] = to_trimesh(hw["j1"])
        parts.pop("wires_at_exit")
    if cable_on():                                           # R7 asks 1.0 mm for the cable
        parts["qwiic_cable"] = cable_mesh(sections=48)
    parts = {k: v for k, v in parts.items() if v is not None}
    out = {}
    if (WORK / "clearance.json").exists() and not a.force:
        out = json.loads((WORK / "clearance.json").read_text())
    for name, mesh in parts.items():
        if name in out:
            print(name, out[name], flush=True)
            continue
        t0 = time.time()
        P = sample_part(mesh)
        # only points whose lower bound (radius and y about the PIP axis, and the swept keep-out) lies under `reach` are
        # turned through the range; the minimum is exact when it lies under reach, a lower bound otherwise
        reach = 4.0 if name == "qwiic_cable" else 1.6
        v, i, q, part = clearance_exact(P, reach=reach, workers=2)
        out[name] = dict(min_mm=round(v, 3), exact=bool(q is not None), at_F=np.round(P[i], 2).tolist() if i is not None else None,
                         pip_deg=q, part=part, n_points=int(len(P)), reach_mm=reach, s=round(time.time() - t0, 1))
        print(name, out[name], flush=True)
        (WORK / "clearance.json").write_text(json.dumps(out, indent=1))
    if "board_to_servo" not in out or a.force:               # R10: the board no closer to the PIP servo than 6.3 mm
        t0 = time.time()
        P = sample_part(parts["board_with_tab"], spacing=0.5)
        v, i, q, part = clearance_exact(P, reach=9.0, workers=2, parts=("servo_pip",))
        out["board_to_servo"] = dict(min_mm=round(v, 3), exact=bool(q is not None), at_F=np.round(P[i], 2).tolist() if i is not None
                                     else None, pip_deg=q, part="servo_pip", n_points=int(len(P)), reach_mm=9.0,
                                     s=round(time.time() - t0, 1))
        print("board_to_servo", out["board_to_servo"], flush=True)
        (WORK / "clearance.json").write_text(json.dumps(out, indent=1))
    if "chips_to_servo" not in out or a.force:               # the five MLX90393 packages alone
        t0 = time.time()
        P = sample_part(to_trimesh(hw["chips"]), spacing=0.3)
        v, i, q, part = clearance_exact(P, reach=12.0, workers=2, parts=("servo_pip",))
        out["chips_to_servo"] = dict(min_mm=round(v, 3), exact=bool(q is not None), at_F=np.round(P[i], 2).tolist() if i is not None
                                     else None, pip_deg=q, part="servo_pip", n_points=int(len(P)), reach_mm=12.0,
                                     s=round(time.time() - t0, 1))
        print("chips_to_servo", out["chips_to_servo"], flush=True)
        (WORK / "clearance.json").write_text(json.dumps(out, indent=1))


# ================================================================================================= sensing

def fe_rows():
    return [json.loads(l) for l in open(FE_ROWS)]


def gain_setting(bmax_mT, headroom=1.25):
    """Highest GAIN_SEL whose range covers headroom x the largest field component, at RES 0, else RES 1."""
    sys.path.insert(0, str(REPO / "scripts"))
    import mlx90393_model as MLX
    for res in (0, 1):
        for g in range(7, -1, -1):
            rxy, rz = MLX.range_mT(g, res)
            rng = min(rxy, rz, 4.8 * MLX.SENS[g][res][0] / MLX.SENS[7][0][0])
            if rng >= headroom * bmax_mT:
                return g, res, rng
    return None, None, None


def sensing_point(row, gap, noise=None):
    """Field at the chips, force bounds and the false force of a 1 um board shift, for one FE row and sensing plane
    (gap: mm behind the whole tip's back face x_F 6.45, negative inside it; tpu_tip_magnet_sensing's convention)."""
    import numpy as np
    sys.path.insert(0, str(REPO / "scripts"))
    import mlx90393_model as MLX
    import tpu_tip_design as TD
    import tpu_tip_magnet_sensing as MS
    ch = MS.chips(row, "board", gap)
    B = MS.B0(row, ch) * 1e3
    bmax = float(np.abs(B).max())
    g, res, rng = gain_setting(bmax)
    saved = MS.NOISE
    if noise == "gain" and g is not None:
        nxy, nz = MLX.noise_uT(g, res, 0, 5)
        MS.NOISE = np.array([nz, nxy, nxy])
    try:
        a = MS.analyse(row, "board", gap)
        ys, zs, Gp = MS.patch_grid(row)
        zc = min(zs, key=lambda z: abs(z - TD.Z_C))
        i0, j0 = ys.index(0.0), zs.index(zc)
        Sg = {k: np.array([[MS.signal(row, Gp[(y, z)][k], ch) for z in zs] for y in ys]) for k in ("normal", "shear_y", "shear_z")}
        J = np.concatenate([np.stack([Sg["normal"][i0, j0], Sg["shear_y"][i0, j0], Sg["shear_z"][i0, j0],
                                      MS.deriv(Sg["normal"][:, j0], ys, i0), MS.deriv(Sg["normal"][i0, :], zs, j0)], 1),
                            np.tile(np.eye(3), (len(ch), 1))], 1)
        sig = np.tile(MS.NOISE, len(ch))
        noise_used = [float(MS.NOISE[1]), float(MS.NOISE[0])]
    finally:
        MS.NOISE = saved
    feq = {}
    for k, ax in enumerate("xyz"):          # tip frame x_t = -x_F: a shift toward the magnets is -x_F = +x_t
        e = np.zeros(3)
        e[k] = 1e-6
        dB = (MS.B0(row, [p + e for p in ch]) - MS.B0(row, [p - e for p in ch])).ravel() / 2 * 1e6
        est = np.linalg.lstsq(J / sig[:, None], dB / sig, rcond=None)[0]
        feq[ax] = dict(normal=round(abs(float(est[0])), 4), shear=round(float(np.hypot(est[1], est[2])), 4))
    both = [p["sigF_n_unk_bg"] for p in a["points"]]
    tr = [float(np.linalg.norm(t)) for t, _ in row["per_force"]["1"]["magnets"]]
    return dict(gap_mm=round(gap, 3), B_chips_mT=np.round(B, 2).tolist(), B_U1_mT=round(float(np.linalg.norm(B[0])), 2),
                B_max_component_mT=round(bmax, 2), gain_sel=g, res=res, range_mT=None if rng is None else round(rng, 1),
                noise_uT_xy_z=[round(v, 3) for v in noise_used],
                centre_force_N=round(a["centre"]["sigF_n_unk_bg"], 4), median_force_N=round(float(np.median(both)), 4),
                range_force_N=[round(min(both), 4), round(max(both), 4)], centre_sig_y_mm=round(a["centre"]["sig_y_mm_bg"], 3),
                centre_sig_z_mm=round(a["centre"]["sig_z_mm_bg"], 3), magnet_travel_um_per_N=round(float(np.mean(tr)), 2),
                approach_1N_um=round(row["per_force"]["1"]["delta_um"], 1), E_star_MPa=round(row["E_best_MPa"], 2),
                false_force_N_per_um={"x (toward the magnets)" if k == "x" else k: v for k, v in feq.items()})


def as_printed_five(r0):
    """An FE row of the five-magnet block with its corner magnets moved to the 2.9 mm print set's spacing (7.4 mm along the
    finger instead of 7.0, layout five_z7p4) and their motions per newton kept, as scripts/dip_link_board_mount.py
    cmd_sense does for concept a."""
    r = json.loads(json.dumps(r0))
    z5 = sorted({round(TIP["z_mount"] - m["centre_F"][2], 2) for m in tip_records()["block5"]["magnets"]["items"]})
    for mg in r["magnets"]:
        if abs(mg[3] - 7.36) < 0.01:
            mg[3] = min(z5)
        elif abs(mg[3] - 14.36) < 0.01:
            mg[3] = max(z5)
    r["as_printed"] = "five_z7p4"
    return r


def sensing_table():
    """Concept b at every FE depth and the whole block at the touch and plate placements (coarse grid), with the
    chips' datasheet noise at GAIN_SEL 7 (the spec's convention) and with the noise at the gain the field needs. With the
    pads of the 2.9 mm print set, the five-magnet rows carry the printed corner spacing (as_printed_five)."""
    rows = fe_rows()
    out = []
    for r in sorted(rows, key=lambda r: (r["layout"], -r["t_behind_mm"])):
        if DESIGN.get("pads") in ("seat2p9", "v2_five") and r["layout"].startswith("five"):
            r = as_printed_five(r)
        tb = r["t_behind_mm"]
        x_slot = MAG_BACK_BLOCK + tb
        cases = [("concept_b" if tb < 10 else "touch", x_slot + SPOT - TIP["x_back"])]
        if tb == 10:
            cases.append(("plate", 3.42))
        for place, gap in cases:
            rec = dict(layout=r["layout"], magnets=len(r["magnets"]), t_behind_mm=tb, x_slot_F=round(x_slot, 2),
                       pad_depth_mm=round(x_slot - TIP["x_palmar"], 2), placement=place, as_printed=r.get("as_printed"))
            rec["gain7_noise"] = sensing_point(r, gap)
            rec["gain_noise"] = sensing_point(r, gap, noise="gain")
            out.append(rec)
            print(r["layout"], tb, place, rec["gain7_noise"]["centre_force_N"], rec["gain_noise"]["centre_force_N"],
                  rec["gain_noise"]["gain_sel"], rec["gain_noise"]["res"], flush=True)
    return out


def static_fields():
    """Static field of each pad's magnets at the five chips (point dipoles, N52 1.45 T, readout frame: x toward the palmar
    face), the gain it needs (gain_setting) and the distances from each chip's sensing point to the nearest magnet's centre
    and surface, at this design's sensing plane (X_SLOT + SPOT), for the tips of the pad set."""
    import numpy as np
    sys.path.insert(0, str(REPO / "scripts"))
    import tpu_tip_design as TD
    import tpu_tip_magnet_sensing as MS
    gap = X_SLOT + SPOT - TIP["x_back"]
    chip_uv = [(0, 0), (0, 7), (7, 0), (0, -7), (-7, 0.051)]
    out = {}
    for key, t in tip_records().items():
        mg = t["magnets"]
        Dm, tm_ = mg["size_mm"]
        mom = np.array([MS.BR * math.pi * (Dm / 2e3) ** 2 * (tm_ / 1e3) / (4e-7 * math.pi), 0.0, 0.0])
        mags_t = [np.array([-m["centre_F"][0] - 2.05, m["centre_F"][1], TIP["z_mount"] - m["centre_F"][2]]) * 1e-3
                  for m in mg["items"]]
        chips_t = [np.array([TD.X_BACK - gap, u, TD.Z_C + v]) * 1e-3 for u, v in chip_uv]
        B = np.array([sum(MS.dipole(c, pm, mom) for pm in mags_t) for c in chips_t]) * 1e3
        bmax = float(np.abs(B).max())
        g, res, rng = gain_setting(bmax)
        dc, ds = [], []
        for u, v in chip_uv:
            c = np.array([X_SLOT + SPOT, u, Z_CONTACT - v])
            dc.append(min(float(np.linalg.norm(c - np.array(m["centre_F"]))) for m in mg["items"]))
            ds.append(min(math.hypot(max(m["x_F"][0] - c[0], 0.0, c[0] - m["x_F"][1]),
                                     max(math.hypot(c[1] - m["centre_F"][1], c[2] - m["centre_F"][2]) - Dm / 2, 0.0))
                          for m in mg["items"]))
        out[key] = dict(gap_from_tip_back_mm=round(gap, 3), B_chips_mT=np.round(B, 3).tolist(),
                        B_U1_mT=round(float(np.linalg.norm(B[0])), 3), B_max_component_mT=round(bmax, 3), gain_sel=g, res=res,
                        chip_to_magnet_centre_mm=[round(v, 2) for v in dc], chip_to_magnet_surface_mm=[round(v, 2) for v in ds],
                        magnets_top_depth_mm=mg["top_depth_mm"], magnets=len(mg["items"]))
        print(f"static {key:7s}: U1 {out[key]['B_U1_mT']:.2f} mT, max component {bmax:.2f} mT (GAIN_SEL {g}, RES {res}), "
              f"chip to magnet surface {min(ds):.2f} mm", flush=True)
    return out


def chip_magnet_distances(key):
    """Distance (mm) from each chip's sensing spot to the nearest magnet's centre and to its surface."""
    import numpy as np
    pts = {k: (X_SLOT + SPOT, *uv_to_yz(*c)) for k, c in
           {"U1": (0, 0), "U2": (0, 7), "U3": (7, 0), "U4": (0, -7), "U5": (-7, 0.051)}.items()}
    out = {}
    for k, p in pts.items():
        best = None
        for c, x0, x1 in magnets(key):
            d = math.dist(p, c)
            # to the cylinder's surface (radius 1.5, x0..x1)
            dx = max(x0 - p[0], 0, p[0] - x1)
            dr = max(math.hypot(p[1] - c[1], p[2] - c[2]) - 1.5, 0)
            ds = math.hypot(dx, dr)
            if best is None or d < best[0]:
                best = (d, ds)
        out[k] = dict(centre_mm=round(best[0], 2), surface_mm=round(best[1], 2))
    return out


# ================================================================================================= renders

COLORS = dict(dip_link="#c9c9c4", cover="#3b6fc4", fr4="#2f7d4f", board_parts="#202020", pad="#d9c08c",
              magnets="#8a8f99", screws="#c8a24a", nuts="#b08a30", mcp_link="#9fb6cf", servo_pip="#4a4a4a",
              qwiic="#f0efe6", cable="#2f5bd8")
LABEL = dict(dip_link="DIP link (PLA)", cover="Cover (PLA)", fr4="ReSkin 5X board", board_parts="MLX90393 chips, 0402 parts",
             pad="85A pad, block", magnets="Magnets 3 x 2 mm", screws="Brass M1.6 x 5 screws", nuts="Brass M1.6 nuts",
             mcp_link="MCP link (PIP 0 deg)", servo_pip="PIP servo", qwiic="Qwiic connector",
             cable="Qwiic cable, 2.0 mm bundle")


def _pv(mesh):
    import numpy as np
    import pyvista as pv
    f = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, float), f)


def _plotter(size=(1600, 1250)):
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
    p.set_background("white")
    try:
        p.enable_anti_aliasing("ssaa")
    except Exception:                                  # noqa: BLE001
        pass
    return p


def _legend_png(path, keys, title=None):
    """A legend strip appended under a render (matplotlib), so its text is at least 13 px at the displayed size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image
    img = Image.open(path)
    w, h = img.size
    dpi = 100
    labels = [_labels()[k] for k in keys]
    ncol = 3 if max(len(t) for t in labels) <= 26 else 2           # long labels in two columns, so none is cut at the edge
    fig = plt.figure(figsize=(w / dpi, 0.5 * (1 + (len(keys) - 1) // ncol) + 0.35), dpi=dpi)
    fig.legend([Patch(facecolor=COLORS[k], edgecolor="#333333") for k in keys], labels, loc="center",
               ncol=ncol, frameon=False, fontsize=24)
    fig.savefig(str(path) + ".legend.png", dpi=dpi)
    plt.close(fig)
    leg = Image.open(str(path) + ".legend.png").convert("RGB")
    out = Image.new("RGB", (w, h + leg.size[1] + (60 if title else 0)), "white")
    y0 = 0
    if title:
        tf = plt.figure(figsize=(w / dpi, 0.6), dpi=dpi)
        tf.text(0.5, 0.5, title, ha="center", va="center", fontsize=27, family="serif")
        tf.savefig(str(path) + ".title.png", dpi=dpi)
        plt.close(tf)
        t = Image.open(str(path) + ".title.png").convert("RGB")
        out.paste(t.resize((w, 60)), (0, 0))
        y0 = 60
        Path(str(path) + ".title.png").unlink()
    out.paste(img.convert("RGB"), (0, y0))
    out.paste(leg.resize((w, leg.size[1])), (0, y0 + h))
    out.save(path)
    Path(str(path) + ".legend.png").unlink()


def render_iso(M, path, view="palmar"):
    import numpy as np
    p = _plotter()
    for k, m in M.items():
        kk = "pad" if k.startswith("pad") else k
        op = 0.28 if kk == "mcp_link" else (0.55 if kk == "pad" else 1.0)
        p.add_mesh(_pv(m), color=COLORS[kk], opacity=op, smooth_shading=True, split_sharp_edges=True, specular=0.2)
    c = np.array([0.0, -2.0, Z_CONTACT + 7.16 + 0.5 * NECK])      # the longer link is framed whole
    off = np.array([-62.0, -70.0, 30.0]) if view == "palmar" else np.array([62.0, -70.0, 34.0])
    p.camera.position = tuple(c + off)
    p.camera.focal_point = tuple(c)
    p.camera.up = (0, 0, 1)
    p.camera.zoom(1.05 * 34.0 / (34.0 + NECK))
    p.screenshot(str(path))
    p.close()
    keys = [("pad" if k.startswith("pad") else k) for k in M]
    _legend_png(path, keys, f"{_title()} at PIP 0 deg, " + ("palmar side" if view == "palmar" else "dorsal side"))


def _title():
    if DESIGN.get("pads") == "v2_five":
        return "Concept b, v2"
    return "Concept b, final" if DESIGN.get("qwiic", "none") != "none" else "Concept b"


def _labels():
    """LABEL with the parts of the design in use (v2: M2 screws without nuts, J1 kept, the 11/22 pad)."""
    out = dict(LABEL)
    if DESIGN.get("board_screws") == "self_forming":
        out["screws"] = f"M{SCREW_D:g} x {SCREW_L:g} screws"
    if DESIGN.get("qwiic") == "j1_kept":
        out["qwiic"] = "J1 and its plug" if cable_on() else "J1, plug and cable start"
    if cable_on():
        out["cable"] = f"Qwiic cable, {cp('cable_d'):g} mm bundle"
    if DESIGN.get("pads") == "v2_five":
        out["pad"] = "85A pad, 11/22 half-ellipsoid"
        out["magnets"] = "Magnets 3 x 2 mm, five"
    return out


def render_qwiic(M, path):
    """The Qwiic header, its plug and the pigtail on the tongue's back, close up from the dorsal side and distal."""
    import numpy as np
    p = _plotter((1600, 1150))
    for k, m in M.items():
        kk = "pad" if k.startswith("pad") else k
        if kk in ("mcp_link",):
            continue
        op = 0.55 if kk == "pad" else 1.0
        p.add_mesh(_pv(m), color=COLORS[kk], opacity=op, smooth_shading=True, split_sharp_edges=True, specular=0.2)
    parts = {n: (np.array(lo), np.array(hi)) for n, lo, hi in qwiic_header_b()[0]}
    lo, hi = parts["q_header"]
    c = np.array([(lo[0] + hi[0]) / 2, 1.0, -79.0])
    p.camera.position = tuple(c + np.array([52.0, -34.0, 22.0]))
    p.camera.focal_point = tuple(c)
    p.camera.up = (0.0, 0.0, 1.0)
    p.camera.zoom(1.45)
    p.screenshot(str(path))
    p.close()
    keys = [k for k in ("qwiic", "dip_link", "nuts", "fr4", "cover", "pad", "servo_pip") if k in
            [("pad" if n.startswith("pad") else n) for n in M]]
    _legend_png(path, keys, "Qwiic header in the tongue's back, plug mated, pigtail in its groove (PIP 0 deg)")


def render_cable(M, path):
    """The Qwiic cable's route seen from the back and the slotted arm's side: out of the plug toward the back, along the
    groove to the arm and up its outer face to the anchor; the MCP link and the PIP servo at PIP 0 see-through."""
    import numpy as np
    p = _plotter((1600, 1250))
    for k, m in M.items():
        kk = "pad" if k.startswith("pad") else k
        op = {"mcp_link": 0.20, "servo_pip": 0.30, "pad": 0.55}.get(kk, 1.0)
        p.add_mesh(_pv(m), color=COLORS[kk], opacity=op, smooth_shading=True, split_sharp_edges=True, specular=0.2)
    c = np.array([5.0, 8.0, -63.5])
    p.camera.position = tuple(c + np.array([46.0, 46.0, 14.0]))
    p.camera.focal_point = tuple(c)
    p.camera.up = (0, 0, 1)
    p.camera.zoom(1.7)
    p.screenshot(str(path))
    p.close()
    keys = [k for k in ("cable", "qwiic", "dip_link", "fr4", "cover", "pad", "mcp_link", "servo_pip")
            if k in [("pad" if n.startswith("pad") else n) for n in M]]
    _legend_png(path, keys, f"{_title()}: the Qwiic cable's route to the slotted arm, PIP 0 deg")


def render_exploded(M, path):
    import numpy as np
    shift = dict(dip_link=0.0, nuts=10.0, fr4=-6.0, board_parts=-6.0, cover=-12.0, screws=-19.0, pad=-31.0, magnets=-31.0,
                 qwiic=-6.0 if DESIGN.get("qwiic") == "j1_kept" else 0.0, cable=0.0)      # J1 kept: on the board; the
    #                                                                                      cable stays in the link's groove
    p = _plotter((1700, 1150))
    for k, m in M.items():
        kk = "pad" if k.startswith("pad") else k
        if kk in ("mcp_link", "servo_pip"):
            continue
        mm = m.copy()
        mm.apply_translation([shift[kk], 0.0, 0.0])
        p.add_mesh(_pv(mm), color=COLORS[kk], opacity=0.6 if kk == "pad" else 1.0, smooth_shading=True,
                   split_sharp_edges=True, specular=0.2)
    c = np.array([-10.0, -2.0, Z_CONTACT + 4.16 + 0.5 * NECK])
    p.camera.position = tuple(c + np.array([-30.0, -95.0, 36.0]))
    p.camera.focal_point = tuple(c)
    p.camera.up = (0, 0, 1)
    p.camera.zoom(1.15 * 34.0 / (34.0 + NECK))
    p.screenshot(str(path))
    p.close()
    keys = [k for k in ("pad", "magnets", "screws", "cover", "board_parts", "fr4", "dip_link", "nuts", "qwiic", "cable")
            if k in [("pad" if n.startswith("pad") else n) for n in M]]
    _legend_png(path, keys, f"{_title()} exploded along x")


def section_paths(sec, i, j):
    """matplotlib Paths of a trimesh section, each closed region with its holes (even-odd fill), in axes (i, j)."""
    import numpy as np
    from matplotlib.path import Path as MPath
    out = []
    try:
        planar, T = sec.to_2D() if hasattr(sec, "to_2D") else sec.to_planar()
        polys = planar.polygons_full
        for poly in polys:
            rings = [np.asarray(poly.exterior.coords)] + [np.asarray(r.coords) for r in poly.interiors]
            verts, codes = [], []
            for r in rings:
                R3 = np.column_stack([r, np.zeros(len(r)), np.ones(len(r))]) @ T.T
                verts.append(R3[:, [i, j]])
                codes += [MPath.MOVETO] + [MPath.LINETO] * (len(r) - 2) + [MPath.CLOSEPOLY]
            out.append(MPath(np.vstack(verts), codes))
    except Exception:                                  # noqa: BLE001
        for poly in sec.discrete:
            out.append(MPath(poly[:, [i, j]], closed=True))
    return out


def render_sections(M, path, spec_rows):
    """Two cuts through the chips: the plane y = 0 (U1, U2, U4, the tab and the bridge) and the plane z = -78.16 mm
    (U1, U3, U5 and the rails), each part filled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import PathPatch, Rectangle
    fig, axs = plt.subplots(1, 2, figsize=(17.5, 10.2), gridspec_kw=dict(width_ratios=[1.05, 1.0]))
    order = ["mcp_link", "servo_pip", "dip_link", "cover", "fr4", "board_parts", "screws", "nuts", "pad", "magnets", "qwiic",
             "cable"]
    planes = [("y = 0 mm (U1, U2, U4)", [0, 0.0, 0], [0, 1, 0], (0, 2), (-14, 13), (TIP["z_end"] - 2.29, -56)),
              (f"z = {Z_CONTACT:.2f} mm (U1, U3, U5)", [0, 0, Z_CONTACT], [0, 0, 1], (1, 0), (-12.5, 12.5), (-12.5, 11))]
    for ax, (title, o, n, (i, j), xl, yl) in zip(axs, planes):
        for k in order:
            kk = "pad" if k.startswith("pad") else k
            mk = [m for name, m in M.items() if (name.startswith("pad") if kk == "pad" else name == k)]
            if not mk:
                continue
            s = mk[0].section(plane_origin=o, plane_normal=n)
            if s is None:
                continue
            for P in section_paths(s, i, j):
                ax.add_patch(PathPatch(P, facecolor=COLORS[kk], edgecolor="#222222", lw=0.6,
                                       alpha=0.35 if kk == "mcp_link" else 0.95, zorder=2))
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=19, family="serif")
        ax.tick_params(labelsize=16)
        ax.grid(True, lw=0.3, color="#bbbbbb", zorder=0)
    tip = next(name[4:] for name in M if name.startswith("pad_"))
    mags = magnets(tip)
    ax = axs[0]
    ax.set_xlabel("x in F (mm), dorsal to the right", fontsize=17, family="serif")
    ax.set_ylabel("z in F (mm)", fontsize=17, family="serif")
    for c, x0, x1 in mags:                       # magnets off this plane: projected outlines
        ax.add_patch(Rectangle((x0, c[2] - 1.5), x1 - x0, 3.0, fill=False, ls="--", lw=1.2, ec="#555b66", zorder=5))
    ys = sorted({abs(round(c[1], 1)) for c, _, _ in mags} - {0.0})
    ax.text(-9.9, Z_CONTACT + 6.96, "magnets at\ny = \u00b1" + ", ".join(f"{y:g}" for y in ys) + " mm\n(projected)", fontsize=15,
            family="serif", color="#3c4250")
    xs = X_SLOT + SPOT
    ax.axvline(xs, color="#c03030", lw=1.1, ls="--", zorder=5)
    ax.annotate(f"sensing plane\nx {xs:.2f} mm, 0.42 mm\nbehind the pad", xy=(xs, Z_CONTACT - 8.34), xytext=(6.6, Z_CONTACT - 10.44),
                fontsize=15, family="serif", color="#902020", arrowprops=dict(arrowstyle="->", color="#902020", lw=0.9))
    ax.axvline(TIP["x_back"], color="#777777", lw=0.9, ls=":", zorder=5)
    ax.text(TIP["x_back"] + 0.25, Z_CONTACT - 0.84, "old back\nface x 6.45", fontsize=15, family="serif", color="#555555")
    ax = axs[1]
    ax.set_xlabel("y in F (mm)", fontsize=17, family="serif")
    ax.set_ylabel("x in F (mm), dorsal up", fontsize=17, family="serif")
    for c, x0, x1 in mags:
        ax.add_patch(Rectangle((c[1] - 1.5, x0), 3.0, x1 - x0, fill=False, ls="--", lw=1.2, ec="#555b66", zorder=5))
    ax.text(-1.6, -7.6, "magnets\n(projected)", fontsize=15, family="serif", color="#3c4250", ha="center")
    ax.axhline(xs, color="#c03030", lw=1.1, ls="--", zorder=5)
    ax.text(-12.2, xs + 0.3, "sensing plane", fontsize=15, family="serif", color="#902020")
    ax.text(RAIL_Y, X_SLOT - RAIL_H - 1.4, "rails", fontsize=15, family="serif", color="#1f3f80", ha="center")
    from matplotlib.patches import Patch
    keys = (["pad", "cover", "fr4", "board_parts", "dip_link", "mcp_link", "servo_pip"] + (["qwiic"] if "qwiic" in M else [])
            + (["cable"] if "cable" in M else []))
    fig.legend([Patch(facecolor=COLORS[k], edgecolor="#333333") for k in keys], [_labels()[k] for k in keys], loc="lower center",
               ncol=4, frameon=False, fontsize=17)
    what = (f"{_labels()['pad'][9:]} pad, {len(mags)} magnets" if DESIGN.get("pads") == "v2_five" else "block tip, four magnets")
    fig.suptitle(f"Sections through the chips, {_title().lower()} at PIP 0 deg ({what})", fontsize=21, family="serif")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def cmd_render(a):
    MEDIA.mkdir(parents=True, exist_ok=True)
    M = assembly_meshes("ell11_5" if "ell11_5" in TIPS else "block4")
    render_iso(M, MEDIA / f"{PREFIX}_iso.png", "palmar")
    render_iso(M, MEDIA / f"{PREFIX}_iso_dorsal.png", "dorsal")
    render_exploded(M, MEDIA / f"{PREFIX}_exploded.png")
    render_sections(M, MEDIA / f"{PREFIX}_section_chips.png", None)
    if DESIGN.get("qwiic") == "header_on_tongue":
        render_qwiic(M, MEDIA / f"{PREFIX}_qwiic.png")
    if cable_on():
        render_cable(M, MEDIA / f"{PREFIX}_cable_route.png")
    import trimesh
    asm = trimesh.util.concatenate(list(M.values()))
    asm.export(CAD_DIR / f"{PREFIX}_assembly_pip0.stl")
    print("renders and assembly written", flush=True)


# ================================================================================================= evaluation

def torsion_constant(polys, h=0.05):
    """Saint-Venant torsion constant J (mm^4) of a plane section given as polygons (union), from Prandtl's stress
    function (laplacian phi = -2 inside, phi = 0 on the boundary; J = 2 * integral of phi) on an h-mm grid."""
    import numpy as np
    import scipy.sparse as sps
    import scipy.sparse.linalg as spl
    from matplotlib.path import Path as MPath
    P = np.vstack(polys)
    lo, hi = P.min(0) - 2 * h, P.max(0) + 2 * h
    xs = np.arange(lo[0], hi[0], h)
    zs = np.arange(lo[1], hi[1], h)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Z.ravel()])
    inside = np.zeros(len(pts), bool)
    for poly in polys:
        inside |= MPath(poly).contains_points(pts)
    inside = inside.reshape(X.shape)
    idx = -np.ones(X.shape, int)
    idx[inside] = np.arange(inside.sum())
    n = int(inside.sum())
    rows, cols, vals = [], [], []
    I, K = np.nonzero(inside)
    for di, dk in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = idx[np.clip(I + di, 0, X.shape[0] - 1), np.clip(K + dk, 0, X.shape[1] - 1)]
        ok = nb >= 0
        rows.append(idx[I, K][ok])
        cols.append(nb[ok])
        vals.append(np.full(ok.sum(), -1.0))
    rows.append(idx[I, K])
    cols.append(idx[I, K])
    vals.append(np.full(n, 4.0))
    A = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    phi = spl.spsolve(A.tocsc(), np.full(n, 2.0 * h * h))
    return float(2.0 * phi.sum() * h * h), float(n * h * h)


def section_polys(mesh, y):
    """Closed polygons (x, z) of a mesh's section by the plane y."""
    s = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
    return [] if s is None else [p[:, [0, 2]] for p in s.discrete]


def cmd_eval(a):
    import numpy as np
    import trimesh
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import dip_link_geometry as G
    S = spec()
    fc = json.loads((WORK / "fc_build.json").read_text())
    clr = json.loads((WORK / "clearance.json").read_text())
    sens = json.loads((WORK / "sensing.json").read_text())
    prep = json.loads((WORK / "prep.json").read_text())
    fe = fe_rows()
    hw = hardware()
    # ---- sensing at the design point
    def pick(layout, tb, place):
        return next(r for r in sens if r["layout"] == layout and r["t_behind_mm"] == tb and r["placement"] == place)
    d4, d5 = pick("four_d3x2_d5", T_BEHIND, "concept_b"), pick("five_d3x2_d5", T_BEHIND, "concept_b")
    t4, t5 = pick("four_d3x2_d5", 10.0, "touch"), pick("five_d3x2_d5", 10.0, "touch")
    p4, p5 = pick("four_d3x2_d5", 10.0, "plate"), pick("five_d3x2_d5", 10.0, "plate")
    fine = {r["layout"] + "|" + r["placement"]: r for r in S["sensing"]["rows"] if r["recipe"] == "w85_30"}
    # ---- mass
    vol = {k: v["volume_mm3"] for k, v in fc.items() if "volume_mm3" in v}
    screw_v = float(hw["screws"].volume()) / 4
    nut_v = float(hw["nuts"].volume()) / 4
    m_dip_100 = vol["dip_link"] * RHO["pla"] / 1e3
    m_dip_15 = vol["dip_link"] * RHO["pla_15"] / 1e3
    m_cover = vol["cover"] * RHO["pla"] / 1e3
    m_fast = 4 * (screw_v + nut_v) * RHO["brass"] / 1e3
    m_board = S["board"]["mass_g"]
    tips_mass = S["mass"]["tips_g"]
    pad_mass = {}
    for key in TIPS:
        nmag = 5 if key == "block5" else 4
        whole_tpu = tips_mass[key] - nmag * MAG_MASS_G
        pad_mass[key] = round(whole_tpu * prep[f"pad_{key}"]["volume_mm3"] / prep[f"pad_{key}"]["whole_volume_mm3"]
                              + nmag * MAG_MASS_G, 2)
    # ---- lateral protrusion over the tip's length
    M = assembly_meshes("block4")
    zr = (TIP["z_end"], TIP["z_mount"])
    width = {}
    for k in ("dip_link", "cover", "fr4", "screws", "pad_block4"):
        V = M[k].vertices
        sel = (V[:, 2] >= zr[0] - 1e-6) & (V[:, 2] <= zr[1] + 1e-6)
        width[k] = dict(y=[round(float(V[sel, 1].min()), 2), round(float(V[sel, 1].max()), 2)],
                        x=[round(float(V[sel, 0].min()), 2), round(float(V[sel, 0].max()), 2)])
    new_body = load_cad("dip_link_new_body")
    Vb = new_body.vertices
    selb = (Vb[:, 2] >= zr[0]) & (Vb[:, 2] <= zr[1]) & (Vb[:, 2] < -70.06)
    width["dip_link_below_webs"] = dict(y=[round(float(Vb[selb, 1].min()), 2), round(float(Vb[selb, 1].max()), 2)])
    # ---- the bridge's torsion constant between the pad and the horn arm, new and current
    old = G.load_part("dip_link")
    dip = load_cad("dip_link")
    J = {}
    for name, mesh, ys in (("current", old, (-12.0, -15.0)), ("concept_b", dip, (-12.0, -15.0))):
        J[name] = {f"{y:g}": round(torsion_constant(section_polys(mesh, y))[0], 1) for y in ys}
    # ---- FE trade-off
    trade = []
    for r in sorted(fe, key=lambda r: (r["layout"], -r["t_behind_mm"])):
        tr = [float(np.linalg.norm(t)) for t, _ in r["per_force"]["1"]["magnets"]]
        trade.append(dict(layout=r["layout"], t_behind_mm=r["t_behind_mm"], x_slot_F=r["x_slot_F"], pad_depth_mm=r["pad_depth_mm"],
                          E_star_MPa=round(r["E_best_MPa"], 2), approach_1N_um=round(r["per_force"]["1"]["delta_um"], 1),
                          approach_3N_um=round(r["per_force"]["3"]["delta_um"], 1), magnet_travel_um_per_N=round(float(np.mean(tr)), 2),
                          n_dof=r["n_dof"]))
    base4 = next(t for t in trade if t["layout"] == "four_d3x2_d5" and t["t_behind_mm"] == 10.0)
    base5 = next(t for t in trade if t["layout"] == "five_d3x2_d5" and t["t_behind_mm"] == 10.0)
    for t in trade:
        b = base4 if t["layout"].startswith("four") else base5
        t["E_star_ratio"] = round(t["E_star_MPa"] / b["E_star_MPa"], 3)
        t["approach_1N_ratio"] = round(t["approach_1N_um"] / b["approach_1N_um"], 3)
        t["travel_ratio"] = round(t["magnet_travel_um_per_N"] / b["magnet_travel_um_per_N"], 3)
    dp = next(t for t in trade if t["layout"] == "four_d3x2_d5" and t["t_behind_mm"] == T_BEHIND)
    dp5 = next(t for t in trade if t["layout"] == "five_d3x2_d5" and t["t_behind_mm"] == T_BEHIND)
    tpu_behind = {k: round(X_SLOT - xb, 2) for k, (_, _, xb) in TIPS.items()}
    dist4 = chip_magnet_distances("block4")
    dist5 = chip_magnet_distances("block5")
    rigid = {k: v for k, v in clr.items() if k not in ("wires_at_exit", "pad_block4", "arms_kept")}
    worst = min(rigid.items(), key=lambda kv: kv[1]["min_mm"])
    # R4: the pad's rail grooves and head notches against its magnet pockets, per tip (pocket meshes from the print STLs)
    g_lo = np.array([X_SLOT - RAIL_H - GROOVE_FLOOR, RAIL_Y - RAIL_TOP / 2 + GROOVE_FIT, RAIL_Z[0] - 0.2])
    g_hi = np.array([X_SLOT, RAIL_Y + RAIL_TOP / 2 - GROOVE_FIT, RAIL_Z[1] + 0.2])

    def box_dist(P, lo, hi):
        d = np.maximum(np.maximum(lo - P, P - hi), 0.0)
        return np.linalg.norm(d, axis=1)
    r4 = {}
    for key in TIPS:
        pm = trimesh.load(WORK / f"pad_{key}.stl")
        voids = [b for b in pm.split(only_watertight=False) if abs(b.volume) < 100]
        Vv = np.vstack([b.vertices for b in voids])
        Vm = Vv * np.array([1, -1, 1])                     # the rails are symmetric in y
        dg = float(min(box_dist(Vv, g_lo, g_hi).min(), box_dist(Vm, g_lo, g_hi).min()))
        dn = min(float(np.min(np.maximum(np.hypot(Vv[:, 1] - y, Vv[:, 2] - z) - HEAD_NOTCH_D / 2, 0.0)
                               + 0 * Vv[:, 0])) for y, z in screw_yz())
        r4[key] = dict(groove_to_pocket_mm=round(dg, 2), notch_to_pocket_mm=round(dn, 2),
                       pocket_dorsal_x_F=round(float(Vv[:, 0].max()), 2), tpu_behind_pockets_mm=round(X_SLOT - float(Vv[:, 0].max()), 2))
    ev = dict(
        title="DIP link redesign, concept b: the board in a rigid cassette in place of the tip's dorsal part",
        date=DATE, status="design; model values; nothing printed or measured",
        generator="scripts/dip_link_concept_b.py (fe, prep, cad, clearance, sensing, render, eval)",
        spec="docs/experiments/20261018-dip_link_board_mount/spec.json",
        terms=dict(
            F=S["terms"]["F"], slot_plane="x_F of the pad's back face, the cover's front face and the chips' package tops",
            gap="distance from the pad's back face to the chips' sensing plane (0.42 mm under the package top), mm",
            force_resolution=S["terms"]["Force resolution"],
            E_star="pressure-law modulus fitted to the FE approach of the screwdriver cylinder (R 12.5 mm) at 0.5, 1 and 3 N, "
                   "with the pads' foundation depth 8.5 mm (scripts/tpu_tip_magnet.py best_modulus), MPa",
            magnet_travel="mean rigid translation of the four (five) magnets per newton of the cylinder's normal load at the "
                          "contact point, um/N",
            approach="indentation of the cylinder at 1 N, um",
            false_force=S["terms"]["False force"], clearance=S["terms"]["Clearance"],
            torsion_constant="Saint-Venant torsion constant of the DIP link's section between the pad and the horn arm (a plane "
                             "y = const), mm^4; the twist per unit length is T / (G J)"),
        design=dict(
            x_slot_F=round(X_SLOT, 3), t_behind_block_mm=T_BEHIND, pad_depth_mm=round(X_SLOT - TIP["x_palmar"], 2),
            stack_x_F=dict(pad=[TIP["x_palmar"], round(X_SLOT, 2)], cover_web=[round(X_SLOT, 2), round(X_COMP, 2)],
                           fr4=[round(X_COMP, 2), round(X_SEAT, 2)], seat_rim=[round(X_SEAT - RIM_H, 2), round(X_SEAT, 2)],
                           tongue=[round(X_SEAT, 2), round(X_TONGUE_BACK, 2)], spine=[round(X_TONGUE_BACK, 2), X_SPINE_BACK],
                           sensing_plane=round(X_SLOT + SPOT, 2)),
            chip_height_mm=CHIP_H, chip_height_src="MLX90393 0.80-1.00 mm; the cover's web is printed at the measured height (pending)",
            tpu_behind_magnets_mm=tpu_behind,
            tab=dict(cut_at_v=TAB_CUT_V, length_mm=round(-10.10 - TAB_CUT_V, 2), keeps="R1, R2 and the four vias at v -10.2 .. -10.3",
                     removes="J1 and its pads, C6 (unpopulated)"),
            fasteners="4 brass M1.6 x 5 cheese head (DIN 84, head 3.0 x 1.0 mm) through cover, board and tongue into 4 brass M1.6 "
                      "hex nuts (ISO 4032, 3.2 mm across flats, 1.3 mm) in pockets on the tongue's back",
            rails=dict(y_centres=[-RAIL_Y, RAIL_Y], base_width=RAIL_BASE, top_width=RAIL_TOP, height=RAIL_H, z=list(RAIL_Z),
                       groove_fit_per_flank=GROOVE_FIT, groove_floor_clearance=GROOVE_FLOOR),
            bridge="between the arms, from the pad's mount plane z -67.30 up to the keep-out of the MCP link and PIP servo less "
                   f"{KEEP_MARGIN} mm (x -10.55 .. {X_SPINE_BACK}), webs at |y| 10.5 .. 17.05 and 10.5 .. 13.40 down to z {Z_WEB_LOW}",
            parts={k: v for k, v in fc.items()}),
        chip_to_magnet_mm=dict(
            four_magnets=dist4, five_magnets=dist5,
            min_centre_four=min(v["centre_mm"] for v in dist4.values()), min_centre_five=min(v["centre_mm"] for v in dist5.values()),
            touch_placement_min_centre_four=round(math.dist((TIP["x_back"] + SPOT, 0, Z_CONTACT), (-4.55, 3.0, -74.66)), 2),
            plate_placement_min_centre_four=round(math.dist((TIP["x_back"] + 3.42, 0, Z_CONTACT), (-4.55, 3.0, -74.66)), 2)),
        board_to_tip_gap_mm=dict(concept_b=SPOT, current_plate=3.42,
                                 note="the sensing plane lies 0.42 mm behind the pad's back face; the pad's back face lies "
                                      f"{TIP['x_back'] - X_SLOT:.2f} mm in front of the whole tip's back face, so in the spec's "
                                      f"convention (gap from x_F 6.45) the gap is {X_SLOT + SPOT - TIP['x_back']:.2f} mm"),
        fe_tradeoff=dict(rows=trade, grid="coarse grid shared by every depth (scripts/dip_link_concept_b.py FE_GRID, auxiliary voxel "
                                          "0.1 mm); on the readiness study's fine grid the whole block gives E* 7.55 MPa, approach "
                                          "60.8 um and travel 13.18 um/N (four magnets, 85A 30 %)",
                         model="readiness FE (scripts/sim2real_fingertip_fe.py): 85A solid 12 MPa, gyroid 30 %, two 0.45 mm walls at "
                               "every outline, bonded magnets, back and mount faces fixed; the pad's back face is the cover",
                         design_point=dict(four=dp, five=dp5)),
        sensing=dict(
            method="scripts/tpu_tip_magnet_sensing.py analyse (point dipoles N52 1.45 T, the five MLX90393 of the ReSkin 5X, "
                   "one 141 Hz sample, Cramer-Rao bound of the normal force with the contact position and a uniform background "
                   "field unknown) on this concept's FE magnet motions; inputs changed: the FE rows (coarse grid, pad cut at the "
                   "slot plane, back face fixed there) and the sensing plane (gap -5.58 mm from the whole tip's back face); "
                   "'gain_noise' adds the quantization of the gain and resolution the field needs (1.25 x headroom), "
                   "'gain7_noise' keeps GAIN_SEL 7's noise as the spec did",
            design_point=dict(four=d4, five=d5), touch_whole_block=dict(four=t4, five=t5), plate_whole_block=dict(four=p4, five=p5),
            spec_fine_grid=dict(four_touch=fine["four_d3x2_d5|touch"]["centre_force_N"], four_plate=fine["four_d3x2_d5|plate"]["centre_force_N"],
                                five_touch=fine["five_d3x2_d5|touch"]["centre_force_N"], five_plate=fine["five_d3x2_d5|plate"]["centre_force_N"]),
            all_rows=sens),
        clearance=dict(rows=clr, min_rigid_mm=worst[1]["min_mm"], min_rigid_part=worst[0],
                       method="dip_link_geometry.points_clearance_exact: surface samples at 0.25 mm and voxel centres at 0.5 mm of "
                              "each part, against the MCP link and the PIP servo, PIP -18..92 deg every 0.5 deg (k-d tree)",
                       requirement="R7: 0.5 mm for rigid parts, 1.0 mm for the cable"),
        protrusion=dict(width_over_tip_length=width, tip_half_width=TIP["y_half"],
                        board_past_tip_per_side_mm=round(10.0 - TIP["y_half"], 2), rim_past_tip_per_side_mm=round(Y_RIM - TIP["y_half"], 2),
                        band_x_F=[round(X_SLOT - HEAD_H, 2), X_TONGUE_BACK],
                        band_from_palmar_face_mm=round(X_SLOT - HEAD_H - TIP["x_palmar"], 2)),
        mass_g=dict(dip_link_100pct=round(m_dip_100, 2), dip_link_15pct=round(m_dip_15, 2), cover=round(m_cover, 2), board=m_board,
                    fasteners=round(m_fast, 2), total_without_tip_100pct=round(m_dip_100 + m_cover + m_board + m_fast, 2),
                    total_without_tip_15pct=round(m_dip_15 + m_cover + m_board + m_fast, 2), limit_R12=7.6,
                    current=dict(dip_link=4.92, board=0.64), pads=pad_mass, whole_tips=tips_mass,
                    densities_g_cm3=RHO),
        torsion_constant_mm4=J,
        r4_distances=r4,
        retention=dict(
            board=dict(
                x="clamped normal to its plane at the four holes between the cover's web (screw heads) and the tongue's seat by "
                  "four brass M1.6 screws into brass nuts; between the holes the cover's web lies on the component face around "
                  "every part and the pad presses web and chips onto the seat",
                y="located against the seat pocket's datum wall at y_F -10.00 (the board pushed against it at assembly), then held "
                  "by friction under the screw preload; the +y side has 0.05 mm clearance",
                z="located against the pocket's proximal wall at z_F -68.06 (the main square's edge beside the tab), then held "
                  "by friction; the distal side has 0.05 mm clearance",
                play="no free play: the datum walls fix the in-plane position at assembly and the preload (about 100 N per screw at "
                     "0.04-0.05 N m) gives about 4 x 100 N x 0.3 = 120 N of friction against an in-plane share of the 5 N design "
                     "shear; the screws pass the 2.40 mm holes with 0.40 mm radial clearance and do not locate",
                tolerances="PLA holes print 0.1-0.2 mm undersize: screw holes 1.80 mm for M1.6, nut pockets 3.35 mm across flats for "
                           "3.2 mm nuts, the pocket 0.05 mm over the board on its free sides; the board's routing tolerance is taken "
                           "at the datum walls"),
            pad=dict(
                x="two dovetail rails on the cover (1.30 mm wide at the cover, 1.90 mm at the top, 1.00 mm tall) snap into "
                  "grooves of the pad's back face that are 0.05 mm narrower per flank; the wedge presses the pad's back face onto "
                  "the cover and the chip tops",
                y="the rails, and the four corner notches around the screw heads (0.20 mm clearance)",
                z="the four head notches, and the pad's mount face on the bridge at z_F -67.30",
                removal="the pad peels off palmar; the board's screws stay"),
            note="the TPU's preload and the snap fit are unmeasured; a pad that lifts off the cover moves its magnets relative to "
                 "the chips, which reads as force"),
        contact_geometry=dict(
            unchanged="the pad's palmar face, its edge rounds and its side faces palmar of x_F -0.75 mm (the capsule's and "
                      "half-ellipsoids' curved surfaces included)",
            changed=["the TPU dorsal of x_F 0.45 mm (6.0 mm of the 17.0 mm tip) is the rigid cassette: the cover's web, the board "
                     "and the tongue, 21.0 mm wide (|y| <= 10.5) against the tip's 14.8 mm",
                     "four corner notches in the pad's back edges, 3.4 mm in diameter and 1.2 mm deep, at y_F +-8.0 and z_F -70.06 "
                     "and -86.06; the screw heads (3.0 mm, |y| 6.5..9.5) stand 1.0 mm palmar of the cover there",
                     "two dovetail grooves in the pad's back face (internal, under the rails)",
                     "the 21.0 mm band starts 10.0 mm behind the palmar face (screw heads) and 11.0 mm (cover), against 17.0 mm for "
                     "a board on the whole tip's back face and 20.0 mm for the current plate",
                     f"the pad is stiffer: E* {100 * (dp['E_star_ratio'] - 1):+.0f} % (four magnets) and "
                     f"{100 * (dp5['E_star_ratio'] - 1):+.0f} % (five), approach at 1 N {100 * (dp['approach_1N_ratio'] - 1):+.0f} % "
                     f"and {100 * (dp5['approach_1N_ratio'] - 1):+.0f} %, magnet travel {100 * (dp['travel_ratio'] - 1):+.0f} % and "
                     f"{100 * (dp5['travel_ratio'] - 1):+.0f} % (coarse FE, the same grid for the whole block)"]),
        printability=dict(
            dip_link="PLA or PETG on the Prusa XL, on its side with the horn arm's outer face (y_F -23.25) on the bed: the bridge's "
                     "profile and the arms print as layers, every hole along y is vertical; the tongue's -y face, 12.75 mm above "
                     "the bed, needs a support (tree support, about 18.7 x 4.1 mm); the seat (x_F 1.95) and the mount stop "
                     "(z_F -67.30) are vertical walls, not top or bottom surfaces (spec printer rule): print 0.10 mm layers there, "
                     "put the seam away from the seat and check it with a straightedge (0.05 mm), lapping on 400 grit if needed. "
                     "100 % infill in the tongue and bridge (modifier), 4 perimeters elsewhere. Thinnest features: the bridge 2.3 mm "
                     "(z -67.30..-65.00 at x 0), the seat rim 0.50 mm wide and 0.45 mm tall (one extrusion line; R5's 21.0 mm leaves "
                     "no more), the tie roof 1.0 mm, the floor under each nut 2.15 mm",
                     cover="PLA or PETG, the clamp face (back, x_F 1.35) on the bed so it prints flat, 0.10 mm layers (the 0.90 mm web is "
                           "nine layers; reprint at the measured chip height), 100 % infill, no supports; the dovetail flanks lean "
                           "16.7 deg; the narrowest wall between windows is 0.85 mm (U1 to C1)",
                     pad="TPU 85A on tool 2 with the readiness recipes: the block pads on their back face (the rail grooves on the bed, "
                         "their 1.8 mm ceilings bridged), the capsule and half-ellipsoid pads upright on the mount face; the magnet "
                         "pockets as in the print sheets with new pause heights (the block's pocket floor 4.0 mm above the bed); the "
                         "print sheets need re-slicing (pending)"),
        assembly_steps=[
            "Before cutting, find VDD, GND, SDA and SCL with a multimeter from J1's four pins to the four vias at v -10.2..-10.3 "
            "and to R1/R2 (GND is the bottom copper pour).",
            "Cut the board's tab 1.60 mm from the main square (v -11.70), keeping R1 and R2; file the edge.",
            "Solder four 30 AWG wires on the bare side at the vias and the GND pour; keep the joints under 0.4 mm proud.",
            "Press four brass M1.6 nuts into the hex pockets on the tongue's back (the proximal two through the spine).",
            "Pass the wires through the exit slot from the seat side; lay the board in the seat pocket, chips facing palmar, and push "
            "it against the -y datum wall and the proximal wall.",
            "Lay the cover on the board (windows over the parts, rails palmar); insert four brass M1.6 x 5 cheese-head screws from the "
            "palmar side and tighten crosswise to about 0.04-0.05 N m.",
            "Tie the wires to the tie block on the tongue's back with a 2.5 mm cable tie.",
            "Press the pad onto the cover: notches over the screw heads, mount face against the bridge, until both rails snap in.",
            "Read the zero-load field of the five chips. Tip exchange: peel the pad off and press the next one on. Board exchange: pad "
            "off, four screws out, cover off."],
        parts_list=[
            dict(part="DIP link", qty=1, material="PLA or PETG", file=f"cad/{PREFIX}_dip_link.stl (.step.zip)"),
            dict(part="Cover", qty=1, material="PLA or PETG", file=f"cad/{PREFIX}_cover.stl (.step)"),
            dict(part="Pad, one per tip", qty=1, material="Siraya Tech TPU 85A, gyroid 30 % or 50 %",
                 file=f"cad/{PREFIX}_pad_<tip>.stl (block4, block5, cap8.5, ell16, ell11)"),
            dict(part="ReSkin 5X board, tab cut at v -11.70", qty=1),
            dict(part="M1.6 x 5 cheese-head screw, DIN 84, brass", qty=4),
            dict(part="M1.6 hex nut, ISO 4032, brass", qty=4),
            dict(part="30 AWG wire (length pending, to the inline connector on the MCP link side)", qty=4),
            dict(part="Cable tie, 2.5 mm", qty=1),
            dict(part="N52 magnets 3 x 2 mm (as the tips)", qty="4 or 5")],
        pending=["Chip height and FR-4 thickness on the owner's board (the cover's web is printed at the chip height)",
                 "The board's flatness (a warped board can move under load where only the pad presses it)",
                 "The four nets' solder points on the cut tab",
                 "The cable's size and its route across the PIP joint (R8)",
                 "The pad's preload and the snap fit's pull-off force; the zero-load field before and after 50 presses at 3 N",
                 "Fine-grid FE of the pads (6-9 GB per run) and the calibration branch's fit for the pad depth",
                 "Re-sliced print sheets of the pads (pause heights)",
                 "The PIP servo's motor field at the chips over the PIP range"],
    )
    if DESIGN.get("qwiic", "none") != "none":
        final_sections(ev, clr)
    ev["files"] = dict(
        generator="scripts/dip_link_concept_b.py", params=str(PARAMS.relative_to(REPO)),
        print=sorted(str(f.relative_to(REPO)) for f in PRINT_OUT.glob("*")) if PRINT_OUT.exists() else [],
        cad=sorted(str(f.relative_to(REPO)) for f in CAD_DIR.glob("*")),
        media=sorted(str(f.relative_to(REPO)) for f in MEDIA.glob("*.png")),
        fe=str(FE_ROWS.relative_to(REPO)), evaluation=str(EVAL.relative_to(REPO)),
        note="STEP files over 1.5 MB are zipped (git LFS stores docs/**/*.zip); the DIP link's STEP keeps the arms as the "
             "STL's facets, the B-rep of the redesigned body alone is *_dip_link_new_body.step")
    EVAL.write_text(json.dumps(ev, indent=1))
    print(json.dumps(dict(clearance=ev["clearance"]["min_rigid_mm"], part=ev["clearance"]["min_rigid_part"],
                          four=d4["gain_noise"]["centre_force_N"], five=d5["gain_noise"]["centre_force_N"], J=J,
                          mass=ev["mass_g"]["total_without_tip_100pct"], width=width), indent=1))


def _qwiic_why(rows):
    prox = [r for r in rows if r["orientation"] == "tab proximal"]
    other = [r for r in rows if r["orientation"] != "tab proximal"]
    out = [max(r["beyond_tip_end_mm"], r["beyond_band_mm"]) for r in other]
    return (f"With the tab proximal, J1, its plug and the cable's first 3 mm overlap the PIP servo by "
            f"{min(-r['connector_clearance_mm'] for r in prox):.1f}-{max(-r['connector_clearance_mm'] for r in prox):.1f} mm and "
            f"the bare tab by {max(-r['tab_clearance_mm'] for r in prox):.1f} mm; distal or lateral they clear the links but "
            f"stand {min(out):.1f}-{max(out):.1f} mm beyond the tip's end or the 21 mm band, where grasped objects meet them.")


def final_sections(ev, clr):
    """The parts of b's final evaluation that the committed concept b lacks: the Qwiic study and the header, the static
    fields of the 2.9 mm pads, the pads' print files and pauses, and the print settings (also written beside the
    evaluation)."""
    q = json.loads((FINAL / f"{FINAL_PFX}_qwiic.json").read_text())
    parts, cuts = qwiic_header_b()
    pb = {n: (lo, hi) for n, lo, hi in parts}
    ev["title"] = ("DIP link concept b, final: the board in a cassette in place of the pad's dorsal 6 mm, the Qwiic header in "
                   "the tongue's back, pads of the 2.9 mm print set")
    ev["status"] = "design; model values; nothing printed or measured"
    ev["qwiic"] = dict(
        study=[dict(orientation=("flipped, " if r["flipped"] else "") + r["orientation"],
                    connector_clearance_mm=r["connector_clearance_mm"], tab_clearance_mm=r["tab_clearance_mm"],
                    beyond_tip_end_mm=r["beyond_tip_end_mm"], beyond_band_mm=r["beyond_band_mm"], into_pad_mm=r["into_pad_mm"],
                    fits=bool(r["clears_links"] and r["inside_envelope"])) for r in q["rows"]],
        decision=("J1 kept" if q["j1_kept"] else "J1 cut off with the tab; a pigtail to an SM04B-SRSS-TB header in a pocket of "
                                                 "the tongue's back"),
        why=_qwiic_why(q["rows"]),
        header=dict(where=f"pocket {P['qwiic_pocket_depth']:.2f} mm deep in the tongue's back (x_F {X_TONGUE_BACK:.2f}), centred "
                          f"at y {P['qwiic_y']:.2f}, its back at z {P['qwiic_z_back']:.2f}, mating face proximal",
                    header_box_F=[list(pb["q_header"][0]), list(pb["q_header"][1])],
                    plug_box_F=[list(pb["q_plug"][0]), list(pb["q_plug"][1])],
                    tongue_under_pocket_mm=round(X_TONGUE_BACK - P["qwiic_pocket_depth"] - X_SEAT, 2),
                    pigtail=f"four wires from the vias through the exit slot (z {WIRE_SLOT['z'][0]:.2f}..{WIRE_SLOT['z'][1]:.2f}) "
                            f"and down a {P['qwiic_groove_depth']:.2f} mm groove at y {P['qwiic_groove_y0']:.2f}.."
                            f"{P['qwiic_groove_y1']:.2f} to the header's leads",
                    cable=("the Qwiic cable leaves the plug toward the PIP joint, rises over the tie block (top x_F "
                           f"{TIE['x_top']:.2f}) and is held there by a 2.5 mm cable tie through the block's tunnel"),
                    clearance=clr.get("qwiic_connector")))
    ev["static_fields_2p9"] = json.loads((WORK / "static_fields.json").read_text())
    pads = {}
    for key, (name, orient, _) in TIPS.items():
        pads[key] = dict(file=str((PRINT_OUT / f"{PREFIX}_pad_{key.replace('.', 'p')}_print.stl").relative_to(REPO)),
                         sheet=f"docs/experiments/20261016-sim2real_fingertip_readiness/print/{name}_g30.json",
                         orientation="cut back face on the bed, palmar face up" if orient == "back" else
                                     "upright, mount face on the bed", pauses=pad_pauses(key))
    ev["pads_print"] = pads
    ps = dict(
        printer="Original Prusa XL 5T, 0.4 mm nozzles (PrusaSlicer profile 'Original Prusa XL - 5T Input Shaper 0.4 nozzle')",
        dip_link=dict(file=f"cad/{PREFIX}_dip_link.stl", tool="a PLA tool (3, 4 or 5 in the owner's v17 template)",
                      material="PLA (the links: PLA, 15 % infill) or PETG",
                      orientation="on its side, the horn arm's outer face (y_F -23.25) on the bed: the bridge's profile and "
                                  "the arms print as layers, every hole along y is vertical",
                      layer_mm=0.10, perimeters=4, infill="100 % in the tongue and the bridge (modifier), 15 % gyroid elsewhere",
                      supports="a tree support under the tongue's -y face (12.75 mm above the bed); none in the header pocket "
                               "or the groove, which open upward on the tongue's back",
                      after="check the seat (x_F 1.95) with a straightedge (0.05 mm); press the four M1.6 nuts in; clear the "
                            "header pocket and the groove of strings"),
        cover=dict(file=f"cad/{PREFIX}_cover.stl", tool="a PLA tool", material="PLA or PETG, solid",
                   orientation="the clamp face (back, x_F 1.35) on the bed", layer_mm=0.10, infill="100 %", supports="none",
                   note="the web is as thick as the chips (chip_h); reprint at the measured package height"),
        pads=dict(tool="tool 2", material="Siraya Tech TPU 85A", profile="Siraya Tech TPU 85A @XL 0.4mm Nozzle (the print "
                  "procedure page)", infill="gyroid 30 % (50 % for the stiff variants), two perimeters, two around each pocket",
                  seat_mm=2.9, per_pad=pads,
                  note="the block pads print on their cut back face, so their pause is 6.0 mm lower than the whole block's "
                       "(12.4 mm on the 2.9 mm sheet); the curved pads print upright and keep the sheets' pauses; slice each "
                       "pad's STL with its sheet's settings and these pauses"),
        assembly_extra=["Cut the tab at v -11.70 (J1 goes with it) and solder four wires (VDD, GND, SDA, SCL; AWG 30-32, "
                        "about 25 mm) to the vias on the bare side",
                        "Pass the wires through the exit slot, lay them in the groove of the tongue's back and solder them to "
                        "the header's leads in the Qwiic pin order (GND, VDD, SDA, SCL from pin 1)",
                        "Glue the SM04B-SRSS-TB header into its pocket (CA glue on the floor), mating face proximal",
                        "Plug the Qwiic cable in, lay it over the tie block and tie it there; route it across the PIP joint "
                        "outside the slotted arm (y > 17.4 mm)"],
        when=time.strftime("%Y-%m-%d %H:%M"))
    (EVAL.parent / f"{PREFIX}_print_settings.json").write_text(json.dumps(ps, indent=1))
    ev["print_settings"] = str((EVAL.parent / f"{PREFIX}_print_settings.json").relative_to(REPO))


def cmd_sensing(a):
    t = sensing_table()
    (WORK / "sensing.json").write_text(json.dumps(t, indent=1))
    (WORK / "static_fields.json").write_text(json.dumps(static_fields(), indent=1))


# ================================================================================================= the Qwiic connection
# The final of concept b (concept_b/final/): the pads of the 2.9 mm print set and a Qwiic connection. The study places the
# board's own J1 (JST SH side-entry header, its mated SHR-04V-S housing and the cable's first 3 mm; the connector model of
# scripts/dip_link_board_mount.py) in eight board orientations at b's placement and checks each against the MCP link and
# the PIP servo over the PIP range and against the envelope the cassette already occupies.

FINAL = OUT / "final"
FINAL_PFX = f"{DATE}-concept_b_final"


def board_to_F_b(P_b, theta_deg, flipped):
    """Board frame (u, v, w; w the height above the component face) to F for a board rotated in its plane by theta about
    U1 (0: tab proximal, toward the PIP joint; 180: tab distal; 90: tab toward -y; 270: tab toward +y) at b's placement
    (component face at X_COMP, parts toward the pad), or flipped (bare face on the pad's back face X_SLOT, parts dorsal)."""
    import numpy as np
    th = math.radians(theta_deg)
    u, v, w = P_b[:, 0], P_b[:, 1], P_b[:, 2]
    if flipped:
        u = -u
    ur = u * math.cos(th) - v * math.sin(th)
    vr = u * math.sin(th) + v * math.cos(th)
    x = X_SLOT + FR4 + w if flipped else X_COMP - w
    return np.column_stack([x, ur, Z_CONTACT - vr])


def cmd_qwiic(a):
    """Where the board's own Qwiic connector J1 can go at b's placement: the eight orientations of
    dip_link_board_mount.cmd_qwiic (tab proximal, distal, toward -y and +y; component side toward the pad or flipped), each
    against the MCP link and the PIP servo over PIP -18..92 deg (points_clearance_exact, 0.25 mm samples) and against b's
    envelope: the tip's end z -88.71, the 21 mm band |y| <= Y_RIM and the pad palmar of X_SLOT. J1 stays if an
    orientation clears by 0.5 mm inside the envelope. -> concept_b/final/<FINAL_PFX>_qwiic.json"""
    import numpy as np
    import trimesh
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import dip_link_board_mount as DLF
    import dip_link_geometry as G
    t0 = time.time()
    rows = []
    for flipped in (False, True):
        for theta, label in ((0, "tab proximal"), (180, "tab distal"), (90, "tab toward -y"), (270, "tab toward +y")):
            meshes = {}
            for name, lo, hi in DLF.j1_parts_board():
                bm = DLF.box_mesh(lo, hi)
                tm = trimesh.Trimesh(board_to_F_b(np.asarray(bm.vertices), theta, flipped), np.asarray(bm.faces), process=False)
                if tm.volume < 0:
                    tm.invert()
                meshes[name] = tm
            conn = trimesh.util.concatenate([m for k, m in meshes.items() if k != "tab_fr4"])
            Pc = G.surface_points(conn, 0.25)
            v, i, q, part = G.points_clearance_exact(Pc, reach=3.0)
            Pt = G.surface_points(meshes["tab_fr4"], 0.25)
            vt, _, qt, pt_ = G.points_clearance_exact(Pt, reach=3.0)
            V = np.asarray(conn.vertices)
            beyond_end = float(max(0.0, TIP["z_end"] - V[:, 2].min()))
            beyond_side = float(max(0.0, np.abs(V[:, 1]).max() - Y_RIM))
            pad_zone = (V[:, 2] >= TIP["z_end"]) & (V[:, 2] <= TIP["z_mount"]) & (np.abs(V[:, 1]) <= TIP["y_half"])
            into_pad = float(max(0.0, X_SLOT - V[pad_zone, 0].min())) if pad_zone.any() else 0.0
            ok_link = bool(v >= 0.5 and vt >= 0.5)
            ok_env = beyond_end <= 0.0 and beyond_side <= 0.0 and into_pad <= 0.0
            rows.append(dict(orientation=label, theta_deg=theta, flipped=flipped, connector_clearance_mm=round(v, 3),
                             connector_clearance_at=dict(pip_deg=q, part=part, at_F=np.round(Pc[i], 2).tolist()),
                             tab_clearance_mm=round(vt, 3), tab_clearance_at=dict(pip_deg=qt, part=pt_),
                             beyond_tip_end_mm=round(beyond_end, 2), beyond_band_mm=round(beyond_side, 2),
                             into_pad_mm=round(into_pad, 2), clears_links=ok_link, inside_envelope=ok_env,
                             bounds_F=np.round(conn.bounds, 2).tolist()))
            print(f"{'flipped' if flipped else 'toward pad':10s} {label:14s}: connector {v:+.2f} mm ({part}, PIP {q}), tab "
                  f"{vt:+.2f} mm; beyond the end {beyond_end:.1f}, beyond the band {beyond_side:.1f}, into the pad "
                  f"{into_pad:.1f} mm [{time.time() - t0:.0f} s]", flush=True)
    keep = [r for r in rows if r["clears_links"] and r["inside_envelope"]]
    out = dict(model=dict(source="scripts/dip_link_board_mount.py j1_parts_board (JST SH SM04B-SRSS-TB on the board, "
                                 "SHR-04V-S housing mated, Qwiic cable 4 x 28 AWG as 0.8 mm wires for 3.0 mm)",
                          placement=f"component face at x_F {X_COMP:.2f} (parts toward the pad) or flipped (bare face at "
                                    f"x_F {X_SLOT:.2f}, parts dorsal); U1 over the contact point",
                          envelope=dict(z_end=TIP["z_end"], half_w=Y_RIM, pad_back_x=X_SLOT), need_mm=0.5),
               rows=rows, j1_kept=bool(keep), kept_orientation=keep[0]["orientation"] if keep else None,
               params=str(Path(os.environ.get("DIPB_PARAMS") or PARAMS).relative_to(REPO)),
               when=time.strftime("%Y-%m-%d %H:%M"))
    FINAL.mkdir(parents=True, exist_ok=True)
    dst = FINAL / f"{FINAL_PFX}_qwiic.json"
    dst.write_text(json.dumps(out, indent=1, default=float))
    print(f"J1 kept: {bool(keep)} -> {dst.relative_to(REPO)} [{time.time() - t0:.0f} s]", flush=True)


# ================================================================================================= command line

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fe")
    f.add_argument("--h-aux", type=float, default=0.1, help="auxiliary voxel of the wall and skin classes, mm")
    f.add_argument("--only", nargs="*", help="layout@t_behind, e.g. four_d3x2_d5@4")
    sub.add_parser("prep", help="arms, bridge profile and pads -> logs/20261018-dip_link/concept_b/")
    sub.add_parser("cad", help="freecadcmd: STEP and STL of the DIP link, the cover and the pads -> concept_b/cad/")
    c = sub.add_parser("clearance", help="exact clearance of every new part over the PIP range -> WORK/clearance.json")
    c.add_argument("--force", action="store_true")
    sub.add_parser("sensing", help="field and force resolution per FE depth -> WORK/sensing.json")
    sub.add_parser("render", help="iso, exploded and section figures -> concept_b/media/; assembly STL")
    sub.add_parser("eval", help="the evaluation -> concept_b/evaluation.json")
    sub.add_parser("qwiic", help="J1 in eight orientations at b's placement -> concept_b/final/*_qwiic.json")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"fe": cmd_fe, "prep": cmd_prep, "cad": cmd_cad, "clearance": cmd_clearance, "sensing": cmd_sensing,
     "render": cmd_render, "eval": cmd_eval, "qwiic": cmd_qwiic}[a.cmd](a)


if os.environ.get("DIPB_FREECAD"):
    fc_build()
elif __name__ == "__main__":
    main()
