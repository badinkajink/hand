#!/usr/bin/env python3
"""DIP link concept (a), window plate and dorsal clamp: a parametric generator and its checks (2026-10-18).

The DIP link's plate behind the printed TPU tip becomes a window plate as thick as the chips' packages less 0.05 mm, so the
five MLX90393 of the ReSkin 5X board reach through their windows and stand 0.05 mm proud of the tip's back face. The board
lies on the window plate's dorsal face; a 3.0 mm dorsal clamp plate presses it there. Two printed pins of the clamp plate
pass through the board's proximal holes into the window plate and locate board and clamp; two M1.6 countersunk brass
screws through the board's distal holes clamp window plate, board and clamp plate together; two M2 brass screws pull the
clamp plate's ears toward the DIP link's shoulders beside the connector tab, which clamps the board's proximal margin and
the tab. The tip slides onto two dovetail keys of the DIP link (its mount face onto the bridge) and a detent on each key
stops it sliding off. The clevis arms are the STL's own facets. Specification: docs/experiments/20261018-dip_link_board_mount/spec.json.

Frame F of the spec (scripts/dip_link_geometry.py): mm, x toward the dorsal side, y along the PIP axis, z up; the tip's
back face at x 6.45, its mount face at z -67.30, its end at z -88.71, the contact point (-10.55, 0, -78.16).

    E=~/miniconda3/envs/femtip/bin/python
    RG="$HOME/.claude/bin/resguard.sh run --mem 2.5G --cpu 200 -- taskset -c 12-15 env NUMPY_MADVISE_HUGEPAGE=0"
    $RG $E scripts/dip_link_concept_a.py build      # parts as meshes (manifold3d) -> concept_a/*.stl, parts.json
    $RG env DIPA_CMD=step DIPA_REPO=$PWD freecadcmd scripts/dip_link_concept_a.py   # STEP of each printed part
    $RG $E scripts/dip_link_concept_a.py check      # clearance over the PIP range, walls, masses -> check.json
    $RG .venv/bin/python scripts/dip_link_concept_a.py sense    # field and force resolution -> sense.json
    $RG .venv/bin/python scripts/dip_link_concept_a.py render   # iso, exploded and section figures -> media/
    .venv/bin/python scripts/dip_link_concept_a.py eval         # evaluation.json from the three above
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:                                   # freecadcmd executes the file without __file__
    REPO = Path(os.environ["DIPA_REPO"])
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

STUDY = REPO / "docs/experiments/20261018-dip_link_board_mount"
OUT = STUDY / "concept_a"
MEDIA = OUT / "media"
SPEC = STUDY / "spec.json"
DATE = "20261018"
PFX = f"{DATE}-concept_a"
READY = REPO / "docs/experiments/20261016-sim2real_fingertip_readiness"
CTX = REPO / "logs/20261018-dip_link/concept_a"          # context meshes for the figures (untracked)

# ----------------------------------------------------------------------------------------------- frame F (spec)
X_BACK, X_PALM, Y_HALF = 6.45, -10.55, 7.4
Z_MOUNT, Z_END, Z_C = -67.30, -88.71, -78.16
PIP = np.array([0.0, 0.0, -51.55])
ARM_Y = (-17.05, 13.40)                         # the clevis arms lie at y <= -17.05 and y >= 13.40 (R6)
FUSE = 0.005                                    # overlap of new solids into the arms' cut faces, mm

# ----------------------------------------------------------------------------------------------- parameters
P = dict(
    chip_h=0.90,            # MLX90393 package height, mm (datasheet 0.80-1.00; the owner's board is to be measured)
    chip_proud=0.05,        # chip tops stand this far proud of the window plate, into the TPU
    fr4=0.60,               # board thickness (ReSkin README, rigid order; to be measured)
    clamp_t=3.0,            # clamp plate over the board's bare face
    ear_t=1.5,              # clamp-plate ears over the DIP link's shoulders
    ear_gap=0.30,           # ear to shoulder: the M2 screws' preload goes into the board, not into this face
    half_w=10.5,            # |y| of the window plate and the clamp plate: the board's 10.0 plus the 0.5 mm rim of R5
    clear_chip=0.20,        # window clearance around each MLX90393
    clear_passive=0.25,     # around C1-C5 and R1-R2 (solder fillets)
    tab_len=4.9,            # connector tab cut to this length past the board's main square (J1 removed, wires soldered)
    pin_d=2.35,             # clamp-plate pins in the board's 2.40 mm proximal holes
    pin_hole_d=2.40,        # their holes in the window plate (reamed)
    pin_into_plate=0.55,    # how far each pin enters the window plate
    diamond_half=0.60,      # half-width along y of the second (diamond) pin: it locates z only
    cs_d=1.6, cs_k=0.96, cs_dk=3.0,     # M1.6 countersunk (DIN 963 / ISO 2009): shank, head height, head diameter
    nut_s=3.2, nut_m=1.3,   # M1.6 hex nut (ISO 4032)
    nut_clear=0.00,         # nut pressed into the pocket (warm it with the soldering iron if tight): walls >= 0.8 mm
    cs_len=4.0,             # M1.6 x 4: its end stays inside the nut, 0.45 mm under the clamp plate's back
    m2_y=8.1, m2_z=-65.35,  # M2 screws through the ears into the shoulders
    m2_pilot=1.7, m2_depth=3.5, m2_head_d=3.8, m2_head_k=1.3, m2_len=5.0, m2_clear_d=2.4,
    bridge_r=13.35,         # bridge's top: an arc about the PIP axis (MCP link's end at 12.70) ...
    bridge_flat=-63.10,     # ... cut flat on the palmar side
    shoulder_r=13.45, shoulder_cap=-61.70, shoulder_y=(6.0, 10.5), shoulder_z0=-67.86,
    ear_top=-62.60, tab_cover_top=-63.00,
    key_yc=3.6, key_root=2.4, key_head=3.6, key_depth=1.8, key_len=5.0,
    detent_h=0.25, detent_z=-71.60,
    fit_flank=0.05, fit_floor=0.10, fit_len=0.10, groove_r=0.35,
    wire_channel=dict(z=(-65.70, -64.50), x0=6.55),
    j1_kept=0.0,            # v2 (2026-10-18 18:45): 1 keeps the board's whole tab with its Qwiic connector J1; the bridge is
                            # slotted for J1, its plug and the cable (tab_len 9.0)
    cs_thread_d=0.0,        # v2: > 0 drops the M1.6 nuts; the distal screws form their threads in a hole of this diameter in
                            # the clamp plate (owner: M2 into 1.8 mm, M3 into 2.75 mm)
    asm_tip="block4",       # the tip of the assembly STL, the renders and the overlap checks
    cable_route=0.0,        # v2 (owner, 2026-10-19): 1 routes the Qwiic cable from J1's plug in a groove over the plug to the
                            # slotted arm and up its outer face (y 17.40), where neither the MCP link nor the PIP servo
                            # reaches at any PIP angle; 0 models the plug and the cable's first 3 mm only
    cable_d=2.0,            # the four 28 AWG wires as one round bundle
    cable_r=2.0,            # bend radius of the bundle's centre line
    cable_lead=0.5,         # straight out of the plug before the first bend
    cable_clear=0.2,        # the groove's clearance around the bundle
    cable_face_y=17.40,     # the bundle's centre line on the slotted arm's outer face (half of the bundle in its groove)
    cable_anchor_x=4.0, cable_anchor_z=-56.5,   # where the face groove ends, below the PIP axis (F; the arm stays put)
)
# JST SH catalogue sizes (eSH) of the board's Qwiic connector J1 and its mated plug: side-entry header SM04B-SRSS-TB 6.0 mm
# wide, 4.25 mm deep plus 0.7 mm of leads, 2.9 mm tall; housing SHR-04V-S 5.0 mm wide, 2.96 mm tall mated, 2.0 mm out of the
# header's mating face; Qwiic cable 4 x 28 AWG modelled as 0.8 mm wires, straight for 3.0 mm out of the housing.
QW = dict(hdr_w=6.0, hdr_d=4.25, lead=0.7, hdr_h=2.9, plug_w=5.0, plug_out=2.0, plug_h=2.96, wire_d=0.8, wire_run=3.0)


def j1_parts_board():
    """J1 as soldered on the ReSkin 5X (spec.json: body u -2.95..3.05, v -18.09..-13.14 including the leads, mating face
    toward the tab's end), the mated SHR-04V-S housing and the first 3 mm of the Qwiic cable, as (name, lo, hi) boxes in
    the board frame (u, v, w), w the height above the component face; and the tab's FR-4."""
    v_mate = -18.09
    uc = 0.05
    out = [("tab_fr4", (-5.0, -19.10, -0.6), (5.0, -10.10, 0.0)),
           ("j1_header", (-2.95, v_mate, 0.0), (3.05, -13.14, QW["hdr_h"])),
           ("plug", (uc - QW["plug_w"] / 2, v_mate - QW["plug_out"], 0.0), (uc + QW["plug_w"] / 2, v_mate, QW["plug_h"]))]
    v0 = v_mate - QW["plug_out"]
    if P.get("cable_route"):                    # the routed cable replaces the straight 3 mm (cable_segments)
        return out
    for k in range(4):
        u = uc + (k - 1.5) * 1.0
        out.append((f"wire{k}", (u - QW["wire_d"] / 2, v0 - QW["wire_run"], 1.4 - QW["wire_d"] / 2),
                    (u + QW["wire_d"] / 2, v0, 1.4 + QW["wire_d"] / 2)))
    return out


def j1_boxes_F(p=P):
    """J1, its plug and the cable's first 3 mm kept on the board at this design's placement (component face at the window
    plate's dorsal face, parts toward the tip, tab proximal): (name, lo, hi) boxes in F."""
    x_cf = levels(p)["x_cf"]
    out = []
    for name, lo, hi in j1_parts_board():
        if name == "tab_fr4":
            continue
        out.append((name, (x_cf - hi[2], lo[0], Z_C - hi[1]), (x_cf - lo[2], hi[0], Z_C - lo[1])))
    return out


ARM_OUT = 17.40                                 # outer face of the slotted arm (fingertip_mount_DIP_link.stl)


def cable_segments(p=P):
    """Centre line of the Qwiic cable's bundle from the plug's rear face to the anchor on the slotted arm's outer face, in F:
    straight out of the plug toward the PIP joint, a turn to +y over the plug, a run along +y to the arm, a turn to +z into
    the arm's outer face and a line up the face to the anchor below the PIP axis. Segments ('line', a, b) and
    ('arc', a, b, centre, radius, plane, t0_deg, t1_deg) with plane 'yz' (angles of (y, z) about the centre)."""
    L = levels(p)
    R, lead = p["cable_r"], p["cable_lead"]
    x = L["x_cf"] - 1.4                          # the plug's wires leave at mid-height of the housing
    y0 = 0.05                                    # J1's centre line (spec.json: body u -2.95 .. 3.05)
    z0 = Z_C + 18.09 + QW["plug_out"]            # the plug's rear face (board v -20.09)
    a1 = np.array([x, y0, z0 + lead])
    c1 = np.array([x, y0 + R, a1[2]])
    b1 = np.array([x, y0 + R, a1[2] + R])
    yf = p["cable_face_y"]
    z_run = b1[2]
    b2 = np.array([x, yf - R, z_run])
    c2 = np.array([x, yf - R, z_run + R])
    b3 = np.array([x, yf, z_run + R])
    A_ = np.array([p["cable_anchor_x"], yf, p["cable_anchor_z"]])
    return [("line", np.array([x, y0, z0]), a1), ("arc", a1, b1, c1, R, "yz", 180.0, 90.0), ("line", b1, b2),
            ("arc", b2, b3, c2, R, "yz", -90.0, 0.0), ("line", b3, A_)]


def cable_polyline(p=P, step=0.2):
    """The centre line as points about `step` mm apart."""
    pts = []
    for s in cable_segments(p):
        if s[0] == "line":
            a, b = s[1], s[2]
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            seg = [a + (b - a) * t for t in np.linspace(0, 1, n + 1)]
        else:
            _, a, b, c, r, plane, t0, t1 = s
            n = max(2, int(np.ceil(abs(np.radians(t1 - t0)) * r / step)))
            seg = [np.array([c[0], c[1] + r * np.cos(t), c[2] + r * np.sin(t)])
                   for t in np.radians(np.linspace(t0, t1, n + 1))]
        pts += seg if not pts else seg[1:]
    return np.array(pts)


def cable_mesh(p=P, sections=24):
    """The bundle as a closed tube (trimesh) along the centre line, frames carried along the line."""
    import trimesh
    C = cable_polyline(p, 0.25)
    r = p["cable_d"] / 2
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    u = np.cross(T[0], [1.0, 0, 0]) if abs(T[0][0]) < 0.9 else np.cross(T[0], [0, 1.0, 0])
    u /= np.linalg.norm(u)
    th = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    V = []
    for k, t in enumerate(T):
        u = u - np.dot(u, t) * t
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        V.append(C[k] + r * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v)))
    V = np.vstack(V + [C[:1], C[-1:]])
    n, m = len(C), sections
    F = []
    for k in range(n - 1):
        for j in range(m):
            a, b, c, d = k * m + j, k * m + (j + 1) % m, (k + 1) * m + j, (k + 1) * m + (j + 1) % m
            F += [(a, c, b), (b, c, d)]
    s0, s1 = n * m, n * m + 1
    for j in range(m):
        F += [(s0, j, (j + 1) % m), (s1, (n - 1) * m + (j + 1) % m, (n - 1) * m + j)]
    tm = trimesh.Trimesh(V, np.array(F), process=True)
    if tm.volume < 0:
        tm.invert()
    return tm


def cable_cuts(K, p=P):
    """The groove that takes the bundle: open toward the dorsal side over the plug, along the DIP link's dorsal face and
    through the slotted arm's lower corner (the cable is laid in, its far plug never threads through), then a groove
    half the bundle deep up the arm's outer face to the anchor."""
    L = levels(p)
    rc = p["cable_d"] / 2 + p["cable_clear"]
    segs = cable_segments(p)
    x = segs[0][1][0]
    x_open = 11.5                                 # past the DIP link's and the slotted arm's dorsal edges
    out = []

    def sector(c, r, t0, t1, w0, w1, n=32):
        t = np.radians(np.linspace(t0, t1, n + 1))
        outer = [(c[1] + (r + rc) * np.cos(a), c[2] + (r + rc) * np.sin(a)) for a in t]
        inner = [(c[1] + max(r - rc, 0.05) * np.cos(a), c[2] + max(r - rc, 0.05) * np.sin(a)) for a in t[::-1]]
        return K.prism(outer + inner, w0, w1, "yzx")
    _, a0, a1 = segs[0]
    out.append(K.box((x - rc, a0[1] - rc, a0[2] - 0.05), (x_open, a0[1] + rc, a1[2] + 0.01)))
    _, a, b, c, r, _, t0, t1 = segs[1]
    out.append(sector(c, r, t0, t1, x - rc, x_open))
    _, b1, b2 = segs[2]
    out.append(K.box((x - rc, b1[1] - 0.01, b1[2] - rc), (x_open, b2[1] + 0.01, b1[2] + rc)))
    _, a, b, c, r, _, t0, t1 = segs[3]
    out.append(sector(c, r, t0, t1, x - rc, x_open))
    _, b3, A_ = segs[4]
    d = np.array([A_[0] - b3[0], A_[2] - b3[2]])
    n_ = np.array([-d[1], d[0]]) / np.linalg.norm(d) * rc
    strip = [(b3[0] + n_[0], b3[2] - 0.05 + n_[1]), (A_[0] + n_[0], A_[2] + n_[1]), (A_[0] - n_[0], A_[2] - n_[1]),
             (b3[0] - n_[0], b3[2] - 0.05 - n_[1])]
    out.append(K.prism(strip, p["cable_face_y"] - rc, ARM_OUT + 1.0, "xzy"))
    out.append(K.cyl("y", (A_[0], A_[2]), rc, p["cable_face_y"] - rc, ARM_OUT + 1.0, seg=48))   # the groove's round end
    return K.union(out)


def levels(p=P):
    """x levels of the stack (mm, F)."""
    t_w = p["chip_h"] - p["chip_proud"]
    x_cf = X_BACK + t_w                         # board's component face = window plate's dorsal face
    x_bb = x_cf + p["fr4"]                      # board's bare face = clamp plate's palmar face
    x_c1 = x_bb + p["clamp_t"]
    x_ear0 = x_c1 - p["ear_t"]
    x_sh1 = x_ear0 - p["ear_gap"]
    return dict(t_w=t_w, x_cf=x_cf, x_bb=x_bb, x_c1=x_c1, x_ear0=x_ear0, x_sh1=x_sh1,
                chip_top=x_cf - p["chip_h"], sense_x=x_cf - p["chip_h"] + 0.42,
                gap=x_cf - p["chip_h"] + 0.42 - X_BACK)


def spec():
    return json.loads(SPEC.read_text())


def uv_to_yz(u, v):
    return u, -(np.asarray(v, float)) + Z_C


# board features in F
def board_items(sp=None):
    sp = sp or spec()
    b = sp["board"]
    holes = [(h["uv"][0], float(uv_to_yz(*h["uv"])[1]), h["d_mm"]) for h in b["mounting_holes"]]
    parts = {}
    for k, v in b["parts"].items():
        if not v["populated"] or k == "J1":
            continue
        y, z = uv_to_yz(*v["centre_uv"])
        parts[k] = dict(kind=v["kind"], y=float(y), z=float(z), sy=v["size_uv"][0], sz=v["size_uv"][1],
                        h=float(np.mean(v["height_mm"])))
    pads = [(pd["centre_uv"][0], float(uv_to_yz(*pd["centre_uv"])[1]), pd["size_mm"]) for pd in b["parts"]["J1"]["pads"]
            if pd["size_mm"][0] < 1.0]              # the four signal pads (0.6 x 1.55 mm) kept on the cut tab
    return dict(holes=holes, parts=parts, pads=pads)


# ----------------------------------------------------------------------------------------------- solid kernels
def _axes_T(axes):
    """3 x 4 transform taking (u, v, w) to F; axes names the F axis of u, v, w, e.g. 'zxy'."""
    T = np.zeros((3, 4))
    for i, a in enumerate(axes):
        T["xyz".index(a), i] = 1.0
    return T


def _T_F_from_stl(part="dip_link"):
    A = json.loads((REPO / "assets/mjcf/experimental/20261013-cad_v1/assembly.json").read_text())
    return np.array(A["T_F_from_part_design_mm"][part]) @ np.array(A["T_part_design_from_stl"][part])


class MeshK:
    """manifold3d: the meshes of the STL files and of every check."""
    name = "mesh"

    def __init__(self):
        import manifold3d
        self.M = manifold3d

    def box(self, lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return self.M.Manifold.cube(list(hi - lo)).translate(list(lo))

    def prism(self, poly_uv, w0, w1, axes):
        poly = np.asarray(poly_uv, float)
        a = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
        if a < 0:
            poly = poly[::-1]
        s = self.M.Manifold.extrude(self.M.CrossSection([poly]), w1 - w0).translate([0, 0, w0])
        return s.transform(_axes_T(axes))

    def cyl(self, axis, centre, r0, x0, x1, r1=None, seg=96):
        """Cylinder (or cone r0 -> r1) along 'x', 'y' or 'z' from x0 to x1 through centre (the other two coordinates)."""
        r1 = r0 if r1 is None else r1
        s = self.M.Manifold.cylinder(x1 - x0, r0, r1, seg).translate([0, 0, x0])
        c = list(centre)
        if axis == "z":
            return s.translate([c[0], c[1], 0])
        if axis == "x":
            return s.transform(_axes_T("yzx")).translate([0, c[0], c[1]])
        return s.transform(_axes_T("zxy")).translate([c[0], 0, c[1]])

    def union(self, parts):
        return self.M.Manifold.batch_boolean(list(parts), self.M.OpType.Add)

    def sub(self, a, b):
        return a - b

    def inter(self, a, b):
        return a ^ b

    def arms(self):
        """The clevis arms of fingertip_mount_DIP_link.stl in F, cut at y = -17.05 and 13.40 (R6)."""
        import dip_link_geometry as G
        dip = from_trimesh(G.load_part("dip_link"))
        big = 200.0
        a = dip ^ self.box((-big, -big, -big), (big, ARM_Y[0], big))
        b = dip ^ self.box((-big, ARM_Y[1], -big), (big, big, big))
        return self.clean(a + b)

    def clean(self, s):
        """Drop zero-volume remnants of the arms' cut planes."""
        return self.union([c for c in s.decompose() if c.volume() > 1.0])


class BrepK:
    """FreeCAD Part, inside freecadcmd: exact solids for the STEP files."""
    name = "brep"

    def __init__(self):
        import FreeCAD
        import Part
        self.F, self.Part = FreeCAD, Part

    def V(self, q):
        return self.F.Vector(*(float(c) for c in q))

    def box(self, lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        d = hi - lo
        return self.Part.makeBox(float(d[0]), float(d[1]), float(d[2]), self.V(lo))

    def prism(self, poly_uv, w0, w1, axes):
        T = _axes_T(axes)[:, :3]
        pts = [self.V(T @ np.array([u, v, w0])) for u, v in poly_uv]
        face = self.Part.Face(self.Part.makePolygon(pts + [pts[0]]))
        return face.extrude(self.V(T @ np.array([0.0, 0.0, w1 - w0])))

    def cyl(self, axis, centre, r0, x0, x1, r1=None, seg=None):
        d = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
        c0, c1 = centre
        base = {"x": (x0, c0, c1), "y": (c0, x0, c1), "z": (c0, c1, x0)}[axis]
        h = float(x1 - x0)
        if r1 is None or abs(r1 - r0) < 1e-9:
            return self.Part.makeCylinder(float(r0), h, self.V(base), self.V(d))
        return self.Part.makeCone(float(r0), float(r1), h, self.V(base), self.V(d))

    def union(self, parts):
        parts = list(parts)
        s = parts[0].fuse(parts[1:]) if len(parts) > 1 else parts[0]
        return s.removeSplitter()

    def sub(self, a, b):
        return a.cut(b)

    def inter(self, a, b):
        return a.common(b)

    def arms(self):
        import Mesh
        m = Mesh.Mesh(str(REPO / "assets/cad_v1/fingertip_mount_DIP_link.stl"))
        sh = self.Part.Shape()
        sh.makeShapeFromMesh(m.Topology, 0.001)
        so = self.Part.Solid(self.Part.Shell(sh.Faces)).removeSplitter()
        T = _T_F_from_stl("dip_link")
        so.transformShape(self.F.Matrix(*[float(v) for v in T.ravel()]))
        big = 200.0
        # cut FUSE/2 inside the new body's overlap: a box face on the arm's own inner face makes OCC drop the arm
        a = so.common(self.box((-big, -big, -big), (big, ARM_Y[0] + FUSE / 2, big)))
        b = so.common(self.box((-big, ARM_Y[1] - FUSE / 2, -big), (big, big, big)))
        self.arm_volumes = (round(a.Volume, 2), round(b.Volume, 2))
        print(f"arms: horn side {a.Volume:.2f} mm3 valid {a.isValid()}, slotted side {b.Volume:.2f} mm3 valid {b.isValid()}",
              flush=True)
        return a.fuse(b)

    def clean(self, s):
        return s


def hexagon_x(K, centre_yz, s_af, x0, x1, rot_deg=0.0):
    """Hexagonal prism along x, across flats s_af."""
    R = s_af / math.sqrt(3.0)
    pts = [(R * math.cos(math.radians(rot_deg + 60 * k)), R * math.sin(math.radians(rot_deg + 60 * k))) for k in range(6)]
    return K.prism([(centre_yz[0] + a, centre_yz[1] + b) for a, b in pts], x0, x1, "yzx")


def to_trimesh(man):
    import trimesh
    m = man.to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3].astype(float), np.asarray(m.tri_verts), process=False)


def from_trimesh(tm):
    import manifold3d as M
    return M.Manifold(M.Mesh(vert_properties=np.ascontiguousarray(tm.vertices, np.float32),
                             tri_verts=np.ascontiguousarray(tm.faces, np.uint32)))


def box(lo, hi):
    return MeshK().box(lo, hi)


def arc_z(x, r):
    return PIP[2] - np.sqrt(np.maximum(r * r - np.asarray(x, float) ** 2, 0.0))


# ----------------------------------------------------------------------------------------------- the parts
def bridge_profile(p, L):
    """The bridge's section in (z, x), extruded along y between the arms: the tip's mount face below (z -67.30), the
    palmar face flush with the tip's palmar face, the board seat at x_cf, and above it an arc about the PIP axis cut flat
    on the palmar side."""
    xf = -math.sqrt(p["bridge_r"] ** 2 - (p["bridge_flat"] - PIP[2]) ** 2)
    xs = np.linspace(L["x_cf"], xf, 80)
    top = [(float(arc_z(x, p["bridge_r"])), float(x)) for x in xs]
    return [(Z_MOUNT, X_PALM), (Z_MOUNT, L["x_cf"])] + top + [(p["bridge_flat"], X_PALM)]


def shoulder_profile(p, L):
    xs = np.linspace(X_BACK - 0.6, L["x_sh1"], 40)
    top = [(float(min(arc_z(x, p["shoulder_r"]), p["shoulder_cap"])), float(x)) for x in xs]
    return [(p["shoulder_z0"], X_BACK - 0.6), (p["shoulder_z0"], L["x_sh1"])] + top[::-1]


def window_boxes(K, p, L, items):
    """Windows through the window plate for the parts on the board's main square."""
    out = []
    for k, v in items["parts"].items():
        if v["z"] >= Z_MOUNT:
            continue
        c = p["clear_chip"] if v["kind"] == "MLX90393" else p["clear_passive"]
        hy, hz = v["sy"] / 2 + c, v["sz"] / 2 + c
        out.append(K.box((X_BACK - 0.2, v["y"] - hy, v["z"] - hz), (L["x_cf"] + 0.05, v["y"] + hy, v["z"] + hz)))
    return out


def key_poly(p, yc, flank=0.0, floor=0.0, x_top=None):
    """Dovetail key section in (x, y): root width at the back face, head width key_depth below it; flank > 0 shrinks the
    flanks inward (the tip's channel with interference), floor > 0 deepens the head face."""
    d, wr, wh = p["key_depth"], p["key_root"] / 2 - flank, p["key_head"] / 2 - flank
    slope = (wh - wr) / d                         # widening per mm toward the palmar side
    x_h = X_BACK - d - floor
    x_t = X_BACK + 0.05 if x_top is None else x_top
    half = lambda x: wr + slope * (X_BACK - x)    # noqa: E731
    return [(x_t, yc - half(x_t)), (x_t, yc + half(x_t)), (x_h, yc + half(x_h)), (x_h, yc - half(x_h))]


def keys(K, p):
    out = []
    for s in (-1, 1):
        yc = s * p["key_yc"]
        k = K.prism(key_poly(p, yc), Z_MOUNT - p["key_len"], Z_MOUNT + 0.1, "xyz")
        # detent: a ramp on the distal side, steeper on the proximal side, across the head face
        x_h = X_BACK - p["key_depth"]
        z0 = p["detent_z"]
        prof = [(z0 - 0.45, x_h + 0.01), (z0, x_h - p["detent_h"]), (z0 + 0.25, x_h + 0.01)]
        hw = p["key_head"] / 2 - 0.2
        out.append(K.union([k, K.prism(prof, yc - hw, yc + hw, "zxy")]))
    return out


def tip_cutter(K, p=P):
    """What the tip loses: two dovetail channels from the mount face, flanks 0.05 mm narrower than the keys, floor
    0.10 mm deeper, 0.10 mm longer, with a groove for each detent."""
    out = []
    for s in (-1, 1):
        yc = s * p["key_yc"]
        ch = K.prism(key_poly(p, yc, flank=p["fit_flank"], floor=p["fit_floor"], x_top=X_BACK + 0.3),
                     Z_MOUNT - p["key_len"] - p["fit_len"], Z_MOUNT + 0.3, "xyz")
        x_f = X_BACK - p["key_depth"] - p["fit_floor"]
        g = K.cyl("y", (x_f, p["detent_z"]), p["groove_r"], yc - p["key_head"] / 2, yc + p["key_head"] / 2, seg=48)
        out.append(K.union([ch, g]))
    return K.union(out)


def dip_link_new(K, p=P):
    """Everything of the DIP link but the clevis arms: bridge, window plate, shoulders, keys, with windows, pockets, wire
    channels, pin holes, countersinks and M2 pilot holes. It overlaps each arm's cut face by FUSE."""
    L = levels(p)
    it = board_items()
    parts = [K.prism(bridge_profile(p, L), ARM_Y[0] - FUSE, ARM_Y[1] + FUSE, "zxy"),
             K.box((X_BACK, -p["half_w"], Z_END), (L["x_cf"], p["half_w"], Z_MOUNT + 0.2))]
    sp = shoulder_profile(p, L)
    for s in (-1, 1):
        y0, y1 = p["shoulder_y"]
        parts.append(K.prism(sp, *(sorted((s * y0, s * y1))), "zxy"))
    parts += keys(K, p)
    body = K.union(parts)
    cuts = window_boxes(K, p, L, it)
    # one pocket under the tab for R1, R2 and the four wires' solder joints, as deep as the window plate (the bridge
    # backs it above the tip's mount face)
    tabp = [v for v in it["parts"].values() if v["z"] >= Z_MOUNT]
    py = [y for y, _, _ in it["pads"]]
    c = p["clear_passive"]
    y_lo = min([min(py) - 0.65] + [v["y"] - v["sy"] / 2 - c for v in tabp])
    y_hi = max([max(py) + 0.65] + [v["y"] + v["sy"] / 2 + c for v in tabp])
    z_lo = min(v["z"] - v["sz"] / 2 - c for v in tabp)
    cuts.append(K.box((X_BACK, y_lo, max(z_lo, Z_MOUNT + 0.04)), (L["x_cf"] + 0.05, y_hi, Z_C + 15.0 + 0.5)))
    wz = p["wire_channel"]["z"]
    for s in (-1, 1):
        y0, y1 = sorted((s * 1.5, s * p["shoulder_y"][0]))
        cuts.append(K.box((p["wire_channel"]["x0"], y0, wz[0]), (L["x_cf"] + 0.05, y1, wz[1])))
    # the clamp-plate pins' holes (proximal board holes, through) and the countersinks (distal board holes)
    r_face = (p["cs_d"] + 2 * p["cs_k"]) / 2
    for y, z, d in it["holes"]:
        if z > Z_C:
            cuts.append(K.cyl("x", (y, z), p["pin_hole_d"] / 2, X_BACK - 0.05, L["x_cf"] + 0.05))
        else:
            x0, x1 = X_BACK - 0.05, L["x_cf"] + 0.05
            cuts.append(K.cyl("x", (y, z), r_face + 0.05, x0, x1, r1=r_face - (x1 - X_BACK)))
    for s in (-1, 1):                                      # M2 pilot holes in the shoulders
        cuts.append(K.cyl("x", (s * p["m2_y"], p["m2_z"]), p["m2_pilot"] / 2, L["x_sh1"] - p["m2_depth"],
                          L["x_sh1"] + 0.05))
    if p.get("j1_kept"):                                   # a slot for J1, its plug and the cable, open through the bridge's top
        B = np.array([np.r_[lo, hi] for _, lo, hi in j1_boxes_F(p)])
        lo, hi = B[:, :3].min(0) - 0.3, B[:, 3:].max(0) + 0.3
        cuts.append(K.box((lo[0], lo[1], lo[2]), (L["x_cf"] + 0.05, hi[1], PIP[2] + 5.0)))
    return K.sub(body, K.union(cuts))


def dip_link(K, p=P):
    D = K.union([K.arms(), dip_link_new(K, p)])
    if p.get("cable_route"):                    # the cable's groove crosses the slotted arm's lower corner and outer face
        D = K.sub(D, cable_cuts(K, p))
    return K.clean(D)


def clamp_plate(K, p=P):
    L = levels(p)
    it = board_items()
    hw = p["half_w"]
    z_bp = Z_C + 10.1                                       # the board's main square, proximal edge
    parts = [K.box((L["x_bb"], -hw, Z_END), (L["x_c1"], hw, z_bp)),
             K.box((L["x_bb"], -5.05, z_bp - 0.05), (L["x_c1"], 5.05, p["tab_cover_top"]))]
    for s in (-1, 1):
        y0, y1 = sorted((s * p["shoulder_y"][0], s * hw))
        parts.append(K.box((L["x_ear0"], y0, z_bp - 0.05), (L["x_c1"], y1, p["ear_top"])))
    for y, z, d in it["holes"]:
        if z > Z_C:                                          # proximal holes: pins, a 0.2 mm lead-in at the tip
            x_tip = L["x_cf"] - p["pin_into_plate"]
            r = p["pin_d"] / 2
            pin = K.union([K.cyl("x", (y, z), r, x_tip + 0.2, L["x_bb"] + 0.05),
                           K.cyl("x", (y, z), r - 0.2, x_tip, x_tip + 0.2, r1=r)])
            if y > 0:                                         # diamond pin: bears at +-z only
                pin = K.inter(pin, K.box((x_tip - 1, y - p["diamond_half"], z - 2), (L["x_bb"] + 1, y + p["diamond_half"], z + 2)))
            parts.append(pin)
    C = K.union(parts)
    cuts = []
    for y, z, d in it["holes"]:
        if z < Z_C and p.get("cs_thread_d"):                 # distal holes: the screw forms its thread in the PLA
            cuts.append(K.cyl("x", (y, z), p["cs_thread_d"] / 2, L["x_bb"] - 0.05, L["x_c1"] + 0.05))
        elif z < Z_C:                                        # distal holes: M1.6 clearance and a nut pocket on the back
            cuts.append(K.cyl("x", (y, z), (p["cs_d"] + 0.1) / 2, L["x_bb"] - 0.05, L["x_c1"] + 0.05))
            cuts.append(hexagon_x(K, (y, z), p["nut_s"] + p["nut_clear"], L["x_c1"] - p["nut_m"] - 0.05, L["x_c1"] + 0.05, 30.0))
    for s in (-1, 1):
        cuts.append(K.cyl("x", (s * p["m2_y"], p["m2_z"]), p["m2_clear_d"] / 2, L["x_ear0"] - 0.05, L["x_c1"] + 0.05))
    return K.sub(C, K.union(cuts))


def board_model(K, p=P):
    """The ReSkin 5X at this placement: FR-4 with the tab cut to tab_len, the populated parts and the solder joints of the
    four wires."""
    L = levels(p)
    it = board_items()
    v_cut = -10.1 - p["tab_len"]
    outline_uv = [(-10, -10.1), (-5, -10.1), (-5, v_cut), (5, v_cut), (5, -10.1), (10, -10.1), (10, 10.051), (-10, 10.051)]
    fr4 = K.prism([(u, -v + Z_C) for u, v in outline_uv], L["x_cf"], L["x_bb"], "yzx")
    fr4 = K.sub(fr4, K.union([K.cyl("x", (y, z), d / 2, L["x_cf"] - 0.1, L["x_bb"] + 0.1) for y, z, d in it["holes"]]))
    comps = []
    for k, v in it["parts"].items():
        comps.append(K.box((L["x_cf"] - v["h"], v["y"] - v["sy"] / 2, v["z"] - v["sz"] / 2),
                           (L["x_cf"] + 0.01, v["y"] + v["sy"] / 2, v["z"] + v["sz"] / 2)))
    wires = []
    for y, z, (sx, sz) in it["pads"]:
        comps.append(K.box((L["x_cf"] - 0.45, y - sx / 2, z - sz / 2), (L["x_cf"] + 0.01, y + sx / 2, z + sz / 2)))
        wires.append((y, z))
    return K.union([fr4] + comps), wires


def wires_model(K, p, wires, side=+1):
    """Four 0.30 mm wires from the pads along the channel to the side of the tab, then dorsally through the clamp plate's
    notch to its back face: the start of the cable (its route beyond the DIP link is pending)."""
    L = levels(p)
    segs = []
    wz = np.mean(p["wire_channel"]["z"])
    yg = side * 0.5 * (5.05 + p["shoulder_y"][0])
    for k, (y, z) in enumerate(sorted(wires, key=lambda t: t[0])):
        x_w = L["x_cf"] - 0.35
        zz = wz + (k - 1.5) * 0.30
        pts = [(x_w, y, z), (x_w, y, zz), (x_w, yg, zz), (L["x_c1"] + 0.6, yg, zz)]
        for a, b in zip(pts[:-1], pts[1:]):
            a, b = np.array(a), np.array(b)
            d = b - a
            if float(np.linalg.norm(d)) < 1e-6:
                continue
            i = int(np.argmax(np.abs(d)))
            c = [a[j] for j in range(3) if j != i]
            segs.append(K.cyl("xyz"[i], c, 0.15, min(a[i], b[i]) - 0.15, max(a[i], b[i]) + 0.15, seg=16))
    return K.union(segs)


def fasteners(K, p=P):
    """Brass fasteners as solids: two M1.6 x 4 countersunk with their nuts, two M2 x 5 cheese heads."""
    L = levels(p)
    it = board_items()
    out = []
    r_face = (p["cs_d"] + 2 * p["cs_k"]) / 2
    for y, z, d in it["holes"]:
        if z < Z_C:
            x_r = X_BACK + (r_face - p["cs_dk"] / 2)              # the head's cylindrical rim, then its 90 deg cone
            head = K.union([K.cyl("x", (y, z), p["cs_dk"] / 2, X_BACK, x_r, seg=48),
                            K.cyl("x", (y, z), p["cs_dk"] / 2, x_r, X_BACK + p["cs_k"], r1=p["cs_d"] / 2, seg=48)])
            shank = K.cyl("x", (y, z), p["cs_d"] / 2, X_BACK + p["cs_k"] - 0.05, X_BACK + p["cs_len"], seg=32)
            nut = K.sub(hexagon_x(K, (y, z), p["nut_s"], L["x_c1"] - p["nut_m"], L["x_c1"], 30.0),
                        K.cyl("x", (y, z), p["cs_d"] / 2, L["x_c1"] - 2, L["x_c1"] + 1))
            out.append(("cs_screw", K.union([head, shank])))
            if not p.get("cs_thread_d"):
                out.append(("nut", nut))
    for s in (-1, 1):
        c = (s * p["m2_y"], p["m2_z"])
        head = K.cyl("x", c, p["m2_head_d"] / 2, L["x_c1"], L["x_c1"] + p["m2_head_k"], seg=48)
        shank = K.cyl("x", c, 1.0, L["x_c1"] - p["m2_len"], L["x_c1"] + 0.05, seg=32)
        out.append(("m2_screw", K.union([head, shank])))
    return out


def tip_F(key, sp=None):
    """A printed tip's STL (print frame) in F, and the map back."""
    import trimesh
    sp = sp or spec()
    t = sp["tips"][key]
    m = trimesh.load(REPO / t["stl"], force="mesh")
    V = np.asarray(m.vertices, float)
    if t["shape"] == "block":                                 # X = z_t from the end, Y = y, Z = depth from the back face
        P_ = np.column_stack([X_BACK - V[:, 2], V[:, 1], V[:, 0] + Z_END])
        inv = lambda Q: np.column_stack([Q[:, 2] - Z_END, Q[:, 1], X_BACK - Q[:, 0]])          # noqa: E731
    else:                                                     # upright: x_t, -y, z_t from the mount face
        P_ = np.column_stack([-V[:, 0] - 2.05, -V[:, 1], Z_MOUNT - V[:, 2]])
        inv = lambda Q: np.column_stack([-Q[:, 0] - 2.05, -Q[:, 1], Z_MOUNT - Q[:, 2]])      # noqa: E731
    tm = trimesh.Trimesh(P_, np.asarray(m.faces), process=True)
    if tm.volume < 0:
        tm.invert()
    return tm, inv


# ----------------------------------------------------------------------------------------------- build
def fname(stem, ext="stl"):
    return OUT / f"{PFX}_{stem}.{ext}"


def cmd_build(a):
    import trimesh
    OUT.mkdir(parents=True, exist_ok=True)
    sp = spec()
    L = levels()
    t0 = time.time()
    K = MeshK()
    D = dip_link(K)
    C = clamp_plate(K)
    B, wires = board_model(K)
    W = None if P.get("j1_kept") else wires_model(K, P, wires, +1)       # J1 kept: no soldered wires
    F_ = fasteners(K)
    cutter = tip_cutter(K)
    rec = dict(parameters=P, levels={k: round(v, 4) for k, v in L.items()}, parts={}, when=time.strftime("%Y-%m-%d %H:%M"))
    for name, man, role in (("dip_link", D, "DIP link: clevis arms (STL), bridge, window plate, shoulders, tip keys; PLA"),
                            ("clamp_plate", C, "dorsal clamp plate with the two locating pins; PLA, solid"),
                            ("board", B, "ReSkin 5X with the tab cut to 4.9 mm (model)"),
                            ("wires", W, "the four wires from the pads to the clamp plate's back (model)"),
                            ("tip_cutter", cutter, "volume removed from each tip: two dovetail channels and detent grooves")):
        if man is None:
            continue
        tm = to_trimesh(man)
        tm.export(fname(name))
        rec["parts"][name] = dict(file=str(fname(name).relative_to(REPO)), role=role, volume_mm3=round(float(tm.volume), 1),
                                  bounds_F=np.round(tm.bounds, 3).tolist(), watertight=bool(tm.is_watertight),
                                  triangles=int(len(tm.faces)), status=str(man.status()))
        print(f"{name}: {tm.volume:.1f} mm3, {len(tm.faces)} triangles, watertight {tm.is_watertight}, "
              f"bounds {np.round(tm.bounds, 2).tolist()}", flush=True)
    fast = []
    for k, (kind, man) in enumerate(F_):
        fast.append(to_trimesh(man))
        rec["parts"].setdefault("fasteners", dict(items=[]))["items"].append(
            dict(kind=kind, volume_mm3=round(float(fast[-1].volume), 2)))
    fm = trimesh.util.concatenate(fast)
    fm.export(fname("fasteners"))
    CTX.mkdir(parents=True, exist_ok=True)
    for kind in ("cs_screw", "nut", "m2_screw"):
        if any(k == kind for k, _ in F_):
            trimesh.util.concatenate([t for (k, _), t in zip(F_, fast) if k == kind]).export(CTX / f"fasteners_{kind}.stl")
    rec["parts"]["fasteners"]["file"] = str(fname("fasteners").relative_to(REPO))
    # the tips with their channels, in F (assembly) and in their print frames (slicing)
    rec["tips"] = {}
    for key in sp["tips"]:
        tm, inv = tip_F(key, sp)
        mt = from_trimesh(tm)
        if mt.status().name != "NoError":
            raise SystemExit(f"tip {key}: {mt.status()}")
        cut = mt - cutter
        tF = to_trimesh(cut)
        if key == P["asm_tip"]:                               # the assembly's tip; the others only in their print frame
            tF.export(fname(f"tip_{key.replace('.', 'p')}_F"))
        tP = trimesh.Trimesh(inv(np.asarray(tF.vertices)), np.asarray(tF.faces), process=False)
        if tP.volume < 0:
            tP.invert()
        tP.export(fname(f"tip_{key.replace('.', 'p')}_print"))
        rec["tips"][key] = dict(file_F=str(fname(f"tip_{key.replace('.', 'p')}_F").relative_to(REPO)) if key == P["asm_tip"] else None,
                                file_print=str(fname(f"tip_{key.replace('.', 'p')}_print").relative_to(REPO)),
                                source=sp["tips"][key]["stl"], print_orientation=sp["tips"][key]["print_orientation"],
                                volume_before_mm3=round(float(tm.volume), 1), volume_after_mm3=round(float(tF.volume), 1),
                                removed_mm3=round(float(tm.volume - tF.volume), 2), shells=len(tF.split(only_watertight=False)))
        print(f"tip {key}: removed {tm.volume - tF.volume:.2f} mm3, shells {rec['tips'][key]['shells']}", flush=True)
    # the PIP servo and the MCP link in F at PIP 0 (context for the assembly)
    import dip_link_geometry as G
    CTX.mkdir(parents=True, exist_ok=True)
    for n in ("servo_pip", "mcp_link"):
        G.load_part(n).export(CTX / f"context_{n}.stl")
    tipF = trimesh.load(fname(f"tip_{P['asm_tip'].replace('.', 'p')}_F"))
    asm = trimesh.util.concatenate([to_trimesh(D), to_trimesh(C), to_trimesh(B)] + ([to_trimesh(W)] if W is not None else []) + [fm, tipF,
                                    G.load_part("servo_pip")])
    asm.export(fname("assembly"))
    rec["parts"]["assembly"] = dict(file=str(fname("assembly").relative_to(REPO)),
                                    role="DIP link, clamp plate, board, wires, fasteners, block tip (four magnets) and the "
                                         "PIP servo in F at PIP 0")
    (OUT / f"{PFX}_parts.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"build {time.time() - t0:.1f} s -> {OUT.relative_to(REPO)}", flush=True)


# ----------------------------------------------------------------------------------------------- STEP (freecadcmd)
def freecad_step():
    """Run inside freecadcmd: the printed parts as exact solids (the arms from the original STL's facets), STEP files and a
    volume comparison with the meshes."""
    K = BrepK()
    import Mesh
    rec = {}
    t0 = time.time()
    newb = dip_link_new(K)
    cc = cable_cuts(K) if P.get("cable_route") else None     # the cable's groove crosses the slotted arm too
    parts = {"dip_link_new_features": newb.cut(cc).removeSplitter() if cc is not None else newb,
             "clamp_plate": clamp_plate(K), "tip_cutter": tip_cutter(K)}
    if cc is not None:
        parts["cable_groove_cutter"] = cc
    arms_ = K.arms()
    fused = arms_.fuse(newb)
    parts["dip_link"] = (fused.cut(cc) if cc is not None else fused).removeSplitter()
    for stem, so in parts.items():
        # the fused DIP link keeps the arms' STL facets (about 7000 faces, 23 MB): written to logs/, regenerated on demand
        dst = (CTX / f"{PFX}_{stem}.step") if stem == "dip_link" else fname(stem, "step")
        dst.parent.mkdir(parents=True, exist_ok=True)
        so.exportStep(str(dst))
        mesh_stem = {"dip_link_new_features": None, "cable_groove_cutter": None}.get(stem, stem)
        v_mesh = Mesh.Mesh(str(fname(mesh_stem))).Volume if mesh_stem else None
        rec[stem] = dict(file=str(dst.relative_to(REPO)), tracked=stem != "dip_link", valid=bool(so.isValid()),
                         solids=len(so.Solids), faces=len(so.Faces),
                         volume_mm3=round(so.Volume, 2), mesh_volume_mm3=None if v_mesh is None else round(v_mesh, 2),
                         bytes=dst.stat().st_size)
        print(f"{stem}: valid {so.isValid()}, {len(so.Solids)} solid(s), {len(so.Faces)} faces, {so.Volume:.2f} mm3 "
              f"(mesh {v_mesh}), {dst.stat().st_size / 1e6:.2f} MB [{time.time() - t0:.0f} s]", flush=True)
    rec["arms_volume_mm3"] = getattr(K, "arm_volumes", None)
    rec["note"] = ("dip_link_new_features: the exact solid of everything new (bridge, window plate, shoulders, keys and their "
                   "cuts); it overlaps the clevis arms' cut faces by 5 um. The clevis arms are those of "
                   "assets/cad_v1/fingertip_mount_DIP_link.stl at y <= -17.05 and y >= 13.40 mm. The fused DIP link (arms "
                   "and new features, one valid solid) goes to logs/ because the arms' facets make it 23 MB; the STL of the "
                   "whole part is in the study folder."
                   + (" With the routed cable, cable_groove_cutter is the groove's solid: subtract it from the DIP link (it "
                      "crosses the slotted arm's lower corner and outer face, which dip_link_new_features does not include)."
                      if cc is not None else ""))
    (OUT / f"{PFX}_step.json").write_text(json.dumps(rec, indent=1))


# ----------------------------------------------------------------------------------------------- checks
def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def section_min_wall(tm, origin, normal, res=0.02, box3=None):
    """Least wall thickness (mm) in one planar section: twice the distance transform on the section's medial ridge,
    rasterized at res; box3 = (lo, hi) in F keeps only ridge points inside that box."""
    from matplotlib.path import Path as MPath
    from scipy.ndimage import distance_transform_edt, maximum_filter
    s = tm.section(plane_origin=origin, plane_normal=normal)
    if s is None:
        return None
    p2, T = s.to_2D()
    lo, hi = p2.bounds
    nx, ny = int((hi[0] - lo[0]) / res) + 3, int((hi[1] - lo[1]) / res) + 3
    if nx * ny > 8e6:
        return None
    gx = lo[0] + res * (np.arange(nx) - 1)
    gy = lo[1] + res * (np.arange(ny) - 1)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = np.zeros(len(pts), bool)
    for poly in p2.polygons_full:
        ins = MPath(np.asarray(poly.exterior.coords)).contains_points(pts)
        for h in poly.interiors:
            ins &= ~MPath(np.asarray(h.coords)).contains_points(pts)
        inside |= ins
    inside = inside.reshape(nx, ny)
    d = distance_transform_edt(inside) * res
    ridge = (d == maximum_filter(d, size=5)) & (d > 0.1)
    Q = (np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size), np.ones(X.size)]) @ T.T)[:, :3].reshape(nx, ny, 3)
    if box3 is not None:
        b0, b1 = np.asarray(box3[0], float), np.asarray(box3[1], float)
        ridge &= np.all((Q >= b0) & (Q <= b1), axis=-1)
    if not ridge.any():
        return None
    k = np.unravel_index(int(np.argmin(np.where(ridge, d, np.inf))), d.shape)
    return dict(min_wall_mm=round(float(2 * d[k]), 3), at_F=np.round(Q[k], 2).tolist())


def cmd_check(a):
    import trimesh
    import dip_link_geometry as G
    t0 = time.time()
    L = levels()
    D = trimesh.load(fname("dip_link"))
    C = trimesh.load(fname("clamp_plate"))
    B = trimesh.load(fname("board"))
    W = trimesh.load(fname("wires")) if fname("wires").exists() else None        # none when J1 is kept
    Fm = trimesh.load(fname("fasteners"))
    res = dict(when=time.strftime("%Y-%m-%d %H:%M"))
    # R6: arms identical to the STL
    stl = G.load_part("dip_link")
    big = 200.0
    def arm_vol(tm, lo_y, hi_y):
        man = from_trimesh(tm) ^ box((-big, lo_y, -big), (big, hi_y, big))
        return man
    dv = []
    for lo_y, hi_y in ((-big, ARM_Y[0]), (ARM_Y[1], big)):
        a1, a2 = arm_vol(D, lo_y, hi_y), arm_vol(stl, lo_y, hi_y)
        dv.append(round(float((a1 - a2).volume() + (a2 - a1).volume()), 4))
    res["arms_symmetric_difference_mm3"] = dv
    # clearance over the PIP range (exact method of the spec) per part; the arms are the STL's and are reported apart
    parts = {k: v for k, v in {"dip_link_new": D, "clamp_plate": C, "board": B, "fasteners": Fm, "wires": W}.items()
             if v is not None}
    files = {"dip_link_new": "dip_link", "clamp_plate": "clamp_plate", "board": "board", "fasteners": "fasteners",
             "wires": "wires"}
    old_chk = OUT / f"{PFX}_check.json"
    prev = json.loads(old_chk.read_text()).get("clearance", {}) if (old_chk.exists() and a.reuse) else {}
    clear = {}
    for name, tm in parts.items():
        h = _sha(fname(files[name]))
        if a.skip_clearance:
            clear[name] = dict(prev.get(name, {}), stale=prev.get(name, {}).get("sha256") != h)
            continue
        if prev.get(name, {}).get("sha256") == h:
            clear[name] = prev[name]
            print(f"clearance {name}: reused {clear[name]['min_mm']} mm", flush=True)
            continue
        Pt = G.surface_points(tm, 0.25)
        if name == "dip_link_new":
            Pt = Pt[(Pt[:, 1] > ARM_Y[0] + 1e-6) & (Pt[:, 1] < ARM_Y[1] - 1e-6)]
        v, i, q, part = G.points_clearance_exact(Pt)
        clear[name] = dict(min_mm=round(v, 3), at_F=np.round(Pt[i], 2).tolist(), pip_deg=q, part=part, n_points=int(len(Pt)),
                           sha256=h)
        print(f"clearance {name}: {v:.3f} mm ({part}, PIP {q}) at {np.round(Pt[i], 2).tolist()} [{len(Pt)} pts, "
              f"{time.time() - t0:.0f} s]", flush=True)
    if "clevis_arms_unchanged" in prev:
        clear["clevis_arms_unchanged"] = prev["clevis_arms_unchanged"]
    elif not a.arms:
        clear["clevis_arms_unchanged"] = dict(
            min_mm=None, note="not recomputed (pass --arms): the arms are the STL's; the slotted arm runs against the MCP "
                              "link's bearing wall by design; the spec's keep-out gives the current DIP link 0.25 mm at "
                              "(-10.02, -16.94, -62.71), on the old bridge that this design replaces")
    else:
        Pa = G.surface_points(stl, 0.25)
        Pa = Pa[(Pa[:, 1] <= ARM_Y[0] + 1e-6) | (Pa[:, 1] >= ARM_Y[1] - 1e-6)]
        v, i, q, part = G.points_clearance_exact(Pa)
        clear["clevis_arms_unchanged"] = dict(min_mm=round(v, 3), at_F=np.round(Pa[i], 2).tolist(), pip_deg=q, part=part,
                                              n_points=int(len(Pa)))
        print(f"clearance arms: {v:.3f} mm ({part}, PIP {q})", flush=True)
    res["clearance"] = clear
    from scipy.spatial import cKDTree
    Ps, _ = G.parent_samples(("servo_pip",))
    tree = cKDTree(Ps)
    it = board_items()
    chips = {k: np.array([X_BACK + L["gap"], v["y"], v["z"]]) for k, v in it["parts"].items() if v["kind"] == "MLX90393"}
    ys, zs = np.meshgrid(np.linspace(-10, 10, 41), np.linspace(Z_C - 10.05, Z_C + 10.1, 41))
    sq = np.column_stack([np.full(ys.size, L["x_cf"]), ys.ravel(), zs.ravel()])
    best_c, best_s = (np.inf, None, None), (np.inf, None, None)
    for q in G.q_grid():
        Xc = G.rot_pip(np.stack(list(chips.values())), -q)
        dc, _ = tree.query(Xc)
        if dc.min() < best_c[0]:
            best_c = (float(dc.min()), list(chips)[int(np.argmin(dc))], float(q))
        ds, _ = tree.query(G.rot_pip(sq, -q))
        if ds.min() < best_s[0]:
            best_s = (float(ds.min()), np.round(sq[int(np.argmin(ds))], 2).tolist(), float(q))
    now = np.stack([np.array([X_BACK + 3.42, v["y"], v["z"]]) for v in it["parts"].values() if v["kind"] == "MLX90393"])
    d_now = min(float(tree.query(G.rot_pip(now, -q))[0].min()) for q in G.q_grid())
    res["to_pip_servo"] = dict(chips_min_mm=round(best_c[0], 2), chip=best_c[1], chip_pip_deg=best_c[2],
                               chips_current_plate_min_mm=round(d_now, 2),
                               main_square_min_mm=round(best_s[0], 2), main_square_at_F=best_s[1], main_square_pip_deg=best_s[2],
                               spec_R10_mm=6.3)
    print(f"to PIP servo: chips {best_c}, main square {best_s}", flush=True)
    win = 0.0
    for k, v in it["parts"].items():
        if v["z"] < Z_MOUNT:
            c = P["clear_chip"] if v["kind"] == "MLX90393" else P["clear_passive"]
            y0, y1 = max(v["y"] - v["sy"] / 2 - c, -Y_HALF), min(v["y"] + v["sy"] / 2 + c, Y_HALF)
            win += max(0.0, y1 - y0) * (v["sz"] + 2 * c)
    res["windows_in_back_face_mm2"] = dict(area=round(win, 1), back_face_block=round(2 * Y_HALF * (Z_MOUNT - Z_END), 1),
                                           spec_R2=55.0)
    # static gaps between the new parts and the horn's slot (the horn sits in y -21.25..-19.25 of the horn-side arm)
    res["horn_slot_to_new_parts_mm"] = round(float(min(D.vertices[(D.vertices[:, 1] > ARM_Y[0] + 1e-6), 1].min(),
                                                       C.vertices[:, 1].min(), B.vertices[:, 1].min()) - (-19.25)), 3)
    # walls in sections through the thin features
    walls = {}
    inner = ((-11.0, ARM_Y[0] + 0.3, -95.0), (14.0, ARM_Y[1] - 0.3, -40.0))      # the new features, between the arms
    sec = {"window_plate_x6.88": (D, (X_BACK + L["t_w"] / 2, 0, 0), (1, 0, 0), inner),
           f"keys_z{Z_MOUNT - 2.7:g}": (D, (0, 0, Z_MOUNT - 2.7), (0, 0, 1), inner),
           "shoulders_z-65.35": (D, (0, 0, P["m2_z"]), (0, 0, 1), inner),
           "bridge_y-5": (D, (0, -5.0, 0), (0, 1, 0), inner),
           "bridge_y0": (D, (0, 0.0, 0), (0, 1, 0), inner),
           "bridge_y12": (D, (0, 12.0, 0), (0, 1, 0), inner),
           "clamp_z-86.06": (C, (0, 0, Z_C - 7.9), (0, 0, 1), None),
           "clamp_z-65.35": (C, (0, 0, P["m2_z"]), (0, 0, 1), None),
           "clamp_y8.1": (C, (0, P["m2_y"], 0), (0, 1, 0), None)}
    for k, (tm, o, n, b3) in sec.items():
        walls[k] = section_min_wall(tm, o, n, box3=b3)
        print(f"wall {k}: {walls[k]}", flush=True)
    res["walls"] = walls
    # TPU between each tip's key channels and its outer surface, in sections near the mount face
    cutter = tip_cutter(MeshK())
    tw = {}
    for key in [k for k in spec()["tips"] if k != "block5"]:          # dip_link_board_mount adds the five-magnet block
        tF = to_trimesh(from_trimesh(tip_F(key)[0]) - cutter)
        tw[key] = {f"z{z:g}": section_min_wall(tF, (0, 0, z), (0, 0, 1)) for z in (Z_MOUNT - 0.3, Z_MOUNT - 2.2, Z_MOUNT - 4.5)}
        print(f"tip walls {key}: {tw[key]}", flush=True)
    res["tip_walls_near_channels"] = tw
    # masses
    vD, vC = float(D.volume), float(C.volume)
    rho_link = 0.9482e-3        # g/mm^3, the current DIP link at its sliced density (spec)
    rho_pla = 1.24e-3
    rho_brass = 8.5e-3
    vF = float(Fm.volume)
    res["mass_g"] = dict(dip_link_as_built_density=round(vD * rho_link, 2), dip_link_solid=round(vD * rho_pla, 2),
                         clamp_plate_solid=round(vC * rho_pla, 2), fasteners_brass=round(vF * rho_brass, 3),
                         board=0.64, total_without_tip=round(vD * rho_link + vC * rho_pla + vF * rho_brass + 0.64, 2),
                         total_solid_dip=round(vD * rho_pla + vC * rho_pla + vF * rho_brass + 0.64, 2),
                         limit_R12=7.6, current_dip_link_with_board=round(4.92 + 0.64, 2))
    res["volumes_mm3"] = dict(dip_link=round(vD, 1), clamp_plate=round(vC, 1), fasteners=round(vF, 2))
    # widths over the tip's length and dorsal extent
    def over_tip(tm):
        return tm.vertices[(tm.vertices[:, 2] < Z_MOUNT - 0.01) & (tm.vertices[:, 2] > Z_END - 0.01)]

    def width(V):
        return [round(float(V[:, 1].min()), 3), round(float(V[:, 1].max()), 3)] if len(V) else None
    Vd = over_tip(D)
    new_d = Vd[(Vd[:, 1] > ARM_Y[0] + 0.01) & (Vd[:, 1] < ARM_Y[1] - 0.01)]
    arm_d = Vd[Vd[:, 1] <= ARM_Y[0] + 0.01]
    res["width_over_tip_length_y"] = dict(
        dip_link_new=width(new_d), clamp_plate=width(over_tip(C)), board=width(over_tip(B)), fasteners=width(over_tip(Fm)),
        horn_side_arm_below_mount_face=dict(y=width(arm_d), z_min=round(float(arm_d[:, 2].min()), 2) if len(arm_d) else None,
                                            note="the unchanged clevis arm of the STL reaches 2.75 mm below the tip's mount "
                                                 "face plane, 9.6 mm beside the tip"))
    res["dorsal_extent_x"] = dict(over_tip_length=round(float(max(over_tip(C)[:, 0].max(), over_tip(Fm)[:, 0].max())), 3),
                                  clamp_plate=round(float(C.vertices[:, 0].max()), 3),
                                  m2_heads_above_mount_face=round(float(Fm.vertices[:, 0].max()), 3),
                                  readiness_board_back=11.05)
    # interference between the assembled parts (should be zero apart from intended contacts)
    tk = P["asm_tip"].replace(".", "p")
    tipF = trimesh.load(fname(f"tip_{tk}_F"))
    pairs = {}
    for n1, m1, n2, m2 in (("dip_link", D, "board", B), ("dip_link", D, "clamp_plate", C), ("clamp_plate", C, "board", B),
                           ("dip_link", D, f"tip_{tk}", tipF), ("board", B, f"tip_{tk}", tipF),
                           ("clamp_plate", C, f"tip_{tk}", tipF)):
        v = float((from_trimesh(m1) ^ from_trimesh(m2)).volume())
        pairs[f"{n1}|{n2}"] = round(v, 4)
    res["overlap_mm3"] = pairs
    print(f"overlaps {pairs}", flush=True)
    (OUT / f"{PFX}_check.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"check {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------------------------- sensing (.venv)
def cmd_sense(a):
    """The readiness study's readout model (point dipoles over the 85A finite elements' magnet motions, five chips, one
    141 Hz sample, Cramer-Rao bound with the contact position and a uniform background unknown) at this design's sensing
    plane; the only input changed from the spec's touch row is the gap."""
    import sim2real_fingertip_sensing as SS
    import tpu_tip_design as TD
    import tpu_tip_magnet_sensing as MS
    L = levels()
    rows = [json.loads(l) for l in open(READY / "fe/fem.jsonl")]
    gaps = {"design": L["gap"], "chips_0.80": L["gap"] + (P["chip_h"] - 0.80), "chips_1.00": L["gap"] - (1.00 - P["chip_h"]),
            "plate_now": 3.42}
    out = []
    for r in rows:
        if r["layout"].endswith("_w1"):
            continue
        ys, zs, Gp = MS.patch_grid(r)
        zc = min(zs, key=lambda z: abs(z - TD.Z_C))
        i0, j0 = ys.index(0.0), zs.index(zc)
        for name, gap in gaps.items():
            an = MS.analyse(r, "board", gap)
            ch = MS.chips(r, "board", gap)
            Bc = MS.B0(r, ch) * 1e3
            bmax = float(np.abs(Bc).max())
            g, rng = SS.gain_for(bmax)
            both = [q["sigF_n_unk_bg"] for q in an["points"]]
            Sg = {k: np.array([[MS.signal(r, Gp[(y, z)][k], ch) for z in zs] for y in ys]) for k in ("normal", "shear_y", "shear_z")}
            J = np.concatenate([np.stack([Sg["normal"][i0, j0], Sg["shear_y"][i0, j0], Sg["shear_z"][i0, j0],
                                          MS.deriv(Sg["normal"][:, j0], ys, i0), MS.deriv(Sg["normal"][i0, :], zs, j0)], 1),
                                np.tile(np.eye(3), (len(ch), 1))], 1)
            sig = np.tile(MS.NOISE, len(ch))
            feq = {}
            for k, ax in enumerate("xyz"):
                e = np.zeros(3)
                e[k] = 1e-6
                dB = (MS.B0(r, [q + e for q in ch]) - MS.B0(r, [q - e for q in ch])).ravel() / 2 * 1e6
                est = np.linalg.lstsq(J / sig[:, None], dB / sig, rcond=None)[0]
                feq[ax] = dict(normal=round(abs(float(est[0])), 4), shear=round(float(np.hypot(est[1], est[2])), 4))
            # chip sensing points in F and the distance to the nearest magnet centre
            mags = [np.array([-(TD.X_FACE - r["depth_mm"] - t / 2) - 2.05, y, Z_MOUNT - z])      # design frame to F
                    for d, t, y, z in r["magnets"]]
            chipsF = [np.array([X_BACK + gap, u, Z_C - v]) for u, v in
                      [(0, 0), (0, 7), (7, 0), (0, -7), (-7, 0.051)]]
            dist = [round(float(min(np.linalg.norm(c - m) for m in mags)), 3) for c in chipsF]
            out.append(dict(layout=r["layout"], recipe=r["recipe"], magnets=len(r["magnets"]), case=name, gap_mm=round(gap, 3),
                            centre_force_N=round(an["centre"]["sigF_n_unk_bg"], 4),
                            median_force_N=round(float(np.median(both)), 4),
                            range_force_N=[round(min(both), 4), round(max(both), 4)],
                            B_chips_mT=np.round(Bc, 3).tolist(), B_U1_mT=round(float(np.linalg.norm(Bc[0])), 3),
                            B_max_component_mT=round(bmax, 3), gain_sel=g,
                            headroom=None if rng is None else round(rng / bmax, 2),
                            chip_to_nearest_magnet_centre_mm=dist, false_force_N_per_um=feq,
                            magnets_F=[np.round(m, 3).tolist() for m in mags]))
            print(f"{r['layout']:14s} {r['recipe']} {name:10s} gap {gap:.3f}: centre {an['centre']['sigF_n_unk_bg']:.4f} N, "
                  f"B max {bmax:.2f} mT (GAIN_SEL {g}), chip-magnet {min(dist):.2f} mm", flush=True)
    # static field of every printed tip's magnets at the five chips (design frame of the readout: x toward the palmar
    # face, the chip normal), at the spec's 0.42 mm (check against its print sheets) and at this design's gap
    sp = spec()
    static = {}
    for key, t in sp["tips"].items():
        mg = t["magnets"]
        Dm, tm_ = mg["size_mm"]
        Vm = math.pi * (Dm / 2e3) ** 2 * (tm_ / 1e3)
        mom = np.array([MS.BR * Vm / (4e-7 * math.pi), 0.0, 0.0])
        mags_t = [np.array([-m["centre_F"][0] - 2.05, m["centre_F"][1], Z_MOUNT - m["centre_F"][2]]) * 1e-3
                  for m in mg["items"]]                      # F to the readout's design frame
        rowk = {}
        for name, gap in (("spec_touch_0.42", 0.42), ("design", L["gap"])):
            chips_t = [np.array([TD.X_BACK - gap, u, TD.Z_C + v]) * 1e-3 for u, v in
                       [(0, 0), (0, 7), (7, 0), (0, -7), (-7, 0.051)]]
            B = np.array([sum(MS.dipole(c, pm, mom) for pm in mags_t) for c in chips_t]) * 1e3
            chipsF = [np.array([X_BACK + gap, u, Z_C - v]) for u, v in [(0, 0), (0, 7), (7, 0), (0, -7), (-7, 0.051)]]
            dist = [round(float(min(np.linalg.norm(c - np.array(m["centre_F"])) for m in mg["items"])), 3) for c in chipsF]
            bmax = float(np.abs(B).max())
            g, rng = SS.gain_for(bmax)
            rowk[name] = dict(gap_mm=round(gap, 3), B_chips_mT=np.round(B, 3).tolist(), B_U1_mT=round(float(np.linalg.norm(B[0])), 3),
                              B_max_component_mT=round(bmax, 3), gain_sel=g, chip_to_nearest_magnet_centre_mm=dist)
        rowk["spec_chip_field_touch_mT"] = t["chip_field_touch"]["B_chips_mT"]
        static[key] = rowk
        print(f"static {key:7s}: U1 {rowk['design']['B_U1_mT']:.2f} mT at {L['gap']:.2f} mm (spec 0.42: "
              f"{rowk['spec_touch_0.42']['B_U1_mT']:.2f}, sheet {t['chip_field_touch']['B_U1_mT']:.2f}), chip-magnet "
              f"{min(rowk['design']['chip_to_nearest_magnet_centre_mm']):.2f} mm", flush=True)
    (OUT / f"{PFX}_sense.json").write_text(json.dumps(dict(
        rows=out, static_fields=static, gaps_mm={k: round(v, 3) for k, v in gaps.items()},
        model="scripts/tpu_tip_magnet_sensing.py analyse (as scripts/dip_link_spec.py sensing): point dipoles N52 1.45 T, "
              "magnet motions of docs/experiments/20261016-sim2real_fingertip_readiness/fe/fem.jsonl, five MLX90393 at "
              "GAIN_SEL 7 noise, one 141 Hz sample, Cramer-Rao bound with the contact position and a uniform background "
              "field unknown",
        changed_input="gap only: the chips' sensing plane at this design's window plate (package height less 0.05 mm) "
                      "instead of the spec's 0.42 mm; chip positions in the board's plane as the spec",
        when=time.strftime("%Y-%m-%d %H:%M")), indent=1, default=float))


# ----------------------------------------------------------------------------------------------- figures (.venv)
COLORS = {"dip_link": "0.90 0.90 0.86 1", "clamp_plate": "0.36 0.56 0.85 1", "board": "0.12 0.48 0.24 1",
          "wires": "0.85 0.30 0.10 1", "fasteners": "0.80 0.65 0.25 1", "tip": "0.92 0.78 0.55 1",
          "servo": "0.25 0.25 0.28 1", "mcp": "0.62 0.62 0.66 0.28"}
W_IMG, H_IMG, FOVY = 1600, 1200, 28.0


def _mjcf(meshes, w, h):
    assets, geoms = [], []
    for k, (name, path, rgba, off) in enumerate(meshes):
        assets.append(f'<mesh name="m{k}" file="{path}" scale="0.001 0.001 0.001"/>')
        geoms.append(f'<geom type="mesh" mesh="m{k}" rgba="{rgba}" pos="{off[0] * 1e-3:.5f} {off[1] * 1e-3:.5f} '
                     f'{off[2] * 1e-3:.5f}" contype="0" conaffinity="0"/>')
    return f"""<mujoco><compiler inertiafromgeom="false"/>
<visual><global offwidth="{w}" offheight="{h}" fovy="{FOVY}"/><quality shadowsize="4096" offsamples="8"/>
<headlight ambient="0.42 0.42 0.42" diffuse="0.48 0.48 0.48" specular="0.05 0.05 0.05"/></visual>
<asset><texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>{''.join(assets)}</asset>
<worldbody><light pos="0.04 -0.06 0.02" dir="-0.4 0.6 -0.3" diffuse="0.40 0.40 0.40" castshadow="false"/>
<light pos="0.05 0.05 -0.03" dir="-0.5 -0.5 0.2" diffuse="0.22 0.22 0.22" castshadow="false"/>
{''.join(geoms)}</worldbody></mujoco>"""


def _obj(path):
    """An OBJ copy of a mesh for the renders, its vertices split at edges sharper than 30 deg, with vertex normals: MuJoCo
    would otherwise average normals across the long thin triangles of a flat face and smear its shading."""
    import trimesh
    src = Path(path)
    dst = CTX / "render" / (src.stem + ".obj")
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        dst.parent.mkdir(parents=True, exist_ok=True)
        m = trimesh.load(src, force="mesh")
        sm = trimesh.graph.smooth_shade(m, angle=np.radians(30.0), facet_minarea=None)
        dst.write_text(trimesh.exchange.obj.export_obj(sm, include_normals=True, include_texture=False))
    return str(dst)


def _render(meshes, cam, dst, w=W_IMG, h=H_IMG):
    import mujoco
    from PIL import Image
    meshes = [(n, _obj(pth), rgba, off) for n, pth, rgba, off in meshes]
    lookat, distance, azimuth, elevation = cam
    m = mujoco.MjModel.from_xml_string(_mjcf(meshes, w, h))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    c = mujoco.MjvCamera()
    c.lookat[:] = np.asarray(lookat) * 1e-3
    c.distance, c.azimuth, c.elevation = distance * 1e-3, azimuth, elevation
    with mujoco.Renderer(m, h, w) as ren:
        ren.update_scene(d, camera=c)
        img = ren.render()
    Image.fromarray(img).save(dst, optimize=True)
    return dst


def _project(P, cam, w=W_IMG, h=H_IMG):
    """Pixel coordinates (u right, v down) of points P (n, 3) in F (mm) for MuJoCo's free camera."""
    lookat, dist, az, el = cam
    a, e = math.radians(az), math.radians(el)
    fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    up = np.array([-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e)])
    right = np.cross(fwd, up)
    eye = np.asarray(lookat, float) - dist * fwd
    f = (h / 2) / math.tan(math.radians(FOVY) / 2)
    Q = np.atleast_2d(P) - eye
    zc = Q @ fwd
    return np.column_stack([w / 2 + f * (Q @ right) / zc, h / 2 - f * (Q @ up) / zc])


def _annotate(path, cam, labels, title=None, callout=False):
    """Labels on a render, each (text, point in F (mm), text position as fractions of the image). With callout, a numbered
    circle marks each point and a legend under the image gives the texts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    img = Image.open(path).convert("RGB")
    W0, H0 = img.size                                          # the camera's image size, for the projection
    a_ = np.asarray(img)
    rows = np.where((a_ < 245).any(axis=(1, 2)))[0]
    cut = int(min(a_.shape[0], rows.max() + 40)) if len(rows) else a_.shape[0]
    img = img.crop((0, 0, a_.shape[1], cut))                  # white margin under the parts
    W, H = img.size
    line = 46.0
    extra = (len(labels) + 0.6) * line if callout else 0.0
    fig = plt.figure(figsize=(W / 150, (H + extra) / 150), dpi=150)
    ax = fig.add_axes([0, extra / (H + extra), 1, H / (H + extra)])
    ax.imshow(img)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    for k, (text, pF, tpos) in enumerate(labels, 1):
        u, v = _project(np.array([pF]), cam, W0, H0)[0]
        tx, ty = tpos if tpos is not None else (u / W, v / H)
        if callout:
            ax.plot([u], [v], "o", ms=27, mfc="white", mec="k", mew=1.6)
            ax.text(u, v, str(k), ha="center", va="center", fontsize=17, family="serif", weight="bold")
        else:
            ax.annotate(text, xy=(u, v), xytext=(tx * W, ty * H), fontsize=17, family="serif",
                        arrowprops=dict(arrowstyle="-", lw=1.3, color="k", shrinkA=0, shrinkB=2),
                        ha="left" if tx < 0.5 else "right", va="center")
            ax.plot([u], [v], "o", ms=4, color="k")
    if callout:
        lg = fig.add_axes([0, 0, 1, extra / (H + extra)])
        lg.set_xlim(0, W)
        lg.set_ylim(extra, 0)
        lg.axis("off")
        for k, (text, _, _) in enumerate(labels, 1):
            lg.text(0.03 * W, (k - 0.4) * line, f"{k}   {text}", fontsize=17, family="serif", va="center")
    if title:
        ax.text(0.02 * W, 0.035 * H, title, fontsize=19, family="serif", va="top")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cmd_render(a):
    MEDIA.mkdir(parents=True, exist_ok=True)
    rel = lambda s_: str(fname(s_))         # noqa: E731
    ctx = lambda s_: str(CTX / f"{s_}.stl")  # noqa: E731
    L = levels()
    it = board_items()
    hd = sorted([(y, z) for y, z, _ in it["holes"] if z < Z_C])          # distal holes, -y first
    C_ = COLORS
    asm = [("dip_link", rel("dip_link"), C_["dip_link"], (0, 0, 0)),
           ("clamp_plate", rel("clamp_plate"), C_["clamp_plate"], (0, 0, 0)),
           ("board", rel("board"), C_["board"], (0, 0, 0)),
           ("wires", rel("wires"), C_["wires"], (0, 0, 0)),
           ("fasteners", rel("fasteners"), C_["fasteners"], (0, 0, 0)),
           ("tip", rel("tip_block4_F"), C_["tip"], (0, 0, 0)),
           ("servo", ctx("context_servo_pip"), C_["servo"], (0, 0, 0)),
           ("mcp", ctx("context_mcp_link"), C_["mcp"], (0, 0, 0))]
    # assembled, from the dorsal side and the slotted arm's side
    cam = ((5.0, 0.0, -71.0), 112, 228, -20)
    iso = _render(asm, cam, MEDIA / f"{PFX}_iso.png")
    _annotate(iso, cam, [
        ("Clamp plate, 3.0 mm", (L["x_c1"], -3.0, -80.0), (0.03, 0.60)),
        ("M2 screw into a shoulder", (L["x_c1"] + P["m2_head_k"], -P["m2_y"], P["m2_z"]), (0.03, 0.42)),
        ("M1.6 nut in the clamp plate", (L["x_c1"], hd[0][0], hd[0][1]), (0.03, 0.80)),
        ("Board edge, 0.5 mm inside the rim", (L["x_cf"] + 0.3, 10.0, -82.0), (0.97, 0.78)),
        ("Window plate, 0.85 mm", (X_BACK + 0.4, P["half_w"], -86.5), (0.97, 0.92)),
        ("Tip, 85A block", (-2.0, Y_HALF, -80.0), (0.97, 0.64)),
        ("Wires to the cable", (L["x_c1"] + 0.6, 5.5, -65.1), (0.97, 0.46)),
        ("Clevis arm as in the STL", (0.0, 17.4, -56.0), (0.97, 0.13)),
        ("PIP servo", (21.95, -6.0, -52.0), (0.03, 0.22))],
        title="Assembled at PIP 0 deg, seen from the dorsal side")
    # exploded: countersunk screws palmar, board, clamp plate, nuts and M2 screws dorsal, tip distal
    ex = [("dip_link", rel("dip_link"), C_["dip_link"], (0, 0, 0)),
          ("cs", ctx("fasteners_cs_screw"), C_["fasteners"], (-12, 0, 0)),
          ("tip", rel("tip_block4_F"), C_["tip"], (0, 0, -18)),
          ("board", rel("board"), C_["board"], (9, 0, 0)),
          ("wires", rel("wires"), C_["wires"], (9, 0, 0)),
          ("clamp_plate", rel("clamp_plate"), C_["clamp_plate"], (18, 0, 0)),
          ("nut", ctx("fasteners_nut"), C_["fasteners"], (25, 0, 0)),
          ("m2", ctx("fasteners_m2_screw"), C_["fasteners"], (27, 0, 0))]
    cam = ((9.0, 0.0, -78.0), 150, 110, -12)
    exploded = _render(ex, cam, MEDIA / f"{PFX}_exploded.png")
    _annotate(exploded, cam, [
        ("DIP link: window plate with chip windows", (L["x_cf"], -6.0, -83.0), None),
        ("Shoulder with an M2 pilot hole", (L["x_sh1"], -P["m2_y"], P["m2_z"] + 1.6), None),
        ("M1.6 countersunk screws, driven from the palmar side", (X_BACK - 12 + 0.3, hd[0][0], hd[0][1]), None),
        ("Tip with two key channels", (-1.0, -Y_HALF, -80.0 - 18), None),
        ("Board, chips toward the tip, wires", (L["x_bb"] + 9, -6.0, -84.0), None),
        ("Clamp plate with two pins", (L["x_c1"] + 18, -6.0, -84.0), None),
        ("M1.6 nuts, pressed into the clamp plate", (L["x_c1"] + 25, hd[0][0], hd[0][1]), None),
        ("M2 screws", (L["x_c1"] + 27 + P["m2_head_k"], -P["m2_y"], P["m2_z"]), None)],
        title="Exploded: window plate, board and clamp plate", callout=True)
    # the DIP link alone from the palmar side, below: window plate, keys, bridge, countersinks
    cam = ((4.0, 0.0, -75.0), 95, 25, 28)
    dl = _render([("dip_link", rel("dip_link"), C_["dip_link"], (0, 0, 0))], cam, MEDIA / f"{PFX}_dip_link_palmar.png")
    _annotate(dl, cam, [
        ("Bridge: the tip's mount face rests here", (-6.0, -12.0, Z_MOUNT), None),
        ("Dovetail keys with detents", (X_BACK - P["key_depth"], -P["key_yc"], -71.0), None),
        ("Pin hole for the clamp plate, 2.40 mm", (X_BACK, 8.0, Z_C + 8.1), None),
        ("U1 window; the chip stands 0.05 mm proud", (X_BACK, 0.0, Z_C - 1.2), None),
        ("Countersink for an M1.6 screw", (X_BACK, hd[1][0], hd[1][1]), None)],
        title="DIP link from the palmar side", callout=True)
    section_figure(MEDIA / f"{PFX}_section.png")
    print("figures ->", MEDIA.relative_to(REPO), flush=True)


def section_figure(dst):
    """Three sections: y = 0 through U4, U1, U2; y = 3.0 through two magnets and a key; z = -78.16 through U5, U1, U3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from matplotlib.patches import Polygon as MPoly, Rectangle
    L = levels()
    sp = spec()
    mags = sp["tips"]["block4"]["magnets"]["items"]
    parts = [("Tip, 85A block", fname("tip_block4_F"), "#e8c88a"), ("DIP link", fname("dip_link"), "#d6d6cc"),
             ("Board (FR-4, chips, parts)", fname("board"), "#1f7a3a"), ("Clamp plate", fname("clamp_plate"), "#5b8fd9"),
             ("Brass screws and nuts", fname("fasteners"), "#c8a33c"), ("Wires", fname("wires"), "#d9531e")]
    meshes = [(n, trimesh.load(p), c) for n, p, c in parts]
    plt.rcParams.update({"font.family": "serif", "font.size": 16})
    fig, axs = plt.subplots(1, 3, figsize=(19.5, 9.0), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.15]))
    panels = [((0, 0.0, 0), (0, 1, 0), (0, 2), "Section y = 0 (U4, U1, U2)", (-11.5, 12.5), (-90.0, -61.0)),
              ((0, 3.0, 0), (0, 1, 0), (0, 2), "Section y = 3.0 mm (magnets, key)", (-11.5, 12.5), (-90.0, -61.0)),
              ((0, 0, Z_C), (0, 0, 1), (1, 0), "Section z = -78.16 mm (U5, U1, U3)", (-12.0, 12.0), (-11.5, 12.5))]
    for ax, (orig, nrm, (h_ax, v_ax), title, xl, yl) in zip(axs, panels):
        for n, tm, c in meshes:
            s = tm.section(plane_origin=orig, plane_normal=nrm)
            if s is None:
                continue
            for poly in s.discrete:
                ax.add_patch(MPoly(poly[:, [h_ax, v_ax]], closed=True, fc=c, ec="k", lw=0.5))
        side = nrm[1] == 1
        if side:
            ax.plot([X_PALM], [Z_C], "kx", ms=12, mew=2)
            for m in mags:
                (x0, x1), y, z = m["x_F"], m["centre_F"][1], m["centre_F"][2]
                if abs(y - orig[1]) < 1.5:
                    ax.add_patch(Rectangle((x0, z - 1.5), (x1 - x0) / 2, 3.0, fc="#cc2222", ec="k", lw=0.5))
                    ax.add_patch(Rectangle((x0 + (x1 - x0) / 2, z - 1.5), (x1 - x0) / 2, 3.0, fc="#9a9a9a", ec="k", lw=0.5))
            ax.axvline(X_BACK + L["gap"], color="k", lw=1.0, ls=":")
            ax.axvline(X_BACK + 3.42, color="#777777", lw=1.0, ls="--")
            ax.set_xlabel("x in F (mm), dorsal to the right")
            ax.set_ylabel("z in F (mm)")
        else:
            for m in mags:
                (x0, x1), y = m["x_F"], m["centre_F"][1]
                ax.add_patch(Rectangle((y - 1.5, x0), 3.0, x1 - x0, fc="none", ec="#cc2222", lw=1.2, ls="--"))
            ax.axhline(X_BACK + L["gap"], color="k", lw=1.0, ls=":")
            ax.set_xlabel("y in F (mm)")
            ax.set_ylabel("x in F (mm), dorsal up")
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=17)
        ax.grid(True, lw=0.3, alpha=0.5)
    a0 = axs[0]
    a0.annotate(f"Sensing plane, {L['gap']:.2f} mm\nbehind the back face", xy=(X_BACK + L["gap"], -73.4), xytext=(-9.8, -71.0),
                fontsize=15, arrowprops=dict(arrowstyle="->", lw=1.0))
    a0.annotate("Sensing plane of the\ncurrent plate, 3.42 mm", xy=(X_BACK + 3.42, -88.0), xytext=(-10.5, -89.6),
                fontsize=15, va="bottom", arrowprops=dict(arrowstyle="->", lw=1.0, color="#555555"), color="#333333")
    axs[1].annotate("Dovetail key\nwith detent", xy=(X_BACK - 1.0, -70.5), xytext=(-9.0, -66.5), fontsize=15,
                    arrowprops=dict(arrowstyle="->", lw=1.0))
    axs[1].annotate("Magnets (N half red)", xy=(-4.55, -81.66), xytext=(-10.5, -87.8), fontsize=15,
                    arrowprops=dict(arrowstyle="->", lw=1.0))
    axs[2].annotate("Magnets at y = +-3 mm\n(outline, z -74.66 / -81.66)", xy=(3.0, -4.6), xytext=(-11.5, 1.5), fontsize=15,
                    arrowprops=dict(arrowstyle="->", lw=1.0))
    from matplotlib.lines import Line2D
    leg = [MPoly([[0, 0]], fc=c, ec="k", label=n) for n, _, c in parts] + \
          [Rectangle((0, 0), 1, 1, fc="#cc2222", ec="k", label="Magnet 3 x 2 mm"),
           Line2D([0], [0], color="k", marker="x", ls="", ms=10, mew=2, label="Contact point")]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=16, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(dst, dpi=110)
    plt.close(fig)
    print(dst.relative_to(REPO), flush=True)


def cmd_eval(a):
    """evaluation.json from parts.json, check.json and sense.json (model values; bench values pending)."""
    L = levels()
    sp = spec()
    parts = json.loads((OUT / f"{PFX}_parts.json").read_text())
    chk = json.loads((OUT / f"{PFX}_check.json").read_text())
    sen = json.loads((OUT / f"{PFX}_sense.json").read_text())
    rel = lambda q: str(Path(q).relative_to(REPO)) if Path(q).is_absolute() else str(q)        # noqa: E731

    def row(layout, recipe, case):
        return next(r for r in sen["rows"] if r["layout"] == layout and r["recipe"] == recipe and r["case"] == case)

    res = {}
    for nm, lay in (("four", "four_d3x2_d5"), ("five", "five_d3x2_d5")):
        for rc in ("w85_30", "w85_50"):
            d, now = row(lay, rc, "design"), row(lay, rc, "plate_now")
            lo, hi = row(lay, rc, "chips_1.00"), row(lay, rc, "chips_0.80")
            res[f"{nm}_{rc}"] = dict(centre_force_N=d["centre_force_N"], median_force_N=d["median_force_N"],
                                     range_over_15_positions_N=d["range_force_N"],
                                     current_plate_centre_force_N=now["centre_force_N"],
                                     chips_1p00_to_0p80_centre_force_N=[lo["centre_force_N"], hi["centre_force_N"]],
                                     B_U1_mT=d["B_U1_mT"], B_max_component_mT=d["B_max_component_mT"], gain_sel=d["gain_sel"],
                                     headroom=d["headroom"], false_force_N_per_um=d["false_force_N_per_um"])
    static = sen["static_fields"]
    c2m = {k: dict(per_chip_U1_U2_U3_U4_U5_mm=v["design"]["chip_to_nearest_magnet_centre_mm"],
                   min_mm=min(v["design"]["chip_to_nearest_magnet_centre_mm"]),
                   current_plate_min_mm=round(min(v["design"]["chip_to_nearest_magnet_centre_mm"]) + 0.0, 3))
           for k, v in static.items()}
    # the current plate's chip-to-magnet distance for the same tips (sensing plane 3.42 mm behind the back face)
    for k, t in sp["tips"].items():
        chipsF = [np.array([X_BACK + 3.42, u, Z_C - v]) for u, v in [(0, 0), (0, 7), (7, 0), (0, -7), (-7, 0.051)]]
        c2m[k]["current_plate_min_mm"] = round(float(min(min(np.linalg.norm(c - np.array(m["centre_F"]))
                                                              for m in t["magnets"]["items"]) for c in chipsF)), 3)
    # beam estimate of the support under the tip: window plate, board and clamp plate as a cantilever from the bridge,
    # 10 N at the contact point; bounds without and with shear connection between the layers; the current 3.0 mm plate
    E_pla, E_fr4, F_, Lc = 3500.0, 20000.0, 10.0, (P["m2_z"] - 0.65) - Z_C
    lay = [(E_pla, 2 * P["half_w"], L["t_w"], X_BACK), (E_fr4, 20.0, P["fr4"], L["x_cf"]), (E_pla, 2 * P["half_w"], P["clamp_t"], L["x_bb"])]
    EI_sep = sum(E * b * t ** 3 / 12 for E, b, t, _ in lay)
    EA = [(E * b * t, x0 + t / 2) for E, b, t, x0 in lay]
    xc = sum(a * x for a, x in EA) / sum(a for a, _ in EA)
    EI_comp = EI_sep + sum(a * (x - xc) ** 2 for a, x in EA)
    EI_now = E_pla * 14.8 * 3.0 ** 3 / 12
    L_now = -64.0 - Z_C
    stiff = dict(load_N=F_, lever_mm=round(Lc, 2),
                 deflection_um=[round(1e3 * F_ * Lc ** 3 / (3 * EI_comp), 1), round(1e3 * F_ * Lc ** 3 / (3 * EI_sep), 1)],
                 current_plate_um=round(1e3 * F_ * L_now ** 3 / (3 * EI_now), 1),
                 note="Euler-Bernoulli cantilever from the M2 screws' line, E 3.5 GPa (PLA) and 20 GPa (FR-4); the first value "
                      "with the layers bonded by the clamp, the second with them free to slide; root compliance not included")
    clear = chk["clearance"]
    new_parts = {k: v for k, v in clear.items() if k != "clevis_arms_unchanged"}
    rigid = {k: v for k, v in new_parts.items() if k != "wires"}
    kmin = min(rigid, key=lambda k: rigid[k]["min_mm"])
    mass = chk["mass_g"]
    E = dict(
        title="DIP link with a window plate and a dorsal clamp plate (concept a)",
        date=DATE, status="model; nothing printed; bench values pending",
        script="scripts/dip_link_concept_a.py", spec=str(SPEC.relative_to(REPO)),
        concept=(f"The DIP link's plate behind the tip becomes a {L['t_w']:.2f} mm window plate. The five MLX90393 reach "
                 f"through its windows and stand {P['chip_proud']:.2f} mm proud of the tip's back face, so their sensing "
                 f"plane lies {L['gap']:.2f} mm behind it (3.42 mm on the current plate). The board lies on the window "
                 "plate's dorsal face; a 3.0 mm dorsal clamp plate locates it with two pins through its proximal holes into "
                 "the window plate and clamps it with two M1.6 countersunk brass screws through its distal holes and two M2 "
                 "brass screws through the clamp plate's ears into the DIP link's shoulders beside the cut tab. The tip "
                 "slides onto two dovetail keys under the bridge; a detent on each key stops it sliding off. The clevis "
                 "arms are the STL's own facets."),
        terms=dict(
            F="finger frame of the spec (index finger, zero joint angles), mm: x dorsal, y along the PIP axis, z up",
            gap="distance from the tip's undeformed back face (x 6.45) to the chips' sensing plane, 0.42 mm under the "
                "package top, mm",
            force_resolution="Cramer-Rao bound (one standard deviation, N) of the normal force at the contact point from "
                             "one 141 Hz sample of the five chips, contact position and a uniform background field unknown",
            clearance="least signed distance (mm) from a part fixed to the DIP link to the MCP link and the PIP servo over "
                      "PIP -18 to 92 deg (dip_link_geometry.points_clearance_exact, 0.25 mm samples, 0.5 deg steps)",
            false_force="normal force or shear (N) the readout's estimator reads per micrometre of rigid board motion "
                        "relative to the magnets"),
        files=dict(
            step={k: dict(file=v["file"], tracked=v["tracked"], valid=v["valid"], faces=v["faces"])
                  for k, v in json.loads((OUT / f"{PFX}_step.json").read_text()).items() if isinstance(v, dict)},
            stl={k: v["file"] for k, v in parts["parts"].items() if "file" in v},
            tips_print_frame={k: v["file_print"] for k, v in parts["tips"].items()},
            figures=[rel(q) for q in sorted(MEDIA.glob("*.png"))],
            parts_json=rel(OUT / f"{PFX}_parts.json"), check_json=rel(OUT / f"{PFX}_check.json"),
            sense_json=rel(OUT / f"{PFX}_sense.json")),
        stack_x_F=dict(tip_back_face=X_BACK, chip_tops=round(L["chip_top"], 3), window_plate=[X_BACK, round(L["x_cf"], 3)],
                       board_fr4=[round(L["x_cf"], 3), round(L["x_bb"], 3)], clamp_plate=[round(L["x_bb"], 3), round(L["x_c1"], 3)],
                       clamp_ears=[round(L["x_ear0"], 3), round(L["x_c1"], 3)], shoulders_to=round(L["x_sh1"], 3),
                       sensing_plane=round(X_BACK + L["gap"], 3)),
        chip_to_magnet_mm=dict(
            definition="distance from each chip's sensing point to the nearest magnet's centre, mm",
            design_min_block_four=c2m["block4"]["min_mm"], current_plate_min_block_four=c2m["block4"]["current_plate_min_mm"],
            per_tip=c2m),
        board_to_tip_gap_mm=dict(
            design=round(L["gap"], 3), current_plate=3.42,
            chip_tops_proud_of_back_face=P["chip_proud"], fr4_component_face_behind_back_face=round(L["t_w"], 3),
            with_chips_1p00_or_0p80_and_this_window_plate=[round(L["gap"] - 0.10, 3), round(L["gap"] + 0.10, 3)],
            note=("the window plate is the measured package height less 0.05 mm (parameter chip_h); 0.80-1.00 mm is the "
                  "datasheet range")),
        field_at_chips_mT=dict(
            frame="readout frame: x toward the palmar face (the chip normal), y, z along the finger; rows U1, U2, U3, U4, U5",
            per_tip={k: dict(B_chips_mT=v["design"]["B_chips_mT"], B_U1_mT=v["design"]["B_U1_mT"],
                             B_max_component_mT=v["design"]["B_max_component_mT"], gain_sel=v["design"]["gain_sel"],
                             B_U1_at_spec_touch_mT=v["spec_touch_0.42"]["B_U1_mT"]) for k, v in static.items()}),
        force_resolution_N=dict(
            at_contact_point=res,
            method=sen["model"], inputs_changed=sen["changed_input"],
            summary=(f"four magnets, 85A 30 % gyroid: {res['four_w85_30']['centre_force_N']:.3f} N "
                     f"(current plate {res['four_w85_30']['current_plate_centre_force_N']:.3f} N); five magnets: "
                     f"{res['five_w85_30']['centre_force_N']:.3f} N (current plate "
                     f"{res['five_w85_30']['current_plate_centre_force_N']:.3f} N)")),
        retention=dict(
            board=dict(
                x=("clamped between the window plate (component face, around the windows) and the clamp plate (bare face): "
                   "two M1.6 x 4 countersunk brass screws through the window plate's countersinks and the board's distal "
                   "holes into brass nuts in the clamp plate's back, and two M2 x 5 brass screws through the clamp plate's "
                   f"ears into the shoulders; the ears stop {P['ear_gap']:.1f} mm short of the shoulders so the M2 tension "
                   "presses the clamp plate onto the board's proximal margin and the tab instead of onto the shoulders. The "
                   f"chips stand {P['chip_proud']:.2f} mm proud, so the mounted tip presses each chip, and the board under "
                   "it, onto the clamp plate"),
                y_z=("two pins of the clamp plate (2.35 mm) through the board's proximal holes (2.40 mm) into 2.40 mm "
                     "reamed holes of the window plate: 0.025 mm radial clearance; the pin at +y is relieved to "
                     f"+-{P['diamond_half']:.1f} mm in y (diamond) and fixes rotation only; friction from the screw preload "
                     "stops slip inside the pin clearance"),
                play_removal=("no designed clearance in the clamp stack (window plate, FR-4, clamp plate in contact); the "
                              "0.1-0.2 mm PLA tolerances fall into the 0.3 mm ear gap, the self-centring countersinks, the "
                              "M1.6 clearance holes (1.7 mm) and the reamed pin holes; the pin diameter is set from a test "
                              "print against the board's measured holes"),
                no_force_on_parts="the clamp plate bears on the bare face only; the window plate touches the component "
                                  "face outside windows that clear the chips by 0.20 mm and C1-C5, R1-R2 by 0.25 mm"),
            tip=dict(
                x="back face on the window plate and the chip tops; the dovetail flanks (channel 0.05 mm narrower per flank) "
                  "wedge the tip toward the window plate and resist pull-off",
                y="key flanks",
                z=f"+z: mount face on the bridge; -z: a {P['detent_h']:.2f} mm detent on each key in a {P['groove_r']:.2f} mm "
                  "groove of the channel floor",
                removal="pull the tip along -z; the board's four screws stay untouched",
                to_magnets_mm=round(float(X_BACK - P["key_depth"] - P["fit_floor"] - P["groove_r"] -
                                          max(m["x_F"][1] for t in sp["tips"].values() for m in t["magnets"]["items"])), 2)),
            fasteners=["2 x M1.6 x 4 countersunk, DIN 963 / ISO 2009, brass", "2 x M1.6 hex nut, ISO 4032, brass, pressed in",
                       "2 x M2 x 5 cheese head, DIN 84, brass, thread-forming into 1.7 mm pilots"],
            tolerances="PLA 0.1-0.2 mm: reamed 2.40 mm pin holes, 1.7 mm M2 pilots, 1.7 mm M1.6 clearance, 0.15 mm nut-pocket "
                       "clearance, 0.3 mm ear gap"),
        clearance=dict(
            min_new_rigid_mm=rigid[kmin]["min_mm"], at=kmin, part=rigid[kmin]["part"], pip_deg=rigid[kmin]["pip_deg"],
            per_part=new_parts, clevis_arms_unchanged=clear.get("clevis_arms_unchanged"),
            horn=dict(min_mm=chk["horn_slot_to_new_parts_mm"],
                      note="static: the horn sits in the horn-side arm's slot (y -21.25..-19.25) and turns with the DIP link"),
            to_pip_servo=chk["to_pip_servo"], requirement="0.5 mm rigid, 1.0 mm cable (R7)"),
        lateral=dict(board_half_width=10.0, tip_half_width=Y_HALF, board_past_tip_per_side=round(10.0 - Y_HALF, 2),
                     rigid_rim_half_width=P["half_w"], rigid_past_tip_per_side=round(P["half_w"] - Y_HALF, 2),
                     width_over_tip_length=chk["width_over_tip_length_y"], total_width_mm=round(2 * P["half_w"], 2)),
        contact_geometry_change=(
            "Tip outer surface unchanged. Each tip gains two dovetail channels at its back face, 5.1 mm long from the "
            "mount face, 1.9 mm deep, 2.3-3.5 mm wide, at y +-3.6 mm (58.3 mm^3 of TPU), which needs a reprint. Over the "
            f"tip's length the rigid parts span |y| <= {P['half_w']:.1f} mm (21.0 mm, the R5 limit; the readiness scenes' "
            "board was 20.0 mm) and the FR-4 edge lies 0.5 mm inside them; the clamp plate's back is at x "
            f"{L['x_c1']:.2f} mm over the tip's length (the readiness board's back at 11.05 mm); the M1.6 screws end inside "
            "their nuts and the M2 heads (to x 12.20 mm) sit above the tip's mount face."),
        printability=dict(
            dip_link=("PLA, upright with the tip end up: the bridge's mount face is a top surface and the window plate "
                      f"({L['t_w']:.2f} mm, two perimeters), the keys and the shoulders rise from it as vertical walls; chip "
                      "windows bridge at most 3.4 mm; supports only under the clevis arms and the bridge's face toward the "
                      "PIP axis. The window plate's faces print as perimeters, not as top or bottom surfaces (the spec's "
                      "rule for seat faces): their flatness is the printer's XY accuracy. Ream the pin holes to 2.40 mm and "
                      "the pilots to 1.7 mm"),
            clamp_plate="PLA or PETG, solid, back face on the bed: the board face is a top surface, the pins print vertical, "
                        "the nut pockets open on the bed side; no supports",
            tips="block: back face on the bed as now, the channels open to the bed and widen upward, their floors bridge 3.5 "
                 "mm; curved tips: upright as now, channels vertical",
            min_walls=chk["walls"], tip_tpu_walls_near_channels=chk.get("tip_walls_near_channels"), walls_rule_mm=0.8),
        mass_g=mass,
        support_stiffness=stiff,
        assembly=[
            "Measure the chips' package height h on the owner's board and set chip_h = h in the generator (window plate "
            "h - 0.05 mm); print the DIP link and the clamp plate.",
            "Ream the window plate's two pin holes to 2.40 mm and the shoulders' pilots to 1.7 mm; check that the clamp "
            "plate's pins slide into the board's proximal holes without play (reprint at +-0.05 mm if not).",
            "Desolder J1, cut the tab 4.9 mm from the main square (v = -15.0 mm), solder four wires (VDD, GND, SDA, SCL; "
            "AWG 32-36) to the signal pads, leading to the +y side.",
            "Lay the board component side down on the window plate: chips in their windows, R1-R2 and the solder joints in "
            "the bridge's pockets, the wires in the channel and up through the gap beside the tab.",
            "Put on the clamp plate (pins through the board into the window plate) and drop two M1.6 nuts into its back "
            "pockets.",
            "With the tip off, drive two M1.6 x 4 countersunk screws from the palmar side through the window plate and the "
            "board's distal holes into the nuts; snug, about 0.05 N m.",
            "Drive two M2 x 5 screws through the clamp plate's ears into the shoulders; snug, the ears still clear of the "
            "shoulders.",
            "Fix the wires in the channel with a dab of hot glue or Kapton tape (removable, so the board lifts out with its "
            "pigtail) so a cable pull loads the DIP link; route the cable "
            "along the clamp plate's proximal edge to y > 17.4 mm and across the PIP joint outside the slotted arm.",
            "Slide the tip along +z onto the keys until its mount face meets the bridge and both detents engage; read the "
            "zero-load field. Tip exchange: pull along -z, the board's screws stay in."],
        parts_list=[
            dict(item="DIP link", material="PLA", qty=1, file=parts["parts"]["dip_link"]["file"]),
            dict(item="Clamp plate", material="PLA or PETG, solid", qty=1, file=parts["parts"]["clamp_plate"]["file"]),
            dict(item="M1.6 x 4 countersunk screw, DIN 963 / ISO 2009", material="brass", qty=2),
            dict(item="M1.6 hex nut, ISO 4032", material="brass", qty=2),
            dict(item="M2 x 5 cheese-head screw, DIN 84", material="brass", qty=2),
            dict(item="ReSkin 5X board, tab cut to 4.9 mm, J1 removed, four wires", material="FR-4 0.6 mm", qty=1),
            dict(item="85A tip with two key channels (one per tip family)", material="Siraya TPU 85A", qty=1,
                 file=parts["tips"]["block4"]["file_print"]),
            dict(item="Hot glue or Kapton tape for the wires' strain relief", material="", qty=1)],
        requirements=dict(
            R1=dict(status="met", value=f"sensing plane {L['gap']:.2f} mm behind the back face (limit 0.52)"),
            R2=dict(status="met in the model; flatness pending a print",
                    value=f"windows {chk['windows_in_back_face_mm2']['area']} mm^2 of the block's "
                          f"{chk['windows_in_back_face_mm2']['back_face_block']} mm^2 (spec 55 mm^2 with 0.20 mm around "
                          "C1-C5; 0.25 mm here for their solder fillets); chip tops 0.05 mm proud"),
            R3=dict(status="met in the model; slip and creep pending the bench", value="four screws, two pins, 0.025 mm radial"),
            R4=dict(status="met in the model", value="dovetail keys, bridge, detents; tip off without touching the board"),
            R5=dict(status="met", value="outer surface unchanged; 21.0 mm wide over the tip's length; FR-4 edge 0.5 mm inside"),
            R6=dict(status="met", value=f"arms' symmetric difference {chk['arms_symmetric_difference_mm3']} mm^3 "
                                        f"(5 um fusion overlap)"),
            R7=dict(status="met" if rigid[kmin]["min_mm"] >= 0.5 and new_parts["wires"]["min_mm"] >= 1.0 else "not met",
                    value=f"rigid {rigid[kmin]['min_mm']} mm ({kmin}), wires {new_parts['wires']['min_mm']} mm"),
            R8=dict(status="met on the DIP link; cable route past the joint pending",
                    value="tab 4.9 mm, wires taped or hot-glued in the DIP link's channel"),
            R9=dict(status="met", value="tip and board change with a screwdriver, DIP link stays on the finger; one part "
                                        "for all fingers"),
            R10=dict(status="met for the chips; the FR-4's proximal corner is 0.15 mm under the 6.3 mm threshold",
                     value=f"brass fasteners; chips {chk['to_pip_servo']['chips_min_mm']} mm from the PIP servo over the range "
                           f"(current plate {chk['to_pip_servo']['chips_current_plate_min_mm']} mm); FR-4 corner "
                           f"{chk['to_pip_servo']['main_square_min_mm']} mm at PIP {chk['to_pip_servo']['main_square_pip_deg']} "
                           "deg, as for any board at the touch placement"),
            R11=dict(status="met except the seat faces print as perimeters", value="see printability"),
            R12=dict(status="met at the links' print settings" if mass["total_without_tip"] <= 7.6 else "not met",
                     value=f"{mass['total_without_tip']} g with the DIP link at the current part's sliced density (15 % "
                           f"infill); {mass['total_solid_dip']} g if it prints solid")),
        designed_overlaps_mm3=dict(
            keys_in_tip_channels=chk["overlap_mm3"]["dip_link|tip_block4"],
            chips_into_tip=chk["overlap_mm3"]["board|tip_block4"],
            note="intended: the channels are 0.05 mm narrower per flank than the keys, and the chips stand 0.05 mm proud "
                 "of the window plate; every other pair of parts has zero overlap"),
        risks=[
            f"The window plate is a {L['t_w']:.2f} mm wall printed vertically; its flatness and the chips' 0.05 mm "
            "protrusion depend on the printer and on the measured package height. A chip more than about 0.1 mm proud "
            "lifts the TPU off the window plate around it.",
            "The M2 screws form threads in PLA: a few board changes before the threads wear; PLA creep lowers the preload "
            "(bench check of the spec: zero-load field before and after 50 presses at 3 N and after 1 h at 5 N).",
            "The window plate, board and clamp plate form a cantilever from the bridge; at 10 N it deflects tens of "
            "micrometres (beam estimate, as the current 3.0 mm plate), elastic and inside the tip's calibration.",
            "Every tip needs a reprint with the two channels; the tips printed on 2026-10-16 have none.",
            "The cable past the PIP joint is not modelled; the bend beside the tab needs thin wire (AWG 32-36).",
            "The detent's retention force in 85A TPU is unmeasured."],
        owner_decisions=["Tab cut to 4.9 mm with soldered wires (this concept needs it; J1 overlaps the MCP link at this "
                         "placement).", "Tip reprint with two dovetail channels at the back face.",
                         "Brass M1.6 countersunk screws and nuts, M2 cheese-head screws (stock pending)."],
        pending=["Package height of the five chips and FR-4 thickness on the owner's board (sets the window plate).",
                 "Flatness of a printed window plate (vertical perimeters).", "Detent pull-off force on an 85A print.",
                 "Retention and creep of the clamp (spec's bench check).", "Cable gauge and route past the PIP joint."],
        when=time.strftime("%Y-%m-%d %H:%M"))
    (OUT / f"{PFX}_evaluation.json").write_text(json.dumps(E, indent=1, default=float))
    print(json.dumps(dict(gap=E["board_to_tip_gap_mm"]["design"], c2m=E["chip_to_magnet_mm"]["design_min_block_four"],
                          force=E["force_resolution_N"]["summary"], clearance=E["clearance"]["min_new_rigid_mm"],
                          mass=mass["total_without_tip"]), indent=1), flush=True)


def main():
    if os.environ.get("DIPA_CMD") == "step":
        freecad_step()
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "check", "sense", "render", "eval"):
        sp_ = sub.add_parser(c)
        if c == "check":
            sp_.add_argument("--reuse", action="store_true", help="reuse clearance rows whose part STL is unchanged")
            sp_.add_argument("--arms", action="store_true", help="also sweep the unchanged clevis arms (slow)")
            sp_.add_argument("--skip-clearance", action="store_true", help="walls, masses and overlaps only")
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check, "sense": cmd_sense, "render": cmd_render,
     "eval": lambda a: cmd_eval(a)}[a.cmd](a)


if __name__ == "__main__" or os.environ.get("DIPA_CMD") == "step":
    main()
