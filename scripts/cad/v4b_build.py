#!/usr/bin/env python3
"""v4b index finger in mixed manufacture (2026-11-20): printed bodies, a laser-cut PIP linkage, a yaw stage held at
both ends of the yaw servo and a carriage for the index gantry. It replaces the all-acrylic build of 2026-11-19
(commit 823a33fbb).

The yaw servo sits in the printed carriage right behind the gimbal G, shaft toward the palm, long end up. The printed
yoke wraps around it: its front arm is bolted to the servo's disc horn, its rear arm turns in a ball bearing on a dowel
held by the carriage behind the servo's base, and a web under the servo joins the two arms. The MCP servo stands in the
yoke's collar with its shaft on the MCP axis; the printed proximal link is bolted to its disc horn and turns on a dowel
in the yoke's idler arm on the other side. The PIP servo lies in the link's cradle, shaft dorsal; its horn drives the DIP
link through the acrylic crank H1, a printed knuckle (pins a and b) and the acrylic rod R1 with a spherical plain bearing
on the lug pin. The carriage bolts to the gantry's MGN7C block and lead nut where the v3 carriage did, so the travel box
and the stroke stay as built.

Frames and units: mm, the finger's mount frame (x dorsal, y lateral, z up; origin at v3's yaw-axis point at the centre of
the index travel box; the carriage moves along x). Each part is placed in its body's frame at the zero pose. A printed
part is a union of extruded outlines and boxes; the clearance check cuts every outline into 1 mm strips (boxes).

check [--fine]  clearances between the bodies over the yaw, MCP and PIP ranges; the module's half-width
stroke          the carriage and the finger against the gantry (beam, bearing blocks, coupler, stepper) over the stroke
pip             the PIP drive along the PIP angle
loads           the yaw stage's support forces and the levers from the lead nut, against v3 and the acrylic v4b
dxf             the acrylic parts on a sheet (cut/20261119-v4b_acrylic_3mm.dxf)
printed         STEP (assets/cad_v4/v4b_build/step, local) and STL (data/) of the printed parts, through FreeCAD
bom, keepout, render, mjcf
Outputs go to docs/experiments/20261119-v4b_build/ and assets/mjcf/experimental/20261120-v4b_mixed/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402

from sr2hand.geometry.rotations import mat_to_quat, rot_x, rot_y  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/20261119-v4b_build"
DATA, MEDIA, CUT = OUT / "data", OUT / "media", OUT / "cut"
STEP_DIR = ROOT / "assets/cad_v4/v4b_build/step"  # CAD exchange files stay on the workstation (.gitignore)
MJCF_DIR = ROOT / "assets/mjcf/experimental/20261120-v4b_mixed"
GANTRY_MESHES = ROOT / "assets/mjcf/experimental/20261104-cad_v3/meshes"
PARAMS = ROOT / "assets/cad_v4/concepts/20261118-layout_params.json"
PJ = json.loads(PARAMS.read_text())
S32 = PJ["servos"]["sts3032"]
T = 3.175   # 1/8-inch acrylic
TP = 3.5    # printed plate
X, Y, Z = np.eye(3)
CHORD = 0.3  # mm, arc discretisation
P_INDEX = np.array([56.1812, 58.3997, 0.0])  # the index mount (travel-box centre) in the palm frame P (assets/cad_v3)

# ------------------------------------------------------------------------------------------------- parameters
B = {
    "Gz": -7.0,             # gimbal centre G below v3's yaw axis (mount frame z)
    "proximal": 66.0,       # G to the PIP axis (the acrylic build had 60)
    "half_width": 28.4,     # index and middle mounts come within 56.8 mm
    "yaw_spline_x": 17.0,   # yaw servo's spline top on the yaw axis, shaft toward -x, long end up (the acrylic build had 37)
    "mcp_spline_y": -18.0,  # MCP servo's spline top on the MCP axis, shaft toward -y, long end up
    "idler_gap": 2.5,       # MCP servo's base to the link's idler side
    "l2_y3_gap": 0.9,       # link idler side to the yoke's idler arm
    "cross_r": 13.0,        # the link's merge block: its top face below G
    "block_depth": 9.0,     # and its depth
    "yaw_rear_gap": 2.5,    # yaw servo's base to the yoke's rear arm
    "web": {"r": 12.5, "t": 3.0, "w": 14.0},  # the yoke's web under the yaw servo: inner face below the yaw axis, thickness, width
    "rear_wall_t": 4.0,
    "cut_clear": 0.1,       # servo case cutouts and pockets: clearance per side
    "pip": {"spline_x": 13.0, "spline_z": -34.0, "spacer": 5.0, "r": 12.0, "lug_r": 9.0, "lug_y": 0.0, "pip_rest": 0.0,
            "knuckle_e": 5.0, "rod_offset_c": 5.5, "horn_limit_deg": 80.0, "long_end": "down"},  # PIP servo's long end toward the PIP
    "disc": {"d": 18.0, "flange": 2.0, "hub_d": 7.0, "hub_h": 2.5, "bolt_r": 6.0},
    "mf63": {"D": 6.0, "B": 2.5, "flange_d": 7.2, "flange_t": 0.6},
    "sph": {"D": 12.0, "W": 4.8, "H": 3.6},
    "shoulder": {"d": 3.0, "head_d": 5.0, "head_h": 1.3, "thread": 4.0},  # McMaster 90323A: low head, precision, M2 x 4
    "nut": {"corners": 4.62, "t": 1.6},  # M2 hex nut: inside a bearing's 6 mm bore it bears on the inner ring alone
    "boss": {"d": 4.0, "h": 0.5},  # printed boss that bears on a bearing's inner ring and keeps the part off the flange
    "dip_cheek_y": (3.7, 6.1),  # printed DIP link's cheeks (|y|): 0.3 from the PIP bearings' flanges
    "lower_y": 7.0,         # the link's PIP cheeks: inner faces (|y|)
    "cheek_out_y": 10.5,    # and outer faces
    "tip_box": {"x": (-11.0, 9.0), "y": 10.5, "z": (-44.0, -20.0)},  # cassette and cap below the PIP axis (PIP 0)
    # the carriage (mount frame; the travel box's centre)
    "carriage": {"x0": -10.5, "col_z": (20.0, 46.9), "plate_x1": 5.3, "plate_y": (-19.5, -14.5), "plate_top": 65.0,
                 "rail_y": (11.0, 15.0), "rail_top": 40.0, "rail_low": 0.5, "roof_top": 21.4, "front_roof_x0": 15.8,
                 "pocket_y": (6.1, 11.0), "pocket_low_rel": -4.0, "tab_gap_rel": 22.0},
}
# the gantry's parts that the carriage attaches to (mount frame at the travel box's centre; scripts/cad/cad_v3 meshes)
LEAD_NUT = {"axis": (0.0, 31.96), "barrel_x": (-1.7, 5.0), "barrel_d": 8.4, "flange_x": (5.0, 8.3), "flange_d": 20.3,
            "hole_r": 6.3, "hole_deg": (90.0, 210.0, 330.0), "hole_d": 2.75}
BLOCK = {"lo": (-14.8, -14.5, 47.4), "hi": (7.7, -8.0, 64.4), "holes_xz": ((-6.7, 49.95), (1.3, 49.95), (-6.7, 61.95), (1.3, 61.95))}
STROKE = (-32.58, 30.0)  # carriage travel along x from the box centre: the far stop and the motor-end stop (home)
GANTRY = [  # the index gantry's beam-side parts below the beam, as boxes (mount frame with the carriage at the box centre)
    ("beam", (-55.7, -7.5, 47.4), (101.2, 7.5, 71.8)),
    ("rail", (-55.5, -11.3, 52.4), (44.5, -6.5, 59.4)),
    ("palmar end", (-55.7, -7.5, 41.9), (-50.0, 7.5, 47.4)),
    ("palmar bearing block", (-50.0, -7.5, 24.4), (-44.0, 7.5, 47.4)),
    ("dorsal bearing block", (38.0, -7.5, 24.4), (44.0, 7.5, 47.4)),
    ("motor end", (44.0, -10.0, 41.9), (101.2, 10.0, 47.4)),
    ("stepper plate", (70.0, -10.0, 21.9), (74.0, 10.0, 41.9)),
    ("coupler", (44.0, -9.5, 22.4), (69.0, 9.5, 41.4)),
    ("stepper", (72.6, -10.1, 21.9), (101.2, 9.9, 45.4)),
    ("lead screw", (-53.9, -2.5, 29.1), (46.6, 2.5, 34.1)),
]
YAWS = (-85.0, -60.0, -30.0, 0.0, 30.0, 60.0, 85.0)
MCP_R = (-15.0, 92.0)
PIPS = (-25.0, 0.0, 20.0, 45.0, 70.0, 92.0)


# ------------------------------------------------------------------------------------------------- 2D outlines
def arc(c, r, a0, a1):
    n = max(2, int(math.ceil(abs(a1 - a0) * r / CHORD)) + 1)
    t = np.linspace(a0, a1, n)
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1)


def hull_circles(circles):
    """Convex hull of circles [(cx, cy, r), ...] as a dense ccw polygon."""
    from scipy.spatial import ConvexHull
    pts = np.concatenate([arc((cx, cy), r, 0.0, 2 * math.pi)[:-1] for cx, cy, r in circles])
    return pts[ConvexHull(pts).vertices]


def rrect(u0, v0, u1, v1, r=1.0):
    r = max(1e-3, min(r, (u1 - u0) / 2 - 1e-6, (v1 - v0) / 2 - 1e-6))
    pts = [arc((u1 - r, v0 + r), r, -math.pi / 2, 0.0), arc((u1 - r, v1 - r), r, 0.0, math.pi / 2),
           arc((u0 + r, v1 - r), r, math.pi / 2, math.pi), arc((u0 + r, v0 + r), r, math.pi, 1.5 * math.pi)]
    return np.concatenate(pts)


def poly_round(verts, r=1.0):
    """A polygon (ccw vertices) with every corner rounded by up to r (dense)."""
    V = np.asarray(verts, float)
    n = len(V)
    out = []
    for i in range(n):
        p0, p1, p2 = V[i - 1], V[i], V[(i + 1) % n]
        a, b = p0 - p1, p2 - p1
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        a, b = a / la, b / lb
        ang = math.acos(float(np.clip(a @ b, -1, 1)))
        if ang < 1e-3 or abs(ang - math.pi) < 1e-3 or r <= 0:
            out.append(p1[None])
            continue
        rr = min(r, 0.45 * la * math.tan(ang / 2), 0.45 * lb * math.tan(ang / 2))
        d = rr / math.tan(ang / 2)
        bis = (a + b) / np.linalg.norm(a + b)
        c = p1 + bis * rr / math.sin(ang / 2)
        s, e = p1 + a * d, p1 + b * d
        a0, a1 = math.atan2(s[1] - c[1], s[0] - c[0]), math.atan2(e[1] - c[1], e[0] - c[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi  # the short way round
        out.append(arc(c, rr, a0, a0 + da))
    return np.concatenate(out)


def signed_area(P):
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def intervals(P, v):
    """Intervals of u where the horizontal line at v lies inside polygon P."""
    u1, v1 = P[:, 0], P[:, 1]
    u2, v2 = np.roll(u1, -1), np.roll(v1, -1)
    m = ((v1 <= v) & (v < v2)) | ((v2 <= v) & (v < v1))
    xs = np.sort(u1[m] + (v - v1[m]) * (u2[m] - u1[m]) / (v2[m] - v1[m]))
    return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]


def subtract(iv, cut):
    out = []
    for a, b in iv:
        segs = [(a, b)]
        for c0, c1 in cut:
            nxt = []
            for s0, s1 in segs:
                if c1 <= s0 or c0 >= s1:
                    nxt.append((s0, s1))
                else:
                    if c0 > s0:
                        nxt.append((s0, c0))
                    if c1 < s1:
                        nxt.append((c1, s1))
            segs = nxt
        out += segs
    return out


class Flat:
    """A plate: outline (polygon in (u, v)), round holes (u, v, d) and polygon holes, placed in its body's frame. One
    face lies on o + u U + v V; the plate extends t along +N = U x V, or along -N with flip=True. kind is "acrylic"
    (laser-cut, t = 1/8 in) or "printed" (a piece of the printed part `part`)."""

    def __init__(self, name, body, outline, o, U, V, holes=(), polys=(), qty=1, flip=False, note="", t=T, kind="acrylic",
                 part=None):
        self.name, self.body, self.qty, self.note, self.kind = name, body, qty, note, kind
        self.part = part or name
        self.printed = kind == "printed"
        self.outline = outline if signed_area(outline) > 0 else outline[::-1]
        self.holes = [tuple(float(x) for x in h) for h in holes]
        self.polys = [np.asarray(p, float) for p in polys]
        self.o, self.U, self.V = np.asarray(o, float), np.asarray(U, float), np.asarray(V, float)
        self.N = np.cross(self.U, self.V)
        self.t = t
        self.w0 = -t if flip else 0.0

    def strips(self, dv=1.0):
        """The outline cut into boxes dv tall, with the holes left out, in the body frame."""
        P = self.outline
        v0, v1 = P[:, 1].min(), P[:, 1].max()
        out = []
        R = np.stack([self.U, self.V, self.N], 1)
        for v in np.arange(v0 + dv / 2, v1, dv):
            iv = intervals(P, v)
            for hu, hv, hd in self.holes:
                if abs(v - hv) < hd / 2:
                    w = math.sqrt((hd / 2) ** 2 - (v - hv) ** 2)
                    iv = subtract(iv, [(hu - w, hu + w)])
            for hp in self.polys:
                iv = subtract(iv, intervals(hp, v))
            for a, b in iv:
                if b - a < 0.05:
                    continue
                c = self.o + (a + b) / 2 * self.U + v * self.V + (self.w0 + self.t / 2) * self.N
                out.append({"name": self.name, "c": c, "R": R, "h": np.array([(b - a) / 2, dv / 2, self.t / 2]), "r": 0.0})
        return out

    def spec(self):
        def clean(P):
            P = np.round(np.asarray(P, float), 4)
            keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-3]
            P = P[keep]
            return P[:-1] if np.linalg.norm(P[0] - P[-1]) <= 1e-3 else P
        return {"type": "flat", "name": self.name, "outline": clean(self.outline).tolist(), "holes": self.holes,
                "polys": [clean(p).tolist() for p in self.polys], "o": self.o.tolist(), "U": self.U.tolist(),
                "V": self.V.tolist(), "N": self.N.tolist(), "w0": self.w0, "t": self.t}

    def prism(self):
        """The outline (no holes) extruded, as a mesh (vertices, faces) in the body frame: convex pieces only."""
        P = self.outline[:-1] if np.allclose(self.outline[0], self.outline[-1]) else self.outline
        n = len(P)
        lo = self.o[None] + P[:, :1] * self.U + P[:, 1:] * self.V + self.w0 * self.N
        hi = lo + self.t * self.N
        Vt = np.concatenate([lo, hi])
        F = [[0, i + 1, i] for i in range(1, n - 1)] + [[n, n + i, n + i + 1] for i in range(1, n - 1)]
        F += [[i, (i + 1) % n, n + (i + 1) % n] for i in range(n)] + [[i, n + (i + 1) % n, n + i] for i in range(n)]
        return Vt, np.array(F)


class Block:
    """A printed box (axis-aligned in its body's frame), a piece of the printed part `part`."""

    def __init__(self, name, body, lo, hi, part):
        self.name, self.body, self.part, self.kind = name, body, part, "printed"
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        assert np.all(self.hi > self.lo), name

    def strips(self, dv=1.0):
        return [box_prim(self.name, self.lo, self.hi)]

    def spec(self):
        return {"type": "box", "name": self.name, "lo": self.lo.tolist(), "hi": self.hi.tolist()}

    def prism(self):
        c = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
        Vt = self.lo + c * (self.hi - self.lo)
        F = [[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2],
             [1, 3, 7], [1, 7, 5]]
        return Vt, np.array(F)


# ------------------------------------------------------------------------------------------------- solids
def box_prim(name, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return {"name": name, "c": (lo + hi) / 2, "R": np.eye(3), "h": (hi - lo) / 2, "r": 0.0}


def frame_prim(name, origin, axes, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return {"name": name, "c": origin + axes @ ((lo + hi) / 2), "R": axes, "h": (hi - lo) / 2, "r": 0.0}


def disc_prims(name, c, axis, d, t):
    """A disc as three rectangles 60 deg apart (half-sizes R and R tan 30): their union covers it, corners at 1.155 R."""
    axis = np.asarray(axis, float)
    u = np.cross(axis, X if abs(axis @ X) < 0.9 else Y)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    R_ = d / 2
    out = []
    for ang in (0.0, math.pi / 3, 2 * math.pi / 3):
        uu = math.cos(ang) * u + math.sin(ang) * v
        out.append({"name": name, "c": np.asarray(c, float), "R": np.stack([uu, np.cross(axis, uu), axis], 1),
                    "h": np.array([R_, R_ * math.tan(math.pi / 6), t / 2]), "r": 0.0})
    return out


def cyl_prims(name, a, b, d):
    """A flat-ended cylinder of diameter d from a to b (disc_prims along the segment: exact in length, at most 15 %
    wide at the corners)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    return disc_prims(name, (a + b) / 2, (b - a) / L, d, L)


def servo_boxes(name, spline, h_dir, long_dir):
    """An STS3032's case, tabs and spline boss as boxes (spline top at `spline`, shaft along h_dir from base to
    spline, long end along long_dir; axis-aligned directions)."""
    h_dir, long_dir = np.asarray(h_dir, float), np.asarray(long_dir, float)
    t_dir = np.abs(np.cross(long_dir, h_dir))
    sp = np.asarray(spline, float)

    def box(l_rng, h_rng, half_t):
        pts = [sp + l * long_dir + (h - S32["h_spline"]) * h_dir + t * half_t * t_dir for l in l_rng for h in h_rng for t in (-1, 1)]
        return np.min(pts, 0), np.max(pts, 0)
    return [box_prim(name + "_case", *box(S32["len"], (0.0, S32["h_case"]), S32["half_t"] + 0.05)),
            box_prim(name + "_tabs", *box(S32["tab_len"], S32["h_tab"], S32["half_t"])),
            box_prim(name + "_boss", *box((-3.5, 3.5), (S32["h_case"], S32["h_spline"]), 3.5))]


# ------------------------------------------------------------------------------------------------- the layout
def geometry():
    """Derived positions (mm, mount frame) that several parts share."""
    Gz, xY, yM = B["Gz"], B["yaw_spline_x"], B["mcp_spline_y"]
    hs, hc, ht = S32["h_spline"], S32["h_case"], S32["h_tab"]
    fl, mf = B["disc"]["flange"], B["mf63"]
    g = {"G": np.array([0.0, 0.0, Gz]), "Gz": Gz, "xY": xY, "yM": yM}
    # yaw servo (carriage): spline at xY, case from xY + 4.4 to its base, tabs 20.5-22 above the base
    g["yaw_case_x"] = (xY + hs - hc, xY + hs)
    g["yaw_tab_x"] = (xY + hs - ht[1], xY + hs - ht[0])
    g["Y1_x"] = (xY - fl - TP, xY - fl)  # the yoke's front arm on the disc horn's face
    g["Y4_x"] = (g["yaw_case_x"][1] + B["yaw_rear_gap"], g["yaw_case_x"][1] + B["yaw_rear_gap"] + TP)  # rear arm
    g["rear_x"] = (g["Y4_x"][1] + mf["flange_t"] + 0.6, g["Y4_x"][1] + mf["flange_t"] + 0.6 + B["rear_wall_t"])
    # MCP servo (yoke): spline at yM, base at yM + 31.9, tabs 20.5-22 above the base
    g["mcp_tab_y"] = (yM + hs - ht[1], yM + hs - ht[0])
    g["mcp_base_y"] = yM + hs
    g["L1_y"] = (yM - fl - 4.0, yM - fl)  # the link's horn side (4 mm: the horn screws' heads sit on it)
    g["L2_y"] = (g["mcp_base_y"] + B["idler_gap"], g["mcp_base_y"] + B["idler_gap"] + TP)
    g["Y3_y"] = (g["L2_y"][1] + B["l2_y3_gap"], g["L2_y"][1] + B["l2_y3_gap"] + TP)
    g["Y2_y"] = (g["mcp_tab_y"][1], g["mcp_tab_y"][1] + TP)  # the yoke's collar, on the base side of the MCP servo's tabs
    # the link: merge block under the MCP servo, PIP servo, PIP axis
    g["cz1"] = Gz - B["cross_r"]
    g["cz0"] = g["cz1"] - B["block_depth"]
    g["pz"] = Gz - B["proximal"]
    g["zs"] = Gz + B["pip"]["spline_z"]
    g["xs"] = B["pip"]["spline_x"]
    g["pip_case_x"] = (g["xs"] - hs, g["xs"] - hs + hc)
    g["pip_tab_x"] = (g["xs"] - hs + ht[0], g["xs"] - hs + ht[1])
    sg = -1.0 if B["pip"].get("long_end", "down") == "down" else 1.0  # the long end's direction along z
    g["pip_long"] = sg
    g["pip_case_z"] = tuple(sorted((g["zs"] + sg * S32["len"][1], g["zs"] + sg * S32["len"][0])))
    g["pip_tab_z"] = tuple(sorted((g["zs"] + sg * S32["tab_len"][1], g["zs"] + sg * S32["tab_len"][0])))
    g["pip_tab_holes_z"] = (g["zs"] + sg * 19.79, g["zs"] - sg * 8.21)  # long end's hole, short end's hole
    return g


def carriage_pieces():
    """The printed carriage (body "carriage"): a column that bolts to the MGN7C block and the lead nut like the v3
    carriage, two side rails beside the beam's lower parts, a roof and pocket around the yaw servo under the stepper, and
    a rear wall that holds the yaw stage's rear dowel."""
    g = geometry()
    C, Gz, xY = B["carriage"], g["Gz"], g["xY"]
    x0, (cz0, cz1), cl = C["x0"], C["col_z"], B["cut_clear"]
    xn = LEAD_NUT["flange_x"][0]
    yr0, yr1 = C["rail_y"]
    yp0, yp1 = C["pocket_y"]
    xc0, xc1 = g["yaw_case_x"]
    xt0 = g["yaw_tab_x"][0]
    xr0, xr1 = g["rear_x"]
    top = C["roof_top"]
    P = "P1 carriage"
    pcs = [Block("C column", "carriage", (x0, C["plate_y"][0], cz0), (xn, yr1, cz1), P),
           Block("C block plate", "carriage", (x0, C["plate_y"][0], cz1), (C["plate_x1"], C["plate_y"][1], C["plate_top"]), P)]
    for s in (-1, 1):
        lo_y, hi_y = sorted((s * yr0, s * yr1))
        pcs += [Block(f"C rail front {'+' if s > 0 else '-'}y", "carriage", (xn, lo_y, cz0), (C["front_roof_x0"], hi_y, C["rail_top"]), P),
                Block(f"C rail {'+' if s > 0 else '-'}y", "carriage", (C["front_roof_x0"], lo_y, C["rail_low"]), (xr1, hi_y, C["rail_top"]), P)]
        lo_y, hi_y = sorted((s * yp0, s * yp1))
        pcs.append(Block(f"C pocket side {'+' if s > 0 else '-'}y", "carriage", (xc0, lo_y, Gz + C["pocket_low_rel"]),
                         (xc1, hi_y, Gz + C["tab_gap_rel"]), P))
    horn_top = B["disc"]["d"] / 2 + 1.9  # the disc horn turns with the yoke; its boxed corners reach 1.155 r
    pcs += [Block("C roof over horn", "carriage", (C["front_roof_x0"], -yr0, Gz + horn_top), (xc0, yr0, top), P),
            Block("C front roof", "carriage", (xc0, -yp0, Gz + S32["len"][1] + cl), (xt0 - 0.1, yp0, top), P),
            Block("C roof", "carriage", (xt0 - 0.1, -yr0, Gz + C["tab_gap_rel"]), (xr1, yr0, top), P),
            Block("C rear wall", "carriage", (xr0, -yr0, Gz - 6.0), (xr1, yr0, top), P)]
    cuts = []
    cy, cz = LEAD_NUT["axis"]
    cuts.append({"type": "cyl", "a": [x0 - 1, cy, cz], "b": [xn + 1, cy, cz], "d": LEAD_NUT["barrel_d"] + 0.2, "name": "nut barrel"})
    for a in LEAD_NUT["hole_deg"]:
        yy, zz = cy + LEAD_NUT["hole_r"] * math.cos(math.radians(a)), cz + LEAD_NUT["hole_r"] * math.sin(math.radians(a))
        cuts.append({"type": "cyl", "a": [x0 - 1, yy, zz], "b": [xn + 1, yy, zz], "d": LEAD_NUT["hole_d"], "name": "nut M3"})
    for bx, bz in BLOCK["holes_xz"]:
        cuts.append({"type": "cyl", "a": [bx, C["plate_y"][0] - 1, bz], "b": [bx, C["plate_y"][1] + 1, bz], "d": 2.4, "name": "block M2"})
    for sy in (-1, 1):  # yaw servo's long-end tab screws into the front roof; access holes through the rear wall
        cuts.append({"type": "cyl", "a": [xc0 - 0.5, sy * 3.5, Gz + 19.79], "b": [xt0 + 0.5, sy * 3.5, Gz + 19.79], "d": 1.8, "name": "tab M2"})
        cuts.append({"type": "cyl", "a": [xr0 - 1, sy * 3.5, Gz + 19.79], "b": [xr1 + 1, sy * 3.5, Gz + 19.79], "d": 4.2, "name": "access"})
    cuts.append({"type": "cyl", "a": [xr0 - 1, 0, Gz], "b": [xr1 + 1, 0, Gz], "d": 3.4, "name": "rear dowel (bonded)"})
    cuts.append({"type": "box", "lo": [xc1 - 9.0, -3.0, Gz + C["tab_gap_rel"] - 1], "hi": [xc1 - 1.0, 3.0, top + 1], "name": "cable slot"})
    return pcs, cuts


def yoke_pieces():
    """The printed yoke (body "yoke"): front arm Y1 on the yaw servo's disc horn, web under the yaw servo, rear arm Y4
    with the rear bearing, the collar Y2 that holds the MCP servo by its tabs, and the idler arm Y3 with the MCP idler's
    dowel."""
    g = geometry()
    Gz = g["Gz"]
    x1a, x1b = g["Y1_x"]
    x4a, x4b = g["Y4_x"]
    w, rw, tw = B["web"]["w"], B["web"]["r"], B["web"]["t"]
    y2a, y2b = g["Y2_y"]
    y3a, y3b = g["Y3_y"]
    cl = B["cut_clear"]
    P = "P2 yoke"
    f1 = dict(o=(x1b, 0, 0), U=Y, V=Z, flip=True, t=TP, kind="printed", part=P)  # Y1 extends from the horn toward -x
    f4 = dict(o=(x4a, 0, 0), U=Y, V=Z, t=TP, kind="printed", part=P)
    pcs = [Flat("Y1 front arm", "yoke", hull_circles([(0.0, Gz, 10.5)]), **f1),
           Flat("Y1 web root", "yoke", rrect(-w / 2, Gz - rw - tw, w / 2, Gz, 1.0), **f1),
           Flat("Y1 idler bar", "yoke", rrect(-3.0, Gz - 3.5, y3b, Gz + 3.5, 1.0), **f1),
           Flat("Y1 collar root", "yoke", rrect(y2a, Gz - 7.0, y2b, Gz + 12.0, 0.8), **f1),
           Block("Y web", "yoke", (x1a, -w / 2, Gz - rw - tw), (x4b, w / 2, Gz - rw), P),
           Flat("Y4 rear arm", "yoke", hull_circles([(0.0, Gz, 6.5)]), **f4),
           Flat("Y4 web root", "yoke", rrect(-w / 2, Gz - rw - tw, w / 2, Gz, 1.0), **f4)]
    # the collar: a frame around the MCP servo's case at its tabs, its lower corners within 12 mm of G
    hz = S32["half_t"] + cl
    outline = poly_round([(-8.5, Gz - 8.0), (-5.5, Gz - 10.6), (5.5, Gz - 10.6), (8.5, Gz - 8.0), (8.5, Gz + 22.5), (-8.5, Gz + 22.5)], 0.8)
    case = rrect(-hz, Gz + S32["len"][0] - cl, hz, Gz + S32["len"][1] + cl, 0.5)
    holes = [(sx * 3.5, Gz + zz, 1.8) for sx in (-1, 1) for zz in (19.79, -8.21)]
    pcs.append(Flat("Y2 collar", "yoke", outline, (0, y2a, 0), X, Z, holes=holes, polys=[case], flip=True, t=TP, kind="printed", part=P))
    pcs.append(Flat("Y2 collar root", "yoke", rrect(8.0, Gz - 7.0, x1b, Gz + 12.0, 0.8), (0, y2a, 0), X, Z, flip=True, t=TP,
                    kind="printed", part=P))
    pcs.append(Flat("Y3 idler arm", "yoke", hull_circles([(0.0, Gz, 6.0), (x1b - 1.0, Gz - 2.5, 1.0), (x1b - 1.0, Gz + 2.5, 1.0)]),
                    (0, y3a, 0), X, Z, flip=True, t=TP, kind="printed", part=P))
    cuts = [{"type": "cyl", "a": [x1a - 1, B["disc"]["bolt_r"] * math.cos(math.radians(a)), Gz + B["disc"]["bolt_r"] * math.sin(math.radians(a))],
             "b": [x1b + 1, B["disc"]["bolt_r"] * math.cos(math.radians(a)), Gz + B["disc"]["bolt_r"] * math.sin(math.radians(a))],
             "d": 2.4, "name": "horn M2"} for a in (45.0, 90.0, 270.0, 315.0)]
    cuts.append({"type": "cyl", "a": [x1a - 1, 0, Gz], "b": [x1b + 1, 0, Gz], "d": 5.0, "name": "horn screw"})
    mf = B["mf63"]
    cuts.append({"type": "cyl", "a": [x4b - mf["B"], 0, Gz], "b": [x4b + 1, 0, Gz], "d": mf["D"], "name": "rear bearing"})
    cuts.append({"type": "cyl", "a": [x4a - 1, 0, Gz], "b": [x4b, 0, Gz], "d": 4.5, "name": "rear dowel clearance"})
    cuts.append({"type": "cyl", "a": [0, y3a - 1, Gz], "b": [0, y3b + 1, Gz], "d": 2.95, "name": "idler dowel (press)"})
    return pcs, cuts


def link_pieces():
    """The printed proximal link (body "link"): the horn side L1 bolted to the MCP servo's disc horn, the idler side L2
    with its bearing, a merge block under the MCP servo, the cradle that holds the PIP servo by its tabs and the PIP
    cheeks with the PIP bearings."""
    g = geometry()
    Gz, cz0, cz1, pz = g["Gz"], g["cz0"], g["cz1"], g["pz"]
    (l1a, l1b), (l2a, l2b) = g["L1_y"], g["L2_y"]
    (xc0, xc1), (zc0, zc1) = g["pip_case_x"], g["pip_case_z"]
    xt0 = g["pip_tab_x"][0]
    (zt0, zt1) = g["pip_tab_z"]
    yi, yo, wy = B["lower_y"], B["cheek_out_y"], S32["half_t"] + B["cut_clear"]
    xb = 6.5  # the merge block's half-width along x
    P = "P3 link"
    pcs = [Flat("L1 horn side", "link", hull_circles([(0.0, Gz, 10.0), (-xb + 1, cz0 + 1, 1.0), (xb - 1, cz0 + 1, 1.0)]),
                (0, l1b, 0), X, Z, t=l1b - l1a, kind="printed", part=P),
           Flat("L2 idler side", "link", hull_circles([(0.0, Gz, 6.5), (-xb + 1, cz0 + 1, 1.0), (xb - 1, cz0 + 1, 1.0)]),
                (0, l2a, 0), X, Z, flip=True, t=TP, kind="printed", part=P),
           Block("L merge block", "link", (-xb, l1a, cz0), (xb, l2b, cz1), P)]
    for s in (-1, 1):
        sg = '+' if s > 0 else '-'
        lo_y, hi_y = sorted((s * wy, s * yo))
        pcs.append(Block(f"L cradle side {sg}y", "link", (xc0, lo_y, zc0 - 1.0), (xc1 + 0.6, hi_y, cz0 + 0.5), P))
        lo_y, hi_y = sorted((s * yi, s * yo))
        pcs.append(Block(f"L cheek root {sg}y", "link", (-8.0, lo_y, pz + 7.0), (8.0, hi_y, zc0 - 0.5), P))
        pcs.append(Flat(f"L PIP cheek {sg}y", "link", hull_circles([(0.0, pz, 8.0), (-7.0, pz + 8.0, 1.0), (7.0, pz + 8.0, 1.0)]),
                        (0, s * yi, 0), X, Z, flip=(s > 0), t=yo - yi, kind="printed", part=P))
    pcs += [Block("L tab frame upper", "link", (xt0 - TP, -yo, zc1 + B["cut_clear"]), (xt0, yo, cz0 + 0.5), P),
            Block("L tab frame lower", "link", (xt0 - TP, -yo, zt0 - 1.0), (xt0, yo, zc0 - B["cut_clear"]), P)]
    mf = B["mf63"]
    cuts = [{"type": "cyl", "a": [B["disc"]["bolt_r"] * math.cos(math.radians(a)), l1a - 1, Gz + B["disc"]["bolt_r"] * math.sin(math.radians(a))],
             "b": [B["disc"]["bolt_r"] * math.cos(math.radians(a)), l1b + 1, Gz + B["disc"]["bolt_r"] * math.sin(math.radians(a))],
             "d": 2.4, "name": "horn M2"} for a in (45.0, 135.0, 225.0, 315.0)]
    cuts.append({"type": "cyl", "a": [0, l1a - 1, Gz], "b": [0, l1b + 1, Gz], "d": 5.0, "name": "horn screw"})
    cuts.append({"type": "cyl", "a": [0, l2a - 0.01, Gz], "b": [0, l2a + mf["B"], Gz], "d": mf["D"], "name": "idler bearing"})
    cuts.append({"type": "cyl", "a": [0, l2a + mf["B"] - 0.01, Gz], "b": [0, l2b + 1, Gz], "d": 4.5, "name": "idler dowel clearance"})
    for s in (-1, 1):
        cuts.append({"type": "cyl", "a": [0, s * (yi - 0.01), pz], "b": [0, s * (yi + mf["B"]), pz], "d": mf["D"], "name": "PIP bearing"})
        cuts.append({"type": "cyl", "a": [0, s * (yi + mf["B"] - 0.01), pz], "b": [0, s * (yo + 1), pz], "d": 4.5, "name": "PIP dowel clearance"})
    for sy in (-1, 1):  # the PIP servo's tab screws into the frame (self-forming M2)
        for zz in g["pip_tab_holes_z"]:
            cuts.append({"type": "cyl", "a": [xt0 - TP - 0.5, sy * 3.5, zz], "b": [xt0 + 0.5, sy * 3.5, zz], "d": 1.8, "name": "tab M2"})
    case = {"type": "box", "lo": [xc0 - 1.0, -wy, zc0 - B["cut_clear"]], "hi": [xc1 + 1.0, wy, zc1 + B["cut_clear"]], "name": "PIP servo case"}
    cuts.append(case)
    return pcs, cuts


def acrylic_flats():
    """The laser-cut PIP linkage: the crank H1 on the PIP servo's disc horn (phorn frame: the servo axis along x through
    (0, 0, zs)) and the rod R1 (its own frame)."""
    g = geometry()
    zs, xs = g["zs"], g["xs"]
    r = B["pip"]["r"]
    bt = B["disc"]["bolt_r"]  # bolted through the disc's holes at 180 and 135 deg, on the side away from the rod
    br = bt / math.sqrt(2)
    outline = hull_circles([(-bt, zs, 3.2), (-br, zs + br, 3.2), (0.0, zs, 4.5), (r, zs, 4.75)])
    holes = [(-bt, zs, 2.2), (-br, zs + br, 2.2), (0.0, zs, 5.0), (r, zs, B["mf63"]["D"])]
    H1 = Flat("H1 PIP crank", "phorn", outline, (xs + B["disc"]["flange"], 0, 0), Y, Z, holes=holes,
              note="2 x M2 into the PIP servo's disc horn (holes at 180 and 135 deg); MF63ZZ for pin a")
    _S, gk = pip_kin(np.array([B["pip"]["pip_rest"]]))
    return [H1, rod_flat(gk["L"])]


def printed_parts():
    """{part: (pieces, cuts, body)} for the three printed bodies; the knuckle is built separately (knuckle_spec)."""
    out = {}
    for fn in (carriage_pieces, yoke_pieces, link_pieces):
        pcs, cuts = fn()
        out[pcs[0].part] = (pcs, cuts, pcs[0].body)
    return out


def dogbone(L, r0, r1, w):
    """Two round ends (radii r0 at u = 0 and r1 at u = L) joined by a bar of half-width w (w < r0, r1), as a dense ccw
    polygon."""
    a0, a1 = math.asin(w / r0), math.asin(w / r1)
    return np.concatenate([arc((L, 0.0), r1, a1 - math.pi, math.pi - a1), arc((0.0, 0.0), r0, a0, 2 * math.pi - a0)])


def rod_flat(L_rod):
    """The PIP rod in its own frame: u from the top bearing (pin b, an MF63ZZ) toward the lug's spherical bearing."""
    outline = dogbone(L_rod, 5.0, 8.0, 3.5)
    return Flat("R1 PIP rod", "rod", outline, (0, 0, -T / 2), X, Y, holes=[(0.0, 0.0, B["mf63"]["D"]), (L_rod, 0.0, B["sph"]["D"])],
                note="MF63ZZ at pin b; spherical plain bearing (d 3, D 12) bonded in at the lug")


def solids():
    """Bought parts and the DIP link's envelope as boxes per body: servos, disc horns, bearing flanges, dowels, the lead
    nut and the MGN7C block, the printed DIP link and the fingertip."""
    g = geometry()
    Gz, xY, yM, dk, mf = g["Gz"], g["xY"], g["yM"], B["disc"], B["mf63"]
    S = {k: [] for k in ("carriage", "yoke", "link", "dip", "phorn")}
    S["carriage"] += servo_boxes("yawservo", (xY, 0, Gz), (-1, 0, 0), (0, 0, 1))
    cy, cz = LEAD_NUT["axis"]
    S["carriage"] += cyl_prims("leadnut", (LEAD_NUT["barrel_x"][0], cy, cz), (LEAD_NUT["flange_x"][0], cy, cz), LEAD_NUT["barrel_d"])
    S["carriage"] += cyl_prims("leadnut", (LEAD_NUT["flange_x"][0], cy, cz), (LEAD_NUT["flange_x"][1], cy, cz), LEAD_NUT["flange_d"])
    S["carriage"].append(box_prim("mgn7c", BLOCK["lo"], BLOCK["hi"]))
    x4a, x4b = g["Y4_x"]
    S["carriage"] += cyl_prims("rear_dowel", (x4b - mf["B"] - 0.5, 0, Gz), (g["rear_x"][1], 0, Gz), 3.0)
    S["yoke"] += disc_prims("yawdisc", (xY - dk["flange"] / 2, 0, Gz), X, dk["d"], dk["flange"])
    S["yoke"] += disc_prims("flange_Y4", (x4b + mf["flange_t"] / 2, 0, Gz), X, mf["flange_d"], mf["flange_t"])
    S["yoke"] += servo_boxes("mcpservo", (0, yM, Gz), (0, -1, 0), (0, 0, 1))
    y_out = g["Y3_y"][1]
    S["yoke"] += cyl_prims("idler_dowel", (0, y_out - 8.0, Gz), (0, y_out, Gz), 3.0)
    S["link"] += disc_prims("mcpdisc", (0, yM - dk["flange"] / 2, Gz), Y, dk["d"], dk["flange"])
    S["link"] += disc_prims("flange_L2", (0, g["L2_y"][0] - mf["flange_t"] / 2, Gz), Y, mf["flange_d"], mf["flange_t"])
    S["link"] += servo_boxes("pipservo", (g["xs"], 0, g["zs"]), (1, 0, 0), (0, 0, g["pip_long"]))
    S["phorn"] += disc_prims("pipdisc", (g["xs"] + dk["flange"] / 2, 0, g["zs"]), X, dk["d"], dk["flange"])
    S["phorn"] += disc_prims("flange_H1", (g["xs"] + dk["flange"] + T + mf["flange_t"] / 2, B["pip"]["r"], g["zs"]), X,
                             mf["flange_d"], mf["flange_t"])
    pz = g["pz"]
    for sy in (1, -1):  # the PIP bearings' flanges on the cheeks' inner faces
        S["link"] += disc_prims("flange_PIP", (0.0, sy * (B["lower_y"] - mf["flange_t"] / 2), pz), Y, mf["flange_d"], mf["flange_t"])
    y0, y1 = B["dip_cheek_y"]
    for sy in (1, -1):
        a, b = sorted((sy * y0, sy * y1))
        S["dip"] += [{**p, "name": "dip_cheek"} for p in disc_prims("dip_cheek", (0.0, (a + b) / 2, pz), Y, 13.0, b - a)]
        S["dip"].append(box_prim("dip_cheek", (-6.5, a, pz - 18.0), (6.5, b, pz)))
    tb = B["tip_box"]
    S["dip"].append(box_prim("tip", (tb["x"][0], -tb["y"], pz + tb["z"][0]), (tb["x"][1], tb["y"], pz + tb["z"][1])))
    return S


SHOULDER_LENGTHS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0)  # McMaster 90323A, 3 mm shoulder


def pin_stack(floor, ring_boss, ring_nut):
    """A shoulder screw along a coordinate that falls from its head: the head bears on `floor`, the bearing's inner
    ring spans ring_boss (its face on the boss) to ring_nut (its face on the nut). The longest catalogue shoulder that
    keeps at least 1.5 mm of its length in the ring and ends 0.3 mm short of the nut's face, so the nut clamps the ring
    against the boss; the nut and the thread's end."""
    ok = [L for L in SHOULDER_LENGTHS if ring_nut + 0.3 < floor - L < ring_boss - 1.5]
    L = max(ok)
    return {"shoulder": L, "shoulder_end": floor - L, "nut": (ring_nut, ring_nut - B["nut"]["t"]),
            "thread_end": floor - L - B["shoulder"]["thread"]}


def knuckle_spec():
    """The printed knuckle in its own frame: x along pin a (from the crank H1 toward the far face), y along k (from pin
    a toward pin b), z along pin b; origin at pin a's foot Pa. Each pin is a shoulder screw whose head sits on the
    knuckle's far face (pin a) or in a counterbore (pin b) and whose M2 nut sits in the bearing's bore recess (in H1 or
    R1), where it bears on the inner ring alone; a 4 mm boss on the knuckle bears on the ring's other face."""
    _S, g = pip_kin(np.array([B["pip"]["pip_rest"]]))
    pp, mf, sh, bo = B["pip"], B["mf63"], B["shoulder"], B["boss"]
    e, c, xa = pp["knuckle_e"], g["c"], g["x_a"]
    x_H1 = pp["spline_x"] + B["disc"]["flange"] + T
    a_boss, a_nut = x_H1 + mf["flange_t"] - xa, x_H1 + mf["flange_t"] - mf["B"] - xa
    s = math.copysign(1.0, c)
    b_boss = -(abs(c) - T / 2 - mf["flange_t"])
    b_nut = b_boss - mf["B"]
    hz = -b_boss - bo["h"]
    box = {"x": (a_boss + bo["h"], 4.5), "y": (-3.5, e + 4.0), "z": (-hz, hz)}
    return {"e": e, "c": c, "x_a": xa, "s": s, "box": box,
            "a": {"ring": (a_boss, a_nut), **pin_stack(box["x"][1], a_boss, a_nut)},
            "b": {"ring": (b_boss, b_nut), **pin_stack(hz - sh["head_h"], b_boss, b_nut)}}


def knuckle_prims(spec):
    """The knuckle, its bosses and the two pins' screws and nuts as primitives in the knuckle frame."""
    bx, e, s = spec["box"], spec["e"], spec["s"]
    r_n, bo = B["nut"]["corners"] / 2, B["boss"]
    a, b = spec["a"], spec["b"]
    Zb = lambda t: -s * t  # noqa: E731, pin b's coordinate back to z
    sh = B["shoulder"]
    return ([box_prim("knuckle", (bx["x"][0], bx["y"][0], bx["z"][0]), (bx["x"][1], bx["y"][1], bx["z"][1]))]
            + cyl_prims("knuckle_boss", (a["ring"][0], 0, 0), (a["ring"][0] + bo["h"], 0, 0), bo["d"])
            + cyl_prims("pin_a", (a["thread_end"], 0, 0), (bx["x"][1], 0, 0), sh["d"])
            + cyl_prims("pin_a_head", (bx["x"][1], 0, 0), (bx["x"][1] + sh["head_h"], 0, 0), sh["head_d"])
            + cyl_prims("pin_a_nut", (a["nut"][1], 0, 0), (a["nut"][0], 0, 0), 2 * r_n)
            + cyl_prims("knuckle_boss", (0, e, Zb(b["ring"][0])), (0, e, Zb(b["ring"][0] + bo["h"])), bo["d"])
            + cyl_prims("pin_b", (0, e, Zb(b["thread_end"])), (0, e, Zb(bx["z"][1])), sh["d"])
            + cyl_prims("pin_b_nut", (0, e, Zb(b["nut"][1])), (0, e, Zb(b["nut"][0])), 2 * r_n))


def rod_solids(L_rod):
    """The rod's bearings beyond its plate, in the rod frame: pin b's flange toward the knuckle and the spherical
    bearing's outer ring (3.6 mm wide in the 3.175 mm plate) at the lug."""
    _S, g = pip_kin(np.array([B["pip"]["pip_rest"]]))
    s, mf = -math.copysign(1.0, g["c"]), B["mf63"]
    return (disc_prims("flange_R1", (0.0, 0.0, s * (T / 2 + mf["flange_t"] / 2)), Z, mf["flange_d"], mf["flange_t"])
            + disc_prims("sph_ring", (L_rod, 0.0, 0.0), Z, B["sph"]["D"], B["sph"]["H"]))


def pip_kin(pip_deg):
    """The PIP drive of scripts/cad/v4c_hardware.py with this layout's numbers, in the link frame (G at (0, 0, Gz)):
    lug A, rod top Q, pin a's foot Pa, knuckle direction k, pin b, horn angle th (rad), closure ok."""
    import v4c_hardware as V
    V.HW["pip"] = {**V.HW["pip"], **B["pip"]}
    V.HW["proximal"] = B["proximal"]
    S, g = V.pip_solve(np.asarray(pip_deg, float))
    sh = np.array([0.0, 0.0, B["Gz"]])
    return {k: (S[k] + sh if k in ("A", "Q", "Pa") else S[k]) for k in S}, g


def dip_lug_prims(A_dip):
    """The DIP link's lug arm between the cheeks: from the PIP axis toward the lug (a box), in the dip frame at PIP 0."""
    g = geometry()
    Op = np.array([0.0, 0.0, g["pz"]])
    d = A_dip - Op
    L = float(np.linalg.norm(d[[0, 2]]))
    u = np.array([d[0], 0.0, d[2]]) / L
    w = np.cross(Y, u)
    y0, y1 = B["dip_cheek_y"]
    return [frame_prim("dip_lugarm", Op, np.stack([u, Y, w], 1), (0.0, -y1, -4.5), (L + 4.5, -y0, 4.5)),
            frame_prim("dip_lugarm", Op, np.stack([u, Y, w], 1), (0.0, y0, -4.5), (L + 4.5, y1, 4.5))]


# ------------------------------------------------------------------------------------------------- poses and check
def body_poses(yaw, mcp, pip, S, i):
    """Rotation and translation of each moving body's frame (mount frame at the zero pose) at one pose; S is the PIP
    solve and i its index."""
    g = geometry()
    G = g["G"]
    Rx = rot_x(math.radians(yaw))
    Rl = Rx @ rot_y(math.radians(mcp))
    tl = G - Rl @ G
    Op = np.array([0.0, 0.0, g["pz"]])
    Rp = rot_y(math.radians(pip))
    Cs = np.array([0.0, 0.0, g["zs"]])
    Rh = rot_x(float(S["th"][i]))
    return {"carriage": (np.eye(3), np.zeros(3)), "yoke": (Rx, G - Rx @ G), "link": (Rl, tl),
            "dip": (Rl @ Rp, Rl @ (Op - Rp @ Op) + tl), "phorn": (Rl @ Rh, Rl @ (Cs - Rh @ Cs) + tl)}


def bodies_at(prims, yaw, mcp, pip, S, i):
    """World primitives per body at one pose."""
    T_ = body_poses(yaw, mcp, pip, S, i)

    def tf(ps, R_, t_):
        return [{"name": p["name"], "c": R_ @ p["c"] + t_, "R": R_ @ p["R"], "h": p["h"], "r": p["r"]} for p in ps]
    out = {b: tf(prims[b], *T_[b]) for b in ("carriage", "yoke", "link", "dip", "phorn")}
    A, Qp, Pa, k, pb = S["A"][i], S["Q"][i], S["Pa"][i], S["k"][i], S["pb"][i]
    Rk = np.stack([X, k, pb], 1)
    kn = [{**p, "c": Pa + Rk @ p["c"], "R": Rk @ p["R"]} for p in S["knuckle_prims"]]
    d = (A - Qp) / np.linalg.norm(A - Qp)
    w = np.cross(pb, d)
    rod = [{**p, "c": Qp + np.stack([d, w, pb], 1) @ p["c"], "R": np.stack([d, w, pb], 1) @ p["R"]} for p in S["rod_prims"]]
    out["knuckle"] = tf(kn, *T_["link"])
    out["rod"] = tf(rod, *T_["link"])
    return out


EXCLUDE = {("carriage", "yoke"): [("yawservo_boss", "yawdisc"), ("rear_dowel", "Y4"), ("rear_dowel", "flange_Y4")],
           ("yoke", "link"): [("mcpservo_boss", "mcpdisc"), ("idler_dowel", "L2"), ("idler_dowel", "flange_L2")],
           ("link", "phorn"): [("pipservo_boss", "pipdisc"), ("pipservo_case", "pipdisc")],
           }
ATTACHED = {("phorn", "knuckle"), ("knuckle", "rod")}


def as_set(ps):
    return (np.array([p["c"] for p in ps]), np.array([p["R"] for p in ps]), np.array([p["h"] for p in ps]),
            np.array([p["r"] for p in ps]))


def all_prims(L_rod, dv=1.0):
    """Primitives per body in their own frames: printed pieces and acrylic plates (strips) plus solids; the rod and the
    knuckle in their own frames."""
    P = solids()
    for pcs, _cuts, body in printed_parts().values():
        for pc in pcs:
            P[body] += pc.strips(dv)
    H1 = acrylic_flats()[0]
    P[H1.body] += H1.strips(dv)
    P["rod_prims"] = rod_flat(L_rod).strips(dv) + rod_solids(L_rod)
    P["knuckle_prims"] = knuckle_prims(knuckle_spec())
    return P


def dip_lug(g):
    gg = geometry()
    Op = np.array([0.0, 0.0, gg["pz"]])
    return Op + rot_y(-math.radians(B["pip"]["pip_rest"])) @ (g["A0"] + np.array([0.0, 0.0, B["Gz"]]) - Op)


def check(step_mcp=7.0, yaws=YAWS, pips=PIPS):
    import cad_v4_layouts as L
    S, g = pip_kin(np.array(pips))
    P = all_prims(g["L"])
    S["rod_prims"], S["knuckle_prims"] = P.pop("rod_prims"), P.pop("knuckle_prims")
    P["dip"] += dip_lug_prims(dip_lug(g))
    mcps = np.arange(MCP_R[0], MCP_R[1] + 0.01, step_mcp)
    least, width = {}, (0.0, None)
    for yaw, mcp in itertools.product(yaws, mcps):
        for i, pip in enumerate(pips):
            W = bodies_at(P, yaw, mcp, pip, S, i)
            for p in W["carriage"] + W["yoke"]:
                cs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * p["h"]
                yy = float(np.abs((p["c"] + cs @ p["R"].T)[:, 1]).max()) + p["r"]
                if yy > width[0]:
                    width = (round(yy, 2), (p["name"], yaw, float(mcp), pip))
            names = list(W)
            pairs = [(a, b) for a, b in itertools.combinations(names, 2) if (a, b) not in ATTACHED and (b, a) not in ATTACHED]
            for a, b in pairs:
                A_, B_ = W[a], W[b]
                ex = EXCLUDE.get((a, b), [])
                gaps, I, J = L.set_gap(as_set(A_), as_set(B_), cutoff=6.0, every=True)
                for gv, ii, jj in zip(gaps, I, J):
                    na, nb = A_[ii]["name"], B_[jj]["name"]
                    if any((na.startswith(e0) and nb.startswith(e1)) or (na.startswith(e1) and nb.startswith(e0)) for e0, e1 in ex):
                        continue
                    key = f"{a}:{na} | {b}:{nb}"
                    if key not in least or gv < least[key][0]:
                        least[key] = (round(float(gv), 2), yaw, float(mcp), pip)
    worst = sorted(least.items(), key=lambda kv: kv[1][0])
    return {"module_half_width_mm": width, "pip_closes": bool(S["ok"].all()), "pip_rod_mm": round(g["L"], 2), "least": worst}


def gantry_prims(s):
    """The gantry's parts as boxes in the module frame with the carriage displaced by s along x (the module sees the
    gantry move by -s)."""
    return [box_prim(n, np.array(lo) - [s, 0, 0], np.array(hi) - [s, 0, 0]) for n, lo, hi in GANTRY]


def stroke_check(poses=None, ss=None):
    """The least gap between the gantry and (a) the carriage over the stroke, (b) the moving bodies at a grid of poses
    over the stroke; the lead nut and block move with the carriage and are left out of (a)."""
    import cad_v4_layouts as L
    yaws, mcps, pips = (-85.0, -45.0, 0.0, 45.0, 85.0), (-15.0, 20.0, 50.0, 92.0), (-25.0, 40.0, 92.0)
    S, g = pip_kin(np.array(pips))
    P = all_prims(g["L"], dv=2.0)
    S["rod_prims"], S["knuckle_prims"] = P.pop("rod_prims"), P.pop("knuckle_prims")
    P["dip"] += dip_lug_prims(dip_lug(g))
    ss = ss if ss is not None else np.r_[np.linspace(STROKE[0], STROKE[1], 13)]
    least = {}
    for s in ss:
        Gp = gantry_prims(s)
        for yaw, mcp in itertools.product(yaws, mcps):
            for i, pip in enumerate(pips):
                W = bodies_at(P, yaw, mcp, pip, S, i)
                for b, ps in W.items():
                    if b == "carriage" and (yaw, mcp, i) != (yaws[0], mcps[0], 0):
                        continue
                    ps = [p for p in ps if not p["name"].startswith(("leadnut", "mgn7c"))]
                    if b == "carriage":  # the lead screw runs through the nut and the column's bore (8.6 mm) by design
                        Gq = [q for q in Gp if q["name"] != "lead screw"]
                    else:
                        Gq = Gp
                    if not ps:
                        continue
                    gaps, I, J = L.set_gap(as_set(Gq), as_set(ps), cutoff=8.0, every=True)
                    for gv, ii, jj in zip(gaps, I, J):
                        key = f"{Gq[ii]['name']} | {b}:{ps[jj]['name']}"
                        if key not in least or gv < least[key][0]:
                            least[key] = (round(float(gv), 2), round(float(s), 2), yaw, mcp, pips[i])
    return sorted(least.items(), key=lambda kv: kv[1][0])



def curl_limit(step=2.0):
    """The least total flexion MCP + PIP at which the DIP link or fingertip reaches the gantry, over yaw and the
    stroke (the finger curled back toward the palm)."""
    import cad_v4_layouts as L
    best = None
    for mcp in np.arange(60.0, MCP_R[1] + 0.01, step):
        pips = np.arange(40.0, 92.01, step)
        S, g = pip_kin(pips)
        P = solids()
        dipp = P["dip"] + dip_lug_prims(dip_lug(g))
        for i, pip in enumerate(pips):
            if best is not None and mcp + pip >= best[0]:
                break
            for yaw in (-85.0, -45.0, 0.0, 45.0, 85.0):
                Rt = body_poses(yaw, mcp, pip, S, i)["dip"]
                W = [{**q, "c": Rt[0] @ q["c"] + Rt[1], "R": Rt[0] @ q["R"]} for q in dipp]
                for s in (STROKE[0], 0.0, STROKE[1]):
                    gaps, _I, _J = L.set_gap(as_set(gantry_prims(s)), as_set(W), cutoff=1.0, every=True)
                    if len(gaps) and min(gaps) <= 0.0:
                        best = (float(mcp + pip), float(mcp), float(pip), yaw, s)
                        break
                if best is not None and best[0] == mcp + pip:
                    break
    return best


# ------------------------------------------------------------------------------------------------- outputs
def fit_coupon():
    """C1, the acrylic fit coupon: ball-bearing holes and the spherical bearing's hole around the nominal sizes."""
    outline = rrect(0.0, 0.0, 84.0, 22.0, 2.0)
    holes, labels = [], []
    for k, d in enumerate((5.8, 5.9, 6.0, 6.1)):
        holes.append((6.0 + 9.0 * k, 6.0, d))
        labels.append((3.0 + 9.0 * k, 11.5, f"{d:.1f}"))
    for k, d in enumerate((11.8, 11.9, 12.0)):
        holes.append((46.0 + 15.0 * k, 9.0, d))
        labels.append((42.5 + 15.0 * k, 17.5, f"{d:.1f}"))
    f = Flat("C1 fit coupon", "none", outline, (0, 0, 0), X, Y, holes=holes, note="cut first: pick the bearing holes")
    f.labels = labels
    return f


def print_coupon_pieces():
    """P0, the printed fit coupon: MF63ZZ bores, 3 mm dowel holes and M2 holes for self-forming threads around the
    nominal sizes, in a 4 mm plate."""
    pcs = [Block("P0 plate", "none", (0.0, 0.0, 0.0), (64.0, 24.0, 4.0), "P0 print fit coupon")]
    cuts = []
    for k, d in enumerate((5.9, 6.0, 6.1, 6.2)):
        cuts.append({"type": "cyl", "a": [7.0 + 10.0 * k, 7.0, -1.0], "b": [7.0 + 10.0 * k, 7.0, 5.0], "d": d, "name": f"bore {d}"})
    for k, d in enumerate((2.85, 2.9, 2.95, 3.0)):
        cuts.append({"type": "cyl", "a": [7.0 + 10.0 * k, 17.0, -1.0], "b": [7.0 + 10.0 * k, 17.0, 5.0], "d": d, "name": f"dowel {d}"})
    for k, d in enumerate((1.7, 1.8, 1.9)):
        cuts.append({"type": "cyl", "a": [47.0 + 7.0 * k, 7.0, -1.0], "b": [47.0 + 7.0 * k, 7.0, 5.0], "d": d, "name": f"M2 {d}"})
    for k, d in enumerate((3.3, 3.4, 3.5)):
        cuts.append({"type": "cyl", "a": [47.0 + 7.0 * k, 17.0, -1.0], "b": [47.0 + 7.0 * k, 17.0, 5.0], "d": d, "name": f"bond {d}"})
    return pcs, cuts


def knuckle_pieces():
    """The knuckle (P4) as FreeCAD pieces in its own frame: the block and two bosses on the bearings' inner rings;
    3.1 mm holes for the shoulders and a 5.6 mm counterbore 1.3 mm deep for pin b's head."""
    K = knuckle_spec()
    bx, e, s, a_, b_ = K["box"], K["e"], K["s"], K["a"], K["b"]
    bo, hh = B["boss"], B["shoulder"]["head_h"]
    zb = lambda t: -s * t  # noqa: E731
    pcs = [Block("K block", "knuckle", (bx["x"][0], bx["y"][0], bx["z"][0]), (bx["x"][1], bx["y"][1], bx["z"][1]), "P4 knuckle")]
    pieces = [{"type": "cyl", "name": "K boss a", "a": [a_["ring"][0], 0, 0], "b": [bx["x"][0] + 0.01, 0, 0], "d": bo["d"]},
              {"type": "cyl", "name": "K boss b", "a": [0, e, zb(b_["ring"][0])], "b": [0, e, zb(b_["ring"][0] + bo["h"] + 0.01)], "d": bo["d"]}]
    cuts = [{"type": "cyl", "a": [bx["x"][0] - 1, 0, 0], "b": [bx["x"][1] + 1, 0, 0], "d": 3.1, "name": "pin a"},
            {"type": "cyl", "a": [0, e, bx["z"][0] - 1], "b": [0, e, bx["z"][1] + 1], "d": 3.1, "name": "pin b"},
            {"type": "cyl", "a": [0, e, zb(bx["z"][1] - hh)], "b": [0, e, zb(bx["z"][1]) - 0.1 * s], "d": 5.6, "name": "pin b head"}]
    return pcs, pieces, cuts


def freecad_spec():
    """The JSON spec of every printed part and of the two acrylic parts (for the viewer and the MJCF)."""
    parts = {}
    for name, (pcs, cuts, body) in printed_parts().items():
        parts[name] = {"file": "20261119-v4b_" + name.split()[1], "body": body, "pieces": [pc.spec() for pc in pcs], "cuts": cuts}
    pcs, extra, cuts = knuckle_pieces()
    parts["P4 knuckle"] = {"file": "20261119-v4b_knuckle", "body": "knuckle", "pieces": [pc.spec() for pc in pcs] + extra, "cuts": cuts}
    pcs, cuts = print_coupon_pieces()
    parts["P0 print fit coupon"] = {"file": "20261119-v4b_print_fit_coupon", "body": "none", "pieces": [pc.spec() for pc in pcs], "cuts": cuts}
    for f in acrylic_flats():
        parts[f.name] = {"file": "20261119-v4b_" + f.name.split()[0], "body": f.body, "pieces": [f.spec()], "cuts": [], "acrylic": True}
    return parts


def cmd_printed():
    """STEP (local) and STL of every part through FreeCAD; a JSON summary in data/."""
    import tempfile
    DATA.mkdir(parents=True, exist_ok=True)
    parts = freecad_spec()
    with tempfile.TemporaryDirectory() as td:
        spec = {"parts": parts, "step_dir": str(STEP_DIR), "stl_dir": str(td), "summary": str(Path(td) / "summary.json")}
        sp = Path(td) / "spec.json"
        sp.write_text(json.dumps(spec))
        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
        env.update(PATH="/usr/bin:/bin", V4B_SPEC=str(sp))
        r = subprocess.run(["freecadcmd", str(ROOT / "scripts/cad/v4b_freecad.py")], env=env, capture_output=True, text=True)
        if r.returncode or not (Path(td) / "summary.json").exists():
            raise SystemExit(r.stdout[-3000:] + r.stderr[-3000:])
        summary = json.loads((Path(td) / "summary.json").read_text())
        import shutil
        for name, rec in summary.items():
            dst = (MJCF_DIR / "meshes") if parts[name].get("acrylic") else DATA
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(td) / (rec["file"] + ".stl"), dst / (rec["file"] + ".stl"))
            rec["stl"] = str((dst / (rec["file"] + ".stl")).relative_to(ROOT))
            rec["step_local"] = str((STEP_DIR / (rec["file"] + ".step")).relative_to(ROOT))
            if not parts[name].get("acrylic"):  # the MJCF reads the printed parts from the same files
                (MJCF_DIR / "meshes").mkdir(parents=True, exist_ok=True)
                shutil.copy(Path(td) / (rec["file"] + ".stl"), MJCF_DIR / "meshes" / (rec["file"] + ".stl"))
    res = {"script": "scripts/cad/v4b_build.py", "command": "printed", "parts": summary,
           "knuckle": {"c_mm": knuckle_spec()["c"], "x_a_mm": knuckle_spec()["x_a"],
                       "pin_a": knuckle_spec()["a"], "pin_b": knuckle_spec()["b"]}}
    (DATA / "20261119-v4b_printed.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def dxf_entities(f, du, dv, rot):
    """DXF R12 entity text for one part, its (u, v) moved by (du, dv) after a rotation of rot * 90 deg."""
    def tr(P):
        P = np.asarray(P, float)
        if rot:
            P = np.stack([-P[:, 1], P[:, 0]], 1)
        return P + np.array([du, dv])
    out = []

    def poly(P, layer="CUT"):
        P = tr(P)
        s_ = [f"0\nPOLYLINE\n8\n{layer}\n66\n1\n70\n1\n10\n0.0\n20\n0.0\n30\n0.0"]
        s_ += [f"0\nVERTEX\n8\n{layer}\n10\n{x:.4f}\n20\n{y:.4f}\n30\n0.0" for x, y in P]
        s_.append(f"0\nSEQEND\n8\n{layer}")
        out.append("\n".join(s_))
    poly(f.outline)
    for hp in f.polys:
        poly(hp)
    for hu, hv, hd in f.holes:
        (x, y), = tr([[hu, hv]])
        out.append(f"0\nCIRCLE\n8\nCUT\n10\n{x:.4f}\n20\n{y:.4f}\n30\n0.0\n40\n{hd / 2:.4f}")
    return out


def text_entity(x, y, h, txt):
    return f"0\nTEXT\n8\nENGRAVE\n10\n{x:.3f}\n20\n{y:.3f}\n30\n0.0\n40\n{h:.2f}\n1\n{txt}"


def label_point(f, rot):
    """A point inside the part, away from its holes, for the engraved ID (the strip with the widest interval)."""
    best = None
    for p in f.strips(1.0):
        if best is None or p["h"][0] > best["h"][0]:
            best = p
    c = best["c"] - f.o
    u, v = float(c @ f.U), float(c @ f.V)
    return (u, v) if not rot else (-v, u)


def least_web(f, step=0.1):
    """The narrowest strip of plate between two cut contours of a part (outline, round holes, slots), in mm."""
    from scipy.spatial import cKDTree

    def dense(P):
        P = np.asarray(P)
        Q = np.roll(P, -1, 0)
        return np.concatenate([a + np.linspace(0, 1, max(1, int(math.ceil(np.linalg.norm(b - a) / step))), endpoint=False)[:, None] * (b - a)
                               for a, b in zip(P, Q)])
    cs = [("outline", dense(f.outline))] + [(f"slot {k + 1}", dense(p)) for k, p in enumerate(f.polys)]
    cs += [(f"hole {d:g} mm", dense(arc((u, v), d / 2, 0, 2 * math.pi))) for u, v, d in f.holes]
    best = (math.inf, None)
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            dmin = float(cKDTree(cs[j][1]).query(cs[i][1])[0].min())
            if dmin < best[0]:
                best = (dmin, (cs[i][0], cs[j][0]))
    return best


def cmd_dxf(sheet_w=130.0, gap=4.0):
    """The acrylic parts (two of each: one spare) and the fit coupon on one sheet, a preview and a parts summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H1, R1 = acrylic_flats()
    parts = [H1, H1, R1, R1, fit_coupon()]
    placed, x, y, row_h = [], 0.0, 0.0, 0.0
    for f in sorted(parts, key=lambda f_: -(f_.outline[:, 0].max() - f_.outline[:, 0].min())):
        P = f.outline
        w, h = P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min()
        rot = h > w * 1.6 and h > 40.0
        if rot:
            w, h = h, w
        if x + w > sheet_w:
            x, y, row_h = 0.0, y + row_h + gap, 0.0
        u0 = (-P[:, 1].max() if rot else P[:, 0].min())
        v0 = (P[:, 0].min() if rot else P[:, 1].min())
        placed.append((f, x - u0, y - v0, rot, (x, y, w, h)))
        x += w + gap
        row_h = max(row_h, h)
    ents = []
    for f, du, dv, rot, _bb in placed:
        ents += dxf_entities(f, du, dv, rot)
        lu, lv = label_point(f, rot)
        tag = f.name.split()[0]
        ents.append(text_entity(lu + du - 1.2 * len(tag) * 0.6, lv + dv - 0.9, 1.8, tag))
        for (tu, tv, txt) in getattr(f, "labels", []):
            ents.append(text_entity(tu + du, tv + dv, 1.5, txt))
    H = y + row_h
    ents.append(text_entity(0.0, -6.0, 3.0, "v4b PIP linkage, 1/8 in (3.175 mm) acrylic, 2026-11-20; CUT red, ENGRAVE blue"))
    head = ("0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n"
            "0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n70\n2\n"
            "0\nLAYER\n2\nCUT\n70\n0\n62\n1\n6\nCONTINUOUS\n0\nLAYER\n2\nENGRAVE\n70\n0\n62\n5\n6\nCONTINUOUS\n"
            "0\nENDTAB\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n")
    CUT.mkdir(parents=True, exist_ok=True)
    out = CUT / "20261119-v4b_acrylic_3mm.dxf"
    out.write_text(head + "\n".join(ents) + "\n0\nENDSEC\n0\nEOF\n")
    fig, ax = plt.subplots(figsize=(10, 10 * (H + 12) / sheet_w))
    for f, du, dv, rot, _bb in placed:
        def tr(P):
            P = np.asarray(P, float)
            if rot:
                P = np.stack([-P[:, 1], P[:, 0]], 1)
            return P + np.array([du, dv])
        P = tr(f.outline)
        ax.fill(P[:, 0], P[:, 1], color="#e8eef7", lw=0)
        ax.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], color="#c0392b", lw=0.9)
        for hu, hv, hd in f.holes:
            (cx, cy), = tr([[hu, hv]])
            ax.add_patch(plt.Circle((cx, cy), hd / 2, fill=False, color="#c0392b", lw=0.8))
        lu, lv = label_point(f, rot)
        ax.text(lu + du, lv + dv, f.name.split()[0], color="#1f3f8f", fontsize=13, ha="center", va="center", family="serif")
    ax.plot([0, sheet_w, sheet_w, 0, 0], [-2, -2, H + 2, H + 2, -2], color="0.6", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xlim(-5, sheet_w + 5)
    ax.set_ylim(-8, H + 6)
    ax.set_xlabel("mm", fontsize=13)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / "20261119-v4b_cut_sheet.png", dpi=110)
    plt.close(fig)
    summary, seen = [], set()
    for f, *_r, bb in placed:
        if f.name in seen:
            continue
        seen.add(f.name)
        w, pair = least_web(f)
        summary.append({"id": f.name.split()[0], "name": f.name, "body": f.body, "qty": sum(1 for q in placed if q[0].name == f.name),
                        "note": f.note, "size_mm": [round(bb[2], 1), round(bb[3], 1)], "holes": len(f.holes),
                        "least_web_mm": round(w, 2) if math.isfinite(w) else None, "least_web_between": pair})
    res = {"script": "scripts/cad/v4b_build.py", "command": "dxf", "dxf": str(out.relative_to(ROOT)),
           "sheet_mm": [sheet_w, round(H, 1)], "thickness_mm": T, "parts": summary}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "20261119-v4b_parts.json").write_text(json.dumps(res, indent=1))
    return res


BOM = [  # item, qty, specification, where, example source, note
    ("STS3032 servo", 3, "Feetech STS3032, 6 V, 4.5 kg cm, 25-tooth spline", "yaw, MCP, PIP", "on hand",
     "each comes with an 18 mm aluminium disc horn and its M2.3 horn screw"),
    ("Carriage P1 (printed)", 1, "PLA, 4 walls, 40 % infill: data/20261119-v4b_carriage.stl", "gantry block and lead nut; yaw servo; rear dowel", "printed", ""),
    ("Yoke P2 (printed)", 1, "PLA, 4 walls, 40 % infill: data/20261119-v4b_yoke.stl", "yaw stage, MCP servo, MCP idler", "printed", ""),
    ("Proximal link P3 (printed)", 1, "PLA, 4 walls, 40 % infill: data/20261119-v4b_link.stl", "MCP to PIP; PIP servo", "printed", ""),
    ("Knuckle P4 (printed)", 1, "PLA or PETG, 100 % infill: data/20261119-v4b_knuckle.stl", "between H1 and R1", "printed", ""),
    ("Print fit coupon P0 (printed)", 1, "PLA: data/20261119-v4b_print_fit_coupon.stl", "print first", "printed",
     "picks the bearing bores, the dowel press fit and the M2 hole"),
    ("Acrylic sheet", 1, "1/8 in (3.175 mm) cast acrylic, at least 130 x 60 mm", "H1, R1 (with spares) and C1", "", ""),
    ("Flanged ball bearing", 6, "MF63ZZ: 3 x 6 x 2.5 mm, flange 7.2 x 0.6 mm",
     "P3 idler side, P3 PIP cheeks (2), P2 rear arm, H1 (pin a), R1 (pin b)", "any MF63ZZ", "order 10"),
    ("Spherical plain bearing", 1, "3 mm bore, 12 mm outside, inner ring 4.8 mm, outer ring 3.6 mm wide, steel", "R1 at the lug",
     "RS PRO 382-380", "order 2; bonded into R1"),
    ("Shoulder screw 3 x 12", 1, "3 mm precision shoulder, 12 mm long, M2 x 4 thread, low head 5 x 1.3 mm", "pin a (knuckle to H1)",
     "McMaster 90323A609", ""),
    ("Shoulder screw 3 x 7", 1, "as above, 7 mm shoulder", "pin b (knuckle to R1)", "McMaster 90323A606", ""),
    ("Dowel pin 3 x 8", 2, "3 mm steel dowel, 8 mm long, h8 or m6", "MCP idler (pressed into P2's idler arm); yaw rear dowel (bonded into P1's rear wall)",
     "ISO 2338 or ISO 8734", "each slides in a bearing's inner ring"),
    ("Dowel pin 3 x 6", 2, "3 mm steel dowel, 6 mm long, h8 or m6", "PIP pins, pressed into the DIP link's cheeks", "ISO 2338 or ISO 8734",
     "slide in the PIP bearings' inner rings"),
    ("Dowel pin 3 x 12", 1, "3 mm steel dowel, 12 mm long", "lug pin, through both cheeks and the spherical bearing", "ISO 2338 or ISO 8734", ""),
    ("M2 hex nut", 2, "M2, 4 mm across flats, 1.6 mm thick", "pins a and b, inside the bearings' bores", "", ""),
    ("M2 x 6 screw", 10, "M2 x 6 socket head", "yaw servo tabs into P1 (2); disc horns: P2 to the yaw horn (4), P3 to the MCP horn (4)", "",
     "into the horns' tapped holes or 1.8 mm printed holes"),
    ("M2 x 5 screw", 10, "M2 x 5 socket head", "MCP servo tabs into P2 (4), PIP servo tabs into P3 (4), H1 to the PIP horn (2)", "",
     "self-forming threads in 1.8 mm printed holes"),
    ("Gantry screws", 7, "4 x M2 into the MGN7C block, 3 x M3 x 10 into the lead nut's flange", "P1 to the index or middle gantry", "as on the v3 carriage",
     "M2 length: the 5 mm plate plus the block's thread depth"),
    ("Retaining compound or epoxy", 1, "Loctite 638 or 5-minute epoxy", "spherical bearing in R1; yaw rear dowel in P1; any loose bearing", "", ""),
    ("DIP link (printed)", 1, "b v4's tongue, board seat and column on new cheeks (the page's interface table)", "PIP to fingertip",
     "printed", "to be drawn from b v4 in Inventor"),
    ("Fingertip", 1, "v4b tip: cassette 3T and cap TW15H3 with the ReSkin board", "on the DIP link", "existing prints", ""),
]


def cmd_bom():
    import csv
    K = knuckle_spec()
    names = " ".join(r[0] for r in BOM)
    for pin in ("a", "b"):
        assert f"Shoulder screw 3 x {K[pin]['shoulder']:.0f}" in names, f"BOM lacks pin {pin}'s shoulder screw"
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "20261119-v4b_bom.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "qty", "specification", "where", "example source", "note"])
        for row in BOM:
            w.writerow(row)
    return str(out.relative_to(ROOT))


def cmd_pip():
    """The PIP drive along the PIP angle (v4c_hardware's pip_range with this layout's numbers): where the rod closes with
    the horn within its limit, the joint torque per SCS0009 limit and the spherical bearing's tilt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import v4c_hardware as V
    pip_kin(np.array([0.0]))  # patches V.HW with this layout's PIP numbers
    R = V.pip_range()
    PIP, lo, hi = V.PIP, R["lo"], R["hi"]
    th = np.degrees(R["S"]["th"])
    want = (PIP >= -25) & (PIP <= 92) & R["ok"]
    res = {"script": "scripts/cad/v4b_build.py", "command": "pip", "params": B["pip"], "proximal_mm": B["proximal"],
           "rod_mm": round(R["g"]["L"], 3), "range_deg": R["range"], "grid_deg": [float(PIP[0]), float(PIP[-1])],
           "torque_scs0009_min_m25_92": round(float(R["tq"][want].min()), 2),
           "torque_scs0009_median_m25_92": round(float(np.median(R["tq"][want])), 2),
           "torque_scs0009_min_range": round(float(R["tq"][lo:hi + 1].min()), 2),
           "bearing_tilt_deg_max": round(float(R["tilt"][lo:hi + 1].max()), 1),
           "horn_deg_at_ends": [round(float(th[lo]), 1), round(float(th[hi]), 1)],
           "curve": {"pip": PIP[lo:hi + 1].tolist(), "torque": np.round(R["tq"][lo:hi + 1], 3).tolist(),
                     "horn": np.round(th[lo:hi + 1], 2).tolist(), "tilt": np.round(R["tilt"][lo:hi + 1], 2).tolist()}}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "20261119-v4b_pip.json").write_text(json.dumps(res, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = PIP[lo:hi + 1]
    ax[0].plot(x, R["tq"][lo:hi + 1], color="#1f3f8f", lw=2)
    ax[0].axhline(1.0, color="0.4", ls="--", lw=1)
    ax[0].set_ylabel("PIP torque per SCS0009 limit", fontsize=13)
    ax[1].plot(x, th[lo:hi + 1], color="#1f3f8f", lw=2, label="horn angle")
    ax[1].plot(x, R["tilt"][lo:hi + 1], color="#c0392b", lw=2, label="bearing tilt")
    ax[1].axhline(V.HW["hardware"]["spherical_bearing"]["tilt_deg"], color="#c0392b", ls=":", lw=1)
    ax[1].set_ylabel("angle (deg)", fontsize=13)
    ax[1].legend(fontsize=12, frameon=False)
    for a_ in ax:
        a_.axvspan(-18, 92, color="0.92", zorder=0)
        a_.set_xlabel("PIP angle (deg)", fontsize=13)
        a_.tick_params(labelsize=12)
    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / "20261119-v4b_pip_drive.png", dpi=110)
    plt.close(fig)
    return {k: v for k, v in res.items() if k != "curve"}


def yaw_supports(pad, F, horn_x, rear_x, Gz):
    """Forces on the yaw stage's supports from a force F (N) at the pad (mm): with the rear bearing, the radial forces at
    the horn and at the rear dowel; without it, the force and bending moment that the servo's output takes at the horn."""
    h = np.array([horn_x, 0.0, Gz])
    M = np.cross((pad - h) * 1e-3, F)  # N m about the horn
    out = {"bending_Nm_at_horn": round(float(np.hypot(M[1], M[2])), 3), "yaw_torque_Nm": round(float(M[0]), 3)}
    if rear_x is not None:
        Ls = (rear_x - horn_x) * 1e-3
        Fr = np.array([0.0, -M[2] / Ls, M[1] / Ls])
        Fh = -F - Fr
        out.update({"span_mm": round(rear_x - horn_x, 1), "rear_radial_N": round(float(np.hypot(Fr[1], Fr[2])), 2),
                    "horn_radial_N": round(float(np.hypot(Fh[1], Fh[2])), 2)})
    else:
        out.update({"horn_radial_N": round(float(np.hypot(F[1], F[2])), 2)})
    return out


def cmd_loads(F_pinch=(3.0, 12.1)):
    """The yaw stage's supports and the levers from the lead nut, for this build against the acrylic v4b (yaw servo's
    spline 37 mm behind G, no rear bearing, proximal 60 mm) and v3 (yaw axis at the mount, yaw link 20.75 mm, proximal
    30.8 mm). The pad is 7.4 mm palmar of the finger's axis and 36.36 mm below the PIP axis at MCP and PIP 0. F_pinch:
    3 N (a light pinch) and 12.1 N (the STS3032's stall at the MCP)."""
    g = geometry()
    nut_z = LEAD_NUT["axis"][1]
    designs = {"v3": {"Gz": 0.0, "pip_z": -51.55, "horn_x": None, "rear_x": None},
               "v4b acrylic (2026-11-19)": {"Gz": -7.0, "pip_z": -67.0, "horn_x": 36.0, "rear_x": None},
               "v4b mixed (this build)": {"Gz": g["Gz"], "pip_z": g["pz"], "horn_x": g["xY"] - B["disc"]["flange"] / 2,
                                          "rear_x": g["Y4_x"][1] - B["mf63"]["B"] / 2}}
    res = {"script": "scripts/cad/v4b_build.py", "command": "loads", "nut_axis_z_mm": nut_z, "designs": {}}
    for name, d in designs.items():
        pad = np.array([-7.4, 0.0, d["pip_z"] - 36.36])
        rec = {"pad_mm": pad.tolist(), "nut_to_pad_mm": round(nut_z - pad[2], 1), "gimbal_to_pad_mm": round(d["Gz"] - pad[2], 1)}
        if d["horn_x"] is not None:
            rec["horn_reach_mm"] = round(d["horn_x"], 1)
            for F in F_pinch:
                for lab, vec in (("pinch", np.array([F, 0.0, 0.0])), ("lateral", np.array([0.0, F, 0.0])), ("vertical", np.array([0.0, 0.0, F]))):
                    rec[f"{lab}_{F:g}N"] = yaw_supports(pad, vec, d["horn_x"], d["rear_x"], d["Gz"])
        res["designs"][name] = rec
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "20261119-v4b_loads.json").write_text(json.dumps(res, indent=1))
    return res


def cmd_keepout(step=2.0):
    """The rod's sweep in the DIP link's frame over the PIP drive's closing range, as overlapping boxes in one STL
    (data/20261119-v4b_dip_keepout.stl): origin on the PIP axis, x dorsal, y along the PIP axis, z toward the MCP at
    PIP 0, mm. Boxes farther than 25 mm from the PIP axis are left out."""
    import trimesh
    import v4c_hardware as Vh
    pip_kin(np.array([0.0]))
    lo, hi = Vh.pip_range()["range"]
    pips = np.arange(lo, hi + 1e-9, step)
    S, g = pip_kin(pips)
    prims = rod_flat(g["L"]).strips() + rod_solids(g["L"])
    Op = np.array([0.0, 0.0, geometry()["pz"]])
    meshes = []
    for i, pip in enumerate(pips):
        A, Qp, pb = S["A"][i], S["Q"][i], S["pb"][i]
        d = (A - Qp) / np.linalg.norm(A - Qp)
        Rr = np.stack([d, np.cross(pb, d), pb], 1)
        Rd = rot_y(-math.radians(pip))
        for p in prims:
            c = Rd @ (Qp + Rr @ p["c"] - Op)
            if np.linalg.norm(c[[0, 2]]) > 25.0:
                continue
            m = trimesh.creation.box(extents=2 * p["h"])
            Tm = np.eye(4)
            Tm[:3, :3], Tm[:3, 3] = Rd @ Rr @ p["R"], c
            m.apply_transform(Tm)
            meshes.append(m)
    km = trimesh.util.concatenate(meshes)
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "20261119-v4b_dip_keepout.stl"
    km.export(out)
    ext = km.bounds
    return {"stl": str(out.relative_to(ROOT)), "pip_range_deg": [float(lo), float(hi)], "step_deg": step, "boxes": len(meshes),
            "bounds_mm": np.round(ext, 2).tolist(), "y_extent_mm": round(float(np.abs(ext[:, 1]).max()), 2)}


# ------------------------------------------------------------------------------------------------- MJCF
COLORS = {"carriage": (0.80, 0.80, 0.82, 1), "yoke": (0.45, 0.62, 0.85, 1), "link": (0.86, 0.56, 0.34, 1), "dip": (0.55, 0.75, 0.45, 1),
          "phorn": (0.92, 0.92, 0.95, 0.75), "knuckle": (0.85, 0.75, 0.25, 1), "rod": (0.92, 0.92, 0.95, 0.75),
          "servo": (0.12, 0.12, 0.14, 1), "metal": (0.70, 0.70, 0.72, 1), "tip": (0.15, 0.15, 0.15, 1)}
MASS_G = {"carriage": 45.0, "yoke": 40.0, "link": 45.0, "phorn": 4.0, "knuckle": 1.5, "rod": 3.0, "dip": 15.0}


def pip_rest_pose():
    """The PIP drive solved at the rest PIP angle (link frame, G at (0, 0, Gz)): the MJCF's zero pose. The horn sits a
    few degrees off the crank's drawn direction there, since the solver also keeps the rod square to pin b."""
    S, g = pip_kin(np.array([B["pip"]["pip_rest"]]))
    k, pb = S["k"][0], S["pb"][0]
    d = (S["A"][0] - S["Q"][0]) / np.linalg.norm(S["A"][0] - S["Q"][0])
    return {"th": float(S["th"][0]), "psi": float(S["psi"][0]), "Pa": S["Pa"][0], "Q": S["Q"][0], "A": S["A"][0],
            "Rk": np.stack([X, k, pb], 1), "Rr": np.stack([d, np.cross(pb, d), pb], 1), "L": g["L"], "g": g}


def pip_joint_angles(pip_deg):
    """The closed PIP drive's joint angles (rad) at PIP angles, from the MJCF's zero pose (pip_rest_pose): the horn,
    pin a (knuckle on the horn) and pin b (rod on the knuckle); closure flag."""
    pip_deg = np.atleast_1d(np.asarray(pip_deg, float))
    S, _g = pip_kin(pip_deg)
    R0 = pip_rest_pose()
    d0k = R0["Rk"].T @ R0["Rr"][:, 0]
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi  # noqa: E731
    out = []
    for i in range(len(pip_deg)):
        Rk = np.stack([X, S["k"][i], S["pb"][i]], 1)
        d = (S["A"][i] - S["Q"][i]) / np.linalg.norm(S["A"][i] - S["Q"][i])
        dk = Rk.T @ d
        qh = float(S["th"][i]) - R0["th"]
        qa = wrap(float(S["psi"][i]) - R0["psi"] - qh)
        qb = wrap(math.atan2(dk[1], dk[0]) - math.atan2(d0k[1], d0k[0]))
        out.append((wrap(qh), qa, qb, bool(S["ok"][i])))
    return out


def build_mjcf(fingers=("index", "middle"), with_gantry=True):
    """An MjSpec of the v4b modules on their gantries: the carriage slides along x (stroke), the yoke turns about x
    (yaw), the link about y (mcp), the horn about the PIP servo's shaft; pins a and b close the PIP drive onto the DIP
    link through a ball joint (an equality connect at the lug). Printed and acrylic parts are their FreeCAD STLs (visual);
    every printed piece is also a convex collision mesh, the servos are boxes."""
    import mujoco
    g = geometry()
    R0 = pip_rest_pose()
    gk = R0["g"]
    m = 1e-3
    spec = mujoco.MjSpec()
    spec.modelname = "v4b_mixed"
    spec.compiler.degree = False
    spec.option.timestep = 0.002
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT  # the yoke against the link and the carriage against the yoke count
    spec.meshdir = str(MJCF_DIR) + "/"
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[1, 1, 1], rgb2=[0.85, 0.88, 0.92], width=64, height=64)
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.93, 0.93, 0.93], rgb2=[0.86, 0.86, 0.86], width=256, height=256)
    mat = spec.add_material(name="grid", texrepeat=[8, 8])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.4, 0.4, 0.01], material="grid", contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[0.1, -0.3, 0.8], dir=[-0.1, 0.3, -1.0], diffuse=[0.7, 0.7, 0.7], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.worldbody.add_light(pos=[-0.3, 0.3, 0.6], dir=[0.3, -0.3, -0.6], diffuse=[0.35, 0.35, 0.35], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    meshdir = MJCF_DIR / "meshes"
    files = {"carriage": "20261119-v4b_carriage", "yoke": "20261119-v4b_yoke", "link": "20261119-v4b_link",
             "knuckle": "20261119-v4b_knuckle", "phorn": "20261119-v4b_H1", "rod": "20261119-v4b_R1"}
    for b_, f in files.items():
        spec.add_mesh(name=f"mesh_{b_}", file=f"meshes/{f}.stl", scale=[m, m, m])
    pieces = {}
    for _name, (pcs, _cuts, body) in printed_parts().items():
        for k, pc in enumerate(pcs):
            Vt, _F = pc.prism()
            nm = f"col_{body}_{k}"
            spec.add_mesh(name=nm, uservert=(Vt * m).ravel().tolist())
            pieces.setdefault(body, []).append(nm)
    gant = ["print", "steel", "stepper", "alu", "brass"]
    if with_gantry:
        for f in fingers:
            for nm in gant:
                spec.add_mesh(name=f"g_{f}_{nm}", file=f"../20261104-cad_v3/meshes/v3_bridge_{f}_{nm}.stl")
        for nm in ("plate", "print", "steel", "stepper", "mount"):
            p_ = GANTRY_MESHES / f"v3_palm_{nm}.stl"
            if p_.exists():
                spec.add_mesh(name=f"palm_{nm}", file=f"../20261104-cad_v3/meshes/v3_palm_{nm}.stl")
        for nm in gant:
            spec.add_mesh(name=f"g_thumb_{nm}", file=f"../20261104-cad_v3/meshes/v3_bridge_thumb_{nm}.stl")
    palm = spec.worldbody.add_body(name="palm", pos=[0.0, 0.0, 0.32])
    if with_gantry:
        gcol = {"print": (0.55, 0.57, 0.60, 1), "steel": (0.7, 0.7, 0.72, 1), "stepper": (0.15, 0.15, 0.17, 1),
                "alu": (0.8, 0.8, 0.82, 1), "brass": (0.78, 0.65, 0.35, 1)}
        for nm in ("plate", "print", "steel", "stepper", "mount"):
            if (GANTRY_MESHES / f"v3_palm_{nm}.stl").exists():
                palm.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"palm_{nm}", rgba=gcol.get(nm, (0.6, 0.6, 0.62, 1)),
                              contype=0, conaffinity=0, group=0)  # group 0: the palm and the thumb's gantry (renders hide it)
        for nm in gant:
            palm.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"g_thumb_{nm}", rgba=gcol[nm], contype=0, conaffinity=0, group=0)
    PF = {"index": P_INDEX, "middle": np.array([56.1813, -58.3998, 0.0])}
    G = g["G"]
    Cs = np.array([g["xs"], 0.0, g["zs"]])
    Op = np.array([0.0, 0.0, g["pz"]])
    Rk0, Rr0, Pa0, Q0, A0 = R0["Rk"], R0["Rr"], R0["Pa"], R0["Q"], R0["A"]  # pip_kin's points already include Gz
    Rh0 = rot_x(R0["th"])
    Sd = solids()
    dip_boxes = Sd["dip"] + dip_lug_prims(dip_lug(gk))

    def add_prims(body, prims, origin, rgba, collide=True, Rb=np.eye(3)):
        for p in prims:
            if np.any(p["h"] <= 0):
                continue
            c = Rb.T @ (p["c"] - origin)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(p["h"] * m).tolist(), pos=(c * m).tolist(),
                          quat=mat_to_quat(Rb.T @ p["R"]).tolist(), rgba=rgba, contype=1 if collide else 0,
                          conaffinity=1 if collide else 0, group=2 if collide else 1, density=0)

    def add_mesh(body, mesh, origin, rgba, Rb=np.eye(3)):
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, pos=(-(Rb.T @ origin) * m).tolist(),
                      quat=mat_to_quat(Rb.T).tolist(), rgba=rgba, contype=0, conaffinity=0, group=1, density=0)

    def add_cols(body, b_, origin):
        for nm in pieces.get(b_, []):
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=nm, pos=(-origin * m).tolist(), rgba=(1, 0, 0, 0.3),
                          contype=1, conaffinity=1, group=3, density=0)

    def inertial(body, b_):
        body.mass = MASS_G[b_] * 1e-3
        body.inertia = [2e-6, 2e-6, 2e-6]
        body.explicitinertial = True

    servo_like = lambda n: n.endswith(("_case", "_tabs", "_boss")) or n.startswith(("yawdisc", "mcpdisc", "pipdisc"))  # noqa: E731
    for f in fingers:
        base = palm.add_body(name=f"{f}_mount", pos=(PF[f] * m).tolist())
        if with_gantry:
            for nm in gant:
                base.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"g_{f}_{nm}", pos=(-PF[f] * m).tolist(), rgba=gcol[nm],
                              contype=0, conaffinity=0, group=1)
            cy, cz = 0.0, 31.6
            base.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[-53.9 * m, cy, cz * m, 46.6 * m, cy, cz * m], size=[2.5 * m, 0, 0],
                          rgba=gcol["steel"], contype=0, conaffinity=0, group=1)
        car = base.add_body(name=f"{f}_carriage")
        inertial(car, "carriage")
        car.add_joint(name=f"{f}_stroke", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[STROKE[0] * m, STROKE[1] * m],
                      limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=5.0, armature=0.05)
        add_mesh(car, "mesh_carriage", np.zeros(3), COLORS["carriage"])
        add_cols(car, "carriage", np.zeros(3))
        for p in Sd["carriage"]:
            col = COLORS["servo"] if p["name"].startswith("yawservo") else (COLORS["metal"] if not p["name"].startswith("leadnut") else (0.78, 0.65, 0.35, 1))
            add_prims(car, [p], np.zeros(3), col, collide=not servo_like(p["name"]) and not p["name"].startswith(("leadnut", "mgn7c", "rear_dowel")))
        yoke = car.add_body(name=f"{f}_yoke", pos=(G * m).tolist())
        inertial(yoke, "yoke")
        yoke.add_joint(name=f"{f}_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-math.radians(85), math.radians(85)],
                       limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.02, armature=0.002)
        add_mesh(yoke, "mesh_yoke", G, COLORS["yoke"])
        add_cols(yoke, "yoke", G)
        for p in Sd["yoke"]:
            col = COLORS["servo"] if p["name"].startswith("mcpservo") else COLORS["metal"]
            add_prims(yoke, [p], G, col, collide=not servo_like(p["name"]) and not p["name"].startswith(("idler_dowel", "flange")))
        link = yoke.add_body(name=f"{f}_link")
        inertial(link, "link")
        link.add_joint(name=f"{f}_mcp", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[math.radians(MCP_R[0]), math.radians(MCP_R[1])],
                       limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.02, armature=0.002)
        add_mesh(link, "mesh_link", G, COLORS["link"])
        add_cols(link, "link", G)
        for p in Sd["link"]:
            col = COLORS["servo"] if p["name"].startswith("pipservo") else COLORS["metal"]
            add_prims(link, [p], G, col, collide=not servo_like(p["name"]) and not p["name"].startswith("flange"))
        horn = link.add_body(name=f"{f}_horn", pos=((Cs - G) * m).tolist(), quat=mat_to_quat(Rh0).tolist())
        inertial(horn, "phorn")
        horn.add_joint(name=f"{f}_pip_horn", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0],
                       range=[-math.radians(B["pip"]["horn_limit_deg"]), math.radians(B["pip"]["horn_limit_deg"])],
                       limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.01, armature=0.001)
        add_mesh(horn, "mesh_phorn", Cs, COLORS["phorn"])  # drawn at horn angle 0; the body starts at Rh0
        for p in Sd["phorn"]:
            add_prims(horn, [p], Cs, COLORS["metal"], collide=False)
        kn = horn.add_body(name=f"{f}_knuckle", pos=((Rh0.T @ (Pa0 - Cs)) * m).tolist(), quat=mat_to_quat(Rh0.T @ Rk0).tolist())
        inertial(kn, "knuckle")
        kn.add_joint(name=f"{f}_pin_a", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], damping=0.002, armature=0.0005)
        kn.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="mesh_knuckle", rgba=COLORS["knuckle"], contype=0, conaffinity=0, group=1, density=0)
        add_prims(kn, [p for p in knuckle_prims(knuckle_spec()) if p["name"].startswith("pin_")], np.zeros(3), COLORS["metal"], collide=False)
        rod = kn.add_body(name=f"{f}_rod", pos=((Rk0.T @ (Q0 - Pa0)) * m).tolist(), quat=mat_to_quat(Rk0.T @ Rr0).tolist())
        inertial(rod, "rod")
        rod.add_joint(name=f"{f}_pin_b", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0.002, armature=0.0005)
        rod.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="mesh_rod", rgba=COLORS["rod"], contype=0, conaffinity=0, group=1, density=0)
        add_prims(rod, rod_solids(gk["L"]), np.zeros(3), COLORS["metal"], collide=False)
        rod.add_site(name=f"{f}_lug_rod", pos=[R0["L"] * m, 0, 0], size=[0.0008, 0, 0], rgba=(1, 0, 0, 1))
        dip = link.add_body(name=f"{f}_dip", pos=((Op - G) * m).tolist())
        inertial(dip, "dip")
        dip.add_joint(name=f"{f}_pip", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[math.radians(-45), math.radians(96)],
                      limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.01, armature=0.001)
        for p in dip_boxes:
            add_prims(dip, [p], Op, COLORS["tip"] if p["name"] == "tip" else COLORS["dip"], collide=True)
        dip.add_site(name=f"{f}_lug_dip", pos=((A0 - Op) * m).tolist(), size=[0.0008, 0, 0], rgba=(0, 0, 1, 1))
        eq = spec.add_equality(name=f"{f}_lug", type=mujoco.mjtEq.mjEQ_CONNECT, name1=f"{f}_rod", name2=f"{f}_dip",
                               objtype=mujoco.mjtObj.mjOBJ_BODY)
        eq.data[0:3] = [R0["L"] * m, 0.0, 0.0]
        eq.solref = [0.004, 1.0]
        for jn, kp, rng in (("stroke", 400.0, STROKE), ("yaw", 3.0, (-85, 85)), ("mcp", 3.0, MCP_R),
                            ("pip_horn", 1.5, (-B["pip"]["horn_limit_deg"], B["pip"]["horn_limit_deg"]))):
            lo_, hi_ = (rng[0] * m, rng[1] * m) if jn == "stroke" else (math.radians(rng[0]), math.radians(rng[1]))
            a = spec.add_actuator(name=f"{f}_{jn}", target=f"{f}_{jn}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=kp, kv=0.1 * kp if jn != "stroke" else 20.0)
            a.ctrlrange = [lo_, hi_]
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    return spec


def cmd_mjcf():
    """assets/mjcf/experimental/20261120-v4b_mixed/: v4b_mixed_index.xml (the index module on its gantry) and
    v4b_mixed_pair.xml (index and middle with the palm and the thumb's gantry); keys home, flexed, abducted, curled."""
    import mujoco
    MJCF_DIR.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, fingers in (("v4b_mixed_index", ("index",)), ("v4b_mixed_pair", ("index", "middle"))):
        spec = build_mjcf(fingers)
        model = spec.compile()
        keys = {"home": (0.0, 0.0, 0.0, 0.0), "flexed": (0.0, 0.0, 60.0, 60.0), "abducted": (-20.0, 45.0, 20.0, 20.0), "curled": (25.0, -30.0, 85.0, 85.0)}
        for kname, (s_, yaw, mcp, pip) in keys.items():
            q = np.zeros(model.nq)
            ctrl = np.zeros(model.nu)
            th, qa, qb, _ok = pip_joint_angles([pip])[0]
            for f in fingers:
                vals = {"stroke": s_ * 1e-3, "yaw": math.radians(yaw), "mcp": math.radians(mcp), "pip_horn": th, "pin_a": qa, "pin_b": qb,
                        "pip": math.radians(pip)}
                for jn, v in vals.items():
                    q[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{jn}")]] = v
                for an in ("stroke", "yaw", "mcp", "pip_horn"):
                    ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{f}_{an}")] = vals[an]
            k = spec.add_key(name=kname, qpos=q.tolist(), ctrl=ctrl.tolist())
        model = spec.compile()
        import re
        xml = re.sub(r' meshdir="[^"]*"', "", spec.to_xml())  # mesh paths stay relative to the XML's folder
        out = MJCF_DIR / f"{name}.xml"
        out.write_text(xml)
        # closure check: at each key, the lug points of the rod and the DIP link coincide
        d = mujoco.MjData(model)
        errs = []
        for kk in range(model.nkey):
            mujoco.mj_resetDataKeyframe(model, d, kk)
            mujoco.mj_forward(model, d)
            for f in fingers:
                a = d.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{f}_lug_rod")]
                b = d.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{f}_lug_dip")]
                errs.append(float(np.linalg.norm(a - b) * 1e3))
        res[name] = {"xml": str(out.relative_to(ROOT)), "nbody": model.nbody, "ngeom": model.ngeom, "nu": model.nu,
                     "closure_mm_max": round(max(errs), 4), "keys": list(keys)}
    res.update(mjcf_drive_test())
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "20261119-v4b_mjcf.json").write_text(json.dumps({"script": "scripts/cad/v4b_build.py", "command": "mjcf", **res}, indent=1))
    return res


def mjcf_drive_test(targets=((20.0, 30.0, 45.0, 60.0), (-25.0, -40.0, 80.0, 85.0), (10.0, 60.0, -10.0, -20.0)), secs=3.0):
    """The pair scene under physics: from home, each target (stroke mm, yaw, MCP, PIP deg) for `secs` seconds through
    the four position actuators. The worst tracking error at the end of each move, the worst lug separation while moving
    and the deepest interpenetration."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MJCF_DIR / "v4b_mixed_pair.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    jid = lambda n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]  # noqa: E731
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)  # noqa: E731
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)  # noqa: E731
    track, sep, pen = 0.0, 0.0, 0.0
    for s_, yaw, mcp, pip in targets:
        th = pip_joint_angles([pip])[0][0]
        for f in ("index", "middle"):
            data.ctrl[aid(f + "_stroke")] = s_ * 1e-3
            data.ctrl[aid(f + "_yaw")] = math.radians(yaw)
            data.ctrl[aid(f + "_mcp")] = math.radians(mcp)
            data.ctrl[aid(f + "_pip_horn")] = th
        for _ in range(int(secs / model.opt.timestep)):
            mujoco.mj_step(model, data)
            for f in ("index", "middle"):
                sep = max(sep, float(np.linalg.norm(data.site_xpos[sid(f + "_lug_rod")] - data.site_xpos[sid(f + "_lug_dip")]) * 1e3))
            for i in range(data.ncon):
                pen = max(pen, -float(data.contact[i].dist) * 1e3)
        for f in ("index", "middle"):
            for jn, want in (("yaw", yaw), ("mcp", mcp), ("pip", pip)):
                track = max(track, abs(math.degrees(data.qpos[jid(f"{f}_{jn}")]) - want))
    return {"drive_targets": [list(t) for t in targets], "drive_secs": secs, "tracking_deg_max": round(track, 2),
            "closure_mm_moving": round(sep, 3), "penetration_mm_max": round(pen, 3)}


FLUSH_VARIANTS = [  # the PIP servo's base flush with the fingertip's palmar face (x = -11): spline 21 mm dorsal of G's axis
    ("as built", dict(spline_x=13.0)),
    ("2 mm dorsal", dict(spline_x=15.0)),
    ("8 mm dorsal, drive centred at PIP 20", dict(spline_x=21.0, spline_z=-35.0, pip_rest=20.0, rod_offset_c=4.3)),
    ("8 mm dorsal, centred, link 3 mm longer", dict(spline_x=21.0, spline_z=-38.0, pip_rest=20.0, rod_offset_c=4.3, proximal=69.0)),
    ("8 mm dorsal, long end toward the MCP", dict(spline_x=21.0, spline_z=-42.0, long_end="up", lug_r=7.0, pip_rest=20.0, rod_offset_c=4.3)),
]


def cmd_flush():
    """How far the PIP servo's base stands proud of the fingertip's palmar face, and what moving it flush costs: the PIP
    drive's range and least torque over the driver's PIP range (-18..92 deg) and the least clearance for the variants in
    FLUSH_VARIANTS, and searches over the drive's proportions with the servo flush. data/20261119-v4b_flush.json."""
    import copy
    import v4c_hardware as Vh
    base = copy.deepcopy(B)

    def setp(v):
        B.clear()
        B.update(copy.deepcopy(base))
        v = dict(v)
        if "proximal" in v:
            B["proximal"] = v.pop("proximal")
        B["pip"].update(v)

    def drive():
        pip_kin(np.array([B["pip"]["pip_rest"]]))
        R = Vh.pip_range()
        want = (Vh.PIP >= -18) & (Vh.PIP <= 92)
        closes = bool(R["range"][0] <= -18 and R["range"][1] >= 92)
        return R, closes, (round(float(R["tq"][want].min()), 2) if closes else None)

    designed = lambda k, gv: (("flange" in k and ("dip_cheek" in k or "dip_lugarm" in k) and abs(gv - 0.3) < 0.02)  # noqa: E731
                              or ("flange_Y4" in k and "rear wall" in k))
    res = {"script": "scripts/cad/v4b_build.py", "command": "flush", "tip_palmar_x": B["tip_box"]["x"][0], "variants": [], "searches": {}}
    try:
        for name, v in FLUSH_VARIANTS:
            setp(v)
            R, closes, tq = drive()
            chk = check(pips=(-18.0, 0.0, 20.0, 45.0, 70.0, 92.0))
            moving = [(k, val) for k, val in chk["least"] if not designed(k, val[0])]
            res["variants"].append({"name": name, "params": v, "servo_base_x": round(B["pip"]["spline_x"] - S32["h_spline"], 2),
                                    "pip_range_deg": R["range"], "closes_m18_92": closes, "torque_min_m18_92": tq,
                                    "rod_mm": round(R["g"]["L"], 2), "least": moving[0], "next": moving[1:4]})
            print(name, res["variants"][-1]["pip_range_deg"], tq, moving[0], flush=True)
        # searches with the servo flush: which proportions close over -18..92, and the best one's clearance
        for tag, grid in (("rest 0-40, r 9-14, lug 7-13", dict(long_end=("down", "up"), r=(9.0, 10.0, 12.0, 14.0), lug_r=(7.0, 9.0, 11.0, 13.0),
                                                                pip_rest=(0.0, 20.0, 40.0))),
                          ("larger lug, r 12-18, lug 11-17", dict(long_end=("down", "up"), r=(12.0, 14.0, 16.0, 18.0), lug_r=(11.0, 13.0, 15.0, 17.0),
                                                                    pip_rest=(10.0, 20.0, 30.0)))):
            n, ok = 0, []
            for le, r, lug, rest in itertools.product(*grid.values()):
                n += 1
                setp(dict(spline_x=21.0, spline_z=-35.0 if le == "down" else -42.0, long_end=le, r=r, lug_r=lug, pip_rest=rest, rod_offset_c=4.3))
                try:
                    _R, closes, tq = drive()
                except Exception:  # noqa: BLE001, no rest pose
                    continue
                if closes:
                    ok.append({"long_end": le, "r": r, "lug_r": lug, "pip_rest": rest, "torque_min": tq})
            ok.sort(key=lambda d: -d["torque_min"])
            best_by = {}
            for le in ("down", "up"):
                cand = [d for d in ok if d["long_end"] == le]
                if cand:
                    d = cand[0]
                    setp(dict(spline_x=21.0, spline_z=-35.0 if le == "down" else -42.0, long_end=le, r=d["r"], lug_r=d["lug_r"],
                              pip_rest=d["pip_rest"], rod_offset_c=4.3))
                    chk = check(pips=(-18.0, 0.0, 20.0, 45.0, 70.0, 92.0))
                    moving = [(k, val) for k, val in chk["least"] if not designed(k, val[0])]
                    best_by[le] = {**d, "least": moving[0]}
            res["searches"][tag] = {"combinations": n, "close": len(ok), "close_long_end_up": sum(1 for d in ok if d["long_end"] == "up"),
                                    "best": best_by}
            print(tag, n, len(ok), best_by, flush=True)
    finally:
        B.clear()
        B.update(base)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "20261119-v4b_flush.json").write_text(json.dumps(res, indent=1, default=float))
    return res



SOFT_GL = {"MUJOCO_GL": "egl", "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
           "EGL_PLATFORM": "surfaceless", "LP_NUM_THREADS": "4"}  # Mesa's llvmpipe: the GPU stays with the queues


def set_pose(model, data, fingers, s_, yaw, mcp, pip):
    """Joint positions of the closed drive at one pose (mm, deg) for each finger, then forward kinematics."""
    import mujoco
    th, qa, qb, _ok = pip_joint_angles([pip])[0]
    for f in fingers:
        for jn, v in (("stroke", s_ * 1e-3), ("yaw", math.radians(yaw)), ("mcp", math.radians(mcp)), ("pip_horn", th),
                      ("pin_a", qa), ("pin_b", qb), ("pip", math.radians(pip))):
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{jn}")]] = v
    mujoco.mj_forward(model, data)


def side_silhouettes(mod, yaw, mcp, pip):
    """x-z silhouettes (convex hulls of each box's corners) per body of a build module (this one or the acrylic one
    from git) at one pose, with the yaw servo's spline x."""
    from scipy.spatial import ConvexHull
    S, g = mod.pip_kin(np.array([pip]))
    P = mod.all_prims(g["L"])
    S["rod_prims"], S["knuckle_prims"] = P.pop("rod_prims"), P.pop("knuckle_prims")
    W = mod.bodies_at(P, yaw, mcp, pip, S, 0)
    cs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    out = {}
    for body, ps in W.items():
        polys = []
        for p in ps:
            pts = (p["c"] + (cs * p["h"]) @ p["R"].T)[:, [0, 2]]
            try:
                polys.append(pts[ConvexHull(pts).vertices])
            except Exception:  # noqa: BLE001, a box seen edge-on
                continue
        out[body] = polys
    return out


def cmd_render():
    """media/: stills of the index module at four poses (side and palmar views, carriage drawn translucent), the side
    view of the acrylic build against this one at MCP -15 and PIP -25 deg, and a film of the index and middle modules
    through their ranges (media/20261119-v4b_mixed_range.mp4)."""
    os.environ.update(SOFT_GL)
    import mujoco
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = {}
    # ---- side view against the acrylic build of 2026-11-19 (commit 823a33fbb)
    import importlib.util
    import tempfile
    src = subprocess.run(["git", "-C", str(ROOT), "show", "823a33fbb:scripts/cad/v4b_build.py"], capture_output=True, text=True, check=True).stdout
    src = src.replace("ROOT = Path(__file__).resolve().parents[2]", f"ROOT = Path({str(ROOT)!r})")
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "v4b_acrylic.py"
        fp.write_text(src)
        spec_ = importlib.util.spec_from_file_location("v4b_acrylic", fp)
        old = importlib.util.module_from_spec(spec_)
        sys.modules["v4b_acrylic"] = old
        spec_.loader.exec_module(old)
        sil_old = side_silhouettes(old, 0.0, -15.0, -25.0)
    sil_new = side_silhouettes(sys.modules[__name__], 0.0, -15.0, -25.0)
    col = {"mount": "#b8bcc4", "carriage": "#b8bcc4", "yoke": "#6f9fd8", "link": "#e0975a", "dip": "#8dbf72", "phorn": "#9b6fb0",
           "knuckle": "#d9bf3f", "rod": "#cc4c4c"}
    fig, axs = plt.subplots(1, 2, figsize=(13, 7.4), sharey=True)
    for ax, sil, title, xy, gz in ((axs[0], sil_old, "Acrylic build (2026-11-19): yaw servo's spline 37 mm behind G", 37.0, -7.0),
                                   (axs[1], sil_new, "This build: spline 17 mm behind G, proximal link 6 mm longer", B["yaw_spline_x"], B["Gz"])):
        for body in ("mount", "carriage", "yoke", "link", "dip", "phorn", "rod", "knuckle"):
            for poly in sil.get(body, []):
                ax.fill(poly[:, 0], poly[:, 1], color=col[body], alpha=0.55 if body in ("mount", "carriage") else 0.8, lw=0)
        ax.plot([0], [gz], "k+", ms=14, mew=2)
        ax.annotate("G", (0, gz), (-9, gz + 6), fontsize=14, family="serif")
        ax.axvline(xy, color="0.3", ls=":", lw=1)
        ax.set_title(title, fontsize=13, family="serif")
        ax.set_aspect("equal")
        ax.set_xlim(-45, 112)
        ax.set_ylim(-125, 70)
        ax.set_xlabel("x, dorsal (mm)", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("z (mm)", fontsize=13)
    from matplotlib.patches import Patch
    names = {"carriage": "carriage or mount", "yoke": "yoke", "link": "proximal link", "phorn": "crank H1", "knuckle": "knuckle",
             "rod": "rod R1", "dip": "DIP link and tip"}
    axs[1].legend(handles=[Patch(color=col[k], label=v) for k, v in names.items()], fontsize=12, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(MEDIA / "20261119-v4b_side_compare.png", dpi=100)
    plt.close(fig)
    out["side_compare"] = "media/20261119-v4b_side_compare.png"
    # ---- stills of the index module
    model = mujoco.MjModel.from_xml_path(str(MJCF_DIR / "v4b_mixed_index.xml"))
    data = mujoco.MjData(model)
    car_geoms = [i for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name == "index_carriage" and model.geom_group[i] == 1]
    for i in car_geoms:
        model.geom_rgba[i, 3] = 0.35
    base = np.array([P_INDEX[0], P_INDEX[1], 320.0]) * 1e-3
    poses = ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, -15.0, -25.0), (0.0, 30.0, 70.0, 80.0), (0.0, -60.0, 20.0, 40.0))
    vis = mujoco.MjvOption()
    vis.geomgroup[:] = [0, 1, 1, 0, 0, 0]  # without the floor, the palm and the thumb's gantry
    imgs = []
    with mujoco.Renderer(model, 720, 720) as r:
        for s_, yaw, mcp, pip in poses:
            set_pose(model, data, ("index",), s_, yaw, mcp, pip)
            row = []
            for az, el in ((90.0, -12.0), (20.0, -15.0)):
                cam = mujoco.MjvCamera()
                cam.lookat[:] = base + np.array([0.012, 0.0, -0.045])
                cam.distance, cam.azimuth, cam.elevation = 0.25, az, el
                r.update_scene(data, cam, scene_option=vis)
                row.append(r.render())
            imgs.append(row)
    fig, axes = plt.subplots(len(poses), 2, figsize=(10, 5 * len(poses)))
    for i, (s_, yaw, mcp, pip) in enumerate(poses):
        for j in range(2):
            axes[i, j].imshow(imgs[i][j])
            axes[i, j].set_axis_off()
            axes[i, j].set_title(f"yaw {yaw:g}, MCP {mcp:g}, PIP {pip:g} deg ({'side' if j == 0 else 'palmar side'})", fontsize=15, family="serif")
    fig.tight_layout()
    fig.savefig(MEDIA / "20261119-v4b_assembly.png", dpi=80)
    plt.close(fig)
    out["stills"] = "media/20261119-v4b_assembly.png"
    # ---- the yaw stage close up, from behind and below
    with mujoco.Renderer(model, 900, 700) as r:
        set_pose(model, data, ("index",), 0.0, 25.0, 30.0, 20.0)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = base + np.array([0.032, 0.0, -0.010])
        cam.distance, cam.azimuth, cam.elevation = 0.14, 230.0, -32.0
        r.update_scene(data, cam, scene_option=vis)
        plt.imsave(MEDIA / "20261119-v4b_yaw_stage.png", r.render())
    out["yaw_stage"] = "media/20261119-v4b_yaw_stage.png"
    # ---- film: index and middle through stroke, yaw, MCP and PIP
    import imageio.v2 as imageio
    model = mujoco.MjModel.from_xml_path(str(MJCF_DIR / "v4b_mixed_pair.xml"))
    data = mujoco.MjData(model)
    keys = [(0, 0, 0, 0), (30, 0, 0, 0), (-30, 0, 0, 0), (0, 0, 0, 0), (0, 60, 0, 0), (0, -60, 0, 0), (0, 0, 0, 0), (0, 0, 90, 0),
            (0, 0, -15, 0), (0, 0, 0, 0), (0, 0, 0, 92), (0, 0, 0, -25), (0, 0, 0, 0), (15, -20, 45, 50), (-10, 25, 30, 70), (0, 0, 0, 0)]
    fps, seg = 30, 0.9
    frames = []
    pair_base = np.array([56.18, 0.0, 320.0]) * 1e-3
    with mujoco.Renderer(model, 960, 720) as r:
        for a, b in zip(keys[:-1], keys[1:]):
            for t in np.linspace(0, 1, int(seg * fps), endpoint=False):
                w = 0.5 - 0.5 * math.cos(math.pi * t)
                q = [ai + w * (bi - ai) for ai, bi in zip(a, b)]
                set_pose(model, data, ("index", "middle"), *q)
                cam = mujoco.MjvCamera()
                cam.lookat[:] = pair_base + np.array([0.0, 0.0, -0.055])
                cam.distance, cam.azimuth, cam.elevation = 0.36, 18.0, -12.0
                r.update_scene(data, cam, scene_option=vis)
                frames.append(r.render())
    film = MEDIA / "20261119-v4b_mixed_range.mp4"
    imageio.mimwrite(film, frames, fps=fps, quality=7, macro_block_size=8)
    imageio.imwrite(film.with_suffix(".jpg"), frames[int(13.4 * seg * fps)], quality=85)  # poster: the first combined pose
    out["film"] = str(film.relative_to(OUT))
    out["film_frames"] = len(frames)
    return out



def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        fine = "--fine" in sys.argv
        res = check(3.0, tuple(np.arange(-85.0, 85.1, 15.0)) + (85.0,), tuple(np.arange(-25.0, 92.1, 6.0)) + (92.0,)) if fine else check()
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / f"20261119-v4b_check{'_fine' if fine else ''}.json").write_text(json.dumps(
            {"script": "scripts/cad/v4b_build.py", "command": "check" + (" --fine" if fine else ""), "params": B,
             "module_half_width_mm": res["module_half_width_mm"], "pip_closes": res["pip_closes"], "pip_rod_mm": res["pip_rod_mm"],
             "least": [[k, v] for k, v in res["least"][:60]]}, indent=1, default=float))
        print("half width", res["module_half_width_mm"], "pip closes", res["pip_closes"], "rod", res["pip_rod_mm"])
        for k, v in res["least"][:30]:
            print(f"{v[0]:7.2f}  {k:62s} yaw {v[1]:5.0f} mcp {v[2]:5.0f} pip {v[3]:5.0f}")
    elif cmd == "stroke":
        res = stroke_check()
        curl = curl_limit()
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "20261119-v4b_stroke.json").write_text(json.dumps(
            {"script": "scripts/cad/v4b_build.py", "command": "stroke", "stroke_mm": STROKE, "gantry": GANTRY,
             "curl_reaches_gantry": curl, "least": [[k, v] for k, v in res[:40]]}, indent=1, default=float))
        print("curl reaching the gantry (MCP + PIP, MCP, PIP, yaw, s):", curl)
        for k, v in res[:24]:
            print(f"{v[0]:7.2f}  {k:55s} s {v[1]:7.2f} yaw {v[2]:5.0f} mcp {v[3]:5.0f} pip {v[4]:5.0f}")
    elif cmd == "dxf":
        print(json.dumps(cmd_dxf(), indent=1))
    elif cmd == "bom":
        print(cmd_bom())
    elif cmd == "keepout":
        print(json.dumps(cmd_keepout()))
    elif cmd == "pip":
        print(json.dumps(cmd_pip(), indent=1))
    elif cmd == "loads":
        print(json.dumps(cmd_loads(), indent=1))
    elif cmd == "printed":
        print(json.dumps(cmd_printed(), indent=1, default=float))
    elif cmd == "mjcf":
        print(json.dumps(cmd_mjcf(), indent=1))
    elif cmd == "flush":
        cmd_flush()
    elif cmd == "render":
        print(json.dumps(cmd_render(), indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
