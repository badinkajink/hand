#!/usr/bin/env python3
"""v4b built from parts (2026-11-19 night): cad_v4's serial gimbal MCP with STS3032s and the single-rod PIP drive of
scripts/cad/v4c_hardware.py, as 1/8-inch laser-cut plates, two printed parts and stock hardware.

The yaw servo sits behind the gimbal on the yaw axis, shaft toward the palm, its aluminium disc horn bolted to the
yoke's back plate. The MCP servo stands on the yoke with its shaft on the MCP axis; its disc horn is bolted to the MCP
link's horn plate, and the link's idler plate turns in a ball bearing on a shoulder screw held by the yoke, coaxial with
the shaft. Below the MCP servo a cross plate joins the two top plates to the narrow lower plates, which hold the PIP
servo flat by its tabs. The PIP servo's horn drives the DIP link through a knuckle (pin a along the horn axis, pin b
square to it) and a rod with a spherical plain bearing on a bolt along the PIP axis.

Every laser-cut part is one closed outline with holes, placed in its body's frame (mm; the finger's mount frame: x
dorsal, y lateral, z up; origin at v3's yaw axis on the mount). The clearance check cuts each outline into 1 mm strips
(boxes); the DXF is written from the same outlines.

check   clearances between the parts over the yaw, MCP and PIP ranges, against the x-stage beam (underside 21.9 mm
        above v3's yaw axis) and the module's half-width at the carriage (index and middle mounts come within 56.8 mm)
dxf     the plates laid out on a sheet (cut/20261119-v4b_acrylic_3mm.dxf, R12: CUT layer polylines and circles,
        ENGRAVE layer labels) and a preview
all     both
Outputs go to docs/experiments/20261119-v4b_build/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

from sr2hand.geometry.rotations import rot_x, rot_y  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/20261119-v4b_build"
DATA, MEDIA, CUT = OUT / "data", OUT / "media", OUT / "cut"
PARAMS = ROOT / "assets/cad_v4/concepts/20261118-layout_params.json"
PJ = json.loads(PARAMS.read_text())
S32 = PJ["servos"]["sts3032"]
T = 3.175  # 1/8-inch sheet
X, Y, Z = np.eye(3)
CHORD = 0.3  # mm, arc discretisation

# ------------------------------------------------------------------------------------------------- parameters
B = {
    "Gz": -7.0,             # gimbal centre below v3's yaw axis (mount frame z)
    "proximal": 60.0,       # G to the PIP axis
    "beam_z": 21.9, "beam_y": (-12.5, 11.3), "half_width": 28.4,
    "yaw_spline_x": 37.0,   # yaw servo's spline top on the yaw axis, shaft toward -x, long end up
    "mcp_spline_y": -18.0,  # MCP servo's spline top on the MCP axis, shaft toward -y, long end up
    "idler_gap": 2.5,       # MCP servo's base to the link's idler plate, which carries its bearing's flange on this side
    "l2_y3_gap": 0.925,     # link idler plate to the printed yoke idler plate Y3, which holds the idler's dowel
    "cross_r": 14.5,        # the cross plate's top face below G (it sweeps an annulus about G as the MCP turns)
    "cut_clear": 0.1,       # servo case cutouts: clearance per side
    "pip": {"spline_x": 13.0, "spline_z": -28.0, "spacer": 5.0, "r": 12.0, "lug_r": 9.0, "lug_y": 0.0, "pip_rest": 0.0,
            "knuckle_e": 5.0, "rod_offset_c": 5.5, "horn_limit_deg": 80.0},
    "disc": {"d": 18.0, "flange": 2.0, "hub_d": 7.0, "hub_h": 2.5, "bolt_r": 6.0},
    "mf63": {"D": 6.0, "B": 2.5, "flange_d": 7.2, "flange_t": 0.6},
    "sph": {"D": 12.0, "W": 4.8, "H": 3.6},
    "shoulder": {"d": 3.0, "head_d": 5.0, "head_h": 1.3, "thread": 4.0},  # McMaster 90323A: low head, precision, M2 x 4
    "nut": {"corners": 4.62, "t": 1.6},  # M2 hex nut: inside a bearing's 6 mm bore it bears on the inner ring alone
    "boss": {"d": 4.0, "h": 0.5},  # printed boss that bears on a bearing's inner ring and keeps the part off the flange
    "dip_cheek_y": (3.7, 6.1),  # printed DIP link's cheeks (|y|): 0.3 from the PIP bearings' flanges on the plates' inner faces
    "lower_y": 7.0,         # the lower plates' inner faces (|y|)
    "tip_box": {"x": (-11.0, 9.0), "y": 10.5, "z": (-44.0, -20.0)},  # cassette and cap below the PIP axis (PIP 0)
}
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
    """A laser-cut part: outline (polygon in (u, v)), round holes (u, v, d) and polygon holes, placed in its body's
    frame. One face lies on o + u U + v V; the plate extends T along +N = U x V, or along -N with flip=True."""

    def __init__(self, name, body, outline, o, U, V, holes=(), polys=(), qty=1, flip=False, note="", printed=False):
        self.name, self.body, self.qty, self.note, self.printed = name, body, qty, note, printed
        self.outline = outline if signed_area(outline) > 0 else outline[::-1]
        self.holes = [tuple(float(x) for x in h) for h in holes]
        self.polys = [np.asarray(p, float) for p in polys]
        self.o, self.U, self.V = np.asarray(o, float), np.asarray(U, float), np.asarray(V, float)
        self.N = np.cross(self.U, self.V)
        self.w0 = -T if flip else 0.0

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
                c = self.o + (a + b) / 2 * self.U + v * self.V + (self.w0 + T / 2) * self.N
                out.append({"name": self.name, "c": c, "R": R, "h": np.array([(b - a) / 2, dv / 2, T / 2]), "r": 0.0})
        return out


# ------------------------------------------------------------------------------------------------- solids
def box_prim(name, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return {"name": name, "c": (lo + hi) / 2, "R": np.eye(3), "h": (hi - lo) / 2, "r": 0.0}


def frame_prim(name, origin, axes, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return {"name": name, "c": origin + axes @ ((lo + hi) / 2), "R": axes, "h": (hi - lo) / 2, "r": 0.0}


def cap_prim(name, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    z = d / L
    x = np.cross(z, X if abs(z @ X) < 0.9 else Y)
    x /= np.linalg.norm(x)
    return {"name": name, "c": (a + b) / 2, "R": np.stack([x, np.cross(z, x), z], 1), "h": np.array([0.0, 0.0, L / 2]), "r": r}


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


# ------------------------------------------------------------------------------------------------- the parts
def geometry():
    """Derived positions (mm, mount frame) that several parts share."""
    Gz, xY, yM = B["Gz"], B["yaw_spline_x"], B["mcp_spline_y"]
    g = {"G": np.array([0.0, 0.0, Gz]), "Gz": Gz, "xY": xY, "yM": yM}
    g["yaw_tab_x"] = (xY + S32["h_spline"] - S32["h_tab"][1], xY + S32["h_spline"] - S32["h_tab"][0])  # spline side, base side
    g["mcp_tab_y"] = (yM + S32["h_spline"] - S32["h_tab"][1], yM + S32["h_spline"] - S32["h_tab"][0])
    g["mcp_base_y"] = yM + S32["h_spline"]
    g["L2_y"] = g["mcp_base_y"] + B["idler_gap"]  # link idler plate, first face
    g["Y3_y"] = g["L2_y"] + T + B["l2_y3_gap"]  # yoke idler plate (printed), first face
    g["L1_y"] = yM - B["disc"]["flange"]  # link horn plate's face on the disc (the plate lies toward -y)
    g["Y1_x"] = xY - B["disc"]["flange"]  # yoke back plate's face on the disc (the plate lies toward -x)
    g["cz1"] = Gz - B["cross_r"]
    g["cz0"] = g["cz1"] - T
    g["pz"] = Gz - B["proximal"]
    g["zs"] = Gz + B["pip"]["spline_z"]
    g["xs"] = B["pip"]["spline_x"]
    g["pip_tab_x"] = (g["xs"] - S32["h_spline"] + S32["h_tab"][0], g["xs"] - S32["h_spline"] + S32["h_tab"][1])
    # the PIP servo's tab strips (long end down): L6a from the case's short end up to the cross plate, glued to its
    # underside; L6b from 1.2 mm below the long end's tabs up to the case's end
    g["L6a_z"] = (g["zs"] - S32["len"][0], g["cz0"])
    g["L6b_z"] = (g["zs"] - S32["tab_len"][1] - 1.2, g["zs"] - S32["len"][1])
    # tabs into the lower plates: L6a's at its lower end, 1.9 mm of plate between the slot and the plates' top edge
    g["L6_tab_z"] = {"L6a_z": (g["L6a_z"][0] + 0.5, g["L6a_z"][0] + 2.5),
                     "L6b_z": (sum(g["L6b_z"]) / 2 - 1.5, sum(g["L6b_z"]) / 2 + 1.5)}
    return g


def flats():
    """Every laser-cut part (Flat), in its body's frame."""
    g = geometry()
    Gz, xY, yM = g["Gz"], g["xY"], g["yM"]
    c = B["cut_clear"]
    br = B["disc"]["bolt_r"] / math.sqrt(2)
    out = []
    # ---- mount: yaw servo plate (normal +x, behind the yaw servo's tabs) and the top plate
    x0 = g["yaw_tab_x"][1]
    z_top = B["beam_z"] - 1.0 - T  # the top plate's lower face
    outline = poly_round([(-15, Gz - 13.0), (15, Gz - 13.0), (15, z_top), (12, z_top), (12, z_top + T), (6, z_top + T),
                          (6, z_top), (-6, z_top), (-6, z_top + T), (-12, z_top + T), (-12, z_top), (-15, z_top)], 1.0)
    case = rrect(-S32["half_t"] - c, Gz - 5.9 - c, S32["half_t"] + c, Gz + 17.3 + c, 0.5)
    holes = [(sy * 3.5, Gz + zz, 2.2) for sy in (-1, 1) for zz in (19.79, -8.21)]
    out.append(Flat("M1 yaw servo plate", "mount", outline, (x0, 0, 0), Y, Z, holes=holes, polys=[case],
                    note="behind the yaw servo's tabs; tabs into M2"))
    outline = rrect(xY + 2.0, -15.0, xY + 70.0, 15.0, 3.0)
    slots = [rrect(x0 - 0.05, sy * a - 0.05 * sy, x0 + T + 0.05, sy * b + 0.05 * sy, 0.1) if sy > 0 else
             rrect(x0 - 0.05, -b - 0.05, x0 + T + 0.05, -a + 0.05, 0.1) for sy in (1, -1) for a, b in ((6, 12),)]
    holes = [(xY + 60.0, sy * 10.0, 3.4) for sy in (-1, 1)] + [(xY + 44.0, sy * 10.0, 3.4) for sy in (-1, 1)]
    out.append(Flat("M2 top plate", "mount", outline, (0, 0, z_top), X, Y, holes=holes, polys=slots,
                    note="mounting face (test stand, later the carriage)"))
    for sy in (1, -1):
        tri = poly_round([(x0 + T, z_top), (x0 + T + 16.0, z_top), (x0 + T, z_top - 16.0)], 1.0)
        out.append(Flat(f"M3 gusset ({'+' if sy > 0 else '-'}y)", "mount", tri, (0, sy * 12.0, 0), X, Z,
                        flip=(sy > 0), note="glued to M1 and M2"))
    # ---- yoke: back plate (normal -x, on the yaw disc), MCP servo plate, idler plate
    y2a, y2b = g["mcp_tab_y"][1], g["mcp_tab_y"][1] + T
    y3a, y3b = g["Y3_y"], g["Y3_y"] + T
    outline = hull_circles([(0.0, Gz, 10.5), (y3a + 1.6, Gz, 5.0), (y2a, Gz, 6.0), (y2a + T / 2, Gz + 7.0, 5.5)])
    # four of the disc's eight holes, clear of Y2's slot: at 45, 90, 270 and 315 deg from +y
    holes = [(br, Gz + sz * br, 2.2) for sz in (-1, 1)] + [(0.0, Gz + sz * B["disc"]["bolt_r"], 2.2) for sz in (-1, 1)] + [(0.0, Gz, 5.0)]
    slots = [rrect(y2a - 0.05, Gz + 3.95, y2b + 0.05, Gz + 10.05, 0.1), rrect(y3a - 0.05, Gz - 3.05, y3b + 0.05, Gz + 3.05, 0.1)]
    out.append(Flat("Y1 yoke back plate", "yoke", outline, (g["Y1_x"], 0, 0), Y, Z, holes=holes, polys=slots, flip=True,
                    note="4 x M2 into the yaw servo's disc horn"))
    xb = g["Y1_x"]  # the arms' tabs reach through Y1
    # the tab into Y1 runs 4 to 10 mm above G, clear of Y1's centre hole for the horn screw
    outline = poly_round([(-6.6, Gz - 12.3), (6.6, Gz - 12.3), (6.6, Gz - 9.0), (8.5, Gz - 7.5), (8.5, Gz + 4.0), (xb, Gz + 4.0),
                          (xb, Gz + 10.0), (8.5, Gz + 10.0), (8.5, Gz + 22.5), (-8.5, Gz + 22.5), (-8.5, Gz - 7.5), (-6.6, Gz - 9.0)], 0.8)
    case = rrect(-S32["half_t"] - c, Gz - 5.9 - c, S32["half_t"] + c, Gz + 17.3 + c, 0.5)
    holes = [(sx * 3.5, Gz + zz, 2.2) for sx in (-1, 1) for zz in (19.79, -8.21)]
    out.append(Flat("Y2 MCP servo plate", "yoke", outline, (0, y2a, 0), X, Z, holes=holes, polys=[case], flip=True,
                    note="behind the MCP servo's tabs; tab into Y1"))
    outline = poly_round([(-6.0, Gz - 6.0), (0.0, Gz - 6.0), (8.0, Gz - 3.0), (xb, Gz - 3.0), (xb, Gz + 3.0), (8.0, Gz + 3.0),
                          (0.0, Gz + 6.0), (-6.0, Gz + 6.0)], 2.0)
    out.append(Flat("Y3 yoke idler plate", "yoke", outline, (0, y3a, 0), X, Z, holes=[(0.0, Gz, 2.95)], flip=True, printed=True,
                    note="printed: holds the idler's 3 x 8 dowel (press fit); tab glued into Y1"))
    # ---- link: horn plate, idler plate, cross plate, lower plates, PIP servo tab strips
    cz0, cz1, pz = g["cz0"], g["cz1"], g["pz"]
    outline = hull_circles([(0.0, Gz, 10.0), (-8.0, cz1 - 1.0, 1.0), (8.0, cz1 - 1.0, 1.0)])
    outline = np.concatenate([outline[outline[:, 1] >= cz1 + 0.5], np.array([[8.0, cz1], [3.0, cz1], [3.0, cz0], [-3.0, cz0], [-3.0, cz1], [-8.0, cz1]])])
    outline = poly_round(outline[np.argsort(np.arctan2(outline[:, 1] - Gz, outline[:, 0]))], 0.0)
    holes = [(sx * br, Gz + sz * br, 2.2) for sx in (-1, 1) for sz in (-1, 1)] + [(0.0, Gz, 5.0)]
    out.append(Flat("L1 link horn plate", "link", outline, (0, g["L1_y"], 0), X, Z, holes=holes,
                    note="4 x M2 into the MCP servo's disc horn; tab into L3"))
    outline = hull_circles([(0.0, Gz, 6.5), (-8.0, cz1 - 1.0, 1.0), (8.0, cz1 - 1.0, 1.0)])
    outline = np.concatenate([outline[outline[:, 1] >= cz1 + 0.5], np.array([[8.0, cz1], [3.0, cz1], [3.0, cz0], [-3.0, cz0], [-3.0, cz1], [-8.0, cz1]])])
    outline = poly_round(outline[np.argsort(np.arctan2(outline[:, 1] - Gz, outline[:, 0]))], 0.0)
    out.append(Flat("L2 link idler plate", "link", outline, (0, g["L2_y"], 0), X, Z, holes=[(0.0, Gz, B["mf63"]["D"])], flip=True,
                    note="MF63ZZ bearing on the idler shoulder screw; tab into L3"))
    y_lo, y_hi = g["L1_y"] - T, g["L2_y"] + T
    outline = rrect(-8.0, y_lo - 2.5, 8.0, y_hi + 2.5, 1.0)  # 2.5 mm of plate beyond the L1 and L2 slots
    slots = [rrect(-3.05, g["L1_y"] - T - 0.05, 3.05, g["L1_y"] + 0.05, 0.1), rrect(-3.05, g["L2_y"] - 0.05, 3.05, g["L2_y"] + T + 0.05, 0.1)]
    ly = B["lower_y"]
    slots += [rrect(-3.05, sy * ly - (T if sy < 0 else 0) - 0.05, 3.05, sy * ly + (T if sy > 0 else 0) + 0.05, 0.1) for sy in (1, -1)]
    out.append(Flat("L3 link cross plate", "link", outline, (0, 0, cz0), X, Y, polys=slots, note="joins L1, L2 above and L4, L5 below"))
    Re = 8.0  # the lower plates: 16 mm wide, a round end of radius Re about the PIP pin, a tab up into L3
    for sy in (1, -1):
        outline = np.concatenate([arc((0.0, pz), Re, math.pi, 2 * math.pi),
                                  np.array([[Re, cz0], [3.0, cz0], [3.0, cz0 + T], [-3.0, cz0 + T], [-3.0, cz0], [-Re, cz0]])])
        sl = [rrect(g["pip_tab_x"][1] - 0.05, zz0 - 0.05, g["pip_tab_x"][1] + T + 0.05, zz1 + 0.05, 0.1)
              for zz0, zz1 in g["L6_tab_z"].values()]
        out.append(Flat(f"L{4 if sy > 0 else 5} lower plate ({'+' if sy > 0 else '-'}y)", "link", outline, (0, sy * B["lower_y"], 0), X, Z,
                        holes=[(0.0, pz, B["mf63"]["D"])], polys=sl, flip=(sy > 0),
                        note="MF63ZZ bearing for the PIP pin; slots for L6a, L6b"))
    xt = g["pip_tab_x"][1]
    for nm, key, hz in (("L6a PIP servo strip, short end", "L6a_z", g["zs"] + 8.21),
                        ("L6b PIP servo strip, long end", "L6b_z", g["zs"] - 19.79)):
        ly = B["lower_y"]
        (zz0, zz1), (t0, t1) = g[key], g["L6_tab_z"][key]
        tabs = [(-ly - T, t0, -ly, t1), (ly, t0, ly + T, t1)]
        outline = poly_round([(-ly, zz0), (ly, zz0), (ly, tabs[1][1]), (ly + T, tabs[1][1]), (ly + T, tabs[1][3]), (ly, tabs[1][3]),
                              (ly, zz1), (-ly, zz1), (-ly, tabs[0][3]), (-ly - T, tabs[0][3]), (-ly - T, tabs[0][1]), (-ly, tabs[0][1])], 0.3)
        out.append(Flat(nm, "link", outline, (xt, 0, 0), Y, Z, holes=[(sy * 3.5, hz, 2.2) for sy in (-1, 1)],
                        note="behind the PIP servo's tabs; tabs into L4, L5"))
    # ---- PIP horn arm (phorn frame: the servo axis along x through (0, 0, zs)) and the rod (drawn in its own frame)
    zs, xs = g["zs"], g["xs"]
    r = B["pip"]["r"]
    bt = B["disc"]["bolt_r"]  # bolted through the disc's holes at 180 and 135 deg, on the side away from the rod
    outline = hull_circles([(-bt, zs, 3.2), (-br, zs + br, 3.2), (0.0, zs, 4.5), (r, zs, 4.75)])
    holes = [(-bt, zs, 2.2), (-br, zs + br, 2.2), (0.0, zs, 5.0), (r, zs, B["mf63"]["D"])]
    out.append(Flat("H1 PIP horn arm", "phorn", outline, (xs + B["disc"]["flange"], 0, 0), Y, Z, holes=holes,
                    note="2 x M2 into the PIP servo's disc horn (holes at 180 and 135 deg); MF63ZZ for pin a"))
    return out


def dogbone(L, r0, r1, w):
    """Two round ends (radii r0 at u = 0 and r1 at u = L) joined by a bar of half-width w (w < r0, r1), as a dense ccw
    polygon: the lug end's far side, the bar's upper edge, the top end's far side; the lower edge closes it."""
    a0, a1 = math.asin(w / r0), math.asin(w / r1)
    return np.concatenate([arc((L, 0.0), r1, a1 - math.pi, math.pi - a1), arc((0.0, 0.0), r0, a0, 2 * math.pi - a0)])


def rod_flat(L_rod):
    """The PIP rod in its own frame: u from the top bearing (pin b, an MF63ZZ) toward the lug's spherical bearing; a
    3.5 mm half-width bar between the two bosses."""
    outline = dogbone(L_rod, 5.0, 8.0, 3.5)
    return Flat("R1 PIP rod", "rod", outline, (0, 0, -T / 2), X, Y, holes=[(0.0, 0.0, B["mf63"]["D"]), (L_rod, 0.0, B["sph"]["D"])],
                note="MF63ZZ at pin b; spherical plain bearing (d 3, D 12) pressed in at the lug")


def solids():
    """Non-flat parts as boxes per body: servos, disc horns, the knuckle, the printed DIP link and the fingertip."""
    g = geometry()
    Gz, xY, yM, dk = g["Gz"], g["xY"], g["yM"], B["disc"]
    S = {k: [] for k in ("mount", "yoke", "link", "dip", "phorn")}
    S["mount"] += servo_boxes("yawservo", (xY, 0, Gz), (-1, 0, 0), (0, 0, 1))
    S["yoke"] += disc_prims("yawdisc", (xY - dk["flange"] / 2, 0, Gz), X, dk["d"], dk["flange"])
    S["yoke"] += servo_boxes("mcpservo", (0, yM, Gz), (0, -1, 0), (0, 0, 1))
    # MCP idler: a 3 x 8 dowel pressed into the printed Y3, flush with its outer face, slides in L2's bearing (flange
    # toward the servo); the MCP servo's horn locates the link along the axis
    mf = B["mf63"]
    y_out = g["Y3_y"] + T
    S["yoke"] += cyl_prims("idler_dowel", (0, y_out - 8.0, Gz), (0, y_out, Gz), 3.0)
    S["link"] += disc_prims("mcpdisc", (0, yM - dk["flange"] / 2, Gz), Y, dk["d"], dk["flange"])
    S["link"] += disc_prims("flange_L2", (0, g["L2_y"] - mf["flange_t"] / 2, Gz), Y, mf["flange_d"], mf["flange_t"])
    S["link"] += servo_boxes("pipservo", (g["xs"], 0, g["zs"]), (1, 0, 0), (0, 0, -1))
    S["phorn"] += disc_prims("pipdisc", (g["xs"] + dk["flange"] / 2, 0, g["zs"]), X, dk["d"], dk["flange"])
    # pin a's bearing sits in H1 with its flange toward the knuckle
    S["phorn"] += disc_prims("flange_H1", (g["xs"] + dk["flange"] + T + mf["flange_t"] / 2, B["pip"]["r"], g["zs"]), X,
                             mf["flange_d"], mf["flange_t"])
    pz = g["pz"]
    for sy in (1, -1):  # the PIP bearings' flanges on the lower plates' inner faces
        S["link"] += disc_prims("flange_L45", (0.0, sy * (B["lower_y"] - mf["flange_t"] / 2), pz), Y, mf["flange_d"], mf["flange_t"])
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
    """A shoulder screw along a coordinate that falls from its head: the head bears on `floor` (a counterbore's floor),
    the bearing's inner ring spans ring_boss (its face on the boss) to ring_nut (its face on the nut). The longest
    catalogue shoulder that keeps at least 1.5 mm of its length in the ring and ends 0.3 mm short of the nut's face, so
    the nut clamps the ring against the boss; the nut and the thread's end."""
    ok = [L for L in SHOULDER_LENGTHS if ring_nut + 0.3 < floor - L < ring_boss - 1.5]
    L = max(ok)
    return {"shoulder": L, "shoulder_end": floor - L, "nut": (ring_nut, ring_nut - B["nut"]["t"]),
            "thread_end": floor - L - B["shoulder"]["thread"]}


def knuckle_spec():
    """The printed knuckle in its own frame: x along pin a (from the horn arm H1 toward the far face), y along k (from
    pin a toward pin b), z along pin b; origin at pin a's foot Pa. Each pin is a shoulder screw whose head sits in a
    counterbore on the knuckle's far face and whose M2 nut sits in the bearing's bore recess (in H1 or R1), where it
    bears on the inner ring alone; a 4 mm boss on the knuckle bears on the ring's other face. Pin b's numbers are along
    t = -s z, which falls from its head; s is the rod's side along pin b."""
    _S, g = pip_kin(np.array([B["pip"]["pip_rest"]]))
    pp, mf, sh, bo = B["pip"], B["mf63"], B["shoulder"], B["boss"]
    e, c, xa = pp["knuckle_e"], g["c"], g["x_a"]
    x_H1 = pp["spline_x"] + B["disc"]["flange"] + T  # H1's face toward the knuckle (mount frame x); its bearing's flange on it
    a_boss, a_nut = x_H1 + mf["flange_t"] - xa, x_H1 + mf["flange_t"] - mf["B"] - xa
    s = math.copysign(1.0, c)
    b_boss = -(abs(c) - T / 2 - mf["flange_t"])  # R1's bearing: flange toward the knuckle
    b_nut = b_boss - mf["B"]
    hz = -b_boss - bo["h"]
    box = {"x": (a_boss + bo["h"], 4.5), "y": (-3.5, e + 4.0), "z": (-hz, hz)}
    # pin a's head sits on the far x face (a counterbore would leave no wall: the knuckle is 2 hz thick along pin b);
    # pin b's head sits in a counterbore on the far face along pin b
    return {"e": e, "c": c, "x_a": xa, "s": s, "box": box,
            "a": {"ring": (a_boss, a_nut), **pin_stack(box["x"][1], a_boss, a_nut)},
            "b": {"ring": (b_boss, b_nut), **pin_stack(hz - sh["head_h"], b_boss, b_nut)}}


def knuckle_prims(spec):
    """The knuckle, its bosses and the two pins' screws and nuts as primitives in the knuckle frame (see knuckle_spec)."""
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
    """The rod's bearings beyond its plate, in the rod frame: pin b's flange toward the knuckle (which lies at
    z = -c) and the spherical bearing's outer ring (3.6 mm wide in the 3.175 mm plate) at the lug."""
    _S, g = pip_kin(np.array([B["pip"]["pip_rest"]]))
    s, mf = -math.copysign(1.0, g["c"]), B["mf63"]
    return (disc_prims("flange_R1", (0.0, 0.0, s * (T / 2 + mf["flange_t"] / 2)), Z, mf["flange_d"], mf["flange_t"])
            + disc_prims("sph_ring", (L_rod, 0.0, 0.0), Z, B["sph"]["D"], B["sph"]["H"]))


def pip_kin(pip_deg):
    """The PIP drive of scripts/cad/v4c_hardware.py with this layout's numbers, in this script's link frame (G at
    (0, 0, Gz)): lug A, rod top Q, pin a's foot Pa, knuckle direction k, pin b, horn angle th (rad), closure ok."""
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
def bodies_at(prims, yaw, mcp, pip, S, i):
    """World primitives per body at one pose; S is the PIP solve and i its index."""
    g = geometry()
    G = g["G"]
    Rx = rot_x(math.radians(yaw))
    Rl = Rx @ rot_y(math.radians(mcp))
    tl = G - Rl @ G
    tyk = G - Rx @ G
    Op = np.array([0.0, 0.0, g["pz"]])
    Rp = rot_y(math.radians(pip))
    Rd, td = Rl @ Rp, Rl @ (Op - Rp @ Op) + tl
    Cs = np.array([0.0, 0.0, g["zs"]])
    Rh = rot_x(float(S["th"][i]))
    Rph, tph = Rl @ Rh, Rl @ (Cs - Rh @ Cs) + tl

    def tf(ps, R_, t_):
        return [{"name": p["name"], "c": R_ @ p["c"] + t_, "R": R_ @ p["R"], "h": p["h"], "r": p["r"]} for p in ps]
    out = {"mount": prims["mount"], "yoke": tf(prims["yoke"], Rx, tyk), "link": tf(prims["link"], Rl, tl),
           "dip": tf(prims["dip"], Rd, td), "phorn": tf(prims["phorn"], Rph, tph)}
    A, Qp, Pa, k, pb = S["A"][i], S["Q"][i], S["Pa"][i], S["k"][i], S["pb"][i]
    Rk = np.stack([X, k, pb], 1)
    kn = [{**p, "c": Pa + Rk @ p["c"], "R": Rk @ p["R"]} for p in S["knuckle_prims"]]
    d = (A - Qp) / np.linalg.norm(A - Qp)
    w = np.cross(pb, d)
    rod = [{**p, "c": Qp + np.stack([d, w, pb], 1) @ p["c"], "R": np.stack([d, w, pb], 1) @ p["R"]}
           for p in S["rod_prims"]]
    out["knuckle"] = tf(kn, Rl, tl)
    out["rod"] = tf(rod, Rl, tl)
    return out


EXCLUDE = {("mount", "yoke"): [("yawservo_boss", "yawdisc"), ("yawservo_case", "Y1"), ("yawservo_boss", "Y1")],
           ("yoke", "link"): [("mcpservo_boss", "mcpdisc"), ("idler_dowel", "L2"), ("idler_dowel", "flange_L2")],
           ("link", "phorn"): [("pipservo_boss", "pipdisc"), ("pipservo_case", "pipdisc")],
           }
ATTACHED = {("phorn", "knuckle"), ("knuckle", "rod")}


def as_set(ps):
    return (np.array([p["c"] for p in ps]), np.array([p["R"] for p in ps]), np.array([p["h"] for p in ps]),
            np.array([p["r"] for p in ps]))


def all_prims(S_rest_L):
    """Primitives per body in their own frames: flats (strips) plus solids; the rod's flat in its own frame."""
    P = solids()
    for f in flats():
        P[f.body] += f.strips()
    P["rod_prims"] = rod_flat(S_rest_L).strips() + rod_solids(S_rest_L)
    P["knuckle_prims"] = knuckle_prims(knuckle_spec())
    return P


def check(step_mcp=7.0, yaws=YAWS, pips=PIPS):
    import cad_v4_layouts as L
    S, g = pip_kin(np.array(pips))
    P = all_prims(g["L"])
    S["rod_prims"], S["knuckle_prims"] = P.pop("rod_prims"), P.pop("knuckle_prims")
    gg = geometry()
    Op = np.array([0.0, 0.0, gg["pz"]])
    A0_dip = Op + rot_y(-math.radians(B["pip"]["pip_rest"])) @ (g["A0"] + np.array([0.0, 0.0, B["Gz"]]) - Op)
    P["dip"] += dip_lug_prims(A0_dip)
    beam = [box_prim("beam", (-60.0, B["beam_y"][0], B["beam_z"]), (110.0, B["beam_y"][1], B["beam_z"] + 50.0))]
    mcps = np.arange(MCP_R[0], MCP_R[1] + 0.01, step_mcp)
    least, width = {}, (0.0, None)
    for yaw, mcp in itertools.product(yaws, mcps):
        for i, pip in enumerate(pips):
            W = bodies_at(P, yaw, mcp, pip, S, i)
            near = W["mount"] + W["yoke"]
            for p in near:
                cs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * p["h"]
                yy = float(np.abs((p["c"] + cs @ p["R"].T)[:, 1]).max()) + p["r"]
                if yy > width[0]:
                    width = (round(yy, 2), (p["name"], yaw, float(mcp), pip))
            names = list(W)
            pairs = [(a, b) for a, b in itertools.combinations(names, 2) if (a, b) not in ATTACHED and (b, a) not in ATTACHED]
            for a, b in pairs + [("beam", n) for n in names if n not in ("mount", "dip")]:
                A_ = beam if a == "beam" else W[a]
                B_ = W[b]
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


# ------------------------------------------------------------------------------------------------- DXF and BOM
def fit_coupon():
    """A test strip for the press fits and the tab slots: bearing holes and slot widths around the nominal sizes."""
    outline = rrect(0.0, 0.0, 92.0, 22.0, 2.0)
    holes, slots, labels = [], [], []
    for k, d in enumerate((5.8, 5.9, 6.0, 6.1)):
        holes.append((6.0 + 9.0 * k, 6.0, d))
        labels.append((3.0 + 9.0 * k, 11.5, f"{d:.1f}"))
    for k, d in enumerate((11.8, 11.9, 12.0)):
        holes.append((48.0 + 14.0 * k, 9.0, d))
        labels.append((44.5 + 14.0 * k, 16.5, f"{d:.1f}"))
    for k, w in enumerate((3.0, 3.1, 3.2, 3.3)):
        slots.append(rrect(4.0 + 9.0 * k - w / 2, 14.0, 4.0 + 9.0 * k + w / 2, 20.0, 0.1))
    f = Flat("C1 fit coupon", "none", outline, (0, 0, 0), X, Y, holes=holes, polys=slots, note="cut first: pick the bearing holes and slot width")
    f.labels = labels
    return f


def servo_coupon():
    """One STS3032 seat as in M1, Y2 and the L6 strips: the case cutout and the four tab holes, to try a servo and the
    webs beside its tab holes before the plates are cut."""
    c = B["cut_clear"]
    a, b = S32["len"]
    case = rrect(-S32["half_t"] - c, a - c, S32["half_t"] + c, b + c, 0.5)
    holes = [(sx * 3.5, zz, 2.2) for sx in (-1, 1) for zz in (-8.21, 19.79)]
    return Flat("C2 servo seat coupon", "none", rrect(-9.5, -11.5, 9.5, 23.0, 2.0), (0, 0, 0), X, Y, holes=holes, polys=[case],
                note="cut first: an STS3032 must drop in and screw down without cracking the webs")


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
    """The narrowest strip of plate between two cut contours of a part (outline, round holes, slots), in mm, and the
    two contours."""
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


def cmd_dxf(sheet_w=300.0, gap=4.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _S, g = pip_kin(np.array([B["pip"]["pip_rest"]]))
    parts = [f for f in flats() if not f.printed] + [rod_flat(g["L"]), fit_coupon(), servo_coupon()]
    placed, x, y, row_h = [], 0.0, 0.0, 0.0
    for f in sorted(parts, key=lambda f_: -(f_.outline[:, 1].max() - f_.outline[:, 1].min())):
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
    ents.append(text_entity(0.0, -6.0, 3.0, "v4b index finger, 1/8 in (3.175 mm) acrylic, 2026-11-19; CUT red, ENGRAVE blue; nominal outlines"))
    head = ("0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n"
            "0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n70\n2\n"
            "0\nLAYER\n2\nCUT\n70\n0\n62\n1\n6\nCONTINUOUS\n0\nLAYER\n2\nENGRAVE\n70\n0\n62\n5\n6\nCONTINUOUS\n"
            "0\nENDTAB\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n")
    CUT.mkdir(parents=True, exist_ok=True)
    out = CUT / "20261119-v4b_acrylic_3mm.dxf"
    out.write_text(head + "\n".join(ents) + "\n0\nENDSEC\n0\nEOF\n")
    # preview
    fig, ax = plt.subplots(figsize=(14, 14 * (H + 12) / sheet_w))
    for f, du, dv, rot, _bb in placed:
        def tr(P):
            P = np.asarray(P, float)
            if rot:
                P = np.stack([-P[:, 1], P[:, 0]], 1)
            return P + np.array([du, dv])
        P = tr(f.outline)
        ax.fill(P[:, 0], P[:, 1], color="#e8eef7", lw=0)
        ax.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], color="#c0392b", lw=0.9)
        for hp in f.polys:
            Q = tr(hp)
            ax.plot(np.r_[Q[:, 0], Q[0, 0]], np.r_[Q[:, 1], Q[0, 1]], color="#c0392b", lw=0.8)
        for hu, hv, hd in f.holes:
            (cx, cy), = tr([[hu, hv]])
            ax.add_patch(plt.Circle((cx, cy), hd / 2, fill=False, color="#c0392b", lw=0.8))
        lu, lv = label_point(f, rot)
        ax.text(lu + du, lv + dv, f.name.split()[0], color="#1f3f8f", fontsize=11, ha="center", va="center", family="serif")
    ax.plot([0, sheet_w, sheet_w, 0, 0], [-2, -2, H + 2, H + 2, -2], color="0.6", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xlim(-5, sheet_w + 5)
    ax.set_ylim(-8, H + 6)
    ax.set_xlabel("mm")
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / "20261119-v4b_cut_sheet.png", dpi=110)
    plt.close(fig)
    summary = []
    for f, *_r, bb in placed:
        w, pair = least_web(f)
        summary.append({"id": f.name.split()[0], "name": f.name, "body": f.body, "qty": f.qty, "note": f.note,
                        "size_mm": [round(bb[2], 1), round(bb[3], 1)], "holes": len(f.holes), "slots": len(f.polys),
                        "least_web_mm": round(w, 2) if math.isfinite(w) else None, "least_web_between": pair})
    res = {"script": "scripts/cad/v4b_build.py", "command": "dxf", "dxf": str(out.relative_to(ROOT)),
           "sheet_mm": [sheet_w, round(H, 1)], "thickness_mm": T, "parts": summary}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "20261119-v4b_parts.json").write_text(json.dumps(res, indent=1))
    return res


BOM = [  # item, qty, specification, where, example source, note
    ("STS3032 servo", 3, "Feetech STS3032, 6 V, 4.5 kg cm, 25-tooth spline", "yaw, MCP, PIP", "on hand",
     "each comes with an 18 mm aluminium disc horn and its M2.3 horn screw"),
    ("Acrylic sheet", 1, "1/8 in (3.175 mm) cast acrylic, at least 300 x 110 mm", "every plate but Y3", "", ""),
    ("Flanged ball bearing", 5, "MF63ZZ: 3 x 6 x 2.5 mm, flange 7.2 x 0.6 mm", "L2 (MCP idler), L4 and L5 (PIP), H1 (pin a), R1 (pin b)",
     "any MF63ZZ", "order 10; the fit coupon C1 picks the hole for a press fit"),
    ("Spherical plain bearing", 1, "3 mm bore, 12 mm outside, inner ring 4.8 mm, outer ring 3.6 mm wide, steel", "R1 at the lug",
     "RS PRO 382-380", "order 2; bonded into R1"),
    ("Shoulder screw 3 x 12", 1, "3 mm precision shoulder, 12 mm long, M2 x 4 thread, low head 5 x 1.3 mm", "pin a (knuckle to H1)",
     "McMaster 90323A609", ""),
    ("Shoulder screw 3 x 7", 1, "as above, 7 mm shoulder", "pin b (knuckle to R1)", "McMaster 90323A606", ""),
    ("Dowel pin 3 x 8", 1, "3 mm steel dowel, 8 mm long, h8 or m6", "MCP idler, pressed into Y3", "ISO 2338 or ISO 8734",
     "slides in L2's bearing"),
    ("Dowel pin 3 x 6", 2, "3 mm steel dowel, 6 mm long, h8 or m6", "PIP pins, pressed into the DIP link's cheeks",
     "ISO 2338 or ISO 8734", "slides in the PIP bearings' inner rings"),
    ("Dowel pin 3 x 12", 1, "3 mm steel dowel, 12 mm long", "lug pin, pressed through both cheeks and the spherical bearing", "ISO 2338 or ISO 8734", ""),
    ("M2 hex nut", 14, "M2, 4 mm across flats, 1.6 mm thick", "pins a and b (2, inside the bearings' bores), servo tabs (12)", "", ""),
    ("M2 x 8 screw", 12, "M2 x 8 socket or pan head", "servo tabs: M1, Y2, L6a and L6b (4 per servo)", "", ""),
    ("M2 x 5 screw", 10, "M2 x 5 socket head", "disc horns: Y1 (4), L1 (4), H1 (2)", "", "into the disc horns' tapped holes"),
    ("M3 x 10 screw and nut", 4, "M3 x 10", "M2 to the test stand", "", ""),
    ("Acrylic cement", 1, "solvent cement (Weld-On 4 or similar)", "tab-and-slot joints", "", ""),
    ("Retaining compound or epoxy", 1, "Loctite 638 or 5-minute epoxy", "spherical bearing in R1; any loose bearing", "", ""),
    ("Cyanoacrylate", 1, "medium CA glue", "Y3's tab into Y1 (printed into acrylic)", "", ""),
    ("Knuckle (printed)", 1, "PETG or PLA, 100 % infill: data/20261119-v4b_knuckle.stl", "between H1 and R1", "printed", ""),
    ("Yoke idler plate Y3 (printed)", 1, "PETG or PLA, 100 % infill: data/20261119-v4b_y3_idler_plate.stl", "yoke, MCP idler", "printed",
     "ream the 2.95 mm hole to a press fit for the dowel"),
    ("DIP link (printed)", 1, "b v4's tongue, board seat and column on new cheeks (the page's interface table)", "PIP to fingertip",
     "printed", "to be drawn from b v4 in Inventor"),
    ("Fingertip", 1, "v4b tip: cassette 3T and cap TW15H3 with the ReSkin board", "on the DIP link", "existing prints", ""),
]


def cmd_pip():
    """The PIP drive along the PIP angle (scripts/cad/v4c_hardware.py's pip_range with this layout's numbers): where the
    rod closes with the horn within its limit, the joint torque per SCS0009 limit and the spherical bearing's tilt;
    data/20261119-v4b_pip.json and media/20261119-v4b_pip_drive.png."""
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
           "horn_deg_span": [round(float(th[lo:hi + 1].min()), 1), round(float(th[lo:hi + 1].max()), 1)],
           "upper_end_cause": None,
           "curve": {"pip": PIP[lo:hi + 1].tolist(), "torque": np.round(R["tq"][lo:hi + 1], 3).tolist(),
                     "horn": np.round(th[lo:hi + 1], 2).tolist(), "tilt": np.round(R["tilt"][lo:hi + 1], 2).tolist()}}
    if hi + 1 < len(PIP):
        j = hi + 1
        causes = {"closure": not bool(R["S"]["ok"][j]), "horn_limit": abs(th[j]) > B["pip"]["horn_limit_deg"],
                  "torque": bool(R["tq"][j] < 1.0), "tilt": bool(R["tilt"][j] > V.HW["hardware"]["spherical_bearing"]["tilt_deg"])}
        res["upper_end_cause"] = [k for k, v in causes.items() if v]
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


def cmd_printed():
    """The printed parts as STLs: the knuckle in its own frame (see knuckle_spec; through holes of 3.1 mm for the
    shoulders, a 5.6 mm counterbore 1.3 mm deep for pin b's head, 4 mm bosses 0.5 mm tall on the bearings' inner rings) and
    the yoke idler plate Y3 (its outline extruded 1/8 in, a 2.95 mm hole for the idler's dowel)."""
    import trimesh
    K = knuckle_spec()
    bx, e, s = K["box"], K["e"], K["s"]
    lo, hi = np.array([v[0] for v in bx.values()]), np.array([v[1] for v in bx.values()])
    body = trimesh.creation.box(extents=hi - lo)
    body.apply_translation((lo + hi) / 2)

    def place(m, a, b, axis, at):
        if axis == "x":
            m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (0, 1, 0)))
            m.apply_translation(((a + b) / 2, at[0], at[1]))
        else:
            m.apply_translation((at[0], at[1], (a + b) / 2))
        return m

    def cyl(r, a, b, axis, at):
        return place(trimesh.creation.cylinder(radius=r, height=abs(b - a), sections=64), a, b, axis, at)

    def tube(r0, r1, a, b, axis, at):
        return place(trimesh.creation.annulus(r_min=r0, r_max=r1, height=abs(b - a), sections=64), a, b, axis, at)
    a_, b_ = K["a"], K["b"]
    bo, hh = B["boss"], B["shoulder"]["head_h"]
    zb = lambda t: -s * t  # noqa: E731, pin b's coordinate to z
    # holes first, then the bosses as tubes (Blender's boolean leaves a non-volume when a boss is cut after a union)
    for m in (cyl(1.55, bx["x"][0] - 0.1, bx["x"][1] + 0.1, "x", (0.0, 0.0)),
              cyl(1.55, bx["z"][0] - 0.1, bx["z"][1] + 0.1, "z", (0.0, e)),
              cyl(2.8, zb(bx["z"][1] - hh), zb(bx["z"][1]) - 0.1 * s, "z", (0.0, e))):
        body = body.difference(m, check_volume=False)
    body = body.union(tube(1.6, bo["d"] / 2, a_["ring"][0], bx["x"][0] + 0.01, "x", (0.0, 0.0)), check_volume=False)
    body = body.union(tube(1.6, bo["d"] / 2, zb(b_["ring"][0]), zb(-bx["z"][1] - 0.01), "z", (0.0, e)), check_volume=False)
    DATA.mkdir(parents=True, exist_ok=True)
    out_k = DATA / "20261119-v4b_knuckle.stl"
    body.export(out_k)
    y3 = next(f for f in flats() if f.name.startswith("Y3"))
    hu, hv, hd = y3.holes[0]
    P = y3.outline[:-1] if np.allclose(y3.outline[0], y3.outline[-1]) else y3.outline
    V2 = np.concatenate([[[hu, hv]], P])  # the outline is star-shaped about the insert's centre: a fan, then the hole
    F = np.array([[0, i, i % len(P) + 1] for i in range(1, len(P) + 1)])
    m3 = trimesh.creation.extrude_triangulation(V2, F, height=T)
    m3 = m3.difference(cyl(hd / 2, -0.1, T + 0.1, "z", (hu, hv)))
    area = 0.5 * abs(float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])))
    assert abs(m3.volume - (area - math.pi * (hd / 2) ** 2) * T) < 0.02 * area * T, "Y3 fan triangulation is not star-shaped"
    out_y = DATA / "20261119-v4b_y3_idler_plate.stl"
    m3.export(out_y)
    res = {"script": "scripts/cad/v4b_build.py", "command": "printed",
           "knuckle_stl": str(out_k.relative_to(ROOT)), "knuckle_mm": [round(float(x), 3) for x in hi - lo],
           "knuckle_watertight": bool(body.is_watertight), "knuckle_c_mm": K["c"], "x_a_mm": K["x_a"],
           "pin_a": {k: v for k, v in a_.items()}, "pin_b": {k: v for k, v in b_.items()},
           "y3_stl": str(out_y.relative_to(ROOT)), "y3_watertight": bool(m3.is_watertight)}
    (DATA / "20261119-v4b_printed.json").write_text(json.dumps(res, indent=1, default=float))
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
        Rd = rot_y(-math.radians(pip))  # link frame to the DIP link's frame, about the PIP axis
        for p in prims:
            c = Rd @ (Qp + Rr @ p["c"] - Op)
            if np.linalg.norm(c[[0, 2]]) > 25.0:
                continue
            m = trimesh.creation.box(extents=2 * p["h"] if p["r"] == 0 else 2 * np.r_[p["r"], p["r"], p["h"][2] + p["r"]])
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


COLORS = {"mount": (0.72, 0.74, 0.78, 1), "yoke": (0.45, 0.62, 0.85, 1), "link": (0.86, 0.56, 0.34, 1), "dip": (0.55, 0.75, 0.45, 1),
          "phorn": (0.7, 0.45, 0.7, 1), "knuckle": (0.85, 0.75, 0.25, 1), "rod": (0.85, 0.3, 0.3, 1)}


def cmd_render(poses=((0.0, 46.0, 0.0), (0.0, -15.0, -25.0), (30.0, 80.0, 70.0), (-60.0, 20.0, 40.0))):
    """Stills of the assembly (every primitive as a box) at four poses, from the side and the front."""
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pips = np.array([p[2] for p in poses])
    S, g = pip_kin(pips)
    P = all_prims(g["L"])
    S["rod_prims"], S["knuckle_prims"] = P.pop("rod_prims"), P.pop("knuckle_prims")
    gg = geometry()
    Op = np.array([0.0, 0.0, gg["pz"]])
    A0 = Op + rot_y(-math.radians(B["pip"]["pip_rest"])) @ (g["A0"] + np.array([0.0, 0.0, B["Gz"]]) - Op)
    P["dip"] += dip_lug_prims(A0)
    from sr2hand.geometry.rotations import mat_to_quat
    images = []
    for i, (yaw, mcp, pip) in enumerate(poses):
        W = bodies_at(P, yaw, mcp, pip, S, i)
        spec = mujoco.MjSpec()
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 900, 900
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                         rgb1=[1, 1, 1], rgb2=[1, 1, 1], width=16, height=16)
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.worldbody.add_light(pos=[0.05, -0.2, 0.3], dir=[-0.2, 0.6, -1.0], diffuse=[0.8, 0.8, 0.8])
        spec.worldbody.add_light(pos=[0.2, 0.2, 0.1], dir=[-1, -1, -0.3], diffuse=[0.4, 0.4, 0.4])
        for body, ps in W.items():
            for p in ps:
                if np.any(p["h"] <= 0):
                    continue
                if p["r"] > 0:
                    gq = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[p["r"] * 1e-3, max(p["h"][2], 1e-4) * 1e-3, 0],
                                                 pos=(p["c"] * 1e-3).tolist(), quat=mat_to_quat(p["R"]).tolist(), rgba=COLORS[body])
                else:
                    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(p["h"] * 1e-3).tolist(), pos=(p["c"] * 1e-3).tolist(),
                                            quat=mat_to_quat(p["R"]).tolist(), rgba=COLORS[body])
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        row = []
        with mujoco.Renderer(m, 700, 700) as r:
            for az, el in ((90.0, -10.0), (0.0, -10.0)):
                cam = mujoco.MjvCamera()
                cam.lookat[:] = [0.01, 0.0, -0.03]
                cam.distance, cam.azimuth, cam.elevation = 0.2, az, el
                r.update_scene(d, cam)
                row.append(r.render())
        images.append(row)
    fig, axes = plt.subplots(len(poses), 2, figsize=(10, 5 * len(poses)))
    for i, (yaw, mcp, pip) in enumerate(poses):
        for j in range(2):
            axes[i, j].imshow(images[i][j])
            axes[i, j].set_axis_off()
            axes[i, j].set_title(f"yaw {yaw:g}, MCP {mcp:g}, PIP {pip:g} deg ({'side' if j == 0 else 'palmar side'})", fontsize=16, family="serif")
    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = MEDIA / "20261119-v4b_assembly.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return str(out.relative_to(ROOT))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        fine = "--fine" in sys.argv
        res = check(3.0, tuple(np.arange(-85.0, 85.1, 15.0)) + (85.0,), tuple(np.arange(-25.0, 92.1, 6.0)) + (92.0,)) if fine else check()
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / f"20261119-v4b_check{'_fine' if fine else ''}.json").write_text(json.dumps(
            {"script": "scripts/cad/v4b_build.py", "command": "check" + (" --fine" if fine else ""), "params": B,
             "module_half_width_mm": res["module_half_width_mm"], "pip_closes": res["pip_closes"], "pip_rod_mm": res["pip_rod_mm"],
             "least": [[k, v] for k, v in res["least"][:40]]}, indent=1, default=float))
        print("half width", res["module_half_width_mm"], "pip closes", res["pip_closes"], "rod", res["pip_rod_mm"])
        for k, v in res["least"][:30]:
            print(f"{v[0]:7.2f}  {k:62s} yaw {v[1]:5.0f} mcp {v[2]:5.0f} pip {v[3]:5.0f}")
    elif cmd == "dxf":
        print(json.dumps(cmd_dxf(), indent=1))
    elif cmd == "bom":
        print(cmd_bom())
    elif cmd == "keepout":
        print(json.dumps(cmd_keepout()))
    elif cmd == "pip":
        print(json.dumps(cmd_pip(), indent=1))
    elif cmd == "printed":
        print(json.dumps(cmd_printed(), default=float))
    elif cmd == "render":
        print(cmd_render())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
