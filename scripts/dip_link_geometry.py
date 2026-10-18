#!/usr/bin/env python3
"""Shared geometry of the DIP-link redesign for the five-chip magnetometer board (2026-10-18).

Frames, the SR2 finger's CAD parts (assets/cad_v1/, placed by the assembly of scripts/build_cad_v1_finger.py), the printed
85A fingertips, the ReSkin 5X board read from its fabrication files, and the keep-out that the MCP link and the PIP servo
sweep through the DIP link's frame over the PIP range. scripts/dip_link_spec.py builds the specification from these; the
design stage imports the same functions to check a candidate part.

Frame F (the index finger at every joint angle zero, mm; scripts/build_cad_v1_finger.py): origin at the yaw centre, x along
the yaw axis toward the dorsal side, y along the PIP axis, z up. The straight finger hangs along -z. The PIP axis passes
through (0, y, -51.55); the tip's palmar face lies at x = -10.55 and its back face at x = 6.45; the contact point of the
bench grasp is (-10.55, 0, -78.16). The horn-side clevis arm of the DIP link lies at y < -17.05 and the slotted arm at
y > 13.40. The PIP hinge turns about +y: a positive angle moves the tip toward -x (flexion). The DIP link and everything
fixed to it (tip, board, clamp, cable anchor) keep their F coordinates at every PIP angle; in that frame the MCP link and the
PIP servo turn about the PIP axis by -q.

Board frame (u, v, w) of the ReSkin 5X files, mm, U1's centre at the origin: u across the board's 20.0 mm width, v along it
with U2 at v = +7.00 and the connector tab at v < -10.10, w out of the component side (every part is on that side). On the
finger, with U1 over the contact point: u = y_F, v = -(z_F + 78.16) (U2 toward the tip's end, the tab toward the PIP joint).

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 200 -- taskset -c 12-15 ~/miniconda3/envs/femtip/bin/python \\
        scripts/dip_link_geometry.py keepout        # rebuild keepout_pip.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

STUDY = REPO / "docs/experiments/20261018-dip_link_board_mount"
LOGS = REPO / "logs/20261018-dip_link"
RESKIN_URL = "https://github.com/raunaqbhirangi/reskin_sensor/tree/main/circuits"
RESKIN_DL = LOGS / "reskin_download"            # 5X_ReSkin.Zip, 5X_ReSkin_BOM.xlsx, README.md as downloaded
RESKIN_X = LOGS / "reskin_5x_extracted"         # the zip extracted; read as data only
ASSEMBLY = REPO / "assets/mjcf/experimental/20261013-cad_v1/assembly.json"
CAD = REPO / "assets/cad_v1"
KEEPOUT = STUDY / "keepout_pip.npz"

FILES = {"carriage": "carriage.stl", "yaw_link": "mcp_mount_YAW-link.stl", "mcp_link": "pip_mount_MCP_link.stl",
         "dip_link": "fingertip_mount_DIP_link.stl", "tip": "fingertip.stl", "servo": "feetech scs0009.stl",
         "tip_and_dip": "fingertip_and_DIP_link.stl"}

# finger frame F
PIP_C = np.array([0.0, 0.0, -51.55])            # a point on the PIP axis
PIP_RANGE_DEG = (-18.0, 92.0)                   # build_cad_v1_finger.ROM["pip"]: -0.314159 .. 1.605703 rad
TIP = dict(x_palmar=-10.55, x_back=6.45, y_half=7.4, z_mount=-67.30, z_end=-88.71, contact=(-10.55, 0.0, -78.16))
Z_CONTACT = -78.16


def tip_design_to_F(p):
    """scripts/tpu_tip_design.py's tip frame (palmar face x = +8.5, back x = -8.5, z = 0 at the mount face, 21.41 at the
    end) to F (index finger)."""
    p = np.asarray(p, float)
    return np.stack([-p[..., 0] - 2.05, p[..., 1], -67.30 - p[..., 2]], axis=-1)


def board_to_F(uvw, x_comp, comp_toward_tip=True):
    """Board coordinates (u, v, w) to F. x_comp is the x_F of the FR-4's component face; with comp_toward_tip the
    components point toward -x (the tip), otherwise toward +x."""
    uvw = np.asarray(uvw, float)
    s = -1.0 if comp_toward_tip else 1.0
    return np.stack([x_comp + s * uvw[..., 2], uvw[..., 0], -uvw[..., 1] + Z_CONTACT], axis=-1)


def rot_pip(points, q_deg):
    """Points fixed to the MCP link (F at q = 0) as seen from the DIP link at PIP angle q: a turn by -q about the PIP axis."""
    phi = math.radians(-q_deg)
    c, s = math.cos(phi), math.sin(phi)
    p = np.asarray(points, float) - PIP_C
    x, z = p[..., 0], p[..., 2]
    out = np.stack([x * c + z * s, p[..., 1], -x * s + z * c], axis=-1)   # rotation about +y by phi
    return out + PIP_C


# ---------------------------------------------------------------------------------------------------------- CAD parts

@lru_cache(maxsize=1)
def assembly() -> dict:
    return json.loads(ASSEMBLY.read_text())


def load_part(name: str):
    """A CAD part in F at q = 0: carriage, yaw_link, mcp_link, dip_link, tip, servo_yaw, servo_mcp, servo_pip."""
    import trimesh
    A = assembly()
    src = "servo" if name.startswith("servo") else name
    m = trimesh.load(CAD / FILES[src], force="mesh")
    m.apply_transform(np.array(A["T_part_design_from_stl"][src]))
    m.apply_transform(np.array(A["T_F_from_part_design_mm"][name]))
    return m


def ray_intervals(mesh, origin, direction):
    """Material intervals of a closed mesh along the line origin + t direction, as (t0, t1) pairs."""
    from morphohand.geometry import cad_features as cf
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    ts = cf._ray_all(mesh.triangles, np.asarray(origin, float), d)
    return [(float(a), float(b)) for a, b in zip(ts[0::2], ts[1::2])]


def line_material(mesh, axis: str, at: dict, lo=-60.0):
    """Material intervals along one F axis ('x', 'y' or 'z') through the point fixed by `at`, in F coordinates."""
    k = "xyz".index(axis)
    o = np.zeros(3)
    for a, v in at.items():
        o["xyz".index(a)] = v
    o[k] = lo if axis != "z" else 0.0
    d = np.zeros(3)
    d[k] = 1.0 if axis != "z" else -1.0
    iv = ray_intervals(mesh, o, d)
    if axis == "z":
        return [(round(-b, 3), round(-a, 3)) for a, b in iv][::-1]
    return [(round(lo + a, 3), round(lo + b, 3)) for a, b in iv]


# ------------------------------------------------------------------------------------------------- the ReSkin 5X board

MIL = 0.0254


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _gerber_val(tok: str) -> float:
    """FSLAX25Y25 with MOIN: an integer of 1e-5 inch, returned in mil."""
    return int(tok) / 100.0


def gerber_paths(path: Path):
    """Draws of a Gerber layer: ('line', (x0, y0), (x1, y1), aperture) and ('circle', centre, radius) in mil, and flashes
    ('flash', (x, y), aperture)."""
    txt = path.read_text()
    out, x, y, mode, ap = [], 0.0, 0.0, "G01", None
    for raw in txt.split("*"):
        s = raw.strip()
        if not s or s.startswith("%") or s.startswith("G04") or s.startswith("M02") or re.match(r"^\d+,", s):
            continue
        if s in ("G01", "G02", "G03"):
            mode = s
            continue
        m = re.fullmatch(r"(G0[123])?(?:X(-?\d+))?(?:Y(-?\d+))?(?:I(-?\d+))?(?:J(-?\d+))?(D0?[123])?", s)
        if m is None:
            m2 = re.fullmatch(r"D(\d+)", s)
            if m2 and int(m2.group(1)) >= 10:
                ap = int(m2.group(1))
            continue
        g, X, Y, I, J, D = m.groups()
        if g:
            mode = g
        nx = _gerber_val(X) if X else x
        ny = _gerber_val(Y) if Y else y
        if D in ("D01", "D1"):
            if mode == "G01":
                out.append(("line", (x, y), (nx, ny), ap))
            else:
                cx, cy = x + (_gerber_val(I) if I else 0.0), y + (_gerber_val(J) if J else 0.0)
                if abs(nx - x) < 1e-9 and abs(ny - y) < 1e-9:
                    out.append(("circle", (cx, cy), math.hypot(x - cx, y - cy)))
                else:
                    out.append(("arc", (x, y), (nx, ny), (cx, cy), mode))
        elif D in ("D03", "D3"):
            out.append(("flash", (nx, ny), ap))
        x, y = nx, ny
    return out


def gerber_apertures(path: Path) -> dict:
    """Aperture number -> (shape, sizes in mm along x and y)."""
    txt = path.read_text()
    ap = {}
    for m in re.finditer(r"%ADD(\d+)([A-Z]+)(?:,([0-9.X]+))?\*%", txt):
        d, shape, par = int(m.group(1)), m.group(2), m.group(3)
        if shape in ("R", "C", "O") and par:
            dims = [float(v) * 1000 * MIL for v in par.split("X")]
            ap[d] = (shape, dims if len(dims) == 2 else dims * 2)
    for m in re.finditer(r"DCode=(\d+)\|XSize=([0-9.]+)mil\|YSize=([0-9.]+)mil.*?Rotation=([0-9.]+)", txt):
        d, sx, sy, rot = int(m.group(1)), float(m.group(2)) * MIL, float(m.group(3)) * MIL, float(m.group(4))
        if abs(rot - 90) < 1 or abs(rot - 270) < 1:
            sx, sy = sy, sx
        ap[d] = ("rounded rectangle", [sx, sy])
    return ap


def excellon_holes(path: Path):
    """Holes of an Excellon file in inch with leading zeros kept and the 2:5 format: [(diameter mm, x mil, y mil)]."""
    txt = path.read_text()
    tools, cur, x, y, out = {}, None, None, None, []
    fmt = re.search(r"FILE_FORMAT=(\d):(\d)", txt)
    ni = int(fmt.group(1)) if fmt else 2

    def val(tok):
        sign = -1.0 if tok.startswith("-") else 1.0
        tok = tok.lstrip("+-")
        return sign * float(tok[:ni] + "." + tok[ni:]) * 1000.0        # inch -> mil

    for line in txt.splitlines():
        s = line.strip()
        m = re.fullmatch(r"T(\d+)F\d+S\d+C([0-9.]+)", s)
        if m:
            tools[int(m.group(1))] = float(m.group(2)) * 25.4
            continue
        m = re.fullmatch(r"T(\d+)", s)
        if m:
            cur = int(m.group(1))
            continue
        m = re.fullmatch(r"(?:X([-+]?\d+))?(?:Y([-+]?\d+))?", s)
        if m and (m.group(1) or m.group(2)) and cur is not None:
            x = val(m.group(1)) if m.group(1) else x
            y = val(m.group(2)) if m.group(2) else y
            out.append((tools[cur], x, y))
    return out


def pick_place(path: Path) -> dict:
    """Designator -> footprint, centre (mil), pad 1 (mil), side, rotation, comment."""
    rows = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^(\S+)\s+(\S+)\s+([\d.]+)mil\s+([\d.]+)mil\s+([\d.]+)mil\s+([\d.]+)mil\s+([\d.]+)mil\s+([\d.]+)mil\s+"
                     r"([TB])\s+([\d.]+)\s+(.*)$", line.strip())
        if m:
            g = m.groups()
            rows[g[0]] = dict(footprint=g[1], mid_mil=(float(g[2]), float(g[3])), pad1_mil=(float(g[6]), float(g[7])),
                              side=g[8], rotation_deg=float(g[9]), comment=g[10].strip())
    return rows


# catalogue heights above the FR-4's component face, mm (not in the fabrication files)
PART_HEIGHT = {
    "MLX90393": (0.80, 1.00, "Melexis MLX90393 datasheet rev. 012, Fig. 26 (QFN-16 3 x 3 mm, 0.80-1.00 mm)"),
    "CAP0402": (0.45, 0.55, "Yageo CC0402 series: 1.0 x 0.5 x 0.5 +- 0.05 mm"),
    "RES0402": (0.30, 0.40, "Yageo RC0402 series: 1.0 x 0.5 x 0.35 +- 0.05 mm"),
    "J1": (2.9, 2.9, "JST SH side-entry header SM04B-SRSS-TB, about 2.9 mm; to be measured on the owner's board"),
}


@lru_cache(maxsize=1)
def board() -> dict:
    """The ReSkin 5X board in its own frame (u, v), mm, U1 at the origin, from the downloaded fabrication files."""
    G = RESKIN_X / "Gerber"
    pp = pick_place(RESKIN_X / "Pick Place" / "Pick Place for PCB.txt")
    u1 = np.array(pp["U1"]["mid_mil"])

    def uv(p):
        return [round((p[0] - u1[0]) * MIL, 3), round((p[1] - u1[1]) * MIL, 3)]

    outline = gerber_paths(G / "PCB.Outline")
    segs = [(uv(a), uv(b)) for k, a, b, *_ in [o for o in outline if o[0] == "line"]]
    circles = [dict(centre_uv=uv(c), radius_mm=round(r * MIL, 3)) for k, c, r in [o for o in outline if o[0] == "circle"]]
    us = [p[0] for s in segs for p in s]
    vs = [p[1] for s in segs for p in s]
    main_v = sorted({p[1] for s in segs for p in s if abs(abs(p[0]) - 10.0) < 0.05})
    holes = excellon_holes(RESKIN_X / "NC Drill" / "PCB.TXT")
    mount = [dict(d_mm=round(d, 3), uv=uv((x, y))) for d, x, y in holes if d > 1.0]
    vias = [dict(d_mm=round(d, 3), uv=uv((x, y))) for d, x, y in holes if d <= 1.0]
    # parts
    parts = {}
    for k, r in pp.items():
        c = uv(r["mid_mil"])
        kind = ("MLX90393" if r["comment"].startswith("Melexis") else "J1" if k == "J1" else
                "RES0402" if k.startswith("R") else "CAP0402")
        size = {"MLX90393": (3.0, 3.0), "CAP0402": (1.0, 0.5), "RES0402": (1.0, 0.5), "J1": (None, None)}[kind]
        rot = r["rotation_deg"] % 180
        if size[0] is not None and abs(rot - 90) < 1:
            size = (size[1], size[0])
        populated = not (k.startswith("C_") and int(k.split("_")[1]) >= 6)    # BOM: C6-C9 left unpopulated
        parts[k] = dict(kind=kind, centre_uv=c, size_uv=list(size), populated=populated,
                        height_mm=list(PART_HEIGHT[kind][:2]), side="component" if r["side"] == "T" else "back",
                        part=r["comment"])
    # J1: its body from the top overlay (the rectangle on the tab), its pads from the paste layer
    ov = [o for o in gerber_paths(G / "PCB.GTO") if o[0] == "line"]
    on_tab = [(uv(a), uv(b)) for _, a, b, _ in ov if uv(a)[1] < -12.0 and uv(b)[1] < -12.0 and abs(uv(a)[0]) < 5.5]
    jb = np.array([p for s in on_tab for p in s])
    aps = gerber_apertures(G / "PCB.GTP")
    pads = []
    for _, p, a in [o for o in gerber_paths(G / "PCB.GTP") if o[0] == "flash"]:
        q = uv(p)
        if q[1] < -12.5 and abs(q[0]) < 5.5 and a in aps:
            pads.append(dict(centre_uv=q, size_mm=[round(v, 3) for v in aps[a][1]]))
    j1 = parts["J1"]
    j1.update(body_u=[round(float(jb[:, 0].min()), 3), round(float(jb[:, 0].max()), 3)],
              body_v=[round(float(jb[:, 1].min()), 3), round(float(jb[:, 1].max()), 3)],
              pads=sorted(pads, key=lambda d: (d["centre_uv"][1], d["centre_uv"][0])),
              mating_face="toward -v (the tab's end): the plug enters along +v; signal pads at v = -13.70, mounting pads at "
                          "v = -17.45")
    j1["size_uv"] = [round(j1["body_u"][1] - j1["body_u"][0], 3), round(j1["body_v"][1] - j1["body_v"][0], 3)]
    files = {p.name: _sha(p) for p in sorted(RESKIN_DL.glob("*"))}
    return dict(
        source=RESKIN_URL + " (5X_ReSkin.Zip: Altium CircuitMaker Gerber, NC drill and pick-and-place of 2021-03-10; "
                            "5X_ReSkin_BOM.xlsx; README.md)",
        files_sha256=files, local_copy=str(RESKIN_DL.relative_to(REPO)),
        thickness_mm=0.6, thickness_src="ReSkin README, rigid order: FR-4, 2 layers, 0.6 mm (a 0.1 mm polyimide flex version "
                                        "is also described); to be measured on the owner's board",
        outline=dict(u=[round(min(us), 3), round(max(us), 3)], v_main=[min(main_v), round(max(vs), 3)],
                     v_side_edges=[min(main_v), max(main_v)],
                     tab_u=[-5.0, 5.0], tab_v=[round(min(vs), 3), min(main_v)], segments_uv=segs, circles=circles,
                     note="the top edge is drawn at v = +10.05 and the side edges end at v = +9.90; a 1.2 mm radius "
                          "circle on the outline layer at (-8.00, +8.05) lies 0.15 mm from the drilled hole at (-8.00, +7.90)"),
        mounting_holes=mount, vias=dict(count=len(vias), d_mm=vias[0]["d_mm"] if vias else None),
        mounting_holes_note="four plated holes, drill 2.40 mm, from the NC drill file; centres 16.00 mm apart across and "
                            "along, their centre 0.10 mm proximal of U1",
        parts=parts, component_side="all parts on one side (bottom paste and overlay layers are empty)",
        sensing_spot_mm=dict(below_package_top=0.42, tol=0.10, lateral_tol=0.20,
                             src="MLX90393 datasheet Fig. 26: 0.42 +- 0.10 mm under the package top, die centred to +-0.20 mm"),
        bus="five MLX90393 at I2C addresses 0x0C-0x10 on one 400 kHz bus; J1 is a 4-pin JST SH (1.0 mm pitch)",
        mass_g=0.64, mass_src="scripts/sim2real_fingertip_scenes.py BOARD (FR-4 0.44 g, five chips 0.10 g, J1 0.10 g)")


# ------------------------------------------------------------------------------------------------------- the keep-out

class KeepOut:
    """Occupancy of the MCP link and the PIP servo swept over a PIP range, on a regular grid in F (the DIP link's frame).
    Cell (i, j, k) has its centre at lo + pitch * (i, j, k)."""

    def __init__(self, occ, lo, pitch, q_range, parts):
        self.occ, self.lo, self.pitch = occ, np.asarray(lo, float), float(pitch)
        self.q_range, self.parts = tuple(q_range), list(parts)
        self._sd = None

    @property
    def signed(self):
        """Signed distance per cell, mm: positive outside the swept parts (distance to the nearest occupied cell centre),
        negative inside (depth)."""
        if self._sd is None:
            from scipy.ndimage import distance_transform_edt
            out = distance_transform_edt(~self.occ).astype(np.float32)
            ins = distance_transform_edt(self.occ).astype(np.float32)
            self._sd = (out - ins) * self.pitch
        return self._sd

    def query(self, P):
        """Signed distance (mm) at points P (n, 3) in F, nearest cell; points outside the grid get +inf."""
        P = np.asarray(P, float)
        ijk = np.rint((P - self.lo) / self.pitch).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.occ.shape)), axis=1)
        out = np.full(len(P), np.inf, np.float32)
        sd = self.signed
        out[ok] = sd[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        return out

    def box_clearance(self, lo, hi, step=None):
        """Least signed distance over a solid box [lo, hi] in F (sampled through its volume): (value mm, point)."""
        step = step or self.pitch
        axes = [np.linspace(a, b, max(2, int(math.ceil((b - a) / step)) + 1)) for a, b in zip(lo, hi)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        d = self.query(P)
        i = int(np.argmin(d))
        return float(d[i]), P[i]

    def save(self, path=KEEPOUT):
        np.savez_compressed(path, bits=np.packbits(self.occ.ravel()), shape=np.array(self.occ.shape), lo=self.lo,
                            pitch=self.pitch, q_range_deg=np.array(self.q_range), parts=np.array(self.parts))

    @classmethod
    def load(cls, path=KEEPOUT):
        z = np.load(path)
        shape = tuple(int(v) for v in z["shape"])
        occ = np.unpackbits(z["bits"])[:int(np.prod(shape))].reshape(shape).astype(bool)
        return cls(occ, z["lo"], float(z["pitch"]), tuple(z["q_range_deg"]), [str(p) for p in z["parts"]])


def build_keepout(q_range=PIP_RANGE_DEG, parts=("mcp_link", "servo_pip"), pitch=0.25, lo=(-22.0, -26.0, -94.0),
                  hi=(30.0, 20.0, -30.0), src_pitch=0.2, dr=0.2, dy=0.2, dth_deg=0.25) -> KeepOut:
    """Sweep the parent-side parts about the PIP axis. The parts are voxelized (scanlines, src_pitch), their cells binned in
    cylindrical coordinates about the PIP axis (r, y, theta = atan2(x, z - z_PIP)); a turn by -q adds -q to theta, so the
    swept set is each (r, y) row dilated along theta over [-q_max, -q_min]. Each row is first dilated by the angle that
    one source cell subtends at its radius, so a solid stays solid in the binning. The F grid then looks up its cells."""
    from morphohand.geometry import cad_features as cf
    pts = np.concatenate([cf.voxelize_solid(load_part(p), src_pitch).centres() for p in parts])
    rel = pts - PIP_C
    r = np.hypot(rel[:, 0], rel[:, 2])
    th = np.degrees(np.arctan2(rel[:, 0], rel[:, 2]))
    nth = int(round(360.0 / dth_deg))
    r_max = float(r.max()) + 2 * dr
    y0 = float(pts[:, 1].min()) - 2 * dy
    nr, ny = int(math.ceil(r_max / dr)) + 1, int(math.ceil((pts[:, 1].max() + 2 * dy - y0) / dy)) + 1
    ir = np.clip(np.rint(r / dr).astype(np.int64), 0, nr - 1)
    iy = np.clip(np.rint((pts[:, 1] - y0) / dy).astype(np.int64), 0, ny - 1)
    it = np.rint((th % 360.0) / dth_deg).astype(np.int64) % nth
    O = np.zeros((nr, ny, nth), bool)
    O[ir, iy, it] = True
    del pts, rel, th, ir, iy, it

    def roll_or(A, k):
        """OR of A rolled by 0..k-1 along theta (k >= 1), by doubling."""
        out, P, off, bit = None, A, 0, 1
        while k:
            if k & 1:
                sh = np.roll(P, off, axis=-1)
                out = sh if out is None else (out | sh)
                off += bit
            k >>= 1
            if k:
                P = P | np.roll(P, bit, axis=-1)
                bit *= 2
        return out

    for i in range(nr):                              # fill the binning gaps of each radius
        ri = max(i * dr, dr)
        k = int(math.ceil(math.degrees(src_pitch * 1.5 / ri) / dth_deg))
        if k > 1:
            O[i] = np.roll(roll_or(O[i], k), -(k // 2), axis=-1)
    lo_shift, hi_shift = -q_range[1], -q_range[0]     # theta' = theta + phi, phi in [-q_max, -q_min]
    L = int(round((hi_shift - lo_shift) / dth_deg)) + 1
    B = roll_or(O, L)                                 # B[t] = OR O[t - j], j = 0..L-1
    del O
    S = np.roll(B, int(round(lo_shift / dth_deg)), axis=-1)   # S[t'] = B[t' - lo_shift]
    del B
    # the F grid
    lo = np.asarray(lo, float)
    shape = tuple(int(math.ceil((h - l) / pitch)) + 1 for l, h in zip(lo, hi))
    gx, gy, gz = (lo[k] + pitch * np.arange(shape[k]) for k in range(3))
    occ = np.zeros(shape, bool)
    X, Z = np.meshgrid(gx, gz, indexing="ij")
    rr = np.hypot(X - PIP_C[0], Z - PIP_C[2])
    tt = np.degrees(np.arctan2(X - PIP_C[0], Z - PIP_C[2])) % 360.0
    jr = np.rint(rr / dr).astype(np.int64)
    jt = np.rint(tt / dth_deg).astype(np.int64) % nth
    inside_r = jr < nr
    for j, yv in enumerate(gy):
        jy = int(round((yv - y0) / dy))
        if 0 <= jy < ny:
            sl = np.zeros_like(inside_r)
            sl[inside_r] = S[jr[inside_r], jy, jt[inside_r]]
            occ[:, j, :] = sl
    return KeepOut(occ, lo, pitch, q_range, parts)


def keepout() -> KeepOut:
    return KeepOut.load(KEEPOUT)


# ------------------------------------------------------------------------------- exact clearance over the PIP range

def surface_points(mesh, spacing=0.25):
    """Points on a mesh's surface (exact, deterministic): every edge sampled at most `spacing` apart, and on each triangle
    whose inradius exceeds spacing / 2 a barycentric grid with n = ceil(longest edge / spacing) steps per edge. Every
    surface point then lies within about 0.7 spacing of a sample: a thinner triangle (a CAD sliver) is covered by its
    edges' samples."""
    V = mesh.vertices
    E = mesh.edges_unique
    a, b = V[E[:, 0]], V[E[:, 1]]
    n_e = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / spacing).astype(int))
    out = [V]
    for n in np.unique(n_e):
        t = (np.arange(1, n) / n)[None, :, None]
        if t.size:
            sel = n_e == n
            out.append((a[sel][:, None, :] * (1 - t) + b[sel][:, None, :] * t).reshape(-1, 3))
    T = mesh.triangles
    side = np.linalg.norm(T[:, [1, 2, 0]] - T, axis=2)
    r_in = 2 * mesh.area_faces / np.maximum(side.sum(axis=1), 1e-12)
    fat = r_in > spacing / 2
    n_of = np.maximum(1, np.ceil(side.max(axis=1) / spacing).astype(int))
    for n in np.unique(n_of[fat]):
        tri = T[fat & (n_of == n)]
        i, j = np.meshgrid(np.arange(1, n), np.arange(1, n), indexing="ij")
        keep = i + j < n                                            # interior points only
        if not keep.any():
            continue
        W = np.stack([1 - (i[keep] + j[keep]) / n, i[keep] / n, j[keep] / n], 1)
        step = max(1, 200000 // len(W))
        for s in range(0, len(tri), step):
            out.append(np.einsum("kw,twd->tkd", W, tri[s:s + step]).reshape(-1, 3))
    return np.unique(np.round(np.vstack(out), 5), axis=0)


@lru_cache(maxsize=4)
def parent_samples(parts=("mcp_link", "servo_pip"), spacing=0.25):
    """Surface points (F at q = 0) of the parts fixed to the MCP link, with the part index; cached in logs/."""
    cache = LOGS / f"parent_surface_{'_'.join(parts)}_{spacing:g}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["P"], z["idx"]
    P, idx = [], []
    for k, name in enumerate(parts):
        P.append(surface_points(load_part(name), spacing))
        idx.append(np.full(len(P[-1]), k))
    P, idx = np.vstack(P), np.concatenate(idx)
    LOGS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, P=P, idx=idx)
    return P, idx


@lru_cache(maxsize=4)
def parent_solids(parts=("mcp_link", "servo_pip"), pitch=0.2):
    from morphohand.geometry import cad_features as cf
    return [cf.voxelize_solid(load_part(p), pitch) for p in parts]


def q_grid(q_range=PIP_RANGE_DEG, step=0.5):
    n = int(round((q_range[1] - q_range[0]) / step)) + 1
    return np.linspace(q_range[0], q_range[1], n)


def boxes_clearance_exact(boxes, q_range=PIP_RANGE_DEG, step=0.5, parts=("mcp_link", "servo_pip"), reach=4.0):
    """Least signed distance from the surface samples of the parent-side parts to each box fixed to the DIP link, over the
    PIP range (negative: a part's surface enters the box, by that depth): {name: {part: (mm, q_deg)}}, boxes given as
    {name: (lo, hi)} in F. The definition of docs/experiments/20261016-sim2real_fingertip_readiness/scenes/
    interference.json. A turn about the PIP axis keeps each sample's y and its radius from the axis, so only the samples
    within `reach` mm of a box in y and in radius are turned; a box farther than `reach` from every part reports +inf."""
    P, idx = parent_samples(parts)
    rel = P - PIP_C
    r = np.hypot(rel[:, 0], rel[:, 2])
    out = {}
    for name, (lo, hi) in boxes.items():
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c, h = (lo + hi) / 2, (hi - lo) / 2
        a2 = PIP_C[[0, 2]]
        r_lo = float(np.linalg.norm(np.maximum(np.maximum(lo[[0, 2]] - a2, a2 - hi[[0, 2]]), 0.0)))   # axis to box, x-z
        r_hi = float(max(np.hypot(x - a2[0], z - a2[1]) for x in (lo[0], hi[0]) for z in (lo[2], hi[2])))
        sel = ((P[:, 1] >= lo[1] - reach) & (P[:, 1] <= hi[1] + reach) & (r >= r_lo - reach) & (r <= r_hi + reach))
        res = {p: (np.inf, None) for p in parts}
        if sel.any():
            Ps, ids = P[sel], idx[sel]
            for q in q_grid(q_range, step):
                d = np.abs(rot_pip(Ps, q) - c) - h
                sd = np.where(np.all(d <= 0, axis=1), d.max(axis=1), np.linalg.norm(np.maximum(d, 0.0), axis=1))
                for k, p in enumerate(parts):
                    m = ids == k
                    if m.any():
                        v = float(sd[m].min())
                        if v < res[p][0]:
                            res[p] = (v, float(q))
        out[name] = {p: (round(v, 3) if np.isfinite(v) else None, q) for p, (v, q) in res.items()}
    return out


def box_clearance_exact(lo, hi, **kw):
    return boxes_clearance_exact({"box": (lo, hi)}, **kw)["box"]


def points_clearance_exact(points, q_range=PIP_RANGE_DEG, step=0.5, parts=("mcp_link", "servo_pip"), reach=4.0):
    """Least distance (mm) from points fixed to the DIP link to the parent-side parts over the PIP range; negative where a
    point lies inside a part (by the distance to its surface). Returns (value, point index, q_deg, part). Pass the surface
    samples and the voxel centres of a candidate part to check it."""
    from scipy.spatial import cKDTree
    P, idx = parent_samples(parts)
    X0 = np.asarray(points, float)
    # a turn about the PIP axis keeps radius and y, so the distance in the (radius, y) plane bounds the 3-D distance below;
    # the swept keep-out's distance field (within 0.3 mm) then drops the points that stay far at every angle
    ry = lambda A: np.column_stack([np.hypot(A[:, 0] - PIP_C[0], A[:, 2] - PIP_C[2]), A[:, 1]])  # noqa: E731
    d_lb, _ = cKDTree(ry(P)).query(ry(X0), k=1)
    if KEEPOUT.exists() and tuple(parts) == ("mcp_link", "servo_pip"):
        K = keepout()
        if K.q_range == tuple(q_range):
            d_k = K.query(X0)
            d_lb = np.maximum(d_lb, np.where(np.isfinite(d_k), d_k - 0.3, d_lb))
    near = np.flatnonzero(d_lb < reach)
    if not len(near):
        return (float(d_lb.min()), int(np.argmin(d_lb)), None, None)
    X = X0[near]
    tree = cKDTree(P)
    sol = parent_solids(parts)
    best = (np.inf, None, None, None)
    for q in q_grid(q_range, step):
        Xp = rot_pip(X, -q)                     # the DIP-fixed points in the parent's frame at angle q
        d, j = tree.query(Xp, k=1)
        ins = np.zeros(len(Xp), bool)
        for s in sol:
            ins |= s.contains(Xp)
        sd = np.where(ins, -d, d)
        i = int(np.argmin(sd))
        if sd[i] < best[0]:
            best = (float(sd[i]), int(near[i]), float(q), parts[int(idx[j[i]])])
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keepout", help="sweep the MCP link and the PIP servo over the PIP range -> keepout_pip.npz")
    sub.add_parser("board", help="print the board read from the fabrication files")
    a = ap.parse_args()
    if a.cmd == "keepout":
        t0 = time.time()
        K = build_keepout()
        STUDY.mkdir(parents=True, exist_ok=True)
        K.save()
        print(f"keep-out {K.occ.shape} at {K.pitch} mm, {K.occ.sum() * K.pitch ** 3:.0f} mm^3 occupied, "
              f"{time.time() - t0:.1f} s -> {KEEPOUT.relative_to(REPO)}", flush=True)
    elif a.cmd == "board":
        print(json.dumps(board(), indent=1, default=float))


if __name__ == "__main__":
    main()
