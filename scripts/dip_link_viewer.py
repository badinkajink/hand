#!/usr/bin/env python3
"""Local viewer of the DIP-link designs a and b and their parameter files (2026-10-18).

Owner, 2026-10-18 18:15: "inevitably i will need to tweak sizes, spacing. so either we make these things cad-interfacable
(for a human) or enable programmatic reconfiguration (with a viewer, ideally)".

A web server on 127.0.0.1 (Python standard library) serves scripts/dip_link_viewer.html. The page lists every parameter of
a parameter file (docs/experiments/20261018-dip_link_board_mount/params/) as a numeric field with its range; a change
regenerates the parts in this process with manifold3d (concept a: dip_link_concept_a's MeshK through
scripts/dip_link_params.py; concept b: dip_link_concept_b's MeshB), in well under a second, and updates the readouts:
least clearance to the MCP link and the PIP servo over PIP -18..92 deg from the swept keep-out (keepout_pip.npz, within
0.3 mm), the clearance at the slider's PIP angle (the parents' surface samples, exact to the 0.25 mm sampling), the
board-to-tip gap, the chip-to-magnet distance, the least wall in the checked sections and the mass. The page saves a
parameter set as a dated file under params/, and starts the full clearance check and the export of a saved set as jobs
under ~/.claude/bin/resguard.sh (scripts/dip_link_params.py check, export).

Start it from a VS Code terminal on the workstation; VS Code forwards the port to the browser:

    cd ~/Programs/hand
    ~/.claude/bin/resguard.sh run --mem 3G --cpu 400 -- taskset -c 8-11 env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python scripts/dip_link_viewer.py          # then open http://127.0.0.1:8780/
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import scipy.spatial  # noqa: F401  (the environment's libstdc++ before manifold3d)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402

import dip_link_params as DP  # noqa: E402

PAGE = REPO / "scripts/dip_link_viewer.html"
JOBS_DIR = REPO / "logs/20261018-dip_link/viewer_jobs"
LOCK = threading.Lock()                      # the generators keep their parameters in module state: one build at a time
STATE = {}                                   # the last build: concept, sample points for the per-angle clearance
JOBS = {}

TIPS_A = ["block4", "block5", "cap8.5", "ell16", "ell11"]
TIP_LABEL = {"block4": "Block, four magnets", "block5": "Block, five magnets", "cap8.5": "Capsule r 8.5 mm",
             "ell16": "Half-ellipsoid 16/32 mm", "ell11": "Half-ellipsoid 11/22 mm",
             "ell11_5": "Half-ellipsoid 11/22 mm, five magnets"}
COLOR = dict(dip_link="#d4d4cc", clamp_plate="#5b8fd9", cover="#3b6fc4", board="#1f7a3a", wires="#d9531e",
             fasteners="#c8a33c", qwiic="#f4f3ea", tip="#e8c88a", magnets="#b03a2e", mcp_link="#9fb6cf", servo_pip="#3a3a3e")


# ----------------------------------------------------------------------------------------------- geometry helpers
def _mesh_payload(name, label, tm, color, opacity=1.0, group="design"):
    V = np.ascontiguousarray(np.asarray(tm.vertices, np.float32))
    F = np.ascontiguousarray(np.asarray(tm.faces, np.uint32))
    return dict(name=name, label=label, color=color, opacity=opacity, group=group, triangles=int(len(F)),
                positions=base64.b64encode(V.tobytes()).decode(), indices=base64.b64encode(F.tobytes()).decode())


def _tm(man):
    import trimesh
    m = man.to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3].astype(float), np.asarray(m.tri_verts), process=False)


def _boxes(M3, boxes):
    return M3.Manifold.batch_boolean([M3.Manifold.cube([float(h - l) for l, h in zip(lo, hi)]).translate([float(v) for v in lo])
                                      for _, lo, hi in boxes], M3.OpType.Add)


def _magnets(M3, items, d):
    """3 x 2 mm disc magnets with their axes along x: [{centre_F, x_F}] -> one manifold."""
    out = []
    for m in items:
        x0, x1 = m["x_F"]
        c = M3.Manifold.cylinder(float(x1 - x0), d / 2, d / 2, 48).rotate([0.0, 90.0, 0.0])
        out.append(c.translate([float(x0), float(m["centre_F"][1]), float(m["centre_F"][2])]))
    return M3.Manifold.batch_boolean(out, M3.OpType.Add)


def disc_distance(P, centre, x0, x1, R):
    r = np.hypot(P[:, 1] - centre[1], P[:, 2] - centre[2])
    dx = np.maximum.reduce([x0 - P[:, 0], np.zeros(len(P)), P[:, 0] - x1])
    return np.hypot(dx, np.maximum(r - R, 0.0))


class Geometry:
    """The parts, context and checks shared by every build of this process (loaded once)."""

    def __init__(self):
        import manifold3d as M3
        import dip_link_geometry as G
        self.M3, self.G = M3, G
        self.A, self.DLF = DP.modules_a()
        import dip_link_concept_b as B
        self.B = B
        self.keepout = G.keepout()
        _ = self.keepout.signed                                  # the distance field (1.7 s, 80 MB)
        self.parents, self.parent_idx = G.parent_samples(("mcp_link", "servo_pip"))
        self.tree = scipy.spatial.cKDTree(self.parents)
        self.solids = G.parent_solids(("mcp_link", "servo_pip"))
        self.context = {n: G.load_part(n) for n in ("mcp_link", "servo_pip")}
        self._arms_a = None
        self._tips_a = {}
        self.tips_a = self.DLF.spec_final()["tips"]
        self.tip = None

    # concept a ---------------------------------------------------------------------------------------------------
    def kernel_a(self):
        geo = self

        class MeshKc(self.A.MeshK):
            def arms(self_k):
                if geo._arms_a is None:
                    geo._arms_a = self.A.MeshK.arms(self_k)
                return geo._arms_a
        return MeshKc()

    def tip_a(self, key):
        ck = (key, round(float(self.A.Z_MOUNT), 6), json.dumps(self.DLF.spec_final()["tips"][key]["magnets"]["items"]))
        if ck not in self._tips_a:
            tm, _ = self.A.tip_F(key, self.DLF.spec_final())
            self._tips_a[ck] = self.A.from_trimesh(tm)
        return self._tips_a[ck]

    def tips_of(self, concept):
        """The tips (a) or pads (b) of the design just applied."""
        return list(self.DLF.spec_final()["tips"]) if concept == "a" else list(self.B.TIPS)

    def build_a(self, vals, des, tip):
        A, DLF, M3 = self.A, self.DLF, self.M3
        DP.apply_a(vals, des)
        self.tips_a = DLF.spec_final()["tips"]
        tip = tip if tip in self.tips_a else ("ell11_5" if "ell11_5" in self.tips_a else next(iter(self.tips_a)))
        self.tip = tip
        j1 = (des or {}).get("qwiic") == "j1_kept"
        K = self.kernel_a()
        L = A.levels()
        D = A.dip_link(K)
        C = A.clamp_plate(K)
        Bm, wires = A.board_model(K)
        W = None if j1 else A.wires_model(K, A.P, wires, +1)
        F = K.union([m for _, m in A.fasteners(K)])
        T = self.tip_a(tip) - A.tip_cutter(K)
        Q = _boxes(M3, A.j1_boxes_F() if j1 else DLF.qwiic_header_final()[0])
        mags = self.tips_a[tip]["magnets"]
        Mg = _magnets(M3, mags["items"], mags["size_mm"][0])
        tms = dict(dip_link=_tm(D), clamp_plate=_tm(C), board=_tm(Bm), fasteners=_tm(F), qwiic=_tm(Q), tip=_tm(T),
                   magnets=_tm(Mg))
        if W is not None:
            tms["wires"] = _tm(W)
        parts = [_mesh_payload("dip_link", "DIP link (clevis arms of the STL, bridge, window plate, keys)", tms["dip_link"],
                               COLOR["dip_link"]),
                 _mesh_payload("clamp_plate", "Clamp plate" + ("" if j1 else " with the Qwiic pocket"), tms["clamp_plate"],
                               COLOR["clamp_plate"]),
                 _mesh_payload("board", "Board: FR-4, chips, parts" + (" (whole tab)" if j1 else " (tab cut)"), tms["board"],
                               COLOR["board"], group="board"),
                 _mesh_payload("qwiic", "J1, plug and cable start" if j1 else "Qwiic header, plug, cable start, pigtail",
                               tms["qwiic"], COLOR["qwiic"], group="board")]
        if W is not None:
            parts.append(_mesh_payload("wires", "Wires from J1's pads", tms["wires"], COLOR["wires"], group="board"))
        parts += [
                 _mesh_payload("fasteners", "Screws (brass)", tms["fasteners"], COLOR["fasteners"], group="hardware"),
                 _mesh_payload("tip", f"Tip, 85A: {TIP_LABEL.get(tip, tip)}, key channels cut", tms["tip"], COLOR["tip"], 0.55,
                               "tip"),
                 _mesh_payload("magnets", "Magnets 3 x 2 mm (as printed)", tms["magnets"], COLOR["magnets"], group="tip")]
        # readouts
        Dn = tms["dip_link"]
        new_pts = self.G.surface_points(Dn, 0.35)
        new_pts = new_pts[(new_pts[:, 1] > A.ARM_Y[0] + 1e-6) & (new_pts[:, 1] < A.ARM_Y[1] - 1e-6)]
        fixed = {"dip_link_new": new_pts}
        for k in ("clamp_plate", "board", "fasteners", "wires", "qwiic"):
            if k in tms:
                fixed[k] = self.G.surface_points(tms[k], 0.35)
        gap = L["gap"]
        chips = np.array([[A.X_BACK + gap, u, A.Z_C - v] for u, v in [(0, 0), (0, 7), (7, 0), (0, -7), (-7, 0.051)]])
        c2m = min(float(disc_distance(chips, m["centre_F"], m["x_F"][0], m["x_F"][1], mags["size_mm"][0] / 2).min())
                  for m in mags["items"])
        tmC = tms["clamp_plate"]
        inner = ((-11.0, A.ARM_Y[0] + 0.3, -95.0), (14.0, A.ARM_Y[1] - 0.3, -40.0))
        P = A.P
        sections = {"Window plate, x mid-plate": (Dn, (A.X_BACK + L["t_w"] / 2, 0, 0), (1, 0, 0), inner),
                    "Keys, z -70": (Dn, (0, 0, -70.0), (0, 0, 1), inner),
                    f"Shoulders, z {P['m2_z']:.2f}": (Dn, (0, 0, P["m2_z"]), (0, 0, 1), inner),
                    "Bridge, y -5": (Dn, (0, -5.0, 0), (0, 1, 0), inner), "Bridge, y 0": (Dn, (0, 0.0, 0), (0, 1, 0), inner),
                    "Bridge, y 12": (Dn, (0, 12.0, 0), (0, 1, 0), inner),
                    "Clamp plate, z -86.06": (tmC, (0, 0, A.Z_C - 7.9), (0, 0, 1), None),
                    f"Clamp plate, z {P['m2_z']:.2f}": (tmC, (0, 0, P["m2_z"]), (0, 0, 1), None),
                    f"Clamp plate, y {P['m2_y']:.2f}": (tmC, (0, P["m2_y"], 0), (0, 1, 0), None),
                    "Clamp plate, y 0": (tmC, (0, DP._A["qwiic"]["y"], 0), (0, 1, 0), None)}
        walls = self.walls(sections)
        vD, vC, vF = float(Dn.volume), float(tmC.volume), float(tms["fasteners"].volume)
        mass = vD * 0.9482e-3 + vC * 1.24e-3 + vF * 8.5e-3 + 0.64
        read = dict(gap=dict(value=round(gap, 3), unit="mm", label="Board-to-tip gap", limit="R1: at most 0.52 mm",
                             ok=bool(gap <= 0.52 + 1e-9),
                             note="from the tip's back face (x_F 6.45) to the chips' sensing plane, 0.42 mm under the "
                                  "package top"),
                    chip_to_magnet=dict(value=round(c2m, 2), unit="mm", label="Chip to magnet",
                                        note=f"least distance from a chip's sensing point to a magnet's surface, "
                                             f"{TIP_LABEL.get(tip, tip).lower()}"),
                    mass=dict(value=round(mass, 2), unit="g", label="Mass without the tip", limit="R12: at most 7.6 g",
                              ok=bool(mass <= 7.6),
                              note="DIP link at the links' sliced density (PLA, 15 % infill), clamp plate solid PLA, brass "
                                   "fasteners, board 0.64 g"))
        return parts, fixed, read, walls

    # concept b ---------------------------------------------------------------------------------------------------
    def build_b(self, vals, des, tip):
        B, M3 = self.B, self.M3
        DP.apply_b(vals, des)
        tip = tip if tip in B.TIPS else ("ell11_5" if "ell11_5" in B.TIPS else next(iter(B.TIPS)))
        self.tip = tip
        K = B.MeshB()
        body, dip = B.dip_link_parts(K)
        cover = B.cover_part(K)
        hw = B.hardware()
        pad = B.make_pad(tip)[0]
        items = B.tip_records()[tip]["magnets"]
        Mg = _magnets(M3, items["items"], items["size_mm"][0])
        if B.DESIGN.get("qwiic") == "header_on_tongue":
            Q = _boxes(M3, B.qwiic_header_b()[0])
            qlabel = "Qwiic header, plug, cable start, pigtail"
        elif B.DESIGN.get("qwiic") == "j1_kept":
            Q = hw["j1"]
            qlabel = "J1, plug and cable start"
        else:
            Q = B.box((B.X_SEAT, B.WIRE_SLOT["y"][0] + 0.2, B.WIRE_SLOT["z"][0] + 0.1),
                      (B.X_TONGUE_BACK + 0.6, B.WIRE_SLOT["y"][1] - 0.2, B.WIRE_SLOT["z"][1] - 0.1))
            qlabel = "Wires in the exit slot"
        tms = dict(dip_link=_tm(dip), body=_tm(body), cover=_tm(cover), board=_tm(hw["fr4"] + hw["board_parts"]),
                   fasteners=_tm(hw["screws"] + hw["nuts"] if "nuts" in hw else hw["screws"]), qwiic=_tm(Q), tip=_tm(pad),
                   magnets=_tm(Mg))
        parts = [_mesh_payload("dip_link", "DIP link (clevis arms of the STL, bridge, tongue, spine)", tms["dip_link"],
                               COLOR["dip_link"]),
                 _mesh_payload("cover", "Cover with rails", tms["cover"], COLOR["cover"]),
                 _mesh_payload("board", "Board: FR-4, chips, parts" + (" (whole tab)" if B.TAB_CUT_V <= -19.1 + 1e-9
                                                                        else " (tab cut)"), tms["board"], COLOR["board"],
                               group="board"),
                 _mesh_payload("qwiic", qlabel, tms["qwiic"], COLOR["qwiic"], group="board"),
                 _mesh_payload("fasteners", "Screws" + (" and nuts" if "nuts" in hw else "") + " (brass)", tms["fasteners"],
                               COLOR["fasteners"], group="hardware"),
                 _mesh_payload("tip", f"Pad, 85A: {TIP_LABEL.get(tip, tip)}", tms["tip"], COLOR["tip"], 0.55, "tip"),
                 _mesh_payload("magnets", "Magnets 3 x 2 mm", tms["magnets"], COLOR["magnets"], group="tip")]
        fixed = {k: self.G.surface_points(tms[k], 0.35) for k in ("body", "cover", "board", "fasteners", "qwiic", "tip")}
        d = B.chip_magnet_distances(tip)
        c2m = min(v["surface_mm"] for v in d.values())
        tb, tc = tms["body"], tms["cover"]
        zp, zd = B.Z_CONTACT + 8.1, B.Z_CONTACT - 7.9                     # the proximal and distal screws
        sections = {"Cover web, x mid-web": (tc, (B.X_SLOT + B.CHIP_H / 2, 0, 0), (1, 0, 0), None),
                    "Body, y 0": (tb, (0, 0.0, 0), (0, 1, 0), None),
                    f"Body, z {zp:.2f} (proximal screws)": (tb, (0, 0, zp), (0, 0, 1), None),
                    f"Body, z {zd:.2f} (distal screws)": (tb, (0, 0, zd), (0, 0, 1), None),
                    "Body, x mid-tongue": (tb, ((B.X_SEAT + B.X_TONGUE_BACK) / 2, 0, 0), (1, 0, 0),
                                           ((-11.0, -10.6, B.TIP["z_end"] - 0.3), (12.0, 10.6, B.TIP["z_mount"] - 0.1)))}
        if B.DESIGN.get("qwiic") == "header_on_tongue":
            sections[f"Body, z {B.P['qwiic_z_back'] - B.NECK + 2.4:.2f} (header pocket)"] = (
                tb, (0, 0, B.P["qwiic_z_back"] - B.NECK + 2.4), (0, 0, 1), None)
        walls = self.walls(sections)
        m = (float(tms["dip_link"].volume) * 0.948e-3 + float(tc.volume) * 1.24e-3 + float(tms["fasteners"].volume) * 8.5e-3
             + 0.64)
        read = dict(gap=dict(value=round(B.SPOT, 3), unit="mm", label="Board-to-pad gap", limit="R1: at most 0.52 mm",
                             ok=True, note=f"from the pad's back face (x_F {B.X_SLOT:.2f}) to the chips' sensing plane; in the "
                                           f"tip's frame the plane lies {B.X_SLOT + B.SPOT - 6.45:+.2f} mm from x_F 6.45"),
                    chip_to_magnet=dict(value=round(c2m, 2), unit="mm", label="Chip to magnet",
                                        note=f"least distance from a chip's sensing point to a magnet's surface, "
                                             f"{TIP_LABEL.get(tip, tip).lower()}"),
                    mass=dict(value=round(m, 2), unit="g", label="Mass without the pad", limit="R12: at most 7.6 g",
                              ok=bool(m <= 7.6), note="DIP link at the links' sliced density, cover solid PLA, brass "
                                                      "fasteners, board 0.64 g"))
        return parts, fixed, read, walls

    # checks ------------------------------------------------------------------------------------------------------
    def walls(self, sections):
        out = {}
        for k, (tm, o, n, b3) in sections.items():
            try:
                r = self.A.section_min_wall(tm, o, n, res=0.04, box3=b3)
            except Exception as e:                       # noqa: BLE001
                r = dict(error=str(e))
            out[k] = r
        vals = [(v["min_wall_mm"], k) for k, v in out.items() if v and "min_wall_mm" in v]
        least = min(vals) if vals else (None, None)
        return dict(least_mm=least[0], at=least[1], sections=out)

    def clearance_keepout(self, fixed):
        rows = {}
        for k, P in fixed.items():
            if not len(P):
                continue
            d = self.keepout.query(P)
            i = int(np.argmin(d))
            rows[k] = dict(min_mm=round(float(d[i]), 2), at_F=np.round(P[i], 2).tolist())
        k = min(rows, key=lambda q: rows[q]["min_mm"])
        return dict(least_mm=rows[k]["min_mm"], part=k, rows=rows)

    def near_points(self, fixed, reach=3.0):
        """The DIP-fixed sample points within reach of the swept parents (for the per-angle clearance)."""
        out, owner = [], []
        for k, P in fixed.items():
            if k == "tip":
                continue
            d = self.keepout.query(P)
            sel = d < reach
            out.append(P[sel])
            owner += [k] * int(sel.sum())
        return (np.vstack(out) if out else np.zeros((0, 3))), owner

    def clearance_at(self, q):
        X, owner = STATE.get("near", (np.zeros((0, 3)), []))
        if not len(X):
            return dict(q_deg=q, min_mm=None, note="no part within 3 mm of the MCP link or the PIP servo at any angle")
        Xp = self.G.rot_pip(X, -q)
        d, j = self.tree.query(Xp, k=1)
        ins = np.zeros(len(Xp), bool)
        for s in self.solids:
            ins |= s.contains(Xp)
        sd = np.where(ins, -d, d)
        i = int(np.argmin(sd))
        return dict(q_deg=q, min_mm=round(float(sd[i]), 2), part=owner[i],
                    parent=("mcp_link", "servo_pip")[int(self.parent_idx[j[i]])], at_F=np.round(X[i], 2).tolist())


GEO: Geometry | None = None


# ----------------------------------------------------------------------------------------------- requests
def list_files():
    out = []
    for f in sorted(DP.PARAMS.glob("*.json"), reverse=True):
        try:
            d = json.loads(f.read_text())
        except Exception:                                # noqa: BLE001
            continue
        if "parameters" in d:
            out.append(dict(file=f.name, concept=d.get("concept"), title=d.get("title", ""),
                            committed=bool(d.get("outputs")), saved=d.get("saved")))
    return out


def build(req):
    concept, vals, des, tip = req["concept"], {k: float(v) for k, v in req["values"].items()}, req.get("design", {}), req["tip"]
    probs = DP.problems(concept, vals, des)
    t0 = time.time()
    with LOCK:
        if concept == "a":
            parts, fixed, read, walls = GEO.build_a(vals, des, tip)
        else:
            parts, fixed, read, walls = GEO.build_b(vals, des, tip)
        tips, tip = GEO.tips_of(concept), GEO.tip
        t_build = time.time() - t0
        clear = GEO.clearance_keepout({k: v for k, v in fixed.items() if k != "tip"})
        STATE["near"] = GEO.near_points(fixed)
        STATE["concept"] = concept
        at = GEO.clearance_at(float(req.get("q", 0.0)))
    read["clearance"] = dict(value=clear["least_mm"], unit="mm", label="Least clearance over PIP -18..92 deg",
                             limit="R7: 0.5 mm rigid, 1.0 mm cable", ok=bool(clear["least_mm"] >= 0.5),
                             note=f"keep-out estimate (keepout_pip.npz, 0.25 mm cells, within 0.3 mm; it reads about 0.13 mm "
                                  f"under the exact check), at the {clear['part'].replace('_', ' ')}", rows=clear["rows"])
    read["wall"] = dict(value=walls["least_mm"], unit="mm", label="Least wall", limit="R11: 0.9 mm (0.8 mm two lines)",
                        ok=bool(walls["least_mm"] is not None and walls["least_mm"] >= 0.8 - 1e-9),
                        note=f"in {walls['at']}", sections=walls["sections"])
    return dict(parts=parts, readouts=read, angle=at, problems=probs, seconds=round(time.time() - t0, 2),
                build_seconds=round(t_build, 2), tips=tips, tip=tip, tip_labels={k: TIP_LABEL.get(k, k) for k in tips})


def context():
    parts = [_mesh_payload("mcp_link", "MCP link", GEO.context["mcp_link"], COLOR["mcp_link"], 0.35, "context"),
             _mesh_payload("servo_pip", "PIP servo", GEO.context["servo_pip"], COLOR["servo_pip"], 0.85, "context")]
    return dict(parts=parts, pip_axis_point=[0.0, 0.0, -51.55], pip_range=[-18.0, 92.0], tips=TIPS_A, tip_labels=TIP_LABEL)


def save(req):
    src = DP.PARAMS / Path(req["file"]).name
    doc = json.loads(src.read_text())
    name = re.sub(r"[^a-z0-9_]+", "_", req.get("name", "").lower()).strip("_")[:40] or "set"
    for q in doc["parameters"]:
        if q["key"] in req["values"]:
            q["value"] = round(float(req["values"][q["key"]]), 4)
    for d in doc.get("design", []):
        if d["key"] in req.get("design", {}):
            d["value"] = req["design"][d["key"]]
    doc.pop("outputs", None)
    doc["derived_from"] = src.name
    doc["saved"] = time.strftime("%Y-%m-%d %H:%M")
    doc["title"] = f"{doc['title']} ({name})" if "(" not in doc["title"][-30:] else doc["title"]
    changed = {q["key"]: [q["default"], q["value"]] for q in doc["parameters"] if abs(q["value"] - q["default"]) > 1e-9}
    doc["changed_from_default"] = changed
    dst = DP.PARAMS / f"{time.strftime('%Y%m%d-%H%M')}-concept_{doc['concept']}-{name}.json"
    dst.write_text(json.dumps(doc, indent=1) + "\n")
    return dict(file=dst.name, path=str(dst.relative_to(REPO)), changed=changed,
                problems=DP.problems(doc["concept"], {q["key"]: float(q["value"]) for q in doc["parameters"]}))


def start_job(kind, file):
    path = DP.PARAMS / Path(file).name
    if not path.exists():
        raise FileNotFoundError(file)
    jid = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{path.stem}"
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log = JOBS_DIR / f"{jid}.log"
    cmd = [str(DP.RESGUARD), "run", "--mem", "3G", "--cpu", "400", "--", "taskset", "-c", "8-11", "env",
           "NUMPY_MADVISE_HUGEPAGE=0", "OMP_NUM_THREADS=1", str(DP.E_FEMTIP), "-u", str(REPO / "scripts/dip_link_params.py"),
           kind, "--params", str(path)]
    fh = open(log, "w")
    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    JOBS[jid] = dict(kind=kind, file=path.name, log=str(log), proc=proc, started=time.strftime("%H:%M:%S"))
    return dict(job=jid, log=str(log.relative_to(REPO)))


def job_status(jid):
    j = JOBS[jid]
    rc = j["proc"].poll()
    tail = Path(j["log"]).read_text()[-4000:] if Path(j["log"]).exists() else ""
    out = dict(job=jid, kind=j["kind"], file=j["file"], started=j["started"], running=rc is None, returncode=rc, log_tail=tail)
    m = re.findall(r"RESULT (\S+\.json)", tail)
    if rc is not None and m:
        res = REPO / m[-1]
        if res.exists():
            out["result_file"] = str(res.relative_to(REPO))
            out["result"] = json.loads(res.read_text())
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):            # quiet: one line per API call
        if "/api/" in self.path:
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {self.path.split('?')[0]}\n")

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/files":
                return self._send(200, list_files())
            if u.path == "/api/params":
                return self._send(200, json.loads((DP.PARAMS / Path(q["file"]).name).read_text()))
            if u.path == "/api/context":
                return self._send(200, context())
            if u.path == "/api/angle":
                with LOCK:
                    return self._send(200, GEO.clearance_at(float(q["q"])))
            if u.path == "/api/job":
                return self._send(200, job_status(q["id"]))
            return self._send(404, dict(error="not found"))
        except Exception as e:                           # noqa: BLE001
            traceback.print_exc()
            return self._send(500, dict(error=f"{type(e).__name__}: {e}"))

    def do_POST(self):
        u = urlparse(self.path)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if u.path == "/api/build":
                return self._send(200, build(req))
            if u.path == "/api/save":
                return self._send(200, save(req))
            if u.path in ("/api/check", "/api/export"):
                return self._send(200, start_job(u.path.rsplit("/", 1)[1], req["file"]))
            return self._send(404, dict(error="not found"))
        except Exception as e:                           # noqa: BLE001
            traceback.print_exc()
            return self._send(500, dict(error=f"{type(e).__name__}: {e}"))


def main():
    global GEO
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8780)
    a = ap.parse_args()
    t0 = time.time()
    GEO = Geometry()
    print(f"geometry loaded in {time.time() - t0:.1f} s", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"DIP-link viewer: http://127.0.0.1:{a.port}/  (Ctrl-C stops it)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
