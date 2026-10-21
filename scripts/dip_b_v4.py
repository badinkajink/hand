#!/usr/bin/env python3
"""DIP link concept b v4 and the wall-height sweep of 2026-10-21 16:20.

Owner, 2026-10-21 16:20, after the second revision's covers: "the 0.15mm inset is still perfectly fine. this is pretty locked
in there now. so now, let's try to lower the heights on both the top cover and the cassette side walls. let's print a few
out and see how low we can go while maintaining this retaining"; "there's no requirement for the dip link to have a long
continuous edge. it just serves as a hard stop for the servo. but it also prevents large objects from contacting the
fingertip--it'll hit the edge first. maybe we can recess/sculpt this edge of the dip link the same way we did for the column
under the fingertip, while making sure our servo and clevis connections are solid"; then "honestly i think we can make the
underlying column even smaller radius too. then the edge detections / bottom face can be usable too".

v4 is built by the concept b pipeline from params/20261021-v4_concept_b.json (design bridge = recessed, col_margin 3.0,
col_side_margin 1.45, the cover I15):

    DLV2_PARAMS_B=docs/experiments/20261018-dip_link_board_mount/params/20261021-v4_concept_b.json \\
        ~/miniconda3/envs/femtip/bin/python scripts/dip_link_v2.py b build check step

Commands of this script (frame F; outputs under docs/experiments/20261018-dip_link_board_mount/v4/b/):
  link     v3 against v4: the first contact of the MCP link and the PIP servo past both ends of the PIP range, by region of
           the link; the lead of the pad over the rigid parts for a flat object approaching from the palmar side, tilted
           toward the proximal side or across the finger; the neck's section; the least walls; mass; v4 inside v3 (the UR5e
           mount's leg molds) -> 20261021-dip_link_v4b_checks.json
  heights  covers I15 with walls 4, 3 and 2 mm tall and caps with solid side blocks 4, 3 and 2 mm tall at 0.15 mm
           interference, their checks and print STLs -> 20261021-dip_link_v4b_heights.json
  step     the covers and caps as B-rep in freecadcmd (without the engraved IDs): cad/*.step
  figures  media/20261021-dip_link_b_v3_v4_palmar.png, _flat_object.png, _heights.png

    unset VIRTUAL_ENV PYTHONPATH
    ~/.claude/bin/resguard.sh run --mem 4G --cpu 200 -- taskset -c 14,15 env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python -u scripts/dip_b_v4.py link heights step figures
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
DIP = REPO / "docs/experiments/20261018-dip_link_board_mount"
PARAMS4 = DIP / "params/20261021-v4_concept_b.json"
os.environ["DIPB_PARAMS"] = str(PARAMS4)
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import dip_link_concept_b as B  # noqa: E402
import dip_b_tip_cap as TC  # noqa: E402

V4 = DIP / "v4/b"
PFX = "20261021-dip_link_v4b"
WORK = REPO / "logs/20261021-dip_link/v4/b/rev3"
CHECKS = V4 / f"{PFX}_checks.json"
HEIGHTS = V4 / f"{PFX}_heights.json"
LINKS = {"v3": DIP / "v3/b/cad/20261020-dip_link_v3b_dip_link.stl", "v4": V4 / f"cad/{PFX}_dip_link.stl"}
INTERFERENCE = 0.15                 # mm per side, the owner's I15 ("perfectly fine", 2026-10-21)
COVER_H = (4.0, 3.0, 2.0)           # wall heights of the sweep; I15 itself is 6.0
CAP_H = (4.0, 3.0, 2.0)             # side-block heights of the sweep; TW2 is 5.0 at 0.05 mm
SEAT_MAX = 3.0                      # the proximal heads' seat: 3 mm up (M2 x 8), or on the wall's top when it is lower
M2 = (6.0, 8.0, 10.0, 12.0)         # M2 lengths in the owner's kits (M2 x 7 and x 9 are rare)
ENGRAVE = dict(depth=0.3, cap_cover=2.2, cap_cap=4.0)
RHO_PLA, RHO_LINK = 1.24, 0.948     # g/cm3: solid, and the DIP link's sliced density (dip_link_concept_b.RHO)


# ------------------------------------------------------------------------------------------------ the link
def surface(path, spacing):
    import trimesh
    import dip_link_geometry as G
    return G.surface_points(trimesh.load(path, force="mesh"), spacing)


def link_region(p):
    x, y, z = p
    if y > B.Y_BRIDGE[1]:
        return "slotted arm"
    if y < B.Y_BRIDGE[0]:
        return "horn arm"
    return "bridge, palmar of x_F -6" if x < -6.0 else "bridge"


def stops(path, q_flex=(88.0, 112.0), q_ext=(-36.0, -14.0), step=0.5, touch=0.05):
    """First PIP angle past each end of the range at which each region of the link comes within `touch` of the MCP link or
    the PIP servo (their surfaces at 0.25 mm, the link's at 0.3 mm), and the least distance at the range's ends. The arms'
    inner faces, which run beside the MCP link's clevis 0.1 mm off at every angle, are left out (a band of 0.2 mm)."""
    import numpy as np
    from scipy.spatial import cKDTree
    import dip_link_geometry as G
    P = surface(path, 0.3)
    keep = (P[:, 2] > -71.0) & ~((P[:, 1] > 13.30) & (P[:, 1] < 13.50)) & ~((P[:, 1] > -17.15) & (P[:, 1] < -16.95))
    P = P[keep]
    reg = np.array([link_region(p) for p in P])
    Q0, idx = G.parent_samples()
    tree = cKDTree(Q0)
    out = {}
    for name, (lo, hi), sgn in (("flexion", q_flex, 1), ("extension", q_ext, -1)):
        first, at_end = {}, {}
        qs = np.arange(lo, hi + 1e-9, step)[::sgn]
        end = 92.0 if sgn > 0 else -18.0
        for q in qs:
            d, j = tree.query(G.rot_pip(P, -q), k=1, distance_upper_bound=3.0)
            for r in np.unique(reg):
                s = (reg == r) & np.isfinite(d)
                if not s.any():
                    continue
                k = np.flatnonzero(s)[np.argmin(d[s])]
                if abs(q - end) < 1e-9:
                    at_end[r] = round(float(d[k]), 3)
                if d[k] < touch and r not in first:
                    first[r] = dict(q_deg=float(q), at_F=np.round(P[k], 2).tolist(),
                                    part=("MCP link", "PIP servo")[int(idx[j[k]])])
        out[name] = dict(first_contact=dict(sorted(first.items(), key=lambda kv: sgn * kv[1]["q_deg"])),
                         range_end_q_deg=end, least_mm_at_range_end=at_end)
    return out


def support(V, u):
    import numpy as np
    return float(np.max(V @ u))


def lead_curves(links, pad, cover, joint0, tilts):
    """The pad's lead over the rigid parts (mm) for a flat object approaching from the palmar side: the object's plane
    first meets the point of largest u.p, u its outward normal seen from the finger; lead = h_pad(u) - h_rigid(u). u tilts
    toward the proximal side (u_z > 0) or across the finger (u_y)."""
    import numpy as np
    out = {}
    for key, Vl in links.items():
        rows = {"proximal_tilt": [], "side_tilt": []}
        for kind in rows:
            for t in tilts:
                a = math.radians(t)
                u = np.array([-math.cos(a), 0.0, math.sin(a)]) if kind == "proximal_tilt" else np.array(
                    [-math.cos(a), math.sin(a), 0.0])
                hp = support(pad, u)
                rows[kind].append(dict(tilt_deg=float(t), link=round(hp - support(Vl, u), 3),
                                       link_cover=round(hp - max(support(Vl, u), support(cover, u)), 3),
                                       with_joint_pip0=round(hp - max(support(Vl, u), support(cover, u), support(joint0, u)), 3)))
        out[key] = rows
    return out


def largest_tilt(rows, col, need):
    """The largest tilt from 0 up to which the lead stays at or above `need` (on the positive side)."""
    best = None
    for r in rows:
        if r["tilt_deg"] < 0:
            continue
        if r[col] < need:
            break
        best = r["tilt_deg"]
    return best


def section_props(path, z):
    """Area (mm2) and second moments (mm4) of a link's section at z_F, about its centroid: I_y for bending by a palmar load
    (about the PIP axis' direction), I_x across the finger."""
    import numpy as np
    import trimesh
    m = trimesh.load(path, force="mesh")
    sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    p2, _ = sec.to_planar(to_2D=np.eye(4))                       # x, y as they are (normal +z)
    A = Sx = Sy = Ixx = Iyy = 0.0
    for poly in p2.polygons_full:
        for ring, s in [(np.asarray(poly.exterior.coords), 1.0)] + [(np.asarray(h.coords), -1.0) for h in poly.interiors]:
            x, y = ring[:, 0], ring[:, 1]
            cr = x[:-1] * y[1:] - x[1:] * y[:-1]
            w = s * np.sign(cr.sum())                              # each ring counter-clockwise, holes subtracted
            A += w * 0.5 * cr.sum()
            Sx += w * np.sum((x[:-1] + x[1:]) * cr) / 6.0
            Sy += w * np.sum((y[:-1] + y[1:]) * cr) / 6.0
            Ixx += w * np.sum((x[:-1] ** 2 + x[:-1] * x[1:] + x[1:] ** 2) * cr) / 12.0
            Iyy += w * np.sum((y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2) * cr) / 12.0
    xc, yc = Sx / A, Sy / A
    # Ixx here is the integral of x^2 (bending about the y axis), Iyy of y^2 (about the x axis)
    return dict(z_F=z, area_mm2=round(A, 1), centroid_x_F=round(xc, 2), centroid_y_F=round(yc, 2),
                I_about_y_mm4=round(Ixx - A * xc ** 2, 0), I_about_x_mm4=round(Iyy - A * yc ** 2, 0),
                x_range_F=[round(float(p2.bounds[0][0]), 2), round(float(p2.bounds[1][0]), 2)])


def cmd_link():
    """v3 against v4 -> CHECKS."""
    import numpy as np
    import trimesh
    import dip_link_geometry as G
    import dip_link_v2 as L2
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    rec = dict(owner=__doc__.split("\n\n")[1].replace("\n", " "), params=str(PARAMS4.relative_to(REPO)),
               design=dict(bridge=B.DESIGN.get("bridge"), col_margin_mm=B.rp("col_margin"),
                           col_side_margin_mm=float(B.P.get("col_side_margin", B.rp("col_margin"))),
                           bridge_chamfer_mm=B.rp("bridge_chamfer"), bridge_x_palmar_F=round(B.bridge_x_palmar(), 3)))
    k = math.sqrt(1.0 - ((B.TIP["z_mount"] - B.Z_CONTACT) / B.ELL["azp"]) ** 2)
    ay = B.ELL["ay"] * k - float(B.P.get("col_side_margin", B.rp("col_margin")))
    ax = B.ELL["ax"] * k - B.rp("col_margin")
    rec["column"] = dict(v3=dict(half_width_mm=round(B.ELL["ay"] * k - 0.6, 3), palmar_x_F=round(-2.05 - (B.ELL["ax"] * k - 0.6), 3)),
                         v4=dict(half_width_mm=round(ay, 3), palmar_x_F=round(-2.05 - ax, 3)),
                         pad_mount_face=dict(half_width_mm=round(B.ELL["ay"] * k, 3), palmar_x_F=round(-2.05 - B.ELL["ax"] * k, 3)),
                         j1_slot_y_F=[-3.85, 3.95])
    rec["column"]["v4"]["pad_overhang_palmar_mm"] = round(rec["column"]["v4"]["palmar_x_F"] - rec["column"]["pad_mount_face"]["palmar_x_F"], 3)
    rec["column"]["v4"]["pad_overhang_side_mm"] = round(rec["column"]["pad_mount_face"]["half_width_mm"] - ay, 3)
    rec["column"]["v4"]["pla_beside_j1_mm"] = round(ay - 3.95, 3)
    # volumes and mass, v4 inside v3 (the UR5e mount's leg molds were cut around v2's and v3's legs)
    mans = {kk: B.to_manifold(trimesh.load(p, force="mesh")) for kk, p in LINKS.items()}
    rec["volume_mm3"] = {kk: round(float(m.volume()), 1) for kk, m in mans.items()}
    rec["mass_g"] = {kk: dict(solid=round(v * RHO_PLA / 1e3, 2), sliced=round(v * RHO_LINK / 1e3, 2)) for kk, v in rec["volume_mm3"].items()}
    rec["v4_outside_v3_mm3"] = round(float((mans["v4"] - mans["v3"]).volume()), 3)
    rec["removed_from_v3_mm3"] = round(float((mans["v3"] - mans["v4"]).volume()), 1)
    # the bridge's top at its least radius about the PIP axis: the UR5e mount's inner block (R 13.40) rests there
    for kk, p in LINKS.items():
        P = surface(p, 0.25)
        s = (P[:, 1] > -16.9) & (P[:, 1] < 13.3) & (P[:, 2] > -66.0) & (P[:, 2] < -60.0) & (P[:, 0] > -12) & (P[:, 0] < 9.5)
        r = np.hypot(P[s, 0] - G.PIP_C[0], P[s, 2] - G.PIP_C[2])
        rec.setdefault("bridge_top_least_radius_mm", {})[kk] = round(float(r.min()), 3)
    print("volumes", rec["volume_mm3"], "v4 outside v3", rec["v4_outside_v3_mm3"], rec["bridge_top_least_radius_mm"], flush=True)
    # sections of the neck between the pad's mount face and the bridge
    rec["neck_sections"] = {kk: [section_props(p, z) for z in (-72.30, -76.0)] for kk, p in LINKS.items()}
    print("neck", json.dumps(rec["neck_sections"]), flush=True)
    # walls (the v2 pipeline's sections, on v4's new body and cover)
    import dip_link_concept_a as A
    body = B.load_cad("dip_link_new_body")
    zj = -72.30
    neck = ((B.TIP["x_palmar"] - 0.1, -B.Y_RIM - 0.1, B.TIP["z_mount"]), (B.X_SPINE_BACK + 0.1, B.Y_RIM + 0.1, -60.0))
    rec["walls"] = {f"neck_z{zj:.2f}_around_J1": A.section_min_wall(body, (0, 0, zj), (0, 0, 1), box3=neck),
                    "neck_y0": A.section_min_wall(body, (0, 0.0, 0), (0, 1, 0), box3=neck)}
    print("walls", rec["walls"], flush=True)
    # clearance over the PIP range (the pipeline's exact sweep), and the stops past its ends
    clr = json.loads((REPO / "logs/20261021-dip_link/v4/b/clearance.json").read_text())
    clr3 = json.loads((REPO / "logs/20261020-dip_link/v3/b/clearance.json").read_text())
    rec["clearance_pip_range"] = {kk: {p: c[p]["min_mm"] for p in ("dip_link_new_body", "cover", "arms_kept", "qwiic_cable")}
                                  for kk, c in (("v3", clr3), ("v4", clr))}
    rec["stops"] = {}
    for kk, p in LINKS.items():
        rec["stops"][kk] = stops(p)
        print("stops", kk, json.dumps(rec["stops"][kk]["flexion"]["first_contact"]), json.dumps(rec["stops"][kk]["extension"]["first_contact"]),
              flush=True)
    # the flat object
    pad = np.asarray(trimesh.load(V4 / f"cad/{PFX}_pad_ell11_5.stl", force="mesh").vertices)
    cover = np.asarray(trimesh.load(V4 / f"cad/{PFX}_cover.stl", force="mesh").vertices)
    joint0 = np.vstack([np.asarray(G.load_part(n).vertices) for n in ("mcp_link", "servo_pip")])
    links = {kk: np.asarray(trimesh.load(p, force="mesh").vertices) for kk, p in LINKS.items()}
    tilts = np.arange(-30.0, 30.01, 1.0)
    rec["flat_object"] = dict(curves=lead_curves(links, pad, cover, joint0, tilts),
                              note="lead = how far the pad stands out of the rigid parts toward a flat object; the pad's FE approach "
                                   "at a 3 N contact is 0.154 mm (concept_b/fe/fem_coarse.jsonl, five magnets, 6.45 mm behind)")
    summ = {}
    for kk in links:
        c = rec["flat_object"]["curves"][kk]
        r0 = next(r for r in c["proximal_tilt"] if r["tilt_deg"] == 0.0)
        summ[kk] = dict(lead_head_on_mm=r0["link"], lead_head_on_with_joint_mm=r0["with_joint_pip0"],
                        proximal_tilt_lead_ge_1mm_deg=largest_tilt(c["proximal_tilt"], "link_cover", 1.0),
                        proximal_tilt_lead_ge_0_deg=largest_tilt(c["proximal_tilt"], "link_cover", 0.0),
                        proximal_tilt_with_joint_ge_0_deg=largest_tilt(c["proximal_tilt"], "with_joint_pip0", 0.0),
                        side_tilt_lead_ge_1mm_deg=largest_tilt(c["side_tilt"], "link_cover", 1.0),
                        side_tilt_lead_ge_0_deg=largest_tilt(c["side_tilt"], "link_cover", 0.0),
                        side_tilt_with_joint_ge_0_deg=largest_tilt(c["side_tilt"], "with_joint_pip0", 0.0))
        # the other side across the finger
        neg = [dict(r, tilt_deg=-r["tilt_deg"]) for r in c["side_tilt"] if r["tilt_deg"] <= 0][::-1]
        neg.sort(key=lambda r: r["tilt_deg"])
        summ[kk]["side_tilt_other_side_lead_ge_0_deg"] = largest_tilt(neg, "link_cover", 0.0)
    rec["flat_object"]["summary"] = summ
    print("flat object", json.dumps(summ), flush=True)
    # print STL of the link for the P1S (the v3 link's orientation and settings)
    import dip_link_print_project as PP
    (V4 / "print").mkdir(parents=True, exist_ok=True)
    rec["bambu"] = L2.bambu_stl(trimesh.load(LINKS["v4"], force="mesh"), PP.R_LINK, V4 / f"print/{PFX}_dip_link_pla_bambu.stl")
    problems = []
    if rec["v4_outside_v3_mm3"] > 0.05:
        problems.append(f"v4 outside v3 by {rec['v4_outside_v3_mm3']} mm3")
    if rec["clearance_pip_range"]["v4"]["dip_link_new_body"] < 0.5:
        problems.append(f"clearance {rec['clearance_pip_range']['v4']}")
    if rec["column"]["v4"]["pla_beside_j1_mm"] < 0.8:
        problems.append("PLA beside J1 under 0.8 mm")
    fl3, fl4 = (rec["stops"][kk]["flexion"]["first_contact"] for kk in ("v3", "v4"))
    if min(v["q_deg"] for v in fl4.values()) != min(v["q_deg"] for v in fl3.values()):
        problems.append("the flexion stop moved")
    rec.update(problems=problems, when=time.strftime("%Y-%m-%d %H:%M"), script="scripts/dip_b_v4.py link",
               seconds=round(time.time() - t0, 1))
    CHECKS.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print("problems:", problems, f"[{time.time() - t0:.0f} s]", flush=True)


# ------------------------------------------------------------------------------------------------ the height sweep
def screw_len(head_up):
    """The longest M2 of the kit whose tip stays inside the tongue (x_F <= 7.90, 0.1 mm short of its back) with the head
    `head_up` above the cover's front face; and the tip's x_F."""
    L = max(l for l in M2 if B.X_SLOT - head_up + l <= B.X_SLOT + B.SCREW_L + 1e-6)
    return L, round(B.X_SLOT - head_up + L, 3)


def engrave(man, text, cap, at, normal, up, depth):
    """`text` engraved `depth` deep into a face of `man` whose outward normal is `normal` (a unit axis of F), centred on the
    face point `at`, its letters' up along `up`, read from outside the face."""
    import numpy as np
    import dip_fit_print_plates as D
    m3 = B._m3()
    n, v = np.asarray(normal, float), np.asarray(up, float)
    r = np.cross(v, n)                                  # right, for a reader looking at the face against its normal
    cs = D.text_cs(text, cap, 0.0, 0.0)
    solid = m3.Manifold.extrude(cs, depth + 0.05).translate([0.0, 0.0, -depth])     # w from -depth to +0.05
    T = np.column_stack([r, v, n, np.asarray(at, float)])                          # (u, v, w) -> F; w along the normal
    if np.linalg.det(T[:, :3]) < 0:
        raise ValueError("left-handed frame")
    return man - solid.transform(T)


def cover_h(K, h, engraved=True):
    """Cover I15 with solid walls h mm tall; the proximal heads on solid wall min(h, SEAT_MAX) up; the wall height engraved
    in each wall's top over its distal part."""
    sr = min(h, SEAT_MAX)
    with TC.b_state(design=dict(cover_walls="solid"), params=dict(wall_clear=-INTERFERENCE, wall_h=h, seat_raise=sr)):
        cov = B.cover_part(K)
    if engraved:
        xt = B.X_SLOT - h
        for s in (-1, 1):
            cov = engrave(cov, f"{h:g}", ENGRAVE["cap_cover"], (xt, s * 8.35, -84.0), (-1, 0, 0), (0, 0, -1), ENGRAVE["depth"])
    return cov, sr


class cap_dims:
    """CAP2's side-block height and gap set for a sweep cap, then restored."""

    def __init__(self, H, clear):
        self.new = dict(block_h=H, block_clear=clear)

    def __enter__(self):
        self.old = {k: TC.CAP2[k] for k in self.new}
        TC.CAP2.update(self.new)

    def __exit__(self, *exc):
        TC.CAP2.update(self.old)
        return False


def cap_h(K, H, engraved=True):
    """A cap with TW2's solid side blocks H mm tall at INTERFERENCE, TW2's plate, chamfers and pin; the height engraved on the
    end plate's outer face."""
    with cap_dims(H, -INTERFERENCE):
        cap = TC.cap_part2(K, dict(side="blocks"))
    if engraved:
        ze = TC.ZE - TC.CAP2["t_end"]
        cap = engrave(cap, f"{H:g}", ENGRAVE["cap_cap"], (4.3, 0.0, ze), (0, 0, -1), (1, 0, 0), ENGRAVE["depth"])
    return cap


def screws_sweep(seat, head_cap):
    """The four board screws: the proximal pair on seats `seat` up, the distal pair on the cap's blocks `head_cap` up, each
    of its kit length."""
    m3 = B._m3()
    out = []
    for y, z in B.screw_yz():
        up = seat if z > B.Z_CONTACT else head_cap
        L, _ = screw_len(up)
        xh = B.X_SLOT - up
        out.append(B.cyl_x(y, z, B.HEAD_D / 2, xh - B.HEAD_H, xh) + B.cyl_x(y, z, B.SCREW_D / 2, xh - 0.01, xh + L))
    return m3.Manifold.batch_boolean(out, m3.OpType.Add)


def cmd_heights():
    import numpy as np
    import dip_link_print_project as PP
    import dip_link_v2 as L2
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    (V4 / "cad").mkdir(parents=True, exist_ok=True)
    (V4 / "print").mkdir(parents=True, exist_ok=True)
    K = B.MeshB()
    hw = B.hardware()
    link = B.to_manifold(B.load_cad("dip_link"))
    board = hw["fr4"] + hw["board_parts"]
    mags = hw["magnets_ell11_5"]
    pad_r = TC.pad_rev2()
    rec = dict(owner="2026-10-21 16:20: 'the 0.15mm inset is still perfectly fine. this is pretty locked in there now. so now, let's "
                     "try to lower the heights on both the top cover and the cassette side walls. let's print a few out and see how "
                     "low we can go while maintaining this retaining'", interference_mm=INTERFERENCE, covers={}, caps={},
               reference=dict(cover_I15=dict(wall_h_mm=6.0, seat_mm=3.0), cap_TW2=dict(block_h_mm=5.0, interference_mm=0.05)),
               pad_flank_straight_mm=round(B.X_SLOT - (B.TIP["x_palmar"] + B.ELL["ax"]), 3),
               engraving=dict(ENGRAVE, note="the height in mm, on each wall's top (covers) or the end plate's outer face (caps)"))
    for h in COVER_H:
        cid = f"I15H{h:g}"
        cov, sr = cover_h(K, h)
        stl = V4 / "cad" / f"{PFX}_cover_{cid}.stl"
        rt = TC.round_trip(cov, stl)
        scr = screws_sweep(sr, 3.0)
        tm = B.to_trimesh(cov)
        bam = L2.bambu_stl(tm, PP.R_CLAMP, V4 / "print" / f"{PFX}_cover_{cid}_pla_bambu.stl")
        Lp, tip = screw_len(sr)
        prox = B._m3().Manifold.batch_boolean([B.cyl_x(y, z, B.HEAD_D / 2, B.X_SLOT - sr - B.HEAD_H, B.X_SLOT - sr)
                                               for y, z in B.screw_yz() if z > B.Z_CONTACT], B._m3().OpType.Add)
        bare = cover_h(K, h, engraved=False)[0]
        r = dict(wall_h_mm=h, seat_mm=sr, proximal_screws=f"M2 x {Lp:g}", proximal_tip_x_F=tip,
                 thread_in_tongue_mm=round(tip - B.X_SEAT, 2), stl=PP.rel(stl), bambu=bam, round_trip=rt,
                 volume_mm3=rt["volume_mm3"], mass_solid_g=round(rt["volume_mm3"] * RHO_PLA / 1e3, 3),
                 width_mm=round(float(tm.bounds[1][1] - tm.bounds[0][1]), 3),
                 engraving_mm3=round(float(bare.volume() - cov.volume()), 3),
                 interference_with_B00R_mm3=round(float((cov ^ pad_r).volume()), 3),
                 overlap_mm3=dict(dip_link_v4=round(float((cov ^ link).volume()), 4), board=round(float((cov ^ board).volume()), 4),
                                  screws=round(float((cov ^ scr).volume()), 4)),
                 heads_to_B00R_mm=round(float(prox.min_gap(pad_r, 3.0)), 3), heads_overlap_B00R_mm3=round(float((prox ^ pad_r).volume()), 4),
                 flank_held_mm=round(min(h, rec["pad_flank_straight_mm"]), 2))
        rec["covers"][cid] = r
        print(cid, json.dumps(r, default=float), flush=True)
    cov6, _ = cover_h(K, 6.0, engraved=False)
    for H in CAP_H:
        vid = f"TW15H{H:g}"
        cap = cap_h(K, H)
        bare = cap_h(K, H, engraved=False)
        stl = V4 / "cad" / f"{PFX}_tip_cap_{vid}.stl"
        rt = TC.round_trip(cap, stl)
        scr = screws_sweep(3.0, H)
        Ld, tip = screw_len(H)
        others = dict(cover_I15=cov6, dip_link_v4=link, board=board, magnets=mags, screws=scr)
        ov = {k: round(float((cap ^ o).volume()), 4) for k, o in others.items()}
        ov["pad_B00R"] = round(float((cap ^ pad_r).volume()), 3)
        side = (cap ^ B.box((-20, -20, TC.ZE + 0.3), (20, 20, -80.0))) - B.box((TC.CAP["pin_x"] - 1.6, -1.6, TC.ZE),
                                                                              (TC.CAP["pin_x"] + 1.6, 1.6, TC.ZE + TC.CAP["pin_l"] + 0.5))
        tm = B.to_trimesh(cap)
        pts = B.sample_part(tm)
        v_c, i_c, q_c, part_c = B.clearance_exact(pts, reach=1.6, workers=2)
        bam = L2.bambu_stl(tm, np.eye(3), V4 / "print" / f"{PFX}_tip_cap_{vid}_pla_bambu.stl")
        r = dict(block_h_mm=H, interference_mm=INTERFERENCE, block_proximal_end_z_F=TC.wall_z0(), distal_screws=f"M2 x {Ld:g}",
                 distal_tip_x_F=tip, thread_in_tongue_mm=round(tip - B.X_SEAT, 2), stl=PP.rel(stl), bambu=bam, round_trip=rt,
                 volume_mm3=rt["volume_mm3"], mass_solid_g=round(rt["volume_mm3"] * RHO_PLA / 1e3, 3),
                 width_mm=round(float(tm.bounds[1][1] - tm.bounds[0][1]), 3), engraving_mm3=round(float(bare.volume() - cap.volume()), 3),
                 overlap_mm3=ov, interference_sides_B00R_mm3=round(float((side ^ pad_r).volume()), 3),
                 pip_sweep=dict(min_mm=round(v_c, 3), exact=q_c is not None, pip_deg=q_c, part=part_c, reach_mm=1.6),
                 flank_held_mm=round(min(H, rec["pad_flank_straight_mm"]), 2))
        rec["caps"][vid] = r
        print(vid, json.dumps(r, default=float), flush=True)
    problems = []
    for grp in ("covers", "caps"):
        for k, r in rec[grp].items():
            if r["round_trip"]["reread_mm3"] < 0.999 * r["round_trip"]["volume_mm3"] or not r["round_trip"]["watertight"]:
                problems.append(f"{k}: STL round trip {r['round_trip']}")
            bad = {q: x for q, x in r["overlap_mm3"].items() if not q.startswith("pad_") and x > 0.05}
            if bad:
                problems.append(f"{k}: overlaps {bad}")
            if r["width_mm"] > 21.0 + 1e-6:
                problems.append(f"{k}: width {r['width_mm']}")
            if r.get("pip_sweep", {}).get("min_mm", 9.0) < 0.5:
                problems.append(f"{k}: PIP sweep {r['pip_sweep']}")
            if r.get("heads_overlap_B00R_mm3", 0.0) > 0:
                problems.append(f"{k}: heads overlap B00R")
            if r["thread_in_tongue_mm"] < 2.4:
                problems.append(f"{k}: {r['thread_in_tongue_mm']} mm of thread")
    rec.update(problems=problems, when=time.strftime("%Y-%m-%d %H:%M"), script="scripts/dip_b_v4.py heights",
               seconds=round(time.time() - t0, 1))
    old = json.loads(HEIGHTS.read_text()) if HEIGHTS.exists() else {}
    rec["step"] = old.get("step")
    HEIGHTS.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print("problems:", problems, f"[{time.time() - t0:.0f} s]", flush=True)


# ------------------------------------------------------------------------------------------------ STEP
def fc_main():
    """Inside freecadcmd: the sweep's covers and caps as B-rep (no engraving), STEP into v4/b/cad."""
    K = B.BrepB()
    log = {}
    parts = [(f"I15H{h:g}", lambda h=h: cover_h(K, h, engraved=False)[0], f"{PFX}_cover_I15H{h:g}") for h in COVER_H]
    parts += [(f"TW15H{H:g}", lambda H=H: cap_h(K, H, engraved=False), f"{PFX}_tip_cap_TW15H{H:g}") for H in CAP_H]
    for pid, build, name in parts:
        s = build()
        step = V4 / "cad" / f"{name}.step"
        s.exportStep(str(step))
        log[pid] = dict(step=str(step.relative_to(REPO)), volume_mm3=round(s.Volume, 3), valid=bool(s.isValid()),
                        solids=len(s.Solids), faces=len(s.Faces))
        print("FC", pid, json.dumps(log[pid]), flush=True)
    (WORK / "fc_rev3.json").write_text(json.dumps(log, indent=1))
    print("FC done", flush=True)


def cmd_step():
    WORK.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    env["DIPB4_FREECAD"] = "1"
    r = subprocess.run(["freecadcmd", str(Path(__file__).resolve())], env=env, cwd=str(REPO), capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith("FC")), flush=True)
    if r.returncode != 0 or "FC done" not in r.stdout:
        print(r.stderr[-3000:])
        raise SystemExit("freecadcmd failed")
    log = json.loads((WORK / "fc_rev3.json").read_text())
    rec = json.loads(HEIGHTS.read_text())
    for pid, l in log.items():
        grp = "covers" if pid.startswith("I15") else "caps"
        mv = rec[grp][pid]["volume_mm3"] + rec[grp][pid]["engraving_mm3"]
        l["mesh_volume_without_engraving_mm3"] = round(mv, 3)
        l["volume_diff_pct"] = round(100 * (l["volume_mm3"] / mv - 1), 3)
    rec["step"] = log
    HEIGHTS.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    bad = [k for k, l in log.items() if not l["valid"] or l["solids"] != 1 or abs(l["volume_diff_pct"]) > 1.0]
    if bad:
        raise SystemExit(f"STEP problems: {bad} {log}")


# ------------------------------------------------------------------------------------------------ figures
def cmd_figures():
    """The links from the palmar side and from the slotted arm's side, the pad's lead over the rigid parts, the sweep."""
    import numpy as np
    import trimesh
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import dip_b_v3_figures as F
    (V4 / "media").mkdir(parents=True, exist_ok=True)
    tmp = WORK / "fig"
    tmp.mkdir(parents=True, exist_ok=True)
    ld = lambda p: trimesh.load(p, force="mesh")                                                       # noqa: E731
    v = {"v3": dict(link=ld(LINKS["v3"]), cover=ld(DIP / "v3/b/cad/20261021-dip_link_v3b_cover_I15.stl")),
         "v4": dict(link=ld(LINKS["v4"]), cover=ld(V4 / f"cad/{PFX}_cover.stl"))}
    pad = B.to_trimesh(TC.pad_rev2())
    c = np.array([-2.0, 0.0, -72.0])
    files = []
    for kk in ("v3", "v4"):
        for view, cam, zoom in (("palmar", (tuple(c + [-95.0, -28.0, 6.0]), tuple(c), (0, 0, -1)), 1.2),
                                ("side", (tuple(c + [0.0, 110.0, 0.0]), tuple(c), (0, 0, -1)), 1.15)):
            f = tmp / f"{view}_{kk}.png"
            F.shot([(v[kk]["link"], F.COL["link"], 1.0), (v[kk]["cover"], F.COL["cover"], 1.0), (pad, F.COL["pad"], 1.0)], f, cam,
                   zoom=zoom)
            files.append((f, f"{kk}: {'from the palmar side' if view == 'palmar' else 'from the slotted arm'}"))
    fig = plt.figure(figsize=(13.0, 13.0), dpi=100)
    for i, (f, title) in enumerate(files):
        ax = fig.add_axes([0.5 * (i // 2), 0.5 - 0.48 * (i % 2), 0.5, 0.46])
        ax.imshow(Image.open(f).convert("RGB"))
        ax.set_axis_off()
        ax.set_title(title, fontsize=17, family="serif")
    fig.suptitle("DIP link b v3 and v4 with cover I15 and pad B00R, finger up", fontsize=19, family="serif")
    fig.savefig(V4 / f"media/{PFX}_v3_v4.png", facecolor="white")
    plt.close(fig)
    # the pad's lead over the rigid parts against the object's tilt
    rec = json.loads(CHECKS.read_text())
    cur = rec["flat_object"]["curves"]
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=100)
    for ax, kind, xl in ((axs[0], "proximal_tilt", "Tilt toward the proximal side (deg)"),
                         (axs[1], "side_tilt", "Tilt across the finger, toward the slotted arm (deg)")):
        for kk, col in (("v3", "#888888"), ("v4", "#c0392b")):
            t = [r["tilt_deg"] for r in cur[kk][kind]]
            ax.plot(t, [r["link_cover"] for r in cur[kk][kind]], color=col, lw=2.2, label=f"{kk}: DIP link and cover")
            ax.plot(t, [r["with_joint_pip0"] for r in cur[kk][kind]], color=col, lw=1.4, ls="--",
                    label=f"{kk}: with the MCP link and servo, PIP 0")
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xlabel(xl, fontsize=14)
        ax.set_ylabel("Pad's lead over the rigid parts (mm)", fontsize=14)
        ax.tick_params(labelsize=13)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-6, 8)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=11.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(V4 / f"media/{PFX}_flat_object.png", facecolor="white")
    plt.close(fig)
    # the sweep: covers and caps side by side, wall tops up
    hrec = json.loads(HEIGHTS.read_text())
    cols = dict(B.COLORS, cap="#e07b24")
    shots = []
    for cid in hrec["covers"]:
        f = tmp / f"{cid}.png"
        F.shot([(ld(V4 / f"cad/{PFX}_cover_{cid}.stl"), cols["cover"], 1.0)], f,
               (tuple(np.array([2.0, 0.0, -87.0]) + [-48.0, -30.0, -30.0]), (2.0, 0.0, -87.0), (0, 0, -1)), zoom=1.35)
        shots.append((f, f"Cover {cid}: walls {hrec['covers'][cid]['wall_h_mm']:g} mm"))
    for vid in hrec["caps"]:
        f = tmp / f"{vid}.png"
        F.shot([(ld(V4 / f"cad/{PFX}_tip_cap_{vid}.stl"), cols["cap"], 1.0), (pad, F.COL["pad"], 0.45)], f,
               (tuple(np.array([0.0, 0.0, -94.0]) + [-50.0, -36.0, -26.0]), (0.0, 0.0, -94.0), (0, 0, -1)), zoom=1.7)
        shots.append((f, f"Cap {vid}: blocks {hrec['caps'][vid]['block_h_mm']:g} mm"))
    fig = plt.figure(figsize=(15.0, 10.4), dpi=100)
    for i, (f, title) in enumerate(shots):
        ax = fig.add_axes([(i % 3) / 3.0, 0.5 - 0.49 * (i // 3), 1 / 3.0, 0.45])
        ax.imshow(Image.open(f).convert("RGB"))
        ax.set_axis_off()
        ax.set_title(title, fontsize=16, family="serif")
    fig.savefig(V4 / f"media/{PFX}_heights.png", facecolor="white")
    plt.close(fig)
    print("figures ->", (V4 / "media").relative_to(REPO), flush=True)


def main():
    for c in sys.argv[1:] or ["link", "heights", "step", "figures"]:
        t0 = time.time()
        {"link": cmd_link, "heights": cmd_heights, "step": cmd_step, "figures": cmd_figures}[c]()
        print(f"== {c} [{time.time() - t0:.0f} s]", flush=True)


if os.environ.get("DIPB4_FREECAD") == "1":
    fc_main()
elif __name__ == "__main__":
    main()
