#!/usr/bin/env python3
"""The cad_v2 finger: the CAD hand of scripts/build_cad_v1_finger.py with the fingertip on the UR5 calibration rig since
2026-10-23 (assets/cad_v2): the DIP link b v4, the cassette 3T, the top cover TW15H3 trimmed at x_F -0.10 with its four
M2 button heads (the owner's trim of 2026-10-27, so that the pad's top edge is exposed; scripts/ur5_cb3_cell.py part_F),
and the 11/22 half-ellipsoid TPU pad.

Frame F (assets/cad_v2/README.md) is the cad_v1 assembly's finger frame: x palmar (-) to dorsal (+), y along the PIP
axis, z along the finger, the PIP axis at z_F = -51.55 mm. The new link replaces the old one without moving it; it is
9.75 mm longer, so the pad's centre (its ellipsoid's centre plane, where the apex lies 10.55 mm palmar of the finger's
axis) is 36.36 mm below the PIP axis, against 26.61 mm for the TPU block. The tip body moves there; its pads are
reorient_backends.replace_tips' sphere pads of the key tcd_ell7.4x8.5x11x22 (tip_codesign_geometry.CurvedTip: semi-axes
7.4 mm across, 8.5 mm deep, 11 mm distal and 22 mm proximal of the apex, apex 10.55 mm from the axis, 21.41 mm long),
which is the v4b pad's dome in the tip frame (pad STL bounds x_F -10.55, z_F -98.46 to -77.05).

The DIP link, cassette and trimmed cap with heads replace the cad_v1 DIP link's visual mesh and collision spheres in
<f>_pip_frame. Their collision spheres are inscribed in each part (no sphere stands proud of its part): candidate centres
on a grid inside the part, radius the distance to the part's surface (a k-d tree on dense surface samples), chosen
greedily by the surface samples they bring within COVER_MM of a sphere until COVER_SHARE of them are covered or the
part's budget is spent. Masses and inertias are the parts' solid volumes (PLA 1240, TPU 1210 kg/m^3, as cad_v1).

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/build_cad_v2_finger.py
Writes assets/mjcf/experimental/20261030-cad_v2/: scene_screwdriver_medium.xml (cad_v1's with the new distal parts),
meshes/ (the new parts in the pip_frame body frame, m; cad_v1's other meshes are referenced in place) and spheres.json
(spheres per part, coverage).
"""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SRC = ROOT / "assets/mjcf/experimental/20261013-cad_v1"
OUT = ROOT / "assets/mjcf/experimental/20261030-cad_v2"
FINGERS = ("thumb", "index", "middle")
PIP_F_MM = np.array([0.0, 0.0, -51.55])          # the PIP axis' centre in F; the pip_frame body's origin
TIP_BODY_MM = 36.36                              # PIP axis to the pad's centre plane (z_F -87.91)
DENSITY = {"dip_link": 1240.0, "cassette": 1240.0, "cap": 1240.0, "pad": 1210.0}
BUDGET = {"dip_link": 160, "cassette": 140, "cap": 110}
COVER_MM, COVER_SHARE, GRID_MM, R_MAX_MM = 1.0, 0.97, 0.5, 6.0


def parts_F():
    """The parts in F (mm): the DIP link, cassette and pad; the cap as printed (cap_printed, watertight, packed and then
    cut back to the trim plane), the cap trimmed with its heads (cap, the visual) and the four heads (heads)."""
    os.environ.setdefault("UR5_CAP_TRIM", "-0.10")
    os.environ.setdefault("UR5_SCREW_HEADS", "button")
    import ur5_cb3_cell as CELL
    pad = max(CELL.part_F("pad").split(only_watertight=False), key=lambda m: m.volume)     # the pad without its pockets
    return {"dip_link": CELL.part_F("dip_link"), "cassette": CELL.part_F("cover"), "cap": CELL.part_F("cap"),
            "cap_printed": CELL._load(CELL.TOOL_PARTS["cap"][0]), "heads": CELL.heads_mesh_F(), "pad": pad,
            "trim_x": CELL.CAP_TRIM_X}


def to_body(m, origin_mm):
    """A part in F (mm) to a body frame at origin_mm with F's axes (m)."""
    m = m.copy()
    m.apply_translation(-np.asarray(origin_mm, float))
    m.apply_scale(1e-3)
    return m


def pack(m, budget, seed=0):
    """Inscribed spheres covering the surface of a watertight part (mm): [(centre, radius)], covered share."""
    import trimesh
    from scipy.spatial import cKDTree
    S, fi = trimesh.sample.sample_surface(m, 60000, seed=seed)
    Nf = m.face_normals[fi]
    tree = cKDTree(S)
    lo, hi = m.bounds
    g = [np.arange(lo[k] + GRID_MM / 2, hi[k], GRID_MM) for k in range(3)]
    P = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    # inside: the offsets from the 8 nearest surface samples run against those samples' outward face normals, by
    # majority (one nearest sample misjudges points beside an edge; trimesh's ray-cast contains needs gigabytes on the
    # link's 15k faces)
    dist, j = tree.query(P, k=8)
    vote = (np.einsum("ikj,ikj->ik", P[:, None, :] - S[j], Nf[j]) < 0.0).sum(1)
    inside = vote >= 5
    C = P[inside]
    R = np.minimum(dist[inside, 0], R_MAX_MM)
    keep = R >= 0.4
    C, R = C[keep], R[keep]
    ctree = cKDTree(C)
    covered = np.zeros(len(S), bool)
    out = []
    rng = np.random.default_rng(seed)
    # take an uncovered surface sample; of the candidate spheres that bring it within COVER_MM, keep the one that
    # covers the most uncovered samples (one neighbour query per sphere tried, so memory stays small)
    while len(out) < budget and covered.mean() < COVER_SHARE:
        s = S[rng.choice(np.flatnonzero(~covered))]
        cand = np.asarray(ctree.query_ball_point(s, R_MAX_MM + COVER_MM), int)
        cand = cand[(R[cand] > 0) & (np.linalg.norm(C[cand] - s, axis=1) <= R[cand] + COVER_MM)]
        if not len(cand):
            covered[np.asarray(tree.query_ball_point(s, COVER_MM), int)] = True     # no sphere reaches it: skip it
            continue
        best, bi = None, None
        for i in cand[np.argsort(-R[cand])][:24]:
            ix = np.asarray(tree.query_ball_point(C[i], R[i] + COVER_MM), int)
            g = int(np.count_nonzero(~covered[ix]))
            if best is None or g > best[0]:
                best, bi = (g, ix), i
        # exact checks of the chosen sphere: its centre inside the part (one ray) and its radius at most the exact
        # distance to the surface; a failed centre is dropped from the candidates
        if not bool(m.contains(C[bi][None])[0]):
            R[bi] = 0.0
            continue
        r = min(float(R[bi]), float(m.nearest.on_surface(C[bi][None])[1][0]))
        if r < 0.4:
            R[bi] = 0.0
            continue
        out.append((C[bi], r))
        covered[np.asarray(tree.query_ball_point(C[bi], r + COVER_MM), int)] = True
    return out, float(covered.mean())


def mass_props(meshes_m):
    """Mass, centre of mass and inertia about it (body frame) of parts [(mesh in m, density)]."""
    J = np.zeros((3, 3))
    props = []
    for m, rho in meshes_m:
        mm = m.copy()
        mm.density = rho
        props.append((mm.mass, mm.center_mass, mm.moment_inertia))
    M = sum(p[0] for p in props)
    c = sum(p[0] * p[1] for p in props) / M
    for mass, cm, Ic in props:
        d = cm - c
        J += Ic + mass * (d @ d * np.eye(3) - np.outer(d, d))
    return M, c, J


def fmt(v, nd=6):
    return " ".join(f"{x:.{nd}f}".rstrip("0").rstrip(".") if abs(x) > 0 else "0" for x in np.atleast_1d(v))


def inertial(M, c, J, mirror):
    sg = np.array([-1.0, -1.0, 1.0]) if mirror else np.ones(3)
    fi = np.array([J[0, 0], J[1, 1], J[2, 2], J[0, 1], J[0, 2], J[1, 2]])
    if mirror:                                           # 180 deg about z flips Ixz and Iyz
        fi = fi * np.array([1, 1, 1, 1, -1, -1])
    return {"pos": fmt(c * sg), "mass": f"{M:.6f}", "fullinertia": fmt(fi, 10)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "meshes").mkdir(exist_ok=True)
    F = parts_F()
    rigid = ("dip_link", "cassette", "cap")
    body = {k: to_body(F[k], PIP_F_MM) for k in rigid}
    pad_tip = to_body(F["pad"], PIP_F_MM + np.array([0.0, 0.0, -TIP_BODY_MM]))
    for k, m in body.items():
        m.export(OUT / "meshes" / f"v4b_{k}.stl")
    spheres, report = {}, {}
    for k in rigid:
        mm = F["cap_printed" if k == "cap" else k].copy()
        mm.apply_translation(-PIP_F_MM)
        sp, cov = pack(mm, BUDGET[k])
        if k == "cap":                                   # the trim: nothing of the cap palmar of x_F = trim_x
            sp = [(c, min(r, c[0] - F["trim_x"])) for c, r in sp]
            sp = [(c, r) for c, r in sp if r >= 0.4]
            hm = F["heads"].copy()
            hm.apply_translation(-PIP_F_MM)
            sp += pack(hm, 40)[0]
        spheres[k] = [[*(np.asarray(c) / 1000).round(7).tolist(), round(r / 1000, 7)] for c, r in sp]
        report[k] = {"spheres": len(sp), "covered_share": round(cov, 4), "watertight": bool(mm.is_watertight),
                     "r_mm_range": [round(min(r for _, r in sp), 2), round(max(r for _, r in sp), 2)],
                     "palmar_extent_x_F_mm": round(min(c[0] - r for c, r in sp), 2)}
        print(k, report[k], flush=True)
    M_pip, c_pip, I_pip = mass_props([(body[k], DENSITY[k]) for k in rigid])
    M_tip, c_tip, I_tip = mass_props([(pad_tip, DENSITY["pad"])])
    (OUT / "spheres.json").write_text(json.dumps({"pip_frame": spheres, "report": report,
                                                  "mass_kg": {"pip_frame": M_pip, "tip": M_tip}}, indent=1))
    root = ET.parse(SRC / "scene_screwdriver_medium.xml").getroot()
    asset = root.find("asset")
    for a in list(asset.findall("mesh")):              # cad_v1's unchanged meshes by path; its DIP link and tip go
        if a.get("name") in ("cad_dip_link", "cad_tip"):
            asset.remove(a)
        elif a.get("file"):
            a.set("file", f"../../{SRC.name}/meshes/{a.get('file')}")
    for k in rigid:
        ET.SubElement(asset, "mesh", name=f"cad_v4b_{k}", file=f"v4b_{k}.stl")
    for f in FINGERS:
        mirror = f == "thumb"
        sg = np.array([-1.0, -1.0, 1.0]) if mirror else np.ones(3)
        q = {"quat": "0 0 0 1"} if mirror else {}
        pf = root.find(f".//body[@name='{f}_pip_frame']")
        for g in list(pf.findall("geom")):
            pf.remove(g)
        pf.find("inertial").attrib.update(inertial(M_pip, c_pip, I_pip, mirror))
        tip = pf.find(f"body[@name='{f}_tip']")
        at = list(pf).index(tip)
        new = []
        for k in rigid:
            new.append(ET.Element("geom", {"name": f"{f}_{k}_visual", "class": "cad_visual", "mesh": f"cad_v4b_{k}",
                                           "material": "cad_link", **q}))
        for k in rigid:
            for i, (x, y, z, r) in enumerate(spheres[k]):
                c = np.array([x, y, z]) * sg
                new.append(ET.Element("geom", {"name": f"{f}_{k}_s{i}", "class": "cad_collision", "pos": fmt(c, 7),
                                               "size": f"{r:.6f}"}))
        for i, e in enumerate(new):
            pf.insert(at + i, e)
        tip.set("pos", f"0 0 {-TIP_BODY_MM / 1000:.5f}")
        for g in list(tip.findall("geom")):
            tip.remove(g)
        tip.find("inertial").attrib.update(inertial(M_tip, c_tip, I_tip, mirror))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(OUT / "scene_screwdriver_medium.xml")
    (OUT / "assembly.json").write_text(json.dumps({
        "source": str(SRC.relative_to(ROOT)), "tip_key": "tcd_ell7.4x8.5x11x22", "tip_body_mm": TIP_BODY_MM,
        "pip_F_mm": PIP_F_MM.tolist(), "cap_trim_x_F": float(os.environ["UR5_CAP_TRIM"]),
        "screw_heads": os.environ["UR5_SCREW_HEADS"], "spheres": report,
        "mass_kg": {"pip_frame": round(M_pip, 6), "tip": round(M_tip, 6)},
        "parts_dir": "docs/experiments/20261018-dip_link_board_mount/v4/b/cad"}, indent=1))
    print("wrote", (OUT / "scene_screwdriver_medium.xml").relative_to(ROOT), f"pip_frame {M_pip * 1000:.2f} g, "
          f"pad {M_tip * 1000:.2f} g", flush=True)


if __name__ == "__main__":
    main()
