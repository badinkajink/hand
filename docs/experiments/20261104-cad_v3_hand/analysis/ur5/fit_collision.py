#!/usr/bin/env python3
"""Collision primitives (cylinders, capsules, boxes) for the UR5 (CB3) links, fitted to ur_description's collision meshes
in each URDF link frame, and how well they cover them: the share of mesh vertices and surface samples inside the union
(0.5 mm tolerance), the largest distance of an uncovered point, and the union's volume against the mesh's (Monte Carlo).

Each primitive takes the points a selector picks; its radius is their largest distance from the axis and its length their
extent along it. Writes collision_primitives.json (metres, link frames) for build_ur5_mjcf.py.

    ~/.claude/bin/resguard.sh run --mem 3G -- .venv/bin/python logs/20261104-cad_v3/ur5/fit_collision.py
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path("/home/humanoid/Programs/hand")
sys.path.insert(0, str(ROOT / "src"))
from morphohand.geometry import rotations as R  # noqa: E402

HERE = Path(__file__).resolve().parent
UR = Path.home() / "Programs/rh56_controller/magpie_force_control/src/ur_description"
AX = {"x": 0, "y": 1, "z": 2}

# per link: (name, type, axis, centre in the plane normal to the axis (mm) or None for the selection's bbox centre, selector)
SPEC = {
    "base_link_inertia": [
        ("base_disc", "cyl", "z", (0.0, 0.0), lambda P: P[:, 1] < 74.0, None),
        ("base_cable_outlet", "box", None, None, lambda P: P[:, 1] >= 74.0, None),
    ],
    "shoulder_link": [
        ("shoulder_j1", "cyl", "z", (0.0, 0.0), lambda P: P[:, 1] >= 0.0, None),
        ("shoulder_j2_side", "box", None, None, lambda P: P[:, 1] < 0.0, None),
    ],
    "upper_arm_link": [   # housings about the J2 and J3 axes (z); the tube along x between them, and its flares
        ("upperarm_j2", "cyl", "z", (0.0, 0.0), lambda P: np.hypot(P[:, 0], P[:, 1]) <= 60.5, None),
        ("upperarm_tube", "capsule", "x", None,
         lambda P: (P[:, 0] > -345.0) & (P[:, 0] < -80.0) & (np.hypot(P[:, 0], P[:, 1]) > 60.5)
         & (np.hypot(P[:, 0] + 425.0, P[:, 1]) > 60.5), (-425.0, 0.0)),
        ("upperarm_flare_j2", "cyl", "x", None, lambda P: (P[:, 0] >= -80.0) & (np.hypot(P[:, 0], P[:, 1]) > 60.5), None),
        ("upperarm_flare_j3", "cyl", "x", None,
         lambda P: (P[:, 0] <= -345.0) & (np.hypot(P[:, 0] + 425.0, P[:, 1]) > 60.5), None),
        ("upperarm_j3", "cyl", "z", (-425.0, 0.0), lambda P: np.hypot(P[:, 0] + 425.0, P[:, 1]) <= 60.5, None),
    ],
    "forearm_link": [     # the J3 housing (z), the tube along x and its flares, the J4 housing about the wrist 1 axis
        ("forearm_j3", "cyl", "z", (0.0, 0.0), lambda P: np.hypot(P[:, 0], P[:, 1]) <= 59.0, None),
        ("forearm_tube", "capsule", "x", None,
         lambda P: (P[:, 0] > -345.0) & (P[:, 0] < -80.0) & (np.hypot(P[:, 0], P[:, 1]) > 59.0)
         & (np.hypot(P[:, 0] + 392.25, P[:, 1]) > 39.6), (-392.25, 0.0)),
        ("forearm_flare_j3", "cyl", "x", None, lambda P: (P[:, 0] >= -80.0) & (np.hypot(P[:, 0], P[:, 1]) > 59.0), None),
        ("forearm_flare_j4", "cyl", "x", None,
         lambda P: (P[:, 0] <= -345.0) & (np.hypot(P[:, 0] + 392.25, P[:, 1]) > 39.6), None),
        ("forearm_j4", "cyl", "z", (-392.25, 0.0), lambda P: np.hypot(P[:, 0] + 392.25, P[:, 1]) <= 39.6, None),
    ],
    "wrist_1_link": [     # the J5 housing about y; the J4 joint's ring about z below it
        ("wrist1_j5_housing", "cyl", "y", (0.0, 0.0), lambda P: np.hypot(P[:, 0], P[:, 2]) <= 39.5, None),
        ("wrist1_j4_ring", "cyl", "z", (0.0, 0.0), lambda P: np.hypot(P[:, 0], P[:, 2]) > 39.5, (None, 0.0)),
    ],
    "wrist_2_link": [     # the J6 housing about y; the J5 joint's ring about z below it
        ("wrist2_j6_housing", "cyl", "y", (0.0, 0.0), lambda P: np.hypot(P[:, 0], P[:, 2]) <= 39.5, None),
        ("wrist2_j5_ring", "cyl", "z", (0.0, 0.0), lambda P: np.hypot(P[:, 0], P[:, 2]) > 39.5, (None, 0.0)),
    ],
    "wrist_3_link": [     # the flange housing about z (tool0's axis); the tool connector's boss on -y
        ("wrist3_flange", "cyl", "z", (0.0, 0.0), lambda P: np.hypot(P[:, 0], P[:, 1]) <= 38.5, None),
        ("wrist3_connector", "box", None, None, lambda P: np.hypot(P[:, 0], P[:, 1]) > 38.5, None),
    ],
}


def origin(el):
    o = el.find("origin")
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0.0] * 3
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0] * 3
    return R.tf(R.euler_to_mat(rpy, "xyz"), np.array(xyz))


def link_meshes():
    root = ET.parse(UR / "ur5.urdf").getroot()
    out = {}
    for ln in root.findall("link"):
        col = ln.find("collision")
        if col is None:
            continue
        m = trimesh.load(UR / col.find("geometry/mesh").get("filename").replace("ur_description/", ""), force="mesh")
        m.apply_transform(origin(col))
        m.apply_scale(1e3)
        out[ln.get("name")] = m
    return out


def fit(kind, axis, centre, P, span=None):
    if kind == "box":
        lo, hi = P.min(0), P.max(0)
        return dict(type="box", pos=((lo + hi) / 2).tolist(), half=((hi - lo) / 2).tolist())
    a = AX[axis]
    o = [i for i in range(3) if i != a]
    c = np.array(centre) if centre is not None else (P[:, o].min(0) + P[:, o].max(0)) / 2
    r = float(np.hypot(P[:, o[0]] - c[0], P[:, o[1]] - c[1]).max())
    t0, t1 = float(P[:, a].min()), float(P[:, a].max())
    if span is not None:                                   # an axial range given (None keeps the points' end)
        t0 = t0 if span[0] is None else min(t0, span[0]) if kind == "cyl" else span[0]
        t1 = t1 if span[1] is None else max(t1, span[1]) if kind == "cyl" else span[1]
    if kind == "cyl":
        pos = np.zeros(3)
        pos[o] = c
        pos[a] = (t0 + t1) / 2
        return dict(type="cylinder", axis=axis, pos=pos.tolist(), r=r, half=(t1 - t0) / 2)
    # capsule along the axis whose round ends stay inside the selection's extent plus the housings
    p0, p1 = np.zeros(3), np.zeros(3)
    p0[o], p1[o] = c, c
    p0[a], p1[a] = t0, t1
    return dict(type="capsule", axis=axis, fromto=p0.tolist() + p1.tolist(), r=r)


def inside(prim, P, tol=0.5):
    if prim["type"] == "box":
        return np.all(np.abs(P - np.array(prim["pos"])) <= np.array(prim["half"]) + tol, axis=1)
    if prim["type"] == "cylinder":
        a = AX[prim["axis"]]
        o = [i for i in range(3) if i != a]
        c = np.array(prim["pos"])
        rad = np.hypot(P[:, o[0]] - c[o[0]], P[:, o[1]] - c[o[1]])
        return (rad <= prim["r"] + tol) & (np.abs(P[:, a] - c[a]) <= prim["half"] + tol)
    f = np.array(prim["fromto"])
    A, B = f[:3], f[3:]
    u = B - A
    t = np.clip(((P - A) @ u) / (u @ u), 0.0, 1.0)
    return np.linalg.norm(P - (A + t[:, None] * u), axis=1) <= prim["r"] + tol


def main():
    ms = link_meshes()
    rng = np.random.default_rng(0)
    out, report = {}, {}
    for ln, spec in SPEC.items():
        m = ms[ln]
        S = trimesh.sample.sample_surface(m, 40000, seed=0)[0]
        P = np.vstack([m.vertices, S])
        prims = []
        for name, kind, axis, centre, sel, span in spec:
            Q = P[sel(P)]
            pr = fit(kind, axis, centre, Q, span)
            pr["name"] = name
            prims.append(pr)
        cov = np.zeros(len(P), bool)
        for pr in prims:
            cov |= inside(pr, P)
        # uncovered points: their distance to the union (approximate: the least excess over the primitives' bounds)
        miss = P[~cov]
        worst = 0.0
        if len(miss):
            for tol in np.arange(0.5, 40.0, 0.5):
                c2 = np.zeros(len(miss), bool)
                for pr in prims:
                    c2 |= inside(pr, miss, tol)
                if c2.all():
                    worst = float(tol)
                    break
        # volumes by Monte Carlo in the union's bounding box
        lo, hi = m.bounds[0] - 25.0, m.bounds[1] + 25.0
        X = rng.uniform(lo, hi, size=(200000, 3))
        inu = np.zeros(len(X), bool)
        for pr in prims:
            inu |= inside(pr, X, 0.0)
        box_vol = float(np.prod(hi - lo))
        v_union = inu.mean() * box_vol
        report[ln] = dict(covered_frac=round(float(cov.mean()), 5), n_points=int(len(P)), worst_uncovered_mm=worst,
                          mesh_volume_cm3=round(float(m.volume) / 1e3, 1), union_volume_cm3=round(v_union / 1e3, 1),
                          union_over_mesh=round(v_union / float(m.volume), 3))
        out[ln] = [dict({k: (np.round(np.array(v) * 1e-3, 6).tolist() if k in ("pos", "half", "fromto") else
                             (round(v * 1e-3, 6) if k == "r" else v)) for k, v in pr.items()}) for pr in prims]
        print(ln, json.dumps(report[ln]))
        for pr in prims:
            print("   ", json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, (list, np.ndarray)) else
                                         (round(v, 2) if isinstance(v, float) else v)) for k, v in pr.items()}))
    (HERE / "collision_primitives.json").write_text(json.dumps(dict(units="m, URDF link frames", links=out,
                                                                    coverage=report), indent=1))


if __name__ == "__main__":
    main()
