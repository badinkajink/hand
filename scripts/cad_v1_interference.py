#!/usr/bin/env python3
"""Interpenetration of the cad_v1 finger's CAD parts, posed by the compiled model.

Every part mesh of assets/mjcf/experimental/20261013-cad_v1/meshes/ is voxelised once in its body
frame (cad_features.voxelize_solid). For a joint configuration q, MuJoCo forward kinematics of
finger.xml gives the body poses; the voxel centres of part A are carried into the body frame of
part B and counted inside B. Flush faces (the servo in its 12.1 mm pocket, a 25.38 mm boss in a
25.40 mm cradle) put a one-voxel film inside the other part, so a collision is counted only for
voxels deeper than one voxel: inside B eroded by one voxel. Penetration volume = count x pitch^3,
the larger of the two directions.

Outputs, in the model folder:
  interference.json   overlap per part pair at q = 0, the three servo seats, single-joint sweeps
                      over the real_v1 range of motion, and an MCP x PIP grid
Run:
  ~/.claude/bin/resguard.sh run --mem 1G --cpu 200 -- .venv/bin/python scripts/cad_v1_interference.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from morphohand.geometry.cad_features import SolidGrid, voxelize_solid  # noqa: E402
from scipy import ndimage  # noqa: E402
from build_cad_v1_finger import BODY_PARTS, OUT, ROM  # noqa: E402

THRESH_MM3 = 0.1  # penetration deeper than one voxel: collision above this volume (mm^3) or area (mm^2)
PITCH = 0.3


def main():
    m = mujoco.MjModel.from_xml_path(str(OUT / "finger.xml"))
    d = mujoco.MjData(m)
    jid = {j: m.joint(f"index_{j}").qposadr[0] for j in ("yaw", "mcp", "pip")}
    body_of = {p: b for b, ps in BODY_PARTS.items() for p in ps}
    bid = {b: m.body(f"index_{b}").id for b in BODY_PARTS}
    grids, deep = {}, {}
    for p in body_of:
        mesh = trimesh.load(OUT / "meshes" / f"{p}.stl")
        mesh.apply_scale(1000.0)  # mm
        g = voxelize_solid(mesh, PITCH)
        grids[p] = g
        deep[p] = SolidGrid(g.origin, g.pitch, ndimage.binary_erosion(g.occ, iterations=1))
    centres = {p: g.centres() for p, g in grids.items()}
    # boundary voxels: a penetration deeper than one voxel always puts some of them inside the other
    # part's eroded solid, so the sweeps test only these (about a fifth of the solid)
    shell = {}
    for p, g in grids.items():
        b = g.occ & ~ndimage.binary_erosion(g.occ, iterations=1)
        shell[p] = g.origin + g.pitch * np.argwhere(b)

    def poses(q):
        d.qpos[:] = 0
        for j, v in q.items():
            d.qpos[jid[j]] = v
        mujoco.mj_kinematics(m, d)
        return {b: (d.xpos[bid[b]] * 1000.0, d.xmat[bid[b]].reshape(3, 3).copy()) for b in BODY_PARTS}

    def overlap(a, b, P, kind="deep"):
        """'deep'/'film': solid voxels of one part inside the other (eroded or not), mm^3;
        'shell': boundary voxels of one part inside the other's eroded solid, as an area in mm^2."""
        G = grids if kind == "film" else deep
        src = shell if kind == "shell" else centres
        (pa, Ra), (pb, Rb) = P[body_of[a]], P[body_of[b]]
        X = src[a] @ Ra.T + pa  # A voxels in world
        n_ab = int(G[b].contains((X - pb) @ Rb).sum())  # into B's body frame
        X = src[b] @ Rb.T + pb
        n_ba = int(G[a].contains((X - pa) @ Ra).sum())
        return max(n_ab, n_ba) * PITCH ** (2 if kind == "shell" else 3)

    parts = list(body_of)
    cross = [(a, b) for a, b in itertools.combinations(parts, 2) if body_of[a] != body_of[b]]
    same = [(a, b) for a, b in itertools.combinations(parts, 2) if body_of[a] == body_of[b]]
    out = {"pitch_mm": PITCH, "threshold_mm3": THRESH_MM3}

    P0 = poses({})
    out["rest_film"] = {f"{a}|{b}": round(overlap(a, b, P0, "film"), 3) for a, b in cross + same}
    out["rest"] = {f"{a}|{b}": round(overlap(a, b, P0), 3) for a, b in cross + same}
    print(f"rest pose: overlap mm^3 (any voxel) / penetration deeper than {PITCH} mm:")
    for k in out["rest"]:
        print(f"  {k:28s} {out['rest_film'][k]:8.3f} {out['rest'][k]:8.3f}")

    sweeps = {}
    deg = np.degrees
    for j in ("yaw", "mcp", "pip"):
        lo, hi = ROM[j]
        qs = np.radians(np.arange(np.floor(deg(lo)), np.ceil(deg(hi)) + 1, 1.0))
        rows = []
        for v in qs:
            P = poses({j: v})
            rows.append({f"{a}|{b}": round(overlap(a, b, P, "shell"), 2) for a, b in cross})
        sweeps[j] = {"q_deg": np.round(deg(qs), 2).tolist(), "overlap_mm3": rows}
        hits = {}
        for k in rows[0]:
            bad = [float(np.round(deg(qs[i]), 1)) for i, r in enumerate(rows) if r[k] > THRESH_MM3]
            if bad:
                hits[k] = [min(bad), max(bad), len(bad)]
        sweeps[j]["colliding_pairs_deg"] = hits
        print(f"{j} sweep (penetrating boundary, mm^2) {deg(lo):.0f}..{deg(hi):.0f} deg: colliding pairs (min, max, count) {hits}")
    out["sweeps"] = sweeps

    # MCP x PIP grid at 3 deg, yaw 0: the curl that matters for self-contact
    mq = np.radians(np.arange(np.floor(deg(ROM["mcp"][0])), np.ceil(deg(ROM["mcp"][1])) + 1, 3.0))
    pq = np.radians(np.arange(np.floor(deg(ROM["pip"][0])), np.ceil(deg(ROM["pip"][1])) + 1, 3.0))
    G = np.zeros((len(mq), len(pq)))
    worst = {}
    for i, a1 in enumerate(mq):
        for k, a2 in enumerate(pq):
            P = poses({"mcp": a1, "pip": a2})
            tot = 0.0
            for a, b in cross:
                v = overlap(a, b, P, "shell")
                if v > THRESH_MM3:
                    tot += v
                    key = f"{a}|{b}"
                    worst[key] = max(worst.get(key, 0.0), v)
            G[i, k] = tot
    out["grid"] = {"mcp_deg": np.round(deg(mq), 1).tolist(), "pip_deg": np.round(deg(pq), 1).tolist(),
                   "total_overlap_mm3": np.round(G, 2).tolist(), "worst_by_pair_mm3": {k: round(v, 2) for k, v in worst.items()},
                   "colliding_fraction": float((G > THRESH_MM3).mean())}
    print(f"MCP x PIP grid: {100 * (G > THRESH_MM3).mean():.1f} % of {G.size} poses collide; worst {out['grid']['worst_by_pair_mm3']}")
    (OUT / "interference.json").write_text(json.dumps(out))
    print("wrote", OUT / "interference.json")


if __name__ == "__main__":
    main()
