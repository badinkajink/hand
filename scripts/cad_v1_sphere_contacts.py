#!/usr/bin/env python3
"""Self-contacts of the cad_v1 finger's collision spheres over the range of motion (2026-10-13).

finger.xml as built (balanced MorphIt spheres unless the build used another sphere file): each joint swept at 1 deg
with the others at zero, then the MCP x PIP grid at 3 deg; a contact counts when its distance is zero or less. Writes
sphere_contacts.json beside the model.

    .venv/bin/python scripts/cad_v1_sphere_contacts.py
"""
import mujoco, numpy as np, collections, json
m = mujoco.MjModel.from_xml_path("assets/mjcf/experimental/20261013-cad_v1/finger.xml"); d = mujoco.MjData(m)
adr = {j: m.joint(f"index_{j}").qposadr[0] for j in ("yaw", "mcp", "pip")}
def contacts(q):
    d.qpos[:] = 0
    for j, v in q.items(): d.qpos[adr[j]] = np.radians(v)
    mujoco.mj_forward(m, d)
    out = collections.Counter()
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist > 0: continue
        b = tuple(sorted((mujoco.mj_id2name(m, 1, m.geom_bodyid[c.geom1]), mujoco.mj_id2name(m, 1, m.geom_bodyid[c.geom2]))))
        out[b] += 1
    return out
res = {}
for j, (lo, hi) in {"yaw": (-85, 85), "mcp": (-15, 92), "pip": (-18, 92)}.items():
    hits = collections.defaultdict(list)
    for v in range(lo, hi + 1):
        for b, n in contacts({j: v}).items(): hits[" / ".join(b)].append(v)
    res[j] = {k: [min(v), max(v), len(v)] for k, v in hits.items()}
    print(j, res[j])
grid = collections.Counter(); cells = collections.defaultdict(list); n = 0
for a in range(-15, 94, 3):  # the grid of scripts/cad_v1_interference.py: MCP -15..93, PIP -18..93 deg
    for b in range(-18, 94, 3):
        n += 1
        for k in contacts({"mcp": a, "pip": b}):
            grid[" / ".join(k)] += 1
            cells[" / ".join(k)].append([a, b])
print("grid", n, dict(grid))
json.dump({"sweeps": res, "grid": {"n": n, "mcp_deg": [-15, 93, 3], "pip_deg": [-18, 93, 3], "pairs": dict(grid), "cells": dict(cells)}}, open("assets/mjcf/experimental/20261013-cad_v1/sphere_contacts.json", "w"))
