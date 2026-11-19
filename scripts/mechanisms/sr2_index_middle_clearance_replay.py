#!/usr/bin/env python3
"""Index-middle clearance of the SR2 hand over the logged bench trials, for the hand as built and two actuator layouts.

Every second control period of the HOM bench trials (docs/experiments/20261115-hom_bench_trials/runs/*.jsonl) is put
on the cad_v3 model (assets/mjcf/experimental/20261104-cad_v3/scene_screwdriver_medium.xml): the mounts are the plan's
d6_P_mm, the joints the servo readings mapped by the driver's JOINT_SIGN (manta_hand.plan). The clearance is the least
distance between the index's and the middle's collision spheres (MorphIt-B, larger than the printed parts), so a
negative value is an overlap of the spheres.

Variants:
  as_built   the spheres as modelled.
  slim       the PIP servo moved off the finger: spheres of the DIP link, and of the MCP link more than 12 mm below the
             MCP axis, kept only inside |y| <= 10.5 mm of the finger's plane (the sensor cassette's half-width).
  sts_inner  STS3032 servos with their 3.7 mm of extra case depth on the middle finger's inner (+y, servo base) side:
             the middle's yaw-link, MCP-link and DIP-link spheres on the +y half moved 3.7 mm along +y.

Usage (a mesh load, so through resguard):
  ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/mechanisms/sr2_index_middle_clearance_replay.py
Writes docs/experiments/20261118-sr2_actuator_layout/20261118-index_middle_clearance_replay.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

import ast  # noqa: E402
import collections  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "assets/mjcf/experimental/20261104-cad_v3/scene_screwdriver_medium.xml"
RUNS = ROOT / "docs/experiments/20261115-hom_bench_trials/runs"
OUT = ROOT / "docs/experiments/20261118-sr2_actuator_layout/20261118-index_middle_clearance_replay.json"
FINGERS = ("index", "middle")
SIGN = {(f, j): s for f in FINGERS for j, s in (("yaw", -1.0), ("mcp", 1.0), ("pip", 1.0))}  # manta_hand.plan
SLAB_MM, MCP_CLEAR_MM, STS_EXTRA_MM = 10.5, 12.0, 3.7
VARIANTS = ("as_built", "slim", "sts_inner")
STRIDE = 2


def spheres(m):
    out = {}
    for f in FINGERS:
        rows = []
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if (nm.startswith(f + "_") and "_s" in nm and not nm.startswith(f + "_mount")
                    and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE):
                rows.append((g, nm[len(f) + 1:nm.rindex("_s")], m.geom_pos[g] * 1e3, m.geom_size[g, 0] * 1e3))
        out[f] = rows
    return out


def kept(variant, part, p, r):
    if variant != "slim":
        return True
    if part == "dip_link" or (part == "mcp_frame" and p[2] < -MCP_CLEAR_MM):
        return abs(p[1]) + r <= SLAB_MM
    return True


def offset(variant, finger, part, p):
    if variant == "sts_inner" and finger == "middle" and part in ("yaw_frame", "mcp_frame", "dip_link") and p[1] > 0:
        return np.array([0.0, STS_EXTRA_MM, 0.0])
    return np.zeros(3)


def main() -> int:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    qadr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]  # noqa: E731
    mount0 = {f: m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{f}_mount")] * 1e3 for f in FINGERS}
    S = spheres(m)
    sel = {v: {f: [k for k, (g, part, p, r) in enumerate(S[f]) if kept(v, part, p, r)] for f in FINGERS}
           for v in VARIANTS}
    local = {v: {f: np.array([S[f][k][2] + offset(v, f, S[f][k][1], S[f][k][2]) for k in sel[v][f]]) for f in FINGERS}
             for v in VARIANTS}
    body = {v: {f: np.array([m.geom_bodyid[S[f][k][0]] for k in sel[v][f]]) for f in FINGERS} for v in VARIANTS}
    rad = {v: {f: np.array([S[f][k][3] for k in sel[v][f]]) for f in FINGERS} for v in VARIANTS}
    part = {v: {f: [S[f][k][1] for k in sel[v][f]] for f in FINGERS} for v in VARIANTS}

    res = {v: [] for v in VARIANTS}
    close = {v: collections.Counter() for v in VARIANTS}
    per_run, plans = [], {}
    for path in sorted(glob.glob(str(RUNS / "*.jsonl"))):
        lines = Path(path).read_text().splitlines()
        plan = json.loads(lines[0]).get("plan")
        if plan not in plans:
            try:
                plans[plan] = json.loads((ROOT / plan).read_text())["meta"]["hom"]["d6_P_mm"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                plans[plan] = None
        d6 = plans[plan]
        if d6 is None:
            continue
        xy = {"index": d6[2:4], "middle": d6[4:6]}
        run_min = {v: math.inf for v in VARIANTS}
        for k, line in enumerate(lines[1:]):
            if k % STRIDE:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            j = rec.get("joints")
            if not j:
                continue
            if isinstance(j, str):
                j = ast.literal_eval(j)
            d.qpos[:] = m.qpos0
            for f in FINGERS:
                d.qpos[qadr(f"{f}_x")] = (xy[f][0] - mount0[f][0]) * 1e-3
                d.qpos[qadr(f"{f}_y")] = (xy[f][1] - mount0[f][1]) * 1e-3
                for sj, sv in (("yaw", "aa"), ("mcp", "fe1"), ("pip", "fe2")):
                    d.qpos[qadr(f"{f}_{sj}")] = math.radians(j[f][sv] / SIGN[(f, sj)])
            mujoco.mj_kinematics(m, d)
            for v in VARIANTS:
                W = {}
                for f in FINGERS:
                    b = body[v][f]
                    W[f] = d.xpos[b] * 1e3 + np.einsum("nij,nj->ni", d.xmat[b].reshape(-1, 3, 3), local[v][f])
                D = (np.linalg.norm(W["index"][:, None, :] - W["middle"][None, :, :], axis=2)
                     - rad[v]["index"][:, None] - rad[v]["middle"][None, :])
                a, c = np.unravel_index(np.argmin(D), D.shape)
                dm = float(D[a, c])
                res[v].append(dm)
                run_min[v] = min(run_min[v], dm)
                if dm < 5.0:
                    close[v][f"{part[v]['index'][a]} | {part[v]['middle'][c]}"] += 1
        per_run.append({"run": Path(path).name, "plan": Path(plan).name,
                        **{v: round(run_min[v], 2) for v in VARIANTS}})

    summary = {}
    for v in VARIANTS:
        x = np.array(res[v])
        summary[v] = {"periods": int(x.size), "min_mm": round(float(x.min()), 2),
                      "p1_mm": round(float(np.percentile(x, 1)), 2), "p10_mm": round(float(np.percentile(x, 10)), 2),
                      "median_mm": round(float(np.median(x)), 2), "share_overlap": round(float(np.mean(x < 0)), 4),
                      "share_under_5mm": round(float(np.mean(x < 5)), 4),
                      "closest_parts_under_5mm (index | middle)": close[v].most_common(8)}
        print(v, json.dumps(summary[v]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"script": "scripts/mechanisms/sr2_index_middle_clearance_replay.py",
                               "scene": str(SCENE.relative_to(ROOT)), "runs": str(RUNS.relative_to(ROOT)),
                               "stride": STRIDE, "slab_mm": SLAB_MM, "mcp_clear_mm": MCP_CLEAR_MM,
                               "sts_extra_mm": STS_EXTRA_MM, "summary": summary, "per_run": per_run}, indent=1))
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
