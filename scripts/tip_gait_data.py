#!/usr/bin/env python3
"""Collect the 2026-09-16 bench partial-reorientation study into one JSON for the page.

    python3 scripts/tip_gait_data.py
    python3 scripts/tip_gait_page.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
E = ROOT / "docs/experiments/20260916-tip_gait"
OUT = E / "tip_gait_data.json"
HW = {"D1": ("6/7", 51.9), "D2": ("10/10", 48.2), "D3": ("4/10", 42.1), "D4": ("9/10", 52.8),
      "D5": ("3/10", 69.6), "D6": ("2/10", 35.5), "D7": ("10/10", 33.3), "D8": ("4/10", 44.2)}


def deg(c):
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def contact_trace():
    """Where each pad touches the shaft through the D7 tip on `cal` (plate at 0): axial mm
    from the tool's centre, normal force, and the tool's shaft direction."""
    import mujoco
    from real_v1_tip_gait import plant_scene
    from real_v1_turn_mechanism import FINGERS, JOINTS, OBJECT
    plan_p = ROOT / "docs/experiments/20260902-residual-bench/deploy/rv05_manual_b85_plan.json"
    plan = json.loads(plan_p.read_text())
    traj = plan_p.with_name("rv05_manual_b85_traj.csv")
    scene = plant_scene(Path(plan["meta"]["scene"]), E / "scenes", "rv05_manual", "cal")
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    poses = {p["name"]: p["joints"] for p in plan["poses"]}
    aid = {(f, j): mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{f}_{j}") for f in FINGERS for j in JOINTS}
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJECT)
    obj = {i for i in range(m.ngeom) if m.geom_bodyid[i] == bid}
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm_pose")
    who = {}
    for i in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
        for f in FINGERS:
            if bn.startswith(f):
                who[i] = f
        if m.geom_bodyid[i] == pb:
            who[i] = "plate"
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "") == "tool_post":
            who[i] = "post"
    d.qpos[:] = plan["meta"]["replay_initial_qpos"]
    d.ctrl[:] = plan["meta"]["replay_base_ctrl"]
    mujoco.mj_forward(m, d)

    def hold(pose, s):
        for f in FINGERS:
            for j in JOINTS:
                d.ctrl[aid[(f, j)]] = np.deg2rad(pose[f][j])
        for _ in range(int(s / m.opt.timestep)):
            mujoco.mj_step(m, d)

    def snap(phase):
        R = d.xmat[bid].reshape(3, 3)
        ax = R[:, 2]
        c = d.xpos[bid]
        out = {}
        c6 = np.zeros(6)
        for i in range(d.ncon):
            ct = d.contact[i]
            if ct.geom1 in obj or ct.geom2 in obj:
                other = ct.geom2 if ct.geom1 in obj else ct.geom1
                k = who.get(other, "other")
                mujoco.mj_contactForce(m, d, i, c6)
                s = float((ct.pos - c) @ ax)
                prev = out.get(k)
                fn = float(abs(c6[0]))
                if prev is None or fn > prev["N"]:
                    out[k] = {"axial_mm": round(s * 1e3, 1), "N": round(fn, 2)}
        return {"phase": phase, "t": round(float(d.time), 2), "cos": round(float(R[2, 2]), 3),
                "from_vertical_deg": round(deg(R[2, 2]), 1), "z_mm": round(float(c[2] * 1e3), 1),
                "contacts": out}

    rows = []
    hold(poses["grip"], 0.8)
    rows.append(snap("grip"))
    tr = list(csv.DictReader(open(traj)))
    span = float(tr[-1]["t_s"])
    for k, r in enumerate(tr):
        hold({f: {j: float(r[f"{f}_{j}_deg"]) for j in JOINTS} for f in FINGERS}, span / len(tr))
        if k in (50, 100, 150, 199):
            rows.append(snap(f"turn {k + 1}/{len(tr)}"))
    hold({f: {j: float(tr[-1][f"{f}_{j}_deg"]) for j in JOINTS} for f in FINGERS}, 1.5)
    rows.append(snap("rest"))
    return rows


def main() -> int:
    plate = json.load(open(E / "plate_replay.json"))
    for r in plate:
        r["from_vertical_deg"] = round(deg(r["cos_end"]), 1)
        r["lost_at_grip"] = bool(r["z_grip"] < 0.06)
        r["held"] = bool((not r["dropped"]) and r["n_pad_end"] >= 2 and not r["lost_at_grip"])
    angle = json.load(open(E / "plan_angle_sweep.json"))
    for r in angle:
        r["from_vertical_deg"] = round(deg(r["cos_end"]), 1)
    gait = json.load(open(E / "tip_gait.json"))
    for r in gait:
        r.pop("trace", None)
    data = {"plate": plate, "angle": angle, "gait": gait, "trace_d7": contact_trace(), "hw": HW,
            "n": {"plate": len(plate), "angle": len(angle), "gait": len(gait)}}
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT}: {data['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
