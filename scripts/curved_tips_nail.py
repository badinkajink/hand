#!/usr/bin/env python3
"""Fingernail on the quarter-ellipsoid tips (docs/handoff/20261015-curved_tips.md, ledger step 13): a rigid plate on the
tip's flat back reaching past its distal end, and whether the bench tasks touch it.

The nail is a box geom on each tip body (rigid, friction as the scene's): `--thick` mm thick behind the back plane, 10 mm
wide, from the mount face to `--over` mm past the distal end. It sits behind the rigid board, so the pads' foundation
(builder v2, the TPU between each pad and the back plane) does not change. Each run replays one deployed hand's bench plan
(D1-D8, servo-limited plant, mu 1.0, the 1 mm pads at the planner's numerics, as scripts/curved_tips_traj.py) and counts,
every 1 ms step, the nail's contacts with the tool, with the floor or the post, and with the other fingers, with the
largest normal force of each.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/curved_tips_nail.py --tips tcd_qell7.4x8.5x11x22_u0.1 --over 1.5 --seeds 0 1 2
Rows: docs/experiments/20261015-curved_tips/nail/nail.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curved_tips_common as C  # noqa: E402
import fingertip_geometry as G  # noqa: E402
import pad_usability_backends as PUB  # noqa: E402
import reorient_backends as RB  # noqa: E402
import tip_codesign_geometry as TG  # noqa: E402

OUT = C.D / "nail" / "nail.jsonl"
PUB.SCN = C.LOGS / "scenes"


def add_nail(scene: Path, tip: str, over_mm: float, thick_mm: float) -> Path:
    sh = TG.shape(tip)
    root = ET.parse(scene).getroot()
    x1, x0 = sh.xc, sh.xc - thick_mm * 1e-3
    z0, z1 = TG.Z_END - over_mm * 1e-3, TG.Z_MOUNT
    for f in RB.FINGERS:
        sgn = G.FACE_SIGN[f]
        tipb = root.find(f".//body[@name='{f}_tip']")
        cx = sgn * 0.5 * (x0 + x1)
        ET.SubElement(tipb, "geom", name=f"{f}_nail", type="box",
                      size=f"{0.5 * (x1 - x0):.6f} 0.005 {0.5 * (z1 - z0):.6f}", pos=f"{cx:.6f} 0 {0.5 * (z0 + z1):.6f}",
                      rgba="0.95 0.92 0.85 1", condim="3", mass="0")
    out = scene.with_name(scene.stem + f"_nail{over_mm:g}.xml")
    ET.ElementTree(root).write(out)
    return out


def run(scene, meta, plan, traj, seed):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    tb = m.body(RB.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    tool7 = np.asarray(meta["tool7"], float)
    dx, dy, dyaw = RB.jitter(seed)
    d.qpos[qa:qa + 3] = tool7[:3] + [dx, dy, 0.0]
    d.qpos[qa + 3:qa + 7] = RB._yaw_quat(tool7[3:], dyaw)
    act = {}
    for f in RB.FINGERS:
        for j in RB.JOINTS:
            n = f"{f}_{j}"
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = meta["q0"][n]
            act[n] = m.actuator(f"a_{n}").id
    mujoco.mj_forward(m, d)
    nails = {m.geom(f"{f}_nail").id: f for f in RB.FINGERS}
    tool = {i for i in range(m.ngeom) if m.geom_bodyid[i] == tb}
    fingerb = {}
    for i in range(m.ngeom):
        bn = m.body(m.geom_bodyid[i]).name
        for f in RB.FINGERS:
            if bn.startswith(f + "_"):
                fingerb[i] = f
    f6 = np.zeros(6)
    tally = {"tool": [0, 0.0], "env": [0, 0.0], "finger": [0, 0.0]}
    steps = 0
    for t_end, tgt, _ in RB.schedule(plan, traj):
        for n_, a in act.items():
            d.ctrl[a] = tgt[n_]
        while d.time < t_end - 1e-9:
            mujoco.mj_step(m, d)
            steps += 1
            hit = {"tool": 0.0, "env": 0.0, "finger": 0.0}
            for i in range(d.ncon):
                g1, g2 = int(d.contact[i].geom[0]), int(d.contact[i].geom[1])
                if g1 not in nails and g2 not in nails:
                    continue
                o = g2 if g1 in nails else g1
                kind = "tool" if o in tool else ("finger" if o in fingerb else "env")
                mujoco.mj_contactForce(m, d, i, f6)
                hit[kind] = max(hit[kind], float(f6[0]))
            for k, v in hit.items():
                if v > 1e-6:
                    tally[k][0] += 1
                    tally[k][1] = max(tally[k][1], v)
            if not np.all(np.isfinite(d.qpos)):
                break
    return {"steps": steps, **{f"nail_{k}_steps": v[0] for k, v in tally.items()},
            **{f"nail_{k}_max_N": round(v[1], 4) for k, v in tally.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tips", nargs="+", required=True)
    ap.add_argument("--over", type=float, default=1.5)
    ap.add_argument("--thick", type=float, default=0.6)
    ap.add_argument("--hands", nargs="+", default=list(RB.HANDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for tip in a.tips:
        for hand in a.hands:
            plan, traj = RB.load_plan(hand)
            scene, meta = PUB.build(hand, tip, "pads", C.SERVO, "bed", 100.0, "1.0")
            ns = add_nail(Path(scene), tip, a.over, a.thick)
            for s in a.seeds:
                t0 = time.time()
                r = run(ns, meta, plan, traj, s)
                row = {"tip": tip, "hand": hand, "seed": s, "over_mm": a.over, "thick_mm": a.thick, **r,
                       "scene": str(ns.relative_to(ROOT)), "script": "scripts/curved_tips_nail.py",
                       "wall_s": round(time.time() - t0, 1), "when": time.strftime("%Y-%m-%d %H:%M")}
                PUB.append_row(OUT, row)
                print(f"{tip} {hand} s{s}: nail on tool {r['nail_tool_steps']} steps (max {r['nail_tool_max_N']} N), "
                      f"env {r['nail_env_steps']}, fingers {r['nail_finger_steps']} of {r['steps']}", flush=True)


if __name__ == "__main__":
    main()
