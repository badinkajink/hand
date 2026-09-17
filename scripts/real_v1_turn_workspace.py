#!/usr/bin/env python3
"""Which grasps leave the fingers room to rotate the tool: a kinematic map.

    python3 scripts/real_v1_turn_workspace.py --hand sv1_u0308_b050 --out DIR

For each grasp (palm depth above the shaft centre x pad-ring elevation x straddle) the fitter
authors the grip on the rigid design scene; from that grip pose the contact-fixed rotation's
tip targets (each tip rotated about a pivot between the driver pads) are solved by per-finger
IK, and the residual is the distance the finger cannot close. A rotation is REACHABLE when
every finger's residual is under `--tol` mm. The chain's own grasp (depth 62.5, elevation 0,
straddle 40 on D6) leaves 6 mm of residual at 10 deg -- no rotation at all. This map finds the
grasps that leave some, per hand, before any dynamics enter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FINGERS = {"thumb": ("thumb_yaw", "thumb_mcp", "thumb_pip"),
           "index": ("index_yaw", "index_mcp", "index_pip"),
           "middle": ("middle_yaw", "middle_mcp", "middle_pip")}
TIPS = {"thumb": "thumb_tip", "index": "index_tip", "middle": "middle_tip"}
ANGLES = (-10, -20, -30, -45, -60, -75, -90)


def grasp_pose(flat: Path, straddle: float, depth: float, elevation: float, thumb_axial: float,
               squeeze: float):
    import mujoco
    import probe_real_v1_carry as pc
    built = pc._grip_from_fit(flat, straddle, 0.0, squeeze, "screwdriver_medium", depth,
                              thumb_axial, elevation)
    if built is None:
        return None
    m, open_qpos, grip_ctrl, depth_mm = built
    d = mujoco.MjData(m)
    # the GRIP pose on the rigid model: joints at the grip command (this is what a rigid plant
    # achieves; the compliant one sits a few degrees straighter under load)
    d.qpos[:] = open_qpos
    for a in range(m.nu):
        jid = m.actuator_trnid[a, 0]
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = grip_ctrl[a]
    mujoco.mj_forward(m, d)
    return m, d, depth_mm


def arcs(m, d, tol_mm: float):
    from morphohand.tools.keyframe_ik import ik_finger
    import mujoco
    tip0 = {f: d.body(TIPS[f]).xpos.copy() for f in FINGERS}
    cen = np.mean([tip0[f] for f in FINGERS], axis=0)
    span = abs(tip0["index"][1] - tip0["middle"][1]) / 2.0
    q0 = d.qpos.copy()
    out = {}
    for k in (0.0, 0.3, 0.6, 1.0):
        c = cen.copy(); c[2] += k * span
        row = {}
        for deg in ANGLES:
            a = np.radians(deg)
            Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
            res = {}
            for f in FINGERS:
                d.qpos[:] = q0
                res[f] = ik_finger(m, d, f, c + Rx @ (tip0[f] - c), iters=300) * 1e3
            row[deg] = {f: round(v, 2) for f, v in res.items()}
        d.qpos[:] = q0
        mujoco.mj_forward(m, d)
        reach = max((abs(deg) for deg in ANGLES if max(row[deg].values()) <= tol_mm), default=0)
        out[f"k{k:g}"] = {"reach_deg": reach, "res": row}
    q = {j: round(float(np.degrees(d.qpos[m.jnt_qposadr[m.joint(j).id]])), 1) for js in FINGERS.values() for j in js}
    return {"span_mm": round(span * 1e3, 1), "q_grip_deg": q, "pivots": out,
            "reach_deg": max(v["reach_deg"] for v in out.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", default="sv1_u0308_b050")
    ap.add_argument("--depths", default="62.5,58,54,50,46")
    ap.add_argument("--elevations", default="0,-15,-30,-45")
    ap.add_argument("--straddles", default="40,30,20")
    ap.add_argument("--squeeze-mm", type=float, default=10.0)
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import real_v1_chain_hands as H
    h = next(x for x in H.hands("AB") if x["tag"] == a.hand)
    fit0 = H.prepare(h, squeeze_mm=a.squeeze_mm)      # makes the flat scene
    flat = H.SCENES / f"{h['tag']}_sq{a.squeeze_mm:g}__flat.xml"
    if not flat.exists():
        flat = H.SCENES / f"{h['tag']}__flat.xml"
    rows = []
    for dep in (float(v) for v in a.depths.split(",")):
        for el in (float(v) for v in a.elevations.split(",")):
            for st in (float(v) for v in a.straddles.split(",")):
                g = grasp_pose(flat, st / 1000, dep / 1000, el, h["thumb_axial"], a.squeeze_mm / 1000)
                row = {"hand": a.hand, "depth_mm": dep, "elevation_deg": el, "straddle_mm": st}
                if g is None:
                    row["fit"] = False
                    print(f"depth {dep:5.1f} elev {el:5.0f} straddle {st:3.0f}: no grasp fits")
                else:
                    m, d, depth_mm = g
                    row.update({"fit": True, "depth_fit_mm": depth_mm, **arcs(m, d, a.tol)})
                    ks = " ".join(f"k{k}:{row['pivots'][f'k{k}']['reach_deg']:>3}" for k in ("0", "0.3", "0.6", "1"))
                    print(f"depth {dep:5.1f} elev {el:5.0f} straddle {st:3.0f}: fit at {depth_mm:5.1f}  reach {row['reach_deg']:>3} deg  [{ks}]  "
                          f"mcp ix/md {row['q_grip_deg']['index_mcp']:5.1f}/{row['q_grip_deg']['middle_mcp']:5.1f}")
                rows.append(row)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.hand}_workspace.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out / (a.hand + '_workspace.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
