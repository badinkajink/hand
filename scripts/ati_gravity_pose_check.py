#!/usr/bin/env python3
"""Which gravity stations of a UR5 calibration plan are safe with the ATI stack and the hook cross on the tool (the ATI
calibration plan of 2026-10-23, docs/experiments/20261023-ati_ft_calibration/): the tool's points (mount v3's adapter, the
ATI base, the Mini45, the ATI top, the hook cross; scripts/ati_stack_mount.py) against the robot's own forearm and wrist
links, the HEX-E's cable port and the cell (indenters, 2x4, table), at each station's joints as the plan recorded them.
The masses hanging from the hooks are not modelled.

    .venv/bin/python scripts/ati_gravity_pose_check.py <plan.json> [--out check.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ati_stack_mount as AS  # noqa: E402
import ur5_cb3_cell as CELL  # noqa: E402
import ur5_tip_calibration as C  # noqa: E402

D = ROOT / "docs/experiments/20261023-ati_ft_calibration"


def stack_points(spacing=2.0):
    import trimesh
    parts = [(CELL.MOUNT / f"{CELL._MOUNT_STEM}_adapter.stl", 0.0),
             (D / "cad/20261023-ati_stack_ati_base.stl", 15.0),
             (None, 15.0 + AS.BASE_T),
             (D / "cad/20261023-ati_stack_ati_top.stl", 15.0 + AS.BASE_T + AS.ATI_H),
             (D / "cad/20261023-ati_stack_hook_cross.stl", 15.0 + AS.BASE_T + AS.ATI_H + AS.TOP_T)]
    out = []
    for f, dz in parts:
        if f is None:
            m = trimesh.creation.cylinder(radius=AS.ATI_D / 2, height=AS.ATI_H, sections=48)
            m.apply_translation([0.0, 0.0, AS.ATI_H / 2])
        else:
            m = trimesh.load(f, force="mesh")
        m.apply_translation([0.0, 0.0, dz])
        p, _ = trimesh.sample.sample_surface_even(m, int(max(300, m.area / spacing ** 2)), seed=0)
        out.append(p)
    return np.vstack(out)


def main(plan_path, out=None):
    plan = json.loads(Path(plan_path).read_text())
    ck = plan["checks"]
    CELL.BOARD.update(ck.get("board") or {})
    CELL.OTHER[:] = [(np.array(o["T_BI"], float), o["kind"]) for o in ck.get("other", [])]
    x, y, z, yaw = ck["placement_mm_deg"]
    T_BI = C.Tm(C.rot(C.EZ, yaw), [x, y, z])
    T_EM = np.array(plan["geometry"]["T_EM"], float)
    P = stack_points()
    CELL.tool_points = lambda spacing=2.0: {"stack": P, "hex_e": CELL.hex_e_mesh(None).sample(1500)}   # noqa: E731
    CELL._self_tool_points = lambda: (CELL.hex_e_mesh(None).sample(800), P[::4])                       # noqa: E731
    rows = []
    for a in plan["actions"]:
        if a["stage"] not in ("gravity", "final_gravity") or not a.get("check"):
            continue
        q = np.radians(a["check"][0]["q_deg"])
        sc = CELL.self_clearance(q, T_EM)
        T_BE = CELL.link_frames(q)["tool0"]
        cl = CELL.clearance(q, T_BE, T_BI, T_EM, plan["indenter"])
        g_M = (T_BE @ T_EM)[:3, :3].T @ np.array([0.0, 0.0, -1.0])
        rows.append(dict(id=a["id"], q_deg=a["check"][0]["q_deg"], gravity_in_M=np.round(g_M, 3).tolist(),
                         self_tool_mm=round(sc["tool"], 1), self_port_mm=round(sc["port"], 1),
                         cell_stack_mm=round(cl["stack"], 1), cell_hex_e_mm=round(cl["hex_e"], 1)))
        r = rows[-1]
        ok = r["self_tool_mm"] >= 15 and r["self_port_mm"] >= 15 and r["cell_stack_mm"] >= 10
        r["ok"] = bool(ok)
        print(f"{a['id']:10s} gravity in M {r['gravity_in_M']}: stack to the arm {r['self_tool_mm']:6.1f} mm, port "
              f"{r['self_port_mm']:6.1f}, stack to the cell {r['cell_stack_mm']:6.1f} -> {'ok' if ok else 'NO'}", flush=True)
    G = np.array([r["gravity_in_M"] for r in rows if r["ok"]])
    sv = np.linalg.svd(G, compute_uv=False) if len(G) else []
    print(f"{sum(r['ok'] for r in rows)} of {len(rows)} stations clear; spread of their gravity directions (singular values) "
          f"{np.round(sv, 2).tolist()}")
    res = dict(plan=str(plan_path), rows=rows, ok=int(sum(r["ok"] for r in rows)),
               gravity_singular_values=np.round(sv, 3).tolist(),
               note="self: the stack and cross to the forearm and wrist links (need 15 mm) and the HEX-E's port; cell: to the "
                    "indenters, 2x4 and table (need 10 mm); the hanging masses not modelled")
    if out:
        Path(out).write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan")
    ap.add_argument("--out", default=str(D / "20261023-ati_gravity_pose_check.json"))
    a = ap.parse_args()
    main(a.plan, a.out)
