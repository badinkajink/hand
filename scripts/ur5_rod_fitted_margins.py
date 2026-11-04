#!/usr/bin/env python3
"""The rod's sweeps against their tip's body with the rod as a registration fitted it (lab, 17:47 on 11-03: seat 2's
fit turned the rod -5.40 deg about z_M and 0.77 deg across, its c_M 1.17 mm off the plan's). The program aims the fitted
c_M at the planned place and keeps the plan's R_PM, so against the tip the rod is the planned one turned about its axis
point. Per sweep: the exact distance of the rod's solid to the tip body (ur5_inverted_rig.body_dist, negative inside) at
the deep pose (max_depth_mm past the expected contact), each azimuth's largest offset and each slide's end, for the
nominal rod and the fitted turn with the across turn of either sign.

    python3 scripts/ur5_rod_fitted_margins.py SEAT [--turn-z DEG] [--across DEG] [--first ID,ID] [--out OUT]
"""
import argparse
import json, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ur5_inverted_rig as RIG
C = RIG.C
RIG.PLATE[0] = "block"
RIG.OUT, RIG.STEM = RIG.BLOCK_OUT, RIG.BLOCK_STEM
RIG.use_cell("table", None)
D = Path("docs/experiments/20261030-inverted_rig_seat_block/data")
ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
ap.add_argument("seat", type=int, help="1, 2 or 3")
ap.add_argument("--turn-z", type=float, default=-5.40, help="the fit's turn about z_M (deg)")
ap.add_argument("--across", type=float, default=0.77, help="the fit's turn across (deg), tried with either sign")
ap.add_argument("--first", default="", help="sweep ids printed first")
ap.add_argument("--plan", default="sweep8", choices=("sweep8", "extras"), help="the program plan: the 8 N sweeps or "
                "the night's extras (each press's deep pose, each roll's turns)")
ap.add_argument("--plan-file", default="", help="a program plan in place of --plan's draft (the lab's halves)")
ap.add_argument("--ids", default="", help="only these actions (comma-separated)")
ap.add_argument("--out", default="")
a = ap.parse_args()
s_, v_ = a.seat, 7 - a.seat
first = [f for f in a.first.split(",") if f]
prog = json.loads((Path(a.plan_file) if a.plan_file else
                   D / f"20261103-plan_probe_table_rod_seat{s_}_v3id{v_}_{a.plan}_draft.json").read_text())
src = json.loads((RIG.ROOT / prog["checks"]["source"]).read_text())
seat = int(src["seat"])
RIG.HOME_KEY[0] = int(prog["homing"]["key"])
ctx = RIG.context(dict(orient=src["orient"], seat_h=RIG.SEAT_BLOCK["seat_h"]), tool=src["tool"])
ctx["seat"] = seat
T_AP = np.asarray(prog["geometry"]["seat"]["T_AP"], float)
c0 = np.array([0.0, 0.0, RIG.tool_end_z("rod")])
T_EM, T_ME = ctx["T_EM"], C.inv(ctx["T_EM"])
var = {"nominal": np.eye(3)}
for tl in (a.across, -a.across):
    var[f"fit z{a.turn_z:+g} x{tl:+g}"] = C.rot([1, 0, 0], tl) @ C.rot([0, 0, 1], a.turn_z)
Tf = {k: C.Tm(R, c0 - R @ c0) for k, R in var.items()}
acts = [x for x in prog["actions"] if x["stage"] == "grid" and x["kind"] in ("sweep", "press", "roll", "twist")
        and (not a.ids or x["id"] in a.ids.split(","))]
acts.sort(key=lambda x: (x["id"] not in first, prog["actions"].index(x)))
grid = [x["id"] for x in prog["actions"] if x["stage"] == "grid"]
out = []
for x in acts:
    st = float(x.get("standoff", 1.5))
    T_r, d_B, n_B = RIG.program_rest(prog, ctx, x, st)
    T_e = C.Tm(T_r[:3, :3], T_r[:3, 3] + (st + float(x["max_depth_mm"])) * d_B)
    if x["kind"] == "sweep":
        t1_B = ctx["T_BA"][:3, :3] @ T_AP[:3, :3] @ np.asarray(x["target"]["t1"], float)
        t2_B = np.cross(n_B, t1_B)
        far = {}
        for lev in x["offsets"]:
            for r_, z in lev:
                far[z] = max(far.get(z, 0.0), r_)
        moves = [("deep", 0.0, 0.0)] + [(f"offset az {z:g}", z, r_) for z, r_ in far.items()] + \
            [(f"slide az {sl['az_deg']:g}", sl["az_deg"], sl["mm"]) for sl in x["slides"] if sl["at_N"] == x["slides"][0]["at_N"]]
        poses = [C.Tm(T_e[:3, :3], T_e[:3, 3] + r_ * (math.cos(math.radians(z)) * t1_B + math.sin(math.radians(z)) * t2_B))
                 for _, z, r_ in moves]
    elif x["kind"] == "roll":                           # about the plan's axis (the fitted one is within the turn)
        poses = [T_e] + RIG.roll_poses(ctx, dict(roll_degs=x["moves"][0]["degs"]), T_e, n_B)
        moves = [("deep", 0, 0)] + [(f"roll pose {k}", 0, 0) for k in range(len(poses) - 1)]
    elif x["kind"] == "twist":                          # about n through the axis point, each degree to the largest
        c_B = C.apply(T_e @ T_EM, c0)
        degs = [d for mv in x["moves"] for d in mv["degs"]]
        lo, hi = min(degs + [0.0]), max(degs + [0.0])
        ang = [float(g) for g in np.arange(math.floor(lo), math.ceil(hi) + 1.0, 1.0)]
        poses = [T_e] + [C.turn_about(T_e, c_B, n_B, g) for g in ang]
        moves = [("deep", 0, 0)] + [(f"twist {g:+.0f}", 0, 0) for g in ang]
    else:
        poses, moves = [T_e], [("deep", 0, 0)]
    row = dict(id=x["id"], grid=grid.index(x["id"]) + 1)
    for k, T_fit in Tf.items():
        d = RIG.body_dist([T @ T_EM @ T_fit @ T_ME for T in poses], ctx, seat, "rod")
        j = int(np.argmin(d))
        row[k] = dict(deep=round(float(d[0]), 2), least=round(float(d[j]), 2), at=moves[j][0],
                      under=[(moves[i][0], round(float(d[i]), 2)) for i in np.argsort(d)[:4] if d[i] < 1.0])
    out.append(row)
    w = min(row[k]["least"] for k in Tf if k != "nominal")
    if w < 1.0 or x["id"] in first:
        print(json.dumps(row), flush=True)
Path(a.out or D / f"20261103-strip_table_pads_rod_{a.plan}_fitted_margins_seat{s_}.json").write_text(
    json.dumps(dict(turn_z_deg=a.turn_z, across_deg=a.across, rows=out), indent=1) + "\n")
print("done", len(out), "actions")
