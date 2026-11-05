#!/usr/bin/env python3
"""The rod's sweeps against their tip's body with the rod as a registration fitted it (lab, 17:47 on 11-03: seat 2's
fit turned the rod -5.40 deg about z_M and 0.77 deg across, its c_M 1.17 mm off the plan's). The program aims the fitted
c_M at the planned place and keeps the plan's R_PM, so against the tip the rod is the planned one turned about its axis
point. Per sweep: the exact distance of the rod's solid to the tip body (ur5_inverted_rig.body_dist, negative inside) at
the deep pose (max_depth_mm past the expected contact), each azimuth's largest offset and each slide's end, for the
nominal rod and the fitted turn with the across turn of either sign.

--joint JSON (ur5_rod_joint_registration.py, 11-03 23:5x): the rod as the joint fit found it, against what the seat's own
fit aimed: turned about z_M and across as fitted (the fit's order, about z_M after across), its axis point moved by the
joint c_M less the own, and the pad moved by the joint T_AP against the own; also each corner of the turns at
--joint-sd sd.

    python3 scripts/ur5/ur5_rod_fitted_margins.py SEAT [--turn-z DEG] [--across DEG] [--first ID,ID] [--out OUT]
"""
import argparse
import json, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import ur5_inverted_rig as RIG
C = RIG.C
RIG.PLATE[0] = "block"
RIG.OUT, RIG.STEM = RIG.BLOCK_OUT, RIG.BLOCK_STEM
RIG.use_cell("table", None)
D = Path("docs/experiments/20261030-inverted_rig_seat_block/data")
ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
ap.add_argument("seat", type=int, help="1, 2 or 3")
ap.add_argument("--turn-z", default="-5.40", help="the fit's turn about z_M (deg); a comma-separated list for an envelope "
                "(11-03: seats 2 and 3 fitted -5.40 and +10.75 on the same rod)")
ap.add_argument("--across", type=float, default=0.77, help="the fit's turn across (deg), tried with either sign")
ap.add_argument("--first", default="", help="sweep ids printed first")
ap.add_argument("--plan", default="sweep8", choices=("sweep8", "extras"), help="the program plan: the 8 N sweeps or "
                "the night's extras (each press's deep pose, each roll's turns)")
ap.add_argument("--plan-file", default="", help="a program plan in place of --plan's draft (the lab's halves)")
ap.add_argument("--ids", default="", help="only these actions (comma-separated)")
ap.add_argument("--joint", default="", help="a joint rod registration: the rod and pad as it fitted them")
ap.add_argument("--joint-sd", type=float, default=2.0, help="the corners of the joint turns at this many sd")
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
for tz in [float(v) for v in str(a.turn_z).split(",")]:
    for tl in (a.across, -a.across):
        var[f"fit z{tz:+g} x{tl:+g}"] = C.rot([1, 0, 0], tl) @ C.rot([0, 0, 1], tz)
Tf = {k: C.Tm(R, c0 - R @ c0) for k, R in var.items()}
Tpad = {k: np.eye(4) for k in Tf}                       # the pad's error in P: the true pad against the aimed one
if a.joint:
    J = json.loads(Path(a.joint).read_text())
    jr = next(r for r in J["rows"] if r["seat"] == s_)
    reg_o = json.loads((RIG.ROOT / next(x for x in J["sessions"] if x.rstrip("/").endswith(jr["session"]))
                        / "registration.json").read_text())
    dc = np.asarray(jr["joint"]["c_M"], float) - np.asarray(reg_o["c_M"], float)
    dP = C.inv(np.asarray(jr["T_AP_joint"], float)) @ np.asarray(reg_o["T_AP"], float)
    (t0, t1), (s0, s1) = J["turn_deg"], J["turn_sd_deg"]
    k_ = a.joint_sd
    for name, z_, x_ in [("joint", t0, t1)] + [(f"joint z{t0 + i * k_ * s0:+.2f} x{t1 + j * k_ * s1:+.2f}",
                                                t0 + i * k_ * s0, t1 + j * k_ * s1) for i in (-1, 1) for j in (-1, 1)]:
        R = C.rot([0, 0, 1], z_) @ C.rot([1, 0, 0], x_)
        Tf[name], Tpad[name] = C.Tm(R, c0 - R @ c0 + dc), dP
    print(f"joint: seat {s_} {jr['session']}: c_M joint less own {np.round(dc, 3).tolist()} mm; the pad "
          f"{jr['joint']['T_AP_vs_own_mm_deg']} mm, deg off the own; turns {t0:+.2f} (sd {s0:.2f}), {t1:+.2f} (sd {s1:.2f})",
          flush=True)
T_BP_plan = ctx["T_BA"] @ T_AP
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
        W = T_BP_plan @ Tpad[k] @ C.inv(T_BP_plan)
        d = RIG.body_dist([W @ T @ T_EM @ T_fit @ T_ME for T in poses], ctx, seat, "rod")
        j = int(np.argmin(d))
        row[k] = dict(deep=round(float(d[0]), 2), least=round(float(d[j]), 2), at=moves[j][0],
                      under=[(moves[i][0], round(float(d[i]), 2)) for i in np.argsort(d)[:4] if d[i] < 1.0])
    out.append(row)
    w = min(row[k]["least"] for k in Tf if k != "nominal")
    if w < 1.0 or x["id"] in first:
        print(json.dumps(row), flush=True)
Path(a.out or D / f"20261103-strip_table_pads_rod_{a.plan}_fitted_margins_seat{s_}.json").write_text(
    json.dumps(dict(turn_z_deg=str(a.turn_z), across_deg=a.across, joint=a.joint or None, joint_sd=a.joint_sd,
                    rows=out), indent=1) + "\n")
print("done", len(out), "actions")
