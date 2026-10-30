#!/usr/bin/env python3
"""Program plans for the probe mode of ur5_tip_calibration.py (the inverted rig: a probe on the HEX-E presses a fingertip
held still on its seat) from the workstation's probe plans (scripts/ur5_inverted_rig.py replay, --plate block:
docs/experiments/20261030-inverted_rig_seat_block/data/20261030-seat_block_plan_up_xm_seat{1,2,3}.json and their
_paths_ files).

Frame I of the program is the strip's frame A in probe mode: homing seats the probe's socket (the fork's, v7's) on a key
of the strip, T_IM_home is that key's T_AM_home, and the program's homing then gives T_BI = T_BA. The pad frame P of the
held tip stands at T_AP on the strip (the seat's CAD, refined by the program's probe fit), the probe's sphere centre at
c_M on the tool axis (tool_end_z).

Each action keeps the workstation's choices as directions in P: the pad point p and its outward normal n, the push
direction d (along -n, or theta off it), and the mount's rotation R_PM at the press (the stem and the turn about it that
the replay found for joint 6). The program places the sphere's centre at T_BP (p + R n - standoff d) with the mount at
T_BP's rotation times R_PM, from the registered T_BP and c_M, so it presses the same pad point the same way when the seat
or the probe stands off its CAD. With the plan's own T_BA and T_EM the rest poses come out as the replay's T_rest (the
script checks them). Each action takes the replay's travel (its _paths_ row: straight, joint, joint-liftNN from the
homing, or hub) and presses at most its deep_clear_mm past contact, the deepest press the workstation's check cleared.

Pass 1 (stage pad) is a set of the plan's presses taken as touch-offs, chosen greedily on the probe fit's Jacobian at
the CAD: each added touch-off is the one that most lowers the expected squared miss of every press in the plan (where the
sphere's centre stands in P against where it was aimed), through the fit's posterior covariance. A twist action presses to 3 N, turns, holds 1 s at each angle, and is followed at the same point by the
same twist at 1 N (owner, 13:4x on 10-30: stronger twists; workstation: hold about 1 s for the Tz noise).

    /usr/bin/python3 scripts/ur5_probe_plan.py PLAN [--out OUT] [--pass1 N] [--pass1-standoff MM] [--held-channel K]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_inverted_rig as RIG  # noqa: E402

C = RIG.C
LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
GEOM_PLAN = LAB / "20261029-plan_sphere_lab_pass1_night_v3id3.json"       # the pad's surface, T_ES and the mount's face
KEY_OF_SEAT = {0: 0, 1: 0, 2: 1}                                          # key 1 serves seats 1 and 2, key 2 seat 3
HELD_CHANNEL = {0: 3, 1: 1, 2: 0}   # the held boards' mux channels on b4:3a:45:b0:d9:94 (owner, 14:3x on 10-30, the strip's
#                                     README step 5): seat 1 v3 ID 1 on 3, seat 2 v3 ID 3 on 1, seat 3 v3 ID 2 on 0
TWISTS = ((3.0, ""), (1.0, "-1N"))                                       # (N, id suffix): 3 N first, then 1 N there
TOUCH_STOP_N, TOUCH_FIT_N = 1.0, 0.8     # a probe pushes mostly along its stem, the HEX-E's z, whose noise (0.21 N a sample)
#                                          is five times x's and y's, where the tip mode's touches push: twice its 0.5, 0.4 N
TWIST_HOLD_S = 1.0
PRIOR = np.r_[np.radians([1.0, 1.0, 0.5]), [0.5] * 3, [0.3] * 3]            # the program's REG_SIGMA_PROBE
SIG_PAD = 0.03


def jacobians(acts, T_BA, T_AP, T_EM, c_M, R, geo0):
    """At the CAD, per action (its onset pose along the push): the touch residual's row d r / d x (x: the seat's 6, the
    sphere centre's 3), and the 3 x 9 change of where the sphere's centre stands in P when the truth is off by x and the
    program aims from the CAD (the press's miss)."""
    T_BP = T_BA @ T_AP

    def centre_P(x, T_on):
        return C.apply(C.inv(T_BA @ C.perturb(T_AP, x[:6])) @ T_on @ T_EM, c_M + x[6:9])

    J, G = np.zeros((len(acts), 9)), np.zeros((len(acts), 3, 9))
    for i, a in enumerate(acts):
        tg = a["target"]
        c = C.apply(T_BP, np.asarray(tg["p"], float) + R * np.asarray(tg["n"], float))
        R_BM = T_BP[:3, :3] @ np.asarray(tg["R_PM"], float)
        T_on = C.Tm(R_BM, c - R_BM @ c_M) @ C.inv(T_EM)
        q0 = centre_P(np.zeros(9), T_on)
        r0 = geo0.surface_distance(q0)[0] - R
        for j in range(9):
            dx = np.zeros(9)
            dx[j] = 1e-5
            q = centre_P(dx, T_on)
            J[i, j] = (geo0.surface_distance(q)[0] - R - r0) / 1e-5
            G[i, :, j] = (q - q0) / 1e-5
    return J, G


def pass1_ids(acts, n, J, G):
    """The first reference, then greedily the press whose touch-off most lowers the expected squared miss over every
    press of the plan, trace(Sigma sum G^T G), Sigma the fit's posterior covariance (priors PRIOR, touches SIG_PAD)."""
    H = np.diag(1.0 / PRIOR ** 2)
    M = np.einsum("kai,kaj->ij", G, G) / len(G)
    first = next((i for i, a in enumerate(acts) if a.get("ws_kind") == "reference"), 0)
    pick = [first]
    H = H + np.outer(J[first], J[first]) / SIG_PAD ** 2
    while len(pick) < min(n, len(acts)):
        best = None
        for k in range(len(acts)):
            if k in pick:
                continue
            v = float(np.trace(np.linalg.inv(H + np.outer(J[k], J[k]) / SIG_PAD ** 2) @ M))
            if best is None or v < best[0]:
                best = (v, k)
        pick.append(best[1])
        H = H + np.outer(J[best[1]], J[best[1]]) / SIG_PAD ** 2
    S = np.linalg.inv(H)
    return [acts[k]["id"] for k in sorted(pick)], np.sqrt(np.diag(S)), float(np.sqrt(np.trace(S @ M)))


def replay_travel(ws, seat, key):
    """The replay's hub pose over the seat, its start (12 mm out of the key, where the program's homing leaves the
    robot) and its lift out of the key before the first joint move (RIG.LIFT: the first of travel_options that passes,
    as pick_travel takes it; checked against the first _paths_ row's joint-liftNN)."""
    RIG.HOME_KEY[0] = key
    ctx = RIG.context(dict(orient=ws["orient"], seat_h=RIG.SEAT_BLOCK["seat_h"]), tool=ws["tool"])
    ctx["seat"] = seat
    ctx["hub"] = RIG.hub(ctx, seat)
    x = {k: (np.array(v, float) if k in ("T_rest", "T_deep", "q_rest", "n_A", "d_A", "u_A") else v)
         for k, v in ws["actions"][0].items()}
    q = RIG.start_q(ctx)
    lift = None
    for i, (w, segs) in enumerate(RIG.travel_options(ctx, None, q, x)):
        sm, why = RIG.sample_all(segs, q, False)
        cl = RIG.clear_of(ctx, sm, None) if not why else {}
        if not list(why) + RIG.clear_flags(cl):
            lift = dict(way=w, lift_mm=RIG.LIFT[min(i, len(RIG.LIFT) - 1)])
            break
    if lift is None:
        raise SystemExit("no way from the homing to the first action passes the replay's checks")
    lift["q_start"] = np.asarray(q, float).round(6).tolist()
    return dict(T_BE=ctx["hub"][0].round(6).tolist(), q=np.asarray(ctx["hub"][1], float).round(6).tolist()), lift


def travel_of(name):
    """(way, homing lift) from a _paths_ row's travel: straight, joint, joint-liftNN, hub."""
    if name.startswith("joint-lift"):
        return "joint", float(name[len("joint-lift"):])
    return name, None


def build(ws, rows, n_pass1=20, st1=None, held_channel=None, twists=TWISTS):
    RIG.PLATE[0] = "block"
    g = RIG.plate()
    seat = int(ws["seat"])
    key = KEY_OF_SEAT[seat]
    hub, first = replay_travel(ws, seat, key)
    way = {r["id"]: r["travel"] for r in rows}
    w0, lift0 = travel_of(way[ws["actions"][0]["id"]])
    if lift0 is not None and abs(lift0 - first["lift_mm"]) > 1e-6:
        raise SystemExit(f"the replay's first lift {lift0} mm is not the {first['lift_mm']} mm found here")
    T_AP = np.array(g["seats"][seat]["T_AF"], float) @ RIG.T_FP
    T_AM_home = np.array(g["keys"][key]["T_AM_home"], float)
    T_BA, T_EM = np.array(ws["T_BA"], float), np.array(ws["T_EM"], float)
    tool = ws["tool"]
    if tool not in RIG.SPHERES:
        raise SystemExit(f"tool {tool}: the probe mode presses with a sphere only")
    R = float(RIG.PROBE[RIG.SPHERES[tool]]["R"])
    c_M = np.array([0.0, 0.0, RIG.tool_end_z(tool)])
    base = json.loads(GEOM_PLAN.read_text())
    geo0 = C.Geometry(base)                                   # the pad's surface (the tip mode's geometry has it too)
    geo = copy.deepcopy(base["geometry"])
    geo.pop("indenter", None)
    geo["T_EM"] = T_EM.tolist()
    geo["T_IM_home"] = T_AM_home.tolist()
    geo["probe"] = dict(kind="sphere", tool=tool, centre_M=c_M.tolist(), radius=R)
    ch = HELD_CHANNEL[seat] if held_channel is None else int(held_channel)
    geo["seat"] = dict(index=seat, key=key, T_AP=T_AP.tolist(), apex_A=ws["seat_apex_A"], orient=ws["orient"],
                       held_channel=ch)
    R_AP = T_AP[:3, :3]
    st = float(ws.get("standoff_mm") or RIG.ST)
    acts, worst = [], 0.0
    for a in ws["actions"]:
        T_rest = np.array(a["T_rest"], float)
        R_PM = R_AP.T @ T_BA[:3, :3].T @ T_rest[:3, :3] @ T_EM[:3, :3]
        d_P = R_AP.T @ np.asarray(a["d_A"], float)
        d_P /= np.linalg.norm(d_P)
        tg = dict(frame="P", p=a["p_P"], n=a["n_P"], d=d_P.round(6).tolist(), R_PM=R_PM.round(7).tolist())
        T_BP = T_BA @ T_AP                                     # the check: the replay's rest pose from the directions in P
        c = C.apply(T_BP, np.asarray(a["p_P"], float) + R * np.asarray(a["n_P"], float) - st * d_P)
        R_BM = T_BP[:3, :3] @ R_PM
        T_BE = C.Tm(R_BM, c - R_BM @ c_M) @ C.inv(T_EM)
        worst = max(worst, C.pose_error(T_BE, T_rest)[0])
        w, lift = travel_of(way.get(a["id"], "joint"))
        deep = float(a.get("deep_clear_mm") or ws.get("deep_mm") or RIG.DEEP)
        x = dict(id=a["id"], stage="grid", kind="press", region=a["region"], i=a["i"], target=tg, standoff=st,
                 max_depth_mm=deep, levels=a["levels"], stop_N=max(a["levels"]), v=0.0005, hold_s=0.6,
                 ws_kind=a["kind"], theta=a["theta"], azimuth=a["azimuth"], q_rest=a["q_rest"], travel_ws=w)
        if lift is not None:
            x["homing_lift_mm"] = lift
        if a["kind"] != "twist":
            acts.append(x)
            continue
        for k, (N, sfx) in enumerate(twists):                 # the replay's own moves: at depth, the sphere turned about
            y = dict(x, id=a["id"] + sfx, kind="twist", press_N=N, hold_s=TWIST_HOLD_S,     # the pad's normal through
                     moves=[dict(degs=[float(a["twist"])])])                               # its centre, and back
            for kk in ("levels", "stop_N"):
                y.pop(kk)
            if k:
                y["travel_ws"] = "straight"                    # from the same rest
                y.pop("homing_lift_mm", None)
            acts.append(y)
    if worst > 0.01:
        raise SystemExit(f"the rest poses from the directions in P stand {worst:.3f} mm off the replay's")
    presses = [a for a in acts if a["kind"] == "press"]
    J, G = jacobians(presses, T_BA, T_AP, T_EM, c_M, R, geo0)
    p1, sd1, miss1 = pass1_ids(presses, n_pass1, J, G)
    p1 = set(p1)
    touches = []
    for a in acts:
        if a["id"] in p1:
            t = dict(a, id="pad-" + a["id"][len("probe-"):], stage="pad", kind="touch",
                     beyond_mm=min(1.2, a["max_depth_mm"]), stop_N=TOUCH_STOP_N, fit_N=TOUCH_FIT_N, v=0.0005,
                     travel_ws="joint")
            if st1 is not None:
                t["standoff"] = float(st1)
            for k in ("levels", "max_depth_mm", "hold_s", "homing_lift_mm"):
                t.pop(k, None)
            touches.append(t)
    touches[0].update(travel_ws=first["way"], homing_lift_mm=first["lift_mm"])   # pass 1 starts from the homing
    yaw = math.degrees(math.atan2(T_BA[1, 0], T_BA[0, 0]))
    regions = {}
    for t in touches:
        regions[t["region"]] = regions.get(t["region"], 0) + 1
    note = (f"From {ws.get('source', 'the workstation plan')} (seat {seat + 1}, tool {tool}, mux channel {ch}) by "
            f"scripts/ur5_probe_plan.py: {len(touches)} touch-offs (pass 1: {regions}; at the CAD the fit's sd would be "
            f"{np.degrees(sd1[:3]).round(2).tolist()} deg and {sd1[3:6].round(3).tolist()} mm for the seat, "
            f"{sd1[6:9].round(3).tolist()} mm for the sphere's centre, a press's expected miss {miss1:.3f} mm rms) and "
            f"{len(acts)} presses and twists; rest poses "
            f"within {worst * 1e3:.1f} um of the replay's at the plan's T_BA and T_EM.")
    plan = dict(
        version=1, mode="probe", created=ws["created"], notes=[note] + list(ws.get("notes", [])),
        geometry=geo,
        homing=dict(repeats=3, key=key),
        travel=dict(speed=1.0, fast_v=0.005, slow_mm_touch=2.5, slow_mm_press=0.5, turn_v=0.004,   # 0.23 deg/s at 0.30
                    probe=dict(retract_mm=RIG.RETRACT, safe_mm=RIG.SAFE, hub=hub,
                               note="the replay's travel_options: straight (back retract_mm along the stem, across, in), "
                                    "joint (back safe_mm, a joint move, in), hub (back safe_mm, joint moves by way of "
                                    "the hub, in); from the homing out of the key along -x_M by homing_lift_mm first")),
        safety=dict(abort_N=10.0, abort_Nm=1.5, box_mm=400.0, travel_abort_N=8.0),
        payload=dict(kg=0.403, cog_mm=[0.0, -0.5, 34.3], note="the tip's mount; the probe's to be measured"),
        speed_slider=0.3,
        checks=dict(placement_mm_deg=[float(T_BA[0, 3]), float(T_BA[1, 3]), float(T_BA[2, 3]), yaw],
                    source=ws.get("source")),
        sim=dict(T_BI=T_BA.tolist(), q_seed=first["q_start"], q_start=first["q_start"], start=None),
        actions=touches + acts)
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pass1", type=int, default=20, help="touch-offs in pass 1")
    ap.add_argument("--pass1-standoff", type=float, default=None, help="mm: the touch-offs' standoff (default the plan's, "
                    "the replay's checked rest)")
    ap.add_argument("--held-channel", type=int, default=None, help="the seat's board's mux channel (default HELD_CHANNEL)")
    ap.add_argument("--no-1n-twists", action="store_true", help="the twists at 3 N only (each 1 N twist adds about 2 min)")
    a = ap.parse_args()
    ws = json.loads(Path(a.plan).read_text())
    src = Path(a.plan).resolve()
    ws["source"] = str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src)
    if ws.get("kind") != "probe":
        raise SystemExit(f"{a.plan} is not a probe plan (kind {ws.get('kind')})")
    paths = Path(a.plan).with_name(Path(a.plan).name.replace("_plan_", "_paths_"))
    if not paths.exists():
        raise SystemExit(f"no {paths.name}: the replay's travels are needed")
    rows = json.loads(paths.read_text())["rows"]
    plan = build(ws, rows, a.pass1, a.pass1_standoff, a.held_channel, TWISTS[:1] if a.no_1n_twists else TWISTS)
    out = Path(a.out) if a.out else Path(a.plan).with_name(Path(a.plan).stem.replace("_plan_", "_program_plan_") + ".json")
    out.write_text(json.dumps(plan, indent=1))
    n1 = sum(1 for x in plan["actions"] if x["stage"] == "pad")
    print(f"{out}: {n1} touch-offs, {len(plan['actions']) - n1} presses and twists; {plan['notes'][0]}")


if __name__ == "__main__":
    main()
