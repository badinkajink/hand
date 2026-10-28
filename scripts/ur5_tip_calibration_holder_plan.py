#!/usr/bin/env python3
"""UR5 plan on the fingertip holder (owner, 2026-10-27): the crosstalk field map, with the robot tip's apex held without
contact at known poses off two held tips that carry their own boards, and the touch-offs that fix its height. The held
tips and their boards are ur5_cb3_cell's (TIPHOLD, TIP_F, HELD_BOARD), the poses scripts/fingertip_holder.py's, the
hover action the lab program's (85ddc425: the robot pad's target point held at at_I with its normal down, turned twist
about the vertical, between two dwells at a reference ref_up_mm above).

Stages of the plan:
  pad    three touch-offs of the robot apex on the +y held apex (tip up). A sphere of the pad's apex radius across the tip
         (7.4^2 / 8.5 = 6.44 mm) stands in for it, its top at the held apex, so that the program's touch-off runs as on an
         indenter. Their onsets give the held apex's height: rebuild with --z-offset (onset minus expected, mm).
  hover  facing on the +y station: W 0.5 to 30 mm at twists -20 to +20 deg, twelve offsets of 3 mm across or along at
         W 1, 3 and 10 mm, and W 45 mm as a far pose; side by side on the -y station: the poses of the holder's reach
         check that pass (g 13 to 25 mm, s 0, 2 and 5 mm) and g 40 mm as a far pose. The first 20 repeat at the end.
         Each pose dwells 2 s, 6 s at W 15 mm and over or g 20 mm and over; the reference dwells 1.5 s, 40 mm above,
         lowered where the cell needs it. The contact guard stays at 0.5 N: the magnets' pull between the tips is at most
         1.0 mN at these poses (fingertip_crosstalk.py pull).

Each pose and its reference are checked with tall_indenter._pose_ok on the cell's "tiphold" kind (holder, held tips and
boards, the robot's trimmed cap and screw heads), with joint 6 within --j6-min and --j6-max and joint 5 within 34 to 98.

The program computes the holder's frame I from its homing, the mount seated by hand on the holder's key (T_BI of the
session's registration), and every target of the plan is in that frame. --true-frame names a registration.json whose T_BI
is where the holder stands, measured by touch-offs (scripts/ur5_tip_calibration_holder_apex_fit.py --out): the apexes,
the hovers' offsets and gaps (along that frame's vertical) are placed by it and written in the program's frame, and the
checks put the holder, its tips and boards there. The pad's orientation stays the program's: at homing its axes are the
key's (x_P along -z_I of the program's frame), so a seating 4.3 deg from the holder's frame tilts the pad as much.
--touch-station -y puts the touch-offs (and the sphere that stands in for the held pad) on the -y apex.

    .venv/bin/python scripts/ur5_tip_calibration_holder_plan.py --session <registration folder> [--z-offset 0]
        [--true-frame FOLDER/registration.json] [--touch-station +y|-y]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("UR5_MOUNT", "v7")            # the mount on the robot since R25 (2026-10-26)
import fingertip_holder as FH  # noqa: E402
import tall_indenter as TI  # noqa: E402
import ur5_cb3_cell as CELL  # noqa: E402

C = CELL.C
LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
APEX_P, N_P = [8.5, 0.0, 10.86], [1.0, 0.0, 0.0]          # the robot pad's apex and its normal, pad frame P
R_APEX = 7.4 ** 2 / 8.5                                    # the pad's radius of curvature at the apex, across the tip
FACING_W = (0.5, 1, 2, 3, 5, 10, 15, 20, 30)
TWISTS = (-20, -10, 0, 10, 20)
WIDTH = 14.8                                               # the pads' width: centre lines 14.8 + g apart side by side


def setup(session, base, add_other=(), drop_other=()):
    ck = base["checks"]
    CELL.BOARD.update(ck["board"])
    CELL.BOARD["half_mm"] = 400.0                          # the mat table under the wrist, as the reach maps
    other = [o for o in ck.get("other", [])]
    for spec in add_other:                                 # kind:session, the fixture at that session's registered T_BI
        kind, ses = spec.split(":", 1)
        T = json.loads((Path(ses) / "registration.json").read_text())["T_BI"]
        other = [o for o in other if o["kind"] != kind] + [dict(kind=kind, T_BI=T)]
    other = [o for o in other if o["kind"] not in drop_other]
    base["checks"]["other"] = other
    CELL.OTHER[:] = [(np.array(o["T_BI"], float), o["kind"]) for o in other]
    geo = C.Geometry(base)
    reg = C.Registration(geo).load(json.loads((Path(session) / "registration.json").read_text()))
    return reg, np.asarray(reg.q_home, float)


def check(reg, qh, p, tw, reg_true=None):
    """(ok, least clearance per group, joints 5 and 6) with the robot apex at p (the program's I) turned tw, the holder
    where reg_true puts it (default: where the program's frame does)."""
    T_IM = FH.T_IF_moving(p, tw) @ np.linalg.inv(CELL.T_M_FROM_F)
    T_BE = reg.T_BI @ T_IM @ C.inv(reg.T_EM)
    return TI._pose_ok(CELL, C, CELL.SIM, reg_true or reg, "tiphold", T_BE, None, qh)


def rel(path):
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def plan_other(base):
    return [dict(kind=o["kind"], origin_B=np.round(np.array(o["T_BI"], float)[:3, 3], 1).tolist())
            for o in base["checks"]["other"]]


def hover(aid, p, tw, dwell, ref_up):
    return dict(id=aid, stage="hover", kind="hover", target=dict(frame="P", p=APEX_P, n=N_P, index=-1, region="face"),
                at_I=[round(float(v), 3) for v in p], twist=float(tw), ref_up_mm=float(ref_up), slow_mm=5.0, v_mm_s=10.0,
                v_slow_mm_s=2.0, dwell_ref_s=1.5, dwell_s=float(dwell), contact_N=0.5)


def run(a):
    base = json.loads(Path(a.base_plan).read_text())
    reg, qh = setup(a.session, base, a.add_other, a.drop_other)
    TI.J6 = (a.j6_min, a.j6_max)
    st = CELL.TIPHOLD["stations"]
    T_true = np.array(json.loads(Path(a.true_frame).read_text())["T_BI"], float) if a.true_frame else reg.T_BI
    M = C.inv(reg.T_BI) @ T_true                                          # the holder's frame to the program's

    def P(v):                                                             # a point of the holder's frame in the program's
        return M[:3, :3] @ np.asarray(v, float) + M[:3, 3]

    reg_true = copy.copy(reg)
    reg_true.T_BI = T_true
    apex_t = {i: FH.held_apex(CELL, i) + np.array([0.0, 0.0, a.z_offset]) for i in range(len(st))}
    apex = {i: P(v) for i, v in apex_t.items()}
    ts = next(i for i in range(len(st)) if (st[i][0] > 0) == (a.touch_station == "+y"))
    up = apex[ts]                                                         # the touch-offs' station, tip up
    cand = []                                                             # (id, p, tw, dwell)
    for i in sorted(range(len(st)), key=lambda k: -st[k][0]):            # +y first
        ys, mode = st[i]
        if mode != "up":
            continue
        tag = "" if i == 1 else "m"                                       # hov-f-... on +y, hov-fm-... on -y
        mw = a.min_w if i == 1 else max(a.min_w, a.min_w_other)          # no touch-offs on -y: its height from +y's
        for tw in TWISTS:
            for W in sorted((w for w in FACING_W if w >= mw), reverse=True):
                cand.append((f"hov-f{tag}-W{W:g}-t{tw:+d}", P(apex_t[i] + [0.0, 0.0, W]), tw, 6.0 if W >= 15 else 2.0))
        for W in (w for w in (1, 3, 10) if w >= mw):
            for dx, dy, nm in ((3, 0, "x+3"), (-3, 0, "x-3"), (0, 3, "y+3"), (0, -3, "y-3")):
                cand.append((f"hov-f{tag}-W{W:g}-{nm}", P(apex_t[i] + [dx, dy, W]), 0, 2.0))
        cand.append((f"hov-f{tag}-W45-far", P(apex_t[i] + [0.0, 0.0, 45.0]), 0, 6.0))
    if st[0][1] == "down":                                                # side by side needs the -y tip under its block
        down = apex_t[0] - np.array([0.0, 0.0, a.z_offset])
        side = json.loads((FH.OUT / "data" / f"{FH.STEM}_reach_side.json").read_text())["poses"]
        for r in side:
            if r["ok"]:
                g, s_, tw = r["g"], r["s"], r["tw"]
                cand.append((f"hov-s-g{g:g}-s{s_:g}-t{tw:+d}", P(down + [s_, -(WIDTH + g), 0.0]), tw,
                             6.0 if g >= 20 else 2.0))
        cand.append(("hov-s-g40-far", P(down + [0.0, -(WIDTH + 40.0), 0.0]), 0, 6.0))
    acts, dropped, least, links = [], [], {}, {}
    for aid, p, tw, dwell in cand:
        ok, worst, j = check(reg, qh, p, tw, reg_true)
        if not ok:
            dropped.append(dict(id=aid, why="pose", joints_56=None if j is None else [round(v, 1) for v in j],
                                clear=worst and {k: round(v, 2) for k, v in worst.items()}))
            continue
        ref = next((h for h in (40.0, 30.0, 20.0, 10.0) if check(reg, qh, p + [0.0, 0.0, h], tw, reg_true)[0]), None)
        if ref is None:
            dropped.append(dict(id=aid, why="reference"))
            continue
        act = hover(aid, p, tw, dwell, ref)
        act["check"] = dict(clear={k: round(v, 2) for k, v in worst.items()}, joints_56=[round(v, 1) for v in j])
        for T in (p, p + [0.0, 0.0, ref]):                  # the robot's links one by one, at the pose and the reference
            T_IM = FH.T_IF_moving(T, tw) @ np.linalg.inv(CELL.T_M_FROM_F)
            T_BE = reg.T_BI @ T_IM @ C.inv(reg.T_EM)
            q = CELL.SIM.ik(T_BE, qh)
            for k, v in CELL.clearance(q, T_BE, T_true, reg.T_EM, "tiphold").items():
                if k.endswith("_link"):
                    if v < links.get(k, (1e9, ""))[0]:
                        links[k] = (round(float(v), 2), aid)
        acts.append(act)
        for k, v in worst.items():
            least[k] = min(least.get(k, 1e9), v)
    if a.keep:                                            # a part of the map: the hovers whose ids match
        import re
        acts = [x for x in acts if re.search(a.keep, x["id"])]
    if a.first:                                           # the watched run: these hovers first, in this order
        first = [x for f in a.first.split(",") for x in acts if x["id"] == f]
        acts = first + [x for x in acts if x not in first]
    reps = [dict(copy.deepcopy(x), id=x["id"] + "-rep") for x in acts[:20]]
    touches = []
    offs = [tuple(float(v) for v in o.split(",")) for o in a.touch_offsets.split(";")] if a.touch_offsets else \
        [(0.0, 0.0)] * (0 if a.no_touch else 3)
    for k, (dx, dy) in enumerate(offs):                    # the program's offset moves the contact point by -offset in I
        t = dict(id=f"hold-touch-{'abcdefghijkl'[k]}" + (f"-x{dx:+g}-y{dy:+g}" if (dx or dy) else ""), stage="pad",
                 kind="touch", target=dict(frame="P", p=APEX_P, n=N_P, index=-1, region="face"), twist=0.0,
                 standoff=float(a.touch_standoff),
                 beyond_mm=float(a.touch_beyond), v=0.00015, stop_N=0.5, fit_N=0.4, offset=[-dx, -dy])
        touches.append(t)
    if a.touch_only:
        acts, reps = [], []
    plan = {k: copy.deepcopy(v) for k, v in base.items() if k != "actions"}
    plan["created"] = a.created
    plan["indenter"] = "tiphold"
    plan["checks"].pop("placement_mm_deg", None)          # the base plan's fixture placement: the program would compare the
    #                                                       holder's homing with it (and its 2 deg tilt limit fails on the
    #                                                       2x4's bow); the plan's checks are built on the seating itself
    centre = up - np.array([0.0, 0.0, R_APEX])
    plan["geometry"]["indenter"] = dict(kind="sphere", centre=[round(float(v), 3) for v in centre], radius=round(R_APEX, 3))
    plan["geometry"]["T_IM_home"] = np.round(CELL.tiphold_parts()["frame"]["T_IM_home"], 4).tolist()
    key = {i: ("+y" if st[i][0] > 0 else "-y") for i in apex}
    plan["holder"] = dict(held_apex_I={key[i]: np.round(apex[i], 3).tolist() for i in apex},
                          held_apex_true_I={key[i]: np.round(apex_t[i], 3).tolist() for i in apex},
                          true_frame=rel(a.true_frame) if a.true_frame else None,
                          T_BI_true=np.round(T_true, 5).tolist(), touch_station=a.touch_station,
                          stations=[list(x) for x in st],
                          z_offset_mm=a.z_offset, registration=rel(a.session),
                          cap_trim_x_F=CELL.CAP_TRIM_X, screw_heads=CELL.SCREW_HEADS, held_board=CELL.HELD_BOARD,
                          j6=[a.j6_min, a.j6_max], least_clearance_mm={k: round(v, 2) for k, v in least.items()},
                          least_link_clearance_mm={k: list(v) for k, v in links.items()}, obstacles=plan_other(base),
                          dropped=dropped)
    plan["notes"] = list(plan.get("notes", [])) + [
        f"{a.created}, workstation: scripts/ur5_tip_calibration_holder_plan.py on {Path(a.session).name}'s registration"
        + (f", the holder placed by {rel(a.true_frame)}" if a.true_frame else "") + ". "
        f"Stage pad: touch-offs of the robot apex on the {a.touch_station} held apex (a 6.44 mm sphere stands in for the "
        "held pad, its top at the apex); their onsets go to scripts/ur5_tip_calibration_holder_apex_fit.py. Stage hover: "
        "the crosstalk map; plan['indenter'] tiphold names the cell's model for the checks."]
    plan["actions"] = touches + acts + reps
    Path(a.out).write_text(json.dumps(plan, indent=1, default=C.jsonable) + "\n")
    n_f = sum(x["id"].startswith("hov-f-") for x in acts)
    n_fm = sum(x["id"].startswith("hov-fm-") for x in acts)
    n_s = sum(x["id"].startswith("hov-s") for x in acts)
    t_est = sum(1.5 + 1.5 + x["dwell_s"] + 2 * ((x["ref_up_mm"] - 5) / 10 + 5 / 2) + 6 for x in acts + reps) / 60
    print(f"{len(touches)} touch-offs, {n_f} facing hovers on +y and {n_fm} on -y, {n_s} side-by-side, {len(reps)} repeats; "
          f"{len(dropped)} dropped; about {t_est:.0f} min of hovers")
    print("least clearances (mm):", {k: round(v, 2) for k, v in least.items()})
    print("least clearance of each robot link (mm, action):", links)
    for d in dropped:
        print("  dropped", d)
    print("->", rel(a.out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", default=str(LAB / "20261027-sphchk-b-hybrid"),
                    help="a folder whose registration.json gives the robot's mount and pad and the holder's place (T_BI)")
    ap.add_argument("--base-plan", default=str(LAB / "20261027-plan_sphere_lab_check_shear_v2id1.json"))
    ap.add_argument("--add-other", action="append", default=[], help="kind:session, a fixture on the cell at that session's "
                    "registered T_BI (the low sphere: sphere:<192158-v2id1's folder>)")
    ap.add_argument("--drop-other", action="append", default=[], help="a fixture kind taken off the cell for this plan")
    ap.add_argument("--min-w", type=float, default=0.0, help="facing gaps under this (mm) are left out: without the "
                    "touch-offs' measured apex height, 3 mm (the seatings' 1 mm and 1 deg spread, the pad's 0.5 mm)")
    ap.add_argument("--min-w-other", type=float, default=2.0, help="the least facing gap on the -y station (mm), whose "
                    "height comes from the +y touch-offs: the -y tip's own board and screws add about 0.5 mm")
    ap.add_argument("--keep", default="", help="keep only the hovers whose ids match this regular expression (a part of "
                    "the map, e.g. the close -y hovers: '^hov-fm-W(0.5|1|2|3|5)-')")
    ap.add_argument("--first", default="", help="hover ids to run first, comma-separated (the owner's watched run)")
    ap.add_argument("--touch-offsets", default="", help="dx,dy;... (mm in the program's I) of the touch-offs from the "
                    "touched apex (default three at the apex)")
    ap.add_argument("--touch-beyond", type=float, default=1.2, help="mm past the expected contact a touch-off goes")
    ap.add_argument("--touch-standoff", type=float, default=5.0, help="mm above the expected contact a touch-off starts "
                    "(more when the apex may stand higher than expected)")
    ap.add_argument("--touch-only", action="store_true", help="only the touch-offs (the apexes' search)")
    ap.add_argument("--no-touch", action="store_true", help="no touch-offs (the rod fixture in place: wrist 3 at 0.7 mm)")
    ap.add_argument("--z-offset", type=float, default=0.0, help="mm added to the held apexes' height (touch-offs)")
    ap.add_argument("--true-frame", default=None, help="a registration.json whose T_BI is where the holder stands (the "
                    "apex fit's --out); targets are placed by it and written in the program's frame")
    ap.add_argument("--touch-station", default="+y", choices=("+y", "-y"), help="the held apex the touch-offs search")
    ap.add_argument("--j6-min", type=float, default=-62.0)
    ap.add_argument("--j6-max", type=float, default=45.0)
    ap.add_argument("--created", default="2026-10-27")
    ap.add_argument("--out", default=str(LAB / "20261027-plan_holder_lab_map_v3id1.json"))
    run(ap.parse_args())
