#!/usr/bin/env python3
"""Layout of the indenter plate for the UR5 cell (owner, 2026-10-28: "the big ass indenter plate i printed wont actually
work though, since the big sphere is in the way of the side-to-side tests ... can you analyze the 1027-indenter_array task
and file and spec out a plan so that each indenter and motion per indenter has enough space? and that they all still fit
on the bambu print plate").

Each indenter's motions are the lab's own plans for v3 ID 1, moved onto that indenter of a layout. The plan's indenter frame
I goes to the plate's indenter by a translation: I and the plate's frame A share their axes (x away from the robot, z up).
The plate stands where the low sphere's frame stood at v3 ID 1's registration (lab/20261027-225213-spherev3id1), and the
tool frames are that registration's. The moves are those of scripts/ur5_tip_calibration_holder_paths.py: the travel, the
approach, the depth at the largest force, and the turns of slides, rolls and twists; touch-offs stop at first contact. At
each sample the joints are checked against the program's windows, and every robot link and tool part against every part of
the plate, the held tips, the 2x4s and the table, with the margins of the lab's checks (robot 8, HEX-E 6, mount 3 (2.5 for
touch-offs), finger 0.8, pad 0.3 mm). Each row names the plate part nearest to each group.

Suites (the plans are in docs/experiments/20261022-ur5_tip_calibration/lab/):
  sphere5   sphere pass 1, stage 2 and the morning plan (the cell's low sphere, R 5 mm) on the 5 mm sphere
  tall5     tall pass 1, stage 2, the morning plan and the top edge (the tall indenter's 5 mm sphere) on the 5 mm sphere
  sphere12  the sphere suite on the 12.5 mm sphere, R set to 12.5 mm in the plans (the top stays where it was)
  tall12    the tall suite on the 12.5 mm sphere
  rod       rod pass 1 and stage 2 on the half rod
  facing    the holder builder's facing hovers over the up station's tip (W 0.5 to 45 mm, twists -20 to +20 deg, offsets)
  side      side-by-side hovers beside the down station's tip (g 13 to 40 mm, s 0, 2 and 5 mm, twists -30 to +20 deg)
  keys      homing on each key, seated and 12 mm out

Layouts (LAYOUTS): "array", the plate printed on 2026-10-27 (scripts/indenter_array.py ARRAY) with its -y station's pillar
cut and that tip pad-down; and the candidates that follow it.

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 200 --"
    $RG .venv/bin/python scripts/indenter_plate_layout.py replay --layout array --suite sphere5 [--quick]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("UR5_MOUNT", "v7")
os.environ["UR5_TIPHOLD_STATIONS"] = "down,up"      # the plate's holder: the -y tip pad-down (side by side), +y up (facing)
import ur5_cb3_cell as CELL  # noqa: E402
import ur5_tip_calibration_holder_paths as HP  # noqa: E402
import indenter_array as IA  # noqa: E402

C = CELL.C
LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
OUT = ROOT / "docs/experiments/20261028-indenter_plate_layout"
STEM = "20261028-indenter_plate_layout"
WORK = ROOT / "logs/20261028-indenter_plate"
TOOLREG = LAB / "20261027-225213-spherev3id1/registration.json"
NEED = dict(robot=8.0, mount=3.0, finger=0.8, hex_e=6.0, pad=0.3)
NEED_TOUCH_MOUNT = 2.5
GROUPS = dict(mount=("adapter", "fork", "clamp_plate", "stop_wedge"), finger=("dip_link", "cover", "cap"),
              hex_e=("hex_e",), pad=("pad",))
HP.J6_WIN = (-65.0, 60.0)                              # the HEX-E cable's window; past +45 deg the owner is present
J6_UNATTENDED = 45.0

SPHERE = ("superseded/20261027-plan_sphere_lab_pass1_v3id1.json", "superseded/20261027-plan_sphere_lab_stage2_v3id1.json",
          "superseded/20261027-plan_sphere_lab_morning_v3id1.json")
TALL = ("superseded/20261027-plan_tall_lab_pass1_v3id1.json", "superseded/20261027-plan_tall_lab_stage2_v3id1.json",
        "superseded/20261027-plan_tall_lab_morning_v3id1.json", "superseded/20261028-plan_tall_lab_top_edge_v3id1_on143529.json")
ROD = ("superseded/20261027-plan_rod_lab_pass1_search_v3id1.json", "superseded/20261027-plan_rod_lab_stage2_v3id1.json")
HOLDER_BASE = "superseded/20261027-plan_holder_lab_map_v3id1.json"
SUITES = dict(sphere5=dict(ind="sphere5", plans=SPHERE), tall5=dict(ind="sphere5", plans=TALL),
              sphere12=dict(ind="sphere12", plans=SPHERE, R=12.5), tall12=dict(ind="sphere12", plans=TALL, R=12.5),
              rod=dict(ind="rod", plans=ROD), facing=dict(ind="holder"), side=dict(ind="holder"), keys=dict(ind="keys"))

# Layouts: overrides of indenter_array.ARRAY (A frame, mm). "array" is the print of 2026-10-27.
LAYOUTS = dict(
    array=dict(),
    # v2a: the 5 mm sphere where the tall indenter's plans work (+y, back), the holder at the -y back corner with its down
    # station toward the edge (the side-by-side poses past the plate), the rod and the 12.5 mm sphere in the front row
    v2a=dict(sphere5=dict(c=(70.0, 85.0), top=160.0), holder=dict(c=(70.0, -72.0)),
             rod=dict(c=(-15.0, 50.0), top=120.0), sphere12=dict(c=(-15.0, -45.0), top=140.0)),
    v2b=dict(sphere5=dict(c=(70.0, 85.0), top=120.0), holder=dict(c=(70.0, -72.0)),
             rod=dict(c=(-15.0, 50.0), top=120.0), sphere12=dict(c=(-15.0, -45.0), top=140.0)),
    # v2c: v2a's back corners; the 12.5 mm sphere in the back middle, where the back corners' suites leave 167 mm and more
    # free; the rod in the front left at 100 mm, under the side-by-side wrist (115 mm free)
    v2c=dict(sphere5=dict(c=(70.0, 85.0), top=160.0), holder=dict(c=(70.0, -72.0)),
             rod=dict(c=(0.0, -85.0), top=100.0), sphere12=dict(c=(80.0, 5.0), top=150.0)),
    # v2d: v2c with the 12.5 mm sphere in the front right at 95 mm: its suite's tool reaches 85 mm sideways and back at
    # up to 40 mm over its top, more than the back middle leaves between the 5 mm sphere and the holder
    v2d=dict(sphere5=dict(c=(70.0, 85.0), top=160.0), holder=dict(c=(70.0, -72.0)),
             rod=dict(c=(0.0, -85.0), top=100.0), sphere12=dict(c=(-15.0, 60.0), top=95.0)),
)

EXCL = dict(sphere5=4.0, sphere12=12.0, rod=12.0)      # the pad's points this near its contact meet the pressed tip


# ------------------------------------------------------------------------------------------------ the plate
def plate(layout):
    """The plate's geometry in A for a layout: indenter_array.geometry with the layout's overrides and the holder's
    stations as CELL.TIPHOLD has them (-y down, +y up). A down station's block hangs on its arm and gusset, without the
    pillar that the print of 2026-10-27 stands each block on: its tip hangs under it."""
    a = copy.deepcopy(IA.ARRAY)
    for k, v in LAYOUTS[layout].items():
        a[k] = dict(a[k], **v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    g = IA.geometry(a)
    h = g["holder"]
    hx, hy = h["c"]
    src = CELL.tiphold_parts(dict(CELL.TIPHOLD, z_bot=h["z_bot"]))["stations"]
    for st, s in zip(h["stations"], src):
        st["mode"] = s["mode"]
        if s["mode"] == "down":
            st["block"] = [(p[0] + hx, p[1] + hy, p[2]) for p in s["block"]]
    return g


def part_sdfs(P, g, tip=True, pressed=None, skip_key=None, cell=None):
    """Signed distance (mm) of points P (A) to each part of the plate: {name: array}. tip=False leaves out the pressed
    indenter's tip (the pad's points around its contact)."""
    a = g["params"]
    x0, x1, y0, y1, tb = a["base"]
    out = dict(base=CELL.sd_box(P, (x0, y0, 0.0), (x1, y1, tb)))
    kd = g["key"]
    bx, by = kd["k0"][0] + kd["margin"], kd["k0"][1] + kd["margin"]
    for i, k in enumerate(g["keys"]):
        if i == skip_key:
            continue
        L = (P - [k["c"][0], k["c"][1], 0.0]) @ IA._Rz(k["yaw"])
        pts = [(sx * hx_, sy * hy_, z) for (hx_, hy_), z in ((kd["k0"], k["body_top"]), (kd["k1"], k["pin_top"]))
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        out[f"key {i + 1}"] = np.minimum(CELL.sd_box(L, (-bx, -by, tb), (bx, by, k["body_top"])), CELL._convex_sdf(L, pts))
    for name in ("sphere5", "sphere12"):
        s = g[name]
        Q = P - [s["c"][0], s["c"][1], 0.0]
        d = np.minimum(CELL.sd_cyl_z(Q, s["col_r"], tb, s["z_col"]),
                       CELL.sd_capped_cone(Q, s["z_col"], s["z_neck"], s["r_col"], s["neck_r"]))
        if tip or pressed != name:
            d = np.minimum(d, np.linalg.norm(Q - [0.0, 0.0, s["top"] - s["R"]], axis=1) - s["R"])
        out[name] = d
    r = g["rod"]
    (cx, cy), hl = r["c"], r["half_len"]
    Q = P - [cx, cy, 0.0]
    d = np.minimum(CELL.sd_capped_cone(Q, tb, r["box_z0"], r["col_r"], r["r_top"]),
                   CELL.sd_box(P, (cx - r["R"], cy - hl, r["box_z0"]), (cx + r["R"], cy + hl, r["z_axis"])))
    if tip or pressed != "rod":
        d = np.minimum(d, CELL.sd_cyl_y(P, r["R"], cy - hl, cy + hl, xz=(cx, r["z_axis"])))
    out["rod"] = d
    h = g["holder"]
    (hx, hy), zb, zt = h["c"], h["z_bot"], h["z_top"]
    Q = P - [hx, hy, 0.0]
    zc = zb - h["cone_below"]
    out["holder column"] = np.minimum(CELL.sd_capped_cone(Q, tb, zc, h["col_r0"], h["col_r"]), CELL.sd_cyl_z(Q, h["col_r"], zc, zt))
    for st in h["stations"]:
        nm = f"holder {'-y' if st['y_s'] < hy else '+y'} {st.get('mode', 'up')} block"
        out[nm] = np.minimum.reduce([CELL.sd_box(P, *st["block"]), CELL.sd_box(P, *st["arm"]), CELL._convex_sdf(P, st["gusset"])])
    out["held tips"] = CELL.held_tip_sdf(Q, dict(CELL.TIPHOLD, **h["tiphold"]))
    if cell is not None:
        out["2x4s, table"] = CELL.cell_sdf(P, cell)
    return out


def part_mins(P, ctx, **kw):
    if not len(P):
        return {}
    return {k: float(v.min()) for k, v in part_sdfs(P, ctx["g"], pressed=ctx.get("pressed"), cell=ctx["T_BA"], **kw).items()}


_PTS = {}
SWEEP = dict(on=False, lo=(-260.0, -260.0), cell=2.0, n=260)   # the lowest z (mm, A) the robot and tool reach over each
#                                                                2 mm cell of the plan view, per run


def sweep_add(PA):
    if not SWEEP["on"] or not len(PA):
        return
    ij = np.floor((PA[:, :2] - SWEEP["lo"]) / SWEEP["cell"]).astype(int)
    ok = ((ij >= 0) & (ij < SWEEP["n"])).all(axis=1)
    np.minimum.at(SWEEP["zmin"], (ij[ok, 0], ij[ok, 1]), PA[ok, 2])


def points():
    if not _PTS:
        _PTS["robot"] = {ln: p for ln, p in CELL.robot_points().items()}
        _PTS["tool"] = dict(CELL.tool_points())
    return _PTS["robot"], _PTS["tool"]


def clearance(q, T_BE, ctx, contact_P=None, skip_key=None):
    """{group: (least distance (mm), the plate part)} for the robot's links and the tool's groups at joints q."""
    rp, tp = points()
    T_AB = C.inv(ctx["T_BA"])
    LF = CELL.link_frames(q)
    res = {}

    def upd(grp, mins):
        for part, v in mins.items():
            if v < res.get(grp, (1e9, ""))[0]:
                res[grp] = (v, part)

    for ln, p in rp.items():
        PA = C.apply(T_AB @ LF[ln], p)
        sweep_add(PA)
        upd("robot", part_mins(PA, ctx, skip_key=skip_key))
    T_AM = T_AB @ T_BE @ ctx["T_EM"]
    for name, p in tp.items():
        grp = next((gg for gg, names in GROUPS.items() if name in names), "robot")
        PA = C.apply(T_AM, p)
        sweep_add(PA)
        if name == "pad" and contact_P is not None:          # the pad presses its indenter's tip and deforms there
            upd(grp, part_mins(PA, ctx, tip=False, skip_key=skip_key))
        else:
            upd(grp, part_mins(PA, ctx, skip_key=skip_key))
    return res


# ------------------------------------------------------------------------------------------------ placing the plans
def tool():
    r = json.loads(TOOLREG.read_text())
    return np.array(r["T_BI"], float), np.array(r["T_EM"], float), r


def home(g, T_BA, T_EM, q0, i=1):
    """Flange pose and joints with the mount seated on key i."""
    T_AM = np.array(g["keys"][i]["T_AM_home"], float)
    T_BE = T_BA @ T_AM @ C.inv(T_EM)
    return T_BE, CELL.SIM.ik(T_BE, q0)


def frame_map(suite, g, plan):
    """T_AI: the plan's indenter frame I on the plate (a translation), and the plan with R changed where the suite asks."""
    s = SUITES[suite]
    plan = copy.deepcopy(plan)
    ind = plan["geometry"]["indenter"]
    if s["ind"] in ("sphere5", "sphere12"):
        c_I = np.asarray(ind["centre"], float)
        if s.get("R"):
            c_I = c_I + [0.0, 0.0, float(ind["radius"]) - s["R"]]
            ind["centre"], ind["radius"] = c_I.tolist(), s["R"]
        sp = g[s["ind"]]
        c_A = np.array([sp["c"][0], sp["c"][1], sp["top"] - sp["R"]])
        return C.Tm(None, c_A - c_I), plan
    if s["ind"] == "rod":
        r = g["rod"]
        c_A = np.array([r["c"][0], r["c"][1], r["z_axis"]])
        return C.Tm(None, c_A - np.asarray(ind["centre"], float)), plan
    h = g["holder"]                                          # the cell's holder (z_bot 40) raised to the plate's
    return C.Tm(None, [h["c"][0], h["c"][1], h["z_bot"] - CELL.TIPHOLD["z_bot"]]), plan


def session(ctx, suite, T_BI):
    """A registration folder for holder_paths.Moves: v3 ID 1's tool frames, the plan's I where the plate puts it, and the
    homing on key 2."""
    d = WORK / f"{ctx['layout']}_{suite}"
    d.mkdir(parents=True, exist_ok=True)
    r = copy.deepcopy(ctx["reg"])
    r["T_BI"] = np.round(T_BI, 6).tolist()
    r["T_BE_home"] = np.round(ctx["T_BE_home"], 6).tolist()
    r["q_home"] = [float(v) for v in ctx["q_home"]]
    (d / "registration.json").write_text(json.dumps(r, indent=1))
    return d


# ------------------------------------------------------------------------------------------------ hovers on the holder
def hover_candidates(test):
    """The holder builder's hovers in the cell holder's frame (z_bot 40): facing over the up station, side by side at -y
    of the down station."""
    import fingertip_holder as FH
    import ur5_tip_calibration_holder_plan as HPL
    out = []
    if test == "facing":
        up = FH.held_apex(CELL, 1)
        for tw in HPL.TWISTS:
            for W in sorted(HPL.FACING_W, reverse=True):
                out.append((f"hov-f-W{W:g}-t{tw:+d}", up + [0.0, 0.0, W], tw, 6.0 if W >= 15 else 2.0))
        for W in (1, 3, 10):
            for dx, dy, nm in ((3, 0, "x+3"), (-3, 0, "x-3"), (0, 3, "y+3"), (0, -3, "y-3")):
                out.append((f"hov-f-W{W:g}-{nm}", up + [dx, dy, W], 0, 2.0))
        out.append(("hov-f-W45-far", up + [0.0, 0.0, 45.0], 0, 6.0))
    else:
        down = FH.held_apex(CELL, 0)
        for g_ in (13, 14, 15, 17, 20, 25, 30, 35, 40):
            for s_ in (0, 2, 5):
                for tw in (-30, -20, -10, 0, 10, 20):
                    out.append((f"hov-s-g{g_:g}-s{s_:g}-t{tw:+d}", down + [s_, -(HPL.WIDTH + g_), 0.0], tw,
                                6.0 if g_ >= 20 else 2.0))
    return [HPL.hover(aid, p, tw, dwell, 40.0) for aid, p, tw, dwell in out]


# ------------------------------------------------------------------------------------------------ the replay
def judge(m, cl, kind):
    why = []
    if m["sin5"] < 0.5:
        why.append(f"sin5 {m['sin5']:.2f}")
    if m["q3"] < 20.0:
        why.append(f"q3 {m['q3']:.1f} deg")
    if m["shoulder"] < 150.0:
        why.append(f"shoulder {m['shoulder']:.0f} mm")
    if m["lim"] < 10.0:
        why.append(f"joint limit {m['lim']:.1f} deg")
    if m["j5_margin"] < 0.0 or m["j6_margin"] < 0.0:
        why.append(f"window j5 {m['j5_margin']:.1f} j6 {m['j6_margin']:.1f}")
    for grp, need in NEED.items():
        if grp == "mount" and kind == "touch":
            need = NEED_TOUCH_MOUNT
        d, part = cl.get(grp, (1e9, ""))
        if d < need:
            why.append(f"{grp} {d:.2f} mm to the {part}")
    return why


def replay_actions(plan, ctx, suite, quick, label):
    """Rows of the plan's actions on the plate."""
    T_AI, plan = frame_map(suite, ctx["g"], plan)
    T_BI = ctx["T_BA"] @ T_AI
    mv = HP.Moves(plan, session(ctx, suite, T_BI))
    reg = mv.reg
    HP.clear_groups = lambda q, T, reg_, cP: {k: v[0] for k, v in clearance(q, T, ctx, cP).items()}   # noqa: E731
    q = np.asarray(ctx["q_home"], float)
    T_prev = np.asarray(ctx["T_BE_home"], float)
    T_prev = C.Tm(T_prev[:3, :3], T_prev[:3, 3] + T_prev[:3, :3] @ reg.T_EM[:3, :3] @ np.array([-12.0, 0.0, 0.0]))
    q = CELL.SIM.ik(T_prev, q)
    rows = []
    for ai, act in enumerate(plan["actions"]):
        if act["kind"] not in ("touch", "hover", "press", "slide", "twist", "roll"):
            continue
        segs, T_end, cP, i_own = mv.action_moves(act, T_prev)
        m = dict(sin5=9.0, q3=999.0, shoulder=1e9, lim=999.0, j5_margin=999.0, j6_margin=999.0)
        cl, j6r, bad = {}, [1e9, -1e9], None
        if quick:                                           # the own moves' ends and the travel's joint moves' ends
            trav = [s for s in segs[:i_own] if s[0] == "J"]
            segs, i_own = trav + segs[i_own:], len(trav)
        for si, seg in enumerate(segs):
            own = si >= i_own
            if act["kind"] == "touch" and si == i_own:      # the approach stops at the first contact
                seg, _ = HP.to_contact(seg, q, reg)
                segs[-1] = ("L", seg[2], segs[-1][2])
            if quick:
                qq = CELL.SIM.ik(seg[2], q)
                sm = None if qq is None else [(seg[2], qq)]
            else:
                sm = HP.samples(seg, q)
            if sm is None:
                bad = f"no IK on segment {si}"
                break
            for k, (T, qq) in enumerate(sm):
                mk = HP.metrics(qq)
                for key in m:
                    m[key] = min(m[key], mk[key])
                j6r = [min(j6r[0], math.degrees(qq[5])), max(j6r[1], math.degrees(qq[5]))]
                if quick or k % 5 == 0 or k == len(sm) - 1:
                    TT = T if T is not None else CELL.SIM.fk(qq)
                    for grp, v in clearance(qq, TT, ctx, cP if own else None).items():
                        if v[0] < cl.get(grp, (1e9, ""))[0]:
                            cl[grp] = v
            q = sm[-1][1]
        why = ([bad] if bad else []) + judge(m, cl, act["kind"])
        rows.append(dict(id=act["id"], kind=act["kind"], plan=label, flags=why,
                         clear={k: [round(v[0], 2), v[1]] for k, v in cl.items()},
                         j6_deg=[round(j6r[0], 1), round(j6r[1], 1)], attended=bool(j6r[1] > J6_UNATTENDED),
                         **{k: round(v, 3) for k, v in m.items()}))
        if why:
            print(f"  {act['id']}: {'; '.join(why)}", flush=True)
        T_prev = T_end
    return rows


def replay_keys(ctx):
    rows = []
    q0 = np.asarray(ctx["reg"]["q_home"], float)
    for i in range(len(ctx["g"]["keys"])):
        T_BE, q = home(ctx["g"], ctx["T_BA"], ctx["T_EM"], q0, i)
        for lift in (0.0, 12.0):
            T = C.Tm(T_BE[:3, :3], T_BE[:3, 3] + T_BE[:3, :3] @ ctx["T_EM"][:3, :3] @ np.array([-lift, 0.0, 0.0]))
            qq = CELL.SIM.ik(T, q if q is not None else q0)
            if qq is None:
                rows.append(dict(id=f"key {i + 1}, {lift:g} mm out", kind="home", flags=["no IK"], clear={}))
                continue
            m = HP.metrics(qq)
            cl = clearance(qq, T, ctx, skip_key=i if lift == 0.0 else None)
            why = judge(m, cl, "home")
            rows.append(dict(id=f"key {i + 1}, {lift:g} mm out", kind="home", flags=why,
                             clear={k: [round(v[0], 2), v[1]] for k, v in cl.items()},
                             j_deg=np.degrees(qq).round(1).tolist(), **{k: round(v, 3) for k, v in m.items()}))
    return rows


def context(layout):
    g = plate(layout)
    T_BA, T_EM, reg = tool()
    CELL.BOARD.update(json.loads((LAB / SPHERE[1]).read_text())["checks"]["board"])
    x0, x1, y0, y1, _ = g["params"]["base"]
    CELL.BOARD["half_mm"] = 400.0
    CELL.BOARD["block_half_mm"] = [max(-x0, x1) + 10.0, 500.0]
    CELL.OTHER[:] = []
    T_home, q_home = home(g, T_BA, T_EM, np.asarray(reg["q_home"], float), 1)
    return dict(layout=layout, g=g, T_BA=T_BA, T_EM=T_EM, reg=reg, T_BE_home=T_home, q_home=q_home)


def cmd_replay(a):
    ctx = context(a.layout)
    t0 = time.time()
    s = SUITES[a.suite]
    ctx["pressed"] = s["ind"] if s["ind"] in EXCL else None
    ctx["excl"] = EXCL.get(s["ind"], 4.0)
    HP.DEPTH, HP.TURNS = "model", True
    SWEEP.update(on=True, zmin=np.full((SWEEP["n"], SWEEP["n"]), np.inf))
    rows = []
    if a.suite == "keys":
        rows = replay_keys(ctx)
    elif a.suite in ("facing", "side"):
        plan = json.loads((LAB / HOLDER_BASE).read_text())
        plan["actions"] = hover_candidates(a.suite)
        if a.limit:
            plan["actions"] = plan["actions"][::max(1, len(plan["actions"]) // a.limit)]
        rows = replay_actions(plan, ctx, a.suite, a.quick, a.suite)
    else:
        for f in s["plans"]:
            plan = json.loads((LAB / f).read_text())
            if a.limit:
                plan["actions"] = plan["actions"][::max(1, len(plan["actions"]) // a.limit)]
            print(f"{f}: {len(plan['actions'])} actions", flush=True)
            rows += replay_actions(plan, ctx, a.suite, a.quick, f)
    n_bad = sum(bool(r["flags"]) for r in rows)
    least = {}
    for r in rows:
        for k, (v, part) in r["clear"].items():
            if v < least.get(k, (1e9, ""))[0]:
                least[k] = (v, part)
    by = {}
    for r in rows:
        for w in r["flags"]:
            key = w.split(" mm to the ")[-1] if " mm to the " in w else w.split()[0]
            by[key] = by.get(key, 0) + 1
    n_att = sum(bool(r.get("attended")) and not r["flags"] for r in rows)
    out = dict(layout=a.layout, suite=a.suite, quick=bool(a.quick), n=len(rows), flagged=n_bad, attended=n_att, least=least,
               flags_by=by,
               seconds=round(time.time() - t0, 1), rows=rows, params=ctx["g"]["params"])
    dst = OUT / "data" / f"{STEM}_{a.layout}_{a.suite}{'_quick' if a.quick else ''}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=0, default=float) + "\n")
    np.savez_compressed(dst.with_suffix(".npz"), zmin=SWEEP["zmin"].astype(np.float32), lo=SWEEP["lo"], cell=SWEEP["cell"])
    print(f"{a.layout} {a.suite}: {len(rows)} actions, {n_bad} flagged, {n_att} more need the owner (j6 past +45); "
          f"least {json.dumps(least)}; flags by {by}; "
          f"{time.time() - t0:.0f} s -> {dst.relative_to(ROOT)}", flush=True)


# ------------------------------------------------------------------------------------------------ the joints alone
def joints_rows(plan, ctx, suite):
    """Per action: the joints' flags at the own moves' ends and the travel's joint moves' ends (no clearances), and the
    largest joint 6 (deg)."""
    T_AI, plan = frame_map(suite, ctx["g"], plan)
    mv = HP.Moves(plan, session(ctx, suite, ctx["T_BA"] @ T_AI))
    q = np.asarray(ctx["q_home"], float)
    T_prev = np.asarray(ctx["T_BE_home"], float)
    out = []
    for act in plan["actions"]:
        if act["kind"] not in ("touch", "hover", "press", "slide", "twist", "roll"):
            continue
        segs, T_end, cP, i_own = mv.action_moves(act, T_prev)
        poses = [sg[2] for sg in segs[:i_own] if sg[0] == "J"] + [sg[1] for sg in segs[i_own:i_own + 1]] + \
            [sg[2] for sg in segs[i_own:]]
        m = dict(sin5=9.0, q3=999.0, shoulder=1e9, lim=999.0, j5_margin=999.0, j6_margin=999.0)
        j6max, why = -1e9, []
        for T in poses:
            qq = CELL.SIM.ik(T, q)
            if qq is None:
                why = ["no IK"]
                break
            mk = HP.metrics(qq)
            for key in m:
                m[key] = min(m[key], mk[key])
            j6max = max(j6max, math.degrees(qq[5]))
            q = qq
        why = why or judge(m, {}, act["kind"])
        out.append((act["id"], why, j6max))
        T_prev = T_end
    return out


def cmd_reach(a):
    """Joints alone over a grid of positions of one suite's indenter (A, mm): the share of actions inside every window, and
    the share that needs the owner (joint 6 past +45 deg)."""
    s = SUITES[a.suite]
    ind = s["ind"]
    xs = [float(v) for v in a.x.split(",")]
    ys = [float(v) for v in a.y.split(",")]
    hs = [float(v) for v in a.h.split(",")]
    if a.suite in ("facing", "side"):
        plan = json.loads((LAB / HOLDER_BASE).read_text())
        plan["actions"] = hover_candidates(a.suite)
        plans = [(a.suite, plan)]
    else:
        plans = [(f, json.loads((LAB / f).read_text())) for f in s["plans"]]
    rows, t0 = [], time.time()
    for h in hs:
        for x in xs:
            for y in ys:
                LAYOUTS["_probe"] = {ind: (dict(c=(x, y), z_bot=h) if ind == "holder" else dict(c=(x, y), top=h))}
                ctx = context("_probe")
                res = []
                for _, plan in plans:
                    res += joints_rows(copy.deepcopy(plan), ctx, a.suite)
                n = len(res)
                ok = sum(1 for _, w, j in res if not w and j <= J6_UNATTENDED)
                att = sum(1 for _, w, j in res if not w and j > J6_UNATTENDED)
                why = {}
                for _, w, _ in res:
                    for x_ in w:
                        k = x_.split()[0]
                        why[k] = why.get(k, 0) + 1
                rows.append(dict(x=x, y=y, h=h, n=n, ok=ok, attended=att, fail=n - ok - att, why=why))
                print(f"{a.suite} x {x:5.0f} y {y:5.0f} h {h:4.0f}: {ok}/{n} pass, {att} attended, {n - ok - att} fail {why}",
                      flush=True)
    dst = OUT / "data" / f"{STEM}_reach_{a.suite}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(dict(suite=a.suite, indenter=ind, rows=rows, seconds=round(time.time() - t0, 1)), indent=0) + "\n")
    print("->", dst.relative_to(ROOT))


# ------------------------------------------------------------------------------------------------ figures
SUITE_TITLE = dict(sphere5="Sphere suite, 5 mm sphere", tall5="Tall suite, 5 mm sphere",
                   sphere12="Sphere suite, 12.5 mm sphere", rod="Rod suite, half rod",
                   facing="Facing hovers, pad-up tip", side="Side-by-side hovers, pad-down tip",
                   tall12="Tall suite, 12.5 mm sphere", keys="Homing on the keys")


def draw_plate(ax, g, hot=(), labels=True):
    """The plate's parts in plan view (A, mm): x to the right (away from the robot), y up. Parts named in hot are drawn
    red."""
    from matplotlib.patches import Circle, Polygon, Rectangle
    a = g["params"]
    x0, x1, y0, y1, _ = a["base"]
    ax.add_patch(Rectangle((-128, -128), 256, 256, fill=False, ls=(0, (4, 3)), lw=1.0, ec="0.45"))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, fc="0.93", ec="0.3", lw=1.0, zorder=0))

    def col(name):
        return "#c0392b" if name in hot else "0.25"

    kd = g["key"]
    bx, by = kd["k0"][0] + kd["margin"], kd["k0"][1] + kd["margin"]
    for i, k in enumerate(g["keys"]):
        R = IA._Rz(k["yaw"])[:2, :2]
        pts = np.array([[sx * bx, sy * by] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) @ R.T + k["c"]
        ax.add_patch(Polygon(pts, closed=True, fc="0.75", ec=col(f"key {i + 1}"), lw=1.2, zorder=3))
        if labels:
            ax.text(k["c"][0] - 14, k["c"][1], f"k{i + 1}", ha="right", va="center", fontsize=12)
    for name in ("sphere5", "sphere12"):
        sp = g[name]
        ax.add_patch(Circle(sp["c"], sp["col_r"], fc="0.80", ec=col(name), lw=1.2, zorder=3))
        ax.add_patch(Circle(sp["c"], sp["R"], fc="0.55", ec=col(name), lw=1.0, zorder=4))
        if labels:
            ax.text(sp["c"][0] + sp["col_r"] + 2, sp["c"][1], "R5" if name == "sphere5" else "R12.5",
                    ha="left", va="center", fontsize=12)
    r = g["rod"]
    ax.add_patch(Circle(r["c"], r["col_r"], fc="0.80", ec=col("rod"), lw=1.2, zorder=3))
    ax.add_patch(Rectangle((r["c"][0] - r["R"], r["c"][1] - r["half_len"]), 2 * r["R"], 2 * r["half_len"], fc="0.55",
                           ec=col("rod"), lw=1.0, zorder=4))
    if labels:
        ax.text(r["c"][0] - r["col_r"] - 2, r["c"][1], "rod", ha="right", va="center", fontsize=12)
    h = g["holder"]
    ax.add_patch(Circle(h["c"], h["col_r0"], fc="0.80", ec=col("holder column"), lw=1.2, zorder=3))
    for st in h["stations"]:
        nm = f"holder {'-y' if st['y_s'] < h['c'][1] else '+y'} {st.get('mode', 'up')} block"
        (bx0, by0, _), (bx1, by1, _) = st["block"]
        ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fc="0.55" if st.get("mode") == "down" else "0.70",
                               ec=col(nm), lw=1.0, zorder=4))
        (ax0, ay0, _), (ax1, ay1, _) = st["arm"]
        ax.add_patch(Rectangle((ax0, ay0), ax1 - ax0, ay1 - ay0, fc="0.70", ec=col(nm), lw=0.8, zorder=4))
    for i, ap in enumerate(h["apex"]):
        mode = CELL.TIPHOLD["stations"][i][1]
        ax.plot(ap[0], ap[1], marker="v" if mode == "down" else "^", ms=6, color=col("held tips"), zorder=6)
    if labels:
        ax.text(h["c"][0] + h["col_r0"] + 2, h["c"][1], "holder", ha="left", va="center", fontsize=12)
    ax.set_aspect("equal")


def cmd_figures(a):
    """Plan views of a layout: each suite's sweep (the lowest point of the robot and tool over each 2 mm cell, from
    replay's .npz) over the plate, the parts that a suite's flags name drawn red."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LAYOUTS.setdefault(a.layout, {})
    g = plate(a.layout)
    suites = [s for s in a.suites.split(",")]
    n = len(suites)
    nc = min(2, n)
    nr = int(math.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5.4 * nc, 5.6 * nr), squeeze=False, constrained_layout=True)
    im = None
    for k, s in enumerate(suites):
        ax = axes[k // nc][k % nc]
        f = OUT / "data" / f"{STEM}_{a.layout}_{s}_quick.json"
        hot, title = set(), SUITE_TITLE.get(s, s)
        if f.exists():
            d = json.loads(f.read_text())
            for r in d["rows"]:
                for w in r["flags"]:
                    if " mm to the " in w:
                        hot.add(w.split(" mm to the ")[1])
            title += f"\n{d['n'] - d['flagged']} of {d['n']} clear"
            npz = f.with_suffix(".npz")
            if npz.exists():
                z = np.load(npz)
                zmin = np.ma.masked_invalid(z["zmin"].T)
                lo, c = z["lo"], float(z["cell"])
                ext = (lo[0], lo[0] + c * zmin.shape[1], lo[1], lo[1] + c * zmin.shape[0])
                im = ax.imshow(zmin, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=300, alpha=0.75, zorder=2)
        draw_plate(ax, g, hot=hot, labels=(k == 0))
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_title(title, fontsize=15)
        ax.tick_params(labelsize=13)
        if k // nc == nr - 1:
            ax.set_xlabel("x (mm), away from the robot", fontsize=14)
        if k % nc == 0:
            ax.set_ylabel("y (mm)", fontsize=14)
    for k in range(n, nr * nc):
        axes[k // nc][k % nc].axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.6, location="bottom")
        cb.set_label("lowest robot or tool point (mm above the plate's underside)", fontsize=14)
        cb.ax.tick_params(labelsize=13)
    dst = OUT / "media" / f"{STEM}_{a.layout}_plan.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=110, bbox_inches="tight")
    print("->", dst.relative_to(ROOT))


# ------------------------------------------------------------------------------------------------ CAD
def cmd_build(a):
    """The plate of a layout as one STL (indenter_array.solid on this layout's geometry: the down station's block on its
    arm and gusset), with its parts JSON (geometry, volume, bounds)."""
    import ur5_calibration_mount as UM
    g = plate(a.layout)
    tm = UM.to_trimesh(IA.solid(UM.MeshK(), g))
    cad = OUT / "cad"
    cad.mkdir(parents=True, exist_ok=True)
    dst = cad / f"{STEM}_{a.layout}.stl"
    tm.export(dst)
    vol = float(tm.volume)
    rec = dict(layout=a.layout, geometry=g, stl=str(dst.relative_to(ROOT)), volume_cm3=round(vol / 1e3, 1),
               mass_g_solid=round(vol / 1e3 * 1.24, 0), bounds_mm=np.round(tm.bounds, 2).tolist(),
               watertight=bool(tm.is_watertight), bodies=len(tm.split(only_watertight=False)),
               when=time.strftime("%Y-%m-%d %H:%M"))
    (cad / f"{STEM}_{a.layout}_parts.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: v for k, v in rec.items() if k != "geometry"}, indent=1))


# ------------------------------------------------------------------------------------------------ renders
RENDER_PICKS = (("side", "hov-s-g15-s0-t+0"), ("facing", "hov-f-W2-t+0"), ("tall5", "tall-flank-"), ("sphere5", "ref-00"),
                ("sphere12", "grid-0"), ("rod", "rod-ref-a"), ("keys", "key 2, 0 mm out"))


def pose_of(ctx, suite, aid):
    """Joints at an action's own pose on the plate: a hover's pose, a press's depth, the seating on a key."""
    q0 = np.asarray(ctx["q_home"], float)
    if suite == "keys":
        i = int(aid.split()[1].rstrip(",")) - 1
        return home(ctx["g"], ctx["T_BA"], ctx["T_EM"], np.asarray(ctx["reg"]["q_home"], float), i)[1]
    if suite in ("facing", "side"):
        plan = json.loads((LAB / HOLDER_BASE).read_text())
        plan["actions"] = [x for x in hover_candidates(suite) if x["id"] == aid]
    else:
        plan = None
        for f in SUITES[suite]["plans"]:
            pl = json.loads((LAB / f).read_text())
            hit = [x for x in pl["actions"] if x["id"] == aid]
            if hit:
                pl["actions"] = hit
                plan = pl
                break
    T_AI, plan = frame_map(suite, ctx["g"], plan)
    mv = HP.Moves(plan, session(ctx, suite, ctx["T_BA"] @ T_AI))
    segs, _, _, i_own = mv.action_moves(plan["actions"][0], np.asarray(ctx["T_BE_home"], float))
    return CELL.SIM.ik(segs[i_own][2], q0)


def cmd_render(a):
    """Stills of a layout's plate in the cell (scripts/ur5_cell_render.py's scene: the robot, the tool, the 2x4s, the table),
    the plate in place of the four fixtures, its held tips on its holder, the robot at one clear action of each suite."""
    import tempfile
    import mujoco
    import trimesh
    from PIL import Image
    import ur5_cell_render as R
    ctx = context(a.layout)
    g, T_BA = ctx["g"], ctx["T_BA"]
    stl = OUT / "cad" / f"{STEM}_{a.layout}.stl"
    work = Path(tempfile.mkdtemp(prefix="plate_render_"))
    dummy = WORK / "dummy_0p01mm.stl"                     # inside the repository: the render logs paths relative to it
    WORK.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(0.01, 0.01, 0.01)).apply_translation((0.0, 0.0, -3000.0)).export(dummy)
    fx = R.fixtures()
    CELL.INDENTER_STL["sphere"], fx["sphere"] = stl, T_BA
    CELL.INDENTER_STL["rod"], R.TALL_STL, R.HOLDER_STL = dummy, dummy, dummy
    h = g["holder"]
    corr = C.inv(fx["holder"]) @ T_BA @ C.Tm(None, [h["c"][0], h["c"][1], h["z_bot"] - CELL.TIPHOLD["z_bot"]])
    orig = CELL.tiphold_T_IF
    CELL.tiphold_T_IF = lambda t=None, i=0: corr @ orig(t, i)               # noqa: E731
    c = C.apply(T_BA, np.array([0.0, 0.0, 90.0]))
    R.CAMERAS["plate"] = (tuple(C.apply(T_BA, np.array([-60.0, -640.0, 430.0]))), tuple(c), 42.0)   # from the -y side
    R.CAMERAS["plate_top"] = (tuple(C.apply(T_BA, np.array([0.0, 0.0, 1500.0]))), tuple(C.apply(T_BA, np.array([0.0, 1.0, 0.0]))), 22.0)
    w, hh = (int(v) for v in a.size.lower().split("x"))
    xml, log, _ = R.build(work, fx, w, hh)
    CELL.tiphold_T_IF = orig
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    media = OUT / "media"
    media.mkdir(parents=True, exist_ok=True)
    rows = {}
    for suite, prefix in RENDER_PICKS:
        f = OUT / "data" / f"{STEM}_{a.layout}_{suite}_quick.json"
        cand = [r["id"] for r in json.loads(f.read_text())["rows"] if not r["flags"] and r["id"].startswith(prefix)] \
            if f.exists() else []
        if cand:
            rows[suite] = cand[0]
    ren = mujoco.Renderer(m, hh, w)
    try:
        for suite, aid in rows.items():
            q = pose_of(ctx, suite, aid)
            if q is None:
                print(f"{suite} {aid}: no IK")
                continue
            R.set_pose(m, d, q)
            for cam in ("plate",) + (("plate_top",) if suite == "side" else ()):
                ren.update_scene(d, camera=cam)
                dst = media / f"{STEM}_{a.layout}_{suite}_{cam}.png"
                Image.fromarray(ren.render()).save(dst)
                print(f"{suite} {aid} -> {dst.relative_to(ROOT)}")
    finally:
        ren.close()


FILM_PICKS = (("side", "hov-s-g15-s0-t+0"), ("side", "hov-s-g25-s2-t+10"), ("facing", "hov-f-W2-t+10"),
              ("tall5", "tall-slide-flank-"), ("sphere5", "slide-dense-"), ("sphere12", "grid-0"), ("rod", "rod-slide"))


def joint_path(ctx, suite, aid, T_prev, q):
    """The joints along one action's moves on the plate (its travel from T_prev, then its own moves), every sample."""
    if suite in ("facing", "side"):
        plan = json.loads((LAB / HOLDER_BASE).read_text())
        plan["actions"] = [x for x in hover_candidates(suite) if x["id"] == aid]
    else:
        plan = None
        for f in SUITES[suite]["plans"]:
            pl = json.loads((LAB / f).read_text())
            hit = [x for x in pl["actions"] if x["id"] == aid]
            if hit:
                pl["actions"], plan = hit, pl
                break
    T_AI, plan = frame_map(suite, ctx["g"], plan)
    mv = HP.Moves(plan, session(ctx, suite, ctx["T_BA"] @ T_AI))
    segs, T_end, _, _ = mv.action_moves(plan["actions"][0], T_prev)
    qs = []
    for seg in segs:
        sm = HP.samples(seg, q)
        if sm is None:
            return None, T_prev, q
        qs += [qq for _, qq in sm]
        q = sm[-1][1]
    return qs, T_end, q


def cmd_film(a):
    """A film of one clear action of each of several suites on a layout's plate, each under a label bar."""
    import shutil
    import subprocess
    import tempfile
    import mujoco
    import trimesh
    from PIL import Image, ImageDraw, ImageFont
    import ur5_cell_render as R
    ctx = context(a.layout)
    g, T_BA = ctx["g"], ctx["T_BA"]
    work = Path(tempfile.mkdtemp(prefix="plate_film_"))
    WORK.mkdir(parents=True, exist_ok=True)
    dummy = WORK / "dummy_0p01mm.stl"
    trimesh.creation.box(extents=(0.01, 0.01, 0.01)).apply_translation((0.0, 0.0, -3000.0)).export(dummy)
    fx = R.fixtures()
    CELL.INDENTER_STL["sphere"], fx["sphere"] = OUT / "cad" / f"{STEM}_{a.layout}.stl", T_BA
    CELL.INDENTER_STL["rod"], R.TALL_STL, R.HOLDER_STL = dummy, dummy, dummy
    h = g["holder"]
    corr = C.inv(fx["holder"]) @ T_BA @ C.Tm(None, [h["c"][0], h["c"][1], h["z_bot"] - CELL.TIPHOLD["z_bot"]])
    orig = CELL.tiphold_T_IF
    CELL.tiphold_T_IF = lambda t=None, i=0: corr @ orig(t, i)               # noqa: E731
    R.CAMERAS["plate"] = (tuple(C.apply(T_BA, np.array([-60.0, -640.0, 430.0]))),
                          tuple(C.apply(T_BA, np.array([0.0, 0.0, 90.0]))), 42.0)
    w, hh = (int(v) for v in a.size.lower().split("x"))
    xml, _, _ = R.build(work, fx, w, hh)
    CELL.tiphold_T_IF = orig
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    frames = work / "frames"
    frames.mkdir()
    q = np.asarray(ctx["q_home"], float)
    T_prev = np.asarray(ctx["T_BE_home"], float)
    k = 0
    ren = mujoco.Renderer(m, hh, w)
    try:
        for suite, prefix in FILM_PICKS:
            f = OUT / "data" / f"{STEM}_{a.layout}_{suite}_quick.json"
            cand = [r["id"] for r in json.loads(f.read_text())["rows"] if not r["flags"] and r["id"].startswith(prefix)]
            if not cand:
                continue
            qs, T_prev, q = joint_path(ctx, suite, cand[0], T_prev, q)
            if not qs:
                continue
            step = max(1, len(qs) // a.frames_per_action)
            label = f"{SUITE_TITLE[suite]}: {cand[0]}"
            for qq in qs[::step] + [qs[-1]] * 8:
                R.set_pose(m, d, qq)
                ren.update_scene(d, camera="plate")
                im = Image.fromarray(ren.render())
                bar = Image.new("RGB", (w, 44), (255, 255, 255))
                ImageDraw.Draw(bar).text((12, 8), label, fill=(0, 0, 0), font=font)
                out = Image.new("RGB", (w, hh + 44), (255, 255, 255))
                out.paste(bar, (0, 0))
                out.paste(im, (0, 44))
                out.save(frames / f"f{k:05d}.png")
                k += 1
            print(f"{suite} {cand[0]}: {len(qs)} samples", flush=True)
    finally:
        ren.close()
    dst = OUT / "media" / f"{STEM}_{a.layout}_film.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(frames / "f%05d.png"), "-vf",
                    f"scale={w}:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", str(dst)], check=True)
    shutil.copy(frames / f"f{k // 2:05d}.png", OUT / "media" / f"{STEM}_{a.layout}_film_poster.png")
    shutil.rmtree(work, ignore_errors=True)
    print(f"{k} frames -> {dst.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--layout", default="array", choices=sorted(LAYOUTS))
    r.add_argument("--suite", required=True, choices=sorted(SUITES))
    r.add_argument("--quick", action="store_true", help="each action's own moves' ends and the travel's joint moves only")
    r.add_argument("--limit", type=int, default=0, help="about this many actions of each plan, evenly spread")
    rc = sub.add_parser("reach")
    rc.add_argument("--suite", required=True, choices=sorted(SUITES))
    rc.add_argument("--x", default="-40,0,40,80,110")
    rc.add_argument("--y", default="-110,-60,0,60,110")
    rc.add_argument("--h", default="120,160,180", help="tops (mm) of a sphere or the rod; the holder's z_bot")
    fg = sub.add_parser("figures")
    fg.add_argument("--layout", default="array")
    fg.add_argument("--suites", default="sphere5,tall5,rod,sphere12,facing,side")
    bd = sub.add_parser("build")
    bd.add_argument("--layout", default="array")
    rn = sub.add_parser("render")
    rn.add_argument("--layout", default="v2c")
    rn.add_argument("--size", default="1200x800")
    fm = sub.add_parser("film")
    fm.add_argument("--layout", default="v2d")
    fm.add_argument("--size", default="960x640")
    fm.add_argument("--fps", type=int, default=20)
    fm.add_argument("--frames-per-action", type=int, default=60)
    a = ap.parse_args()
    dict(replay=cmd_replay, reach=cmd_reach, figures=cmd_figures, build=cmd_build, render=cmd_render, film=cmd_film)[a.cmd](a)


if __name__ == "__main__":
    main()
