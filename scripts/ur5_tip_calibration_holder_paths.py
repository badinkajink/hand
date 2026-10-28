#!/usr/bin/env python3
"""The moves of a holder plan as the lab program makes them, checked for the UR5's singularities, its joint limits and the
cell's clearances (owner, 2026-10-27, through the lab: "a lot of the rotated motions where the wrist links rotate out
right will be really close to a singularity/joint limit where the rod indenter currently is").

From the homing pose lifted 12 mm out of the key (where the program's freedrive homing leaves the robot), each action's
travel follows the program's travel(): up 10 mm, across, down; or, past travel's turn_deg or far_mm, up until the pad
stands travel.lift_mm over the indenter's top, across (a joint move past joint_deg), down. A hover then goes straight
down to its pose and back up. A press, slide, twist or roll goes from its rest (standoff, 1.5 mm when absent) to the
depth at its largest force (--depth) and back. A slide, roll or twist makes its turns at that depth as the program's
contact_turns does (lab, 2026-10-28 00:00: a slide's turns of 2.3 deg took joint 6 about 2 deg past its press; --no-turns
leaves them out). A touch-off goes from its rest toward
beyond_mm past the expected contact and stops where
the robot's pad first meets the held pad (the program stops at the contact's onset), found by bisection on the pad's
clearance with the holder where --true-frame puts it; contact_mm in each row is that depth below the rest. The first
action's first move, out of the seating where the mount rests on the key, is left out of the clearances. Straight moves are sampled every 2 mm and 1 deg, joint moves every 1 deg of the largest joint's
travel, the inverse kinematics continued from sample to sample. Per action, over its travel and its own moves:

  sin5      the least |sin q5|: the wrist singularity at q5 = 0 or 180 deg
  q3        the least |q3| (deg): the elbow singularity, arm straight, at q3 = 0
  shoulder  the least distance (mm) of the wrist (wrist_1_link's origin) from joint 1's axis: the shoulder singularity
  limits    the least margin (deg) of any joint to +-360 deg, of joint 5 to the program's wrist window (30 to 100) and of
            joint 6 to the plan's -62 to +45
  clear     the least clearance (mm) of the robot's links, the mount, the finger and the HEX-E, every 5th sample and at
            the ends, with every fixture of --other (a kind with several folders, as the rod's three seatings, counts
            each, so the worst is kept)

An action is flagged when sin5 < 0.5 (q5 within 30 deg of 0 or 180), q3 < 20 deg, the shoulder distance < 150 mm, a
joint's margin < 10 deg, joint 5 or 6 leaves its window, or a clearance falls under the plan checks' needs (robot 8,
mount 3, finger 0.8, HEX-E 6 mm). --filtered writes the plan without the flagged hovers.

    .venv/bin/python scripts/ur5_tip_calibration_holder_paths.py PLAN --session FOLDER \\
        --other sphere:FOLDER --other rod:F1,F2,F3 --other tall:FOLDER [--filtered OUT]
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
import ur5_cb3_cell as CELL  # noqa: E402

C = CELL.C
SIM = CELL.SIM
EZ = np.array([0.0, 0.0, 1.0])
NEED = dict(robot=8.0, mount=3.0, finger=0.8, hex_e=6.0)
GROUPS = dict(mount=("adapter", "fork", "clamp_plate", "stop_wedge"), finger=("dip_link", "cover", "cap"), hex_e=("hex_e",),
              pad=("pad",))
J5_WIN, J6_WIN = (30.0, 100.0), (-62.0, 45.0)


def rot_z(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])


def rot_from_to(a, b):
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


class Moves:
    def __init__(self, plan, session):
        self.plan = plan
        self.geo = C.Geometry(plan)
        self.reg = C.Registration(self.geo).load(json.loads((Path(session) / "registration.json").read_text()))
        self.T_EP = self.reg.T_EM @ self.reg.T_MP
        self.R_ref = np.asarray(plan["geometry"]["T_IM_home"], float)[:3, :3]
        ind = plan["geometry"]["indenter"]
        self.c_I = np.asarray(ind["centre"], float)
        self.top_I = self.c_I + float(ind["radius"]) * EZ
        self.cfg = dict(turn_deg=20.0, far_mm=60.0, lift_mm=60.0, joint_deg=45.0)
        self.cfg.update(plan.get("travel", {}))
        self.pts_P = np.asarray(plan["geometry"]["surface"]["pts"], float)

    def T_BE(self, p_P, n_P, twist, at_I):
        R = rot_z(twist) @ self.R_ref @ rot_from_to(n_P, [1.0, 0.0, 0.0])
        T_IP = C.Tm(R, np.asarray(at_I, float) - R @ np.asarray(p_P, float))
        return self.reg.T_BI @ T_IP @ C.inv(self.T_EP)

    def up(self, T, mm):
        return C.Tm(T[:3, :3], T[:3, 3] + self.reg.T_BI[:3, :3] @ EZ * mm)

    def lifted(self, T, h):
        pts = C.apply(C.inv(self.reg.T_BI) @ T @ self.T_EP, self.pts_P)
        return self.up(T, max(10.0, h + self.top_I[2] - float(pts[:, 2].min())))

    def travel(self, Tc, Tt):
        """The segments ('L' or 'J', from, to) of the program's travel from Tc to Tt."""
        mm, deg = C.pose_error(Tc, Tt)
        if mm < 1e-3 and deg < 1e-3:
            return []
        if deg > self.cfg["turn_deg"] or mm > self.cfg["far_mm"]:
            a, b = self.lifted(Tc, self.cfg["lift_mm"]), self.lifted(Tt, self.cfg["lift_mm"])
            c = self.lifted(Tt, 0.0)
            return [("L", Tc, a), ("J" if deg > self.cfg["joint_deg"] else "L", a, b), ("L", b, c), ("L", c, Tt)]
        a, b = self.up(Tc, 10.0), self.up(Tt, 10.0)
        return [("L", Tc, a), ("L", a, b), ("L", b, Tt)]

    def turns(self, a, T0):
        """The straight steps of the program's contact_turns from the pressed pose T0: for each of a['moves'], turns to each
        of its degs and back to 0 in steps of at most 0.5 deg (run_turns), about the sphere's centre and the horizontal
        across dir_deg (slide), the same axis through the top (roll) or the vertical through the top (twist)."""
        R_BI = self.reg.T_BI[:3, :3]
        c_B, top_B = C.apply(self.reg.T_BI, self.c_I), C.apply(self.reg.T_BI, self.top_I)
        segs, T = [], T0
        for mv in a.get("moves", []):
            ang = math.radians(mv.get("dir_deg", 0.0))
            u_I = np.array([math.cos(ang), math.sin(ang), 0.0])
            centre, axis = dict(slide=(c_B, R_BI @ np.cross(EZ, u_I)), roll=(top_B, R_BI @ np.cross(EZ, u_I)))\
                .get(a["kind"], (top_B, R_BI @ EZ))
            cur = 0.0
            for d in list(mv["degs"]) + [0.0]:
                n = max(1, int(math.ceil(abs(d - cur) / 0.5)))
                for j in range(1, n + 1):
                    T1 = C.turn_about(T0, centre, axis, cur + (d - cur) * j / n)
                    segs.append(("L", T, T1))
                    T = T1
                cur = d
        return segs

    def action_moves(self, a, T_prev):
        """(the action's segments including its travel, the pose it ends at, the contact point in P or None, the index of
        the action's first own segment after its travel)."""
        tg = a["target"]
        if a["kind"] == "hover":
            T_pose = self.T_BE(tg["p"], tg["n"], a.get("twist", 0.0), a["at_I"])
            h = float(a.get("ref_up_mm", 40.0))
            T_ref = self.up(T_pose, h)
            tr = self.travel(T_prev, T_ref)
            return tr + [("L", T_ref, T_pose), ("L", T_pose, T_ref)], T_ref, None, len(tr)
        st = float(a.get("standoff", 5.0 if a["kind"] == "touch" else 1.5))
        off = a.get("offset", (0.0, 0.0))                  # the program moves the contact point by -offset in I
        at = self.top_I + EZ * st - np.array([off[0], off[1], 0.0])
        T_rest = self.T_BE(tg["p"], tg["n"], a.get("twist", 0.0), at)
        if a["kind"] == "touch":
            deep = float(a.get("beyond_mm", 1.2))
        elif DEPTH == "max":
            deep = float(a.get("max_depth_mm", 2.0))
        else:                                                  # the depth at the action's largest force, and 0.3 mm
            F = float(a.get("stop_N") or a.get("press_N") or max(a.get("levels") or [2.0]))
            reg_ = (tg.get("region") or "flank")
            d_top, f_top = PRESS_DEPTH.get(reg_, PRESS_DEPTH["flank"])
            deep = min(float(a.get("max_depth_mm", 2.5)), d_top * (F / f_top) ** (2.0 / 3.0) + 0.3)
        T_deep = self.up(T_rest, -(st + deep))
        tr = self.travel(T_prev, T_rest)
        own = [("L", T_rest, T_deep)] + (self.turns(a, T_deep) if TURNS and a["kind"] in ("slide", "roll", "twist") else [])
        return tr + own + [("L", T_deep, T_rest)], T_rest, tg["p"], len(tr)


def slerp_T(T0, T1, s):
    from scipy.spatial.transform import Rotation, Slerp
    r = Slerp([0.0, 1.0], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))(s).as_matrix()
    return C.Tm(r, (1 - s) * T0[:3, 3] + s * T1[:3, 3])


def samples(seg, q0):
    kind, T0, T1 = seg
    if kind == "J":
        q1 = SIM.ik(T1, q0)
        if q1 is None:
            return None
        n = max(2, int(math.ceil(math.degrees(np.max(np.abs(q1 - q0))))) + 1)
        return [(None, q0 + (q1 - q0) * s) for s in np.linspace(0.0, 1.0, n)]
    mm, deg = C.pose_error(T0, T1)
    n = max(2, int(math.ceil(max(mm / 2.0, deg))) + 1)
    out, q = [], q0
    for s in np.linspace(0.0, 1.0, n):
        T = slerp_T(T0, T1, s)
        q = SIM.ik(T, q)
        if q is None:
            return None
        out.append((T, q))
    return out


def metrics(q):
    q = np.asarray(q, float)
    lf = CELL.link_frames(q)
    w = lf["wrist_1_link"][:3, 3]
    j5, j6 = math.degrees(q[4]), math.degrees(q[5])
    return dict(sin5=abs(math.sin(q[4])), q3=abs(math.degrees(math.atan2(math.sin(q[2]), math.cos(q[2])))),
                shoulder=float(np.hypot(w[0], w[1])), lim=float(np.min(360.0 - np.abs(np.degrees(q)))),
                j5_margin=min(j5 - J5_WIN[0], J5_WIN[1] - j5), j6_margin=min(j6 - J6_WIN[0], J6_WIN[1] - j6))


def to_contact(seg, q0, reg):
    """The approach segment cut at the first contact of the robot's pad with the holder (reg.T_BI), and its depth (mm)."""
    _, T0, T1 = seg

    def pad(s):
        T = slerp_T(T0, T1, s)
        qq = SIM.ik(T, q0)
        return 1e9 if qq is None else clear_groups(qq, T, reg, None).get("pad", 1e9)

    L = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
    if pad(1.0) > 0.0:
        return seg, None
    if pad(0.0) <= 0.0:
        return ("L", T0, T0), 0.0
    lo, hi = 0.0, 1.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if pad(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return ("L", T0, slerp_T(T0, T1, hi)), hi * L


KIND = "tiphold"                                            # the plan's indenter kind (plan["indenter"]), set in run
DEPTH = "model"                                             # --depth: "model" (the force's depth) or "max" (max_depth_mm)
TURNS = True                                                # --no-turns: slides, rolls and twists without their turns
# the pad's depth (mm) at a force (N) by region, as R29's builder took it (scripts/ur5_tip_calibration_tall_pass.py: the
# depth at the largest force), scaled as F^(2/3) to other forces
PRESS_DEPTH = dict(face=(1.773, 5.0), edge=(1.238, 3.0), flank=(1.0, 2.0), tip=(1.0, 2.0))


def clear_groups(q, T, reg, contact_P):
    cl = CELL.clearance(q, T, reg.T_BI, reg.T_EM, KIND, contact_P=contact_P)
    g = {}
    for k, v in cl.items():
        grp = next((n for n, names in GROUPS.items() if k in names), "robot")
        g[grp] = min(g.get(grp, 1e9), float(v))
    return g


def run(a):
    global KIND, DEPTH, TURNS
    DEPTH, TURNS = a.depth, not a.no_turns
    plan = json.loads(Path(a.plan).read_text())
    KIND = plan.get("indenter", "tiphold")
    mv = Moves(plan, a.session)
    reg = mv.reg
    CELL.BOARD.update(plan["checks"]["board"])
    CELL.BOARD["half_mm"] = 400.0
    other = [o for o in plan["checks"].get("other", [])]
    for spec in a.other:
        kind, folders = spec.split(":", 1)
        other = [o for o in other if o["kind"] != kind]
        for f in folders.split(","):
            other.append(dict(kind=kind, T_BI=json.loads((Path(f) / "registration.json").read_text())["T_BI"]))
    if a.only:
        other = [o for o in other if o["kind"] in a.only]
    CELL.OTHER[:] = [(np.array(o["T_BI"], float), o["kind"]) for o in other]
    T_true = np.array(json.loads(Path(a.true_frame).read_text())["T_BI"], float) if a.true_frame else \
        np.array(plan.get("holder", {}).get("T_BI_true") or reg.T_BI, float)
    reg_true = copy.copy(reg)                               # the holder where it stands; the moves stay in the program's I
    reg_true.T_BI = T_true
    q = np.asarray(reg.q_home, float)
    T_prev = np.asarray(reg.T_BE_home, float)
    T_prev = C.Tm(T_prev[:3, :3], T_prev[:3, 3] + T_prev[:3, :3] @ reg.T_EM[:3, :3] @ np.array([-12.0, 0.0, 0.0]))
    q = SIM.ik(T_prev, q)
    rows, flagged = [], []
    skipped = []
    for ai, act in enumerate(plan["actions"]):
        if act["kind"] not in ("touch", "hover", "press", "slide", "twist", "roll"):   # gravity poses: the lab's simulator
            skipped.append(act["id"])
            continue
        segs, T_end, cP, i_own = mv.action_moves(act, T_prev)
        m = dict(sin5=9.0, q3=999.0, shoulder=1e9, lim=999.0, j5_margin=999.0, j6_margin=999.0)
        cl = {}
        bad = None
        j6r = [1e9, -1e9]
        depth = None
        for si, seg in enumerate(segs):
            if act["kind"] == "touch" and si == i_own:            # the approach stops at the first contact
                seg, depth = to_contact(seg, q, reg_true)
                segs[-1] = ("L", seg[2], segs[-1][2])
            sm = samples(seg, q)
            if sm is None:
                bad = f"no IK on segment {si}"
                break
            own = si >= i_own                               # the action's own moves; the contact is excluded there
            for k, (T, qq) in enumerate(sm):
                mk = metrics(qq)
                for key in m:
                    m[key] = min(m[key], mk[key])
                j6r = [min(j6r[0], math.degrees(qq[5])), max(j6r[1], math.degrees(qq[5]))]
                if (k % 5 == 0 or k == len(sm) - 1) and (ai, si) != (0, 0):
                    TT = T if T is not None else SIM.fk(qq)
                    g = clear_groups(qq, TT, reg_true, cP if own else None)
                    for key, v in g.items():
                        cl[key] = min(cl.get(key, 1e9), v)
            q = sm[-1][1]
        why = [bad] if bad else []
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
        for key, need in NEED.items():
            if cl.get(key, 1e9) < need:
                why.append(f"{key} {cl[key]:.2f} mm")
        row = dict(id=act["id"], **{k: round(v, 3) for k, v in m.items()}, clear={k: round(v, 2) for k, v in cl.items()},
                   flags=why)
        row["j6_deg"] = [round(j6r[0], 1), round(j6r[1], 1)]
        if act["kind"] == "touch":
            row["contact_mm"] = None if depth is None else round(depth, 2)
        rows.append(row)
        if why:
            flagged.append(act["id"])
            print(f"  {act['id']}: {'; '.join(why)}", flush=True)
        T_prev = T_end
    keys = ("sin5", "q3", "shoulder", "lim", "j5_margin", "j6_margin")
    least = {k: min(r[k] for r in rows) for k in keys}
    lc = {}
    for r in rows:
        for k, v in r["clear"].items():
            lc[k] = min(lc.get(k, 1e9), v)
    print(f"{len(rows)} actions, {len(flagged)} flagged; least over all: "
          + ", ".join(f"{k} {v:.2f}" for k, v in least.items()) + "; clear " + str({k: round(v, 2) for k, v in lc.items()}))
    if skipped:
        print(f"{len(skipped)} actions of other kinds not replayed: {sorted(set(x.split('-')[0] for x in skipped))}")
    out = dict(plan=str(a.plan), session=str(a.session), kind=KIND, skipped=skipped,
               replay=f"touch-offs to first contact; presses, slides, twists and rolls to the depth at their largest force "
                      f"({DEPTH}), {'with' if TURNS else 'without'} the turns of slides, rolls and twists", holder_T_BI=np.round(T_true, 5).tolist(), other=[dict(kind=o["kind"],
               origin_B=np.round(np.array(o["T_BI"])[:3, 3], 1).tolist()) for o in other], least=least, least_clear=lc,
               rows=rows, flagged=flagged)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("->", a.out)
    if a.filtered:
        p2 = copy.deepcopy(plan)
        drop = set(flagged) | {x + "-rep" for x in flagged}
        p2["actions"] = [x for x in plan["actions"] if x["id"] not in drop or x["kind"] != "hover"]
        p2["checks"]["other"] = other
        p2.setdefault("notes", []).append(f"paths checked by scripts/ur5_tip_calibration_holder_paths.py: {len(flagged)} "
                                          f"actions flagged, the hovers among them dropped")
        Path(a.filtered).write_text(json.dumps(p2, indent=1, default=C.jsonable) + "\n")
        print("filtered ->", a.filtered, len(p2["actions"]), "actions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan")
    ap.add_argument("--session", required=True)
    ap.add_argument("--other", action="append", default=[], help="kind:folder[,folder...] (each folder's registration T_BI)")
    ap.add_argument("--out", default=str(ROOT / "logs/20261027-holder_paths.json"))
    ap.add_argument("--filtered", default=None)
    ap.add_argument("--depth", default="model", choices=("model", "max"), help="a press's replayed depth: at its largest "
                    "force by PRESS_DEPTH and 0.3 mm (model), or its max_depth_mm (max)")
    ap.add_argument("--no-turns", action="store_true", help="replay slides, rolls and twists without their turns")
    ap.add_argument("--true-frame", default=None, help="a registration.json whose T_BI places the holder (default: the "
                    "plan's holder.T_BI_true, else the program's frame); to check another hypothesis of where it stands")
    ap.add_argument("--only", action="append", default=[], help="keep only these fixture kinds (the rod alone: --only rod)")
    run(ap.parse_args())
