#!/usr/bin/env python3
"""Plan of the fingertip's calibration session on the lab's UR5 (CB3) with the OptoForce HEX-E, for
src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py: the nominal frames of the tool stack, the pad's surface, the
targets of every stage in the pad and indenter frames, and their checks on an assumed placement of the indenter (reach of
every pose by the CB3's kinematics, clearance of the robot's links, the HEX-E, the mount and the finger to the indenter
and the plywood, the turn of the tool along each travel).

The pad's targets are FE surface points (docs/experiments/20261020-fingertip_contact_estimation/data/
20261020-b_v2_surface_basis.npz) that the sphere reaches along their normal with the finger's cover walls, cap and link
clear of the indenter, chosen by farthest-point sampling in four regions (face: n_x > 0.9; edge: 0.5-0.9; flank:
|n_y| > 0.7; tip end: n_z > 0.7). Each target turns about the vertical through its contact (twist) until the robot reaches
all its poses and clears the cell.

    .venv/bin/python scripts/ur5_tip_calibration_plan.py [--indenter sphere|rod] [--placement x,y,z,yaw] [--order joint6]
            [--gravity-roll-offset 30]
        placement: the indenter's base in the robot's base frame (mm, deg; x_I points away from the robot)
        gravity-roll-offset: deg added to every roll of the gravity stations; with 30 none sits half a turn from the
        homing pose, where the IK nearest the homing branch may take either side
        order joint6: each stage's actions ordered by joint 6 at their rest poses, solved nearest the homing branch as
        the session solves its joint moves, so that the HEX-E's cable is turned less (2026-10-22: the sensor dropped
        out twice while joint 6 turned). The gravity sweeps go out and back (every other station in order of joint 6,
        then the rest in reverse), which keeps the direction of gravity apart from time for the fit's drift term; the
        pad and grid stages are sorted, the grid's reference presses kept at their places. The travel checks run on
        the new order.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_cb3_cell as CELL  # noqa: E402

C, SIM = CELL.C, CELL.SIM
OUTDIR = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz"
PAYLOAD_V3 = ROOT / "docs/experiments/20261021-ur5_calibration_mount_v3/20261021-ur5_calibration_mount_v3_payload.json"
FACE_CENTRE = np.array([8.5, 0.0, 10.86])            # the contact point of the mount's design (the centre magnet's z)
LEVELS = dict(face=[0.25, 0.5, 1.0, 2.0, 3.5, 5.0], edge=[0.25, 0.5, 1.0, 2.0, 3.0], flank=[0.25, 0.5, 1.0, 2.0],
              tip=[0.25, 0.5, 1.0, 2.0])
COUNTS = dict(grid=dict(face=30, edge=22, flank=16, tip=6), touch=dict(face=12, edge=10, flank=10, tip=4))
TWISTS = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0)
ROBOT_LINKS = ("forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")
CLEAR = dict(robot=8.0, hex_e=6.0, mount=3.0, finger=0.8, pad=0.3, ply=5.0, travel=3.0)
# Joint 5 at least 30 deg from the wrist's singularity (0 and 180 deg). On 2026-10-22 a travel took it to 173.5 deg and
# pressed the HEX-E's cable port against the forearm (the owner stopped the robot); the limit was 7 deg (sin 0.12).
WRIST_SIN = 0.5
# Joint 5 inside the program's --wrist-window on the robot's branch (deg). With joint 5 past about 100 deg the tool folds
# back toward the forearm; the twisted touch-offs reached 149 deg, which the program refuses.
WRIST_WINDOW = (30.0, 100.0)


def nominal(kind, mount_turn_deg=0.0):
    """mount_turn_deg: x_M's turn about the tool axis from tool0's x (the lab's mount measured -92.15 on 2026-10-22: the
    palmar side along tool0's -y, from two hand seatings on the keys, logs/20261022-ur5_homing_by_hand)."""
    B = np.load(BASIS)
    T_IM = np.eye(4)
    T_IM[:3, :3] = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]       # z_M -> x_I, y_M -> y_I, x_M -> -z_I
    a_M = C.apply(CELL.T_M_FROM_P, FACE_CENTRE)                              # (10.55, 0, 93.36)
    top = 46.0 if kind == "sphere" else 42.5
    T_IM[:3, 3] = np.array([0.0, 30.0, top + 5.0]) - T_IM[:3, :3] @ a_M      # homing: contact 30 mm to +y, 5 mm up
    ind = (dict(kind="sphere", centre=[0.0, 0.0, 41.0], radius=5.0) if kind == "sphere"
           else dict(kind="rod", centre=[0.0, 0.0, 30.0], radius=12.5, axis=[0.0, 1.0, 0.0], half_length=10.0))
    return dict(
        T_EM=C.Tm(C.rot(C.EZ, mount_turn_deg), [0.0, 0.0, CELL.HEX["height_mm"]]), T_MP=CELL.T_M_FROM_P,
        T_ES=C.Tm(None, [0.0, 0.0, CELL.HEX["height_mm"]]),
        T_IM_home=T_IM, indenter=ind,
        surface=dict(pts=B["pts"], normals=B["normals"], tris=B["tris"]),
        mount_face=dict(point=[17.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0]),
        notes=dict(
            T_EM="the HEX-E's tool face 35.0 mm from the UR flange (OptoForce HEX-E V1.0 data sheet, built-in UR adapter); "
                 + ("x_M toward the HEX-E's dowel, assumed on tool0's x until the session fits it" if not mount_turn_deg
                    else f"x_M turned {mount_turn_deg:+.2f} deg about the tool axis from tool0's x, as measured by hand "
                         "seatings on the keys; the session fits it"),
            T_MP="the pad's tip frame in the mount frame of UR5 mount v3 with DIP link b v4: a translation",
            T_ES="the HEX-E's moment origin assumed on its tool face's centre, its axes on the flange's",
            T_IM_home="the mount at homing in the indenter frame: socket on the key, the face centre 30 mm to +y and 5 mm "
                      "above the indenter's top (scripts/ur5_calibration_mount.py homing())"))


def payload():
    """Mass and centre of gravity past the UR flange for set_payload: mount v3's parts (its payload record without its
    HEX-E QC line) and the OptoForce HEX-E, 260 g at half its height."""
    d = json.loads(PAYLOAD_V3.read_text())
    items = [(i["mass_g"], np.array(i["cog_M_mm"])) for i in d["items"] if i["item"] != "hex_e"]
    m = sum(w for w, _ in items)
    c = sum(w * p for w, p in items) / m + np.array([0.0, 0.0, CELL.HEX["height_mm"]])
    mh = CELL.HEX["mass_kg"] * 1e3
    tot = m + mh
    cog = (m * c + mh * np.array([0.0, 0.0, CELL.HEX["height_mm"] / 2])) / tot
    return dict(kg=round(tot / 1e3, 4), cog_mm=[round(float(v), 1) for v in cog], mount_and_finger_g=round(m, 1),
                hex_e_g=mh, note="mount v3 parts, DIP link b v4, cover, board, pad and screws (mount v3 payload record) + "
                                 "OptoForce HEX-E 260 g (data sheet V1.0) at 17.5 mm")


def regions(N):
    reg = np.full(len(N), "", object)
    reg[N[:, 0] > 0.9] = "face"
    reg[(N[:, 0] > 0.5) & (N[:, 0] <= 0.9)] = "edge"
    reg[(np.abs(N[:, 1]) > 0.7) & (N[:, 0] <= 0.5)] = "flank"
    reg[(N[:, 2] > 0.7) & (N[:, 0] <= 0.5)] = "tip"
    return reg


def fps(P, k, seed_pt=None):
    """Farthest-point sampling of k of the points P (indices)."""
    if len(P) == 0 or k <= 0:
        return []
    i0 = int(np.argmin(np.linalg.norm(P - (seed_pt if seed_pt is not None else P.mean(0)), axis=1)))
    sel = [i0]
    d = np.linalg.norm(P - P[i0], axis=1)
    while len(sel) < min(k, len(P)):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return sel


class Checker:
    """Reach and clearance of the plan's poses on an assumed indenter placement T_BI."""

    def __init__(self, geo, T_BI, kind):
        self.geo, self.T_BI, self.kind = geo, T_BI, kind
        self.T_EM, self.T_EP = geo.T_EM0, geo.T_EM0 @ geo.T_MP0
        self.q = None

    def T_BE(self, T_IP):
        return self.T_BI @ T_IP @ C.inv(self.T_EP)

    def ik(self, T, q0=None):
        q0 = self.q if q0 is None else q0
        q = SIM.ik(T, q0)
        if q is None or np.any(np.abs(q) > 2 * math.pi - 0.05):
            return None
        return q

    def seed(self, T):
        """The IK branch at the homing pose: the upper arm above the table (shoulder lift in [-180, 0] deg), wrist and
        elbow away from their singularities, the most clearance of the robot's links, then the least wrist-3 turn."""
        best = None
        seen = []
        for q2 in (-0.4, -1.0, -1.6, -2.2, -2.8):
            for q3 in (2.4, 1.6, 0.8, -0.8, -1.6, -2.4):
                for q4 in (-2.5, -1.6, -0.6, 0.6, 1.6):
                    for q5 in (1.57, -1.57):
                        for q1 in np.linspace(-math.pi, math.pi, 13)[:-1]:
                            q = SIM.ik(T, np.array([q1, q2, q3, q4, q5, 0.0]))
                            if q is None:
                                continue
                            q = (q + math.pi) % (2 * math.pi) - math.pi
                            if not (-math.pi <= q[1] <= 0.0) or abs(math.sin(q[4])) < 0.3 or abs(math.sin(q[2])) < 0.2:
                                continue
                            if any(np.abs(q - s_).max() < 1e-3 for s_ in seen):
                                continue
                            seen.append(q)
        for q in seen:
            cl = CELL.clearance(q, T, self.T_BI, self.T_EM, self.kind)
            rob = min(v for k, v in cl.items() if k in ROBOT_LINKS)
            cost = -min(rob, 60.0) + 5.0 * abs(q[5])
            if best is None or cost < best[0]:
                best = (cost, q, rob)
        if best is None:
            raise SystemExit("no IK branch with the upper arm above the table at the homing pose: move the indenter")
        self.q = best[1]
        self.seed_info = dict(branches=len(seen), robot_clear_mm=round(best[2], 1), q_deg=np.degrees(best[1]).round(1).tolist())
        return self.q

    def pose_ok(self, T, contact_P=None, need=CLEAR, extra=0.0, contact_M=None):
        q = self.ik(T)
        if q is None:
            return False, dict(reason="no IK"), None
        if not (-math.pi - 0.2 <= q[1] <= 0.2):
            return False, dict(reason="upper arm below the table"), q
        if abs(math.sin(q[4])) < WRIST_SIN:
            return False, dict(reason="wrist within 30 deg of its singularity", q_deg=np.degrees(q).round(1).tolist()), q
        if not WRIST_WINDOW[0] <= math.degrees(q[4]) <= WRIST_WINDOW[1]:
            return False, dict(reason="joint 5 outside the wrist window", q_deg=np.degrees(q).round(1).tolist()), q
        sok, sc = CELL.self_ok(q, self.T_EM)
        if not sok:
            return False, dict(reason="the tool near the robot's own links", self={k: round(v, 1) for k, v in sc.items()},
                               q_deg=np.degrees(q).round(1).tolist()), q
        cl = CELL.clearance(q, T, self.T_BI, self.T_EM, self.kind, contact_P=contact_P, contact_M=contact_M)
        worst = {}
        for k, v in cl.items():
            grp = ("robot" if k in ROBOT_LINKS else "hex_e" if k == "hex_e" else
                   "mount" if k in ("adapter", "fork", "clamp_plate") else "pad" if k == "pad" else "finger")
            worst[grp] = min(worst.get(grp, 1e9), v)
        ok = all(worst[g] >= need[g] + extra for g in worst)
        return ok, dict(clear={k: round(v, 2) for k, v in worst.items()}, self={k: round(v, 1) for k, v in sc.items()},
                        q_deg=np.degrees(q).round(1).tolist()), q

    def joint_path_ok(self, q0, q1, n=12, need=CLEAR["travel"]):
        for s in np.linspace(0, 1, n)[1:-1]:
            q = q0 + (q1 - q0) * s
            T = SIM.fk(q)
            if abs(math.sin(q[4])) < WRIST_SIN or not WRIST_WINDOW[0] <= math.degrees(q[4]) <= WRIST_WINDOW[1]:
                return False, f"joint 5 at {math.degrees(q[4]):.0f} deg"
            sok, sc = CELL.self_ok(q, self.T_EM)
            if not sok:
                return False, "tool near the arm: " + ", ".join(f"{k} {v:.1f} mm" for k, v in sc.items())
            cl = CELL.clearance(q, T, self.T_BI, self.T_EM, self.kind)
            if min(cl.values()) < need:
                return False, round(min(cl.values()), 2)
        return True, None


def lifted(geo, T_BI, T_BE, T_EP, h_mm):
    pts = C.apply(C.inv(T_BI) @ T_BE @ T_EP, geo.pts)
    top = (geo.c_I + geo.R * C.EZ)[2]
    dz = max(10.0, h_mm + top - float(pts[:, 2].min()))
    return C.Tm(T_BE[:3, :3], T_BE[:3, 3] + T_BI[:3, :3] @ C.EZ * dz)


def tip_reachable(geo, kind, idx, twist=0.0):
    """The finger's parts and the rest of the pad clear the indenter when the sphere (the rod) touches pad point idx
    along its normal, and 2 mm deeper."""
    p, n = geo.pts[idx], geo.nrm[idx]
    out = {}
    for depth in (0.0, -2.0):
        T_IP = geo.T_IP(p, n, twist, depth)
        T_IM = T_IP @ C.inv(CELL.T_M_FROM_P)
        for name, pts in CELL.tool_points().items():
            if name in ("hex_e", "adapter"):
                continue
            PI = C.apply(T_IM, pts)
            if name == "pad":
                pP = C.apply(C.inv(CELL.T_M_FROM_P), pts)
                near = np.linalg.norm(pP - p, axis=1) < (4.0 if depth == 0 else 6.0)
                d = min(float(CELL.indenter_sdf(PI[~near], kind).min()),
                        float(CELL.indenter_sdf(PI[near], kind, tip=False).min()))
            else:
                d = float(CELL.indenter_sdf(PI, kind).min())
            out[name] = min(out.get(name, 1e9), d)
    ok = (out["pad"] >= -0.05 and min(out[k] for k in ("dip_link", "cover", "cap")) >= CLEAR["finger"]
          and min(out[k] for k in ("fork", "clamp_plate")) >= CLEAR["mount"])
    return ok, out


def build(kind="sphere", placement=(0.0, -450.0, 56.0, -90.0), out=None, block_mm=38.0, mount_turn_deg=0.0,
          block_half_mm=(50.0, 30.0), order_by="plan", roll_offset_deg=0.0, q_seed_deg=None, touch_twist_deg=0.0):
    """placement: the indenter's base in the base frame (x, y, z mm, yaw deg); block_mm: the block under the indenter;
    block_half_mm: its half sizes along x_I, y_I (a 2 by 4 under the indenter runs far along y_I); order_by: "plan" (the
    order in which the actions are made) or "joint6" (module docstring); roll_offset_deg: added to every roll of the
    gravity stations; q_seed_deg: the robot's joints at its seating on the key, whose IK branch the session follows
    (the program solves its moves nearest them). Without it the builder picks the branch of most clearance, which on
    2026-10-22 was another branch from the lab robot's (joint 1 -92 against 59 deg), so its checks described another
    arm. touch_twist_deg: the flank and edge touch-offs turned by +-t, else +-2t, else 0 (step 5)."""
    t_start = time.time()
    CELL.BOARD["block_mm"] = float(block_mm)
    CELL.BOARD["block_half_mm"] = tuple(float(v) for v in block_half_mm)
    g = nominal(kind, mount_turn_deg)
    plan = dict(version=C.PLAN_VERSION, created=time.strftime("%Y-%m-%d %H:%M"), indenter=kind,
                geometry=g, payload=payload(),
                travel=dict(turn_deg=20.0, far_mm=60.0, lift_mm=60.0, joint_deg=45.0),
                safety=dict(abort_N=10.0, abort_Nm=1.5, box_mm=400.0),
                homing=dict(repeats=3), actions=[])
    geo = C.Geometry(json.loads(json.dumps(plan, default=C.jsonable)))
    x, y, z, yaw = placement
    T_BI = C.Tm(C.rot(C.EZ, yaw), [x, y, z])
    chk = Checker(geo, T_BI, kind)
    reg = regions(geo.nrm)
    i_ref = int(np.argmin(np.linalg.norm(geo.pts - FACE_CENTRE, axis=1)))
    p_ref, n_ref = geo.pts[i_ref], geo.nrm[i_ref]
    # 1. which surface points the indenter reaches
    reach = np.zeros(len(geo.pts), bool)
    why = {}
    for i in range(len(geo.pts)):
        if not reg[i]:
            continue
        ok, o = tip_reachable(geo, kind, i)
        reach[i] = ok
        why[i] = o
    print(f"reachable surface points: {reach.sum()} of {len(reach)} ({', '.join(f'{r} {((reg == r) & reach).sum()}/{(reg == r).sum()}' for r in ('face', 'edge', 'flank', 'tip'))})",
          flush=True)
    # 2. homing pose and the IK branch
    T_home = T_BI @ geo.T_IM_home @ C.inv(geo.T_EM0)
    if q_seed_deg is not None:
        q_s = chk.ik(T_home, np.radians(np.asarray(q_seed_deg, float)))
        if q_s is None or np.abs(q_s - np.radians(np.asarray(q_seed_deg, float))).max() > math.radians(5.0):
            raise SystemExit(f"the homing pose has no IK within 5 deg of --q-seed {q_seed_deg}")
        chk.q = q_s
        chk.seed_info = dict(branches=1, given_deg=list(q_seed_deg), q_deg=np.degrees(q_s).round(1).tolist())
    else:
        chk.seed(T_home)
    q_home = chk.q.copy()
    checks = dict(homing=chk.pose_ok(T_home)[1], seed=chk.seed_info)
    acts, dropped = [], []

    def add(a, poses, contact_P=None, need_extra=0.0, contact_M=None, need=CLEAR):
        """Append action a when every pose in poses reaches and clears (trying the twists when a['twist'] is None: those
        of a['twists_first'] first, then 0)."""
        first = tuple(a.pop("twists_first", ()))
        rest = (0.0,) if first else tuple(TWISTS)
        twists = (first + rest) if a.get("twist") is None else (a["twist"],)
        best = None
        for tw in twists:
            a2 = dict(a, twist=tw)
            q_save = chk.q.copy()
            ok_all, info = True, []
            for k, T in enumerate(poses(a2)):
                if k == 0:
                    chk.q = q_home.copy()          # the session's travel solves its joint moves nearest the homing branch
                ok, inf, q = chk.pose_ok(T, contact_P=contact_P, extra=need_extra, contact_M=contact_M, need=need)
                inf["pose"] = k
                info.append(inf)
                if not ok:
                    ok_all = False
                    break
                chk.q = q
            if ok_all:
                a2["check"] = info
                acts.append(a2)
                return True
            chk.q = q_save
            worst = min(info[-1].get("clear", {"none": -1e3}).values()) if info else -1e3
            if best is None or worst > best[0]:
                best = (worst, tw, info[-1])
        dropped.append(dict(id=a["id"], twist=best[1] if best else None, why=best[2] if best else None))
        return False

    def press_poses(a):
        tg = a["target"]
        st = a.get("standoff", 1.5)
        T_IP = geo.T_IP(tg["p"], tg["n"], a["twist"], st, a.get("offset", (0.0, 0.0)))
        T_rest = chk.T_BE(T_IP)
        d = -(T_BI[:3, :3] @ C.EZ)
        depth = st + a.get("max_depth_mm", a.get("beyond_mm", 2.5))
        T_deep = C.Tm(T_rest[:3, :3], T_rest[:3, 3] + d * min(depth, st + 1.0))   # 5 N takes about 0.7 mm on the face
        T_lift = lifted(geo, T_BI, T_rest, chk.T_EP, 60.0)
        return [T_lift, T_rest, T_deep]

    def tgt(i, region=None):
        return dict(frame="P", p=[round(float(v), 4) for v in geo.pts[i]], n=[round(float(v), 5) for v in geo.nrm[i]],
                    index=int(i), region=region or str(reg[i]))

    # 3. gravity sweep: the face centre 150 mm above the indenter's top; roll_offset_deg turns every roll, so that no
    # station sits half a turn from the homing pose, where the IK nearest the homing branch may take either side (the
    # lab's first run, 2026-10-22: grav-03 at roll 180 reached +166.6 deg of joint 6, 22.8 deg off its orientation, and
    # the next station turned joint 6 by 285 deg)
    st_I = (geo.c_I + (geo.R + 150.0) * C.EZ).tolist()
    grav = []
    for tilt in (0.0, 45.0, 90.0, -30.0):
        rolls = {0.0: (0, 60, 120, 180, 240, 300), 45.0: (0, 90, 180, 270), 90.0: (0, 90), -30.0: (0, 180)}[tilt]
        for roll in rolls:
            if roll_offset_deg:
                roll = (roll + roll_offset_deg + 180.0) % 360.0 - 180.0
            R = C.rot([0.0, 1.0, 0.0], tilt) @ C.rot([1.0, 0.0, 0.0], roll)
            w = C.log_so3(R)
            ang = float(np.degrees(np.linalg.norm(w)))
            ax = (w / np.linalg.norm(w)).tolist() if ang > 1e-9 else [0.0, 0.0, 1.0]
            grav.append((tilt, roll, ax, ang))

    def grav_poses(a):
        R = C.rot(a["axis"], a["deg"]) @ geo.R_IP([1.0, 0.0, 0.0], 0.0)
        T_IP = C.Tm(R, np.array(a["station_I"]) - R @ np.array(a["p"]))
        return [chk.T_BE(T_IP)]

    for k, (tilt, roll, ax, ang) in enumerate(grav):
        add(dict(id=f"grav-{k:02d}", stage="gravity", kind="orient", axis=ax, deg=ang, tilt_deg=tilt, roll_deg=roll,
                 station_I=st_I, p=p_ref.tolist(), twist=0.0), grav_poses, need_extra=5.0)
    chk.q = q_home.copy()
    # 4. mount face touches (rigid). mount-0 and mount-2 push on to 5 N: on the rigid column, the push past the onset is the
    # HEX-E's and the robot's deflection along x_M, the direction of the face presses; at two heights (z_M 52 and 68 mm)
    # it separates the HEX-E's shift from its tilt, which the ground truth takes from the data sheet otherwise (0.19 mm
    # of the 0.77 mm indentation of a 5 N face press)
    for k, (yy, zz) in enumerate(((-16.0, 52.0), (-16.0, 60.0), (-16.0, 68.0), (6.0, 62.0), (6.0, 68.0), (-6.0, 68.0))):
        a = dict(id=f"mount-{k}", stage="mount", kind="touch", target=dict(frame="M", p=[17.0, yy, zz], n=[1.0, 0.0, 0.0]),
                 twist=0.0, standoff=2.0, beyond_mm=1.0, v=0.0001, stop_N=5.0 if k in (0, 2) else 0.8, fit_N=0.6)
        if k in (0, 2):
            a["role"] = "compliance"

        def mposes(a):
            tg = a["target"]
            R = C.rot(C.EZ, a["twist"]) @ geo.R_ref
            c = geo.contact_point_I(a["standoff"])
            T_IM = C.Tm(R, c - R @ np.array(tg["p"]))
            T_rest = T_BI @ T_IM @ C.inv(geo.T_EM0)
            d = -(T_BI[:3, :3] @ C.EZ)
            return [C.Tm(T_rest[:3, :3], T_rest[:3, 3] - d * 10.0), T_rest,
                    C.Tm(T_rest[:3, :3], T_rest[:3, 3] + d * a["standoff"])]       # rigid: the touch stops at 0.8 N
        add(dict(a, twist=None), mposes, contact_M=a["target"]["p"], need=dict(CLEAR, mount=2.0))   # after homing: 0.3 mm known
    # 5. pad touches over the regions. At twist 0 the finger lies along x_I, and a shift of the sphere along x_I slides it
    # along the face and the flanks without changing any gap: the touches leave the sphere's place along the finger to the
    # homing's prior (0.25 mm). A flank's or an edge's touch at twist t sees that shift by its normal's y_P part times
    # sin t, so touch_twist_deg turns them by +-t, else +-2t, else 0. Not recommended: over four simulator truths on the
    # lab's branch (2026-10-22, homing spread 0.25 mm along the key) it took the presses' contact error from 0.30 to
    # 0.11 and from 0.32 to 0.02 mm, but from 0.38 to 0.53 and from 0.47 to 0.61 mm: the twisted touches stand at other
    # arm configurations, whose kinematic error (about 0.1 mm) they see at half the sensitivity (QUEUE.md R7)
    for r_, n_ in COUNTS["touch"].items():
        idx = np.nonzero((reg == r_) & reach)[0]
        for j, s in enumerate(fps(geo.pts[idx], n_, p_ref)):
            i = int(idx[s])
            a = dict(id=f"pad-{r_}-{j:02d}", stage="pad", kind="touch", target=tgt(i), twist=None, standoff=1.5,
                     beyond_mm=1.2, v=0.00015, stop_N=0.5, fit_N=0.4)
            if touch_twist_deg and r_ in ("flank", "edge"):
                t_, sg = float(touch_twist_deg), (1.0 if j % 2 == 0 else -1.0)
                a["twists_first"] = (sg * t_, -sg * t_, sg * 2 * t_, -sg * 2 * t_)
            add(a, press_poses, contact_P=geo.pts[i])
    # 6. pivot: the face centre, four 0.3 mm offsets, four turns
    t1 = np.array([0.0, 0.0, 1.0]) - n_ref[2] * n_ref
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n_ref, t1)
    plan["pivot"] = dict(ref_index=i_ref, t1=t1.tolist(), t2=t2.tolist(), level_N=3.0)
    piv = [("ref", (0.0, 0.0), 0.0)] + [("offset", o, 0.0) for o in ((0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5))]
    for k, (role, o, tw) in enumerate(piv):
        p = p_ref + o[0] * t1 + o[1] * t2
        add(dict(id=f"pivot-{k}", stage="pivot", kind="press", role=role, offset_P=list(o),
                 target=dict(frame="P", p=p.tolist(), n=n_ref.tolist(), index=i_ref, region="face"), twist=tw, standoff=1.5,
                 levels=[1.5, 3.0], stop_N=3.0, hold_s=1.5, max_depth_mm=2.5), press_poses, contact_P=p_ref)
    sgn = None                                    # one turning direction: the other swings the mount into the homing post
    for s_ in (1.0, -1.0):
        if all(chk.pose_ok(T, contact_P=p_ref)[0] for T in press_poses(dict(target=dict(p=p_ref.tolist(), n=n_ref.tolist()),
                                                                                  twist=60.0 * s_, standoff=1.5))[1:]):
            sgn = s_
            break
    for j, tw in enumerate((15.0, 30.0, 45.0, 60.0)):
        add(dict(id=f"pivot-{5 + j}", stage="pivot", kind="press", role="turn", offset_P=[0.0, 0.0],
                 target=dict(frame="P", p=p_ref.tolist(), n=n_ref.tolist(), index=i_ref, region="face"),
                 twist=tw * (sgn or 1.0), standoff=1.5, levels=[1.5, 3.0], stop_N=3.0, hold_s=1.5, max_depth_mm=2.5),
            press_poses, contact_P=p_ref)
    # 7. grid with the reference press every 15
    order = []
    for r_, n_ in COUNTS["grid"].items():
        idx = np.nonzero((reg == r_) & reach)[0]
        order += [(r_, int(idx[s])) for s in fps(geo.pts[idx], n_, p_ref)]
    # visit by region, then nearest neighbour, so the tool turns little between presses
    seq, last = [], p_ref
    for r_ in ("face", "edge", "flank", "tip"):
        rest = [o for o in order if o[0] == r_]
        while rest:
            j = int(np.argmin([np.linalg.norm(geo.pts[i] - last) + 3.0 * np.linalg.norm(geo.nrm[i] - geo.nrm[seq[-1][1]]
                                                                                           if seq else 0.0) for _, i in rest]))
            seq.append(rest.pop(j))
            last = geo.pts[seq[-1][1]]
    ref = dict(stage="grid", kind="press", role="ref", target=tgt(i_ref, "face"), twist=0.0, standoff=1.5,
               levels=[0.5, 1.0, 2.0], stop_N=2.0, max_depth_mm=2.5)
    n_ref_press = 0
    for k, (r_, i) in enumerate(seq):
        if k % 15 == 0:
            add(dict(ref, id=f"ref-{n_ref_press:02d}"), press_poses, contact_P=geo.pts[i_ref])
            n_ref_press += 1
        lv = LEVELS[r_]
        add(dict(id=f"grid-{k:03d}", stage="grid", kind="press", target=tgt(i), twist=None, standoff=1.5, levels=lv,
                 stop_N=lv[-1], max_depth_mm=2.5, hold_s=0.6), press_poses, contact_P=geo.pts[i])
    add(dict(ref, id=f"ref-{n_ref_press:02d}"), press_poses, contact_P=geo.pts[i_ref])
    # 8. shear (slide about the sphere's centre), roll, twist
    R_ind = geo.R
    s_deg = [round(math.degrees(s / R_ind), 3) for s in (0.1, 0.2, 0.3)]
    slip_deg = [round(math.degrees(s / R_ind), 3) for s in (0.5, 1.0)]
    face_idx = np.nonzero((reg == "face") & reach)[0]
    pick = lambda q: int(face_idx[np.argmin(np.linalg.norm(geo.pts[face_idx] - q, axis=1))])   # noqa: E731
    shear_pts = [i_ref, pick(FACE_CENTRE + [0, 4.0, 0]), pick(FACE_CENTRE + [0, -4.0, 0]), pick(FACE_CENTRE + [0, 0, 4.0])]
    for r_ in ("edge", "flank"):
        idx = np.nonzero((reg == r_) & reach)[0]
        if len(idx):
            shear_pts.append(int(idx[np.argmin(np.linalg.norm(geo.pts[idx] - (FACE_CENTRE + [-3.0, 6.0, 0.0]), axis=1))]))
    for k, i in enumerate(shear_pts):
        for F in ((1.0, 2.0) if k == 0 else (1.0,)):
            moves = [dict(dir_deg=float(dd), degs=s_deg) for dd in range(0, 360, 45)]
            moves += [dict(dir_deg=0.0, degs=slip_deg + [-slip_deg[-1]]), dict(dir_deg=90.0, degs=slip_deg + [-slip_deg[-1]])]
            add(dict(id=f"shear-{k}-{F:.0f}N", stage="shear", kind="slide", target=tgt(i), twist=None, standoff=1.5, press_N=F,
                     moves=moves, hold_s=0.4), press_poses, contact_P=geo.pts[i])
    for k, i in enumerate(shear_pts[:4]):
        add(dict(id=f"roll-{k}", stage="roll", kind="roll", target=tgt(i), twist=None, standoff=1.5, press_N=1.5,
                 moves=[dict(dir_deg=0.0, degs=[3.0, 6.0, -3.0, -6.0]), dict(dir_deg=90.0, degs=[3.0, 6.0, -3.0, -6.0])],
                 hold_s=0.4), press_poses, contact_P=geo.pts[i])
        for F in (1.0, 2.5):
            add(dict(id=f"twist-{k}-{F:.1f}N", stage="twist", kind="twist", target=tgt(i), twist=None, standoff=1.5, press_N=F,
                     moves=[dict(degs=[5.0, 10.0, 20.0, -5.0, -10.0, -20.0])], hold_s=0.5), press_poses, contact_P=geo.pts[i])
    # 9. final: the reference press and six gravity orientations
    add(dict(ref, id="final-ref", stage="final"), press_poses, contact_P=geo.pts[i_ref])
    for k, (tilt, roll, ax, ang) in enumerate([grav[j] for j in (0, 1, 3, 6, 8, 11)]):
        add(dict(id=f"fgrav-{k:02d}", stage="final_gravity", kind="orient", axis=ax, deg=ang, tilt_deg=tilt, roll_deg=roll,
                 station_I=st_I, p=p_ref.tolist(), twist=0.0), grav_poses, need_extra=5.0)
    # order by joint 6 (--order joint6), before the travel checks so that they cover each action's new neighbours
    def joint6(a):
        """Joint 6 (rad) at the action's rest pose, solved nearest the homing branch as the session's joint moves are,
        and wrapped to within half a turn of the homing pose's."""
        if a["stage"] in ("gravity", "final_gravity"):
            T = grav_poses(a)[0]
        elif a.get("target", {}).get("frame") == "P":
            T = press_poses(a)[1]
        else:
            return None
        q = chk.ik(T, q_home)
        if q is None:
            return None
        return float(q_home[5] + (q[5] - q_home[5] + math.pi) % (2 * math.pi) - math.pi)

    q6 = {a["id"]: joint6(a) for a in acts}
    for a in acts:
        if q6[a["id"]] is not None:
            a["joint6_deg"] = round(math.degrees(q6[a["id"]]), 1)

    def j6_travel(seq):
        v = [q6[a["id"]] for a in seq if q6[a["id"]] is not None]
        return round(float(np.degrees(np.abs(np.diff(v)).sum())), 1) if len(v) > 1 else 0.0

    order_info = {}
    new = []
    for stg in dict.fromkeys(a["stage"] for a in acts):          # the stages in their order; each one's actions adjoin
        ga = [a for a in acts if a["stage"] == stg]
        key = lambda a: q6[a["id"]] if q6[a["id"]] is not None else float(q_home[5])   # noqa: E731
        ga2 = ga
        if order_by == "joint6" and stg in ("gravity", "final_gravity"):
            srt = sorted(ga, key=key)
            ga2 = srt[0::2] + srt[1::2][::-1]
        elif order_by == "joint6" and stg in ("pad", "grid"):
            fixed = {k for k, a in enumerate(ga) if a.get("role") == "ref"}
            free = iter(sorted([a for a in ga if a.get("role") != "ref"], key=key))
            ga2 = [ga[k] if k in fixed else next(free) for k in range(len(ga))]
        order_info[stg] = dict(plan_deg=j6_travel(ga), ordered_deg=j6_travel(ga2))
        new += ga2
    acts[:] = new
    if order_by == "joint6":
        print("joint-6 travel by stage, deg (plan order -> by joint 6): " +
              ", ".join(f"{s} {v['plan_deg']:.0f} -> {v['ordered_deg']:.0f}" for s, v in order_info.items()), flush=True)
    # travel between consecutive contact actions: joint paths of the big turns; between gravity orientations: moveJ to
    # each station's IK nearest the homing branch, as the session solves them
    travel_bad = []
    for stg in ("gravity", "final_gravity"):
        ga = [a for a in acts if a["stage"] == stg]
        qs = []
        for a in ga:
            qs.append(chk.ik(grav_poses(a)[0], q_home))
        for a0, a1, q0, q1 in zip(ga[:-1], ga[1:], qs[:-1], qs[1:]):
            if q0 is not None and q1 is not None:
                ok, worst = chk.joint_path_ok(q0, q1, need=10.0)
                if not ok:
                    travel_bad.append((a0["id"], a1["id"], worst))
    prev = None
    for a in acts:
        if a["stage"] in ("gravity", "final_gravity"):
            prev = None
            continue
        if prev is not None:
            Tp = press_poses(prev)[1] if prev["target"]["frame"] == "P" else None
            Tn = press_poses(a)[1] if a["target"]["frame"] == "P" else None
            if Tp is not None and Tn is not None:
                mm, deg = C.pose_error(Tp, Tn)
                if deg > plan["travel"]["joint_deg"]:
                    q0 = chk.ik(lifted(geo, T_BI, Tp, chk.T_EP, 60.0), q_home)
                    q1 = chk.ik(lifted(geo, T_BI, Tn, chk.T_EP, 60.0), q0 if q0 is not None else q_home)
                    if q0 is None or q1 is None:
                        travel_bad.append((prev["id"], a["id"], "no IK"))
                    else:
                        ok, worst = chk.joint_path_ok(q0, q1)
                        if not ok:
                            travel_bad.append((prev["id"], a["id"], worst))
        prev = a if "target" in a else None
    plan["actions"] = acts
    plan["sim"] = dict(T_BI=T_BI, start=dict(p=p_ref.tolist(), n=n_ref.tolist(), standoff=80.0), q_seed=q_home.tolist(),
                       placement_mm_deg=list(placement))
    plan["checks"] = dict(placement_mm_deg=list(placement), block_mm=float(block_mm), board=dict(CELL.BOARD),
                          other=[dict(kind=k, T_BI=np.round(T, 4).tolist()) for T, k in CELL.OTHER],
                          mount_turn_deg=float(mount_turn_deg),
                          homing=checks["homing"], seed=checks["seed"],
                          reachable_points=int(reach.sum()),
                          reachable_by_region={r: int(((reg == r) & reach).sum()) for r in ("face", "edge", "flank", "tip")},
                          reachable_index=np.nonzero(reach)[0].tolist(),
                          points_by_region={r: int((reg == r).sum()) for r in ("face", "edge", "flank", "tip")},
                          dropped=dropped, travel_joint_paths_bad=travel_bad, clearances_needed_mm=CLEAR,
                          order=dict(mode=order_by, joint6_travel_deg=order_info), gravity_roll_offset_deg=roll_offset_deg,
                          touch_twist_deg=touch_twist_deg,
                          counts={s: sum(1 for a in acts if a["stage"] == s) for s in
                                  ("gravity", "mount", "pad", "pivot", "grid", "shear", "roll", "twist", "final", "final_gravity")},
                          seconds=round(time.time() - t_start, 1))
    out = Path(out or OUTDIR / f"plan/{time.strftime('%Y%m%d')}-plan_{kind}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, default=C.jsonable))
    print(f"{len(acts)} actions ({plan['checks']['counts']}), {len(dropped)} dropped, {len(travel_bad)} travel paths short of "
          f"clearance -> {out} ({out.stat().st_size / 1e6:.1f} MB, {time.time() - t_start:.0f} s)")
    for d in dropped[:12]:
        print("  dropped", d)
    return plan, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--indenter", default="sphere", choices=("sphere", "rod"))
    ap.add_argument("--placement", default="0,-450,56,-90", help="x,y,z (mm), yaw (deg) of the indenter's base in the "
                    "robot's base frame (z: above the table the robot stands on)")
    ap.add_argument("--block", type=float, default=38.0, help="mm of block under the indenter on the plywood")
    ap.add_argument("--block-half", default="50,30", help="half sizes of the block along x_I, y_I (mm)")
    ap.add_argument("--mount-turn", type=float, default=0.0, help="x_M's turn about the tool axis from tool0's x (deg)")
    ap.add_argument("--other", action="append", default=[], help="another indenter on the table as an obstacle: "
                    "kind,x,y,z,yaw (its base in the base frame, mm and deg); repeatable")
    ap.add_argument("--order", default="plan", choices=("plan", "joint6"), help="joint6: each stage's actions ordered by "
                    "joint 6, the gravity sweeps out and back (module docstring)")
    ap.add_argument("--gravity-roll-offset", type=float, default=0.0, help="deg added to every roll of the gravity "
                    "stations; 30 keeps every station at least 30 deg from half a turn off the homing pose")
    ap.add_argument("--q-seed", default=None, help="deg, comma separated: the robot's joints at its seating on the key "
                    "(lab/poses.jsonl); the checks then follow the IK branch the session uses")
    ap.add_argument("--touch-twist", type=float, default=0.0, help="deg: the flank and edge touch-offs turned by +-t, "
                    "else +-2t, else 0, so that the registration sees the sphere's place along the finger; not "
                    "recommended (step 5's comment: it helped in two simulator truths and hurt in two)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    for o in a.other:
        k, *v = o.split(",")
        x, y, z, yaw = (float(s) for s in v)
        CELL.OTHER.append((C.Tm(C.rot(C.EZ, yaw), [x, y, z]), k))
    build(a.indenter, tuple(float(v) for v in a.placement.split(",")), a.out, a.block, a.mount_turn,
          tuple(float(v) for v in a.block_half.split(",")), a.order, a.gravity_roll_offset,
          None if a.q_seed is None else [float(v) for v in a.q_seed.split(",")], a.touch_twist)


if __name__ == "__main__":
    main()
