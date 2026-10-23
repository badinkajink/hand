#!/usr/bin/env python3
"""Robot plan of the ATI Mini45's (or, with --parts, the Mini40's) calibration with a weighed mass on the lab's UR5 (CB3):
the stations, their order and
their checks, for scripts/ati_calibration_session.py (owner, 2026-10-23: "design a mount for the ATI for the HEX-E and
build up a calibration plan").

The ATI sits between the HEX-E and the hook cross (scripts/ati_stack_mount.py). A mass hangs from one eye of the cross
on a short line; it hangs free only while the eye's arm points downward (the line bends at most 75 deg at the eye), so
every eye gets stations of its own. A station is a joint vector that turns the ATI's frame about a fixed point in front
of the robot, P0, until gravity has the wanted direction in the ATI's frame. Every candidate is checked as the session
will meet it: reach by the CB3's kinematics on the branch of the lab's homing pose; joint 5 at least 30 deg from its
singularity and within 30..130 deg; joint 6 within 150 deg of the homing pose's (the cables); the tool (HEX-E and its
cable port, v3's adapter, the ATI base, the Mini45, the top, the cross) against the forearm and wrist links; the tool
and the links against the cell of the tip calibration (table, plywood, 2x4, both indenters, as the rod plan of
2026-10-23 placed them); and the line and the mass hanging from the eye in use against the tool, every link from the shoulder on and the cell.
From the clear candidates a greedy D-optimal choice takes eight stations per eye: each next station is the one that most
raises the log-determinant of the six wrench directions' information (torques over 0.1 m).

The session runs in blocks, one per eye: the reference station with no mass (drift), the eye's stations with no mass, the
loading station (the operator hangs the mass), the same stations with the mass, the loading station again (the operator
takes it off). Moves are joint moves between checked stations; every move's joint path is checked as well, with the
mass hanging along gravity at each step in the loaded sweeps.

    .venv/bin/python scripts/ati_calibration_plan.py [--p0 x,y,z] [--per-eye 8] [--out plan.json]
(through ~/.claude/bin/resguard.sh run --mem 3G: it loads the robot's and the tool's meshes)
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
D = ROOT / "docs/experiments/20261023-ati_ft_calibration"
PARTS = D / "20261023-ati_stack_parts.json"
CELL_PLAN = ROOT / "docs/experiments/20261022-ur5_tip_calibration/plan/20261023-plan_rod_lab_reg.json"
HOME_Q_DEG = (48.294, -86.25, 109.546, -24.819, 47.342, 6.43)    # lab/poses.jsonl "home_after_2x4_rotation" (10-22)
MOUNT_TURN = -92.15                                                 # x_M from tool0's x (lab, 2026-10-22)
G_LOCAL = 9.796                                                     # m/s2, Boulder (1655 m, 40.0 N): Helmert + free air
MASS = dict(nominal_kg=0.5, line_mm=45.0, r_mm=25.0, h_mm=80.0)     # pigtail and S hook; the mass's envelope
BEND_MIN = 0.26                                                     # g . u at least this: the line bends at most 75 deg
WRIST5 = (30.0, 130.0)                                              # deg
Q6_SPAN = 150.0                                                     # deg either side of the homing pose's joint 6
NEED = dict(self_tool=15.0, port=15.0, tool_cell=30.0, links_cell=30.0, mass_line=1.5, mass_tool=8.0, mass_links=20.0,
            mass_cell=50.0)                                         # mm; mass_line: the line from 6 mm past the eye
NEED_MOVE = dict(NEED, mass_line=-1e9, mass_cell=30.0)                 # during a move: the line may brush the eye's arm
SPEED = dict(v=0.3, a=0.3)                                          # rad/s, rad/s2 for every move (the swing)
TIME = dict(settle_s=4.0, rest_s=3.0, settle_max_s=15.0, operator_s=60.0)


def load_cell():
    """The cell as the rod plan of 2026-10-23 placed it: the rod's indenter frame (its plywood, the 2x4 under it, the
    table) and the sphere's as an obstacle."""
    p = json.loads(CELL_PLAN.read_text())
    ck = p["checks"]
    CELL.BOARD.update(ck.get("board") or {})
    CELL.OTHER[:] = [(np.array(o["T_BI"], float), o["kind"]) for o in ck.get("other", [])]
    x, y, z, yaw = ck["placement_mm_deg"]
    return C.Tm(C.rot(C.EZ, yaw), [x, y, z]), p["geometry"]["indenter"]["kind"]


def geometry():
    rec = json.loads(PARTS.read_text())
    T_EM = C.Tm(C.rot(C.EZ, MOUNT_TURN), [0.0, 0.0, CELL.HEX["height_mm"]])
    T_MA = np.array(rec["T_MA"], float)
    return rec, T_EM, T_MA


def tool_points_M(rec, spacing=2.5):
    """Surface samples (M, mm) of the tool: v3's adapter, the base, the Mini45, the top, the cross; and the HEX-E."""
    import trimesh
    zM = rec["z_M"]
    parts = []
    ad = trimesh.load(CELL.MOUNT / f"{CELL._MOUNT_STEM}_adapter.stl", force="mesh")
    parts.append(ad)
    for nm, z in (("ati_base", zM["adapter_top"]), ("ati_top", zM["ati_origin"]), ("hook_cross", zM["top_face"])):
        m = trimesh.load(ROOT / rec["parts"][nm]["file"], force="mesh")
        m.apply_translation([0.0, 0.0, z])
        parts.append(m)
    sd = rec.get("ati", dict(d_mm=45.0, h_mm=15.7))
    ati = trimesh.creation.cylinder(radius=sd["d_mm"] / 2, height=sd["h_mm"], sections=64)
    ati.apply_translation([0.0, 0.0, zM["ati_mount_face"] + sd["h_mm"] / 2])
    parts.append(ati)
    P = []
    for m in parts:
        p, _ = trimesh.sample.sample_surface_even(m, int(max(300, m.area / spacing ** 2)), seed=0)
        P.append(p)
    hx = CELL.hex_e_mesh(None)
    ph, _ = trimesh.sample.sample_surface_even(hx, 1500, seed=0)
    return np.vstack(P), ph


def hanging_points(p_eye, g, n_line=12):
    """The line from 6 mm past the eye along g, and the mass (a cylinder, axis along g) below it: two point sets in the
    eye's frame. At the steepest bend allowed (75 deg) the line passes the arm's lower edge 2.1 mm off."""
    L, r, h = MASS["line_mm"], MASS["r_mm"], MASS["h_mm"]
    g = np.asarray(g, float) / np.linalg.norm(g)
    line = p_eye + np.outer(np.linspace(6.0, L, n_line), g)
    a = np.cross(g, [1.0, 0.0, 0.0])
    a = a if np.linalg.norm(a) > 1e-3 else np.cross(g, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(g, a)
    th = np.linspace(0.0, 2 * math.pi, 16, endpoint=False)
    rim = np.outer(np.cos(th), a) + np.outer(np.sin(th), b)
    pts = []
    for s in np.linspace(0.0, h, 7):
        pts.append(p_eye + g * (L + s) + r * rim)
    for rr in (0.0, r / 2):
        pts.append(p_eye + g * (L + h) + rr * rim)
    return line, np.vstack(pts)


def arm_points(spacing=8.0):
    """Surface samples (link frame, mm) of the shoulder and the upper arm, which ur5_cb3_cell.robot_points leaves out."""
    import trimesh
    _, meshes = CELL.urdf_model()
    out = {}
    for ln in ("shoulder_link", "upper_arm_link"):
        T, f = meshes[ln]
        m = CELL._load(f)
        p, _ = trimesh.sample.sample_surface_even(m, int(max(200, m.area / spacing ** 2)), seed=0)
        out[ln] = C.apply(T, p)
    return out


class Checker:
    def __init__(self):
        from scipy.spatial import cKDTree
        self.rec, self.T_EM, self.T_MA = geometry()
        self.T_EA = self.T_EM @ self.T_MA
        self.T_BI, self.kind = load_cell()
        self.tool_M, self.hex_M = tool_points_M(self.rec)
        self.tool_tree = cKDTree(np.vstack([self.tool_M, self.hex_M]))
        self.eyes = self.rec["eyes"]
        self.robot = CELL.robot_points()
        self.arm = arm_points()                                   # for the hanging mass only: the shoulder sits on the table
        self.trees = {ln: cKDTree(p) for ln, p in list(self.robot.items()) + list(self.arm.items())}
        self.q_home = np.radians(HOME_Q_DEG)

    def cell_dist(self, P_B):
        P_I = C.apply(C.inv(self.T_BI), P_B)
        return float(min(CELL.cell_sdf(P_I, self.T_BI).min(), CELL.indenter_sdf(P_I, self.kind).min(),
                         CELL.other_sdf(P_B).min()))

    def links_dist(self, P_B, LF, links):
        return min(float(self.trees[ln].query(C.apply(C.inv(LF[ln]), P_B))[0].min()) for ln in links)

    def ok_q(self, q):
        """The joint windows of this plan, all on the branch of the lab's homing pose: the base within 75 deg of the
        homing pose's, the upper arm above the table, the elbow up and 15 deg from straight, wrist 1 in -120..170 deg
        (no wrap), joint 5 in WRIST5, joint 6 within Q6_SPAN of the homing pose's (the cables)."""
        d = np.degrees(q)
        if abs(d[0] - HOME_Q_DEG[0]) > 75.0:
            return "base turned too far"
        if not -180.0 <= d[1] <= 0.0 or not 15.0 <= d[2] <= 165.0:
            return "branch (upper arm or elbow)"
        if not -120.0 <= d[3] <= 170.0:
            return "wrist 1 outside -120..170 deg"
        if not WRIST5[0] <= d[4] <= WRIST5[1]:
            return f"joint 5 at {d[4]:.0f} deg"
        if abs(d[5] - HOME_Q_DEG[5]) > Q6_SPAN:
            return f"joint 6 at {d[5]:.0f} deg"
        return None

    def clear(self, q, eyes=()):
        """Clearances (mm) of the tool, the links and the masses hanging from the given eyes at joints q."""
        LF = CELL.link_frames(q)
        T_BE = LF["tool0"]
        T_BM = T_BE @ self.T_EM
        tool_B = C.apply(T_BM, self.tool_M[::2])
        hex_B = C.apply(T_BM, self.hex_M[::2])
        out = dict(self_tool=self.links_dist(tool_B, LF, CELL.SELF_LINKS),
                   port=self.links_dist(C.apply(T_BE, CELL.port_points()), LF, CELL.SELF_LINKS),
                   tool_cell=self.cell_dist(np.vstack([tool_B, hex_B])),
                   links_cell=min(self.cell_dist(C.apply(LF[ln], p[::3])) for ln, p in self.robot.items()))
        g_M = T_BM[:3, :3].T @ np.array([0.0, 0.0, -1.0])
        for e in eyes:
            line_M, body_M = hanging_points(np.array(e["p_M"]), g_M)
            out[f"mass_line_{e['name']}"] = float(self.tool_tree.query(line_M)[0].min())
            out[f"mass_tool_{e['name']}"] = float(self.tool_tree.query(body_M)[0].min())
            P_M = np.vstack([line_M, body_M])
            P_B = C.apply(T_BM, P_M)
            out[f"mass_links_{e['name']}"] = self.links_dist(P_B, LF, tuple(self.robot) + tuple(self.arm))
            out[f"mass_cell_{e['name']}"] = self.cell_dist(P_B)
        return out

    def passes(self, cl, need=NEED):
        for k, v in cl.items():
            key = k.split("_")[0] + "_" + k.split("_")[1] if k.startswith("mass_") else k
            if v < need[key]:
                return False
        return True

    def g_A(self, q):
        T_BA = SIM.fk(q) @ self.T_EA
        return T_BA[:3, :3].T @ np.array([0.0, 0.0, -1.0])


def fib_sphere(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = math.pi * (1 + 5 ** 0.5) * k
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def rot_to(a, b):
    """The least rotation taking unit a to unit b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:
        p = np.cross(a, [1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.cross(a, [0.0, 1.0, 0.0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    K = C.skew(v)
    return np.eye(3) + K + K @ K / (1.0 + c)


def wrench(g_A, eye, m=1.0):
    """The mass's wrench at the ATI's origin (N, N m) for gravity direction g_A (unit, ATI frame), with the line's
    departure from the eye's edge (radius of the hole) as in the analysis."""
    f = m * G_LOCAL * np.asarray(g_A, float)
    p = np.array(eye["p_A"], float) * 1e-3
    return np.r_[f, np.cross(p, f)]


def candidates(chk, P0, n_dir=160, yaws=np.arange(0.0, 360.0, 15.0)):
    """Clear (q, g_A, eyes) for gravity directions on a sphere and turns about the vertical, the ATI's origin at P0."""
    p_EA = chk.T_EA[:3, 3]
    out, why = [], {}
    for g in fib_sphere(n_dir):
        valid = [e for e in chk.eyes if g @ np.array(e["u_A"]) >= BEND_MIN]
        if not valid:
            continue
        R0 = rot_to(g, [0.0, 0.0, -1.0])
        best = None
        for yw in yaws:
            R_BA = C.rot(C.EZ, yw) @ R0
            R_BE = R_BA @ chk.T_EA[:3, :3].T
            T_BE = C.Tm(R_BE, P0 - R_BE @ p_EA)
            q = SIM.ik(T_BE, chk.q_home)
            if q is None:
                why["no IK"] = why.get("no IK", 0) + 1
                continue
            q = chk.q_home + (q - chk.q_home + math.pi) % (2 * math.pi) - math.pi
            r = chk.ok_q(q)
            if r:
                why[r.split(" at")[0]] = why.get(r.split(" at")[0], 0) + 1
                continue
            cl = chk.clear(q, valid)
            if not chk.passes({k: v for k, v in cl.items() if not k.startswith("mass_")}):
                why["tool or links"] = why.get("tool or links", 0) + 1
                continue
            ok_eyes = [e["name"] for e in valid
                       if all(cl[f"{s}_{e['name']}"] >= NEED[s] for s in ("mass_line", "mass_tool", "mass_links", "mass_cell"))]
            if not ok_eyes:
                why["hanging mass"] = why.get("hanging mass", 0) + 1
                continue
            cost = abs(math.degrees(q[5] - chk.q_home[5])) + 0.5 * float(np.degrees(np.abs(q - chk.q_home)).sum())
            if best is None or len(ok_eyes) > len(best[2]) or (len(ok_eyes) == len(best[2]) and cost < best[3]):
                best = (q, g, ok_eyes, cost, cl)
        if best is not None:
            q, g, ok_eyes, cost, cl = best
            out.append(dict(q=q, g_A=chk.g_A(q), eyes=ok_eyes, clear=cl))
    return out, why


def choose(cands, chk, per_eye, exclude=()):
    """Greedy D-optimal: (station, eye) pairs, at most per_eye per eye, each raising log det of the scaled information
    of the mass's wrenches the most."""
    S = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    eyes = {e["name"]: e for e in chk.eyes}
    pairs = [(i, n) for i, c in enumerate(cands) for n in c["eyes"] if (i, n) not in exclude]
    W = {(i, n): S @ wrench(cands[i]["g_A"], eyes[n], MASS["nominal_kg"]) for i, n in pairs}
    info = 1e-6 * np.eye(6)
    count = {n: 0 for n in eyes}
    chosen = []
    while len(chosen) < per_eye * len(eyes):
        best = None
        for p in pairs:
            if p in chosen or count[p[1]] >= per_eye:
                continue
            w = W[p]
            val = float(np.linalg.slogdet(info + np.outer(w, w))[1])
            if best is None or val > best[0]:
                best = (val, p)
        if best is None:
            break
        chosen.append(best[1])
        info += np.outer(W[best[1]], W[best[1]])
        count[best[1][1]] += 1
    sv = np.linalg.svd(np.array([W[p] for p in chosen]), compute_uv=False)
    return chosen, count, sv


def path_ok(chk, q0, q1, eye=None, n=10):
    """The joint path q0 -> q1 (moveJ) at n points: the tool and links clear, and with a mass on `eye` the line never
    bends back over the arm (g . u >= 0) and the mass clear of the tool, the links and the cell (NEED_MOVE)."""
    worst = {}
    for s in np.linspace(0.0, 1.0, n + 2)[1:-1]:
        q = q0 + (q1 - q0) * s
        cl = chk.clear(q, [eye] if eye else [])
        for k, v in cl.items():
            worst[k] = min(worst.get(k, 1e9), v)
        if eye is not None:
            ga = chk.g_A(q)
            worst["bend"] = min(worst.get("bend", 1.0), float(ga @ np.array(eye["u_A"])))
    ok = chk.passes({k: v for k, v in worst.items() if k != "bend"}, NEED_MOVE) and worst.get("bend", 1.0) >= 0.0
    return ok, {k: round(v, 2) for k, v in worst.items()}


def jdist(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def order_block(chk, eye, load, others, qs):
    """The order of a block's stations: a cycle from the loading station through the others and back, every move
    clear with the mass hanging (path_ok with the eye), least total joint travel. A move that is not clear may go by
    way of the loading station when both its halves are. Returns (the other stations in order, with "via" marks where a
    move goes by the loading station, or None; the station to drop when no cycle exists)."""
    from itertools import permutations
    nodes = [load] + list(others)
    ok = {}
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            ok[a, b] = ok[b, a] = path_ok(chk, qs[nodes[a]], qs[nodes[b]], eye)[0]
    def edge(a, b):
        if ok[a, b]:
            return jdist(qs[nodes[a]], qs[nodes[b]]), [b]
        if a and b and ok[a, 0] and ok[0, b]:
            return jdist(qs[nodes[a]], qs[nodes[0]]) + jdist(qs[nodes[0]], qs[nodes[b]]), ["via", b]
        return math.inf, [b]

    best = None
    for perm in permutations(range(1, len(nodes))):
        cost, seq, cur = 0.0, [], 0
        for b in list(perm) + [0]:
            c, e = edge(cur, b) if b else ((jdist(qs[nodes[cur]], qs[nodes[0]]) if ok[cur, 0] else math.inf), [0])
            cost += c + (5.0 if e[0] == "via" else 0.0)
            seq += e
            cur = b
            if best is not None and cost >= best[0]:
                break
        else:
            if cost < math.inf:
                best = (cost, seq)
    if best is None:                                  # no cycle: drop the station with the fewest clear moves
        deg = {k: sum(ok[k, j] for j in range(len(nodes)) if j != k) for k in range(1, len(nodes))}
        return None, [nodes[min(deg, key=deg.get)]]
    out = [nodes[k] if k != "via" else "via" for k in best[1][:-1]]      # the final return to load is the unload step
    return out, []


def move_s(q0, q1):
    d = float(np.abs(q1 - q0).max())
    v, a = SPEED["v"], SPEED["a"]
    return d / v + v / a if d > v * v / a else 2 * math.sqrt(d / a)


def payload(rec):
    """Mass and centre of gravity past the UR flange for set_payload (E frame, mm), without and with the mass: the HEX-E
    (260 g, data sheet), v3's adapter, M6 screws and inserts (v3's payload record), the printed parts at 0.70 g/cm3 (the
    draft plate's slicing: 142 g for 204 cm3), the sensor (the Mini45 92 g, the Mini40 50 g; to be weighed), 20 g of M3
    screws."""
    import trimesh
    zM = rec["z_M"]
    sd = rec.get("ati", dict(d_mm=45.0, h_mm=15.7, mass_g=92.0))
    items = [("hex_e", 260.0, -17.5), ("adapter", 24.9, 7.4), ("m6_screws_inserts", 29.2, 11.0),
             (f"ati_{rec.get('sensor', 'Mini45').lower()}", sd["mass_g"], zM["ati_mount_face"] + sd["h_mm"] / 2),
             ("m3_screws", 20.0, zM["ati_origin"])]
    for nm, z in (("ati_base", zM["adapter_top"]), ("ati_top", zM["ati_origin"]), ("hook_cross", zM["top_face"])):
        m = trimesh.load(ROOT / rec["parts"][nm]["file"], force="mesh")
        items.append((nm, float(m.volume) * 0.70e-3, z + float(m.center_mass[2])))
    m = sum(w for _, w, _ in items)
    z = sum(w * zz for _, w, zz in items) / m
    zc = zM["top_face"] + 50.0                                   # the mass, about the eyes' centre
    mm = MASS["nominal_kg"] * 1e3
    z2 = (m * z + mm * zc) / (m + mm)
    E = CELL.HEX["height_mm"]
    return dict(kg=round(m / 1e3, 4), cog_mm=[0.0, 0.0, round(z + E, 1)],
                with_mass=dict(kg=round((m + mm) / 1e3, 4), cog_mm=[0.0, 0.0, round(z2 + E, 1)]),
                items={nm: round(w, 1) for nm, w, _ in items})


def build(P0, per_eye, out):
    t0 = time.time()
    chk = Checker()
    cands, why = candidates(chk, np.asarray(P0, float))
    print(f"{len(cands)} clear gravity directions (rejections: {why}) [{time.time() - t0:.0f} s]", flush=True)
    eyes = {e["name"]: e for e in chk.eyes}
    qs = {i: c["q"] for i, c in enumerate(cands)}
    exclude, orders = set(), {}
    for attempt in range(6):
        chosen, count, sv = choose(cands, chk, per_eye, exclude)
        print(f"attempt {attempt}: chosen per eye {count}; scaled singular values {np.round(sv, 2)}", flush=True)
        orders, dropped = {}, []
        for n in ("+z", "+x", "-y", "-x", "+y"):
            idx = [i for i, m in chosen if m == n]
            if not idx:
                continue
            load = max(idx, key=lambda i: float(cands[i]["g_A"] @ np.array(eyes[n]["u_A"])))
            order, bad = order_block(chk, eyes[n], load, [i for i in idx if i != load], qs)
            if bad:
                dropped += [(i, n) for i in bad]
            else:
                orders[n] = (load, order)
        print(f"  {len(dropped)} stations with no clear move from their loading station: {dropped} "
              f"[{time.time() - t0:.0f} s]", flush=True)
        if not dropped:
            break
        exclude |= set(dropped)
    # the reference station: no mass, the tool pointing down as nearly as the candidates allow, near the loading stations
    ref = max(range(len(cands)), key=lambda i: cands[i]["g_A"][2] - 0.002 * jdist(cands[i]["q"], chk.q_home))
    used = sorted({i for i, _ in chosen} | {ref})
    sid = {i: f"st-{k:02d}" for k, i in enumerate(used)}
    stations = {}
    for i in used:
        c = cands[i]
        stations[i] = dict(id=sid[i], q_deg=np.round(np.degrees(c["q"]), 3).tolist(), g_A=np.round(c["g_A"], 5).tolist(),
                           eyes=c["eyes"], clear={k: round(v, 1) for k, v in c["clear"].items()})
    blocks, travel_bad, t_total, cur, cur_i = [], [], 0.0, chk.q_home, None

    def leg(q0, q1, i1, eye=None):
        """A move clear of the cell, or by way of the homing pose."""
        ok, worst = path_ok(chk, q0, q1, eye)
        if ok or eye is not None:
            return ok, worst, []
        ok1, w1 = path_ok(chk, q0, chk.q_home)
        ok2, w2 = path_ok(chk, chk.q_home, q1)
        return ok1 and ok2, w1 if not ok1 else w2, [dict(station="home", mass=None, role="via")]

    for n in ("+z", "+x", "-y", "-x", "+y"):
        if n not in orders:
            continue
        load, order = orders[n]
        visit = [i for i in order if i != "via"]
        steps = []
        for i1, mass, role, prompt in ([(ref, "off", "reference", None), (load, "off", None, None)]
                                        + [(i, "off", None, None) for i in order]
                                        + [(load, "on", "load", f"hang the mass on the {n} eye")]
                                        + [(i, "on", None, None) for i in order]
                                        + [(load, "on", "unload", f"take the mass off the {n} eye")]):
            if i1 == "via":                                   # by way of the loading station, no rest
                steps.append(dict(station=sid[load], mass=mass, role="via"))
                q1 = cands[load]["q"]
            else:
                q1 = cands[i1]["q"]
                loaded = mass == "on" and role != "load"
                ok, worst, via = leg(cur, q1, i1, eyes[n] if loaded else None)
                if not ok:
                    travel_bad.append(dict(eye=n, to=sid[i1], mass=mass, worst=worst))
                steps += via
                st = dict(station=sid[i1], mass=mass)
                if role:
                    st["role"] = role
                if prompt:
                    st["prompt"] = prompt
                steps.append(st)
                t_total += TIME["settle_s"] + TIME["rest_s"] + (TIME["operator_s"] if prompt else 0.0)
            t_total += move_s(cur, q1)
            cur = q1
        blocks.append(dict(eye=n, stations=[sid[load]] + [sid[i] for i in visit], steps=steps))
    ok, worst, via = leg(cur, cands[ref]["q"], ref)
    blocks.append(dict(eye=None, steps=via + [dict(station=sid[ref], mass="off", role="reference")]))
    t_total += move_s(cur, cands[ref]["q"]) + TIME["rest_s"]
    end_home, _ = path_ok(chk, cands[ref]["q"], chk.q_home)
    t_total += move_s(cands[ref]["q"], chk.q_home) if end_home else 0.0
    rec, T_EM, T_MA = chk.rec, chk.T_EM, chk.T_MA
    plan = dict(version=1, kind="ati_mass_calibration", made=time.strftime("%Y-%m-%d %H:%M"), sensor=rec.get("sensor", "Mini45"),
                note="stations for scripts/ati_calibration_session.py; built by scripts/ati_calibration_plan.py",
                geometry=dict(T_EM=T_EM, T_MA=T_MA, T_EA=chk.T_EA, eyes=chk.eyes, eye=rec["eye"], z_M=rec["z_M"],
                              mount_turn_deg=MOUNT_TURN, parts=str(PARTS.relative_to(ROOT)), clock_deg=rec["clock_deg"]),
                mass=MASS, g_local=G_LOCAL, p0_mm=list(P0), home_q_deg=list(HOME_Q_DEG), speed=SPEED, time=TIME,
                payload=payload(rec), end_home=bool(end_home),
                stations=[stations[i] for i in used], blocks=blocks,
                checks=dict(need_mm=NEED, wrist5_deg=WRIST5, q6_span_deg=Q6_SPAN, bend_min=BEND_MIN,
                            cell=str(CELL_PLAN.relative_to(ROOT)), rejections=why, travel_bad=travel_bad,
                            scaled_singular_values=np.round(sv, 3).tolist(), per_eye=count, n_candidates=len(cands),
                            excluded=[[sid.get(i, i), n] for i, n in sorted(exclude, key=str)]),
                minutes=round(t_total / 60.0, 1), seconds_to_build=round(time.time() - t0, 1))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=1, default=C.jsonable) + "\n")
    print(f"{len(used)} stations, {sum(len(b['steps']) for b in blocks)} steps, about {plan['minutes']} min; "
          f"{len(travel_bad)} moves short of clearance -> {out} [{time.time() - t0:.0f} s]")
    for t in travel_bad[:10]:
        print("  ", t)
    return plan


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--p0", default="-180,-420,430", help="mm in the base frame: where the ATI's origin stays")
    ap.add_argument("--per-eye", type=int, default=8)
    ap.add_argument("--out", default=str(D / "plan/20261023-ati_plan_mass.json"))
    ap.add_argument("--parts", default=str(PARTS), help="the stack's parts record (scripts/ati_stack_mount.py --sensor)")
    a = ap.parse_args()
    PARTS = Path(a.parts).resolve()
    build([float(v) for v in a.p0.split(",")], a.per_eye, a.out)
