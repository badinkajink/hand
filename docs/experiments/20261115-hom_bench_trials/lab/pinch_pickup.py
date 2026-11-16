"""Pick up the sphere lying on its side with a two-finger pinch and let it pivot upright between the pads (owner,
2026-11-16 15:2x: "carefully try to pick up the sphere and run a controlled slip to have it rotate to more level, and
try placing it? would be a very interesting skill to build. controlled friction is the name of the game after all";
"this grasp actually will be a very clean test of the magnetic field sensing fingertips and the contact model").

The pinch: the thumb (x_P < 0) against the middle or the index (x_P > 0), their pads' normals along x_P, the contacts
--lat-deg above the ball's equator. The fingers' flexion planes all contain x_P (yaw turns a finger about x_P), so a
pinch along x_P needs no wrist turn when the object's axis lies across x_P. The object can then turn only about the
pinch axis, the line through the two contacts: lifted slowly, its block drags on the table and the object stands up
between the pads, its tilt set by the lift (the controlled slip); lifted clear, it hangs about atan(e / 15 mm) from
vertical, e being the pinch axis' height above the ball's centre at the grasp (r sin lat; the centre of mass lies about
15 mm below the ball's centre). The third finger is curled out of the way on the side away from the block.

    plan     the object measured (lying: tilt over 60 deg, its axis within --max-axis-deg of across x_P); the pinch's
             centre in P and the poses solved on the hand's model (pads as the v4b half-ellipsoids): open (pads --gap
             mm out), touch, grip (the PIPs --squeeze-deg past touch), the third finger's pose; every finger part's
             clearance from the table, the ball, the stem and the block; the plan file for putdown and letgo
             (plans/<date>-pinch_pickup_<pair>_plan.json). No motion
    grasp    the fingers to the open pose (the hand high), the lateral move that puts the ball under the pinch's
             centre, the descent in two legs (fast, then the last --slow-mm at --v-slow), a fine correction, and the
             pinch closed in small steps; arm moves stop on the HEX-E's change (--stop-n), the tracker (the object
             moving before the close) and the fingers' joints (a joint pushed --deflect-deg off its reading before the
             leg: a finger on the table or the ball)
    lift     the arm up along base z by --lift-mm at --v-lift (default: the ball's centre to the object's standing
             height + --lift-extra), stopped by the HEX-E or the ball slipping down the pads (--slip-mm along z_P);
             the object's tilt, its centre in P, the joints, the servos' loads and the HEX-E logged at the telemetry rate
    slip     the controlled slip: the PIPs' squeeze lowered in --step-deg steps while the object hangs more than
             --tilt-goal deg from vertical; stops at the goal, when the ball slides down the pads (--slip-mm), or at
             --min-squeeze
    status   the object, the joints and the loads, the HEX-E

Then hom_setup.py putdown --plan <the plan file> (tracked touchdown, seat check, release check, release, straighten,
lift). Logs: logs/20261116-hom_setup/<date>_pinch_pickup.jsonl (steps) and logs/20261116-hom_trials/<stamp>_pinch_<step>
_telemetry.jsonl (streams). Every command takes --video-dir (both cameras).

    /usr/bin/python3 pinch_pickup.py plan --object sphere --map ../maps/20261116-sphere_fitted_map_O.json
    /usr/bin/python3 pinch_pickup.py grasp --plan PLAN --object sphere --map MAP --go --video-dir DIR
"""
import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hom_setup as HS  # noqa: E402
from hom_setup import quat_to_mat, tf, tf_inv, T_EP  # noqa: E402
from sr2hand.robot.tool_chain import ToolChain  # noqa: E402

H = HS.HAND
ELL = (8.5, 7.4, 11.0, 22.0)        # the v4b pad, surfaces.HalfEllipsoid(ax depth, ay across, az distal, azp proximal), mm
REACH = 10.55                       # the finger's axis to the pad's apex (sr2_hand.json reach_mm)
R_TAPED = 20.75                     # mm: the 40 mm sphere under ~0.75 mm of tape (owner, 2026-11-16)
COM_MM = 15.1                       # mm the object's centre of mass lies below the ball's centre along its axis
# simulator-degree ranges: the servos' calibrated ranges (manta_hand.servos.FINGER_JOINTS) over JOINT_SIGN
LIM = {"thumb": {"yaw": (-85.0, 85.0), "mcp": (-12.3, 89.06), "pip": (-18.0, 86.72)},
       "index": {"yaw": (-85.0, 85.0), "mcp": (-15.0, 64.75), "pip": (-2.93, 92.0)},
       "middle": {"yaw": (-85.0, 85.0), "mcp": (-15.0, 69.73), "pip": (-16.11, 77.05)}}
MCP_MAX = 50.0                      # deg: well short of the pots' dead zone past ~62 deg (sr2-mcp-pot-dead-zone)
J = ("yaw", "mcp", "pip")
LOG = HS.LOG / f"{time.strftime('%Y%m%d')}_pinch_pickup.jsonl"
STREAMS = HS.VIDEO_DIR


def note(rec):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}, default=float) + "\n")


# ------------------------------------------------------------------------------------------------ the fingertip boards

MUX_DIR = HS.ROOT / "logs/20261116-three_finger_mux"
BOARD = {"middle": 0, "index": 1, "thumb": 2}       # boards in the splitter's log: mux channels 0, 1, 3 (ch3 the thumb)
NB = 3
REC = 8 + 80 * NB                                   # reskin5x_mux.py's log: host time f64, then 5 x (t x y z) f32 a board
REST = HS.LOG / "pinch_pads_rest.json"


class Pads:
    """The splitter's newest log (reskin5x_mux.py, started apart), tailed: each board's 15 field components and its
    change from a rest (|B - B_rest| over the 15 components, the mean of the last --pad-avg records)."""

    def __init__(self, avg=8):
        logs = sorted(MUX_DIR.glob("*-mux.bin"), key=lambda q: q.stat().st_mtime)
        if not logs or time.time() - logs[-1].stat().st_mtime > 5.0:
            raise SystemExit("no fingertip board log written in the last 5 s: start reskin5x_mux.py")
        self.path, self.avg = logs[-1], avg
        self.f = open(self.path, "rb")
        size = self.path.stat().st_size
        self.f.seek(max(0, size - size % REC - REC * 200))
        self.buf, self.last, self.rest, self.lock = b"", None, None, threading.Lock()
        self.hist = []
        self.read()

    def read(self):
        """Read what is new; returns the newest mean field (NB, 15) or None."""
        with self.lock:
            self.buf += self.f.read()
            n = len(self.buf) // REC
            if n:
                raw = np.frombuffer(self.buf[:n * REC], dtype=np.uint8).reshape(n, REC)
                self.buf = self.buf[n * REC:]
                t = raw[:, :8].copy().view(np.float64)[:, 0]
                v = raw[:, 8:].copy().view(np.float32).reshape(n, NB, 5, 4)[:, :, :, 1:].reshape(n, NB, 15)
                self.hist.extend(zip(t, v))
                self.hist = self.hist[-200:]
            if len(self.hist) < self.avg:
                return None
            self.last = np.mean([h[1] for h in self.hist[-self.avg:]], axis=0)
            return self.last

    def set_rest(self, seconds=0.5):
        time.sleep(seconds)
        self.read()
        with self.lock:
            k = max(self.avg, int(160 * seconds))
            self.rest = np.mean([h[1] for h in self.hist[-k:]], axis=0)
        REST.write_text(json.dumps({"t": time.time(), "log": str(self.path), "rest": self.rest.tolist()}))
        return self.rest

    def load_rest(self):
        r = json.loads(REST.read_text())
        self.rest = np.asarray(r["rest"], float)
        return time.time() - r["t"]

    def change(self):
        """{finger: |B - B_rest| uT} (the field change), or None before a rest."""
        b = self.read()
        if b is None or self.rest is None:
            return None
        return {f: float(np.linalg.norm(b[k] - self.rest[k])) for f, k in BOARD.items()}


def pads_text(d):
    return "pads " + ("(none)" if d is None else " ".join(f"{f} {d[f]:.0f}" for f in ("thumb", "index", "middle"))) + " uT"


# ------------------------------------------------------------------------------------------------ the hand's model

def mounts(d6):
    return {"thumb": d6[0:2], "index": d6[2:4], "middle": d6[4:6]}


def frames(f, m, q):
    """(T_tip, T_pip) in P of finger f on mount m (P mm) at joints q (simulator deg, yaw mcp pip)."""
    c, _ = H.box(f)
    ch = ToolChain(f, H.finger_links(f, m[0] - c[0], m[1] - c[1], *(math.radians(v) for v in q)))
    return ch.transform(upto=f"{f}_tip"), ch.transform(upto=f"{f}_pip_frame")


def pad_point(f, m, q, n_world):
    """(point in P, on the palmar side, z in the tip frame): the pad's surface point whose outward normal is n_world."""
    T, _ = frames(f, m, q)
    n = T[:3, :3].T @ np.asarray(n_world, float)
    s = H.fingers[f].face_sign
    a = np.array([ELL[0], ELL[1], ELL[2] if n[2] < 0 else ELL[3]])
    x = a ** 2 * n / math.sqrt(float(np.sum(a ** 2 * n ** 2)))
    loc = np.array([s * (REACH - ELL[0]), 0.0, 0.0]) + x
    return T[:3, :3] @ loc + T[:3, 3], float(s * x[0]) > 0.0, float(x[2])


def pad_gap(f, m, q, c, r):
    """mm between the pad (the half-ellipsoid, both halves) and a ball (c, r) in P: the pad's support point toward c."""
    T, _ = frames(f, m, q)
    u = np.asarray(c, float) - T[:3, 3]
    u /= np.linalg.norm(u)
    p, _, _ = pad_point(f, m, q, u)
    return float(np.linalg.norm(np.asarray(c) - p) - r) if (np.asarray(c) - p) @ u > 0 else -float(r)


def parts(f, m, q):
    """[(name, point in P, radius)]: the tip end (11 mm past the tip, r 6; conservative), the PIP joint (r 8), the
    pad's lowest point (r 0)."""
    T, Tp = frames(f, m, q)
    lo, _, _ = pad_point(f, m, q, [0.0, 0.0, -1.0])
    return [("tip end", (T @ np.array([0.0, 0.0, -11.0, 1.0]))[:3], 6.0), ("PIP", Tp[:3, 3], 8.0), ("pad", lo, 0.0)]


def lowest(f, m, q):
    return min((p[2] - rad, name) for name, p, rad in parts(f, m, q))


def solve(f, m, c, u, r, q0, keep=0.0):
    """Joints that put the pad's point of outward normal -u at c + r u (u: the contact's direction from the ball's
    centre), keep deg inside the servo's range; (q, residual mm, the contact's z in the tip frame)."""
    target = np.asarray(c, float) + r * np.asarray(u, float)
    lo = [LIM[f][k][0] + keep for k in J]
    hi = [(min(LIM[f][k][1], MCP_MAX) if k == "mcp" else LIM[f][k][1]) - keep for k in J]

    def res(q):
        p, ok, _ = pad_point(f, m, q, -np.asarray(u))
        return (p - target) + (0.0 if ok else 50.0)
    sol = least_squares(res, np.clip(q0, lo, hi), bounds=(lo, hi))
    return sol.x, float(np.linalg.norm(sol.fun)), pad_point(f, m, sol.x, -np.asarray(u))[2]


def margin(f, q):
    """deg from the nearest servo limit (MCP: from MCP_MAX on the flexion side)."""
    out = math.inf
    for k, v in zip(J, q):
        lo, hi = LIM[f][k]
        hi = min(hi, MCP_MAX) if k == "mcp" else hi
        out = min(out, v - lo, hi - v)
    return out


def object_caps(c, axis_dir, h):
    """The object (P mm) as capsules [(name, a, b, radius)]: ball, stem and block along axis_dir (the object's up, y_O)
    from the ball's centre c; h the centre's height above the block's bottom (hom_setup.map_height)."""
    up = np.asarray(axis_dir, float) / np.linalg.norm(axis_dir)
    c = np.asarray(c, float)
    return [("ball", c, c, R_TAPED), ("stem", c - R_TAPED * up, c - (h - HS.BLOCK_H) * up, HS.STEM_R),
            ("block", c - (h - HS.BLOCK_H) * up, c - h * up, HS.BLOCK_R)]


def clearances(m, pose, c, axis_dir, h, table_z, skip_pad_ball=()):
    """{finger: (least gap to the table, to the stem or block, to the ball (pad ellipsoid for the pinch fingers, the
    parts' balls for the others))} in mm, the hand in the grasp position (the ball's centre at c in P, the table at
    table_z in P)."""
    caps = object_caps(c, axis_dir, h)
    out = {}
    for f, q in pose.items():
        qq = [q[k] for k in J]
        ps = parts(f, m[f], qq)
        tab = min(p[2] - rad - table_z for _, p, rad in ps)
        sb = min(HS.seg_dist(p, a_, b_) - cr - rad for _, p, rad in ps for name, a_, b_, cr in caps if name != "ball")
        if f in skip_pad_ball:
            ball = pad_gap(f, m[f], qq, c, R_TAPED)
        else:
            T, _ = frames(f, m[f], qq)
            ball = min([pad_gap(f, m[f], qq, c, R_TAPED)]
                       + [float(np.linalg.norm(p - c)) - R_TAPED - rad for _, p, rad in ps if rad > 0])
        out[f] = (tab, sb, ball)
    return out


def descent_clear(m, pose, c):
    """{finger: mm}: the least gap between each finger's parts and the ball while the hand comes straight down to the
    grasp's height with this pose: a part ending below the ball's centre passes the equator (its horizontal distance
    from the ball's axis counts), one ending above it is nearest at the end; the pad as its ellipsoid."""
    c = np.asarray(c, float)
    out = {}
    for f, q in pose.items():
        qq = [q[k] for k in J]
        gaps = [pad_gap(f, m[f], qq, c, R_TAPED)]
        T, _ = frames(f, m[f], qq)
        cz = c.copy()
        cz[2] = T[2, 3]                                 # the ball's centre at the tip frame's height: the pad's widest pass
        if T[2, 3] < c[2]:
            gaps.append(pad_gap(f, m[f], qq, cz, R_TAPED))
        for name, p, rad in parts(f, m[f], qq):
            if rad == 0.0:
                continue
            if p[2] < c[2]:
                gaps.append(float(np.hypot(*(p[:2] - c[:2]))) - R_TAPED - rad)
            else:
                gaps.append(float(np.linalg.norm(p - c)) - R_TAPED - rad)
        out[f] = min(gaps)
    return out


def finger_pair(stem_P):
    """(pinch partner, third finger): the partner on the block's side of y_P, so the third finger stays away from it."""
    return ("middle", "index") if stem_P[1] < 0 else ("index", "middle")


def third_pose(f, m, c, table_z, axis_dir, h):
    """The third finger curled out of the way: the pose with the largest least clearance (table - 12, ball, block)."""
    best = None
    sgn = 1.0 if m[1] > 0 else -1.0                      # yaw away from the ball's side (+y for the index)
    for yaw in (0.0, 10.0, 20.0, 30.0):
        for mcp in (0.0, 15.0, 30.0, 45.0):
            for pip in (0.0, 20.0, 40.0, 60.0, 80.0):
                q = {"yaw": sgn * yaw, "mcp": mcp, "pip": pip}
                cl = clearances({f: m}, {f: q}, c, axis_dir, h, table_z)[f]
                score = min(cl[0] - 12.0, cl[1], cl[2]) - 0.02 * (yaw + mcp + pip)
                if best is None or score > best[0]:
                    best = (score, q, cl)
    return best[1], best[2]


def plan_pinch(d6, p_ball, stem_P, ball_above_table, h, lat_deg, gap, squeeze, grid=5.0):
    """Search the pinch's centre c (P mm) near the partner's side for the largest table clearance with joint margins;
    returns the chosen solution dict."""
    m = mounts(d6)
    partner, third = finger_pair(stem_P)
    L = math.radians(lat_deg)
    ut, up_ = np.array([-math.cos(L), 0.0, math.sin(L)]), np.array([math.cos(L), 0.0, math.sin(L)])
    y_mid = 0.5 * (m["thumb"][1] + m[partner][1])
    axis_dir = -np.asarray(stem_P, float)
    sols = []
    for xc in np.arange(-10.0, 15.1, grid):
        for yc in np.arange(y_mid - 10.0, y_mid + 10.1, grid):
            for zc in np.arange(-90.0, -64.9, grid):
                c = np.array([xc, yc, zc])
                qt, et, zt = solve("thumb", m["thumb"], c, ut, R_TAPED, [0.0, 0.0, 50.0])
                qp, ep, zp = solve(partner, m[partner], c, up_, R_TAPED, [15.0, 0.0, 50.0])
                if et > 0.3 or ep > 0.3:
                    continue
                qto, eto, _ = solve("thumb", m["thumb"], c, ut, R_TAPED + gap, qt, keep=3.0)
                qpo, epo, _ = solve(partner, m[partner], c, up_, R_TAPED + gap, qp, keep=3.0)
                if eto > 0.3 or epo > 0.3:
                    continue
                table_z = zc - ball_above_table
                pose = {"thumb": dict(zip(J, qt)), partner: dict(zip(J, qp))}
                cl = clearances(m, pose, c, axis_dir, h, table_z, skip_pad_ball=("thumb", partner))
                po = {"thumb": dict(zip(J, qto)), partner: dict(zip(J, qpo))}
                clo = clearances(m, po, c, axis_dir, h, table_z, skip_pad_ball=("thumb", partner))
                mg = min(margin("thumb", qt), margin(partner, qp), margin("thumb", qto), margin(partner, qpo),
                         margin("thumb", np.add(qt, [0, 0, squeeze])), margin(partner, np.add(qp, [0, 0, squeeze])))
                tab = min(cl["thumb"][0], cl[partner][0], clo["thumb"][0], clo[partner][0])
                end = min(zt, zp)                         # the contact's z in the tip frame (the pad's end at -11)
                dc = descent_clear(m, po, c)
                dmin = min(dc.values())
                score = (tab + min(0.0, mg - 4.0) * 2.0 + min(0.0, end + 8.5) * 3.0 + min(0.0, dmin - 3.0) * 3.0)
                sols.append({"score": score, "c": c, "q_touch": pose, "q_open": po, "table_mm": tab, "margin_deg": mg,
                             "contact_z_tip": (zt, zp), "clear_touch": cl, "clear_open": clo, "table_z": table_z,
                             "descent_mm": dc})
    if not sols:
        raise SystemExit("no pinch found on the model")
    best = max(sols, key=lambda s: s["score"])
    q3, cl3 = third_pose(third, m[third], best["c"], best["table_z"], axis_dir, h)
    best.update(partner=partner, third=third, q_third=q3, clear_third=cl3, axis_dir=axis_dir, lat_deg=lat_deg,
                gap=gap, squeeze=squeeze)
    return best


def pose_full(sol, which, squeeze=0.0):
    """{finger: {yaw, mcp, pip}} for the station: the pinch fingers' 'open' or 'touch' joints (the PIPs + squeeze), the
    third finger's pose."""
    src = sol["q_open"] if which == "open" else sol["q_touch"]
    out = {f: {k: float(v) for k, v in q.items()} for f, q in src.items()}
    for f in out:
        out[f]["pip"] += squeeze
    out[sol["third"]] = {k: float(v) for k, v in sol["q_third"].items()}
    return out


def write_plan(sol, path, base_plan, h):
    """The plan file putdown and letgo read: the mounts, the poses (open first), meta.hom's object and nominal pose
    (the ball's centre at the pinch's centre, upright)."""
    c = sol["c"]
    palm_m = (np.array([0.0, 0.0, HS.Z_OBJ]) - c / 1e3).tolist()
    plan = {"design": f"pinch_pickup_thumb_{sol['partner']}", "mounts_palm_mm": base_plan["mounts_palm_mm"],
            "poses": [{"name": "open", "joints": pose_full(sol, "open")},
                      {"name": "touch", "joints": pose_full(sol, "touch")},
                      {"name": "grip", "joints": pose_full(sol, "touch", sol["squeeze"])}],
            "meta": {"source": f"pinch_pickup.py plan ({time.strftime('%Y-%m-%d %H:%M')}): thumb-{sol['partner']} pinch "
                               f"along x_P, contacts {sol['lat_deg']:g} deg above the equator, pads {sol['gap']:g} mm "
                               f"out when open, PIPs +{sol['squeeze']:g} deg in the grip; {sol['third']} curled away",
                     "written": time.strftime("%Y-%m-%d %H:%M"),
                     "hom": {"shape": "sphere", "d6_P_mm": base_plan["meta"]["hom"]["d6_P_mm"],
                             "setup": {"palm_m": palm_m, "quat": [0.70710678, 0.70710678, 0.0, 0.0],
                                       "r_m": R_TAPED / 1e3, "hl_m": 0.0},
                             "pinch": {"c_P_mm": c.tolist(), "partner": sol["partner"], "third": sol["third"],
                                       "lat_deg": sol["lat_deg"], "gap_mm": sol["gap"], "squeeze_deg": sol["squeeze"],
                                       "table_mm": sol["table_mm"], "margin_deg": sol["margin_deg"],
                                       "descent_mm": sol["descent_mm"],
                                       "h_mm": h}}}}
    Path(path).write_text(json.dumps(plan, indent=1) + "\n")
    return plan


# ------------------------------------------------------------------------------------------------ live helpers

class Stream:
    """The station's joints and servo loads, the tracked object and the HEX-E at the telemetry rate, into a JSON-lines
    file, until stop()."""

    def __init__(self, st, meas, arm, path, period=0.05, pads=None):
        self.st, self.meas, self.arm, self.path, self.period = st, meas, arm, Path(path), period
        self.pads = pads
        self.run, self.n = True, 0
        self.marks = []
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def mark(self, text):
        self.marks.append((time.time(), text))

    def _loop(self):
        with open(self.path, "a") as fo:
            while self.run:
                t0 = time.time()
                rec = {"t": t0}
                try:
                    s = self.st.state()
                    tel = s.get("telemetry", {})
                    rec["joints"] = HS.sim_joints(s)
                    rec["load"] = tel.get("servo_load")
                    rec["torque"] = s.get("servo_torque")
                except (OSError, SystemExit) as e:
                    rec["station_error"] = str(e)[:120]
                with self.meas.lock:
                    smp = self.meas.samples[-1] if self.meas.samples else None
                if smp is not None and t0 - smp[0] < 0.3:
                    R = quat_to_mat(smp[5])
                    rec["obj"] = {"age": round(t0 - smp[0], 3), "p_P": np.round(smp[1], 2).tolist(),
                                  "p_B": np.round(smp[4], 2).tolist(), "q_B": np.round(smp[5], 5).tolist(),
                                  "tilt": round(math.degrees(math.acos(float(np.clip(R[2, 1], -1, 1)))), 2)}
                if self.arm is not None:
                    try:
                        rec["palm_B"] = np.round(self.arm.T_BP()[:3, 3], 2).tolist()
                        if self.arm.ft is not None or self.arm.ur_ft:
                            _, F = self.arm.force()
                            rec["F"] = None if F is None else np.round(F, 3).tolist()
                    except Exception as e:                  # noqa: BLE001 (a log line, never a stop)
                        rec["arm_error"] = str(e)[:120]
                if self.pads is not None:
                    try:
                        b = self.pads.read()
                        if b is not None:
                            rec["pads_uT"] = {f: np.round(b[k], 1).tolist() for f, k in BOARD.items()}
                            d = self.pads.change()
                            if d is not None:
                                rec["pads_dB"] = {f: round(v, 1) for f, v in d.items()}
                    except Exception as e:                  # noqa: BLE001
                        rec["pads_error"] = str(e)[:120]
                while self.marks:
                    rec.setdefault("mark", []).append(self.marks.pop(0)[1])
                fo.write(json.dumps(rec, default=float) + "\n")
                self.n += 1
                time.sleep(max(0.0, self.period - (time.time() - t0)))

    def stop(self):
        self.run = False
        self.th.join(timeout=2.0)
        print(f"   telemetry: {self.n} records -> {self.path}", flush=True)


def obj_now(meas, n=5, max_age=0.5):
    """(p_P, p_B, tilt deg) of the newest tracked samples, or None."""
    with meas.lock:
        last = meas.samples[-n:]
    if len(last) < n or time.time() - last[0][0] > max_age:
        return None
    R = quat_to_mat(last[-1][5])
    return (np.median([g[1] for g in last], axis=0), np.median([g[4] for g in last], axis=0),
            math.degrees(math.acos(float(np.clip(R[2, 1], -1.0, 1.0)))))


def obj_watch(meas, info, mm=1.5, turn_deg=3.0, lost_s=1.5):
    """check() for Arm.move before the close: a reason when the object moves mm or turns turn_deg in the base frame B
    (a finger pushing it; it rests on the table, so any arm move leaves it still in B), or when the tracker has not
    seen it for lost_s (the hand may hide tags for a moment). info: measure_object's (p_B, q_B)."""
    p_ref, q_ref = np.asarray(info["p_B"], float), np.asarray(info["q_B"], float)

    def check():
        with meas.lock:
            got = meas.samples[-3:]
        if not got or time.time() - got[-1][0] > lost_s:
            return f"the tracker lost the object for {lost_s:g} s"
        pb = np.median([g[4] for g in got], axis=0)
        d = float(np.linalg.norm(pb - p_ref))
        turn = math.degrees(float(np.linalg.norm(HS.quat_to_rotvec(HS.quat_mul(HS.quat_conj(q_ref), got[-1][5])))))
        if d > mm or turn > turn_deg:
            return f"the object moved {d:.2f} mm and turned {turn:.2f} deg in B (limits {mm:g}, {turn_deg:g})"
        return None
    return check


def joint_watch(st, j_ref, deflect, fingers=HS.FINGERS):
    """check() for Arm.move: a reason when a finger joint lies more than deflect deg off its reading at the leg's start
    (a finger pressed by the table or the object)."""
    last = [0.0]

    def check():
        if time.time() - last[0] < 0.1:
            return None
        last[0] = time.time()
        j = HS.sim_joints(st.state())
        if j is None:
            return None
        worst = max((abs(j[f][k] - j_ref[f][k]), f, k) for f in fingers for k in J)
        return (f"the {worst[1]} {worst[2]} moved {worst[0]:.1f} deg from its reading at the start (limit {deflect:g}): "
                "a finger touches something") if worst[0] > deflect else None
    return check


def both(*checks):
    def check():
        for c in checks:
            why = c()
            if why:
                return why
        return None
    return check


def ramp(st, j_from, j_to, seconds, speed, check=None):
    """The station's manual/joints in steps of 0.1 s from j_from to j_to; check() between steps may stop it (returns
    the reason)."""
    n = max(1, int(round(10 * seconds)))
    for i in range(1, n + 1):
        u = i / n
        step = {f: {k: j_from[f][k] + (j_to[f][k] - j_from[f][k]) * u for k in J} for f in j_to}
        st.call("POST", "/api/v1/manual/joints", {"joints": step, "servo_speed": speed})
        time.sleep(0.1)
        if check is not None:
            why = check()
            if why:
                return why, step
    return None, j_to


# ------------------------------------------------------------------------------------------------ commands

def measure_object(meas, a):
    p, q, info = meas.pose(a.measure_s, settle=0.0)
    R_P, R_B = quat_to_mat(q), quat_to_mat(np.asarray(info["q_B"], float))
    tilt = math.degrees(math.acos(float(np.clip(R_B[2, 1], -1.0, 1.0))))
    return p, q, info, -R_P[:, 1], tilt


def cmd_plan(a):
    base = HS.load_plan(a.base_plan)
    h = HS.map_height(a.map)
    arm = HS.Arm()
    meas = HS.Measure(a)
    try:
        p, q, info, stem_P, tilt = measure_object(meas, a)
        T_BP = arm.T_BP()
    finally:
        meas.stop()
        arm.close()
    above = float(info["p_B"][2]) - a.table_z
    sh = stem_P[:2] / np.linalg.norm(stem_P[:2])
    off = math.degrees(math.asin(min(1.0, abs(float(sh[0])))))
    print(f"object: ball centre in P {np.round(p, 1).tolist()} mm, {above:.1f} mm above the table; axis {tilt:.1f} deg "
          f"from vertical; the stem in P {np.round(stem_P, 3).tolist()} ({off:.1f} deg off across x_P)", flush=True)
    if tilt < 60.0:
        raise SystemExit("the object is not lying on its side: this skill picks up a lying object")
    if off > a.max_axis_deg:
        raise SystemExit(f"the object's axis is {off:.1f} deg off across x_P (limit {a.max_axis_deg:g}): a pinch along "
                         "x_P would not let it turn upright")
    sol = plan_pinch(base["meta"]["hom"]["d6_P_mm"], p, stem_P, above, h, a.lat_deg, a.gap, a.squeeze)
    c = sol["c"]
    e = R_TAPED * math.sin(math.radians(a.lat_deg))
    print(f"pinch: thumb-{sol['partner']}, centre in P {np.round(c, 1).tolist()} mm, contacts {a.lat_deg:g} deg above "
          f"the equator (hang about {math.degrees(math.atan2(e, COM_MM)):.1f} deg from vertical once clear); joint "
          f"margin {sol['margin_deg']:.1f} deg; contacts at z {sol['contact_z_tip'][0]:.1f} / {sol['contact_z_tip'][1]:.1f}"
          f" mm in the tip frames (the pad ends at -11)")
    for name in ("open", "touch"):
        print("   " + HS.show_pose(name, pose_full(sol, name)))
    print("   " + HS.show_pose("grip", pose_full(sol, "touch", a.squeeze)))
    for name, cl in (("touch", sol["clear_touch"]), ("open", sol["clear_open"])):
        print(f"   at {name}: " + "; ".join(f"{f} table {v[0]:.1f} stem/block {v[1]:.1f} ball {v[2]:.1f}"
                                            for f, v in cl.items()))
    print("   coming down in the open pose, the least gap to the ball: "
          + "; ".join(f"{f} {v:.1f} mm" for f, v in sol["descent_mm"].items()))
    v = sol["clear_third"]
    print(f"   {sol['third']} out of the way: table {v[0]:.1f} stem/block {v[1]:.1f} ball {v[2]:.1f} mm")
    out = Path(a.out or (HERE.parent / f"plans/{time.strftime('%Y%m%d')}-pinch_pickup_thumb_{sol['partner']}_plan.json"))
    write_plan(sol, out, base, h)
    d = p - c
    print(f"plan -> {out}\nmoves: lateral {np.hypot(d[0], d[1]):.1f} mm in P's xy, then down {c[2] - p[2]:+.1f} mm "
          f"along z_P (palm {T_BP[2, 3]:.1f} -> {T_BP[2, 3] - (c[2] - p[2]):.1f} mm in B)", flush=True)
    note({"cmd": "plan", "p_P": p.tolist(), "p_B": info["p_B"], "tilt": tilt, "stem_P": stem_P.tolist(),
          "c": c.tolist(), "partner": sol["partner"], "table_mm": sol["table_mm"], "plan": str(out)})
    return 0


def load_sol(plan):
    pp = plan["meta"]["hom"]["pinch"]
    poses = {p["name"]: p["joints"] for p in plan["poses"]}
    return pp, poses


def cmd_grasp(a):
    plan = HS.load_plan(a.plan)
    pp, poses = load_sol(plan)
    c = np.asarray(pp["c_P_mm"], float)
    m = mounts(plan["meta"]["hom"]["d6_P_mm"])
    h = HS.map_height(a.map)
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    bad = HS.mounts_off(plan, s)
    if bad:
        raise SystemExit(f"the station's mounts are not the plan's: {bad}")
    if a.go and s.get("servo_torque") != 1:
        raise SystemExit("servo torque is off")
    j0 = HS.sim_joints(s)
    print("   " + HS.show_pose("measured", j0))
    pads = None if a.no_pads else Pads()
    arm = HS.Arm(a.ur_ft, a.blank_s)
    meas = HS.Measure(a, video=HS.recorder(a, "pinch_grasp"))
    stream = None
    try:
        p, q, info, stem_P, tilt = measure_object(meas, a)
        above = float(info["p_B"][2]) - a.table_z
        T_BP = arm.T_BP()
        palm_above = T_BP[2, 3] - a.table_z
        d = p - c
        print(f"object: ball centre in P {np.round(p, 1).tolist()} ({above:.1f} mm above the table), tilt {tilt:.1f} deg; "
              f"the pinch's centre {np.round(c, 1).tolist()}: lateral {np.hypot(d[0], d[1]):.1f} mm, down "
              f"{c[2] - p[2]:.1f} mm", flush=True)
        if np.hypot(d[0], d[1]) > a.max_lat:
            raise SystemExit(f"lateral {np.hypot(d[0], d[1]):.1f} mm is more than --max-lat {a.max_lat:g}")
        # the open fingers high over everything before any lateral move: their lowest part over the object's top
        top = above + R_TAPED                                    # the ball's top above the table (the block lies lower)
        low_open = min(lowest(f, m[f], [poses["open"][f][k] for k in J])[0] for f in HS.FINGERS)
        over = palm_above + low_open - top
        print(f"   with the open pose here the fingers' lowest part is {over:.1f} mm over the ball's top", flush=True)
        if over < a.over_mm:
            raise SystemExit(f"the hand is too low to move sideways ({over:.1f} mm over the ball, {a.over_mm:g} wanted): "
                             "raise it first (hom_setup.py place)")
        if not a.go:
            return 0
        arm.connect(a.stop_n is not None)
        stream = Stream(st, meas, arm, STREAMS / f"{time.strftime('%Y%m%d-%H%M%S')}_pinch_grasp_telemetry.jsonl",
                        pads=pads)
        stream.mark("open")
        why, _ = ramp(st, j0, poses["open"], a.open_s, a.speed)
        time.sleep(1.0)
        # lateral: the ball's centre under the pinch's centre, at this height
        for leg in ("lateral", "fine"):
            p, q, info, stem_P, tilt = measure_object(meas, a)
            d = p - c
            lat = float(np.hypot(d[0], d[1]))
            if leg == "fine" and lat < 0.5:
                break
            if leg == "fine" and lat > 4.0:
                raise SystemExit(f"still {lat:.1f} mm off after the lateral move")
            C = tf(None, [d[0], d[1], 0.0])
            stream.mark(leg)
            why, _ = arm.move(arm.T_BP() @ C @ tf_inv(T_EP), a.v_lat, a.acc, a.stop_n, lat / (1e3 * a.v_lat) + 6.0,
                              f"{leg} {lat:.1f} mm", check=obj_watch(meas, info))
            if why:
                raise SystemExit(f"the {leg} move stopped: {why}")
        # the descent: fast to --slow-mm over the grasp's height, then slowly, each leg watched
        for leg in ("fast", "slow"):
            p, q, info, stem_P, tilt = measure_object(meas, a)
            dz = float(c[2] - p[2])                               # the palm down by dz puts the ball at c's height in P
            dist = dz - a.slow_mm if leg == "fast" else dz
            v = a.v_down if leg == "fast" else a.v_slow
            if dist < 0.5:
                continue
            jr = HS.sim_joints(st.state())
            stream.mark(f"descent {leg}")
            T_goal = arm.T_BE()
            T_goal[2, 3] -= dist
            why, _ = arm.move(T_goal, v, a.acc, a.stop_n, dist / (1e3 * v) + 6.0, f"descent, {leg} leg, {dist:.1f} mm",
                              check=both(obj_watch(meas, info), joint_watch(st, jr, a.deflect_deg)))
            if why:
                raise SystemExit(f"the descent stopped: {why}; look before anything else")
        p, q, info, stem_P, tilt = measure_object(meas, a)
        d = p - c
        print(f"   at the grasp's height: the ball {np.round(d, 2).tolist()} mm from the pinch's centre", flush=True)
        if np.hypot(d[0], d[1]) > 0.7:
            if np.hypot(d[0], d[1]) > 3.0:
                raise SystemExit("more than 3 mm off at the grasp's height: look")
            stream.mark("fine at height")
            why, _ = arm.move(arm.T_BP() @ tf(None, [d[0], d[1], 0.0]) @ tf_inv(T_EP), a.v_slow, a.acc, a.stop_n, 8.0,
                              f"fine {np.hypot(d[0], d[1]):.1f} mm at the grasp's height",
                              check=obj_watch(meas, info))
            if why:
                raise SystemExit(f"the fine move stopped: {why}")
        # the close: open -> touch in small steps (the object watched: a finger arriving first pushes it), then the squeeze
        if pads is not None:
            pads.set_rest(0.6)                            # the pads' rest: the fingers open round the ball, nothing touching
        p0 = obj_now(meas)
        jo = HS.sim_joints(st.state())
        stream.mark("close")

        def moved():
            o = obj_now(meas)
            if o is None or p0 is None:
                return None
            dd = float(np.linalg.norm(o[0] - p0[0]))
            return f"the object moved {dd:.1f} mm during the close (limit {a.close_mm:g})" if dd > a.close_mm else None
        why, at = ramp(st, {f: jo[f] for f in HS.FINGERS}, poses["touch"], a.close_s, a.speed, check=moved)
        if why:
            print(f"   the close stopped: {why}; holding there", flush=True)
            raise SystemExit("look before anything else")
        time.sleep(0.8)
        print("   at touch: " + pads_text(None if pads is None else pads.change()), flush=True)
        grip = {f: {k: poses["touch"][f][k] + (a.squeeze if (k == "pip" and f in ("thumb", pp["partner"])) else 0.0)
                    for k in J} for f in HS.FINGERS}
        stream.mark("squeeze")
        ramp(st, poses["touch"], grip, a.squeeze_s, a.speed)
        time.sleep(1.5)
        p1 = obj_now(meas)
        j1 = HS.sim_joints(st.state())
        load = st.state().get("telemetry", {}).get("servo_load")
        dB = None if pads is None else pads.change()
        print("   " + HS.show_pose("gripped", j1) + f"\n   loads {load}; " + pads_text(dB), flush=True)
        if p1 is not None and p0 is not None:
            print(f"   the object moved {np.round(p1[0] - p0[0], 2).tolist()} mm in P over the close; tilt {p1[2]:.1f} deg",
                  flush=True)
        note({"cmd": "grasp", "plan": a.plan, "c": c.tolist(), "joints": j1, "load": load, "pads_dB": dB,
              "obj": None if p1 is None else {"p_P": p1[0].tolist(), "p_B": p1[1].tolist(), "tilt": p1[2]}})
        print("GRASPED: the pinch closed round the ball; next: lift", flush=True)
    finally:
        if stream is not None:
            stream.stop()
        meas.stop()
        arm.close()
    return 0


def cmd_lift(a):
    plan = HS.load_plan(a.plan)
    pp, poses = load_sol(plan)
    h = HS.map_height(a.map)
    st = HS.Station(a.station, a.token_file)
    pads = None
    if not a.no_pads:
        pads = Pads()
        print(f"   pads: the rest taken {pads.load_rest():.0f} s ago; " + pads_text(pads.change()), flush=True)
    arm = HS.Arm(a.ur_ft, a.blank_s)
    meas = HS.Measure(a, video=HS.recorder(a, "pinch_lift"))
    stream = None
    try:
        p, q, info, stem_P, tilt = measure_object(meas, a)
        above = float(info["p_B"][2]) - a.table_z
        lift = a.lift_mm if a.lift_mm is not None else (h + a.lift_extra) - above
        print(f"object: ball centre {above:.1f} mm above the table, tilt {tilt:.1f} deg; lift {lift:.1f} mm at "
              f"{1e3 * a.v_lift:g} mm/s (the ball's centre to {above + lift:.1f} mm; standing it is at {h:.1f})", flush=True)
        if lift > a.max_lift:
            raise SystemExit(f"{lift:.1f} mm is more than --max-lift {a.max_lift:g}")
        if not a.go:
            return 0
        arm.connect(a.stop_n is not None)
        stream = Stream(st, meas, arm, STREAMS / f"{time.strftime('%Y%m%d-%H%M%S')}_pinch_lift_telemetry.jsonl",
                        pads=pads)
        z_ref = float(p[2])

        def slipping():
            o = obj_now(meas, 3, 0.4)
            if o is None:
                return None                                    # the tags may turn away as the object stands up
            dz = float(o[0][2] - z_ref)
            return (f"the ball slid {dz:+.1f} mm along z_P in the pinch (limit {a.slip_mm:g})"
                    if abs(dz) > a.slip_mm else None)
        T_goal = arm.T_BE()
        T_goal[2, 3] += lift
        stream.mark("lift")
        why, moved = arm.move(T_goal, a.v_lift, a.acc, a.stop_n, lift / (1e3 * a.v_lift) + 8.0, f"lift {lift:.1f} mm",
                              check=slipping)
        time.sleep(1.5)
        p, q, info, stem_P, tilt = measure_object(meas, a)
        above = float(info["p_B"][2]) - a.table_z
        print(f"   after the lift ({moved:.1f} mm{', STOPPED: ' + why if why else ''}): ball centre {above:.1f} mm above "
              f"the table, z_P {p[2] - z_ref:+.1f} mm in the pinch, tilt {tilt:.1f} deg; the block's bottom "
              f"{above - h:+.1f} mm from the table if upright", flush=True)
        dB = None if pads is None else pads.change()
        print("   " + pads_text(dB), flush=True)
        note({"cmd": "lift", "moved": moved, "stopped": why, "tilt": tilt, "above": above, "dz_pinch": float(p[2] - z_ref),
              "pads_dB": dB})
        print(f"{'LIFTED' if not why else 'LIFT STOPPED'}: tilt {tilt:.1f} deg", flush=True)
    finally:
        if stream is not None:
            stream.stop()
        meas.stop()
        arm.close()
    return 0


def cmd_slip(a):
    plan = HS.load_plan(a.plan)
    pp, poses = load_sol(plan)
    st = HS.Station(a.station, a.token_file)
    pads = None
    if not a.no_pads:
        pads = Pads()
        print(f"   pads: the rest taken {pads.load_rest():.0f} s ago; " + pads_text(pads.change()), flush=True)
    meas = HS.Measure(a, video=HS.recorder(a, "pinch_slip"))
    stream = None
    try:
        p, q, info, stem_P, tilt = measure_object(meas, a)
        j = HS.sim_joints(st.state())
        cur = {f: dict(j[f]) for f in HS.FINGERS}
        target = {f: {k: poses["touch"][f][k] for k in J} for f in HS.FINGERS}
        pinch = ("thumb", pp["partner"])
        # the squeeze now: the commanded PIPs past touch (the station holds the last command; the plan's grip)
        sq = a.from_squeeze if a.from_squeeze is not None else float(pp["squeeze_deg"])
        print(f"object: tilt {tilt:.1f} deg (goal {a.tilt_goal:g}); squeeze {sq:g} deg on the PIPs; steps of "
              f"{a.step_deg:g} deg every {a.dwell_s:g} s down to {a.min_squeeze:g}", flush=True)
        if not a.go or tilt <= a.tilt_goal:
            return 0
        stream = Stream(st, meas, None, STREAMS / f"{time.strftime('%Y%m%d-%H%M%S')}_pinch_slip_telemetry.jsonl",
                        pads=pads)
        z_ref = float(p[2])
        hist = []
        while sq - a.step_deg >= a.min_squeeze - 1e-9:
            sq -= a.step_deg
            cmd = {f: {k: target[f][k] + (sq if (k == "pip" and f in pinch) else 0.0) for k in J} for f in HS.FINGERS}
            stream.mark(f"squeeze {sq:.2f}")
            st.call("POST", "/api/v1/manual/joints", {"joints": cmd, "servo_speed": a.speed})
            time.sleep(a.dwell_s)
            o = obj_now(meas, 5, 0.5)
            if o is None:
                print(f"   squeeze {sq:5.2f}: the tracker does not see the object", flush=True)
                continue
            dz = float(o[0][2] - z_ref)
            dB = None if pads is None else pads.change()
            hist.append((sq, o[2], dz, dB))
            print(f"   squeeze {sq:5.2f} deg: tilt {o[2]:5.1f} deg, the ball {dz:+.2f} mm along z_P; " + pads_text(dB),
                  flush=True)
            if abs(dz) > a.slip_mm:
                back = sq + 2 * a.step_deg
                cmd = {f: {k: target[f][k] + (back if (k == "pip" and f in pinch) else 0.0) for k in J}
                       for f in HS.FINGERS}
                st.call("POST", "/api/v1/manual/joints", {"joints": cmd, "servo_speed": a.speed})
                print(f"   the ball slides in the pinch: squeeze back to {back:.2f} deg", flush=True)
                break
            if o[2] <= a.tilt_goal:
                if a.resqueeze > 0:
                    back = sq + a.resqueeze
                    cmd = {f: {k: target[f][k] + (back if (k == "pip" and f in pinch) else 0.0) for k in J}
                           for f in HS.FINGERS}
                    st.call("POST", "/api/v1/manual/joints", {"joints": cmd, "servo_speed": a.speed})
                    print(f"   at the goal: squeeze back up to {back:.2f} deg", flush=True)
                break
        time.sleep(1.0)
        p, q, info, stem_P, tilt = measure_object(meas, a)
        note({"cmd": "slip", "history": hist, "tilt_end": tilt})
        print(f"SLIP DONE: tilt {tilt:.1f} deg", flush=True)
    finally:
        if stream is not None:
            stream.stop()
        meas.stop()
    return 0


def cmd_status(a):
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    print("   " + HS.show_pose("measured", HS.sim_joints(s)))
    print(f"   loads {s.get('telemetry', {}).get('servo_load')}")
    meas = HS.Measure(a)
    try:
        p, q, info, stem_P, tilt = measure_object(meas, a)
    finally:
        meas.stop()
    print(f"   object: ball centre in P {np.round(p, 1).tolist()}, in B {info['p_B']}, "
          f"{info['p_B'][2] - a.table_z:.1f} mm above the table, tilt {tilt:.1f} deg")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--object", default="sphere", choices=("sphere",))
        p.add_argument("--map", required=True)
        p.add_argument("--station", default="http://10.99.99.2:8765")
        p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
        p.add_argument("--table-z", type=float, default=3.0)
        p.add_argument("--measure-s", type=float, default=1.0)
        p.add_argument("--go", action="store_true")
        p.add_argument("--stop-n", type=float, default=3.0)
        p.add_argument("--ur-ft", action="store_true")
        p.add_argument("--blank-s", type=float, default=0.4)
        p.add_argument("--acc", type=float, default=0.05)
        p.add_argument("--speed", type=int, default=40)
        p.add_argument("--video-dir", default=None)
        p.add_argument("--no-pads", action="store_true", help="without the fingertip boards' log")

    p = sub.add_parser("plan")
    common(p)
    p.add_argument("--base-plan", default=str(HERE.parent / "plans/20261116-film_x105_y70_twist_g10_lat12_plan.json"),
                   help="a plan at the station's layout (mounts, d6_P_mm)")
    p.add_argument("--lat-deg", type=float, default=3.0, help="the contacts above the ball's equator")
    p.add_argument("--gap", type=float, default=10.0, help="mm the pads stand off the ball in the open pose")
    p.add_argument("--squeeze", type=float, default=8.0, help="deg the PIPs go past touch in the grip")
    p.add_argument("--max-axis-deg", type=float, default=10.0)
    p.add_argument("--out", default=None)
    p = sub.add_parser("grasp")
    common(p)
    p.add_argument("--plan", required=True)
    p.add_argument("--max-lat", type=float, default=120.0)
    p.add_argument("--over-mm", type=float, default=20.0, help="mm the open fingers keep over the ball's top sideways")
    p.add_argument("--open-s", type=float, default=3.0)
    p.add_argument("--v-lat", type=float, default=0.015)
    p.add_argument("--v-down", type=float, default=0.008)
    p.add_argument("--v-slow", type=float, default=0.003)
    p.add_argument("--slow-mm", type=float, default=15.0)
    p.add_argument("--deflect-deg", type=float, default=4.0)
    p.add_argument("--close-s", type=float, default=4.0)
    p.add_argument("--close-mm", type=float, default=5.0, help="the object's motion that stops the close")
    p.add_argument("--squeeze", type=float, default=8.0)
    p.add_argument("--squeeze-s", type=float, default=1.5)
    p = sub.add_parser("lift")
    common(p)
    p.add_argument("--plan", required=True)
    p.add_argument("--lift-mm", type=float, default=None)
    p.add_argument("--lift-extra", type=float, default=12.0, help="mm over the standing height the ball's centre goes")
    p.add_argument("--max-lift", type=float, default=90.0)
    p.add_argument("--v-lift", type=float, default=0.003)
    p.add_argument("--slip-mm", type=float, default=5.0)
    p = sub.add_parser("slip")
    common(p)
    p.add_argument("--plan", required=True)
    p.add_argument("--tilt-goal", type=float, default=6.0)
    p.add_argument("--from-squeeze", type=float, default=None)
    p.add_argument("--step-deg", type=float, default=0.5)
    p.add_argument("--dwell-s", type=float, default=1.5)
    p.add_argument("--min-squeeze", type=float, default=0.0)
    p.add_argument("--slip-mm", type=float, default=2.0)
    p.add_argument("--resqueeze", type=float, default=1.0)
    p = sub.add_parser("status")
    common(p)
    a = ap.parse_args()
    return {"plan": cmd_plan, "grasp": cmd_grasp, "lift": cmd_lift, "slip": cmd_slip, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
