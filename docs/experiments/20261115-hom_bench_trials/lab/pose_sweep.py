"""The palm clear of the object before a gantry move, and a short free-air pose sweep at each new layout (2026-11-19;
the owner, relayed 18:1x: "if we can do a short free-air pose sweep at each layout in like <1 minute, it's worth it, at
least initially, just to see if it actually matters"). For hom_series.py between layouts: the boards record in the
splitter's file meanwhile, and the sweeps are the data that show whether the own-pose term depends on the layout.

    clear   the palm straight up in base z, guarded by the HEX-E, until every finger part (tip pad, tip end, PIP joint;
            hom_setup's balls) keeps --clear-mm above the object's top, with the fingers straight and in every sweep
            pose and along the sweep's paths, the object taken to be anywhere under the hand. The object's top: the
            table (--table-z), the object's centre height from its tag map and the plan's radius. Nothing moves when
            the palm is high enough already; refused past --max-lift. A gantry moves the fingers along x_P and y_P
            only, so the plan's mounts stand for the present ones in height. Prints "CLEAR OK".
    sweep   at the plan's mounts (the station's must match): the 12 poses of POSES, the plan's open pose first, each
            checked on the hand model before it moves: every finger part at least --ff-mm from every other finger's
            parts along the joints' path from the previous pose, every part --clear-mm above the object's top at the
            palm's present height, and every target inside the station's safe range (servo_limits) with the MCP at
            most --mcp-max and MCP + PIP at most --curl-max. A pose that fails is skipped and logged. Each pose is
            reached in --ramp-s by the station's manual joints (ten steps a second) and held --hold-s; then the fingers
            go back to where the sweep began. A joint more than --max-err deg (yaw --max-err-yaw) from its target at
            the end of a hold, or torque off, stops the sweep and takes the fingers back. One JSON line per pose to
            --out (the hold's start and end, target, measured joints). Prints "SWEEP DONE".
            The thumb's DIP tag (24, 14 mm) is tracked by both cameras through the sweep (--no-thumb-tag to skip; the
            owner, relayed 11-19 18:1x: "maybe tracking the thumb fingertip tag would help us understand slop as
            well?"): each pose's line carries the tag's median centre and normal in P over the hold against the hand
            model's at the measured joints (fingertip_tag_check.fk). The cameras see it 12-39 deg off its normal in
            half the poses (zero, mcp 30, the yaws, the open pose; 2.6 mm and 2.1 deg from the model at zero, 21:00).
            Best effort: without cameras the sweep runs without it.

    /usr/bin/python3 pose_sweep.py clear --plan PLAN --map MAP [--go]
    /usr/bin/python3 pose_sweep.py sweep --plan PLAN --map MAP [--go]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "src")]
import hom_setup as HS  # noqa: E402
from sr2hand.control import servo_limits as SL  # noqa: E402
from sr2hand.geometry.rotations import quat_to_mat, tf, tf_inv  # noqa: E402

FINGERS = HS.FINGERS
J = ("yaw", "mcp", "pip")
# The same 12 poses at every layout (simulator degrees, (yaw, mcp, pip)); "open" is the plan's open pose. Each joint
# alone, together and against the other fingers, inside the 11-04 sweep's caps (MCP <= 50, MCP + PIP <= 120, |yaw| <= 25).
POSES = [
    ("open", None),
    ("zero", {f: (0, 0, 0) for f in FINGERS}),
    ("mcp 30", {f: (0, 30, 0) for f in FINGERS}),
    ("pip 45", {f: (0, 0, 45) for f in FINGERS}),
    ("curl 40/40", {f: (0, 40, 40) for f in FINGERS}),
    ("yaw +15", {f: (15, 20, 0) for f in FINGERS}),
    ("yaw -15", {f: (-15, 20, 0) for f in FINGERS}),
    ("thumb curl", {"thumb": (0, 40, 50), "index": (0, 0, 0), "middle": (0, 0, 0)}),
    ("index curl", {"thumb": (0, 0, 0), "index": (0, 40, 50), "middle": (0, 0, 0)}),
    ("middle curl", {"thumb": (0, 0, 0), "index": (0, 0, 0), "middle": (0, 40, 50)}),
    ("mixed A", {"thumb": (15, 10, 60), "index": (15, 40, 20), "middle": (15, 25, 50)}),
    ("mixed B", {"thumb": (-15, 45, 10), "index": (15, 10, 60), "middle": (-15, 45, 30)}),
]
OUT = ROOT / "logs/20261119-pose_sweeps"


def pose_dict(plan, name, spec):
    """The pose's targets, clipped 0.5 deg inside the station's safe ranges (the index's PIP starts at +1.8 deg: its
    pot's dead zone, 11-17), as the HOM targets are."""
    j = HS.open_pose(plan) if spec is None else {f: {k: float(v) for k, v in zip(J, spec[f])} for f in FINGERS}
    for f in FINGERS:
        for k in J:
            lo, hi = SL.safe_range_sim(f, k)
            j[f][k] = float(np.clip(j[f][k], lo + 0.5, hi - 0.5))
    return j


def object_top_B(a, plan):
    """mm: the highest point of the object resting on its block, in base z."""
    r = 1e3 * float(plan["meta"]["hom"]["setup"]["r_m"])
    return a.table_z + HS.map_height(a.map) + r


def lowest_B(plan, T_BP, poses):
    """(mm, where): the lowest finger part (its ball's bottom) over the poses and the joints' paths between them, in
    base z, at the palm pose T_BP."""
    best = (math.inf, "")
    for (n0, j0), (n1, j1) in zip(poses[:-1], poses[1:]):
        for f, u, part, pt, rad in HS.path_pts(plan, j0, j1):
            z = float((T_BP[:3, :3] @ pt + T_BP[:3, 3])[2]) - rad
            if z < best[0]:
                best = (z, f"{f} {part} between {n0} and {n1} ({u:.0%})")
    return best


def finger_gap(plan, j0, j1, n=10):
    """(mm, where): the least gap between parts of different fingers along the joints' straight path j0 -> j1."""
    best = (math.inf, "")
    for i in range(n + 1):
        u = i / n
        q = {f: {k: j0[f][k] + (j1[f][k] - j0[f][k]) * u for k in J} for f in FINGERS}
        pts = {f: HS.finger_pts(plan, f, q[f]) for f in FINGERS}
        for a_, b_ in (("thumb", "index"), ("thumb", "middle"), ("index", "middle")):
            for (pa, ra), na in zip(pts[a_], HS.PARTS):
                for (pb, rb), nb in zip(pts[b_], HS.PARTS):
                    d = float(np.linalg.norm(pa - pb)) - ra - rb
                    if d < best[0]:
                        best = (d, f"{a_} {na} and {b_} {nb} at {u:.0%}")
    return best


def range_ok(a, j):
    for f in FINGERS:
        for k in J:
            lo, hi = SL.safe_range_sim(f, k)
            if not lo <= j[f][k] <= hi:
                return f"{f} {k} {j[f][k]:+.1f} outside {lo:+.1f} .. {hi:+.1f}"
        if j[f]["mcp"] > a.mcp_max or j[f]["mcp"] + j[f]["pip"] > a.curl_max:
            return f"{f} mcp {j[f]['mcp']:+.1f}, mcp + pip {j[f]['mcp'] + j[f]['pip']:+.1f} over the caps"
    return None


def sweep_poses(plan):
    return [(n, pose_dict(plan, n, s)) for n, s in POSES]


def cmd_clear(a):
    plan = HS.load_plan(a.plan)
    arm = HS.Arm(a.ur_ft)
    T_BP = arm.T_BP()
    top = object_top_B(a, plan)
    poses = [("zero", HS.ZERO)] + sweep_poses(plan) + [("zero", HS.ZERO)]
    low, where = lowest_B(plan, T_BP, poses)
    need = top + a.clear_mm - low
    print(f"object's top at z_B {top:.1f} mm; the fingers' lowest point {low:.1f} mm ({where}): "
          f"{low - top:.1f} mm above it, {a.clear_mm:g} wanted", flush=True)
    if need <= 0.5:
        print("CLEAR OK: no lift needed", flush=True)
        return 0
    dz = need + 2.0
    if dz > a.max_lift:
        raise SystemExit(f"a lift of {dz:.1f} mm is more than --max-lift {a.max_lift:g}: stopping")
    print(f"lift {dz:.1f} mm straight up in base z" + ("" if a.go else " (dry)"), flush=True)
    if not a.go:
        return 0
    arm.connect(a.stop_n is not None)
    try:
        T_goal = tf(T_BP[:3, :3], T_BP[:3, 3] + np.array([0.0, 0.0, dz])) @ tf_inv(HS.T_EP)
        why, moved = arm.move(T_goal, a.v_up, a.acc, a.stop_n, dz / (1e3 * a.v_up) + 5.0, f"lift {dz:.1f} mm")
    finally:
        arm.close()
    if why:
        raise SystemExit(f"the lift stopped: {why}")
    low2, _ = lowest_B(plan, arm.T_BP(), poses)
    print(f"CLEAR OK: lifted {dz:.1f} mm; the fingers' lowest point {low2 - top:.1f} mm above the object's top", flush=True)
    HS.note({"cmd": "pose_sweep clear", "plan": a.plan, "lift_mm": round(dz, 1)})
    return 0


class ThumbTag:
    """The thumb's DIP tag from both cameras through the sweep (live.Tracker with tag 24's one-tag map, as
    fingertip_tag_check.py measure); at() gives a hold's median against the model. A failure leaves self.error set."""

    def __init__(self, plan):
        import threading
        self.got, self.lock, self.arm, self.tracker, self.error = [], threading.Lock(), None, None, None
        try:
            import fingertip_tag_check as F
            import hom_trial as HT
            self.F, self.M = F, F.mounts_P(plan)

            class _A:
                map = F.tag_map_file(24)
            self.arm, self.tracker = HT.start_tracker(_A, self.on)
        except Exception as e:                              # noqa: BLE001
            self.error = repr(e)[:200]
            print(f"   thumb tag: no tracker ({self.error})", flush=True)

    def on(self, smp):
        m = smp.msg
        if m is not None and m.n_tags > 0 and np.all(np.isfinite(m.p_PO)):
            with self.lock:
                self.got.append((time.time(), np.array(m.p_PO, float), np.array(m.q_PO, float), int(m.n_tags)))

    def at(self, t0, t1, joints):
        if self.error or not joints:
            return {"seen": 0, "error": self.error}
        try:
            with self.lock:
                g = [x for x in self.got if t0 <= x[0] <= t1]
            q = [joints["thumb"][k] for k in J]
            p_pred, n_pred, _, _ = self.F.fk("thumb", self.M["thumb"], q)
            if len(g) < 3:
                return {"seen": len(g), "joints": q}
            P = np.array([x[1] for x in g])
            p_m = np.median(P, axis=0)
            n_m = -np.mean([quat_to_mat(x[2])[:, 2] for x in g], axis=0)
            n_m /= np.linalg.norm(n_m)
            d = p_m - p_pred
            return {"seen": len(g), "n_tags": int(np.median([x[3] for x in g])), "joints": q,
                    "p_meas": np.round(p_m, 3).tolist(), "p_pred": np.round(p_pred, 3).tolist(),
                    "d_mm": np.round(d, 3).tolist(), "n_meas": np.round(n_m, 4).tolist(),
                    "n_pred": np.round(n_pred, 4).tolist(),
                    "normal_deg": round(math.degrees(math.acos(float(np.clip(n_m @ n_pred, -1, 1)))), 2),
                    "sd_mm": np.round(P.std(axis=0), 3).tolist(),
                    # one camera's depth scatters 2.5 mm (sd) along its view, both cameras' 0.3 mm (21:05, 11-19)
                    **self.two(g, p_pred, n_pred)}
        except Exception as e:                              # noqa: BLE001
            return {"seen": 0, "error": repr(e)[:200]}

    @staticmethod
    def two(g, p_pred, n_pred):
        g2 = [x for x in g if x[3] >= 2]
        if len(g2) < 3:
            return {"seen_2cam": len(g2)}
        P = np.array([x[1] for x in g2])
        n_m = -np.mean([quat_to_mat(x[2])[:, 2] for x in g2], axis=0)
        n_m /= np.linalg.norm(n_m)
        return {"seen_2cam": len(g2), "d_mm_2cam": np.round(np.median(P, axis=0) - p_pred, 3).tolist(),
                "normal_deg_2cam": round(math.degrees(math.acos(float(np.clip(n_m @ n_pred, -1, 1)))), 2),
                "sd_mm_2cam": np.round(P.std(axis=0), 3).tolist()}

    def stop(self):
        for x in (self.tracker, self.arm):
            try:
                if x is not None:
                    x.stop()
            except Exception:                               # noqa: BLE001
                pass


def cmd_sweep(a):
    plan = HS.load_plan(a.plan)
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    off = HS.mounts_off(plan, s)
    if off and a.go:
        raise SystemExit(f"the station's mounts are not the plan's: {off}")
    if off:
        print(f"(dry: the station's mounts are not the plan's; the checks use the plan's: {sorted(off)})", flush=True)
    if a.go and s.get("servo_torque") != 1:
        raise SystemExit("servo torque is not on")
    j_start = HS.sim_joints(s)
    if j_start is None:
        raise SystemExit("no joint telemetry")
    arm = HS.Arm(a.ur_ft)
    T_BP = arm.T_BP()
    top = object_top_B(a, plan)
    out = Path(a.out) if a.out else OUT / f"{time.strftime('%Y%m%d-%H%M%S')}_{plan['design']}_sweep.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    plan_name = Path(a.plan).name
    rows, prev, prev_name = [], j_start, "start"
    for name, j in sweep_poses(plan):
        why = range_ok(a, j)
        gap, gwhere = finger_gap(plan, prev, j)
        low, lwhere = lowest_B(plan, T_BP, [(prev_name, prev), (name, j)])
        if why is None and gap < a.ff_mm:
            why = f"fingers {gap:.1f} mm apart ({gwhere}), under --ff-mm {a.ff_mm:g}"
        if why is None and low - top < a.clear_mm:
            why = f"{low - top:.1f} mm above the object's top ({lwhere}), under --clear-mm {a.clear_mm:g}"
        print(f"   {name:12s} " + ("skipped: " + why if why else f"ok (fingers {gap:.1f} mm apart, {low - top:.1f} mm "
                                                                     "above the object)"), flush=True)
        rows.append((name, j, why, gap, low - top))
        if why is None:
            prev, prev_name = j, name
    run = [r for r in rows if r[2] is None]
    print(f"{len(run)} of {len(rows)} poses at {plan['design']}: {len(run) * (a.ramp_s + a.hold_s) + 2 * a.ramp_s:.0f} s"
          + ("" if a.go else " (dry)"), flush=True)
    if not a.go:
        return 0
    log = open(out, "a")
    log.write(json.dumps({"what": "pose sweep", "plan": plan_name, "design": plan["design"],
                          "mounts": s.get("mount_positions"), "start": j_start, "T_BP": np.round(T_BP, 3).tolist(),
                          "skipped": [{"name": n, "why": w} for n, _, w, _, _ in rows if w], "t": time.time()}) + "\n")
    cur, stopped = j_start, None
    tag = ThumbTag(plan) if a.thumb_tag else None
    try:
        for name, j, why, gap, clr in rows:
            if why:
                continue
            t0 = time.time()
            HS.ramp_joints(argparse.Namespace(speed=a.speed, wait_s=0.0), st, cur, j, a.ramp_s)
            th0 = time.time()
            time.sleep(a.hold_s)
            th1 = time.time()
            s2 = st.state()
            m = HS.sim_joints(s2)
            err = {f: {k: round(m[f][k] - j[f][k], 2) for k in J} for f in FINGERS} if m else None
            tt = tag.at(th0 + 0.3, th1, m) if tag is not None else None
            log.write(json.dumps({"pose": name, "target": j, "measured": m, "error": err, "t_move": t0, "t_hold": th0,
                                  "t_end": th1, "torque": s2.get("servo_torque"), "gap_mm": round(gap, 1),
                                  "above_object_mm": round(clr, 1), "thumb_tag": tt}) + "\n")
            log.flush()
            cur = j
            if s2.get("servo_torque") != 1:
                stopped = "torque is off"
                break
            if err is not None:
                bad = [f"{f} {k} {err[f][k]:+.1f}" for f in FINGERS for k in J
                       if abs(err[f][k]) > (a.max_err_yaw if k == "yaw" else a.max_err)]
                if bad:
                    stopped = f"{name}: off target " + ", ".join(bad)
                    break
            seen = "" if not tt else (f"; thumb tag {tt['seen']} poses" + (f", {np.linalg.norm(tt['d_mm']):.1f} mm "
                                                                           f"and {tt['normal_deg']:.1f} deg from the model"
                                                                           if "d_mm" in tt else ""))
            print(f"   {name:12s} held {th1 - th0:.1f} s{seen}", flush=True)
    finally:
        if tag is not None:
            tag.stop()
        HS.ramp_joints(argparse.Namespace(speed=a.speed, wait_s=0.5), st, cur, j_start, a.ramp_s)
        log.write(json.dumps({"pose": "back", "target": j_start, "t": time.time(), "stopped": stopped}) + "\n")
        log.close()
    HS.note({"cmd": "pose_sweep sweep", "plan": a.plan, "out": str(out), "stopped": stopped})
    if stopped:
        raise SystemExit(f"the sweep stopped ({stopped}); the fingers went back to where it began")
    print(f"SWEEP DONE: {len(run)} poses -> {out}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("clear", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True)
        p.add_argument("--map", required=True)
        p.add_argument("--table-z", type=float, default=3.0, help="the table's height in the base frame (mm)")
        p.add_argument("--clear-mm", type=float, default=25.0)
        p.add_argument("--ur-ft", action="store_true")
        p.add_argument("--go", action="store_true")
    p = sub.choices["clear"]
    p.add_argument("--max-lift", type=float, default=70.0)
    p.add_argument("--v-up", type=float, default=0.02, help="m/s")
    p.add_argument("--acc", type=float, default=0.1)
    p.add_argument("--stop-n", type=float, default=3.0)
    p = sub.choices["sweep"]
    p.add_argument("--station", default="http://10.99.99.2:8765")
    p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    p.add_argument("--ff-mm", type=float, default=8.0)
    p.add_argument("--mcp-max", type=float, default=50.0)
    p.add_argument("--curl-max", type=float, default=110.0)
    p.add_argument("--ramp-s", type=float, default=1.0)
    p.add_argument("--hold-s", type=float, default=3.0)
    p.add_argument("--speed", type=int, default=40)
    p.add_argument("--max-err", type=float, default=6.0)
    p.add_argument("--max-err-yaw", type=float, default=10.0)
    p.add_argument("--no-thumb-tag", dest="thumb_tag", action="store_false")
    p.add_argument("--out", default=None)
    a = ap.parse_args()
    return {"clear": cmd_clear, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
