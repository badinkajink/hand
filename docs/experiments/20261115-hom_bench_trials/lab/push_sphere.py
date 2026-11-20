"""Push the lying sphere with one fingertip (owner, 2026-11-20 13:2x, after the 13:19 snapshot of the sphere lying on the
cables: "alternative: instead of pinch, slowly push, with one of the index or middle finger, the sphere. either along the
stem's axis or perpendicular to it"), to free it from the cables and bring it where the pick-up plans.

The object is measured: its ball's centre and the stem's direction in B. The push goes along the stem (either way) or
across it, whichever of the four is nearest the way to the table's centre (or --dir). The pushing finger (--finger) stays
straight and the other two curl up (--curl MCP PIP). The palm stays level. It moves at its height until the fingertip's
lowest point stands --behind mm behind the ball on the push's line, then straight down until that point is --tip-z above
the table, then along the push by --push mm at --v-push, then straight up by --rise. The HEX-E guards every move
(--stop-n). Then the object is measured again. Without --go it prints the plan.

    /usr/bin/python3 push_sphere.py --plan PLAN --map MAP [--finger middle] [--push 60] --go --video-dir DIR
"""
import argparse
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "src")]
import hom_setup as HS  # noqa: E402
import pinch_pickup as PP  # noqa: E402

TABLE_C = np.array([-150.0, -375.0])          # mm in B: the table's centre (hom_setup's carry)


def lowest(plan, f, q):
    """(point in P, mm) the lowest point of finger f's parts at joints q, and that part's radius."""
    best = None
    for pt, rad in HS.finger_pts(plan, f, q):
        z = float(np.asarray(pt, float)[2]) - rad
        if best is None or z < best[0]:
            best = (z, np.asarray(pt, float), rad)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True, help="a plan with the station's mounts (the finger geometry)")
    ap.add_argument("--map", required=True)
    ap.add_argument("--finger", choices=("index", "middle"), default="middle")
    ap.add_argument("--curl", nargs=2, type=float, default=[50.0, 60.0], help="MCP PIP deg of the other two fingers")
    ap.add_argument("--dir", choices=("auto", "toward-block", "away-from-block", "across+", "across-"), default="auto")
    ap.add_argument("--behind", type=float, default=10.0, help="mm between the fingertip and the ball before the push")
    ap.add_argument("--tip-z", type=float, default=8.0, help="mm of the fingertip's lowest point above the table")
    ap.add_argument("--push", type=float, default=60.0, help="mm the fingertip travels along the push")
    ap.add_argument("--v", type=float, default=0.01, help="m/s of the moves to the start")
    ap.add_argument("--v-push", type=float, default=0.005, help="m/s of the push")
    ap.add_argument("--acc", type=float, default=0.05)
    ap.add_argument("--stop-n", type=float, default=3.0)
    ap.add_argument("--rise", type=float, default=60.0)
    ap.add_argument("--table-z", type=float, default=3.0)
    ap.add_argument("--measure-s", type=float, default=1.0)
    ap.add_argument("--station", default="http://10.99.99.2:8765")
    ap.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    plan = HS.load_plan(a.plan)
    st = HS.Station(a.station, a.token_file)
    j = HS.sim_joints(st.state())
    if j is None or max(abs(j[f][k]) for f in j for k in j[f]) > 5.0:
        raise SystemExit("the fingers are not straight: run hom_setup.py zero first")
    pose = {f: ({"yaw": 0.0, "mcp": 0.0, "pip": 0.0} if f == a.finger else
                {"yaw": 0.0, "mcp": a.curl[0], "pip": a.curl[1]}) for f in HS.FINGERS}
    tip_z, tip_pt, tip_r = lowest(plan, a.finger, pose[a.finger])
    others = min(lowest(plan, f, pose[f])[0] for f in HS.FINGERS if f != a.finger)
    print(f"the {a.finger} straight, the others curled to MCP {a.curl[0]:g} PIP {a.curl[1]:g}: their lowest point "
          f"{others - tip_z:.1f} mm above the {a.finger}'s tip", flush=True)
    if others - tip_z < 25.0:
        raise SystemExit("the curled fingers hang within 25 mm of the pushing tip: curl them more (--curl)")
    a2 = types.SimpleNamespace(map=a.map, object="sphere", video_dir=a.video_dir, plan=a.plan, go=a.go, measure_s=a.measure_s)
    arm = HS.Arm(False, 0.4)
    meas = HS.Measure(a2, video=HS.recorder(a2, "push"), map_file=None)
    try:
        p, q, info, stem_P, tilt = PP.measure_object(meas, a2)
        T_BP = arm.T_BP()
        R = T_BP[:3, :3]
        c = np.asarray(info["p_B"], float)
        u = (R @ np.asarray(stem_P, float))[:2]
        u = u / np.linalg.norm(u)
        perp = np.array([-u[1], u[0]])
        cands = {"toward-block": u, "away-from-block": -u, "across+": perp, "across-": -perp}
        to_c = TABLE_C - c[:2]
        name = (max(cands, key=lambda k: float(cands[k] @ to_c)) if a.dir == "auto" else a.dir)
        d = cands[name]
        tip_P = tip_pt - np.array([0.0, 0.0, tip_r])                 # the tip part's lowest point in P
        s_xy = c[:2] - d * (PP.R_TAPED + tip_r + a.behind)
        z_tip = a.table_z + a.tip_z
        t_goal = np.array([s_xy[0], s_xy[1], z_tip]) - R @ tip_P
        tip_now = T_BP[2, 3] + (R @ tip_P)[2]
        print(f"the ball's centre at {np.round(c, 1).tolist()} mm in B ({c[2] - a.table_z:.1f} mm above the table), the "
              f"stem {np.round(u, 2).tolist()} in B's xy, tilt {tilt:.1f} deg; {np.linalg.norm(to_c):.1f} mm from the table's "
              f"centre", flush=True)
        print(f"push {name} ({np.round(d, 2).tolist()}, {math.degrees(math.acos(float(np.clip(d @ to_c / np.linalg.norm(to_c), -1, 1)))):.0f}"
              f" deg off the way to the centre): the {a.finger}'s tip to {np.round(s_xy, 1).tolist()} at {z_tip:.1f} mm, "
              f"{a.behind:g} mm behind the ball; then {a.push:g} mm at {1e3 * a.v_push:g} mm/s; then up {a.rise:g} mm", flush=True)
        print(f"   the tip now {tip_now - (c[2] + PP.R_TAPED):+.1f} mm over the ball's top; the move to the start is "
              f"{np.linalg.norm(t_goal[:2] - T_BP[:2, 3]):.1f} mm across at this height, then {T_BP[2, 3] - t_goal[2]:.1f} mm "
              f"down", flush=True)
        if tip_now - (c[2] + PP.R_TAPED) < 15.0:
            raise SystemExit("the tip is less than 15 mm over the ball's top: lift first (pose_sweep.py clear)")
        if not a.go:
            return 0
        st.call("POST", "/api/v1/manual/joints", {"joints": pose, "servo_speed": 40})
        time.sleep(3.0)
        arm.connect(True)
        Tp = T_BP.copy()
        for label, t, v in (("across to the start", np.array([t_goal[0], t_goal[1], T_BP[2, 3]]), a.v),
                            ("down to the start", t_goal, a.v),
                            ("push", t_goal + np.array([d[0] * a.push, d[1] * a.push, 0.0]), a.v_push),
                            ("up", t_goal + np.array([d[0] * a.push, d[1] * a.push, a.rise]), a.v)):
            if label == "up":
                t = arm.T_BP()[:3, 3] + np.array([0.0, 0.0, a.rise])
            Tp[:3, 3] = t
            dist = float(np.linalg.norm(t - arm.T_BP()[:3, 3]))
            why, moved = arm.move(Tp @ HS.tf_inv(HS.T_EP), v, a.acc, a.stop_n, dist / (1e3 * v) + 6.0, f"{label}, {dist:.1f} mm")
            if why and label != "push":
                raise SystemExit(f"the {label} stopped: {why}; look before anything else")
            if why:
                print(f"   the push stopped after {moved:.1f} mm: {why}", flush=True)
        p2, q2, info2, stem2, tilt2 = PP.measure_object(meas, a2)
        c2 = np.asarray(info2["p_B"], float)
        print(f"PUSHED: the ball's centre at {np.round(c2, 1).tolist()} mm in B ({c2[2] - a.table_z:.1f} mm above the table), "
              f"moved {np.linalg.norm(c2[:2] - c[:2]):.1f} mm; tilt {tilt2:.1f} deg; {np.linalg.norm(TABLE_C - c2[:2]):.1f} mm "
              f"from the table's centre", flush=True)
    finally:
        meas.stop()
        arm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
