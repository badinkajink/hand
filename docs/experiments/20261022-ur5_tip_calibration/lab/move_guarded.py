"""Guarded moves for the TCP check of 2026-10-22 on the lab's UR5, at the owner's request ("from this right indenter
you go home, above both indenters ... bring the fingertip to 2mm +Z of the left ... go home ... now, 2mm +Z of the right").

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/move_guarded.py plan|go home|over-sphere|over-rod [--gap 2]

plan: prints the waypoints (flange poses), their joints by IK and the least clearances of the tool and the robot's links
to the indenters, the 2x4 and the table (scripts/ur5_cb3_cell.py), without connecting to the robot's control.
go: the same check, then ur_rtde's control (TCP set to the flange, as the calibration program does), the HEX-E polled at
100 Hz with its 50 Hz filter, and each waypoint as a linear move: up 25 mm at 10 mm/s when the tool is low, travel at
25 mm/s, the last 30 mm down at 5 mm/s. A change of the HEX-E's force past --stop-N (3 N) from its reading at the start,
a stale HEX-E, or a protective stop stops the robot (stopL). Every move goes to moves.jsonl.

The indenters' poses: the hand seatings in poses.jsonl, the mount turned -92.15 deg about the tool axis (indenters.py),
levelled about the key. The target puts the pad's contact point (a_M = (10.55, 0, 93.36) mm, the homing's contact
point) --gap mm above the indenter's top, the palmar side down and the finger along x_I (the homing's orientation).
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
sys.path[:0] = [str(HERE), str(ROOT / "scripts"), str(ROOT / "src/morphohand/driver/reskin5x/host")]
import indenters as I                    # noqa: E402
import ur5_cb3_cell as CELL              # noqa: E402
import ur5_tip_calibration as C          # noqa: E402
import ur5_tip_calibration_sim as SIM    # noqa: E402

ROBOT, FT = "192.168.0.4", "192.168.0.6"
TURN = -92.15
SEAT = {"sphere": "left_indenter_seated", "rod": "right_indenter_seated"}
A_M = np.array([10.55, 0.0, 93.36])
MIN_CLEAR = dict(robot=10.0, hex_e=5.0, mount=3.0, finger=2.0, pad=1.0)
PART_CLASS = dict(forearm_link="robot", wrist_1_link="robot", wrist_2_link="robot", wrist_3_link="robot", hex_e="hex_e",
                  adapter="mount", fork="mount", clamp_plate="mount", dip_link="finger", cover="finger", cap="finger",
                  pad="pad")


def load():
    plan = json.loads(I.PLAN.read_text())
    recs = {r["label"]: r for r in map(json.loads, (HERE / "poses.jsonl").read_text().splitlines())}
    T_EM = np.array(plan["geometry"]["T_EM"], float)
    T_EM[:3, :3] = C.rot(C.EZ, TURN)
    return plan, recs, T_EM


def T_BI_of(kind, plan, recs, T_EM):
    T, _, _ = I.indenter_pose(recs[SEAT[kind]]["T_tcp_mm"], kind, plan, T_EM)
    return I.levelled(T, kind)


def over_target(kind, gap, plan, recs, T_EM):
    """Flange pose with the pad's contact point gap mm above the indenter's top, in the homing's orientation."""
    T_BI = T_BI_of(kind, plan, recs, T_EM)
    R = np.array(plan["geometry"]["T_IM_home"], float)[:3, :3]
    c = np.array([0.0, 0.0, I.TOP[kind] + gap])
    T_IM = C.Tm(R, c - R @ A_M)
    return T_BI @ T_IM @ C.inv(T_EM), T_BI


def clearances(T_BE, q_seed, T_EM, plan, recs):
    """Least clearance (mm) per part over both indenters and the cell, with the joints by IK."""
    q = SIM.ik(T_BE, np.asarray(q_seed, float))
    out = {}
    for kind in ("sphere", "rod"):
        T_BI = T_BI_of(kind, plan, recs, T_EM)
        cl = CELL.clearance(q, T_BE, T_BI, T_EM, kind)
        for k, v in cl.items():
            out[k] = min(out.get(k, 1e9), float(v))
    return q, out


def waypoints(what, gap, cur_T, cur_q, plan, recs, T_EM, mode="go"):
    home = C.ur_to_T(recs["home_after_2x4_rotation"]["tcp_pose_m_rad"])
    wps = []                                    # (name, flange pose, speed m/s, stop on a change of N, parts not checked)
    if what == "home":
        if cur_T[2, 3] < home[2, 3] - 40.0:                     # low, maybe on a key: straight up first
            off, up = cur_T.copy(), cur_T.copy()
            off[2, 3] += 10.0       # the key stands 4.5 mm into the socket; the cell model's fork has no socket, so it
            up[2, 3] += 25.0        # clears the key only from 10 mm. A seating leaves the hand's push on the key; lifting
            wps += [("lift off the key 10 mm", off, 0.003, 25.0, {"fork"}),   # releases it (every face drafted 20 deg)
                    ("up 15 mm more", up, 0.010, 3.0, set())]                 # while a jam would climb
        wps.append(("home", home, 0.025, 3.0, set()))
    else:
        kind = what.split("-")[1]
        if C.pose_error(cur_T, home)[0] > 20.0:
            if mode == "go":
                raise SystemExit(f"the tool is {C.pose_error(cur_T, home)[0]:.0f} mm from home: go home first")
            cur_T, cur_q = home, np.radians(recs["home_after_2x4_rotation"]["q_deg"])   # plan it from home
        T_t, _ = over_target(kind, gap, plan, recs, T_EM)
        above = T_t.copy()
        above[2, 3] += 30.0
        wps += [(f"30 mm above the {gap:g} mm point over the {kind}", above, 0.025, 3.0, set()),
                (f"pad {gap:g} mm over the {kind}'s top", T_t, 0.005, 2.0, set())]
    q = cur_q
    rows = []
    for name, T, v, stop_N, exempt in wps:
        q, cl = clearances(T, q, T_EM, plan, recs)
        rows.append(dict(name=name, T=T, v=v, q=q, cl=cl, stop_N=stop_N, exempt=exempt))
    return rows


def print_rows(rows):
    for r in rows:
        p = r["T"][:3, 3]
        low = {k: round(v, 1) for k, v in r["cl"].items()}
        print(f"  {r['name']}: flange ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm at {r['v'] * 1e3:.0f} mm/s, joints "
              f"{np.round(np.degrees(r['q']), 1)} deg\n    least clearances mm {low}")


def check_ok(rows):
    bad = []
    for r in rows:
        for k, v in r["cl"].items():
            if k in r["exempt"]:
                continue
            lim = MIN_CLEAR[PART_CLASS.get(k, "robot")]                  # an unknown part gets the robot's 10 mm
            if v < lim:
                bad.append(f"{r['name']}: {k} {v:.1f} mm < {lim:.0f}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("plan", "go"))
    ap.add_argument("what", choices=("home", "over-sphere", "over-rod"))
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--stop-N", type=float, default=3.0)
    a = ap.parse_args()
    import rtde_receive
    plan, recs, T_EM = load()
    r = rtde_receive.RTDEReceiveInterface(ROBOT, 125.0)
    cur_T, cur_q = C.ur_to_T(r.getActualTCPPose()), np.array(r.getActualQ())
    rows = waypoints(a.what, a.gap, cur_T, cur_q, plan, recs, T_EM, a.mode)
    print(f"{a.mode} {a.what} from flange {np.round(cur_T[:3, 3], 1)} mm (speed slider {r.getTargetSpeedFraction():.2f}):")
    print_rows(rows)
    bad = check_ok(rows)
    if bad:
        print("REFUSED, clearance below the limits:\n  " + "\n  ".join(bad))
        sys.exit(2)
    if a.mode == "plan":
        return
    import rtde_control
    ft = C.OptoForceStream(FT, rate=100, bias=False, filt=3)
    ft.configure()
    ft.start()
    c = rtde_control.RTDEControlInterface(ROBOT)
    log = []
    try:
        c.setTcp([0.0] * 6)
        for row in rows:
            time.sleep(0.6)                                     # at rest: this move's tare
            _, v0 = ft.ring.since(time.time() - 0.4)
            F0 = v0[:, :3].mean(axis=0)
            stop_N = min(a.stop_N, row["stop_N"]) if row["stop_N"] <= 3.0 else row["stop_N"]
            t0, fmax, why = time.time(), 0.0, None
            c.moveL(C.T_to_ur(row["T"]), row["v"], 0.1, True)
            time.sleep(0.05)
            while True:
                t, v = ft.ring.latest()
                if t is None or time.time() - t > 0.1:
                    why = "no HEX-E sample for 100 ms"
                else:
                    fmax = max(fmax, float(np.linalg.norm(v[:3] - F0)))
                    if fmax > stop_N:
                        why = f"HEX-E force changed {fmax:.2f} N (limit {stop_N:g})"
                if why is None and (r.isProtectiveStopped() or r.isEmergencyStopped()):
                    why = "protective or emergency stop"
                if why:
                    c.stopL(1.0)
                    break
                if c.getAsyncOperationProgress() < 0 and time.time() - t0 > 0.3:
                    break
                if time.time() - t0 > 60.0:
                    why = "move over 60 s"
                    c.stopL(1.0)
                    break
                time.sleep(0.01)
            got = C.ur_to_T(r.getActualTCPPose())
            err = C.pose_error(got, row["T"])
            _, v1 = ft.ring.since(time.time() - 0.2)
            rec = dict(time=time.strftime("%H:%M:%S"), what=a.what, name=row["name"], target_flange=row["T"].round(3).tolist(),
                       reached_flange=got.round(3).tolist(), error_mm_deg=[round(err[0], 3), round(err[1], 3)],
                       max_dF_N=round(fmax, 3), stop_N=stop_N, F0_N=F0.round(3).tolist(),
                       F_end_N=v1[:, :3].mean(axis=0).round(3).tolist() if len(v1) else None, stopped=why)
            log.append(rec)
            print(f"  {row['name']}: {'STOPPED, ' + why if why else 'reached'}; {err[0]:.2f} mm, {err[1]:.2f} deg from the "
                  f"target; HEX-E force changed at most {fmax:.2f} N")
            if why:
                break
    finally:
        try:
            c.stopScript()
            c.disconnect()
        except Exception:                                   # noqa: BLE001
            pass
        ft.stop = True
        with open(HERE / "moves.jsonl", "a") as f:
            for rec in log:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
