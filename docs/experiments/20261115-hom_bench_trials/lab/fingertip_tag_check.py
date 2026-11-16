"""The fingertips against the hand's model by their DIP tags (owner, 2026-11-16: "go to a morphology, actuate the servos
so their fingertip apriltags are visible, and do this check"). Tags 24 (thumb), 25 (index) and 26 (middle), 14 mm, sit
on each DIP link's dorsal face: centre at x_L = -face_sign * 8.10 mm, z_L = -36.205 mm in <finger>_pip_frame
(assets/tracking/20261115-dip_link_tags), the outward normal along -face_sign * x_L.

    poses    for each finger, the joint angles (simulator degrees) that turn its tag's outward normal toward a camera
             (the smallest view angle), with the finger's lowest point --table-clear mm above the table and the tips
             apart; the arm's current pose, the station's mounts. Prints; no motion
    move     the station's manual/joints to the chosen pose of --finger (the others keep their measured angles), in
             small steps over --ramp-s; only with --go
    measure  each visible tag's pose from both cameras at once (live.Tracker with a one-tag map, a median over
             --seconds), against the model's prediction at the station's measured joints and mounts: the centre's
             error in P (mm) and the normal's angle (deg)

    /usr/bin/python3 fingertip_tag_check.py poses --plan ../plans/20261116-film_x70_y110_twist_plan.json
    /usr/bin/python3 fingertip_tag_check.py move --plan PLAN --finger index --go
    /usr/bin/python3 fingertip_tag_check.py measure --plan PLAN --fingers index middle thumb

Log: logs/20261116-hom_setup/20261116_fingertip_tags.jsonl.
"""
import argparse
import json
import math
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]
import hom_setup as HS  # noqa: E402
import hom_trial as HT  # noqa: E402
from sr2hand.geometry.rotations import quat_to_mat, tf_from_ur, tf_inv  # noqa: E402
from sr2hand.robot.tool_chain import SR2Hand, ToolChain  # noqa: E402
from sr2hand.tracking import live  # noqa: E402

TAG = {"thumb": 24, "index": 25, "middle": 26}
TAG_C = (8.10, -36.205)                  # mm: |x_L| and z_L of the sticker's centre in <finger>_pip_frame
LOG = ROOT / "logs/20261116-hom_setup/20261116_fingertip_tags.jsonl"
H = SR2Hand.load()
FINGERS = ("thumb", "index", "middle")


def note(rec):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}) + "\n")


def mounts_P(plan):
    d6 = plan["meta"]["hom"]["d6_P_mm"]
    return {"thumb": d6[0:2], "index": d6[2:4], "middle": d6[4:6]}


def fk(finger, mount, q_deg):
    """(tag centre in P, outward normal in P, tip pad centre in P, contact apex in P) at joints q_deg (yaw, mcp, pip)."""
    c, _ = H.box(finger)
    links = H.finger_links(finger, mount[0] - c[0], mount[1] - c[1], *(math.radians(v) for v in q_deg))
    T_pip = ToolChain(finger, links).transform(upto=f"{finger}_pip_frame")
    T_tip = ToolChain(finger, links).transform(upto=f"{finger}_tip")
    T_con = ToolChain(finger, links).transform()
    s = H.fingers[finger].face_sign
    p = (T_pip @ np.array([-s * TAG_C[0], 0.0, TAG_C[1], 1.0]))[:3]
    n = T_pip[:3, :3] @ np.array([-s, 0.0, 0.0])
    return p, n, T_tip[:3, 3], T_con[:3, 3]


def cameras_P(T_BP):
    cams = live.load_cameras(HS.HT.CAL / "20261113-lens_board_k3_arm.json",
                             HS.HT.CAL / "20261113-handeye_tcp_fitted_lens_moved836.json")
    T_PB = tf_inv(T_BP)
    return {sn: (cam, T_PB @ T_BC) for sn, (cam, T_BC) in cams.items()}


def view(p, n, cam, T_PC):
    """(view angle deg, pixel (u, v) or None if outside the image) of a tag at p with normal n seen by cam."""
    c = T_PC[:3, 3]
    d = c - p
    ang = math.degrees(math.acos(float(np.clip(n @ d / np.linalg.norm(d), -1, 1))))
    T_CP = tf_inv(T_PC)
    x = T_CP[:3, :3] @ p + T_CP[:3, 3]
    if x[2] <= 0:
        return ang, None
    K = cam.K
    u, v = K[0, 0] * x[0] / x[2] + K[0, 2], K[1, 1] * x[1] / x[2] + K[1, 2]
    ok = 60 < u < cam.width - 60 and 60 < v < cam.height - 60
    return ang, (u, v, float(np.linalg.norm(d))) if ok else None


def station_state(a):
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    return st, s, HS.sim_joints(s)


def cmd_poses(a):
    plan = HS.load_plan(a.plan)
    M = mounts_P(plan)
    st, s, j = station_state(a)
    import rtde_receive
    r = rtde_receive.RTDEReceiveInterface(HT.ROBOT, 125.0)
    T_BP = tf_from_ur(r.getActualTCPPose()) @ HS.T_EP
    r.disconnect()
    cams = cameras_P(T_BP)
    print(f"station mounts {s.get('mount_positions')}; plan {plan['design']} mounts in P {M}")
    print(f"palm at {np.round(T_BP[:3, 3], 1).tolist()} mm in B; table z {a.table_z:g} mm")
    for sn, (cam, T_PC) in cams.items():
        print(f"camera {sn}: at {np.round(T_PC[:3, 3], 0).tolist()} mm in P, looking along "
              f"{np.round(T_PC[:3, 2], 2).tolist()}")
    if j:
        print("   " + HS.show_pose("measured", j))
    zB = lambda p: float((T_BP @ np.append(p, 1.0))[2])          # noqa: E731
    out = {}
    for f in a.fingers:
        best = []
        others = [g for g in FINGERS if g != f]
        tips_o = {g: fk(g, M[g], [j[g][k] for k in ("yaw", "mcp", "pip")])[2] for g in others} if j else {}
        for yaw in np.arange(-a.max_yaw, a.max_yaw + 1e-9, 5.0):
            for mcp in np.arange(a.mcp[0], a.mcp[1] + 1e-9, 5.0):
                for pip in np.arange(a.pip[0], a.pip[1] + 1e-9, 5.0):
                    q = (float(yaw), float(mcp), float(pip))
                    p, n, tip, con = fk(f, M[f], q)
                    low = min(zB(tip) - a.pad_r, zB(con), zB(p) - 10.0)
                    if low < a.table_z + a.table_clear:
                        continue
                    if any(np.linalg.norm(tip - t) < a.tip_clear for t in tips_o.values()):
                        continue
                    for sn, (cam, T_PC) in cams.items():
                        ang, px = view(p, n, cam, T_PC)
                        if px is None or ang > a.max_view:
                            continue
                        cost = ang + a.w_move * (abs(yaw) + abs(mcp) + abs(pip)) / 10.0
                        best.append((cost, ang, sn, q, p, px, low))
        best.sort(key=lambda b: b[0])
        if not best:
            print(f"{f}: no pose within the limits turns tag {TAG[f]} within {a.max_view:g} deg of a camera")
            continue
        print(f"{f} (tag {TAG[f]}): the best poses")
        for cost, ang, sn, q, p, px, low in best[:a.top]:
            print(f"   yaw {q[0]:+5.0f} mcp {q[1]:+5.0f} pip {q[2]:+5.0f}: camera {sn} at {ang:4.1f} deg, "
                  f"{px[2]:.0f} mm away, pixel ({px[0]:.0f}, {px[1]:.0f}); tag at {np.round(p, 1).tolist()} in P; "
                  f"lowest point {low - a.table_z:.0f} mm above the table")
        out[f] = {"yaw": best[0][3][0], "mcp": best[0][3][1], "pip": best[0][3][2], "camera": best[0][2],
                  "view_deg": best[0][1]}
    pth = LOG.parent / "20261116_fingertip_tag_poses.json"
    pth.write_text(json.dumps({"plan": a.plan, "poses": out, "T_BP": T_BP.tolist()}, indent=1))
    note({"cmd": "poses", "plan": a.plan, "poses": out})
    print(f"chosen poses -> {pth}")
    return 0


def cmd_move(a):
    st, s, j = station_state(a)
    if j is None:
        raise SystemExit("no joint telemetry")
    tgt = {f: dict(j[f]) for f in FINGERS}
    if a.pose == "zero":
        tgt = {f: {k: 0.0 for k in ("yaw", "mcp", "pip")} for f in FINGERS}
    elif a.pose:
        plan = HS.load_plan(a.plan)
        tgt = next({f: {k: float(v) for k, v in js.items()} for f, js in p["joints"].items()}
                   for p in plan["poses"] if p["name"] == a.pose)
    else:
        chosen = json.loads((LOG.parent / "20261116_fingertip_tag_poses.json").read_text())["poses"]
        for f in a.finger:
            tgt[f] = {k: (float(a.q[i]) if a.q else float(chosen[f][k])) for i, k in enumerate(("yaw", "mcp", "pip"))}
    print("   " + HS.show_pose("measured", j))
    print("   " + HS.show_pose("target", tgt))
    if not a.go:
        return 0
    if s.get("servo_torque") != 1:
        raise SystemExit("servo torque is not on: run hom_setup.py torque-on first")
    n = max(1, int(round(10 * a.ramp_s)))
    for i in range(1, n + 1):
        u = i / n
        step = {f: {k: j[f][k] + (tgt[f][k] - j[f][k]) * u for k in tgt[f]} for f in tgt}
        st.call("POST", "/api/v1/manual/joints", {"joints": step, "servo_speed": a.speed})
        time.sleep(0.1)
    time.sleep(a.wait_s)
    j2 = HS.sim_joints(st.state())
    print("   " + HS.show_pose("measured", j2))
    note({"cmd": "move", "target": tgt, "after": j2})
    return 0


def tag_map_file(tag):
    m = {"family": "tag36h11", "size_mm": 14.0,
         "corners_mm": {str(tag): [[7.0, -7.0, 0.0], [-7.0, -7.0, 0.0], [-7.0, 7.0, 0.0], [7.0, 7.0, 0.0]]},
         "what": f"tag {tag} alone (14 mm) in its own frame: right x, down y, centre at 0"}
    f = Path(tempfile.gettempdir()) / f"20261116_tag{tag}_map.json"
    f.write_text(json.dumps(m))
    return str(f)


def cmd_measure(a):
    plan = HS.load_plan(a.plan)
    M = mounts_P(plan)
    st, s, j = station_state(a)
    rows = {}
    for f in a.fingers:
        q = [j[f][k] for k in ("yaw", "mcp", "pip")]
        p_pred, n_pred, _, _ = fk(f, M[f], q)
        got, lock = [], threading.Lock()

        def on(smp):
            m = smp.msg
            if m is not None and m.n_tags > 0 and np.all(np.isfinite(m.p_PO)):
                with lock:
                    got.append((np.array(m.p_PO, float), np.array(m.q_PO, float), int(m.n_tags)))

        class _A:
            map = tag_map_file(TAG[f])
        arm, tracker = HT.start_tracker(_A, on)
        try:
            time.sleep(1.5 + a.seconds)
        finally:
            tracker.stop()
            arm.stop()
        if len(got) < 3:
            print(f"{f:6s} tag {TAG[f]}: {len(got)} poses in {a.seconds:g} s -- not seen (joints "
                  f"{np.round(q, 1).tolist()})")
            rows[f] = {"seen": len(got), "joints": q}
            continue
        P = np.array([g[0] for g in got])
        p_m = np.median(P, axis=0)
        z_axes = np.array([quat_to_mat(g[1])[:, 2] for g in got])
        n_m = -np.mean(z_axes, axis=0)
        n_m /= np.linalg.norm(n_m)
        ang = math.degrees(math.acos(float(np.clip(n_m @ n_pred, -1, 1))))
        d = p_m - p_pred
        print(f"{f:6s} tag {TAG[f]}: joints yaw {q[0]:+.1f} mcp {q[1]:+.1f} pip {q[2]:+.1f}; measured "
              f"{np.round(p_m, 2).tolist()} mm in P ({len(got)} poses, sd {np.round(P.std(axis=0), 2).tolist()}, "
              f"{int(np.median([g[2] for g in got]))} tag); minus the model {np.round(d, 2).tolist()} "
              f"(|{np.linalg.norm(d):.2f}| mm); normal {ang:.1f} deg from the model's", flush=True)
        rows[f] = {"seen": len(got), "joints": q, "p_meas": p_m.tolist(), "p_pred": p_pred.tolist(), "d_mm": d.tolist(),
                   "n_meas": n_m.tolist(), "n_pred": n_pred.tolist(), "normal_deg": ang,
                   "sd_mm": P.std(axis=0).tolist()}
    note({"cmd": "measure", "plan": a.plan, "mounts_P": M, "station_mounts": s.get("mount_positions"), "rows": rows})
    return 0


def cmd_wrist(a):
    """The arm's last joint by --deg (moveJ of joint 6 alone, at --speed rad/s), stopped by the HEX-E's force change
    past --stop-n or a protective stop: the hand turns about the flange's axis and the palm stays within ~10 mm, so
    the cameras see the other side of the fingers."""
    arm = HS.Arm()
    q0 = list(arm.r.getActualQ())
    q1 = list(q0)
    q1[5] += math.radians(a.deg)
    print(f"joint 6 {math.degrees(q0[5]):.1f} -> {math.degrees(q1[5]):.1f} deg at {a.speed:g} rad/s")
    if not a.go:
        arm.close()
        return 0
    try:
        arm.connect(True)
        F0 = arm.force_mean(0.4)
        arm.c.moveJ(q1, a.speed, a.acc, True)
        t0, fmax, why = time.time(), 0.0, None
        time.sleep(0.05)
        while True:
            t, F = arm.force()
            if time.time() - t0 > arm.blank_s:
                if t is None or time.time() - t > 0.1:
                    why = "no HEX-E sample for 100 ms"
                else:
                    fmax = max(fmax, float(np.linalg.norm(F - F0)))
                    if fmax > a.stop_n:
                        why = f"HEX-E changed {fmax:.2f} N (limit {a.stop_n:g})"
            if why is None and (arm.r.isProtectiveStopped() or arm.r.isEmergencyStopped()):
                why = "protective or emergency stop"
            if why is None and time.time() - t0 > abs(math.radians(a.deg)) / a.speed + 8.0:
                why = "took too long"
            if why:
                arm.c.stopJ(2.0)
                break
            if arm.c.getAsyncOperationProgress() < 0 and time.time() - t0 > 0.3:
                break
            time.sleep(0.01)
        q2 = arm.r.getActualQ()
        print(f"   {'STOPPED, ' + why if why else 'done'}: joint 6 at {math.degrees(q2[5]):.1f} deg, HEX-E change up to "
              f"{fmax:.2f} N", flush=True)
        note({"cmd": "wrist", "deg": a.deg, "q0": q0, "q_end": list(q2), "stopped": why, "max_dF_N": fmax})
    finally:
        arm.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("wrist")
    p.add_argument("--deg", type=float, required=True)
    p.add_argument("--speed", type=float, default=0.25, help="rad/s")
    p.add_argument("--acc", type=float, default=0.3, help="rad/s^2")
    p.add_argument("--stop-n", type=float, default=3.0)
    p.add_argument("--go", action="store_true")
    for name in ("poses", "move", "measure"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=name != "move")
        p.add_argument("--station", default="http://10.99.99.2:8765")
        p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
        if name == "poses":
            p.add_argument("--fingers", nargs="+", default=["index", "middle"])
            p.add_argument("--table-z", type=float, default=3.0)
            p.add_argument("--table-clear", type=float, default=25.0)
            p.add_argument("--pad-r", type=float, default=11.0)
            p.add_argument("--tip-clear", type=float, default=35.0)
            p.add_argument("--max-yaw", type=float, default=30.0)
            p.add_argument("--mcp", type=float, nargs=2, default=(-10.0, 55.0))
            p.add_argument("--pip", type=float, nargs=2, default=(0.0, 70.0))
            p.add_argument("--max-view", type=float, default=45.0)
            p.add_argument("--w-move", type=float, default=0.5)
            p.add_argument("--top", type=int, default=5)
        if name == "move":
            p.add_argument("--finger", nargs="+", default=[], choices=list(FINGERS))
            p.add_argument("--pose", help="zero, or a pose of --plan (open, grip) for all three fingers")
            p.add_argument("--q", type=float, nargs=3, help="yaw mcp pip (deg) instead of the chosen pose")
            p.add_argument("--speed", type=int, default=40)
            p.add_argument("--ramp-s", type=float, default=2.0)
            p.add_argument("--wait-s", type=float, default=1.5)
            p.add_argument("--go", action="store_true")
        if name == "measure":
            p.add_argument("--fingers", nargs="+", default=["index", "middle", "thumb"])
            p.add_argument("--seconds", type=float, default=2.0)
    a = ap.parse_args()
    return {"poses": cmd_poses, "move": cmd_move, "measure": cmd_measure, "wrist": cmd_wrist}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
