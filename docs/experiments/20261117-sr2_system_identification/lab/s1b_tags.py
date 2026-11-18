"""S1(b) of the identification queue (QUEUE.md), 2026-11-17: the DIP tags 24-26 at about nine poses per finger, spread
over the joints' ranges, each measured with both cameras against the hand's model at the station's measured joints and
mounts. The workstation fits each joint's zero offset and each finger mount's position from the rows (W2). The tools
are fingertip_tag_check.py's (20261115-hom_bench_trials/lab): fk, view and its measure; the other fingers stay straight.

    /usr/bin/python3 s1b_tags.py choose --plan PLAN [--n 9]
        every pose (yaw, mcp, pip on a 5 deg grid) that turns the finger's tag within --max-view of a camera, with the
        finger's lowest point --table-clear above the table and its tip --tip-clear from the others'; then --n of them
        per finger by farthest-point sampling in joint space, from the best view. Writes OUT/poses.json; no motion
    uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python s1b_tags.py check --out OUT
        the CAD hand's finger-to-finger clearance (handgeo, as make_sysid_plan.py) at the station's mounts, at each pose
        and along the straight path from zero (1 deg steps); poses under --min-mm are dropped. Rewrites poses.json
    /usr/bin/python3 s1b_tags.py run --plan PLAN --out OUT --go
        per pose: the finger to the pose in small steps over --ramp-s, --wait-s, the tag measured over --seconds, the
        finger back to zero; rows to OUT/rows.jsonl (one per pose), a summary at the end

PLAN must carry the station's current mounts (meta.hom.d6_P_mm); on 2026-11-17 the roll press plan's (85, 170).
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
LAB = ROOT / "docs/experiments/20261115-hom_bench_trials/lab"
XT = ROOT / "docs/experiments/20261104-sr2_hand_crosstalk/lab"
FINGERS = ("thumb", "index", "middle")
SCALE = {"yaw": 30.0, "mcp": 65.0, "pip": 70.0}          # deg: the joint-space distance's units


def cmd_choose(a):
    sys.path[:0] = [str(ROOT / "src"), str(LAB)]
    import fingertip_tag_check as FT
    import hom_setup as HS
    from sr2hand.geometry.rotations import tf_from_ur
    plan = HS.load_plan(a.plan)
    M = FT.mounts_P(plan)
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    import rtde_receive
    r = rtde_receive.RTDEReceiveInterface(FT.HT.ROBOT, 125.0)
    T_BP = tf_from_ur(r.getActualTCPPose()) @ HS.T_EP
    r.disconnect()
    cams = FT.cameras_P(T_BP)
    zB = lambda p: float((T_BP @ np.append(p, 1.0))[2])          # noqa: E731
    zero = (0.0, 0.0, 0.0)
    out = {"plan": a.plan, "station_mounts": s.get("mount_positions"), "mounts_P": M, "T_BP": T_BP.tolist(),
           "poses": {}}
    for f in FINGERS:
        tips_o = {g: FT.fk(g, M[g], zero)[2] for g in FINGERS if g != f}
        cand = []
        for yaw in np.arange(-a.max_yaw, a.max_yaw + 1e-9, 5.0):
            for mcp in np.arange(a.mcp[0], a.mcp[1] + 1e-9, 5.0):
                for pip in np.arange(a.pip[0], a.pip[1] + 1e-9, 5.0):
                    q = (float(yaw), float(mcp), float(pip))
                    p, n, tip, con = FT.fk(f, M[f], q)
                    low = min(zB(tip) - a.pad_r, zB(con), zB(p) - 10.0)
                    if low < a.table_z + a.table_clear:
                        continue
                    if any(np.linalg.norm(tip - t) < a.tip_clear for t in tips_o.values()):
                        continue
                    views = [(FT.view(p, n, cam, T_PC), sn) for sn, (cam, T_PC) in cams.items()]
                    views = [(ang, sn) for (ang, px), sn in views if px is not None and ang <= a.max_view]
                    if views:
                        ang, sn = min(views)
                        cand.append((ang, sn, q, round(low - a.table_z, 1)))
        if not cand:
            print(f"{f}: no pose turns its tag within {a.max_view:g} deg of a camera")
            continue
        cand.sort(key=lambda c: c[0])
        X = np.array([[c[2][0] / SCALE["yaw"], c[2][1] / SCALE["mcp"], c[2][2] / SCALE["pip"]] for c in cand])
        pick = [0]
        dmin = np.linalg.norm(X - X[0], axis=1)
        while len(pick) < min(a.n, len(cand)):
            k = int(np.argmax(dmin))
            pick.append(k)
            dmin = np.minimum(dmin, np.linalg.norm(X - X[k], axis=1))
        out["poses"][f] = [{"q": list(cand[k][2]), "view_deg": round(cand[k][0], 1), "camera": cand[k][1],
                            "low_mm": cand[k][3]} for k in pick]
        print(f"{f}: {len(cand)} candidate poses; chosen:")
        for k in pick:
            ang, sn, q, low = cand[k]
            print(f"   yaw {q[0]:+5.0f} mcp {q[1]:+5.0f} pip {q[2]:+5.0f}: camera {sn} at {ang:4.1f} deg, lowest point "
                  f"{low:.0f} mm above the table")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "poses.json").write_text(json.dumps(out, indent=1))
    print(f"-> {Path(a.out) / 'poses.json'}")
    return 0


def cmd_check(a):
    sys.path.insert(0, str(XT))
    import make_sysid_plan as MS
    pth = Path(a.out) / "poses.json"
    d = json.loads(pth.read_text())
    sm = d["station_mounts"]
    MS.HOME = {f: (float(sm[f]["x"]), float(sm[f]["y"])) for f in FINGERS}
    loc = MS.local(MS.HOME["middle"])
    ck = MS.Check()
    zero = {f: {k: 0.0 for k in ("yaw", "mcp", "pip")} for f in FINGERS}
    kept = {}
    for f, rows in d["poses"].items():
        kept[f] = []
        for row in rows:
            least = math.inf
            for u in np.linspace(0.0, 1.0, int(max(abs(v) for v in row["q"])) + 2):
                q = {g: dict(zero[g]) for g in FINGERS}
                q[f] = {k: u * v for k, v in zip(("yaw", "mcp", "pip"), row["q"])}
                for g in FINGERS:
                    if g != f:
                        least = min(least, ck.ev(loc, q, f, g))
            row["clear_mm"] = round(least, 1)
            ok = least >= a.min_mm
            print(f"{f} {row['q']}: least clearance {least:.1f} mm on the path from zero{'' if ok else ' -- DROPPED'}")
            if ok:
                kept[f].append(row)
    d["poses"] = kept
    d["checked"] = {"min_mm": a.min_mm, "mounts_palm_mm": MS.HOME, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    pth.write_text(json.dumps(d, indent=1))
    print(f"kept {sum(len(v) for v in kept.values())} poses -> {pth}")
    return 0


def cmd_run(a):
    sys.path[:0] = [str(ROOT / "src"), str(LAB)]
    import threading
    import fingertip_tag_check as FT
    import hom_setup as HS
    from sr2hand.geometry.rotations import quat_to_mat
    d = json.loads((Path(a.out) / "poses.json").read_text())
    if "checked" not in d:
        raise SystemExit("poses.json has no clearance check: run check under uv first")
    plan = HS.load_plan(a.plan)
    M = FT.mounts_P(plan)
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    sm = s.get("mount_positions") or {}
    if any(abs(sm[f]["x"] - d["station_mounts"][f]["x"]) > 1.0 or abs(sm[f]["y"] - d["station_mounts"][f]["y"]) > 1.0
           for f in FINGERS):
        raise SystemExit(f"the station's mounts {sm} are not the checked ones {d['station_mounts']}")
    if not a.go:
        print(f"{sum(len(v) for v in d['poses'].values())} poses would run")
        return 0
    zero = {f: {k: 0.0 for k in ("yaw", "mcp", "pip")} for f in FINGERS}

    def ramp(j_from, j_to, seconds):
        n = max(1, int(round(10 * seconds)))
        for i in range(1, n + 1):
            u = i / n
            step = {f: {k: j_from[f][k] + (j_to[f][k] - j_from[f][k]) * u for k in j_to[f]} for f in j_to}
            st.call("POST", "/api/v1/manual/joints", {"joints": step, "servo_speed": a.speed})
            time.sleep(0.1)
        time.sleep(a.wait_s)
        return HS.sim_joints(st.state())

    rows_f = Path(a.out) / "rows.jsonl"
    j = HS.sim_joints(st.state())
    if max(abs(j[f][k]) for f in j for k in j[f]) > 3.0:
        j = ramp(j, zero, 3.0)
    summary = []
    for f, poses in d["poses"].items():
        for row in poses:
            tgt = {g: dict(zero[g]) for g in FINGERS}
            tgt[f] = {k: float(v) for k, v in zip(("yaw", "mcp", "pip"), row["q"])}
            j = ramp(HS.sim_joints(st.state()), tgt, a.ramp_s)
            q = [j[f][k] for k in ("yaw", "mcp", "pip")]
            p_pred, n_pred, _, _ = FT.fk(f, M[f], q)
            got, lock = [], threading.Lock()

            def on(smp):
                m = smp.msg
                if m is not None and m.n_tags > 0 and np.all(np.isfinite(m.p_PO)):
                    with lock:
                        got.append((np.array(m.p_PO, float), np.array(m.q_PO, float), int(m.n_tags)))

            class _A:
                map = FT.tag_map_file(FT.TAG[f])
            arm, tracker = FT.HT.start_tracker(_A, on)
            try:
                time.sleep(1.5 + a.seconds)
            finally:
                tracker.stop()
                arm.stop()
            rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "finger": f, "q_cmd": row["q"], "q_meas": q,
                   "joints_all": j, "station_mounts": sm, "mounts_P": M[f], "seen": len(got),
                   "p_pred": p_pred.tolist(), "n_pred": n_pred.tolist(), "view_deg_planned": row["view_deg"],
                   "camera_planned": row["camera"]}
            if len(got) >= 3:
                P = np.array([g[0] for g in got])
                p_m = np.median(P, axis=0)
                nz = -np.mean([quat_to_mat(g[1])[:, 2] for g in got], axis=0)
                nz /= np.linalg.norm(nz)
                dd = p_m - p_pred
                ang = math.degrees(math.acos(float(np.clip(nz @ n_pred, -1, 1))))
                rec.update({"p_meas": p_m.tolist(), "n_meas": nz.tolist(), "d_mm": dd.tolist(), "normal_deg": ang,
                            "sd_mm": P.std(axis=0).tolist(), "n_tags": int(np.median([g[2] for g in got]))})
                summary.append((f, float(np.linalg.norm(dd)), ang))
                print(f"{f:6s} q {np.round(q, 1).tolist()}: tag minus model {np.round(dd, 2).tolist()} mm "
                      f"(|{np.linalg.norm(dd):.2f}|), normal {ang:.1f} deg ({len(got)} poses)", flush=True)
            else:
                print(f"{f:6s} q {np.round(q, 1).tolist()}: tag not seen ({len(got)} poses)", flush=True)
            with rows_f.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            ramp(HS.sim_joints(st.state()), zero, a.ramp_s)
    for f in FINGERS:
        e = [x for x in summary if x[0] == f]
        if e:
            print(f"{f}: {len(e)} poses seen; |tag - model| median {np.median([x[1] for x in e]):.2f} mm, max "
                  f"{max(x[1] for x in e):.2f}; normal median {np.median([x[2] for x in e]):.1f} deg")
    print("S1B DONE")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("choose")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", default=str(ROOT / "logs/20261117-sr2_s1b_tags"))
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--station", default="http://10.99.99.2:8765")
    p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    p.add_argument("--table-z", type=float, default=3.0)
    p.add_argument("--table-clear", type=float, default=25.0)
    p.add_argument("--pad-r", type=float, default=11.0)
    p.add_argument("--tip-clear", type=float, default=35.0)
    p.add_argument("--max-yaw", type=float, default=30.0)
    p.add_argument("--mcp", type=float, nargs=2, default=(-10.0, 55.0))
    p.add_argument("--pip", type=float, nargs=2, default=(0.0, 70.0))
    p.add_argument("--max-view", type=float, default=40.0)
    p = sub.add_parser("check")
    p.add_argument("--out", default=str(ROOT / "logs/20261117-sr2_s1b_tags"))
    p.add_argument("--min-mm", type=float, default=12.0)
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", default=str(ROOT / "logs/20261117-sr2_s1b_tags"))
    p.add_argument("--station", default="http://10.99.99.2:8765")
    p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    p.add_argument("--speed", type=int, default=40)
    p.add_argument("--ramp-s", type=float, default=2.0)
    p.add_argument("--wait-s", type=float, default=1.5)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--go", action="store_true")
    a = ap.parse_args()
    return {"choose": cmd_choose, "check": cmd_check, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
