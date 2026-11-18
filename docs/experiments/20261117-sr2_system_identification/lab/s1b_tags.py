"""S1(b) of the identification queue (QUEUE.md), 2026-11-17: the DIP tags 24-26 at about nine poses per finger, spread
over the joints' ranges, each measured with both cameras against the hand's model at the station's measured joints and
mounts. The workstation fits each joint's zero offset and each finger mount's position from the rows (W2). The tools
are fingertip_tag_check.py's (20261115-hom_bench_trials/lab): fk, view and its measure; the other fingers stay straight.

    /usr/bin/python3 s1b_tags.py choose --plan PLAN [--n 9] [--fingers middle] [--both] [--avoid-map MAP:R ...]
        every pose (yaw, mcp, pip on a 5 deg grid) that turns the finger's tag within --max-view of a camera (of both
        cameras with --both), with the finger's lowest point --table-clear above the table and its tip --tip-clear from
        the others'; then --n of them per finger by farthest-point sampling in joint space, from the best view. Each
        --avoid-map object is measured by the tracker, and a pose whose straight path from zero brings the finger's tip
        pad, contact apex or tag within the object's radius R plus --avoid-clear of its centre is dropped. Writes
        OUT/poses.json; no motion
    uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python s1b_tags.py check --out OUT
        the CAD hand's finger-to-finger clearance (handgeo, as make_sysid_plan.py) at the station's mounts, at each pose
        and along the straight path from zero (1 deg steps); poses under --min-mm are dropped. Rewrites poses.json
    /usr/bin/python3 s1b_tags.py mount --finger thumb --x X [--y Y] --go
        one finger's gantry to (X, Y) in the station's palm frame (its manual mount move), inside the station's limits
        less --margin, with every finger within 3 deg of zero and the mount --min-sep from the others'
    /usr/bin/python3 s1b_tags.py run --plan PLAN --out OUT [--fingers middle] --go
        refuses if the arm, a mount or an --avoid-map object moved since choose; then per pose: the finger to the pose
        in small steps over --ramp-s, --wait-s, the tag measured over --seconds, the finger back to zero; rows to
        OUT/rows.jsonl (one per pose), a summary at the end

PLAN gives the offset between the model's mounts (meta.hom.d6_P_mm) and the station's (mounts_palm_mm), the same in
every plan; choose takes the mounts from the station, so a gantry moved off the plan's layout is modelled where it is.
On 2026-11-17 the roll press plan's (85, 170). 2026-11-18 (owner, with the arm lowered 43 mm and switched off and the
sphere moved aside: "the middle finger tag should be visible from at least one camera now, when it is curled into the
tag-visible joint config. if you yaw the thumb finger toward +x and also step the thumb +x, it should be visible by
both"): --fingers, --both, --avoid-map, the station's mounts and mount.
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


def model_mounts(plan, station_mounts):
    """Each finger's mount in the model's frame P (fingertip_tag_check.fk's): the station's mount (its palm frame) plus
    the plan's offset between meta.hom.d6_P_mm and mounts_palm_mm (thumb (-6.18, -2.40), index (6.18, 3.40), middle
    (6.18, -3.40) mm in every plan)."""
    d6 = plan["meta"]["hom"]["d6_P_mm"]
    out = {}
    for i, f in enumerate(FINGERS):
        off = (d6[2 * i] - plan["mounts_palm_mm"][f][0], d6[2 * i + 1] - plan["mounts_palm_mm"][f][1])
        out[f] = [float(station_mounts[f]["x"]) + off[0], float(station_mounts[f]["y"]) + off[1]]
    return out


def measure_objects(FT, specs, seconds=2.0):
    """[{map, r_mm, p_P, n}]: each MAP:R object's centre in P (mm, the tracker's median over seconds; None if seen in
    under three samples)."""
    import threading
    out = []
    for spec in specs:
        path, r = spec.rsplit(":", 1)
        got, lock = [], threading.Lock()

        def on(smp, got=got, lock=lock):
            m = smp.msg
            if m is not None and m.n_tags > 0 and np.all(np.isfinite(m.p_PO)):
                with lock:
                    got.append(np.array(m.p_PO, float))

        class _A:
            map = path
        arm, tracker = FT.HT.start_tracker(_A, on)
        try:
            time.sleep(1.5 + seconds)
        finally:
            tracker.stop()
            arm.stop()
        p = np.median(got, axis=0).round(1).tolist() if len(got) >= 3 else None
        out.append({"map": path, "r_mm": float(r), "p_P": p, "n": len(got)})
    return out


def path_clear(FT, f, mount, q, objs, pad_r, step=2.0):
    """The least distance (mm) from an object's surface (its centre less its radius) to the finger's tip pad (pad_r),
    contact apex or tag (12 mm, the DIP link round it) on the straight joint path from zero to q."""
    least = math.inf
    for u in np.linspace(0.0, 1.0, int(max(abs(v) for v in q) // step) + 2):
        p, _, tip, con = FT.fk(f, mount, tuple(float(u * v) for v in q))
        for o in objs:
            c = np.array(o["p_P"])
            least = min(least, np.linalg.norm(tip - c) - pad_r - o["r_mm"], np.linalg.norm(con - c) - o["r_mm"],
                        np.linalg.norm(p - c) - 12.0 - o["r_mm"])
    return float(least)


def arm_T_BP(FT, HS):
    from sr2hand.geometry.rotations import tf_from_ur
    import rtde_receive
    r = rtde_receive.RTDEReceiveInterface(FT.HT.ROBOT, 125.0)
    T = tf_from_ur(r.getActualTCPPose()) @ HS.T_EP
    mode = r.getRobotMode()
    r.disconnect()
    return T, mode


def cmd_choose(a):
    sys.path[:0] = [str(ROOT / "src"), str(LAB)]
    import fingertip_tag_check as FT
    import hom_setup as HS
    plan = HS.load_plan(a.plan)
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    M = model_mounts(plan, s["mount_positions"])
    T_BP, mode = arm_T_BP(FT, HS)
    cams = FT.cameras_P(T_BP)
    zB = lambda p: float((T_BP @ np.append(p, 1.0))[2])          # noqa: E731
    zero = (0.0, 0.0, 0.0)
    objs = measure_objects(FT, a.avoid_map) if a.avoid_map else []
    for o in objs:
        print(f"object {Path(o['map']).name}: " + (f"centre P {o['p_P']} mm ({o['n']} samples), kept {o['r_mm']:g} + "
                                                   f"{a.avoid_clear:g} mm clear" if o["p_P"] else "NOT SEEN"))
    if any(o["p_P"] is None for o in objs):
        raise SystemExit("an --avoid-map object is not seen: no poses chosen")
    fingers = [f for f in FINGERS if f in a.fingers.split(",")]
    out = {"plan": a.plan, "station_mounts": s.get("mount_positions"), "mounts_P": M, "T_BP": T_BP.tolist(),
           "arm_mode": mode, "avoid": objs, "avoid_clear_mm": a.avoid_clear, "both": a.both, "poses": {}}
    print(f"arm mode {mode}, palm origin B {np.round(T_BP[:3, 3], 1).tolist()} mm; mounts (model) "
          + ", ".join(f"{f} ({M[f][0]:.1f}, {M[f][1]:.1f})" for f in FINGERS))
    for f in fingers:
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
                    if not views or (a.both and len(views) < len(cams)):
                        continue
                    clear = path_clear(FT, f, M[f], q, objs, a.pad_r) if objs else None
                    if clear is not None and clear < a.avoid_clear:
                        continue
                    if a.both:                          # the worse camera's view angle ranks the pose
                        ang, sn = max(views)[0], "+".join(sorted(v[1] for v in views))
                    else:
                        ang, sn = min(views)
                    cand.append((ang, sn, q, round(low - a.table_z, 1), clear))
        if not cand:
            print(f"{f}: no pose turns its tag within {a.max_view:g} deg of {'both cameras' if a.both else 'a camera'}"
                  + (" clear of the objects" if objs else ""))
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
                            "low_mm": cand[k][3], "obj_clear_mm": None if cand[k][4] is None else round(cand[k][4], 1)}
                           for k in pick]
        print(f"{f}: {len(cand)} candidate poses; chosen:")
        for k in pick:
            ang, sn, q, low, clear = cand[k]
            print(f"   yaw {q[0]:+5.0f} mcp {q[1]:+5.0f} pip {q[2]:+5.0f}: camera {sn} at {ang:4.1f} deg, lowest point "
                  f"{low:.0f} mm above the table" + ("" if clear is None else f", objects {clear:.0f} mm clear"))
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
    M = d.get("mounts_P") or FT.mounts_P(plan)          # poses.json's before 2026-11-18 took the plan's mounts
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    sm = s.get("mount_positions") or {}
    if any(abs(sm[f]["x"] - d["station_mounts"][f]["x"]) > 1.0 or abs(sm[f]["y"] - d["station_mounts"][f]["y"]) > 1.0
           for f in FINGERS):
        raise SystemExit(f"the station's mounts {sm} are not the checked ones {d['station_mounts']}")
    T_BP, mode = arm_T_BP(FT, HS)
    T0 = np.array(d["T_BP"])
    dp = float(np.linalg.norm(T_BP[:3, 3] - T0[:3, 3]))
    dr = math.degrees(math.acos(float(np.clip((np.trace(T0[:3, :3].T @ T_BP[:3, :3]) - 1) / 2, -1, 1))))
    if dp > 2.0 or dr > 1.0:
        raise SystemExit(f"the palm moved {dp:.1f} mm and {dr:.2f} deg since choose: choose again")
    for o in measure_objects(FT, [f"{o['map']}:{o['r_mm']}" for o in d.get("avoid", [])]):
        o0 = next(x for x in d["avoid"] if x["map"] == o["map"])
        moved = None if o["p_P"] is None else float(np.linalg.norm(np.subtract(o["p_P"], o0["p_P"])))
        print(f"object {Path(o['map']).name}: " + ("not seen" if moved is None else f"moved {moved:.1f} mm since choose"))
        if moved is None or moved > a.moved_mm:
            raise SystemExit("the poses were chosen clear of this object where it was: choose again")
    fingers = [f for f in FINGERS if f in a.fingers.split(",")]
    d["poses"] = {f: v for f, v in d["poses"].items() if f in fingers}
    if not a.go:
        print(f"{sum(len(v) for v in d['poses'].values())} poses would run ({', '.join(d['poses'])})")
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
                        got.append((np.array(m.p_PO, float), np.array(m.q_PO, float), int(m.n_tags), "+".join(sorted(smp.dets))))

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
                   "camera_planned": row["camera"],
                   "cams": {k: sum(1 for g in got if g[3] == k) for k in sorted({g[3] for g in got})}}
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
                      f"(|{np.linalg.norm(dd):.2f}|), normal {ang:.1f} deg ({len(got)} poses; cameras "
                      + ", ".join(f"{k.replace('832412070344', '0344').replace('836612071918', '1918')} {v}"
                                  for k, v in rec["cams"].items()) + ")", flush=True)
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


def cmd_mount(a):
    sys.path[:0] = [str(ROOT / "src"), str(LAB)]
    import hom_setup as HS
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    lim = st.call("GET", "/api/v1/manual/limits")["mounts"][a.finger]
    cur = s["mount_positions"][a.finger]
    x = float(cur["x"] if a.x is None else a.x)
    y = float(cur["y"] if a.y is None else a.y)
    for k, v in (("x", x), ("y", y)):
        lo, hi = lim[k]
        if not lo + a.margin <= v <= hi - a.margin:
            raise SystemExit(f"{a.finger} {k} {v:+.1f} mm is outside the station's range [{lo:g}, {hi:g}] less "
                             f"{a.margin:g} mm")
    j = HS.sim_joints(s)
    dj = max(abs(j[f][k]) for f in j for k in j[f])
    if dj > 3.0:
        raise SystemExit(f"a finger is {dj:.1f} deg from zero: the gantry moves with every finger straight")
    sep = min(math.hypot(x - s["mount_positions"][g]["x"], y - s["mount_positions"][g]["y"])
              for g in FINGERS if g != a.finger)
    print(f"{a.finger}: ({cur['x']:+.2f}, {cur['y']:+.2f}) -> ({x:+.2f}, {y:+.2f}) mm in the station's palm frame; the "
          f"nearest other mount {sep:.0f} mm away; fingers within {dj:.1f} deg of zero")
    if sep < a.min_sep:
        raise SystemExit(f"under --min-sep {a.min_sep:g} mm")
    if not a.go:
        return 0
    st.call("POST", "/api/v1/manual/mounts", {"mounts": {a.finger: {"x": x, "y": y}}}, timeout=30.0)
    s = st.wait_idle("mount", timeout=120.0)
    print(f"mounts now {s.get('mount_positions')}, error {s.get('last_error')}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("choose")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", default=str(ROOT / "logs/20261117-sr2_s1b_tags"))
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--fingers", default="thumb,index,middle")
    p.add_argument("--both", action="store_true", help="keep only poses that both cameras see")
    p.add_argument("--avoid-map", action="append", default=[], metavar="MAP:R",
                   help="an object to keep clear of: its tag map and its radius (mm) round the map's origin")
    p.add_argument("--avoid-clear", type=float, default=20.0)
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
    p = sub.add_parser("mount")
    p.add_argument("--finger", required=True, choices=FINGERS)
    p.add_argument("--x", type=float)
    p.add_argument("--y", type=float)
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument("--min-sep", type=float, default=50.0)
    p.add_argument("--station", default="http://10.99.99.2:8765")
    p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    p.add_argument("--go", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", default=str(ROOT / "logs/20261117-sr2_s1b_tags"))
    p.add_argument("--fingers", default="thumb,index,middle")
    p.add_argument("--moved-mm", type=float, default=10.0)
    p.add_argument("--station", default="http://10.99.99.2:8765")
    p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    p.add_argument("--speed", type=int, default=40)
    p.add_argument("--ramp-s", type=float, default=2.0)
    p.add_argument("--wait-s", type=float, default=1.5)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--go", action="store_true")
    a = ap.parse_args()
    return {"choose": cmd_choose, "check": cmd_check, "mount": cmd_mount, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
