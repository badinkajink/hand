"""S1(a) of the identification queue (QUEUE.md), 2026-11-17: each joint's reachable range inside the station's limits.
One joint at a time, the other eight straight; the target steps from zero toward each end in --step deg steps at
--speed, --dwell s a step; the reach is the last measured angle before the joint stops following (the target leads the
reading by more than --lag deg (2.5; the yaws sag 1-1.5 deg under gravity at 45 deg) on two steps running) or before its load passes --load-max. Then the joint goes back to
zero. The ends: servo_limits' safe ranges (5 deg inside the potentiometers' ends), and tonight's caps (owner's rules:
never curl an MCP past about 60 deg, where its potentiometer jumps; large yaws and full curls pulled a fingertip cable on
2026-11-04): yaw within +-45 deg, MCP at most 58 deg, PIP at most 80 deg.

    uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python s1a_ranges.py check
        the CAD hand's finger-to-finger clearance at the station's mounts at every end pose and on the way (1 deg
        steps), as make_sysid_plan.py; an end under --min-mm is dropped. Writes OUT/ends.json
    /usr/bin/python3 s1a_ranges.py run --go
        the sweeps of OUT/ends.json; rows to OUT/rows.jsonl (one per step), OUT/summary.json at the end

Per joint and end the summary gives the station's calibrated limit, the safe limit, tonight's cap, the target reached,
the last measured angle and why the sweep stopped.
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
JOINTS = ("yaw", "mcp", "pip")
CAP = {"yaw": (-45.0, 45.0), "mcp": (-90.0, 58.0), "pip": (-90.0, 80.0)}


def ends_list():
    sys.path[:0] = [str(ROOT / "src")]
    from sr2hand.control import servo_limits as SL
    out = []
    for f in FINGERS:
        for k in JOINTS:
            lo_s, hi_s = SL.safe_range_sim(f, k)
            lo, hi = max(lo_s, CAP[k][0]), min(hi_s, CAP[k][1])
            for end, val in (("low", lo), ("high", hi)):
                out.append({"finger": f, "joint": k, "end": end, "target_deg": round(val, 2),
                            "safe_deg": round(lo_s if end == "low" else hi_s, 2),
                            "cap_deg": CAP[k][0] if end == "low" else CAP[k][1]})
    return out


def cmd_check(a):
    sys.path.insert(0, str(XT))
    import make_sysid_plan as MS
    sys.path[:0] = [str(LAB), str(ROOT / "src")]
    import urllib.request
    tok = Path(a.token_file).read_text().strip()
    req = urllib.request.Request(a.station + "/api/v1/state", headers={"Authorization": f"Bearer {tok}"})
    try:
        sm = json.loads(urllib.request.urlopen(req, timeout=10).read())["mount_positions"]
    except Exception:
        sm = json.loads(Path(a.mounts_json).read_text()) if a.mounts_json else None
    if not sm:
        raise SystemExit("no mounts: pass --mounts-json")
    MS.HOME = {f: (float(sm[f]["x"]), float(sm[f]["y"])) for f in FINGERS}
    loc = MS.local(MS.HOME["middle"])
    ck = MS.Check()
    zero = {f: {k: 0.0 for k in JOINTS} for f in FINGERS}
    kept = []
    for e in ends_list():
        least = math.inf
        for u in np.linspace(0.0, 1.0, int(abs(e["target_deg"])) + 2):
            q = {g: dict(zero[g]) for g in FINGERS}
            q[e["finger"]][e["joint"]] = u * e["target_deg"]
            for g in FINGERS:
                if g != e["finger"]:
                    least = min(least, ck.ev(loc, q, e["finger"], g))
        e["clear_mm"] = round(least, 1)
        ok = least >= a.min_mm
        print(f"{e['finger']:6s} {e['joint']} {e['end']:4s} to {e['target_deg']:+7.2f} (safe {e['safe_deg']:+7.2f}): "
              f"least {least:5.1f} mm{'' if ok else ' -- DROPPED'}")
        if ok:
            kept.append(e)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "ends.json").write_text(json.dumps({"mounts": sm, "min_mm": a.min_mm, "ends": kept}, indent=1))
    print(f"kept {len(kept)} ends -> {Path(a.out) / 'ends.json'}")
    return 0


def cmd_run(a):
    sys.path[:0] = [str(LAB), str(ROOT / "src")]
    import hom_setup as HS
    from sr2hand.control import servo_limits as SL
    P, SV = SL._driver()
    d = json.loads((Path(a.out) / "ends.json").read_text())
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    sm = s.get("mount_positions") or {}
    if any(abs(sm[f]["x"] - d["mounts"][f]["x"]) > 1.0 or abs(sm[f]["y"] - d["mounts"][f]["y"]) > 1.0 for f in FINGERS):
        raise SystemExit(f"the station's mounts {sm} are not the checked ones {d['mounts']}")
    print(f"{len(d['ends'])} sweeps, {a.step:g} deg steps at speed {a.speed}, {a.dwell:g} s a step")
    if not a.go:
        return 0
    if s.get("servo_torque") != 1:
        raise SystemExit("servo torque is not on")
    zero = {f: {k: 0.0 for k in JOINTS} for f in FINGERS}
    rows_f = Path(a.out) / "rows.jsonl"

    def send(q, speed):
        st.call("POST", "/api/v1/manual/joints", {"joints": q, "servo_speed": speed})

    def read():
        s = st.state()
        return HS.sim_joints(s), (s.get("telemetry") or {}).get("servo_load") or {}

    def tmax():
        dg = st.call("POST", "/api/v1/servos/diagnostics")
        t = [float(v) for v in ((dg or {}).get("temperature") or {}).values()] if isinstance(dg, dict) else []
        return max(t) if t else None

    j, _ = read()
    if max(abs(j[f][k]) for f in j for k in j[f]) > 3.0:
        raise SystemExit("the fingers are not straight: zero them first (hom_setup.py zero)")
    summary = []
    for e in d["ends"]:
        f, k = e["finger"], e["joint"]
        t = tmax()
        if t is not None and t >= 50.0:
            print(f"   {t:.0f} C: waiting for every servo under 40 C", flush=True)
            for _ in range(45):
                time.sleep(60)
                t = tmax()
                if t is not None and t < 40.0:
                    break
            else:
                raise SystemExit("the servos did not cool under 40 C in 45 min")
        sid = SV.FINGER_JOINTS[P.FINGER_ID[f]][P.SIM_JOINT_TO_SERVO[k]][0]
        sgn = 1.0 if e["target_deg"] >= 0 else -1.0
        n = int(abs(e["target_deg"]) / a.step)
        targets = [sgn * a.step * i for i in range(1, n + 1)] + [e["target_deg"]]
        lagging, why, last = 0, "reached the end", None
        for tg in targets:
            q = {g: dict(zero[g]) for g in FINGERS}
            q[f][k] = tg
            send(q, a.speed)
            time.sleep(a.dwell)
            js, loads = read()
            meas = js[f][k]
            load = float(loads.get(str(sid), 0.0) or 0.0)
            with rows_f.open("a") as fh:
                fh.write(json.dumps({"t": time.time(), "finger": f, "joint": k, "end": e["end"], "target": tg,
                                     "measured": meas, "load": load, "sid": sid}) + "\n")
            if sgn * (tg - meas) > a.lag:
                lagging += 1
            else:
                lagging, last = 0, (tg, meas)
            if lagging >= 2:
                why = f"stopped following ({sgn * (tg - meas):.1f} deg behind)"
                break
            if abs(load) > a.load_max:
                why = f"load {load:.0f} over {a.load_max:g}"
                last = (tg, meas)
                break
        # back to zero in 1 deg steps
        js, _ = read()
        cur = js[f][k]
        for v in np.linspace(cur, 0.0, int(abs(cur)) + 2)[1:]:
            q = {g: dict(zero[g]) for g in FINGERS}
            q[f][k] = float(v)
            send(q, a.speed)
            time.sleep(0.05)
        time.sleep(1.0)
        rec = {**e, "reached_target_deg": last[0] if last else None, "reached_measured_deg": last[1] if last else None,
               "stopped": why}
        summary.append(rec)
        print(f"{f:6s} {k} {e['end']:4s}: {why}; last following at target {last[0] if last else float('nan'):+.1f}, "
              f"measured {last[1] if last else float('nan'):+.1f} (safe {e['safe_deg']:+.1f}, cap {e['cap_deg']:+.1f})",
              flush=True)
    (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=1))
    print("S1A DONE")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=str(ROOT / "logs/20261117-sr2_s1a_ranges"))
        p.add_argument("--station", default="http://10.99.99.2:8765")
        p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
    p = sub.choices["check"]
    p.add_argument("--min-mm", type=float, default=12.0)
    p.add_argument("--mounts-json", default=None)
    p = sub.choices["run"]
    p.add_argument("--speed", type=int, default=20)
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--dwell", type=float, default=0.35)
    p.add_argument("--lag", type=float, default=2.5)
    p.add_argument("--load-max", type=float, default=600.0)
    p.add_argument("--go", action="store_true")
    a = ap.parse_args()
    return {"check": cmd_check, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
