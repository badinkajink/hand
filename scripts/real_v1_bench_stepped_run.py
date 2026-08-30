#!/usr/bin/env python3
"""Instrumented, arrival-gated replay of a plan's turn segment.

The service's own /reorient writes at 50Hz and owns the servo bus for the whole
2.78s, so the telemetry loop is suspended start to finish: the 2026-08-30 run log
has 14 'during' samples and NOT ONE of them contains a servo reading.  Commanded
vs achieved was measurable only at the two endpoints.

This replays the same grip -> turn_end interpolation the service uses
(plan.run_trajectory, linear in u = i/n) but hands the bus back after every frame
and reads before advancing, so every step has a measured position and load.  With
--gate it also WAITS for arrival instead of assuming it.

Usage:
  python3 stepped_run.py --plan g12 --steps 55 --dwell 0.12
  python3 stepped_run.py --plan g12 --gate 1.5 --gate-timeout 1.0
"""
import argparse, json, sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import mh

SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}
FID = {"thumb": "0", "index": "1", "middle": "2"}
YAW_SERVO = {"thumb": 0, "index": 3, "middle": 6}
JOINTS = ("yaw", "mcp", "pip")
KEY = {"yaw": "aa", "mcp": "fe1", "pip": "fe2"}
DEPLOY = "/home/humanoid/Programs/hand/docs/experiments/20260829-real_v1_deploy/deploy"


def lerp(a, b, u):
    return {f: {j: a[f][j] + (b[f][j] - a[f][j]) * u for j in JOINTS} for f in a}


def command(pose, speed=80):
    tok = mh.post("/stream/start", {"timeout_s": 2.0})["token"]
    try:
        mh.post("/stream/frame", {"token": tok, "joints": pose, "servo_speed": speed})
    finally:
        mh.post("/stream/end", {"token": tok})


def read_fresh(max_age=0.25, tries=40):
    for _ in range(tries):
        t = mh.get("/state")["telemetry"]
        if not t.get("servo_polling_suspended") and (t.get("servo_age_s") or 9) < max_age:
            return t
        time.sleep(0.05)
    raise RuntimeError("no fresh, unsuspended servo sample")


def sample(t, want):
    out = {}
    for f in SIGN:
        got = {j: t["servos"][FID[f]][KEY[j]] / (SIGN[f] if j == "yaw" else 1.0)
               for j in JOINTS}
        out[f] = {"cmd": {j: round(want[f][j], 3) for j in JOINTS},
                  "got": {j: round(got[j], 3) for j in JOINTS},
                  "err": {j: round(got[j] - want[f][j], 3) for j in JOINTS},
                  "yaw_load": (t.get("servo_load") or {}).get(str(YAW_SERVO[f]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="g12")
    ap.add_argument("--steps", type=int, default=55, help="same count the 50Hz service uses")
    ap.add_argument("--dwell", type=float, default=0.12, help="seconds to settle per step")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="if >0, wait for max |yaw err| below this many degrees")
    ap.add_argument("--gate-timeout", type=float, default=1.0)
    ap.add_argument("--speed", type=int, default=80)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hold", type=float, default=0.0,
                    help="after the last step, keep sampling at the final pose for this "
                         "many seconds -- the 2026-08-30 run ended the instant the "
                         "trajectory did, so nothing recorded whether the shaft was still "
                         "in the hand a second later")
    ap.add_argument("--from-pose", default="grip")
    ap.add_argument("--to-pose", default="turn_end")
    ap.add_argument("--load-delta", type=float, default=0.0,
                    help="ABORT if |yaw load| rises this far above the per-finger baseline "
                         "measured at step 0.  An absolute threshold is useless here: index's "
                         "yaw reads |load| 165-375 in poses where nothing is touching it, so "
                         "the signal is the RISE, not the level.")
    ap.add_argument("--load-settle", type=float, default=0.0,
                    help="before advancing, wait (up to --gate-timeout) for |yaw load| to fall "
                         "back within this much of baseline -- load-gated stepping, so each "
                         "step seats before the next one is commanded")
    ap.add_argument("--regrip", default=None,
                    help="JSON from regrip.py.  Its mcp values are the angles at which the "
                         "fingers ACTUALLY meet the shaft; the plan's are 7-10 degrees "
                         "deeper, and position control turns that difference into clamping "
                         "force.  The per-finger difference is subtracted from the mcp of "
                         "every pose on the path, so the turn's mcp profile is preserved "
                         "relative to real contact instead of relative to the sim's.")
    ap.add_argument("--mcp-scale", type=float, default=1.0,
                    help="scale the mcp CHANGE from the grip pose along the path.  The carry "
                         "extends the fingers 14-18 degrees because in sim the object has "
                         "rotated away underneath them; if the real shaft turns less than "
                         "that, the same extension is just an opening hand.  <1 keeps contact.")
    ap.add_argument("--stall-deg", type=float, default=0.5,
                    help="a load abort only fires if the yaw also failed to advance this far "
                         "over --stall-window steps.  Load alone is not a fault: the turn is "
                         "supposed to push the object, so load RISES when it is working.")
    ap.add_argument("--stall-window", type=int, default=5)
    ap.add_argument("--csv", action="store_true",
                    help="replay <plan>_traj.csv instead of interpolating the set-points. "
                         "These are NOT the same path: the plan JSON keeps three poses and "
                         "run_trajectory draws a straight line between them, but in the CSV "
                         "middle_yaw/middle_pip finish their roll by u=0.4 while index_yaw "
                         "tracks the chord.  Measured in the g12 scene, the chord closes "
                         "index_tip<->middle_tip to 0.0mm at step 42 of 55; the CSV path "
                         "never goes below 12.9mm.  The set-point export dropped per-joint "
                         "timing that was load-bearing.")
    args = ap.parse_args()

    plan = json.load(open(f"{DEPLOY}/{args.plan}_plan.json"))
    poses = {p["name"]: p["joints"] for p in plan["poses"]}
    mcp_off = {f: 0.0 for f in SIGN}
    if args.regrip:
        rg = json.load(open(args.regrip))
        mcp_off = {f: poses["grip"][f]["mcp"] - rg[f]["mcp"] for f in SIGN}
        print("  mcp relief applied to the whole path: "
              + "  ".join(f"{f}=-{mcp_off[f]:.1f}deg" for f in SIGN))
    a, b = poses[args.from_pose], poses[args.to_pose]
    csv_path = None
    if args.csv:
        import csv as _csv
        rows = list(_csv.DictReader(open(f"{DEPLOY}/{args.plan}_traj.csv")))
        # the turn ramp only: everything at or after the grip set-point
        t_grip = plan["poses"][0]["ramp_s"] + plan["poses"][1]["ramp_s"] + plan["poses"][1]["hold_s"]
        t_end = t_grip + plan["poses"][2]["ramp_s"]
        csv_path = [{f: {j: float(r[f"{f}_{j}_deg"]) for j in JOINTS} for f in SIGN}
                    for r in rows if t_grip - 1e-9 <= float(r["t_s"]) <= t_end + 1e-9]
        print(f"  replaying {len(csv_path)} CSV samples (t {t_grip:.2f}..{t_end:.2f}s)")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or f"logs/{stamp}-{args.plan}-stepped.jsonl"
    import os; os.makedirs(os.path.dirname(out), exist_ok=True)
    fh = open(out, "w")
    meta = {"kind": "meta", "plan": args.plan, "steps": args.steps, "csv": args.csv, "dwell_s": args.dwell,
            "gate_deg": args.gate, "gate_timeout_s": args.gate_timeout,
            "servo_speed": args.speed, "from": args.from_pose, "to": args.to_pose,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "plan_meta": plan["meta"]}
    fh.write(json.dumps(meta) + "\n")

    print(f"  {args.from_pose} -> {args.to_pose}, {args.steps} steps, dwell {args.dwell}s"
          + (f", gate {args.gate} deg" if args.gate else "") + f"  -> {out}")
    print("  step   u      thumb yaw c/g/e      index yaw c/g/e      middle yaw c/g/e     loads")
    t_start = time.monotonic()
    worst = {f: 0.0 for f in SIGN}
    base = None
    aborted = None
    yaw_hist = {f: [] for f in SIGN}

    def loads(row):
        return {f: abs(row[f]["yaw_load"] or 0) for f in SIGN}

    def over(row, margin):
        """fingers whose yaw load has risen more than `margin` above baseline"""
        if base is None:
            return []
        L = loads(row)
        return [f for f in SIGN if L[f] - base[f] > margin]

    n_steps = (len(csv_path) - 1) if csv_path else args.steps
    for i in range(0, n_steps + 1):
        u = i / n_steps
        want = csv_path[i] if csv_path else lerp(a, b, u)
        if args.mcp_scale != 1.0:
            g0 = poses["grip"]
            want = {f: {**want[f],
                        "mcp": g0[f]["mcp"] + (want[f]["mcp"] - g0[f]["mcp"]) * args.mcp_scale}
                    for f in SIGN}
        if args.regrip:
            want = {f: {**want[f], "mcp": want[f]["mcp"] - mcp_off[f]} for f in SIGN}
        command(want, args.speed)
        time.sleep(args.dwell)
        t = read_fresh()
        row = sample(t, want)
        waits = 0
        if args.gate:
            deadline = time.monotonic() + args.gate_timeout
            while time.monotonic() < deadline:
                if max(abs(row[f]["err"]["yaw"]) for f in SIGN) <= args.gate:
                    break
                time.sleep(0.06); waits += 1
                row = sample(read_fresh(), want)
        if base is None:
            base = loads(row)
            print("  yaw-load baseline: " + "  ".join(f"{f}={base[f]:.0f}" for f in SIGN))

        settle_waits = 0
        if args.load_settle:
            deadline = time.monotonic() + args.gate_timeout
            while over(row, args.load_settle) and time.monotonic() < deadline:
                time.sleep(0.06); settle_waits += 1
                row = sample(read_fresh(), want)

        for f in SIGN:
            worst[f] = max(worst[f], abs(row[f]["err"]["yaw"]))
        rec = {"kind": "step", "i": i, "u": round(u, 4),
               "t_s": round(time.monotonic() - t_start, 3),
               "gate_waits": waits, "settle_waits": settle_waits,
               "load_base": base, "joints": row}
        fh.write(json.dumps(rec) + "\n")

        for f in SIGN:
            yaw_hist[f].append(row[f]["got"]["yaw"])

        def stalled(f):
            h = yaw_hist[f][-(args.stall_window + 1):]
            return len(h) > args.stall_window and (max(h) - min(h)) < args.stall_deg

        if args.load_delta:
            hot = [f for f in over(row, args.load_delta) if stalled(f)]
            if hot:
                L = loads(row)
                aborted = {"i": i, "u": round(u, 4), "fingers": hot,
                           "load": L, "base": base}
                print(f"\n  !! STALL ABORT at step {i} (u={u:.2f}): "
                      + ", ".join(f"{f} load {L[f]:.0f} vs base {base[f]:.0f}, yaw moved "
                                  f"<{args.stall_deg}deg in {args.stall_window} steps"
                                  for f in hot))
                fh.write(json.dumps({"kind": "abort", **aborted}) + "\n")
                if i > 0:
                    back = csv_path[max(0, i - 3)] if csv_path else lerp(a, b, max(0, i - 3) / n_steps)
                    print("  backing off 3 steps")
                    command(back, args.speed); time.sleep(0.3)
                break
        if i % 5 == 0 or i == n_steps:
            print(f"  {i:4d} {u:5.2f}  " + "  ".join(
                f"{row[f]['cmd']['yaw']:+6.1f}/{row[f]['got']['yaw']:+6.1f}/"
                f"{row[f]['err']['yaw']:+5.2f}" for f in SIGN)
                + "   " + " ".join(f"{row[f]['yaw_load']:+6.0f}" for f in SIGN))
    if args.hold > 0 and not aborted:
        print(f"  holding {args.hold:.1f}s at the final pose...")
        want = csv_path[-1] if csv_path else lerp(a, b, 1.0)
        end = time.monotonic() + args.hold
        k = 0
        while time.monotonic() < end:
            command(want, args.speed)
            time.sleep(0.15)
            row = sample(read_fresh(), want)
            fh.write(json.dumps({"kind": "hold", "i": k,
                                 "t_s": round(time.monotonic() - t_start, 3),
                                 "joints": row}) + "\n")
            if k % 4 == 0:
                print(f"  hold {time.monotonic()-end+args.hold:5.1f}s  " + "  ".join(
                    f"{f[:3]} {row[f]['got']['yaw']:+6.1f} L{row[f]['yaw_load']:+5.0f}"
                    for f in SIGN))
            k += 1
    fh.write(json.dumps({"kind": "end", "aborted": aborted,
                         "worst_yaw_err": worst}) + "\n")
    fh.close()
    print(f"  done in {time.monotonic() - t_start:.1f}s   worst |yaw err|: "
          + "  ".join(f"{f}={worst[f]:.2f}" for f in worst))
    print(f"  log: {out}")


if __name__ == "__main__":
    main()
