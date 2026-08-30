#!/usr/bin/env python3
"""Relieve a position-control over-clamp down to a target load band.

The exported grip pose is a POSITION the sim reaches with the object contacted
there.  On hardware the fingers meet the shaft 7-10 degrees earlier and then keep
driving to that position, so the leftover travel becomes clamping force nobody
asked for (thumb fe1 at -525 while middle sat at -210).  A shaft held that
rigidly cannot be turned by the yaw couple -- index yaw did not move at all
across ten commanded steps.

So: start each finger's mcp command AT its measured stall angle (commanding the
unreachable target is what created the force) and walk it back until |fe1 load|
enters the band.  Yaw and pip are held at the grip pose throughout.
"""
import argparse, json, sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import mh

DEPLOY = "/home/humanoid/Programs/hand/docs/experiments/20260829-real_v1_deploy/deploy"
JOINTS = ("yaw", "mcp", "pip")
SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}
FID = {"thumb": "0", "index": "1", "middle": "2"}
KEY = {"yaw": "aa", "mcp": "fe1", "pip": "fe2"}
MCP_SERVO = {"thumb": 1, "index": 4, "middle": 7}

ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="g12")
ap.add_argument("--target", type=float, default=220.0, help="|fe1 load| to relieve down to")
ap.add_argument("--target-thumb", type=float, default=None)
ap.add_argument("--target-index", type=float, default=None)
ap.add_argument("--target-middle", type=float, default=None,
                help="per-finger override.  The fingers do not have the same job: thumb and "
                     "index hold the shaft, middle has to SLIDE against it to yaw, and its "
                     "own grip is what stalls it -- the one run whose middle yaw completed "
                     "was the one where middle alone was relieved to load 0.")
ap.add_argument("--step", type=float, default=0.4, help="degrees of mcp per iteration")
ap.add_argument("--max-give", type=float, default=6.0, help="degrees of relief allowed per finger")
ap.add_argument("--preload-start", type=float, default=5.0,
                help="degrees ABOVE the stall angle to begin the walk-down from.  Relieving "
                     "all the way to first contact left middle at load 0 and the shaft fell "
                     "out during the turn; the walk stops at the first command whose load is "
                     "within --target, so a higher start lands a deliberate preload.")
ap.add_argument("--iters", type=int, default=30)
args = ap.parse_args()

plan = json.load(open(f"{DEPLOY}/{args.plan}_plan.json"))
grip = {p["name"]: p["joints"] for p in plan["poses"]}["grip"]


def read_fresh(max_age=0.25, tries=40):
    for _ in range(tries):
        t = mh.get("/state")["telemetry"]
        if not t.get("servo_polling_suspended") and (t.get("servo_age_s") or 9) < max_age:
            return t
        time.sleep(0.05)
    raise RuntimeError("no fresh, unsuspended servo sample")


def command(pose, speed=40):
    tok = mh.post("/stream/start", {"timeout_s": 2.0})["token"]
    try:
        mh.post("/stream/frame", {"token": tok, "joints": pose, "servo_speed": speed})
    finally:
        mh.post("/stream/end", {"token": tok})


t = read_fresh()
# anchor at the STALL angle, not the unreachable command
stall = {f: t["servos"][FID[f]][KEY["mcp"]] for f in SIGN}
cmd = {f: {"yaw": grip[f]["yaw"], "pip": grip[f]["pip"],
           "mcp": min(stall[f] + args.preload_start, grip[f]["mcp"])} for f in SIGN}
start = {f: cmd[f]["mcp"] for f in SIGN}
print("  stall angle: " + "  ".join(f"{f}={stall[f]:.1f}" for f in SIGN)
      + f"   starting {args.preload_start:.1f}deg above it")

for it in range(args.iters):
    command(cmd)
    time.sleep(0.25)
    t = read_fresh()
    ld = t.get("servo_load") or {}
    load = {f: abs(ld.get(str(MCP_SERVO[f])) or 0) for f in SIGN}
    got = {f: t["servos"][FID[f]][KEY["mcp"]] for f in SIGN}
    tgt = {f: (getattr(args, f"target_{f}") or args.target) for f in SIGN}
    hot = [f for f in SIGN if load[f] > tgt[f]
           and (start[f] - cmd[f]["mcp"]) < args.max_give]
    print(f"  {it:2d}  " + "  ".join(
        f"{f[:3]} cmd{cmd[f]['mcp']:+6.1f} got{got[f]:+6.1f} L{load[f]:+5.0f}" for f in SIGN)
        + ("   relieving: " + ",".join(hot) if hot else "   -> in band"))
    if not hot:
        break
    for f in hot:
        cmd[f]["mcp"] -= args.step

print("\n  final grip: " + json.dumps({f: {k: round(v, 2) for k, v in cmd[f].items()} for f in SIGN}))
json.dump(cmd, open("logs/regrip_pose.json", "w"))
print("  wrote logs/regrip_pose.json")
