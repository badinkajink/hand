#!/usr/bin/env python3
"""Seat the grip: ramp open -> grip on the loaded plan, then report the pose and loads.

This is the first step of every bench turn.  It exists as a ramp rather than a
single set-joints because the grip pose is 30+ degrees from open on all three
mcp axes and the servos will slam into the shaft if handed the endpoint.  The
seated pose it prints is what real_v1_bench_regrip.py then relieves, and what
real_v1_bench_stepped_run.py starts the turn from.
"""
import json, sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import mh

DEPLOY = "/home/humanoid/Programs/hand/docs/experiments/20260829-real_v1_deploy/deploy"
JOINTS = ("yaw", "mcp", "pip")
SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}
FID = {"thumb": "0", "index": "1", "middle": "2"}
KEY = {"yaw": "aa", "mcp": "fe1", "pip": "fe2"}
YAW_SERVO = {"thumb": 0, "index": 3, "middle": 6}

plan = json.load(open(f"{DEPLOY}/{sys.argv[1] if len(sys.argv)>1 else 'g12'}_plan.json"))
poses = {p["name"]: p["joints"] for p in plan["poses"]}
a, b = poses["open"], poses["grip"]

def read_fresh(max_age=0.25, tries=40):
    for _ in range(tries):
        t = mh.get("/state")["telemetry"]
        if not t.get("servo_polling_suspended") and (t.get("servo_age_s") or 9) < max_age:
            return t
        time.sleep(0.05)
    raise RuntimeError("no fresh, unsuspended servo sample")

def show(tag, t):
    sv, ld = t["servos"], (t.get("servo_load") or {})
    print(f"  {tag}")
    for f in SIGN:
        v = sv[FID[f]]
        print(f"    {f:7s} yaw {v[KEY['yaw']]/SIGN[f]:+7.2f}  mcp {v[KEY['mcp']]:+7.2f}"
              f"  pip {v[KEY['pip']]:+7.2f}   yaw-load {ld.get(str(YAW_SERVO[f])):+6.0f}")

show("before (open):", read_fresh())

N, DT = 25, 0.04          # ~1.0 s ramp, well under the 2 s stream timeout
tok = mh.post("/stream/start", {"timeout_s": 2.0})["token"]
try:
    for i in range(N + 1):
        u = i / N
        pose = {f: {j: a[f][j] + (b[f][j] - a[f][j]) * u for j in JOINTS} for f in SIGN}
        mh.post("/stream/frame", {"token": tok, "joints": pose, "servo_speed": 60})
        time.sleep(DT)
finally:
    mh.post("/stream/end", {"token": tok})

time.sleep(0.8)           # let it seat before reading
t = read_fresh()
show("after (grip, seated):", t)
print("\n  commanded grip: " + "  ".join(
    f"{f} y{b[f]['yaw']:+.1f} m{b[f]['mcp']:+.1f} p{b[f]['pip']:+.1f}" for f in SIGN))
