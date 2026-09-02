#!/usr/bin/env python3
"""Walk the deploy plan's own grip->turn_end chord forward and back, gripping the object.

    python3 scripts/servo_hysteresis_under_load.py \
        --plan docs/experiments/20260902-residual-bench/deploy/rv05_manual_b85_plan.json \
        --out docs/experiments/20260902-servo-sysid/underload.json

WHY.  `servo_hysteresis_sweep.py` found a friction cone of 0.70-1.50 deg on the free-hanging
hand, but a free finger only reaches servo load 0-150 and a grasp reaches 990.  Everything known
about the cone is therefore measured in the leftmost 15 % of the range that matters, and BAM's
load-dependent friction (arXiv:2410.08650, M3) is untested rather than refuted.  If the cone
widens with load it could be several degrees at the 585-735 a real grasp produces -- the size of
the unexplained within-design scatter on `middle_yaw` -- and MuJoCo, whose `frictionloss` is a
constant, could not express it.

This repeats the hysteresis measurement at grasp load.  The object supplies the load, so the
hand must be holding it, which also means the swept path cannot be invented: moving one joint
alone through 30 deg would leave the trajectory the clearance gate cleared and risk both the
grip and the fingers.  Instead all nine joints are interpolated along the plan's OWN chord
between `grip` and `turn_end`, stepped up and then back down, dwelling at each step so what is
read is an equilibrium.  That is the cleared path, traversed in both directions.

Drives the station over HTTP rather than the raw bus, so the control service keeps running and
keeps logging.  Load is the in-band check that the object is still held: it collapses toward the
free-hanging values if the shaft is dropped, and a run where that happens is void, not zero.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

FINGERS = ("thumb", "index", "middle")
JOINTS = ("yaw", "mcp", "pip")
# `servo_load` is keyed by SERVO id 0-8; `servos` is keyed by FINGER id 0-2 with aa/fe1/fe2
# under it.  They are different indexings of the same nine joints and mixing them silently
# yields None for every achieved angle.
SERVO_ID = {(f, j): i * 3 + k for i, f in enumerate(FINGERS) for k, j in enumerate(JOINTS)}
FINGER_ID = {f: i for i, f in enumerate(FINGERS)}
SERVO_JOINT = {"yaw": "aa", "mcp": "fe1", "pip": "fe2"}


def api(base: str, token: str, path: str, body=None, timeout=25):
    url = f"{base}/api/v1/{path}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json",
                                          "X-Manta-Token": token})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def lerp(a: dict, b: dict, u: float) -> dict:
    return {f: {j: a[f][j] + (b[f][j] - a[f][j]) * u for j in JOINTS} for f in FINGERS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="http://10.99.99.2:8765")
    ap.add_argument("--token", default="5dbd5b618d7d02af4cb571cf813cebdc")
    ap.add_argument("--steps", type=int, default=11, help="set-points from grip to turn_end")
    ap.add_argument("--dwell", type=float, default=0.7)
    ap.add_argument("--servo-speed", type=int, default=60,
                    help="slower than a run: each step must settle, not track")
    ap.add_argument("--abort-temp", type=float, default=52.0)
    a = ap.parse_args()

    poses = {p["name"]: p["joints"] for p in json.loads(a.plan.read_text())["poses"]}
    grip, end = poses["grip"], poses["turn_end"]
    us = [i / (a.steps - 1) for i in range(a.steps)]
    schedule = [("up", u) for u in us] + [("down", u) for u in reversed(us)]

    print(f"plan {a.plan.name}: {a.steps} set-points along grip->turn_end, up then down")
    records = []
    for direction, u in schedule:
        target = lerp(grip, end, u)
        api(a.base, a.token, "manual/joints",
            {"joints": target, "servo_speed": a.servo_speed})
        time.sleep(a.dwell)
        st = api(a.base, a.token, "state")
        tel = st.get("telemetry", {}) or {}
        servos = tel.get("servos") or {}
        load = tel.get("servo_load") or {}
        temp = tel.get("servo_temperature") or {}
        if temp and max(temp.values()) >= a.abort_temp:
            print(f"  ABORT: {max(temp.values())} C >= {a.abort_temp}")
            break
        rec = {"direction": direction, "u": u, "t": time.time(), "joints": {}}
        for f in FINGERS:
            for j in JOINTS:
                sid = str(SERVO_ID[(f, j)])
                got = (servos.get(str(FINGER_ID[f])) or {}).get(SERVO_JOINT[j])
                sign = -1.0 if (j == "yaw" and f in ("index", "middle")) else 1.0
                rec["joints"][f"{f}_{j}"] = {
                    "commanded": target[f][j],
                    "achieved": None if got is None else got * sign,
                    "load": load.get(sid), "temp": temp.get(sid)}
        records.append(rec)
        my = rec["joints"]["middle_yaw"]
        print(f"  {direction:<4} u={u:4.2f}  middle_yaw cmd {my['commanded']:6.2f}"
              f"  got {my['achieved'] if my['achieved'] is None else round(my['achieved'],2)!s:>7}"
              f"  load {my['load']!s:>6}  maxT {max(temp.values()) if temp else '?'}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"plan": str(a.plan), "steps": a.steps,
                                 "dwell": a.dwell, "records": records}, indent=1))
    print(f"\nwrote {a.out}  ({len(records)} set-points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
