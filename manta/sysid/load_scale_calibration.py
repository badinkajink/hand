#!/usr/bin/env python3
"""Kitchen-scale calibration of present_load: index fe1 curls the fingertip into a scale in
small steps; at each step the operator types the scale reading in grams.

Interactive. Run it on the CB1 with a TTY so input() works, e.g.
    scp -o ControlPath=~/.ssh/cb1.sock load_scale_calibration.py irlab@10.99.99.2:/tmp/
    ssh -t -S ~/.ssh/cb1.sock irlab@10.99.99.2 \
        'cd ~/hand/src/morphohand/driver/manta/host && python3 /tmp/load_scale_calibration.py'

Protocol: scale tared empty, fingertip resting lightly on it at the start pose. fe1 steps
+STEP_DEG (curl, into the scale) until |load| >= LOAD_CAP or MAX_STEPS, then steps back down
the same targets (unloading, to see friction hysteresis). fe2 holds its start angle with
torque on so the finger stays rigid; its load is logged too.

Safety:
  * goal = current position before torque is enabled, for both servos (no jump)
  * every target is bounds-checked by sync_set_joints before any write
  * wall-clock backstop turns BOTH servos' torque OFF after BACKSTOP_S no matter what
    (also covers the operator walking away mid-prompt)
  * abort if |fe1 load| > LOAD_ABORT for ABORT_HOLD_S, or either temperature > TEMP_ABORT_C
  * torque OFF for both in `finally`
Writes /tmp/load_scale_cal.csv (time series) and /tmp/load_scale_cal_steps.csv (one row per step).
"""
from __future__ import annotations

import csv
import glob
import os
import math
import threading
import time

from manta_hand.servos import FINGER_JOINTS, TORQUE_OFF, ServoBus

FINGER = 1
# ~2 position counts per step (~30 load units when blocked). DIRECTION=-1 presses by OPENING fe1,
# for a finger bent out to ~50 deg over a flat scale (opening drives the tip down).
STEP_DEG = 0.6 * float(os.environ.get("DIRECTION", 1))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 12))
LOAD_CAP = 300            # stop loading once the settled |load| reaches this
LOAD_ABORT = 500          # sustained above this -> abort
ABORT_HOLD_S = 0.3
SPEED = 80
SETTLE_S = 1.5
MEASURE_S = 1.0
BACKSTOP_S = float(os.environ.get("BACKSTOP_S", 240.0))
TEMP_ABORT_C = 45.0
# NONINTERACTIVE=1: no prompts; each step simply holds STEP_HOLD_S so the operator can note
# the scale's plateau readings in order and report them afterwards.
NONINTERACTIVE = os.environ.get("NONINTERACTIVE") == "1"
STEP_HOLD_S = 4.0

fe1_id, fe1_zero, (fe1_lo, fe1_hi) = FINGER_JOINTS[FINGER]["fe1"]
fe2_id, fe2_zero, (fe2_lo, fe2_hi) = FINGER_JOINTS[FINGER]["fe2"]
# The U2D2 re-enumerates as ttyUSB1, ttyUSB2... if it is replugged; take the env var or the only one.
_ports = sorted(glob.glob("/dev/ttyUSB*"))
PORT = os.environ.get("SERVO_PORT") or (_ports[0] if len(_ports) == 1 else "/dev/ttyUSB0")
print(f"servo port {PORT}", flush=True)
bus = ServoBus(PORT)
c = bus.controller
s1, s2 = bus.servo(fe1_id), bus.servo(fe2_id)
rows: list[tuple] = []
steps: list[dict] = []
t0 = time.monotonic()
off_reason: list[str] = []
over_since: list[float | None] = [None]


def torque_off(reason: str) -> None:
    if off_reason:
        return
    off_reason.append(reason)
    for s in (s1, s2):
        try:
            s._call(c.write_torque_enable, TORQUE_OFF)
        except Exception as exc:  # noqa: BLE001
            print(f"!! torque-off write failed on {s.id} ({exc}); retrying", flush=True)
            s.disable()
    print(f"\nTORQUE OFF: {reason}", flush=True)


def rel1(rad: float) -> float:
    return math.degrees(rad) - fe1_zero


def rel2(rad: float) -> float:
    return math.degrees(rad) - fe2_zero


def sample(phase: str, target: float) -> tuple[float, float, float]:
    if off_reason:
        raise RuntimeError(f"torque already off ({off_reason[0]})")
    p1 = rel1(s1._call(c.read_present_position)[0])
    l1 = s1._call(c.read_present_load)[0]
    p2 = rel2(s2._call(c.read_present_position)[0])
    l2 = s2._call(c.read_present_load)[0]
    now = time.monotonic()
    rows.append((round(now - t0, 4), phase, round(target, 3), round(p1, 3), l1, round(p2, 3), l2))
    if abs(l1) > LOAD_ABORT:
        over_since[0] = over_since[0] or now
        if now - over_since[0] > ABORT_HOLD_S:
            raise RuntimeError(f"|fe1 load| {l1} > {LOAD_ABORT} for {ABORT_HOLD_S}s")
    else:
        over_since[0] = None
    return p1, l1, l2


def check_temp() -> None:
    for s in (s1, s2):
        temp = s._call(c.read_present_temperature)[0]
        if temp > TEMP_ABORT_C:
            raise RuntimeError(f"servo {s.id} at {temp} C > {TEMP_ABORT_C}")


def do_step(k: int, direction: str, target: float, fe2_hold: float) -> dict | None:
    bus.sync_set_joints({FINGER: {"fe1": target, "fe2": fe2_hold}}, speed=SPEED)
    t = time.monotonic()
    while time.monotonic() - t < SETTLE_S:
        sample(f"{direction}{k}_settle", target)
    meas = []
    t = time.monotonic()
    while time.monotonic() - t < MEASURE_S:
        meas.append(sample(f"{direction}{k}_measure", target))
    check_temp()
    p1 = sum(m[0] for m in meas) / len(meas)
    l1 = sum(m[1] for m in meas) / len(meas)
    l2 = sum(m[2] for m in meas) / len(meas)
    print(f"[{direction} {k:2d}] fe1 target {target:6.2f}  at {p1:6.2f} deg  "
          f"fe1 load {l1:7.1f}  fe2 load {l2:7.1f}", flush=True)
    if NONINTERACTIVE:
        t = time.monotonic()
        while time.monotonic() - t < STEP_HOLD_S - SETTLE_S - MEASURE_S:
            sample(f"{direction}{k}_hold", target)
        ans = ""
    else:
        ans = input("   scale reading in grams (Enter = skip, q = stop loading): ").strip()
    if ans.lower() == "q":
        return None
    try:
        grams = float(ans) if ans else float("nan")
    except ValueError:
        print("   (not a number; recorded as blank)")
        grams = float("nan")
    rec = {"dir": direction, "k": k, "target": round(target, 3), "fe1_pos": round(p1, 3),
           "fe1_load": round(l1, 1), "fe2_load": round(l2, 1), "grams": grams}
    steps.append(rec)
    return rec


def main() -> int:
    pos1 = s1._call(c.read_present_position)[0]
    pos2 = s2._call(c.read_present_position)[0]
    start, fe2_hold = rel1(pos1), rel2(pos2)
    top = start + MAX_STEPS * STEP_DEG
    print(f"fe1 (servo {fe1_id}) start {start:.2f} deg, max target {top:.2f} "
          f"(limits [{fe1_lo}, {fe1_hi}]); fe2 (servo {fe2_id}) held at {fe2_hold:.2f} deg")
    if not fe1_lo <= top <= fe1_hi:
        raise ValueError("max target outside software limits -- nothing moved")
    print("\nSetup: tare the scale EMPTY, then place it so the index fingertip rests lightly on it.\n"
          "Curling (fe1 +) must press the fingertip INTO the scale. Torque is still off.")
    if not NONINTERACTIVE and input("Press Enter to enable torque and start (q to quit): ").strip().lower() == "q":
        print("quit -- nothing moved")
        return 0
    pos1 = s1._call(c.read_present_position)[0]    # re-read: finger may have been bumped
    pos2 = s2._call(c.read_present_position)[0]
    start, fe2_hold = rel1(pos1), rel2(pos2)
    # fe2 often rests right at its soft limit; a one-count wobble below it would make
    # sync_set_joints refuse the hold target. Clamp (moves fe2 at most a few tenths of a degree).
    fe2_hold = min(max(fe2_hold, fe2_lo + 0.3), fe2_hi)
    if not fe1_lo <= start + MAX_STEPS * STEP_DEG <= fe1_hi:
        raise ValueError("max target outside software limits -- nothing moved")

    for s, pos in ((s1, pos1), (s2, pos2)):
        s._call(c.write_goal_position, pos)       # hold-in-place goal BEFORE torque on
        s.set_speed(SPEED)
    backstop = threading.Timer(BACKSTOP_S, torque_off, args=("wall-clock backstop",))
    backstop.daemon = True
    backstop.start()
    reached = 0
    try:
        s1.enable()
        s2.enable()
        for k in range(MAX_STEPS + 1):
            rec = do_step(k, "up", start + k * STEP_DEG, fe2_hold)
            if rec is None:
                break
            reached = k
            if abs(rec["fe1_load"]) >= LOAD_CAP:
                print(f"   load cap {LOAD_CAP} reached; unloading")
                break
        for k in range(reached - 1, -1, -1):
            if do_step(k, "down", start + k * STEP_DEG, fe2_hold) is None:
                break
    finally:
        backstop.cancel()
        torque_off("end of test")
        print(f"bus timeouts {bus.timeouts}/{bus.transactions}", flush=True)
        with open("/tmp/load_scale_cal.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "phase", "fe1_target", "fe1_pos", "fe1_load", "fe2_pos", "fe2_load"])
            w.writerows(rows)
        with open("/tmp/load_scale_cal_steps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dir", "k", "target", "fe1_pos", "fe1_load",
                                              "fe2_load", "grams"])
            w.writeheader()
            w.writerows(steps)
        print(f"wrote {len(rows)} samples, {len(steps)} steps to /tmp/load_scale_cal*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
