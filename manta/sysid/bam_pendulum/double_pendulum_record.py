#!/usr/bin/env python3
"""Double-pendulum validation recordings (BAM paper style: the tip traces a circle, a square, a square
wave and a triangle wave, at a high and a low P gain). Validation only: the logs are replayed later in
MuJoCo with the BAM models fitted on the single pendulum.

    python3 double_pendulum_record.py --check              # print joint ranges per path, no motion
    python3 double_pendulum_record.py [--paths circle square] [--gains 15 4] [--logdir data_raw/double]

Geometry: q = 0 is hanging plumb (horns centered 2026-10-29). Tip = the weight's center:
  x = L1 sin q1 + L2 sin(q1 + q2),  y = -L1 cos q1 - L2 cos(q1 + q2)      (mm, joint 1 axis at origin)

Safety, per run:
  * every IK target checked against Q1_SAFE / Q2_SAFE before torque on; goal = current position first
  * slow joint-space approach from hanging and return to hanging
  * abort -> torque OFF: tracking error > jam threshold (scaled with P gain) for JAM_S, |load| >
    LOAD_ABORT for LOAD_S, temperature > 45 C, position jump > JUMP_DEG (sensor blind zone)
  * wall-clock backstop timer; P gain restored to the default and torque off in `finally`
  * cool-down pause between runs while either servo is above COOL_C
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time

from rustypot import Scs0009PyController

PORT = "/dev/cu.usbmodem5B8E1150821"
J1, J2 = 20, 21
IDS = (J1, J2)
TORQUE_ON, TORQUE_OFF = 1, 2
P_DEFAULT = 15
D_DEFAULT = 15
# servo 21 is mounted facing the other way: +q2 (IK convention) = negative servo-21 reading.
# Logs are in the IK convention; raw = SIGN * logged. (2026-10-29 elbow-up runs used +1.)
SIGN = {J1: 1.0, J2: -1.0}
L1, L2 = 60.0, 90.0
Q1_SAFE = (-110.0, 65.0)     # stored limits -149.7/+76.2, minus gravity lag; cable tug past ~ -120
Q2_SAFE = (-120.0, 120.0)
DURATION_S = 15.0
APPROACH_S = 3.0
LOAD_ABORT, LOAD_S = 900, 0.5
TEMP_ABORT_C, COOL_C = 45, 38
JUMP_DEG = 25.0
JAM_S = 1.0

# rig metadata for the replay model (g, mm, g*mm^2 about each joint's own axis)
RIG = {
    "L1_mm": L1, "L2_mm": L2,
    "link1": {"mass_g": 11.9, "com_mm": 20.0, "izz_g_mm2": 12690.0},
    "servo2": {"mass_g": 13.0, "com_from_axis1_mm": 55.0, "note": "estimate; weigh it"},
    "link2_arm": {"part": "5-hole identification arm (id5), weight in the 90 mm hole", "mass_g": 10.8, "com_mm": 37.0, "izz_g_mm2": 24088.0},
    "link2_weight": {"mass_g": 48.08, "at_mm": 90.0, "dia_mm": 19.88, "len_mm": 19.88, "note": "steel cylinder + bolt, weighed"},
    "vin_supply": "hand 5 V supply (servos read ~4.8 V)",
}

CENTER = (35.0, -112.0)     # ~117 mm out, tilted toward the side joint 1 has room on
R_CIRCLE, HALF_SQ, X_SWEEP, Y_AMP = 25.0, 22.0, 35.0, 15.0
STEP_EASE_S = 0.3


def tip_path(name: str, t: float) -> tuple[float, float]:
    """Tip target (mm) at time t in [0, DURATION_S]."""
    cx, cy = CENTER
    u = t / DURATION_S
    if name == "circle":                       # 3 laps
        a = 2 * math.pi * 3 * u
        return cx + R_CIRCLE * math.sin(a), cy - R_CIRCLE * math.cos(a)
    if name == "square":                       # 2 laps, constant speed
        s = (u * 2 % 1) * 4
        k, f = int(s), s - int(s)
        h = HALF_SQ
        corners = [(-h, -h), (h, -h), (h, h), (-h, h), (-h, -h)]
        (x0, y0), (x1, y1) = corners[k], corners[k + 1]
        return cx + x0 + (x1 - x0) * f, cy + y0 + (y1 - y0) * f
    x = cx + X_SWEEP * math.sin(2 * math.pi * u)    # one slow side-to-side sweep
    ph = (u * 4) % 1                           # 4 vertical cycles
    if name == "square_wave":                  # steps eased over STEP_EASE_S (instant steps saturated both
        e = STEP_EASE_S * 4 / DURATION_S       # servos and dropped the bus, 2026-10-29)
        def ease(f):
            f = min(max(f / e, 0.0), 1.0)
            return 3 * f * f - 2 * f ** 3
        level = 1 - 2 * ease(ph - 0.5) if ph >= 0.5 else -1 + 2 * ease(ph) if ph < e else 1.0
        return x, cy + Y_AMP * level
    if name == "triangle_wave":
        return x, cy + Y_AMP * (1 - 4 * abs(ph - 0.5))
    raise ValueError(name)


PATHS = ("circle", "square", "square_wave", "triangle_wave")


def ik(x: float, y: float) -> tuple[float, float]:
    """Joint angles (deg) for a tip target; elbow branch q2 >= 0."""
    r2 = x * x + y * y
    c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1:
        raise ValueError(f"unreachable {x:.1f},{y:.1f}")
    q2 = math.acos(c2)
    q1 = math.atan2(x, -y) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return math.degrees(q1), math.degrees(q2)


def joint_traj(name: str, dt: float = 0.005):
    ts = [k * dt for k in range(int(DURATION_S / dt) + 1)]
    return ts, [ik(*tip_path(name, t)) for t in ts]


def check(paths) -> bool:
    ok = True
    for p in paths:
        _, qs = joint_traj(p)
        q1 = [q[0] for q in qs]
        q2 = [q[1] for q in qs]
        good = (Q1_SAFE[0] <= min(q1) and max(q1) <= Q1_SAFE[1] and Q2_SAFE[0] <= min(q2) and max(q2) <= Q2_SAFE[1])
        ok &= good
        print(f"{p:14s} q1 {min(q1):+7.1f} .. {max(q1):+6.1f}   q2 {min(q2):+7.1f} .. {max(q2):+6.1f}   "
              f"{'OK' if good else 'OUT OF SAFE RANGE'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--paths", nargs="+", default=list(PATHS), choices=PATHS)
    ap.add_argument("--gains", nargs="+", type=int, default=[P_DEFAULT, 4])
    ap.add_argument("--kd", type=int, default=0, help="firmware D gain (BAM fits use 0; servo default 15)")
    ap.add_argument("--logdir", default="data_raw/double_v2")
    a = ap.parse_args()
    if not check(a.paths):
        print("refusing to run: a path leaves the safe joint range")
        return 1
    if a.check:
        return 0
    os.makedirs(a.logdir, exist_ok=True)

    c = Scs0009PyController(serial_port=PORT, baudrate=1_000_000, timeout=0.2)
    time.sleep(0.5)
    lock = threading.Lock()
    off: list[str] = []
    retries = [0]

    def call(f, *args):
        for attempt in range(4):
            try:
                with lock:
                    return f(*args)
            except Exception as exc:  # noqa: BLE001
                retries[0] += 1
                if attempt == 3:
                    raise RuntimeError(f"{getattr(f, '__name__', f)}{args}: {exc}") from exc
                time.sleep(0.005)

    def all_off(reason: str) -> None:
        if not off:
            off.append(reason)
        for i in IDS:
            for _ in range(4):
                try:
                    with lock:
                        c.write_torque_enable(i, TORQUE_OFF)
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(0.01)
        print(f"  TORQUE OFF ({reason})", flush=True)

    def set_gain(p: int, d: int) -> None:
        for i in IDS:
            call(c.write_p_coefficient, i, p)
            call(c.write_d_coefficient, i, d)
            got = (call(c.read_p_coefficient, i)[0], call(c.read_d_coefficient, i)[0])
            if got != (p, d):
                raise RuntimeError(f"id {i}: gains wrote P {p} D {d}, read {got}")

    def read_state():
        pos = [SIGN[i] * math.degrees(call(c.read_present_position, i)[0]) for i in IDS]
        load = [SIGN[i] * call(c.read_present_load, i)[0] for i in IDS]
        spd = [SIGN[i] * math.degrees(call(c.read_present_speed, i)[0]) for i in IDS]
        return pos, load, spd

    def run(path: str, kp: int) -> None:
        off.clear()
        ts, qs = joint_traj(path)
        jam_deg = 8.0 * P_DEFAULT / kp + 4.0          # more lag allowed at low gain
        backstop_s = 2 * APPROACH_S + DURATION_S + 20
        backstop = threading.Timer(backstop_s, all_off, args=("wall-clock backstop",))
        backstop.daemon = True
        entries = []
        state = {"jam": [None, None], "over": [None, None], "last": None, "vin": None}
        t0 = time.monotonic()

        def step(phase: str, goal: list[float]) -> None:
            if off:
                raise RuntimeError(off[0])
            for i, g in zip(IDS, goal):
                call(c.write_goal_position, i, math.radians(SIGN[i] * g))
            pos, load, spd = read_state()
            now = time.monotonic()
            if len(entries) % 100 == 0:
                state["vin"] = [call(c.read_present_voltage, i)[0] / 10 for i in IDS]
            entries.append({"timestamp": round(now - t0, 5), "phase": phase, "goal_position": goal,
                            "position": pos, "speed": spd, "load": load, "input_volts": state["vin"],
                            "torque_enable": True})
            for j in range(2):
                if state["last"] and abs(pos[j] - state["last"][j]) > JUMP_DEG:
                    raise RuntimeError(f"id {IDS[j]}: position jump {state['last'][j]:+.1f} -> {pos[j]:+.1f}")
                if abs(load[j]) > LOAD_ABORT:
                    state["over"][j] = state["over"][j] or now
                    if now - state["over"][j] > LOAD_S:
                        raise RuntimeError(f"id {IDS[j]}: |load| {load[j]}")
                else:
                    state["over"][j] = None
                if abs(goal[j] - pos[j]) > jam_deg:
                    state["jam"][j] = state["jam"][j] or now
                    if now - state["jam"][j] > JAM_S:
                        raise RuntimeError(f"id {IDS[j]}: tracking error {goal[j] - pos[j]:+.1f} deg (jam?)")
                else:
                    state["jam"][j] = None
            state["last"] = pos

        def joint_ramp(qa, qb, dur, phase):
            t_s = time.monotonic()
            while (f := (time.monotonic() - t_s) / dur) < 1:
                s = 3 * f * f - 2 * f ** 3                # smooth start and stop
                step(phase, [qa[k] + (qb[k] - qa[k]) * s for k in range(2)])
            step(phase, list(qb))

        status = "ok"
        backstop.start()
        try:
            set_gain(kp, a.kd)
            pos0, _, _ = read_state()
            for i, p in zip(IDS, pos0):
                call(c.write_goal_position, i, math.radians(SIGN[i] * p))
                call(c.write_torque_enable, i, TORQUE_ON)
            joint_ramp(pos0, list(qs[0]), APPROACH_S, "approach")
            t_s = time.monotonic()
            while (t := time.monotonic() - t_s) < DURATION_S:
                k = min(int(t / 0.005), len(qs) - 1)
                step("path", list(qs[k]))
            joint_ramp(list(qs[-1]), [0.0, 0.0], APPROACH_S, "return")
            t_e = time.monotonic() + 0.5
            while time.monotonic() < t_e:
                step("return_hold", [0.0, 0.0])
        except Exception as exc:  # noqa: BLE001
            status = f"ABORT: {exc}"
            print(f"  !! {status}", flush=True)
        finally:
            backstop.cancel()
            try:
                set_gain(P_DEFAULT, D_DEFAULT)
            except Exception as exc:  # noqa: BLE001
                print(f"  !! could not restore P gain: {exc}", flush=True)
            all_off(off[0] if off else "end of run")
        path_e = [e for e in entries if e["phase"] == "path"]
        err = [max(abs(e["goal_position"][j] - e["position"][j]) for e in path_e) for j in range(2)] if path_e else [0, 0]
        rms = [math.sqrt(sum((e["goal_position"][j] - e["position"][j]) ** 2 for e in path_e) / max(1, len(path_e)))
               for j in range(2)]
        dur = entries[-1]["timestamp"] if entries else 0
        print(f"  {path} kp {kp}: {status}; {len(entries)} samples in {dur:.1f} s ({len(entries) / max(dur, 1e-9):.0f} Hz); "
              f"path tracking error rms {rms[0]:.1f}/{rms[1]:.1f} deg, max {err[0]:.1f}/{err[1]:.1f} deg (j1/j2); "
              f"max |load| {max((abs(e['load'][0]) for e in entries), default=0)}/{max((abs(e['load'][1]) for e in entries), default=0)}",
              flush=True)
        log = {"rig": RIG, "motor": "scs0009", "ids": list(IDS), "kp": kp, "kd": a.kd, "raw_sign": [SIGN[i] for i in IDS], "path": path, "status": status,
               "trajectory": path, "duration_s": DURATION_S, "tip_center_mm": CENTER, "entries": entries}
        fn = os.path.join(a.logdir, f"{time.strftime('%Y%m%d_%H%M%S')}_{path}_kp{kp}_kd{a.kd}.json")
        with open(fn, "w") as f:
            json.dump(log, f)
        print(f"  saved {fn}", flush=True)
        if status != "ok":
            raise SystemExit(f"stopping after abort in {path} kp {kp}")

    for kp in a.gains:
        for path in a.paths:
            while True:
                temps = [call(c.read_present_temperature, i)[0] for i in IDS]
                if max(temps) <= COOL_C:
                    break
                print(f"  cooling: {temps} C", flush=True)
                time.sleep(10)
            print(f"run {path}, P gain {kp} (temps {temps} C)", flush=True)
            run(path, kp)
            time.sleep(2.0)
    print(f"all runs done; bus retries {retries[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
