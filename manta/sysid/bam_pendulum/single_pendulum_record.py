#!/usr/bin/env python3
"""BAM (Better Actuator Models) single-pendulum recorder for the SCS0009 (rustypot Scs0009PyController).

One call = one setup (weight stack at one hole): it records every trajectory at every P gain, like
BAM's feetech/all_record.py, and writes one BAM-format JSON log per recording.

    python3 single_pendulum_record.py --check --stack-g 80.3 --hole 60      # plan + parameters, no motion
    python3 single_pendulum_record.py --stack-g 80.3 --hole 60 [--arm-g 11.0] [--kps 15 8 4 2] [--kd 0]
        [--trajectories sin_time_square sin_sin up_and_down lift_and_drop] [--release-mode 3]

Trajectories are BAM's own (bam/trajectory.py, 6 s each, 0 = hanging plumb). Log format follows
bam/feetech/record.py: top level mass, length, kp, vin, motor, trajectory, entries[]; each entry
position, speed, load, input_volts, temp, timestamp, goal_position, torque_enable (SI units, rad).
`mass` / `arm_mass` hold the stock-BAM equivalents from bam_rig_params.py (they reproduce the real
gravity torque and inertia in bam/testbench.py); the real values are under `real_pendulum` for the
planned Pendulum / testbench_mujoco patch.

Safety (no tracking-error guard: BAM's trajectories outrun the servo on purpose at low gain):
  * gains written to RAM only (EEPROM stays locked) and verified; restored to P=15, D=15 in `finally`
  * goal = current position before torque on; slow approach to the start and slow return to hanging
  * abort -> torque OFF: |position| > Q_ABORT_DEG, |load| > LOAD_ABORT for LOAD_S (stall),
    temperature > 45 C, position jump > JUMP_DEG between samples (sensor blind zone)
  * wall-clock backstop timer per recording; torque off in `finally`; cool-down between recordings
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import threading
import time

import numpy as np
from rustypot import Scs0009PyController

from bam_rig_params import ARMS, props

PORT = "/dev/cu.usbmodem5B8E1150821"
TORQUE_ON, TORQUE_OFF, TORQUE_FREE = 1, 2, 3     # manta_hand/servos.py
P_DEFAULT, D_DEFAULT = 15, 15
STALL_NM_6V = 0.2256
Q_ABORT_DEG = 135.0
LOAD_ABORT, LOAD_S = 950, 1.5
TEMP_ABORT_C, COOL_C = 45, 38
JUMP_DEG = 25.0
APPROACH_S, RETURN_RATE_DEG_S = 1.5, 45.0
G = 9.80665


# ---- BAM's trajectories (bam/trajectory.py, copied so this runs without the bam package) ----
def _cubic(keyframes, t):
    if t < keyframes[0][0]:
        return keyframes[0][1]
    if t > keyframes[-1][0]:
        return keyframes[-1][1]
    for (t0, x0, v0), (t1, x1, v1) in zip(keyframes, keyframes[1:]):
        if t0 <= t <= t1:
            A = [[1, t0, t0**2, t0**3], [0, 1, 2 * t0, 3 * t0**2], [1, t1, t1**2, t1**3], [0, 1, 2 * t1, 3 * t1**2]]
            w = np.linalg.solve(A, [x0, v0, x1, v1])
            return float(w[0] + w[1] * t + w[2] * t**2 + w[3] * t**3)


TRAJ = {
    "lift_and_drop": lambda t: (_cubic([[0.0, 0.0, 0.0], [2.0, -np.pi / 2, 0.0]], t), t < 2.0),
    "sin_time_square": lambda t: (float(np.sin(t**2)), True),
    "up_and_down": lambda t: (_cubic([[0.0, 0.0, 0.0], [3.0, np.pi / 2, 0.0], [6.0, 0.8 * np.pi / 2, 0.0]], t), True),
    "sin_sin": lambda t: (float(np.sin(t) * np.pi / 2 + np.sin(5.0 * t) * 0.5 * np.sin(t * 2.0)), True),
}
DURATION_S = 6.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print the plan and parameters, no motion")
    ap.add_argument("--id", type=int, default=20)
    ap.add_argument("--stack-g", type=float, required=True, help="weighed bolt + washers + nut, grams")
    ap.add_argument("--arm", default="id5", choices=list(ARMS), help="id5 = the 5-hole arm (default)")
    ap.add_argument("--hole", type=float, required=True, help="weight hole distance, mm")
    ap.add_argument("--arm-g", type=float, default=None, help="arm mass if weighed (default: computed from the STL)")
    ap.add_argument("--washer-od", type=float, default=30.0)
    ap.add_argument("--kps", type=int, nargs="+", default=[15, 8, 4, 2])
    ap.add_argument("--kd", type=int, default=0, help="D gain during recording (BAM's recorder uses 0)")
    ap.add_argument("--trajectories", nargs="+", default=list(TRAJ), choices=list(TRAJ))
    ap.add_argument("--release-mode", type=int, default=TORQUE_OFF, choices=[TORQUE_OFF, TORQUE_FREE],
                    help="torque mode when a trajectory releases the motor (lift_and_drop). 2 = motor unpowered (load reads 0); 3 still drives (load ~190), so not a release")
    ap.add_argument("--zero-deg", type=float, default=0.0,
                    help="servo reading when hanging plumb; goals are offset by it and logged positions are relative to it")
    ap.add_argument("--vin-every", type=int, default=50, help="read supply voltage every N samples")
    ap.add_argument("--logdir", default="data_raw/single")
    a = ap.parse_args()

    if a.hole not in ARMS[a.arm][3]:
        ap.error(f"arm {a.arm} has holes {ARMS[a.arm][3]}")
    first, inertia, i_stack, m_eq, a_eq = props(a.stack_g, a.hole, a.arm, a.arm_g, a.washer_od)
    peak_nm = first * 1e-6 * G
    print(f"setup: stack {a.stack_g} g at {a.hole:.0f} mm, arm {a.arm} {a.arm_g or ARMS[a.arm][0]} g -> peak gravity "
          f"{peak_nm * 1000:.0f} mN·m = {peak_nm / (STALL_NM_6V * 4.8 / 6) * 100:.0f}% of stall at 4.8 V")
    print(f"       BAM mass {m_eq / 1000:.4f} kg, arm_mass {a_eq / 1000:.4f} kg, length {a.hole / 1000:.3f} m; "
          f"real first moment {first * 1e-6:.6f} kg·m, inertia {inertia * 1e-9:.4e} kg·m²")
    print(f"       {len(a.kps)} gains {a.kps} (D={a.kd}) x {len(a.trajectories)} trajectories = "
          f"{len(a.kps) * len(a.trajectories)} recordings, ~{len(a.kps) * len(a.trajectories) * 12 / 60:.0f} min")
    for name in a.trajectories:
        qs = [math.degrees(TRAJ[name](k * 0.005)[0]) for k in range(int(DURATION_S / 0.005) + 1)]
        print(f"       {name:16s} goal {min(qs):+6.1f} .. {max(qs):+6.1f} deg")
    if a.check:
        return 0
    os.makedirs(a.logdir, exist_ok=True)

    c = Scs0009PyController(serial_port=PORT, baudrate=1_000_000, timeout=0.2)
    time.sleep(0.5)
    lock = threading.Lock()
    off: list[str] = []
    retries = [0]
    i = a.id
    Z = math.radians(a.zero_deg)
    _rd = c.read_present_position
    _wr = c.write_goal_position

    def read_pos(idx):                       # positions relative to hanging plumb
        return [v - Z for v in _rd(idx)]

    def write_goal(idx, q):
        return _wr(idx, q + Z)

    read_pos.__name__, write_goal.__name__ = "read_present_position", "write_goal_position"
    c_pos, c_goal = read_pos, write_goal

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

    def torque_off(reason: str) -> None:
        if not off:
            off.append(reason)
        for _ in range(4):
            try:
                with lock:
                    c.write_torque_enable(i, TORQUE_OFF)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.01)
        print(f"  TORQUE OFF ({reason})", flush=True)

    def set_gains(p: int, d: int) -> None:
        call(c.write_p_coefficient, i, p)
        call(c.write_d_coefficient, i, d)
        got = (call(c.read_p_coefficient, i)[0], call(c.read_d_coefficient, i)[0])
        if got != (p, d):
            raise RuntimeError(f"gains wrote P={p} D={d}, read {got}")

    def record(name: str, kp: int) -> str:
        off.clear()
        backstop = threading.Timer(DURATION_S + 2 * APPROACH_S + 200 / RETURN_RATE_DEG_S + 15,
                                   torque_off, args=("wall-clock backstop",))
        backstop.daemon = True
        entries: list[dict] = []
        st = {"over": None, "last": None, "vin": 4.8, "temp": 0, "n": 0}

        def sample(goal: float, enabled: bool, t0: float) -> dict:
            if off:
                raise RuntimeError(off[0])
            ta = time.monotonic() - t0
            pos = call(c_pos, i)[0]
            spd = call(c.read_present_speed, i)[0]
            load = call(c.read_present_load, i)[0]
            if st["n"] % a.vin_every == 0:
                st["vin"] = call(c.read_present_voltage, i)[0] / 10
                st["temp"] = call(c.read_present_temperature, i)[0]
            st["n"] += 1
            tb = time.monotonic() - t0
            q = math.degrees(pos)
            if abs(q) > Q_ABORT_DEG:
                raise RuntimeError(f"position {q:+.1f} deg beyond ±{Q_ABORT_DEG}")
            if st["last"] is not None and abs(q - st["last"]) > JUMP_DEG:
                raise RuntimeError(f"position jump {st['last']:+.1f} -> {q:+.1f} (blind zone?)")
            st["last"] = q
            if st["temp"] > TEMP_ABORT_C:
                raise RuntimeError(f"temperature {st['temp']} C")
            now = time.monotonic()
            if abs(load) > LOAD_ABORT and enabled:
                st["over"] = st["over"] or now
                if now - st["over"] > LOAD_S:
                    raise RuntimeError(f"|load| {load} for {LOAD_S} s (stall)")
            else:
                st["over"] = None
            return {"position": float(pos), "speed": float(spd), "load": float(load),
                    "input_volts": float(st["vin"]), "temp": float(st["temp"]),
                    "timestamp": (ta + tb) / 2, "goal_position": float(goal), "torque_enable": bool(enabled)}

        status = "ok"
        backstop.start()
        try:
            set_gains(kp, a.kd)
            p0 = call(c_pos, i)[0]
            call(c_goal, i, p0)
            call(c.write_torque_enable, i, TORQUE_ON)
            # approach the trajectory's start (BAM holds trajectory(0) for 1 s before recording)
            g0 = TRAJ[name](0.0)[0]
            ts = time.monotonic()
            while (f := (time.monotonic() - ts) / APPROACH_S) < 1:
                s = 3 * f * f - 2 * f**3
                call(c_goal, i, p0 + (g0 - p0) * s)
                sample(p0 + (g0 - p0) * s, True, ts)
            call(c_goal, i, g0)
            time.sleep(0.5)
            # recording
            t0 = time.monotonic()
            enabled = True
            while (t := time.monotonic() - t0) < DURATION_S:
                goal, en = TRAJ[name](t)
                if en != enabled:
                    if en:
                        call(c_goal, i, call(c_pos, i)[0])
                    call(c.write_torque_enable, i, TORQUE_ON if en else a.release_mode)
                    enabled = en
                if enabled:
                    call(c_goal, i, goal)
                entries.append(sample(goal, enabled, t0))
            # slow return to hanging (torque back on from where it is)
            p = call(c_pos, i)[0]
            call(c_goal, i, p)
            call(c.write_torque_enable, i, TORQUE_ON)
            step = math.radians(RETURN_RATE_DEG_S) * 0.02
            while abs(p) > 1e-3:
                p = max(0.0, p - step) if p > 0 else min(0.0, p + step)
                call(c_goal, i, p)
                time.sleep(0.02)
                if off:
                    raise RuntimeError(off[0])
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001
            status = f"ABORT: {exc}"
            print(f"  !! {status}", flush=True)
        finally:
            backstop.cancel()
            if status != "ok":
                torque_off(f"abort: {status}")       # first, before any diagnosis
                try:                                 # did the servo reset? (RAM D gain back to its EEPROM 15)
                    d_now = call(c.read_d_coefficient, i)[0]
                    te = call(c.read_torque_enable, i)[0]
                    reset = d_now != a.kd and a.kd != D_DEFAULT
                    status += f" | after: D={d_now} torque_enable={te} -> " + (
                        "SERVO RESET (brown-out?)" if reset else "servo did not reset (bus/adapter dropout)")
                    print(f"  diagnosis: {status.split(' | ')[1]}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  diagnosis failed: {exc}", flush=True)
            try:
                set_gains(P_DEFAULT, D_DEFAULT)
            except Exception as exc:  # noqa: BLE001
                print(f"  !! could not restore gains: {exc}", flush=True)
            torque_off(off[0] if off else "end of recording")

        dur = entries[-1]["timestamp"] if entries else 0.0
        err = [abs(e["goal_position"] - e["position"]) for e in entries if e["torque_enable"]]
        print(f"  {name} kp {kp}: {status}; {len(entries)} samples ({len(entries) / max(dur, 1e-9):.0f} Hz); "
              f"tracking error rms {math.degrees(math.sqrt(sum(x * x for x in err) / max(1, len(err)))):.1f} deg; "
              f"max |load| {max((abs(e['load']) for e in entries), default=0):.0f}; temp {st['temp']} C", flush=True)
        log = {
            "mass": m_eq / 1000, "arm_mass": a_eq / 1000, "length": a.hole / 1000, "kp": kp, "kd": a.kd,
            "vin": float(np.mean([e["input_volts"] for e in entries])) if entries else 4.8,
            "motor": "scs0009", "trajectory": name, "id": i, "status": status,
            "release_mode": a.release_mode, "zero_deg": a.zero_deg,
            "real_pendulum": {"stack_g": a.stack_g, "hole_mm": a.hole, "arm": a.arm,
                              "arm_g": a.arm_g or ARMS[a.arm][0], "washer_od_mm": a.washer_od,
                              "first_moment_kg_m": first * 1e-6, "inertia_kg_m2": inertia * 1e-9,
                              "stack_spin_inertia_kg_m2": i_stack * 1e-9},
            "entries": entries,
        }
        date = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")
        fn = os.path.join(a.logdir, f"{date}_m{a.stack_g:.0f}g_l{a.hole:.0f}_kp{kp}_{name}.json")
        with open(fn, "w") as fh:
            json.dump(log, fh)
        if status != "ok":
            raise SystemExit(f"stopping after abort ({fn})")
        return fn

    for kp in a.kps:
        for name in a.trajectories:
            while (temp := call(c.read_present_temperature, i)[0]) > COOL_C:
                print(f"  cooling: {temp} C", flush=True)
                time.sleep(10)
            print(f"kp {kp}, {name} (temp {temp} C)", flush=True)
            record(name, kp)
            time.sleep(1.5)
    print(f"setup done; bus retries {retries[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
