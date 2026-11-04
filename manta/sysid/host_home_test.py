#!/usr/bin/env python3
"""Prototype of a debounced StallGuard home, run from the CB1, to check it is
repeatable before patching the firmware.

    python3 /tmp/host_home_test.py <joint> [cycles=5] [hold_ms=50] [backoff_mm=20] [--sgt N]

Mirrors the firmware HOME (same speed 12000 sps / accel 2000, same unchecked
window = ramp + 0.8 s), with one change: instead of stopping on the first
flagged DRV_STATUS read, it stops only when the stall flag stays set for
hold_ms. Polls DRV_STATUS + position every ~2 ms (firmware: 3 ms).

Each cycle: MOVEMM out to backoff_mm, home, STOP+DIS the instant the stall is
confirmed, report where it stopped, ZERO there (as a real home would). If
homing is repeatable every cycle after the first stops at ~0 mm.

Safety: the home is a position move to MAX_OVERTRAVEL_MM past the current
zero, so even if no stall is ever confirmed the motor stops by itself after at
most that much grinding at the wall. Wall-clock cap, STOP+DIS on any exit.

Also logs every poll to /tmp/host_home_j<joint>_<ts>.csv and reports the
longest flag run that did NOT become a stall -- the data for choosing the
firmware's debounce.
"""

import csv
import signal
import sys
import time

sys.path.insert(1, "/tmp")
import gantry_rom_sweep as g  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402

GRACE_S = g.HOME_VELOCITY / g.HOME_ACCEL + 0.8  # same as firmware Stepper_Home
MAX_OVERTRAVEL_MM = 3.0
DEFAULT_IHOLD_IRUN = 0x00060100  # firmware default: IHOLD=0, IRUN=1 (~192 mA RMS), IHOLDDELAY=6
BACKOFF_CONFIRM_SAMPLES = 8  # ~0.3 s at the sweep's 25 Hz sampling


# A position move decelerates over its last v^2/2a (11.2 mm at 12000 sps / 2000
# sps^2), and StallGuard reads "stalled" at low speed. 2026-11-03: aiming the move
# at the travel limit itself made every "stall" land in that slowdown, short of
# the wall. So the move is aimed past the limit by the decel distance (it is
# still at full speed when it reaches the limit) and THIS loop enforces the
# limit with STOP+DIS. If this process dies, the move still ends by itself
# DECEL_MM + 2 mm further on.
DECEL_MM = g.HOME_VELOCITY ** 2 / (2 * g.HOME_ACCEL) / abs(g.STEPS_PER_MM)


def home(joint, hold_s, writer, cycle, overtravel_mm=None):
    """Home toward step 0. Stops on a stall flag held hold_s while at cruise speed,
    or once the count passes -overtravel_mm (the wall should have been found by then)."""
    overtravel_mm = MAX_OVERTRAVEL_MM if overtravel_mm is None else overtravel_mm
    jt = g.driver.joints[joint]
    p0 = jt.status.position / g.STEPS_PER_MM
    target = -(overtravel_mm + DECEL_MM + 2.0)
    jt.move_to_mm(target, g.HOME_VELOCITY, g.HOME_ACCEL)
    cap = GRACE_S + abs(p0 + overtravel_mm) * abs(g.STEPS_PER_MM) / g.HOME_VELOCITY + 3.0
    t0 = time.monotonic()
    run_start = None
    false_runs = []  # durations (s) of flag runs after the grace window that cleared
    while True:
        now = time.monotonic() - t0
        sg, stall = g.drv_status(joint)
        st = jt.status
        mm = st.position / g.STEPS_PER_MM
        if mm <= -overtravel_mm:
            g.disable_all()
            return "overtravel", p0, mm, now, false_runs
        # After the firmware-equivalent grace window the axis is at cruise until it
        # passes -overtravel_mm (the decel starts DECEL_MM + 2 mm beyond that).
        armed = now >= GRACE_S
        writer.writerow({"cycle": cycle, "t": f"{now:.4f}", "mm": f"{mm:.3f}", "sg": sg,
                         "stall": int(stall), "armed": int(armed)})
        if armed and stall:
            run_start = run_start if run_start is not None else now
            if now - run_start >= hold_s:
                g.disable_all()
                mm = jt.status.position / g.STEPS_PER_MM
                return "stall", p0, mm, now, false_runs
        else:
            if run_start is not None:
                false_runs.append(now - run_start)
            run_start = None
        if not st.moving:
            g.disable_all()
            return "overtravel", p0, mm, now, false_runs
        if now > cap:
            g.disable_all()
            return "timeout", p0, mm, now, false_runs


def set_run_current(joint, ihold_irun):
    """SETRUN5160: steady-state IHOLD_IRUN, also what the firmware's start-of-move
    kick restores to. (IHOLD_IRUN is write-only, so it can't be read back.)"""
    g.driver.send("SETRUN5160", f"J{joint}", f"{ihold_irun:X}")


def main():
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in ("--sgt", "--overtravel", "--irun")]
    joint = int(args[0])
    cycles = int(args[1]) if len(args) > 1 else 5
    hold_s = (float(args[2]) if len(args) > 2 else 50.0) / 1000
    backoff = float(args[3]) if len(args) > 3 else 20.0
    sgt = int(sys.argv[sys.argv.index("--sgt") + 1]) if "--sgt" in sys.argv else g.HOME_COOLCONF[joint] >> 16
    # --overtravel N: allow the wall up to N mm past the current zero (when the zero is
    # known to be short, e.g. after a false-early home). Default MAX_OVERTRAVEL_MM.
    overtravel = float(sys.argv[sys.argv.index("--overtravel") + 1]) if "--overtravel" in sys.argv else None
    # --irun N: run current for the homing move only (IRUN=1 ~192 mA default, 2 ~287 mA,
    # 3 ~383 mA RMS on 200 mA motors). Restored to the default right after each home.
    irun = int(sys.argv[sys.argv.index("--irun") + 1]) if "--irun" in sys.argv else None
    # --zero-at-limit: treat reaching -overtravel as reaching the wall (J1's StallGuard
    # can't see its wall, 2026-11-03) and ZERO there instead of stopping the test.
    zero_at_limit = "--zero-at-limit" in sys.argv
    if irun is not None and not 1 <= irun <= 3:
        sys.exit("--irun must be 1-3")
    if any(v <= 0 for v in g.FULL_EXTENSION_MM.values()):
        sys.exit("stale driver: non-positive FULL_EXTENSION_MM -- refusing to move")
    g.active_joints.append(joint)
    signal.signal(signal.SIGHUP, g.on_signal)
    signal.signal(signal.SIGTERM, g.on_signal)
    g.driver = MantaHandDriver(g.PORT)
    jt = g.driver.joints[joint]
    out = f"/tmp/host_home_j{joint}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"J{joint}: SGT {sgt}, IRUN {irun or 1} while homing, hold {hold_s * 1000:.0f} ms, from {backoff} mm, {cycles} cycles", flush=True)
    try:
        jt.set_scale(g.STEPS_PER_MM)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cycle", "t", "mm", "sg", "stall", "armed"])
            w.writeheader()
            drift = 0.0  # where each stop is relative to the zero this run started with
            for c in range(1, cycles + 1):
                jt.enable()
                # Back-off away from the wall: J1 flagged a 0.1 s stall at ~5 mm going out
                # (2026-11-02) with nothing visibly sticking, so require ~0.3 s here.
                g.STALL_CONFIRM_SAMPLES = BACKOFF_CONFIRM_SAMPLES
                g.move_and_log(joint, backoff, None, {})
                jt.write_reg5160(0x6D, sgt << 16)
                if irun is not None:
                    set_run_current(joint, (DEFAULT_IHOLD_IRUN & ~0x1F00) | (irun << 8))
                try:
                    result, p0, p1, t, false_runs = home(joint, hold_s, w, c, overtravel)
                finally:
                    if irun is not None:
                        set_run_current(joint, DEFAULT_IHOLD_IRUN)
                f.flush()
                longest = max(false_runs, default=0) * 1000
                drift += p1
                print(f"  cycle {c}: {result:10s} stopped at {p1:+.2f} mm ({drift:+.2f} vs starting zero; "
                      f"travelled {p0 - p1:.1f} mm in {t:.1f}s); "
                      f"{len(false_runs)} flag runs cleared before stopping, longest {longest:.0f} ms", flush=True)
                if result != "stall" and not (zero_at_limit and result == "overtravel"):
                    print(f"  stopping: home ended by {result}, not a confirmed stall -- zero NOT changed", flush=True)
                    break
                jt.zero()
        print(f"wrote {out}")
    finally:
        g.disable_all()
        set_run_current(joint, DEFAULT_IHOLD_IRUN)
        print(jt.status)
        g.driver.close()


if __name__ == "__main__":
    main()
