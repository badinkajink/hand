#!/usr/bin/env python3
"""Relative-StallGuard home, run from the CB1 (prototype of a firmware change).

    python3 /tmp/sg_home.py J0 [J1 ...] [--cycles N] [--backoff MM] [--overtravel MM]
                                        [--speed SPS] [--sgt N] [--ratio R] [--window S]
                                        [--zero] [--trusted] [--no-backoff] [--log CSV]
--log: append every poll (cycle, t, joint, mm, sg, phase) to CSV for later plotting.
--trusted: the current zero is known good, so a stop more than 1.5 mm from it is
reported as a FALSE stall and not zeroed.

Why: on the 24 V supply the TMC5160's SG_RESULT no longer collapses toward 0 when
an axis hits its home wall -- it steps DOWN by ~20% and stays there (2026-11-04,
all six axes, SGT 10: J0 110->87, J2 104->83, J3 94->74, J4 86->50, J5 102->77),
while free running is steady to about +-5. The firmware's single-sample flag
(sg <= threshold set by SGT) can't separate those; a drop relative to the axis's
own free-running level can.

Per axis: write COOLCONF SGT, move toward home aimed past the travel limit by the
decel distance (still at cruise when it gets there), and after the axis has been
at cruise for SETTLE_S, take the median of BASELINE_S of sg as its baseline.
Stop (STOP+DIS) once the median of the last WINDOW_S of sg is below ratio x
baseline. If the count passes -overtravel first, stop there instead ("limit" --
a wall push, the carriage is at the wall if the zero was within overtravel).

Several axes home at once if listed together (round-robin polling: one
DRV_STATUS + one STAT per axis per pass). With --zero, each axis is ZEROed where
it stopped (stall or limit). Each cycle first backs off to --backoff mm (all
listed axes, one at a time, normal move) unless --no-backoff. Reports the stop
position relative to the zero at the start of each cycle.
"""

import signal
import statistics
import sys
import time

sys.path.insert(1, "/tmp")
import gantry_rom_sweep as g  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402

SETTLE_S = 0.3      # after reaching cruise, before the baseline starts
BASELINE_S = 0.4    # baseline = median sg over this long
WINDOW_S = 0.04     # stop when the median of the last WINDOW_S of sg < ratio * baseline
# J1 and J4 have a strong once-per-leadscrew-turn load ripple (1 turn = ~0.27 s at
# 12000 sps): a 40 ms window sees one dip as the wall. Average over a full turn.
AXIS_WINDOW_S = {1: 0.27, 4: 0.27}
AXIS_RATIO = {1: 0.8, 4: 0.8}
MAX_ZERO_JUMP_MM = 1.5  # with a trusted zero, a stop further than this from it is a false stall
ACCEL = 8000


def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


TRACE = []  # (t, joint, mm, sg, phase) per poll, filled when --log is given


def home(joints, speed, sgt, ratio, overtravel, window=None):
    d = g.driver
    TRACE.clear()
    decel_mm = speed ** 2 / (2 * ACCEL) / abs(g.STEPS_PER_MM)
    ax = {}
    for j in joints:
        jt = d.joints[j]
        jt.write_reg5160(0x6D, sgt << 16)
        p0 = jt.status.position / g.STEPS_PER_MM
        jt.enable()
        ax[j] = {"p0": p0, "cruise_since": None, "prev": None, "base": [], "baseline": None,
                 "win": [], "done": None, "t_base": None}
    t0 = time.monotonic()
    for j in joints:
        d.joints[j].move_to_mm(-(overtravel + decel_mm + 2.0), speed, ACCEL)
    longest = max((a["p0"] + overtravel) * abs(g.STEPS_PER_MM) / speed for a in ax.values())
    cap = longest + speed / ACCEL + 5.0
    while any(a["done"] is None for a in ax.values()):
        if time.monotonic() - t0 > cap:
            g.disable_all()
            raise RuntimeError(f"home exceeded {cap:.1f}s cap -- all disabled")
        for j, a in ax.items():
            if a["done"]:
                continue
            now = time.monotonic()
            st = d.joints[j].status
            sg, _ = g.drv_status(j)
            mm = st.position / g.STEPS_PER_MM
            phase = "accel" if a["cruise_since"] is None else "settle" if now - a["cruise_since"] < SETTLE_S \
                else "base" if a["baseline"] is None else "watch"
            TRACE.append((now - t0, j, mm, sg, phase))
            if mm <= -overtravel:
                d.send("STOP", f"J{j}"); d.send("DIS", f"J{j}")
                a["done"] = ("limit", mm, now - t0)
                continue
            if a["prev"]:
                vel = abs(st.position - a["prev"][0]) / (now - a["prev"][1])
                if vel >= 0.85 * speed:
                    a["cruise_since"] = a["cruise_since"] or now
            a["prev"] = (st.position, now)
            if a["cruise_since"] is None or now - a["cruise_since"] < SETTLE_S:
                continue
            if a["baseline"] is None:
                a["base"].append(sg)
                if now - a["cruise_since"] >= SETTLE_S + BASELINE_S:
                    a["baseline"] = statistics.median(a["base"])
                    a["t_base"] = mm
                continue
            w = window or AXIS_WINDOW_S.get(j, WINDOW_S)
            r = ratio or AXIS_RATIO.get(j, 0.85)
            a["win"] = [(t, v) for t, v in a["win"] if now - t <= w] + [(now, sg)]
            if len(a["win"]) >= 3 and now - a["win"][0][0] >= 0.8 * w and \
                    statistics.median(v for _, v in a["win"]) < r * a["baseline"]:
                d.send("STOP", f"J{j}"); d.send("DIS", f"J{j}")
                a["done"] = ("stall", d.joints[j].status.position / g.STEPS_PER_MM, now - t0)
    return ax


def main():
    joints = [int(a.upper().lstrip("J")) for a in sys.argv[1:] if a.upper().startswith("J")]
    cycles = arg("--cycles", 1, int)
    backoff = arg("--backoff", 20.0)
    overtravel = arg("--overtravel", 3.0)
    speed = arg("--speed", 12000, int)
    sgt = arg("--sgt", 10, int)
    ratio = arg("--ratio", None)
    window = arg("--window", None)
    zero = "--zero" in sys.argv
    log = arg("--log", None, str)
    if any(v <= 0 for v in g.FULL_EXTENSION_MM.values()):
        sys.exit("stale driver: non-positive FULL_EXTENSION_MM -- refusing to move")
    g.active_joints.extend(joints)
    signal.signal(signal.SIGHUP, g.on_signal)
    signal.signal(signal.SIGTERM, g.on_signal)
    g.driver = MantaHandDriver(g.PORT)
    try:
        for j in joints:
            g.driver.joints[j].set_scale(g.STEPS_PER_MM)
        for c in range(1, cycles + 1):
            if "--no-backoff" not in sys.argv:
                for j in joints:
                    g.driver.joints[j].enable()
                    g.STALL_CONFIRM_SAMPLES = 8
                    g.move_and_log(j, backoff, None, {})
            ax = home(joints, speed, sgt, ratio, overtravel, window)
            if log:
                with open(log, "a") as f:
                    f.writelines(f"{c},{t:.4f},{j},{mm:.3f},{sg},{ph}\n" for t, j, mm, sg, ph in TRACE)
            for j, a in ax.items():
                how, mm, t = a["done"]
                false = "--trusted" in sys.argv and abs(mm) > MAX_ZERO_JUMP_MM
                if false:
                    how = "FALSE"
                elif zero:
                    g.driver.joints[j].zero()
                print(f"  cycle {c} J{j}: {how:5s} at {mm:+6.2f} mm in {t:4.1f}s | baseline sg {a['baseline']} "
                      f"(taken by {a['t_base'] if a['t_base'] is None else round(a['t_base'], 1)} mm)"
                      + ("  NOT zeroed (false stall)" if false else "  zeroed" if zero else ""), flush=True)
    finally:
        g.disable_all()
        g.driver.close()


if __name__ == "__main__":
    main()
