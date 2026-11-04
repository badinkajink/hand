#!/usr/bin/env python3
"""Walk one finger's xy stepper gantry through its full range and log a
binding profile along the way.

For each of the finger's two axes, sweeps out and back over its measured
travel (minus END_MARGIN_MM at each hardstop) with the other axis parked
near, mid and far. Logs TMC5160 StallGuard2 sg_result (DRV_STATUS bits 9:0)
at ~25 Hz: near-max = running freely, lower = more load, 0 + stall flag =
stalled. A dip at the same position across passes is a binding spot.

After the sweeps, each axis is re-homed and its step count at the hardstop
compared with the original zero -- a non-zero offset means steps were lost
(the motor skipped somewhere, usually at a tight spot).

Safety: aborts (STOP + DIS) on the stall flag during cruise, on a per-move
wall-clock cap, on SIGHUP/SIGTERM (SSH dropped), and always disables both
axes on exit. The axis not moving is disabled while the other sweeps
(leadscrews hold position unpowered).

Run on the CB1 as a file, one finger per run:
    python3 /tmp/gantry_rom_sweep.py <finger 0|1|2> [--quick] [--joint N]
--quick parks the other axis at mid only (1 pass each instead of 3).
--joint N sweeps only that joint of the finger (both axes are still homed).
Writes /tmp/gantry_rom_f<finger>_<timestamp>.csv.
"""

from __future__ import annotations

import csv
import signal
import sys
import time

import manta_hand
from manta_hand import MantaHandDriver
from manta_hand.kinematics import (
    FULL_EXTENSION_MM,
    HOME_ACCEL,
    HOME_COOLCONF,
    HOME_DIRECTION,
    HOME_VELOCITY,
    STEPPER_JOINTS,
    STEPS_PER_MM,
    _home_one_axis,
    _home_timeout_ms,
)

PORT = "/dev/ttyACM0"
SWEEP_VELOCITY = 12000  # sps, same as homing: StallGuard is validated at this speed
SWEEP_ACCEL = 8000  # short ramps (~2.8 mm) so most of the travel is at cruise
END_MARGIN_MM = 2.0  # stay this far off each hardstop (MOVEMM has no stall check)
SAMPLE_PERIOD_S = 0.04
CRUISE_FRACTION = 0.85  # samples below this fraction of cruise speed are tagged as ramp
# Abort only when the stall flag holds for this many consecutive cruise samples
# (~0.12 s at 25 Hz). The SGT values are tuned for homing into a hardstop; mid-
# travel the free-running sg is only ~40-50 and noise dips to single digits, so
# a single flagged sample can be noise (seen on J1). A real bind persists.
STALL_CONFIRM_SAMPLES = 3
# J1's sg spikes to 0 right as it reaches cruise (the same spike behind its
# false-early HOME stalls), so the stall check ignores the first part of cruise.
CRUISE_SETTLE_S = 0.3
LOST_STEP_START_MM = 20.0
NAMES = {0: "thumb", 1: "index", 2: "middle"}

driver: MantaHandDriver | None = None
active_joints: list[int] = []


def disable_all():
    if driver is None:
        return
    for j in active_joints:
        for cmd in ("STOP", "DIS"):
            try:
                driver.send(cmd, f"J{j}")
            except Exception:
                pass


def on_signal(signum, _frame):
    disable_all()
    print(f"\nsignal {signum}: stopped and disabled", flush=True)
    sys.exit(1)


def drv_status(j: int) -> tuple[int, bool]:
    raw = int(driver.send("RREG5160", f"J{j}", "6F").split()[0], 16)
    return raw & 0x3FF, bool(raw & (1 << 24))


def travel(j: int) -> tuple[float, float]:
    """(near, far) commandable mm for joint j, sign-aware (J4 runs negative)."""
    full = FULL_EXTENSION_MM[j]
    sign = 1.0 if full > 0 else -1.0
    return sign * END_MARGIN_MM, full - sign * END_MARGIN_MM


def move_cap_s(distance_mm: float) -> float:
    return abs(distance_mm) * abs(STEPS_PER_MM) / SWEEP_VELOCITY + SWEEP_VELOCITY / SWEEP_ACCEL + 4.0


def move_and_log(j: int, target_mm: float, writer, tag: dict) -> int:
    """Move joint j to target_mm, sampling sg_result. Returns sample count.
    Raises after STOP+DIS on stall flag at cruise or wall-clock cap."""
    joint = driver.joints[j]
    start_mm = joint.status.position / STEPS_PER_MM
    cap = move_cap_s(target_mm - start_mm)
    joint.move_to_mm(target_mm, SWEEP_VELOCITY, SWEEP_ACCEL)
    t0 = time.monotonic()
    prev_pos, prev_t = None, None
    n = 0
    stall_run = 0
    cruise_since = None
    while True:
        now = time.monotonic()
        st = joint.status
        sg, stall = drv_status(j)
        mm = st.position / STEPS_PER_MM
        vel = 0.0 if prev_pos is None else (st.position - prev_pos) / (now - prev_t)
        cruise = abs(vel) >= CRUISE_FRACTION * SWEEP_VELOCITY
        prev_pos, prev_t = st.position, now
        if writer is not None:
            writer.writerow({**tag, "t": f"{now - t0:.3f}", "mm": f"{mm:.3f}", "sps": f"{vel:.0f}",
                             "cruise": int(cruise), "sg": sg, "stall": int(stall)})
            n += 1
        cruise_since = (cruise_since or now) if cruise else None
        settled = cruise and now - cruise_since >= CRUISE_SETTLE_S
        stall_run = stall_run + 1 if (stall and settled) else 0
        if stall and settled:
            print(f"    J{j}: stall flag at {mm:.2f} mm (sg={sg}), {stall_run}/{STALL_CONFIRM_SAMPLES}", flush=True)
        if stall_run >= STALL_CONFIRM_SAMPLES:
            disable_all()
            raise RuntimeError(f"J{j}: STALL held {stall_run} samples, at {mm:.2f} mm -- stopped and disabled")
        if not st.moving:
            return n
        if now - t0 > cap:
            disable_all()
            raise RuntimeError(f"J{j}: move to {target_mm} mm exceeded {cap:.1f}s cap -- disabled")
        time.sleep(SAMPLE_PERIOD_S)


def park(j: int, mm: float):
    driver.joints[j].enable()
    move_and_log(j, mm, None, {})
    driver.joints[j].disable()


def lost_steps_check(j: int) -> float:
    """Back off, re-home, and return (stall position - 0) in mm before re-zeroing."""
    joint = driver.joints[j]
    joint.enable()
    # Start beyond HOME's unchecked grace window (ramp to 12000 sps at 2000
    # sps^2 = 11.2 mm, plus 0.8 s ~= 3 mm). From 10 mm the axis reached the wall
    # inside that window and ground ~4.2 mm into it, which read as lost steps.
    move_and_log(j, LOST_STEP_START_MM, None, {})
    joint.write_reg5160(0x6D, HOME_COOLCONF[j])
    timeout_ms = _home_timeout_ms(j)
    joint.home(HOME_DIRECTION, HOME_VELOCITY, HOME_ACCEL, timeout_ms)
    t0 = time.monotonic()
    st = joint.status
    while st.homing_result == 1:
        if time.monotonic() - t0 > timeout_ms / 1000 + 5:
            disable_all()
            raise RuntimeError(f"J{j}: re-home exceeded cap -- disabled")
        time.sleep(0.2)
        st = joint.status
    offset_mm = st.position / STEPS_PER_MM
    joint.zero()
    joint.disable()
    return offset_mm, st.homing_result


def main():
    global driver
    finger = int(sys.argv[1])
    quick = "--quick" in sys.argv
    only = int(sys.argv[sys.argv.index("--joint") + 1]) if "--joint" in sys.argv else None
    jx, jy = STEPPER_JOINTS[finger]
    active_joints.extend([jx, jy])
    signal.signal(signal.SIGHUP, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    print(f"driver: {manta_hand.__file__}")
    # Corrected driver (~/hand, 2026-08-29): every axis travels positive mm from
    # home. A negative entry means a stale copy that would aim MOVEMM into the
    # home hardstop with no stall check.
    if any(v <= 0 for v in FULL_EXTENSION_MM.values()):
        sys.exit(f"stale driver: FULL_EXTENSION_MM={FULL_EXTENSION_MM} has a non-positive axis -- refusing to move")
    print(f"finger {finger} ({NAMES[finger]}): x=J{jx} {travel(jx)}, y=J{jy} {travel(jy)} mm")

    out = f"/tmp/gantry_rom_f{finger}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    driver = MantaHandDriver(PORT)
    try:
        for j in (jx, jy):
            driver.joints[j].set_scale(STEPS_PER_MM)
            driver.joints[j].enable()
            _home_one_axis(driver, j)
            driver.joints[j].disable()

        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["finger", "joint", "other_joint", "other_mm", "dir",
                                              "t", "mm", "sps", "cruise", "sg", "stall"])
            w.writeheader()
            for sweep_j, other_j in ((jx, jy), (jy, jx)):
                if only is not None and sweep_j != only:
                    continue
                o_near, o_far = travel(other_j)
                parks = [(o_near + o_far) / 2] if quick else [o_near, (o_near + o_far) / 2, o_far]
                near, far = travel(sweep_j)
                for other_mm in parks:
                    park(other_j, other_mm)
                    joint = driver.joints[sweep_j]
                    joint.enable()
                    move_and_log(sweep_j, near, None, {})
                    for direction, target in (("out", far), ("back", near)):
                        tag = {"finger": finger, "joint": sweep_j, "other_joint": other_j,
                               "other_mm": f"{other_mm:.1f}", "dir": direction}
                        t0 = time.monotonic()
                        n = move_and_log(sweep_j, target, w, tag)
                        print(f"  J{sweep_j} {direction:4s} {near:6.1f}->{far:6.1f} mm, "
                              f"J{other_j} at {other_mm:6.1f}: {time.monotonic() - t0:4.1f}s, {n} samples",
                              flush=True)
                    joint.disable()
                    f.flush()
            park(jx, travel(jx)[0])  # leave the finger near home

        for j in (jx, jy) if only is None else (only,):
            offset, result = lost_steps_check(j)
            how = "stall" if result == 2 else "timeout" if result == 3 else f"result={result}"
            print(f"  J{j} lost-step check: home found at {offset:+.3f} mm vs original zero ({how})")
        print(f"wrote {out}")
    finally:
        disable_all()
        driver.close()


if __name__ == "__main__":
    main()
