#!/usr/bin/env python3
"""Move gantry axes to absolute mm-from-home targets, disabling each the moment
it arrives (leadscrews hold position unpowered).

    python3 /tmp/gantry_goto.py J0=21.26 J1=25.68 J2=42.92 ... [--together]
    python3 /tmp/gantry_goto.py home [--together]      (every axis to 2 mm off its wall)

Default: one axis at a time, with the StallGuard abort. --together: all axes
started back to back, then no traffic until done (the firmware drops commands
while several axes step); NO stall check; deadline-capped.

Trusts the current zeros (home first if unsure). Each target is bounds-checked
against FULL_EXTENSION_MM minus END_MARGIN_MM before anything moves. A stall
held at cruise (STALL_CONFIRM_SAMPLES in a row, after CRUISE_SETTLE_S) or a
wall-clock cap stops and disables. The stall check uses each axis's
HOME_COOLCONF SGT, except J1 at SGT 12 (at SGT 10 its flag is on ~50% of the
time while running freely). Afterwards checks every driver's CHOPCONF/GSTAT
for a chip reset (J2's TMC5160 reset silently on 2026-11-03).
"""

import signal
import sys
import time

sys.path.insert(1, "/tmp")
import gantry_rom_sweep as g  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402
from manta_hand.protocol import MantaHandError  # noqa: E402

SGT_OVERRIDE = {1: 12}
GOOD_CHOPCONF = 0x140100C3
POLL_S = 0.2  # together mode: one STATALL per 0.2 s
STAT_TRIES = 3
# Together mode speed: at 12000 sps with six axes stepping the firmware fell behind
# (J4/J5 at about half their commanded rate) and stopped answering commands.
TOGETHER_VELOCITY = 6000
TOGETHER_ACCEL = 8000  # with several axes stepping the board can drop a reply (2 s timeout each)
STAGGER_S = 0.5  # >= the firmware's 500 ms start kick, so kicks never overlap


def retry(fn, what):
    """Run an idempotent command (absolute MOVEMM, DIS, STATALL) with retries."""
    for attempt in range(1, STAT_TRIES + 1):
        try:
            return fn()
        except MantaHandError as e:
            print(f"    {what} attempt {attempt}/{STAT_TRIES} failed: {e}", flush=True)
    g.disable_all()
    raise RuntimeError(f"board stopped answering ({what}) -- all axes stopped and disabled")


def statall():
    """STATALL with retries: under multi-axis stepping the firmware sometimes
    misses a command (no reply within 2 s, no USB disconnect). The driver clears
    its input buffer before each send, so a retry can't pick up a stale reply."""
    return retry(g.driver.get_all_status, "STATALL")


def move_together(targets, vel=TOGETHER_VELOCITY):
    """Start every axis back to back while all are at standstill, then send NOTHING
    until the slowest should be done, then check once and disable all.

    2026-11-03: with several axes stepping at 12000 sps the firmware drops or
    delays commands (no reply within the driver's 2 s timeout, no USB
    disconnect) -- polling, MOVEMMs and DIS sent mid-move all failed, and a late
    reply corrupted the next STATALL. Starts while nothing is moving go through.
    All start kicks overlap: fine on the 24 V / 6 A supply (it browned out the old
    19 V / 65 W brick)."""
    d = g.driver
    st0 = d.get_all_status()
    for j in targets:
        d.joints[j].enable()
    t0 = time.monotonic()
    for j, mm in targets.items():
        d.joints[j].move_to_mm(mm, vel, TOGETHER_ACCEL)
    longest = max(abs(mm - st0[j].position / g.STEPS_PER_MM) * abs(g.STEPS_PER_MM) / vel + vel / TOGETHER_ACCEL
                  for j, mm in targets.items())
    time.sleep(longest + 0.5)
    deadline = t0 + 2 * longest + 10.0
    while True:
        try:
            all_st = d.get_all_status()
            if not any(all_st[j].moving for j in targets):
                break
        except (MantaHandError, ValueError) as e:
            print(f"    STATALL failed ({e}); retrying", flush=True)
        if time.monotonic() > deadline:
            g.disable_all()
            raise RuntimeError("axes still moving past the deadline -- all stopped and disabled")
        time.sleep(0.5)
    for j in targets:
        d.joints[j].disable()
    for j, mm in targets.items():
        print(f"  J{j}: {st0[j].position / g.STEPS_PER_MM:6.2f} -> {all_st[j].position / g.STEPS_PER_MM:6.2f} mm "
              f"(target {mm:.2f}), disabled", flush=True)
    print(f"  all arrived in {time.monotonic() - t0:.1f}s (expected {longest:.1f}s at {vel} sps)", flush=True)


def check_drivers(joints):
    bad = []
    for j in joints:
        c = int(g.driver.send("RREG5160", f"J{j}", "6C").split()[0], 16)
        gs = int(g.driver.send("RREG5160", f"J{j}", "01").split()[0], 16)
        if c != GOOD_CHOPCONF or gs & 1:
            bad.append(f"J{j} (CHOPCONF {c:08x}, GSTAT {gs:x})")
    print("driver check: " + ("all OK, no resets" if not bad else "RESET/BAD: " + ", ".join(bad)), flush=True)


def main():
    together = "--together" in sys.argv
    vel = int(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else TOGETHER_VELOCITY
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--speed"]
    targets = {}
    if args == ["home"]:
        targets = {j: g.travel(j)[0] for j in range(6)}
    else:
        for arg in args:
            j, mm = arg.upper().lstrip("J").split("=")
            targets[int(j)] = float(mm)
    if any(v <= 0 for v in g.FULL_EXTENSION_MM.values()):
        sys.exit("stale driver: non-positive FULL_EXTENSION_MM -- refusing to move")
    for j, mm in targets.items():
        near, far = g.travel(j)
        if not near <= mm <= far:
            sys.exit(f"J{j}={mm} mm is outside [{near}, {far}] -- nothing moved")
    g.active_joints.extend(sorted(targets))
    signal.signal(signal.SIGHUP, g.on_signal)
    signal.signal(signal.SIGTERM, g.on_signal)
    g.driver = MantaHandDriver(g.PORT)
    try:
        for j in sorted(targets):
            jt = g.driver.joints[j]
            jt.set_scale(g.STEPS_PER_MM)
            jt.write_reg5160(0x6D, SGT_OVERRIDE.get(j, g.HOME_COOLCONF[j] >> 16) << 16)
        if together:
            move_together(targets, vel)
        else:
            for j in sorted(targets):
                jt = g.driver.joints[j]
                start = jt.status.position / g.STEPS_PER_MM
                jt.enable()
                g.move_and_log(j, targets[j], None, {})
                jt.disable()
                print(f"  J{j}: {start:6.2f} -> {jt.status.position / g.STEPS_PER_MM:6.2f} mm, disabled", flush=True)
        check_drivers(sorted(targets))
    finally:
        g.disable_all()
        g.driver.close()


if __name__ == "__main__":
    main()
