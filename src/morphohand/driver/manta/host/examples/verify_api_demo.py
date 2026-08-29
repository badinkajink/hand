#!/usr/bin/env python3
"""Hands-on verification script for the Hand/HandFinger API
(manta_hand.hand). Exercises, in order:

  1. home        -- Hand.home_all(): homes all 6 stepper axes, then zeros
                     all 9 servos.
  2. global x/y   -- HandFinger.move_to_global(): position a finger's
                     gantry in the shared palm-fixed frame {P}.
  3. local x/y    -- HandFinger.move_to_local(): position the same
                     finger's gantry in its OWN local frame (q_Fx, q_Fy).
  4. joint sweep  -- Hand.set_joints_fast(): real-time-capable aa/fe1/fe2
                     control loop (the fast path -- see servos.py's
                     ServoBus.sync_set_joints for why this exists instead
                     of HandFinger.move_to_local/global for anything
                     resembling a control loop).

Needs BOTH the Manta M8P (steppers, USB-CDC) and the U2D2 (servos,
USB-serial) connected on the same machine this runs from -- edit
STEPPER_PORT/SERVO_PORT below to match. As of this session both are
wired to the CB1 (ssh irlab@10.99.99.2), so this is most easily run there;
running it from a Mac would need both devices moved over (the U2D2 alone
isn't enough -- 'home' and the x/y calls need the M8P too).

Run: python3 verify_api_demo.py
"""
import math
import sys
import time

sys.path.insert(0, "..")
from manta_hand import Hand, MantaHandDriver, ServoBus

STEPPER_PORT = "/dev/ttyACM0"
SERVO_PORT = "/dev/ttyUSB0"

DEMO_FINGER = 0  # thumb -- origin (-50, 0) in {P}, local box roughly x:[-30,+26.2] y:[-55,+55]


def main():
    with MantaHandDriver(STEPPER_PORT) as stepper_driver, ServoBus(SERVO_PORT) as servo_bus:
        for sid in range(9):
            servo_bus.servo(sid).enable()

        hand = Hand(stepper_driver, servo_bus)  # Gantry enables + sets stepper scale internally

        # 1. HOME -- all 6 steppers, then all 9 servos to their zero references
        print("=== 1. home_all() ===", flush=True)
        hand.home_all()

        finger = hand.finger(DEMO_FINGER)

        # 2. GLOBAL x/y -- position in the shared palm frame {P}
        print("\n=== 2. move_to_global (finger 0, thumb) ===", flush=True)
        finger.move_to_global(x_mm=-40.0, y_mm=-20.0, fe1=10)
        time.sleep(1.5)
        print("  moved to global (x=-40, y=-20)", flush=True)

        # 3. LOCAL x/y -- position in this finger's own local frame
        #    (thumb's origin is (-50,0) in {P}, so local (10,-20) ==
        #    the same physical point as global (-40,-20) above)
        print("\n=== 3. move_to_local (finger 0, thumb) ===", flush=True)
        finger.move_to_local(x_mm=10.0, y_mm=-20.0, fe1=10)
        time.sleep(1.5)
        print("  moved to local (x=10, y=-20) -- same physical point as step 2", flush=True)

        # 4. JOINT SWEEP -- real-time-capable fast path, all 9 servos,
        #    oscillating each through its calibrated FINGER_JOINTS range
        print("\n=== 4. joint sweep via set_joints_fast (10s, 20Hz) ===", flush=True)
        from manta_hand.servos import FINGER_JOINTS

        params = {}
        phase_i = 0
        for fid, joints in FINGER_JOINTS.items():
            for name, (sid, zero_deg, (lo, hi)) in joints.items():
                center, amp = (lo + hi) / 2.0, (hi - lo) / 2.0
                params[(fid, name)] = (center, amp, phase_i)
                phase_i += 1

        duration_s, target_hz, cycle_s = 10.0, 20, 6.0
        period = 1.0 / target_hz
        start = time.monotonic()
        n = 0
        while time.monotonic() - start < duration_s:
            t = time.monotonic() - start
            pose = {0: {}, 1: {}, 2: {}}
            for (fid, name), (center, amp, phase_i) in params.items():
                phase = phase_i * (2 * math.pi / 9)
                pose[fid][name] = center + amp * math.sin(2 * math.pi * t / cycle_s + phase)
            hand.set_joints_fast(pose)
            n += 1
            next_tick = start + n * period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
        print(f"  sent {n} updates over {time.monotonic()-start:.1f}s", flush=True)

        # settle back to zero for a clean end state
        hand.set_joints_fast({fid: {name: 0.0 for name in joints} for fid, joints in FINGER_JOINTS.items()})
        time.sleep(2.0)
        print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
