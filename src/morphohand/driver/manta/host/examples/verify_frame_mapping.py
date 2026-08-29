#!/usr/bin/env python3
"""Settle the three things about this hand that only the hardware can answer.

    python3 verify_frame_mapping.py --all
    python3 verify_frame_mapping.py --travel        # just the ruler check
    python3 verify_frame_mapping.py --signs         # just the servo directions

`manta_hand.plan` converts a simulated hand design into gantry positions and servo angles. Most
of that conversion is verified on paper -- the mount origins are numerically identical on both
sides, and the sim's joint limits match the servo contract exactly and uniquely. Three things
are not, and each of them silently produces a hand that looks right and grasps nothing:

1. FINGER IDENTITY. `plan.py` believes finger 0 is the thumb at palm x -50, finger 1 the index
   at +y, finger 2 the middle at -y. That comes from a drawing, not from watching a block move.
   Swap index and middle and every asymmetric design is mirrored.

2. aa SIGN, per finger. The flexion joints give themselves away -- their calibrated ranges are
   [-15, +92]-shaped, which only "a little hyperextension, a lot of flexion" explains. aa's
   range is symmetric, so it carries no such fingerprint, and the scene gives all three fingers
   the SAME yaw axis while mirroring mcp/pip, which is not what a rigid 180-degree-rotated thumb
   module would do. A flipped sign rolls a pad off the tool: g12 grips with the thumb at
   +17.7 deg of yaw.

3. J1/J3/J5 TRAVEL. `FULL_EXTENSION_MM` says these three reach 56.2 / 56.0 / 54.1 mm against a
   60 mm nominal, and that 4-6 mm shortfall is what disqualifies the most compact 11 of the 108
   sampled designs. But those three axes' `STEPS_PER_MM` was, per its own comment,
   "back-calculated from a known-good 10mm move and hasn't been individually ruler-checked" --
   so the shortfall may be a scale error rather than a wall. The two cases need opposite
   responses and a caliper tells them apart in a minute:

     commanded 45 mm moves ~45 mm   -> the scale is right, the rails really are short,
                                       those 11 designs are unbuildable, stop trying
     commanded 45 mm moves ~48-50 mm-> the scale is 6-10 percent high, every mm this driver has
                                       ever commanded on these axes was short by that much, and
                                       fixing STEPS_PER_MM recovers the whole design box

   The second case matters beyond the design set: mount positions are commanded in the same mm.

Nothing here is destructive -- every move is well inside a calibrated range, and the travel
probe stops short of both hypotheses' far hardstop. It does drive real hardware, so it asks
before each stage and needs --yes to skip the prompts.
"""
from __future__ import annotations

import argparse
import sys
import time

from manta_hand import Hand, MantaHandDriver, ServoBus
from manta_hand.kinematics import (
    FULL_EXTENSION_MM, STEPPER_ACCEL, STEPPER_JOINTS, STEPPER_VELOCITY, STEPS_PER_MM,
)
from manta_hand.plan import FINGER_ID, SIM_JOINT_TO_SERVO
from manta_hand.servos import FINGER_JOINTS

PROBE_MM = 45.0     # long enough for a caliper to resolve a 6% scale error (2.7mm), short of
                    # the far hardstop under BOTH hypotheses (45mm commanded is at most ~50mm
                    # travelled if the scale is 10% high, against a >=54mm rail)
PROBE_DEG = 20.0    # inside every joint's measured range on every finger


def ask(prompt: str, yes: bool) -> str:
    if yes:
        print(f"  {prompt} [auto-yes]")
        return ""
    return input(f"  {prompt} ").strip()


def stage_identity(hand, yes: bool) -> None:
    """Which physical gantry block is finger 0/1/2."""
    print("\n1. FINGER IDENTITY")
    print("   Each finger moves 10 mm along its own local x, one at a time. Watch which block")
    print("   moves and where it sits relative to the palm.")
    for name, fid in FINGER_ID.items():
        if ask(f"move the block plan.py calls '{name}' (finger {fid})? [enter to go, s to skip]",
               yes) == "s":
            continue
        f = hand.finger(fid)
        f.move_to_local(0.0, 0.0)
        time.sleep(1.5)
        f.move_to_local(10.0, 0.0)
        time.sleep(1.5)
        seen = ask(f"which block moved, and is it at palm "
                   f"{'x -50' if name == 'thumb' else ('+y' if name == 'index' else '-y')}? "
                   f"[y / describe]", yes)
        if seen and not seen.lower().startswith("y"):
            print(f"   !! finger {fid} is NOT {name}: {seen}")
            print("      Fix STEPPER_JOINTS / FINGER_GEOMETRY in kinematics.py before going on;")
            print("      every design below is mirrored until you do.")
        f.move_to_local(0.0, 0.0)
        time.sleep(1.0)


def stage_signs(hand, yes: bool) -> dict[tuple[str, str], float]:
    """Which physical direction a positive servo command produces, per finger per joint."""
    print("\n2. JOINT SIGNS")
    print("   Each joint goes to zero, then to +20 deg. For aa, the question is which way the")
    print("   FINGERTIP swings in the palm frame {P} (+y is 'up' on the drawing, toward the")
    print("   index side). For fe1/fe2 it is simply whether the finger curls toward the palm")
    print("   centre -- if any fe answer is 'away', stop: the zero references are wrong, not")
    print("   just the signs.")
    signs: dict[tuple[str, str], float] = {}
    for name, fid in FINGER_ID.items():
        for sim_joint, servo in SIM_JOINT_TO_SERVO.items():
            _sid, _zero, (lo, hi) = FINGER_JOINTS[fid][servo]
            if not (lo <= PROBE_DEG <= hi):
                print(f"   {name}.{servo}: +{PROBE_DEG:.0f} deg outside measured "
                      f"[{lo:.1f},{hi:.1f}] -- skipped, probe it by hand")
                continue
            if ask(f"{name}.{servo} (sim {name}_{sim_joint}): 0 -> +{PROBE_DEG:.0f} deg? "
                   f"[enter / s]", yes) == "s":
                continue
            hand.set_joints_fast({fid: {servo: 0.0}}, speed=60)
            time.sleep(1.5)
            hand.set_joints_fast({fid: {servo: PROBE_DEG}}, speed=60)
            time.sleep(1.5)
            if sim_joint == "yaw":
                a = ask("did the fingertip move toward +y_P (index side)? [y/n]", yes)
                # The scene rotates every finger's yaw about +x_P, so positive sim yaw swings
                # every tip toward +y_P. Matching that is sign +1; the opposite is -1.
                signs[(name, sim_joint)] = 1.0 if a.lower().startswith("y") or yes else -1.0
            else:
                a = ask("did the finger curl toward the palm centre? [y/n]", yes)
                signs[(name, sim_joint)] = 1.0 if a.lower().startswith("y") or yes else -1.0
                if signs[(name, sim_joint)] < 0:
                    print(f"   !! {name}.{servo} flexes the wrong way at positive command --"
                          f" check that servo's zero reference, not just its sign")
            hand.set_joints_fast({fid: {servo: 0.0}}, speed=60)
            time.sleep(1.0)
    return signs


def stage_travel(driver, yes: bool) -> None:
    """Is the y-axis shortfall a scale error or a wall."""
    print("\n3. J1/J3/J5 TRAVEL -- scale error, or a real wall?")
    print(f"   Each axis homes, then is commanded to +{PROBE_MM:.0f} mm. Measure the actual")
    print("   travel with a caliper against a fixed reference and type it in.")
    for j in (1, 3, 5):
        if ask(f"home and probe J{j} (believed travel {FULL_EXTENSION_MM[j]:.1f} mm, "
               f"scale {STEPS_PER_MM[j]:.1f} steps/mm)? [enter / s]", yes) == "s":
            continue
        from manta_hand.kinematics import _home_one_axis
        _home_one_axis(driver, j)
        driver.joints[j].move_to_mm(PROBE_MM, STEPPER_VELOCITY, STEPPER_ACCEL)
        while driver.joints[j].status.moving:
            time.sleep(0.2)
        raw = ask(f"measured travel from home, in mm? [number / s]", yes)
        try:
            measured = float(raw)
        except ValueError:
            print("   skipped")
            continue
        ratio = measured / PROBE_MM
        true_scale = STEPS_PER_MM[j] / ratio
        true_travel = FULL_EXTENSION_MM[j] * ratio
        print(f"   J{j}: commanded {PROBE_MM:.1f}, moved {measured:.1f} -> "
              f"{100 * (ratio - 1):+.1f}% scale error")
        if abs(ratio - 1) < 0.01:
            print(f"       scale is right. This rail really does stop at "
                  f"{FULL_EXTENSION_MM[j]:.1f} mm; the compact designs needing it are out.")
        else:
            print(f"       STEPS_PER_MM[{j}] should be {true_scale:.1f}, and the real travel is "
                  f"~{true_travel:.1f} mm.")
            print(f"       Every mm ever commanded on J{j} was off by {100*(ratio-1):+.1f}% -- "
                  f"that includes mount positions.")
        driver.joints[j].move_to_mm(10.0, STEPPER_VELOCITY, STEPPER_ACCEL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--signs", action="store_true")
    ap.add_argument("--travel", action="store_true")
    ap.add_argument("--stepper-port", default="/dev/ttyACM0")
    ap.add_argument("--servo-port", default="/dev/ttyUSB0")
    ap.add_argument("--yes", action="store_true", help="don't prompt (answers count as 'as expected')")
    args = ap.parse_args()
    if not (args.all or args.identity or args.signs or args.travel):
        ap.error("pick at least one of --all/--identity/--signs/--travel")

    want = {k: args.all or getattr(args, k) for k in ("identity", "signs", "travel")}
    with MantaHandDriver(args.stepper_port) as driver:
        if want["identity"] or want["signs"]:
            with ServoBus(args.servo_port) as bus:
                hand = Hand(driver, bus)
                if want["identity"]:
                    print("homing all axes first -- identity and signs both start from home")
                    hand.home_all()
                    stage_identity(hand, args.yes)
                if want["signs"]:
                    signs = stage_signs(hand, args.yes)
                    if signs:
                        print("\nPaste into manta_hand/plan.py, and set SIGNS_MEASURED = True:")
                        print("JOINT_SIGN = {")
                        for (f, j), s in signs.items():
                            print(f'    ("{f}", "{j}"): {s:+.1f},')
                        print("}")
        if want["travel"]:
            stage_travel(driver, args.yes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
