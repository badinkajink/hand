# 3-servo finger bring-up notes (2026-08-28)

## Zero references redefined (all servos, current bus state)

All 9 servos (0-8) were on the bus together for the first time and every
zero reference below was redefined from freshly-read live positions at
this point, superseding the individual per-servo zero values recorded
earlier in this file (kept below for history/context on each unit's
individual bring-up, but these are now the current values to use):

| Servo | Zero (raw) |
|---|---|
| 0 | -41.0156 |
| 1 | -34.2773 |
| 2 | **-43.9453** (first zero ever recorded for this servo -- previously flagged as missing) |
| 3 | -10.2539 |
| 4 | 84.9609 |
| 5 | -146.7773 |
| 6 | -149.70703125 (confirmed again; a disable/re-enable torque cycle fixed a stuck-at-wrong-position state where direct move commands weren't taking effect despite real load -- worth trying that cycle first if this servo gets stuck again, before a full power-cycle) |
| 7 | 81.7383 |
| 8 | 72.6562 |

## Joint role mapping (all 3 fingers)

Same joint-role range convention across all three fingers: **aa**
(adduction/abduction), **fe1** (phalanx 1 flexion), **fe2** (phalanx 2
flexion), each range relative to that servo's own zero reference above:

- aa: [-85 deg, +85 deg]
- fe1: [-15 deg, +92 deg]
- fe2: [-18 deg, +92 deg]

| Finger | aa | fe1 | fe2 | stepper (X) | stepper (Y) |
|---|---|---|---|---|---|
| 0 (thumb) | ID 0 | ID 1 | ID 2 | stepper0: [0, 110mm] | stepper1: [0, 59.9mm] |
| 1 (index) | ID 3 | ID 4 | ID 5 | stepper2: [0, 60mm] | stepper3: [0, 59.9mm] |
| 2 (middle) | ID 6 | ID 7 | ID 8 | stepper4: [0, -60mm] | stepper5: [0, 59.9mm] |

## Confirmed full-flexion positions (fe1/fe2 only, aa excluded)

User-confirmed real full-flexion raw positions per servo -- NOT the
theoretical zero+92 relative value (which was physically unreachable for
every unit tested; see the "full flexion" attempt in this session's
transcript). aa (adduction/abduction) is excluded from "full flexion"
entirely -- it's a separate DOF and should not move for a flexion command.

| Finger | fe1 servo | fe1 full flexion (raw) | fe2 servo | fe2 full flexion (raw) |
|---|---|---|---|---|
| 0 (thumb) | ID 1 | 54.7852 | ID 2 | 45.9961 |
| 1 (index) | ID 4 | 35.1562 | ID 5 | -59.1797 |
| 2 (middle) | ID 7 | 149.7070 | ID 8 | 149.7070 |

## Confirmed full-extension positions (fe1/fe2 only)

User-confirmed real full-extension raw positions per servo, read
immediately after the full-flexion positions above (note how different
these are from the earlier zero references -- these fe1/fe2 servos moved a
lot between reads without an explicit command from this session each time;
treat all of these positions as needing periodic reconfirmation, not fixed
constants):

| Finger | fe1 servo | fe1 full extension (raw) | fe2 servo | fe2 full extension (raw) |
|---|---|---|---|---|
| 0 (thumb) | ID 1 | -45.9961 | ID 2 | -62.4023 |
| 1 (index) | ID 4 | 74.7070 | ID 5 | 39.8438 |
| 2 (middle) | ID 7 | 70.8984 | ID 8 | 55.9570 |

Note: finger 2's stepper4 X range is negative ([0, -60mm]) unlike fingers
0/1 -- likely a mirrored rail direction for that finger's physical
position on the hand, not a typo. Ranges above are as given by the user,
not yet independently re-verified against the ranges/stall data recorded
per-servo elsewhere in this file (several servos, e.g. 0 and 6, showed
real stalls well short of their full claimed aa/fe1/fe2 range during
earlier probing -- worth reconciling before trusting these as safe to
drive to blindly).

## Global coordinate system {P}

Palm-fixed global frame `{P}`, +x_P right, +y_P up (top-down view of the
hand). Each finger has its own local, axis-aligned (not rotated) frame
with origin given relative to `{P}`:

| Finger | Local frame | Origin rel. to {P} (mm) | Workspace box (W x H, mm) | +flex direction |
|---|---|---|---|---|
| Thumb | q_Tx, q_Ty | (-50, 0) | 60 x 110 | +x_P (toward center/right) |
| Index | q_Ix, q_Iy | (50, 55) | 60 x 60 | -x_P (toward center/left) |
| Middle | q_Mx, q_My | (50, -55) | 60 x 60 | -x_P (toward center/left) |

Auxiliary dimensions shown in the source drawing: 40mm horizontal from the
{P} vertical centerline to the index/middle box region, 50mm vertical from
the {P} horizontal centerline to the boundary between the index and middle
boxes -- recorded as depicted, not independently re-derived from the
origin coordinates above.

**Home positions** (green circles in the drawing, one per finger, at a
specific corner of that finger's workspace box):

- Thumb: bottom-left corner of its box.
- Index: top-right corner of its box.
- Middle: bottom-right corner of its box.

Exact numeric (x, y) home coordinates per finger were not pinned down
precisely from the drawing -- if commanding actual home moves, confirm
each corner's precise local coordinates first (derivable from each
finger's origin +/- half box width/height once corner selection is
confirmed) rather than assuming from this description alone.

Resolved: the thumb's box is 110mm tall (along q_Ty) x 60mm wide (along
q_Tx) in this drawing, while the joint-role-mapping table's `stepper0`
("x", [0, 110mm]) / `stepper1` ("y", [0, 59.9mm]) axis labels don't match
that orientation directly -- confirmed intentional, not a conflict. Each
finger is on its own independent gantry, and the stepper firmware's x/y
axis labels are flipped relative to this drawing's q_Fx/q_Fy convention.
Keep both namings as-is; don't try to force them into agreement.

Working notes from live U2D2 bring-up of the new independent-phalanx finger
servos (see `docs/servos.md`'s "Upcoming hardware" section). Not yet wired
into `manta_hand/servos.py`'s `FINGERS` table -- joint roles (adduction /
phalanx1 / phalanx2) aren't decided/assigned yet. All positions are raw
`status.position_deg` readings (servo's own zero-offset, not comparable
between units) unless noted as "relative to zero", which means relative to
that specific servo's own recorded zero reference below.

All EEPROM min/max_angle_limit resets are to raw (0, 1023), which reads
back as (-149.70703125, 76.171875) degrees in this raw-to-degree convention
-- fixed across every unit tested, not per-unit.

## Servo 0

- Zero reference (raw): **-43.9453** (also called "full extension" by hand)
- User-specified target range (relative to zero): -85 to +85
- Real stall points found during automated hardstop sweep (small-step,
  present-position-verified): **-115.43 raw (relative -71.48)** stepping
  negative, then **-90.23 raw (relative -46.29)** stepping positive away
  from that -- two separate stall points ~25 deg apart, never confirmed to
  reach the full claimed +/-85 range.
- One serious incident: a buggy stall-detector looped for several minutes
  commanding against a genuine stall before being caught and killed. No
  lasting damage observed (temp normal) but flagging in case of long-term
  wear.

## Servo 1

- Zero reference (raw): **-27.8320**
- True probed range (hand-verified extremes via incremental step probe):
  **-148.54 (real stop) to +89.36 (untested safe cap, not a confirmed true
  max -- probe was capped at +90, same caveat as finger 0's adduction in
  docs/servos.md)**
- User-specified target range (relative to zero): -15 to +92 -- not
  re-verified against the corrected sweep script.
- Note: reusing another servo's raw-degree numbers as "center" does NOT
  work -- each unit's own zero-offset differs even when EEPROM limits are
  reset identically. Confirmed the hard way (a reused center landed at a
  real endstop for this unit).

## Servo 2

- Never got a clean full-range probe -- two attempts interrupted (one by a
  real stall/comms timeout mid-probe, one by user "stop"/unplug).
- Confirmed-working move: +15 deg relative step from -64.75 to -50.39 raw,
  clean, no stall.
- User-specified target range (relative to a defined zero): -18 to +92 --
  zero reference for this servo was not durably recorded before it was
  swapped out; needs re-establishing before trusting this range.

## Servo 6

- Zero reference (raw): **-149.70703125** (redefined multiple times this
  session after ID reassignment and EEPROM limit resets -- treat this as
  current, not the earlier transient values).
- Known unreliable / flaky unit this session:
  - `torque_enable` register has read back values (including `3`=FREE)
    that did NOT match actual physical behavior (present_load stayed
    ~300 while supposedly free-spinning) -- a real register/behavior
    mismatch, not user error.
  - Present position has changed between checks without any commanded
    move traceable to it (e.g. jumped -149.71 -> 101.37 unexplained;
    another jump to 98.73 after a failed out-of-range write). Root cause
    unresolved -- possibly a leftover goal-register creep after a script
    exits while torque stays enabled, possibly manual handling, unclear.
  - Repeated real stalls at DIFFERENT raw positions across attempts
    (100.5, -84, -81.7, -142.97) rather than one consistent hardstop --
    pattern suggests insufficient torque to push through ordinary
    mechanism friction rather than a single fixed mechanical limit.
  - **present_voltage consistently reads ~50 (likely 5.0V)**, close to
    this servo's own configured min_voltage_limit (45 = 4.5V) and well
    under max_voltage_limit (90 = 9.0V). SCS0009 typically wants more
    like 6-7.4V for rated torque. Worth checking the actual supply/rail
    voltage under load before trusting further range data from this unit.
  - Direct single `move_to_deg()` calls (higher speed, ~2.5s settle) have
    worked more reliably than small-step incremental sweeps for this
    servo specifically -- unclear why, noted as an observation not an
    explanation.
- User-observed extremes (by hand, approximate, not software-confirmed
  clean): one extreme called "+85 deg" landed around raw -80 to -84;
  the other called "-85 deg" landed around raw 98-103. These do not
  cleanly bracket the zero reference above -- take with caution given the
  unit's overall flakiness this session.

## Servo 7

- Zero reference (raw): **82.03125** ("upper position", hand-set)
- Note: this zero sits *outside* the standard reset EEPROM max_angle_limit
  (76.171875) -- this unit's usable window differs from the others even
  after the same factory-range reset.
- Hand-positioned points:
  - Target -15 deg (relative) -> measured **71.19 raw (relative -10.84)**
  - Target ~+92 deg (relative, rough) -> measured **149.70703125 raw
    (relative +67.68)**
- Clean commanded return to zero confirmed (no stall): landed at 81.74 raw.

## Servo 3

- Independent unit, unrelated to the earlier "servo 7" entry below despite
  the bus ID reassignment path used to get here.
- Zero reference (raw): **-16.11328125** (user-refined; superseded an
  earlier -12.3046875 reading)
- Hand-positioned points:
  - Right hardstop, target +85 deg (relative, rough) -> measured
    **63.57421875 raw (relative +75.88)**.
  - Left hardstop -> measured **-88.76953125 raw (relative -76.46)**.

## Servo 4

- Independent unit (repurposed hardware; unrelated to the earlier "servo 6"
  entry below despite the bus ID reassignment path used to get here).
- Zero reference (raw): **88.18359375** (user-defined)
- Hand-positioned points:
  - Hardstop, target -15 deg (relative, rough) -> measured
    **74.70703125 raw (relative -13.48)**.
  - Other hardstop, past zero in the same (negative) direction, rough
    target ~92 deg magnitude -> measured **35.15625 raw (relative
    -53.03)**. Both hardstops are on the negative side of zero for this
    unit -- zero is not centered between them.

## Servo 5

- Zero reference (raw): **-149.70703125** (hand-positioned)
- EEPROM min/max_angle_limit reset to factory full range (0, 1023 raw ->
  -149.70703125 to 76.171875 deg).
- Note: zero sits exactly at this unit's own min_angle_limit floor -- same
  edge case that caused a lot of confusion with servo 6 (any move toward
  more negative values than zero is inherently impossible from here; only
  positive-direction moves have room). Worth confirming this is actually
  the joint's real physical stop before treating it as reliable.
- Hand-positioned point: moved past zero, target ~92 deg (relative, rough,
  user-relabeled from an initial "-18 deg" attempt) -> measured
  **40.4296875 raw**.

## Real full-range probe (2026-08-29, careful small-step + present-position verified)

Automated probe of all 9 servos from their zero reference, in both
directions, using actual present-position tracking (not move_to_deg's own
write-verify, which only checks the goal register and has repeatedly
missed real stalls this session). Results, relative to each servo's zero:

| Servo | Real range (relative deg) | Confidence |
|---|---|---|
| 0 (finger 0 aa) | [-72.95, +8.20] | credible |
| 1 (finger 0 fe1) | [-12.60, +89.36] | credible, close to declared (-15,92) |
| 2 (finger 0 fe2) | [-19.92, +39.84] | credible (positive side well short of declared 92, real hardware limit) |
| 3 (finger 1 aa) | [+1.17, +1.17] | **UNRELIABLE** -- identical stuck position in both directions, essentially zero range, inconsistent with an aa joint having real freedom. Servo unresponsive during this probe, not a real measurement. Needs a clean re-test. |
| 4 (finger 1 fe1) | [-10.84, +58.59] | credible -- also contradicts this session's earlier (wrong) theory that this servo's zero sits past a hard ~76deg ceiling; it does not. |
| 5 (finger 1 fe2) | [-1.46, +85.84] | credible -- negative side tiny because zero sits close to this servo's real floor, consistent with earlier findings. |
| 6 (finger 2 aa) | [+0.00, +0.00] | **UNRELIABLE** -- identical stuck position in both directions. Matches this servo's well-documented flakiness all session (unexplained jumps, needs disable/re-enable or power-cycle to recover). Needs a clean re-test once the unit is behaving. |
| 7 (finger 2 fe1) | [-10.55, -0.88] | **UNRELIABLE** -- directly contradicts a successful verified move to +66.8 relative (148.5 raw) minutes earlier in the same session. Servo malfunctioned during this specific probe. Needs a clean re-test. |
| 8 (finger 2 fe2) | [-17.29, +74.71] | credible -- consistent with the same servo's recent successful full-flexion test (~+76.5 relative). |

Servos 3, 6, and 7's numbers above should NOT be used to update
`FINGER_JOINTS`' declared ranges -- they reflect a malfunctioning servo
during that specific probe run, not real hardware limits. The other six
(0, 1, 2, 4, 5, 8) are reasonably trustworthy and could inform tightening
`FINGER_JOINTS`' declared ranges to match real hardware, particularly
finger 0's fe2 (positive side much shorter than declared) and finger 1's
fe2 (negative side much shorter than declared).

All 9 servos were re-zeroed after this probe; servo 6 again needed a
disable/re-enable cycle to actually reach zero (present position was
stuck ~98deg away from target despite a correct goal-register write) --
same recovery pattern documented earlier in this file.

## Manual torque-free range sweep (2026-08-29) -- superseded FINGER_JOINTS

Every servo was freed one at a time (all others held at their zero) and
hand-moved through its full range while logging present_position_deg at
10Hz -- raw data in `host/examples/servo_manual_range.csv`. This avoids
every unreliability issue documented above (commanded-move stalls, phantom
loads, register/behavior mismatches) since nothing was ever commanded --
just read back while a human moved it.

**Important finding**: `Servo.free()` (writes `TORQUE_FREE=3`, per
AmazingHand's reference convention) does NOT reliably produce genuine
backdrivable behavior on this SCS0009 firmware -- servo 0 read back
`torque_enable=3` and `present_load=0` (both looking correct) while still
visibly springing back to a held position when released by hand. A full
`reboot()` didn't fix it either. Writing the RAW value `0` directly
(`write_torque_enable(id, 0)`, bypassing this project's `TORQUE_FREE=3`
constant) did produce genuine backdrivable behavior, confirmed by hand.
**Use raw `0`, not `Servo.free()`, for real hand-backdrivable testing on
this hardware** until `servos.py`'s `TORQUE_FREE` constant itself is
revisited.

Measured min/max (raw absolute degrees) and derived relative-to-zero
range, intersected with the originally-declared nominal contract to get
the EFFECTIVE range now used in `FINGER_JOINTS`:

| Servo | Zero (raw) | Measured min/max (raw) | Measured relative | Declared | Effective (used in FINGER_JOINTS) |
|---|---|---|---|---|---|
| 0 (f0 aa) | -41.0156 | -111.04 / 33.69 | [-70.02, +74.71] | [-85, 85] | [-70.02, +74.71] |
| 1 (f0 fe1) | -34.2773 | -46.58 / 54.79 | [-12.30, +89.06] | [-15, 92] | [-12.30, +89.06] |
| 2 (f0 fe2) | -43.9453 | -62.40 / 42.77 | [-18.46, +86.72] | [-18, 92] | [-18.00, +86.72] |
| 3 (f1 aa) | -10.2539 | -89.94 / 64.16 | [-79.69, +74.41] | [-85, 85] | [-79.69, +74.41] |
| 4 (f1 fe1) | 84.9609 | 34.57 / 149.71 | [-50.39, +64.75] | [-15, 92] | [-15.00, +64.75] |
| 5 (f1 fe2) | -146.7773 | -149.71 / 39.84 | [-2.93, +186.62] | [-18, 92] | [-2.93, +92.00] |
| 6 (f2 aa) | -149.70703125 | -149.71 / 149.71 | [+0.00, +299.41] | [-85, 85] | [+0.00, +85.00] |
| 7 (f2 fe1) | **79.98046875** (re-zeroed this session, was 81.7383) | 37.21 / 149.71 | [-42.77, +69.73] | [-15, 92] | [-15.00, +69.73] |
| 8 (f2 fe2) | 72.6562 | 56.54 / 149.71 | [-16.12, +77.05] | [-18, 92] | [-16.11, +77.05] |

Note some measured ranges exceed 90 degrees of real travel and/or exceed
the declared contract on one side (e.g. servo 5's positive side, servo 6's
whole range) -- the effective range still caps at the declared contract in
those cases, since the instruction was "never command further than either
the declared range or the physical range, whichever is smaller."

This table is now authoritative -- `FINGER_JOINTS` in `manta_hand/servos.py`
uses these effective ranges directly, superseding every earlier
range/zero value recorded elsewhere in this file for these 9 servos.

## Servo 6 REPLACED (2026-08-29) -- old unit had a broken position sensor

The original servo 6 (finger 2 aa) developed a hard fault mid-session:
`present_position` froze at a fixed bit-exact value (`-2.6128806086985423`
rad) across every read regardless of actual physical movement, while other
telemetry (voltage, temperature) continued updating normally and the bus
still responded to ping. This is a genuine hardware fault in that unit's
position sensor/encoder, not a software or communication issue -- it
explains most of that servo's confusing behavior earlier in this file
(commands appearing to "stall" at inconsistent positions, register vs.
present mismatches) since every write-verify check was comparing against
meaningless frozen feedback the whole time. The old unit was physically
replaced.

**New unit's calibration**, via a proper commanded sweep (small-step,
present-position-verified, not just goal-register write-verify) -- this
servo tracked cleanly and monotonically at every single step, a stark
contrast to the old unit:

- Bare-servo (pre-assembly) true range: -96.68 deg to +148.83 deg (raw),
  center 26.07 deg.
- **Assembled-in-finger** true range (torque-free hand sweep, logged at
  10Hz, same method used for the original 9-servo calibration): **-62.40
  deg to +88.18 deg (raw)**, notably narrower than the bare-servo range
  (expected -- the finger housing/linkage constrains travel further).
  True assembled center: **12.89 deg**.
- `FINGER_JOINTS[2]["aa"]` updated to `(6, 12.89, (-75.29, 75.29))` --
  zero_deg=12.89 (the assembled true center), range = measured relative
  range intersected with the declared +/-85 contract. This finally gives
  finger 2's aa a genuinely symmetric bidirectional range, unlike the old
  unit which could only ever reach 0 to +85 (its zero sat at its own
  floor) -- see the now-superseded entries above this section for that
  unit's whole troubleshooting history.

## Servo 8

- Zero reference (raw): **76.7578125** (hand-positioned)
- EEPROM min/max_angle_limit reset to factory full range (0, 1023 raw ->
  -149.70703125 to 76.171875 deg).
- Note: like servo 7, this zero sits right at/outside the standard reset
  max_angle_limit (76.171875) -- another unit whose usable window edges
  right up against (or past) the generic factory range.
- Hand-positioned points:
  - One hardstop, target -18 deg (relative, rough) -> measured
    **55.66 raw (relative -21.09)**.
  - Other hardstop, target ~+92 deg (relative, rough) -> measured
    **149.70703125 raw (relative +72.95)** -- exact same raw value as
    servo 7's analogous "~92 deg" extreme. Likely a shared physical/
    electrical rotation limit common to this servo type (both units hit
    it exactly), not a coincidence.
