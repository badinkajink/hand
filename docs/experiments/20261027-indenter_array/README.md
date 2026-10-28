# Indenter array with three homing keys for the UR5 cell

One PLA print puts the four indenters and three homing keys on a single base (owner, 2026-10-27). It is 220 x 236 mm,
180 mm tall and 1122.5 cm³ solid. The robot homes on every key on cb3t's registration with at least 42.2 mm of robot
clearance and 18.7 mm at the HEX-E. In front, the 5 mm sphere and the 12.5 mm half rod have their tops at 120 mm. At the
back, a 12.5 mm sphere and the fingertip holder's two stations have their tops at 180 mm. The keys are mount v3's wedge key,
which homed with less scatter than the walled seat of the tall indenter and the holder (owner, 2026-10-27).

## Terms and metrics

- Frame A: z up from the base's underside, x away from the robot, origin at the base's centre. The checks put A where
  cb3t's registration (`lab/20261026-134522-cb3t`) put the low sphere's frame, with 2x4s under the whole base.
- Key: the drafted key of `ur5_calibration_mount.indenter` on a post. Its long faces seat in the mount's socket (8 x 18 mm
  at the mouth, 20 deg draft), its ends stand 0.15 mm short of the socket's, and the palmar face stays 1 mm over the post.
  A key's yaw is the turn of its long axis from +x about z; at homing the tool axis lies along it.
- Mouth height: the height of the socket's mouth above the base's underside at homing, 30 mm.
- Clearance: the least distance (mm) from a group of parts (robot links, mount, finger, pad, HEX-E) to the array, the
  held tips and the 2x4s, as `ur5_cb3_cell.clearance`. A pose passes with joint 5 in 34 to 98 deg, joint 6 in -65 to
  +50 deg, the cable port 15 mm from the arm and the margins of `tall_indenter.NEED` (robot 8, HEX-E 6, mount 3, finger 0.8,
  pad 0.3 mm).
- Press: the pad's apex 1.5 mm off an indenter's surface and then 1 mm into it, the pad's normal along the surface normal,
  at twists -20 to +40 deg in 10 deg steps. Polar angle is the surface normal's angle from vertical. A point counts as
  pressed when at least one twist passes. Pad points within 12 mm of the apex are checked against the array without the
  pressed tip.

## Layout

| Item | Centre x, y (mm) | Top (mm) | Shape |
|---|---|---|---|
| Key 1 | -95, -80 | 34.5 | yaw -30 deg |
| Key 2 | -95, 0 | 34.5 | yaw 0 deg |
| Key 3 | -95, 80 | 34.5 | yaw +15 deg |
| 5 mm sphere | -15, -42 | 120 | neck r 3 mm, 15 deg cone to r 24 mm |
| Half rod, R 12.5 mm | -15, 45 | 120 | 20 mm long along y on a 25 x 20 mm block, cone to r 24 mm |
| 12.5 mm sphere | 70, -45 | 180 | neck r 7.5 mm, 15 deg cone to r 26 mm |
| Fingertip holder | 70, 50 | 180 (held apexes) | both stations up, turned 0 and 180 deg (`ur5_cb3_cell.TIPHOLD`); each station block a 21 x 21 mm pillar from the base |

The 12.5 mm sphere replaces the tall indenter's second 5 mm sphere. It has the rod's radius, so sphere and cylinder
contact can be compared at one radius, and two sphere radii at one shape. The yaws are the ends of the ranges where joint 5
stays inside its window at each key: -30 to 0 deg at y -80, -15 to 0 deg at y 0 and 0 to +15 deg at y 80. The three long
axes span 45 deg, so the keys fix x as well as y and yaw.

## Homing on each key

| Key | Joints (deg) | Robot | Mount | Finger | Pad | HEX-E |
|---|---|---|---|---|---|---|
| 1 | 53.9, -76.6, 124.5, -48.4, 86.4, 1.4 | 42.2 | 6.5 | 34.5 | 30.5 | 18.7 |
| 2 | 51.4, -78.9, 127.6, -49.0, 53.8, 1.5 | 42.2 | 8.7 | 19.0 | 24.3 | 27.0 |
| 3 | 57.1, -79.4, 128.3, -49.1, 44.6, 1.4 | 42.2 | 7.7 | 26.0 | 30.5 | 24.7 |

Clearances are in mm, with the other two keys present and the seated key's own contact left out.

## Presses on each indenter

| Indenter | Points | Pressed | Top | Polar 30 deg | Polar 60 deg | Polar 90 deg |
|---|---|---|---|---|---|---|
| 5 mm sphere | 25 | 13 | 1/1 | 8/8 | 4/8 | 0/8 |
| 12.5 mm sphere | 25 | 13 | 1/1 | 8/8 | 4/8 | 0/8 |
| Half rod | 5 | 2 | 1/1 | +45 deg 1/1, -45 deg 0/1 | | 0/2 |
| Held tips, W 2 and 30 mm | 4 | 4 | | | | |

Joints 5 and 6 left their windows in 120 of the 5 mm sphere's failed poses and in 113 of the 12.5 mm sphere's. The
array itself stopped 7 and 14 of their poses, through the HEX-E, mount or robot. The rod's side toward the robot fails on
the HEX-E. Four variants of the layout changed the 5 mm sphere's count by 3 points at most: mouth height 50 or 30 mm, tops
100/160 or 120/180 mm (`data/20261027-indenter_array_press_{a,b,c,d}.json`; the 12.5 mm sphere and rod rows of those runs
used a 4 mm contact zone on the pad).

## Print and fitting

- One P1S plate, base down, PLA, without supports: the holder's station blocks stand on pillars to the base (owner, 10-27).
  The part was not sliced here.
- Six wood screws (4 x 25 mm) through the counterbored holes into the 2x4s, with the keys' edge toward the robot.
- The held tips go on as in `docs/experiments/20261027-fingertip_holder/README.md`: M2 screws through the cassette and the
  board, forming their threads in the 1.9 mm holes. The holes are blind, 10 mm deep from the block's top; M2 x 8 reaches
  4.1 mm into the block.
- Homing: seat the mount on each key in turn and record the flange pose; each key gives the array's pose.

## Future work

- Registration over several keys: fit A's pose to the three homing poses and report each key's residual. The registration
  and the plan builder read one key today, and `ur5_cb3_cell` has no indenter kind for the array (it is defined in
  `indenter_array._setup`).
- The full reach map of `tall_indenter.py reach` on each indenter of the array.
- The 180 mm columns' lateral compliance under a press (the tall indenter's was 0.0040 to 0.0050 mm/N at 160 mm, solid).

## Files

- `scripts/indenter_array.py` (build, homing, press)
- `cad/20261027-indenter_array.stl`
- `20261027-indenter_array_parts.json`
- `data/20261027-indenter_array_homing.json`, `data/20261027-indenter_array_press.json`
- `media/20261027-indenter_array_views.png`
