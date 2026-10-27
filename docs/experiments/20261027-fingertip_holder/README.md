# Fingertip holder for the crosstalk test on the UR5 cell

The fingertip holder is a one-piece PLA print, 52 mm tall and 172.9 cm³, shaped as a T. Two v2 fingertips (pad, cassette
3T and top cover TW15H3, no board) sit on station blocks opposite each other on one column. The tip on the −y station faces
down for the side-by-side poses, and the tip on the +y station faces up for the facing poses. A night run then covers both
without a change of tips. In the reach check on mount v7 and cb3t's registration, v2 ID 1 reaches every facing gap from
0.5 to 30 mm at twists of −20 to +20 deg (45 of 45 poses). Side by side, with the two apexes level, it reaches gaps of
13 mm and more at twist 0, and 15 to 20 mm at twists of −10 to +10 deg.

The homing seat locates the fork more closely than the tall indenter's. Its walls stand 9.5 to 14 mm tall at 0.2 mm from the
fork, where they stood 4 mm tall at 0.4 mm. A stop faces the fork's palmar end at 0.1 mm. The key's edges are rounded to
0.8 mm to clear the printed socket's rounded corners, and the key reaches 5.0 mm into the 5.3 mm socket. The other
station's tip adds 4.0 to 13.4 uT at v2 ID 1's chips when it has five magnets, against 46 to 483 uT from the tip under
test, so the test's field model includes both held tips.

Builder: `scripts/fingertip_holder.py`, from the numbers in `ur5_cb3_cell.TIPHOLD` and `TIP_F`, which the cell's
clearance checks read too.

![The holder and the test poses](media/20261027-fingertip_holder_poses.png)

*Figure 1. Left, the holder with both held tips (orange) and the seat on its tower behind. Middle, v2 ID 1 (blue, its fork
cut at z_M 60) 2 mm above the +y station's tip. Right, side by side at 13 mm with the −y station's tip; the fork's back
over v2 ID 1's tip is level with the station block.*

## Terms and metrics

- v2 ID 1 is the tip on the robot's mount, whose board streams. A held tip is a v2 tip on one of the holder's two stations.
- Frame I has z up from the 2x4's top and x along the tool axis toward the tip at homing, as for the tall indenter. Frame
  F is the DIP link's frame of `assets/cad_v2/README.md`; the cassette's back face lies at x_F = 3.80 mm, where the
  board's component face lies on the finger. Frame M is the mount's, with z_M along the tool axis.
- The apex is the centre of a pad's face.
- W is the gap between the two apexes with the pads facing, on the +y station.
- g is the gap between the two pads' flanks with the tips side by side and their apexes level, on the −y station. The
  pads are 14.8 mm wide, so their centre lines lie 14.8 + g mm apart. v2 ID 1 lies at −y_I of the held tip, the side of
  the crosstalk budget's poses. s moves v2 ID 1 toward its tip end, in mm.
- A twist turns v2 ID 1 about the vertical through its apex, in deg. At 0 all tips point along +x_I.
- A pose passes when two conditions hold. First, the IK finds a solution near cb3t's homing branch with joint 5 within
  34 to 98 deg and joint 6 within −65 to +50 deg (the HEX-E's cable), and the cable port stands 15 mm from the arm. Second,
  these clearances hold to the holder, both held tips, the 2x4, the table, the rod fixture and the robot: mount 3.0, finger
  0.8, HEX-E 6.0, robot links 8.0 and pad 0.3 mm. These are the tall indenter's checks, with the held tips as obstacles.
- The field is the norm over v2 ID 1's 15 chip components, in uT, from the crosstalk budget's model
  (`scripts/fingertip_crosstalk.py`): a five-magnet tip with v2 ID 1's poles, or four N52 magnets in a 5.0 mm square.

## The part

| Feature | Value |
|---|---|
| Station blocks | Two, 21.3 x 21.0 x 12 mm, their faces' centres at y = ±28.5 mm; top faces 52 mm and undersides 40 mm above the 2x4 |
| Holes | Four of 1.9 mm through each block in the cassette's pattern: ±8.0 mm across the tip, 8.10 mm toward its base and 7.90 mm toward its end from the face's centre |
| Held tips' apexes | 66.35 mm above the 2x4 on the +y station (tip up), 25.65 mm on the −y station (tip down) |
| Arms and column | Each block's inner side joins a column of radius 14 mm, on a 45 deg gusset |
| Homing seat | Mouth plane 30 mm above the 2x4, 22 mm toward the robot and 75 mm to +y; on a tower from the base |
| Base plate | 80 x 128 x 8 mm, four 4.6 mm holes counterbored 9 mm, 3 mm deep |
| Volume | 172.9 cm³ (214 g if solid); the STEP reads back as one valid solid of the same volume |

Each block ends 0.1 mm short of its cassette's end. There the top cover's end plate reaches 4.2 mm past the cassette's
back, over the block's end face.

## Homing seat

The seat keeps the tall indenter's three raised flats, 4 mm square and 0.6 mm proud of the floor, under the fork's palmar
face. They set the depth, the roll and the pitch. The other features changed as follows.

- The key keeps the socket's shape (8 x 18 mm at the mouth, drafted 20 deg) at 0.05 mm inside each face. It now reaches
  5.0 mm into the 5.3 mm socket, against 4.5 mm. The socket in the CAD has sharp corners, but a printed pocket's corners
  come out rounded. So the key's four edges along the socket's axis are rounded to 0.8 mm, and its faces bear before its
  corners.
- The walls stand 0.2 mm from the fork's sides, against 0.4 mm. Along the column's narrow part (z_M 31 to 45) they are
  14 mm tall. Above z_M 46.5 the −y wall is 11.5 mm tall, below the DIP link's horn-side arm, and stops at z_M 59.5
  before the clamp plate. The +y wall is 9.5 mm tall there, below the slotted arm, and runs on to the stop.
- The stop is a wall 4 mm thick and 9.5 mm tall. It faces the end of the fork's palmar column (z_M 72.95) at 0.1 mm,
  across the full width, and joins the two walls.

At homing, the seat check samples the fork, the clamp plate and the DIP link every 0.3 mm. The slotted arm's M3 head and
DIN 9021 washer are modelled as a 9 mm cylinder at the PIP axis. The flats touch the palmar face. The key stands 0.041 mm
inside the socket, the walls 0.195 and 0.196 mm from the fork, and the stop 0.096 mm from its end. The DIP link stays at
least 1.04 mm from the walls and the stop, and the screw head and washer 3.0 mm.

The tall indenter's walls along the column's narrow part ended at z_M 45.5, but the column widens at z_M 45.45. They
overlapped the fork by 0.05 mm there. On both parts they now end at z_M 45.0.

## Mounting the tips

The pad, cassette and top cover come off the finger as one unit. On the finger the cassette clamps the board onto the
tongue. On the holder it clamps onto a block, whose face lies where the board's component face lies. The two proximal
screws pass through the cassette, the two distal ones through the top cover's ears and the cassette, as on the finger.
On the +y station the screws enter the top face from above; on the −y station they enter the underside from below. An
M2 x 8 reaches 4.1 mm into the block, and each block is 12 mm thick, so either face of either block takes a tip.

The cassette's holes are 2.4 mm across, so each M2 screw has 0.2 mm of play. A held tip's place on its block is known
from the CAD to about ±0.2 mm.

## Reach of the test poses

Pads facing, on the +y station: all 45 poses pass, at W 0.5, 1, 2, 3, 5, 10, 15, 20 and 30 mm and twists of −20, −10,
0, +10 and +20 deg. Joint 5 is at 62 deg at twist 0 and 34.2 deg at +20 deg, its limit being 34 deg. At W 0.5 mm and
twist 0 the least clearances are mount 9.6, finger 4.6, HEX-E 53.4 and robot 72.6 mm.

Side by side, on the −y station; the twists that pass (deg) at each shift s:

| g (mm) | s 0 | s 2 mm | s 5 mm |
|---|---|---|---|
| 9 to 12 | none | none | none |
| 13 | 0 | 0 | 0 |
| 14 | 0, +10 | 0, +10 | 0, +10 |
| 15 | −10, 0, +10 | −10, 0, +10 | 0, +10 |
| 20 | −10, 0, +10 | −10, 0 | −10, 0 |
| 25 | −20, −10, 0 | −10, 0 | −10, 0 |

The limit at small g is v2 ID 1's mount. From z_M 90 to 105 mm the v7 fork's back covers the tongue to the tip's end. It
starts 18.4 mm above the apex and reaches y_M = 13.3 mm, 2.8 mm past the cassette's side. The station block spans 14.35 to
26.35 mm above the held apex, so with the apexes level the fork's back is level with the block. The mount clears the block
by g − 9.0 mm at twist 0, and the 3 mm margin is met from g = 12 mm. At g 25 mm and twist +10 deg, the fork comes within
1.4 mm of the rod fixture's homing post, 114 to 128 mm along −y_I.

## Field of the two held tips

The field at v2 ID 1's chips, at twist 0 and s 0, from the tip under test and from the other station's tip:

| Pose | Tip under test: five / four magnets (uT) | Other tip: five / four magnets (uT) |
|---|---|---|
| Facing, W 0.5 mm | 483 / 33.7 | 13.4 / 0.31 |
| Facing, W 30 mm | 46.5 / 0.35 | 7.0 / 0.08 |
| Side by side, g 13 mm | 231 / 38.9 | 6.0 / 0.06 |
| Side by side, g 25 mm | 64.9 / 3.4 | 4.0 / 0.03 |

Side by side at g 20 mm the model gives 102 uT, as in the crosstalk budget for the same pose. A five-magnet tip on the
other station gives 3 % of the field under test at the closest gaps and 15 % at W 30 mm. A four-magnet tip there gives at
most 0.31 uT. A press of 0.4 to 0.7 N gives 23 uT.

## Homing

At homing, cb3t's branch puts the joints at (67.0, −72.5, 118.8, −46.6, 69.5, 1.4) deg. The robot, the HEX-E, the finger
and the pad clear the holder and both held tips by 42.2, 20.2, 1.1 and 5.5 mm. Homing registers the holder, so the tips
can move between stations or faces without homing again.

## Bill of materials and print

- The holder, `cad/20261027-fingertip_holder.stl` (STEP beside it), in PLA, base on the bed. Only the two blocks'
  undersides (21 x 21 mm each, 40 mm up) need support; the gussets under the arms lean 45 deg. The 1.9 mm holes print
  vertically, as on the DIP link.
- Four wood screws, 4 x 30 mm, pan head, at (30, −20), (−30, −20), (30, 75) and (−30, 30) mm in I. In every test pose
  that passes they stand at least 43.7 mm from v2 ID 1's board.
- Each held tip's own four M2 x 8 screws.

## Fitting it to the 2x4

1. Take the sphere fixture off and put the holder where it stood, centred across the 2x4, with the seat tower toward the
   robot and the −y station toward the rod fixture.
2. Drive the four screws snug in a cross pattern, then tighten them.
3. Home after fastening: in teach mode, seat the socket on the key with the palmar side down, press it down onto the
   three flats and forward against the stop, end teach mode and record the seating (`record_pose.py`).
4. Screw one tip on the +y station's top face and one under the −y station's block.

## Future work

- The test's plan: facing on the +y station at W 0.5 to 30 mm (twists −20 to +20 deg) and side by side on the −y station
  at g 13 to 20 mm, v2 ID 1's board streaming, then the same with pads 05R and 05R4 exchanged between the stations. The
  six pose parameters of each held tip are fitted from the readings with the field model of both tips. A residual above
  1 % of the field at held-out poses falsifies the model at that accuracy.
- A seat check on the robot: ten seatings by hand, each recorded. A spread in roll under 0.15 deg and along the tool axis
  under 0.05 mm confirms the flats, the stop and the key's fit.

## Files

- `cad/20261027-fingertip_holder.step`, `cad/20261027-fingertip_holder.stl`: the part in frame I, mm.
- `20261027-fingertip_holder_parts.json`: the resolved geometry, the volume and the STEP's read-back.
- `data/20261027-fingertip_holder_reach_facing.json`, `data/20261027-fingertip_holder_reach_side.json`: the test poses
  with their joints and clearances.
- `data/20261027-fingertip_holder_homing.json`: the homing pose with the seat check.
- `data/20261027-fingertip_holder_farfield.json`: the fields of the two held tips.
- `media/20261027-fingertip_holder_poses.png`: Figure 1.
