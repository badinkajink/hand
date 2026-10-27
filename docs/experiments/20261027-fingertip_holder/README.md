# Fingertip holder for the crosstalk test on the UR5 cell

The fingertip holder is a one-piece PLA print, 52 mm tall and 142.7 cm³, against the tall indenter's 391.7 cm³. It fixes
a second v2 fingertip (pad, cassette 3T and top cover TW15H3, no board) in the cell by the tip's four M2 screws. Their
threads form in 1.9 mm holes that pass through a station block, so the tip faces up from the block's top face or down from
its underside. In the reach check on mount v7 and cb3t's registration, v2 ID 1 reaches every facing gap from 0.5 to 30 mm
at twists of −20 to +10 deg. Side by side, with the two apexes level, it reaches gaps of 13 mm and more at twist 0, and 15
to 20 mm at twists of −10 to +10 deg.

Below 13 mm side by side, the v7 fork's back meets the station block. That back covers v2 ID 1's tongue to the tip's end
and stands 2.8 mm wider than the cassette. The homing seat is the tall indenter's: three raised flats, the key at 0.05 mm
on every face and guide walls.

Builder: `scripts/fingertip_holder.py`, from the numbers in `ur5_cb3_cell.TIPHOLD` and `TIP_F`, which the cell's
clearance checks read too.

![The holder and the test poses](media/20261027-fingertip_holder_poses.png)

*Figure 1. Left, the holder with the held tip up. Middle, v2 ID 1 (blue, its fork cut at z_M 60) 2 mm above the held tip
(orange). Right, side by side at 13 mm with the held tip down; the fork's back over v2 ID 1's tip is level with the
station block.*

## Terms and metrics

- v2 ID 1 is the tip on the robot's mount, whose board streams. The held tip is the second v2 tip on the holder.
- Frame I has z up from the 2x4's top and x along the tool axis toward the tip at homing, as for the tall indenter. Frame
  F is the DIP link's frame of `assets/cad_v2/README.md`; the cassette's back face lies at x_F = 3.80 mm, where the
  board's component face lies on the finger.
- The apex is the centre of a pad's face.
- W is the gap between the two apexes with the pads facing (held tip up).
- g is the gap between the two pads' flanks with the tips side by side and their apexes level (held tip down). The pads
  are 14.8 mm wide, so their centre lines lie 14.8 + g mm apart. v2 ID 1 lies at −y_I of the held tip, the side of the
  crosstalk budget's poses. s moves v2 ID 1 toward its tip end, in mm.
- A twist turns v2 ID 1 about the vertical through its apex, in deg. At 0 both tips point along +x_I.
- A pose passes when two conditions hold. First, the IK finds a solution near cb3t's homing branch with joint 5 within
  34 to 98 deg and joint 6 within −65 to +50 deg (the HEX-E's cable), and the cable port stands 15 mm from the arm. Second,
  these clearances hold to the holder, the held tip, the 2x4, the table, the rod fixture and the robot: mount 3.0, finger
  0.8, HEX-E 6.0, robot links 8.0 and pad 0.3 mm. These are the tall indenter's checks, with the held tip as an obstacle.

## The part

| Feature | Value |
|---|---|
| Station block | 21.3 x 21.0 x 12 mm; top face 52 mm and underside 40 mm above the 2x4; face centre at (0, −28.5) mm in I |
| Holes | Four of 1.9 mm through the block in the cassette's pattern: ±8.0 mm across the tip, 8.10 mm toward its base and 7.90 mm toward its end from the face's centre |
| Held tip's apex | 66.35 mm above the 2x4 with the tip up, 25.65 mm with it down |
| Arm and column | The block's +y side joins a column of radius 14 mm, on a 45 deg gusset |
| Homing seat | The tall indenter's seat; its mouth plane 30 mm above the 2x4, 22 mm toward the robot and 52 mm to +y; on a tower from the base |
| Base plate | 80 x 106 x 8 mm, four 4.6 mm holes counterbored 9 mm, 3 mm deep |
| Volume | 142.7 cm³ (177 g if solid); the STEP reads back as one valid solid of the same volume |

The block ends 0.1 mm short of the cassette's end. There the top cover's end plate reaches 4.2 mm past the cassette's
back, over the block's end face.

## Mounting the tip

The pad, cassette and top cover come off the finger as one unit. On the finger the cassette clamps the board onto the
tongue. On the holder it clamps onto the block, whose face lies where the board's component face lies. The two proximal
screws pass through the cassette, the two distal ones through the top cover's ears and the cassette, as on the finger.
With the tip up, the screws enter the top face from above; with the tip down, they enter the underside from below. An
M2 x 8 reaches 4.1 mm into the block, and the block is 12 mm thick, so the threads cut from the two faces stay apart.

The cassette's holes are 2.4 mm across, so each M2 screw has 0.2 mm of play. The held tip's place on the block is known
from the CAD to about ±0.2 mm.

## Reach of the test poses

Pads facing, held tip up:

| W (mm) | Twists that pass (deg) |
|---|---|
| 0.5, 1, 2, 3, 5, 10, 15, 20, 30 | −20, −10, 0, +10 |

At +20, +30 and +40 deg, joint 5 falls to 28.3, 16.5 and 5.7 deg, under its 34 deg limit. At W 0.5 mm and twist 0 the
least clearances are mount 9.6, finger 4.6, HEX-E 53.4 and robot 72.6 mm. Joint 5 is then at 54.9 deg and joint 6 at 1.5
deg.

Side by side, held tip down; the twists that pass (deg) at each shift s:

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
26.35 mm above the held apex, so with the apexes level the fork's back is level with the block. The mount clears the block by
g − 9.0 mm at twist 0, and the 3 mm margin is met from g = 12 mm. At g 25 mm and twist +10 deg, the fork comes within
1.4 mm of the rod fixture's homing post, 114 to 128 mm along −y_I.

The crosstalk budget (`../20261027-fingertip_crosstalk/data/budget.json`) gives the field at v2 ID 1's chips for these
poses. With the pads facing, a five-magnet neighbour gives 513 uT at W 0 and 46 uT at 30 mm. Side by side, it gives
367 uT at 10 mm and 102 uT at 20 mm, and four magnets give 88 and 8 uT. A press of 0.4 to 0.7 N gives 23 uT.

## Homing

At homing, cb3t's branch puts the joints at (64.0, −72.2, 118.3, −46.4, 66.4, 1.4) deg. The robot, the HEX-E, the finger
and the pad clear the holder by 42.2, 20.2, 6.6 and 14.6 mm, with the held tip up or down. The mount sits at −0.05 mm,
the key's fit in the socket. Homing registers the holder, so the tip can move between the two faces without homing
again.

## Bill of materials and print

- The holder, `cad/20261027-fingertip_holder.stl` (STEP beside it), in PLA, base on the bed. Only the station block's
  underside (21 x 21 mm, 40 mm up) needs support; the gusset under the arm leans 45 deg. The 1.9 mm holes print
  vertically, as on the DIP link.
- Four wood screws, 4 x 30 mm, pan head, at (30, −20), (−30, −20), (30, 60) and (−30, 10) mm in I. In every test pose
  that passes they stand at least 43.7 mm from v2 ID 1's board.
- The held tip's own four M2 x 8 screws.

## Fitting it to the 2x4

1. Take the sphere fixture off and put the holder where it stood, centred across the 2x4, with the seat tower toward the
   robot and the station toward the rod fixture.
2. Drive the four screws snug in a cross pattern, then tighten them.
3. Home after fastening: in teach mode, seat the socket on the key with the palmar side down, press it down onto the
   three flats, end teach mode and record the seating (`record_pose.py`).
4. Screw the held tip onto the top face for the facing poses, or under the block for side by side.

## Future work

- The test's plan: facing at W 0.5 to 30 mm (twists −20 to +10 deg) and side by side at g 13 to 20 mm, with pads 05R
  and 05R4 on the held tip in turn and v2 ID 1's board streaming. The held tip's six pose parameters are fitted from the
  readings with the field model. A residual above 1 % of the field at held-out poses falsifies the model at that accuracy.
- A seat check on the robot: ten seatings by hand, each recorded. A spread in roll under 0.15 deg and along the tool axis
  under 0.05 mm confirms the flats and the key's fit.

## Files

- `cad/20261027-fingertip_holder.step`, `cad/20261027-fingertip_holder.stl`: the part in frame I, mm.
- `20261027-fingertip_holder_parts.json`: the resolved geometry, the volume and the STEP's read-back.
- `data/20261027-fingertip_holder_reach_up.json`, `data/20261027-fingertip_holder_reach_down.json`: the test poses with
  their joints and clearances; `data/20261027-fingertip_holder_homing.json`: the homing pose.
- `media/20261027-fingertip_holder_poses.png`: Figure 1.
