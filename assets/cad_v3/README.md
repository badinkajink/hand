# CAD of the whole SR2 hand on the UR5 CB3, version 3

This folder holds the owner's Onshape export of the whole hand (2026-11-04), the frames that place it on the lab's UR5
(CB3) and an inventory of its 171 solids. The MuJoCo models built from them are in
[`../mjcf/experimental/20261104-cad_v3`](../mjcf/experimental/20261104-cad_v3), written by
[`scripts/build_cad_v3_hand.py`](../../scripts/build_cad_v3_hand.py). The analysis behind the numbers is on the result
page `docs/experiments/20261104-cad_v3_hand/20261104-cad_v3_hand_on_ur5.html`.

The hand hangs from the UR5 through the OptoForce HEX-E, which stays on the flange. The adaptor bolts to the HEX-E's tool
face, 35.0 mm out from the flange. The tool axis lies 4.2 mm off the gantries' centre line, the adaptor face is 103.9 mm
above the plane of the yaw axes, and the robot's flange is 138.9 mm above it. The adaptor has no dowel, so it bolts to the
HEX-E in four clockings k, 90 deg apart. The fingers' travel boxes in
the CAD are centred about 6 mm outside the positions the driver and the simulator have used.

## Files

| File | What it is |
|---|---|
| `step/hand_v3_anthro_config.step` | the assembly "hand v3 (anthro config)": base plate, robot mount, UR5 adaptor, three gantries, three fingers, camera bracket, tag plate |
| `step/gantry_v3.step` | the index/middle gantry with x-stepper mounts on both sides, as modelled |
| `step/gantry_v3_thumb.step` | the thumb gantry |
| `step/base_plate.step`, `step/robot_mount.step`, `step/ur5_adaptor.step` | the three plates on their own |
| `frames.json` | frames A, P and E, the four flange transforms, the travel boxes, heights, camera and tag poses |
| `inventory.json` | one row per solid: kind, rigid body, material, mass, bounding box |

The STEP files are copies of `assets/cad_v2/full_cad` (same SHA-256). Every part in them is named "Part n", so the
inventory names each solid from its volume, faces and holes. The finger parts in the export are the cad_v1 parts: the
original DIP link and the TPU block tip. The models use cad_v2's fingers with the v4b tip instead.

## Frames

- **A** is the frame of the assembly, in mm. Its z axis points up toward the robot. The base plate spans z_A 0 to 3, the
  robot mount 3 to 28, and the gantries hang below.
- **P** is the palm frame of the simulator and of the driver's {P}. Its x axis points from the thumb toward the
  index/middle pair, and its y axis from the middle finger toward the index. Its origin lies on the plane of the three
  yaw-axis centres, midway between the travel-box centres. P is A turned by +90 deg about z_A:
  x_P = -y_A + 121.44, y_P = x_A + 111.00, z_P = z_A + 75.90 (mm).
- **E** is tool0 of the UR5 CB3, the controller's flange frame. Its z axis points out of the flange into the hand. The
  flange's four M6 holes lie at 45, 135, 225 and 315 deg from x_E, and the tool connector on -y_E. Seen from the tool
  side, with z_E toward the viewer, x_E is 90 deg clockwise from y_E.
- **H** is the HEX-E's tool face, 35.0 mm out along z_E (data sheet). Its dowel was measured at -92.15 deg from x_E (lab
  README of the tip calibration, mean of two seatings, +-0.58 deg), so its four M6 holes lie at -47.15 + 90 j deg from x_E.
  Its cable port points along +y_E, opposite the robot's tool connector.

The adaptor's four 4.4 mm plate holes coincide with the robot mount's holes. Their isosceles trapezoid allows only one
way to fit the adaptor to the plate. The tool axis therefore passes through (-111.0, 116.0) mm in A, and the adaptor's
face is z_A = 28. The adaptor's M6 holes lie at -21, 69, 159 and 249 deg from x_A. In clocking k, x_A lies at
theta_k = 21.85 + 90 k deg from x_E about +z_E, and z_A = -z_E. Without the HEX-E it would be 24 + 90 k deg, with the
adaptor face on the flange.

## The four clockings

The table gives P's origin in tool0 and the UR TCP that puts the tool frame on it (PolyScope's x, y, z in mm and
rotation vector). `.venv/bin/python scripts/build_cad_v3_hand.py tcp --k K --point X Y Z` prints the TCP of any point
of P. The last two columns name the gantry that the HEX-E's cable port (+y_E) and the robot's tool connector (-y_E) point
at, and how far off.

| k | theta_k (deg) | P's origin in tool0 (mm) | rotation vector (rad) | HEX-E port faces | tool connector faces |
|---|---|---|---|---|---|
| 0 | 21.85 | (2.02, -5.05, 138.90) | (1.7602, 2.6022, 0) | index, 23 deg off | thumb, 22 deg off |
| 1 | 111.85 | (5.05, 2.02, 138.90) | (0.5954, -3.0847, 0) | between thumb and index | middle, 23 deg off |
| 2 | 201.85 | (-2.02, 5.05, 138.90) | (2.6022, -1.7602, 0) | thumb, 22 deg off | index, 23 deg off |
| 3 | 291.85 | (-5.05, -2.02, 138.90) | (3.0847, 0.5954, 0) | middle, 23 deg off | between thumb and index |

The index gantry is the one beside the tag plate and the bracket arm. In the owner's photo of 2026-11-04, the black
plug at the flange points at the thumb gantry. If that plug is the HEX-E's cable port, the hand is at k = 2; if it is the
robot's tool connector, at k = 0. k = 2 is the more likely reading and is not yet confirmed.

## Gantries

Each gantry's x stage (the beam, firmware J0, J2, J4) runs along x_A and carries the y stage (the finger's carriage, J1,
J3, J5) along y_A. Every axis homes at the end that carries its stepper. From the hard stops in the CAD, the strokes are
114.8 mm for the thumb's x stage, 64.8 mm for the index and middle x stages and 62.58 mm for every y stage. The firmware's
110 and 60 mm from home leave 4.8 and 2.58 mm before the far stops.

Anchored at the motor-end stops, the travel boxes are centred in P at:

| Finger | CAD box centre (mm) | driver's {P} (mm) | difference (mm) | box (mm) |
|---|---|---|---|---|
| thumb | (-56.18, -2.40) | (-50, 0) | (-6.18, -2.40) | 60 x 110 |
| index | (56.18, 58.40) | (50, 55) | (6.18, 3.40) | 60 x 60 |
| middle | (56.18, -58.40) | (50, -55) | (6.18, -3.40) | 60 x 60 |

These centres assume the thumb gantry sits on the base plate's holes, 2.5 mm along +y_A from where the export places
it. As exported, the thumb beam and its rail block overlap the pair's by 1.6 mm. On the holes, every overlap clears by at
least 0.9 mm.

Heights above the yaw plane (z_P, mm): UR5 flange 138.9, adaptor face 103.9, robot mount underside 78.9, base plate underside 75.9, gantry
frame lowest point 53.9, beam underside 47.4, beam bearing blocks 24.4, beam motor end 21.9, carriage bottom -12.8.

## Camera and tag

The camera bracket (`d435_plate`, solid 000) runs under the base plate's +x_A edge and ends 52.6 mm past it in a fork
for a 20 x 20 mm extrusion. The D435 is no longer on the hand (owner, 2026-11-04: it stands on a tripod on the table), so
the models carry the bracket and no camera. The tag plate (`static_tag_mount`, solid 005) carries tag36h11 id 6, a 40 mm
black square, on the face opposite the thumb's palmar side (+y_A, -x_P; owner, 2026-11-04).

## Assumed or not yet measured

- the clocking k (k = 2 is the likely reading of the owner's photo);
- the HEX-E's turn on the flange (-2.15 deg, measured with the calibration mount in October) and its 35.0 mm height
  (data sheet);
- the 0.93 deg tilt of each carriage's yaw bore against its rail-block bolt pattern in the CAD, which the cad_v1 model
  removed as an export artefact;
- the materials: the export has none, and the 2.98 kg total (without the HEX-E's 0.26 kg) uses solid PLA and datasheet
  masses.

The owner confirmed on 2026-11-04 that the thumb gantry sits on the base plate's holes and that the real hand has no
collisions between its parts. The finger links are drawn metallic silver, as the owner painted them; the fingertips and
servos are black.

## Rebuild

    env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/bin:/bin CADV3_CMD=extract CADV3_REPO=$PWD freecadcmd scripts/cad_v3_assets.py
    .venv/bin/python scripts/build_cad_v3_hand.py frames
    .venv/bin/python scripts/build_cad_v3_hand.py meshes
    .venv/bin/python scripts/build_cad_v3_hand.py scenes
    .venv/bin/python scripts/build_cad_v3_hand.py check

Run each mesh-loading step under `~/.claude/bin/resguard.sh run --mem 3G --cpu 100 --`.
