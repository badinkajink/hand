# CAD of the DIP link, fingertip and bench fixtures, version 2

This folder has the parts made since `assets/cad_v1`, in the versions used on 2026-10-26. The DIP link b v4 and its
fingertip parts replace the original DIP link and fingertip. The bench fixtures are the UR5e calibration mount v7 and the
parts that calibrate an ATI Mini45 or Mini40 on the UR5e. Every rigid part has an exact STEP file for CAD and an STL. The
STEP files are made of planes, cylinders and cones, with a few B-spline faces; the TPU pad has its print STL only. The other
finger parts (yaw and MCP links, carriage, SCS0009 servo) are unchanged and stay in [`assets/cad_v1`](../cad_v1).
[`manifest.json`](manifest.json) gives each file's source in `docs/experiments/` and its SHA-256. [`checks.json`](checks.json) compares the STEP files
built for this folder with the meshes they replace.

## DIP link and fingertip

The design is the 3T set that the owner kept on 2026-10-23. The STEP files are in frame D (Frames, below); the STLs are
oriented for printing, with the part's ID engraved.

| Files | Part | Material and printer | Page |
|---|---|---|---|
| `dip_link_v4b.step`, `dip_link_v4b_print.stl` | DIP link b v4 | PLA, Bambu Lab P1S | [DIP link b v4](../../docs/experiments/20261021-dip_link_b_v4/20261021-dip_link_b_v4_and_wall_heights.html) |
| `cassette_3T.step`, `cassette_3T_print.stl` | Cassette 3T (cover I15H3 3T on the pages) | PLA, P1S | same page, section "Chosen heights and tighter rails" |
| `top_cover_TW15H3.step`, `top_cover_TW15H3_print.stl` | Top cover TW15H3 (tip cap on the pages) | PLA, P1S | same section |
| `pad_B05R_print.stl` | Fingertip pad B05R | Siraya Tech 85A on the Prusa XL; magnets at pauses 7.3, 12.2 and 17.1 mm | same section |

Four M2 x 8 screws fasten the set. The fit-test plates and print settings are in the
[DIP runbook](../../docs/experiments/20261018-dip_link_board_mount/README.md).

### Changes from the original CAD

The original DIP link, [`fingertip_mount_DIP_link.stl`](../cad_v1/fingertip_mount_DIP_link.stl), ends 37.16 mm from the PIP
axis. Its fingertip is a TPU block, [`fingertip.stl`](../cad_v1/fingertip.stl), 17.0 x 14.8 x 21.41 mm with 2.7 mm fillets,
that slides onto the link's mount ([CAD finger model](../../docs/experiments/20261013-cad_v1_finger/20261013-cad_v1_finger_model.html)).

The redesign of 2026-10-18 brings the ReSkin 5X magnetometer board as close to the pad as possible and fixes it so that it
cannot move (owner: "it's important that the board does not wiggle at all"). The two clevis arms of the original are kept
unchanged. One is the horn-side arm with the 2.0 mm slot for the SCS0009 horn, the other the slotted arm with its 5.2 mm
bore. Everything distal of the arms is new:

- A tongue with a seat pocket for the board (20 x 20 mm of 0.6 mm FR-4 with five MLX90393 chips). The board slides in along
  the finger, its Qwiic connector J1 first, and J1 stays on the board. The Qwiic cable leaves in a groove past the slotted arm
  to an anchor 5.0 mm below the PIP axis.
- A link 9.75 mm longer, so that J1, its plug and the cable fit: the PIP axis lies 46.91 mm from the tip's end.
- In v4, a recessed bridge edge (at x_F = -6.46 mm) and a narrower column under the pad. A flat object meets the pad 4.0 mm
  before any rigid part (v3: 0.0 mm). The joint still stops on the horn-side arm at 93.0 deg.

The fingertip is now three parts on the tongue:

- The cassette 3T clamps the board onto the tongue with the four screws, whose threads form in the PLA. Its 3 mm side walls
  run along the pad's flanks. Its dovetail rails press 0.10 mm into each wall of the pad's grooves at the cassette and
  0.16 mm at their top. The pad is locked in and stands slightly proud of the cassette; the owner kept this fit for the
  retention.
- The pad B05R is a half-ellipsoid of TPU, the 11/22 tip of the curved-tip study, with five 3 x 2 mm magnets set at print
  pauses. It slides onto the rails from the tip end.
- The top cover TW15H3 closes the tip end. It has an end plate and 3 mm side blocks along the pad's distal flanks. The two distal
  screws pass through its ears, and its 2.0 mm pin enters the pad's 1.6 mm hole.

The concepts, their force resolution and the cable route are on the
[DIP page](../../docs/experiments/20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html); v3's side
walls and column are on the [v3 page](../../docs/experiments/20261020-ur5_calibration_mount_v2/20261020-dip_link_b_v3_and_ur5_mount_v2.html).

## UR5e calibration mount (`ur5_mount/`)

The mount fixes a DIP link, with its board and pad, to the OptoForce HEX-E force sensor on the lab's UR5 (CB3). The robot
presses the pad onto an indenter fixed in its cell while the HEX-E measures the force. The board's readings are then
calibrated against that force ([protocol](../../docs/experiments/20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_protocol.html)).
The robot has run mount v7 since 2026-10-25. Its STEP and STL files are in frame M.

| Files | Part | Version | Page |
|---|---|---|---|
| `adapter.step`, `.stl` | Adapter | v3, inserts of 2026-10-22 | [mount v3](../../docs/experiments/20261021-ur5_calibration_mount_v3/20261021-ur5_calibration_mount_v3.html), section "Brass inserts in the adapter" |
| `standoff_fork_v7.step`, `.stl` | Stand-off fork | v7, 2026-10-25 | no page; records in [`20261025-ur5_calibration_mount_v7/`](../../docs/experiments/20261025-ur5_calibration_mount_v7) |
| `clamp_plate.step`, `.stl` | Clamp plate | v3 | [mount v1](../../docs/experiments/20261018-ur5_calibration_mount/20261018-ur5_calibration_mount.html) |
| `indenter_sphere.step`, `.stl`; `indenter_rod.step`, `.stl` | Indenters | v3 | mount v1 |

- The adapter is 15 mm of PLA on the HEX-E's tool plate, fastened by four M6 x 20 screws. Four brass M3 heat-set inserts
  (5.7 mm long, in 4.0 mm bores) take the fork's foot or the base of an ATI stack.
- The stand-off fork grips the DIP link by its clevis arms. An inner block sits between the arms and a 1.6 mm key enters the horn
  slot. Walls surround both legs. In v7 a solid back below the dorsal ear presses on the tongue's back from z_F = -97.5 to
  -83.5 mm with 0.2 mm of interference. Its stiffness at the contact is 141.5 N/mm by a beam estimate. Mounts v4 to v6 (a stop
  plate with set screws, a wedge, a thin flexing plate) are superseded.
- The clamp plate presses the horn-side arm onto the fork's inner block with two M3 x 16 screws. An M3 x 12 screw with a DIN
  9021 washer pulls the slotted arm onto the block's other face.
- The indenters are a 5 mm sphere and a 12.5 mm half rod, printed in PLA and fixed in the cell.

## ATI Mini45 and Mini40 calibration fixtures (`ati_mini40_45/`)

These parts calibrate one of the lab's ATI sensors, the Mini45 (FT11452) or the Mini40 (FT16552), on the UR5 with a weighed
mass. A calibrated ATI would replace the HEX-E as the reference force
([calibration plan](../../docs/experiments/20261023-ati_ft_calibration/20261023-ati_ft_calibration_plan.html)). The stack
sits on the adapter, coaxial with the tool axis, and was drawn to ATI's drawings 9230-05-1094 (Mini45) and 9230-05-1278 and
-1314 (Mini40). The STEP and STL files of the stack are in frame M.

| Files | Part | Page section |
|---|---|---|
| `mini45_base`, `mini45_top` (`.step`, `.stl`) | Stack around the Mini45 | "The stack on the HEX-E" |
| `mini40_base`, `mini40_top` | Stack around the Mini40 | "The Mini40's interface and stack" |
| `hook_cross` | Hook cross with five eyes | "The hook cross and the hanging mass" |
| `mass_cup`, `mass_cap` | Holder for the lab's 100 g knob weight | same section |

- The base screws into the adapter's inserts with four M3 x 20 screws. The sensor's mounting side screws onto the base from
  below with M3 x 12 screws, six on the Mini45 and three on the Mini40.
- The top screws onto the sensor's tool side with M3 x 10 screws. Never use longer ones: that thread is 3.0 mm deep. Four
  brass M3 inserts in the top take the hook cross or the fork. The stack lengthens the tool by 46.7 mm (Mini45) or 43.27 mm
  (Mini40).
- The mass hangs from one eye of the cross at a time, on a thread knotted through the eye. Use thread: a 0.5 mm copper wire
  bends over about 18 mm and moves the line of action off the eye.
- The holder is a cup with crush ribs and a cap with a pocket for the weight's 14 x 11 mm knob and a 2 mm tie hole. Two
  M3 x 12 screws join them.

## Frames

- Frame D, used for the DIP-link STEP files, is the coordinate system of `assets/cad_v1/fingertip_mount_DIP_link.stl`. The PIP
  axis is its y axis and x points to the palmar side. z runs along the finger; the tip's end is at z_D = 46.91 mm. The
  new link replaces the old one in an assembly without moving it. The link, cassette and top cover are placed as assembled.
- The pipelines and pages use frame F, with x_D = -x_F, y_D = y_F + 21.25 mm and z_D = -z_F - 51.55 mm.
- Frame M is used for the mount and the ATI stack. Its origin is at the centre of the adapter's face on the HEX-E, z runs
  along the tool axis toward the fingertip and x points to the palmar side. On the v7 fork, frame D sits at
  (0, -21.25, 57.0) mm in M with the same axes, so the PIP axis is 57.0 mm from the adapter's face. With an ATI stack under the fork, add the stack's 46.7 or
  43.27 mm to z.
- The indenters are drawn in the table frame and the mass holder in its own frame.

## Accuracy of the STEP files

- The DIP link's arms come from the original STL. Each arm is a prism along the PIP axis between a few steps, so each band was
  fitted from its middle section with lines, arcs and circles. No section vertex lies more than 2 um from its fitted curve.
  The bands were extruded and fused with the exact new body by the concept b pipeline. The link has 481 faces, 268 planes and
  213 cylinders; the pipeline's own STEP had 6,866, of which 6,608 were triangles. 99 % of the surface lies within 0.0014 mm of
  the pipeline's mesh. The exception is a round of about 1 mm on the lower inner edge of the horn-side arm. That edge is sharp
  in this file, so up to 0.37 mm of extra material sits there, over 0.12 % of the surface.
- The cassette and top cover are the pipeline's exact STEP files, moved into frame D. Their walls follow the pad's
  elliptical outline in flat facets, as the pipeline draws them, and the engraved IDs are left out.
- The pad has no STEP file. Its dome is an ellipsoid, and it prints from the STL. A faceted STEP of the nominal pad is
  `docs/experiments/20261018-dip_link_board_mount/v4/b/cad/20261021-dip_link_v4b_pad_ell11_5.step.zip`.
- The v7 fork comes from the mount script's exact export of 2026-10-26. Its walls around the legs are flat facets, 349 of its
  383 faces, cut around the link's legs.
- The ATI parts were built for this folder by `ati_stack_mount`'s own geometry functions on FreeCAD Part. They lie within
  0.019 mm of the meshes, which is the meshes' own chord error.

## Importing into Onshape

STEP is the route; no import into Onshape keeps a feature history.

- Import the `.step` files themselves through Insert new tab, then Import, selecting several at once. Each becomes a Part
  Studio, or one Part Studio with "Combine to a single Part Studio".
- Onshape stores a zip of STEP or STL files untranslated, with the message seen on 2026-10-18.
- Leave "Orient imported models with Y Axis Up" off: the files are in millimetres with z up.
- The imported parts keep their planes and cylinders, so mate connectors snap to the PIP bore's axis and to the faces.
- Imported solids can be edited with Move face, Replace face, Delete face and Modify fillet.
- After a part is regenerated, update its import tab with the new file; Onshape then keeps the references where it can.
- An STL imports as a mesh part. It can be placed and mated at points on the mesh but not converted to a solid.

Parametric parts in Onshape would mean porting each generator to FeatureScript, or writing features through the REST API's
`addPartStudioFeature`. In Inventor, open a STEP file with File, Open, then Options, Convert Model.

## Regenerating this folder

`scripts/cad_v2_assets.py` builds and copies everything. Run each step through `~/.claude/bin/resguard.sh run --mem 4G --`.

    ~/miniconda3/envs/femtip/bin/python scripts/cad_v2_assets.py arms
    env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/bin:/bin CADV2_CMD=step CADV2_REPO=$PWD freecadcmd scripts/cad_v2_assets.py
    env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/bin:/bin UR5M_REV=v7 UR5M_ONLY=standoff_fork UR5M_CMD=step UR5M_REPO=$PWD \
        freecadcmd scripts/ur5_calibration_mount.py
    ~/miniconda3/envs/femtip/bin/python scripts/cad_v2_assets.py check
    python3 scripts/cad_v2_assets.py copy

`python3 scripts/cad_v2_assets.py copy --check` lists the copies whose source in `docs/experiments/` has changed since. The
`arms` step reads the arms as the concept b pipeline cut them, `logs/20261021-dip_link/v4/b/arm_*.stl`; the v4 build in the
[DIP runbook](../../docs/experiments/20261018-dip_link_board_mount/README.md) writes them again.
