# Object seating and external tracking for SR2 hand experiments on the UR5

Plan of 2026-11-05 for a dedicated session. The owner decides the items under "Decisions for the owner". The aim is to
run the HOM study's four tasks on the real hand with the simulation's protocol: pin-push, press, roll and twist
(`scripts/hom/hom_tasks.py`). External cameras measure the object's motion.

## Summary

Each object gets a short stem below it that ends in a printed hexagonal prism, the tag block, with a 13 mm AprilTag on
each side face. The block's top lies 15 mm below the lowest finger point of the five layouts' grasps, so the stem is
35.6 mm long on the cylinder and 25.2 mm on the sphere. The object's pose is solved from all visible tag corners at
once, in the images of two D435 colour cameras mounted on their sides, with each camera's pose in the palm frame taken
from the arm. The block rests on a flat seat fixed to the table. At 1.6 s, when the simulation removes its post, the arm
lifts the hand 40 mm; after the task it lowers the hand and the fingers set the object down (owner, 2026-11-05).

The targets are 0.3 mm and 0.3° RMS for the object's pose in the palm frame, at 30 Hz, in at least 99 % of the frames
of a trial. The first series runs the five layouts of the task-conflict study, 10 trials per layout and task, against
simulations of the objects as built.

## Simulation of steps 1 to 3 (2026-11-05)

The owner reviewed the first version of this plan the same day: a seat that drops away needs parts the lab does not
have, so the arm lifts the hand instead, and the ballast had no purpose. Steps 1 to 3 then ran in simulation; the page is
`docs/experiments/20261105-object_tracking_sim/20261105-object_tracking_sim.html`. Its results changed this plan in six
places:

- The seat is flat. A pocket round the block's lower end breaks the cylinder tasks: the pin-push scores 0 at four of
  five layouts. On the flat seat the largest loss of a cylinder pick against the task-conflict study is 0.011.
- The objects have no ballast. Infill alone makes 25 g; the stem and block lower the centre of mass by 7.5 mm
  (cylinder) and 7.0 mm (sphere).
- The stems are longer, because the press's middle finger reaches 20.6 mm below the cylinder.
- A stem on the sphere works at the generalist (70,110) and the pin-push specialist (75,55) only. The HOM tripod lets
  the sphere roll, and at the other three layouts the stem touches a finger in 3 to 5 of 5 rollouts.
- The cameras stand on their sides at azimuths 0° and 260° from x_P, at the block's height in the task.
- Each camera's pose in P comes from the arm after a hand-eye calibration. With it the object is tracked to 0.03 to
  0.20 mm and 0.04 to 0.13° RMS. From one 40 mm hand tag the camera's tilt is only good to 0.1 to 1.0°, which moves
  the object by 0.3 to 5.4 mm in P.

## Terms

- P: the palm frame of `assets/cad_v3/README.md`. x points from the thumb toward the index and middle fingers, y from
  the middle toward the index, z up toward the flange.
- O: the object frame of the HOM scenes, at the object's centre. The cylinder's axis lies along y_P at the start of
  every task. The cad_v3 keyframe `over_object` lays it along x_P; the HOM convention applies here.
- $T_{AB}$: the pose of frame B in frame A, a 4 × 4 transform in mm.
- Tag block: the printed prism on the object's stem that bears the object's tags.
- Tag map: the 3-D coordinates of every tag corner of one rigid body (object, hand or seat) in that body's frame.
- Seated pose: $T_{PO}$ while the object rests in the seat.
- Frame age: the time from a camera's exposure to the arrival, at the controller, of the pose computed from that frame.
- Task scores (`scripts/hom/hom_landscape.py:48-58`) are 0 unless the object is held. Pin-push and press score
  $1-\theta_\mathrm{RMSE}/\theta_\mathrm{amp}$: the RMS error of the rotation about the pinch axis over its amplitude
  (15° and 20°). Roll scores $1/(1+\omega_\mathrm{RMSE}/40\,°\mathrm{s^{-1}})$, from the RMS error of the angular
  velocity. Twist scores the largest twist held on three pads divided by 90°.

## Requirements

| Quantity | Target | Source of the target |
|---|---|---|
| Object pose in P | 0.3 mm and 0.3° RMS | the HOM "sense" variant assumes 0.5 mm and 1° (`hom_tasks.py`, `VARIANTS`) |
| Rate | 30 Hz | the camera; the HOM controller steps at 50 or 100 Hz |
| Tracked frames in a trial | at least 99 % | the HOM controller takes the object pose at every step (`src/sr2hand/control/hom.py`, `Measurement`) |
| Frame age | 50 ms median | the "sense" variant delays every measurement by 40 ms |
| Start pose | measured before each trial; within 1 mm and 2° of nominal, else the arm moves the hand | the simulation jitters the start by up to 1 mm and 2° |
| Arm motion | a 40 mm lift at 1.6 s and its return, in each trial; nothing else within a series | the hand's cables; the lift replaces a seat that drops away (owner, 2026-11-05) |
| Ferromagnetic parts | none in the objects or the seat | the fingertip boards are magnetometers |

## Measurement chain

For each camera C the object's pose in the palm frame is $T_{PO} = T_{CP}^{-1}\,T_{CO}$. $T_{CO}$ is solved from the
tag block's corners. Both cameras enter one least-squares solve of $T_{PO}$ over the reprojection error of all block
corners (`sr2hand.tracking.pose.solve_multi`). Each camera's pose in its own image of the block is good across the
image (0.01 to 0.02 mm) and weak in depth: the detector puts corners about 0.06 px outside the true ones, which brings
the block 0.5 mm closer at 0.3 m. For two cameras about 90° apart, each one's depth is a lateral offset of the other.
Together, with their poses known, they reach 0.03 to 0.20 mm and 0.04 to 0.13° RMS per pick in simulation.

$T_{CP}$ comes from the arm. Each camera's pose in the base frame and the hand tags' poses in tool0 are solved once, by
moving the arm through about 15 poses planned with the cell model (`scripts/ur5/ur5_cb3_cell.py`; OpenCV
`calibrateRobotWorldHandEye`). In each frame $T_{CP}$ then follows from the joint readings through
`sr2hand.robot.ur_kinematics` and `sr2hand.robot.tool_chain.SR2Hand`, the lift included. The hand tags check this route:
a bumped camera shows as a jump between the two. They cannot replace it. One 40 mm tag seen nearly head-on fixes its
own tilt to 0.1 to 1.0°, and the block lies about 170 mm from it, so $T_{PO}$ from the hand tags alone was off by 0.3
to 5.4 mm in simulation.

The hand has one tag now: tag36h11 id 6, 40 mm, on the static tag mount. Its centre is at (16.1, 133.6, 43.9) mm in P,
and it faces −x_P (`assets/cad_v3/frames.json`, `sensors.tag`). One more 40 mm tag per camera goes on a static part of
the hand, facing that camera; the simulation placed them on the hand's outline 110 mm from P's z axis and 44 mm above
P. The tags' poses in P start from CAD and are refined by the same arm poses.

Both cameras connect to corvus, which records the servo, board and HEX-E logs, so every stream uses one clock. Frames
are stamped with librealsense's global time. The frame age comes from the hand tags' first motion after the lift's
command (section "Seat and the arm's lift").

## Objects and tag blocks

| | Cylinder (pin-push, press) | Sphere (roll, twist) |
|---|---|---|
| Body | 25 mm × 100 mm, axis along y_P; printed at 23.4 mm under 0.8 mm grip tape | 40 mm, bare |
| Stem | 4 mm, 35.6 mm long, from the middle of the underside | 4 mm, 25.7 mm long, from the lower pole |
| Top of the tag block | 15 mm below the lowest finger point of the five layouts' grasps and the measured servo's five best per task | same |
| Tags | block ids 10 to 15; end faces 22 (+z_O) and 23 (−z_O), 13 mm | block ids 16 to 21 |
| Mass | 25 g with stem, block and tape, by infill; no ballast | same |
| Centre of mass | 7.5 mm below O | 7.0 mm below O |
| Seat interface | the block's flat underside | same |

The objects as printed on 2026-11-12 are in `assets/tracking/20261112-objects/` (`scripts/hardware/tracking_objects_print.py`;
owner: stems for both layout sets, about 0.8 mm grip tape on the cylinder, the sphere bare and at the generalist and the
pin-push specialist only). Each block is printed in one piece with its stem, upright, so the stem lies on the block's
axis; the stem ends in a 45° cone, a flat seat 10 mm across and a 2.4 × 5 mm key, glued into a matching flat and slot
on the object, and the CAD assembly shares no volume. The cylinder is printed standing, so it comes out round; its two
end tags enter a bundle adjustment with the block's, which measures the cylinder against its block. The owner printed
it without end tags on 2026-11-13; the plate then takes 80.0 g of PLA against 93.8 g in the slicer. The first attempt
was cancelled after 4 min: the project files left the one-part objects without a filament on the object, Bambu Studio's
window put them on filament 1, mapped to the black spool, and the slice made 359 filament changes in 12 h instead of 67
in 3.9 h. The projects now put black on filament 1 and white on 2. Its map is then its block's, placed by the seat and key,
as for the sphere. A plate pressed on the face of tag 10 or 13 lies parallel to its axis, and equal gaps to the
cylinder at both ends set its turn when it is glued. The sphere is two
hemispheres on three dowels. The tracker's maps of the assembled objects are `object_tracking_mjcf.object_map`. The
measured servo's best layouts (grids of 2026-11-10, press, roll and twist) reach 18.5 mm below the cylinder and 10.7 mm
below the sphere (`rows/stem_sv25_sv55db.json`), so the cylinder's stem stays at 35.6 mm and the sphere's grows by
0.5 mm.

The sizes, the 25 g and the axis direction are those of the HOM scenes (`scripts/hom/hand_object_scale_sim.py:70`,
`hom_tasks.py:77-81`). At the open and touch poses of the five layouts' S3 grasps the fingers reach up to 20.6 mm below
the cylinder, where the press's middle finger touches 30° below the equator, and up to 9.9 mm below the sphere
(`docs/experiments/20261105-object_tracking_sim/rows/stem.json`). The earlier recipes
(`docs/experiments/20261008-hand_object_scale/objects.json`) put steel at the centre only to reach 25 g with thin walls
and sparse infill, and a 2.5 g flag with one 40 mm tag on each object. A denser infill reaches 25 g without ballast: the
cylinder's body is 21.4 g, 35 % of solid PLA, and the sphere's 21.6 g, 52 %. The tag block replaces the flag.

The tag block is a hexagonal prism, 30 mm across flats and 17 mm tall, hollow with 0.8 mm walls, about 3 g. Each of
its six side faces is 17.3 mm wide and bears a 13 mm tag36h11, the largest whose one-cell white border fits the face.
The tags are printed into the faces in matte black and white PLA on the P1S with the AMS, flush with the face, so
their geometry follows the CAD. Laser-printed matte labels are the fallback. The lower face is flat. Each body has its
own tag IDs: hand 6 to 8, seat 9, cylinder 10 to 15 and its end faces 22 and 23, sphere 16 to 21. Each tag map is refined by a bundle adjustment
over about 40 views (step 3).

The tags are placed below the fingers because the pads cover the object's sides. On the 40 mm sphere, tags printed on
the curved surface would be about 8 mm across, and some would lie under the pads. Below the fingers the tags are flat,
13 mm and outside every finger's path on the cylinder. The sphere is different. The tripods of the roll and the twist
let it roll on every pad (w_roll = 0), and in the study's own rollouts its vertical axis turns by 2 to 104° (median
per pick) before the task starts. A stem follows that turn. At the roll and press specialists' layouts, and in the
twist at the twist specialist's, the stem or block touches a finger in 3 to 5 of 5 rollouts and those picks lose up to
0.50. At the generalist and the pin-push specialist it clears by 7.1 mm or more. Tags on the sphere's surface remain the
option for the other three layouts (decision 2).

## Seat and the arm's lift

The seat is a flat printed plate on a pedestal fixed to the table, under the block. At 1.6 s into a trial, when the
simulation removes its post (`hom_tasks.py`, `Schedule`), the arm lifts the hand 40 mm in 0.4 s on a minimum-jerk
profile, with a peak acceleration of 1.44 m/s². After the task it lowers the hand by the same distance and the fingers
open. The hand tags start to move in the images when the lift starts, so the delay between the lift's command and
that first motion gives the frame age in every trial.

The seat locates nothing on purpose. A pocket 4 mm deep round the block's lower end, with 0.3 mm of clearance, broke the
cylinder tasks in simulation. The grasp forces come on at 0.8 s while the block is caught, the pinch pushes the object
1 to 3 mm against the walls, and after the lift the cylinder swings about the pinch axis and falls. On the study's post
the object is supported but free to slide and tilt, and before 1.6 s it travels 1.9 to 30.3 mm as the fingers close.
The flat seat reproduces that travel. The tracker measures the start pose, and the arm moves the hand by the measured
offset when it exceeds 1 mm or 2°.

In simulation the lift's acceleration made no difference: on three picks the scores with and without it were within
0.01. The block stayed at least 31.8 mm above the seat during every task, so it never came down more than 8.2 mm below
its seated height; a 15 mm lift would leave 6.8 mm.

## Cameras

Two D435s are used: the one on corvus's USB hub (serial 836612071918) and a second. Each records its colour stream at
1920 × 1080 and 30 fps, with fixed exposure of at most 5 ms, fixed gain and white balance, and an LED panel beside it.
At 0.3 m a 13 mm tag then spans about 60 px, 7.6 px per cell. The earlier tracker used the infrared stream at
1280 × 720 (`scripts/hardware/real_v1_tag_tracker.py`). That stream has a global shutter but half the angular
resolution (focal length 674 px against 1397 px). At the roll's 40°/s the colour sensor's rolling shutter shears the
block by about 0.2°.

The cameras stand on their sides, 1080 × 1920, 42° across and 69° high, fixed to the table 0.3 m from the block in the
task, at its height, at azimuths 0° and 260° from x_P. Upright, 42° high, a camera cannot see the seated block and a
hand tag 44 mm above P in one frame. In the simulated rollouts this pair sees two or more block tags in 99.8 % of the
task's frames and every frame of 19 of the 20 picks; alone, the cameras cover 77 % and 86 %. Cameras 10° or more above
the block see it as well but lose their hand tags behind the fingers. If corvus cannot run both at 1080p, 1280 × 720
gives 5.0 px per cell at 0.3 m, and the 13 mm tag still decodes to 65° there.

## Tracking software

A new package `src/sr2hand/tracking/`, with rotations from `sr2hand.geometry.rotations`:
- capture: `pyrealsense2`, both cameras, global time stamps;
- detection: `pupil_apriltags` at full resolution in a window around the block's last pose, and on the whole frame at
  `quad_decimate=2` when the block is lost;
- pose: all corners of one body in one solve (`cv2.solvePnP` with SQPNP, then Levenberg-Marquardt), dropping a tag
  seen more than 65° from its normal (bench, 2026-11-09) or whose corners reproject worse than 1 px, and both cameras
  in one solve of $T_{PO}$;
- calibration: the hand tags against the arm, and a bundle adjustment of each tag map;
- output: $T_{PO}$ with its time stamp, the number of tags used and the reprojection RMS, sent to the trial runner at
  30 Hz; H.264 video of both cameras and every detection written to disk.

The detector settings and the fixed-exposure handling of `scripts/hardware/real_v1_tag_tracker.py` are reused. Corvus
runs Python 3.10 with numpy 1.26.4 and scipy 1.12.0; `pyrealsense2`, `pupil_apriltags` and OpenCV are checked there
first.

## Trial sequence

| Time | Step |
|---|---|
| before | The object rests on the flat seat. The tracker measures its pose; when it is more than 1 mm or 2° from nominal, the arm moves the hand by the offset. The fingers are at the open pose, 8 mm off the object. |
| 0.1 to 0.8 s | The fingers approach at 20 mm/s. |
| 0.8 s | The HOM controller turns on the grasp forces. |
| 1.6 s | The arm lifts the hand 40 mm in 0.4 s. |
| 2.0 s | The task starts; the controller receives $T_{PO}$ from the tracker. |
| 4.9 to 7.8 s | The task ends (4.9 s roll, 5.5 s press, 6.5 s pin-push, 7.8 s twist); the forces stay on. |
| return | The arm lowers the hand 40 mm, the fingers open and the object rests on the seat. Its pose is measured again. |

The four tasks end close to their start pose. The pin-push turns +15°, −30° and +15°, the press goes up 20° and back,
and the roll's steps return to rest. The twist leaves the sphere turned by up to 90°, which the flat seat accepts. The
runner pauses when the object ends more than 10 mm from its start or drops. For a manual reseat the arm
lifts the hand 80 mm, the operator places the object, and the arm returns. A catch tray around the seat keeps a dropped
object near it.

Between layouts the object stays on the seat, the fingers flex away from it and the gantries move. The arm then
shifts the hand so that the next grasp's object centre lies at the seat, a move of at most a few centimetres.

## Reproduction of the simulated experiments

The first series uses the five layouts of the task-conflict study on the SCS0009 plant
(`docs/experiments/20261103-hom_task_conflict/20261104-conflict_sv25.json`). They are the generalist (70,110) and the
specialists for twist (100,70), roll (85,170), pin-push (75,55) and press (70,150). Each layout runs its S3 grasp and
controller parameters for each task, 10 trials each: 200 trials, about an hour plus reseats.

The simulation is run again first on the setup as built, at seeds 0 to 4. The objects enter it with stem and block,
at their measured mass, centre of mass and inertia. The flat seat and the arm's lift are in
`scripts/hardware/object_tracking_rollouts.py`; the measured friction and the plant of the hand's servos are added. The
"sense" variant runs at the tracker's measured rate, frame age and noise.

For each task, the score distributions (median and interquartile range) and the Spearman correlation of the five
layouts' median scores are compared. The study's claim that (70,110) reaches 0.84 of each task's best layout is tested
as well. One trial per layout and task is overlaid on its simulated angle trajectory.

The series needs three items outside this plan. The HOM controller must run on the hand; its `Measurement` takes the
joint angles, the object pose and each finger's centre of pressure at every step. The study's mount coordinates must be
converted to the driver's, whose origins lie 6.2 mm from the CAD travel-box centres. The yaw servos' sag of 2.4 to 6.2°
under gravity enters the controller as joint error.

## In-hand pose estimation

Every trial records the servo angles, the boards' raw fields with their contact estimates
(`src/sr2hand/driver/reskin5x/host/wrench_viewer.py`), the HEX-E wrench and $T_{PO}$. The bus reads nine servo
positions at up to 111 Hz, and three boards give about 160 records/s. The baseline estimator places each contact from
the finger's forward kinematics and the pad's contact location. It then fits the sphere's centre or the cylinder's
axis. The sphere's orientation and the cylinder's turn about its own axis follow only from the contacts' motion over
time.

The predicted error comes from the same estimator on the simulated rollouts with the measured sensor errors. The
contact location error is 0.25 to 0.53 mm at 1.5 N and above, 0.59 to 0.75 mm below 1 N, and 1.1 to 1.6 mm at 0.5 N
and below. The on-hand residual after the own-angle correction is 4.4 µT, and the yaw sag adds joint error. The
measured error comes from the trials against the tracker, whose 0.3 mm target is below every contact location error.

## Implementation steps

Each step names its acceptance test. Steps 1 to 3 need no time on the hand or the arm.

1. Tag size test. Print a plate with tag36h11 and tag25h9 at 10, 12, 13, 14 and 16 mm. Record it with the colour stream
   at 1080p and 720p, at 0.25, 0.3 and 0.4 m, and at 0, 30, 45 and 60° from the tag normal. Choose the smallest family
   and size that decodes in at least 99 % of frames up to 50°, with corner jitter of at most 0.15 px at 0.3 m. The 13 mm
   tag36h11 of this plan changes if the test requires it. Simulated on 2026-11-05: at 1080p and 0.3 m every tag of
   10 mm or more decodes in 99 % of frames to 65°, the 13 mm tag36h11 to 65° out to 0.5 m, with corner jitter of 0.03
   to 0.06 px. `sr2hand.tracking.pose.detect` subtracts the detector's half-pixel offset.
2. Clearance and visibility in simulation. Add the stem, block and seat to the HOM scenes and replay the five layouts'
   grasps and tasks. Accept when the stem and block clear every finger by 3 mm in every rollout and the scores change
   by at most 0.02. Then render the block's tags from candidate camera poses over the same rollouts. Choose the two
   poses that give, in every frame, at least 2 fully visible tags within 60° of their normal in one camera or both. The
   hand's tags must be visible to both cameras. Done on 2026-11-05 for the flat seat and the 40 mm lift: on the
   cylinder the stem clears by 1.6 mm or more (5.3 mm at nine of ten picks) and no pick loses more than 0.011; on the
   sphere it touches at three layouts. The pair at 0° and 260° covers 99.8 % of the task's frames. Left: the hand tags'
   brackets in the cad_v3 model, and the sphere's tags at three layouts (decision 2).
3. Tag block and turntable. Print a block and refine its tag map by bundle adjustment from about 40 views. Track it on
   a turntable driven by an STS3032 (12-bit encoder) through 360° in 5° steps, level and on a 20° wedge. Accept 0.3°
   and 0.3 mm RMS against the encoder and the turntable's fitted axis, with at least 99 % of frames tracked. Simulated:
   the adjustment brings an assumed print error of 0.11 mm RMS down to 0.02 mm; with the refined map the block is
   tracked in every frame to 0.06 to 0.08° against the encoder and 0.14 to 0.20 mm from its fitted path, and at 40°/s
   with a rolling shutter to 0.20° and 0.55 mm against the truth. One camera's error is mostly in depth (0.5 mm).
   Bench, 2026-11-09 (the lab session: the printed block of `assets/tracking/20261107-tag_prints/`, two D435s at 0.28
   to 0.29 m, cv2.aruco, 45 frames each). Per camera and frame the block's pose varies by 0.14 to 0.16 mm, mostly
   along the view axis, and by 0.11° RMS. A tag seen at 80° decoded in half the frames and raised the depth's spread
   to 1.05 mm; the tracker now drops tags beyond 65°, which in simulation of this setup lowers the spread from 0.52 to
   0.14 mm. With the CAD map the corners reproject at 0.75 px RMS, and a fit of the across-flats and the square size
   brings that to 0.33 to 0.38 px, with the squares 2.5 % large relative to the face spacing. Only that ratio is seen
   by a camera whose pose is free. The across-flats measured with calipers sets the scale, and the adjustment must
   free each tag's size, which `pose.bundle_adjust` held at the map's. The owner's calipers give 29.95 to 30.02 mm
   across the three pairs of flats, so the faces lie where the CAD puts them and the black squares measure about
   13.33 mm, 0.16 mm of black spread per edge. With that map (`HexBlock(across_flats=29.985, tag_size=13.33)`) the
   bench corners reproject at 0.33 and 0.38 px, and the block lies 4.1 to 4.4 mm farther than with the CAD map. The
   D435 depth maps put the faces 3.6 to 6.9 mm nearer than that solution, and 0.5 to 2.6 mm nearer than the CAD one:
   either the depth is biased at 0.29 m, near its minimum range (25 and 54 % of pixels valid), or the colour cameras'
   factory model, which has no distortion, misses lens distortion where the block lay, 420 to 500 px from the image
   centre. A checkerboard calibration of each colour camera with distortion, and arm steps of the block seen across
   the image, decide it; both belong before step 5. `bundle_adjust` now frees each tag's size and takes the scale from
   measured lengths, on the block the three across-flats. In simulation of this print (40 views) it brings the map to
   0.028 mm RMS of the print and a single camera's depth bias from 5.3 mm (CAD map) and 7.6 mm (rigid adjustment) to
   0.44 mm. The frames and corner bursts (19 MB) are on the workstation in `logs/20261109-tag_block_bench/cams/`.
   The lens and arm calibration of 2026-11-13 measured both
   (`docs/experiments/20261113-camera_arm_calibration/20261113-camera_arm_calibration.html`). With the fitted lenses,
   tag 13 triangulates at 13.02 mm over 17 arm poses. Against the board solved in IR1, the depth reads 0.9 and 1.8 mm
   farther. The two cameras' block centres differ by 8.13 mm with 13.33 mm tags and by 2.71 mm with 13.0 mm tags. On
   that page the 13.33 mm is attributed to the factory model, which has no distortion. A two-camera bundle adjustment
   of the block map (`pose.bundle_adjust_multi`, 35 views, the cameras' poses held) gave tags of 12.98 to 13.02 mm,
   0.29 to 0.35 mm inside the caliper faces. With it the cameras' block centres lie 0.36 mm apart over the 17 arm
   poses, down from 2.71 mm. The map for the tracker is
   `docs/experiments/20261113-camera_arm_calibration/data/fits/20261113-hex_block_map.json` (`TagMap.to_json`). It
   is the 11-07 test block's; the new cylinder block bears the same ids 10 to 15 and needs its own run. The owner
   moved camera 836612071918 at about 16:00 on 2026-11-13, and its pose was re-solved against the unmoved camera from
   31 board views. In a test on the old place the method landed 0.42 mm and 0.14 deg from the arm's fit. The cameras'
   poses for the tracker are in
   `docs/experiments/20261113-camera_arm_calibration/data/fits/20261113-handeye_tcp_fitted_lens_moved836.json`.
4. Objects and seat. Print the cylinder and the sphere at the infill that makes 25 g, with the chosen contact surface
   (decision 1). Measure their mass and centre of mass. Measure the pad's friction on the surface with the hand on the
   UR5. One fingertip presses the seated object at 1 and 2 N while the arm slides 5 mm, and the HEX-E gives the forces.
   Set each object down on the flat seat 30 times with the hand, and accept a start pose within 1 mm and 2° of nominal
   in 29 of 30.
5. Cameras and hand frame in the cell. Mount the cameras on their sides at 0° and 260°, the lights and the hand tags
   (decisions 3 and 4), and add them to `cell.json`. Calibrate the cameras against the arm's base with the hand tags
   (about 15 poses). The ChArUco board on the mount's −y_P face is in camera 260's view at turns of 0 to +45° about
   z_P and in camera 0's at +90 to +135°, which is −90 to −135° on joint 6; the −x_P face faces away from both
   cameras in the task orientation. A paper checkerboard can replace the board if it is flat and measured, and fixed
   to the hand for the hand-eye poses (`docs/experiments/20261113-tracking_prints/`, 2026-11-13). Accept when the two routes to $T_{CP}$ agree within 1 mm and 0.5°, the tracked motion of a seated
   object's block matches arm steps of 10 to 40 mm within 0.3 mm, and the frame age from the hand tags is at most 50 ms
   median.
6. The arm's lift. Run the lift and its return 100 times with an object held. Accept a return within 0.1 mm of the
   start (the UR5's repeatability), and the boards' readings within their noise (3.4 µT) while the arm moves.
7. Trial runner on corvus. Run 50 consecutive trials of one task on one layout. Accept at most 2 manual reseats and
   every stream on one clock.
8. First series and comparison, as in "Reproduction of the simulated experiments", once the HOM controller runs on the
   hand.
9. In-hand estimation: the predicted and measured errors on the series' data.

The lab's rules apply to every arm motion. There is no freedrive with the SR2 hand on the UR5, joint 6 stays at most
+45° without the owner, and every path is checked with the cell model. Results go into dated HTML pages under
`docs/experiments/`, as the repository's CLAUDE.md requires, with films from the tracking cameras.

## Parts

| Part | Made by | Notes |
|---|---|---|
| Tag test plate | session, printed | step 1; STLs in `assets/tracking/20261107-tag_prints/` (2026-11-07) |
| Tag blocks: cylinder, sphere, spare | session, printed | P1S with AMS, matte black and white PLA; the step-3 block (ids 10-15, 4.2 mm bore) in the same folder |
| Cylinder and sphere | session, printed | `assets/tracking/20261112-objects/` (2026-11-12): blocks with their stems, the cylinder, the sphere's hemispheres and dowels; infill for 25 g |
| ChArUco board | session, printed | `assets/tracking/20261112-charuco_board/` (2026-11-12): 7 × 5 squares of 21 mm, for each colour camera's lens and the hand-eye calibration |
| Flat seat on a pedestal | session, printed | fixed to the table under the block |
| Turntable adapter and 20° wedge | session, printed | step 3 |
| Catch tray around the seat | session, printed | |
| Mount for the ChArUco board on the hand | session, printed | `assets/tracking/20261112-charuco_board/` (2026-11-12): a post in the notch at the end of the hand's camera arm, bolted through its M5 hole; the board hangs facing −x_P or −y_P, at least 16 mm from every other solid of the hand's CAD; replaces the two hand tag mounts (owner: one board for both calibrations); both cameras see the −y_P face, with the hand turned (step 5) |
| Camera and light mounts | owner | fixed to the table |
| USB 3 cable and port for the second D435 | bought if needed | |
| Two LED panels | bought | |
| High-friction surface | bought | decision 1 |

## Decisions for the owner

1. Contact surface. The HOM results use friction 1.0, which stands for grip tape on the bench tools. Bare PLA gives
   0.37 to 0.44 against the pads, and at 0.4 the twist's median best-of-set score falls from 0.91 to 0.08 in
   simulation. Recommended: grip tape or a rubber coating on the contact surfaces, measured in step 4. The tags are on
   the block, so the surface can be covered entirely. Decided 2026-11-12: grip tape of about 0.8 mm on the cylinder,
   the sphere bare for now.
2. The sphere's tags at the roll, press and twist specialists' layouts, where the stem touches a finger: tags on the
   sphere's surface, or a sphere series at the generalist and the pin-push specialist only. Decided 2026-11-12: the stem,
   at those two layouts.
3. Hand tag mounts: two 40 mm tags on static parts of the hand, facing the cameras at 0° and 260°. They check the arm's
   route to $T_{CP}$ and give the frame age. Decided 2026-11-12: one ChArUco board, moved in front of each camera for
   its lens and then clipped to the hand for the hand-eye poses.
4. Camera host: corvus, which already has a D435, unless the tracker delays the servo or board logs there. Otherwise a
   second computer synchronised to corvus by chrony.
5. Object pose for the HOM controller: the tracker, as in the "sense" variant, is recommended first. The tactile and
   joint estimate is then evaluated against it in step 9.

## Options considered

| Option | Outcome |
|---|---|
| One tag on a vane (workshop study) | Visible through the whole run in 40 of 74 trials; the vane leaves the camera's view when the cylinder turns about its axis (`paper/transfer_results.tex`) |
| Tags on a cube fixed to the object | Solving each face alone makes the pose jump when the visible face changes. One solve over all corners with a calibrated tag map, used for the tag block, removes the jumps |
| Tags printed on the sphere | About 8 mm on a 40 mm sphere, and some lie under the pads; kept for tasks that turn the object beyond about 45° |
| Slot in the cylinder over a long post | Needs a lift of more than half the cylinder's length and can jam |
| Seat that drops away (this plan's first version) | Needs a carriage, guides and a drive the lab does not have (owner, 2026-11-05); the arm lifts the hand instead |
| Pocket round the block's lower end | In simulation the pin-push scores 0 at four of five layouts: the block is caught while the forces come on |
| Cone boss in a cone socket (first version) | A 20 mm socket mouth round a 10 mm boss puts the socket's upper half inside the block; not simulated |
| Ballast of steel, brass or tungsten | Not needed: infill alone makes 25 g |
| Camera poses from one hand tag each | 0.1 to 1.0° of tilt error and 0.3 to 5.4 mm in P in simulation; kept as a check of the arm's route |
| Cameras mounted upright | 42° high: the seated block and a hand tag 44 mm above P do not fit one frame at 0.3 m |
| Retroreflective balls seen by a D435 | Needs an infrared ring light at the lens, and the 50 mm stereo baseline limits depth |
| OptiTrack | Not installed at the robot, and the hand covers the markers |
