# Object seating and external tracking for SR2 hand experiments on the UR5

Plan of 2026-11-05 for a dedicated session. The owner decides the items under "Decisions for the owner". The aim is to
run the HOM study's four tasks on the real hand with the simulation's protocol: pin-push, press, roll and twist
(`scripts/hom/hom_tasks.py`). External cameras measure the object's motion.

## Summary

Each object gets a short stem below it that ends in a printed hexagonal prism, the tag block, with a 13 mm AprilTag on
each side face. At the HOM grasps the fingertips reach 6 to 8 mm below the object, and the block hangs 15 mm below
them, clear of every finger. The object's pose is solved from all visible tag corners at once, in the images of two
D435 colour cameras at the block's height, relative to tags on the hand. The block's lower end rests in a cone socket
on a retracting seat. The
seat drops away at 1.6 s, when the simulation removes its post, and rises after the task to take the object back. The
arm does not move during a series of trials.

The targets are 0.3 mm and 0.3° RMS for the object's pose in the palm frame, at 30 Hz, in at least 99 % of the frames
of a trial. The first series runs the five layouts of the task-conflict study, 10 trials per layout and task, against
simulations of the objects as built.

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
| Seated pose repeatability | 0.2 mm and 0.3° (1 sd) | the simulation jitters the start by up to 1 mm and 2° |
| Arm motion | none within a series | the hand's cables (owner, 2026-11-05) |
| Ferromagnetic parts | none in the objects or the seat sockets | the fingertip boards are magnetometers |

## Measurement chain

For each camera C the object's pose in the palm frame is $T_{PO} = T_{CP}^{-1}\,T_{CO}$. $T_{CO}$ is solved from the
tag block's corners and $T_{CP}$ from the hand's tags in the same image. Both cameras enter one least-squares solve of
$T_{PO}$ over the reprojection error of all block corners. Each camera's pose relative to the hand comes from its own
images, so the cameras need no calibration against the robot. A bumped camera shows as a jump in $T_{CP}$.

The hand has one tag now: tag36h11 id 6, 40 mm, on the static tag mount. Its centre is at (16.1, 133.6, 43.9) mm in P,
and it faces −x_P (`assets/cad_v3/frames.json`, `sensors.tag`). One more 40 mm tag per camera goes on a static part of
the hand, facing that camera. The tags' poses in P start from CAD. They are refined once by moving the arm through
about 15 poses planned with the cell model (`scripts/ur5/ur5_cb3_cell.py`). Each camera's pose in the base frame and
the tags' poses in tool0 are solved together (OpenCV `calibrateRobotWorldHandEye`). After that, $T_{CP}$ also follows
from the joint readings through `sr2hand.robot.ur_kinematics` and `sr2hand.robot.tool_chain.SR2Hand`. The two routes
are compared in every series, and the second is used for frames in which the hand tags are hidden.

Both cameras connect to corvus, which records the servo, board and HEX-E logs, so every stream uses one clock. Frames
are stamped with librealsense's global time. A 30 mm tag on the seat's carriage moves at the commanded drop time, and
its first motion in the images gives the frame age in every trial.

## Objects and tag blocks

| | Cylinder (pin-push, press) | Sphere (roll, twist) |
|---|---|---|
| Body | 25 mm × 100 mm, axis along y_P | 40 mm |
| Stem | 4 mm, from the middle of the underside | 4 mm, from the lower pole |
| Top of the tag block | 15 mm below the grasp's lowest fingertip point | same |
| Mass | 25 g with stem and block | 25 g with stem and block |
| Ballast | brass or tungsten, centre of mass within 0.5 mm of O | same |
| Seat interface | cone boss and a radial V-groove under the block | cone boss |

The sizes, the 25 g and the axis direction are those of the HOM scenes (`scripts/hom/hand_object_scale_sim.py:70`,
`hom_tasks.py:77-81`). At layout (70,110) the fingertips reach 6.0 to 6.5 mm (tripod) and 6.3 to 8.4 mm (pinch) below
the object's lowest point (`docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b/ground/`), so the
stem is about 22 mm long. The earlier recipes (`docs/experiments/20261008-hand_object_scale/objects.json`) put steel
ballast in both objects and a 2.5 g flag with one 40 mm tag on each. Steel near a fingertip distorts its magnets'
field, so the ballast here is brass (8.5 g/cm³) or tungsten (19.3 g/cm³), and the tag block replaces the flag.

The tag block is a hexagonal prism, 30 mm across flats and 17 mm tall, hollow with 0.8 mm walls, about 3 g. Each of
its six side faces is 17.3 mm wide and bears a 13 mm tag36h11. The tags are printed into the faces in matte black and
white PLA on the P1S with the AMS, so their geometry follows the CAD. Laser-printed matte labels are the fallback. The
lower face has a 60° cone boss, 10 mm across. On the cylinder's block a radial V-groove fixes the yaw. Each body has its
own tag IDs: hand 6 to 8, seat 9, cylinder 10 to 15, sphere 16 to 21. Each tag map is refined by a bundle adjustment
over about 40 views (step 3).

The tags are placed below the fingers because the pads cover the object's sides. On the 40 mm sphere, tags printed on
the curved surface would be about 8 mm across, and some would lie under the pads. Below the fingers the tags are flat,
13 mm and outside every finger's path. The stem and block change the object's mass distribution, so the selected
grasps are simulated again with the objects as built (step 2). Tags on the sphere's surface remain the option for tasks
that turn the object by more than about 45° about a horizontal axis, where a stem would reach the fingers.

## Retracting seat

The seat is a cone socket on a vertical carriage. At 1.6 s into a trial the carriage drops 30 mm in at most 0.2 s,
when the simulation removes the post (`hom_tasks.py`, `Schedule`). After the task it rises and takes the object back.
At the top an over-centre crank presses the carriage against a hard stop, so the seated height does not depend on the
servo's angle. The drive is an STS3032 on its own serial adapter, at least 150 mm from the fingertips. The guides are
carbon or brass rods. The carriage's state and the frame age are read from a 30 mm tag36h11 (id 9) on the carriage.

The socket's cone has a 20 mm mouth and captures the block within 5 mm. The cylinder's socket adds a ridge that fits the
block's V-groove and turns the block into place from up to 15° of yaw. The sphere's socket has no ridge, so the sphere
may rest at any yaw, which the tags measure.

Until the retracting seat is built, a fixed seat with the same socket is used, and the arm lifts the hand 30 mm at
1.6 s. In the simulation the seat replaces the post and the "hold" pin: the block is constrained by the cone until 1.6 s.

## Cameras

Two D435s are used: the one on corvus's USB hub (serial 836612071918) and a second. Each records its colour stream at
1920 × 1080 and 30 fps, with fixed exposure of at most 5 ms, fixed gain and white balance, and an LED panel beside it.
At 0.3 m a 13 mm tag then spans about 60 px, 7.6 px per cell. The earlier tracker used the infrared stream at
1280 × 720 (`scripts/hardware/real_v1_tag_tracker.py`). That stream has a global shutter but half the angular
resolution (focal length 674 px against 1397 px). At the roll's 40°/s the colour sensor's rolling shutter shears the
block by about 0.2°.

The cameras stand about 0.3 m from the block, at its height within ±15°, 90 to 120° apart in azimuth, fixed to the
table. Their positions come from the visibility check of step 2. If corvus cannot run both at 1080p, 1280 × 720 gives
5.0 px per cell at 0.3 m.

## Tracking software

A new package `src/sr2hand/tracking/`, with rotations from `sr2hand.geometry.rotations`:
- capture: `pyrealsense2`, both cameras, global time stamps;
- detection: `pupil_apriltags` at full resolution in a window around the block's last pose, and on the whole frame at
  `quad_decimate=2` when the block is lost;
- pose: all corners of one body in one solve (`cv2.solvePnP` with SQPNP, then Levenberg-Marquardt), dropping a tag
  whose corners reproject worse than 1 px, and both cameras in one solve of $T_{PO}$;
- calibration: the hand tags against the arm, and a bundle adjustment of each tag map;
- output: $T_{PO}$ with its time stamp, the number of tags used and the reprojection RMS, sent to the trial runner at
  30 Hz; H.264 video of both cameras and every detection written to disk.

The detector settings and the fixed-exposure handling of `scripts/hardware/real_v1_tag_tracker.py` are reused. Corvus
runs Python 3.10 with numpy 1.26.4 and scipy 1.12.0; `pyrealsense2`, `pupil_apriltags` and OpenCV are checked there
first.

## Trial sequence

| Time | Step |
|---|---|
| before | The seat is up with the object. The seated pose is checked against nominal (0.5 mm, 1°). The fingers are at the open pose, 8 mm off the object. |
| 0.1 to 0.8 s | The fingers approach at 20 mm/s. |
| 0.8 s | The HOM controller turns on the grasp forces. |
| 1.6 s | The seat drops 30 mm. |
| 2.0 s | The task starts; the controller receives $T_{PO}$ from the tracker. |
| 4.9 to 7.8 s | The task ends (4.9 s roll, 5.5 s press, 6.5 s pin-push, 7.8 s twist); the forces stay on. |
| return | If the block is within 5 mm of the socket, and within 15° for the cylinder, the seat rises to 2 mm below it. The fingers open, the block drops into the cone and the seat completes its rise. The seated pose is checked again. |

The four tasks end close to their start pose. The pin-push turns +15°, −30° and +15°, the press goes up 20° and back,
and the roll's steps return to rest. The twist leaves the sphere turned by up to 90°, which its socket accepts. The
runner pauses when a block ends outside the capture range or the object drops. For a manual reseat the arm
lifts the hand 80 mm, the operator places the object, and the arm returns. A catch tray around the seat keeps a dropped
object near it.

Between layouts the seat stays up with the object, the fingers flex away from it and the gantries move. The arm then
shifts the hand so that the next grasp's object centre lies at the seat, a move of at most a few centimetres.

## Reproduction of the simulated experiments

The first series uses the five layouts of the task-conflict study on the SCS0009 plant
(`docs/experiments/20261103-hom_task_conflict/20261104-conflict_sv25.json`). They are the generalist (70,110) and the
specialists for twist (100,70), roll (85,170), pin-push (75,55) and press (70,150). Each layout runs its S3 grasp and
controller parameters for each task, 10 trials each: 200 trials, about an hour plus reseats.

The simulation is run again first on the setup as built, at seeds 0 to 4. The objects enter it with stem, block and
ballast, at their measured mass, centre of mass and inertia. The seat model, the measured friction and the plant of the
hand's servos are added. The "sense" variant runs at the tracker's measured rate, frame age and noise.

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

1. Tag size test. Print a plate with tag36h11 and tag25h9 at 10, 12, 14 and 16 mm. Record it with the colour stream at
   1080p and 720p, at 0.25, 0.3 and 0.4 m, and at 0, 30, 45 and 60° from the tag normal. Choose the smallest family and
   size that decodes in at least 99 % of frames up to 50°, with corner jitter of at most 0.15 px at 0.3 m. The 13 mm
   tag36h11 of this plan changes if the test requires it.
2. Clearance and visibility in simulation. Add the stem, block and seat to the HOM scenes and replay the five layouts'
   grasps and tasks. Accept when the stem and block clear every finger by 3 mm in every rollout and the scores change
   by at most 0.02. Then render the block's tags in the cad_v3 scene from candidate camera poses over the same rollouts.
   Choose the two poses that give, in every frame, at least 2 fully visible tags within 60° of their normal in one
   camera or both. The hand's tags must be visible to both cameras.
3. Tag block and turntable. Print a block and refine its tag map by bundle adjustment from about 40 views. Track it on
   a turntable driven by an STS3032 (12-bit encoder) through 360° in 5° steps, level and on a 20° wedge. Accept 0.3°
   and 0.3 mm RMS against the encoder and the turntable's fitted axis, with at least 99 % of frames tracked.
4. Objects and seat. Print the cylinder and the sphere with their ballast and the chosen contact surface (decision 1).
   Measure their mass and centre of mass. Measure the pad's friction on the surface with the hand on the UR5. One
   fingertip presses the seated object at 1 and 2 N while the arm slides 5 mm, and the HEX-E gives the forces. Reseat
   each object 30 times by hand in a fixed seat, and accept 0.2 mm and 0.3° (1 sd).
5. Cameras and hand frame in the cell. Mount the cameras, the lights and the hand tags (decisions 3 and 4), and add them
   to `cell.json`. Refine the hand tags' poses with the arm (about 15 poses). Accept when the two routes to $T_{CP}$
   agree within 1 mm and 0.5°, and the frame age from the seat tag is at most 50 ms median.
6. Retracting seat. Build it (decision 2). Accept an up position within 0.05 mm over 100 cycles (dial indicator) and a
   30 mm drop in at most 0.2 s. While the seat cycles under a gripped object, the boards' readings must stay within
   their noise (3.4 µT).
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
| Tag test plate | session, printed | step 1 |
| Tag blocks: cylinder, sphere, spare | session, printed | P1S with AMS, matte black and white PLA |
| Cylinder and sphere | session, printed | halves around the ballast; stem and block in one piece with the lower half |
| Seat sockets: cone, and cone with ridge | session, printed | |
| Turntable adapter and 20° wedge | session, printed | step 3 |
| Catch tray around the seat | session, printed | |
| Retracting seat: carriage, crank and frame | owner, or the session's draft reviewed by the owner | decision 2 |
| Hand tag mounts, two | owner | positions from step 2 |
| Camera and light mounts | owner | fixed to the table |
| Brass or tungsten ballast | bought | about 8 to 15 g per object |
| USB 3 cable and port for the second D435 | bought if needed | |
| Two LED panels | bought | |
| Serial adapter for the seat's STS3032 | bought | its own bus |
| High-friction surface | bought | decision 1 |

## Decisions for the owner

1. Contact surface. The HOM results use friction 1.0, which stands for grip tape on the bench tools. Bare PLA gives
   0.37 to 0.44 against the pads, and at 0.4 the twist's median best-of-set score falls from 0.91 to 0.08 in
   simulation. Recommended: grip tape or a rubber coating on the contact surfaces, measured in step 4. The tags are on
   the block, so the surface can be covered entirely.
2. Retracting seat: designed by the owner in Inventor, or drafted by the session for review. The fixed seat with a
   30 mm arm lift is used until it exists.
3. Hand tag mounts: two 40 mm tags on static parts of the hand, facing the cameras chosen in step 2.
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
| Slot in the cylinder over a long post | Needs a lift of more than half the cylinder's length and can jam; a 10 mm cone locates the object as well |
| Retroreflective balls seen by a D435 | Needs an infrared ring light at the lens, and the 50 mm stereo baseline limits depth |
| OptiTrack | Not installed at the robot, and the hand covers the markers |
