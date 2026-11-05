# Current simulator hand model and the specification of the simplified cad_v3 model

Task "current" of the cad_v3 workflow, 2026-11-04. Analysis only: files were written in this folder and nowhere else.
The scratch scripts and their outputs sit beside this file:

- `probe_scenes.py` compiles the current scenes and writes `probe_scenes.json`;
- `yaw_centres.py` computes the CAD fingers' joint axes from `parts.json` and writes `yaw_centres.json`;
- `bbox_views.py` plots the CAD parts' bounding boxes in `bbox_views.png`.

A mesh render was not made: the resource gate stayed closed on swap (15.5 GB in use against its 15 GB limit).
`render_A.py` renders the hand's meshes in four views and is ready to run under `resguard.sh run --mem 3G`.

## Summary

A drop-in cad_v3 keeps the cad_v2 scene's 21 bodies, 25 joints, 15 actuators and 31-entry qpos vector as they are.
The shared generator indexes that vector by position. About 40 active scripts match names on a finger prefix, and five
of the HOM and clearance scripts assign a geom to a finger that way. Every new part therefore goes on `palm_pose` as
static geometry, named without a finger prefix.

The CAD fixes two numbers that the current models guessed. The yaw-centre plane, the origin of the palm frame P, lies
103.9 mm below the UR5 flange face; the UR5e arm scene used 100 mm. The lowest gantry surface near the palm centre is
24.4 mm above that plane; cad_v2's `palm_plate` underside is at 23.5 mm. The plate's 170 x 180 mm footprint covers 29 %
of the 337 x 312 mm footprint of the real gantries and plates.

The flange-to-palm transform needs one parameter that no current model has: the clocking k of the UR5 adaptor, one of
four positions 90 deg apart. The arm scene fixed it by assumption.

## Terms and frames

- **A**: the frame of the owner's assembly "hand v3 (anthro config)", mm. z_A points up toward the robot. The base
  plate occupies z_A 0..3, the robot mount 3..28, the gantries -54..0.
- **P**: the palm frame of the simulator and of the hand driver ({P} in `manta_hand/kinematics.py` and `plan.py`).
  The origin lies on the plane of the three yaw-axis centres, x_P points from the thumb to the pair, y_P from the
  middle finger to the index, z_P up; the fingers hang along -z_P.
- **E (tool0)**: the UR5 tool flange frame, origin on the flange face, z_E out of the flange toward the tool. In the
  lab's URDF (`~/Programs/rh56_controller/magpie_force_control/src/ur_description/ur5.urdf`), tool0 equals
  `wrist_3_link`: the chain wrist_3 -> flange -> tool0 composes to the identity (computed here from its rpy values).
- **qpos**: MuJoCo's generalised position vector; **nq** its length. A free joint takes 7 entries (position, wxyz
  quaternion), a slide or hinge one.
- **Morph slides**: the passive slide joints `<f>_x` and `<f>_y` that place finger f's mount in P.
- **Bridge**: the moving cross-beam of a finger's gantry. It spans the gantry along y_A and rides on the frame's x-axis
  leadscrew. The y-axis stepper, leadscrew and rail are mounted on it, and the finger's carriage rides along it.
- **Static geom**: a geom on `palm_pose` itself, which moves only with the palm.

## The current model and how it is built

The model in use is `assets/mjcf/experimental/20261030-cad_v2/scene_screwdriver_medium.xml`
(`scripts/build_cad_v2_finger.py`). It is cad_v1's scene (`scripts/build_cad_v1_finger.py`, `write_scene`) with the
v4b fingertip's DIP link, cassette and trimmed cap in `<f>_pip_frame`. cad_v1's scene is real_v1's
(`assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml`, `scripts/build_real_v1_scenes.py`) with the finger bodies
replaced. The palm, the object, the floor, the actuators and the keyframes come unchanged from real_v1. The tips have
no geoms in the file; `reorient_backends.replace_tips(..., "tcd_ell7.4x8.5x11x22", ...)` adds the pads at use.

Measured on the compiled models (`probe_scenes.json`):

| model | nq | nv | nu | nbody | ngeom | contact excludes | keyframes |
|---|---|---|---|---|---|---|---|
| `real_v1/real_hand.xml` (hand only) | 18 | 18 | 9 | 20 | 14 | 21 | open |
| `real_v1/scenes/scene_screwdriver_medium.xml` | 31 | 30 | 15 | 21 | 15 | 21 | open_ik, open |
| `20261030-cad_v2/scene_screwdriver_medium.xml` | 31 | 30 | 15 | 21 | 1371 | 39 | open |

The CAD hand v3's fingers have cad_v2's kinematics. From the 10.2 mm bearing bores in `parts.json`, the yaw-to-MCP
distance is 20.750 mm and the MCP-to-PIP distance 30.800 mm on all three fingers. cad_v1's `assembly.json` gives the
same two values. The yaw bore axes are tilted 0.93 deg out of the horizontal (z-component 0.0162). The same 0.925 deg
rotation was measured in cad_v1's carriage STL and removed there as an export artefact.

## Names and qpos layout that other code reads

Bodies: `palm_pose` (a direct child of the world body), `<f>_mount`, `<f>_yaw_frame`, `<f>_mcp_frame`,
`<f>_len_frame`, `<f>_pip_frame`, `<f>_tip` for f in thumb, index, middle; the object body `screwdriver_medium`
(`reorient_backends.OBJ`; every replaced object keeps this name); `post` in HOM scenes.

Joints: `palm_px`, `palm_py`, `palm_pz` (slides), `palm_rx`, `palm_ry`, `palm_rz` (hinges); per finger `<f>_x`, `<f>_y`,
`<f>_yaw`, `<f>_mcp`, `<f>_len`, `<f>_pip`. The `<f>_len` slide is a zero-travel shim: its range "0 0" compiles as
unlimited (`jnt_limited` false), and it exists only for the generator's index layout.

Actuators: `a_palm_px` .. `a_palm_rz` (class `pose`: kp 4000, kv 100) and `a_<f>_yaw`, `a_<f>_mcp`, `a_<f>_pip` (class
`ctrl`). `scripts/apply_measured_plant.py` rewrites the plant through the `ctrl` default class.

Geoms: `palm_plate`, which must stay the first geom child of `palm_pose` (`real_v1_d1_continuation.palm_plate` asserts
this); `floor`; `<f>_tipgeom` and `<f>_pad<i>` (added by `replace_tips`); the CAD spheres `<f>_<part>_s<i>` and visuals
`<f>_<part>_visual`.

Sites: `workspace_thumb`, `workspace_index`, `workspace_middle` (visual); the RL builder adds `palm_pose_site`, the arm
scene `palm_site`. Default classes: `pose`, `morph`, `ctrl`, `cad_visual` (group 2), `cad_collision` (group 3),
`cad_pad` (group 4). Keyframes: `open` and `open_ik` (the grasp, written per design). The generator copies the `open`
key's first 13 values and its finger angles into every baked design. Without an `open` key of at least 31 values it
writes the baseline hand's default angles.

Name conventions read by code:

- A geom belongs to finger f when its body name, or an ancestor's, starts with f. `hom_tasks.Plant`, `hom_gpu`,
  `hand_object_scale_sim.hold_rollout`, `hand_object_scale_kin.cmd_verify` and `real_v1_trajectory_clearance.finger_geoms`
  use this. A grep of `scripts/` and `src/` finds about 40 active files that call `startswith` on a finger name.
- `hom_gpu` treats a geom whose name contains `_pad` as a fingertip pad, and a geom matching
  `_(dip_link|cassette|cap)_s\d+$` with radius under 2 mm as a fine tip sphere.

The qpos layout of the scene with morph slides (nq 31):

| qpos | entries |
|---|---|
| 0-6 | object free joint (position 0-2, quaternion 3-6) |
| 7-12 | palm_px, palm_py, palm_pz, palm_rx, palm_ry, palm_rz |
| 13-18 | thumb_x, thumb_y, thumb_yaw, thumb_mcp, thumb_len, thumb_pip |
| 19-24 | index_x, index_y, index_yaw, index_mcp, index_len, index_pip |
| 25-30 | middle_x, middle_y, middle_yaw, middle_mcp, middle_len, middle_pip |

The generator (`morphohand.tools.morphology_xml`) bakes a design by adding (x, y) to `<f>_mount`'s pos and len to
`<f>_len_frame`'s pos. It then removes the nine morph joints and keeps qpos indices 2, 3, 5, 8, 9, 11, 14, 15, 17 of the
18-entry finger block. A rigid scene has nq 22 (13 + 9); `rl/reference_trajectory.py` slices it as 0:7, 7:13, 13:22.
A hand-only file has nq 18. The scene test is the presence of a joint named `palm_px`. Any added joint shifts these
indices or leaves keyframes the wrong length, and MuJoCo then refuses to load the generated file.

## The palm body and the palm plate

`palm_pose` sits at (0, 0, 62.5) mm in the world with gravcomp 1. Its only geom, `palm_plate`, is a 170 x 180 x 3 mm box
(x_P by y_P) centred at z_P +25 mm. Its underside is at 23.5 mm and its top at 26.5 mm. Its mass, 0.0918 kg, comes from
the default density of 1000 kg/m^3. The palm subtree weighs 0.354 kg in cad_v2 and 0.222 kg in real_v1. The `open`
key of cad_v2 adds 20.05 mm to palm_pz: 10 mm from real_v1's key and the 10.05 mm `palm_raise_m` for the longer CAD
finger. At that key the palm origin is 82.55 mm above the floor.

The plate is used in the following places:

1. It stands for the palm underside in contact with the object and the floor. The floor plane collides with it, so a
   lowered or tilted palm meets the table there.
2. `hand_object_scale_kin.PLATE_BOTTOM` = 23.5 mm is the analytic plate. A grasp is feasible only with at least 1 mm
   of object-to-plate clearance; `workspace` keeps the object's top below it; `verify` checks the analytic clearance
   against `mj_geomDistance` to the geom named `palm_plate`.
3. The RL bracing reward reads the contact sensor `palm_cube_contact`, which matches every geom on `palm_pose`
   (`rl/env_build.py`). `rl_contact_eval.py` counts palm contacts the same way.
4. `real_v1_d1_continuation.py` and `real_v1_diffmjx_gate.py` switch the first palm geom's collisions off, since MJX 3.6
   has no cylinder-box test, and assert that the held state has no plate contacts.
5. The workspace sites are drawn on its top face (z_P 28 mm).

Contact excludes: `palm_pose` is excluded against `<f>_yaw_frame`, `<f>_mcp_frame`, `<f>_len_frame`, `<f>_pip_frame`
and `<f>_tip` (15 pairs from real_v1) and against `<f>_mount` (3 pairs added by cad_v1). Every geom placed on
`palm_pose` is therefore invisible to the fingers. The CAD carriage stands inside the gantry by construction: its
collision spheres in cad_v2 reach z_P 68.2 mm, and the v3 carriage part reaches 65.0 mm.

## The morph slides and the driver's {P}

The sim's mount positions, converted from metres to millimetres, are {P} coordinates: thumb (-50, 0), index (50, 55),
middle (50, -55). `tests/test_manta_frame_map.py` pins `REAL_V1_MOUNTS` equal to `kinematics.FINGER_GEOMETRY`. Each
slide's zero is the centre of that finger's travel.

| sim joint | range (mm) | driver local axis | firmware axis | mechanism in the CAD |
|---|---|---|---|---|
| `<f>_x` | +-30, all three | q_Fx | J1, J3, J5 (firmware "y") | carriage along the bridge, along y_A |
| `<f>_y` | +-55 thumb, +-30 pair | q_Fy | J0, J2, J4 (firmware "x") | bridge along the frame, along x_A |

Firmware positions follow `_TRANSFORM`. For the thumb, J0 = 55 + local y and J1 = 30 + local x. For the index,
J2 = 30 - local y and J3 = 30 - local x. For the middle finger, J4 = 30 + local y and J5 = 30 - local x. Each axis
travels its nominal range from its homed hard stop: 110 mm on J0, 60 mm on J1-J5.

The CAD is consistent with this assignment. The thumb's frame is 160 mm long along x_A, its leadscrew 150 mm, and its
110 mm axis is J0. Each finger's yaw centre lies on its bridge's y-leadscrew axis in x_A, within 0.1 mm (bounding-box centres).

## P in the CAD frame A

Measured from `parts.json`:

- The three yaw centres lie at z_A = -75.9 mm, so z_P = z_A + 75.9 mm.
- The yaw axes run along y_A; the sim's yaw axis is x_P. The thumb (finger v3 <3>, mirrored yaw axis) sits at
  y_A 159.6 mm, the pair at 80.1 and 83.1 mm. Hence x_P = -y_A.
- The design is symmetric about x_A = -111.0 mm. The thumb frame's centre lies there, as does the midpoint of the pair
  frames (-55 and -167), the centres of the base plate and robot mount, and the adaptor's bolt circle. If the travel
  centres share that symmetry, y_P = 0 at x_A = -111.0.

Inferred: with both frames right-handed and z up, y_P = +x_A. P is then A turned by -90 deg about z_A, and the index is
the finger on gantry <1> (x_A -110..0, finger v3 <2>). This needs the owner's confirmation; see the open questions.

Not yet known: the x_P origin, which is the travel centre of the carriages along y_A. The CAD's anthro configuration is
a compact layout (thumb to pair 78 mm, index to middle 77 mm), so its finger positions are not the travel centres. The
frames meet at y_A = 120.2 mm, and the robot mount is centred at y_A 121.4 mm.

The gantry levels in P, from the bridges' downward planar faces and the part bounding boxes:

| surface | z_A (mm) | z_P (mm) |
|---|---|---|
| y-stepper and bridge end, outer end of each bridge | -54.0 | 21.9 |
| bridge bosses at both ends of the y-leadscrew, 86.8 mm apart; the inner ones 8.7-8.8 mm from y_A 120.2 | -51.5 | 24.4 |
| y-leadscrew underside | -46.8 | 29.1 |
| bridge main underside | -28.5 | 47.4 |
| gantry frame underside | -22.0 | 53.9 |
| base plate underside (frame top) | 0.0 | 75.9 |
| robot mount top, adaptor flange face | 28.0 | 103.9 |

The flange axis passes through (-111.0, 116.0) mm in A. That is 4.2 mm from the frames' meeting line and 5.45 mm from
the plates' centre, toward the pair.

## Code that breaks when the palm body gains geoms or bodies

- `morphohand.tools.morphology_xml` and `scripts/generate_morphology_xml.py`: any new joint breaks the 13 + 18 layout.
  A new body that must follow a design is not moved; only `<f>_mount` and `<f>_len_frame` are.
- `hom_gpu.mocap_model`: `palm_pose` must stay a direct child of the world body, or it raises. Per-world layouts write
  `body_pos` of `<f>_mount` alone, so geometry outside `<f>_mount` does not follow the layout.
- `hand_object_scale_sim.cad_fingers` copies only the `<f>_mount` subtrees from the CAD scene into a palm built by
  `build_real_v1_scenes`. New `palm_pose` geoms would be dropped from every HOM scene unless it copies them.
- `hand_object_scale_kin`: `PLATE_BOTTOM` is a constant, and `cmd_verify` measures to the geom named `palm_plate`. A new
  palm geom that hangs below the plate is missing from the analytic feasibility test.
- `real_v1_trajectory_clearance.finger_geoms` walks up to a body whose name starts with a finger name. Gantry frames
  placed in `<f>_mount` would enter its 5 mm finger-to-finger gate: the two pair frames stand 2 mm apart at the nominal
  mounts.
- `hom_tasks`, `hom_gpu`, `hand_object_scale_sim`: geoms under `<f>_` bodies count as finger links in the
  object-clearance checks and in the finger force sums.
- `rl/env_build.py`, `rl_contact_eval.py`, `real_v1_turn_probe.py`: every geom on `palm_pose` counts as palm or plate
  contact, and the bracing reward changes meaning with it.
- `real_v1_d1_continuation.py`, `real_v1_diffmjx_gate.py`: only the first palm geom is switched off for MJX. A new box
  would meet the cylinder tool in a cylinder-box pair that MJX 3.6 lacks.
- `hom_chain.py` (gravity feed-forward) and `real_v1_bench_study.py` (payload) read the palm subtree mass. Mass added to
  `palm_pose` changes both. The pose actuators (kp 4000 N/m, kv 100 N s/m, armature 0.01 kg) move the palm subtree,
  0.354 kg in cad_v2. Their damping ratio is 1.31 there and falls to 0.68 with 1 kg more.
- Six tests `HAND == "cad_v2"` (in `hand_object_scale_kin`, `hand_object_scale_sim`, `hom_tasks`, `hom_gpu`,
  `fp_sr2_toy`) select the v4b tip and its scene paths. Two tests `HAND in ("cad_v1", "cad_v2")` select the CAD finger.
  A new value `cad_v3` misses all eight, and the value check of `hand_object_scale_kin` rejects it.
  `fingertip_shape_rerun_rig.py` (active hand-off) stops unless `SCALE_HAND=cad_v2` and stays as it is.
- `hand_object_scale_kin.morph_vector` hard-codes the nominal mounts (0.100 m thumb-to-pair, 0.055 m half-spacing).

Unaffected: code that reads only the pose of `palm_pose` (`fit_real_v1_pose.py`, `reorient_backends._palm_and_start`,
the RL observation and reward terms), as long as the palm frame stays where it is.

## The UR5e arm scene and what it got wrong

`scripts/build_real_v1_arm_scene.py` deletes the six palm joints and makes `palm_pose` a child of the menagerie UR5e's
`wrist_3_link` at (0, 0.2, 0) m with quat (0.7071, 0.7071, 0, 0). It adds a cylinder `wrist_stack` of radius 40 mm,
running 100 mm from the palm origin up +z_P, at density 700 kg/m^3 (0.35 kg). Its IK model puts `palm_site` at the same
offset. Studies built on it: `real_v1_arm_study`, `real_v1_bench_study`, `real_v1_chain_hands`, `real_v1_slip_study`,
`probe_real_v1_chain`, `real_v1_chain_policy`.

1. **Robot.** The lab robot is a UR5 CB3; the scene uses the menagerie UR5e. The offsets differ (menagerie UR5e
   against the lab URDF): d1 163 against 89.2 mm, d4 134 against 109.2 mm, d5 100 against 94.7 mm, d6 100 against
   82.3 mm. Masses and meshes differ as well. The lab URDF's `base` frame is `base_link` turned 180 deg about z.
2. **Flange-to-palm distance.** It was 100 mm, written as "to be calipered". The CAD gives 103.9 mm from the flange face
   to the yaw-centre plane.
3. **Lateral offset.** The scene put P's origin on the flange axis. In the CAD the axis lies about 4-5 mm from the
   hand's centre line along x_P, and on y_P = 0.
4. **Clocking.** The scene set x_P equal to the flange's x. The adaptor has no dowel, so the hand can sit at any of four
   positions 90 deg apart. The task context puts the adaptor's bolts at -21 deg + 90 deg i about +z_A from +x_A. UR's
   pattern puts the flange's bolts at 45 deg + 90 deg j about +z_E, with the dowel near -90 deg. With z_E = -z_A,
   tool0's x axis then lies at 24 deg + 90 deg k from +x_A about +z_A. This is inferred from those angles; the ur5 task
   checks them.
5. **Geometry.** A cylinder of 40 mm radius and the 170 x 180 mm plate stand for a structure 337 x 312 mm across and
   104 mm deep.
   That structure includes a 251 x 250 x 25 mm robot mount, a 337 x 243 x 3 mm base plate and the camera bracket, which
   reaches x_P = -175 mm. Arm-to-table and arm-to-hand clearances are wrong by the difference.
6. **Mass.** The scene adds 0.35 kg for the stack and 0.092 kg for the plate. The CAD volume is 1836 cm^3, of which the
   robot mount is 770 cm^3. The materials are not in the STEP files, so the payload needs a weighing.
7. **Frames.** No frame is named tool0, and the IK runs on `palm_site`. The UR controller's TCP and payload are set in
   tool0, so the transform from tool0 to P must be written once in the model and once in PolyScope.
8. **Excludes.** Wrist links are excluded against `palm_pose` and four finger segments. `<f>_mount`, which has 36
   collision spheres per finger in cad_v1 and cad_v2, is not excluded. At 104 mm from the flange this has no effect,
   but a cad_v3 build should add the pairs.
9. **Base.** The arm stands at (-0.5, 0, 0.3) m on a pedestal. The lab cell's base frame and table come from the UR5
   calibration cell (`scripts/ur5_cb3_cell.py`, cell.json).

## Specification of the simplified cad_v3 model

### What stays identical

1. The finger subtrees `<f>_mount` .. `<f>_tip` are copied unchanged from the cad_v2 scene: meshes, the 447 collision
   spheres per finger, inertias, the tip pads built at use with `tcd_ell7.4x8.5x11x22`. All 39 contact excludes stay.
2. Body, joint, actuator, default-class and keyframe names, joint axes and ranges, actuator classes and the qpos layout
   of the section above stay. nq is 31 with morph slides and 22 rigid, nu 15.
3. `palm_pose` stays a direct child of the world body, with the six palm joints in floating scenes, gravcomp 1 and its
   frame P. Its mass stays 0.0918 kg: every new geom has `mass="0"`.
4. `palm_plate` stays the first geom of `palm_pose`, a box with that name, as the flat lower envelope of the gantries.
5. The mount origins stay (-50, 0), (50, 55), (50, -55) mm. If the CAD's travel centres differ, the residuals are
   reported. A change then goes into `REAL_V1_MOUNTS`, `FINGER_GEOMETRY`, `MOUNTS_MM` and `morph_vector` together, after
   the owner agrees.
6. No new joints anywhere, and no new bodies under `palm_pose` in the floating scenes.

### What changes, in the model and in code

1. `palm_plate` takes the gantries' footprint and the bridges' lowest boss height (table below), and
   `hand_object_scale_kin.PLATE_BOTTOM` follows it, from 23.5 to 24.4 mm.
2. `SCALE_HAND=cad_v3` is accepted. The v4b-tip tests read one flag, `V4B = HAND in ("cad_v2", "cad_v3")`, in place of
   the eight equality tests, and `CAD_DIR`, `GEN` and `SCENES` point at the cad_v3 folders.
3. `hand_object_scale_sim.cad_fingers` copies `palm_pose`'s geoms and sites from the cad_v3 scene with the fingers.
4. Scene caches of cad_v2 runs are not reused: the palm geometry differs (`HOS_SCENES`, a new `GEN` folder).
5. The output folder is date-prefixed, `assets/mjcf/experimental/20261104-cad_v3/`, with a hand file and a scene file:
   the generator needs the pair (morphology-scenes skill).

### Frames and transforms

- A site `cad_A` on `palm_pose` at A's origin, with A's axes, so that every CAD coordinate maps to P through it. With
  the inferred axes, x_A = y_P, y_A = -x_P and z_A = z_P: the site's quat is a +90 deg turn about z_P. A's origin lies
  at (y_o, 111.0, 75.9) mm in P, where y_o is the y_A coordinate of the travel-centre line.
- Four sites `tool0_k0` .. `tool0_k3` on `palm_pose`. Each is the pose of tool0 for clocking k. The origin is the flange
  axis on the adaptor face, (-111.0, 116.0, 28.0) mm in A, which is (y_o - 116.0, 0, 103.9) mm in P. z points along
  -z_A, and x at 24 deg + 90 deg k from +x_A once the ur5 task confirms the bolt angles.
- A numeric `<custom>` field `adaptor_clock` records k in the arm scene. For each k the builder prints T(tool0 -> P),
  the TCP and the payload's centre of mass in tool0. They are printed in PolyScope 3.15's form: x, y, z in mm and a
  rotation vector in rad.

### Geoms to add, all static on `palm_pose`

Collision boxes use a new default class `hand_struct` (geom group 3, `mass="0"`, contype and conaffinity 1). Names
start with `hand_` or `gantry_`, never with a finger name, and contain neither `_pad` nor `_s<digit>`. Extents are from
the A bounding boxes in `parts.json` (hand-file indices in brackets) and are converted to P by the transform above.

| geom | source parts | A extent x / y / z (mm) | note |
|---|---|---|---|
| `palm_plate` | swept bridges | union of the three frames, x -222..0, y 0..240.4; underside z -51.5 | the bridges' lowest boss; z_P 24.4 (was 23.5) |
| `hand_base_plate` | [002] | -279.3..57.3 / 0..242.9 / 0..3 | |
| `hand_robot_mount` | [001] | -236.5..14.5 / -3.5..246.4 / 3..28 | covers the adaptor [004], which needs no box |
| `gantry_frame_index` | [074] | -110..0 / 0..120 / -22..0 | if gantry <1> is the index |
| `gantry_frame_middle` | [102] | -222..-112 / 0..120 / -22..0 | |
| `gantry_frame_thumb` | [045] | -191..-31 / 120.4..240.4 / -22..0 | |
| `gantry_xmotor_index` | [070, 071, 077] | -8.5..56.8 / 2.5..22.5 / -24..0 | J2 stepper and bracket, fixed |
| `gantry_xmotor_middle` | [003, 062, 118] | -278.8..-212.6 / 2.5..22.5 / -27.5..0 | J4 stepper and bracket, fixed |
| `gantry_xmotor_thumb` | [051, 060, 061] | -247.8..-182.5 / 218.4..240.4 / -24..0 | J0 stepper and bracket, fixed |
| `gantry_ymotor_index` | [063, 072] swept | bridge x range swept over the travel / -35.9..21.3 / -54..-30.5 | J3 stepper on the bridge |
| `gantry_ymotor_middle` | [103, 115] swept | as above / -35.9..21.3 / -54..-30.5 | J5 stepper |
| `gantry_ymotor_thumb` | [040, 044] swept | as above / 219.1..276.3 / -54..-30.5 | J1 stepper |
| `hand_camera_bracket` | [000] | -14.2..25.8 / 111.0..295.5 / -10..0 | the sensors task adds the D435 and tag sites |
| `hand_side_bracket` | [005] | 0.1..45.1 / 103.3..120.0 / -57..18 | reaches z_P 18.9, outside the fingers' reach |

A swept box covers the stage's whole travel. Its x_A extent is the parts' own x extent (20 mm for the y-stepper) plus
the travel (60 mm pair, 110 mm thumb), centred on the travel centre. The bridges themselves are 22.5 mm (pair) and
24.5 mm (thumb) wide. Their bodies above z_P 47.4 add nothing under the plate and are left out.

The exact alternative places each bridge and its y-stepper at the design's position. That requires a jointless body
`gantry_bridge_<f>` under `palm_pose`, whose y offset the generator bakes and `hom_gpu` writes per world. It is not
drop-in: the generator, `hom_gpu` and `cad_fingers` change, and in unbaked scenes the bridge stays at its nominal
position.

### Mass and inertia

Floating scenes keep the palm's 0.0918 kg. The arm scene adds a jointless body `hand_payload` under `palm_pose` with the
weighed mass, and the centre of mass and inertia from the CAD volumes at that mass. The volume-weighted centre of the
hand file lies at (-112.0, 107.8, -10.4) mm in A for uniform density.

### Arm attachment

The arm is the lab UR5 CB3, built from the URDF named above (the ur5 task of this workflow). It keeps the joint names of
`morphohand.tools.arm_ik.UR5E_JOINTS` and the body names `base`, `shoulder_link` .. `wrist_3_link`. The root body
`base` equals the URDF's `base` frame, the controller's B. A body `tool0` is a child of `wrist_3_link` with the
identity pose. `palm_pose` becomes a child of `tool0`, with no joints, pos and quat equal to T(tool0 -> P) for the
chosen k. The hand subtree is the same file content in floating and arm scenes. The IK model has a site `tool0` and a
site `palm_site` at T(tool0 -> P), so `ArmIK` works unchanged. The arm-to-hand excludes cover `wrist_1_link` ..
`wrist_3_link` against `palm_pose` and against every finger body including `<f>_mount`.

### Checks before use

1. The cad_v3 scene compiles with nq 31, nv 30 and nu 15. The generator bakes a design to nq 22. All joint, actuator
   and body names equal cad_v2's; this is checked by a diff of the name lists.
2. At cad_v2's `open` key, the finger bodies' world poses equal cad_v2's to 1e-9 m.
3. FK of the arm model's `tool0` equals `ur5_cb3_cell.link_frames(q)["tool0"]` within 0.01 mm at 100 random q.
4. The mesh assembly in P overlays the simplified boxes in a render, checked by eye, for each k.
5. Over the servo-calibrated joint ranges (`hand_object_scale_kin.limits`), the closest approach of each finger's
   collision geoms to the new palm geoms is reported. A fully curled CAD finger, MCP and PIP both at 92 deg, puts its
   pad centre 16.6 mm above the yaw plane (computed here). If any distance is negative, the palm-finger excludes hide a
   real contact.

### The full-fidelity model

The full model uses the same body, joint and actuator names, so a state transfers by joint name (as
`build_real_v1_arm_scene.solve_home` copies it). It shows every CAD part as a visual mesh in group 2. If its bridges
move, their slide joints are appended after all other joints and coupled to `<f>_y` by joint equalities, so qpos 0-30
keep their meaning. That model is never a generation source.

## Open questions for the owner

1. The index gantry. The right-handed reading puts the finger the driver calls index (J2/J3) on the +x_A side of the
   CAD, gantry <1>. The owner confirms it or names the other gantry.
2. The hard stops. The travel centres along y_A fix P's origin, and with it the residuals against the mounts (-50, 0)
   and (50, +-55) mm. The frame centres alone give an index-to-middle spacing of 112 mm against the nominal 110 mm.
3. The hand's mass and the materials of the robot mount and base plate, for the payload.
4. The place of the STEP export: it sits in `assets/cad_v2/full_cad/` (untracked), and `assets/cad_v3/` would match the
   model's name.
