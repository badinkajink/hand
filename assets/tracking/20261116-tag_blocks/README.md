# Tag blocks of the tracked cylinder and sphere, second version, 20261116

The two block-and-stem parts of `assets/tracking/20261112-objects/` with the stem's root widened and filleted (owner, 2026-11-16: "we probably shouldve filleted the mounting point between the rod and the base of the apriltag hex block"; no stem had broken). Written by `scripts/hardware/tracking_objects_print.py blocks`. They replace the first version's blocks; the cylinder body, the sphere's halves and the dowels stay those of 2026-11-12.

| Part | Files | Print |
|---|---|---|
| cylinder block and stem, ids 10-15, root widened and filleted | `20261116-cylinder_block_white.stl`, `20261116-cylinder_block_black.stl` | upright, block on the bed |
| sphere block and stem, ids 16-21, root widened and filleted | `20261116-sphere_block_white.stl`, `20261116-sphere_block_black.stl` | upright, block on the bed |

Geometry. In the first version the 4 mm stem met the block's top face at a square corner and reached 0.5 mm into it. Here its root is 6 mm across at the top face and tapers to the stem over 10 mm. A concave fillet of 3 mm joins the taper to the face, so the root's foot is 11.4 mm across. The block, its tags, the stem's length, the cone, the seat and the key are unchanged, and so are the tag maps. The root ends 10 mm above the block's top, 5 mm below the lowest finger point of the plan's five layouts and the measured servo's five best per task, at the grasps' open and touch poses. The simulation's stem stays a plain cylinder.

Bending. A side force on the object bends the stem about its root. The lever arm from the block's top to the object's centre is 48.1 mm on the cylinder and 45.7 mm on the sphere. Per newton of side force, the nominal stress at the first version's root was 7.7 MPa on the cylinder and 7.3 MPa on the sphere, and the square corner raised it by a factor of about 2. In this version the largest nominal stress lies where the taper meets the stem: 6.1 and 5.7 MPa per newton, without a corner. At the root it is 2.3 and 2.2 MPa per newton.

Mass. The root adds 0.13 g (cylinder) and 0.13 g (sphere) of solid PLA. The stem with its root, cone and key is 0.88 g and 0.71 g, against 0.75 g and 0.57 g in the first version. Weigh each assembled object; the simulation takes the measured mass.

Checks. The black cells are those of the first version (shared volume equal to their volume). In the CAD assembly the parts share 0 mm3 with the cylinder and 0 mm3 with the sphere. Every tag face of the black bodies, drawn at 20 px/mm from the mesh, decodes as its own id with corners 0.022 mm RMS from the map.

Clearance in the rollouts. The rollouts of 2026-11-05 with the seat and the arm's lift (configurations built and flat, 200 rollouts; `object_tracking_rollouts.py rootclear`) were replayed with the root drawn by two cylinders that enclose it: 6 mm across up to 10 mm and 11.4 mm across up to 3 mm. On the cylinder's tasks at all five layouts no finger came within 6.8 mm of them (100 rollouts). On the sphere at the generalist (70, 110) and pin-push (75, 55) layouts, where the sphere runs, none came within 9.4 mm (40 rollouts); the trial series runs the pin-push layout at (75, 56.8). At the other three layouts the 4 mm stem already overlapped a finger by up to 1.00 mm, and the enclosing cylinders by up to 1.62 mm. Those rollouts used the sphere's earlier 25.2 mm stem, which put the root 0.5 mm nearer the fingers than the printed 25.7 mm.

Third version (owner, 2026-11-16), the blocks held on their objects without glue by a cotter joint: `assets/tracking/20261116-cotter_joint/`.

Bambu Studio project: `20261116-tag_blocks_p1s.3mf`, both blocks on one plate, the black on filament 1 and the white on filament 2, written on each object. Map them to the black and white spools when sending. The saved project, sliced again by the command line: 3.2 h, 42.8 g white and 10.2 g black, 67 filament loads (33 black, 34 white). Not in git; `tracking_objects_print.py blocks bambu` rebuilds it.
