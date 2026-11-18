# Larger tracked cylinder and sphere with the cotter joint, 20261117

The tracked objects at the size-ranking test's large sizes (`scripts/hom/hom_size_rank.py`), a 45 mm x 100 mm cylinder and a 67 mm sphere, each held on its tag block by the cotter joint of `assets/tracking/20261116-cotter_joint/` (owner, 2026-11-17: "prepare these but with a larger diameter sphere, cylinder, since our current ones aren't broken yet"). The owner asked for one print job in a single PLA filament: "i will glue on a paper apriltag onto the cylinder ends if it turns out being crucial". Written by `scripts/hardware/tracking_objects_print.py --set large large build|bambu`. Their tag blocks and stems are the modular ones of `assets/tracking/20261117-modular_blocks/`, codes C45 and S67.

| Part | File | Print |
|---|---|---|
| cylinder body 45 mm (43.4 mm printed for the tape), no end tags, with the joint's socket and slot | `20261117-cylinder_white.stl` | standing on its -z end |
| sphere 67 mm, lower hemisphere with the joint's socket and slot | `20261117-sphere_lower_white.stl` | flat side down, the socket on top |
| sphere 67 mm, upper hemisphere | `20261117-sphere_upper_white.stl` | flat side down |
| dowel pin of the sphere (print three) | `20261117-sphere_pin_white.stl` | standing; print three |
| cotter 0.25 mm thinner, one dimple | `20261117-cotter_thin_white.stl` | on its side, dimples up |
| cotter, two dimples | `20261117-cotter_mid_white.stl` | on its side, dimples up |
| cotter 0.25 mm thicker, three dimples | `20261117-cotter_thick_white.stl` | on its side, dimples up |

Geometry. The cylinder prints at 43.4 mm, 100 mm long, so 0.8 mm grip tape brings it to 45 mm. Its end faces are bare and flat, 43.4 mm across, for paper tags. The sphere is 67 mm and bare, printed as two hemispheres joined by three dowels 10 mm from its axis. The flat seat is 10 mm across and lies 0.58 mm (cylinder) and 0.38 mm (sphere) below the printed surface.

Joint. The socket, its slot and the stem's taper and post are those of 2026-11-16. The cotter enters on the cylinder's -x side, as on the 25 mm cylinder. On the sphere the joint is turned 180 deg about the stem, so its cotter enters on the -z side: in the replayed rollouts fingers touched the 67 mm sphere at the +z opening (see below). The slot reaches the surface further from the stem on the larger objects, so the cotters are 19.8 mm long in place of 13.7 mm. At its nominal place each end of a cotter lies 2.05 mm inside the cylinder's surface and 4.75 mm inside the sphere's. Their thickness along the taper is unchanged, so the fit coupons of 2026-11-16 test this joint too, and the cotter that fits a coupon fits these objects.

Stems and blocks. The block's top is to lie 31 mm (cylinder) and 16.7 mm (sphere) below the object's taped or bare surface, against 35.6 and 25.7 mm on the smaller objects. These lengths put the block 15 mm below the lowest finger point at the open and touch poses of the ten best layouts per task in the HOM grids at these sizes (`grid2dff_v4b_sv25r`: the SCS0009 plant, 2.5 MPa pads; `object_tracking_rollouts.py --set large stem`). The blocks bear ids 32-37 (cylinder) and 38-43 (sphere), so the tracker tells them from the smaller objects' blocks (`object_tracking_mjcf.OBJECT_SETS`).

Checks. In the CAD assembly the stem's end, the object and the cotter share 0 mm3 on both objects. The shared volume is 0 at the cotter's nominal place and at both ends of its travel, for each of the three cotters. Over its travel each end stays 0.55 to 3.55 mm inside the cylinder's surface and 3.25 to 6.25 mm inside the sphere's. A stem turned by 180 deg stops 1.2 mm short of its seat.

Rollouts. Each of the 40 layouts above ran its task with its best grasp at seeds 20-24 in three scenes: the study's (a 3 mm post under the object until 1.6 s), and the object on its stem and block with the block on a flat seat or in the seat's 4 mm pocket, the arm lifting the hand 40 mm at 1.6 s (`object_tracking_rollouts.py --set large run`, 600 rollouts on the SCS0009 plant, CPU MuJoCo). Scores are means; the finger distance to the stem or block is capped at 10 mm, the range of the contact query.

| Task | Object | Layouts | Score, study's scene | Score, flat seat | Score, pocket seat | Held, pocket seat | Nearest finger, flat seat (mm) | Nearest finger, pocket seat (mm) |
|---|---|---|---|---|---|---|---|---|
| twist | 67 mm sphere | 10 | 0.61 | 0.63 | 0.62 | 100 % | 1.1 | 0.3 |
| roll | 67 mm sphere | 10 | 0.54 | 0.53 | 0.52 | 100 % | 10.0 | 10.0 |
| pin-push | 45 mm cylinder | 10 | 0.73 | 0.65 | 0.67 | 98 % | 10.0 | 10.0 |
| press | 45 mm cylinder | 10 | 0.77 | 0.79 | 0.50 | 78 % | 9.1 | 10.0 |

On the flat seat the scores stay within 0.02 of the study's scene, except the pin-push (0.08 lower). In the pocket the press loses 0.27, as on the 25 mm cylinder, where the series uses the flat seat. The twist's nearest finger came at its fourth layout, (150, 70): late in the twist the middle finger came within 1.1 mm of the stem or block on the flat seat and 0.3 mm in the pocket, without touching. At the other layouts no finger came within 9 mm. The stem's length can change with the modular stem without reprinting the sphere.

Finger contacts near the openings. The rollouts on the two seats were replayed (200 on the cylinder, 200 on the sphere; `object_tracking_rollouts.py --set large touchmap`) and every finger contact within 0.5 mm of the object's surface recorded. The nearest contacts were 16.0 mm (entry) and 1.2 mm (far opening) on the cylinder, and 10.0 and 0.0 mm on the sphere. Rollouts with a contact within 2 mm of an opening: 0 and 7 of 200 on the cylinder, 0 and 4 of 200 on the sphere.

Mass. The simulations take each object with its block, stem and tape as 25 g. Each body was sliced alone in the plate's process (0.20 mm Standard, two walls) at five infills, the sphere with its dowels:

| Infill | Cylinder body (g) | Sphere, halves and dowels (g) |
|---|---|---|
| 0 % gyroid | 16.7 | 23.7 |
| 5 % gyroid | 26.0 | 33.0 |
| 10 % gyroid | 34.2 | 41.1 |
| 10 % lightning | 17.4 | 25.6 |
| 20 % lightning | 18.2 | 25.6 |

Lightning infill supports only the top faces, so it adds 1.5 g (cylinder) and 1.9 g (sphere) to the empty bodies; gyroid adds 1.8 to 1.9 g per percent. With the block (3 g in the plan), a stem of about 1 g and a cotter of 0.15 g, the cylinder at 20 % lightning comes to about 22 g before its tape and the sphere to about 30 g. The sphere is above 25 g even with no infill. Weigh each assembled object; the simulation takes the measured mass.

Bambu Studio project `20261117-large_objects_p1s.3mf`: the cylinder body, both hemispheres, three dowels and two cotters of each size, in white on filament 1, 20 % lightning infill. The saved project, sliced again by the command line: 1.4 h, 44.4 g, one filament and no prime tower. Map filament 1 to the white spool when sending. Not in git; `tracking_objects_print.py --set large large bambu` rebuilds it.
